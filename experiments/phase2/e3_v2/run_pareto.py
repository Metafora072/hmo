"""Frozen 0.8B quality-memory Pareto evaluation for HMO."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from argparse import Namespace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from experiments.phase2.e3_v2.context_query import (
    full_kv_intervention,
    generate_greedy,
    run_post_intervention_prompt,
    tokenize_sample_prompt,
)
from experiments.phase2.e3_v2.coverage_fidelity import (
    ALLOCATION_SCHEMA,
    allocate_coverage_fidelity,
)
from experiments.phase2.e3_v2.coverage_fidelity_cache import (
    GLOBAL_FIXED_CHUNK_SELECTOR,
    build_global_fixed_chunk_topk_position_plan,
    build_raw_exact_slack_position_plan,
    build_retained_position_plan,
    make_coverage_fidelity_intervention,
)
from experiments.phase2.e3_v2.oracle import (
    OracleConfig,
    OracleContractError,
    build_segment_catalog,
)
from experiments.phase2.e3_v2.query_accessibility import (
    collect_hybrid_query_token_probe,
)
from experiments.phase2.e3_v2.real_model_preflight import (
    REFERENCE_BACKEND,
    _force_torch_reference_backend,
    _load_model,
    model_provenance,
)
from experiments.phase2.e3_v2.run_coverage_fidelity import (
    METRICS,
    _append_jsonl,
    _atomic_json,
    _cleanup_cuda,
    _eligible_action_counts,
    _pair_summary,
    restrict_eligible_signals,
)
from experiments.phase2.e3_v2.run_discovery import _build_samples
from experiments.phase2.e3_v2.run_end_task import (
    score_generated_text,
    select_equal_byte_segments,
)
from experiments.utils.memory_accounting import get_active_kv_bytes
from experiments.utils.model_loader import (
    get_full_attention_indices,
    get_linear_attention_indices,
)
from experiments.utils.run_manifest import collect_environment, ensure_run_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_SCHEMA = "hmo.contiguous_cf.pareto_protocol.v1"
RESULT_SCHEMA = "hmo.contiguous_cf.pareto_result.v1"
RESULTS_FILENAME = "pareto_results.jsonl"
SUMMARY_FILENAME = "pareto_summary.json"
SYSTEMS = (
    "contiguous_cf",
    "global_fixed_chunk_topk",
    "raw_alpha_exact_slack",
    "scattered_cf",
    "contiguous_sparse_only",
    "full_kv_reference",
)
EQUAL_BYTE_SYSTEMS = SYSTEMS[:-1]
BUDGET_FRACTIONS = (0.05, 0.1, 0.2)
STAGE_SETS = {
    "smoke": ("smoke",),
    "formal": ("8k", "16k"),
}


def _budget_key(fraction: float) -> str:
    return f"{int(round(fraction * 100)):02d}pct"


def load_pareto_protocol(path: Path) -> tuple[dict, str]:
    try:
        encoded = path.read_bytes()
        payload = json.loads(encoded)
    except (OSError, json.JSONDecodeError) as exc:
        raise OracleContractError(f"cannot read Pareto protocol: {path}") from exc

    method = payload.get("method", {})
    stages = payload.get("stages", {})
    shapes = {
        name: (
            stage.get("datasets"),
            int(stage.get("samples_per_dataset", 0)),
            int(stage.get("context_length", 0)),
            int(stage.get("segment_length", 0)),
            int(stage.get("seed", 0)),
            stage.get("sample_id_prefix"),
        )
        for name, stage in stages.items()
    }
    expected_comparisons = [
        ["contiguous_cf", "global_fixed_chunk_topk"],
        ["contiguous_cf", "scattered_cf"],
    ]
    if (
        payload.get("schema_version") != PROTOCOL_SCHEMA
        or payload.get("status") != "frozen_before_outcomes"
        or tuple(payload.get("systems", ())) != SYSTEMS
        or tuple(payload.get("equal_byte_systems", ())) != EQUAL_BYTE_SYSTEMS
        or payload.get("primary_comparisons") != expected_comparisons
        or payload.get("primary_metric") != "normalized_answer_contains"
        or tuple(payload.get("budget_fractions", ())) != BUDGET_FRACTIONS
        or payload.get("model_id") != "Qwen/Qwen3.5-0.8B"
        or not payload.get("model_revision")
        or int(payload.get("max_new_tokens", 0)) <= 0
        or int(payload.get("inference_seed", 0)) <= 0
        or method
        != {
            "allocator": "attention_led",
            "sparse_selector": "max_mass_window",
            "sparse_width": 16,
            "raw_slack_selector": "global_top_tokens_slack",
            "global_fixed_chunk_width": 16,
            "global_fixed_chunk_slack": "prefix_of_next_ranked_chunk",
            "protected_prefix_segments": 1,
            "protected_suffix_segments": 1,
        }
        or set(stages) != {"smoke", "8k", "16k"}
        or shapes.get("smoke", ())[:4] != ("needle", 1, 8192, 256)
        or shapes.get("8k", ())[:4]
        != ("needle,longeval_lines", 12, 8192, 256)
        or shapes.get("16k", ())[:4]
        != ("needle,longeval_lines", 12, 16384, 256)
        or any(shape[4] <= 0 or not shape[5] for shape in shapes.values())
        or len({shape[4] for shape in shapes.values()}) != len(shapes)
        or shapes["8k"][4:] != (
            20261005,
            "contiguous_cf_confirm_8k_s20261005_",
        )
        or shapes["16k"][4:] != (
            20261006,
            "contiguous_cf_confirm_16k_s20261006_",
        )
    ):
        raise OracleContractError("Pareto protocol mismatch")
    return payload, hashlib.sha256(encoded).hexdigest()


def _load_completed(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise OracleContractError(
                    f"invalid Pareto result at line {line_number}"
                ) from exc
    keys = [
        (row.get("stage"), row.get("sample_id"), row.get("budget_fraction"))
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise OracleContractError("Pareto results contain duplicate budget cases")
    return rows


def summarize_pareto_results(rows: Sequence[Mapping]) -> dict:
    if not rows:
        raise OracleContractError("Pareto summary requires result rows")
    by_budget = {}
    for fraction in sorted({float(row["budget_fraction"]) for row in rows}):
        selected = [
            row for row in rows if float(row["budget_fraction"]) == fraction
        ]
        systems = {
            system: {
                metric: float(
                    np.mean([row["systems"][system][metric] for row in selected])
                )
                for metric in METRICS
            }
            for system in SYSTEMS
        }
        mean_resident = {
            system: float(
                np.mean(
                    [
                        row["systems"][system]["post_query_resident_kv_bytes"]
                        for row in selected
                    ]
                )
            )
            for system in SYSTEMS
        }
        by_budget[_budget_key(fraction)] = {
            "budget_fraction": fraction,
            "case_count": len(selected),
            "systems": systems,
            "by_stage_dataset": {
                f"{stage}/{dataset}": {
                    "case_count": sum(
                        row["stage"] == stage and row["dataset"] == dataset
                        for row in selected
                    ),
                    "systems": {
                        system: {
                            metric: float(
                                np.mean(
                                    [
                                        row["systems"][system][metric]
                                        for row in selected
                                        if row["stage"] == stage
                                        and row["dataset"] == dataset
                                    ]
                                )
                            )
                            for metric in METRICS
                        }
                        for system in SYSTEMS
                    },
                }
                for stage, dataset in sorted(
                    {(row["stage"], row["dataset"]) for row in selected}
                )
            },
            "comparisons": {
                f"contiguous_cf_vs_{system}": _pair_summary(
                    selected, "contiguous_cf", system
                )
                for system in SYSTEMS
                if system != "contiguous_cf"
            },
            "mean_post_query_resident_kv_bytes": mean_resident,
            "mean_resident_fraction_of_full": {
                system: float(
                    np.mean(
                        [
                            row["systems"][system][
                                "post_query_resident_kv_bytes"
                            ]
                            / row["systems"]["full_kv_reference"][
                                "post_query_resident_kv_bytes"
                            ]
                            for row in selected
                        ]
                    )
                )
                for system in SYSTEMS
            },
            "equal_resident_byte_cases": sum(
                len(
                    {
                        row["systems"][system]["post_query_resident_kv_bytes"]
                        for system in EQUAL_BYTE_SYSTEMS
                    }
                )
                == 1
                for row in selected
            ),
        }
    return {
        "budget_case_count": len(rows),
        "sample_case_count": len(
            {(row["stage"], row["sample_id"]) for row in rows}
        ),
        "by_budget": by_budget,
    }


def _generate_system(
    model,
    tokenizer,
    prompt,
    attention_layers,
    recurrent_layers,
    intervention,
    sample,
    max_new_tokens,
):
    state = run_post_intervention_prompt(
        model,
        prompt,
        attention_layer_indices=attention_layers,
        recurrent_layer_indices=recurrent_layers,
        intervention=intervention,
    )
    resident_bytes = get_active_kv_bytes(state.cache, list(attention_layers))
    answer = generate_greedy(
        model,
        tokenizer,
        state,
        max_new_tokens=max_new_tokens,
    )
    payload = {
        **score_generated_text(answer.text, sample),
        "generated_text": answer.text,
        "generated_token_ids": [int(value) for value in answer.token_ids[0].tolist()],
        "post_query_resident_kv_bytes": int(resident_bytes),
    }
    del state
    _cleanup_cuda()
    return payload


def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Pareto evaluation requires exactly one visible GPU")
    protocol, protocol_sha = load_pareto_protocol(Path(args.protocol).resolve())
    if (
        args.model_id != protocol["model_id"]
        or args.model_revision != protocol["model_revision"]
    ):
        raise OracleContractError("model identity disagrees with Pareto protocol")

    stage_names = STAGE_SETS[args.stage_set]
    method = protocol["method"]
    fractions = tuple(float(value) for value in protocol["budget_fractions"])
    run_dir = Path(args.run_dir).resolve()
    model_path = Path(args.model_path).resolve()
    model_identity = model_provenance(
        model_path, args.model_id, revision=args.model_revision
    )
    manifest = ensure_run_manifest(
        run_dir,
        experiment="contiguous_coverage_fidelity_pareto",
        args={
            "model_id": args.model_id,
            "model_revision": args.model_revision,
            "protocol_sha256": protocol_sha,
            "stage_set": args.stage_set,
            "max_new_tokens": protocol["max_new_tokens"],
            "inference_seed": protocol["inference_seed"],
            "recurrent_backend": REFERENCE_BACKEND,
            "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        selections={
            "scope": (
                "operational_smoke_excluded_from_claims"
                if args.stage_set == "smoke"
                else "matched_confirmation_suite_pareto"
            ),
            "systems": list(SYSTEMS),
            "equal_byte_systems": list(EQUAL_BYTE_SYSTEMS),
            "primary_comparisons": protocol["primary_comparisons"],
            "budget_fractions": list(fractions),
            "allocator_schema": ALLOCATION_SCHEMA,
            "fixed_chunk_selector": GLOBAL_FIXED_CHUNK_SELECTOR,
            "method": method,
            "stages": list(stage_names),
            "candidate_search": False,
            "continuation_gate": False,
        },
        model=model_identity,
        project_root=PROJECT_ROOT,
        require_clean=True,
        environment=collect_environment(),
    )

    results_path = run_dir / RESULTS_FILENAME
    completed = _load_completed(results_path)
    if completed and not args.resume:
        raise OracleContractError("Pareto results exist; pass --resume to continue")
    completed_keys = {
        (row["stage"], row["sample_id"], float(row["budget_fraction"]))
        for row in completed
    }
    rows = list(completed)

    torch.manual_seed(protocol["inference_seed"])
    torch.cuda.manual_seed_all(protocol["inference_seed"])
    torch.set_float32_matmul_precision("highest")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model, tokenizer = _load_model(model_path)
    attention_layers = tuple(get_full_attention_indices(model.config))
    recurrent_layers = tuple(get_linear_attention_indices(model.config))
    if not attention_layers or not recurrent_layers:
        raise OracleContractError("Pareto model is not a Hybrid architecture")
    _force_torch_reference_backend(model, recurrent_layers)

    samples_by_stage = {
        stage: _build_samples(tokenizer, Namespace(**protocol["stages"][stage]))
        for stage in stage_names
    }
    cases = [
        (stage, sample)
        for stage in stage_names
        for sample in samples_by_stage[stage]
    ]
    expected_sample_count = sum(
        len(protocol["stages"][stage]["datasets"].split(","))
        * protocol["stages"][stage]["samples_per_dataset"]
        for stage in stage_names
    )
    if len(cases) != expected_sample_count or len(
        {(stage, sample.sample_id) for stage, sample in cases}
    ) != len(cases):
        raise OracleContractError("Pareto sample construction changed")

    for case_index, (stage, sample) in enumerate(cases, start=1):
        missing_fractions = [
            fraction
            for fraction in fractions
            if (stage, sample.sample_id, fraction) not in completed_keys
        ]
        if not missing_fractions:
            continue
        stage_config = protocol["stages"][stage]
        prompt = tokenize_sample_prompt(sample, tokenizer)
        with torch.no_grad():
            context_outputs = model.model(
                prompt.context_ids.to(model.device),
                use_cache=True,
                return_dict=True,
            )
        segments = build_segment_catalog(
            context_outputs.past_key_values,
            attention_layers,
            context_tokens=prompt.context_tokens,
            config=OracleConfig(
                segment_length=stage_config["segment_length"],
                middle_kv_fraction=max(fractions),
            ),
        )
        del context_outputs
        _cleanup_cuda()

        probe = collect_hybrid_query_token_probe(
            model,
            prompt,
            attention_layer_indices=attention_layers,
            recurrent_layer_indices=recurrent_layers,
            segments=segments,
            segment_length=stage_config["segment_length"],
        )
        attention = probe.alpha.as_dict()
        accessibility = probe.accessibility.field_dict("read_share")
        token_attention_mass = tuple(float(value) for value in probe.token_attention_mass)
        allocator_attention, allocator_accessibility = restrict_eligible_signals(
            attention, accessibility, segments
        )
        del probe
        _cleanup_cuda()

        existing = next(
            (
                row
                for row in rows
                if row["stage"] == stage and row["sample_id"] == sample.sample_id
            ),
            None,
        )
        if existing is None:
            full_generated = _generate_system(
                model,
                tokenizer,
                prompt,
                attention_layers,
                recurrent_layers,
                full_kv_intervention,
                sample,
                protocol["max_new_tokens"],
            )
        else:
            full_generated = dict(existing["systems"]["full_kv_reference"])

        eligible = [segment for segment in segments if segment.eligible]
        token_unit_bytes = eligible[0].kv_bytes // eligible[0].token_count
        query_kv_bytes = prompt.query_tokens * token_unit_bytes
        protected_bytes = sum(
            segment.kv_bytes for segment in segments if segment.protected
        )
        full_expected = (
            sum(segment.kv_bytes for segment in segments) + query_kv_bytes
        )
        if full_generated["post_query_resident_kv_bytes"] != full_expected:
            raise OracleContractError("Pareto Full-KV byte accounting failed")

        for fraction in missing_fractions:
            budget_started = time.perf_counter()
            raw_selection = select_equal_byte_segments(
                attention,
                accessibility,
                segments,
                middle_kv_fraction=fraction,
            )
            exact_plan = allocate_coverage_fidelity(
                allocator_attention,
                allocator_accessibility,
                segments,
                middle_kv_fraction=fraction,
                sparse_width=method["sparse_width"],
                use_accessibility=False,
                enable_exact_upgrades=True,
            )
            sparse_plan = allocate_coverage_fidelity(
                allocator_attention,
                allocator_accessibility,
                segments,
                middle_kv_fraction=fraction,
                sparse_width=method["sparse_width"],
                use_accessibility=False,
                enable_exact_upgrades=False,
            )
            if exact_plan.total_charged_bytes != sparse_plan.total_charged_bytes:
                raise OracleContractError("Pareto HMO plans are not equal-byte")

            position_plans = {
                "contiguous_cf": build_retained_position_plan(
                    exact_plan,
                    segments,
                    token_attention_mass,
                    context_tokens=prompt.context_tokens,
                    sparse_selector="max_mass_window",
                ),
                "global_fixed_chunk_topk": (
                    build_global_fixed_chunk_topk_position_plan(
                        segments,
                        token_attention_mass,
                        context_tokens=prompt.context_tokens,
                        target_context_charged_bytes=exact_plan.total_charged_bytes,
                        chunk_width=method["global_fixed_chunk_width"],
                    )
                ),
                "raw_alpha_exact_slack": build_raw_exact_slack_position_plan(
                    segments,
                    raw_selection["raw_alpha_segment_ids"],
                    token_attention_mass,
                    context_tokens=prompt.context_tokens,
                    target_context_charged_bytes=exact_plan.total_charged_bytes,
                ),
                "scattered_cf": build_retained_position_plan(
                    exact_plan,
                    segments,
                    token_attention_mass,
                    context_tokens=prompt.context_tokens,
                    sparse_selector="top_tokens",
                ),
                "contiguous_sparse_only": build_retained_position_plan(
                    sparse_plan,
                    segments,
                    token_attention_mass,
                    context_tokens=prompt.context_tokens,
                    sparse_selector="max_mass_window",
                ),
            }
            if len(
                {
                    plan.context_charged_bytes
                    for plan in position_plans.values()
                }
            ) != 1:
                raise OracleContractError(
                    "Pareto position plans are not context-byte matched"
                )

            generated = {
                name: _generate_system(
                    model,
                    tokenizer,
                    prompt,
                    attention_layers,
                    recurrent_layers,
                    make_coverage_fidelity_intervention(
                        position_plans[name], attention_layers, name=name
                    ),
                    sample,
                    protocol["max_new_tokens"],
                )
                for name in EQUAL_BYTE_SYSTEMS
            }
            generated["full_kv_reference"] = dict(full_generated)
            equal_resident = {
                generated[name]["post_query_resident_kv_bytes"]
                for name in EQUAL_BYTE_SYSTEMS
            }
            compressed_expected = exact_plan.total_charged_bytes + query_kv_bytes
            if equal_resident != {compressed_expected}:
                raise OracleContractError(
                    "Pareto arms differ from the resident-byte target"
                )

            compressed_cap = exact_plan.total_budget_limit_bytes + query_kv_bytes
            for name in EQUAL_BYTE_SYSTEMS:
                generated[name].update(
                    {
                        "post_query_budget_limit_bytes": compressed_cap,
                        "expected_post_query_resident_kv_bytes": compressed_expected,
                    }
                )
            generated["full_kv_reference"].update(
                {
                    "post_query_budget_limit_bytes": full_expected,
                    "expected_post_query_resident_kv_bytes": full_expected,
                }
            )

            protected_ids = {
                segment.segment_id for segment in segments if segment.protected
            }
            middle_retained_tokens = exact_plan.middle_charged_bytes // token_unit_bytes
            plans = {
                "contiguous_cf": {
                    "allocation": exact_plan.to_dict(),
                    "retention": position_plans["contiguous_cf"].to_dict(),
                    "action_counts": _eligible_action_counts(
                        exact_plan, protected_ids
                    ),
                },
                "global_fixed_chunk_topk": {
                    "chunk_width": method["global_fixed_chunk_width"],
                    "boundary_slack_tokens": (
                        middle_retained_tokens
                        % method["global_fixed_chunk_width"]
                    ),
                    "retention": position_plans[
                        "global_fixed_chunk_topk"
                    ].to_dict(),
                },
                "raw_alpha_exact_slack": {
                    "selected_exact_segment_ids": list(
                        raw_selection["raw_alpha_segment_ids"]
                    ),
                    "retention": position_plans[
                        "raw_alpha_exact_slack"
                    ].to_dict(),
                },
                "scattered_cf": {
                    "allocation": exact_plan.to_dict(),
                    "retention": position_plans["scattered_cf"].to_dict(),
                    "action_counts": _eligible_action_counts(
                        exact_plan, protected_ids
                    ),
                },
                "contiguous_sparse_only": {
                    "allocation": sparse_plan.to_dict(),
                    "retention": position_plans[
                        "contiguous_sparse_only"
                    ].to_dict(),
                    "action_counts": _eligible_action_counts(
                        sparse_plan, protected_ids
                    ),
                },
            }
            row = {
                "stage": stage,
                "sample_id": sample.sample_id,
                "dataset": sample.dataset,
                "answer": sample.answer,
                "context_tokens": prompt.context_tokens,
                "query_tokens": prompt.query_tokens,
                "budget_fraction": fraction,
                "budget_key": _budget_key(fraction),
                "raw_alpha_selection": raw_selection,
                "byte_accounting": {
                    "token_kv_bytes": token_unit_bytes,
                    "query_kv_bytes": query_kv_bytes,
                    "protected_context_kv_bytes": protected_bytes,
                    "compressed_post_query_cap_bytes": compressed_cap,
                    "equal_byte_post_query_resident_bytes": compressed_expected,
                },
                "plans": plans,
                "systems": generated,
                "elapsed_seconds": time.perf_counter() - budget_started,
            }
            rows.append(row)
            _append_jsonl(results_path, row)
            print(
                f"[{case_index}/{len(cases)}] {stage} {sample.dataset} "
                f"{sample.sample_id} {_budget_key(fraction)}: "
                f"hmo={generated['contiguous_cf']['normalized_answer_contains']:.0f} "
                "fixed="
                f"{generated['global_fixed_chunk_topk']['normalized_answer_contains']:.0f} "
                "raw+slack="
                f"{generated['raw_alpha_exact_slack']['normalized_answer_contains']:.0f} "
                f"scattered={generated['scattered_cf']['normalized_answer_contains']:.0f} "
                "sparse="
                f"{generated['contiguous_sparse_only']['normalized_answer_contains']:.0f} "
                f"full={generated['full_kv_reference']['normalized_answer_contains']:.0f}",
                flush=True,
            )

    expected_keys = {
        (stage, sample.sample_id, fraction)
        for stage, sample in cases
        for fraction in fractions
    }
    if {
        (row["stage"], row["sample_id"], float(row["budget_fraction"]))
        for row in rows
    } != expected_keys:
        raise OracleContractError("Pareto run did not complete the selected package")
    rows.sort(
        key=lambda row: (
            row["stage"],
            row["dataset"],
            row["sample_id"],
            float(row["budget_fraction"]),
        )
    )
    summary = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete",
        "scope": (
            "operational_smoke_excluded_from_claims"
            if args.stage_set == "smoke"
            else "matched_confirmation_suite_pareto"
        ),
        "manifest_id": manifest["manifest_id"],
        "protocol_sha256": protocol_sha,
        "analysis": summarize_pareto_results(rows),
        "runtime": {
            "elapsed_seconds": time.perf_counter() - started,
            "gpu_name": torch.cuda.get_device_name(0),
            "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
            "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
        "samples": rows,
    }
    _atomic_json(run_dir / SUMMARY_FILENAME, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--stage-set", choices=tuple(STAGE_SETS), default="formal"
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    payload = run(parse_args())
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key != "samples"},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
