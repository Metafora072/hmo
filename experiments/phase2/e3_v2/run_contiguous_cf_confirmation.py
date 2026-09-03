"""Fresh confirmation for contiguous coverage-fidelity KV allocation."""
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
    make_selected_segment_intervention,
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
PROTOCOL_SCHEMA = "hmo.contiguous_cf.confirmation_protocol.v1"
RESULT_SCHEMA = "hmo.contiguous_cf.confirmation_result.v1"
RESULTS_FILENAME = "contiguous_cf_confirmation_results.jsonl"
SUMMARY_FILENAME = "contiguous_cf_confirmation_summary.json"
SYSTEMS = (
    "contiguous_cf",
    "scattered_cf",
    "contiguous_sparse_only",
    "raw_alpha_exact_topk",
    "full_kv_reference",
)
COMPRESSED_SYSTEMS = SYSTEMS[:3]


def load_confirmation_protocol(path: Path) -> tuple[dict, str]:
    try:
        encoded = path.read_bytes()
        payload = json.loads(encoded)
    except (OSError, json.JSONDecodeError) as exc:
        raise OracleContractError(f"cannot read confirmation protocol: {path}") from exc

    method = payload.get("method", {})
    stages = payload.get("stages", {})
    stage_shapes = {
        name: (
            stage.get("datasets"),
            int(stage.get("samples_per_dataset", 0)),
            int(stage.get("context_length", 0)),
            int(stage.get("segment_length", 0)),
            float(stage.get("middle_kv_fraction", 0.0)),
            int(stage.get("seed", 0)),
            stage.get("sample_id_prefix"),
        )
        for name, stage in stages.items()
    }
    if (
        payload.get("schema_version") != PROTOCOL_SCHEMA
        or payload.get("status") != "frozen_before_outcomes"
        or tuple(payload.get("systems", ())) != SYSTEMS
        or payload.get("primary_comparison")
        != ["contiguous_cf", "raw_alpha_exact_topk"]
        or payload.get("primary_metric") != "normalized_answer_contains"
        or payload.get("model_id") != "Qwen/Qwen3.5-0.8B"
        or not payload.get("model_revision")
        or set(stages) != {"8k", "16k"}
        or int(payload.get("max_new_tokens", 0)) <= 0
        or int(payload.get("inference_seed", 0)) <= 0
        or method
        != {
            "allocator": "attention_led",
            "middle_kv_fraction": 0.1,
            "sparse_selector": "max_mass_window",
            "sparse_width": 16,
            "protected_prefix_segments": 1,
            "protected_suffix_segments": 1,
        }
        or stage_shapes["8k"][:5]
        != ("needle,longeval_lines", 12, 8192, 256, 0.1)
        or stage_shapes["16k"][:5]
        != ("needle,longeval_lines", 12, 16384, 256, 0.1)
        or any(shape[5] <= 0 or not shape[6] for shape in stage_shapes.values())
        or stage_shapes["8k"][5] == stage_shapes["16k"][5]
        or stage_shapes["8k"][6] == stage_shapes["16k"][6]
    ):
        raise OracleContractError("contiguous CF confirmation protocol mismatch")
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
                    f"invalid confirmation result at line {line_number}"
                ) from exc
    keys = [(row.get("stage"), row.get("sample_id")) for row in rows]
    if len(keys) != len(set(keys)):
        raise OracleContractError("confirmation results contain duplicate cases")
    return rows


def _group_system_metrics(rows: Sequence[Mapping]) -> dict:
    output = {}
    groups = sorted({(row["stage"], row["dataset"]) for row in rows})
    for stage, dataset in groups:
        selected = [
            row
            for row in rows
            if row["stage"] == stage and row["dataset"] == dataset
        ]
        output[f"{stage}/{dataset}"] = {
            "case_count": len(selected),
            "systems": {
                system: {
                    metric: float(
                        np.mean(
                            [row["systems"][system][metric] for row in selected]
                        )
                    )
                    for metric in METRICS
                }
                for system in SYSTEMS
            },
        }
    return output


def summarize_confirmation_results(rows: Sequence[Mapping]) -> dict:
    if not rows:
        raise OracleContractError("confirmation summary requires result rows")
    systems = {
        system: {
            metric: float(np.mean([row["systems"][system][metric] for row in rows]))
            for metric in METRICS
        }
        for system in SYSTEMS
    }
    mean_resident = {
        system: float(
            np.mean(
                [row["systems"][system]["post_query_resident_kv_bytes"] for row in rows]
            )
        )
        for system in SYSTEMS
    }
    return {
        "case_count": len(rows),
        "systems": systems,
        "by_stage_dataset": _group_system_metrics(rows),
        "comparisons": {
            "contiguous_cf_vs_raw_alpha_exact_topk": _pair_summary(
                rows, "contiguous_cf", "raw_alpha_exact_topk"
            ),
            "contiguous_cf_vs_scattered_cf": _pair_summary(
                rows, "contiguous_cf", "scattered_cf"
            ),
            "contiguous_cf_vs_contiguous_sparse_only": _pair_summary(
                rows, "contiguous_cf", "contiguous_sparse_only"
            ),
            "contiguous_cf_vs_full_kv_reference": _pair_summary(
                rows, "contiguous_cf", "full_kv_reference"
            ),
        },
        "mean_post_query_resident_kv_bytes": mean_resident,
        "mean_resident_fraction_of_full": {
            system: float(
                np.mean(
                    [
                        row["systems"][system]["post_query_resident_kv_bytes"]
                        / row["systems"]["full_kv_reference"]
                        ["post_query_resident_kv_bytes"]
                        for row in rows
                    ]
                )
            )
            for system in SYSTEMS
        },
        "equal_compressed_resident_byte_cases": sum(
            len(
                {
                    row["systems"][system]["post_query_resident_kv_bytes"]
                    for system in COMPRESSED_SYSTEMS
                }
            )
            == 1
            for row in rows
        ),
    }


def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("confirmation requires exactly one visible GPU")
    protocol, protocol_sha = load_confirmation_protocol(Path(args.protocol).resolve())
    if (
        args.model_id != protocol["model_id"]
        or args.model_revision != protocol["model_revision"]
    ):
        raise OracleContractError("model identity disagrees with confirmation protocol")

    method = protocol["method"]
    run_dir = Path(args.run_dir).resolve()
    model_path = Path(args.model_path).resolve()
    model_identity = model_provenance(
        model_path, args.model_id, revision=args.model_revision
    )
    manifest = ensure_run_manifest(
        run_dir,
        experiment="contiguous_coverage_fidelity_confirmation",
        args={
            "model_id": args.model_id,
            "model_revision": args.model_revision,
            "protocol_sha256": protocol_sha,
            "max_new_tokens": protocol["max_new_tokens"],
            "inference_seed": protocol["inference_seed"],
            "recurrent_backend": REFERENCE_BACKEND,
            "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        selections={
            "scope": "fresh_confirmation_no_postselection",
            "systems": list(SYSTEMS),
            "primary_comparison": protocol["primary_comparison"],
            "allocator_schema": ALLOCATION_SCHEMA,
            "method": method,
            "stages": protocol["stages"],
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
        raise OracleContractError("confirmation results exist; pass --resume to continue")
    completed_keys = {(row["stage"], row["sample_id"]) for row in completed}
    rows = list(completed)

    torch.manual_seed(protocol["inference_seed"])
    torch.cuda.manual_seed_all(protocol["inference_seed"])
    torch.set_float32_matmul_precision("highest")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model, tokenizer = _load_model(model_path)
    attention_layers = tuple(get_full_attention_indices(model.config))
    recurrent_layers = tuple(get_linear_attention_indices(model.config))
    _force_torch_reference_backend(model, recurrent_layers)
    samples_by_stage = {
        stage: _build_samples(tokenizer, Namespace(**protocol["stages"][stage]))
        for stage in ("8k", "16k")
    }
    cases = [
        (stage, sample)
        for stage in ("8k", "16k")
        for sample in samples_by_stage[stage]
    ]
    expected_case_count = sum(
        len(stage["datasets"].split(",")) * stage["samples_per_dataset"]
        for stage in protocol["stages"].values()
    )
    if len(cases) != expected_case_count or len(
        {(stage, sample.sample_id) for stage, sample in cases}
    ) != len(cases):
        raise OracleContractError("fresh confirmation sample construction changed")

    for case_index, (stage, sample) in enumerate(cases, start=1):
        if (stage, sample.sample_id) in completed_keys:
            continue
        sample_started = time.perf_counter()
        stage_config = protocol["stages"][stage]
        prompt = tokenize_sample_prompt(sample, tokenizer)
        with torch.no_grad():
            context_outputs = model.model(
                prompt.context_ids.to(model.device), use_cache=True, return_dict=True
            )
        segments = build_segment_catalog(
            context_outputs.past_key_values,
            attention_layers,
            context_tokens=prompt.context_tokens,
            config=OracleConfig(
                segment_length=stage_config["segment_length"],
                middle_kv_fraction=method["middle_kv_fraction"],
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
        allocator_attention, allocator_accessibility = restrict_eligible_signals(
            attention, accessibility, segments
        )
        raw_selection = select_equal_byte_segments(
            attention,
            accessibility,
            segments,
            middle_kv_fraction=method["middle_kv_fraction"],
        )
        exact_plan = allocate_coverage_fidelity(
            allocator_attention,
            allocator_accessibility,
            segments,
            middle_kv_fraction=method["middle_kv_fraction"],
            sparse_width=method["sparse_width"],
            use_accessibility=False,
            enable_exact_upgrades=True,
        )
        sparse_plan = allocate_coverage_fidelity(
            allocator_attention,
            allocator_accessibility,
            segments,
            middle_kv_fraction=method["middle_kv_fraction"],
            sparse_width=method["sparse_width"],
            use_accessibility=False,
            enable_exact_upgrades=False,
        )
        if exact_plan.total_charged_bytes != sparse_plan.total_charged_bytes:
            raise OracleContractError("confirmation compressed plans are not equal-byte")

        position_plans = {
            "contiguous_cf": build_retained_position_plan(
                exact_plan,
                segments,
                probe.token_attention_mass,
                context_tokens=prompt.context_tokens,
                sparse_selector="max_mass_window",
            ),
            "scattered_cf": build_retained_position_plan(
                exact_plan,
                segments,
                probe.token_attention_mass,
                context_tokens=prompt.context_tokens,
                sparse_selector="top_tokens",
            ),
            "contiguous_sparse_only": build_retained_position_plan(
                sparse_plan,
                segments,
                probe.token_attention_mass,
                context_tokens=prompt.context_tokens,
                sparse_selector="max_mass_window",
            ),
        }
        del probe
        _cleanup_cuda()

        arm_specs = [
            (
                name,
                make_coverage_fidelity_intervention(
                    position_plans[name], attention_layers, name=name
                ),
            )
            for name in COMPRESSED_SYSTEMS
        ]
        arm_specs.extend(
            [
                (
                    "raw_alpha_exact_topk",
                    make_selected_segment_intervention(
                        segments,
                        attention_layers,
                        raw_selection["raw_alpha_segment_ids"],
                        context_tokens=prompt.context_tokens,
                        name="raw_alpha_exact_topk",
                    ),
                ),
                ("full_kv_reference", full_kv_intervention),
            ]
        )
        generated = {}
        for system, intervention in arm_specs:
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
                max_new_tokens=protocol["max_new_tokens"],
            )
            generated[system] = {
                **score_generated_text(answer.text, sample),
                "generated_text": answer.text,
                "generated_token_ids": [
                    int(value) for value in answer.token_ids[0].tolist()
                ],
                "post_query_resident_kv_bytes": int(resident_bytes),
            }
            del state
            _cleanup_cuda()
        if len(
            {
                generated[name]["post_query_resident_kv_bytes"]
                for name in COMPRESSED_SYSTEMS
            }
        ) != 1:
            raise OracleContractError("confirmation arms are not equal resident-byte")

        eligible = [segment for segment in segments if segment.eligible]
        token_unit_bytes = eligible[0].kv_bytes // eligible[0].token_count
        query_kv_bytes = prompt.query_tokens * token_unit_bytes
        protected_bytes = sum(segment.kv_bytes for segment in segments if segment.protected)
        compressed_cap = exact_plan.total_budget_limit_bytes + query_kv_bytes
        plan_by_system = {
            "contiguous_cf": exact_plan,
            "scattered_cf": exact_plan,
            "contiguous_sparse_only": sparse_plan,
        }
        for name, plan in plan_by_system.items():
            expected = plan.total_charged_bytes + query_kv_bytes
            if generated[name]["post_query_resident_kv_bytes"] != expected:
                raise OracleContractError(
                    f"{name} post-query bytes disagree with its allocation plan"
                )
            generated[name].update(
                {
                    "post_query_budget_limit_bytes": compressed_cap,
                    "expected_post_query_resident_kv_bytes": expected,
                }
            )
        raw_expected = (
            protected_bytes
            + raw_selection["budget_slots"] * raw_selection["unit_segment_bytes"]
            + query_kv_bytes
        )
        full_expected = sum(segment.kv_bytes for segment in segments) + query_kv_bytes
        if (
            generated["raw_alpha_exact_topk"]["post_query_resident_kv_bytes"]
            != raw_expected
            or generated["full_kv_reference"]["post_query_resident_kv_bytes"]
            != full_expected
        ):
            raise OracleContractError("confirmation baseline byte accounting failed")
        generated["raw_alpha_exact_topk"].update(
            {
                "post_query_budget_limit_bytes": compressed_cap,
                "expected_post_query_resident_kv_bytes": raw_expected,
            }
        )
        generated["full_kv_reference"].update(
            {
                "post_query_budget_limit_bytes": full_expected,
                "expected_post_query_resident_kv_bytes": full_expected,
            }
        )

        protected_ids = {segment.segment_id for segment in segments if segment.protected}
        plans = {
            name: {
                "allocation": plan_by_system[name].to_dict(),
                "retention": position_plans[name].to_dict(),
                "action_counts": _eligible_action_counts(
                    plan_by_system[name], protected_ids
                ),
            }
            for name in COMPRESSED_SYSTEMS
        }
        row = {
            "stage": stage,
            "sample_id": sample.sample_id,
            "dataset": sample.dataset,
            "answer": sample.answer,
            "context_tokens": prompt.context_tokens,
            "query_tokens": prompt.query_tokens,
            "raw_alpha_selection": raw_selection,
            "byte_accounting": {
                "token_kv_bytes": token_unit_bytes,
                "query_kv_bytes": query_kv_bytes,
                "protected_context_kv_bytes": protected_bytes,
                "compressed_post_query_cap_bytes": compressed_cap,
            },
            "plans": plans,
            "systems": generated,
            "elapsed_seconds": time.perf_counter() - sample_started,
        }
        rows.append(row)
        _append_jsonl(results_path, row)
        print(
            f"[{case_index}/{len(cases)}] {stage} {sample.dataset} {sample.sample_id}: "
            f"contiguous={generated['contiguous_cf']['normalized_answer_contains']:.0f} "
            f"scattered={generated['scattered_cf']['normalized_answer_contains']:.0f} "
            f"sparse={generated['contiguous_sparse_only']['normalized_answer_contains']:.0f} "
            f"raw={generated['raw_alpha_exact_topk']['normalized_answer_contains']:.0f} "
            f"full={generated['full_kv_reference']['normalized_answer_contains']:.0f}",
            flush=True,
        )

    expected_keys = {(stage, sample.sample_id) for stage, sample in cases}
    if {(row["stage"], row["sample_id"]) for row in rows} != expected_keys:
        raise OracleContractError("confirmation did not complete the frozen sample set")
    rows.sort(key=lambda row: (row["stage"], row["dataset"], row["sample_id"]))
    summary = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete",
        "scope": "fresh_confirmation_no_postselection",
        "manifest_id": manifest["manifest_id"],
        "protocol_sha256": protocol_sha,
        "analysis": summarize_confirmation_results(rows),
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
