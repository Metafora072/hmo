"""Development end-task run for the CF-HMO coverage-fidelity allocator."""
from __future__ import annotations

import argparse
import gc
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
    CoverageFidelityPlan,
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
from experiments.phase2.e3_v2.run_cf_diagnosis import (
    _assert_selection_reproduced,
    load_changed_cases,
)
from experiments.phase2.e3_v2.run_discovery import _build_samples
from experiments.phase2.e3_v2.run_end_task import (
    _load_protocol,
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
RESULTS_FILENAME = "coverage_fidelity_results.jsonl"
SUMMARY_FILENAME = "coverage_fidelity_summary.json"
SCHEMA_VERSION = "hmo.coverage_fidelity.development_result.v1"
SYSTEMS = (
    "cf_hmo",
    "cf_hmo_no_access",
    "sparse_only",
    "raw_alpha_exact_topk",
    "full_kv_reference",
)
METRICS = ("normalized_answer_contains", "normalized_exact_match", "token_f1")


def _atomic_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
                    f"invalid persisted D1b result at line {line_number}"
                ) from exc
    keys = [(row.get("stage"), row.get("sample_id")) for row in rows]
    if len(keys) != len(set(keys)):
        raise OracleContractError("persisted D1b results contain duplicate cases")
    return rows


def _eligible_action_counts(plan: CoverageFidelityPlan, protected_ids: set[int]) -> dict:
    actions = [
        item.action for item in plan.allocations if item.segment_id not in protected_ids
    ]
    return {
        action: sum(value == action for value in actions)
        for action in ("recurrent_only", "sparse", "exact")
    }


def _pair_summary(rows: Sequence[Mapping], left: str, right: str) -> dict:
    output = {}
    for metric in METRICS:
        deltas = [
            float(row["systems"][left][metric] - row["systems"][right][metric])
            for row in rows
        ]
        output[metric] = {
            "mean_delta": float(np.mean(deltas)),
            "wins": sum(value > 1e-12 for value in deltas),
            "ties": sum(abs(value) <= 1e-12 for value in deltas),
            "losses": sum(value < -1e-12 for value in deltas),
            "stage_mean_delta": {
                stage: float(
                    np.mean(
                        [
                            delta
                            for row, delta in zip(rows, deltas)
                            if row["stage"] == stage
                        ]
                    )
                )
                for stage in sorted({row["stage"] for row in rows})
            },
        }
    return output


def summarize_results(rows: Sequence[Mapping]) -> dict:
    if not rows:
        raise OracleContractError("D1b summary requires result rows")
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
        "comparisons": {
            "cf_hmo_vs_no_access": _pair_summary(
                rows, "cf_hmo", "cf_hmo_no_access"
            ),
            "cf_hmo_vs_sparse_only": _pair_summary(rows, "cf_hmo", "sparse_only"),
            "cf_hmo_vs_raw_alpha_exact_topk": _pair_summary(
                rows, "cf_hmo", "raw_alpha_exact_topk"
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
        "mean_eligible_action_counts": {
            system: {
                action: float(
                    np.mean([row["plans"][system]["action_counts"][action] for row in rows])
                )
                for action in ("recurrent_only", "sparse", "exact")
            }
            for system in ("cf_hmo", "cf_hmo_no_access", "sparse_only")
        },
        "p3_baseline_reproduced_cases": sum(
            row["baseline_reproduction"]["raw_alpha"]
            and row["baseline_reproduction"]["full_kv_reference"]
            for row in rows
        ),
    }


def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("D1b development requires exactly one visible GPU")
    if not 0.0 < args.middle_kv_fraction < 1.0 or args.sparse_width <= 0:
        raise OracleContractError("D1b budget and Sparse width are invalid")
    protocol, protocol_sha = _load_protocol(Path(args.protocol).resolve())
    if (
        args.model_id != protocol["model_id"]
        or args.model_revision != protocol["model_revision"]
    ):
        raise OracleContractError("D1b model identity disagrees with P3 protocol")
    cases, sources = load_changed_cases(
        [Path(value) for value in args.source_result],
        protocol=protocol,
        protocol_sha256=protocol_sha,
        expected_count=args.expected_changed_cases,
    )
    run_dir = Path(args.run_dir).resolve()
    model_path = Path(args.model_path).resolve()
    model_identity = model_provenance(
        model_path, args.model_id, revision=args.model_revision
    )
    scientific_args = {
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "middle_kv_fraction": args.middle_kv_fraction,
        "sparse_width": args.sparse_width,
        "expected_changed_cases": args.expected_changed_cases,
        "max_new_tokens": protocol["max_new_tokens"],
        "recurrent_backend": REFERENCE_BACKEND,
        "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    manifest = ensure_run_manifest(
        run_dir,
        experiment="coverage_fidelity_d1b_development",
        args=scientific_args,
        selections={
            "scope": "development_only_frozen_p3_membership_changed_cases",
            "source_runs": sources,
            "protocol_sha256": protocol_sha,
            "systems": list(SYSTEMS),
            "allocator_schema": ALLOCATION_SCHEMA,
            "coverage_priority": "rank01(query_attention)/sparse_bytes",
            "fidelity_priority": (
                "rank01(query_attention)*(1-rank01(query_read_share))"
                "/incremental_exact_bytes"
            ),
            "sparse_selector": "stable_token_query_attention",
            "candidate_search": False,
        },
        model=model_identity,
        project_root=PROJECT_ROOT,
        require_clean=True,
        environment=collect_environment(),
    )

    results_path = run_dir / RESULTS_FILENAME
    completed = _load_completed(results_path)
    if completed and not args.resume:
        raise OracleContractError("D1b results exist; pass --resume to continue")
    completed_keys = {(row["stage"], row["sample_id"]) for row in completed}
    rows = list(completed)

    torch.manual_seed(20261004)
    torch.cuda.manual_seed_all(20261004)
    torch.set_float32_matmul_precision("highest")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model, tokenizer = _load_model(model_path)
    attention_layers = tuple(get_full_attention_indices(model.config))
    recurrent_layers = tuple(get_linear_attention_indices(model.config))
    _force_torch_reference_backend(model, recurrent_layers)
    samples_by_stage = {
        stage: {
            sample.sample_id: sample
            for sample in _build_samples(tokenizer, Namespace(**protocol["stages"][stage]))
        }
        for stage in ("8k", "16k")
    }

    for case_index, case in enumerate(cases, start=1):
        stage = case["stage"]
        source_row = case["source_row"]
        sample_id = source_row["sample_id"]
        if (stage, sample_id) in completed_keys:
            continue
        sample_started = time.perf_counter()
        sample = samples_by_stage[stage].get(sample_id)
        if sample is None or sample.answer != source_row["answer"]:
            raise OracleContractError(f"cannot reconstruct frozen P3 sample {sample_id}")
        prompt = tokenize_sample_prompt(sample, tokenizer)
        if (
            prompt.context_tokens != source_row["context_tokens"]
            or prompt.query_tokens != source_row["query_tokens"]
        ):
            raise OracleContractError(f"reconstructed prompt changed for {sample_id}")
        segment_length = protocol["stages"][stage]["segment_length"]
        with torch.no_grad():
            context_outputs = model.model(
                prompt.context_ids.to(model.device), use_cache=True, return_dict=True
            )
        segments = build_segment_catalog(
            context_outputs.past_key_values,
            attention_layers,
            context_tokens=prompt.context_tokens,
            config=OracleConfig(
                segment_length=segment_length,
                middle_kv_fraction=args.middle_kv_fraction,
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
            segment_length=segment_length,
        )
        attention = probe.alpha.as_dict()
        accessibility = probe.accessibility.field_dict("read_share")
        historical_selection = select_equal_byte_segments(
            attention,
            accessibility,
            segments,
            middle_kv_fraction=args.middle_kv_fraction,
        )
        if abs(args.middle_kv_fraction - protocol["stages"][stage]["middle_kv_fraction"]) < 1e-12:
            _assert_selection_reproduced(
                historical_selection, source_row["selection"]
            )

        plan_specs = {
            "cf_hmo": dict(use_accessibility=True, enable_exact_upgrades=True),
            "cf_hmo_no_access": dict(
                use_accessibility=False, enable_exact_upgrades=True
            ),
            "sparse_only": dict(
                use_accessibility=False, enable_exact_upgrades=False
            ),
        }
        plans = {
            name: allocate_coverage_fidelity(
                attention,
                accessibility,
                segments,
                middle_kv_fraction=args.middle_kv_fraction,
                sparse_width=args.sparse_width,
                **options,
            )
            for name, options in plan_specs.items()
        }
        charged = {plan.total_charged_bytes for plan in plans.values()}
        if len(charged) != 1:
            raise OracleContractError("D1b compressed arms do not have equal planned bytes")
        covered = {
            name: {
                item.segment_id
                for item in plan.allocations
                if item.action != "recurrent_only"
            }
            for name, plan in plans.items()
        }
        if covered["cf_hmo"] != covered["cf_hmo_no_access"]:
            raise OracleContractError("no-access changed the Sparse coverage set")

        position_plans = {
            name: build_retained_position_plan(
                plan,
                segments,
                probe.token_attention_mass,
                context_tokens=prompt.context_tokens,
            )
            for name, plan in plans.items()
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
            for name in ("cf_hmo", "cf_hmo_no_access", "sparse_only")
        ]
        arm_specs.extend(
            [
                (
                    "raw_alpha_exact_topk",
                    make_selected_segment_intervention(
                        segments,
                        attention_layers,
                        historical_selection["raw_alpha_segment_ids"],
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
        compressed_bytes = {
            generated[name]["post_query_resident_kv_bytes"]
            for name in ("cf_hmo", "cf_hmo_no_access", "sparse_only")
        }
        if len(compressed_bytes) != 1:
            raise OracleContractError("D1b compressed arms are not equal resident-byte")

        raw_matches_source = (
            generated["raw_alpha_exact_topk"]["generated_token_ids"]
            == source_row["systems"]["raw_alpha"]["generated_token_ids"]
        )
        full_reproduced = (
            generated["full_kv_reference"]["generated_token_ids"]
            == source_row["systems"]["full_kv_reference"]["generated_token_ids"]
        )
        source_budget = protocol["stages"][stage]["middle_kv_fraction"]
        raw_reproduction_required = (
            abs(args.middle_kv_fraction - source_budget) < 1e-12
        )
        if (raw_reproduction_required and not raw_matches_source) or not full_reproduced:
            raise OracleContractError("D1b failed to reproduce a frozen P3 baseline")

        eligible = [segment for segment in segments if segment.eligible]
        token_unit_bytes = eligible[0].kv_bytes // eligible[0].token_count
        query_kv_bytes = prompt.query_tokens * token_unit_bytes
        protected_bytes = sum(segment.kv_bytes for segment in segments if segment.protected)
        compressed_cap = plans["cf_hmo"].total_budget_limit_bytes + query_kv_bytes
        for name, plan in plans.items():
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
            + historical_selection["budget_slots"]
            * historical_selection["unit_segment_bytes"]
            + query_kv_bytes
        )
        full_expected = sum(segment.kv_bytes for segment in segments) + query_kv_bytes
        if (
            generated["raw_alpha_exact_topk"]["post_query_resident_kv_bytes"]
            != raw_expected
            or generated["full_kv_reference"]["post_query_resident_kv_bytes"]
            != full_expected
        ):
            raise OracleContractError("D1b baseline resident-byte accounting failed")
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
        plan_records = {
            name: {
                "allocation": plan.to_dict(),
                "retention": position_plans[name].to_dict(),
                "action_counts": _eligible_action_counts(plan, protected_ids),
            }
            for name, plan in plans.items()
        }
        row = {
            "stage": stage,
            "sample_id": sample_id,
            "dataset": sample.dataset,
            "answer": sample.answer,
            "context_tokens": prompt.context_tokens,
            "query_tokens": prompt.query_tokens,
            "historical_selection": historical_selection,
            "source_outcome": {
                "raw_alpha": source_row["systems"]["raw_alpha"]
                ["normalized_answer_contains"],
                "frozen_v2": source_row["systems"]["frozen_v2"]
                ["normalized_answer_contains"],
            },
            "baseline_reproduction": {
                "raw_alpha": (
                    raw_matches_source if raw_reproduction_required else None
                ),
                "full_kv_reference": full_reproduced,
            },
            "byte_accounting": {
                "token_kv_bytes": token_unit_bytes,
                "query_kv_bytes": query_kv_bytes,
                "protected_context_kv_bytes": protected_bytes,
                "compressed_post_query_cap_bytes": compressed_cap,
            },
            "plans": plan_records,
            "systems": generated,
            "elapsed_seconds": time.perf_counter() - sample_started,
        }
        rows.append(row)
        _append_jsonl(results_path, row)
        print(
            f"[{case_index}/{len(cases)}] {stage} {sample_id}: "
            f"cf={generated['cf_hmo']['normalized_answer_contains']:.0f} "
            f"no_access={generated['cf_hmo_no_access']['normalized_answer_contains']:.0f} "
            f"sparse={generated['sparse_only']['normalized_answer_contains']:.0f} "
            f"raw={generated['raw_alpha_exact_topk']['normalized_answer_contains']:.0f}",
            flush=True,
        )

    expected_keys = {(case["stage"], case["source_row"]["sample_id"]) for case in cases}
    actual_keys = {(row["stage"], row["sample_id"]) for row in rows}
    if actual_keys != expected_keys:
        raise OracleContractError("D1b run did not complete the development set")
    rows.sort(key=lambda row: (row["stage"], row["sample_id"]))
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "scope": "development_only_not_fresh_confirmation",
        "manifest_id": manifest["manifest_id"],
        "protocol_sha256": protocol_sha,
        "source_runs": sources,
        "middle_kv_fraction": args.middle_kv_fraction,
        "sparse_width": args.sparse_width,
        "analysis": summarize_results(rows),
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
    parser.add_argument("--source-result", action="append", required=True)
    parser.add_argument("--middle-kv-fraction", type=float, default=0.10)
    parser.add_argument("--sparse-width", type=int, default=16)
    parser.add_argument("--expected-changed-cases", type=int, default=10)
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
