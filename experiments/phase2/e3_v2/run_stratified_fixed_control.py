"""Frozen 16K/10% control isolating free-start window placement."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from experiments.phase2.e3_v2.context_query import tokenize_sample_prompt
from experiments.phase2.e3_v2.coverage_fidelity import (
    ALLOCATION_SCHEMA,
    allocate_coverage_fidelity,
)
from experiments.phase2.e3_v2.coverage_fidelity_cache import (
    STRATIFIED_FIXED_CHUNK_SELECTOR,
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
    _pair_summary,
    restrict_eligible_signals,
)
from experiments.phase2.e3_v2.run_discovery import _build_samples
from experiments.phase2.e3_v2.run_pareto import (
    PROJECT_ROOT,
    _generate_system,
    _load_completed as _load_parent_results,
)
from experiments.utils.model_loader import (
    get_full_attention_indices,
    get_linear_attention_indices,
)
from experiments.utils.run_manifest import collect_environment, ensure_run_manifest


PROTOCOL_SCHEMA = "hmo.stratified_fixed_chunk.control_protocol.v1"
RESULT_SCHEMA = "hmo.stratified_fixed_chunk.control_result.v1"
RESULTS_FILENAME = "stratified_fixed_results.jsonl"
SUMMARY_FILENAME = "stratified_fixed_summary.json"
SYSTEMS = ("contiguous_cf_parent", "stratified_fixed_chunk")
BUDGET_FRACTION = 0.1
SOURCE_STAGE = "16k"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_legacy_parent_protocol(path: Path) -> tuple[dict, str]:
    try:
        encoded = path.read_bytes()
        payload = json.loads(encoded)
    except (OSError, json.JSONDecodeError) as exc:
        raise OracleContractError("cannot read legacy Pareto parent") from exc
    if (
        payload.get("schema_version") != "hmo.contiguous_cf.pareto_protocol.v1"
        or "scattered_cf" not in payload.get("systems", ())
        or set(payload.get("stages", {})) != {"smoke", "8k", "16k"}
    ):
        raise OracleContractError("legacy Pareto parent contract mismatch")
    return payload, hashlib.sha256(encoded).hexdigest()


def load_control_protocol(
    path: Path,
    *,
    parent_protocol_sha256: str,
    parent_results_sha256: str,
) -> tuple[dict, str]:
    try:
        encoded = path.read_bytes()
        payload = json.loads(encoded)
    except (OSError, json.JSONDecodeError) as exc:
        raise OracleContractError(f"cannot read control protocol: {path}") from exc

    parent = payload.get("parent", {})
    expected_method = {
        "allocator": "reuse_contiguous_cf_parent",
        "exact_upgrades": "unchanged",
        "sparse_retained_tokens": "unchanged_including_slack",
        "window_start_alignment": 16,
        "alignment_origin": "segment_start",
        "aligned_window_score": "sum_query_attention_mass",
        "aligned_window_tie_break": "earliest_start",
        "protected_segments": "unchanged",
    }
    expected_execution = {
        "smoke_sample_cases": 1,
        "formal_sample_cases": 24,
        "reuse_parent_hmo_generation": True,
        "recompute_and_verify_parent_hmo_plan": True,
        "candidate_search": False,
        "continuation_gate": False,
    }
    if (
        payload.get("schema_version") != PROTOCOL_SCHEMA
        or payload.get("status") != "frozen_before_outcomes"
        or payload.get("model_id") != "Qwen/Qwen3.5-0.8B"
        or not payload.get("model_revision")
        or tuple(payload.get("systems", ())) != SYSTEMS
        or payload.get("primary_comparison")
        != ["stratified_fixed_chunk", "contiguous_cf_parent"]
        or payload.get("primary_metric") != "normalized_answer_contains"
        or int(payload.get("max_new_tokens", 0)) != 32
        or int(payload.get("inference_seed", 0)) != 20261009
        or payload.get("method") != expected_method
        or payload.get("execution") != expected_execution
        or parent.get("protocol_sha256") != parent_protocol_sha256
        or parent.get("results_sha256") != parent_results_sha256
        or parent.get("source_stage") != SOURCE_STAGE
        or float(parent.get("budget_fraction", -1)) != BUDGET_FRACTION
        or int(parent.get("expected_sample_cases", 0)) != 24
    ):
        raise OracleContractError("stratified fixed-chunk protocol mismatch")
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
                    f"invalid control result at line {line_number}"
                ) from exc
    ids = [row.get("sample_id") for row in rows]
    if len(ids) != len(set(ids)):
        raise OracleContractError("control results contain duplicate samples")
    return rows


def _jaccard(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 1.0


def summarize_control_results(rows: list[dict]) -> dict:
    if not rows:
        raise OracleContractError("cannot summarize empty control results")
    by_dataset = {}
    for dataset in sorted({row["dataset"] for row in rows}):
        selected = [row for row in rows if row["dataset"] == dataset]
        by_dataset[dataset] = {
            "sample_cases": len(selected),
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
    return {
        "sample_case_count": len(rows),
        "systems": {
            system: {
                metric: float(
                    np.mean([row["systems"][system][metric] for row in rows])
                )
                for metric in METRICS
            }
            for system in SYSTEMS
        },
        "comparison": _pair_summary(
            rows, "stratified_fixed_chunk", "contiguous_cf_parent"
        ),
        "by_dataset": by_dataset,
        "equal_resident_byte_cases": sum(
            row["systems"][SYSTEMS[0]]["post_query_resident_kv_bytes"]
            == row["systems"][SYSTEMS[1]]["post_query_resident_kv_bytes"]
            for row in rows
        ),
        "mean_retained_position_jaccard": float(
            np.mean([row["geometry"]["retained_position_jaccard"] for row in rows])
        ),
        "identical_generated_token_cases": sum(
            row["systems"][SYSTEMS[0]]["generated_token_ids"]
            == row["systems"][SYSTEMS[1]]["generated_token_ids"]
            for row in rows
        ),
    }


def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("control evaluation requires exactly one visible GPU")

    parent_protocol_path = Path(args.parent_protocol).resolve()
    parent_results_path = Path(args.parent_results).resolve()
    parent_protocol, parent_protocol_sha = _load_legacy_parent_protocol(
        parent_protocol_path
    )
    parent_results_sha = _sha256(parent_results_path)
    protocol, protocol_sha = load_control_protocol(
        Path(args.protocol).resolve(),
        parent_protocol_sha256=parent_protocol_sha,
        parent_results_sha256=parent_results_sha,
    )
    if (
        args.model_id != protocol["model_id"]
        or args.model_revision != protocol["model_revision"]
    ):
        raise OracleContractError("model identity disagrees with control protocol")

    parent_rows = [
        row
        for row in _load_parent_results(parent_results_path)
        if row["stage"] == SOURCE_STAGE
        and float(row["budget_fraction"]) == BUDGET_FRACTION
    ]
    if len(parent_rows) != protocol["parent"]["expected_sample_cases"]:
        raise OracleContractError("parent Pareto result set is incomplete")
    parent_by_id = {row["sample_id"]: row for row in parent_rows}
    if len(parent_by_id) != len(parent_rows):
        raise OracleContractError("parent Pareto sample ids are not unique")

    run_dir = Path(args.run_dir).resolve()
    model_path = Path(args.model_path).resolve()
    model_identity = model_provenance(
        model_path, args.model_id, revision=args.model_revision
    )
    manifest = ensure_run_manifest(
        run_dir,
        experiment="stratified_fixed_chunk_control",
        args={
            "model_id": args.model_id,
            "model_revision": args.model_revision,
            "protocol_sha256": protocol_sha,
            "parent_protocol_sha256": parent_protocol_sha,
            "parent_results_sha256": parent_results_sha,
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
                else "frozen_16k_free_start_mechanism_control"
            ),
            "systems": list(SYSTEMS),
            "budget_fraction": BUDGET_FRACTION,
            "allocator_schema": ALLOCATION_SCHEMA,
            "sparse_selector": STRATIFIED_FIXED_CHUNK_SELECTOR,
            "method": protocol["method"],
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
        raise OracleContractError("control results exist; pass --resume to continue")
    completed_ids = {row["sample_id"] for row in completed}
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
        raise OracleContractError("control model is not a Hybrid architecture")
    _force_torch_reference_backend(model, recurrent_layers)

    stage_config = parent_protocol["stages"][SOURCE_STAGE]
    samples = _build_samples(tokenizer, Namespace(**stage_config))
    if len(samples) != protocol["execution"]["formal_sample_cases"]:
        raise OracleContractError("control sample construction changed")
    if set(sample.sample_id for sample in samples) != set(parent_by_id):
        raise OracleContractError("control samples disagree with parent Pareto rows")
    if args.stage_set == "smoke":
        samples = [next(sample for sample in samples if sample.dataset == "needle")]

    for case_index, sample in enumerate(samples, start=1):
        if sample.sample_id in completed_ids:
            continue
        parent = parent_by_id[sample.sample_id]
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
                middle_kv_fraction=BUDGET_FRACTION,
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

        allocation = allocate_coverage_fidelity(
            allocator_attention,
            allocator_accessibility,
            segments,
            middle_kv_fraction=BUDGET_FRACTION,
            sparse_width=parent_protocol["method"]["sparse_width"],
            use_accessibility=False,
            enable_exact_upgrades=True,
        )
        hmo_positions = build_retained_position_plan(
            allocation,
            segments,
            token_attention_mass,
            context_tokens=prompt.context_tokens,
            sparse_selector="max_mass_window",
        )
        if (
            parent["plans"]["contiguous_cf"]["allocation"]
            != allocation.to_dict()
            or parent["plans"]["contiguous_cf"]["retention"]
            != hmo_positions.to_dict()
            or parent["context_tokens"] != prompt.context_tokens
            or parent["query_tokens"] != prompt.query_tokens
            or parent["answer"] != sample.answer
        ):
            raise OracleContractError(
                "recomputed HMO plan disagrees with the SHA-pinned parent row"
            )

        control_positions = build_retained_position_plan(
            allocation,
            segments,
            token_attention_mass,
            context_tokens=prompt.context_tokens,
            sparse_selector=STRATIFIED_FIXED_CHUNK_SELECTOR,
            sparse_alignment=protocol["method"]["window_start_alignment"],
        )
        control = _generate_system(
            model,
            tokenizer,
            prompt,
            attention_layers,
            recurrent_layers,
            make_coverage_fidelity_intervention(
                control_positions,
                attention_layers,
                name="stratified_fixed_chunk",
            ),
            sample,
            protocol["max_new_tokens"],
        )
        parent_hmo = dict(parent["systems"]["contiguous_cf"])
        expected_resident = parent_hmo["post_query_resident_kv_bytes"]
        if (
            control["post_query_resident_kv_bytes"] != expected_resident
            or control_positions.context_charged_bytes
            != hmo_positions.context_charged_bytes
        ):
            raise OracleContractError("control and parent HMO are not equal-byte")

        hmo_segments = {item.segment_id: item for item in hmo_positions.segments}
        control_segments = {
            item.segment_id: item for item in control_positions.segments
        }
        sparse_ids = [
            item.segment_id
            for item in allocation.allocations
            if item.action == "sparse"
        ]
        changed_sparse = sum(
            hmo_segments[segment_id].positions
            != control_segments[segment_id].positions
            for segment_id in sparse_ids
        )
        row = {
            "stage": args.stage_set,
            "source_stage": SOURCE_STAGE,
            "sample_id": sample.sample_id,
            "dataset": sample.dataset,
            "answer": sample.answer,
            "context_tokens": prompt.context_tokens,
            "query_tokens": prompt.query_tokens,
            "budget_fraction": BUDGET_FRACTION,
            "plans": {
                "allocation": allocation.to_dict(),
                "contiguous_cf_parent": hmo_positions.to_dict(),
                "stratified_fixed_chunk": control_positions.to_dict(),
            },
            "geometry": {
                "sparse_segment_count": len(sparse_ids),
                "changed_sparse_segment_count": changed_sparse,
                "retained_position_jaccard": _jaccard(
                    hmo_positions.active_positions,
                    control_positions.active_positions,
                ),
            },
            "systems": {
                "contiguous_cf_parent": parent_hmo,
                "stratified_fixed_chunk": control,
            },
            "integrity": {
                "parent_protocol_sha256": parent_protocol_sha,
                "parent_results_sha256": parent_results_sha,
                "parent_hmo_plan_exact_match": True,
                "equal_resident_bytes": True,
            },
        }
        rows.append(row)
        _append_jsonl(results_path, row)
        print(
            f"[{case_index}/{len(samples)}] {sample.dataset} {sample.sample_id}: "
            f"hmo={parent_hmo['normalized_answer_contains']:.0f} "
            f"aligned={control['normalized_answer_contains']:.0f} "
            f"changed_sparse={changed_sparse}/{len(sparse_ids)}",
            flush=True,
        )

    expected_count = protocol["execution"][f"{args.stage_set}_sample_cases"]
    if len(rows) != expected_count:
        raise OracleContractError("control run did not complete the selected package")
    rows.sort(key=lambda row: (row["dataset"], row["sample_id"]))
    summary = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete",
        "scope": (
            "operational_smoke_excluded_from_claims"
            if args.stage_set == "smoke"
            else "frozen_16k_free_start_mechanism_control"
        ),
        "manifest_id": manifest["manifest_id"],
        "protocol_sha256": protocol_sha,
        "parent_protocol_sha256": parent_protocol_sha,
        "parent_results_sha256": parent_results_sha,
        "analysis": summarize_control_results(rows),
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
    parser.add_argument("--parent-protocol", required=True)
    parser.add_argument("--parent-results", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--stage-set", choices=("smoke", "formal"), default="formal")
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
