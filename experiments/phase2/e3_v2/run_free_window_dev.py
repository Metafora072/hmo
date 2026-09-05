"""Run frozen HMO layer-local and free-window development arms on one GPU."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import torch

from experiments.phase2.e3_v2.analyze_free_window_offline import (
    _legacy_plan,
    _load_layer_scores,
)
from experiments.phase2.e3_v2.chunkkv_adapter import build_chunkkv_plan
from experiments.phase2.e3_v2.free_window_allocator import (
    FREE_WINDOW_SCHEMA,
    LAYER_LOCAL_SCHEMA,
    build_free_window_plan,
    build_layer_local_hmo_plan,
    make_layerwise_window_intervention,
)
from experiments.phase2.e3_v2.freeze_free_window_dev_protocol import (
    DATASET_ORDER,
    PROTOCOL_SCHEMA,
)
from experiments.phase2.e3_v2.oracle import OracleConfig, OracleContractError, build_segment_catalog
from experiments.phase2.e3_v2.real_model_preflight import (
    REFERENCE_BACKEND,
    _force_torch_reference_backend,
    _load_model,
    model_provenance,
)
from experiments.phase2.e3_v2.run_coverage_fidelity import (
    _append_jsonl,
    _atomic_json,
    _cleanup_cuda,
)
from experiments.phase2.e3_v2.run_hotpot_paired import METRICS, _generate_system, _load_completed
from experiments.phase2.e3_v2.run_native_tasks import (
    _load_datasets,
    _make_sample,
    load_native_protocol,
)
from experiments.phase2.e3_v2.run_hotpot_solvability import _sha256_file
from experiments.phase2.e3_v2.context_query import tokenize_sample_prompt_aligned
from experiments.utils.eval_harness import get_ground_truths
from experiments.utils.model_loader import get_full_attention_indices, get_linear_attention_indices
from experiments.utils.run_manifest import collect_environment, ensure_run_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULT_SCHEMA = "hmo.free_window_dev_result.v1"
RESULTS_FILENAME = "free_window_dev_results.jsonl"
SUMMARY_FILENAME = "free_window_dev_summary.json"
SYSTEMS = (
    "hmo_legacy",
    "hmo_layer_local",
    "chunkkv",
    "hmo_free_window",
    "full_kv_reference",
)
EQUAL_BYTE_SYSTEMS = SYSTEMS[:-1]
GENERATED_SYSTEMS = ("hmo_layer_local", "hmo_free_window")
REUSED_SYSTEM_MAP = {
    "hmo_legacy": "contiguous_cf",
    "chunkkv": "chunkkv",
    "full_kv_reference": "full_kv_reference",
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_protocol(path: Path) -> tuple[dict, str]:
    encoded = path.read_bytes()
    payload = json.loads(encoded)
    cases = payload.get("cases", [])
    selection = payload.get("selection", {})
    if (
        payload.get("schema_version") != PROTOCOL_SCHEMA
        or payload.get("status") != "frozen_before_new_method_qa_outcomes"
        or payload.get("purpose")
        != "development_only_task_and_context_length_stratified_validation"
        or tuple(payload.get("systems", ())) != SYSTEMS
        or tuple(payload.get("generated_systems", ())) != GENERATED_SYSTEMS
        or tuple(payload.get("reused_systems", ())) != tuple(REUSED_SYSTEM_MAP)
        or payload.get("primary_metric") != "official_qa_f1"
        or float(payload.get("middle_kv_fraction", 0.0)) != 0.1
        or selection.get("uses_qa_outcomes") is not False
        or selection.get("uses_proxy_outcomes") is not False
        or selection.get("development_only") is not True
        or tuple(selection.get("dataset_order", ())) != DATASET_ORDER
        or int(payload.get("case_count", -1)) != len(cases)
        or len({case.get("sample_id") for case in cases}) != len(cases)
        or len({case.get("probe_id") for case in cases}) != len(cases)
    ):
        raise OracleContractError("free-window development protocol mismatch")
    expected_per_dataset = int(selection["per_dataset"])
    expected_per_stratum = int(selection["per_stratum"])
    strata = int(selection["length_strata"])
    for dataset in DATASET_ORDER:
        members = [case for case in cases if case.get("dataset") == dataset]
        if len(members) != expected_per_dataset:
            raise OracleContractError(f"free-window case count changed: {dataset}")
        for stratum in range(strata):
            if sum(case.get("length_stratum") == stratum for case in members) != expected_per_stratum:
                raise OracleContractError(f"free-window length stratum changed: {dataset}")
    return payload, _sha256_bytes(encoded)


def _load_source_rows(path: Path, expected_sha: str, expected_count: int) -> dict[str, dict]:
    if _sha256_file(path) != expected_sha:
        raise OracleContractError("source result SHA disagrees with frozen protocol")
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row["sample_id"])
            if sample_id in rows:
                raise OracleContractError("source results contain duplicate sample IDs")
            rows[sample_id] = row
    if len(rows) != expected_count:
        raise OracleContractError("source result count disagrees with frozen protocol")
    return rows


def _reused_payload(source: Mapping, source_name: str, source_sha: str) -> dict:
    payload = dict(source["systems"][source_name])
    payload["execution_source"] = "sha256_pinned_native_9b_result"
    payload["source_results_sha256"] = source_sha
    return payload


def _mean(values: Sequence[float]) -> float:
    return float(math.fsum(float(value) for value in values) / len(values))


def _comparison(rows: Sequence[Mapping], left: str, right: str) -> dict:
    output = {}
    for metric in METRICS:
        deltas = [
            float(row["systems"][left][metric])
            - float(row["systems"][right][metric])
            for row in rows
        ]
        output[metric] = {
            "mean_delta": _mean(deltas),
            "wins": sum(value > 1e-12 for value in deltas),
            "ties": sum(abs(value) <= 1e-12 for value in deltas),
            "losses": sum(value < -1e-12 for value in deltas),
        }
    return output


def _summary_group(rows: Sequence[Mapping], comparisons: Sequence[Sequence[str]]) -> dict:
    return {
        "case_count": len(rows),
        "systems": {
            system: {
                metric: _mean([row["systems"][system][metric] for row in rows])
                for metric in METRICS
            }
            for system in SYSTEMS
        },
        "comparisons": {
            f"{left}_vs_{right}": _comparison(rows, left, right)
            for left, right in comparisons
        },
        "mean_post_query_resident_kv_bytes": {
            system: _mean(
                [row["systems"][system]["post_query_resident_kv_bytes"] for row in rows]
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
            for row in rows
        ),
    }


def summarize(rows: Sequence[Mapping], comparisons: Sequence[Sequence[str]]) -> dict:
    grouped: dict[str, list[Mapping]] = defaultdict(list)
    for row in rows:
        grouped[str(row["dataset"])].append(row)
    return {
        "overall": _summary_group(rows, comparisons),
        "by_dataset": {
            dataset: _summary_group(grouped[dataset], comparisons)
            for dataset in DATASET_ORDER
            if grouped[dataset]
        },
    }


def _validate_case_source(case: Mapping, source: Mapping, sample, prompt, shift: int) -> None:
    expected = {
        "sample_id": case["sample_id"],
        "dataset": case["dataset"],
        "record_index": case["record_index"],
        "record_sha256": case["record_sha256"],
        "context_tokens": case["context_tokens"],
        "query_tokens": case["query_tokens"],
    }
    if any(source.get(key) != value for key, value in expected.items()):
        raise OracleContractError("frozen case differs from its source result")
    if (
        source["query_probe"]["probe_id"] != case["probe_id"]
        or source["question"] != sample.question
        or list(source["answers"]) != get_ground_truths(sample)
        or int(prompt.context_tokens) != case["context_tokens"]
        or int(prompt.query_tokens) != case["query_tokens"]
        or int(source["construction"]["boundary_shift_characters"]) != shift
    ):
        raise OracleContractError("frozen case reconstruction changed")


def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("free-window development requires exactly one visible GPU")
    protocol_path = Path(args.protocol).resolve()
    protocol, protocol_sha = _load_protocol(protocol_path)
    native_path = Path(args.native_protocol).resolve()
    native, native_sha = load_native_protocol(native_path)
    if (
        native_sha != protocol["native_protocol"]["sha256"]
        or args.model_id != protocol["model_id"]
        or args.model_revision != protocol["model_revision"]
        or native["model_id"] != protocol["model_id"]
        or native["model_revision"] != protocol["model_revision"]
    ):
        raise OracleContractError("free-window model or parent protocol identity changed")

    source_path = Path(args.source_results).resolve()
    source_sha = protocol["source_results"]["sha256"]
    source_rows = _load_source_rows(
        source_path, source_sha, int(protocol["source_results"]["case_count"])
    )
    archive = Path(args.archive).resolve()
    datasets = _load_datasets(archive, native)
    cases = list(protocol["cases"])
    if args.limit is not None:
        if args.limit <= 0:
            raise OracleContractError("free-window limit must be positive")
        cases = cases[: args.limit]

    run_dir = Path(args.run_dir).resolve()
    model_path = Path(args.model_path).resolve()
    model_identity = model_provenance(
        model_path, args.model_id, revision=args.model_revision
    )
    manifest = ensure_run_manifest(
        run_dir,
        experiment="hmo_layerwise_free_window_development",
        args={
            "archive": str(archive),
            "archive_sha256": native["dataset_source"]["archive_sha256"],
            "model_id": args.model_id,
            "model_revision": args.model_revision,
            "native_protocol_sha256": native_sha,
            "protocol_sha256": protocol_sha,
            "source_results_sha256": source_sha,
            "case_limit": args.limit,
            "inference_seed": native["generation"]["inference_seed"],
            "recurrent_backend": REFERENCE_BACKEND,
            "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        selections={
            "scope": (
                "operational_smoke_excluded_from_claims"
                if args.limit is not None
                else protocol["purpose"]
            ),
            "cases": cases,
            "systems": list(SYSTEMS),
            "generated_systems": list(GENERATED_SYSTEMS),
            "primary_comparisons": protocol["primary_comparisons"],
            "method": protocol["method"],
            "outcome_conditioned_selection": False,
        },
        model=model_identity,
        project_root=PROJECT_ROOT,
        require_clean=True,
        environment=collect_environment(),
    )

    results_path = run_dir / RESULTS_FILENAME
    completed = _load_completed(results_path)
    if completed and not args.resume:
        raise OracleContractError("free-window results exist; pass --resume to continue")
    rows = list(completed)
    completed_ids = {str(row["sample_id"]) for row in completed}

    seed = int(native["generation"]["inference_seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("highest")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model, tokenizer = _load_model(model_path)
    attention_layers = tuple(get_full_attention_indices(model.config))
    recurrent_layers = tuple(get_linear_attention_indices(model.config))
    if not attention_layers or not recurrent_layers:
        raise OracleContractError("free-window model is not a Hybrid architecture")
    _force_torch_reference_backend(model, recurrent_layers)

    method = protocol["method"]
    for case_index, case in enumerate(cases, start=1):
        sample_id = str(case["sample_id"])
        if sample_id in completed_ids:
            continue
        case_started = time.perf_counter()
        source = source_rows[sample_id]
        dataset = str(case["dataset"]).removeprefix("longbench_")
        records, raw_lines = datasets[dataset]
        record_index = int(case["record_index"])
        if _sha256_bytes(raw_lines[record_index]) != case["record_sha256"]:
            raise OracleContractError("free-window source record SHA changed")
        sample = _make_sample(dataset, record_index, records[record_index])
        prompt, boundary_shift = tokenize_sample_prompt_aligned(sample, tokenizer)
        _validate_case_source(case, source, sample, prompt, boundary_shift)

        with torch.no_grad():
            context_outputs = model.model(
                prompt.context_ids.to(model.device), use_cache=True, return_dict=True
            )
        segments = build_segment_catalog(
            context_outputs.past_key_values,
            attention_layers,
            context_tokens=prompt.context_tokens,
            config=OracleConfig(
                segment_length=int(method["segment_length"]),
                middle_kv_fraction=float(protocol["middle_kv_fraction"]),
                protected_prefix_segments=int(method["protected_prefix_segments"]),
                protected_suffix_segments=int(method["protected_suffix_segments"]),
            ),
        )
        del context_outputs
        _cleanup_cuda()

        layer_indices, layer_array = _load_layer_scores(source)
        if layer_indices != attention_layers:
            raise OracleContractError("cached probe layer indices changed")
        layer_scores = {
            layer_index: layer_array[offset]
            for offset, layer_index in enumerate(layer_indices)
        }
        legacy = _legacy_plan(source)
        token_bytes = int(source["byte_accounting"]["token_kv_bytes"])
        chunkkv = build_chunkkv_plan(
            segments,
            layer_scores,
            context_tokens=prompt.context_tokens,
            target_context_charged_bytes=legacy.context_charged_bytes,
            context_token_kv_bytes=token_bytes,
            observation_query_tokens=prompt.query_tokens,
            chunk_size=int(method["free_window_width"]),
        )
        if chunkkv.to_dict() != source["plans"]["chunkkv"]:
            raise OracleContractError("free-window ChunkKV reconstruction changed")
        layer_local = build_layer_local_hmo_plan(
            segments,
            layer_scores,
            legacy,
            context_token_kv_bytes=token_bytes,
        )
        free_window = build_free_window_plan(segments, layer_scores, chunkkv)
        if (
            layer_local.schema_version != LAYER_LOCAL_SCHEMA
            or free_window.schema_version != FREE_WINDOW_SCHEMA
            or len(
                {
                    legacy.context_charged_bytes,
                    chunkkv.context_charged_bytes,
                    layer_local.context_charged_bytes,
                    free_window.context_charged_bytes,
                }
            )
            != 1
        ):
            raise OracleContractError("free-window plans are not exactly equal-byte")

        max_new_tokens = int(native["generation"]["max_new_tokens"][dataset])
        generated = {
            "hmo_layer_local": _generate_system(
                model,
                tokenizer,
                prompt,
                attention_layers,
                recurrent_layers,
                make_layerwise_window_intervention(
                    layer_local, name="hmo_layer_local"
                ),
                sample,
                max_new_tokens,
            ),
            "hmo_free_window": _generate_system(
                model,
                tokenizer,
                prompt,
                attention_layers,
                recurrent_layers,
                make_layerwise_window_intervention(
                    free_window, name="hmo_free_window"
                ),
                sample,
                max_new_tokens,
            ),
        }
        for system, source_name in REUSED_SYSTEM_MAP.items():
            generated[system] = _reused_payload(source, source_name, source_sha)
        compressed_expected = int(
            source["byte_accounting"]["equal_byte_post_query_resident_bytes"]
        )
        if {
            int(generated[system]["post_query_resident_kv_bytes"])
            for system in EQUAL_BYTE_SYSTEMS
        } != {compressed_expected}:
            raise OracleContractError("free-window generated arms miss equal-byte target")
        for system in GENERATED_SYSTEMS:
            generated[system].update(
                {
                    "post_query_budget_limit_bytes": int(
                        source["byte_accounting"]["compressed_post_query_cap_bytes"]
                    ),
                    "expected_post_query_resident_kv_bytes": compressed_expected,
                    "execution_source": "current_run",
                }
            )

        row = {
            "schema_version": RESULT_SCHEMA,
            "stage": "smoke" if args.limit is not None else "development",
            "sample_id": sample_id,
            "dataset": str(case["dataset"]),
            "record_index": record_index,
            "record_sha256": case["record_sha256"],
            "length_stratum": int(case["length_stratum"]),
            "question": sample.question,
            "answers": get_ground_truths(sample),
            "context_tokens": prompt.context_tokens,
            "query_tokens": prompt.query_tokens,
            "budget_fraction": float(protocol["middle_kv_fraction"]),
            "byte_accounting": source["byte_accounting"],
            "query_probe": source["query_probe"],
            "plans": {
                "hmo_legacy": source["plans"]["contiguous_cf"],
                "hmo_layer_local": layer_local.to_dict(),
                "chunkkv": source["plans"]["chunkkv"],
                "hmo_free_window": free_window.to_dict(),
            },
            "systems": generated,
            "elapsed_seconds": time.perf_counter() - case_started,
        }
        rows.append(row)
        _append_jsonl(results_path, row)
        print(
            f"[{case_index}/{len(cases)}] {sample_id}: "
            + " ".join(
                f"{system}={generated[system]['official_qa_f1']:.4f}"
                for system in SYSTEMS
            ),
            flush=True,
        )

    expected_ids = {str(case["sample_id"]) for case in cases}
    actual_ids = {str(row["sample_id"]) for row in rows}
    if actual_ids != expected_ids:
        raise OracleContractError("free-window development package is incomplete")
    rows.sort(
        key=lambda row: (
            DATASET_ORDER.index(row["dataset"]),
            int(row["length_stratum"]),
            int(row["record_index"]),
        )
    )
    summary = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete",
        "scope": (
            "operational_smoke_excluded_from_claims"
            if args.limit is not None
            else protocol["purpose"]
        ),
        "manifest_id": manifest["manifest_id"],
        "protocol_sha256": protocol_sha,
        "source_results_sha256": source_sha,
        "analysis": summarize(rows, protocol["primary_comparisons"]),
        "runtime": {
            "elapsed_seconds": time.perf_counter() - started,
            "gpu_name": torch.cuda.get_device_name(0),
            "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
            "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
        "automatic_followup": False,
        "samples": rows,
    }
    _atomic_json(run_dir / SUMMARY_FILENAME, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--native-protocol", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--source-results", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    payload = run(parse_args())
    print(json.dumps({key: value for key, value in payload.items() if key != "samples"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
