"""Generate only missing layer-local HMO outputs for the frozen 506-row table."""
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
    _segments_from_row,
)
from experiments.phase2.e3_v2.chunkkv_adapter import build_chunkkv_plan
from experiments.phase2.e3_v2.context_query import tokenize_sample_prompt_aligned
from experiments.phase2.e3_v2.free_window_allocator import (
    LAYER_LOCAL_SCHEMA,
    build_layer_local_hmo_plan,
    make_layerwise_window_intervention,
)
from experiments.phase2.e3_v2.freeze_free_window_dev_protocol import DATASET_ORDER
from experiments.phase2.e3_v2.freeze_layer_local_completion_protocol import (
    METHOD_VERSION,
    PROTOCOL_SCHEMA,
)
from experiments.phase2.e3_v2.oracle import (
    OracleConfig,
    OracleContractError,
    build_segment_catalog,
)
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
from experiments.phase2.e3_v2.run_free_window_dev import (
    _load_source_rows,
    _reused_payload,
    _validate_case_source,
)
from experiments.phase2.e3_v2.run_hotpot_paired import (
    METRICS,
    _generate_system,
    _load_completed,
)
from experiments.phase2.e3_v2.run_native_tasks import (
    _load_datasets,
    _make_sample,
    load_native_protocol,
)
from experiments.utils.eval_harness import get_ground_truths
from experiments.utils.model_loader import (
    get_full_attention_indices,
    get_linear_attention_indices,
)
from experiments.utils.run_manifest import collect_environment, ensure_run_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULT_SCHEMA = "hmo.layer_local_completion_result.v1"
RESULTS_FILENAME = "layer_local_completion_results.jsonl"
SUMMARY_FILENAME = "layer_local_completion_summary.json"
SYSTEMS = ("hmo_legacy", "hmo_layer_local", "chunkkv", "full_kv_reference")
EQUAL_BYTE_SYSTEMS = SYSTEMS[:3]
REUSED_SYSTEM_MAP = {
    "hmo_legacy": "contiguous_cf",
    "chunkkv": "chunkkv",
    "full_kv_reference": "full_kv_reference",
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_protocol(path: Path) -> tuple[dict, str]:
    try:
        encoded = path.read_bytes()
        payload = json.loads(encoded)
    except (OSError, json.JSONDecodeError) as exc:
        raise OracleContractError(f"cannot read completion protocol: {path}") from exc
    execution = payload.get("execution", {})
    selection = payload.get("selection", {})
    cases = payload.get("cases", [])
    if (
        payload.get("schema_version") != PROTOCOL_SCHEMA
        or payload.get("status") != "frozen_before_remaining_layer_local_outcomes"
        or payload.get("purpose")
        != "development_robustness_completion_on_original_506_main_table"
        or payload.get("method_version") != METHOD_VERSION
        or tuple(payload.get("systems", ())) != SYSTEMS
        or tuple(payload.get("reused_systems", ())) != tuple(REUSED_SYSTEM_MAP)
        or payload.get("generated_system") != "hmo_layer_local"
        or payload.get("primary_comparisons")
        != [["hmo_layer_local", "hmo_legacy"], ["hmo_layer_local", "chunkkv"]]
        or payload.get("primary_metric") != "official_qa_f1"
        or tuple(payload.get("secondary_metrics", ())) != METRICS[1:]
        or float(payload.get("middle_kv_fraction", 0.0)) != 0.1
        or tuple(selection.get("dataset_order", ())) != DATASET_ORDER
        or int(selection.get("length_strata", 0)) != 4
        or selection.get("development_only") is not True
        or selection.get("independent_confirmation") is not False
        or execution.get("reused_layer_local_cases") != 120
        or execution.get("generated_layer_local_cases") != 386
        or execution.get("baseline_generation_cells") != 0
        or execution.get("continuation_gate") is not False
        or len(cases) != 506
        or len({case.get("sample_id") for case in cases}) != 506
        or sum(case.get("layer_local_execution") == "generate_once" for case in cases)
        != 386
    ):
        raise OracleContractError("layer-local completion protocol mismatch")
    return payload, _sha256_bytes(encoded)


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


def _summary_group(rows: Sequence[Mapping]) -> dict:
    if not rows:
        raise OracleContractError("completion summary group is empty")
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
            "hmo_layer_local_vs_hmo_legacy": _comparison(
                rows, "hmo_layer_local", "hmo_legacy"
            ),
            "hmo_layer_local_vs_chunkkv": _comparison(
                rows, "hmo_layer_local", "chunkkv"
            ),
            "hmo_layer_local_vs_full_kv_reference": _comparison(
                rows, "hmo_layer_local", "full_kv_reference"
            ),
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
        "generation_limit_hits": {
            system: sum(
                len(row["systems"][system]["generated_token_ids"])
                >= int(row["max_new_tokens"])
                for row in rows
            )
            for system in SYSTEMS
        },
    }


def summarize(rows: Sequence[Mapping]) -> dict:
    by_dataset: dict[str, list[Mapping]] = defaultdict(list)
    by_stratum: dict[int, list[Mapping]] = defaultdict(list)
    for row in rows:
        by_dataset[str(row["dataset"])].append(row)
        by_stratum[int(row["length_stratum"])].append(row)
    return {
        "overall": _summary_group(rows),
        "by_dataset": {
            dataset: _summary_group(by_dataset[dataset])
            for dataset in DATASET_ORDER
        },
        "by_length_stratum": {
            str(stratum): {
                **_summary_group(by_stratum[stratum]),
                "mean_context_tokens": _mean(
                    [row["context_tokens"] for row in by_stratum[stratum]]
                ),
            }
            for stratum in range(4)
        },
    }


def _without_selector_timing(plan: Mapping) -> dict:
    normalized = json.loads(json.dumps(plan))
    for layer in normalized["layers"]:
        layer.pop("selector_seconds", None)
    return normalized


def _validate_reused_layer_local(source: Mapping, dev: Mapping) -> None:
    segments = _segments_from_row(source, 256)
    layer_indices, layer_array = _load_layer_scores(source)
    legacy = _legacy_plan(source)
    reconstructed = build_layer_local_hmo_plan(
        segments,
        {
            layer_index: layer_array[offset]
            for offset, layer_index in enumerate(layer_indices)
        },
        legacy,
        context_token_kv_bytes=int(source["byte_accounting"]["token_kv_bytes"]),
    )
    if _without_selector_timing(reconstructed.to_dict()) != _without_selector_timing(
        dev["plans"]["hmo_layer_local"]
    ):
        raise OracleContractError("reused layer-local plan changed structurally")


def _combined_row(
    case: Mapping,
    source: Mapping,
    *,
    layer_local_system: Mapping,
    layer_local_plan: Mapping,
    layer_local_source: str,
    source_sha: str,
    max_new_tokens: int,
) -> dict:
    systems = {
        system: _reused_payload(source, source_name, source_sha)
        for system, source_name in REUSED_SYSTEM_MAP.items()
    }
    systems["hmo_layer_local"] = dict(layer_local_system)
    systems["hmo_layer_local"]["execution_source"] = layer_local_source
    compressed_expected = int(
        source["byte_accounting"]["equal_byte_post_query_resident_bytes"]
    )
    if {
        int(systems[system]["post_query_resident_kv_bytes"])
        for system in EQUAL_BYTE_SYSTEMS
    } != {compressed_expected}:
        raise OracleContractError("combined completion row misses equal-byte target")
    return {
        "schema_version": RESULT_SCHEMA,
        "stage": "development_completion",
        "method_version": METHOD_VERSION,
        "sample_id": str(case["sample_id"]),
        "dataset": str(case["dataset"]),
        "record_index": int(case["record_index"]),
        "record_sha256": str(case["record_sha256"]),
        "length_stratum": int(case["length_stratum"]),
        "question": source["question"],
        "answers": list(source["answers"]),
        "context_tokens": int(case["context_tokens"]),
        "query_tokens": int(case["query_tokens"]),
        "max_new_tokens": max_new_tokens,
        "budget_fraction": 0.1,
        "byte_accounting": source["byte_accounting"],
        "query_probe": source["query_probe"],
        "plans": {
            "hmo_legacy": source["plans"]["contiguous_cf"],
            "hmo_layer_local": dict(layer_local_plan),
            "chunkkv": source["plans"]["chunkkv"],
        },
        "systems": systems,
    }


def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("layer-local completion requires exactly one visible GPU")
    protocol, protocol_sha = load_protocol(Path(args.protocol).resolve())
    if args.model_id != protocol["model_id"] or args.model_revision != protocol["model_revision"]:
        raise OracleContractError("completion model identity changed")
    parents = protocol["parents"]
    source_path = Path(args.source_results).resolve()
    dev_path = Path(args.dev_results).resolve()
    native_path = Path(args.native_protocol).resolve()
    source_sha = parents["native_results"]["sha256"]
    dev_sha = parents["free_window_dev_results"]["sha256"]
    source_rows = _load_source_rows(source_path, source_sha, 506)
    dev_rows = _load_source_rows(dev_path, dev_sha, 120)
    native, native_sha = load_native_protocol(native_path)
    if native_sha != parents["native_protocol"]["sha256"]:
        raise OracleContractError("completion native protocol changed")
    cases = list(protocol["cases"])
    generate_cases = [
        case for case in cases if case["layer_local_execution"] == "generate_once"
    ]
    if args.limit is not None:
        if not 0 < args.limit <= len(generate_cases):
            raise OracleContractError("completion limit must select a missing prefix")
        selected_cases = generate_cases[: args.limit]
        seed_reused = False
    else:
        selected_cases = cases
        seed_reused = True

    run_dir = Path(args.run_dir).resolve()
    model_path = Path(args.model_path).resolve()
    archive = Path(args.archive).resolve()
    datasets = _load_datasets(archive, native)
    model_identity = model_provenance(
        model_path, args.model_id, revision=args.model_revision
    )
    manifest = ensure_run_manifest(
        run_dir,
        experiment="layer_local_hmo_original_506_completion",
        args={
            "archive": str(archive),
            "model_id": args.model_id,
            "model_revision": args.model_revision,
            "protocol_sha256": protocol_sha,
            "source_results_sha256": source_sha,
            "dev_results_sha256": dev_sha,
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
            "cases": selected_cases,
            "generated_system": "hmo_layer_local",
            "baseline_generation_cells": 0,
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
        raise OracleContractError("completion results exist; pass --resume")
    rows = list(completed)
    completed_ids = {str(row["sample_id"]) for row in rows}

    if seed_reused:
        case_by_id = {str(case["sample_id"]): case for case in cases}
        for sample_id, dev in dev_rows.items():
            if sample_id in completed_ids:
                continue
            case = case_by_id[sample_id]
            source = source_rows[sample_id]
            _validate_reused_layer_local(source, dev)
            dataset = str(case["dataset"]).removeprefix("longbench_")
            row = _combined_row(
                case,
                source,
                layer_local_system=dev["systems"]["hmo_layer_local"],
                layer_local_plan=dev["plans"]["hmo_layer_local"],
                layer_local_source="sha_pinned_free_window_development",
                source_sha=source_sha,
                max_new_tokens=int(native["generation"]["max_new_tokens"][dataset]),
            )
            rows.append(row)
            completed_ids.add(sample_id)
            _append_jsonl(results_path, row)

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
        raise OracleContractError("completion model is not Hybrid Attention")
    _force_torch_reference_backend(model, recurrent_layers)
    method = protocol["method"]

    generated_index = 0
    for case in selected_cases:
        sample_id = str(case["sample_id"])
        if sample_id in completed_ids:
            continue
        generated_index += 1
        source = source_rows[sample_id]
        dataset = str(case["dataset"]).removeprefix("longbench_")
        records, raw_lines = datasets[dataset]
        record_index = int(case["record_index"])
        if _sha256_bytes(raw_lines[record_index]) != case["record_sha256"]:
            raise OracleContractError("completion source record SHA changed")
        sample = _make_sample(dataset, record_index, records[record_index])
        prompt, shift = tokenize_sample_prompt_aligned(sample, tokenizer)
        _validate_case_source(case, source, sample, prompt, shift)
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
            raise OracleContractError("completion probe layer indices changed")
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
            chunk_size=10,
        )
        if chunkkv.to_dict() != source["plans"]["chunkkv"]:
            raise OracleContractError("completion ChunkKV reconstruction changed")
        layer_local = build_layer_local_hmo_plan(
            segments,
            layer_scores,
            legacy,
            context_token_kv_bytes=token_bytes,
        )
        if (
            layer_local.schema_version != LAYER_LOCAL_SCHEMA
            or layer_local.context_charged_bytes != legacy.context_charged_bytes
            or any(
                len(layer.active_positions) != len(legacy.active_positions)
                for layer in layer_local.layers
            )
        ):
            raise OracleContractError("completion layer-local geometry changed bytes")
        max_new_tokens = int(native["generation"]["max_new_tokens"][dataset])
        generated = _generate_system(
            model,
            tokenizer,
            prompt,
            attention_layers,
            recurrent_layers,
            make_layerwise_window_intervention(layer_local, name="hmo_layer_local"),
            sample,
            max_new_tokens,
        )
        generated.update(
            {
                "post_query_budget_limit_bytes": int(
                    source["byte_accounting"]["compressed_post_query_cap_bytes"]
                ),
                "expected_post_query_resident_kv_bytes": int(
                    source["byte_accounting"]["equal_byte_post_query_resident_bytes"]
                ),
            }
        )
        row = _combined_row(
            case,
            source,
            layer_local_system=generated,
            layer_local_plan=layer_local.to_dict(),
            layer_local_source="current_run",
            source_sha=source_sha,
            max_new_tokens=max_new_tokens,
        )
        rows.append(row)
        completed_ids.add(sample_id)
        _append_jsonl(results_path, row)
        print(
            f"[{generated_index}/{sum(c['layer_local_execution'] == 'generate_once' for c in selected_cases)}] "
            f"{sample_id}: layer_local={generated['official_qa_f1']:.4f} "
            f"legacy={row['systems']['hmo_legacy']['official_qa_f1']:.4f} "
            f"chunkkv={row['systems']['chunkkv']['official_qa_f1']:.4f}",
            flush=True,
        )

    expected_ids = {str(case["sample_id"]) for case in selected_cases}
    actual_ids = {str(row["sample_id"]) for row in rows}
    if actual_ids != expected_ids:
        raise OracleContractError("completion run did not finish its frozen package")
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
        "method_version": METHOD_VERSION,
        "manifest_id": manifest["manifest_id"],
        "protocol_sha256": protocol_sha,
        "source_results_sha256": source_sha,
        "dev_results_sha256": dev_sha,
        "analysis": summarize(rows),
        "execution": {
            "reused_layer_local_rows": sum(
                row["systems"]["hmo_layer_local"]["execution_source"]
                == "sha_pinned_free_window_development"
                for row in rows
            ),
            "current_run_layer_local_rows": sum(
                row["systems"]["hmo_layer_local"]["execution_source"]
                == "current_run"
                for row in rows
            ),
            "baseline_generation_cells": 0,
        },
        "runtime": {
            "elapsed_seconds": time.perf_counter() - started,
            "gpu_name": torch.cuda.get_device_name(0),
            "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        },
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
    parser.add_argument("--dev-results", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    payload = run(parse_args())
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
