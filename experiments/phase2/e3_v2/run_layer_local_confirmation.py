"""Fresh four-task confirmation of layer-local HMO under exact equal bytes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from experiments.phase2.e3_v2.attention_probe import collect_attention_token_probe
from experiments.phase2.e3_v2.attention_probe_cache import (
    ATTENTION_PROBE_AGGREGATION,
    ATTENTION_PROBE_CACHE_SCHEMA,
    get_or_create_attention_probe,
)
from experiments.phase2.e3_v2.chunkkv_adapter import (
    CHUNKKV_ADAPTER_SCHEMA,
    build_chunkkv_plan,
    make_chunkkv_intervention,
)
from experiments.phase2.e3_v2.context_query import (
    full_kv_intervention,
    generate_greedy,
    run_post_intervention_prompt,
    tokenize_sample_prompt_aligned,
)
from experiments.phase2.e3_v2.coverage_fidelity import allocate_coverage_fidelity
from experiments.phase2.e3_v2.coverage_fidelity_cache import (
    build_retained_position_plan,
    make_coverage_fidelity_intervention,
)
from experiments.phase2.e3_v2.free_window_allocator import (
    LAYER_LOCAL_SCHEMA,
    build_layer_local_hmo_plan,
    make_layerwise_window_intervention,
)
from experiments.phase2.e3_v2.freeze_layer_local_confirmation_protocol import (
    CONFIRMATION_TASKS,
    METHOD_VERSION,
    PROTOCOL_SCHEMA,
    build_fresh_inventory,
    prompt_identity_sha256,
    select_stratified_cases,
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
    _eligible_action_counts,
)
from experiments.phase2.e3_v2.run_hotpot_paired import (
    METRICS,
    _load_completed,
    _score_text,
)
from experiments.phase2.e3_v2.run_hotpot_solvability import _sha256_file
from experiments.phase2.e3_v2.run_native_tasks import (
    _load_datasets,
    _make_sample,
    load_native_protocol,
)
from experiments.utils.cache_access import get_cache_layer
from experiments.utils.eval_harness import get_ground_truths
from experiments.utils.memory_accounting import get_active_kv_bytes
from experiments.utils.model_loader import (
    get_full_attention_indices,
    get_linear_attention_indices,
)
from experiments.utils.run_manifest import collect_environment, ensure_run_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULT_SCHEMA = "hmo.layer_local_confirmation_result.v1"
RESULTS_FILENAME = "layer_local_confirmation_results.jsonl"
SUMMARY_FILENAME = "layer_local_confirmation_summary.json"
SYSTEMS = ("hmo_legacy", "hmo_layer_local", "chunkkv", "full_kv_reference")
EQUAL_BYTE_SYSTEMS = SYSTEMS[:3]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_confirmation_protocol(path: Path) -> tuple[dict, str]:
    try:
        encoded = path.read_bytes()
        payload = json.loads(encoded)
    except (OSError, json.JSONDecodeError) as exc:
        raise OracleContractError(f"cannot read confirmation protocol: {path}") from exc
    selection = payload.get("selection", {})
    method = payload.get("method", {})
    if (
        payload.get("schema_version") != PROTOCOL_SCHEMA
        or payload.get("status") != "frozen_before_candidate_qa_outcomes"
        or payload.get("purpose") != "fresh_shorter_context_four_task_confirmation"
        or payload.get("method_version") != METHOD_VERSION
        or tuple(payload.get("systems", ())) != SYSTEMS
        or tuple(payload.get("equal_byte_systems", ())) != EQUAL_BYTE_SYSTEMS
        or payload.get("primary_comparisons")
        != [["hmo_layer_local", "hmo_legacy"], ["hmo_layer_local", "chunkkv"]]
        or payload.get("primary_metric") != "official_qa_f1"
        or tuple(payload.get("secondary_metrics", ())) != METRICS[1:]
        or float(payload.get("middle_kv_fraction", 0.0)) != 0.1
        or selection.get("dataset_order") != list(CONFIRMATION_TASKS)
        or int(selection.get("per_dataset", 0)) != 20
        or int(selection.get("length_strata", 0)) != 4
        or int(selection.get("per_stratum", 0)) != 5
        or selection.get("uses_qa_outcomes") is not False
        or selection.get("uses_probe_scores") is not False
        or int(payload.get("case_count", 0)) != 80
        or len(payload.get("cases", ())) != 80
        or method.get("layer_local_scope") != "sparse_window_placement_only"
        or method.get("recurrent_state_policy") != "unchanged"
        or method.get("byte_policy") != "exact_equal_post_query_resident_bytes"
    ):
        raise OracleContractError("layer-local confirmation protocol mismatch")
    identities = [case.get("prompt_identity_sha256") for case in payload["cases"]]
    sample_ids = [case.get("sample_id") for case in payload["cases"]]
    if (
        len(set(identities)) != 80
        or len(set(sample_ids)) != 80
        or any(not isinstance(value, str) or len(value) != 64 for value in identities)
    ):
        raise OracleContractError("confirmation cases are not unique frozen identities")
    return payload, hashlib.sha256(encoded).hexdigest()


def _layer_bytes(cache, attention_layers: Sequence[int]) -> dict[str, int]:
    output = {}
    for layer_index in attention_layers:
        layer = get_cache_layer(cache, int(layer_index))
        if not layer.has_kv():
            raise OracleContractError(f"Full-Attention layer {layer_index} has no KV")
        output[str(layer_index)] = int(
            layer.keys.numel() * layer.keys.element_size()
            + layer.values.numel() * layer.values.element_size()
        )
    return output


def _layer_lengths(cache, attention_layers: Sequence[int]) -> dict[str, int]:
    return {
        str(layer_index): int(get_cache_layer(cache, int(layer_index)).keys.shape[-2])
        for layer_index in attention_layers
    }


def _generate_system(
    model,
    tokenizer,
    prompt,
    attention_layers,
    recurrent_layers,
    intervention,
    sample,
    max_new_tokens: int,
) -> dict:
    started = time.perf_counter()
    state = run_post_intervention_prompt(
        model,
        prompt,
        attention_layer_indices=attention_layers,
        recurrent_layer_indices=recurrent_layers,
        intervention=intervention,
    )
    prompt_seconds = time.perf_counter() - started
    resident_bytes = get_active_kv_bytes(state.cache, list(attention_layers))
    per_layer_bytes = _layer_bytes(state.cache, attention_layers)
    per_layer_lengths = _layer_lengths(state.cache, attention_layers)
    intervention_metadata = dict(state.intervention.metadata)
    decode_started = time.perf_counter()
    answer = generate_greedy(model, tokenizer, state, max_new_tokens=max_new_tokens)
    decode_seconds = time.perf_counter() - decode_started
    token_ids = [int(value) for value in answer.token_ids[0].tolist()]
    payload = {
        **_score_text(answer.text, sample),
        "generated_text": answer.text,
        "generated_token_ids": token_ids,
        "generated_token_count": len(token_ids),
        "hit_generation_limit": len(token_ids) >= max_new_tokens,
        "post_query_resident_kv_bytes": int(resident_bytes),
        "post_query_layer_kv_bytes": per_layer_bytes,
        "post_query_layer_kv_lengths": per_layer_lengths,
        "intervention_metadata": intervention_metadata,
        "prompt_intervention_seconds": prompt_seconds,
        "decode_seconds": decode_seconds,
        "system_elapsed_seconds": time.perf_counter() - started,
    }
    del state
    _cleanup_cuda()
    return payload


def _pair_summary(rows: Sequence[Mapping], left: str, right: str) -> dict:
    return {
        metric: {
            "mean_delta": float(
                np.mean(
                    [
                        float(row["systems"][left][metric])
                        - float(row["systems"][right][metric])
                        for row in rows
                    ]
                )
            ),
            "wins": sum(
                float(row["systems"][left][metric])
                - float(row["systems"][right][metric])
                > 1e-12
                for row in rows
            ),
            "ties": sum(
                abs(
                    float(row["systems"][left][metric])
                    - float(row["systems"][right][metric])
                )
                <= 1e-12
                for row in rows
            ),
            "losses": sum(
                float(row["systems"][left][metric])
                - float(row["systems"][right][metric])
                < -1e-12
                for row in rows
            ),
        }
        for metric in METRICS
    }


def summarize_results(rows: Sequence[Mapping]) -> dict:
    if not rows:
        raise OracleContractError("confirmation summary requires results")
    systems = {
        system: {
            metric: float(np.mean([row["systems"][system][metric] for row in rows]))
            for metric in METRICS
        }
        for system in SYSTEMS
    }
    return {
        "case_count": len(rows),
        "systems": systems,
        "comparisons": {
            "hmo_layer_local_vs_hmo_legacy": _pair_summary(
                rows, "hmo_layer_local", "hmo_legacy"
            ),
            "hmo_layer_local_vs_chunkkv": _pair_summary(
                rows, "hmo_layer_local", "chunkkv"
            ),
            "hmo_layer_local_vs_full_kv_reference": _pair_summary(
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
        "equal_layer_byte_cases": sum(
            len(
                {
                    json.dumps(
                        row["systems"][system]["post_query_layer_kv_bytes"],
                        sort_keys=True,
                    )
                    for system in EQUAL_BYTE_SYSTEMS
                }
            )
            == 1
            for row in rows
        ),
        "generation_limit_hits": {
            system: sum(row["systems"][system]["hit_generation_limit"] for row in rows)
            for system in SYSTEMS
        },
        "mean_resident_fraction_of_full": {
            system: float(
                np.mean(
                    [
                        row["systems"][system]["post_query_resident_kv_bytes"]
                        / row["systems"]["full_kv_reference"]["post_query_resident_kv_bytes"]
                        for row in rows
                    ]
                )
            )
            for system in SYSTEMS
        },
    }


def _materialize_selected(protocol: Mapping, archive: Path, tokenizer):
    native_path = Path(protocol["native_protocol"]["path_hint"])
    native, native_sha = load_native_protocol(native_path)
    if native_sha != protocol["native_protocol"]["sha256"]:
        raise OracleContractError("native parent protocol SHA changed")
    inventory, audit = build_fresh_inventory(archive, native, tokenizer)
    reproduced = []
    selection = protocol["selection"]
    for dataset in CONFIRMATION_TASKS:
        reproduced.extend(
            select_stratified_cases(
                inventory[dataset],
                dataset=dataset,
                count=selection["per_dataset"],
                strata=selection["length_strata"],
                seed=selection["seed"],
            )
        )
    reproduced.sort(
        key=lambda item: (
            CONFIRMATION_TASKS.index(item["dataset"]),
            int(item["length_stratum"]),
            int(item["record_index"]),
        )
    )
    if reproduced != protocol["cases"]:
        raise OracleContractError("frozen confirmation selection does not reproduce")
    datasets = _load_datasets(archive, native)
    materialized = {}
    for case in reproduced:
        dataset = case["dataset"]
        index = int(case["record_index"])
        records, raw_lines = datasets[dataset]
        sample = _make_sample(dataset, index, records[index])
        prompt, shift = tokenize_sample_prompt_aligned(sample, tokenizer)
        observed = {
            "sample_id": sample.sample_id,
            "record_sha256": _sha256_bytes(raw_lines[index]),
            "context_tokens": prompt.context_tokens,
            "query_tokens": prompt.query_tokens,
            "boundary_shift_characters": shift,
            "prompt_identity_sha256": prompt_identity_sha256(prompt),
        }
        if any(observed[key] != case[key] for key in observed):
            raise OracleContractError(f"frozen confirmation case changed: {sample.sample_id}")
        materialized[sample.sample_id] = (sample, prompt, shift)
    return materialized, audit


def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("layer-local confirmation requires exactly one visible GPU")
    protocol_path = Path(args.protocol).resolve()
    protocol, protocol_sha = load_confirmation_protocol(protocol_path)
    if args.model_id != protocol["model_id"] or args.model_revision != protocol["model_revision"]:
        raise OracleContractError("model identity disagrees with confirmation protocol")
    archive = Path(args.archive).resolve()
    if _sha256_file(archive) != protocol["dataset_source"]["archive_sha256"]:
        raise OracleContractError("LongBench archive SHA changed")
    model_path = Path(args.model_path).resolve()
    for filename, expected in protocol["tokenizer_files_sha256"].items():
        if _sha256_file(model_path / filename) != expected:
            raise OracleContractError(f"tokenizer identity changed: {filename}")
    selected_cases = list(protocol["cases"])
    if args.limit is not None:
        if not 0 < args.limit <= len(selected_cases):
            raise OracleContractError("limit must select a non-empty frozen prefix")
        selected_cases = selected_cases[: args.limit]

    run_dir = Path(args.run_dir).resolve()
    probe_cache_dir = Path(args.probe_cache_dir).resolve()
    model_identity = model_provenance(model_path, args.model_id, revision=args.model_revision)
    manifest = ensure_run_manifest(
        run_dir,
        experiment="layer_local_hmo_fresh_four_task_confirmation",
        args={
            "archive": str(archive),
            "archive_sha256": protocol["dataset_source"]["archive_sha256"],
            "model_id": args.model_id,
            "model_revision": args.model_revision,
            "protocol_sha256": protocol_sha,
            "probe_cache_dir": str(probe_cache_dir),
            "limit": args.limit,
            "inference_seed": protocol["generation"]["inference_seed"],
            "recurrent_backend": REFERENCE_BACKEND,
            "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        selections={
            "scope": (
                "operational_smoke_excluded_from_claims"
                if args.limit is not None
                else protocol["purpose"]
            ),
            "method_version": METHOD_VERSION,
            "systems": list(SYSTEMS),
            "equal_byte_systems": list(EQUAL_BYTE_SYSTEMS),
            "cases": selected_cases,
            "candidate_search": False,
            "outcome_conditioned_selection": False,
            "attention_probe_cache_schema": ATTENTION_PROBE_CACHE_SCHEMA,
            "attention_probe_aggregation": ATTENTION_PROBE_AGGREGATION,
            "chunkkv_adapter_schema": CHUNKKV_ADAPTER_SCHEMA,
            "layer_local_schema": LAYER_LOCAL_SCHEMA,
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
    rows = list(completed)
    completed_ids = {str(row["sample_id"]) for row in rows}

    seed = int(protocol["generation"]["inference_seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("highest")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model, tokenizer = _load_model(model_path)
    attention_layers = tuple(get_full_attention_indices(model.config))
    recurrent_layers = tuple(get_linear_attention_indices(model.config))
    if not attention_layers or not recurrent_layers:
        raise OracleContractError("confirmation model is not Hybrid Attention")
    _force_torch_reference_backend(model, recurrent_layers)
    materialized, selection_audit = _materialize_selected(protocol, archive, tokenizer)

    method = protocol["method"]
    fraction = float(protocol["middle_kv_fraction"])
    for case_index, case in enumerate(selected_cases, start=1):
        sample, prompt, boundary_shift = materialized[case["sample_id"]]
        if sample.sample_id in completed_ids:
            continue
        case_started = time.perf_counter()
        with torch.no_grad():
            context_outputs = model.model(
                prompt.context_ids.to(model.device), use_cache=True, return_dict=True
            )
        segments = build_segment_catalog(
            context_outputs.past_key_values,
            attention_layers,
            context_tokens=prompt.context_tokens,
            config=OracleConfig(
                segment_length=method["segment_length"],
                middle_kv_fraction=fraction,
                protected_prefix_segments=method["protected_prefix_segments"],
                protected_suffix_segments=method["protected_suffix_segments"],
            ),
        )
        full_layer_context_bytes = _layer_bytes(
            context_outputs.past_key_values, attention_layers
        )
        del context_outputs
        _cleanup_cuda()

        probe_started = time.perf_counter()
        cached_probe = get_or_create_attention_probe(
            probe_cache_dir,
            model_identity=model_identity,
            prompt=prompt,
            attention_layer_indices=attention_layers,
            segments=segments,
            segment_length=method["segment_length"],
            collector=lambda: collect_attention_token_probe(
                model,
                prompt,
                attention_layer_indices=attention_layers,
                segments=segments,
            ),
        )
        probe_seconds = time.perf_counter() - probe_started
        probe = cached_probe.result
        attention = probe.alpha.as_dict()
        token_attention_mass = tuple(float(value) for value in probe.token_attention_mass)
        layer_token_attention_mass = probe.layer_scores()
        allocator_attention = {
            segment.segment_id: float(attention[segment.segment_id])
            for segment in segments
            if segment.eligible
        }
        exact_plan = allocate_coverage_fidelity(
            allocator_attention,
            None,
            segments,
            middle_kv_fraction=fraction,
            sparse_width=method["sparse_width"],
            use_accessibility=False,
            enable_exact_upgrades=True,
        )
        eligible = [segment for segment in segments if segment.eligible]
        token_unit_bytes = eligible[0].kv_bytes // eligible[0].token_count
        legacy_plan = build_retained_position_plan(
            exact_plan,
            segments,
            token_attention_mass,
            context_tokens=prompt.context_tokens,
            sparse_selector="max_mass_window",
        )
        layer_local_plan = build_layer_local_hmo_plan(
            segments,
            layer_token_attention_mass,
            legacy_plan,
            context_token_kv_bytes=token_unit_bytes,
        )
        chunkkv_plan = build_chunkkv_plan(
            segments,
            layer_token_attention_mass,
            context_tokens=prompt.context_tokens,
            target_context_charged_bytes=exact_plan.total_charged_bytes,
            context_token_kv_bytes=token_unit_bytes,
            observation_query_tokens=prompt.query_tokens,
            chunk_size=method["chunkkv_chunk_size"],
        )
        if {
            legacy_plan.context_charged_bytes,
            layer_local_plan.context_charged_bytes,
            chunkkv_plan.context_charged_bytes,
        } != {exact_plan.total_charged_bytes}:
            raise OracleContractError("confirmation plans are not exact equal-byte")
        del probe
        _cleanup_cuda()

        interventions = {
            "hmo_legacy": make_coverage_fidelity_intervention(
                legacy_plan, attention_layers, name="hmo_legacy"
            ),
            "hmo_layer_local": make_layerwise_window_intervention(
                layer_local_plan, name="hmo_layer_local"
            ),
            "chunkkv": make_chunkkv_intervention(chunkkv_plan),
            "full_kv_reference": full_kv_intervention,
        }
        max_new_tokens = int(protocol["generation"]["max_new_tokens"][case["dataset"]])
        generated = {
            name: _generate_system(
                model,
                tokenizer,
                prompt,
                attention_layers,
                recurrent_layers,
                interventions[name],
                sample,
                max_new_tokens,
            )
            for name in SYSTEMS
        }
        query_kv_bytes = prompt.query_tokens * token_unit_bytes
        compressed_expected = exact_plan.total_charged_bytes + query_kv_bytes
        full_expected = sum(segment.kv_bytes for segment in segments) + query_kv_bytes
        if {
            generated[name]["post_query_resident_kv_bytes"]
            for name in EQUAL_BYTE_SYSTEMS
        } != {compressed_expected}:
            raise OracleContractError("compressed arms miss post-query byte target")
        if generated["full_kv_reference"]["post_query_resident_kv_bytes"] != full_expected:
            raise OracleContractError("Full-KV arm misses post-query byte target")
        layer_byte_signatures = {
            json.dumps(generated[name]["post_query_layer_kv_bytes"], sort_keys=True)
            for name in EQUAL_BYTE_SYSTEMS
        }
        layer_length_signatures = {
            json.dumps(generated[name]["post_query_layer_kv_lengths"], sort_keys=True)
            for name in EQUAL_BYTE_SYSTEMS
        }
        if len(layer_byte_signatures) != 1 or len(layer_length_signatures) != 1:
            raise OracleContractError("compressed arms differ in per-layer KV residency")
        protected_ids = {segment.segment_id for segment in segments if segment.protected}
        for name in SYSTEMS:
            expected = compressed_expected if name in EQUAL_BYTE_SYSTEMS else full_expected
            generated[name].update(
                {
                    "expected_post_query_resident_kv_bytes": expected,
                    "post_query_budget_limit_bytes": (
                        exact_plan.total_budget_limit_bytes + query_kv_bytes
                        if name in EQUAL_BYTE_SYSTEMS
                        else full_expected
                    ),
                }
            )
        row = {
            "schema_version": RESULT_SCHEMA,
            "stage": "smoke" if args.limit is not None else "formal",
            "method_version": METHOD_VERSION,
            "sample_id": sample.sample_id,
            "dataset": sample.dataset,
            "record_index": int(case["record_index"]),
            "record_sha256": case["record_sha256"],
            "prompt_identity_sha256": case["prompt_identity_sha256"],
            "length_stratum": int(case["length_stratum"]),
            "question": sample.question,
            "answers": get_ground_truths(sample),
            "construction": {
                "source_context_unchanged": True,
                "augmentation": False,
                "truncation": False,
                "memory_context_tokens": prompt.context_tokens,
                "query_tokens": prompt.query_tokens,
                "boundary_shift_characters": boundary_shift,
            },
            "byte_accounting": {
                "context_token_kv_bytes": token_unit_bytes,
                "query_kv_bytes": query_kv_bytes,
                "equal_byte_post_query_resident_bytes": compressed_expected,
                "full_post_query_resident_bytes": full_expected,
                "full_layer_context_kv_bytes": full_layer_context_bytes,
            },
            "query_probe": {
                **cached_probe.provenance(),
                "probe_seconds": probe_seconds,
            },
            "plans": {
                "shared_allocation": {
                    "allocation": exact_plan.to_dict(),
                    "action_counts": _eligible_action_counts(exact_plan, protected_ids),
                },
                "hmo_legacy": legacy_plan.to_dict(),
                "hmo_layer_local": layer_local_plan.to_dict(),
                "chunkkv": chunkkv_plan.to_dict(),
            },
            "systems": generated,
            "elapsed_seconds": time.perf_counter() - case_started,
        }
        rows.append(row)
        _append_jsonl(results_path, row)
        print(
            f"[{case_index}/{len(selected_cases)}] {sample.sample_id}: "
            + " ".join(
                f"{name}={generated[name]['official_qa_f1']:.4f}" for name in SYSTEMS
            ),
            flush=True,
        )

    expected_ids = {case["sample_id"] for case in selected_cases}
    actual_ids = {str(row["sample_id"]) for row in rows}
    if actual_ids != expected_ids:
        raise OracleContractError("confirmation run did not complete its frozen package")
    rows.sort(
        key=lambda row: (
            CONFIRMATION_TASKS.index(row["dataset"].removeprefix("longbench_")),
            int(row["length_stratum"]),
            int(row["record_index"]),
        )
    )
    overall = summarize_results(rows)
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
        "analysis": {
            **overall,
            "by_dataset": {
                dataset: summarize_results(
                    [
                        row
                        for row in rows
                        if row["dataset"] == f"longbench_{dataset}"
                    ]
                )
                for dataset in CONFIRMATION_TASKS
                if any(
                    row["dataset"] == f"longbench_{dataset}" for row in rows
                )
            },
        },
        "selection_audit": selection_audit,
        "runtime": {
            "elapsed_seconds": time.perf_counter() - started,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        },
    }
    _atomic_json(run_dir / SUMMARY_FILENAME, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--probe-cache-dir", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    payload = run(parse_args())
    print(json.dumps({key: value for key, value in payload.items() if key != "selection_audit"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
