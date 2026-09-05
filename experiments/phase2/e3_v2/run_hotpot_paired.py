"""Frozen equal-byte 32K HotpotQA paired pilot for HMO and structured baselines."""
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

from experiments.phase2.e3_v2.context_query import (
    generate_greedy,
    run_post_intervention_prompt,
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
from experiments.phase2.e3_v2.query_probe_cache import (
    QUERY_PROBE_AGGREGATION,
    QUERY_PROBE_CACHE_SCHEMA,
    get_or_create_query_probe,
    retained_positions_sha256,
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
    restrict_eligible_signals,
)
from experiments.phase2.e3_v2.run_end_task import select_equal_byte_segments
from experiments.phase2.e3_v2.run_hotpot_solvability import (
    _load_records,
    _validate_case_records,
    build_augmented_sample,
    load_protocol as load_solvability_protocol,
    validate_longest_base_selection,
)
from experiments.utils.eval_harness import get_ground_truths, score_prediction
from experiments.utils.memory_accounting import get_active_kv_bytes
from experiments.utils.model_loader import (
    get_full_attention_indices,
    get_linear_attention_indices,
)
from experiments.utils.run_manifest import collect_environment, ensure_run_manifest
from experiments.vendor.longbench_metrics import normalize_answer


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_SCHEMA = "hmo.hotpotqa_32k_paired_protocol.v1"
RESULT_SCHEMA = "hmo.hotpotqa_32k_paired_result.v1"
RESULTS_FILENAME = "hotpot_paired_results.jsonl"
SUMMARY_FILENAME = "hotpot_paired_summary.json"
SYSTEMS = (
    "contiguous_cf",
    "global_fixed_chunk_topk",
    "raw_alpha_exact_slack",
    "scattered_cf",
    "full_kv_reference",
)
EQUAL_BYTE_SYSTEMS = SYSTEMS[:-1]
METRICS = (
    "official_qa_f1",
    "normalized_answer_contains",
    "normalized_exact_match",
)
STAGE_CASE_COUNTS = {"smoke": 1, "formal": 4}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_paired_protocol(path: Path) -> tuple[dict, str]:
    try:
        encoded = path.read_bytes()
        payload = json.loads(encoded)
    except (OSError, json.JSONDecodeError) as exc:
        raise OracleContractError(f"cannot read Hotpot paired protocol: {path}") from exc

    method = payload.get("method", {})
    execution = payload.get("execution", {})
    parent = payload.get("full_kv_parent", {})
    expected_comparisons = [
        ["contiguous_cf", "global_fixed_chunk_topk"],
        ["contiguous_cf", "raw_alpha_exact_slack"],
        ["contiguous_cf", "scattered_cf"],
        ["contiguous_cf", "full_kv_reference"],
    ]
    if (
        payload.get("schema_version") != PROTOCOL_SCHEMA
        or payload.get("status") != "frozen_before_compressed_outcomes"
        or payload.get("model_id") != "Qwen/Qwen3.5-0.8B"
        or not payload.get("model_revision")
        or tuple(payload.get("systems", ())) != SYSTEMS
        or tuple(payload.get("equal_byte_systems", ())) != EQUAL_BYTE_SYSTEMS
        or payload.get("primary_comparisons") != expected_comparisons
        or payload.get("primary_metric") != "official_qa_f1"
        or tuple(payload.get("secondary_metrics", ())) != METRICS[1:]
        or float(payload.get("middle_kv_fraction", 0.0)) != 0.1
        or int(payload.get("max_new_tokens", 0)) != 32
        or int(payload.get("inference_seed", 0)) <= 0
        or len(payload.get("solvability_protocol_sha256", "")) != 64
        or any(len(parent.get(key, "")) != 64 for key in ("results_sha256", "summary_sha256"))
        or len(parent.get("manifest_id", "")) != 64
        or len(parent.get("code_commit", "")) != 40
        or method
        != {
            "allocator": "attention_led",
            "sparse_selector": "max_mass_window",
            "sparse_width": 16,
            "raw_slack_selector": "global_top_tokens_slack",
            "global_fixed_chunk_width": 16,
            "global_fixed_chunk_slack": "prefix_of_next_ranked_chunk",
            "segment_length": 256,
            "protected_prefix_segments": 1,
            "protected_suffix_segments": 1,
        }
        or execution
        != {
            "smoke_case_count": 1,
            "formal_case_count": 4,
            "full_kv_reused_from_parent": True,
            "continuation_gate": False,
            "case_filtering_after_outcomes": False,
            "automatic_followup": False,
        }
    ):
        raise OracleContractError("Hotpot paired protocol mismatch")
    return payload, hashlib.sha256(encoded).hexdigest()


def load_full_kv_parent(path: Path, protocol: Mapping) -> dict[str, dict]:
    parent = protocol["full_kv_parent"]
    if _sha256(path) != parent["results_sha256"]:
        raise OracleContractError("Full-KV parent JSONL SHA mismatch")
    summary_path = path.parent / "hotpot_solvability_summary.json"
    if _sha256(summary_path) != parent["summary_sha256"]:
        raise OracleContractError("Full-KV parent summary SHA mismatch")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        manifest = json.loads(
            (path.parent / "run_manifest.json").read_text(encoding="utf-8")
        )
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, json.JSONDecodeError) as exc:
        raise OracleContractError("cannot parse Full-KV parent artifacts") from exc
    if (
        summary.get("status") != "complete"
        or summary.get("scope") != "full_kv_solvability_routing_evidence"
        or summary.get("manifest_id") != parent["manifest_id"]
        or manifest.get("manifest_id") != parent["manifest_id"]
        or manifest.get("run_spec", {}).get("code", {}).get("commit")
        != parent["code_commit"]
        or len(rows) != STAGE_CASE_COUNTS["formal"]
    ):
        raise OracleContractError("Full-KV parent metadata mismatch")
    by_id = {str(row.get("sample_id")): row for row in rows}
    if len(by_id) != len(rows) or any(
        row.get("stage") != "formal"
        or row.get("system") != "full_kv_reference"
        or row.get("official_metric") != "f1"
        for row in rows
    ):
        raise OracleContractError("Full-KV parent rows violate their frozen scope")
    return by_id


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
                    f"invalid Hotpot paired result at line {line_number}"
                ) from exc
    sample_ids = [row.get("sample_id") for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise OracleContractError("Hotpot paired results contain duplicate cases")
    return rows


def _score_text(text: str, sample) -> dict:
    scores = score_prediction(text, sample)
    prediction = normalize_answer(text)
    truths = [normalize_answer(value) for value in get_ground_truths(sample)]
    return {
        "official_metric": scores.primary_metric,
        "official_qa_f1": scores.primary_score,
        "normalized_answer_contains": float(
            any(value and value in prediction for value in truths)
        ),
        "normalized_exact_match": float(prediction in truths),
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
    prompt_started = started
    state = run_post_intervention_prompt(
        model,
        prompt,
        attention_layer_indices=attention_layers,
        recurrent_layer_indices=recurrent_layers,
        intervention=intervention,
    )
    prompt_seconds = time.perf_counter() - prompt_started
    resident_bytes = get_active_kv_bytes(state.cache, list(attention_layers))
    decode_started = time.perf_counter()
    answer = generate_greedy(
        model, tokenizer, state, max_new_tokens=max_new_tokens
    )
    decode_seconds = time.perf_counter() - decode_started
    payload = {
        **_score_text(answer.text, sample),
        "generated_text": answer.text,
        "generated_token_ids": [int(value) for value in answer.token_ids[0].tolist()],
        "post_query_resident_kv_bytes": int(resident_bytes),
        "prompt_intervention_seconds": prompt_seconds,
        "decode_seconds": decode_seconds,
        "system_elapsed_seconds": time.perf_counter() - started,
    }
    del state
    _cleanup_cuda()
    return payload


def _parent_system_payload(row: Mapping, sample, construction: Mapping) -> dict:
    if (
        row.get("question") != sample.question
        or list(row.get("answers", ())) != get_ground_truths(sample)
        or row.get("construction") != construction
    ):
        raise OracleContractError("Full-KV parent sample differs from reconstructed case")
    rescored = _score_text(str(row["generated_text"]), sample)
    if any(
        rescored[metric] != float(row[metric])
        for metric in METRICS
    ):
        raise OracleContractError("Full-KV parent scores do not reproduce")
    return {
        **rescored,
        "generated_text": row["generated_text"],
        "generated_token_ids": list(row["generated_token_ids"]),
        "post_query_resident_kv_bytes": int(row["post_query_resident_kv_bytes"]),
        "source": "sha_pinned_p6_full_kv_parent",
    }


def _pair_summary(rows: Sequence[Mapping], left: str, right: str) -> dict:
    output = {}
    for metric in METRICS:
        deltas = [
            float(row["systems"][left][metric])
            - float(row["systems"][right][metric])
            for row in rows
        ]
        output[metric] = {
            "mean_delta": float(np.mean(deltas)),
            "wins": sum(value > 1e-12 for value in deltas),
            "ties": sum(abs(value) <= 1e-12 for value in deltas),
            "losses": sum(value < -1e-12 for value in deltas),
        }
    return output


def summarize_results(
    rows: Sequence[Mapping],
    systems: Sequence[str] = SYSTEMS,
    equal_byte_systems: Sequence[str] = EQUAL_BYTE_SYSTEMS,
) -> dict:
    if not rows:
        raise OracleContractError("Hotpot paired summary requires results")
    system_names = tuple(systems)
    equal_byte_names = tuple(equal_byte_systems)
    system_metrics = {
        system: {
            metric: float(
                np.mean([float(row["systems"][system][metric]) for row in rows])
            )
            for metric in METRICS
        }
        for system in system_names
    }
    mean_resident = {
        system: float(
            np.mean(
                [row["systems"][system]["post_query_resident_kv_bytes"] for row in rows]
            )
        )
        for system in system_names
    }
    return {
        "case_count": len(rows),
        "systems": system_metrics,
        "comparisons": {
            f"contiguous_cf_vs_{system}": _pair_summary(
                rows, "contiguous_cf", system
            )
            for system in system_names
            if system != "contiguous_cf"
        },
        "mean_post_query_resident_kv_bytes": mean_resident,
        "mean_resident_fraction_of_full": {
            system: float(
                np.mean(
                    [
                        row["systems"][system]["post_query_resident_kv_bytes"]
                        / row["systems"]["full_kv_reference"][
                            "post_query_resident_kv_bytes"
                        ]
                        for row in rows
                    ]
                )
            )
            for system in system_names
        },
        "equal_resident_byte_cases": sum(
            len(
                {
                    row["systems"][system]["post_query_resident_kv_bytes"]
                    for system in equal_byte_names
                }
            )
            == 1
            for row in rows
        ),
    }


def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Hotpot paired pilot requires exactly one visible GPU")
    protocol, protocol_sha = load_paired_protocol(Path(args.protocol).resolve())
    solvability, solvability_sha = load_solvability_protocol(
        Path(args.solvability_protocol).resolve()
    )
    if solvability_sha != protocol["solvability_protocol_sha256"]:
        raise OracleContractError("solvability protocol SHA disagrees with paired protocol")
    if (
        args.model_id != protocol["model_id"]
        or args.model_revision != protocol["model_revision"]
        or solvability["model_id"] != protocol["model_id"]
        or solvability["model_revision"] != protocol["model_revision"]
    ):
        raise OracleContractError("model identity disagrees across paired protocols")

    case_count = STAGE_CASE_COUNTS[args.stage_set]
    selected_cases = solvability["cases"][:case_count]
    archive = Path(args.archive).resolve()
    records, raw_lines = _load_records(archive, solvability)
    _validate_case_records(records, raw_lines, solvability["cases"])
    full_parent = load_full_kv_parent(Path(args.full_kv_results).resolve(), protocol)
    run_dir = Path(args.run_dir).resolve()
    probe_cache_dir = Path(args.probe_cache_dir).resolve()
    model_path = Path(args.model_path).resolve()
    model_identity = model_provenance(
        model_path, args.model_id, revision=args.model_revision
    )
    manifest = ensure_run_manifest(
        run_dir,
        experiment="hotpotqa_32k_aug_equal_byte_paired_pilot",
        args={
            "archive": str(archive),
            "archive_sha256": solvability["dataset"]["archive_sha256"],
            "full_kv_parent_results_sha256": protocol["full_kv_parent"][
                "results_sha256"
            ],
            "model_id": args.model_id,
            "model_revision": args.model_revision,
            "protocol_sha256": protocol_sha,
            "probe_cache_dir": str(probe_cache_dir),
            "solvability_protocol_sha256": solvability_sha,
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
                else "four_case_real_task_paired_pilot"
            ),
            "systems": list(SYSTEMS),
            "equal_byte_systems": list(EQUAL_BYTE_SYSTEMS),
            "primary_comparisons": protocol["primary_comparisons"],
            "primary_metric": protocol["primary_metric"],
            "middle_kv_fraction": protocol["middle_kv_fraction"],
            "method": protocol["method"],
            "case_count": case_count,
            "cases": selected_cases,
            "candidate_search": False,
            "continuation_gate": False,
            "query_probe_cache_schema": QUERY_PROBE_CACHE_SCHEMA,
            "query_probe_aggregation": QUERY_PROBE_AGGREGATION,
        },
        model=model_identity,
        project_root=PROJECT_ROOT,
        require_clean=True,
        environment=collect_environment(),
    )

    results_path = run_dir / RESULTS_FILENAME
    completed = _load_completed(results_path)
    if completed and not args.resume:
        raise OracleContractError("Hotpot paired results exist; pass --resume to continue")
    rows = list(completed)
    completed_ids = {str(row["sample_id"]) for row in completed}

    seed = int(protocol["inference_seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("highest")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model, tokenizer = _load_model(model_path)
    attention_layers = tuple(get_full_attention_indices(model.config))
    recurrent_layers = tuple(get_linear_attention_indices(model.config))
    if not attention_layers or not recurrent_layers:
        raise OracleContractError("Hotpot paired model is not a Hybrid architecture")
    _force_torch_reference_backend(model, recurrent_layers)
    longest_base_audit = validate_longest_base_selection(
        records, solvability["cases"], tokenizer
    )

    method = protocol["method"]
    fraction = float(protocol["middle_kv_fraction"])
    for case_index, case in enumerate(selected_cases, start=1):
        base = records[case["base_index"]]
        donor = records[case["donor_index"]]
        sample, prompt, construction = build_augmented_sample(
            base, donor, case, tokenizer, solvability["construction"]
        )
        if sample.sample_id in completed_ids:
            continue
        if sample.sample_id not in full_parent:
            raise OracleContractError("Full-KV parent lacks a selected Hotpot sample")
        full_generated = _parent_system_payload(
            full_parent[sample.sample_id], sample, construction
        )
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
        del context_outputs
        _cleanup_cuda()

        cached_probe = get_or_create_query_probe(
            probe_cache_dir,
            model_identity=model_identity,
            prompt=prompt,
            attention_layer_indices=attention_layers,
            recurrent_layer_indices=recurrent_layers,
            segments=segments,
            segment_length=method["segment_length"],
            collector=lambda: collect_hybrid_query_token_probe(
                model,
                prompt,
                attention_layer_indices=attention_layers,
                recurrent_layer_indices=recurrent_layers,
                segments=segments,
                segment_length=method["segment_length"],
            ),
        )
        probe = cached_probe.result
        query_probe_provenance = cached_probe.provenance()
        attention = probe.alpha.as_dict()
        accessibility = probe.accessibility.field_dict("read_share")
        token_attention_mass = tuple(float(value) for value in probe.token_attention_mass)
        allocator_attention, allocator_accessibility = restrict_eligible_signals(
            attention, accessibility, segments
        )
        del probe
        _cleanup_cuda()

        raw_selection = select_equal_byte_segments(
            attention, accessibility, segments, middle_kv_fraction=fraction
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
        position_plans = {
            "contiguous_cf": build_retained_position_plan(
                exact_plan,
                segments,
                token_attention_mass,
                context_tokens=prompt.context_tokens,
                sparse_selector="max_mass_window",
            ),
            "global_fixed_chunk_topk": build_global_fixed_chunk_topk_position_plan(
                segments,
                token_attention_mass,
                context_tokens=prompt.context_tokens,
                target_context_charged_bytes=exact_plan.total_charged_bytes,
                chunk_width=method["global_fixed_chunk_width"],
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
        }
        if len({plan.context_charged_bytes for plan in position_plans.values()}) != 1:
            raise OracleContractError("Hotpot compressed position plans are not equal-byte")

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
        generated["full_kv_reference"] = full_generated

        eligible = [segment for segment in segments if segment.eligible]
        token_unit_bytes = eligible[0].kv_bytes // eligible[0].token_count
        query_kv_bytes = prompt.query_tokens * token_unit_bytes
        protected_bytes = sum(segment.kv_bytes for segment in segments if segment.protected)
        compressed_expected = exact_plan.total_charged_bytes + query_kv_bytes
        full_expected = sum(segment.kv_bytes for segment in segments) + query_kv_bytes
        if {
            generated[name]["post_query_resident_kv_bytes"]
            for name in EQUAL_BYTE_SYSTEMS
        } != {compressed_expected}:
            raise OracleContractError("Hotpot compressed arms miss their resident-byte target")
        if generated["full_kv_reference"]["post_query_resident_kv_bytes"] != full_expected:
            raise OracleContractError("Hotpot parent Full-KV byte accounting changed")

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

        protected_ids = {segment.segment_id for segment in segments if segment.protected}
        middle_retained_tokens = exact_plan.middle_charged_bytes // token_unit_bytes
        row = {
            "stage": args.stage_set,
            "sample_id": sample.sample_id,
            "dataset": sample.dataset,
            "base_index": case["base_index"],
            "donor_index": case["donor_index"],
            "question": sample.question,
            "answers": get_ground_truths(sample),
            "construction": construction,
            "context_tokens": prompt.context_tokens,
            "query_tokens": prompt.query_tokens,
            "budget_fraction": fraction,
            "byte_accounting": {
                "token_kv_bytes": token_unit_bytes,
                "query_kv_bytes": query_kv_bytes,
                "protected_context_kv_bytes": protected_bytes,
                "compressed_post_query_cap_bytes": compressed_cap,
                "equal_byte_post_query_resident_bytes": compressed_expected,
                "full_post_query_resident_bytes": full_expected,
            },
            "raw_alpha_selection": raw_selection,
            "query_probe": query_probe_provenance,
            "plans": {
                "contiguous_cf": {
                    "allocation": exact_plan.to_dict(),
                    "retention": position_plans["contiguous_cf"].to_dict(),
                    "active_positions_sha256": retained_positions_sha256(
                        position_plans["contiguous_cf"].active_positions
                    ),
                    "action_counts": _eligible_action_counts(exact_plan, protected_ids),
                },
                "global_fixed_chunk_topk": {
                    "chunk_width": method["global_fixed_chunk_width"],
                    "boundary_slack_tokens": (
                        middle_retained_tokens % method["global_fixed_chunk_width"]
                    ),
                    "retention": position_plans["global_fixed_chunk_topk"].to_dict(),
                    "active_positions_sha256": retained_positions_sha256(
                        position_plans["global_fixed_chunk_topk"].active_positions
                    ),
                },
                "raw_alpha_exact_slack": {
                    "selected_exact_segment_ids": list(
                        raw_selection["raw_alpha_segment_ids"]
                    ),
                    "retention": position_plans["raw_alpha_exact_slack"].to_dict(),
                    "active_positions_sha256": retained_positions_sha256(
                        position_plans["raw_alpha_exact_slack"].active_positions
                    ),
                },
                "scattered_cf": {
                    "allocation": exact_plan.to_dict(),
                    "retention": position_plans["scattered_cf"].to_dict(),
                    "active_positions_sha256": retained_positions_sha256(
                        position_plans["scattered_cf"].active_positions
                    ),
                    "action_counts": _eligible_action_counts(exact_plan, protected_ids),
                },
            },
            "systems": generated,
            "elapsed_seconds": time.perf_counter() - case_started,
        }
        rows.append(row)
        _append_jsonl(results_path, row)
        print(
            f"[{case_index}/{case_count}] {sample.sample_id}: "
            + " ".join(
                f"{name}={generated[name]['official_qa_f1']:.4f}" for name in SYSTEMS
            ),
            flush=True,
        )

    expected_ids = {
        f"hotpotqa_32k_aug_b{case['base_index']:04d}_d{case['donor_index']:04d}"
        for case in selected_cases
    }
    if {str(row["sample_id"]) for row in rows} != expected_ids:
        raise OracleContractError("Hotpot paired run did not complete the selected package")
    rows.sort(key=lambda row: (row["base_index"], row["donor_index"]))
    summary = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete",
        "scope": (
            "operational_smoke_excluded_from_claims"
            if args.stage_set == "smoke"
            else "four_case_real_task_paired_pilot"
        ),
        "manifest_id": manifest["manifest_id"],
        "protocol_sha256": protocol_sha,
        "solvability_protocol_sha256": solvability_sha,
        "analysis": summarize_results(rows),
        "longest_base_selection_audit": longest_base_audit,
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
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--solvability-protocol", required=True)
    parser.add_argument("--full-kv-results", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--probe-cache-dir", required=True)
    parser.add_argument("--stage-set", choices=tuple(STAGE_CASE_COUNTS), default="formal")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    payload = run(parse_args())
    print(json.dumps({key: value for key, value in payload.items() if key != "samples"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
