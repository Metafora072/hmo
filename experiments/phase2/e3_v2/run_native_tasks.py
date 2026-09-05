"""Frozen native LongBench QA evaluation for HMO and equal-byte baselines."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import zipfile
from pathlib import Path
from typing import Mapping, Sequence

import torch

from experiments.phase2.e3_v2.attention_probe import collect_attention_token_probe
from experiments.phase2.e3_v2.attention_probe_cache import (
    ATTENTION_PROBE_AGGREGATION,
    ATTENTION_PROBE_CACHE_SCHEMA,
    get_or_create_attention_probe,
    retained_positions_sha256,
)
from experiments.phase2.e3_v2.c3_protocol import (
    C3_MODEL_ID,
    C3_SCHEMA,
    load_c3_protocol,
    native_protocol_view,
)
from experiments.phase2.e3_v2.chunkkv_adapter import (
    CHUNKKV_ADAPTER_SCHEMA,
    CHUNKKV_CHUNK_SIZE,
    build_chunkkv_plan,
    make_chunkkv_intervention,
)
from experiments.phase2.e3_v2.context_query import (
    full_kv_intervention,
    tokenize_sample_prompt_aligned,
)
from experiments.phase2.e3_v2.coverage_fidelity import allocate_coverage_fidelity
from experiments.phase2.e3_v2.coverage_fidelity_cache import (
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
from experiments.phase2.e3_v2.run_end_task import select_equal_byte_segments
from experiments.phase2.e3_v2.run_hotpot_paired import (
    METRICS,
    _generate_system,
    _load_completed,
    summarize_results,
)
from experiments.phase2.e3_v2.run_hotpot_solvability import _sha256_file
from experiments.utils.dataset_utils import EvalSample
from experiments.utils.eval_harness import get_ground_truths
from experiments.utils.model_loader import (
    get_full_attention_indices,
    get_linear_attention_indices,
)
from experiments.utils.run_manifest import collect_environment, ensure_run_manifest
from experiments.vendor.longbench_metrics import LONG_BENCH_REVISION


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_SCHEMA = "hmo.native_longbench_qa_protocol.v1"
SIX_TASK_PROTOCOL_SCHEMA = "hmo.native_longbench_six_task_protocol.v1"
RESULT_SCHEMA = "hmo.native_longbench_qa_result.v1"
RESULTS_FILENAME = "native_longbench_results.jsonl"
SUMMARY_FILENAME = "native_longbench_summary.json"
SYSTEMS = (
    "contiguous_cf",
    "chunkkv",
    "global_fixed_chunk_topk",
    "raw_alpha_exact_slack",
    "full_kv_reference",
)
EQUAL_BYTE_SYSTEMS = SYSTEMS[:-1]
DATASET_ORDER = ("hotpotqa", "narrativeqa")
STAGE_CASE_COUNTS = {"smoke": 2, "formal": 24}
SIX_TASK_MODEL_ID = "Qwen/Qwen3.5-9B"
SIX_TASK_MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
SIX_TASK_ORDER = (
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_native_protocol(path: Path) -> tuple[dict, str]:
    try:
        encoded = path.read_bytes()
        payload = json.loads(encoded)
    except (OSError, json.JSONDecodeError) as exc:
        raise OracleContractError(f"cannot read native LongBench protocol: {path}") from exc

    is_c3 = payload.get("schema_version") == C3_SCHEMA
    is_six_task = payload.get("schema_version") == SIX_TASK_PROTOCOL_SCHEMA
    if is_c3:
        try:
            c3_payload, protocol_sha, parent = load_c3_protocol(path, PROJECT_ROOT)
        except ValueError as exc:
            raise OracleContractError(str(exc)) from exc
        payload = native_protocol_view(c3_payload, parent)
    else:
        protocol_sha = _sha256_bytes(encoded)

    method = payload.get("method", {})
    selection = payload.get("selection", {})
    execution = payload.get("execution", {})
    datasets = payload.get("datasets", {})
    generation = payload.get("generation", {})
    expected_comparisons = [
        ["contiguous_cf", "chunkkv"],
        ["contiguous_cf", "global_fixed_chunk_topk"],
        ["contiguous_cf", "raw_alpha_exact_slack"],
        ["contiguous_cf", "full_kv_reference"],
    ]
    expected_method = {
        "allocator": "attention_led",
        "sparse_selector": "max_mass_window",
        "sparse_width": 16,
        "chunkkv_chunk_size": 10,
        "chunkkv_observation": "query_suffix_attention",
        "chunkkv_layer_policy": "independent_per_full_layer_shared_across_kv_heads",
        "chunkkv_partial_chunk": "fixed_prefix_of_next_ranked_chunk",
        "raw_slack_selector": "global_top_tokens_slack",
        "global_fixed_chunk_width": 16,
        "global_fixed_chunk_slack": "prefix_of_next_ranked_chunk",
        "segment_length": 256,
        "protected_prefix_segments": 1,
        "protected_suffix_segments": 1,
    }
    if is_six_task:
        selection_expected = {
            "rule": "longest_exact_serialized_memory_context_within_inclusive_token_band_then_record_index",
            "min_memory_context_tokens": 1,
            "max_memory_context_tokens": 16384,
            "maximum_samples_per_dataset": 100,
            "source_context_unchanged": True,
            "augmentation": False,
            "truncation": False,
            "outcome_conditioned_selection": False,
            "boundary_alignment": "if_one_token_crosses_the_semantic_context_boundary_move_that_complete_token_into_memory_context",
            "record_sha256_semantics": "sha256_of_raw_jsonl_record_bytes_without_line_ending",
        }
        stage_sets_expected = {
            "prefix50": {
                "per_dataset_prefix": 50,
                "manifest_group": "prefix100",
                "manifest_per_dataset_prefix": 100,
            },
            "prefix100": {
                "per_dataset_prefix": 100,
                "manifest_group": "prefix100",
                "manifest_per_dataset_prefix": 100,
            },
        }
        expected_counts = {
            "narrativeqa": 61,
            "qasper": 100,
            "multifieldqa_en": 100,
            "hotpotqa": 100,
            "2wikimqa": 100,
            "musique": 45,
        }
        expected_record_counts = {name: 200 for name in expected_counts}
        expected_record_counts["multifieldqa_en"] = 150
        if (
            payload.get("status") != "frozen_before_outcomes"
            or payload.get("purpose")
            != "broad_native_non_augmented_real_task_main_table"
            or payload.get("model_id") != SIX_TASK_MODEL_ID
            or payload.get("model_revision") != SIX_TASK_MODEL_REVISION
            or tuple(payload.get("dataset_order", ())) != SIX_TASK_ORDER
            or tuple(payload.get("systems", ())) != SYSTEMS
            or tuple(payload.get("equal_byte_systems", ())) != EQUAL_BYTE_SYSTEMS
            or payload.get("primary_comparisons") != expected_comparisons
            or payload.get("primary_metric") != "official_qa_f1"
            or tuple(payload.get("secondary_metrics", ())) != METRICS[1:]
            or float(payload.get("middle_kv_fraction", 0.0)) != 0.1
            or method != expected_method
            or selection != selection_expected
            or payload.get("stage_sets") != stage_sets_expected
            or generation.get("decoding") != "greedy"
            or int(generation.get("inference_seed", 0)) <= 0
            or generation.get("max_new_tokens")
            != {
                "narrativeqa": 128,
                "qasper": 128,
                "multifieldqa_en": 64,
                "hotpotqa": 32,
                "2wikimqa": 32,
                "musique": 32,
            }
            or tuple(datasets.get(name, {}) and name for name in SIX_TASK_ORDER)
            != SIX_TASK_ORDER
        ):
            raise OracleContractError("six-task native LongBench protocol mismatch")
        for name, count in expected_counts.items():
            spec = datasets[name]
            cases = spec.get("cases", [])
            if (
                spec.get("member") != f"data/{name}.jsonl"
                or spec.get("record_count") != expected_record_counts[name]
                or spec.get("official_metric") != "qa_f1_score"
                or len(cases) != count
                or len({case.get("index") for case in cases}) != count
            ):
                raise OracleContractError(f"six-task dataset mismatch: {name}")
        execution_expected = {
            "prefix50_case_count": 295,
            "prefix100_case_count": 506,
            "prefix50_generation_cells": 1475,
            "prefix100_generation_cells": 2530,
            "continuation_is_precommitted_prefix": True,
            "continuation_gate": False,
            "resume_after_interruption": True,
            "case_filtering_after_outcomes": False,
        }
        if payload.get("execution") != execution_expected:
            raise OracleContractError("six-task execution count mismatch")
        return payload, protocol_sha

    if (
        payload.get("schema_version") != (C3_SCHEMA if is_c3 else PROTOCOL_SCHEMA)
        or payload.get("status")
        != ("frozen_before_27b_outcomes" if is_c3 else "frozen_before_outcomes")
        or payload.get("purpose") != "native_non_augmented_real_task_external_validity"
        or payload.get("model_id")
        != (C3_MODEL_ID if is_c3 else "Qwen/Qwen3.5-0.8B")
        or not payload.get("model_revision")
        or tuple(payload.get("systems", ())) != SYSTEMS
        or tuple(payload.get("equal_byte_systems", ())) != EQUAL_BYTE_SYSTEMS
        or payload.get("primary_comparisons") != expected_comparisons
        or payload.get("primary_metric") != "official_qa_f1"
        or tuple(payload.get("secondary_metrics", ())) != METRICS[1:]
        or float(payload.get("middle_kv_fraction", 0.0)) != 0.1
        or method != expected_method
        or selection
        != {
            "rule": "longest_exact_serialized_memory_context_within_inclusive_token_band_then_record_index",
            "min_memory_context_tokens": 8192,
            "max_memory_context_tokens": 16384,
            "samples_per_dataset": 12,
            "source_context_unchanged": True,
            "augmentation": False,
            "truncation": False,
            "outcome_conditioned_selection": False,
            "boundary_alignment": "if_one_token_crosses_the_semantic_context_boundary_move_that_complete_token_into_memory_context",
            "record_sha256_semantics": "sha256_of_raw_jsonl_record_bytes_without_line_ending",
        }
        or execution
        != {
            "smoke_case_count": 2,
            "formal_case_count": 24,
            "smoke_selection": "first_frozen_case_per_dataset",
            "continuation_gate": False,
            "resume_after_interruption": True,
            "case_filtering_after_outcomes": False,
        }
        or generation.get("decoding") != "greedy"
        or int(generation.get("inference_seed", 0)) <= 0
        or generation.get("max_new_tokens") != {"hotpotqa": 32, "narrativeqa": 128}
        or payload.get("dataset_source", {}).get("metric_revision") != LONG_BENCH_REVISION
        or len(payload.get("dataset_source", {}).get("archive_sha256", "")) != 64
        or tuple(datasets) != DATASET_ORDER
    ):
        raise OracleContractError("native LongBench protocol mismatch")

    for name in DATASET_ORDER:
        spec = datasets[name]
        cases = spec.get("cases", [])
        if (
            spec.get("member") != f"data/{name}.jsonl"
            or spec.get("record_count") != 200
            or spec.get("official_metric") != "qa_f1_score"
            or len(cases) != selection["samples_per_dataset"]
            or len({case.get("index") for case in cases}) != len(cases)
            or any(
                not isinstance(case.get(key), expected)
                for case in cases
                for key, expected in (
                    ("index", int),
                    ("record_sha256", str),
                    ("context_tokens", int),
                    ("query_tokens", int),
                    ("boundary_shift_characters", int),
                )
            )
        ):
            raise OracleContractError(f"native LongBench dataset mismatch: {name}")
    return payload, protocol_sha


def select_longest_candidates(
    metadata: Sequence[Mapping], minimum: int, maximum: int, count: int
) -> list[dict]:
    eligible = [
        dict(item)
        for item in metadata
        if minimum <= int(item["context_tokens"]) <= maximum
    ]
    return sorted(eligible, key=lambda item: (-int(item["context_tokens"]), int(item["index"])))[:count]


def _dataset_order(protocol: Mapping) -> tuple[str, ...]:
    return tuple(protocol.get("dataset_order", DATASET_ORDER))


def _load_datasets(archive: Path, protocol: Mapping) -> dict[str, tuple[list[dict], list[bytes]]]:
    expected_sha = protocol["dataset_source"]["archive_sha256"]
    observed_sha = _sha256_file(archive)
    if observed_sha != expected_sha:
        raise OracleContractError(f"LongBench archive SHA mismatch: {observed_sha} != {expected_sha}")
    output = {}
    try:
        with zipfile.ZipFile(archive) as handle:
            for name in _dataset_order(protocol):
                raw_lines = handle.read(protocol["datasets"][name]["member"]).splitlines()
                records = [json.loads(line) for line in raw_lines if line.strip()]
                if len(records) != len(raw_lines):
                    raise OracleContractError(f"LongBench {name} contains blank records")
                if len(records) != protocol["datasets"][name]["record_count"]:
                    raise OracleContractError(f"LongBench {name} record count changed")
                output[name] = (records, raw_lines)
    except (OSError, KeyError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        raise OracleContractError(f"cannot load pinned LongBench archive: {archive}") from exc
    return output


def _make_sample(dataset: str, index: int, record: Mapping) -> EvalSample:
    answers = [str(value) for value in record.get("answers", []) if str(value)]
    if not answers:
        raise OracleContractError(f"LongBench {dataset}[{index}] has no answers")
    return EvalSample(
        dataset=f"longbench_{dataset}",
        sample_id=f"longbench_{dataset}_{index:04d}",
        context=str(record["context"]),
        question=str(record["input"]),
        answer=answers[0],
        answers=answers,
        context_length=0,
    )


def validate_frozen_selection(
    dataset: str,
    records: Sequence[Mapping],
    raw_lines: Sequence[bytes],
    spec: Mapping,
    selection: Mapping,
    tokenizer,
) -> tuple[dict[int, tuple[EvalSample, object, int]], dict]:
    materialized = {}
    metadata = []
    for index, record in enumerate(records):
        sample = _make_sample(dataset, index, record)
        prompt, shift = tokenize_sample_prompt_aligned(sample, tokenizer)
        materialized[index] = (sample, prompt, shift)
        metadata.append(
            {
                "index": index,
                "record_sha256": _sha256_bytes(raw_lines[index]),
                "context_tokens": prompt.context_tokens,
                "query_tokens": prompt.query_tokens,
                "boundary_shift_characters": shift,
            }
        )
    expected = select_longest_candidates(
        metadata,
        selection["min_memory_context_tokens"],
        selection["max_memory_context_tokens"],
        selection.get(
            "samples_per_dataset", selection.get("maximum_samples_per_dataset", 0)
        ),
    )
    if expected != spec["cases"]:
        raise OracleContractError(f"frozen {dataset} selection no longer reproduces")
    return materialized, {
        "record_count": len(records),
        "eligible_count": sum(
            selection["min_memory_context_tokens"] <= row["context_tokens"] <= selection["max_memory_context_tokens"]
            for row in metadata
        ),
        "selected": expected,
    }


def _selected_cases(protocol: Mapping, stage_set: str) -> list[tuple[str, Mapping]]:
    order = _dataset_order(protocol)
    if protocol.get("schema_version") == SIX_TASK_PROTOCOL_SCHEMA:
        stage = protocol["stage_sets"].get(stage_set)
        if not isinstance(stage, Mapping):
            raise OracleContractError(f"unknown six-task stage set: {stage_set}")
        count = int(stage["per_dataset_prefix"])
        return [
            (name, case)
            for name in order
            for case in protocol["datasets"][name]["cases"][:count]
        ]
    if stage_set == "smoke":
        return [(name, protocol["datasets"][name]["cases"][0]) for name in order]
    return [
        (name, case)
        for name in order
        for case in protocol["datasets"][name]["cases"]
    ]


def summarize_native_results(
    rows: Sequence[Mapping], dataset_order: Sequence[str] = DATASET_ORDER
) -> dict:
    analysis = summarize_results(rows, SYSTEMS, EQUAL_BYTE_SYSTEMS)
    analysis["by_dataset"] = {
        name: summarize_results(
            [row for row in rows if row["dataset"] == f"longbench_{name}"],
            SYSTEMS,
            EQUAL_BYTE_SYSTEMS,
        )
        for name in dataset_order
    }
    return analysis


def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("native LongBench evaluation requires exactly one visible GPU")
    protocol, protocol_sha = load_native_protocol(Path(args.protocol).resolve())
    if args.model_id != protocol["model_id"] or args.model_revision != protocol["model_revision"]:
        raise OracleContractError("model identity disagrees with native LongBench protocol")

    archive = Path(args.archive).resolve()
    datasets = _load_datasets(archive, protocol)
    selected_cases = _selected_cases(protocol, args.stage_set)
    dataset_order = _dataset_order(protocol)
    if protocol.get("schema_version") == SIX_TASK_PROTOCOL_SCHEMA:
        stage_spec = protocol["stage_sets"][args.stage_set]
        manifest_stage_set = str(stage_spec["manifest_group"])
        manifest_cases = [
            (name, case)
            for name in dataset_order
            for case in protocol["datasets"][name]["cases"][
                : int(stage_spec["manifest_per_dataset_prefix"])
            ]
        ]
    else:
        if args.stage_set not in STAGE_CASE_COUNTS or len(selected_cases) != STAGE_CASE_COUNTS[args.stage_set]:
            raise OracleContractError("native LongBench stage case count changed")
        manifest_stage_set = args.stage_set
        manifest_cases = selected_cases
    run_dir = Path(args.run_dir).resolve()
    probe_cache_dir = Path(args.probe_cache_dir).resolve()
    model_path = Path(args.model_path).resolve()
    model_identity = model_provenance(model_path, args.model_id, revision=args.model_revision)
    manifest = ensure_run_manifest(
        run_dir,
        experiment="native_longbench_qa_equal_byte_paired",
        args={
            "archive": str(archive),
            "archive_sha256": protocol["dataset_source"]["archive_sha256"],
            "model_id": args.model_id,
            "model_revision": args.model_revision,
            "protocol_sha256": protocol_sha,
            "probe_cache_dir": str(probe_cache_dir),
            "stage_set": manifest_stage_set,
            "inference_seed": protocol["generation"]["inference_seed"],
            "recurrent_backend": REFERENCE_BACKEND,
            "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        selections={
            "scope": "operational_smoke_excluded_from_claims" if args.stage_set == "smoke" else "native_non_augmented_real_task_external_validity",
            "systems": list(SYSTEMS),
            "equal_byte_systems": list(EQUAL_BYTE_SYSTEMS),
            "primary_metric": protocol["primary_metric"],
            "middle_kv_fraction": protocol["middle_kv_fraction"],
            "method": protocol["method"],
            "cases": [{"dataset": name, **case} for name, case in manifest_cases],
            "candidate_search": False,
            "outcome_conditioned_selection": False,
            "query_probe_cache_schema": ATTENTION_PROBE_CACHE_SCHEMA,
            "query_probe_aggregation": ATTENTION_PROBE_AGGREGATION,
            "chunkkv_adapter_schema": CHUNKKV_ADAPTER_SCHEMA,
        },
        model=model_identity,
        project_root=PROJECT_ROOT,
        require_clean=True,
        environment=collect_environment(),
    )

    results_path = run_dir / RESULTS_FILENAME
    completed = _load_completed(results_path)
    if completed and not args.resume:
        raise OracleContractError("native LongBench results exist; pass --resume to continue")
    rows = list(completed)
    completed_ids = {str(row["sample_id"]) for row in completed}

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
        raise OracleContractError("native LongBench model is not a Hybrid architecture")
    _force_torch_reference_backend(model, recurrent_layers)

    materialized = {}
    selection_audit = {}
    for name in dataset_order:
        records, raw_lines = datasets[name]
        materialized[name], selection_audit[name] = validate_frozen_selection(
            name,
            records,
            raw_lines,
            protocol["datasets"][name],
            protocol["selection"],
            tokenizer,
        )

    method = protocol["method"]
    fraction = float(protocol["middle_kv_fraction"])
    for case_index, (dataset, case) in enumerate(selected_cases, start=1):
        sample, prompt, boundary_shift = materialized[dataset][case["index"]]
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
        del context_outputs
        _cleanup_cuda()

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
        probe = cached_probe.result
        attention = probe.alpha.as_dict()
        token_attention_mass = tuple(float(value) for value in probe.token_attention_mass)
        layer_token_attention_mass = probe.layer_scores()
        allocator_attention = {
            segment.segment_id: float(attention[segment.segment_id])
            for segment in segments
            if segment.eligible
        }
        del probe
        _cleanup_cuda()

        raw_selection = select_equal_byte_segments(
            attention, None, segments, middle_kv_fraction=fraction
        )
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
        position_plans = {
            "contiguous_cf": build_retained_position_plan(
                exact_plan, segments, token_attention_mass,
                context_tokens=prompt.context_tokens, sparse_selector="max_mass_window"
            ),
            "global_fixed_chunk_topk": build_global_fixed_chunk_topk_position_plan(
                segments, token_attention_mass,
                context_tokens=prompt.context_tokens,
                target_context_charged_bytes=exact_plan.total_charged_bytes,
                chunk_width=method["global_fixed_chunk_width"],
            ),
            "raw_alpha_exact_slack": build_raw_exact_slack_position_plan(
                segments, raw_selection["raw_alpha_segment_ids"], token_attention_mass,
                context_tokens=prompt.context_tokens,
                target_context_charged_bytes=exact_plan.total_charged_bytes,
            ),
        }
        chunkkv_plan = build_chunkkv_plan(
            segments,
            layer_token_attention_mass,
            context_tokens=prompt.context_tokens,
            target_context_charged_bytes=exact_plan.total_charged_bytes,
            context_token_kv_bytes=token_unit_bytes,
            observation_query_tokens=prompt.query_tokens,
            chunk_size=CHUNKKV_CHUNK_SIZE,
        )
        if len({
            plan.context_charged_bytes
            for plan in (*position_plans.values(), chunkkv_plan)
        }) != 1:
            raise OracleContractError("native LongBench compressed plans are not equal-byte")

        max_new_tokens = protocol["generation"]["max_new_tokens"][dataset]
        interventions = {
            name: make_coverage_fidelity_intervention(
                position_plans[name], attention_layers, name=name
            )
            for name in position_plans
        }
        interventions["chunkkv"] = make_chunkkv_intervention(chunkkv_plan)
        generated = {
            name: _generate_system(
                model, tokenizer, prompt, attention_layers, recurrent_layers,
                interventions[name],
                sample, max_new_tokens,
            )
            for name in EQUAL_BYTE_SYSTEMS
        }
        generated["full_kv_reference"] = _generate_system(
            model, tokenizer, prompt, attention_layers, recurrent_layers,
            full_kv_intervention, sample, max_new_tokens,
        )

        query_kv_bytes = prompt.query_tokens * token_unit_bytes
        protected_bytes = sum(segment.kv_bytes for segment in segments if segment.protected)
        compressed_expected = exact_plan.total_charged_bytes + query_kv_bytes
        full_expected = sum(segment.kv_bytes for segment in segments) + query_kv_bytes
        if {generated[name]["post_query_resident_kv_bytes"] for name in EQUAL_BYTE_SYSTEMS} != {compressed_expected}:
            raise OracleContractError("native LongBench compressed arms miss resident-byte target")
        if generated["full_kv_reference"]["post_query_resident_kv_bytes"] != full_expected:
            raise OracleContractError("native LongBench Full-KV byte accounting changed")
        compressed_cap = exact_plan.total_budget_limit_bytes + query_kv_bytes
        for name in EQUAL_BYTE_SYSTEMS:
            generated[name].update({
                "post_query_budget_limit_bytes": compressed_cap,
                "expected_post_query_resident_kv_bytes": compressed_expected,
            })
        generated["full_kv_reference"].update({
            "post_query_budget_limit_bytes": full_expected,
            "expected_post_query_resident_kv_bytes": full_expected,
        })

        protected_ids = {segment.segment_id for segment in segments if segment.protected}
        middle_retained_tokens = exact_plan.middle_charged_bytes // token_unit_bytes
        row = {
            "stage": args.stage_set,
            "sample_id": sample.sample_id,
            "dataset": sample.dataset,
            "record_index": case["index"],
            "record_sha256": case["record_sha256"],
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
            "query_probe": cached_probe.provenance(),
            "plans": {
                "contiguous_cf": {
                    "allocation": exact_plan.to_dict(),
                    "retention": position_plans["contiguous_cf"].to_dict(),
                    "active_positions_sha256": retained_positions_sha256(position_plans["contiguous_cf"].active_positions),
                    "action_counts": _eligible_action_counts(exact_plan, protected_ids),
                },
                "chunkkv": chunkkv_plan.to_dict(),
                "global_fixed_chunk_topk": {
                    "chunk_width": method["global_fixed_chunk_width"],
                    "boundary_slack_tokens": middle_retained_tokens % method["global_fixed_chunk_width"],
                    "retention": position_plans["global_fixed_chunk_topk"].to_dict(),
                    "active_positions_sha256": retained_positions_sha256(position_plans["global_fixed_chunk_topk"].active_positions),
                },
                "raw_alpha_exact_slack": {
                    "selected_exact_segment_ids": list(raw_selection["raw_alpha_segment_ids"]),
                    "retention": position_plans["raw_alpha_exact_slack"].to_dict(),
                    "active_positions_sha256": retained_positions_sha256(position_plans["raw_alpha_exact_slack"].active_positions),
                },
            },
            "systems": generated,
            "elapsed_seconds": time.perf_counter() - case_started,
        }
        rows.append(row)
        _append_jsonl(results_path, row)
        print(
            f"[{case_index}/{len(selected_cases)}] {sample.sample_id}: "
            + " ".join(f"{name}={generated[name]['official_qa_f1']:.4f}" for name in SYSTEMS),
            flush=True,
        )

    expected_ids = {f"longbench_{name}_{case['index']:04d}" for name, case in selected_cases}
    allowed_ids = {f"longbench_{name}_{case['index']:04d}" for name, case in manifest_cases}
    actual_ids = {str(row["sample_id"]) for row in rows}
    if not expected_ids <= actual_ids or not actual_ids <= allowed_ids:
        raise OracleContractError("native LongBench run did not complete selected package")
    rows.sort(key=lambda row: (dataset_order.index(row["dataset"].removeprefix("longbench_")), row["record_index"]))
    summary = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete" if actual_ids == allowed_ids else "stage_complete",
        "scope": "operational_smoke_excluded_from_claims" if args.stage_set == "smoke" else "native_non_augmented_real_task_external_validity",
        "manifest_id": manifest["manifest_id"],
        "protocol_sha256": protocol_sha,
        "analysis": summarize_native_results(rows, dataset_order),
        "selection_audit": selection_audit,
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
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--probe-cache-dir", required=True)
    parser.add_argument("--stage-set", default="formal")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    payload = run(parse_args())
    print(json.dumps({key: value for key, value in payload.items() if key != "samples"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
