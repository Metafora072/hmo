"""Development-only CF-HMO diagnosis on the frozen P3 changed set."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from argparse import Namespace
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from experiments.phase2.e3_v2.context_query import (
    full_kv_intervention,
    run_post_intervention_prompt,
    score_gold_answer_logprob,
    tokenize_answer_continuation,
    tokenize_sample_prompt,
)
from experiments.phase2.e3_v2.coverage_fidelity_cache import (
    select_query_attention_positions,
)
from experiments.phase2.e3_v2.oracle import (
    OracleConfig,
    OracleContractError,
    SegmentSpec,
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
from experiments.phase2.e3_v2.run_discovery import _build_samples
from experiments.phase2.e3_v2.run_end_task import (
    PROTOCOL_SCHEMA,
    _load_protocol,
    make_selected_segment_intervention,
    select_equal_byte_segments,
)
from experiments.utils.kv_ops import select_token_skeleton_positions
from experiments.utils.memory_accounting import get_active_kv_bytes
from experiments.utils.model_loader import (
    get_full_attention_indices,
    get_linear_attention_indices,
)
from experiments.utils.run_manifest import collect_environment, ensure_run_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_FILENAME = "cf_diagnosis_results.jsonl"
SUMMARY_FILENAME = "cf_diagnosis_summary.json"
SCHEMA_VERSION = "hmo.coverage_fidelity.diagnosis.v1"
SELECTORS = ("kv_norm", "query_attention")
SYSTEMS = ("raw_alpha", "frozen_v2", "full_kv_reference")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _load_json(path: Path, description: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OracleContractError(f"cannot read {description}: {path}") from exc
    if not isinstance(value, dict):
        raise OracleContractError(f"{description} must contain a JSON object")
    return value


def load_changed_cases(
    source_paths: Sequence[Path],
    *,
    protocol: Mapping,
    protocol_sha256: str,
    expected_count: int,
) -> tuple[list[dict], list[dict]]:
    """Load and provenance-check the immutable P3 membership-changed rows."""
    if len(source_paths) != 2:
        raise OracleContractError("D0 requires exactly the frozen 8K and 16K summaries")
    cases = []
    sources = []
    seen_stages = set()
    seen_ids = set()
    stage_keys = (
        "datasets",
        "samples_per_dataset",
        "context_length",
        "segment_length",
        "middle_kv_fraction",
        "seed",
        "sample_id_prefix",
    )
    for source_path in source_paths:
        source_path = source_path.resolve()
        summary = _load_json(source_path, "P3 end-task summary")
        manifest_path = source_path.parent / "run_manifest.json"
        manifest = _load_json(manifest_path, "P3 run manifest")
        stage = str(summary.get("stage"))
        run_spec = manifest.get("run_spec", {})
        arguments = run_spec.get("arguments", {})
        if (
            summary.get("schema_version")
            != "hmo.query_accessibility.end_task_result.v1"
            or summary.get("status") != "complete"
            or stage not in {"8k", "16k"}
            or stage in seen_stages
            or summary.get("protocol_sha256") != protocol_sha256
            or summary.get("manifest_id") != manifest.get("manifest_id")
            or run_spec.get("experiment") != "query_accessibility_v2_end_task"
            or arguments.get("stage") != stage
        ):
            raise OracleContractError(f"P3 source contract mismatch: {source_path}")
        frozen_stage = protocol["stages"][stage]
        if any(arguments.get(key) != frozen_stage[key] for key in stage_keys):
            raise OracleContractError(f"P3 source stage arguments changed: {source_path}")
        if (
            arguments.get("model_id") != protocol["model_id"]
            or arguments.get("model_revision") != protocol["model_revision"]
        ):
            raise OracleContractError(f"P3 source model identity changed: {source_path}")
        rows = summary.get("samples", [])
        if len(rows) != int(summary.get("sample_count", -1)):
            raise OracleContractError(f"P3 source sample count mismatch: {source_path}")
        for row in rows:
            selection = row.get("selection", {})
            if not selection.get("membership_changed"):
                continue
            sample_id = str(row.get("sample_id"))
            if sample_id in seen_ids or row.get("dataset") != "longeval_lines":
                raise OracleContractError("D0 changed set must contain unique LongEval rows")
            seen_ids.add(sample_id)
            cases.append({"stage": stage, "source_row": row})
        sources.append(
            {
                "stage": stage,
                "summary_path": str(source_path),
                "summary_sha256": _sha256(source_path),
                "manifest_path": str(manifest_path),
                "manifest_sha256": _sha256(manifest_path),
                "manifest_id": manifest["manifest_id"],
                "code_commit": run_spec.get("code", {}).get("commit"),
            }
        )
        seen_stages.add(stage)
    if seen_stages != {"8k", "16k"} or len(cases) != expected_count:
        raise OracleContractError(
            f"expected {expected_count} changed cases across 8K/16K, found {len(cases)}"
        )
    cases.sort(key=lambda item: (item["stage"], item["source_row"]["sample_id"]))
    sources.sort(key=lambda item: item["stage"])
    return cases, sources


def locate_answer_token_positions(tokenizer, prompt, answer: str) -> dict:
    """Map the unique answer occurrence in memory context to context token positions."""
    context = prompt.text.memory_context
    starts = []
    cursor = 0
    while True:
        start = context.find(answer, cursor)
        if start < 0:
            break
        starts.append(start)
        cursor = start + 1
    if len(starts) != 1:
        raise OracleContractError(
            f"answer must occur exactly once in memory context, found {len(starts)}"
        )
    char_start = starts[0]
    char_end = char_start + len(answer)
    encoded = tokenizer(
        prompt.text.full_prompt,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    token_ids = encoded["input_ids"]
    if not torch.equal(token_ids, prompt.full_ids):
        raise OracleContractError("answer mapping tokenizer pass changed full prompt token ids")
    offsets = encoded["offset_mapping"][0, : prompt.context_tokens].tolist()
    positions = [
        index
        for index, (start, end) in enumerate(offsets)
        if int(start) < char_end and int(end) > char_start
    ]
    if not positions:
        raise OracleContractError("answer character span maps to no context tokens")
    return {
        "answer_char_span": [char_start, char_end],
        "answer_token_positions": positions,
    }


def build_survival_record(
    answer_positions: Sequence[int],
    kept_by_segment: Mapping[int, Sequence[int]],
) -> dict:
    answer_set = set(int(value) for value in answer_positions)
    kept = sorted({int(value) for values in kept_by_segment.values() for value in values})
    retained = sorted(answer_set.intersection(kept))
    return {
        "kept_positions_by_answer_segment": {
            str(segment_id): [int(value) for value in values]
            for segment_id, values in sorted(kept_by_segment.items())
        },
        "answer_token_count": len(answer_set),
        "answer_tokens_retained": retained,
        "answer_token_retained_count": len(retained),
        "answer_token_retained_fraction": len(retained) / len(answer_set),
        "any_answer_token_survived": bool(retained),
        "all_answer_tokens_survived": len(retained) == len(answer_set),
    }


def summarize_diagnosis(rows: Sequence[Mapping], widths: Sequence[int]) -> dict:
    if not rows:
        raise OracleContractError("D0 diagnosis requires result rows")
    teacher = {}
    for system in SYSTEMS:
        values = [row["systems"][system]["mean_logprob"] for row in rows]
        teacher[system] = {"mean_gold_logprob": float(np.mean(values))}
    teacher["frozen_v2_minus_raw_alpha"] = {
        "mean_delta": float(
            np.mean(
                [
                    row["systems"]["frozen_v2"]["mean_logprob"]
                    - row["systems"]["raw_alpha"]["mean_logprob"]
                    for row in rows
                ]
            )
        )
    }
    survival = {}
    for selector in SELECTORS:
        survival[selector] = {}
        for width in widths:
            records = [row["sparse_survival"][selector][str(width)] for row in rows]
            survival[selector][str(width)] = {
                "all_answer_tokens_survived_cases": sum(
                    record["all_answer_tokens_survived"] for record in records
                ),
                "any_answer_token_survived_cases": sum(
                    record["any_answer_token_survived"] for record in records
                ),
                "mean_answer_token_retained_fraction": float(
                    np.mean(
                        [record["answer_token_retained_fraction"] for record in records]
                    )
                ),
            }
    return {
        "case_count": len(rows),
        "stages": {
            stage: sum(row["stage"] == stage for row in rows)
            for stage in sorted({row["stage"] for row in rows})
        },
        "outcomes": {
            "v2_wins": sum(row["source_outcome"]["delta"] > 0 for row in rows),
            "ties": sum(row["source_outcome"]["delta"] == 0 for row in rows),
            "v2_losses": sum(row["source_outcome"]["delta"] < 0 for row in rows),
        },
        "teacher_forced": teacher,
        "sparse_survival": survival,
    }


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
                    f"invalid persisted D0 result at line {line_number}"
                ) from exc
    keys = [(row.get("stage"), row.get("sample_id")) for row in rows]
    if len(keys) != len(set(keys)):
        raise OracleContractError("persisted D0 results contain duplicate cases")
    return rows


def _assert_selection_reproduced(actual: Mapping, stored: Mapping) -> None:
    exact_fields = (
        "eligible_segment_ids",
        "budget_limit_bytes",
        "budget_slots",
        "unit_segment_bytes",
        "gate_enabled",
        "membership_changed",
        "raw_alpha_segment_ids",
        "frozen_v2_segment_ids",
    )
    mismatches = []
    for field in exact_fields:
        actual_value = actual.get(field)
        stored_value = stored.get(field)
        if field.endswith("segment_ids"):
            actual_value = tuple(actual_value or ())
            stored_value = tuple(stored_value or ())
        if actual_value != stored_value:
            mismatches.append(field)
    if mismatches:
        raise OracleContractError(
            "D0 failed to reproduce P3 selection fields: " + ", ".join(mismatches)
        )


def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("D0 diagnosis requires exactly one visible GPU")
    widths = tuple(sorted({int(value) for value in args.sparse_widths.split(",")}))
    if not widths or any(value <= 0 for value in widths):
        raise OracleContractError("sparse widths must be positive")
    protocol_path = Path(args.protocol).resolve()
    protocol, protocol_sha = _load_protocol(protocol_path)
    if protocol.get("schema_version") != PROTOCOL_SCHEMA:
        raise OracleContractError("D0 requires the frozen P3 protocol")
    if (
        args.model_id != protocol["model_id"]
        or args.model_revision != protocol["model_revision"]
    ):
        raise OracleContractError("D0 model identity disagrees with P3 protocol")
    cases, sources = load_changed_cases(
        [Path(value) for value in args.source_result],
        protocol=protocol,
        protocol_sha256=protocol_sha,
        expected_count=args.expected_changed_cases,
    )
    model_path = Path(args.model_path).resolve()
    run_dir = Path(args.run_dir).resolve()
    model_identity = model_provenance(
        model_path, args.model_id, revision=args.model_revision
    )
    scientific_args = {
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "sparse_widths": list(widths),
        "expected_changed_cases": args.expected_changed_cases,
        "stages": ["8k", "16k"],
        "recurrent_backend": REFERENCE_BACKEND,
        "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    manifest = ensure_run_manifest(
        run_dir,
        experiment="coverage_fidelity_d0_diagnosis",
        args=scientific_args,
        selections={
            "scope": "development_only_frozen_p3_membership_changed_cases",
            "source_runs": sources,
            "protocol_sha256": protocol_sha,
            "selectors": list(SELECTORS),
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
        raise OracleContractError("D0 results exist; pass --resume to continue")
    completed_keys = {(row["stage"], row["sample_id"]) for row in completed}
    rows = list(completed)

    torch.manual_seed(20261003)
    torch.cuda.manual_seed_all(20261003)
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
        try:
            sample = samples_by_stage[stage][sample_id]
        except KeyError as exc:
            raise OracleContractError(f"cannot reconstruct P3 sample {sample_id}") from exc
        if sample.answer != source_row["answer"]:
            raise OracleContractError(f"reconstructed answer changed for {sample_id}")
        stage_config = protocol["stages"][stage]
        prompt = tokenize_sample_prompt(sample, tokenizer)
        if (
            prompt.context_tokens != source_row["context_tokens"]
            or prompt.query_tokens != source_row["query_tokens"]
        ):
            raise OracleContractError(f"reconstructed prompt changed for {sample_id}")
        answer_mapping = locate_answer_token_positions(
            tokenizer, prompt, sample.answer
        )

        with torch.no_grad():
            context_outputs = model.model(
                prompt.context_ids.to(model.device), use_cache=True, return_dict=True
            )
        full_cache = context_outputs.past_key_values
        oracle_config = OracleConfig(
            segment_length=stage_config["segment_length"],
            middle_kv_fraction=stage_config["middle_kv_fraction"],
        )
        segments = build_segment_catalog(
            full_cache,
            attention_layers,
            context_tokens=prompt.context_tokens,
            config=oracle_config,
        )
        by_segment = {segment.segment_id: segment for segment in segments}
        answer_segment_ids = sorted(
            {
                position // stage_config["segment_length"]
                for position in answer_mapping["answer_token_positions"]
            }
        )
        if any(segment_id not in by_segment for segment_id in answer_segment_ids):
            raise OracleContractError("answer token mapped outside segment catalog")
        kv_positions = {
            width: {
                segment_id: select_token_skeleton_positions(
                    full_cache,
                    list(attention_layers),
                    by_segment[segment_id].start,
                    by_segment[segment_id].end,
                    width,
                )[0]
                for segment_id in answer_segment_ids
            }
            for width in widths
        }
        del context_outputs, full_cache
        _cleanup_cuda()

        probe = collect_hybrid_query_token_probe(
            model,
            prompt,
            attention_layer_indices=attention_layers,
            recurrent_layer_indices=recurrent_layers,
            segments=segments,
            segment_length=stage_config["segment_length"],
        )
        selection = select_equal_byte_segments(
            probe.alpha.as_dict(),
            probe.accessibility.field_dict("read_share"),
            segments,
            middle_kv_fraction=stage_config["middle_kv_fraction"],
        )
        _assert_selection_reproduced(selection, source_row["selection"])
        query_positions = {
            width: {
                segment_id: select_query_attention_positions(
                    probe.token_attention_mass, by_segment[segment_id], width
                )
                for segment_id in answer_segment_ids
            }
            for width in widths
        }
        _cleanup_cuda()

        answer_ids = tokenize_answer_continuation(tokenizer, prompt, sample.answer)
        arm_specs = (
            (
                "raw_alpha",
                make_selected_segment_intervention(
                    segments,
                    attention_layers,
                    selection["raw_alpha_segment_ids"],
                    context_tokens=prompt.context_tokens,
                    name="raw_alpha_topk",
                ),
            ),
            (
                "frozen_v2",
                make_selected_segment_intervention(
                    segments,
                    attention_layers,
                    selection["frozen_v2_segment_ids"],
                    context_tokens=prompt.context_tokens,
                    name="frozen_v2_topk",
                ),
            ),
            ("full_kv_reference", full_kv_intervention),
        )
        systems = {}
        for system, intervention in arm_specs:
            state = run_post_intervention_prompt(
                model,
                prompt,
                attention_layer_indices=attention_layers,
                recurrent_layer_indices=recurrent_layers,
                intervention=intervention,
            )
            resident_bytes = get_active_kv_bytes(state.cache, list(attention_layers))
            score = score_gold_answer_logprob(model, state, answer_ids)
            systems[system] = {
                **asdict(score),
                "post_query_resident_kv_bytes": int(resident_bytes),
            }
            del state
            _cleanup_cuda()
        if (
            systems["raw_alpha"]["post_query_resident_kv_bytes"]
            != systems["frozen_v2"]["post_query_resident_kv_bytes"]
        ):
            raise OracleContractError("D0 raw alpha and V2 arms are not equal-byte")

        raw_correct = float(
            source_row["systems"]["raw_alpha"]["normalized_answer_contains"]
        )
        v2_correct = float(
            source_row["systems"]["frozen_v2"]["normalized_answer_contains"]
        )
        row = {
            "stage": stage,
            "sample_id": sample_id,
            "dataset": sample.dataset,
            "answer": sample.answer,
            "context_tokens": prompt.context_tokens,
            "query_tokens": prompt.query_tokens,
            **answer_mapping,
            "answer_segment_ids": answer_segment_ids,
            "selection": selection,
            "source_outcome": {
                "raw_alpha": raw_correct,
                "frozen_v2": v2_correct,
                "delta": v2_correct - raw_correct,
            },
            "systems": systems,
            "sparse_survival": {
                "kv_norm": {
                    str(width): build_survival_record(
                        answer_mapping["answer_token_positions"], kv_positions[width]
                    )
                    for width in widths
                },
                "query_attention": {
                    str(width): build_survival_record(
                        answer_mapping["answer_token_positions"], query_positions[width]
                    )
                    for width in widths
                },
            },
            "elapsed_seconds": time.perf_counter() - sample_started,
        }
        rows.append(row)
        _append_jsonl(results_path, row)
        print(
            f"[{case_index}/{len(cases)}] {stage} {sample_id}: "
            f"source_delta={row['source_outcome']['delta']:+.0f} "
            f"gold_delta={systems['frozen_v2']['mean_logprob'] - systems['raw_alpha']['mean_logprob']:+.4f}",
            flush=True,
        )

    expected_keys = {(case["stage"], case["source_row"]["sample_id"]) for case in cases}
    actual_keys = {(row["stage"], row["sample_id"]) for row in rows}
    if actual_keys != expected_keys:
        raise OracleContractError("D0 run did not complete the frozen changed set")
    rows.sort(key=lambda row: (row["stage"], row["sample_id"]))
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "scope": "development_only_not_fresh_confirmation",
        "manifest_id": manifest["manifest_id"],
        "protocol_sha256": protocol_sha,
        "source_runs": sources,
        "sparse_widths": list(widths),
        "analysis": summarize_diagnosis(rows, widths),
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
    parser.add_argument("--sparse-widths", default="8,16")
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
