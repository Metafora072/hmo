"""D2 diagnosis of contiguous query-attention Sparse windows."""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from argparse import Namespace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from experiments.phase2.e3_v2.context_query import tokenize_sample_prompt
from experiments.phase2.e3_v2.coverage_fidelity_cache import (
    SPARSE_SELECTORS,
    select_sparse_positions,
)
from experiments.phase2.e3_v2.oracle import OracleContractError, SegmentSpec
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
    build_survival_record,
    load_changed_cases,
    locate_answer_token_positions,
)
from experiments.phase2.e3_v2.run_discovery import _build_samples
from experiments.phase2.e3_v2.run_end_task import _load_protocol
from experiments.utils.model_loader import (
    get_full_attention_indices,
    get_linear_attention_indices,
)
from experiments.utils.run_manifest import collect_environment, ensure_run_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_FILENAME = "contiguous_window_diagnosis_results.jsonl"
SUMMARY_FILENAME = "contiguous_window_diagnosis_summary.json"
SCHEMA_VERSION = "hmo.coverage_fidelity.contiguous_window_diagnosis.v1"
MIN_COMPLETE_SURVIVAL_CASES = 5


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


def build_probe_segments(
    context_tokens: int, segment_length: int
) -> tuple[SegmentSpec, ...]:
    """Build token-boundary-only segments for a survival diagnostic."""
    if context_tokens <= 0 or segment_length <= 0:
        raise OracleContractError("D2 probe segment sizes must be positive")
    count = math.ceil(context_tokens / segment_length)
    if count < 3:
        raise OracleContractError("D2 requires middle segments")
    segments = []
    for segment_id in range(count):
        start = segment_id * segment_length
        end = min(start + segment_length, context_tokens)
        token_count = end - start
        position = ((start + end) / 2.0) / context_tokens
        segments.append(
            SegmentSpec(
                segment_id=segment_id,
                start=start,
                end=end,
                token_count=token_count,
                kv_bytes=token_count,
                protected=segment_id in {0, count - 1},
                partial=token_count != segment_length,
                normalized_position=float(position),
                position_bin=min(int(position * 4), 3),
            )
        )
    return tuple(segments)


def summarize_survival(rows: Sequence[Mapping], widths: Sequence[int]) -> dict:
    if not rows:
        raise OracleContractError("D2 survival summary requires rows")
    survival = {}
    for selector in SPARSE_SELECTORS:
        survival[selector] = {}
        for width in widths:
            values = [row["survival"][selector][str(width)] for row in rows]
            survival[selector][str(width)] = {
                "all_answer_tokens_survived_cases": sum(
                    value["all_answer_tokens_survived"] for value in values
                ),
                "any_answer_token_survived_cases": sum(
                    value["any_answer_token_survived"] for value in values
                ),
                "mean_answer_token_retained_fraction": float(
                    np.mean(
                        [value["answer_token_retained_fraction"] for value in values]
                    )
                ),
            }
    candidates = []
    for width in widths:
        window = survival["max_mass_window"][str(width)]
        scattered = survival["top_tokens"][str(width)]
        candidates.append(
            {
                "width": width,
                "complete_survival_cases": window[
                    "all_answer_tokens_survived_cases"
                ],
                "complete_survival_gain_vs_top_tokens": (
                    window["all_answer_tokens_survived_cases"]
                    - scattered["all_answer_tokens_survived_cases"]
                ),
                "mean_retained_fraction": window[
                    "mean_answer_token_retained_fraction"
                ],
            }
        )
    selected = max(
        candidates,
        key=lambda item: (
            item["complete_survival_cases"],
            item["mean_retained_fraction"],
            -item["width"],
        ),
    )
    continue_to_generation = (
        selected["complete_survival_cases"] >= MIN_COMPLETE_SURVIVAL_CASES
        and selected["complete_survival_gain_vs_top_tokens"] > 0
    )
    return {
        "case_count": len(rows),
        "survival": survival,
        "selection": {
            "rule": (
                "max complete survival, then mean retained fraction, then smaller width"
            ),
            "minimum_complete_survival_cases": MIN_COMPLETE_SURVIVAL_CASES,
            "selected_selector": "max_mass_window",
            "selected_width": selected["width"],
            "selected_complete_survival_cases": selected[
                "complete_survival_cases"
            ],
            "selected_complete_survival_gain_vs_top_tokens": selected[
                "complete_survival_gain_vs_top_tokens"
            ],
            "continue_to_generation": continue_to_generation,
        },
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
                    f"invalid persisted D2 result at line {line_number}"
                ) from exc
    keys = [(row.get("stage"), row.get("sample_id")) for row in rows]
    if len(keys) != len(set(keys)):
        raise OracleContractError("persisted D2 results contain duplicate cases")
    return rows


def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("D2 diagnosis requires exactly one visible GPU")
    widths = tuple(sorted({int(value) for value in args.sparse_widths.split(",")}))
    if not widths or any(value <= 0 for value in widths):
        raise OracleContractError("D2 Sparse widths must be positive")
    protocol, protocol_sha = _load_protocol(Path(args.protocol).resolve())
    if (
        args.model_id != protocol["model_id"]
        or args.model_revision != protocol["model_revision"]
    ):
        raise OracleContractError("D2 model identity disagrees with P3 protocol")
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
        "sparse_widths": list(widths),
        "expected_changed_cases": args.expected_changed_cases,
        "continuation_minimum_complete_survival_cases": (
            MIN_COMPLETE_SURVIVAL_CASES
        ),
        "recurrent_backend": REFERENCE_BACKEND,
        "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    manifest = ensure_run_manifest(
        run_dir,
        experiment="coverage_fidelity_d2_contiguous_window_diagnosis",
        args=scientific_args,
        selections={
            "scope": "development_only_frozen_p3_membership_changed_cases",
            "source_runs": sources,
            "protocol_sha256": protocol_sha,
            "selectors": list(SPARSE_SELECTORS),
            "window_rule": "contiguous_fixed_width_max_query_attention_mass",
            "tie_break": "earliest_window_start",
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
        raise OracleContractError("D2 results exist; pass --resume to continue")
    completed_keys = {(row["stage"], row["sample_id"]) for row in completed}
    rows = list(completed)

    torch.manual_seed(20261005)
    torch.cuda.manual_seed_all(20261005)
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
        answer_mapping = locate_answer_token_positions(tokenizer, prompt, sample.answer)
        segment_length = protocol["stages"][stage]["segment_length"]
        segments = build_probe_segments(prompt.context_tokens, segment_length)
        by_segment = {segment.segment_id: segment for segment in segments}
        answer_segment_ids = sorted(
            {
                position // segment_length
                for position in answer_mapping["answer_token_positions"]
            }
        )
        probe = collect_hybrid_query_token_probe(
            model,
            prompt,
            attention_layer_indices=attention_layers,
            recurrent_layer_indices=recurrent_layers,
            segments=segments,
            segment_length=segment_length,
        )
        survival = {}
        for selector in SPARSE_SELECTORS:
            survival[selector] = {}
            for width in widths:
                kept = {
                    segment_id: select_sparse_positions(
                        probe.token_attention_mass,
                        by_segment[segment_id],
                        width,
                        selector=selector,
                    )
                    for segment_id in answer_segment_ids
                }
                survival[selector][str(width)] = build_survival_record(
                    answer_mapping["answer_token_positions"], kept
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
            "survival": survival,
            "elapsed_seconds": time.perf_counter() - sample_started,
        }
        rows.append(row)
        _append_jsonl(results_path, row)
        print(
            f"[{case_index}/{len(cases)}] {stage} {sample_id}: "
            + " ".join(
                f"window{width}="
                f"{int(survival['max_mass_window'][str(width)]['all_answer_tokens_survived'])}"
                for width in widths
            ),
            flush=True,
        )
        del probe
        _cleanup_cuda()

    expected_keys = {(case["stage"], case["source_row"]["sample_id"]) for case in cases}
    actual_keys = {(row["stage"], row["sample_id"]) for row in rows}
    if actual_keys != expected_keys:
        raise OracleContractError("D2 run did not complete the development set")
    rows.sort(key=lambda row: (row["stage"], row["sample_id"]))
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "scope": "development_only_not_fresh_confirmation",
        "manifest_id": manifest["manifest_id"],
        "protocol_sha256": protocol_sha,
        "source_runs": sources,
        "sparse_widths": list(widths),
        "analysis": summarize_survival(rows, widths),
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
