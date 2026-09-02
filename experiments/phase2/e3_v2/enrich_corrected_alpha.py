"""Recompute sequential hybrid-query alpha while reusing completed oracle labels."""
from __future__ import annotations

import argparse
import json
import time
from argparse import Namespace
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from experiments.phase2.e3_v2.alpha_probe import collect_isolated_query_alpha
from experiments.phase2.e3_v2.context_query import tokenize_sample_prompt
from experiments.phase2.e3_v2.direct_fusion import evaluate_direct_fusions
from experiments.phase2.e3_v2.oracle import OracleContractError, load_oracle_manifest
from experiments.phase2.e3_v2.real_model_preflight import (
    _force_torch_reference_backend,
    _load_model,
    model_provenance,
)
from experiments.phase2.e3_v2.run_discovery import (
    _build_samples,
    _cleanup_cuda,
    analyze_discovery,
)
from experiments.phase2.e3_v2.statistics import SegmentEvidence, spearman_correlation
from experiments.utils.model_loader import (
    get_full_attention_indices,
    get_linear_attention_indices,
)


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OracleContractError(f"cannot read {path}") from exc


def replace_alpha_rows(
    raw_rows: Sequence[Mapping],
    corrected_alpha: Mapping[int, float],
    *,
    sample_id: str,
) -> tuple[SegmentEvidence, ...]:
    """Replace alpha and every alpha-derived candidate without changing labels."""
    evidence = []
    raw_ids = {int(row["segment_id"]) for row in raw_rows}
    if not raw_ids or not raw_ids.issubset(corrected_alpha):
        raise OracleContractError("corrected alpha and segment evidence disagree")
    for raw in raw_rows:
        segment_id = int(raw["segment_id"])
        alpha = float(corrected_alpha[segment_id])
        candidates = {
            str(name): float(value) for name, value in raw["candidates"].items()
        }
        candidates["phi_sigma_alpha"] = alpha * candidates["sigma_current"]
        candidates["phi_delta_alpha"] = alpha * candidates["delta_update"]
        evidence.append(
            SegmentEvidence(
                sample_id=sample_id,
                dataset=str(raw["dataset"]),
                segment_id=segment_id,
                utility=float(raw["utility"]),
                alpha=alpha,
                normalized_position=float(raw["normalized_position"]),
                candidates=candidates,
            )
        )
    return tuple(evidence)


def _alpha_shift_summary(
    legacy: Sequence[float],
    corrected: Sequence[float],
    *,
    k: int,
) -> dict:
    legacy = np.asarray(legacy, dtype=np.float64)
    corrected = np.asarray(corrected, dtype=np.float64)
    if legacy.shape != corrected.shape or legacy.ndim != 1 or len(legacy) < 2:
        raise OracleContractError("alpha shift arrays must be aligned one-dimensional vectors")
    legacy_topk = set(np.argsort(-legacy, kind="stable")[:k].tolist())
    corrected_topk = set(np.argsort(-corrected, kind="stable")[:k].tolist())
    return {
        "spearman": spearman_correlation(legacy, corrected),
        "topk_overlap": len(legacy_topk & corrected_topk) / k,
        "argmax_changed": int(np.argmax(legacy)) != int(np.argmax(corrected)),
        "legacy_sum": float(legacy.sum()),
        "corrected_sum": float(corrected.sum()),
    }


def enrich(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("corrected-alpha enrichment requires exactly one visible CUDA device")
    model_path = Path(args.model_path).resolve()
    model_identity = model_provenance(
        model_path,
        args.model_id,
        revision=args.model_revision,
    )
    model, tokenizer = _load_model(model_path)
    attention_layers = tuple(get_full_attention_indices(model.config))
    recurrent_layers = tuple(get_linear_attention_indices(model.config))
    _force_torch_reference_backend(model, recurrent_layers)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    evidence = []
    k_by_sample = {}
    sources = []
    alpha_shifts = {}
    for run_index, value in enumerate(args.run_dir):
        run_dir = Path(value).resolve()
        manifest = _read_json(run_dir / "run_manifest.json")
        summary = _read_json(run_dir / "discovery_summary.json")
        if summary.get("status") != "complete":
            raise OracleContractError(f"incomplete source run {run_dir}")
        if manifest["run_spec"]["model"] != model_identity:
            raise OracleContractError("source run model provenance mismatch")
        source_args = manifest["run_spec"]["arguments"]
        samples = _build_samples(
            tokenizer,
            Namespace(
                datasets=source_args["datasets"],
                samples_per_dataset=source_args["samples_per_dataset"],
                context_length=source_args["context_length"],
                seed=source_args["seed"],
                sample_id_prefix=source_args.get("sample_id_prefix", ""),
            ),
        )
        summary_by_id = {row["sample_id"]: row for row in summary["sample_summaries"]}
        namespace = f"run{run_index}_{manifest['manifest_id'][:8]}"
        for sample in samples:
            prompt = tokenize_sample_prompt(sample, tokenizer)
            expected = summary_by_id[sample.sample_id]
            if prompt.context_tokens != expected["context_tokens"]:
                raise OracleContractError("regenerated prompt disagrees with source run")
            plan = load_oracle_manifest(
                run_dir / "samples" / sample.sample_id / "oracle_manifest.json"
            )
            alpha_result = collect_isolated_query_alpha(
                model,
                prompt,
                attention_layer_indices=attention_layers,
                segments=plan.segments,
            )
            corrected = alpha_result.as_dict()
            raw_rows = _read_json(
                run_dir / "samples" / sample.sample_id / "segment_evidence.json"
            )["rows"]
            namespaced_id = f"{namespace}:{sample.sample_id}"
            updated = replace_alpha_rows(
                raw_rows,
                corrected,
                sample_id=namespaced_id,
            )
            evidence.extend(updated)
            eligible_raw = [
                row for row in sorted(raw_rows, key=lambda row: int(row["segment_id"]))
            ]
            eligible_updated = sorted(updated, key=lambda row: row.segment_id)
            k = int(expected["middle_budget_slots"])
            alpha_shifts[namespaced_id] = _alpha_shift_summary(
                [float(row["alpha"]) for row in eligible_raw],
                [row.alpha for row in eligible_updated],
                k=k,
            )
            k_by_sample[namespaced_id] = k
            print(f"recaptured alpha for {namespaced_id}", flush=True)
            _cleanup_cuda()
        sources.append(
            {
                "run_dir": str(run_dir),
                "manifest_id": manifest["manifest_id"],
                "seed": source_args["seed"],
                "sample_count": len(samples),
            }
        )

    shift_values = list(alpha_shifts.values())
    analysis = analyze_discovery(
        evidence,
        k_by_sample,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    direct = evaluate_direct_fusions(
        evidence,
        k_by_sample,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed + 100,
    )
    return {
        "schema_version": "hmo.corrected_alpha_enrichment.v1",
        "status": "complete",
        "scope": "sequential_hybrid_query_alpha_enrichment_reusing_oracle_labels",
        "sources": sources,
        "sample_count": len(k_by_sample),
        "model": model_identity,
        "alpha_shift": {
            "mean_spearman": float(np.mean([row["spearman"] for row in shift_values])),
            "mean_topk_overlap": float(
                np.mean([row["topk_overlap"] for row in shift_values])
            ),
            "argmax_changed_samples": sum(row["argmax_changed"] for row in shift_values),
            "samples": alpha_shifts,
        },
        "analysis": analysis,
        "direct_fusions": direct,
        "k_by_sample": k_by_sample,
        "evidence": [asdict(row) for row in evidence],
        "runtime": {
            "elapsed_seconds": time.perf_counter() - started,
            "gpu_name": torch.cuda.get_device_name(0),
            "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
            "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--model-revision")
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args()
    if args.bootstrap_samples <= 0:
        parser.error("bootstrap-samples must be positive")
    return args


def main() -> int:
    args = parse_args()
    payload = enrich(args)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {key: value for key, value in payload.items() if key != "evidence"}
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
