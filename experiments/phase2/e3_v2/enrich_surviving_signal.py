"""Recapture surviving recurrent contributions for completed P1 oracle labels."""
from __future__ import annotations

import argparse
import json
import time
from argparse import Namespace
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from experiments.phase2.e3_v2.context_query import tokenize_sample_prompt
from experiments.phase2.e3_v2.oracle import OracleContractError
from experiments.phase2.e3_v2.real_model_preflight import (
    REFERENCE_BACKEND,
    _force_torch_reference_backend,
    _load_model,
    model_provenance,
)
from experiments.phase2.e3_v2.run_discovery import (
    _build_samples,
    _capture_sample_signals,
    _cleanup_cuda,
)
from experiments.phase2.e3_v2.statistics import (
    SegmentEvidence,
    evaluate_candidate_grouped_cv,
    ndcg_at_k,
    pairwise_ranking_accuracy,
    sample_grouped_bootstrap_interval,
)
from experiments.utils.model_loader import get_linear_attention_indices


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OracleContractError(f"cannot read {path}") from exc


def _rank01(values) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="stable")
    ranks = np.empty(len(array), dtype=np.float64)
    ranks[order] = np.arange(len(array), dtype=np.float64)
    return ranks / max(len(array) - 1, 1)


def _evaluate_gap(evidence, k_by_sample, bootstrap_samples, seed):
    by_sample = {}
    for row in evidence:
        by_sample.setdefault(row.sample_id, []).append(row)
    pairwise_diff = {}
    ndcg_diff = {}
    datasets = {}
    for sample_id, rows in sorted(by_sample.items()):
        rows = sorted(rows, key=lambda row: row.segment_id)
        alpha = np.asarray([row.alpha for row in rows], dtype=np.float64)
        utility = [row.utility for row in rows]
        surviving_rank = _rank01(
            [row.candidates["surviving_write_norm"] for row in rows]
        )
        score = alpha * (1.0 - surviving_rank)
        pairwise_diff[sample_id] = pairwise_ranking_accuracy(score, utility) - (
            pairwise_ranking_accuracy(alpha, utility)
        )
        ndcg_diff[sample_id] = ndcg_at_k(
            score, utility, k_by_sample[sample_id]
        ) - ndcg_at_k(alpha, utility, k_by_sample[sample_id])
        datasets[sample_id] = rows[0].dataset

    def task_means(values):
        grouped = {}
        for sample_id, value in values.items():
            grouped.setdefault(datasets[sample_id], []).append(value)
        return {
            dataset: float(np.mean(dataset_values))
            for dataset, dataset_values in sorted(grouped.items())
        }

    return {
        "formula": "alpha*(1-within_sample_rank01(surviving_write_norm))",
        "pairwise_improvement": asdict(
            sample_grouped_bootstrap_interval(
                pairwise_diff, n_bootstrap=bootstrap_samples, seed=seed
            )
        ),
        "ndcg_improvement": asdict(
            sample_grouped_bootstrap_interval(
                ndcg_diff, n_bootstrap=bootstrap_samples, seed=seed + 1
            )
        ),
        "task_pairwise_improvement": task_means(pairwise_diff),
        "task_ndcg_improvement": task_means(ndcg_diff),
    }


def enrich(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("signal enrichment requires exactly one visible CUDA device")
    model_path = Path(args.model_path).resolve()
    model_identity = model_provenance(
        model_path, args.model_id, revision=args.model_revision
    )
    model, tokenizer = _load_model(model_path)
    recurrent_layers = tuple(get_linear_attention_indices(model.config))
    _force_torch_reference_backend(model, recurrent_layers)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    evidence = []
    k_by_sample = {}
    sources = []

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
            ),
        )
        summary_by_id = {
            row["sample_id"]: row for row in summary["sample_summaries"]
        }
        namespace = f"run{run_index}_{manifest['manifest_id'][:8]}"
        for sample in samples:
            prompt = tokenize_sample_prompt(sample, tokenizer)
            expected = summary_by_id[sample.sample_id]
            if prompt.context_tokens != expected["context_tokens"]:
                raise OracleContractError("regenerated prompt disagrees with source run")
            outputs, recurrent, _ = _capture_sample_signals(
                model,
                prompt,
                recurrent_layers,
                segment_length=source_args["segment_length"],
            )
            del outputs
            raw_rows = _read_json(
                run_dir / "samples" / sample.sample_id / "segment_evidence.json"
            )["rows"]
            namespaced_id = f"{namespace}:{sample.sample_id}"
            for raw in raw_rows:
                segment_id = int(raw["segment_id"])
                candidates = {
                    str(name): float(value)
                    for name, value in raw["candidates"].items()
                }
                candidates["surviving_write_norm"] = float(
                    recurrent.surviving_write_norm[segment_id]
                )
                evidence.append(
                    SegmentEvidence(
                        sample_id=namespaced_id,
                        dataset=str(raw["dataset"]),
                        segment_id=segment_id,
                        utility=float(raw["utility"]),
                        alpha=float(raw["alpha"]),
                        normalized_position=float(raw["normalized_position"]),
                        candidates=candidates,
                    )
                )
            k_by_sample[namespaced_id] = int(expected["middle_budget_slots"])
            print(f"captured {namespaced_id}", flush=True)
            _cleanup_cuda()
        sources.append(
            {
                "run_dir": str(run_dir),
                "manifest_id": manifest["manifest_id"],
                "seed": source_args["seed"],
                "sample_count": len(samples),
            }
        )

    learned = evaluate_candidate_grouped_cv(
        evidence,
        "surviving_write_norm",
        k_by_sample=k_by_sample,
        folds=min(4, len(k_by_sample)),
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    return {
        "status": "complete",
        "scope": "discovery_signal_enrichment_not_confirmation",
        "sources": sources,
        "sample_count": len(k_by_sample),
        "model": model_identity,
        "learned_incremental_diagnostic": asdict(learned),
        "direct_recurrent_gap": _evaluate_gap(
            evidence, k_by_sample, args.bootstrap_samples, args.seed + 100
        ),
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
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()
    if len(args.run_dir) < 1 or args.bootstrap_samples <= 0:
        parser.error("provide source runs and a positive bootstrap count")
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
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
