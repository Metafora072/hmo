"""Offline evaluation for the frozen safe-in/stressed-out boundary exchange."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from experiments.phase2.e3_v2.analyze_discovery_runs import _read_json
from experiments.phase2.e3_v2.conditional_regime import (
    _rank01,
    load_discovery_evidence,
)
from experiments.phase2.e3_v2.oracle import OracleContractError
from experiments.phase2.e3_v2.statistics import (
    SegmentEvidence,
    ndcg_at_k,
    pairwise_ranking_accuracy,
    sample_grouped_bootstrap_interval,
)


BOUNDARY_EXCHANGE_SCHEMA = "p1-boundary-exchange-v1"
REGIME_PRIORITY = {"SAFE": 0, "NEUTRAL": 1, "STRESSED": 2}


def boundary_exchange_scores(
    alpha: Sequence[float],
    sigma_current: Sequence[float],
    delta_update: Sequence[float],
    *,
    k: int,
) -> tuple[np.ndarray, tuple[str, ...], dict | None]:
    """Replace the lowest-alpha SAFE insider with the highest-alpha STRESSED outsider."""
    alpha = np.asarray(alpha, dtype=np.float64)
    sigma_current = np.asarray(sigma_current, dtype=np.float64)
    delta_update = np.asarray(delta_update, dtype=np.float64)
    if (
        alpha.ndim != 1
        or len(alpha) < 2
        or alpha.shape != sigma_current.shape
        or alpha.shape != delta_update.shape
        or not np.all(np.isfinite(alpha))
        or not np.all(np.isfinite(sigma_current))
        or not np.all(np.isfinite(delta_update))
        or k <= 0
        or k >= len(alpha)
    ):
        raise ValueError("boundary exchange inputs must be aligned, finite, and have 0 < k < n")

    high_sigma = _rank01(sigma_current) >= 0.5
    high_delta = _rank01(delta_update) >= 0.5
    regimes = tuple(
        "SAFE"
        if sigma_high and not delta_high
        else "STRESSED"
        if sigma_high and delta_high
        else "NEUTRAL"
        for sigma_high, delta_high in zip(high_sigma, high_delta)
    )

    alpha_order = [int(index) for index in np.argsort(-alpha, kind="stable")]
    selected = set(alpha_order[:k])
    safe_inside = [index for index in selected if regimes[index] == "SAFE"]
    stressed_outside = [
        index for index in alpha_order[k:] if regimes[index] == "STRESSED"
    ]
    exchange = None
    if safe_inside and stressed_outside:
        donor = min(safe_inside, key=lambda index: (alpha[index], index))
        entrant = max(stressed_outside, key=lambda index: (alpha[index], -index))
        selected.remove(donor)
        selected.add(entrant)
        exchange = {"safe_inside_index": donor, "stressed_outside_index": entrant}

    # The controller is set-valued. Preserve raw-alpha ordering within the revised
    # selected and unselected sets to obtain a deterministic full ranking.
    controller_order = (
        [index for index in alpha_order if index in selected]
        + [index for index in alpha_order if index not in selected]
    )
    scores = np.empty(len(alpha), dtype=np.float64)
    for rank, index in enumerate(controller_order):
        scores[index] = float(len(alpha) - rank)
    return scores, regimes, exchange


def _task_means(values: Mapping[str, float], sample_rows: Mapping[str, dict]) -> dict:
    grouped: dict[str, list[float]] = {}
    for sample_id, value in values.items():
        grouped.setdefault(sample_rows[sample_id]["dataset"], []).append(value)
    return {
        dataset: float(np.mean(task_values))
        for dataset, task_values in sorted(grouped.items())
    }


def _sign_counts(values: Sequence[float]) -> dict[str, int]:
    return {
        "positive": sum(value > 0.0 for value in values),
        "zero": sum(value == 0.0 for value in values),
        "negative": sum(value < 0.0 for value in values),
    }


def evaluate_boundary_exchange(
    evidence: Sequence[SegmentEvidence],
    k_by_sample: Mapping[str, int],
    *,
    bootstrap_samples: int = 5000,
    seed: int = 20260912,
) -> dict:
    """Evaluate the fixed one-swap policy against raw alpha on discovery evidence."""
    by_sample: dict[str, list[SegmentEvidence]] = {}
    for row in evidence:
        by_sample.setdefault(row.sample_id, []).append(row)
    if not by_sample or set(by_sample) != set(k_by_sample):
        raise OracleContractError("boundary evidence and k_by_sample disagree")
    if bootstrap_samples <= 0:
        raise OracleContractError("bootstrap_samples must be positive")

    sample_rows = {}
    pairwise_delta = {}
    ndcg_delta = {}
    topk_mean_utility_delta = {}
    topk_sum_utility_delta = {}
    executable_quality = []
    for sample_id, rows in sorted(by_sample.items()):
        rows = sorted(rows, key=lambda row: row.segment_id)
        datasets = {row.dataset for row in rows}
        if len(datasets) != 1:
            raise OracleContractError("one sample cannot span multiple datasets")
        values = [
            row.candidates.get(name)
            for row in rows
            for name in ("sigma_current", "delta_update")
        ]
        if any(value is None or not math.isfinite(float(value)) for value in values):
            raise OracleContractError("boundary exchange requires finite recurrent signals")

        k = int(k_by_sample[sample_id])
        utility = np.asarray([row.utility for row in rows], dtype=np.float64)
        alpha = np.asarray([row.alpha for row in rows], dtype=np.float64)
        controller, regimes, exchange = boundary_exchange_scores(
            alpha,
            [row.candidates["sigma_current"] for row in rows],
            [row.candidates["delta_update"] for row in rows],
            k=k,
        )
        alpha_selected = np.argsort(-alpha, kind="stable")[:k]
        controller_selected = np.argsort(-controller, kind="stable")[:k]
        base_mean = float(np.mean(utility[alpha_selected]))
        controller_mean = float(np.mean(utility[controller_selected]))
        mean_delta = controller_mean - base_mean
        sum_delta = mean_delta * k
        base_pairwise = pairwise_ranking_accuracy(alpha, utility)
        controller_pairwise = pairwise_ranking_accuracy(controller, utility)
        base_ndcg = ndcg_at_k(alpha, utility, k)
        controller_ndcg = ndcg_at_k(controller, utility, k)

        mapped_exchange = None
        if exchange is not None:
            donor = int(exchange["safe_inside_index"])
            entrant = int(exchange["stressed_outside_index"])
            quality_delta = float(utility[entrant] - utility[donor])
            executable_quality.append(quality_delta)
            mapped_exchange = {
                "safe_inside_segment_id": rows[donor].segment_id,
                "stressed_outside_segment_id": rows[entrant].segment_id,
                "oracle_utility_delta": quality_delta,
            }

        pairwise_delta[sample_id] = controller_pairwise - base_pairwise
        ndcg_delta[sample_id] = controller_ndcg - base_ndcg
        topk_mean_utility_delta[sample_id] = mean_delta
        topk_sum_utility_delta[sample_id] = sum_delta
        sample_rows[sample_id] = {
            "dataset": next(iter(datasets)),
            "k": k,
            "regime_counts": {
                regime: regimes.count(regime) for regime in REGIME_PRIORITY
            },
            "exchange": mapped_exchange,
            "alpha_topk_segment_ids": [rows[index].segment_id for index in alpha_selected],
            "controller_topk_segment_ids": [
                rows[index].segment_id for index in controller_selected
            ],
            "topk_mean_utility_delta": mean_delta,
            "topk_sum_utility_delta": sum_delta,
            "pairwise_delta": pairwise_delta[sample_id],
            "ndcg_delta": ndcg_delta[sample_id],
        }

    executable_count = len(executable_quality)
    sample_count = len(sample_rows)
    task_topk = _task_means(topk_mean_utility_delta, sample_rows)
    task_ndcg = _task_means(ndcg_delta, sample_rows)
    enough_executable = executable_count > sample_count / 2
    task_direction_consistent = all(value >= 0.0 for value in task_topk.values()) and all(
        value >= 0.0 for value in task_ndcg.values()
    )
    overall_topk = float(np.mean(list(topk_mean_utility_delta.values())))
    overall_ndcg = float(np.mean(list(ndcg_delta.values())))
    continue_to_fresh_8k = bool(
        enough_executable
        and overall_topk > 0.0
        and overall_ndcg > 0.0
        and task_direction_consistent
    )

    return {
        "schema_version": BOUNDARY_EXCHANGE_SCHEMA,
        "status": "complete",
        "hypothesis": "replace one alpha-selected SAFE insider with one STRESSED outsider",
        "baseline": "raw alpha TopK",
        "configuration": {
            "normalization": "within_sample_average_rank01",
            "threshold": 0.5,
            "threshold_search": False,
            "max_exchanges_per_sample": 1,
            "safe_inside": "lowest_alpha_SAFE_inside_TopK",
            "stressed_outside": "highest_alpha_STRESSED_outside_TopK",
            "ranking_after_exchange": "selected_then_unselected_with_raw_alpha_order_within_each_set",
        },
        "continue_rule": {
            "executable_samples": "strictly_more_than_half",
            "overall_topk_mean_utility_delta": "positive",
            "overall_ndcg_delta": "positive",
            "every_task_topk_and_ndcg_delta": "nonnegative",
            "confidence_interval_required": False,
        },
        "sample_count": sample_count,
        "executable_exchange_count": executable_count,
        "executable_exchange_sign_counts": _sign_counts(executable_quality),
        "enough_executable_exchanges": enough_executable,
        "topk_mean_utility_improvement": asdict(
            sample_grouped_bootstrap_interval(
                topk_mean_utility_delta,
                n_bootstrap=bootstrap_samples,
                seed=seed,
            )
        ),
        "topk_sum_utility_improvement": asdict(
            sample_grouped_bootstrap_interval(
                topk_sum_utility_delta,
                n_bootstrap=bootstrap_samples,
                seed=seed + 1,
            )
        ),
        "pairwise_improvement": asdict(
            sample_grouped_bootstrap_interval(
                pairwise_delta,
                n_bootstrap=bootstrap_samples,
                seed=seed + 2,
            )
        ),
        "ndcg_improvement": asdict(
            sample_grouped_bootstrap_interval(
                ndcg_delta,
                n_bootstrap=bootstrap_samples,
                seed=seed + 3,
            )
        ),
        "task_topk_mean_utility_improvement": task_topk,
        "task_pairwise_improvement": _task_means(pairwise_delta, sample_rows),
        "task_ndcg_improvement": task_ndcg,
        "task_direction_consistent": task_direction_consistent,
        "continue_to_fresh_8k": continue_to_fresh_8k,
        "decision": (
            "freeze_boundary_exchange_and_run_fresh_8k"
            if continue_to_fresh_8k
            else "stop_boundary_exchange_without_tuning"
        ),
        "samples": sample_rows,
    }


def load_discovery_evidence_and_k(
    run_dirs: Sequence[Path],
) -> tuple[tuple[SegmentEvidence, ...], dict[str, int], list[dict], dict, dict]:
    evidence, sources, compatible_arguments, model = load_discovery_evidence(run_dirs)
    k_by_sample = {}
    for run_index, (run_dir, source) in enumerate(zip(run_dirs, sources)):
        summary = _read_json(Path(run_dir).resolve() / "discovery_summary.json")
        namespace = f"run{run_index}_{source['manifest_id'][:8]}"
        for row in summary["sample_summaries"]:
            k_by_sample[f"{namespace}:{row['sample_id']}"] = int(
                row["middle_budget_slots"]
            )
    if {row.sample_id for row in evidence} != set(k_by_sample):
        raise OracleContractError("loaded evidence and budget slots disagree")
    return evidence, k_by_sample, sources, compatible_arguments, model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args()
    if args.bootstrap_samples <= 0:
        parser.error("bootstrap-samples must be positive")
    return args


def main() -> int:
    args = parse_args()
    run_dirs = [Path(value) for value in args.run_dir]
    evidence, k_by_sample, sources, compatible_arguments, model = (
        load_discovery_evidence_and_k(run_dirs)
    )
    payload = evaluate_boundary_exchange(
        evidence,
        k_by_sample,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    payload["sources"] = sources
    payload["compatible_arguments"] = compatible_arguments
    payload["model"] = model
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + chr(10),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
