"""Training-free fusion diagnostics for E3-v2 segment evidence."""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from experiments.phase2.e3_v2.statistics import (
    SegmentEvidence,
    ndcg_at_k,
    pairwise_ranking_accuracy,
    sample_grouped_bootstrap_interval,
)


def _rank01(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="stable")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks / max(len(array) - 1, 1)


BOUNDED_ADDITIVE_SCHEMA = "p1-bounded-additive-v1"
BOUNDED_ADDITIVE_LAMBDAS = (-0.30, -0.15, 0.15, 0.30)
CONDITIONAL_CONTROLLER_SCHEMA = "p1-conditional-rank-v1"


def _task_means(values, sample_rows):
    grouped = {}
    for sample_id, value in values.items():
        dataset = sample_rows[sample_id]["dataset"]
        grouped.setdefault(dataset, []).append(value)
    return {
        dataset: float(np.mean(dataset_values))
        for dataset, dataset_values in sorted(grouped.items())
    }


def _lambda_key(value: float) -> str:
    return f"lambda_{value:+.2f}"


def evaluate_bounded_additive(
    evidence: Sequence[SegmentEvidence],
    k_by_sample: Mapping[str, int],
    *,
    lambdas: Sequence[float] = BOUNDED_ADDITIVE_LAMBDAS,
    bootstrap_samples: int,
    seed: int,
) -> dict:
    """Evaluate and select the frozen rank(alpha) + lambda*rank(sigma) family."""
    lambda_values = tuple(float(value) for value in lambdas)
    if not lambda_values or len(set(lambda_values)) != len(lambda_values):
        raise ValueError("bounded-additive lambdas must be unique and non-empty")
    if any(not np.isfinite(value) or abs(value) > 0.30 for value in lambda_values):
        raise ValueError("bounded-additive lambdas must be finite and within [-0.30, 0.30]")

    by_sample: dict[str, list[SegmentEvidence]] = {}
    for row in evidence:
        by_sample.setdefault(row.sample_id, []).append(row)
    if set(by_sample) != set(k_by_sample):
        raise ValueError("bounded-additive evidence and k_by_sample disagree")

    sample_rows = {}
    for sample_id, rows in sorted(by_sample.items()):
        rows = sorted(rows, key=lambda row: row.segment_id)
        utility = [row.utility for row in rows]
        alpha_rank = _rank01([row.alpha for row in rows])
        recurrent_centered = _rank01(
            [row.candidates["sigma_current"] for row in rows]
        ) - 0.5
        scores = {"alpha": alpha_rank}
        scores.update(
            {
                _lambda_key(value): alpha_rank + value * recurrent_centered
                for value in lambda_values
            }
        )
        sample_rows[sample_id] = {
            "dataset": rows[0].dataset,
            "metrics": {
                name: {
                    "pairwise_accuracy": pairwise_ranking_accuracy(score, utility),
                    "ndcg": ndcg_at_k(score, utility, k_by_sample[sample_id]),
                }
                for name, score in scores.items()
            },
        }

    results = {}
    for method_index, value in enumerate(lambda_values):
        method = _lambda_key(value)
        pairwise_diff = {
            sample_id: row["metrics"][method]["pairwise_accuracy"]
            - row["metrics"]["alpha"]["pairwise_accuracy"]
            for sample_id, row in sample_rows.items()
        }
        ndcg_diff = {
            sample_id: row["metrics"][method]["ndcg"]
            - row["metrics"]["alpha"]["ndcg"]
            for sample_id, row in sample_rows.items()
        }
        results[method] = {
            "lambda": value,
            "pairwise_improvement": sample_grouped_bootstrap_interval(
                pairwise_diff,
                n_bootstrap=bootstrap_samples,
                seed=seed + method_index * 2,
            ).__dict__,
            "ndcg_improvement": sample_grouped_bootstrap_interval(
                ndcg_diff,
                n_bootstrap=bootstrap_samples,
                seed=seed + method_index * 2 + 1,
            ).__dict__,
            "task_pairwise_improvement": _task_means(pairwise_diff, sample_rows),
            "task_ndcg_improvement": _task_means(ndcg_diff, sample_rows),
        }

    selected_key = max(
        results,
        key=lambda key: (
            results[key]["pairwise_improvement"]["mean"],
            results[key]["ndcg_improvement"]["mean"],
            -abs(results[key]["lambda"]),
            -results[key]["lambda"],
        ),
    )
    return {
        "schema_version": BOUNDED_ADDITIVE_SCHEMA,
        "formula": "rank01(alpha)+lambda*(rank01(sigma_current)-0.5)",
        "baseline": "rank01(alpha), ranking-equivalent to raw alpha",
        "selection_rule": "max_mean_pairwise_then_ndcg_then_smaller_abs_lambda",
        "selected_lambda": results[selected_key]["lambda"],
        "selected_method": selected_key,
        "methods": results,
        "samples": sample_rows,
    }


def evaluate_direct_fusions(
    evidence: Sequence[SegmentEvidence],
    k_by_sample: Mapping[str, int],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict:
    """Compare two fixed scores against raw query-attention alpha."""
    by_sample: dict[str, list[SegmentEvidence]] = {}
    for row in evidence:
        by_sample.setdefault(row.sample_id, []).append(row)
    methods = ("alpha_sigma_product", "alpha_inverse_delta_rank")
    sample_rows = {}
    for sample_id, rows in sorted(by_sample.items()):
        rows = sorted(rows, key=lambda row: row.segment_id)
        utility = [row.utility for row in rows]
        alpha = np.asarray([row.alpha for row in rows], dtype=np.float64)
        sigma = np.asarray(
            [row.candidates["sigma_current"] for row in rows], dtype=np.float64
        )
        delta_rank = _rank01(
            [row.candidates["delta_update"] for row in rows]
        )
        scores = {
            "alpha": alpha,
            "alpha_sigma_product": alpha * sigma,
            "alpha_inverse_delta_rank": alpha * (1.0 - delta_rank),
        }
        sample_rows[sample_id] = {
            "dataset": rows[0].dataset,
            "metrics": {
                name: {
                    "pairwise_accuracy": pairwise_ranking_accuracy(score, utility),
                    "ndcg": ndcg_at_k(score, utility, k_by_sample[sample_id]),
                }
                for name, score in scores.items()
            },
        }

    results = {}
    for method_index, method in enumerate(methods):
        pairwise_diff = {
            sample_id: row["metrics"][method]["pairwise_accuracy"]
            - row["metrics"]["alpha"]["pairwise_accuracy"]
            for sample_id, row in sample_rows.items()
        }
        ndcg_diff = {
            sample_id: row["metrics"][method]["ndcg"]
            - row["metrics"]["alpha"]["ndcg"]
            for sample_id, row in sample_rows.items()
        }

        results[method] = {
            "pairwise_improvement": sample_grouped_bootstrap_interval(
                pairwise_diff,
                n_bootstrap=bootstrap_samples,
                seed=seed + method_index * 2,
            ).__dict__,
            "ndcg_improvement": sample_grouped_bootstrap_interval(
                ndcg_diff,
                n_bootstrap=bootstrap_samples,
                seed=seed + method_index * 2 + 1,
            ).__dict__,
            "task_pairwise_improvement": _task_means(pairwise_diff, sample_rows),
            "task_ndcg_improvement": _task_means(ndcg_diff, sample_rows),
        }
    return {"baseline": "alpha", "methods": results, "samples": sample_rows}


REGIME_PRIORITY = {"SAFE": 0, "NEUTRAL": 1, "STRESSED": 2}
CONDITIONAL_CONTROLLER_FORMULA = (
    "single_top_down_adjacent_regime_inversion_pass"
)
CONDITIONAL_COLLISION_POLICY = (
    "swap_adjacent_alpha_ranks_when_lower_regime_priority_is_higher_"
    "and_neither_segment_has_moved"
)


def conditional_controller_scores(
    alpha: Sequence[float],
    sigma_current: Sequence[float],
    delta_update: Sequence[float],
) -> tuple[np.ndarray, tuple[str, ...], tuple[tuple[int, int], ...]]:
    """Apply one stable adjacent regime swap per segment at most."""
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
    ):
        raise ValueError("conditional controller inputs must be aligned and finite")

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
    priorities = np.asarray(
        [REGIME_PRIORITY[regime] for regime in regimes],
        dtype=np.int64,
    )
    base_order = list(np.argsort(-alpha, kind="stable"))
    scores = alpha.copy()
    moved = set()
    swaps = []
    for position in range(len(base_order) - 1):
        upper = int(base_order[position])
        lower = int(base_order[position + 1])
        if upper in moved or lower in moved:
            continue
        if priorities[lower] > priorities[upper]:
            scores[upper], scores[lower] = scores[lower], scores[upper]
            moved.update((upper, lower))
            swaps.append((upper, lower))
    return scores, regimes, tuple(swaps)


def evaluate_conditional_controller(
    evidence: Sequence[SegmentEvidence],
    k_by_sample: Mapping[str, int],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict:
    """Compare the frozen local regime controller against raw alpha."""
    by_sample: dict[str, list[SegmentEvidence]] = {}
    for row in evidence:
        by_sample.setdefault(row.sample_id, []).append(row)
    if not by_sample or set(by_sample) != set(k_by_sample):
        raise ValueError("conditional controller evidence and k_by_sample disagree")

    sample_rows = {}
    for sample_id, rows in sorted(by_sample.items()):
        rows = sorted(rows, key=lambda row: row.segment_id)
        datasets = {row.dataset for row in rows}
        if len(datasets) != 1:
            raise ValueError("one sample cannot span multiple datasets")
        utility = [row.utility for row in rows]
        alpha = np.asarray([row.alpha for row in rows], dtype=np.float64)
        controller, regimes, swaps = conditional_controller_scores(
            alpha,
            [row.candidates["sigma_current"] for row in rows],
            [row.candidates["delta_update"] for row in rows],
        )
        metrics = {
            "alpha": {
                "pairwise_accuracy": pairwise_ranking_accuracy(alpha, utility),
                "ndcg": ndcg_at_k(alpha, utility, k_by_sample[sample_id]),
            },
            "conditional_controller": {
                "pairwise_accuracy": pairwise_ranking_accuracy(controller, utility),
                "ndcg": ndcg_at_k(controller, utility, k_by_sample[sample_id]),
            },
        }
        sample_rows[sample_id] = {
            "dataset": next(iter(datasets)),
            "regime_counts": {
                regime: regimes.count(regime) for regime in REGIME_PRIORITY
            },
            "adjacent_swaps": [
                [rows[upper].segment_id, rows[lower].segment_id]
                for upper, lower in swaps
            ],
            "metrics": metrics,
        }

    pairwise_diff = {
        sample_id: row["metrics"]["conditional_controller"]["pairwise_accuracy"]
        - row["metrics"]["alpha"]["pairwise_accuracy"]
        for sample_id, row in sample_rows.items()
    }
    ndcg_diff = {
        sample_id: row["metrics"]["conditional_controller"]["ndcg"]
        - row["metrics"]["alpha"]["ndcg"]
        for sample_id, row in sample_rows.items()
    }
    return {
        "schema_version": CONDITIONAL_CONTROLLER_SCHEMA,
        "formula": CONDITIONAL_CONTROLLER_FORMULA,
        "baseline": "raw alpha ranking",
        "configuration": {
            "normalization": "within_sample_average_rank01",
            "threshold": 0.5,
            "threshold_search": False,
            "regime_priority": REGIME_PRIORITY,
            "rank_adjustment": {"SAFE": 1, "NEUTRAL": 0, "STRESSED": -1},
            "collision_policy": CONDITIONAL_COLLISION_POLICY,
        },
        "controller": {
            "pairwise_improvement": sample_grouped_bootstrap_interval(
                pairwise_diff,
                n_bootstrap=bootstrap_samples,
                seed=seed,
            ).__dict__,
            "ndcg_improvement": sample_grouped_bootstrap_interval(
                ndcg_diff,
                n_bootstrap=bootstrap_samples,
                seed=seed + 1,
            ).__dict__,
            "task_pairwise_improvement": _task_means(pairwise_diff, sample_rows),
            "task_ndcg_improvement": _task_means(ndcg_diff, sample_rows),
        },
        "samples": sample_rows,
    }
