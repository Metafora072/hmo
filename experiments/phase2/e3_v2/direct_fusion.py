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
