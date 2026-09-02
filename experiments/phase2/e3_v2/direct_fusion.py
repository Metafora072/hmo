"""Training-free fusion diagnostics for E3-v2 discovery evidence."""
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

        def task_means(values):
            grouped = {}
            for sample_id, value in values.items():
                dataset = sample_rows[sample_id]["dataset"]
                grouped.setdefault(dataset, []).append(value)
            return {
                dataset: float(np.mean(dataset_values))
                for dataset, dataset_values in sorted(grouped.items())
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
            "task_pairwise_improvement": task_means(pairwise_diff),
            "task_ndcg_improvement": task_means(ndcg_diff),
        }
    return {"baseline": "alpha", "methods": results, "samples": sample_rows}
