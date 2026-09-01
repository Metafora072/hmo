"""P0-D sample-grouped ranking and incremental-value statistics."""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from experiments.phase2.e3_v2.oracle import OracleContractError


@dataclass(frozen=True)
class SegmentEvidence:
    sample_id: str
    dataset: str
    segment_id: int
    utility: float
    alpha: float
    normalized_position: float
    candidates: Mapping[str, float]


@dataclass(frozen=True)
class SampleRankingMetrics:
    sample_id: str
    dataset: str
    pairwise_accuracy: float
    ndcg: float


@dataclass(frozen=True)
class BootstrapInterval:
    mean: float
    lower: float
    upper: float
    n_samples: int
    n_bootstrap: int


@dataclass(frozen=True)
class CandidateCVResult:
    candidate: str
    baseline_metrics: tuple[SampleRankingMetrics, ...]
    augmented_metrics: tuple[SampleRankingMetrics, ...]
    pairwise_improvement: BootstrapInterval
    ndcg_improvement: BootstrapInterval
    task_pairwise_improvement: Mapping[str, float]
    task_ndcg_improvement: Mapping[str, float]


def _finite_array(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise OracleContractError(f"{name} must be one-dimensional and finite")
    return array


def pairwise_ranking_accuracy(
    predictor: Sequence[float],
    utility: Sequence[float],
    *,
    tie_credit: float = 0.5,
) -> float:
    """Score every utility-ordered pair; true utility ties are uninformative."""
    predictor = _finite_array(predictor, "predictor")
    utility = _finite_array(utility, "utility")
    if predictor.shape != utility.shape:
        raise OracleContractError("predictor and utility lengths disagree")
    credit = 0.0
    comparisons = 0
    for left in range(len(utility)):
        for right in range(left + 1, len(utility)):
            truth = np.sign(utility[left] - utility[right])
            if truth == 0:
                continue
            prediction = np.sign(predictor[left] - predictor[right])
            comparisons += 1
            if prediction == truth:
                credit += 1.0
            elif prediction == 0:
                credit += tie_credit
    if comparisons == 0:
        raise OracleContractError("pairwise accuracy requires one non-tied utility pair")
    return credit / comparisons


def ndcg_at_k(
    predictor: Sequence[float],
    utility: Sequence[float],
    k: int,
) -> float:
    """NDCG with within-sample min-shifted nonnegative utility relevance."""
    predictor = _finite_array(predictor, "predictor")
    utility = _finite_array(utility, "utility")
    if predictor.shape != utility.shape or len(utility) == 0:
        raise OracleContractError("NDCG inputs must be non-empty and aligned")
    if k <= 0:
        raise OracleContractError("NDCG k must be positive")
    k = min(int(k), len(utility))
    relevance = utility - utility.min()
    discounts = 1.0 / np.log2(np.arange(2, k + 2, dtype=np.float64))
    predicted_order = np.argsort(-predictor, kind="stable")[:k]
    ideal_order = np.argsort(-relevance, kind="stable")[:k]
    dcg = float(np.sum(relevance[predicted_order] * discounts))
    ideal = float(np.sum(relevance[ideal_order] * discounts))
    return 1.0 if ideal <= 1e-12 else dcg / ideal


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def pearson_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    left = _finite_array(left, "left")
    right = _finite_array(right, "right")
    if left.shape != right.shape or len(left) < 2:
        raise OracleContractError("correlation inputs must be aligned with at least two values")
    left = left - left.mean()
    right = right - right.mean()
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return 0.0 if denominator <= 1e-12 else float(np.dot(left, right) / denominator)


def spearman_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    left = _finite_array(left, "left")
    right = _finite_array(right, "right")
    if left.shape != right.shape:
        raise OracleContractError("Spearman inputs must be aligned")
    return pearson_correlation(_average_ranks(left), _average_ranks(right))


def residual_correlation(
    candidate: Sequence[float],
    utility: Sequence[float],
    alpha: Sequence[float],
    position: Sequence[float],
) -> float:
    """Pearson association after linear residualization on alpha and position."""
    candidate = _finite_array(candidate, "candidate")
    utility = _finite_array(utility, "utility")
    alpha = _finite_array(alpha, "alpha")
    position = _finite_array(position, "position")
    if not (candidate.shape == utility.shape == alpha.shape == position.shape):
        raise OracleContractError("residual-correlation inputs must be aligned")
    controls = np.column_stack((np.ones(len(alpha)), alpha, position))
    candidate_fit = controls @ np.linalg.lstsq(controls, candidate, rcond=None)[0]
    utility_fit = controls @ np.linalg.lstsq(controls, utility, rcond=None)[0]
    return pearson_correlation(candidate - candidate_fit, utility - utility_fit)


def alpha_bin_pairwise_accuracy(
    candidate: Sequence[float],
    utility: Sequence[float],
    alpha: Sequence[float],
    *,
    bins: int = 4,
    tie_credit: float = 0.5,
) -> float:
    """Compare candidate ordering only within within-sample alpha quantile bins."""
    candidate = _finite_array(candidate, "candidate")
    utility = _finite_array(utility, "utility")
    alpha = _finite_array(alpha, "alpha")
    if not (candidate.shape == utility.shape == alpha.shape):
        raise OracleContractError("alpha-bin inputs must be aligned")
    if bins <= 1:
        raise OracleContractError("alpha bins must be greater than one")
    quantiles = np.quantile(alpha, np.linspace(0, 1, bins + 1))
    assignments = np.searchsorted(quantiles[1:-1], alpha, side="right")
    total_credit = 0.0
    total_pairs = 0
    for bin_index in range(bins):
        indices = np.flatnonzero(assignments == bin_index)
        for offset, left in enumerate(indices):
            for right in indices[offset + 1 :]:
                truth = np.sign(utility[left] - utility[right])
                if truth == 0:
                    continue
                prediction = np.sign(candidate[left] - candidate[right])
                total_pairs += 1
                if prediction == truth:
                    total_credit += 1.0
                elif prediction == 0:
                    total_credit += tie_credit
    if total_pairs == 0:
        raise OracleContractError("alpha bins contain no informative pair")
    return total_credit / total_pairs


def sample_grouped_bootstrap_interval(
    values_by_sample: Mapping[str, float],
    *,
    n_bootstrap: int = 2000,
    seed: int = 20260901,
) -> BootstrapInterval:
    """Percentile CI from resampling sample IDs, never individual segments."""
    if n_bootstrap <= 0:
        raise OracleContractError("n_bootstrap must be positive")
    sample_ids = sorted(values_by_sample)
    values = _finite_array([values_by_sample[sample_id] for sample_id in sample_ids], "sample values")
    if len(values) < 2:
        raise OracleContractError("sample bootstrap requires at least two samples")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    means = values[draws].mean(axis=1)
    return BootstrapInterval(
        mean=float(values.mean()),
        lower=float(np.quantile(means, 0.025)),
        upper=float(np.quantile(means, 0.975)),
        n_samples=len(values),
        n_bootstrap=n_bootstrap,
    )


def _group_folds(sample_ids: Sequence[str], folds: int, seed: int) -> dict[str, int]:
    unique = sorted(
        set(sample_ids),
        key=lambda sample_id: hashlib.sha256(
            f"{seed}|{sample_id}".encode("utf-8")
        ).hexdigest(),
    )
    folds = min(folds, len(unique))
    if folds < 2:
        raise OracleContractError("grouped CV requires at least two samples")
    return {sample_id: index % folds for index, sample_id in enumerate(unique)}


def _fit_ridge_predict(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    ridge_lambda: float,
) -> np.ndarray:
    mean = train_features.mean(axis=0)
    scale = train_features.std(axis=0)
    scale[scale < 1e-8] = 1.0
    train = (train_features - mean) / scale
    test = (test_features - mean) / scale
    train = np.column_stack((np.ones(len(train)), train))
    test = np.column_stack((np.ones(len(test)), test))
    penalty = np.eye(train.shape[1], dtype=np.float64) * ridge_lambda
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(train.T @ train + penalty, train.T @ train_targets)
    return test @ coefficients


def _validate_evidence(evidence: Sequence[SegmentEvidence], candidate: str) -> None:
    if not evidence:
        raise OracleContractError("candidate evaluation requires segment evidence")
    seen = set()
    samples: dict[str, int] = {}
    for row in evidence:
        key = (row.sample_id, row.segment_id)
        if key in seen:
            raise OracleContractError("segment evidence IDs must be unique within sample")
        seen.add(key)
        samples[row.sample_id] = samples.get(row.sample_id, 0) + 1
        values = (row.utility, row.alpha, row.normalized_position, row.candidates.get(candidate))
        if values[-1] is None or not all(math.isfinite(float(value)) for value in values):
            raise OracleContractError("segment evidence contains missing or non-finite values")
    if any(count < 2 for count in samples.values()):
        raise OracleContractError("each sample needs at least two eligible segments")


def evaluate_candidate_grouped_cv(
    evidence: Sequence[SegmentEvidence],
    candidate: str,
    *,
    k_by_sample: Mapping[str, int],
    folds: int = 5,
    ridge_lambda: float = 0.001,
    bootstrap_samples: int = 2000,
    seed: int = 20260901,
) -> CandidateCVResult:
    """Compare attention+position against the same predictor plus one candidate."""
    evidence = tuple(evidence)
    _validate_evidence(evidence, candidate)
    fold_by_sample = _group_folds([row.sample_id for row in evidence], folds, seed)
    baseline_predictions = np.empty(len(evidence), dtype=np.float64)
    augmented_predictions = np.empty(len(evidence), dtype=np.float64)
    targets = np.asarray([row.utility for row in evidence], dtype=np.float64)
    baseline_features = np.asarray(
        [[row.alpha, row.normalized_position] for row in evidence],
        dtype=np.float64,
    )
    augmented_features = np.asarray(
        [
            [row.alpha, row.normalized_position, float(row.candidates[candidate])]
            for row in evidence
        ],
        dtype=np.float64,
    )
    sample_ids = np.asarray([row.sample_id for row in evidence], dtype=object)
    for fold in sorted(set(fold_by_sample.values())):
        test_mask = np.asarray(
            [fold_by_sample[sample_id] == fold for sample_id in sample_ids],
            dtype=bool,
        )
        train_mask = ~test_mask
        baseline_predictions[test_mask] = _fit_ridge_predict(
            baseline_features[train_mask],
            targets[train_mask],
            baseline_features[test_mask],
            ridge_lambda,
        )
        augmented_predictions[test_mask] = _fit_ridge_predict(
            augmented_features[train_mask],
            targets[train_mask],
            augmented_features[test_mask],
            ridge_lambda,
        )

    baseline_metrics = []
    augmented_metrics = []
    for sample_id in sorted(set(sample_ids)):
        indices = np.flatnonzero(sample_ids == sample_id)
        rows = [evidence[index] for index in indices]
        datasets = {row.dataset for row in rows}
        if len(datasets) != 1:
            raise OracleContractError("one sample cannot span multiple datasets")
        if sample_id not in k_by_sample:
            raise OracleContractError(f"missing NDCG k for sample {sample_id}")
        dataset = next(iter(datasets))
        baseline_metrics.append(
            SampleRankingMetrics(
                sample_id=sample_id,
                dataset=dataset,
                pairwise_accuracy=pairwise_ranking_accuracy(
                    baseline_predictions[indices],
                    targets[indices],
                ),
                ndcg=ndcg_at_k(
                    baseline_predictions[indices],
                    targets[indices],
                    k_by_sample[sample_id],
                ),
            )
        )
        augmented_metrics.append(
            SampleRankingMetrics(
                sample_id=sample_id,
                dataset=dataset,
                pairwise_accuracy=pairwise_ranking_accuracy(
                    augmented_predictions[indices],
                    targets[indices],
                ),
                ndcg=ndcg_at_k(
                    augmented_predictions[indices],
                    targets[indices],
                    k_by_sample[sample_id],
                ),
            )
        )

    baseline_by_id = {metric.sample_id: metric for metric in baseline_metrics}
    augmented_by_id = {metric.sample_id: metric for metric in augmented_metrics}
    pairwise_diff = {
        sample_id: augmented_by_id[sample_id].pairwise_accuracy
        - baseline_by_id[sample_id].pairwise_accuracy
        for sample_id in baseline_by_id
    }
    ndcg_diff = {
        sample_id: augmented_by_id[sample_id].ndcg - baseline_by_id[sample_id].ndcg
        for sample_id in baseline_by_id
    }

    def task_means(values: Mapping[str, float]) -> dict[str, float]:
        grouped: dict[str, list[float]] = {}
        for metric in baseline_metrics:
            grouped.setdefault(metric.dataset, []).append(values[metric.sample_id])
        return {
            dataset: float(np.mean(dataset_values))
            for dataset, dataset_values in sorted(grouped.items())
        }

    return CandidateCVResult(
        candidate=candidate,
        baseline_metrics=tuple(baseline_metrics),
        augmented_metrics=tuple(augmented_metrics),
        pairwise_improvement=sample_grouped_bootstrap_interval(
            pairwise_diff,
            n_bootstrap=bootstrap_samples,
            seed=seed,
        ),
        ndcg_improvement=sample_grouped_bootstrap_interval(
            ndcg_diff,
            n_bootstrap=bootstrap_samples,
            seed=seed + 1,
        ),
        task_pairwise_improvement=task_means(pairwise_diff),
        task_ndcg_improvement=task_means(ndcg_diff),
    )


def select_discovery_candidate(
    results: Sequence[CandidateCVResult],
) -> CandidateCVResult:
    """Apply the frozen discovery-only selection rule before confirmation."""
    if not results:
        raise OracleContractError("candidate selection requires CV results")
    names = [result.candidate for result in results]
    if len(names) != len(set(names)):
        raise OracleContractError("candidate CV result names must be unique")
    return sorted(
        results,
        key=lambda result: (
            -result.pairwise_improvement.mean,
            -result.ndcg_improvement.mean,
            result.candidate,
        ),
    )[0]
