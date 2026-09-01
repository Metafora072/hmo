"""
HMO Research — Metrics
Accuracy, F1, latency, memory metrics for evaluation.
"""
import re
import time
import torch
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field

try:
    from sklearn.metrics import roc_auc_score as _sklearn_roc_auc_score
except Exception:
    _sklearn_roc_auc_score = None

try:
    from scipy.stats import pearsonr as _scipy_pearsonr
    from scipy.stats import spearmanr as _scipy_spearmanr
except Exception:
    _scipy_pearsonr = None
    _scipy_spearmanr = None


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Small average-rank helper used when scipy is unavailable."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def _pearson_fallback(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or y.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0, 1.0
    return float(np.corrcoef(x, y)[0, 1]), 1.0


def _spearman_fallback(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    return _pearson_fallback(_rankdata(x), _rankdata(y))


def _roc_auc_fallback(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = _rankdata(scores)
    rank_sum_pos = float(ranks[pos].sum())
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


@dataclass
class GenerationMetrics:
    """Metrics from a single generation."""
    accuracy: float = 0.0
    f1: float = 0.0
    rouge_l: float = 0.0
    code_sim: float = 0.0
    primary_metric: str = "accuracy"
    primary_score: float = 0.0
    ttft_ms: float = 0.0          # time to first token
    decode_latency_ms: float = 0.0  # total decode time
    tokens_per_sec: float = 0.0
    peak_vram_mb: float = 0.0
    generated_text: str = ""


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower().strip()
    # Remove articles
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    # Remove punctuation
    s = re.sub(r'[^\w\s]', '', s)
    # Remove extra whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 score."""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens or not gt_tokens:
        return float(pred_tokens == gt_tokens)

    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Check if the ground truth appears in the prediction."""
    return float(normalize_answer(ground_truth) in normalize_answer(prediction))


def compute_correlation(scores: np.ndarray, labels: np.ndarray):
    """Compute Pearson, Spearman correlation and AUC."""
    results = {}
    if len(np.unique(labels)) < 2:
        return {"pearson": 0.0, "spearman": 0.0, "auc": 0.5, "warning": "labels have <2 unique values"}

    if _scipy_pearsonr is not None:
        r_p, p_p = _scipy_pearsonr(scores, labels)
    else:
        r_p, p_p = _pearson_fallback(scores, labels)

    if _scipy_spearmanr is not None:
        r_s, p_s = _scipy_spearmanr(scores, labels)
    else:
        r_s, p_s = _spearman_fallback(scores, labels)
    results["pearson"] = float(r_p)
    results["pearson_p"] = float(p_p)
    results["spearman"] = float(r_s)
    results["spearman_p"] = float(p_s)

    # AUC (binarize labels if not already)
    binary_labels = (labels > 0).astype(int)
    if len(np.unique(binary_labels)) >= 2:
        if _sklearn_roc_auc_score is not None:
            results["auc"] = float(_sklearn_roc_auc_score(binary_labels, scores))
        else:
            results["auc"] = _roc_auc_fallback(binary_labels, scores)
    else:
        results["auc"] = 0.5

    return results


class LatencyTracker:
    """Track TTFT, decode latency, throughput."""

    def __init__(self):
        self.start_time = 0.0
        self.first_token_time = 0.0
        self.end_time = 0.0
        self.n_tokens = 0

    def start(self):
        torch.cuda.synchronize()
        self.start_time = time.perf_counter()

    def mark_first_token(self):
        torch.cuda.synchronize()
        self.first_token_time = time.perf_counter()

    def end(self, n_tokens: int):
        torch.cuda.synchronize()
        self.end_time = time.perf_counter()
        self.n_tokens = n_tokens

    @property
    def ttft_ms(self) -> float:
        if self.first_token_time == 0:
            return 0.0
        return (self.first_token_time - self.start_time) * 1000

    @property
    def decode_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000

    @property
    def tokens_per_sec(self) -> float:
        elapsed = self.end_time - self.start_time
        if elapsed <= 0:
            return 0.0
        return self.n_tokens / elapsed


def get_peak_vram_mb(device_id: int = 0) -> float:
    """Get peak VRAM usage in MB."""
    return torch.cuda.max_memory_allocated(device_id) / (1024 ** 2)


def reset_vram_stats(device_id: int = 0):
    """Reset peak VRAM tracking."""
    torch.cuda.reset_peak_memory_stats(device_id)


def compute_rouge_l(prediction: str, ground_truth: str) -> float:
    """ROUGE-L F1 score based on longest common subsequence."""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return float(pred_tokens == gt_tokens)

    m, n = len(gt_tokens), len(pred_tokens)
    # LCS via DP
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if gt_tokens[i - 1] == pred_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs_len = dp[m][n]
    if lcs_len == 0:
        return 0.0
    precision = lcs_len / n
    recall = lcs_len / m
    return 2 * precision * recall / (precision + recall)
