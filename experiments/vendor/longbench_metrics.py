"""Minimal English-metric subset from the official LongBench evaluator.

Source: https://github.com/THUDM/LongBench/blob/
2e00731f8d0bff23dc4325161044d0ed8af94c1e/LongBench/metrics.py

Adapted to omit unused Chinese/classification/retrieval dependencies. Metric
semantics are unchanged. See ``LONGBENCH_LICENSE`` for the upstream MIT terms.
"""
from __future__ import annotations

import re
import string
from collections import Counter

LONG_BENCH_REVISION = "2e00731f8d0bff23dc4325161044d0ed8af94c1e"
LONG_BENCH_SOURCE = (
    "https://github.com/THUDM/LongBench/blob/"
    f"{LONG_BENCH_REVISION}/LongBench/metrics.py"
)


def normalize_answer(text: str) -> str:
    """Lower text and remove ASCII punctuation, articles, and extra whitespace."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def f1_score(prediction: list[str], ground_truth: list[str]) -> float:
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction)
    recall = num_same / len(ground_truth)
    return float((2 * precision * recall) / (precision + recall))


def qa_f1_score(prediction: str, ground_truth: str) -> float:
    return f1_score(
        normalize_answer(prediction).split(),
        normalize_answer(ground_truth).split(),
    )


def code_sim_score(prediction: str, ground_truth: str) -> float:
    try:
        from fuzzywuzzy import fuzz
    except ImportError as exc:
        raise RuntimeError(
            "LongBench LCC scoring requires fuzzywuzzy==0.18.0"
        ) from exc

    candidate = ""
    for line in prediction.lstrip("\n").split("\n"):
        if "`" not in line and "#" not in line and "//" not in line:
            candidate = line
            break
    return float(fuzz.ratio(candidate, ground_truth) / 100)


def rouge_score(prediction: str, ground_truth: str) -> float:
    try:
        from rouge import Rouge
    except ImportError as exc:
        raise RuntimeError("LongBench GovReport scoring requires rouge") from exc

    try:
        scores = Rouge().get_scores([prediction], [ground_truth], avg=True)
    except Exception:
        return 0.0
    return float(scores["rouge-l"]["f"])
