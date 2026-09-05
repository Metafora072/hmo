"""
HMO Research — Evaluation Harness
Unified generation + evaluation pipeline for all experiments.
"""
import json
import os
import torch
from dataclasses import dataclass
from pathlib import Path
from loguru import logger
from datetime import datetime

from .dataset_utils import EvalSample
from .metrics import (
    GenerationMetrics, LatencyTracker,
    compute_exact_match, compute_f1,
    get_peak_vram_mb, reset_vram_stats,
)
from experiments.vendor.longbench_metrics import (
    code_sim_score as longbench_code_sim_score,
    qa_f1_score as longbench_qa_f1_score,
    rouge_score as longbench_rouge_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(
    os.environ.get("HMO_RESULTS_ROOT", str(PROJECT_ROOT / "experiments/results"))
)

LONG_BENCH_PROMPTS = {
    "narrativeqa": (
        "You are given a story, which can be either a novel or a movie script, and a question. "
        "Answer the question as concisely as you can, using a single phrase if possible. "
        "Do not provide any explanation.\n\n"
        "Story: {context}\n\n"
        "Now, answer the question based on the story as concisely as you can, using a single phrase if possible. "
        "Do not provide any explanation.\n\n"
        "Question: {input}\n\n"
        "Answer:"
    ),
    "qasper": (
        "You are given a scientific article and a question. Answer the question as concisely as you can, "
        "using a single phrase or sentence if possible. If the question cannot be answered based on the information "
        "in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", "
        "or \"unanswerable\". Do not provide any explanation.\n\n"
        "Article: {context}\n\n"
        "Answer the question based on the above article as concisely as you can, using a single phrase or sentence "
        "if possible. If the question cannot be answered based on the information in the article, write "
        "\"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". "
        "Do not provide any explanation.\n\n"
        "Question: {input}\n\n"
        "Answer:"
    ),
    "multifieldqa_en": (
        "Read the following text and answer briefly.\n\n"
        "{context}\n\n"
        "Now, answer the following question based on the above text, only give me the answer "
        "and do not output any other words.\n\n"
        "Question: {input}\n"
        "Answer:"
    ),
    "hotpotqa": (
        "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
        "The following are given passages.\n"
        "{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
        "Question: {input}\n"
        "Answer:"
    ),
    "2wikimqa": (
        "Answer the question based on the given passages. Only give me the answer and do not "
        "output any other words.\n\n"
        "The following are given passages.\n"
        "{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer and do not "
        "output any other words.\n\n"
        "Question: {input}\n"
        "Answer:"
    ),
    "musique": (
        "Answer the question based on the given passages. Only give me the answer and do not "
        "output any other words.\n\n"
        "The following are given passages.\n"
        "{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer and do not "
        "output any other words.\n\n"
        "Question: {input}\n"
        "Answer:"
    ),
    "gov_report": (
        "You are given a report by a government agency. Write a one-page summary of the report.\n\n"
        "Report:\n{context}\n\n"
        "Now, write a one-page summary of the report.\n\n"
        "Summary:"
    ),
    "lcc": (
        "Please complete the code given below.\n"
        "{context}Next line of code:\n"
    ),
}

LONG_BENCH_MAX_GEN = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "gov_report": 512,
    "lcc": 64,
}

LONG_BENCH_NO_CHAT_DATASETS = {
    "trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p",
}

CONTEXT_QUERY_BOUNDARY_MARKER = "<|hmo_context_query_boundary_7f3c1d|>"


class PromptBoundaryError(ValueError):
    """Raised when a prompt cannot be serialized into an exact context/query split."""


@dataclass(frozen=True)
class PromptTextParts:
    memory_context: str
    query_suffix: str

    @property
    def full_prompt(self) -> str:
        return self.memory_context + self.query_suffix


def get_benchmark_key(dataset: str) -> str:
    """Map internal dataset names to the official LongBench subset key when applicable."""
    if dataset.startswith("longbench_"):
        return dataset[len("longbench_"):]
    return dataset


def resolve_max_new_tokens(dataset: str, fallback: int) -> int:
    """Use official LongBench decoding lengths when available; otherwise keep caller default."""
    key = get_benchmark_key(dataset)
    return int(LONG_BENCH_MAX_GEN.get(key, fallback))


def get_ground_truths(sample: EvalSample) -> list[str]:
    """Return all valid ground truths for one sample."""
    if sample.answers:
        return [ans for ans in sample.answers if ans]
    return [sample.answer] if sample.answer else []


@dataclass(frozen=True)
class PredictionScores:
    accuracy: float = 0.0
    f1: float = 0.0
    rouge_l: float = 0.0
    code_sim: float = 0.0
    primary_metric: str = "accuracy"
    primary_score: float = 0.0


def get_primary_metric_name(dataset: str) -> str:
    key = get_benchmark_key(dataset)
    if key in {
        "hotpotqa",
        "narrativeqa",
        "qasper",
        "multifieldqa_en",
        "2wikimqa",
        "musique",
    }:
        return "f1"
    if key == "gov_report":
        return "rouge_l"
    if key == "lcc":
        return "code_sim"
    if dataset in {"needle", "longeval_lines"}:
        return "accuracy"
    if dataset.startswith("longbench_"):
        raise ValueError(f"No pinned official LongBench metric for {dataset}")
    raise ValueError(f"No registered metric protocol for {dataset}")


def score_prediction(prediction: str, sample: EvalSample) -> PredictionScores:
    """Score a prediction with the dataset's preregistered metric."""
    primary_metric = get_primary_metric_name(sample.dataset)
    ground_truths = get_ground_truths(sample)
    if not ground_truths:
        return PredictionScores(primary_metric=primary_metric)

    if primary_metric == "f1":
        value = max(longbench_qa_f1_score(prediction, gt) for gt in ground_truths)
        return PredictionScores(
            f1=float(value), primary_metric=primary_metric, primary_score=float(value),
        )
    if primary_metric == "rouge_l":
        value = max(longbench_rouge_score(prediction, gt) for gt in ground_truths)
        return PredictionScores(
            rouge_l=float(value), primary_metric=primary_metric, primary_score=float(value),
        )
    if primary_metric == "code_sim":
        value = max(longbench_code_sim_score(prediction, gt) for gt in ground_truths)
        return PredictionScores(
            code_sim=float(value), primary_metric=primary_metric, primary_score=float(value),
        )

    accuracy = max(compute_exact_match(prediction, gt) for gt in ground_truths)
    f1 = max(compute_f1(prediction, gt) for gt in ground_truths)
    return PredictionScores(
        accuracy=float(accuracy), f1=float(f1),
        primary_metric=primary_metric, primary_score=float(accuracy),
    )


def _build_raw_prompt_parts(sample: EvalSample) -> PromptTextParts:
    dataset_key = get_benchmark_key(sample.dataset)
    if dataset_key in LONG_BENCH_PROMPTS:
        template = LONG_BENCH_PROMPTS[dataset_key]
        if template.count("{context}") != 1:
            raise PromptBoundaryError(
                f"Expected exactly one context slot for {dataset_key}"
            )
        prefix_template, suffix_template = template.split("{context}", 1)
        values = {"context": "", "input": sample.question}
        return PromptTextParts(
            memory_context=prefix_template.format(**values) + sample.context,
            query_suffix=suffix_template.format(**values),
        )

    return PromptTextParts(
        memory_context=sample.context,
        query_suffix=(
            f"\n\nQuestion: {sample.question}\nAnswer directly in a few words:"
        ),
    )


def _apply_chat_template(tokenizer, content: str) -> str:
    messages = [{"role": "user", "content": content}]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def build_prompt_parts(sample: EvalSample, tokenizer) -> PromptTextParts:
    """Serialize an exact memory-context/query-suffix prompt boundary."""
    raw_parts = _build_raw_prompt_parts(sample)
    dataset_key = get_benchmark_key(sample.dataset)
    if (
        dataset_key in LONG_BENCH_PROMPTS
        and dataset_key in LONG_BENCH_NO_CHAT_DATASETS
    ):
        return raw_parts

    marker = CONTEXT_QUERY_BOUNDARY_MARKER
    if marker in raw_parts.full_prompt:
        raise PromptBoundaryError("Boundary marker unexpectedly occurs in prompt content")

    marked_content = raw_parts.memory_context + marker + raw_parts.query_suffix
    marked_prompt = _apply_chat_template(tokenizer, marked_content)
    if marked_prompt.count(marker) != 1:
        raise PromptBoundaryError("Chat template did not preserve the context/query marker")
    memory_context, query_suffix = marked_prompt.split(marker, 1)

    full_prompt = _apply_chat_template(tokenizer, raw_parts.full_prompt)
    if memory_context + query_suffix != full_prompt:
        raise PromptBoundaryError("Removing the boundary marker changed prompt serialization")
    return PromptTextParts(memory_context=memory_context, query_suffix=query_suffix)


def build_prompt(sample: EvalSample, tokenizer) -> str:
    """Build the exact full prompt used by generation and P0-B splitting."""
    return build_prompt_parts(sample, tokenizer).full_prompt
@torch.no_grad()
def generate_and_evaluate(
    model,
    tokenizer,
    sample: EvalSample,
    max_new_tokens: int = 128,
    device_id: int = 0,
) -> GenerationMetrics:
    """
    Generate a response and compute metrics for a single sample.
    """
    prompt = build_prompt(sample, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    input_ids = inputs["input_ids"].to(model.device)
    max_new_tokens = resolve_max_new_tokens(sample.dataset, max_new_tokens)

    reset_vram_stats(device_id)
    tracker = LatencyTracker()

    tracker.start()
    outputs = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
    )
    n_new = outputs.shape[1] - input_ids.shape[1]
    tracker.end(n_new)

    generated_ids = outputs[0, input_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    scores = score_prediction(generated_text, sample)
    peak_vram = get_peak_vram_mb(device_id)

    return GenerationMetrics(
        accuracy=scores.accuracy,
        f1=scores.f1,
        rouge_l=scores.rouge_l,
        code_sim=scores.code_sim,
        primary_metric=scores.primary_metric,
        primary_score=scores.primary_score,
        ttft_ms=tracker.ttft_ms,
        decode_latency_ms=tracker.decode_ms,
        tokens_per_sec=tracker.tokens_per_sec,
        peak_vram_mb=peak_vram,
        generated_text=generated_text,
    )


def run_eval_suite(
    model,
    tokenizer,
    samples: list[EvalSample],
    experiment_name: str,
    method_name: str,
    max_new_tokens: int = 128,
    device_id: int = 0,
    save_results: bool = True,
) -> dict:
    """
    Run evaluation on a list of samples and aggregate results.
    """
    results = []
    total_acc = 0.0
    total_f1 = 0.0
    total_primary = 0.0
    primary_metrics = set()

    for i, sample in enumerate(samples):
        logger.info(f"[{method_name}] {i+1}/{len(samples)} — {sample.sample_id}")
        try:
            metrics = generate_and_evaluate(
                model, tokenizer, sample,
                max_new_tokens=max_new_tokens,
                device_id=device_id,
            )
            total_acc += metrics.accuracy
            total_f1 += metrics.f1
            total_primary += metrics.primary_score
            primary_metrics.add(metrics.primary_metric)

            results.append({
                "sample_id": sample.sample_id,
                "dataset": sample.dataset,
                "context_length": sample.context_length,
                "accuracy": metrics.accuracy,
                "f1": metrics.f1,
                "rouge_l": metrics.rouge_l,
                "code_sim": metrics.code_sim,
                "primary_metric": metrics.primary_metric,
                "primary_score": metrics.primary_score,
                "ttft_ms": metrics.ttft_ms,
                "decode_ms": metrics.decode_latency_ms,
                "tok_per_sec": metrics.tokens_per_sec,
                "peak_vram_mb": metrics.peak_vram_mb,
                "generated": metrics.generated_text[:200],
                "answer": sample.answer[:200],
            })
        except Exception as e:
            logger.error(f"Failed on {sample.sample_id}: {e}")
            results.append({
                "sample_id": sample.sample_id,
                "error": str(e),
            })

    n = len(samples)
    summary = {
        "experiment": experiment_name,
        "method": method_name,
        "n_samples": n,
        "avg_accuracy": total_acc / max(n, 1),
        "avg_f1": total_f1 / max(n, 1),
        "primary_metrics": sorted(primary_metrics),
        "avg_primary": total_primary / max(n, 1),
        "timestamp": datetime.now().isoformat(),
    }

    if save_results:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / f"{experiment_name}_{method_name}.json"
        with open(out_path, "w") as f:
            json.dump({"summary": summary, "details": results}, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {out_path}")

    logger.info(f"[{method_name}] Primary: {summary['avg_primary']:.3f} ({summary['primary_metrics']})")
    return summary
