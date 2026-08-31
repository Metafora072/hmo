"""
HMO Research — Evaluation Harness
Unified generation + evaluation pipeline for all experiments.
"""
import json
import os
import torch
from pathlib import Path
from loguru import logger
from datetime import datetime

from .dataset_utils import EvalSample
from .metrics import (
    GenerationMetrics, LatencyTracker,
    compute_exact_match, compute_f1, compute_rouge_l,
    get_peak_vram_mb, reset_vram_stats,
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
    "hotpotqa": (
        "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
        "The following are given passages.\n"
        "{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
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
    "hotpotqa": 32,
    "gov_report": 512,
    "lcc": 64,
}

LONG_BENCH_NO_CHAT_DATASETS = {
    "trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p",
}


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


def score_prediction(prediction: str, sample: EvalSample) -> tuple[float, float, float]:
    """
    Score one prediction against all available ground truths using the
    official LongBench-style max-over-ground-truths convention.
    """
    ground_truths = get_ground_truths(sample)
    if not ground_truths:
        return 0.0, 0.0, 0.0

    accuracy = max(compute_exact_match(prediction, gt) for gt in ground_truths)
    f1 = max(compute_f1(prediction, gt) for gt in ground_truths)
    rouge_l = max(compute_rouge_l(prediction, gt) for gt in ground_truths)
    return float(accuracy), float(f1), float(rouge_l)


def build_prompt(sample: EvalSample, tokenizer) -> str:
    """
    Build a prompt from an EvalSample.

    For LongBench subsets we align to the official benchmark prompt templates.
    For tasks where the official LongBench code avoids chat wrapping (e.g. LCC),
    we return the raw prompt directly. Otherwise we wrap with the model's chat
    template while disabling thinking mode when supported.
    """
    dataset_key = get_benchmark_key(sample.dataset)
    if dataset_key in LONG_BENCH_PROMPTS:
        prompt = LONG_BENCH_PROMPTS[dataset_key].format(
            context=sample.context,
            input=sample.question,
        )
        if dataset_key in LONG_BENCH_NO_CHAT_DATASETS:
            return prompt
    else:
        prompt = f"{sample.context}\n\nQuestion: {sample.question}\nAnswer directly in a few words:"

    messages = [{"role": "user", "content": prompt}]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    # Disable thinking mode if supported (Qwen3.5)
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


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

    accuracy, f1, _ = score_prediction(generated_text, sample)
    peak_vram = get_peak_vram_mb(device_id)

    return GenerationMetrics(
        accuracy=accuracy,
        f1=f1,
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

            results.append({
                "sample_id": sample.sample_id,
                "dataset": sample.dataset,
                "context_length": sample.context_length,
                "accuracy": metrics.accuracy,
                "f1": metrics.f1,
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
        "timestamp": datetime.now().isoformat(),
    }

    if save_results:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / f"{experiment_name}_{method_name}.json"
        with open(out_path, "w") as f:
            json.dump({"summary": summary, "details": results}, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {out_path}")

    logger.info(f"[{method_name}] Accuracy: {summary['avg_accuracy']:.3f}, F1: {summary['avg_f1']:.3f}")
    return summary
