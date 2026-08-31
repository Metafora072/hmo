"""
E3: Dual-Channel Mechanism Analysis
====================================
Claim: phi=sigma*alpha joint signal predicts refresh value better than either alone.

Model: Qwen3.5-27B BF16
Data: 80% HotpotQA + 20% Needle, 100 samples, 32K
Collects per-segment signals + single-segment refresh oracle labels.

Usage:
    CUDA_VISIBLE_DEVICES=1 python experiments/phase2/e3_mechanism/run.py
"""
import sys
import json
import argparse
import random
import hashlib
import numpy as np
import torch
from pathlib import Path
from loguru import logger
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.utils.model_loader import load_model_and_tokenizer
from experiments.utils.hmo_controller import HMOController, HMOConfig
from experiments.utils.dataset_utils import make_needle_samples, load_longbench_subset
from experiments.utils.metrics import reset_vram_stats
from experiments.utils.eval_harness import score_prediction
from experiments.utils.prototype_runner import (
    build_input_ids,
    collect_sigma_and_segment_costs,
    collect_segment_attention_scores,
    protected_segments,
    middle_segments,
    refresh_drop_rest_actions,
    protected_kv_drop_rest_actions,
    run_forced_actions,
    default_shared_budget_limit,
)
from experiments.utils.saturation import compute_segment_saturation
from experiments.phase2.runner import get_results_dir


def parse_args():
    p = argparse.ArgumentParser(description="E3: Mechanism Analysis")
    p.add_argument("--model", type=str, default="qwen3.5-27b")
    p.add_argument("--n_samples", type=int, default=100)
    p.add_argument("--context_length", type=int, default=32768)
    p.add_argument("--segment_length", type=int, default=512)
    p.add_argument("--keep_ratio", type=float, default=0.5)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--oracle_sample_rate", type=int, default=10,
                   help="Test 1 in N middle segments for oracle refresh gain")
    p.add_argument("--run-name", type=str, default="e3_mechanism",
                   help="Results subdirectory name")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def build_samples(tokenizer, args):
    n_lb = int(args.n_samples * 0.8)
    n_needle = args.n_samples - n_lb
    samples = []
    try:
        lb = load_longbench_subset("hotpotqa", tokenizer, n_lb, args.context_length, args.seed)
        samples.extend(lb)
    except Exception as e:
        logger.warning(f"LongBench load failed: {e}")
    samples.extend(make_needle_samples(tokenizer, n_needle, args.context_length, args.seed))
    if len(samples) < args.n_samples:
        extra = make_needle_samples(tokenizer, args.n_samples - len(samples), args.context_length, args.seed + 1000)
        samples.extend(extra)
    return samples


@torch.no_grad()
def run_sample_mechanism(controller, tokenizer, sample, args):
    """Collect signals + oracle refresh gain for one sample."""
    input_ids = build_input_ids(sample, tokenizer, controller.model.device, max_length=args.context_length + 256)

    # Collect sigma and raw sub-signals (memory-efficient: skip full logits)
    controller.hook_mgr.clear()
    controller.hook_mgr.attach()
    try:
        outputs, cache, _ = controller._prefill_with_cache_last_logits(input_ids)
        signals = dict(controller.hook_mgr.get_signals())
    finally:
        controller.hook_mgr.remove()
    del outputs, cache
    torch.cuda.empty_cache()

    sigma = compute_segment_saturation(
        signals,
        alpha_rho=controller.hmo.alpha_rho,
        alpha_c=controller.hmo.alpha_c,
        alpha_g=controller.hmo.alpha_g,
        segment_length=controller.hmo.segment_length,
        warmup_tokens=controller.hmo.warmup_tokens,
        input_ids=input_ids,
        repeat_ngram_size=controller.hmo.repeat_ngram_size,
        repeat_threshold=controller.hmo.repeat_threshold,
    )

    # Collect per-sub-signal scores (normalized, through saturation pipeline)
    # These show each sub-signal's contribution after normalization/warmup/filtering.
    sigma_rho = compute_segment_saturation(
        signals, alpha_rho=1.0, alpha_c=0.0, alpha_g=0.0,
        segment_length=controller.hmo.segment_length,
        warmup_tokens=controller.hmo.warmup_tokens,
        input_ids=input_ids,
        repeat_ngram_size=controller.hmo.repeat_ngram_size,
        repeat_threshold=controller.hmo.repeat_threshold,
    )
    sigma_c = compute_segment_saturation(
        signals, alpha_rho=0.0, alpha_c=1.0, alpha_g=0.0,
        segment_length=controller.hmo.segment_length,
        warmup_tokens=controller.hmo.warmup_tokens,
        input_ids=input_ids,
        repeat_ngram_size=controller.hmo.repeat_ngram_size,
        repeat_threshold=controller.hmo.repeat_threshold,
    )
    sigma_g = compute_segment_saturation(
        signals, alpha_rho=0.0, alpha_c=0.0, alpha_g=1.0,
        segment_length=controller.hmo.segment_length,
        warmup_tokens=controller.hmo.warmup_tokens,
        input_ids=input_ids,
        repeat_ngram_size=controller.hmo.repeat_ngram_size,
        repeat_threshold=controller.hmo.repeat_threshold,
    )

    # Collect attention fragility (alpha)
    alpha = collect_segment_attention_scores(controller, input_ids)
    n_segs = min(len(sigma), len(alpha))
    sigma = sigma[:n_segs]
    alpha = alpha[:n_segs]
    phi = sigma * alpha

    # Baseline: protected KV + drop rest
    sigma_full, segment_costs = collect_sigma_and_segment_costs(controller, input_ids)
    budget = default_shared_budget_limit(segment_costs, controller.hmo.keep_ratio, n_sigma=len(sigma_full))
    baseline_actions = protected_kv_drop_rest_actions(n_segs)
    baseline_result = run_forced_actions(
        controller, input_ids, baseline_actions,
        max_new_tokens=args.max_new_tokens,
        budget_limit_bytes=budget, segment_costs=segment_costs,
    )
    baseline_acc, baseline_f1, _ = score_prediction(baseline_result.generated_text, sample)

    # Oracle: test single-segment refresh on stratified subset of middle segments
    mid = middle_segments(n_segs)
    # Randomize oracle segment selection to avoid position bias
    # Use deterministic seed from sample_id for reproducibility
    seed_int = int(hashlib.md5(sample.sample_id.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed_int)
    test_segs = rng.sample(mid, min(max(len(mid) // args.oracle_sample_rate, 1), len(mid)))
    oracle_gains = {}
    for seg_idx in test_segs:
        refresh_actions = refresh_drop_rest_actions(n_segs, [seg_idx])
        try:
            refresh_result = run_forced_actions(
                controller, input_ids, refresh_actions,
                max_new_tokens=args.max_new_tokens,
                budget_limit_bytes=budget, segment_costs=segment_costs,
            )
            _, refresh_f1, _ = score_prediction(refresh_result.generated_text, sample)
            oracle_gains[seg_idx] = refresh_f1 - baseline_f1
        except Exception as e:
            logger.warning(f"Oracle refresh failed for seg {seg_idx}: {e}")
            oracle_gains[seg_idx] = 0.0
        torch.cuda.empty_cache()

    # Also extract raw (unnormalized) per-segment signals from hooks for completeness
    raw_rho, raw_c, raw_g = [], [], []
    for seg_idx in range(n_segs):
        rho_vals, c_vals, g_vals = [], [], []
        for layer_idx, seg_sig in signals.items():
            if seg_idx < len(seg_sig.rho_max):
                rho_vals.append(seg_sig.rho_max[seg_idx])
            if seg_idx < len(seg_sig.c_max):
                c_vals.append(seg_sig.c_max[seg_idx])
            if seg_idx < len(seg_sig.g_mag_min):
                g_vals.append(seg_sig.g_mag_min[seg_idx])
        raw_rho.append(max(rho_vals) if rho_vals else 0.0)
        raw_c.append(max(c_vals) if c_vals else 0.0)
        raw_g.append(min(g_vals) if g_vals else 0.0)

    return {
        "sample_id": sample.sample_id,
        "dataset": sample.dataset,
        "n_segments": n_segs,
        "sigma": sigma.tolist(),
        "sigma_rho": sigma_rho[:n_segs].tolist(),
        "sigma_c": sigma_c[:n_segs].tolist(),
        "sigma_g": sigma_g[:n_segs].tolist(),
        "raw_rho": raw_rho,
        "raw_c": raw_c,
        "raw_g": raw_g,
        "alpha": alpha.tolist(),
        "phi": phi.tolist(),
        "baseline_acc": baseline_acc,
        "baseline_f1": baseline_f1,
        "oracle_gains": {str(k): v for k, v in oracle_gains.items()},
    }


def main():
    args = parse_args()
    logger.info(f"E3 Mechanism — {args.model}, n={args.n_samples}, ctx={args.context_length}")

    model, tokenizer, config = load_model_and_tokenizer(args.model, device="cuda", gpu_id=args.gpu_id)
    controller = HMOController(
        model, tokenizer, config,
        hmo_config=HMOConfig(segment_length=args.segment_length, keep_ratio=args.keep_ratio),
        gpu_id=args.gpu_id,
    )

    samples = build_samples(tokenizer, args)
    results_dir = get_results_dir(args.run_name)
    output_path = results_dir / "e3_mechanism.jsonl"

    completed_ids = set()
    if args.resume and output_path.exists():
        with open(output_path) as f:
            for line in f:
                row = json.loads(line.strip())
                completed_ids.add(row["sample_id"])

    for i, sample in enumerate(samples):
        if sample.sample_id in completed_ids:
            continue
        logger.info(f"Sample {i+1}/{len(samples)}: {sample.sample_id}")
        try:
            reset_vram_stats(args.gpu_id)
            result = run_sample_mechanism(controller, tokenizer, sample, args)
            with open(output_path, "a") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        except torch.cuda.OutOfMemoryError:
            logger.error(f"OOM on sample {i+1}")
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Error on sample {i+1}: {e}")

    logger.info(f"E3 complete. Output: {output_path}")


if __name__ == "__main__":
    main()
