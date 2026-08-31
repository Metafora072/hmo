"""
V1fast: Saturation Detector Quick Screen
========================================
Goal: decide quickly whether sigma appears directionally useful.

This fast screen keeps the current prototype semantics but uses:
  - fewer Needle samples
  - shorter context
  - fewer oracle refresh probes per sample
  - lightweight screening criteria instead of final-paper thresholds
"""
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from loguru import logger
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.utils.model_loader import load_model_and_tokenizer
from experiments.utils.hmo_controller import HMOController, HMOConfig
from experiments.utils.dataset_utils import make_needle_samples
from experiments.utils.fidelity_guard import require_theory_faithful
from experiments.utils.metrics import compute_correlation, reset_vram_stats, get_peak_vram_mb
from experiments.utils.fast_screen import safe_mean, top_bottom_split
from experiments.v1_saturation.run import choose_test_segments
from experiments.utils.prototype_runner import (
    all_kv_actions,
    build_answer_ids,
    build_input_ids,
    collect_sigma,
    prepare_cache_with_actions,
    refresh_only_actions,
    score_answer_logprob,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def parse_args():
    p = argparse.ArgumentParser(description="V1fast: saturation detector screen")
    p.add_argument("--model", type=str, default="qwen3.5-0.8b")
    p.add_argument("--n_samples", type=int, default=12)
    p.add_argument("--context_length", type=int, default=16384)
    p.add_argument("--segment_length", type=int, default=512)
    p.add_argument("--gpu_id", type=int, default=1)
    p.add_argument("--max_new_tokens", type=int, default=8)
    p.add_argument("--oracle_sample_rate", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def run_sample_logprob(controller, tokenizer, sample, args):
    """
    Fast V1 retest with a more sensitive oracle:
    compare the gold answer logprob before/after a single-segment refresh.
    """
    input_ids = build_input_ids(
        sample,
        tokenizer,
        controller.model.device,
        max_length=args.context_length + 256,
    )
    answer_ids = build_answer_ids(sample, tokenizer, controller.model.device)

    sigma = collect_sigma(controller, input_ids)
    n_segs = len(sigma)

    baseline_actions = all_kv_actions(n_segs)
    _, baseline_cache, baseline_logits, baseline_rts = prepare_cache_with_actions(
        controller,
        input_ids,
        baseline_actions,
    )
    baseline_score = score_answer_logprob(
        controller,
        input_ids,
        answer_ids,
        baseline_cache,
        baseline_logits,
        baseline_rts,
    )

    oracle_labels = {}
    for seg_idx in choose_test_segments(sigma, args, sample.sample_id):
        forced_actions = refresh_only_actions(n_segs, [seg_idx])
        _, refresh_cache, refresh_logits, refresh_rts = prepare_cache_with_actions(
            controller,
            input_ids,
            forced_actions,
        )
        refresh_score = score_answer_logprob(
            controller,
            input_ids,
            answer_ids,
            refresh_cache,
            refresh_logits,
            refresh_rts,
        )
        oracle_labels[seg_idx] = float(refresh_score - baseline_score)
        torch.cuda.empty_cache()

    return {
        "sigma": sigma,
        "baseline_score": float(baseline_score),
        "oracle_labels": oracle_labels,
        "n_segs": n_segs,
        "seq_len": int(input_ids.shape[1]),
    }


def main():
    args = parse_args()
    require_theory_faithful("v1fast")
    logger.info(f"V1fast — {args.model}, n={args.n_samples}, ctx={args.context_length}")

    model, tokenizer, config = load_model_and_tokenizer(
        args.model,
        device="cuda",
        gpu_id=args.gpu_id,
    )
    controller = HMOController(
        model,
        tokenizer,
        config,
        hmo_config=HMOConfig(segment_length=args.segment_length),
        gpu_id=args.gpu_id,
    )

    samples = make_needle_samples(
        tokenizer,
        n_samples=args.n_samples,
        context_length=args.context_length,
        seed=args.seed,
    )

    all_sigma = []
    all_gain = []
    per_sample = []

    for i, sample in enumerate(samples):
        logger.info(f"Sample {i + 1}/{len(samples)}: {sample.sample_id}")
        reset_vram_stats(args.gpu_id)
        try:
            result = run_sample_logprob(controller, tokenizer, sample, args)
            for seg_idx, gain in result["oracle_labels"].items():
                if seg_idx < len(result["sigma"]):
                    all_sigma.append(float(result["sigma"][seg_idx]))
                    all_gain.append(float(gain))

            per_sample.append({
                "sample_id": sample.sample_id,
                "n_segments": result["n_segs"],
                "n_tested": len(result["oracle_labels"]),
                "baseline_score": result["baseline_score"],
                "peak_vram_mb": get_peak_vram_mb(args.gpu_id),
            })
        except torch.cuda.OutOfMemoryError:
            logger.error(f"OOM on sample {i + 1}")
            torch.cuda.empty_cache()
        except Exception as exc:
            logger.error(f"Error on sample {i + 1}: {exc}")
        finally:
            torch.cuda.empty_cache()

    sigma_arr = np.asarray(all_sigma, dtype=float)
    gain_arr = np.asarray(all_gain, dtype=float)

    if sigma_arr.size >= 10:
        corr = compute_correlation(sigma_arr, gain_arr)
        low_idx, high_idx = top_bottom_split(sigma_arr, top_frac=0.3)
        high_gain = safe_mean(gain_arr[high_idx])
        low_gain = safe_mean(gain_arr[low_idx])
        direction_ok = high_gain > low_gain
        auc_ok = corr.get("auc", 0.0) >= 0.60
        screened_positive = bool(direction_ok or auc_ok)
    else:
        corr = {"warning": f"only {sigma_arr.size} tested segments"}
        high_gain = 0.0
        low_gain = 0.0
        direction_ok = False
        auc_ok = False
        screened_positive = False

    logger.info("=" * 60)
    logger.info(f"V1fast RESULTS — {args.model}")
    logger.info(f"  tested segments: {sigma_arr.size}")
    logger.info(f"  Pearson: {corr.get('pearson', 0.0):.4f}")
    logger.info(f"  Spearman: {corr.get('spearman', 0.0):.4f}")
    logger.info(f"  AUC: {corr.get('auc', 0.0):.4f}")
    logger.info(f"  high-sigma mean gain: {high_gain:.4f}")
    logger.info(f"  low-sigma mean gain: {low_gain:.4f}")
    logger.info(f"  screen-positive: {screened_positive}")
    logger.info("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "experiment": "V1fast_saturation_detection",
        "model": args.model,
        "n_samples": args.n_samples,
        "context_length": args.context_length,
        "segment_length": args.segment_length,
        "oracle_mode": "gold_answer_logprob_gain",
        "timestamp": datetime.now().isoformat(),
        "n_tested_segments": int(sigma_arr.size),
        "correlation": corr,
        "high_sigma_mean_gain": high_gain,
        "low_sigma_mean_gain": low_gain,
        "direction_ok": direction_ok,
        "auc_ok": auc_ok,
        "screened_positive": screened_positive,
        "per_sample": per_sample,
    }
    out_path = RESULTS_DIR / f"v1fast_{args.model.replace('.', '_')}_{args.context_length}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
