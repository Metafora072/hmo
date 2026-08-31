"""
V4: Joint HMO Validation
========================
Kill question: Is HMO-full better than either RTS-only or Refresh-only alone?

This version uses the live prototype paths directly:
1. Full KV = upper bound reference only
2. H2O = controller hard-eviction baseline under the shared byte budget
3. RTS-only = forced KV+RTS policy under the shared byte budget
4. Refresh-only = forced KV+refresh policy under the shared byte budget
5. HMO-full = controller's native action policy under the same byte budget

Note:
The core controller semantics are now aligned, but this script remains
guard-blocked until the "Full KV = upper bound" protocol role is frozen.

Usage:
    CUDA_VISIBLE_DEVICES=1 python experiments/v4_joint/run.py \
        --model qwen3.5-0.8b --n_samples 60 --context_length 16384
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
from experiments.utils.dataset_utils import make_needle_samples, load_longbench_subset
from experiments.utils.fidelity_guard import require_theory_faithful
from experiments.utils.metrics import compute_exact_match, reset_vram_stats
from experiments.utils.prototype_runner import (
    build_input_ids,
    collect_sigma_and_segment_costs,
    default_shared_budget_limit,
    refresh_drop_rest_budgeted_actions,
    rts_budgeted_actions,
    run_forced_actions,
    select_triggered_refresh_segments,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def parse_args():
    p = argparse.ArgumentParser(description="V4: Joint HMO Validation")
    p.add_argument("--model", type=str, default="qwen3.5-0.8b")
    p.add_argument("--n_samples", type=int, default=60)
    p.add_argument("--context_length", type=int, default=16384)
    p.add_argument("--segment_length", type=int, default=512)
    p.add_argument("--keep_ratio", type=float, default=0.5)
    p.add_argument("--skeleton_rank", type=int, default=4)
    p.add_argument("--refresh_budget", type=int, default=3)
    p.add_argument("--saturation_threshold", type=float, default=0.7)
    p.add_argument("--refresh_threshold", type=float, default=0.8)
    p.add_argument("--rts_threshold", type=float, default=0.4)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_samples(tokenizer, args):
    """V4 data: HotpotQA-dominant (70%) + Needle (30%).
    V4 tests H4 (joint HMO > components alone). The joint effect requires
    both detector (H1) and refresh (H2) to work, which need density-varying
    data. Needle is included for RTS component validation."""
    n_lb = int(args.n_samples * 0.7)
    n_needle = args.n_samples - n_lb

    samples = []
    try:
        lb = load_longbench_subset(
            "hotpotqa",
            tokenizer,
            n_samples=n_lb,
            context_length=args.context_length,
            seed=args.seed,
        )
        samples.extend(lb)
    except Exception as e:
        logger.warning(f"LongBench load failed: {e}")

    needle_samples = make_needle_samples(
        tokenizer,
        n_samples=n_needle,
        context_length=args.context_length,
        seed=args.seed,
    )
    samples.extend(needle_samples)

    if len(samples) < args.n_samples:
        missing = args.n_samples - len(samples)
        extra = make_needle_samples(
            tokenizer,
            n_samples=missing,
            context_length=args.context_length,
            seed=args.seed + 1000,
        )
        for idx, sample in enumerate(extra):
            sample.sample_id = f"needle_backfill_{idx:04d}"
        samples.extend(extra)
    return samples


@torch.no_grad()
def run_sample(controller, tokenizer, sample, args):
    """Run one sample under a Full-KV upper bound plus four matched HMO conditions."""
    input_ids = build_input_ids(
        sample,
        tokenizer,
        controller.model.device,
        max_length=args.context_length + 256,
    )
    sigma, segment_costs = collect_sigma_and_segment_costs(controller, input_ids)
    shared_budget_limit_bytes = default_shared_budget_limit(segment_costs, args.keep_ratio, n_sigma=len(sigma))
    triggered_segs = select_triggered_refresh_segments(
        sigma,
        budget=min(args.refresh_budget, len(sigma)),
        threshold=args.saturation_threshold,
    )

    results = {
        "full_kv": controller.run_baseline(input_ids, max_new_tokens=args.max_new_tokens),
        "h2o": controller.run_h2o_baseline(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            budget_limit_bytes=shared_budget_limit_bytes,
        ),
    }

    rts_actions = rts_budgeted_actions(
        len(sigma),
        sigma,
        segment_costs,
        budget_limit_bytes=shared_budget_limit_bytes,
        rts_threshold=controller.hmo.rts_threshold,
    )
    refresh_actions = refresh_drop_rest_budgeted_actions(
        len(sigma),
        triggered_segs,
        segment_costs,
        budget_limit_bytes=shared_budget_limit_bytes,
    )

    results["rts_only"] = run_forced_actions(
        controller,
        input_ids,
        rts_actions,
        max_new_tokens=args.max_new_tokens,
        budget_limit_bytes=shared_budget_limit_bytes,
        segment_costs=segment_costs,
    )
    results["refresh_only"] = run_forced_actions(
        controller,
        input_ids,
        refresh_actions,
        max_new_tokens=args.max_new_tokens,
        budget_limit_bytes=shared_budget_limit_bytes,
        segment_costs=segment_costs,
    )
    results["hmo_full"] = controller.run(input_ids, max_new_tokens=args.max_new_tokens)
    results["shared_budget_limit_bytes"] = shared_budget_limit_bytes
    return results


def main():
    args = parse_args()
    require_theory_faithful("v4")
    logger.info(f"V4 Joint HMO — {args.model}, n={args.n_samples}, ctx={args.context_length}")

    model, tokenizer, config = load_model_and_tokenizer(
        args.model,
        device="cuda",
        gpu_id=args.gpu_id,
    )
    controller = HMOController(
        model,
        tokenizer,
        config,
        hmo_config=HMOConfig(
            segment_length=args.segment_length,
            keep_ratio=args.keep_ratio,
            skeleton_rank=args.skeleton_rank,
            refresh_budget=args.refresh_budget,
            saturation_threshold=args.saturation_threshold,
            refresh_threshold=args.refresh_threshold,
            rts_threshold=args.rts_threshold,
        ),
        gpu_id=args.gpu_id,
    )

    samples = build_samples(tokenizer, args)
    conditions = ["full_kv", "h2o", "rts_only", "refresh_only", "hmo_full"]
    cond_accs = {name: [] for name in conditions}
    cond_vram = {name: [] for name in conditions}
    per_sample = []

    for i, sample in enumerate(samples):
        logger.info(f"Sample {i + 1}/{len(samples)}: {sample.sample_id}")

        try:
            reset_vram_stats(args.gpu_id)
            results = run_sample(controller, tokenizer, sample, args)

            for name in conditions:
                cond_accs[name].append(compute_exact_match(results[name].generated_text, sample.answer))
                cond_vram[name].append(results[name].peak_vram_mb)

            per_sample.append({
                "sample_id": sample.sample_id,
                **{f"{name}_acc": cond_accs[name][-1] for name in conditions},
                **{f"{name}_vram": cond_vram[name][-1] for name in conditions},
                **{f"{name}_bytes": results[name].total_tracked_bytes for name in conditions},
                "hmo_n_kept_kv": results["hmo_full"].n_kept_kv,
                "hmo_n_skeleton": results["hmo_full"].n_skeleton,
                "hmo_n_refresh": results["hmo_full"].n_refresh,
                "hmo_n_dropped": results["hmo_full"].n_dropped,
                "shared_budget_limit_bytes": int(results["shared_budget_limit_bytes"]),
            })
        except torch.cuda.OutOfMemoryError:
            logger.error(f"OOM on sample {i + 1}")
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Error on sample {i + 1}: {e}")

    summary = {}
    for name in conditions:
        arr = np.array(cond_accs[name])
        vram = np.array(cond_vram[name])
        summary[name] = {
            "mean": float(arr.mean()) if len(arr) > 0 else 0.0,
            "std": float(arr.std()) if len(arr) > 0 else 0.0,
            "mean_vram": float(vram.mean()) if len(vram) > 0 else 0.0,
            "mean_tracked_bytes": float(np.mean([row[f"{name}_bytes"] for row in per_sample])) if per_sample else 0.0,
            "n": len(arr),
        }

    hmo = summary["hmo_full"]["mean"]
    rts = summary["rts_only"]["mean"]
    refresh = summary["refresh_only"]["mean"]
    h2o = summary["h2o"]["mean"]

    pass_joint = hmo > max(rts, refresh)
    pass_h2o = hmo > h2o
    passed = pass_joint and pass_h2o

    logger.info("=" * 60)
    logger.info(f"V4 RESULTS — {args.model}")
    logger.info("  Full KV is reported as an upper bound and is not budget-matched.")
    for name, stats in summary.items():
        logger.info(
            f"  {name}: {stats['mean']:.3f} ± {stats['std']:.3f}, "
            f"vram={stats['mean_vram']:.0f}MB, bytes={stats['mean_tracked_bytes']:.0f}"
        )
    logger.info(f"  PASS: HMO > max(RTS,Refresh): {pass_joint} | HMO > H2O: {pass_h2o}")
    logger.info(f"  VERDICT: {'PASS' if passed else 'FAIL'}")
    logger.info("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "experiment": "V4_joint_hmo",
        "model": args.model,
        "n_samples": args.n_samples,
        "context_length": args.context_length,
        "segment_length": args.segment_length,
        "timestamp": datetime.now().isoformat(),
        "summary": summary,
        "passed": passed,
        "prototype_alignment": (
            "all five conditions run through the current HMO controller/cache operators; "
            "H2O, RTS-only, refresh-only, and HMO-full are the budget-matched set, "
            "while Full KV is an upper bound only"
        ),
        "full_kv_role": "upper_bound_only",
        "per_sample": per_sample,
    }
    out_path = RESULTS_DIR / f"v4_{args.model.replace('.', '_')}_{args.context_length}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
