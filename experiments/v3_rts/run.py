"""
V3: RTS Skeleton Validation
===========================
Kill question: Does the prototype's RTS path beat hard eviction under the same
shared byte budget?

This version is aligned with the current HMO prototype:
1. Full KV = upper bound reference only
2. Hard eviction = controller H2O baseline under the shared byte budget
3. RTS skeleton = forced controller policy with KV+RTS only under the same byte budget

Note:
The core RTS mechanism now uses explicit low-rank storage plus decode-time
reconstruction. This script is still guard-blocked until the experiment itself
is frozen around the shared memory accountant and the "Full KV = upper bound"
protocol role.

Usage:
    CUDA_VISIBLE_DEVICES=1 python experiments/v3_rts/run.py \
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
    rts_budgeted_actions,
    run_forced_actions,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def parse_args():
    p = argparse.ArgumentParser(description="V3: RTS Skeleton Validation")
    p.add_argument("--model", type=str, default="qwen3.5-0.8b")
    p.add_argument("--n_samples", type=int, default=60)
    p.add_argument("--context_length", type=int, default=16384)
    p.add_argument("--segment_length", type=int, default=512)
    p.add_argument("--keep_ratio", type=float, default=0.5, help="Fraction of middle segments kept as full KV")
    p.add_argument("--skeleton_rank", type=int, default=4)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_samples(tokenizer, args):
    """V3 data: Needle-dominant (70%) + HotpotQA (30%).
    V3 tests H3 (RTS skeleton > hard eviction). Needle's repetitive filler
    produces low-rank KV matrices, which is favorable for SVD compression.
    HotpotQA is included to verify RTS also works on natural text."""
    n_needle = int(args.n_samples * 0.7)
    n_lb = args.n_samples - n_needle

    samples = make_needle_samples(
        tokenizer,
        n_samples=n_needle,
        context_length=args.context_length,
        seed=args.seed,
    )
    try:
        lb_samples = load_longbench_subset(
            "hotpotqa",
            tokenizer,
            n_samples=n_lb,
            context_length=args.context_length,
            seed=args.seed,
        )
        samples.extend(lb_samples)
    except Exception as e:
        logger.warning(f"Could not load LongBench: {e}, using only Needle")

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
    """Run one sample under full-KV upper bound plus matched H2O/RTS conditions."""
    input_ids = build_input_ids(
        sample,
        tokenizer,
        controller.model.device,
        max_length=args.context_length + 256,
    )
    sigma, segment_costs = collect_sigma_and_segment_costs(controller, input_ids)
    shared_budget_limit_bytes = default_shared_budget_limit(segment_costs, args.keep_ratio, n_sigma=len(sigma))

    full_result = controller.run_baseline(input_ids, max_new_tokens=args.max_new_tokens)
    h2o_result = controller.run_h2o_baseline(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        budget_limit_bytes=shared_budget_limit_bytes,
    )

    forced_actions = rts_budgeted_actions(
        len(sigma),
        sigma,
        segment_costs,
        budget_limit_bytes=shared_budget_limit_bytes,
        rts_threshold=controller.hmo.rts_threshold,
    )
    rts_result = run_forced_actions(
        controller,
        input_ids,
        forced_actions,
        max_new_tokens=args.max_new_tokens,
        budget_limit_bytes=shared_budget_limit_bytes,
        segment_costs=segment_costs,
    )

    return {
        "full_kv": full_result,
        "h2o_eviction": h2o_result,
        "rts_skeleton": rts_result,
        "shared_budget_limit_bytes": shared_budget_limit_bytes,
    }


def main():
    args = parse_args()
    require_theory_faithful("v3")
    logger.info(f"V3 RTS Skeleton — {args.model}, n={args.n_samples}, ctx={args.context_length}")

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
        ),
        gpu_id=args.gpu_id,
    )

    samples = build_samples(tokenizer, args)

    conditions = ["full_kv", "h2o_eviction", "rts_skeleton"]
    cond_accs = {name: [] for name in conditions}
    cond_vram = {name: [] for name in conditions}
    cond_meta = {name: [] for name in conditions}
    per_sample = []

    for i, sample in enumerate(samples):
        logger.info(f"Sample {i + 1}/{len(samples)}: {sample.sample_id}")

        try:
            reset_vram_stats(args.gpu_id)
            results = run_sample(controller, tokenizer, sample, args)

            for name, res in results.items():
                if name == "shared_budget_limit_bytes":
                    continue
                cond_accs[name].append(compute_exact_match(res.generated_text, sample.answer))
                cond_vram[name].append(res.peak_vram_mb)
                cond_meta[name].append({
                    "n_kept_kv": res.n_kept_kv,
                    "n_skeleton": res.n_skeleton,
                    "n_refresh": res.n_refresh,
                    "n_dropped": res.n_dropped,
                    "tracked_bytes": res.total_tracked_bytes,
                    "budget_limit_bytes": res.budget_limit_bytes,
                })

            per_sample.append({
                "sample_id": sample.sample_id,
                **{f"{name}_acc": cond_accs[name][-1] for name in conditions},
                **{f"{name}_vram": cond_vram[name][-1] for name in conditions},
                **{f"{name}_bytes": results[name].total_tracked_bytes for name in conditions},
                "rts_n_kept_kv_segments": results["rts_skeleton"].n_kept_kv,
                "rts_n_skeleton_segments": results["rts_skeleton"].n_skeleton,
                "h2o_kept_tokens": results["h2o_eviction"].n_kept_kv,
                "shared_budget_limit_bytes": int(results["shared_budget_limit_bytes"]),
            })

        except torch.cuda.OutOfMemoryError:
            logger.error(f"OOM on sample {i + 1}")
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Error on sample {i + 1}: {e}")

    summary = {}
    for name in conditions:
        acc = np.array(cond_accs[name])
        vram = np.array(cond_vram[name])
        meta = cond_meta[name]
        summary[name] = {
            "mean_acc": float(acc.mean()) if len(acc) > 0 else 0.0,
            "std_acc": float(acc.std()) if len(acc) > 0 else 0.0,
            "mean_vram": float(vram.mean()) if len(vram) > 0 else 0.0,
            "mean_kept_kv": float(np.mean([m["n_kept_kv"] for m in meta])) if meta else 0.0,
            "mean_skeleton": float(np.mean([m["n_skeleton"] for m in meta])) if meta else 0.0,
            "mean_dropped": float(np.mean([m["n_dropped"] for m in meta])) if meta else 0.0,
            "mean_tracked_bytes": float(np.mean([m["tracked_bytes"] for m in meta])) if meta else 0.0,
            "mean_budget_limit_bytes": float(np.mean([m["budget_limit_bytes"] for m in meta])) if meta else 0.0,
            "n": len(acc),
        }

    rts_acc = summary["rts_skeleton"]["mean_acc"]
    h2o_acc = summary["h2o_eviction"]["mean_acc"]
    passed = rts_acc > h2o_acc + 0.03

    logger.info("=" * 60)
    logger.info(f"V3 RESULTS — {args.model}")
    logger.info("  Full KV is reported as an upper bound and is not budget-matched.")
    for name, stats in summary.items():
        logger.info(
            f"  {name}: acc={stats['mean_acc']:.3f}±{stats['std_acc']:.3f}, "
            f"vram={stats['mean_vram']:.0f}MB, bytes={stats['mean_tracked_bytes']:.0f}/"
            f"{stats['mean_budget_limit_bytes']:.0f}, kept={stats['mean_kept_kv']:.1f}, "
            f"skeleton={stats['mean_skeleton']:.1f}, dropped={stats['mean_dropped']:.1f}"
        )
    logger.info("  PASS condition: RTS_acc > H2O_acc + 3%")
    logger.info(f"  VERDICT: {'PASS' if passed else 'FAIL'}")
    logger.info("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "experiment": "V3_rts_skeleton",
        "model": args.model,
        "n_samples": args.n_samples,
        "context_length": args.context_length,
        "segment_length": args.segment_length,
        "keep_ratio": args.keep_ratio,
        "skeleton_rank": args.skeleton_rank,
        "timestamp": datetime.now().isoformat(),
        "summary": summary,
        "passed": passed,
        "prototype_note": (
            "Full KV is an upper bound only; H2O and RTS-only are the budget-matched "
            "comparison pair and both report tracked bytes explicitly."
        ),
        "full_kv_role": "upper_bound_only",
        "per_sample": per_sample,
    }
    out_path = RESULTS_DIR / f"v3_{args.model.replace('.', '_')}_{args.context_length}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
