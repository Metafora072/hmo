"""
V2: State-to-Token Refresh Validation
=====================================
Kill question: Does saturation-triggered refresh replay improve accuracy?

This version uses the current HMO prototype directly:
1. No intervention = canonical refresh baseline (protected KV, other middle segments dropped)
2. Random refresh = add refresh slots on random middle segments
3. Periodic refresh = add refresh slots on evenly spaced middle segments
4. Triggered refresh = add refresh slots on top H2-weak predictor segments
   (`sigma x attention_score`) on middle segments

All four conditions share one explicit per-sample total-memory budget cap.

Usage:
    CUDA_VISIBLE_DEVICES=1 python experiments/v2_refresh/run.py \
        --model qwen3.5-0.8b --n_samples 80 --context_length 16384
"""
import sys
import json
import argparse
import random
import numpy as np
import torch
from pathlib import Path
from loguru import logger
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.utils.model_loader import load_model_and_tokenizer
from experiments.utils.hmo_controller import HMOController, HMOConfig
from experiments.utils.dataset_utils import make_needle_samples, load_longbench_subset
from experiments.utils.metrics import (
    compute_exact_match,
    reset_vram_stats,
    get_peak_vram_mb,
)
from experiments.utils.fidelity_guard import require_theory_faithful
from experiments.utils.prototype_runner import (
    build_input_ids,
    collect_segment_attention_scores,
    collect_sigma_and_segment_costs,
    estimate_action_bytes,
    middle_segments,
    protected_kv_drop_rest_actions,
    refresh_drop_rest_actions,
    run_forced_actions,
    select_periodic_segments,
    select_triggered_refresh_segments,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def parse_args():
    p = argparse.ArgumentParser(description="V2: Refresh Validation")
    p.add_argument("--model", type=str, default="qwen3.5-0.8b")
    p.add_argument("--n_samples", type=int, default=80)
    p.add_argument("--context_length", type=int, default=16384)
    p.add_argument("--segment_length", type=int, default=512)
    p.add_argument("--refresh_budget", type=int, default=3, help="Max segments to replay")
    p.add_argument("--saturation_threshold", type=float, default=0.7)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def sample_seed(base_seed: int, sample_id: str) -> int:
    return base_seed + sum(ord(ch) for ch in sample_id)


def build_samples(tokenizer, args):
    """
    V2 data: HotpotQA-dominant mix with Needle as a secondary stress test.
    This keeps V2 aligned with the V1 conclusion that the useful trigger is
    H2-weak rather than sigma-only.
    """
    n_lb = int(args.n_samples * 0.8)
    n_needle = args.n_samples - n_lb

    samples = []
    try:
        lb_samples = load_longbench_subset(
            "hotpotqa",
            tokenizer,
            n_samples=n_lb,
            context_length=args.context_length,
            seed=args.seed,
        )
        samples.extend(lb_samples)
        if len(lb_samples) < n_lb:
            logger.warning(
                f"HotpotQA returned {len(lb_samples)}/{n_lb} samples "
                f"(context length filtering). Backfilling with extra Needle."
            )
    except Exception as exc:
        raise RuntimeError(
            f"V2 now requires HotpotQA in the mix because triggered refresh uses "
            f"the V1-validated H2-weak path on information-density-varying data. "
            f"HotpotQA loading failed: {exc}"
        ) from exc

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


def select_triggered_refresh_segments_h2_weak(
    sigma: np.ndarray,
    attention_scores: np.ndarray,
    budget: int,
) -> tuple[list[int], np.ndarray]:
    """Select triggered refresh segments using the V1-validated H2-weak predictor."""
    candidates = middle_segments(len(sigma))
    if budget <= 0 or not candidates:
        return [], sigma * attention_scores

    trigger_scores = sigma * attention_scores
    ranked = sorted(candidates, key=lambda idx: float(trigger_scores[idx]), reverse=True)
    return ranked[:budget], trigger_scores


@torch.no_grad()
def run_sample_all_conditions(controller, tokenizer, sample, args):
    """Run one sample under all four refresh conditions."""
    input_ids = build_input_ids(
        sample,
        tokenizer,
        controller.model.device,
        max_length=args.context_length + 256,
    )
    sigma, segment_costs = collect_sigma_and_segment_costs(controller, input_ids)
    attention_scores = collect_segment_attention_scores(controller, input_ids)

    eligible = middle_segments(len(sigma))
    budget = min(args.refresh_budget, len(eligible))
    triggered_segs, trigger_scores = select_triggered_refresh_segments_h2_weak(
        sigma,
        attention_scores,
        budget=budget,
    )
    n_refresh = len(triggered_segs)

    rng = random.Random(sample_seed(args.seed, sample.sample_id))
    random_segs = rng.sample(eligible, min(n_refresh, len(eligible))) if n_refresh > 0 else []
    periodic_segs = select_periodic_segments(eligible, n_refresh)

    action_maps = {
        "no_intervention": protected_kv_drop_rest_actions(len(sigma)),
        "random_refresh": refresh_drop_rest_actions(len(sigma), random_segs),
        "periodic_refresh": refresh_drop_rest_actions(len(sigma), periodic_segs),
        "triggered_refresh": refresh_drop_rest_actions(len(sigma), triggered_segs),
    }
    shared_budget_limit_bytes = max(
        estimate_action_bytes(segment_costs, actions) for actions in action_maps.values()
    ) if action_maps else 0

    results = {}
    refresh_sets = {
        "no_intervention": [],
        "random_refresh": random_segs,
        "periodic_refresh": periodic_segs,
        "triggered_refresh": triggered_segs,
    }
    for cond_name, forced_actions in action_maps.items():
        refresh_result = run_forced_actions(
            controller,
            input_ids,
            forced_actions,
            max_new_tokens=args.max_new_tokens,
            budget_limit_bytes=shared_budget_limit_bytes,
        )
        results[cond_name] = {
            "accuracy": compute_exact_match(refresh_result.generated_text, sample.answer),
            "n_refresh_segs": len(refresh_sets[cond_name]),
            "refresh_segs": refresh_sets[cond_name],
            "generated": refresh_result.generated_text[:200],
            "tracked_bytes": int(refresh_result.total_tracked_bytes),
        }

    return results, sigma, attention_scores, trigger_scores, shared_budget_limit_bytes


def main():
    args = parse_args()
    require_theory_faithful("v2")
    logger.info(f"V2 Refresh — {args.model}, n={args.n_samples}, ctx={args.context_length}")

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
            refresh_budget=args.refresh_budget,
            saturation_threshold=args.saturation_threshold,
            refresh_threshold=args.saturation_threshold,
        ),
        gpu_id=args.gpu_id,
    )

    samples = build_samples(tokenizer, args)

    conditions = ["no_intervention", "random_refresh", "periodic_refresh", "triggered_refresh"]
    cond_accs = {name: [] for name in conditions}
    per_sample = []

    for i, sample in enumerate(samples):
        logger.info(f"Sample {i + 1}/{len(samples)}: {sample.sample_id}")
        reset_vram_stats(args.gpu_id)

        try:
            results, sigma, attention_scores, trigger_scores, shared_budget_limit_bytes = run_sample_all_conditions(
                controller, tokenizer, sample, args
            )

            for cond_name, res in results.items():
                cond_accs[cond_name].append(res["accuracy"])

            per_sample.append({
                "sample_id": sample.sample_id,
                "dataset": sample.dataset,
                **{f"{name}_acc": results[name]["accuracy"] for name in conditions},
                **{f"{name}_bytes": results[name]["tracked_bytes"] for name in conditions},
                "sigma_max": float(sigma.max()) if len(sigma) > 0 else 0.0,
                "attention_max": float(attention_scores.max()) if len(attention_scores) > 0 else 0.0,
                "trigger_score_max": float(trigger_scores.max()) if len(trigger_scores) > 0 else 0.0,
                "n_refresh": results["triggered_refresh"]["n_refresh_segs"],
                "shared_budget_limit_bytes": int(shared_budget_limit_bytes),
                "peak_vram_mb": get_peak_vram_mb(args.gpu_id),
            })

            if (i + 1) % 20 == 0:
                for name in conditions:
                    avg = np.mean(cond_accs[name]) if cond_accs[name] else 0.0
                    logger.info(f"  {name}: {avg:.3f} (n={len(cond_accs[name])})")

        except torch.cuda.OutOfMemoryError:
            logger.error(f"OOM on sample {i + 1}")
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Error on sample {i + 1}: {e}")

    summary = {}
    for name in conditions:
        arr = np.array(cond_accs[name])
        summary[name] = {
            "mean": float(arr.mean()) if len(arr) > 0 else 0.0,
            "std": float(arr.std()) if len(arr) > 0 else 0.0,
            "n": len(arr),
        }

    trig = summary["triggered_refresh"]["mean"]
    no_int = summary["no_intervention"]["mean"]
    rand_ref = summary["random_refresh"]["mean"]

    passed = (trig > no_int + 0.05) and (trig > rand_ref)

    logger.info("=" * 60)
    logger.info(f"V2 RESULTS — {args.model}")
    for name, stats in summary.items():
        logger.info(f"  {name}: {stats['mean']:.3f} ± {stats['std']:.3f}")
    logger.info("  PASS condition: triggered > no_intervention+5% AND triggered > random")
    logger.info(f"  VERDICT: {'PASS' if passed else 'FAIL'}")
    logger.info("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "experiment": "V2_refresh",
        "model": args.model,
        "n_samples": args.n_samples,
        "context_length": args.context_length,
        "segment_length": args.segment_length,
        "timestamp": datetime.now().isoformat(),
        "summary": summary,
        "passed": passed,
        "prototype_alignment": (
            "refresh conditions use the canonical protected-KV/drop-rest baseline "
            "plus selected refresh segments under one shared per-sample byte cap; "
            "triggered refresh uses the V1-validated H2-weak predictor "
            "(sigma x attention_score)"
        ),
        "per_sample": per_sample,
    }
    out_path = RESULTS_DIR / f"v2_{args.model.replace('.', '_')}_{args.context_length}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
