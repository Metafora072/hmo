"""
V2fast: Refresh Mechanism Quick Screen
=====================================
Goal: check whether saturation-triggered refresh beats random/periodic
often enough to justify running the expensive full validation.
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
from experiments.utils.fast_screen import paired_win_rate
from experiments.v2_refresh.run import run_sample_all_conditions

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def parse_args():
    p = argparse.ArgumentParser(description="V2fast: refresh mechanism screen")
    p.add_argument("--model", type=str, default="qwen3.5-0.8b")
    p.add_argument("--n_samples", type=int, default=24)
    p.add_argument("--context_length", type=int, default=16384)
    p.add_argument("--segment_length", type=int, default=512)
    p.add_argument("--refresh_budget", type=int, default=1)
    p.add_argument("--saturation_threshold", type=float, default=0.7)
    p.add_argument("--gpu_id", type=int, default=1)
    p.add_argument("--max_new_tokens", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    require_theory_faithful("v2fast")
    logger.info(f"V2fast — {args.model}, n={args.n_samples}, ctx={args.context_length}")

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

    samples = make_needle_samples(
        tokenizer,
        n_samples=args.n_samples,
        context_length=args.context_length,
        seed=args.seed,
    )

    conditions = ["no_intervention", "random_refresh", "periodic_refresh", "triggered_refresh"]
    cond_accs = {name: [] for name in conditions}
    per_sample = []

    for i, sample in enumerate(samples):
        logger.info(f"Sample {i + 1}/{len(samples)}: {sample.sample_id}")
        try:
            results, sigma = run_sample_all_conditions(controller, tokenizer, sample, args)
            for name in conditions:
                cond_accs[name].append(results[name]["accuracy"])
            per_sample.append({
                "sample_id": sample.sample_id,
                **{f"{name}_acc": results[name]["accuracy"] for name in conditions},
                "sigma_max": float(sigma.max()) if len(sigma) > 0 else 0.0,
                "n_refresh": results["triggered_refresh"]["n_refresh_segs"],
            })
        except torch.cuda.OutOfMemoryError:
            logger.error(f"OOM on sample {i + 1}")
            torch.cuda.empty_cache()
        except Exception as exc:
            logger.error(f"Error on sample {i + 1}: {exc}")

    summary = {}
    for name in conditions:
        arr = np.asarray(cond_accs[name], dtype=float)
        summary[name] = {
            "mean": float(arr.mean()) if arr.size > 0 else 0.0,
            "std": float(arr.std()) if arr.size > 0 else 0.0,
            "n": int(arr.size),
        }

    trig = cond_accs["triggered_refresh"]
    rand = cond_accs["random_refresh"]
    periodic = cond_accs["periodic_refresh"]
    no_int = cond_accs["no_intervention"]

    trigger_vs_random = paired_win_rate(trig, rand)
    trigger_vs_periodic = paired_win_rate(trig, periodic)
    trigger_vs_no = paired_win_rate(trig, no_int)
    screened_positive = (
        summary["triggered_refresh"]["mean"] >= summary["random_refresh"]["mean"]
        and summary["triggered_refresh"]["mean"] >= summary["periodic_refresh"]["mean"]
        and trigger_vs_random >= 0.55
        and trigger_vs_periodic >= 0.55
    )

    logger.info("=" * 60)
    logger.info(f"V2fast RESULTS — {args.model}")
    for name, stats in summary.items():
        logger.info(f"  {name}: {stats['mean']:.3f} ± {stats['std']:.3f}")
    logger.info(f"  trigger vs random win-rate: {trigger_vs_random:.3f}")
    logger.info(f"  trigger vs periodic win-rate: {trigger_vs_periodic:.3f}")
    logger.info(f"  trigger vs no-intervention win-rate: {trigger_vs_no:.3f}")
    logger.info(f"  screen-positive: {screened_positive}")
    logger.info("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "experiment": "V2fast_refresh",
        "model": args.model,
        "n_samples": args.n_samples,
        "context_length": args.context_length,
        "segment_length": args.segment_length,
        "timestamp": datetime.now().isoformat(),
        "summary": summary,
        "trigger_vs_random_win_rate": trigger_vs_random,
        "trigger_vs_periodic_win_rate": trigger_vs_periodic,
        "trigger_vs_no_intervention_win_rate": trigger_vs_no,
        "screened_positive": screened_positive,
        "per_sample": per_sample,
    }
    out_path = RESULTS_DIR / f"v2fast_{args.model.replace('.', '_')}_{args.context_length}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
