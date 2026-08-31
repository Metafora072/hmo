"""
V4fast: Joint HMO Quick Screen
==============================
Goal: check whether the full HMO orchestration is directionally stronger
than the component-only variants on a small mixed screen.
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
from experiments.utils.fidelity_guard import require_theory_faithful
from experiments.utils.metrics import compute_exact_match
from experiments.utils.fast_screen import paired_win_rate
from experiments.v4_joint.run import build_samples, run_sample

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def parse_args():
    p = argparse.ArgumentParser(description="V4fast: joint HMO screen")
    p.add_argument("--model", type=str, default="qwen3.5-0.8b")
    p.add_argument("--n_samples", type=int, default=16)
    p.add_argument("--context_length", type=int, default=16384)
    p.add_argument("--segment_length", type=int, default=512)
    p.add_argument("--keep_ratio", type=float, default=0.5)
    p.add_argument("--skeleton_rank", type=int, default=2)
    p.add_argument("--refresh_budget", type=int, default=1)
    p.add_argument("--saturation_threshold", type=float, default=0.7)
    p.add_argument("--refresh_threshold", type=float, default=0.8)
    p.add_argument("--rts_threshold", type=float, default=0.4)
    p.add_argument("--gpu_id", type=int, default=1)
    p.add_argument("--max_new_tokens", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    require_theory_faithful("v4fast")
    logger.info(f"V4fast — {args.model}, n={args.n_samples}, ctx={args.context_length}")

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
    per_sample = []

    for i, sample in enumerate(samples):
        logger.info(f"Sample {i + 1}/{len(samples)}: {sample.sample_id}")
        try:
            results = run_sample(controller, tokenizer, sample, args)
            for name in conditions:
                cond_accs[name].append(compute_exact_match(results[name].generated_text, sample.answer))

            per_sample.append({
                "sample_id": sample.sample_id,
                **{f"{name}_acc": cond_accs[name][-1] for name in conditions},
                "hmo_n_kept_kv": results["hmo_full"].n_kept_kv,
                "hmo_n_skeleton": results["hmo_full"].n_skeleton,
                "hmo_n_refresh": results["hmo_full"].n_refresh,
                "hmo_n_dropped": results["hmo_full"].n_dropped,
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

    hmo_vs_h2o = paired_win_rate(cond_accs["hmo_full"], cond_accs["h2o"])
    hmo_vs_rts = paired_win_rate(cond_accs["hmo_full"], cond_accs["rts_only"])
    hmo_vs_refresh = paired_win_rate(cond_accs["hmo_full"], cond_accs["refresh_only"])
    screened_positive = (
        summary["hmo_full"]["mean"] >= summary["h2o"]["mean"]
        and summary["hmo_full"]["mean"] >= summary["rts_only"]["mean"]
        and summary["hmo_full"]["mean"] >= summary["refresh_only"]["mean"]
        and hmo_vs_h2o >= 0.55
    )

    logger.info("=" * 60)
    logger.info(f"V4fast RESULTS — {args.model}")
    for name, stats in summary.items():
        logger.info(f"  {name}: {stats['mean']:.3f} ± {stats['std']:.3f}")
    logger.info(f"  HMO vs H2O win-rate: {hmo_vs_h2o:.3f}")
    logger.info(f"  HMO vs RTS-only win-rate: {hmo_vs_rts:.3f}")
    logger.info(f"  HMO vs refresh-only win-rate: {hmo_vs_refresh:.3f}")
    logger.info(f"  screen-positive: {screened_positive}")
    logger.info("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "experiment": "V4fast_joint",
        "model": args.model,
        "n_samples": args.n_samples,
        "context_length": args.context_length,
        "segment_length": args.segment_length,
        "timestamp": datetime.now().isoformat(),
        "summary": summary,
        "hmo_vs_h2o_win_rate": hmo_vs_h2o,
        "hmo_vs_rts_only_win_rate": hmo_vs_rts,
        "hmo_vs_refresh_only_win_rate": hmo_vs_refresh,
        "screened_positive": screened_positive,
        "per_sample": per_sample,
    }
    out_path = RESULTS_DIR / f"v4fast_{args.model.replace('.', '_')}_{args.context_length}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
