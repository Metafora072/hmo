"""
V3fast: RTS Mechanism Quick Screen
==================================
Goal: decide whether RTS-only at least directionally beats H2O on a
small mixed Needle + HotpotQA screen.
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
from experiments.v3_rts.run import build_samples, run_sample

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def parse_args():
    p = argparse.ArgumentParser(description="V3fast: RTS mechanism screen")
    p.add_argument("--model", type=str, default="qwen3.5-0.8b")
    p.add_argument("--n_samples", type=int, default=16)
    p.add_argument("--context_length", type=int, default=16384)
    p.add_argument("--segment_length", type=int, default=512)
    p.add_argument("--keep_ratio", type=float, default=0.5)
    p.add_argument("--skeleton_rank", type=int, default=2)
    p.add_argument("--gpu_id", type=int, default=1)
    p.add_argument("--max_new_tokens", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    require_theory_faithful("v3fast")
    logger.info(f"V3fast — {args.model}, n={args.n_samples}, ctx={args.context_length}")

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
    per_sample = []

    for i, sample in enumerate(samples):
        logger.info(f"Sample {i + 1}/{len(samples)}: {sample.sample_id}")
        try:
            results = run_sample(controller, tokenizer, sample, args)
            for name, res in results.items():
                acc = compute_exact_match(res.generated_text, sample.answer)
                cond_accs[name].append(acc)
                cond_vram[name].append(res.peak_vram_mb)

            per_sample.append({
                "sample_id": sample.sample_id,
                **{f"{name}_acc": cond_accs[name][-1] for name in conditions},
                "rts_n_kept_kv": results["rts_skeleton"].n_kept_kv,
                "rts_n_skeleton": results["rts_skeleton"].n_skeleton,
            })
        except torch.cuda.OutOfMemoryError:
            logger.error(f"OOM on sample {i + 1}")
            torch.cuda.empty_cache()
        except Exception as exc:
            logger.error(f"Error on sample {i + 1}: {exc}")

    summary = {}
    for name in conditions:
        arr = np.asarray(cond_accs[name], dtype=float)
        vram = np.asarray(cond_vram[name], dtype=float)
        summary[name] = {
            "mean_acc": float(arr.mean()) if arr.size > 0 else 0.0,
            "std_acc": float(arr.std()) if arr.size > 0 else 0.0,
            "mean_vram": float(vram.mean()) if vram.size > 0 else 0.0,
            "n": int(arr.size),
        }

    rts_vs_h2o_win_rate = paired_win_rate(cond_accs["rts_skeleton"], cond_accs["h2o_eviction"])
    screened_positive = (
        summary["rts_skeleton"]["mean_acc"] >= summary["h2o_eviction"]["mean_acc"]
        and rts_vs_h2o_win_rate >= 0.55
    )

    logger.info("=" * 60)
    logger.info(f"V3fast RESULTS — {args.model}")
    for name, stats in summary.items():
        logger.info(
            f"  {name}: acc={stats['mean_acc']:.3f} ± {stats['std_acc']:.3f}, "
            f"vram={stats['mean_vram']:.0f}MB"
        )
    logger.info(f"  RTS vs H2O win-rate: {rts_vs_h2o_win_rate:.3f}")
    logger.info(f"  screen-positive: {screened_positive}")
    logger.info("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "experiment": "V3fast_rts",
        "model": args.model,
        "n_samples": args.n_samples,
        "context_length": args.context_length,
        "segment_length": args.segment_length,
        "timestamp": datetime.now().isoformat(),
        "summary": summary,
        "rts_vs_h2o_win_rate": rts_vs_h2o_win_rate,
        "screened_positive": screened_positive,
        "note": "Fast screen ignores final-paper VRAM compression claims and focuses on directional accuracy only.",
        "per_sample": per_sample,
    }
    out_path = RESULTS_DIR / f"v3fast_{args.model.replace('.', '_')}_{args.context_length}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
