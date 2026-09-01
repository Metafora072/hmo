"""
E2: Component Ablation
======================
Claim: RTS and refresh each contribute, phi-triggered > periodic.

Model: Qwen3.5-27B-GPTQ-Int4
Data: 80% HotpotQA + 20% Needle, 100 samples, 32K
Methods: full_kv / h2o / rts_only / refresh_only / hmo_periodic / hmo_full

Usage:
    CUDA_VISIBLE_DEVICES=1 python experiments/phase2/e2_ablation/run.py
"""
import sys
import json
import argparse
import time
import torch
from pathlib import Path
from loguru import logger
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.utils.model_loader import load_model_and_tokenizer
from experiments.utils.hmo_controller import HMOController, HMOConfig
from experiments.utils.dataset_utils import make_needle_samples, load_longbench_subset
from experiments.phase2.runner import (
    get_named_results_dir,
    initialize_formal_run,
    run_sample_all_methods,
    save_cell,
    load_completed_cells,
    summarize_cells,
)

METHODS = ["full_kv", "h2o", "rts_only", "refresh_only", "hmo_periodic", "hmo_full"]


def parse_args():
    p = argparse.ArgumentParser(description="E2: Component Ablation")
    p.add_argument("--model", type=str, default="qwen3.5-27b-gptq-int4")
    p.add_argument("--n_samples", type=int, default=100)
    p.add_argument("--context_length", type=int, default=32768)
    p.add_argument("--segment_length", type=int, default=512)
    p.add_argument("--keep_ratio", type=float, default=0.5)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--run-name", type=str, default=None)
    return p.parse_args()


# PLACEHOLDER_E2_BUILD


def build_samples(tokenizer, args):
    """E2 data: 80% HotpotQA + 20% Needle at 32K."""
    n_lb = int(args.n_samples * 0.8)
    n_needle = args.n_samples - n_lb
    samples = []
    try:
        lb = load_longbench_subset("hotpotqa", tokenizer, n_lb, args.context_length, args.seed)
        samples.extend(lb)
    except Exception as e:
        logger.warning(f"LongBench load failed: {e}")
    needle = make_needle_samples(tokenizer, n_needle, args.context_length, args.seed)
    samples.extend(needle)
    if len(samples) < args.n_samples:
        extra = make_needle_samples(tokenizer, args.n_samples - len(samples), args.context_length, args.seed + 1000)
        samples.extend(extra)
    return samples


def main():
    args = parse_args()
    logger.info(f"E2 Ablation — {args.model}, n={args.n_samples}, ctx={args.context_length}")

    results_dir = get_named_results_dir("e2_ablation", args.run_name)
    manifest = initialize_formal_run(
        results_dir, "e2_ablation", args,
        {
            "benchmarks": ["longbench_hotpotqa", "needle"],
            "context_lengths": [args.context_length],
            "methods": METHODS,
            "mixture": {"longbench_hotpotqa": 0.8, "needle": 0.2},
        },
    )
    logger.info(f"Run manifest: {manifest['manifest_id']}")

    model, tokenizer, config = load_model_and_tokenizer(args.model, device="cuda", gpu_id=args.gpu_id)
    controller = HMOController(
        model, tokenizer, config,
        hmo_config=HMOConfig(segment_length=args.segment_length, keep_ratio=args.keep_ratio),
        gpu_id=args.gpu_id,
    )

    samples = build_samples(tokenizer, args)
    output_path = results_dir / "e2_ablation.jsonl"
    completed = load_completed_cells(output_path) if args.resume else set()

    for i, sample in enumerate(samples):
        remaining = [m for m in METHODS if (m, sample.dataset, args.context_length, sample.sample_id) not in completed]
        if not remaining:
            continue
        logger.info(f"Sample {i+1}/{len(samples)}: {sample.sample_id} — {remaining}")
        torch.cuda.empty_cache()
        cells = run_sample_all_methods(
            controller, sample, tokenizer,
            methods=remaining, context_length=args.context_length,
            experiment="e2_ablation", max_new_tokens=args.max_new_tokens, gpu_id=args.gpu_id,
        )
        for cell in cells:
            save_cell(cell, output_path)

    summary = summarize_cells(output_path)
    summary_path = results_dir / "e2_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"manifest_id": manifest["manifest_id"], "summary": summary, "timestamp": datetime.now().isoformat()}, f, indent=2, ensure_ascii=False)
    logger.info(f"E2 complete. Summary: {summary_path}")
    for key, stats in sorted(summary.items()):
        logger.info(f"  {key}: acc={stats['mean_acc']:.3f}±{stats['std_acc']:.3f}")


if __name__ == "__main__":
    main()
