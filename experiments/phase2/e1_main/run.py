"""
E1: Main Experiment Table
=========================
HMO vs 6 baselines on 6 benchmarks across 5 context lengths.

Core claim: HMO matches H2O in easy regime (8K-16K) and outperforms in
hard regime (32K+) where H2O starts losing information.

Usage:
    # Timing test first (10 samples, HMO-full only)
    CUDA_VISIBLE_DEVICES=1 python experiments/phase2/e1_main/run.py --timing-test

    # Full run with resume support
    CUDA_VISIBLE_DEVICES=1 python experiments/phase2/e1_main/run.py --resume

    # Specific subset
    CUDA_VISIBLE_DEVICES=1 python experiments/phase2/e1_main/run.py \
        --benchmarks needle,hotpotqa --context-lengths 8192,32768
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
from experiments.utils.run_manifest import read_manifest_id
from experiments.utils.hmo_controller import HMOController, HMOConfig
from experiments.phase2.runner import (
    get_named_results_dir,
    initialize_formal_run,
    load_benchmark_samples,
    run_sample_all_methods,
    save_cell,
    load_completed_cells,
    summarize_cells,
)

METHODS = [
    "full_kv",
    "budgeted_recent_kv",
    "budgeted_uniform_kv",
    "h2o",
    "snapkv",
    "streamingllm",
    "duoattention",
    "pyramidkv_lite",
    "quest_lite",
    "sagekv_lite",
    "hmo_full",
]
BENCHMARKS = ["needle", "longeval_lines", "hotpotqa", "narrativeqa", "gov_report", "lcc"]
CONTEXT_LENGTHS = [8192, 16384, 32768, 65536]

# PLACEHOLDER_PARSE


def parse_args():
    p = argparse.ArgumentParser(description="E1: Main Experiment Table")
    p.add_argument("--model", type=str, default="qwen3.5-0.8b")
    p.add_argument("--n_samples", type=int, default=200)
    p.add_argument("--segment_length", type=int, default=512)
    p.add_argument("--keep_ratio", type=float, default=0.5)
    p.add_argument("--refresh_budget", type=int, default=3)
    p.add_argument("--refresh_min_phi", type=float, default=0.05)
    p.add_argument("--refresh_alpha_mix", type=float, default=0.0)
    p.add_argument("--rts_floor_tokens", type=int, default=1)
    p.add_argument("--rts_phi_mix", type=float, default=0.5)
    p.add_argument("--kv_anchor_budget", type=int, default=0)
    p.add_argument("--kv_anchor_min_phi", type=float, default=0.02)
    p.add_argument("--kv_anchor_diversity", type=float, default=0.15)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--timing-test", action="store_true", help="Run 10-sample timing test only")
    p.add_argument("--resume", action="store_true", help="Skip completed cells")
    p.add_argument("--benchmarks", type=str, default=None, help="Comma-separated benchmark subset")
    p.add_argument("--context-lengths", type=str, default=None, help="Comma-separated context lengths")
    p.add_argument("--methods", type=str, default=None, help="Comma-separated method subset")
    p.add_argument("--run-name", type=str, default=None, help="Optional subdirectory name under experiments/results/e1_main/")
    return p.parse_args()


def resolve_results_dir(args):
    """Return the output directory for this E1 run."""
    return get_named_results_dir("e1_main", args.run_name)


def run_timing_test(controller, tokenizer, args):
    """10 samples x all context lengths x HMO-full only. Estimate total GPUh."""
    logger.info("=" * 60)
    logger.info("E1 TIMING TEST — 10 samples, HMO-full only")
    logger.info("=" * 60)

    results_dir = resolve_results_dir(args)
    ctx_lengths = [int(x) for x in args.context_lengths.split(",")] if args.context_lengths else CONTEXT_LENGTHS
    timing_results = {}

    for ctx_len in ctx_lengths:
        logger.info(f"\n--- Context length: {ctx_len} ---")
        samples = load_benchmark_samples("needle", tokenizer, n_samples=10, context_length=ctx_len, seed=args.seed)

        times = []
        for i, sample in enumerate(samples):
            try:
                torch.cuda.empty_cache()
                t0 = time.perf_counter()
                cells = run_sample_all_methods(
                    controller, sample, tokenizer,
                    methods=["hmo_full"],
                    context_length=ctx_len,
                    experiment="e1_timing",
                    max_new_tokens=args.max_new_tokens,
                    gpu_id=args.gpu_id,
                )
                t1 = time.perf_counter()
                elapsed = t1 - t0
                times.append(elapsed)
                acc = cells[0].accuracy if cells else 0.0
                logger.info(f"  Sample {i+1}/10: {elapsed:.1f}s, acc={acc:.1f}, vram={cells[0].peak_vram_mb:.0f}MB")
            except torch.cuda.OutOfMemoryError:
                logger.error(f"  OOM at ctx={ctx_len}, sample {i+1}")
                torch.cuda.empty_cache()
                times.append(float('inf'))
                break

        if times and times[-1] != float('inf'):
            avg_time = sum(times) / len(times)
            # Estimate: for each sample, we pay 1 shared prefill + N method runs.
            # The timing above measures 1 method (hmo_full) which includes prefill.
            # Approximate: prefill ≈ 40% of total, each additional method ≈ 60%.
            # So N methods ≈ 1 prefill + N * decode_cost ≈ avg_time * (0.4 + N * 0.6)
            n_methods = len(METHODS)
            est_per_sample = avg_time * (0.4 + n_methods * 0.6)
            est_per_ctx = est_per_sample * args.n_samples * len(BENCHMARKS)
            est_hours = est_per_ctx / 3600
            timing_results[ctx_len] = {"avg_sec": avg_time, "est_gpuh": est_hours}
            logger.info(f"  ctx={ctx_len}: avg={avg_time:.1f}s/sample(1 method), est={est_hours:.1f} GPUh for full run")
        else:
            timing_results[ctx_len] = {"avg_sec": float('inf'), "est_gpuh": float('inf'), "note": "OOM"}
            logger.warning(f"  ctx={ctx_len}: OOM — skip this context length in full run")

    total_est = sum(v["est_gpuh"] for v in timing_results.values() if v["est_gpuh"] != float('inf'))
    logger.info("\n" + "=" * 60)
    logger.info(f"TIMING TEST SUMMARY — Total estimated: {total_est:.1f} GPUh")
    for ctx, info in timing_results.items():
        logger.info(f"  {ctx}: {info}")
    if total_est > 50:
        logger.warning(f"Estimated {total_est:.1f} GPUh > 50h budget. Consider reducing n_samples or dropping a context length.")
    else:
        logger.info("Budget OK. Proceed with full run.")
    logger.info("=" * 60)

    # Save timing results
    with open(results_dir / "e1_timing_test.json", "w") as f:
        json.dump({"manifest_id": read_manifest_id(results_dir), "timing": timing_results, "total_est_gpuh": total_est, "timestamp": datetime.now().isoformat()}, f, indent=2)


def main():
    args = parse_args()
    logger.info(f"E1 Main Experiment — model={args.model}, n={args.n_samples}")

    benchmarks = args.benchmarks.split(",") if args.benchmarks else BENCHMARKS
    ctx_lengths = [int(x) for x in args.context_lengths.split(",")] if args.context_lengths else CONTEXT_LENGTHS
    methods = args.methods.split(",") if args.methods else METHODS
    results_dir = resolve_results_dir(args)
    selections = {
        "benchmarks": ["needle"] if args.timing_test else benchmarks,
        "context_lengths": ctx_lengths,
        "methods": ["hmo_full"] if args.timing_test else methods,
        "samples_per_benchmark": 10 if args.timing_test else args.n_samples,
    }
    manifest = initialize_formal_run(results_dir, "e1_timing" if args.timing_test else "e1_main", args, selections)
    logger.info(f"Run manifest: {manifest['manifest_id']}")

    model, tokenizer, config = load_model_and_tokenizer(
        args.model, device="cuda", gpu_id=args.gpu_id,
    )
    controller = HMOController(
        model, tokenizer, config,
        hmo_config=HMOConfig(
            segment_length=args.segment_length,
            keep_ratio=args.keep_ratio,
            refresh_budget=args.refresh_budget,
            refresh_min_phi=args.refresh_min_phi,
            refresh_alpha_mix=args.refresh_alpha_mix,
            rts_floor_tokens=args.rts_floor_tokens,
            rts_phi_mix=args.rts_phi_mix,
            kv_anchor_budget=args.kv_anchor_budget,
            kv_anchor_min_phi=args.kv_anchor_min_phi,
            kv_anchor_diversity=args.kv_anchor_diversity,
        ),
        gpu_id=args.gpu_id,
    )

    if args.timing_test:
        run_timing_test(controller, tokenizer, args)
        return

    # Full run
    output_path = results_dir / "e1_main.jsonl"
    completed = load_completed_cells(output_path) if args.resume else set()
    if completed:
        logger.info(f"Resuming: {len(completed)} cells already completed")

    total_cells = len(ctx_lengths) * len(benchmarks) * len(methods) * args.n_samples
    done_cells = 0
    t_start = time.perf_counter()

    # Outer: context_length ascending (OOM at 128K doesn't lose earlier results)
    for ctx_len in ctx_lengths:
        for bench in benchmarks:
            logger.info(f"\n{'='*40} {bench} @ {ctx_len} {'='*40}")
            try:
                samples = load_benchmark_samples(bench, tokenizer, args.n_samples, ctx_len, args.seed)
            except Exception as e:
                logger.error(f"Failed to load {bench}: {e}")
                continue

            for i, sample in enumerate(samples):
                # Check which methods still need to run for this sample
                remaining_methods = [
                    m for m in methods
                    if (m, sample.dataset, ctx_len, sample.sample_id) not in completed
                ]
                if not remaining_methods:
                    done_cells += len(methods)
                    continue

                logger.info(f"  [{bench}@{ctx_len}] Sample {i+1}/{len(samples)}: {sample.sample_id} — methods: {remaining_methods}")
                torch.cuda.empty_cache()

                cells = run_sample_all_methods(
                    controller, sample, tokenizer,
                    methods=remaining_methods,
                    context_length=ctx_len,
                    experiment="e1_main",
                    max_new_tokens=args.max_new_tokens,
                    gpu_id=args.gpu_id,
                )

                for cell in cells:
                    save_cell(cell, output_path)
                    if not cell.error:
                        completed.add((cell.method, cell.dataset, cell.context_length, cell.sample_id))
                    done_cells += 1

                # Progress
                elapsed = time.perf_counter() - t_start
                rate = done_cells / max(elapsed, 1)
                remaining = (total_cells - done_cells) / max(rate, 0.001)
                logger.info(f"  Progress: {done_cells}/{total_cells} cells, {elapsed/3600:.1f}h elapsed, ~{remaining/3600:.1f}h remaining")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("E1 COMPLETE — Generating summary")
    summary = summarize_cells(output_path)
    summary_path = results_dir / "e1_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"manifest_id": manifest["manifest_id"], "summary": summary, "timestamp": datetime.now().isoformat()}, f, indent=2, ensure_ascii=False)
    logger.info(f"Summary saved to {summary_path}")

    for key, stats in sorted(summary.items()):
        logger.info(
            f"  {key}: {stats.get('primary_metric', 'accuracy')}="
            f"{stats.get('mean_primary', stats['mean_acc']):.3f}±"
            f"{stats.get('std_primary', stats['std_acc']):.3f}, "
            f"acc={stats['mean_acc']:.3f}±{stats['std_acc']:.3f}, "
            f"f1={stats['mean_f1']:.3f}, n={stats['n']}"
        )


if __name__ == "__main__":
    main()
