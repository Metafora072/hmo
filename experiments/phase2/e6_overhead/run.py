"""
E6: Overhead Analysis
=====================
Claim: HMO control logic overhead < 15%.

Model: Qwen3.5-27B BF16
Data: 50 samples from E1, each repeated 3 times
Context: 32K, 64K
Metrics: TTFT, decode tok/s, peak KV memory, per-component cost breakdown

Usage:
    CUDA_VISIBLE_DEVICES=1 python experiments/phase2/e6_overhead/run.py
"""
import sys
import json
import argparse
import time
import numpy as np
import torch
from pathlib import Path
from loguru import logger
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.utils.model_loader import load_model_and_tokenizer
from experiments.utils.hmo_controller import HMOController, HMOConfig
from experiments.utils.dataset_utils import make_needle_samples, load_longbench_subset
from experiments.utils.metrics import compute_exact_match, reset_vram_stats, get_peak_vram_mb
from experiments.utils.prototype_runner import (
    build_input_ids,
    collect_sigma_and_segment_costs,
    default_shared_budget_limit,
)
from experiments.phase2.runner import get_named_results_dir, initialize_formal_run


def parse_args():
    p = argparse.ArgumentParser(description="E6: Overhead Analysis")
    p.add_argument("--model", type=str, default="qwen3.5-27b")
    p.add_argument("--n_samples", type=int, default=50)
    p.add_argument("--n_repeats", type=int, default=3)
    p.add_argument("--segment_length", type=int, default=512)
    p.add_argument("--keep_ratio", type=float, default=0.5)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--run-name", type=str, default=None)
    return p.parse_args()


def build_samples(tokenizer, n_samples, context_length, seed):
    """Mixed samples: 70% HotpotQA + 30% Needle."""
    n_lb = int(n_samples * 0.7)
    n_needle = n_samples - n_lb
    samples = []
    try:
        lb = load_longbench_subset("hotpotqa", tokenizer, n_lb, context_length, seed)
        samples.extend(lb)
    except Exception as e:
        logger.warning(f"LongBench load failed: {e}")
    samples.extend(make_needle_samples(tokenizer, n_needle, context_length, seed))
    if len(samples) < n_samples:
        extra = make_needle_samples(tokenizer, n_samples - len(samples), context_length, seed + 1000)
        samples.extend(extra)
    return samples


@torch.no_grad()
def profile_sample(controller, tokenizer, sample, context_length, max_new_tokens, gpu_id):
    """Profile one sample with detailed per-component timing."""
    input_ids = build_input_ids(
        sample, tokenizer, controller.model.device,
        max_length=context_length + 256,
    )
    seq_len = input_ids.shape[1]

    # 1. Baseline (Full KV) timing
    reset_vram_stats(gpu_id)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    baseline_result = controller.run_baseline(input_ids, max_new_tokens=max_new_tokens)
    torch.cuda.synchronize()
    t_baseline = time.perf_counter() - t0
    baseline_vram = get_peak_vram_mb(gpu_id)
    torch.cuda.empty_cache()

    # 2. HMO with per-component timing
    reset_vram_stats(gpu_id)

    # 2a. Prefill + saturation probe
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    sigma, segment_costs = collect_sigma_and_segment_costs(controller, input_ids)
    torch.cuda.synchronize()
    t_probe = time.perf_counter() - t0
    torch.cuda.empty_cache()

    # 2b. Full HMO run (includes probe + actions + decode)
    reset_vram_stats(gpu_id)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    hmo_result = controller.run(input_ids, max_new_tokens=max_new_tokens)
    torch.cuda.synchronize()
    t_hmo_total = time.perf_counter() - t0
    hmo_vram = get_peak_vram_mb(gpu_id)
    torch.cuda.empty_cache()

    # 3. H2O timing
    budget = default_shared_budget_limit(segment_costs, controller.hmo.keep_ratio, n_sigma=len(sigma))
    reset_vram_stats(gpu_id)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    h2o_result = controller.run_h2o_baseline(
        input_ids, max_new_tokens=max_new_tokens,
        budget_limit_bytes=budget,
    )
    torch.cuda.synchronize()
    t_h2o = time.perf_counter() - t0
    h2o_vram = get_peak_vram_mb(gpu_id)
    torch.cuda.empty_cache()

    # Compute overhead
    overhead_vs_baseline = (t_hmo_total - t_baseline) / max(t_baseline, 1e-6)
    overhead_vs_h2o = (t_hmo_total - t_h2o) / max(t_h2o, 1e-6)

    # Compute actual generated token counts
    baseline_n_tok = baseline_result.generated_ids.shape[-1] if baseline_result.generated_ids is not None else max_new_tokens
    hmo_n_tok = hmo_result.generated_ids.shape[-1] if hmo_result.generated_ids is not None else max_new_tokens
    h2o_n_tok = h2o_result.generated_ids.shape[-1] if h2o_result.generated_ids is not None else max_new_tokens

    return {
        "sample_id": sample.sample_id,
        "dataset": sample.dataset,
        "context_length": context_length,
        "seq_len": seq_len,
        "n_segments": len(sigma),
        # Timing (seconds)
        "baseline_time_s": t_baseline,
        "h2o_time_s": t_h2o,
        "hmo_total_time_s": t_hmo_total,
        "probe_time_s": t_probe,
        "overhead_vs_baseline_pct": overhead_vs_baseline * 100,
        "overhead_vs_h2o_pct": overhead_vs_h2o * 100,
        # VRAM (MB)
        "baseline_vram_mb": baseline_vram,
        "h2o_vram_mb": h2o_vram,
        "hmo_vram_mb": hmo_vram,
        # Throughput (actual generated tokens)
        "baseline_tok_per_s": baseline_n_tok / max(t_baseline, 1e-6),
        "h2o_tok_per_s": h2o_n_tok / max(t_h2o, 1e-6),
        "hmo_tok_per_s": hmo_n_tok / max(t_hmo_total, 1e-6),
        # HMO actions
        "n_kept_kv": hmo_result.n_kept_kv,
        "n_skeleton": hmo_result.n_skeleton,
        "n_refresh": hmo_result.n_refresh,
        "n_dropped": hmo_result.n_dropped,
        "hmo_tracked_bytes": hmo_result.total_tracked_bytes,
        "h2o_tracked_bytes": h2o_result.total_tracked_bytes,
    }


def main():
    args = parse_args()
    logger.info(f"E6 Overhead — {args.model}, n={args.n_samples}, repeats={args.n_repeats}")

    context_lengths = [32768, 65536]
    results_dir = get_named_results_dir("e6_overhead", args.run_name)
    manifest = initialize_formal_run(
        results_dir, "e6_overhead", args,
        {
            "benchmarks": ["longbench_hotpotqa", "needle"],
            "context_lengths": context_lengths,
            "methods": ["full_kv", "h2o", "hmo_full"],
            "repeats": args.n_repeats,
        },
    )
    logger.info(f"Run manifest: {manifest['manifest_id']}")

    model, tokenizer, config = load_model_and_tokenizer(args.model, device="cuda", gpu_id=args.gpu_id)
    controller = HMOController(
        model, tokenizer, config,
        hmo_config=HMOConfig(segment_length=args.segment_length, keep_ratio=args.keep_ratio),
        gpu_id=args.gpu_id,
    )

    output_path = results_dir / "e6_overhead.jsonl"

    completed = set()
    if args.resume and output_path.exists():
        with open(output_path) as f:
            for line in f:
                row = json.loads(line.strip())
                completed.add((row["sample_id"], row["context_length"], row.get("repeat", 0)))

    for ctx_len in context_lengths:
        logger.info(f"\n{'='*40} Context: {ctx_len} {'='*40}")
        samples = build_samples(tokenizer, args.n_samples, ctx_len, args.seed)

        oom_at_ctx = False
        for i, sample in enumerate(samples):
            if oom_at_ctx:
                break
            for rep in range(args.n_repeats):
                if (sample.sample_id, ctx_len, rep) in completed:
                    continue

                logger.info(f"  [{ctx_len}] Sample {i+1}/{len(samples)} rep {rep+1}/{args.n_repeats}: {sample.sample_id}")
                try:
                    result = profile_sample(controller, tokenizer, sample, ctx_len, args.max_new_tokens, args.gpu_id)
                    result["repeat"] = rep
                    result["manifest_id"] = manifest["manifest_id"]
                    with open(output_path, "a") as f:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                except torch.cuda.OutOfMemoryError:
                    logger.error(f"OOM at ctx={ctx_len}, skipping remaining samples at this length")
                    torch.cuda.empty_cache()
                    oom_at_ctx = True
                    break
                except Exception as e:
                    logger.error(f"Error: {e}")

    # Summary
    if output_path.exists():
        from collections import defaultdict
        groups = defaultdict(list)
        with open(output_path) as f:
            for line in f:
                row = json.loads(line.strip())
                ctx = row["context_length"]
                groups[ctx].append(row)

        logger.info("\n" + "=" * 60)
        logger.info("E6 OVERHEAD SUMMARY")
        summary = {}
        for ctx in sorted(groups):
            rows = groups[ctx]
            overhead_baseline = np.mean([r["overhead_vs_baseline_pct"] for r in rows])
            overhead_h2o = np.mean([r["overhead_vs_h2o_pct"] for r in rows])
            probe_frac = np.mean([r["probe_time_s"] / max(r["hmo_total_time_s"], 1e-6) for r in rows])
            summary[ctx] = {
                "n": len(rows),
                "overhead_vs_baseline_pct": float(overhead_baseline),
                "overhead_vs_h2o_pct": float(overhead_h2o),
                "probe_fraction_pct": float(probe_frac * 100),
                "mean_baseline_tok_s": float(np.mean([r["baseline_tok_per_s"] for r in rows])),
                "mean_hmo_tok_s": float(np.mean([r["hmo_tok_per_s"] for r in rows])),
                "mean_hmo_vram_mb": float(np.mean([r["hmo_vram_mb"] for r in rows])),
            }
            logger.info(
                f"  ctx={ctx}: overhead_vs_baseline={overhead_baseline:.1f}%, "
                f"overhead_vs_h2o={overhead_h2o:.1f}%, "
                f"probe={probe_frac*100:.1f}% of total, "
                f"hmo_tok/s={summary[ctx]['mean_hmo_tok_s']:.1f}"
            )

        summary_path = results_dir / "e6_summary.json"
        with open(summary_path, "w") as f:
            json.dump({"manifest_id": manifest["manifest_id"], "summary": summary, "timestamp": datetime.now().isoformat()}, f, indent=2)
        logger.info(f"E6 complete. Summary: {summary_path}")


if __name__ == "__main__":
    main()
