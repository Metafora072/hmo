"""
E4: Hyperparameter Sensitivity + Detector Ablation
===================================================
Claim: HMO is robust to hyperparameters; each sub-signal contributes.

Model: Qwen3.5-27B-GPTQ-Int4
Data: 80% HotpotQA + 20% Needle, 100 samples, 32K
13 settings: threshold x3, budget x3, n_keep x3, detector x4

Usage:
    CUDA_VISIBLE_DEVICES=1 python experiments/phase2/e4_sensitivity/run.py
"""
import sys
import json
import argparse
import torch
from pathlib import Path
from loguru import logger
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.utils.model_loader import load_model_and_tokenizer
from experiments.utils.hmo_controller import HMOController, HMOConfig
from experiments.utils.dataset_utils import make_needle_samples, load_longbench_subset
from experiments.utils.metrics import reset_vram_stats
from experiments.utils.prototype_runner import build_input_ids
from experiments.utils.eval_harness import score_prediction
from experiments.phase2.runner import get_named_results_dir, initialize_formal_run, save_cell, ExperimentCell

# 13 ablation settings — each must change a parameter that actually affects controller.run()
SETTINGS = {
    # Refresh budget sweep (directly controls how many segments get refreshed)
    "budget_1": {"refresh_budget": 1},
    "budget_3": {"refresh_budget": 3},
    "budget_5": {"refresh_budget": 5},
    # Keep ratio sweep (controls total byte budget = protected + keep_ratio * middle)
    "keepratio_30": {"keep_ratio": 0.3},
    "keepratio_50": {"keep_ratio": 0.5},
    "keepratio_70": {"keep_ratio": 0.7},
    # Segment length sweep (controls granularity of segment-level decisions)
    "seglen_256": {"segment_length": 256},
    "seglen_512": {"segment_length": 512},
    "seglen_1024": {"segment_length": 1024},
    # Detector feature ablation (controls sigma sub-signal weights)
    "det_rho_only": {"alpha_rho": 1.0, "alpha_c": 0.0, "alpha_g": 0.0},
    "det_c_only":   {"alpha_rho": 0.0, "alpha_c": 1.0, "alpha_g": 0.0},
    "det_g_only":   {"alpha_rho": 0.0, "alpha_c": 0.0, "alpha_g": 1.0},
    "det_full":     {"alpha_rho": 0.4, "alpha_c": 0.3, "alpha_g": 0.3},
}

# PLACEHOLDER_E4_MAIN


def parse_args():
    p = argparse.ArgumentParser(description="E4: Hyperparameter Sensitivity")
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
    p.add_argument("--settings", type=str, default=None, help="Comma-separated setting subset")
    return p.parse_args()


def build_samples(tokenizer, args):
    n_lb = int(args.n_samples * 0.8)
    n_needle = args.n_samples - n_lb
    samples = []
    try:
        lb = load_longbench_subset("hotpotqa", tokenizer, n_lb, args.context_length, args.seed)
        samples.extend(lb)
    except Exception as e:
        logger.warning(f"LongBench load failed: {e}")
    samples.extend(make_needle_samples(tokenizer, n_needle, args.context_length, args.seed))
    if len(samples) < args.n_samples:
        extra = make_needle_samples(tokenizer, args.n_samples - len(samples), args.context_length, args.seed + 1000)
        samples.extend(extra)
    return samples


def main():
    args = parse_args()
    logger.info(f"E4 Sensitivity — {args.model}, n={args.n_samples}, ctx={args.context_length}")

    settings_to_run = args.settings.split(",") if args.settings else list(SETTINGS.keys())
    results_dir = get_named_results_dir("e4_sensitivity", args.run_name)

    manifest = initialize_formal_run(
        results_dir, "e4_sensitivity", args,
        {
            "benchmarks": ["longbench_hotpotqa", "needle"],
            "context_lengths": [args.context_length],
            "settings": settings_to_run,
        },
    )
    logger.info(f"Run manifest: {manifest['manifest_id']}")

    model, tokenizer, config = load_model_and_tokenizer(args.model, device="cuda", gpu_id=args.gpu_id)
    samples = build_samples(tokenizer, args)
    output_path = results_dir / "e4_sensitivity.jsonl"

    # Load completed for resume
    completed = set()
    if args.resume and output_path.exists():
        with open(output_path) as f:
            for line in f:
                row = json.loads(line.strip())
                if not row.get("error"):
                    completed.add((row["method"], row["sample_id"]))

    for setting_name in settings_to_run:
        overrides = SETTINGS[setting_name]
        logger.info(f"\n{'='*40} Setting: {setting_name} {'='*40}")
        logger.info(f"  Overrides: {overrides}")

        # Build HMOConfig with overrides
        cfg_kwargs = dict(
            segment_length=args.segment_length,
            keep_ratio=args.keep_ratio,
        )
        cfg_kwargs.update(overrides)
        hmo_config = HMOConfig(**cfg_kwargs)

        controller = HMOController(
            model, tokenizer, config,
            hmo_config=hmo_config,
            gpu_id=args.gpu_id,
        )

        for i, sample in enumerate(samples):
            if (setting_name, sample.sample_id) in completed:
                continue
            logger.info(f"  [{setting_name}] Sample {i+1}/{len(samples)}: {sample.sample_id}")
            torch.cuda.empty_cache()
            reset_vram_stats(args.gpu_id)

            try:
                input_ids = build_input_ids(
                    sample, tokenizer, controller.model.device,
                    max_length=args.context_length + 256,
                )
                hmo_result = controller.run(input_ids, max_new_tokens=args.max_new_tokens)
                gen = hmo_result.generated_text or ""
                scores = score_prediction(gen, sample)
                cell = ExperimentCell(
                    experiment="e4_sensitivity",
                    method=setting_name,
                    dataset=sample.dataset,
                    context_length=args.context_length,
                    sample_id=sample.sample_id,
                    accuracy=scores.accuracy,
                    f1=scores.f1,
                    rouge_l=scores.rouge_l,
                    code_sim=scores.code_sim,
                    primary_metric=scores.primary_metric,
                    primary_score=scores.primary_score,
                    peak_vram_mb=hmo_result.peak_vram_mb,
                    tracked_bytes=hmo_result.total_tracked_bytes,
                    budget_limit_bytes=hmo_result.budget_limit_bytes,
                    n_kept_kv=hmo_result.n_kept_kv,
                    n_skeleton=hmo_result.n_skeleton,
                    n_refresh=hmo_result.n_refresh,
                    n_dropped=hmo_result.n_dropped,
                    generated_text=gen[:300],
                    answer=sample.answer[:300],
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                cell = ExperimentCell(
                    experiment="e4_sensitivity", method=setting_name,
                    dataset=sample.dataset, context_length=args.context_length,
                    sample_id=sample.sample_id, error="OOM",
                )
            except Exception as e:
                cell = ExperimentCell(
                    experiment="e4_sensitivity", method=setting_name,
                    dataset=sample.dataset, context_length=args.context_length,
                    sample_id=sample.sample_id, error=str(e)[:200],
                )
            save_cell(cell, output_path)

    # Summary
    from collections import defaultdict
    groups = defaultdict(list)
    with open(output_path) as f:
        for line in f:
            row = json.loads(line.strip())
            if not row.get("error"):
                groups[row["method"]].append(row.get("primary_score", row["accuracy"]))

    logger.info("\n" + "=" * 60)
    logger.info("E4 SUMMARY")
    import numpy as np
    summary = {}
    for name in sorted(groups):
        scores = np.array(groups[name])
        summary[name] = {"mean_score": float(scores.mean()), "std_score": float(scores.std()), "n": len(scores)}
        logger.info(f"  {name}: score={scores.mean():.3f}±{scores.std():.3f} (n={len(scores)})")

    summary_path = results_dir / "e4_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"manifest_id": manifest["manifest_id"], "summary": summary, "timestamp": datetime.now().isoformat()}, f, indent=2)
    logger.info(f"E4 complete. Summary: {summary_path}")


if __name__ == "__main__":
    main()
