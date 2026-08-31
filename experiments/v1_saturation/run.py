"""
V1: DeltaNet Saturation Detection Validation
============================================
Kill question: Can the prototype's saturation score predict when refresh replay helps?

This script is being migrated toward the canonical V1 protocol:
1. Collect sigma with the controller's live hook path
2. Build mixed Needle + HotpotQA samples so detector validity is not judged on
   uniform-density filler alone
3. Use the canonical refresh baseline: keep sink/recent KV, drop all other
   middle segments, then add exactly one refresh segment under the same budget cap
4. Correlate sigma with the measured single-refresh gain
5. If sigma-only fails, automatically rerun the analysis with
   sigma * attention_score as the weak predictor

Note:
The fidelity guard currently blocks this script because the oracle still needs
to be rewritten around the exact pre-decode refresh semantics.

Usage:
    CUDA_VISIBLE_DEVICES=1 python experiments/v1_saturation/run.py \
        --model qwen3.5-0.8b --n_samples 50 --context_length 16384
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
from experiments.utils.metrics import (
    compute_exact_match,
    compute_correlation,
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
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def parse_args():
    p = argparse.ArgumentParser(description="V1: Saturation Detection Validation")
    p.add_argument("--model", type=str, default="qwen3.5-0.8b")
    p.add_argument("--n_samples", type=int, default=50)
    p.add_argument("--context_length", type=int, default=16384)
    p.add_argument("--segment_length", type=int, default=512)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument(
        "--oracle_sample_rate",
        type=int,
        default=15,
        help="Test roughly every Nth segment for refresh gain",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def sample_seed(base_seed: int, sample_id: str) -> int:
    """Stable per-sample seed that does not depend on Python hash randomization."""
    return base_seed + sum(ord(ch) for ch in sample_id)


def choose_test_segments(sigma: np.ndarray, args, sample_id: str) -> list[int]:
    """Pick a stratified subset of segments for oracle refresh evaluation."""
    candidates = middle_segments(len(sigma))
    if not candidates:
        candidates = list(range(len(sigma)))
    if not candidates:
        return []

    k = max(3, len(candidates) // max(args.oracle_sample_rate, 1))
    ranked = sorted(candidates, key=lambda idx: float(sigma[idx]))

    test_indices = set(ranked[:k])
    test_indices.update(ranked[-k:])

    rng = np.random.RandomState(sample_seed(args.seed, sample_id))
    random_idx = rng.choice(candidates, size=min(k, len(candidates)), replace=False)
    test_indices.update(int(idx) for idx in random_idx.tolist())
    return sorted(test_indices)


def build_samples(tokenizer, args):
    """V1 data: HotpotQA-dominant (80%) + Needle (20%).
    V1 tests H1 (detector discriminability) and H2 (sigma predicts refresh value).
    HotpotQA has natural information-density variation across paragraphs,
    which is favorable for H1. Needle is included as a secondary check but
    is known to degenerate sigma on uniform filler (H1 limitation)."""
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
            f"Canonical V1 requires HotpotQA data (H1 known limitation: "
            f"detector degenerates on uniform-density input). "
            f"HotpotQA loading failed: {exc}"
        ) from exc

    n_hotpotqa = len([s for s in samples if s.dataset.startswith("longbench")])
    if n_hotpotqa == 0:
        raise RuntimeError(
            "Canonical V1 requires at least some HotpotQA samples for "
            "information-density-varying validation. Got 0 after filtering."
        )

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


def summarize_correlation(scores: np.ndarray, labels: np.ndarray) -> tuple[dict, bool]:
    """Compute the canonical V1 verdict from predictor scores and oracle labels."""
    if len(scores) < 20:
        return {"warning": f"only {len(scores)} tested segments, need >=20"}, False
    corr = compute_correlation(scores, labels)
    passed = corr.get("pearson", 0.0) > 0.3 or corr.get("auc", 0.0) > 0.65
    return corr, passed


@torch.no_grad()
def run_sample(controller, tokenizer, sample, args):
    """
    For one sample:
    1. Collect prototype sigma
    2. Run canonical no-refresh baseline (protected KV, other middle segments dropped)
    3. Force single-segment refresh replay under the same shared budget cap
    4. Measure refresh gain for correlation
    """
    input_ids = build_input_ids(
        sample,
        tokenizer,
        controller.model.device,
        max_length=args.context_length + 256,
    )

    sigma, segment_costs = collect_sigma_and_segment_costs(controller, input_ids)
    attention_scores = collect_segment_attention_scores(controller, input_ids)
    n_segs = len(sigma)

    baseline_actions = protected_kv_drop_rest_actions(n_segs)
    candidate_actions = {
        seg_idx: refresh_drop_rest_actions(n_segs, [seg_idx])
        for seg_idx in choose_test_segments(sigma, args, sample.sample_id)
    }
    shared_budget_limit_bytes = max(
        [estimate_action_bytes(segment_costs, baseline_actions)]
        + [estimate_action_bytes(segment_costs, actions) for actions in candidate_actions.values()],
        default=0,
    )

    baseline_result = run_forced_actions(
        controller,
        input_ids,
        baseline_actions,
        max_new_tokens=args.max_new_tokens,
        budget_limit_bytes=shared_budget_limit_bytes,
    )
    baseline_acc = compute_exact_match(baseline_result.generated_text, sample.answer)

    oracle_labels = {}
    for seg_idx, forced_actions in candidate_actions.items():
        refresh_result = run_forced_actions(
            controller,
            input_ids,
            forced_actions,
            max_new_tokens=args.max_new_tokens,
            budget_limit_bytes=shared_budget_limit_bytes,
        )
        refresh_acc = compute_exact_match(refresh_result.generated_text, sample.answer)
        oracle_labels[seg_idx] = max(0.0, refresh_acc - baseline_acc)

    return {
        "dataset": sample.dataset,
        "sigma": sigma,
        "attention_scores": attention_scores,
        "baseline_acc": baseline_acc,
        "baseline_gen": baseline_result.generated_text[:200],
        "oracle_labels": oracle_labels,
        "n_segs": n_segs,
        "seq_len": int(input_ids.shape[1]),
        "shared_budget_limit_bytes": int(shared_budget_limit_bytes),
        "baseline_tracked_bytes": int(baseline_result.total_tracked_bytes),
    }


def main():
    args = parse_args()
    require_theory_faithful("v1")
    logger.info(f"V1 Saturation Detection — {args.model}, n={args.n_samples}, ctx={args.context_length}")

    model, tokenizer, config = load_model_and_tokenizer(
        args.model,
        device="cuda",
        gpu_id=args.gpu_id,
    )
    controller = HMOController(
        model,
        tokenizer,
        config,
        hmo_config=HMOConfig(segment_length=args.segment_length),
        gpu_id=args.gpu_id,
    )

    samples = build_samples(tokenizer, args)

    all_sigma_tested = []
    all_attention_tested = []
    all_oracle_tested = []
    per_dataset_sigma = {}
    per_dataset_attention = {}
    per_dataset_oracle = {}
    per_sample_results = []

    for i, sample in enumerate(samples):
        logger.info(f"Sample {i + 1}/{len(samples)}: {sample.sample_id}")
        reset_vram_stats(args.gpu_id)

        try:
            result = run_sample(controller, tokenizer, sample, args)

            for seg_idx, oracle_val in result["oracle_labels"].items():
                if seg_idx < len(result["sigma"]):
                    all_sigma_tested.append(float(result["sigma"][seg_idx]))
                    attention_val = (
                        float(result["attention_scores"][seg_idx])
                        if seg_idx < len(result["attention_scores"])
                        else 0.0
                    )
                    all_attention_tested.append(attention_val)
                    all_oracle_tested.append(float(oracle_val))
                    per_dataset_sigma.setdefault(result["dataset"], []).append(float(result["sigma"][seg_idx]))
                    per_dataset_attention.setdefault(result["dataset"], []).append(attention_val)
                    per_dataset_oracle.setdefault(result["dataset"], []).append(float(oracle_val))

            per_sample_results.append({
                "sample_id": sample.sample_id,
                "dataset": result["dataset"],
                "baseline_acc": result["baseline_acc"],
                "n_segs": result["n_segs"],
                "n_tested": len(result["oracle_labels"]),
                "sigma_mean": float(result["sigma"].mean()) if len(result["sigma"]) > 0 else 0.0,
                "sigma_max": float(result["sigma"].max()) if len(result["sigma"]) > 0 else 0.0,
                "attention_mean": float(result["attention_scores"].mean()) if len(result["attention_scores"]) > 0 else 0.0,
                "attention_max": float(result["attention_scores"].max()) if len(result["attention_scores"]) > 0 else 0.0,
                "peak_vram_mb": get_peak_vram_mb(args.gpu_id),
            })

            logger.info(
                f"  acc={result['baseline_acc']:.1f}, "
                f"tested={len(result['oracle_labels'])}/{result['n_segs']} segs, "
                f"vram={get_peak_vram_mb(args.gpu_id):.0f}MB"
            )

        except torch.cuda.OutOfMemoryError:
            logger.error(f"OOM on sample {i + 1}, skipping")
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Error on sample {i + 1}: {e}")

    sigma_arr = np.array(all_sigma_tested, dtype=float)
    attention_arr = np.array(all_attention_tested, dtype=float)
    oracle_arr = np.array(all_oracle_tested, dtype=float)

    strong_corr, strong_pass = summarize_correlation(sigma_arr, oracle_arr)
    strong_dataset_summary = {}
    for dataset_name in sorted(per_dataset_sigma):
        ds_sigma = np.asarray(per_dataset_sigma[dataset_name], dtype=float)
        ds_oracle = np.asarray(per_dataset_oracle[dataset_name], dtype=float)
        ds_corr, ds_pass = summarize_correlation(ds_sigma, ds_oracle)
        ds_corr["n_tested_segments"] = int(len(ds_sigma))
        ds_corr["passed"] = bool(ds_pass)
        strong_dataset_summary[dataset_name] = ds_corr

    weak_ran = not strong_pass
    weak_scores = sigma_arr * attention_arr if weak_ran else np.array([], dtype=float)
    weak_corr = None
    weak_pass = None
    weak_dataset_summary = {}
    if weak_ran:
        weak_corr, weak_pass = summarize_correlation(weak_scores, oracle_arr)
        for dataset_name in sorted(per_dataset_sigma):
            ds_sigma = np.asarray(per_dataset_sigma[dataset_name], dtype=float)
            ds_attention = np.asarray(per_dataset_attention[dataset_name], dtype=float)
            ds_oracle = np.asarray(per_dataset_oracle[dataset_name], dtype=float)
            ds_corr, ds_pass = summarize_correlation(ds_sigma * ds_attention, ds_oracle)
            ds_corr["n_tested_segments"] = int(len(ds_sigma))
            ds_corr["passed"] = bool(ds_pass)
            weak_dataset_summary[dataset_name] = ds_corr

    final_mode = "h2_strong" if strong_pass else ("h2_weak" if weak_pass else "fail")

    logger.info("=" * 60)
    logger.info(f"V1 RESULTS — {args.model} ({len(sigma_arr)} tested segments)")
    logger.info("  H2-strong (sigma-only):")
    logger.info(f"    Pearson:  {strong_corr.get('pearson', 0.0):.4f}")
    logger.info(f"    Spearman: {strong_corr.get('spearman', 0.0):.4f}")
    logger.info(f"    AUC:      {strong_corr.get('auc', 0.0):.4f}")
    logger.info(f"    PASS:     {strong_pass}")
    for dataset_name, stats in strong_dataset_summary.items():
        if "pearson" in stats:
            logger.info(
                f"    {dataset_name}: Pearson={stats['pearson']:.4f}, "
                f"Spearman={stats['spearman']:.4f}, AUC={stats['auc']:.4f}, "
                f"pass={stats['passed']}, n={stats['n_tested_segments']}"
            )
        else:
            logger.info(f"    {dataset_name}: {stats['warning']}")

    if weak_ran:
        logger.info("  H2-weak (sigma x attention_score):")
        logger.info(f"    Pearson:  {weak_corr.get('pearson', 0.0):.4f}")
        logger.info(f"    Spearman: {weak_corr.get('spearman', 0.0):.4f}")
        logger.info(f"    AUC:      {weak_corr.get('auc', 0.0):.4f}")
        logger.info(f"    PASS:     {weak_pass}")
        for dataset_name, stats in weak_dataset_summary.items():
            if "pearson" in stats:
                logger.info(
                    f"    {dataset_name}: Pearson={stats['pearson']:.4f}, "
                    f"Spearman={stats['spearman']:.4f}, AUC={stats['auc']:.4f}, "
                    f"pass={stats['passed']}, n={stats['n_tested_segments']}"
                )
            else:
                logger.info(f"    {dataset_name}: {stats['warning']}")
    else:
        logger.info("  H2-weak was skipped because H2-strong already passed.")

    logger.info("  PASS condition: Pearson>0.3 OR AUC>0.65")
    logger.info(f"  FINAL VERDICT: {final_mode}")
    logger.info("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "experiment": "V1_saturation_detection",
        "model": args.model,
        "n_samples": args.n_samples,
        "context_length": args.context_length,
        "segment_length": args.segment_length,
        "timestamp": datetime.now().isoformat(),
        "n_tested_segments": len(sigma_arr),
        "decision_tree": {
            "strong_first": True,
            "weak_ran": weak_ran,
            "final_verdict": final_mode,
        },
        "h2_strong": {
            "predictor": "sigma",
            "correlation": strong_corr,
            "per_dataset_correlation": strong_dataset_summary,
            "passed": bool(strong_pass),
        },
        "h2_weak": {
            "predictor": "sigma_times_attention_score",
            "correlation": weak_corr if weak_ran else None,
            "per_dataset_correlation": weak_dataset_summary if weak_ran else {},
            "passed": bool(weak_pass) if weak_ran else None,
            "ran": weak_ran,
        },
        "passed": bool(strong_pass or bool(weak_pass)),
        "prototype_alignment": (
            "single-refresh gain is measured under the canonical protected-KV/drop-rest "
            "baseline; Full KV is treated only as an upper bound and is not budget-matched"
        ),
        "per_sample": per_sample_results,
    }
    out_path = RESULTS_DIR / f"v1_{args.model.replace('.', '_')}_{args.context_length}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
