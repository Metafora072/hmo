"""
Phase 2 Shared Runner — Common infrastructure for all formal experiments.

Provides:
  - ExperimentCell: one result row
  - run_method(): dispatch to controller methods by name
  - load_benchmark_samples(): unified loader for all 6 benchmarks
  - JSONL incremental save + resume support
"""
from __future__ import annotations

import json
import os
import time
import numpy as np
import torch
from dataclasses import dataclass, field, asdict
from pathlib import Path
from loguru import logger

from experiments.utils.dataset_utils import (
    EvalSample,
    make_needle_samples,
    make_longeval_lines_samples,
    load_longbench_subset,
)
from experiments.utils.hmo_controller import HMOController, HMOResult
from experiments.utils.metrics import (
    get_peak_vram_mb, reset_vram_stats,
)
from experiments.utils.prototype_runner import (
    build_input_ids,
    collect_sigma_and_segment_costs,
    default_shared_budget_limit,
    rts_budgeted_actions,
    refresh_drop_rest_budgeted_actions,
    hmo_periodic_actions,
    select_triggered_refresh_segments,
    run_forced_actions,
)
from experiments.utils.eval_harness import score_prediction, resolve_max_new_tokens

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = Path(
    os.environ.get("HMO_RESULTS_ROOT", str(PROJECT_ROOT / "experiments/results"))
)
RESULTS_DIR = RESULTS_ROOT  # backward-compat alias for older scripts


def get_results_dir(experiment_name: str) -> Path:
    """Return the dedicated result directory for one formal Phase-2 experiment."""
    path = RESULTS_ROOT / experiment_name
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class ExperimentCell:
    """One result row in the experiment grid."""
    experiment: str = ""
    method: str = ""
    dataset: str = ""
    context_length: int = 0
    sample_id: str = ""
    accuracy: float = 0.0
    f1: float = 0.0
    rouge_l: float = 0.0
    primary_metric: str = "accuracy"
    primary_score: float = 0.0
    ttft_ms: float = 0.0
    decode_ms: float = 0.0
    tok_per_sec: float = 0.0
    peak_vram_mb: float = 0.0
    tracked_bytes: int = 0
    budget_limit_bytes: int = 0
    n_kept_kv: int = 0
    n_skeleton: int = 0
    n_refresh: int = 0
    n_dropped: int = 0
    generated_text: str = ""
    answer: str = ""
    error: str = ""


def get_primary_metric_name(dataset: str) -> str:
    """
    Return the benchmark's official primary metric when we support it.

    For LongBench subsets, the README specifies:
      - HotpotQA / NarrativeQA: F1
      - GovReport: Rouge-L
    Everything else currently falls back to accuracy / exact match.
    """
    if dataset in ("longbench_hotpotqa", "longbench_narrativeqa"):
        return "f1"
    if dataset == "longbench_gov_report":
        return "rouge_l"
    return "accuracy"


def get_primary_score(dataset: str, accuracy: float, f1: float, rouge_l: float) -> float:
    metric = get_primary_metric_name(dataset)
    if metric == "f1":
        return f1
    if metric == "rouge_l":
        return rouge_l
    return accuracy


def load_benchmark_samples(
    benchmark: str,
    tokenizer,
    n_samples: int,
    context_length: int,
    seed: int = 42,
) -> list[EvalSample]:
    """Unified loader for all 6 benchmarks."""
    if benchmark == "needle":
        return make_needle_samples(tokenizer, n_samples, context_length, seed)
    elif benchmark == "longeval_lines":
        return make_longeval_lines_samples(tokenizer, n_samples, context_length, seed=seed)
    elif benchmark in ("hotpotqa", "narrativeqa", "gov_report", "lcc"):
        return load_longbench_subset(benchmark, tokenizer, n_samples, context_length, seed)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")


def _pad_actions_for_tail(actions: dict[int, str], n_sigma: int, n_costs: int) -> dict[int, str]:
    """Ensure trailing segments beyond sigma coverage are explicitly assigned KV."""
    for seg_idx in range(n_sigma, n_costs):
        if seg_idx not in actions:
            actions[seg_idx] = "KV"
    return actions


def run_method(
    controller: HMOController,
    method: str,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    budget_limit_bytes: int,
    sigma: np.ndarray | None = None,
    segment_costs: dict | None = None,
) -> HMOResult:
    """Dispatch to the correct controller method by name."""
    if method == "full_kv":
        return controller.run_baseline(input_ids, max_new_tokens=max_new_tokens)
    elif method == "budgeted_recent_kv":
        return controller.run_budgeted_recent_kv_baseline(
            input_ids, max_new_tokens=max_new_tokens,
            budget_limit_bytes=budget_limit_bytes,
        )
    elif method == "budgeted_uniform_kv":
        return controller.run_budgeted_uniform_kv_baseline(
            input_ids, max_new_tokens=max_new_tokens,
            budget_limit_bytes=budget_limit_bytes,
        )
    elif method == "h2o":
        return controller.run_h2o_baseline(
            input_ids, max_new_tokens=max_new_tokens,
            budget_limit_bytes=budget_limit_bytes,
        )
    elif method == "snapkv":
        return controller.run_snapkv_baseline(
            input_ids, max_new_tokens=max_new_tokens,
            budget_limit_bytes=budget_limit_bytes,
        )
    elif method == "streamingllm":
        return controller.run_streamingllm_baseline(
            input_ids, max_new_tokens=max_new_tokens,
            budget_limit_bytes=budget_limit_bytes,
        )
    elif method == "duoattention":
        return controller.run_duoattention_baseline(
            input_ids, max_new_tokens=max_new_tokens,
            budget_limit_bytes=budget_limit_bytes,
        )
    elif method == "pyramidkv_lite":
        return controller.run_pyramidkv_lite_baseline(
            input_ids, max_new_tokens=max_new_tokens,
            budget_limit_bytes=budget_limit_bytes,
        )
    elif method == "quest_lite":
        return controller.run_quest_lite_baseline(
            input_ids, max_new_tokens=max_new_tokens,
            budget_limit_bytes=budget_limit_bytes,
        )
    elif method == "sagekv_lite":
        return controller.run_sagekv_lite_baseline(
            input_ids, max_new_tokens=max_new_tokens,
            budget_limit_bytes=budget_limit_bytes,
        )
    elif method == "hmo_full":
        return controller.run(input_ids, max_new_tokens=max_new_tokens)
    elif method == "rts_only":
        if sigma is None or segment_costs is None:
            raise ValueError("rts_only requires sigma and segment_costs")
        actions = rts_budgeted_actions(
            len(sigma), sigma, segment_costs,
            budget_limit_bytes=budget_limit_bytes,
            rts_threshold=controller.hmo.rts_threshold,
        )
        actions = _pad_actions_for_tail(actions, len(sigma), len(segment_costs))
        return run_forced_actions(
            controller, input_ids, actions,
            max_new_tokens=max_new_tokens,
            budget_limit_bytes=budget_limit_bytes,
            segment_costs=segment_costs,
        )
    elif method == "refresh_only":
        if sigma is None or segment_costs is None:
            raise ValueError("refresh_only requires sigma and segment_costs")
        triggered = select_triggered_refresh_segments(
            sigma,
            budget=min(controller.hmo.refresh_budget, len(sigma)),
            threshold=controller.hmo.saturation_threshold,
        )
        actions = refresh_drop_rest_budgeted_actions(
            len(sigma), triggered, segment_costs, budget_limit_bytes,
        )
        actions = _pad_actions_for_tail(actions, len(sigma), len(segment_costs))
        return run_forced_actions(
            controller, input_ids, actions,
            max_new_tokens=max_new_tokens,
            budget_limit_bytes=budget_limit_bytes,
            segment_costs=segment_costs,
        )
    elif method == "hmo_periodic":
        if segment_costs is None:
            raise ValueError("hmo_periodic requires segment_costs")
        n_sig = len(sigma) if sigma is not None else len(segment_costs)
        actions = hmo_periodic_actions(
            n_sig, segment_costs, budget_limit_bytes,
            refresh_budget=controller.hmo.refresh_budget,
            sigma=sigma,
        )
        actions = _pad_actions_for_tail(actions, n_sig, len(segment_costs))
        return run_forced_actions(
            controller, input_ids, actions,
            max_new_tokens=max_new_tokens,
            budget_limit_bytes=budget_limit_bytes,
            segment_costs=segment_costs,
        )
    else:
        raise ValueError(f"Unknown method: {method}")


# PLACEHOLDER_IO


def result_to_cell(
    hmo_result: HMOResult,
    sample: EvalSample,
    experiment: str,
    method: str,
    context_length: int,
) -> ExperimentCell:
    """Convert an HMOResult + sample into an ExperimentCell."""
    gen = hmo_result.generated_text or ""
    ans = sample.answer or ""
    is_summary = sample.dataset in ("longbench_gov_report",)
    accuracy, f1, rouge_l_full = score_prediction(gen, sample)
    rouge_l = rouge_l_full if is_summary else 0.0
    primary_metric = get_primary_metric_name(sample.dataset)
    primary_score = get_primary_score(sample.dataset, accuracy, f1, rouge_l)
    return ExperimentCell(
        experiment=experiment,
        method=method,
        dataset=sample.dataset,
        context_length=context_length,
        sample_id=sample.sample_id,
        accuracy=accuracy,
        f1=f1,
        rouge_l=rouge_l,
        primary_metric=primary_metric,
        primary_score=primary_score,
        peak_vram_mb=hmo_result.peak_vram_mb,
        tracked_bytes=hmo_result.total_tracked_bytes,
        budget_limit_bytes=hmo_result.budget_limit_bytes,
        n_kept_kv=hmo_result.n_kept_kv,
        n_skeleton=hmo_result.n_skeleton,
        n_refresh=hmo_result.n_refresh,
        n_dropped=hmo_result.n_dropped,
        generated_text=gen[:300],
        answer=ans[:300],
    )


def save_cell(cell: ExperimentCell, output_path: Path):
    """Append one cell to JSONL (crash-safe incremental save)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a") as f:
        f.write(json.dumps(asdict(cell), ensure_ascii=False) + "\n")


def load_completed_cells(output_path: Path) -> set[tuple[str, str, int, str]]:
    """Load (method, dataset, context_length, sample_id) tuples for resume."""
    completed = set()
    if not output_path.exists():
        return completed
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if not row.get("error"):
                    completed.add((
                        row["method"], row["dataset"],
                        row["context_length"], row["sample_id"],
                    ))
            except json.JSONDecodeError:
                continue
    return completed


def summarize_cells(output_path: Path) -> dict:
    """Read JSONL and produce aggregated summary grouped by (method, dataset, context_length)."""
    from collections import defaultdict
    groups = defaultdict(list)
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("error"):
                continue
            key = (row["method"], row["dataset"], row["context_length"])
            groups[key].append(row)

    summary = {}
    for key, rows in groups.items():
        method, dataset, ctx = key
        accs = [r["accuracy"] for r in rows]
        f1s = [r["f1"] for r in rows]
        vrams = [r["peak_vram_mb"] for r in rows]
        primary_metric = rows[0].get("primary_metric", get_primary_metric_name(dataset))
        primary_scores = [r.get("primary_score", get_primary_score(dataset, r["accuracy"], r["f1"], r.get("rouge_l", 0.0))) for r in rows]
        summary[f"{method}_{dataset}_{ctx}"] = {
            "method": method,
            "dataset": dataset,
            "context_length": ctx,
            "n": len(rows),
            "primary_metric": primary_metric,
            "mean_primary": float(np.mean(primary_scores)),
            "std_primary": float(np.std(primary_scores)),
            "mean_acc": float(np.mean(accs)),
            "std_acc": float(np.std(accs)),
            "mean_f1": float(np.mean(f1s)),
            "mean_vram": float(np.mean(vrams)),
            "mean_tracked_bytes": float(np.mean([r["tracked_bytes"] for r in rows])),
        }
    return summary


@torch.no_grad()
def run_sample_all_methods(
    controller: HMOController,
    sample: EvalSample,
    tokenizer,
    methods: list[str],
    context_length: int,
    experiment: str,
    max_new_tokens: int = 64,
    gpu_id: int = 0,
) -> list[ExperimentCell]:
    """Run all methods on one sample, sharing sigma/costs computation."""
    max_new_tokens = resolve_max_new_tokens(sample.dataset, max_new_tokens)
    input_ids = build_input_ids(
        sample, tokenizer, controller.model.device,
        max_length=context_length + 256,
    )

    # Compute sigma and segment costs once (shared across methods)
    needs_sigma = any(m in ("rts_only", "refresh_only", "hmo_periodic") for m in methods)
    sigma, segment_costs = None, None
    if needs_sigma:
        sigma, segment_costs = collect_sigma_and_segment_costs(controller, input_ids)
    elif any(m not in ("full_kv",) for m in methods):
        # Need segment_costs for budget computation even without sigma
        sigma, segment_costs = collect_sigma_and_segment_costs(controller, input_ids)

    # Free DeltaNet intermediate tensors from the sigma prefill pass
    import gc; gc.collect(); torch.cuda.empty_cache()

    budget_limit_bytes = 0
    if segment_costs is not None:
        n_sig = len(sigma) if sigma is not None else len(segment_costs)
        budget_limit_bytes = default_shared_budget_limit(segment_costs, controller.hmo.keep_ratio, n_sigma=n_sig)

    cells = []
    for method in methods:
        reset_vram_stats(gpu_id)
        gc.collect(); torch.cuda.empty_cache()
        try:
            hmo_result = run_method(
                controller, method, input_ids, max_new_tokens,
                budget_limit_bytes, sigma, segment_costs,
            )
            cell = result_to_cell(hmo_result, sample, experiment, method, context_length)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            logger.error(f"OOM: {method} on {sample.sample_id} @ {context_length}")
            cell = ExperimentCell(
                experiment=experiment, method=method, dataset=sample.dataset,
                context_length=context_length, sample_id=sample.sample_id,
                error="OOM",
            )
        except Exception as e:
            logger.error(f"Error: {method} on {sample.sample_id}: {e}")
            cell = ExperimentCell(
                experiment=experiment, method=method, dataset=sample.dataset,
                context_length=context_length, sample_id=sample.sample_id,
                error=str(e)[:200],
            )
        cells.append(cell)
    return cells
