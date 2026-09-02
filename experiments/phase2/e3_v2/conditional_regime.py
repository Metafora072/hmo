"""Offline evidence test for the conditional recurrent safe/stressed hypothesis."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from typing import Sequence

import numpy as np

from experiments.phase2.e3_v2.analyze_discovery_runs import (
    COMPATIBILITY_ARGUMENTS,
    _read_json,
)
from experiments.phase2.e3_v2.oracle import OracleContractError
from experiments.phase2.e3_v2.statistics import (
    SegmentEvidence,
    grouped_baseline_residuals,
    sample_grouped_bootstrap_interval,
)


QUADRANTS = {
    (False, False): "Q1_low_sigma_low_delta",
    (False, True): "Q2_low_sigma_high_delta",
    (True, False): "Q3_safe_high_sigma_low_delta",
    (True, True): "Q4_stressed_high_sigma_high_delta",
}
SAFE_QUADRANT = QUADRANTS[(True, False)]
STRESSED_QUADRANT = QUADRANTS[(True, True)]


def _rank01(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise OracleContractError("within-sample ranks require at least two finite values")
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks / (len(values) - 1)


def load_discovery_evidence(
    run_dirs: Sequence[Path],
) -> tuple[tuple[SegmentEvidence, ...], list[dict], dict, dict]:
    """Load and namespace configuration-compatible completed discovery runs."""
    if not run_dirs:
        raise OracleContractError("conditional analysis requires discovery runs")
    evidence = []
    sources = []
    reference_signature = None
    reference_model = None
    for run_index, raw_run_dir in enumerate(run_dirs):
        run_dir = raw_run_dir.resolve()
        manifest = _read_json(run_dir / "run_manifest.json")
        summary = _read_json(run_dir / "discovery_summary.json")
        if summary.get("status") != "complete":
            raise OracleContractError(f"discovery run is incomplete: {run_dir}")
        arguments = manifest["run_spec"]["arguments"]
        signature = {name: arguments.get(name) for name in COMPATIBILITY_ARGUMENTS}
        model = manifest["run_spec"]["model"]
        if reference_signature is None:
            reference_signature = signature
            reference_model = model
        elif signature != reference_signature or model != reference_model:
            raise OracleContractError("discovery runs are not configuration-compatible")

        namespace = f"run{run_index}_{manifest['manifest_id'][:8]}"
        sample_count = 0
        for sample_summary in summary["sample_summaries"]:
            original_id = sample_summary["sample_id"]
            namespaced_id = f"{namespace}:{original_id}"
            payload = _read_json(
                run_dir / "samples" / original_id / "segment_evidence.json"
            )
            for raw in payload["rows"]:
                row = SegmentEvidence(
                    sample_id=str(raw["sample_id"]),
                    dataset=str(raw["dataset"]),
                    segment_id=int(raw["segment_id"]),
                    utility=float(raw["utility"]),
                    alpha=float(raw["alpha"]),
                    normalized_position=float(raw["normalized_position"]),
                    candidates={
                        str(name): float(value)
                        for name, value in raw["candidates"].items()
                    },
                )
                if row.sample_id != original_id:
                    raise OracleContractError("segment evidence sample ID mismatch")
                evidence.append(replace(row, sample_id=namespaced_id))
            sample_count += 1
        sources.append(
            {
                "run_dir": str(run_dir),
                "manifest_id": manifest["manifest_id"],
                "seed": arguments["seed"],
                "sample_count": sample_count,
            }
        )
    return tuple(evidence), sources, reference_signature, reference_model


def _distribution(values: Sequence[float], sample_ids: Sequence[str]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0:
        return {
            "segment_count": 0,
            "sample_count": 0,
            "mean_centered_residual": None,
            "median_centered_residual": None,
            "std_centered_residual": None,
        }
    return {
        "segment_count": len(array),
        "sample_count": len(set(sample_ids)),
        "mean_centered_residual": float(array.mean()),
        "median_centered_residual": float(np.median(array)),
        "std_centered_residual": float(array.std()),
    }


def _sign_counts(values: Sequence[float]) -> dict[str, int]:
    return {
        "positive": sum(value > 0.0 for value in values),
        "zero": sum(value == 0.0 for value in values),
        "negative": sum(value < 0.0 for value in values),
    }


def analyze_conditional_regimes(
    evidence: Sequence[SegmentEvidence],
    *,
    folds: int = 4,
    ridge_lambda: float = 0.001,
    bootstrap_samples: int = 5000,
    seed: int = 20260910,
) -> dict:
    """Test Q4 stressed minus Q3 safe utility without tuning thresholds."""
    evidence = tuple(evidence)
    if bootstrap_samples <= 0:
        raise OracleContractError("bootstrap_samples must be positive")
    for row in evidence:
        values = (
            row.candidates.get("sigma_current"),
            row.candidates.get("delta_update"),
        )
        if any(value is None or not math.isfinite(float(value)) for value in values):
            raise OracleContractError(
                "conditional analysis requires finite sigma_current and delta_update"
            )

    residuals = np.asarray(
        grouped_baseline_residuals(
            evidence,
            folds=folds,
            ridge_lambda=ridge_lambda,
            seed=seed,
        ),
        dtype=np.float64,
    )
    indices_by_sample: dict[str, list[int]] = {}
    for index, row in enumerate(evidence):
        indices_by_sample.setdefault(row.sample_id, []).append(index)

    centered = residuals.copy()
    quadrant_values = {name: [] for name in QUADRANTS.values()}
    quadrant_sample_ids = {name: [] for name in QUADRANTS.values()}
    sample_rows = []
    sample_contrasts = {}
    dataset_by_sample = {}
    for sample_id, indices in sorted(indices_by_sample.items()):
        datasets = {evidence[index].dataset for index in indices}
        if len(datasets) != 1:
            raise OracleContractError("one sample cannot span multiple datasets")
        dataset = next(iter(datasets))
        dataset_by_sample[sample_id] = dataset
        centered[indices] -= centered[indices].mean()
        sigma_rank = _rank01(
            [evidence[index].candidates["sigma_current"] for index in indices]
        )
        delta_rank = _rank01(
            [evidence[index].candidates["delta_update"] for index in indices]
        )
        values_by_quadrant = {name: [] for name in QUADRANTS.values()}
        for local_index, global_index in enumerate(indices):
            quadrant = QUADRANTS[
                (sigma_rank[local_index] >= 0.5, delta_rank[local_index] >= 0.5)
            ]
            value = float(centered[global_index])
            values_by_quadrant[quadrant].append(value)
            quadrant_values[quadrant].append(value)
            quadrant_sample_ids[quadrant].append(sample_id)
        means = {
            name: (float(np.mean(values)) if values else None)
            for name, values in values_by_quadrant.items()
        }
        contrast = None
        if means[SAFE_QUADRANT] is not None and means[STRESSED_QUADRANT] is not None:
            contrast = means[STRESSED_QUADRANT] - means[SAFE_QUADRANT]
            sample_contrasts[sample_id] = contrast
        sample_rows.append(
            {
                "sample_id": sample_id,
                "dataset": dataset,
                "quadrant_counts": {
                    name: len(values) for name, values in values_by_quadrant.items()
                },
                "quadrant_mean_centered_residual": means,
                "q4_stressed_minus_q3_safe": contrast,
            }
        )

    overall = None
    if len(sample_contrasts) >= 2:
        overall = asdict(
            sample_grouped_bootstrap_interval(
                sample_contrasts,
                n_bootstrap=bootstrap_samples,
                seed=seed + 1,
            )
        )
    task_values: dict[str, list[float]] = {}
    for sample_id, value in sample_contrasts.items():
        task_values.setdefault(dataset_by_sample[sample_id], []).append(value)
    task_contrasts = {
        dataset: {
            "mean": float(np.mean(values)),
            "sample_count": len(values),
            "sign_counts": _sign_counts(values),
        }
        for dataset, values in sorted(task_values.items())
    }
    task_direction_consistent = bool(task_contrasts) and all(
        row["mean"] >= 0.0 for row in task_contrasts.values()
    )
    pattern_supported = bool(
        overall is not None
        and overall["mean"] > 0.0
        and task_direction_consistent
    )
    return {
        "schema_version": "hmo.conditional_regime.v1",
        "status": "complete",
        "hypothesis": "Q4 stressed has higher residual exact-KV utility than Q3 safe",
        "residualization": {
            "baseline": "sample-grouped OOF ridge on alpha + normalized_position",
            "folds": min(folds, len(indices_by_sample)),
            "ridge_lambda": ridge_lambda,
            "within_sample_centering": True,
        },
        "regime_definition": {
            "normalization": "within-sample average rank01",
            "threshold": 0.5,
            "threshold_search": False,
            "quadrants": list(QUADRANTS.values()),
        },
        "sample_count": len(indices_by_sample),
        "segment_count": len(evidence),
        "samples_with_q3_and_q4": len(sample_contrasts),
        "quadrant_summary": {
            name: _distribution(quadrant_values[name], quadrant_sample_ids[name])
            for name in QUADRANTS.values()
        },
        "sample_contrasts": sample_rows,
        "q4_stressed_minus_q3_safe": {
            "sample_grouped_bootstrap": overall,
            "sign_counts": _sign_counts(list(sample_contrasts.values())),
            "task_means": task_contrasts,
        },
        "task_direction_consistent": task_direction_consistent,
        "pattern_supported": pattern_supported,
        "decision": (
            "freeze_minimal_three_state_controller"
            if pattern_supported
            else "do_not_tune_thresholds_return_to_openchat"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--ridge-lambda", type=float, default=0.001)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args()
    if args.folds < 2 or args.ridge_lambda < 0 or args.bootstrap_samples <= 0:
        parser.error("invalid CV, ridge, or bootstrap configuration")
    return args


def main() -> int:
    args = parse_args()
    evidence, sources, compatible_arguments, model = load_discovery_evidence(
        [Path(value) for value in args.run_dir]
    )
    payload = analyze_conditional_regimes(
        evidence,
        folds=args.folds,
        ridge_lambda=args.ridge_lambda,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    payload["sources"] = sources
    payload["compatible_arguments"] = compatible_arguments
    payload["model"] = model
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + chr(10),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
