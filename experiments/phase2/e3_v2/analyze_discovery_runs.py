"""Combine compatible E3-v2 discovery runs for sample-grouped analysis."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from experiments.phase2.e3_v2.direct_fusion import evaluate_direct_fusions

from experiments.phase2.e3_v2.oracle import OracleContractError
from experiments.phase2.e3_v2.run_discovery import analyze_discovery
from experiments.phase2.e3_v2.statistics import SegmentEvidence


COMPATIBILITY_ARGUMENTS = (
    "model_id",
    "datasets",
    "context_length",
    "segment_length",
    "middle_kv_fraction",
    "donors_per_segment",
    "backgrounds_per_pair",
    "recurrent_backend",
    "secondary_generation",
)


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OracleContractError(f"cannot read discovery artifact {path}") from exc


def combine_discovery_runs(
    run_dirs: list[Path],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict:
    if len(run_dirs) < 2:
        raise OracleContractError("combined analysis requires at least two runs")
    evidence = []
    k_by_sample = {}
    sources = []
    reference_signature = None
    reference_model = None

    for run_index, run_dir in enumerate(run_dirs):
        run_dir = run_dir.resolve()
        manifest = _read_json(run_dir / "run_manifest.json")
        summary = _read_json(run_dir / "discovery_summary.json")
        if summary.get("status") != "complete":
            raise OracleContractError(f"discovery run is incomplete: {run_dir}")
        run_spec = manifest["run_spec"]
        arguments = run_spec["arguments"]
        signature = {name: arguments.get(name) for name in COMPATIBILITY_ARGUMENTS}
        model = run_spec["model"]
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
            sample_path = run_dir / "samples" / original_id / "segment_evidence.json"
            raw_rows = _read_json(sample_path)["rows"]
            for raw in raw_rows:
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
            k_by_sample[namespaced_id] = int(sample_summary["middle_budget_slots"])
            sample_count += 1
        sources.append(
            {
                "run_dir": str(run_dir),
                "manifest_id": manifest["manifest_id"],
                "seed": arguments["seed"],
                "sample_count": sample_count,
            }
        )

    return {
        "status": "complete",
        "scope": "combined_discovery_only_not_confirmation",
        "sources": sources,
        "compatible_arguments": reference_signature,
        "sample_count": len(k_by_sample),
        "segment_evidence_rows": len(evidence),
        "analysis": analyze_discovery(
            evidence,
            k_by_sample,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        ),
        "direct_fusions": evaluate_direct_fusions(
            evidence,
            k_by_sample,
            bootstrap_samples=bootstrap_samples,
            seed=seed + 100,
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()
    if len(args.run_dir) < 2 or args.bootstrap_samples <= 0:
        parser.error("provide at least two run dirs and a positive bootstrap count")
    return args


def main() -> int:
    args = parse_args()
    payload = combine_discovery_runs(
        [Path(value) for value in args.run_dir],
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
