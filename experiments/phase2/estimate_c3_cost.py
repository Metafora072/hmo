"""Project the full C3 runtime from its first formal 10% result stage."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _mean(values, name: str) -> float:
    numeric = [float(value) for value in values]
    if not numeric or min(numeric) < 0:
        raise ValueError(f"missing or invalid {name} timing values")
    return float(statistics.fmean(numeric))


def estimate(summary: dict, hourly_rate: float | None = None) -> dict:
    """Estimate 432 formal cells without scaling prefill by output length."""
    rows = [
        row
        for row in summary.get("samples", [])
        if row.get("stage") == "formal_32k"
        and float(row.get("budget_fraction", 0.0)) == 0.1
    ]
    if not rows:
        raise ValueError("summary lacks formal_32k 10% result rows")
    full_name = "full_kv_reference"
    if any(full_name not in row.get("systems", {}) for row in rows):
        raise ValueError("formal timing rows lack Full-KV")
    compressed_names = tuple(
        name for name in rows[0]["systems"] if name != full_name
    )
    if len(compressed_names) != 4 or any(
        set(row["systems"]) != set(compressed_names) | {full_name} for row in rows
    ):
        raise ValueError("formal timing rows do not contain four compressed systems")

    prepare = _mean((row["sample_prepare_seconds"] for row in rows), "prepare")
    full_prompt = _mean(
        (
            row["systems"][full_name]["prompt_intervention_seconds"]
            for row in rows
        ),
        "Full prompt",
    )
    full_decode = _mean(
        (row["systems"][full_name]["decode_seconds"] for row in rows),
        "Full decode",
    )
    compressed_prompt = _mean(
        (
            row["systems"][name]["prompt_intervention_seconds"]
            for row in rows
            for name in compressed_names
        ),
        "compressed prompt",
    )
    compressed_decode = _mean(
        (
            row["systems"][name]["decode_seconds"]
            for row in rows
            for name in compressed_names
        ),
        "compressed decode",
    )
    model_load = float(summary.get("runtime", {}).get("model_load_seconds", -1))
    if model_load < 0:
        raise ValueError("summary lacks model-load timing")

    synthetic = 24 * (
        prepare + full_prompt + full_decode + 12 * (compressed_prompt + compressed_decode)
    )
    native_context_ratio = 0.5
    hotpot = 12 * (
        prepare * native_context_ratio
        + full_prompt * native_context_ratio
        + full_decode
        + 4 * (compressed_prompt * native_context_ratio + compressed_decode)
    )
    narrative = 12 * (
        prepare * native_context_ratio
        + full_prompt * native_context_ratio
        + 4 * full_decode
        + 4 * (compressed_prompt * native_context_ratio + 4 * compressed_decode)
    )
    projected = model_load + synthetic + hotpot + narrative
    margin = 1.25
    with_margin = projected * margin
    output = {
        "assumption": "first_formal_32k_10pct_projection_with_separate_prompt_decode_scaling",
        "measured_formal_rows": len(rows),
        "measured_mean_seconds": {
            "model_load": model_load,
            "sample_prepare": prepare,
            "full_prompt_intervention": full_prompt,
            "full_decode_32_tokens": full_decode,
            "compressed_prompt_intervention": compressed_prompt,
            "compressed_decode_32_tokens": compressed_decode,
        },
        "projection_assumptions": {
            "synthetic_samples": 24,
            "compressed_arms_per_budget": 4,
            "synthetic_budgets": 3,
            "native_samples_per_task": 12,
            "native_context_to_32k_ratio": native_context_ratio,
            "hotpot_decode_ratio": 1.0,
            "narrative_decode_ratio": 4.0,
            "prefill_is_not_scaled_by_decode_ratio": True,
        },
        "projected_seconds_before_margin": {
            "synthetic_formal": synthetic,
            "native_hotpotqa": hotpot,
            "native_narrativeqa": narrative,
            "total": projected,
        },
        "margin": margin,
        "projected_gpu_hours": with_margin / 3600,
    }
    if hourly_rate is not None:
        if hourly_rate < 0:
            raise ValueError("hourly rate must be nonnegative")
        output["hourly_rate"] = hourly_rate
        output["projected_cost"] = with_margin / 3600 * hourly_rate
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--hourly-rate", type=float)
    args = parser.parse_args()
    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    print(json.dumps(estimate(payload, args.hourly_rate), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
