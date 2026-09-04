"""Project the C3 mandatory-core runtime from its two-cell 27B preflight."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def estimate(summary: dict, hourly_rate: float | None = None) -> dict:
    samples = summary.get("samples", [])
    if len(samples) != 1:
        raise ValueError("C3 preflight summary must contain exactly one sample")
    row = samples[0]
    if (
        row.get("stage") != "preflight_32k"
        or float(row.get("budget_fraction", 0.0)) != 0.1
        or set(row.get("systems", {}))
        != {"contiguous_cf", "full_kv_reference"}
    ):
        raise ValueError("summary is not the frozen C3 two-cell preflight")
    runtime = summary.get("runtime", {})
    model_load = float(runtime["model_load_seconds"])
    prepare = float(row["sample_prepare_seconds"])
    hmo = float(row["systems"]["contiguous_cf"]["system_elapsed_seconds"])
    full = float(row["systems"]["full_kv_reference"]["system_elapsed_seconds"])
    if min(model_load, prepare, hmo, full) < 0:
        raise ValueError("preflight timing values must be nonnegative")

    # Each 32K synthetic sample generates Full once and four compressed arms at
    # three budgets. HMO time is the conservative proxy for the other arms.
    synthetic = 24 * (prepare + full + 12 * hmo)
    # Native contexts are no longer than 16.3K. Hotpot permits 32 output tokens;
    # Narrative permits 128, so its generation allowance is scaled by four.
    native_hotpot = 12 * (prepare + full + 4 * hmo)
    native_narrative = 12 * (prepare + 4 * full + 16 * hmo)
    projected = model_load + synthetic + native_hotpot + native_narrative
    with_margin = projected * 1.25
    output = {
        "assumption": "27B_preflight_timing_projection_with_25pct_margin",
        "measured_seconds": {
            "model_load": model_load,
            "sample_prepare": prepare,
            "hmo_generation": hmo,
            "full_generation": full,
        },
        "projected_seconds_before_margin": {
            "synthetic_core": synthetic,
            "native_hotpotqa": native_hotpot,
            "native_narrativeqa": native_narrative,
            "total": projected,
        },
        "margin": 1.25,
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
