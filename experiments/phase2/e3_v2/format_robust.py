"""Post-hoc format-robust analysis for frozen HMO retrieval runs."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

from experiments.vendor.longbench_metrics import normalize_answer


SCHEMA_VERSION = "hmo.format_robust_secondary.v1"
_CLOCK_TRUTH = re.compile(r"^\s*(\d{3,4})\s*o['’]?clock\s*$", re.IGNORECASE)
_CLOCK_COLON = re.compile(
    r"(?<!\d)(\d{1,2}):(\d{2})(?:\s*(?:o['’]?clock|a\.?m\.?|p\.?m\.?))?(?!\d)",
    re.IGNORECASE,
)
_CLOCK_SUFFIX = re.compile(
    r"(?<![A-Za-z0-9])(\d{3,4})\s*(?:o['’]?clock|a\.?m\.?|p\.?m\.?)\b",
    re.IGNORECASE,
)


def _clock_truth_key(answer: str) -> str | None:
    match = _CLOCK_TRUTH.fullmatch(answer)
    return match.group(1).lstrip("0") or "0" if match else None


def _clock_prediction_keys(text: str) -> set[str]:
    keys = {
        (match.group(1) + match.group(2)).lstrip("0") or "0"
        for match in _CLOCK_COLON.finditer(text)
    }
    keys.update(
        match.group(1).lstrip("0") or "0"
        for match in _CLOCK_SUFFIX.finditer(text)
    )
    return keys


def format_robust_contains(text: str, answer: str, dataset: str) -> float:
    """Preserve primary containment, then resolve one known Needle format alias."""
    prediction = normalize_answer(text)
    truth = normalize_answer(answer)
    if truth and truth in prediction:
        return 1.0
    if dataset != "needle":
        return 0.0
    clock_key = _clock_truth_key(answer)
    if clock_key is None:
        return 0.0
    return float(clock_key in _clock_prediction_keys(text))


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
    if not rows:
        raise ValueError(f"no result rows in {path}")
    return rows


def _pair_summary(rows: Iterable[Mapping], left: str, right: str) -> dict:
    pairs = [
        (
            row["secondary_scores"][left],
            row["secondary_scores"][right],
        )
        for row in rows
    ]
    count = len(pairs)
    return {
        "left": left,
        "right": right,
        "case_count": count,
        "left_accuracy": sum(left_value for left_value, _ in pairs) / count,
        "right_accuracy": sum(right_value for _, right_value in pairs) / count,
        "mean_delta": sum(left_value - right_value for left_value, right_value in pairs)
        / count,
        "wins": sum(left_value > right_value for left_value, right_value in pairs),
        "ties": sum(left_value == right_value for left_value, right_value in pairs),
        "losses": sum(left_value < right_value for left_value, right_value in pairs),
    }


def analyze_result_file(path: Path, label: str) -> dict:
    raw_rows = _read_jsonl(path)
    scored_rows = []
    systems = tuple(raw_rows[0]["systems"])
    for row in raw_rows:
        if tuple(row["systems"]) != systems:
            raise ValueError(f"system set changed within {path}")
        secondary_scores = {
            system: format_robust_contains(
                values["generated_text"], row["answer"], row["dataset"]
            )
            for system, values in row["systems"].items()
        }
        scored_rows.append({**row, "secondary_scores": secondary_scores})

    system_summary = {}
    for system in systems:
        primary = [
            float(row["systems"][system]["normalized_answer_contains"])
            for row in scored_rows
        ]
        secondary = [row["secondary_scores"][system] for row in scored_rows]
        system_summary[system] = {
            "primary_correct": int(sum(primary)),
            "secondary_correct": int(sum(secondary)),
            "case_count": len(scored_rows),
            "primary_accuracy": sum(primary) / len(primary),
            "secondary_accuracy": sum(secondary) / len(secondary),
            "upgrades": int(sum(new > old for old, new in zip(primary, secondary))),
            "downgrades": int(sum(new < old for old, new in zip(primary, secondary))),
        }

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in scored_rows:
        grouped[f"{row['stage']}/{row['dataset']}"].append(row)

    comparisons = {}
    for right in (
        "scattered_cf",
        "raw_alpha_exact_slack",
        "raw_alpha_exact_topk",
        "contiguous_sparse_only",
        "full_kv_reference",
    ):
        if "contiguous_cf" in systems and right in systems:
            comparisons[f"contiguous_cf_vs_{right}"] = _pair_summary(
                scored_rows, "contiguous_cf", right
            )

    return {
        "label": label,
        "source": str(path.resolve()),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "case_count": len(scored_rows),
        "systems": system_summary,
        "by_stage_dataset": {
            group: {
                system: {
                    "secondary_correct": int(
                        sum(row["secondary_scores"][system] for row in rows)
                    ),
                    "case_count": len(rows),
                }
                for system in systems
            }
            for group, rows in sorted(grouped.items())
        },
        "secondary_comparisons": comparisons,
        "changed_cases": [
            {
                "sample_id": row["sample_id"],
                "dataset": row["dataset"],
                "answer": row["answer"],
                "upgraded_systems": [
                    system
                    for system in systems
                    if row["secondary_scores"][system]
                    > float(row["systems"][system]["normalized_answer_contains"])
                ],
                "generated_text": {
                    system: row["systems"][system]["generated_text"]
                    for system in systems
                    if row["secondary_scores"][system]
                    > float(row["systems"][system]["normalized_answer_contains"])
                },
            }
            for row in scored_rows
            if any(
                row["secondary_scores"][system]
                != float(row["systems"][system]["normalized_answer_contains"])
                for system in systems
            )
        ],
    }


def parse_input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("input must use LABEL=PATH")
    label, raw_path = value.split("=", 1)
    if not label or not raw_path:
        raise argparse.ArgumentTypeError("input must use non-empty LABEL=PATH")
    return label, Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=parse_input)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "metric_role": "post_hoc_secondary_only",
        "primary_metric_unchanged": True,
        "rule": (
            "Use frozen normalized containment first; only for Needle truths in "
            "N o'clock form, treat colonized H:MM predictions with optional "
            "o'clock/AM/PM suffixes as the same clock-format alias. Other "
            "datasets receive no extra aliasing."
        ),
        "runs": [analyze_result_file(path, label) for label, path in args.input],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
