"""Freeze completion of layer-local HMO over the original 506-row main table."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from experiments.phase2.e3_v2.freeze_free_window_dev_protocol import (
    DATASET_ORDER,
    _load_rows,
    _sha256_path,
)
from experiments.phase2.e3_v2.oracle import OracleContractError


PROTOCOL_SCHEMA = "hmo.layer_local_completion_protocol.v1"
METHOD_VERSION = "hmo.layer_local.v1"


def build_cases(
    source_rows: Sequence[Mapping], dev_rows: Sequence[Mapping], *, strata: int
) -> list[dict]:
    if strata <= 0:
        raise OracleContractError("completion length strata must be positive")
    source_ids = {str(row["sample_id"]) for row in source_rows}
    dev_ids = {str(row["sample_id"]) for row in dev_rows}
    if (
        len(source_ids) != len(source_rows)
        or len(dev_ids) != len(dev_rows)
        or not dev_ids < source_ids
    ):
        raise OracleContractError("completion parents are not unique strict subsets")
    output = []
    for dataset in DATASET_ORDER:
        members = sorted(
            (row for row in source_rows if row["dataset"] == dataset),
            key=lambda row: (
                int(row["context_tokens"]),
                int(row["record_index"]),
                str(row["sample_id"]),
            ),
        )
        if not members:
            raise OracleContractError(f"completion parent lacks {dataset}")
        for rank, row in enumerate(members):
            output.append(
                {
                    "dataset": dataset,
                    "sample_id": str(row["sample_id"]),
                    "record_index": int(row["record_index"]),
                    "record_sha256": str(row["record_sha256"]),
                    "context_tokens": int(row["context_tokens"]),
                    "query_tokens": int(row["query_tokens"]),
                    "probe_id": str(row["query_probe"]["probe_id"]),
                    "length_stratum": min(rank * strata // len(members), strata - 1),
                    "layer_local_execution": (
                        "reuse_sha_pinned_development" if row["sample_id"] in dev_ids
                        else "generate_once"
                    ),
                }
            )
    output.sort(
        key=lambda item: (
            DATASET_ORDER.index(item["dataset"]),
            int(item["length_stratum"]),
            int(item["record_index"]),
        )
    )
    return output


def freeze(
    *,
    source_results: Path,
    dev_results: Path,
    native_protocol: Path,
    output: Path,
    strata: int,
) -> dict:
    source_rows = _load_rows(source_results)
    dev_rows = _load_rows(dev_results)
    cases = build_cases(source_rows, dev_rows, strata=strata)
    generated = sum(case["layer_local_execution"] == "generate_once" for case in cases)
    reused = len(cases) - generated
    if len(cases) != 506 or reused != 120 or generated != 386:
        raise OracleContractError("completion package count changed")
    payload = {
        "schema_version": PROTOCOL_SCHEMA,
        "status": "frozen_before_remaining_layer_local_outcomes",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "development_robustness_completion_on_original_506_main_table",
        "method_version": METHOD_VERSION,
        "model_id": "Qwen/Qwen3.5-9B",
        "model_revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "parents": {
            "native_results": {
                "path_hint": str(source_results.resolve()),
                "sha256": _sha256_path(source_results),
                "case_count": len(source_rows),
            },
            "free_window_dev_results": {
                "path_hint": str(dev_results.resolve()),
                "sha256": _sha256_path(dev_results),
                "case_count": len(dev_rows),
            },
            "native_protocol": {
                "path_hint": str(native_protocol.resolve()),
                "sha256": _sha256_path(native_protocol),
            },
        },
        "systems": [
            "hmo_legacy",
            "hmo_layer_local",
            "chunkkv",
            "full_kv_reference",
        ],
        "generated_system": "hmo_layer_local",
        "reused_systems": ["hmo_legacy", "chunkkv", "full_kv_reference"],
        "primary_comparisons": [
            ["hmo_layer_local", "hmo_legacy"],
            ["hmo_layer_local", "chunkkv"],
        ],
        "primary_metric": "official_qa_f1",
        "secondary_metrics": [
            "normalized_answer_contains",
            "normalized_exact_match",
        ],
        "middle_kv_fraction": 0.1,
        "method": {
            "segment_length": 256,
            "protected_prefix_segments": 1,
            "protected_suffix_segments": 1,
            "sparse_selector": "per_full_layer_max_mass_contiguous_window",
            "global_actions_and_counts": "legacy_hmo_frozen_per_sample",
            "layer_policy": "independent_per_full_attention_layer_shared_across_kv_heads",
            "recurrent_state_policy": "unchanged",
            "byte_policy": "exact_equal_post_query_resident_bytes",
        },
        "selection": {
            "dataset_order": list(DATASET_ORDER),
            "case_count": len(cases),
            "length_strata": strata,
            "rule": "all_original_main_table_rows_with_equal_rank_strata_per_task",
            "uses_new_method_qa_outcomes": False,
            "development_only": True,
            "independent_confirmation": False,
            "case_filtering_after_outcomes": False,
        },
        "execution": {
            "reused_layer_local_cases": reused,
            "generated_layer_local_cases": generated,
            "baseline_generation_cells": 0,
            "new_generation_cells": generated,
            "continuation_gate": False,
            "resume_after_interruption": True,
        },
        "case_count": len(cases),
        "cases": cases,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent, prefix=f".{output.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-results", required=True, type=Path)
    parser.add_argument("--dev-results", required=True, type=Path)
    parser.add_argument("--native-protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--length-strata", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = freeze(
        source_results=args.source_results,
        dev_results=args.dev_results,
        native_protocol=args.native_protocol,
        output=args.output,
        strata=args.length_strata,
    )
    print(json.dumps({key: value for key, value in payload.items() if key != "cases"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
