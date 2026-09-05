"""Freeze a task- and length-stratified development set for HMO free windows."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

from experiments.phase2.e3_v2.oracle import OracleContractError


PROTOCOL_SCHEMA = "hmo.free_window_dev_protocol.v1"
DATASET_ORDER = (
    "longbench_narrativeqa",
    "longbench_qasper",
    "longbench_multifieldqa_en",
    "longbench_hotpotqa",
    "longbench_2wikimqa",
    "longbench_musique",
)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    sample_ids = [str(row.get("sample_id")) for row in rows]
    if not rows or len(sample_ids) != len(set(sample_ids)):
        raise OracleContractError("source results are empty or contain duplicate samples")
    return rows


def _stable_key(seed: int, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode()).hexdigest()


def select_cases(
    rows: Sequence[Mapping], *, per_dataset: int, strata: int, seed: int
) -> list[dict]:
    if per_dataset <= 0 or strata <= 0 or per_dataset % strata:
        raise OracleContractError("per-dataset count must be positive and divisible by strata")
    grouped: dict[str, list[Mapping]] = defaultdict(list)
    for row in rows:
        grouped[str(row["dataset"])].append(row)
    if tuple(name for name in DATASET_ORDER if name in grouped) != DATASET_ORDER:
        raise OracleContractError("source results do not contain the six expected tasks")

    selected = []
    per_stratum = per_dataset // strata
    for dataset in DATASET_ORDER:
        unique = {}
        for row in sorted(
            grouped[dataset], key=lambda item: (int(item["record_index"]), item["sample_id"])
        ):
            unique.setdefault(str(row["query_probe"]["probe_id"]), row)
        ordered = sorted(
            unique.values(),
            key=lambda item: (
                int(item["context_tokens"]),
                int(item["record_index"]),
                str(item["sample_id"]),
            ),
        )
        if len(ordered) < per_dataset:
            raise OracleContractError(f"not enough unique probes for {dataset}")
        bins = [[] for _ in range(strata)]
        for rank, row in enumerate(ordered):
            bins[min(rank * strata // len(ordered), strata - 1)].append(row)
        for stratum, candidates in enumerate(bins):
            chosen = sorted(
                candidates,
                key=lambda item: _stable_key(seed, str(item["sample_id"])),
            )[:per_stratum]
            if len(chosen) != per_stratum:
                raise OracleContractError(f"length stratum {stratum} is too small")
            for row in chosen:
                selected.append(
                    {
                        "dataset": dataset,
                        "sample_id": str(row["sample_id"]),
                        "record_index": int(row["record_index"]),
                        "record_sha256": str(row["record_sha256"]),
                        "context_tokens": int(row["context_tokens"]),
                        "query_tokens": int(row["query_tokens"]),
                        "probe_id": str(row["query_probe"]["probe_id"]),
                        "length_stratum": stratum,
                    }
                )
    selected.sort(
        key=lambda item: (
            DATASET_ORDER.index(item["dataset"]),
            item["length_stratum"],
            item["record_index"],
        )
    )
    return selected


def freeze(
    results_path: Path,
    native_protocol_path: Path,
    output_path: Path,
    *,
    per_dataset: int,
    strata: int,
    seed: int,
) -> dict:
    rows = _load_rows(results_path)
    cases = select_cases(rows, per_dataset=per_dataset, strata=strata, seed=seed)
    payload = {
        "schema_version": PROTOCOL_SCHEMA,
        "status": "frozen_before_new_method_qa_outcomes",
        "purpose": "development_only_task_and_context_length_stratified_validation",
        "source_results": {
            "path_hint": str(results_path.resolve()),
            "sha256": _sha256_path(results_path),
            "case_count": len(rows),
        },
        "native_protocol": {
            "path_hint": str(native_protocol_path.resolve()),
            "sha256": _sha256_path(native_protocol_path),
        },
        "model_id": "Qwen/Qwen3.5-9B",
        "model_revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "systems": [
            "hmo_legacy",
            "hmo_layer_local",
            "chunkkv",
            "hmo_free_window",
            "full_kv_reference",
        ],
        "generated_systems": ["hmo_layer_local", "hmo_free_window"],
        "reused_systems": ["hmo_legacy", "chunkkv", "full_kv_reference"],
        "primary_comparisons": [
            ["hmo_free_window", "chunkkv"],
            ["hmo_free_window", "hmo_legacy"],
            ["hmo_layer_local", "hmo_legacy"],
        ],
        "primary_metric": "official_qa_f1",
        "middle_kv_fraction": 0.1,
        "method": {
            "segment_length": 256,
            "protected_prefix_segments": 1,
            "protected_suffix_segments": 1,
            "free_window_width": 10,
            "free_window_selector": "exact_cardinality_constrained_weighted_interval_dp",
            "free_window_boundary_policy": "arbitrary_eligible_token_start",
            "layer_policy": "independent_per_full_attention_layer_shared_across_kv_heads",
            "recurrent_state_policy": "unchanged",
            "layer_local_control": "legacy_hmo_actions_and_counts_with_per_layer_sparse_placement",
            "byte_target": "exactly_the_reconstructed_chunkkv_context_resident_bytes",
        },
        "selection": {
            "rule": "deduplicate_exact_probe_identity_then_equal_rank_length_strata_then_seeded_hash",
            "dataset_order": list(DATASET_ORDER),
            "per_dataset": per_dataset,
            "length_strata": strata,
            "per_stratum": per_dataset // strata,
            "seed": seed,
            "uses_qa_outcomes": False,
            "uses_proxy_outcomes": False,
            "development_only": True,
            "final_confirmation_requires_unseen_source_record_ids": True,
        },
        "case_count": len(cases),
        "cases": cases,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent, prefix=f".{output_path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--native-protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--per-dataset", type=int, default=20)
    parser.add_argument("--length-strata", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260905)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = freeze(
        args.results,
        args.native_protocol,
        args.output,
        per_dataset=args.per_dataset,
        strata=args.length_strata,
        seed=args.seed,
    )
    print(json.dumps({key: value for key, value in payload.items() if key != "cases"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
