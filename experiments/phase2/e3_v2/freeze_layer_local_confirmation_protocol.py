"""Freeze unseen shorter-context LongBench cases for layer-local HMO confirmation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from transformers import AutoTokenizer

from experiments.phase2.e3_v2.context_query import tokenize_sample_prompt_aligned
from experiments.phase2.e3_v2.oracle import OracleContractError
from experiments.phase2.e3_v2.run_hotpot_solvability import _sha256_file
from experiments.phase2.e3_v2.run_native_tasks import (
    _load_datasets,
    _make_sample,
    load_native_protocol,
)


PROTOCOL_SCHEMA = "hmo.layer_local_confirmation_protocol.v1"
METHOD_VERSION = "hmo.layer_local.v1"
CONFIRMATION_TASKS = (
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
)
AUDITED_TASKS = CONFIRMATION_TASKS + ("narrativeqa", "musique")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def prompt_identity_sha256(prompt) -> str:
    digest = hashlib.sha256()
    for label, tensor in (
        (b"context", prompt.context_ids),
        (b"query", prompt.query_ids),
    ):
        values = tensor.detach().cpu().contiguous().numpy()
        digest.update(label)
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(str(tuple(values.shape)).encode("ascii"))
        digest.update(values.tobytes())
    return digest.hexdigest()


def _stable_key(seed: int, dataset: str, sample_id: str) -> str:
    return _sha256_bytes(f"{seed}:{dataset}:{sample_id}".encode())


def select_stratified_cases(
    candidates: Sequence[Mapping],
    *,
    dataset: str,
    count: int,
    strata: int,
    seed: int,
) -> list[dict]:
    if count <= 0 or strata <= 0 or count % strata:
        raise OracleContractError("confirmation count must divide evenly into strata")
    ordered = sorted(
        (dict(item) for item in candidates),
        key=lambda item: (
            int(item["context_tokens"]),
            int(item["record_index"]),
            str(item["sample_id"]),
        ),
    )
    if len(ordered) < count:
        raise OracleContractError(f"not enough fresh identities for {dataset}")
    bins = [[] for _ in range(strata)]
    for rank, item in enumerate(ordered):
        bins[min(rank * strata // len(ordered), strata - 1)].append(item)
    selected = []
    per_stratum = count // strata
    for stratum, members in enumerate(bins):
        chosen = sorted(
            members,
            key=lambda item: _stable_key(seed, dataset, str(item["sample_id"])),
        )[:per_stratum]
        if len(chosen) != per_stratum:
            raise OracleContractError(f"fresh stratum is too small: {dataset}/{stratum}")
        selected.extend({**item, "length_stratum": stratum} for item in chosen)
    return sorted(
        selected,
        key=lambda item: (int(item["length_stratum"]), int(item["record_index"])),
    )


def build_fresh_inventory(
    archive: Path,
    native_protocol: Mapping,
    tokenizer,
) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    datasets = _load_datasets(archive, native_protocol)
    inventory = {}
    audit = {}
    minimum = int(native_protocol["selection"]["min_memory_context_tokens"])
    maximum = int(native_protocol["selection"]["max_memory_context_tokens"])
    for dataset in AUDITED_TASKS:
        records, raw_lines = datasets[dataset]
        used_indices = {
            int(case["index"])
            for case in native_protocol["datasets"][dataset]["cases"]
        }
        eligible = []
        for record_index, record in enumerate(records):
            sample = _make_sample(dataset, record_index, record)
            prompt, boundary_shift = tokenize_sample_prompt_aligned(sample, tokenizer)
            if not minimum <= prompt.context_tokens <= maximum:
                continue
            eligible.append(
                {
                    "dataset": dataset,
                    "sample_id": sample.sample_id,
                    "record_index": record_index,
                    "record_sha256": _sha256_bytes(raw_lines[record_index]),
                    "context_tokens": prompt.context_tokens,
                    "query_tokens": prompt.query_tokens,
                    "boundary_shift_characters": boundary_shift,
                    "prompt_identity_sha256": prompt_identity_sha256(prompt),
                }
            )
        used_identities = {
            item["prompt_identity_sha256"]
            for item in eligible
            if item["record_index"] in used_indices
        }
        fresh = [
            item
            for item in eligible
            if item["record_index"] not in used_indices
            and item["prompt_identity_sha256"] not in used_identities
        ]
        unique = {}
        for item in sorted(fresh, key=lambda value: int(value["record_index"])):
            unique.setdefault(item["prompt_identity_sha256"], item)
        fresh_unique = list(unique.values())
        inventory[dataset] = fresh_unique
        audit[dataset] = {
            "record_count": len(records),
            "eligible_count": len(eligible),
            "used_record_ids": len(used_indices),
            "fresh_record_ids_after_used_identity_exclusion": len(fresh),
            "fresh_unique_prompt_identities": len(fresh_unique),
            "fresh_context_token_range": (
                [
                    min(item["context_tokens"] for item in fresh_unique),
                    max(item["context_tokens"] for item in fresh_unique),
                ]
                if fresh_unique
                else None
            ),
        }
    return inventory, audit


def build_protocol(
    *,
    archive: Path,
    native_protocol_path: Path,
    model_path: Path,
    per_dataset: int,
    strata: int,
    seed: int,
) -> dict:
    native, native_sha = load_native_protocol(native_protocol_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    inventory, audit = build_fresh_inventory(archive, native, tokenizer)
    cases = []
    for dataset in CONFIRMATION_TASKS:
        cases.extend(
            select_stratified_cases(
                inventory[dataset],
                dataset=dataset,
                count=per_dataset,
                strata=strata,
                seed=seed,
            )
        )
    cases.sort(
        key=lambda item: (
            CONFIRMATION_TASKS.index(item["dataset"]),
            int(item["length_stratum"]),
            int(item["record_index"]),
        )
    )
    tokenizer_files = {}
    for filename in ("tokenizer.json", "tokenizer_config.json"):
        path = model_path / filename
        if not path.is_file():
            raise OracleContractError(f"missing tokenizer identity file: {filename}")
        tokenizer_files[filename] = _sha256_file(path)
    return {
        "schema_version": PROTOCOL_SCHEMA,
        "status": "frozen_before_candidate_qa_outcomes",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "fresh_shorter_context_four_task_confirmation",
        "method_version": METHOD_VERSION,
        "model_id": native["model_id"],
        "model_revision": native["model_revision"],
        "tokenizer_files_sha256": tokenizer_files,
        "dataset_source": native["dataset_source"],
        "native_protocol": {
            "path_hint": str(native_protocol_path.resolve()),
            "sha256": native_sha,
        },
        "systems": [
            "hmo_legacy",
            "hmo_layer_local",
            "chunkkv",
            "full_kv_reference",
        ],
        "equal_byte_systems": [
            "hmo_legacy",
            "hmo_layer_local",
            "chunkkv",
        ],
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
            "allocator": "attention_led_coverage_fidelity",
            "segment_length": 256,
            "protected_prefix_segments": 1,
            "protected_suffix_segments": 1,
            "sparse_width": 16,
            "layer_local_scope": "sparse_window_placement_only",
            "global_scope": "segment_actions_counts_exact_and_slack",
            "layer_policy": "independent_per_full_attention_layer_shared_across_kv_heads",
            "recurrent_state_policy": "unchanged",
            "chunkkv_chunk_size": 10,
            "byte_policy": "exact_equal_post_query_resident_bytes",
        },
        "selection": {
            "dataset_order": list(CONFIRMATION_TASKS),
            "per_dataset": per_dataset,
            "length_strata": strata,
            "per_stratum": per_dataset // strata,
            "seed": seed,
            "inclusive_context_token_band": [
                int(native["selection"]["min_memory_context_tokens"]),
                int(native["selection"]["max_memory_context_tokens"]),
            ],
            "excludes_all_native_506_record_ids": True,
            "excludes_prompt_identities_used_by_native_506": True,
            "deduplicates_fresh_prompt_identity": True,
            "selection_rule": "equal_rank_length_strata_then_seeded_hash",
            "uses_qa_outcomes": False,
            "uses_probe_scores": False,
            "shorter_context_transfer": True,
        },
        "generation": {
            "decoding": native["generation"]["decoding"],
            "inference_seed": native["generation"]["inference_seed"],
            "max_new_tokens": {
                dataset: native["generation"]["max_new_tokens"][dataset]
                for dataset in CONFIRMATION_TASKS
            },
        },
        "inventory_audit": audit,
        "case_count": len(cases),
        "cases": cases,
    }


def _atomic_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--native-protocol", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--per-dataset", type=int, default=20)
    parser.add_argument("--length-strata", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260906)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_protocol(
        archive=args.archive.resolve(),
        native_protocol_path=args.native_protocol.resolve(),
        model_path=args.model_path.resolve(),
        per_dataset=args.per_dataset,
        strata=args.length_strata,
        seed=args.seed,
    )
    _atomic_json(args.output, payload)
    print(json.dumps({key: value for key, value in payload.items() if key != "cases"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
