"""Freeze a 9B six-task LongBench protocol from a tokenizer-only inventory."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


SYSTEMS = (
    "contiguous_cf",
    "chunkkv",
    "global_fixed_chunk_topk",
    "raw_alpha_exact_slack",
    "full_kv_reference",
)
MAX_NEW_TOKENS = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
}


def freeze(inventory: dict) -> dict:
    selection = inventory["selection"]
    if (
        inventory.get("schema_version") != "hmo.longbench_six_task_inventory.v1"
        or selection.get("outcome_conditioned") is not False
        or selection.get("augmentation") is not False
        or selection.get("truncation") is not False
        or selection.get("maximum_tokens") != 16384
        or selection.get("requested_per_task") != 100
    ):
        raise ValueError("inventory is not the frozen no-outcome <=16K selection")
    datasets = {}
    for name in MAX_NEW_TOKENS:
        source = inventory["datasets"][name]
        cases = list(source["selected"])
        if not cases or len(cases) > 100:
            raise ValueError(f"invalid selected prefix for {name}")
        datasets[name] = {
            "member": source["member"],
            "record_count": source["record_count"],
            "official_metric": "qa_f1_score",
            "eligible_count": source["eligible_count"],
            "cases": cases,
        }
    prefix50_cases = sum(min(50, len(spec["cases"])) for spec in datasets.values())
    prefix100_cases = sum(len(spec["cases"]) for spec in datasets.values())
    return {
        "schema_version": "hmo.native_longbench_six_task_protocol.v1",
        "status": "frozen_before_outcomes",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "broad_native_non_augmented_real_task_main_table",
        "model_id": inventory["model_id"],
        "model_revision": inventory["model_revision"],
        "dataset_source": {
            "repository": "zai-org/LongBench",
            "revision": "5e628be450b7e67fb7ae6e201bd6d8f7056f7672",
            "archive_type": "zip",
            "archive_sha256": inventory["archive_sha256"],
            "metric_revision": "2e00731f8d0bff23dc4325161044d0ed8af94c1e",
        },
        "dataset_order": list(MAX_NEW_TOKENS),
        "systems": list(SYSTEMS),
        "equal_byte_systems": list(SYSTEMS[:-1]),
        "primary_comparisons": [
            ["contiguous_cf", name] for name in SYSTEMS if name != "contiguous_cf"
        ],
        "primary_metric": "official_qa_f1",
        "secondary_metrics": [
            "normalized_answer_contains",
            "normalized_exact_match",
        ],
        "middle_kv_fraction": 0.1,
        "method": {
            "allocator": "attention_led",
            "sparse_selector": "max_mass_window",
            "sparse_width": 16,
            "chunkkv_chunk_size": 10,
            "chunkkv_observation": "query_suffix_attention",
            "chunkkv_layer_policy": "independent_per_full_layer_shared_across_kv_heads",
            "chunkkv_partial_chunk": "fixed_prefix_of_next_ranked_chunk",
            "raw_slack_selector": "global_top_tokens_slack",
            "global_fixed_chunk_width": 16,
            "global_fixed_chunk_slack": "prefix_of_next_ranked_chunk",
            "segment_length": 256,
            "protected_prefix_segments": 1,
            "protected_suffix_segments": 1,
        },
        "selection": {
            "rule": selection["rule"],
            "min_memory_context_tokens": selection["minimum_tokens"],
            "max_memory_context_tokens": selection["maximum_tokens"],
            "maximum_samples_per_dataset": 100,
            "source_context_unchanged": True,
            "augmentation": False,
            "truncation": False,
            "outcome_conditioned_selection": False,
            "boundary_alignment": "if_one_token_crosses_the_semantic_context_boundary_move_that_complete_token_into_memory_context",
            "record_sha256_semantics": "sha256_of_raw_jsonl_record_bytes_without_line_ending",
        },
        "stage_sets": {
            "prefix50": {
                "per_dataset_prefix": 50,
                "manifest_group": "prefix100",
                "manifest_per_dataset_prefix": 100,
            },
            "prefix100": {
                "per_dataset_prefix": 100,
                "manifest_group": "prefix100",
                "manifest_per_dataset_prefix": 100,
            },
        },
        "execution": {
            "prefix50_case_count": prefix50_cases,
            "prefix100_case_count": prefix100_cases,
            "prefix50_generation_cells": prefix50_cases * len(SYSTEMS),
            "prefix100_generation_cells": prefix100_cases * len(SYSTEMS),
            "continuation_is_precommitted_prefix": True,
            "continuation_gate": False,
            "resume_after_interruption": True,
            "case_filtering_after_outcomes": False,
        },
        "generation": {
            "decoding": "greedy",
            "inference_seed": 20261018,
            "max_new_tokens": MAX_NEW_TOKENS,
        },
        "datasets": datasets,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inventory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    payload = freeze(json.loads(args.inventory.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
