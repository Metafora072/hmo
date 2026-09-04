"""Validation and runner views for the frozen C3 27B experiment package."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Mapping


C3_SCHEMA = "hmo.c3.large_model_protocol.v1"
C3_STATUS = "frozen_before_27b_outcomes"
C3_MODEL_ID = "Qwen/Qwen3.5-27B"
C3_MODEL_REVISION = "fc05daec18b0a78c049392ed2e771dde82bdf654"
C3_SYSTEMS = (
    "contiguous_cf",
    "global_fixed_chunk_topk",
    "raw_alpha_exact_slack",
    "scattered_cf",
    "full_kv_reference",
)
C3_EQUAL_BYTE_SYSTEMS = C3_SYSTEMS[:-1]
C3_BUDGETS = (0.05, 0.1, 0.2)
C3_NATIVE_PARENT_SHA256 = (
    "86ebfa5cfdff0613e559780811887b7537d0485cbd00534193c0aac433b49e2a"
)


class C3ProtocolError(ValueError):
    """Raised when the C3 package no longer matches its frozen contract."""


def _read_json(path: Path) -> tuple[dict, bytes]:
    try:
        encoded = path.read_bytes()
        payload = json.loads(encoded)
    except (OSError, json.JSONDecodeError) as exc:
        raise C3ProtocolError(f"cannot read C3 protocol: {path}") from exc
    if not isinstance(payload, dict):
        raise C3ProtocolError("C3 protocol root must be an object")
    return payload, encoded


def _synthetic_cells(stage: Mapping, default_systems: tuple[str, ...]) -> int:
    systems = tuple(stage.get("systems", default_systems))
    compressed = tuple(name for name in systems if name != "full_kv_reference")
    samples = len(str(stage["datasets"]).split(",")) * int(
        stage["samples_per_dataset"]
    )
    budgets = tuple(float(value) for value in stage["budget_fractions"])
    return samples * (len(compressed) * len(budgets) + 1)


def load_c3_protocol(path: Path, project_root: Path) -> tuple[dict, str, dict]:
    """Load C3, validate all cell counts, and return its pinned native parent."""
    path = path.resolve()
    project_root = project_root.resolve()
    payload, encoded = _read_json(path)
    model = payload.get("model", {})
    hardware = payload.get("hardware", {})
    synthetic = payload.get("synthetic", {})
    stages = synthetic.get("stages", {})
    native = payload.get("native", {})
    method = payload.get("method", {})
    expected_method = {
        "allocator": "attention_led",
        "sparse_selector": "max_mass_window",
        "sparse_width": 16,
        "raw_slack_selector": "global_top_tokens_slack",
        "global_fixed_chunk_width": 16,
        "global_fixed_chunk_slack": "prefix_of_next_ranked_chunk",
        "segment_length": 256,
        "protected_prefix_segments": 1,
        "protected_suffix_segments": 1,
    }
    if (
        payload.get("schema_version") != C3_SCHEMA
        or payload.get("status") != C3_STATUS
        or model
        != {
            "id": C3_MODEL_ID,
            "revision": C3_MODEL_REVISION,
            "dtype": "bfloat16",
            "weight_bytes": 55563022432,
        }
        or hardware
        != {
            "visible_gpu_count": 1,
            "minimum_device_memory_gib": 80,
            "persistent_free_space_gib": 120,
        }
        or tuple(payload.get("systems", ())) != C3_SYSTEMS
        or tuple(payload.get("equal_byte_systems", ())) != C3_EQUAL_BYTE_SYSTEMS
        or payload.get("primary_comparisons")
        != [
            ["contiguous_cf", "global_fixed_chunk_topk"],
            ["contiguous_cf", "scattered_cf"],
        ]
        or method != expected_method
        or tuple(synthetic.get("budget_fractions", ())) != C3_BUDGETS
        or synthetic.get("max_new_tokens") != 32
        or int(synthetic.get("inference_seed", 0)) <= 0
        or synthetic.get("stage_sets")
        != {"preflight": ["preflight_32k"], "core": ["core_32k"]}
        or set(stages) != {"preflight_32k", "core_32k"}
    ):
        raise C3ProtocolError("C3 top-level contract mismatch")

    preflight = stages["preflight_32k"]
    core = stages["core_32k"]
    if (
        preflight
        != {
            "datasets": "needle",
            "samples_per_dataset": 1,
            "context_length": 32768,
            "segment_length": 256,
            "seed": 20261015,
            "sample_id_prefix": "c3_27b_preflight_32k_s20261015_",
            "budget_fractions": [0.1],
            "systems": ["contiguous_cf", "full_kv_reference"],
            "expected_generation_cells": 2,
        }
        or core
        != {
            "datasets": "needle,longeval_lines",
            "samples_per_dataset": 12,
            "context_length": 32768,
            "segment_length": 256,
            "seed": 20261016,
            "sample_id_prefix": "c3_27b_core_32k_s20261016_",
            "budget_fractions": [0.05, 0.1, 0.2],
            "systems": list(C3_SYSTEMS),
            "expected_generation_cells": 312,
        }
    ):
        raise C3ProtocolError("C3 synthetic stage mismatch")
    for stage in stages.values():
        if _synthetic_cells(stage, C3_SYSTEMS) != stage["expected_generation_cells"]:
            raise C3ProtocolError("C3 synthetic generation-cell count mismatch")

    parent_rel = native.get("parent_protocol")
    if not isinstance(parent_rel, str):
        raise C3ProtocolError("C3 native parent path is missing")
    parent_path = (project_root / parent_rel).resolve()
    if project_root not in parent_path.parents:
        raise C3ProtocolError("C3 native parent escapes project root")
    parent, parent_encoded = _read_json(parent_path)
    parent_sha = hashlib.sha256(parent_encoded).hexdigest()
    native_cases = sum(
        len(spec.get("cases", ())) for spec in parent.get("datasets", {}).values()
    )
    if (
        parent_sha != C3_NATIVE_PARENT_SHA256
        or native.get("parent_protocol_sha256") != C3_NATIVE_PARENT_SHA256
        or native.get("stage_set") != "formal"
        or float(native.get("middle_kv_fraction", 0.0)) != 0.1
        or native.get("datasets") != ["hotpotqa", "narrativeqa"]
        or native.get("cases_per_dataset") != 12
        or native.get("expected_generation_cells") != 120
        or native_cases != 24
        or tuple(parent.get("systems", ())) != C3_SYSTEMS
        or native_cases * len(C3_SYSTEMS) != native["expected_generation_cells"]
    ):
        raise C3ProtocolError("C3 native package mismatch")

    mandatory = payload.get("mandatory_core", {})
    if mandatory != {
        "synthetic_generation_cells": 312,
        "native_generation_cells": 120,
        "total_generation_cells": 432,
        "continuation_gate": False,
    }:
        raise C3ProtocolError("C3 mandatory-core count mismatch")
    if (
        payload.get("extension")
        != {
            "increase_synthetic_to_30_per_dataset_cells": 468,
            "native_20pct_compressed_cells_full_reused": 96,
            "hotpotqa_32k_aug_central_cells": 20,
            "small_64k_stress_cells": 40,
            "automatic": False,
        }
        or payload.get("immutability")
        != {
            "method_or_budget_changes_after_launch": False,
            "case_filtering_after_outcomes": False,
            "preflight_is_a_result_gate": False,
            "extension_requires_new_decision": True,
        }
        or len(payload.get("claims", ())) != 2
        or not all(
            isinstance(claim, str) and claim for claim in payload["claims"]
        )
    ):
        raise C3ProtocolError("C3 extension or interpretation contract mismatch")
    return payload, hashlib.sha256(encoded).hexdigest(), parent


def pareto_protocol_view(payload: Mapping) -> dict:
    """Project the C3 master contract into the existing Pareto runner schema."""
    method = dict(payload["method"])
    method.pop("segment_length")
    return {
        "schema_version": C3_SCHEMA,
        "status": payload["status"],
        "model_id": payload["model"]["id"],
        "model_revision": payload["model"]["revision"],
        "systems": list(payload["systems"]),
        "equal_byte_systems": list(payload["equal_byte_systems"]),
        "primary_comparisons": payload["primary_comparisons"],
        "primary_metric": "normalized_answer_contains",
        "budget_fractions": payload["synthetic"]["budget_fractions"],
        "max_new_tokens": payload["synthetic"]["max_new_tokens"],
        "inference_seed": payload["synthetic"]["inference_seed"],
        "method": method,
        "stages": copy.deepcopy(payload["synthetic"]["stages"]),
        "stage_sets": copy.deepcopy(payload["synthetic"]["stage_sets"]),
    }


def native_protocol_view(payload: Mapping, parent: Mapping) -> dict:
    """Reuse C2's frozen records while changing only the pinned model identity."""
    view = copy.deepcopy(parent)
    view["schema_version"] = C3_SCHEMA
    view["status"] = payload["status"]
    view["model_id"] = payload["model"]["id"]
    view["model_revision"] = payload["model"]["revision"]
    view["c3_parent_protocol_sha256"] = C3_NATIVE_PARENT_SHA256
    return view


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    payload, digest, parent = load_c3_protocol(args.protocol, args.project_root)
    print(
        json.dumps(
            {
                "status": "valid",
                "protocol_sha256": digest,
                "model": payload["model"],
                "preflight_generation_cells": payload["synthetic"]["stages"][
                    "preflight_32k"
                ]["expected_generation_cells"],
                "mandatory_core": payload["mandatory_core"],
                "native_case_count": sum(
                    len(spec["cases"]) for spec in parent["datasets"].values()
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
