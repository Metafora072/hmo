"""Offline geometry analysis for layer-local HMO and exact free windows."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from experiments.phase2.e3_v2.chunkkv_adapter import build_chunkkv_plan
from experiments.phase2.e3_v2.coverage_fidelity_cache import (
    RetainedPositionPlan,
    SegmentRetention,
)
from experiments.phase2.e3_v2.free_window_allocator import (
    build_free_window_plan,
    build_layer_local_hmo_plan,
)
from experiments.phase2.e3_v2.oracle import OracleContractError, SegmentSpec
from experiments.phase2.e3_v2.query_probe_cache import retained_positions_sha256


OFFLINE_ANALYSIS_SCHEMA = "hmo.free_window_offline_analysis.v1"


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_rows(path: Path, limit: int | None) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) == limit:
                    break
    if not rows:
        raise OracleContractError("offline analysis received no result rows")
    return rows


def _segments_from_row(row: Mapping, segment_length: int) -> tuple[SegmentSpec, ...]:
    context_tokens = int(row["context_tokens"])
    token_bytes = int(row["byte_accounting"]["token_kv_bytes"])
    count = math.ceil(context_tokens / segment_length)
    if count <= 2:
        raise OracleContractError("offline analysis requires protected middle context")
    segments = []
    for segment_id in range(count):
        start = segment_id * segment_length
        end = min(start + segment_length, context_tokens)
        token_count = end - start
        segments.append(
            SegmentSpec(
                segment_id=segment_id,
                start=start,
                end=end,
                token_count=token_count,
                kv_bytes=token_count * token_bytes,
                protected=segment_id in {0, count - 1},
                partial=token_count != segment_length,
                normalized_position=((start + end) / 2.0) / context_tokens,
                position_bin=min(int(((start + end) / 2.0) / context_tokens * 4), 3),
            )
        )
    protected_bytes = sum(segment.kv_bytes for segment in segments if segment.protected)
    if protected_bytes != int(row["byte_accounting"]["protected_context_kv_bytes"]):
        raise OracleContractError("offline segment reconstruction changed protected bytes")
    return tuple(segments)


def _load_layer_scores(row: Mapping) -> tuple[tuple[int, ...], np.ndarray]:
    provenance = row["query_probe"]
    layer_path = Path(provenance["layer_score_path"])
    metadata_path = Path(provenance["metadata_path"])
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        layers = np.load(layer_path, allow_pickle=False)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise OracleContractError("offline analysis cannot load a probe") from exc
    layer_indices = tuple(int(value) for value in metadata["identity"]["attention_layer_indices"])
    if (
        metadata.get("probe_id") != provenance.get("probe_id")
        or metadata.get("layer_scores_sha256")
        != provenance.get("layer_scores_sha256")
        or _sha256_path(layer_path) != provenance.get("layer_scores_sha256")
        or layers.dtype != np.dtype("<f4")
        or layers.shape != (len(layer_indices), int(row["context_tokens"]))
        or not np.isfinite(layers).all()
        or np.any(layers < 0)
    ):
        raise OracleContractError("offline probe provenance or shape mismatch")
    return layer_indices, layers


def _legacy_plan(row: Mapping) -> RetainedPositionPlan:
    raw = row["plans"]["contiguous_cf"]["retention"]
    plan = RetainedPositionPlan(
        context_tokens=int(raw["context_tokens"]),
        context_charged_bytes=int(raw["context_charged_bytes"]),
        sparse_selector=str(raw["sparse_selector"]),
        active_positions=tuple(int(value) for value in raw["active_positions"]),
        segments=tuple(
            SegmentRetention(
                segment_id=int(item["segment_id"]),
                action=str(item["action"]),
                positions=tuple(int(value) for value in item["positions"]),
            )
            for item in raw["segments"]
        ),
    )
    if retained_positions_sha256(plan.active_positions) != row["plans"]["contiguous_cf"]["active_positions_sha256"]:
        raise OracleContractError("offline legacy HMO hash mismatch")
    return plan


def _jaccard(left: Sequence[int], right: Sequence[int]) -> float:
    a = set(left)
    b = set(right)
    return len(a & b) / len(a | b) if a or b else 1.0


def _regions_touched(positions: Sequence[int], segments: Sequence[SegmentSpec]) -> int:
    active = set(positions)
    return sum(
        bool(active.intersection(range(segment.start, segment.end)))
        for segment in segments
        if segment.eligible
    )


def _mean(values: Sequence[float]) -> float:
    return float(math.fsum(float(value) for value in values) / len(values))


def _summarize(rows: Sequence[Mapping]) -> dict:
    groups: dict[str, list[Mapping]] = defaultdict(list)
    for row in rows:
        groups[str(row["dataset"])].append(row)
    groups["overall"] = list(rows)
    summary = {}
    for name, members in groups.items():
        summary[name] = {
            "sample_count": len(members),
            "mean_context_tokens": _mean([row["context_tokens"] for row in members]),
            "mean_retained_attention_mass": {
                policy: _mean(
                    [row["retained_attention_mass"][policy] for row in members]
                )
                for policy in (
                    "hmo_legacy",
                    "hmo_layer_local",
                    "chunkkv",
                    "hmo_free_window",
                )
            },
            "mean_mass_delta": {
                "layer_local_minus_legacy": _mean(
                    [row["mass_delta"]["layer_local_minus_legacy"] for row in members]
                ),
                "free_window_minus_chunkkv": _mean(
                    [row["mass_delta"]["free_window_minus_chunkkv"] for row in members]
                ),
            },
            "changed_layer_fraction": {
                "layer_local_vs_legacy": _mean(
                    [row["changed_layer_fraction"]["layer_local_vs_legacy"] for row in members]
                ),
                "free_window_vs_chunkkv": _mean(
                    [row["changed_layer_fraction"]["free_window_vs_chunkkv"] for row in members]
                ),
            },
            "mean_position_jaccard": {
                "layer_local_vs_legacy": _mean(
                    [row["mean_position_jaccard"]["layer_local_vs_legacy"] for row in members]
                ),
                "free_window_vs_chunkkv": _mean(
                    [row["mean_position_jaccard"]["free_window_vs_chunkkv"] for row in members]
                ),
            },
            "mean_eligible_regions_touched": {
                policy: _mean(
                    [row["eligible_regions_touched"][policy] for row in members]
                )
                for policy in (
                    "hmo_legacy",
                    "hmo_layer_local",
                    "chunkkv",
                    "hmo_free_window",
                )
            },
            "mean_crossed_macro_boundaries_free_window": _mean(
                [row["crossed_macro_boundaries_free_window"] for row in members]
            ),
            "mean_fixed_fragment_tokens_per_layer": _mean(
                [row["fixed_fragment_tokens_per_layer"] for row in members]
            ),
            "mean_selector_seconds": {
                "hmo_layer_local": _mean(
                    [row["selector_seconds"]["hmo_layer_local"] for row in members]
                ),
                "hmo_free_window": _mean(
                    [row["selector_seconds"]["hmo_free_window"] for row in members]
                ),
            },
        }
    return summary


def analyze(
    results_path: Path,
    *,
    segment_length: int,
    chunk_size: int,
    limit: int | None,
) -> dict:
    source_rows = _load_rows(results_path, limit)
    output_rows = []
    unique_probe_ids = set()
    started = time.perf_counter()
    for index, row in enumerate(source_rows, start=1):
        context_tokens = int(row["context_tokens"])
        segments = _segments_from_row(row, segment_length)
        layer_indices, layer_array = _load_layer_scores(row)
        layer_scores = {
            layer_index: layer_array[offset]
            for offset, layer_index in enumerate(layer_indices)
        }
        unique_probe_ids.add(str(row["query_probe"]["probe_id"]))
        legacy = _legacy_plan(row)
        target_bytes = int(legacy.context_charged_bytes)
        token_bytes = int(row["byte_accounting"]["token_kv_bytes"])
        chunkkv = build_chunkkv_plan(
            segments,
            layer_scores,
            context_tokens=context_tokens,
            target_context_charged_bytes=target_bytes,
            context_token_kv_bytes=token_bytes,
            observation_query_tokens=int(row["query_tokens"]),
            chunk_size=chunk_size,
        )
        stored_chunk = row["plans"]["chunkkv"]
        if chunkkv.to_dict() != stored_chunk:
            raise OracleContractError("offline ChunkKV reconstruction changed history")

        layer_local = build_layer_local_hmo_plan(
            segments,
            layer_scores,
            legacy,
            context_token_kv_bytes=token_bytes,
        )
        free_window = build_free_window_plan(segments, layer_scores, chunkkv)
        if len({target_bytes, layer_local.context_charged_bytes, free_window.context_charged_bytes}) != 1:
            raise OracleContractError("offline policies are not exactly equal-byte")

        protected = set(free_window.protected_positions)
        legacy_regions = _regions_touched(legacy.active_positions, segments)
        masses = defaultdict(list)
        local_changed = []
        free_changed = []
        local_jaccard = []
        free_jaccard = []
        chunk_regions = []
        local_regions = []
        free_regions = []
        crossed = []
        fixed_tokens = []
        for offset, layer_index in enumerate(layer_indices):
            scores = layer_array[offset].astype(np.float64, copy=False)
            baseline_layer = chunkkv.layers[offset]
            local_layer = layer_local.layers[offset]
            free_layer = free_window.layers[offset]
            if not (
                baseline_layer.layer_index
                == local_layer.layer_index
                == free_layer.layer_index
                == layer_index
            ):
                raise OracleContractError("offline layer plans are misordered")
            masses["hmo_legacy"].append(
                math.fsum(float(scores[position]) for position in legacy.active_positions)
            )
            masses["hmo_layer_local"].append(
                math.fsum(float(scores[position]) for position in local_layer.active_positions)
            )
            masses["chunkkv"].append(
                math.fsum(float(scores[position]) for position in baseline_layer.active_positions)
            )
            masses["hmo_free_window"].append(
                math.fsum(float(scores[position]) for position in free_layer.active_positions)
            )
            local_changed.append(local_layer.active_positions != legacy.active_positions)
            free_changed.append(free_layer.active_positions != baseline_layer.active_positions)
            local_jaccard.append(_jaccard(local_layer.active_positions, legacy.active_positions))
            free_jaccard.append(_jaccard(free_layer.active_positions, baseline_layer.active_positions))
            chunk_regions.append(_regions_touched(baseline_layer.active_positions, segments))
            local_regions.append(local_layer.eligible_regions_touched)
            free_regions.append(free_layer.eligible_regions_touched)
            crossed.append(free_layer.crossed_macro_boundaries)
            fixed_tokens.append(
                sum(len(fragment.positions) for fragment in free_layer.fixed_fragments)
            )
            if (
                free_layer.middle_attention_mass
                + 1e-12
                < float(free_layer.baseline_middle_attention_mass)
                or local_layer.middle_attention_mass
                + 1e-12
                < float(local_layer.baseline_middle_attention_mass)
            ):
                raise OracleContractError("offline proxy non-degradation failed")

        policy_mass = {name: _mean(values) for name, values in masses.items()}
        output_rows.append(
            {
                "sample_id": row["sample_id"],
                "dataset": row["dataset"],
                "record_index": row["record_index"],
                "probe_id": row["query_probe"]["probe_id"],
                "context_tokens": context_tokens,
                "attention_layers": len(layer_indices),
                "equal_context_resident_kv_bytes": target_bytes,
                "retained_context_tokens_per_layer": len(legacy.active_positions),
                "retained_attention_mass": policy_mass,
                "mass_delta": {
                    "layer_local_minus_legacy": policy_mass["hmo_layer_local"]
                    - policy_mass["hmo_legacy"],
                    "free_window_minus_chunkkv": policy_mass["hmo_free_window"]
                    - policy_mass["chunkkv"],
                },
                "changed_layer_fraction": {
                    "layer_local_vs_legacy": _mean(local_changed),
                    "free_window_vs_chunkkv": _mean(free_changed),
                },
                "mean_position_jaccard": {
                    "layer_local_vs_legacy": _mean(local_jaccard),
                    "free_window_vs_chunkkv": _mean(free_jaccard),
                },
                "eligible_regions_touched": {
                    "hmo_legacy": float(legacy_regions),
                    "hmo_layer_local": _mean(local_regions),
                    "chunkkv": _mean(chunk_regions),
                    "hmo_free_window": _mean(free_regions),
                },
                "crossed_macro_boundaries_free_window": _mean(crossed),
                "fixed_fragment_tokens_per_layer": _mean(fixed_tokens),
                "selector_seconds": {
                    "hmo_layer_local": math.fsum(
                        layer.selector_seconds for layer in layer_local.layers
                    ),
                    "hmo_free_window": math.fsum(
                        layer.selector_seconds for layer in free_window.layers
                    ),
                },
                "protected_context_tokens": len(protected),
                "exact_equal_bytes": True,
                "chunkkv_reconstruction_exact": True,
            }
        )
        if index % 25 == 0 or index == len(source_rows):
            print(f"[{index}/{len(source_rows)}] offline geometry complete", flush=True)

    return {
        "schema_version": OFFLINE_ANALYSIS_SCHEMA,
        "status": "complete",
        "source_results_path": str(results_path.resolve()),
        "source_case_count": len(source_rows),
        "unique_probe_count": len(unique_probe_ids),
        "segment_length": segment_length,
        "chunk_size": chunk_size,
        "all_equal_bytes": all(row["exact_equal_bytes"] for row in output_rows),
        "all_chunkkv_reconstructions_exact": all(
            row["chunkkv_reconstruction_exact"] for row in output_rows
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "summary": _summarize(output_rows),
        "samples": output_rows,
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
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--segment-length", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = analyze(
        args.results,
        segment_length=args.segment_length,
        chunk_size=args.chunk_size,
        limit=args.limit,
    )
    _atomic_json(args.output, payload)
    print(json.dumps({key: value for key, value in payload.items() if key != "samples"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
