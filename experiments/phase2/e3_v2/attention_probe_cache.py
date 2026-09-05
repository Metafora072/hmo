"""Persistent identity-bound FP32 artifacts for the attention-only probe."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from experiments.phase2.e3_v2.alpha_probe import AlphaProbeResult
from experiments.phase2.e3_v2.attention_probe import AttentionTokenProbeResult
from experiments.phase2.e3_v2.context_query import TokenizedPromptSplit
from experiments.phase2.e3_v2.oracle import OracleContractError, SegmentSpec
from experiments.phase2.e3_v2.query_probe_cache import (
    _atomic_write_json,
    _atomic_write_npy,
    _canonical_json,
    _sha256_path,
    _token_ids_sha256,
    retained_positions_sha256,
)


ATTENTION_PROBE_CACHE_SCHEMA = "hmo.attention_probe_cache.v2"
ATTENTION_PROBE_AGGREGATION = "token_and_layer_fp32_segment_fsum.v2"


@dataclass(frozen=True)
class CachedAttentionProbe:
    result: AttentionTokenProbeResult
    probe_id: str
    token_scores_sha256: str
    layer_scores_sha256: str
    score_path: Path
    layer_score_path: Path
    metadata_path: Path
    cache_hit: bool

    def provenance(self) -> dict[str, Any]:
        return {
            "schema_version": ATTENTION_PROBE_CACHE_SCHEMA,
            "aggregation_version": ATTENTION_PROBE_AGGREGATION,
            "probe_id": self.probe_id,
            "token_scores_sha256": self.token_scores_sha256,
            "layer_scores_sha256": self.layer_scores_sha256,
            "token_scores_dtype": "float32",
            "score_path": str(self.score_path),
            "layer_score_path": str(self.layer_score_path),
            "metadata_path": str(self.metadata_path),
            "cache_hit": self.cache_hit,
        }


def build_attention_probe_identity(
    *,
    model_identity: Mapping[str, Any],
    prompt: TokenizedPromptSplit,
    attention_layer_indices: Sequence[int],
    segments: Sequence[SegmentSpec],
    segment_length: int,
) -> dict[str, Any]:
    layers = tuple(int(value) for value in attention_layer_indices)
    ordered = tuple(sorted(segments, key=lambda item: item.segment_id))
    if (
        not layers
        or len(set(layers)) != len(layers)
        or segment_length <= 0
        or not ordered
        or ordered[0].start != 0
        or ordered[-1].end != prompt.context_tokens
    ):
        raise OracleContractError("invalid attention-probe identity inputs")
    cursor = 0
    rows = []
    for segment in ordered:
        if segment.start != cursor or segment.end <= segment.start:
            raise OracleContractError("attention-probe segments must be contiguous")
        cursor = segment.end
        rows.append(
            {
                "segment_id": int(segment.segment_id),
                "start": int(segment.start),
                "end": int(segment.end),
            }
        )
    model_payload = dict(model_identity)
    if any(
        not model_payload.get(field)
        for field in ("model_id", "revision", "config_sha256")
    ):
        raise OracleContractError("attention-probe model identity is incomplete")
    return {
        "schema_version": ATTENTION_PROBE_CACHE_SCHEMA,
        "aggregation_version": ATTENTION_PROBE_AGGREGATION,
        "model_id": str(model_payload["model_id"]),
        "model_revision": str(model_payload["revision"]),
        "model_fingerprint_sha256": hashlib.sha256(
            _canonical_json(model_payload)
        ).hexdigest(),
        "context_token_ids_sha256": _token_ids_sha256(prompt.context_ids),
        "query_token_ids_sha256": _token_ids_sha256(prompt.query_ids),
        "context_tokens": prompt.context_tokens,
        "query_tokens": prompt.query_tokens,
        "attention_layer_indices": list(layers),
        "segment_length": int(segment_length),
        "segments": rows,
    }


def _validate_collected(
    result: AttentionTokenProbeResult, identity: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    if (
        result.alpha.context_tokens != identity["context_tokens"]
        or result.alpha.query_tokens != identity["query_tokens"]
        or result.alpha.attention_layer_indices
        != tuple(identity["attention_layer_indices"])
        or result.alpha.segment_ids
        != tuple(row["segment_id"] for row in identity["segments"])
    ):
        raise OracleContractError("collected attention probe disagrees with identity")
    aggregate = np.asarray(result.token_attention_mass, dtype="<f4")
    layers = np.asarray(result.layer_token_attention_mass, dtype="<f4")
    expected = int(identity["context_tokens"])
    if (
        aggregate.shape != (expected,)
        or layers.shape != (len(identity["attention_layer_indices"]), expected)
        or not np.isfinite(aggregate).all()
        or not np.isfinite(layers).all()
        or np.any(aggregate < 0)
        or np.any(layers < 0)
    ):
        raise OracleContractError("attention-probe scores must be finite FP32")
    return (
        np.ascontiguousarray(aggregate, dtype="<f4"),
        np.ascontiguousarray(layers, dtype="<f4"),
    )


def _result_from_arrays(
    aggregate: np.ndarray, layers: np.ndarray, identity: Mapping[str, Any]
) -> AttentionTokenProbeResult:
    segment_mass = tuple(
        float(
            math.fsum(
                float(value) for value in aggregate[row["start"] : row["end"]]
            )
        )
        for row in identity["segments"]
    )
    return AttentionTokenProbeResult(
        alpha=AlphaProbeResult(
            context_tokens=int(identity["context_tokens"]),
            query_tokens=int(identity["query_tokens"]),
            attention_layer_indices=tuple(identity["attention_layer_indices"]),
            segment_ids=tuple(row["segment_id"] for row in identity["segments"]),
            attention_mass=segment_mass,
        ),
        token_attention_mass=tuple(float(value) for value in aggregate),
        layer_token_attention_mass=tuple(
            tuple(float(value) for value in row) for row in layers
        ),
    )


def _load_cached(
    *,
    identity: Mapping[str, Any],
    probe_id: str,
    score_path: Path,
    layer_score_path: Path,
    metadata_path: Path,
) -> CachedAttentionProbe:
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        aggregate = np.load(score_path, allow_pickle=False)
        layers = np.load(layer_score_path, allow_pickle=False)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise OracleContractError("cannot load persistent attention probe") from exc
    aggregate_sha = _sha256_path(score_path)
    layers_sha = _sha256_path(layer_score_path)
    if (
        metadata.get("schema_version") != ATTENTION_PROBE_CACHE_SCHEMA
        or metadata.get("aggregation_version") != ATTENTION_PROBE_AGGREGATION
        or metadata.get("probe_id") != probe_id
        or metadata.get("identity") != identity
        or metadata.get("token_scores_file") != score_path.name
        or metadata.get("layer_scores_file") != layer_score_path.name
        or metadata.get("token_scores_sha256") != aggregate_sha
        or metadata.get("layer_scores_sha256") != layers_sha
        or aggregate.dtype != np.dtype("<f4")
        or layers.dtype != np.dtype("<f4")
        or aggregate.shape != (identity["context_tokens"],)
        or layers.shape
        != (len(identity["attention_layer_indices"]), identity["context_tokens"])
        or not np.isfinite(aggregate).all()
        or not np.isfinite(layers).all()
        or np.any(aggregate < 0)
        or np.any(layers < 0)
    ):
        raise OracleContractError("persistent attention probe failed validation")
    return CachedAttentionProbe(
        result=_result_from_arrays(aggregate, layers, identity),
        probe_id=probe_id,
        token_scores_sha256=aggregate_sha,
        layer_scores_sha256=layers_sha,
        score_path=score_path,
        layer_score_path=layer_score_path,
        metadata_path=metadata_path,
        cache_hit=True,
    )


def get_or_create_attention_probe(
    cache_dir: Path,
    *,
    model_identity: Mapping[str, Any],
    prompt: TokenizedPromptSplit,
    attention_layer_indices: Sequence[int],
    segments: Sequence[SegmentSpec],
    segment_length: int,
    collector: Callable[[], AttentionTokenProbeResult],
) -> CachedAttentionProbe:
    """Load one exact v2 artifact or collect and persist it once."""
    identity = build_attention_probe_identity(
        model_identity=model_identity,
        prompt=prompt,
        attention_layer_indices=attention_layer_indices,
        segments=segments,
        segment_length=segment_length,
    )
    probe_id = hashlib.sha256(_canonical_json(identity)).hexdigest()
    cache_dir = Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    score_path = cache_dir / f"{probe_id}.token_attention.f32.npy"
    layer_score_path = cache_dir / f"{probe_id}.layer_attention.f32.npy"
    metadata_path = cache_dir / f"{probe_id}.json"
    existing = (score_path.exists(), layer_score_path.exists(), metadata_path.exists())
    if all(existing):
        return _load_cached(
            identity=identity,
            probe_id=probe_id,
            score_path=score_path,
            layer_score_path=layer_score_path,
            metadata_path=metadata_path,
        )
    if any(existing):
        raise OracleContractError("persistent attention probe is incomplete")

    aggregate, layers = _validate_collected(collector(), identity)
    _atomic_write_npy(score_path, aggregate)
    _atomic_write_npy(layer_score_path, layers)
    aggregate_sha = _sha256_path(score_path)
    layers_sha = _sha256_path(layer_score_path)
    _atomic_write_json(
        metadata_path,
        {
            "schema_version": ATTENTION_PROBE_CACHE_SCHEMA,
            "aggregation_version": ATTENTION_PROBE_AGGREGATION,
            "probe_id": probe_id,
            "identity": identity,
            "token_scores_file": score_path.name,
            "layer_scores_file": layer_score_path.name,
            "token_scores_sha256": aggregate_sha,
            "layer_scores_sha256": layers_sha,
            "token_scores_dtype": "float32",
            "layer_scores_dtype": "float32",
        },
    )
    return CachedAttentionProbe(
        result=_result_from_arrays(aggregate, layers, identity),
        probe_id=probe_id,
        token_scores_sha256=aggregate_sha,
        layer_scores_sha256=layers_sha,
        score_path=score_path,
        layer_score_path=layer_score_path,
        metadata_path=metadata_path,
        cache_hit=False,
    )
