"""Persistent, identity-bound FP32 query probes for final HMO runners."""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from experiments.phase2.e3_v2.alpha_probe import AlphaProbeResult
from experiments.phase2.e3_v2.context_query import TokenizedPromptSplit
from experiments.phase2.e3_v2.oracle import OracleContractError, SegmentSpec
from experiments.phase2.e3_v2.query_accessibility import (
    HybridQueryTokenProbeResult,
    QueryAccessibilityResult,
)


QUERY_PROBE_CACHE_SCHEMA = "hmo.query_probe_cache.v1"
QUERY_PROBE_AGGREGATION = "token_fp32_segment_fsum.v1"


@dataclass(frozen=True)
class CachedQueryProbe:
    result: HybridQueryTokenProbeResult
    probe_id: str
    token_scores_sha256: str
    score_path: Path
    metadata_path: Path
    cache_hit: bool

    def provenance(self) -> dict[str, Any]:
        return {
            "schema_version": QUERY_PROBE_CACHE_SCHEMA,
            "aggregation_version": QUERY_PROBE_AGGREGATION,
            "probe_id": self.probe_id,
            "token_scores_sha256": self.token_scores_sha256,
            "token_scores_dtype": "float32",
            "score_path": str(self.score_path),
            "metadata_path": str(self.metadata_path),
            "cache_hit": self.cache_hit,
        }


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _token_ids_sha256(token_ids: torch.Tensor) -> str:
    if token_ids.ndim != 2 or token_ids.shape[0] != 1:
        raise OracleContractError("query-probe identity requires [1, tokens] ids")
    values = np.ascontiguousarray(
        token_ids.detach().to(device="cpu", dtype=torch.int64).numpy(),
        dtype="<i8",
    )
    digest = hashlib.sha256()
    digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def retained_positions_sha256(positions: Sequence[int]) -> str:
    """Hash an ordered retained-position sequence with an explicit length."""
    values = np.ascontiguousarray(tuple(int(value) for value in positions), dtype="<i8")
    if values.ndim != 1 or np.any(values < 0):
        raise OracleContractError("retained positions must be nonnegative integers")
    digest = hashlib.sha256()
    digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def build_query_probe_identity(
    *,
    model_identity: Mapping[str, Any],
    prompt: TokenizedPromptSplit,
    attention_layer_indices: Sequence[int],
    recurrent_layer_indices: Sequence[int],
    segments: Sequence[SegmentSpec],
    segment_length: int,
) -> dict[str, Any]:
    attention = tuple(int(value) for value in attention_layer_indices)
    recurrent = tuple(int(value) for value in recurrent_layer_indices)
    ordered = tuple(sorted(segments, key=lambda item: item.segment_id))
    if (
        not attention
        or not recurrent
        or len(set(attention)) != len(attention)
        or len(set(recurrent)) != len(recurrent)
        or segment_length <= 0
        or not ordered
        or ordered[0].start != 0
        or ordered[-1].end != prompt.context_tokens
    ):
        raise OracleContractError("invalid query-probe identity inputs")
    cursor = 0
    segment_rows = []
    for segment in ordered:
        if segment.start != cursor or segment.end <= segment.start:
            raise OracleContractError("query-probe segments must be contiguous")
        cursor = segment.end
        segment_rows.append(
            {
                "segment_id": int(segment.segment_id),
                "start": int(segment.start),
                "end": int(segment.end),
            }
        )
    model_payload = dict(model_identity)
    required_model_fields = ("model_id", "revision", "config_sha256")
    if any(not model_payload.get(field) for field in required_model_fields):
        raise OracleContractError("query-probe model identity is incomplete")
    return {
        "schema_version": QUERY_PROBE_CACHE_SCHEMA,
        "aggregation_version": QUERY_PROBE_AGGREGATION,
        "model_id": str(model_payload["model_id"]),
        "model_revision": str(model_payload["revision"]),
        "model_fingerprint_sha256": hashlib.sha256(
            _canonical_json(model_payload)
        ).hexdigest(),
        "context_token_ids_sha256": _token_ids_sha256(prompt.context_ids),
        "query_token_ids_sha256": _token_ids_sha256(prompt.query_ids),
        "context_tokens": prompt.context_tokens,
        "query_tokens": prompt.query_tokens,
        "attention_layer_indices": list(attention),
        "recurrent_layer_indices": list(recurrent),
        "segment_length": int(segment_length),
        "segments": segment_rows,
    }


def _validate_collected_probe(
    result: HybridQueryTokenProbeResult,
    identity: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    segment_ids = tuple(row["segment_id"] for row in identity["segments"])
    attention_layers = tuple(identity["attention_layer_indices"])
    recurrent_layers = tuple(identity["recurrent_layer_indices"])
    if (
        result.alpha.context_tokens != identity["context_tokens"]
        or result.alpha.query_tokens != identity["query_tokens"]
        or result.alpha.attention_layer_indices != attention_layers
        or result.alpha.segment_ids != segment_ids
        or result.accessibility.context_tokens != identity["context_tokens"]
        or result.accessibility.query_tokens != identity["query_tokens"]
        or result.accessibility.recurrent_layer_indices != recurrent_layers
        or result.accessibility.segment_ids != segment_ids
    ):
        raise OracleContractError("collected query probe disagrees with its identity")

    scores = np.asarray(result.token_attention_mass, dtype="<f4")
    if (
        scores.shape != (identity["context_tokens"],)
        or not np.isfinite(scores).all()
        or np.any(scores < 0)
    ):
        raise OracleContractError("query-probe token scores must be finite FP32")

    accessibility = {
        name: [float(value) for value in getattr(result.accessibility, name)]
        for name in ("read_norm", "read_share", "read_alignment")
    }
    if any(
        len(values) != len(segment_ids)
        or any(not math.isfinite(value) for value in values)
        for values in accessibility.values()
    ):
        raise OracleContractError("query-probe accessibility payload is invalid")
    return np.ascontiguousarray(scores, dtype="<f4"), accessibility


def _stable_segment_mass(
    scores: np.ndarray, identity: Mapping[str, Any]
) -> tuple[float, ...]:
    return tuple(
        float(
            math.fsum(
                float(value) for value in scores[row["start"] : row["end"]]
            )
        )
        for row in identity["segments"]
    )


def _result_from_payload(
    scores: np.ndarray,
    identity: Mapping[str, Any],
    accessibility: Mapping[str, Sequence[float]],
) -> HybridQueryTokenProbeResult:
    segment_ids = tuple(row["segment_id"] for row in identity["segments"])
    alpha = AlphaProbeResult(
        context_tokens=int(identity["context_tokens"]),
        query_tokens=int(identity["query_tokens"]),
        attention_layer_indices=tuple(identity["attention_layer_indices"]),
        segment_ids=segment_ids,
        attention_mass=_stable_segment_mass(scores, identity),
    )
    access = QueryAccessibilityResult(
        context_tokens=int(identity["context_tokens"]),
        query_tokens=int(identity["query_tokens"]),
        recurrent_layer_indices=tuple(identity["recurrent_layer_indices"]),
        segment_ids=segment_ids,
        read_norm=tuple(float(value) for value in accessibility["read_norm"]),
        read_share=tuple(float(value) for value in accessibility["read_share"]),
        read_alignment=tuple(
            float(value) for value in accessibility["read_alignment"]
        ),
    )
    return HybridQueryTokenProbeResult(
        alpha=alpha,
        accessibility=access,
        token_attention_mass=tuple(float(value) for value in scores),
    )


def _atomic_write_npy(path: Path, values: np.ndarray) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, values, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        encoded = json.dumps(
            payload, ensure_ascii=True, indent=2, sort_keys=True
        ) + "\n"
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_cached_probe(
    *,
    identity: Mapping[str, Any],
    probe_id: str,
    score_path: Path,
    metadata_path: Path,
) -> CachedQueryProbe:
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        scores = np.load(score_path, allow_pickle=False)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise OracleContractError("cannot load persistent query probe") from exc
    score_sha = _sha256_path(score_path)
    if (
        metadata.get("schema_version") != QUERY_PROBE_CACHE_SCHEMA
        or metadata.get("aggregation_version") != QUERY_PROBE_AGGREGATION
        or metadata.get("probe_id") != probe_id
        or metadata.get("identity") != identity
        or metadata.get("token_scores_file") != score_path.name
        or metadata.get("token_scores_sha256") != score_sha
        or scores.dtype != np.dtype("<f4")
        or scores.shape != (identity["context_tokens"],)
        or not np.isfinite(scores).all()
        or np.any(scores < 0)
    ):
        raise OracleContractError("persistent query probe failed identity or hash validation")
    accessibility = metadata.get("accessibility", {})
    expected_segments = len(identity["segments"])
    if any(
        not isinstance(accessibility.get(name), list)
        or len(accessibility[name]) != expected_segments
        or any(not math.isfinite(float(value)) for value in accessibility[name])
        for name in ("read_norm", "read_share", "read_alignment")
    ):
        raise OracleContractError("persistent query-probe accessibility is invalid")
    return CachedQueryProbe(
        result=_result_from_payload(scores, identity, accessibility),
        probe_id=probe_id,
        token_scores_sha256=score_sha,
        score_path=score_path,
        metadata_path=metadata_path,
        cache_hit=True,
    )


def get_or_create_query_probe(
    cache_dir: Path,
    *,
    model_identity: Mapping[str, Any],
    prompt: TokenizedPromptSplit,
    attention_layer_indices: Sequence[int],
    recurrent_layer_indices: Sequence[int],
    segments: Sequence[SegmentSpec],
    segment_length: int,
    collector: Callable[[], HybridQueryTokenProbeResult],
) -> CachedQueryProbe:
    """Load one exact probe artifact or collect and persist it once."""
    identity = build_query_probe_identity(
        model_identity=model_identity,
        prompt=prompt,
        attention_layer_indices=attention_layer_indices,
        recurrent_layer_indices=recurrent_layer_indices,
        segments=segments,
        segment_length=segment_length,
    )
    probe_id = hashlib.sha256(_canonical_json(identity)).hexdigest()
    cache_dir = Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    score_path = cache_dir / f"{probe_id}.token_attention.f32.npy"
    metadata_path = cache_dir / f"{probe_id}.json"
    existing = (score_path.exists(), metadata_path.exists())
    if all(existing):
        return _load_cached_probe(
            identity=identity,
            probe_id=probe_id,
            score_path=score_path,
            metadata_path=metadata_path,
        )
    if any(existing):
        raise OracleContractError("persistent query probe is incomplete")

    collected = collector()
    scores, accessibility = _validate_collected_probe(collected, identity)
    _atomic_write_npy(score_path, scores)
    score_sha = _sha256_path(score_path)
    metadata = {
        "schema_version": QUERY_PROBE_CACHE_SCHEMA,
        "aggregation_version": QUERY_PROBE_AGGREGATION,
        "probe_id": probe_id,
        "identity": identity,
        "token_scores_file": score_path.name,
        "token_scores_sha256": score_sha,
        "token_scores_dtype": "float32",
        "token_scores_count": int(scores.size),
        "accessibility": accessibility,
    }
    _atomic_write_json(metadata_path, metadata)
    result = _result_from_payload(scores, identity, accessibility)
    return CachedQueryProbe(
        result=result,
        probe_id=probe_id,
        token_scores_sha256=score_sha,
        score_path=score_path,
        metadata_path=metadata_path,
        cache_hit=False,
    )
