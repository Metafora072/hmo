"""Equal-byte ChunkKV adapter for hybrid Full-Attention/DeltaNet models."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from experiments.phase2.e3_v2.context_query import InterventionResult
from experiments.phase2.e3_v2.oracle import OracleContractError, SegmentSpec
from experiments.phase2.e3_v2.query_probe_cache import retained_positions_sha256
from experiments.utils.cache_access import get_cache_layer
from experiments.utils.memory_accounting import get_active_kv_bytes


CHUNKKV_ADAPTER_SCHEMA = "hmo.chunkkv_hybrid_adapter.v1"
CHUNKKV_CHUNK_SIZE = 10


@dataclass(frozen=True)
class ChunkKVLayerPlan:
    layer_index: int
    active_positions: tuple[int, ...]
    selected_chunk_starts: tuple[int, ...]
    partial_chunk_start: int | None
    partial_chunk_tokens: int

    def to_dict(self) -> dict:
        return {
            "layer_index": self.layer_index,
            "retained_context_tokens": len(self.active_positions),
            "active_positions_sha256": retained_positions_sha256(
                self.active_positions
            ),
            "selected_chunk_starts": list(self.selected_chunk_starts),
            "partial_chunk_start": self.partial_chunk_start,
            "partial_chunk_tokens": self.partial_chunk_tokens,
        }


@dataclass(frozen=True)
class ChunkKVPlan:
    context_tokens: int
    context_charged_bytes: int
    context_token_kv_bytes: int
    chunk_size: int
    observation_query_tokens: int
    protected_positions: tuple[int, ...]
    layers: tuple[ChunkKVLayerPlan, ...]

    def to_dict(self) -> dict:
        return {
            "schema_version": CHUNKKV_ADAPTER_SCHEMA,
            "context_tokens": self.context_tokens,
            "context_charged_bytes": self.context_charged_bytes,
            "context_token_kv_bytes": self.context_token_kv_bytes,
            "chunk_size": self.chunk_size,
            "observation_query_tokens": self.observation_query_tokens,
            "protected_context_tokens": len(self.protected_positions),
            "protected_positions_sha256": retained_positions_sha256(
                self.protected_positions
            ),
            "layer_position_policy": "independent_per_full_layer_shared_across_kv_heads",
            "partial_chunk_policy": "fixed_prefix_of_next_ranked_chunk",
            "recurrent_state_policy": "unchanged",
            "layers": [layer.to_dict() for layer in self.layers],
        }


def _chunk_candidates(
    eligible_positions: tuple[int, ...], scores: Sequence[float], chunk_size: int
) -> list[tuple[float, int, tuple[int, ...]]]:
    candidates = []
    for offset in range(0, len(eligible_positions), chunk_size):
        positions = eligible_positions[offset : offset + chunk_size]
        if positions and positions != tuple(range(positions[0], positions[-1] + 1)):
            raise OracleContractError("ChunkKV eligible middle must be contiguous")
        mass = [float(scores[position]) for position in positions]
        if any(not math.isfinite(value) or value < 0 for value in mass):
            raise OracleContractError("ChunkKV scores must be finite and nonnegative")
        candidates.append((-sum(mass), positions[0], positions))
    return sorted(candidates)


def build_chunkkv_plan(
    segments: Sequence[SegmentSpec],
    layer_token_attention_mass: Mapping[int, Sequence[float]],
    *,
    context_tokens: int,
    target_context_charged_bytes: int,
    context_token_kv_bytes: int,
    observation_query_tokens: int,
    chunk_size: int = CHUNKKV_CHUNK_SIZE,
) -> ChunkKVPlan:
    """Build a per-Full-layer chunk ranking under the shared byte target."""
    ordered = tuple(sorted(segments, key=lambda item: item.segment_id))
    if (
        not ordered
        or ordered[0].start != 0
        or ordered[-1].end != context_tokens
        or chunk_size <= 0
        or observation_query_tokens <= 0
        or context_token_kv_bytes <= 0
        or target_context_charged_bytes % context_token_kv_bytes
    ):
        raise OracleContractError("invalid ChunkKV adapter inputs")
    cursor = 0
    protected = []
    eligible = []
    for segment in ordered:
        if segment.start != cursor or segment.end <= segment.start:
            raise OracleContractError("ChunkKV segments must be contiguous")
        cursor = segment.end
        if not segment.eligible and not segment.protected:
            raise OracleContractError("ChunkKV cannot adapt an unprotected partial segment")
        positions = range(segment.start, segment.end)
        if segment.protected:
            protected.extend(positions)
        elif segment.eligible:
            eligible.extend(positions)
    protected_positions = tuple(protected)
    eligible_positions = tuple(eligible)
    retained_tokens = target_context_charged_bytes // context_token_kv_bytes
    middle_tokens = retained_tokens - len(protected_positions)
    if (
        middle_tokens < 0
        or middle_tokens > len(eligible_positions)
        or not layer_token_attention_mass
    ):
        raise OracleContractError("ChunkKV target cannot be matched by eligible context")

    layer_plans = []
    for layer_index, scores in sorted(layer_token_attention_mass.items()):
        if len(scores) != context_tokens:
            raise OracleContractError("ChunkKV layer scores omit context positions")
        remaining = middle_tokens
        active = set(protected_positions)
        selected_starts = []
        partial_start = None
        partial_tokens = 0
        for _, start, positions in _chunk_candidates(
            eligible_positions, scores, chunk_size
        ):
            if remaining <= 0:
                break
            take = min(remaining, len(positions))
            active.update(positions[:take])
            if take == len(positions):
                selected_starts.append(start)
            else:
                partial_start = start
                partial_tokens = take
            remaining -= take
        positions = tuple(sorted(active))
        if remaining or len(positions) != retained_tokens:
            raise OracleContractError("ChunkKV layer failed its token target")
        layer_plans.append(
            ChunkKVLayerPlan(
                layer_index=int(layer_index),
                active_positions=positions,
                selected_chunk_starts=tuple(sorted(selected_starts)),
                partial_chunk_start=partial_start,
                partial_chunk_tokens=partial_tokens,
            )
        )
    return ChunkKVPlan(
        context_tokens=context_tokens,
        context_charged_bytes=target_context_charged_bytes,
        context_token_kv_bytes=context_token_kv_bytes,
        chunk_size=chunk_size,
        observation_query_tokens=observation_query_tokens,
        protected_positions=protected_positions,
        layers=tuple(layer_plans),
    )


def make_chunkkv_intervention(plan: ChunkKVPlan, *, name: str = "chunkkv"):
    """Apply layer-specific positions while leaving recurrent cache state untouched."""
    layer_indices = tuple(layer.layer_index for layer in plan.layers)
    if not layer_indices or len(set(layer_indices)) != len(layer_indices):
        raise OracleContractError("ChunkKV intervention layers must be unique")

    def intervene(cache, context_ids: torch.Tensor) -> InterventionResult:
        if context_ids.shape != (1, plan.context_tokens):
            raise OracleContractError("ChunkKV intervention context mismatch")
        for layer_index in layer_indices:
            layer = get_cache_layer(cache, layer_index)
            if not layer.has_kv() or layer.keys.shape[-2] != plan.context_tokens:
                raise OracleContractError("ChunkKV intervention requires Full-KV")
        before_bytes = get_active_kv_bytes(cache, list(layer_indices))
        expected_full_bytes = plan.context_tokens * plan.context_token_kv_bytes
        if before_bytes != expected_full_bytes:
            raise OracleContractError("ChunkKV Full-KV bytes disagree with token cost")

        for layer_plan in plan.layers:
            positions = torch.tensor(
                layer_plan.active_positions,
                device=context_ids.device,
                dtype=torch.long,
            )
            layer = get_cache_layer(cache, layer_plan.layer_index)
            layer.keys = layer.keys.index_select(-2, positions)
            layer.values = layer.values.index_select(-2, positions)
        after_bytes = get_active_kv_bytes(cache, list(layer_indices))
        if after_bytes != plan.context_charged_bytes:
            raise OracleContractError("ChunkKV resident bytes disagree with target")
        reference = torch.tensor(
            plan.layers[0].active_positions,
            device=context_ids.device,
            dtype=torch.long,
        )
        return InterventionResult(
            name=name,
            active_context_positions=reference,
            metadata={
                "schema_version": CHUNKKV_ADAPTER_SCHEMA,
                "context_resident_bytes": int(after_bytes),
                "retained_context_tokens_per_layer": int(reference.numel()),
                "layer_position_hashes": {
                    str(layer.layer_index): retained_positions_sha256(
                        layer.active_positions
                    )
                    for layer in plan.layers
                },
            },
        )

    return intervene
