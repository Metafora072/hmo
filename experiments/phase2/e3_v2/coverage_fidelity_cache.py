"""Cache intervention for a query-attention coverage-fidelity plan."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence

import torch

from experiments.phase2.e3_v2.context_query import InterventionResult
from experiments.phase2.e3_v2.coverage_fidelity import CoverageFidelityPlan
from experiments.phase2.e3_v2.oracle import OracleContractError, SegmentSpec
from experiments.utils.cache_access import get_cache_layer
from experiments.utils.memory_accounting import get_active_kv_bytes


@dataclass(frozen=True)
class SegmentRetention:
    segment_id: int
    action: str
    positions: tuple[int, ...]


@dataclass(frozen=True)
class RetainedPositionPlan:
    context_tokens: int
    context_charged_bytes: int
    active_positions: tuple[int, ...]
    segments: tuple[SegmentRetention, ...]

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "active_positions": list(self.active_positions),
            "segments": [
                {**asdict(item), "positions": list(item.positions)}
                for item in self.segments
            ],
        }


def select_query_attention_positions(
    token_attention_mass: Sequence[float],
    segment: SegmentSpec,
    width: int,
) -> list[int]:
    """Select stable top-attention positions within one segment."""
    if width <= 0 or len(token_attention_mass) < segment.end:
        raise OracleContractError("invalid query-attention Sparse selection inputs")
    values = [float(value) for value in token_attention_mass[segment.start : segment.end]]
    if len(values) != segment.token_count or any(
        not math.isfinite(value) or value < 0 for value in values
    ):
        raise OracleContractError(
            "query-attention Sparse scores must be finite and nonnegative"
        )
    keep = min(width, segment.token_count)
    order = sorted(range(len(values)), key=lambda index: (-values[index], index))
    return sorted(segment.start + index for index in order[:keep])


def build_retained_position_plan(
    plan: CoverageFidelityPlan,
    segments: Sequence[SegmentSpec],
    token_attention_mass: Sequence[float],
    *,
    context_tokens: int,
) -> RetainedPositionPlan:
    """Materialize exact token positions for every allocator action."""
    ordered = tuple(sorted(segments, key=lambda item: item.segment_id))
    allocations = {item.segment_id: item for item in plan.allocations}
    if (
        not ordered
        or ordered[0].start != 0
        or ordered[-1].end != context_tokens
        or len(token_attention_mass) != context_tokens
        or set(allocations) != {item.segment_id for item in ordered}
    ):
        raise OracleContractError("coverage-fidelity position inputs are misaligned")

    cursor = 0
    charged = 0
    retention = []
    active = []
    for segment in ordered:
        if segment.start != cursor or segment.end <= segment.start:
            raise OracleContractError("coverage-fidelity segments must be contiguous")
        cursor = segment.end
        allocation = allocations[segment.segment_id]
        if allocation.action == "recurrent_only":
            positions = []
        elif allocation.action == "exact":
            positions = list(range(segment.start, segment.end))
        elif allocation.action == "sparse":
            positions = select_query_attention_positions(
                token_attention_mass,
                segment,
                allocation.retained_tokens,
            )
        else:
            raise OracleContractError(
                f"unknown coverage-fidelity action {allocation.action!r}"
            )
        if len(positions) != allocation.retained_tokens:
            raise OracleContractError("retained position count disagrees with allocation")
        unit = segment.kv_bytes // segment.token_count
        if segment.kv_bytes % segment.token_count or len(positions) * unit != allocation.charged_bytes:
            raise OracleContractError("retained positions disagree with charged KV bytes")
        charged += allocation.charged_bytes
        active.extend(positions)
        retention.append(
            SegmentRetention(
                segment_id=segment.segment_id,
                action=allocation.action,
                positions=tuple(positions),
            )
        )
    if charged != plan.total_charged_bytes or active != sorted(set(active)):
        raise OracleContractError("materialized coverage-fidelity plan violates byte/order contract")
    return RetainedPositionPlan(
        context_tokens=context_tokens,
        context_charged_bytes=charged,
        active_positions=tuple(active),
        segments=tuple(retention),
    )


def make_coverage_fidelity_intervention(
    position_plan: RetainedPositionPlan,
    attention_layer_indices: Sequence[int],
    *,
    name: str,
):
    """Build an in-place attention-KV intervention from retained positions."""
    layer_indices = tuple(int(value) for value in attention_layer_indices)
    if not layer_indices or not position_plan.active_positions:
        raise OracleContractError("coverage-fidelity intervention cannot be empty")

    def intervene(cache, context_ids: torch.Tensor) -> InterventionResult:
        if context_ids.shape != (1, position_plan.context_tokens):
            raise OracleContractError("coverage-fidelity intervention context mismatch")
        expected_full_bytes = 0
        for layer_index in layer_indices:
            layer = get_cache_layer(cache, layer_index)
            if not layer.has_kv() or layer.keys.shape[-2] != position_plan.context_tokens:
                raise OracleContractError("coverage-fidelity intervention requires Full-KV")
            expected_full_bytes += int(
                layer.keys.numel() * layer.keys.element_size()
                + layer.values.numel() * layer.values.element_size()
            )
        before_bytes = get_active_kv_bytes(cache, list(layer_indices))
        if before_bytes != expected_full_bytes:
            raise OracleContractError("coverage-fidelity Full-KV byte count is inconsistent")

        positions = torch.tensor(
            position_plan.active_positions,
            device=context_ids.device,
            dtype=torch.long,
        )
        for layer_index in layer_indices:
            layer = get_cache_layer(cache, layer_index)
            layer.keys = layer.keys.index_select(-2, positions)
            layer.values = layer.values.index_select(-2, positions)
        after_bytes = get_active_kv_bytes(cache, list(layer_indices))
        if after_bytes != position_plan.context_charged_bytes:
            raise OracleContractError(
                "coverage-fidelity resident bytes disagree with allocator charge"
            )
        return InterventionResult(
            name=name,
            active_context_positions=positions,
            metadata={
                "context_resident_bytes": int(after_bytes),
                "retained_context_tokens": len(position_plan.active_positions),
            },
        )

    return intervene
