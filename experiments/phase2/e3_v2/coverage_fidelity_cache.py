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


SPARSE_SELECTORS = ("top_tokens", "max_mass_window")
RAW_EXACT_SLACK_SELECTOR = "global_top_tokens_slack"


@dataclass(frozen=True)
class SegmentRetention:
    segment_id: int
    action: str
    positions: tuple[int, ...]


@dataclass(frozen=True)
class RetainedPositionPlan:
    context_tokens: int
    context_charged_bytes: int
    sparse_selector: str
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


def select_max_attention_window_positions(
    token_attention_mass: Sequence[float],
    segment: SegmentSpec,
    width: int,
) -> list[int]:
    """Select the contiguous window with maximum query-attention mass."""
    if width <= 0 or len(token_attention_mass) < segment.end:
        raise OracleContractError("invalid attention-window Sparse selection inputs")
    values = [float(value) for value in token_attention_mass[segment.start : segment.end]]
    if len(values) != segment.token_count or any(
        not math.isfinite(value) or value < 0 for value in values
    ):
        raise OracleContractError(
            "attention-window Sparse scores must be finite and nonnegative"
        )
    keep = min(width, segment.token_count)
    window_sum = sum(values[:keep])
    best_sum = window_sum
    best_start = 0
    for start in range(1, len(values) - keep + 1):
        window_sum += values[start + keep - 1] - values[start - 1]
        if window_sum > best_sum:
            best_sum = window_sum
            best_start = start
    return list(
        range(segment.start + best_start, segment.start + best_start + keep)
    )


def select_sparse_positions(
    token_attention_mass: Sequence[float],
    segment: SegmentSpec,
    width: int,
    *,
    selector: str,
) -> list[int]:
    if selector == "top_tokens":
        return select_query_attention_positions(token_attention_mass, segment, width)
    if selector == "max_mass_window":
        return select_max_attention_window_positions(
            token_attention_mass, segment, width
        )
    raise OracleContractError(f"unknown Sparse selector {selector!r}")


def build_retained_position_plan(
    plan: CoverageFidelityPlan,
    segments: Sequence[SegmentSpec],
    token_attention_mass: Sequence[float],
    *,
    context_tokens: int,
    sparse_selector: str = "top_tokens",
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
            positions = select_sparse_positions(
                token_attention_mass,
                segment,
                allocation.retained_tokens,
                selector=sparse_selector,
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
        sparse_selector=sparse_selector,
        active_positions=tuple(active),
        segments=tuple(retention),
    )


def build_raw_exact_slack_position_plan(
    segments: Sequence[SegmentSpec],
    selected_exact_segment_ids: Sequence[int],
    token_attention_mass: Sequence[float],
    *,
    context_tokens: int,
    target_context_charged_bytes: int,
) -> RetainedPositionPlan:
    """Spend Raw Exact segment-rounding slack on top remaining query tokens."""
    ordered = tuple(sorted(segments, key=lambda item: item.segment_id))
    selected_exact = {int(value) for value in selected_exact_segment_ids}
    eligible_ids = {item.segment_id for item in ordered if item.eligible}
    if (
        not ordered
        or ordered[0].start != 0
        or ordered[-1].end != context_tokens
        or len(token_attention_mass) != context_tokens
        or not selected_exact
        or not selected_exact <= eligible_ids
    ):
        raise OracleContractError("raw Exact+Slack inputs are misaligned")

    cursor = 0
    per_token_costs = set()
    for segment in ordered:
        if segment.start != cursor or segment.end <= segment.start:
            raise OracleContractError("raw Exact+Slack segments must be contiguous")
        cursor = segment.end
        if segment.kv_bytes % segment.token_count:
            raise OracleContractError("raw Exact+Slack segment cost is not token-divisible")
        per_token_costs.add(segment.kv_bytes // segment.token_count)
    if len(per_token_costs) != 1:
        raise OracleContractError("raw Exact+Slack requires one per-token KV cost")
    token_cost = next(iter(per_token_costs))

    exact_ids = selected_exact | {
        item.segment_id for item in ordered if item.protected
    }
    exact_bytes = sum(item.kv_bytes for item in ordered if item.segment_id in exact_ids)
    slack_bytes = target_context_charged_bytes - exact_bytes
    if slack_bytes < 0 or slack_bytes % token_cost:
        raise OracleContractError("raw Exact+Slack target cannot be matched exactly")
    slack_tokens = slack_bytes // token_cost

    candidates = []
    for segment in ordered:
        if not segment.eligible or segment.segment_id in selected_exact:
            continue
        for position in range(segment.start, segment.end):
            mass = float(token_attention_mass[position])
            if not math.isfinite(mass) or mass < 0:
                raise OracleContractError(
                    "raw Exact+Slack token scores must be finite and nonnegative"
                )
            candidates.append((-mass, position))
    if slack_tokens > len(candidates):
        raise OracleContractError("raw Exact+Slack target exceeds remaining context")
    slack_positions = {
        position for _, position in sorted(candidates)[:slack_tokens]
    }

    active = []
    retention = []
    for segment in ordered:
        if segment.segment_id in exact_ids:
            positions = tuple(range(segment.start, segment.end))
            action = "exact"
        else:
            positions = tuple(
                position
                for position in range(segment.start, segment.end)
                if position in slack_positions
            )
            action = "sparse" if positions else "recurrent_only"
        active.extend(positions)
        retention.append(
            SegmentRetention(
                segment_id=segment.segment_id,
                action=action,
                positions=positions,
            )
        )
    if len(active) * token_cost != target_context_charged_bytes:
        raise OracleContractError("raw Exact+Slack failed its byte target")
    return RetainedPositionPlan(
        context_tokens=context_tokens,
        context_charged_bytes=target_context_charged_bytes,
        sparse_selector=RAW_EXACT_SLACK_SELECTOR,
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
