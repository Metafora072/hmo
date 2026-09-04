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
GLOBAL_FIXED_CHUNK_SELECTOR = "global_fixed_chunk_topk_boundary_slack"
STRATIFIED_FIXED_CHUNK_SELECTOR = "stratified_fixed_chunk_aligned_window"


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


def select_max_attention_aligned_window_positions(
    token_attention_mass: Sequence[float],
    segment: SegmentSpec,
    width: int,
    *,
    alignment: int,
) -> list[int]:
    """Select the best window whose start lies on a segment-local boundary."""
    if (
        width <= 0
        or alignment <= 0
        or len(token_attention_mass) < segment.end
    ):
        raise OracleContractError("invalid aligned-window Sparse selection inputs")
    values = [float(value) for value in token_attention_mass[segment.start : segment.end]]
    if len(values) != segment.token_count or any(
        not math.isfinite(value) or value < 0 for value in values
    ):
        raise OracleContractError(
            "aligned-window Sparse scores must be finite and nonnegative"
        )
    keep = min(width, segment.token_count)
    starts = range(0, len(values) - keep + 1, alignment)
    best_start = max(
        starts,
        key=lambda start: (sum(values[start : start + keep]), -start),
    )
    return list(
        range(segment.start + best_start, segment.start + best_start + keep)
    )


def select_sparse_positions(
    token_attention_mass: Sequence[float],
    segment: SegmentSpec,
    width: int,
    *,
    selector: str,
    alignment: int | None = None,
) -> list[int]:
    if selector == "top_tokens":
        return select_query_attention_positions(token_attention_mass, segment, width)
    if selector == "max_mass_window":
        return select_max_attention_window_positions(
            token_attention_mass, segment, width
        )
    if selector == STRATIFIED_FIXED_CHUNK_SELECTOR:
        if alignment is None:
            raise OracleContractError(
                "stratified fixed-chunk selector requires an alignment"
            )
        return select_max_attention_aligned_window_positions(
            token_attention_mass,
            segment,
            width,
            alignment=alignment,
        )
    raise OracleContractError(f"unknown Sparse selector {selector!r}")


def build_retained_position_plan(
    plan: CoverageFidelityPlan,
    segments: Sequence[SegmentSpec],
    token_attention_mass: Sequence[float],
    *,
    context_tokens: int,
    sparse_selector: str = "top_tokens",
    sparse_alignment: int | None = None,
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
                alignment=sparse_alignment,
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


def build_global_fixed_chunk_topk_position_plan(
    segments: Sequence[SegmentSpec],
    token_attention_mass: Sequence[float],
    *,
    context_tokens: int,
    target_context_charged_bytes: int,
    chunk_width: int,
) -> RetainedPositionPlan:
    """Globally rank fixed-boundary chunks and exactly match a byte target.

    Whole chunks are selected in score order. If the target has a remainder,
    the final tokens come from the fixed-boundary prefix of the next ranked
    chunk, preserving deterministic byte equality without a free-start window.
    """
    ordered = tuple(sorted(segments, key=lambda item: item.segment_id))
    if (
        chunk_width <= 0
        or not ordered
        or ordered[0].start != 0
        or ordered[-1].end != context_tokens
        or len(token_attention_mass) != context_tokens
    ):
        raise OracleContractError("global fixed-chunk inputs are misaligned")

    cursor = 0
    per_token_costs = set()
    supported_tokens = 0
    protected_positions = set()
    for segment in ordered:
        if segment.start != cursor or segment.end <= segment.start:
            raise OracleContractError("global fixed-chunk segments must be contiguous")
        cursor = segment.end
        if not segment.eligible and not segment.protected:
            raise OracleContractError(
                "global fixed-chunk has no policy for unprotected partial segments"
            )
        if segment.kv_bytes % segment.token_count:
            raise OracleContractError(
                "global fixed-chunk segment cost is not token-divisible"
            )
        per_token_costs.add(segment.kv_bytes // segment.token_count)
        supported_tokens += segment.token_count
        if segment.protected:
            protected_positions.update(range(segment.start, segment.end))
    if len(per_token_costs) != 1:
        raise OracleContractError("global fixed-chunk requires one per-token KV cost")
    token_cost = next(iter(per_token_costs))
    if (
        target_context_charged_bytes < len(protected_positions) * token_cost
        or target_context_charged_bytes > supported_tokens * token_cost
        or target_context_charged_bytes % token_cost
    ):
        raise OracleContractError("global fixed-chunk target cannot be matched exactly")

    remaining = (
        target_context_charged_bytes // token_cost - len(protected_positions)
    )
    candidates = []
    for segment in ordered:
        if not segment.eligible:
            continue
        for start in range(segment.start, segment.end, chunk_width):
            positions = tuple(range(start, min(start + chunk_width, segment.end)))
            values = [float(token_attention_mass[position]) for position in positions]
            if any(not math.isfinite(value) or value < 0 for value in values):
                raise OracleContractError(
                    "global fixed-chunk scores must be finite and nonnegative"
                )
            candidates.append((-sum(values), start, positions))

    active = set(protected_positions)
    for _, _, positions in sorted(candidates):
        if remaining <= 0:
            break
        take = min(remaining, len(positions))
        active.update(positions[:take])
        remaining -= take
    if remaining:
        raise OracleContractError("global fixed-chunk target exceeds eligible context")

    active_positions = tuple(sorted(active))
    retention = []
    for segment in ordered:
        positions = tuple(
            position
            for position in range(segment.start, segment.end)
            if position in active
        )
        action = (
            "exact"
            if len(positions) == segment.token_count
            else "sparse"
            if positions
            else "recurrent_only"
        )
        retention.append(
            SegmentRetention(
                segment_id=segment.segment_id,
                action=action,
                positions=positions,
            )
        )
    if len(active_positions) * token_cost != target_context_charged_bytes:
        raise OracleContractError("global fixed-chunk failed its byte target")
    return RetainedPositionPlan(
        context_tokens=context_tokens,
        context_charged_bytes=target_context_charged_bytes,
        sparse_selector=GLOBAL_FIXED_CHUNK_SELECTOR,
        active_positions=active_positions,
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
