"""Exact layer-wise free-window allocation under a fixed ChunkKV byte target."""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np
import torch

from experiments.phase2.e3_v2.chunkkv_adapter import ChunkKVPlan
from experiments.phase2.e3_v2.context_query import InterventionResult
from experiments.phase2.e3_v2.coverage_fidelity_cache import RetainedPositionPlan
from experiments.phase2.e3_v2.oracle import OracleContractError, SegmentSpec
from experiments.phase2.e3_v2.query_probe_cache import retained_positions_sha256
from experiments.utils.cache_access import get_cache_layer
from experiments.utils.memory_accounting import get_active_kv_bytes


FREE_WINDOW_SCHEMA = "hmo.free_window_allocator.v1"
LAYER_LOCAL_SCHEMA = "hmo.layer_local_allocator.v1"


@dataclass(frozen=True)
class FixedFragment:
    source_start: int
    positions: tuple[int, ...]


@dataclass(frozen=True)
class LayerWindowPlan:
    layer_index: int
    active_positions: tuple[int, ...]
    window_starts: tuple[int, ...]
    fixed_fragments: tuple[FixedFragment, ...]
    middle_attention_mass: float
    baseline_middle_attention_mass: float | None
    crossed_macro_boundaries: int
    eligible_regions_touched: int
    selector_seconds: float

    def to_dict(self) -> dict:
        return {
            "layer_index": self.layer_index,
            "retained_context_tokens": len(self.active_positions),
            "active_positions_sha256": retained_positions_sha256(
                self.active_positions
            ),
            "window_starts": list(self.window_starts),
            "fixed_fragments": [
                {**asdict(fragment), "positions": list(fragment.positions)}
                for fragment in self.fixed_fragments
            ],
            "middle_attention_mass": self.middle_attention_mass,
            "baseline_middle_attention_mass": self.baseline_middle_attention_mass,
            "crossed_macro_boundaries": self.crossed_macro_boundaries,
            "eligible_regions_touched": self.eligible_regions_touched,
            "selector_seconds": self.selector_seconds,
        }


@dataclass(frozen=True)
class LayerwiseWindowPlan:
    schema_version: str
    method: str
    context_tokens: int
    context_charged_bytes: int
    context_token_kv_bytes: int
    window_width: int
    protected_positions: tuple[int, ...]
    layers: tuple[LayerWindowPlan, ...]

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "method": self.method,
            "context_tokens": self.context_tokens,
            "context_charged_bytes": self.context_charged_bytes,
            "context_token_kv_bytes": self.context_token_kv_bytes,
            "window_width": self.window_width,
            "protected_context_tokens": len(self.protected_positions),
            "protected_positions_sha256": retained_positions_sha256(
                self.protected_positions
            ),
            "layer_position_policy": (
                "independent_per_full_layer_shared_across_kv_heads"
            ),
            "recurrent_state_policy": "unchanged",
            "layers": [layer.to_dict() for layer in self.layers],
        }


def _validated_regions(
    segments: Sequence[SegmentSpec], context_tokens: int
) -> tuple[tuple[SegmentSpec, ...], tuple[int, ...], tuple[int, ...]]:
    ordered = tuple(sorted(segments, key=lambda item: item.segment_id))
    if not ordered or ordered[0].start != 0 or ordered[-1].end != context_tokens:
        raise OracleContractError("free-window segments do not cover the context")
    cursor = 0
    protected = []
    eligible = []
    for segment in ordered:
        if segment.start != cursor or segment.end <= segment.start:
            raise OracleContractError("free-window segments must be contiguous")
        cursor = segment.end
        positions = range(segment.start, segment.end)
        if segment.protected:
            protected.extend(positions)
        elif segment.eligible:
            eligible.extend(positions)
        else:
            raise OracleContractError(
                "free-window allocator cannot use an unprotected partial segment"
            )
    return ordered, tuple(protected), tuple(eligible)


def _validated_scores(scores: Sequence[float], context_tokens: int) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if (
        values.shape != (context_tokens,)
        or not np.isfinite(values).all()
        or np.any(values < 0)
    ):
        raise OracleContractError(
            "free-window scores must be finite, nonnegative, and context aligned"
        )
    return values


def _window_crosses_region(
    start: int, width: int, ordered: Sequence[SegmentSpec]
) -> bool:
    end = start + width - 1
    first = next(segment.segment_id for segment in ordered if segment.start <= start < segment.end)
    last = next(segment.segment_id for segment in ordered if segment.start <= end < segment.end)
    return first != last


def _regions_touched(
    active: Sequence[int], ordered: Sequence[SegmentSpec]
) -> int:
    positions = set(active)
    return sum(
        1
        for segment in ordered
        if segment.eligible
        and any(position in positions for position in range(segment.start, segment.end))
    )


def _selected_chunk_fragments(
    baseline: ChunkKVPlan,
    layer_index: int,
    eligible_positions: tuple[int, ...],
) -> tuple[tuple[FixedFragment, ...], tuple[int, ...]]:
    matches = [layer for layer in baseline.layers if layer.layer_index == layer_index]
    if len(matches) != 1:
        raise OracleContractError("free-window baseline layer is missing or duplicated")
    layer = matches[0]
    chunks = {
        positions[0]: positions
        for offset in range(0, len(eligible_positions), baseline.chunk_size)
        if (positions := eligible_positions[offset : offset + baseline.chunk_size])
    }
    fixed = []
    full_starts = []
    reconstructed = set(baseline.protected_positions)
    for start in layer.selected_chunk_starts:
        positions = chunks.get(start)
        if positions is None:
            raise OracleContractError("ChunkKV selected start is off its source grid")
        reconstructed.update(positions)
        if len(positions) == baseline.chunk_size:
            full_starts.append(start)
        else:
            fixed.append(FixedFragment(source_start=start, positions=positions))
    if layer.partial_chunk_start is not None:
        source = chunks.get(layer.partial_chunk_start)
        if source is None or not 0 < layer.partial_chunk_tokens < len(source):
            raise OracleContractError("ChunkKV partial fragment is invalid")
        positions = source[: layer.partial_chunk_tokens]
        reconstructed.update(positions)
        fixed.append(
            FixedFragment(source_start=layer.partial_chunk_start, positions=positions)
        )
    elif layer.partial_chunk_tokens:
        raise OracleContractError("ChunkKV partial count lacks a source chunk")
    if tuple(sorted(reconstructed)) != layer.active_positions:
        raise OracleContractError("ChunkKV fragment decomposition changed its plan")
    fixed.sort(key=lambda item: (item.source_start, item.positions))
    return tuple(fixed), tuple(sorted(full_starts))


def select_optimal_fixed_width_windows(
    scores: Sequence[float],
    *,
    eligible_positions: Sequence[int],
    fixed_positions: Sequence[int],
    window_width: int,
    window_count: int,
    baseline_starts: Sequence[int] = (),
) -> tuple[tuple[int, ...], float]:
    """Select exactly ``window_count`` disjoint free-start windows.

    The recurrence is cardinality-constrained weighted interval scheduling.
    Its inner prefix maximum is vectorized; the retained backtracking table is
    one boolean per state.
    """
    values = _validated_scores(scores, len(scores))
    eligible = tuple(int(value) for value in eligible_positions)
    fixed = tuple(int(value) for value in fixed_positions)
    if (
        window_width <= 0
        or window_count < 0
        or eligible != tuple(sorted(set(eligible)))
        or fixed != tuple(sorted(set(fixed)))
        or set(fixed) - set(eligible)
    ):
        raise OracleContractError("invalid free-window selection inputs")
    if window_count == 0:
        if baseline_starts:
            raise OracleContractError("zero-window plan cannot have baseline windows")
        return (), 0.0

    allowed = np.zeros(values.size, dtype=np.int8)
    allowed[list(eligible)] = 1
    allowed[list(fixed)] = 0
    prefix = np.concatenate(([0], np.cumsum(allowed, dtype=np.int64)))
    starts = np.flatnonzero(
        prefix[window_width:] - prefix[:-window_width] == window_width
    ).astype(np.int64, copy=False)
    if starts.size == 0:
        raise OracleContractError("free-window plan has no legal window")
    score_prefix = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    weights = score_prefix[starts + window_width] - score_prefix[starts]
    previous = np.searchsorted(starts, starts - window_width, side="right")

    state_count = starts.size + 1
    prior = np.zeros(state_count, dtype=np.float64)
    decisions = np.zeros((window_count + 1, state_count), dtype=np.bool_)
    for selected in range(1, window_count + 1):
        candidates = prior[previous] + weights
        current = np.empty(state_count, dtype=np.float64)
        current[0] = -np.inf
        current[1:] = np.maximum.accumulate(candidates)
        before = np.concatenate(([-np.inf], current[1:-1]))
        decisions[selected, 1:] = candidates > before
        prior = current
    optimum = float(prior[-1])
    if not math.isfinite(optimum):
        raise OracleContractError("free-window target is infeasible")

    selected_starts = []
    selected = window_count
    prefix_count = starts.size
    while selected:
        if prefix_count <= 0:
            raise OracleContractError("free-window backtracking failed")
        if decisions[selected, prefix_count]:
            candidate_index = prefix_count - 1
            selected_starts.append(int(starts[candidate_index]))
            prefix_count = int(previous[candidate_index])
            selected -= 1
        else:
            prefix_count -= 1
    selected_starts.sort()

    baseline = tuple(sorted(int(value) for value in baseline_starts))
    if baseline:
        if len(baseline) != window_count:
            raise OracleContractError("baseline window count differs from target")
        legal = set(int(value) for value in starts.tolist())
        if any(start not in legal for start in baseline) or any(
            left + window_width > right for left, right in zip(baseline, baseline[1:])
        ):
            raise OracleContractError("baseline windows are not feasible")
        baseline_mass = math.fsum(
            float(score_prefix[start + window_width] - score_prefix[start])
            for start in baseline
        )
        tolerance = 1e-12 * max(1.0, abs(optimum), abs(baseline_mass))
        if optimum + tolerance < baseline_mass:
            raise OracleContractError("free-window optimum is below its baseline")
        if abs(optimum - baseline_mass) <= tolerance:
            return baseline, float(baseline_mass)
    return tuple(selected_starts), optimum


def build_free_window_plan(
    segments: Sequence[SegmentSpec],
    layer_token_attention_mass: Mapping[int, Sequence[float]],
    baseline: ChunkKVPlan,
) -> LayerwiseWindowPlan:
    """Expand a concrete ChunkKV plan into exact free-start interval selection."""
    ordered, protected, eligible = _validated_regions(
        segments, baseline.context_tokens
    )
    if protected != baseline.protected_positions or not layer_token_attention_mass:
        raise OracleContractError("free-window inputs disagree with ChunkKV")
    retained_tokens = (
        baseline.context_charged_bytes // baseline.context_token_kv_bytes
    )
    if retained_tokens * baseline.context_token_kv_bytes != baseline.context_charged_bytes:
        raise OracleContractError("free-window baseline bytes are not token divisible")

    layers = []
    for layer_index, raw_scores in sorted(layer_token_attention_mass.items()):
        started = time.perf_counter()
        scores = _validated_scores(raw_scores, baseline.context_tokens)
        fixed_fragments, baseline_starts = _selected_chunk_fragments(
            baseline, int(layer_index), eligible
        )
        fixed_positions = tuple(
            sorted(position for fragment in fixed_fragments for position in fragment.positions)
        )
        middle_tokens = retained_tokens - len(protected)
        if (
            middle_tokens < len(fixed_positions)
            or (middle_tokens - len(fixed_positions)) % baseline.chunk_size
        ):
            raise OracleContractError("free-window fragment remainder changed the budget")
        window_count = (middle_tokens - len(fixed_positions)) // baseline.chunk_size
        if window_count != len(baseline_starts):
            raise OracleContractError("free-window source decomposition lost a full chunk")
        starts, _ = select_optimal_fixed_width_windows(
            scores,
            eligible_positions=eligible,
            fixed_positions=fixed_positions,
            window_width=baseline.chunk_size,
            window_count=window_count,
            baseline_starts=baseline_starts,
        )
        middle = set(fixed_positions)
        for start in starts:
            middle.update(range(start, start + baseline.chunk_size))
        active = tuple(sorted(set(protected) | middle))
        if len(active) != retained_tokens:
            raise OracleContractError("free-window plan changed its token target")
        baseline_middle = set(fixed_positions)
        for start in baseline_starts:
            baseline_middle.update(range(start, start + baseline.chunk_size))
        mass = math.fsum(float(scores[position]) for position in middle)
        baseline_mass = math.fsum(
            float(scores[position]) for position in baseline_middle
        )
        if mass + 1e-12 * max(1.0, abs(mass), abs(baseline_mass)) < baseline_mass:
            raise OracleContractError("free-window retained mass is below ChunkKV")
        layers.append(
            LayerWindowPlan(
                layer_index=int(layer_index),
                active_positions=active,
                window_starts=starts,
                fixed_fragments=fixed_fragments,
                middle_attention_mass=float(mass),
                baseline_middle_attention_mass=float(baseline_mass),
                crossed_macro_boundaries=sum(
                    _window_crosses_region(start, baseline.chunk_size, ordered)
                    for start in starts
                ),
                eligible_regions_touched=_regions_touched(middle, ordered),
                selector_seconds=time.perf_counter() - started,
            )
        )
    if tuple(layer.layer_index for layer in layers) != tuple(
        layer.layer_index for layer in baseline.layers
    ):
        raise OracleContractError("free-window layer order differs from ChunkKV")
    return LayerwiseWindowPlan(
        schema_version=FREE_WINDOW_SCHEMA,
        method="free_window",
        context_tokens=baseline.context_tokens,
        context_charged_bytes=baseline.context_charged_bytes,
        context_token_kv_bytes=baseline.context_token_kv_bytes,
        window_width=baseline.chunk_size,
        protected_positions=protected,
        layers=tuple(layers),
    )


def build_layer_local_hmo_plan(
    segments: Sequence[SegmentSpec],
    layer_token_attention_mass: Mapping[int, Sequence[float]],
    legacy: RetainedPositionPlan,
    *,
    context_token_kv_bytes: int,
) -> LayerwiseWindowPlan:
    """Keep legacy HMO actions/counts and optimize Sparse placement per layer."""
    ordered, protected, _ = _validated_regions(segments, legacy.context_tokens)
    retention = {item.segment_id: item for item in legacy.segments}
    if set(retention) != {segment.segment_id for segment in ordered}:
        raise OracleContractError("layer-local HMO lacks segment retention entries")
    retained_tokens = len(legacy.active_positions)
    if retained_tokens * context_token_kv_bytes != legacy.context_charged_bytes:
        raise OracleContractError("layer-local HMO bytes are not token aligned")

    layers = []
    for layer_index, raw_scores in sorted(layer_token_attention_mass.items()):
        started = time.perf_counter()
        scores = _validated_scores(raw_scores, legacy.context_tokens)
        active = []
        starts = []
        for segment in ordered:
            source = retention[segment.segment_id]
            count = len(source.positions)
            if source.action in {"exact", "recurrent_only"}:
                positions = source.positions
            elif source.action == "sparse":
                width = min(count, segment.token_count)
                prefix = np.concatenate(
                    ([0.0], np.cumsum(scores[segment.start : segment.end]))
                )
                masses = prefix[width:] - prefix[:-width]
                start = segment.start + int(np.argmax(masses))
                positions = tuple(range(start, start + width))
                starts.append(start)
            else:
                raise OracleContractError("layer-local HMO has an unknown action")
            if len(positions) != count:
                raise OracleContractError("layer-local HMO changed a segment count")
            active.extend(positions)
        active_positions = tuple(active)
        if (
            active_positions != tuple(sorted(set(active_positions)))
            or len(active_positions) != retained_tokens
        ):
            raise OracleContractError("layer-local HMO changed its token target")
        middle = tuple(position for position in active_positions if position not in set(protected))
        layers.append(
            LayerWindowPlan(
                layer_index=int(layer_index),
                active_positions=active_positions,
                window_starts=tuple(starts),
                fixed_fragments=(),
                middle_attention_mass=float(
                    math.fsum(float(scores[position]) for position in middle)
                ),
                baseline_middle_attention_mass=float(
                    math.fsum(
                        float(scores[position])
                        for position in legacy.active_positions
                        if position not in set(protected)
                    )
                ),
                crossed_macro_boundaries=0,
                eligible_regions_touched=_regions_touched(middle, ordered),
                selector_seconds=time.perf_counter() - started,
            )
        )
    return LayerwiseWindowPlan(
        schema_version=LAYER_LOCAL_SCHEMA,
        method="layer_local_hmo",
        context_tokens=legacy.context_tokens,
        context_charged_bytes=legacy.context_charged_bytes,
        context_token_kv_bytes=context_token_kv_bytes,
        window_width=0,
        protected_positions=protected,
        layers=tuple(layers),
    )


def make_layerwise_window_intervention(
    plan: LayerwiseWindowPlan, *, name: str
):
    """Apply equal-length layer-specific positions without changing recurrent state."""
    layer_indices = tuple(layer.layer_index for layer in plan.layers)
    retained_counts = {len(layer.active_positions) for layer in plan.layers}
    if (
        not layer_indices
        or len(set(layer_indices)) != len(layer_indices)
        or len(retained_counts) != 1
    ):
        raise OracleContractError("layerwise intervention plan is inconsistent")

    def intervene(cache, context_ids: torch.Tensor) -> InterventionResult:
        if context_ids.shape != (1, plan.context_tokens):
            raise OracleContractError("layerwise intervention context mismatch")
        for layer_index in layer_indices:
            layer = get_cache_layer(cache, layer_index)
            if not layer.has_kv() or layer.keys.shape[-2] != plan.context_tokens:
                raise OracleContractError("layerwise intervention requires Full-KV")
        before_bytes = get_active_kv_bytes(cache, list(layer_indices))
        expected_full = plan.context_tokens * plan.context_token_kv_bytes
        if before_bytes != expected_full:
            raise OracleContractError("layerwise Full-KV bytes disagree with token cost")
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
            raise OracleContractError("layerwise resident bytes disagree with target")
        reference = torch.tensor(
            plan.layers[0].active_positions,
            device=context_ids.device,
            dtype=torch.long,
        )
        return InterventionResult(
            name=name,
            active_context_positions=reference,
            metadata={
                "schema_version": plan.schema_version,
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
