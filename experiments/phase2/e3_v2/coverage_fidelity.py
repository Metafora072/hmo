"""Deterministic byte-accounted allocator for coverage-fidelity HMO."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

from experiments.phase2.e3_v2.oracle import SegmentSpec


ALLOCATION_SCHEMA = "hmo.coverage_fidelity.allocator.v1"


class CoverageFidelityError(ValueError):
    """Raised when allocation inputs cannot satisfy the byte contract."""


@dataclass(frozen=True)
class SegmentAllocation:
    segment_id: int
    action: str
    retained_tokens: int
    charged_bytes: int
    attention_rank: float
    accessibility_rank: float


@dataclass(frozen=True)
class CoverageFidelityPlan:
    middle_kv_fraction: float
    sparse_width: int
    use_accessibility: bool
    enable_exact_upgrades: bool
    protected_kv_bytes: int
    eligible_full_kv_bytes: int
    middle_budget_limit_bytes: int
    total_budget_limit_bytes: int
    middle_charged_bytes: int
    total_charged_bytes: int
    residual_middle_bytes: int
    allocations: tuple[SegmentAllocation, ...]

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "allocations": [asdict(item) for item in self.allocations],
        }


def _rank01(values: Sequence[float]) -> list[float]:
    """Return stable average ranks in [0, 1], preserving ties."""
    if not values:
        return []
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2.0 / max(len(order) - 1, 1)
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def _validate_signal(name: str, signal: Mapping[int, float], ids: tuple[int, ...]) -> None:
    if set(signal) != set(ids):
        raise CoverageFidelityError(
            f"{name} keys must exactly match eligible segment ids {ids}"
        )
    for segment_id, value in signal.items():
        if not math.isfinite(float(value)):
            raise CoverageFidelityError(
                f"{name}[{segment_id}] must be finite, got {value!r}"
            )


def allocate_coverage_fidelity(
    attention: Mapping[int, float],
    accessibility: Mapping[int, float],
    segments: Sequence[SegmentSpec],
    *,
    middle_kv_fraction: float,
    sparse_width: int,
    use_accessibility: bool = True,
    enable_exact_upgrades: bool = True,
) -> CoverageFidelityPlan:
    """Allocate Recurrent-only, Sparse, and Exact actions under one byte cap.

    Coverage is assigned before fidelity. Signals are rank-normalized within the
    sample. Protected boundary segments remain Exact and are charged separately.
    """
    if not 0.0 < middle_kv_fraction < 1.0:
        raise CoverageFidelityError("middle_kv_fraction must lie in (0, 1)")
    if sparse_width <= 0:
        raise CoverageFidelityError("sparse_width must be positive")
    if not segments:
        raise CoverageFidelityError("segments cannot be empty")

    ordered = tuple(sorted(segments, key=lambda item: item.segment_id))
    ids = tuple(item.segment_id for item in ordered)
    if len(set(ids)) != len(ids):
        raise CoverageFidelityError("segment ids must be unique")
    unsupported = [item.segment_id for item in ordered if not item.eligible and not item.protected]
    if unsupported:
        raise CoverageFidelityError(
            "partial unprotected segments have no allocation policy: "
            + ", ".join(map(str, unsupported))
        )

    eligible = tuple(item for item in ordered if item.eligible)
    eligible_ids = tuple(item.segment_id for item in eligible)
    _validate_signal("attention", attention, eligible_ids)
    _validate_signal("accessibility", accessibility, eligible_ids)

    units: dict[int, int] = {}
    for item in ordered:
        if item.token_count <= 0 or item.kv_bytes <= 0:
            raise CoverageFidelityError("segment token and byte counts must be positive")
        if item.kv_bytes % item.token_count:
            raise CoverageFidelityError(
                f"segment {item.segment_id} KV bytes are not token-divisible"
            )
        units[item.segment_id] = item.kv_bytes // item.token_count
    eligible_units = {units[item.segment_id] for item in eligible}
    if len(eligible_units) > 1:
        raise CoverageFidelityError(
            "eligible segments must share one exact per-token KV byte cost"
        )
    if len({item.token_count for item in eligible}) > 1:
        raise CoverageFidelityError(
            "eligible segments must share one full segment token count"
        )
    if any(sparse_width >= item.token_count for item in eligible):
        raise CoverageFidelityError(
            "sparse_width must be smaller than every eligible segment"
        )

    attention_values = [float(attention[item.segment_id]) for item in eligible]
    accessibility_values = [float(accessibility[item.segment_id]) for item in eligible]
    attention_ranks = dict(zip(eligible_ids, _rank01(attention_values)))
    accessibility_ranks = dict(zip(eligible_ids, _rank01(accessibility_values)))

    protected_bytes = sum(item.kv_bytes for item in ordered if item.protected)
    eligible_full_bytes = sum(item.kv_bytes for item in eligible)
    middle_limit = math.floor(eligible_full_bytes * middle_kv_fraction)
    total_limit = protected_bytes + middle_limit
    retained = {item.segment_id: 0 for item in eligible}
    charged = 0

    sparse_costs = {
        item.segment_id: sparse_width * units[item.segment_id] for item in eligible
    }
    all_sparse_cost = sum(sparse_costs.values())
    if all_sparse_cost <= middle_limit:
        coverage_order = list(eligible)
    else:
        coverage_order = sorted(
            eligible,
            key=lambda item: (
                -attention_ranks[item.segment_id] / sparse_costs[item.segment_id],
                item.segment_id,
            ),
        )
    for item in coverage_order:
        cost = sparse_costs[item.segment_id]
        if charged + cost <= middle_limit:
            retained[item.segment_id] = sparse_width
            charged += cost

    if enable_exact_upgrades:
        covered = [item for item in eligible if retained[item.segment_id] > 0]
        fidelity_order = sorted(
            covered,
            key=lambda item: (
                -attention_ranks[item.segment_id]
                * (
                    1.0 - accessibility_ranks[item.segment_id]
                    if use_accessibility
                    else 1.0
                )
                / (item.kv_bytes - sparse_costs[item.segment_id]),
                item.segment_id,
            ),
        )
        for item in fidelity_order:
            segment_id = item.segment_id
            increment = item.kv_bytes - retained[segment_id] * units[segment_id]
            if charged + increment <= middle_limit:
                retained[segment_id] = item.token_count
                charged += increment

    # Spend segment-granularity slack as additional Sparse token slots.
    residual_order = sorted(
        (item for item in eligible if 0 < retained[item.segment_id] < item.token_count),
        key=lambda item: (-attention_ranks[item.segment_id], item.segment_id),
    )
    while residual_order and middle_limit - charged >= next(iter(eligible_units)):
        assigned = False
        for item in residual_order:
            segment_id = item.segment_id
            unit = units[segment_id]
            if retained[segment_id] >= item.token_count:
                continue
            if charged + unit > middle_limit:
                break
            retained[segment_id] += 1
            charged += unit
            assigned = True
        if not assigned:
            break

    allocations = []
    for item in ordered:
        if item.protected:
            action = "exact"
            retained_tokens = item.token_count
            segment_bytes = item.kv_bytes
            attention_rank = 0.0
            accessibility_rank = 0.0
        else:
            retained_tokens = retained[item.segment_id]
            segment_bytes = retained_tokens * units[item.segment_id]
            action = (
                "recurrent_only"
                if retained_tokens == 0
                else "exact"
                if retained_tokens == item.token_count
                else "sparse"
            )
            attention_rank = attention_ranks[item.segment_id]
            accessibility_rank = accessibility_ranks[item.segment_id]
        allocations.append(
            SegmentAllocation(
                segment_id=item.segment_id,
                action=action,
                retained_tokens=retained_tokens,
                charged_bytes=segment_bytes,
                attention_rank=attention_rank,
                accessibility_rank=accessibility_rank,
            )
        )

    if charged > middle_limit:
        raise AssertionError("allocator exceeded the middle KV byte cap")
    common_unit = next(iter(eligible_units), None)
    if common_unit is not None and middle_limit - charged >= common_unit and residual_order:
        raise AssertionError("allocator left an affordable Sparse token slot unused")
    return CoverageFidelityPlan(
        middle_kv_fraction=middle_kv_fraction,
        sparse_width=sparse_width,
        use_accessibility=use_accessibility,
        enable_exact_upgrades=enable_exact_upgrades,
        protected_kv_bytes=protected_bytes,
        eligible_full_kv_bytes=eligible_full_bytes,
        middle_budget_limit_bytes=middle_limit,
        total_budget_limit_bytes=total_limit,
        middle_charged_bytes=charged,
        total_charged_bytes=protected_bytes + charged,
        residual_middle_bytes=middle_limit - charged,
        allocations=tuple(allocations),
    )
