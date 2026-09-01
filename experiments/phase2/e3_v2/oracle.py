"""P0-D: deterministic equal-byte oracle planning and cache interventions."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch

from experiments.phase2.e3_v2.context_query import (
    InterventionResult,
    PostInterventionState,
)
from experiments.phase2.e3_v2.protocol import P0D_PROTOCOL_VERSION
from experiments.utils.cache_access import get_cache_layer
from experiments.utils.memory_accounting import (
    get_active_kv_bytes,
    get_segment_kv_bytes,
)


class OracleContractError(RuntimeError):
    """Raised when an oracle plan or arm violates the preregistered contract."""


class OracleManifestError(OracleContractError):
    pass


@dataclass(frozen=True)
class OracleConfig:
    segment_length: int
    middle_kv_fraction: float = 0.10
    protected_prefix_segments: int = 1
    protected_suffix_segments: int = 1
    donors_per_segment: int = 3
    backgrounds_per_pair: int = 3
    position_bins: int = 4
    seed: int = 20260901

    def validate(self) -> None:
        if self.segment_length <= 0:
            raise OracleContractError("segment_length must be positive")
        if not 0 < self.middle_kv_fraction < 1:
            raise OracleContractError("middle_kv_fraction must lie in (0, 1)")
        if self.protected_prefix_segments < 0 or self.protected_suffix_segments < 0:
            raise OracleContractError("protected segment counts cannot be negative")
        if self.donors_per_segment <= 0:
            raise OracleContractError("donors_per_segment must be positive")
        if self.backgrounds_per_pair <= 0:
            raise OracleContractError("backgrounds_per_pair must be positive")
        if self.position_bins <= 1:
            raise OracleContractError("position_bins must be greater than one")


@dataclass(frozen=True)
class SegmentSpec:
    segment_id: int
    start: int
    end: int
    token_count: int
    kv_bytes: int
    protected: bool
    partial: bool
    normalized_position: float
    position_bin: int

    @property
    def eligible(self) -> bool:
        return not self.protected and not self.partial


@dataclass(frozen=True)
class SwapComparison:
    comparison_id: str
    target_segment: int
    donor_segment: int
    background_segments: tuple[int, ...]
    target_exact_middle: tuple[int, ...]
    donor_exact_middle: tuple[int, ...]
    middle_charged_bytes: int
    context_resident_bytes: int


@dataclass(frozen=True)
class OraclePlan:
    protocol_version: str
    sample_id: str
    context_tokens: int
    attention_layer_indices: tuple[int, ...]
    config: OracleConfig
    segments: tuple[SegmentSpec, ...]
    eligible_segment_ids: tuple[int, ...]
    middle_budget_limit_bytes: int
    middle_budget_slots: int
    protected_kv_bytes: int
    total_context_kv_bytes: int
    comparisons: tuple[SwapComparison, ...]

    @property
    def manifest_id(self) -> str:
        payload = json.dumps(
            self._payload(),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _payload(self) -> dict:
        return {
            "protocol_version": self.protocol_version,
            "sample_id": self.sample_id,
            "context_tokens": self.context_tokens,
            "attention_layer_indices": list(self.attention_layer_indices),
            "config": asdict(self.config),
            "segments": [asdict(segment) for segment in self.segments],
            "eligible_segment_ids": list(self.eligible_segment_ids),
            "middle_budget_limit_bytes": self.middle_budget_limit_bytes,
            "middle_budget_slots": self.middle_budget_slots,
            "protected_kv_bytes": self.protected_kv_bytes,
            "total_context_kv_bytes": self.total_context_kv_bytes,
            "comparisons": [
                {
                    **asdict(comparison),
                    "background_segments": list(comparison.background_segments),
                    "target_exact_middle": list(comparison.target_exact_middle),
                    "donor_exact_middle": list(comparison.donor_exact_middle),
                }
                for comparison in self.comparisons
            ],
        }

    def to_dict(self) -> dict:
        return {"manifest_id": self.manifest_id, **self._payload()}

    def comparison(self, comparison_id: str) -> SwapComparison:
        matches = [
            comparison
            for comparison in self.comparisons
            if comparison.comparison_id == comparison_id
        ]
        if len(matches) != 1:
            raise OracleContractError(
                f"expected one comparison {comparison_id!r}, found {len(matches)}"
            )
        return matches[0]

    @classmethod
    def from_dict(cls, raw: Mapping) -> "OraclePlan":
        try:
            config = OracleConfig(**raw["config"])
            segments = tuple(SegmentSpec(**segment) for segment in raw["segments"])
            comparisons = tuple(
                SwapComparison(
                    comparison_id=item["comparison_id"],
                    target_segment=int(item["target_segment"]),
                    donor_segment=int(item["donor_segment"]),
                    background_segments=tuple(item["background_segments"]),
                    target_exact_middle=tuple(item["target_exact_middle"]),
                    donor_exact_middle=tuple(item["donor_exact_middle"]),
                    middle_charged_bytes=int(item["middle_charged_bytes"]),
                    context_resident_bytes=int(item["context_resident_bytes"]),
                )
                for item in raw["comparisons"]
            )
            plan = cls(
                protocol_version=raw["protocol_version"],
                sample_id=raw["sample_id"],
                context_tokens=int(raw["context_tokens"]),
                attention_layer_indices=tuple(raw["attention_layer_indices"]),
                config=config,
                segments=segments,
                eligible_segment_ids=tuple(raw["eligible_segment_ids"]),
                middle_budget_limit_bytes=int(raw["middle_budget_limit_bytes"]),
                middle_budget_slots=int(raw["middle_budget_slots"]),
                protected_kv_bytes=int(raw["protected_kv_bytes"]),
                total_context_kv_bytes=int(raw["total_context_kv_bytes"]),
                comparisons=comparisons,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise OracleManifestError("invalid oracle manifest structure") from exc
        if raw.get("manifest_id") != plan.manifest_id:
            raise OracleManifestError("oracle manifest hash does not match its payload")
        validate_oracle_plan(plan)
        return plan


@dataclass(frozen=True)
class ByteEqualityAudit:
    comparison_id: str
    middle_charged_bytes: int
    context_resident_bytes: int
    post_query_resident_bytes: int
    resident_context_tokens: int


@dataclass(frozen=True)
class ArmQuality:
    mean_gold_logprob: float
    secondary_score: float | None = None


@dataclass(frozen=True)
class PairObservation:
    oracle_manifest_id: str
    sample_id: str
    comparison_id: str
    target_segment: int
    donor_segment: int
    background_segments: tuple[int, ...]
    delta_logprob: float
    delta_secondary: float | None


@dataclass(frozen=True)
class PairAggregate:
    target_segment: int
    donor_segment: int
    background_count: int
    mean_delta_logprob: float
    mean_delta_secondary: float | None


def _stable_digest(*parts: object) -> str:
    text = "|".join(str(part) for part in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_segment_catalog(
    cache,
    attention_layer_indices: Sequence[int],
    *,
    context_tokens: int,
    config: OracleConfig,
) -> tuple[SegmentSpec, ...]:
    """Measure exact per-segment KV bytes from an isolated context cache."""
    config.validate()
    if context_tokens <= 0:
        raise OracleContractError("context_tokens must be positive")
    layer_indices = tuple(int(index) for index in attention_layer_indices)
    if not layer_indices or len(set(layer_indices)) != len(layer_indices):
        raise OracleContractError("attention layer indices must be non-empty and unique")
    for layer_index in layer_indices:
        layer = get_cache_layer(cache, layer_index)
        if not layer.has_kv() or layer.keys.shape[-2] != context_tokens:
            raise OracleContractError(
                f"attention layer {layer_index} does not contain the full context KV"
            )

    n_segments = math.ceil(context_tokens / config.segment_length)
    if config.protected_prefix_segments + config.protected_suffix_segments >= n_segments:
        raise OracleContractError("protected regions leave no middle segments")
    protected_ids = set(range(config.protected_prefix_segments))
    protected_ids.update(
        range(n_segments - config.protected_suffix_segments, n_segments)
    )

    segments = []
    for segment_id in range(n_segments):
        start = segment_id * config.segment_length
        end = min(start + config.segment_length, context_tokens)
        token_count = end - start
        position = ((start + end) / 2.0) / context_tokens
        position_bin = min(int(position * config.position_bins), config.position_bins - 1)
        segments.append(
            SegmentSpec(
                segment_id=segment_id,
                start=start,
                end=end,
                token_count=token_count,
                kv_bytes=get_segment_kv_bytes(
                    cache,
                    list(layer_indices),
                    start,
                    end,
                ),
                protected=segment_id in protected_ids,
                partial=token_count != config.segment_length,
                normalized_position=float(position),
                position_bin=position_bin,
            )
        )
    return tuple(segments)


def _balanced_background(
    candidates: Sequence[SegmentSpec],
    count: int,
    *,
    sample_id: str,
    pair: tuple[int, int],
    attempt: int,
    seed: int,
) -> tuple[int, ...]:
    if count == 0:
        return ()
    by_bin: dict[int, list[SegmentSpec]] = {}
    for segment in candidates:
        by_bin.setdefault(segment.position_bin, []).append(segment)
    for position_bin, members in by_bin.items():
        members.sort(
            key=lambda segment: _stable_digest(
                seed,
                sample_id,
                pair,
                attempt,
                position_bin,
                segment.segment_id,
            )
        )
    bin_order = sorted(
        by_bin,
        key=lambda position_bin: _stable_digest(
            seed,
            sample_id,
            pair,
            attempt,
            "bin",
            position_bin,
        ),
    )
    selected = []
    offsets = {position_bin: 0 for position_bin in bin_order}
    while len(selected) < count:
        progressed = False
        for position_bin in bin_order:
            offset = offsets[position_bin]
            members = by_bin[position_bin]
            if offset < len(members):
                selected.append(members[offset].segment_id)
                offsets[position_bin] += 1
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            raise OracleContractError("not enough segments to build the background")
    return tuple(sorted(selected))


def build_oracle_plan(
    *,
    sample_id: str,
    context_tokens: int,
    attention_layer_indices: Sequence[int],
    segments: Sequence[SegmentSpec],
    config: OracleConfig,
) -> OraclePlan:
    """Plan deterministic donor/background swaps without observing quality labels."""
    config.validate()
    if not sample_id:
        raise OracleContractError("sample_id must be non-empty")
    segment_tuple = tuple(segments)
    eligible = tuple(segment for segment in segment_tuple if segment.eligible)
    if len(eligible) <= config.donors_per_segment:
        raise OracleContractError("not enough eligible segments for the donor policy")
    unit_costs = {segment.kv_bytes for segment in eligible}
    if len(unit_costs) != 1 or next(iter(unit_costs)) <= 0:
        raise OracleContractError("eligible full segments must have one positive KV byte cost")
    unit_cost = next(iter(unit_costs))
    total_middle_bytes = sum(segment.kv_bytes for segment in eligible)
    budget_limit = math.floor(total_middle_bytes * config.middle_kv_fraction)
    budget_slots = budget_limit // unit_cost
    if budget_slots < 1:
        raise OracleContractError("middle byte budget cannot retain one full segment")
    if budget_slots > len(eligible) - 1:
        raise OracleContractError("middle byte budget leaves no segment available for a swap")
    background_size = budget_slots - 1
    if background_size == 0 and config.backgrounds_per_pair > 1:
        raise OracleContractError(
            "multiple backgrounds require a budget of at least two exact middle segments"
        )

    segment_by_id = {segment.segment_id: segment for segment in segment_tuple}
    eligible_ids = tuple(segment.segment_id for segment in eligible)
    donor_pairs: set[tuple[int, int]] = set()
    for target in eligible:
        donors = [segment for segment in eligible if segment.segment_id != target.segment_id]
        donors.sort(
            key=lambda donor: (
                donor.position_bin == target.position_bin,
                _stable_digest(
                    config.seed,
                    sample_id,
                    "donor",
                    target.segment_id,
                    donor.segment_id,
                ),
            )
        )
        for donor in donors[: config.donors_per_segment]:
            donor_pairs.add(tuple(sorted((target.segment_id, donor.segment_id))))

    degree = {segment_id: 0 for segment_id in eligible_ids}
    for left, right in donor_pairs:
        degree[left] += 1
        degree[right] += 1
    if any(value < config.donors_per_segment for value in degree.values()):
        raise OracleContractError("deterministic donor graph does not meet minimum degree")

    protected_bytes = sum(segment.kv_bytes for segment in segment_tuple if segment.protected)
    total_context_bytes = sum(segment.kv_bytes for segment in segment_tuple)
    comparisons = []
    for pair in sorted(donor_pairs):
        candidates = [
            segment
            for segment in eligible
            if segment.segment_id not in pair
        ]
        if background_size > len(candidates):
            raise OracleContractError("not enough background candidates for a donor pair")
        backgrounds: set[tuple[int, ...]] = set()
        for attempt in range(512):
            background = _balanced_background(
                candidates,
                background_size,
                sample_id=sample_id,
                pair=pair,
                attempt=attempt,
                seed=config.seed,
            )
            backgrounds.add(background)
            if len(backgrounds) == config.backgrounds_per_pair:
                break
        if len(backgrounds) != config.backgrounds_per_pair:
            raise OracleContractError("could not construct enough unique backgrounds")

        for background in sorted(backgrounds):
            target_exact = tuple(sorted((*background, pair[0])))
            donor_exact = tuple(sorted((*background, pair[1])))
            charged = sum(segment_by_id[index].kv_bytes for index in target_exact)
            donor_charged = sum(segment_by_id[index].kv_bytes for index in donor_exact)
            if charged != donor_charged or charged != budget_slots * unit_cost:
                raise OracleContractError("planned swap arms are not equal-byte")
            comparison_id = "cmp_" + _stable_digest(
                P0D_PROTOCOL_VERSION,
                sample_id,
                pair,
                background,
                config.seed,
            )[:20]
            comparisons.append(
                SwapComparison(
                    comparison_id=comparison_id,
                    target_segment=pair[0],
                    donor_segment=pair[1],
                    background_segments=background,
                    target_exact_middle=target_exact,
                    donor_exact_middle=donor_exact,
                    middle_charged_bytes=charged,
                    context_resident_bytes=protected_bytes + charged,
                )
            )

    plan = OraclePlan(
        protocol_version=P0D_PROTOCOL_VERSION,
        sample_id=sample_id,
        context_tokens=context_tokens,
        attention_layer_indices=tuple(int(index) for index in attention_layer_indices),
        config=config,
        segments=segment_tuple,
        eligible_segment_ids=eligible_ids,
        middle_budget_limit_bytes=budget_limit,
        middle_budget_slots=budget_slots,
        protected_kv_bytes=protected_bytes,
        total_context_kv_bytes=total_context_bytes,
        comparisons=tuple(sorted(comparisons, key=lambda item: item.comparison_id)),
    )
    validate_oracle_plan(plan)
    return plan


def validate_oracle_plan(plan: OraclePlan) -> None:
    if plan.protocol_version != P0D_PROTOCOL_VERSION:
        raise OracleContractError("unsupported oracle protocol version")
    plan.config.validate()
    if not plan.sample_id or plan.context_tokens <= 0:
        raise OracleContractError("oracle sample and context must be non-empty")
    if (
        not plan.attention_layer_indices
        or len(set(plan.attention_layer_indices)) != len(plan.attention_layer_indices)
        or any(index < 0 for index in plan.attention_layer_indices)
    ):
        raise OracleContractError(
            "attention layer indices must be non-empty, unique and nonnegative"
        )
    if tuple(segment.segment_id for segment in plan.segments) != tuple(range(len(plan.segments))):
        raise OracleContractError("segment IDs must be contiguous and recoverable")
    expected_segment_count = math.ceil(plan.context_tokens / plan.config.segment_length)
    if len(plan.segments) != expected_segment_count:
        raise OracleContractError("segment count disagrees with context and segment length")
    cursor = 0
    for segment in plan.segments:
        if segment.start != cursor or segment.end <= segment.start:
            raise OracleContractError("segment positions do not exactly cover the context")
        expected_end = min(segment.start + plan.config.segment_length, plan.context_tokens)
        expected_position = ((segment.start + segment.end) / 2.0) / plan.context_tokens
        expected_bin = min(
            int(expected_position * plan.config.position_bins),
            plan.config.position_bins - 1,
        )
        expected_protected = (
            segment.segment_id < plan.config.protected_prefix_segments
            or (
                plan.config.protected_suffix_segments > 0
                and segment.segment_id
                >= len(plan.segments) - plan.config.protected_suffix_segments
            )
        )
        if (
            segment.end != expected_end
            or segment.token_count != segment.end - segment.start
            or segment.partial != (segment.token_count != plan.config.segment_length)
            or segment.protected != expected_protected
            or segment.position_bin != expected_bin
            or not math.isclose(segment.normalized_position, expected_position)
            or segment.kv_bytes <= 0
        ):
            raise OracleContractError("segment catalog fields are not recoverable")
        cursor = segment.end
    if cursor != plan.context_tokens:
        raise OracleContractError("segment positions do not cover all context tokens")
    eligible = tuple(segment.segment_id for segment in plan.segments if segment.eligible)
    if eligible != plan.eligible_segment_ids:
        raise OracleContractError("eligible segment IDs disagree with the catalog")
    eligible_costs = {plan.segments[index].kv_bytes for index in eligible}
    if len(eligible_costs) != 1:
        raise OracleContractError("eligible segments do not have an equal byte cost")
    unit_cost = next(iter(eligible_costs))
    total_middle_bytes = sum(plan.segments[index].kv_bytes for index in eligible)
    expected_limit = math.floor(total_middle_bytes * plan.config.middle_kv_fraction)
    if plan.middle_budget_limit_bytes != expected_limit:
        raise OracleContractError("middle byte limit is not recoverable from the catalog")
    if plan.middle_budget_slots != expected_limit // unit_cost:
        raise OracleContractError("middle budget slots are not recoverable from bytes")
    expected_protected_bytes = sum(
        segment.kv_bytes for segment in plan.segments if segment.protected
    )
    if plan.protected_kv_bytes != expected_protected_bytes:
        raise OracleContractError("protected KV bytes disagree with the catalog")
    if plan.total_context_kv_bytes != sum(segment.kv_bytes for segment in plan.segments):
        raise OracleContractError("total context KV bytes disagree with the catalog")
    comparison_ids = [comparison.comparison_id for comparison in plan.comparisons]
    if not comparison_ids or len(comparison_ids) != len(set(comparison_ids)):
        raise OracleContractError("comparison IDs must be unique")
    eligible_set = set(plan.eligible_segment_ids)
    pair_backgrounds: dict[tuple[int, int], set[tuple[int, ...]]] = {}
    for comparison in plan.comparisons:
        target_set = set(comparison.target_exact_middle)
        donor_set = set(comparison.donor_exact_middle)
        background_set = set(comparison.background_segments)
        pair = (comparison.target_segment, comparison.donor_segment)
        if (
            pair[0] >= pair[1]
            or pair[0] not in eligible_set
            or pair[1] not in eligible_set
            or len(background_set) != len(comparison.background_segments)
            or pair[0] in background_set
            or pair[1] in background_set
        ):
            raise OracleContractError("oracle donor/background identity is invalid")
        if not target_set <= eligible_set or not donor_set <= eligible_set:
            raise OracleContractError("oracle arms reference ineligible segments")
        if target_set != background_set | {comparison.target_segment}:
            raise OracleContractError("target arm is not recoverable from R and i")
        if donor_set != background_set | {comparison.donor_segment}:
            raise OracleContractError("donor arm is not recoverable from R and j")
        if len(target_set) != plan.middle_budget_slots or len(donor_set) != plan.middle_budget_slots:
            raise OracleContractError("oracle arm does not use the budget-defined slot count")
        target_bytes = sum(plan.segments[index].kv_bytes for index in target_set)
        donor_bytes = sum(plan.segments[index].kv_bytes for index in donor_set)
        if target_bytes != donor_bytes or target_bytes != comparison.middle_charged_bytes:
            raise OracleContractError("oracle manifest contains unequal byte charges")
        if comparison.middle_charged_bytes > plan.middle_budget_limit_bytes:
            raise OracleContractError("oracle arm exceeds the middle byte budget")
        if comparison.context_resident_bytes != plan.protected_kv_bytes + target_bytes:
            raise OracleContractError("resident context bytes are not recoverable")
        expected_id = "cmp_" + _stable_digest(
            P0D_PROTOCOL_VERSION,
            plan.sample_id,
            pair,
            comparison.background_segments,
            plan.config.seed,
        )[:20]
        if comparison.comparison_id != expected_id:
            raise OracleContractError("comparison ID is not recoverable")
        pair_backgrounds.setdefault(pair, set()).add(comparison.background_segments)
    if any(
        len(backgrounds) != plan.config.backgrounds_per_pair
        for backgrounds in pair_backgrounds.values()
    ):
        raise OracleContractError("donor pairs do not have the required unique backgrounds")
    degree = dict.fromkeys(eligible, 0)
    for left, right in pair_backgrounds:
        degree[left] += 1
        degree[right] += 1
    if any(value < plan.config.donors_per_segment for value in degree.values()):
        raise OracleContractError("manifest donor graph does not meet minimum degree")


def _exact_segment_ids(
    plan: OraclePlan,
    comparison: SwapComparison,
    arm: str,
) -> tuple[int, ...]:
    if arm == "target":
        middle = comparison.target_exact_middle
    elif arm == "donor":
        middle = comparison.donor_exact_middle
    else:
        raise OracleContractError("arm must be 'target' or 'donor'")
    protected = tuple(segment.segment_id for segment in plan.segments if segment.protected)
    return tuple(sorted((*protected, *middle)))


def make_oracle_intervention(
    plan: OraclePlan,
    comparison_id: str,
    arm: str,
):
    """Build a P0-B intervention callback for one recoverable oracle arm."""
    comparison = plan.comparison(comparison_id)
    exact_segment_ids = _exact_segment_ids(plan, comparison, arm)

    def intervene(cache, context_ids: torch.Tensor) -> InterventionResult:
        if context_ids.ndim != 2 or context_ids.shape != (1, plan.context_tokens):
            raise OracleContractError("oracle context tokens do not match the manifest")
        before_bytes = get_active_kv_bytes(cache, list(plan.attention_layer_indices))
        if before_bytes != plan.total_context_kv_bytes:
            raise OracleContractError(
                f"context KV bytes changed: expected {plan.total_context_kv_bytes}, got {before_bytes}"
            )
        selected_segments = [plan.segments[index] for index in exact_segment_ids]
        positions = torch.cat(
            [
                torch.arange(segment.start, segment.end, device=context_ids.device)
                for segment in selected_segments
            ]
        ).to(dtype=torch.long)
        positions = positions.sort().values
        for layer_index in plan.attention_layer_indices:
            layer = get_cache_layer(cache, layer_index)
            if not layer.has_kv() or layer.keys.shape[-2] != plan.context_tokens:
                raise OracleContractError("attention cache no longer matches the planned context")
            layer.keys = layer.keys.index_select(-2, positions)
            layer.values = layer.values.index_select(-2, positions)
        after_bytes = get_active_kv_bytes(cache, list(plan.attention_layer_indices))
        if after_bytes != comparison.context_resident_bytes:
            raise OracleContractError(
                f"resident KV bytes {after_bytes} do not match {comparison.context_resident_bytes}"
            )
        return InterventionResult(
            name=f"oracle_{arm}",
            active_context_positions=positions,
            metadata={
                "oracle_manifest_id": plan.manifest_id,
                "comparison_id": comparison.comparison_id,
                "arm": arm,
                "exact_segment_ids": exact_segment_ids,
                "background_segments": comparison.background_segments,
                "target_segment": comparison.target_segment,
                "donor_segment": comparison.donor_segment,
                "middle_charged_bytes": comparison.middle_charged_bytes,
                "context_resident_bytes": comparison.context_resident_bytes,
            },
        )

    return intervene


def audit_equal_byte_pair(
    target: PostInterventionState,
    donor: PostInterventionState,
    attention_layer_indices: Sequence[int],
) -> ByteEqualityAudit:
    """Fail unless charged and decode-resident bytes are exactly equal."""
    target_meta = target.intervention.metadata
    donor_meta = donor.intervention.metadata
    for field in (
        "oracle_manifest_id",
        "comparison_id",
        "middle_charged_bytes",
        "context_resident_bytes",
    ):
        if target_meta.get(field) != donor_meta.get(field):
            raise OracleContractError(f"oracle arms disagree on {field}")
    if target_meta.get("arm") != "target" or donor_meta.get("arm") != "donor":
        raise OracleContractError("oracle arm roles are invalid")
    target_bytes = get_active_kv_bytes(target.cache, list(attention_layer_indices))
    donor_bytes = get_active_kv_bytes(donor.cache, list(attention_layer_indices))
    if target_bytes != donor_bytes:
        raise OracleContractError("post-query decode-resident KV bytes are unequal")
    if target.active_context_positions.numel() != donor.active_context_positions.numel():
        raise OracleContractError("oracle arms retain different context token counts")
    return ByteEqualityAudit(
        comparison_id=str(target_meta["comparison_id"]),
        middle_charged_bytes=int(target_meta["middle_charged_bytes"]),
        context_resident_bytes=int(target_meta["context_resident_bytes"]),
        post_query_resident_bytes=target_bytes,
        resident_context_tokens=int(target.active_context_positions.numel()),
    )


def ensure_oracle_manifest(path: Path, plan: OraclePlan) -> dict:
    """Create one immutable oracle manifest or verify its exact reuse."""
    validate_oracle_plan(plan)
    expected = plan.to_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            actual = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise OracleManifestError("existing oracle manifest is unreadable") from exc
        if actual != expected:
            raise OracleManifestError("existing oracle manifest does not match the plan")
        OraclePlan.from_dict(actual)
        return actual
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(expected, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError:
        return ensure_oracle_manifest(path, plan)
    return expected


def load_oracle_manifest(path: Path) -> OraclePlan:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OracleManifestError("oracle manifest is unreadable") from exc
    return OraclePlan.from_dict(raw)


def build_pair_observation(
    plan: OraclePlan,
    comparison_id: str,
    target_quality: ArmQuality,
    donor_quality: ArmQuality,
) -> PairObservation:
    comparison = plan.comparison(comparison_id)
    values = (target_quality.mean_gold_logprob, donor_quality.mean_gold_logprob)
    if not all(math.isfinite(value) for value in values):
        raise OracleContractError("primary oracle quality must be finite")
    delta_secondary = None
    if target_quality.secondary_score is not None or donor_quality.secondary_score is not None:
        if target_quality.secondary_score is None or donor_quality.secondary_score is None:
            raise OracleContractError("secondary quality must be present in both arms")
        if not math.isfinite(target_quality.secondary_score) or not math.isfinite(
            donor_quality.secondary_score
        ):
            raise OracleContractError("secondary oracle quality must be finite")
        delta_secondary = target_quality.secondary_score - donor_quality.secondary_score
    return PairObservation(
        oracle_manifest_id=plan.manifest_id,
        sample_id=plan.sample_id,
        comparison_id=comparison.comparison_id,
        target_segment=comparison.target_segment,
        donor_segment=comparison.donor_segment,
        background_segments=comparison.background_segments,
        delta_logprob=target_quality.mean_gold_logprob - donor_quality.mean_gold_logprob,
        delta_secondary=delta_secondary,
    )


def aggregate_pair_observations(
    plan: OraclePlan,
    observations: Iterable[PairObservation],
    *,
    require_complete: bool = True,
) -> tuple[tuple[PairAggregate, ...], dict[int, float]]:
    """Mean backgrounds per pair, then derive mean signed segment utility."""
    observations = tuple(observations)
    by_id = {observation.comparison_id: observation for observation in observations}
    if len(by_id) != len(observations):
        raise OracleContractError("duplicate comparison observations are not allowed")
    expected_ids = {comparison.comparison_id for comparison in plan.comparisons}
    if require_complete and set(by_id) != expected_ids:
        missing = sorted(expected_ids - set(by_id))
        extra = sorted(set(by_id) - expected_ids)
        raise OracleContractError(f"incomplete oracle results: missing={missing}, extra={extra}")
    if not set(by_id) <= expected_ids:
        raise OracleContractError("observation references an unknown comparison")

    grouped: dict[tuple[int, int], list[PairObservation]] = {}
    for observation in observations:
        if observation.oracle_manifest_id != plan.manifest_id or observation.sample_id != plan.sample_id:
            raise OracleContractError("observation provenance does not match the oracle plan")
        comparison = plan.comparison(observation.comparison_id)
        if (
            observation.target_segment != comparison.target_segment
            or observation.donor_segment != comparison.donor_segment
            or observation.background_segments != comparison.background_segments
        ):
            raise OracleContractError("observation cannot be recovered from its comparison")
        grouped.setdefault(
            (comparison.target_segment, comparison.donor_segment),
            [],
        ).append(observation)

    aggregates = []
    signed: dict[int, list[float]] = {segment_id: [] for segment_id in plan.eligible_segment_ids}
    for pair, rows in sorted(grouped.items()):
        delta_logp = sum(row.delta_logprob for row in rows) / len(rows)
        secondary_values = [row.delta_secondary for row in rows]
        if all(value is None for value in secondary_values):
            delta_secondary = None
        elif any(value is None for value in secondary_values):
            raise OracleContractError("secondary labels are incomplete across backgrounds")
        else:
            delta_secondary = sum(float(value) for value in secondary_values) / len(rows)
        aggregates.append(
            PairAggregate(
                target_segment=pair[0],
                donor_segment=pair[1],
                background_count=len(rows),
                mean_delta_logprob=delta_logp,
                mean_delta_secondary=delta_secondary,
            )
        )
        signed[pair[0]].append(delta_logp)
        signed[pair[1]].append(-delta_logp)

    utility = {
        segment_id: sum(values) / len(values)
        for segment_id, values in signed.items()
        if values
    }
    return tuple(aggregates), utility
