"""Attention-only query probe for final HMO and external baselines."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from experiments.phase2.e3_v2.alpha_probe import AlphaProbeResult, _eager_attention
from experiments.phase2.e3_v2.context_query import TokenizedPromptSplit
from experiments.phase2.e3_v2.oracle import OracleContractError, SegmentSpec
from experiments.utils.cache_access import get_cache_layer


@dataclass(frozen=True)
class AttentionTokenProbeResult:
    alpha: AlphaProbeResult
    token_attention_mass: tuple[float, ...]
    layer_token_attention_mass: tuple[tuple[float, ...], ...]

    def layer_scores(self) -> dict[int, tuple[float, ...]]:
        return dict(zip(self.alpha.attention_layer_indices, self.layer_token_attention_mass))


@torch.no_grad()
def collect_attention_token_probe(
    model,
    prompt: TokenizedPromptSplit,
    *,
    attention_layer_indices: Sequence[int],
    segments: Sequence[SegmentSpec],
) -> AttentionTokenProbeResult:
    """Collect aggregate and per-Full-layer query-to-context attention mass."""
    layer_indices = tuple(int(index) for index in attention_layer_indices)
    segment_tuple = tuple(segments)
    if not layer_indices or len(set(layer_indices)) != len(layer_indices):
        raise OracleContractError("attention probe layers must be non-empty and unique")
    if not segment_tuple or segment_tuple[0].start != 0:
        raise OracleContractError("attention probe segments must start at zero")
    cursor = 0
    for segment in segment_tuple:
        if segment.start != cursor or segment.end <= segment.start:
            raise OracleContractError("attention probe segments must be contiguous")
        cursor = segment.end
    if cursor != prompt.context_tokens:
        raise OracleContractError("attention probe segments do not cover the context")

    device = model.device
    base_model = getattr(model, "model", model)
    context_outputs = base_model(
        prompt.context_ids.to(device), use_cache=True, return_dict=True
    )
    private_cache = context_outputs.past_key_values
    for layer_index in layer_indices:
        layer = get_cache_layer(private_cache, layer_index)
        if not layer.has_kv() or layer.keys.shape[-2] != prompt.context_tokens:
            raise OracleContractError("attention probe lacks Full-KV context")

    flat_token_mass = []
    per_layer: dict[int, list[torch.Tensor]] = {index: [] for index in layer_indices}
    with _eager_attention(model, base_model):
        query_ids = prompt.query_ids.to(device)
        for query_offset in range(prompt.query_tokens):
            logical_position = torch.tensor(
                [[prompt.context_tokens + query_offset]], device=device, dtype=torch.long
            )
            cache_position = torch.tensor(
                [prompt.context_tokens + query_offset], device=device, dtype=torch.long
            )
            outputs = base_model(
                query_ids[:, query_offset : query_offset + 1],
                past_key_values=private_cache,
                use_cache=True,
                output_attentions=True,
                position_ids=logical_position,
                cache_position=cache_position,
                return_dict=True,
            )
            private_cache = outputs.past_key_values
            attentions = getattr(outputs, "attentions", None)
            if attentions is None:
                raise OracleContractError("attention probe produced no attention")
            if len(attentions) == len(layer_indices):
                indexed = tuple(zip(layer_indices, attentions))
            elif max(layer_indices) < len(attentions):
                indexed = tuple((index, attentions[index]) for index in layer_indices)
            else:
                raise OracleContractError("attention tuple cannot be mapped to model layers")
            for layer_index, weights in indexed:
                if weights is None or weights.ndim != 4 or weights.shape[-2] != 1:
                    raise OracleContractError(
                        f"attention probe layer {layer_index} is invalid"
                    )
                if weights.shape[-1] < prompt.context_tokens:
                    raise OracleContractError("attention probe omits context keys")
                mass = (
                    weights[..., : prompt.context_tokens]
                    .to(torch.float32)
                    .mean(dim=(0, 1, 2))
                )
                flat_token_mass.append(mass)
                per_layer[layer_index].append(mass)

    token_mass = torch.stack(flat_token_mass).mean(dim=0)
    layer_mass = tuple(
        torch.stack(per_layer[index]).mean(dim=0) for index in layer_indices
    )
    values = (token_mass, *layer_mass)
    if any(
        value.shape != (prompt.context_tokens,)
        or not torch.isfinite(value).all()
        or torch.any(value < 0)
        for value in values
    ):
        raise OracleContractError("attention probe token mass is invalid")
    segment_mass = tuple(
        float(token_mass[segment.start : segment.end].sum().item())
        for segment in segment_tuple
    )
    return AttentionTokenProbeResult(
        alpha=AlphaProbeResult(
            context_tokens=prompt.context_tokens,
            query_tokens=prompt.query_tokens,
            attention_layer_indices=layer_indices,
            segment_ids=tuple(segment.segment_id for segment in segment_tuple),
            attention_mass=segment_mass,
        ),
        token_attention_mass=tuple(float(value) for value in token_mass.tolist()),
        layer_token_attention_mass=tuple(
            tuple(float(value) for value in scores.tolist()) for scores in layer_mass
        ),
    )
