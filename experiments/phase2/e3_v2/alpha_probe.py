"""P0-D isolated Full-KV query-attention probe for alpha controls."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Sequence

import torch

from experiments.phase2.e3_v2.context_query import TokenizedPromptSplit
from experiments.phase2.e3_v2.oracle import OracleContractError, SegmentSpec
from experiments.utils.cache_access import get_cache_layer


@dataclass(frozen=True)
class AlphaProbeResult:
    context_tokens: int
    query_tokens: int
    attention_layer_indices: tuple[int, ...]
    segment_ids: tuple[int, ...]
    attention_mass: tuple[float, ...]

    def as_dict(self) -> dict[int, float]:
        return dict(zip(self.segment_ids, self.attention_mass))


@contextmanager
def _eager_attention(model, base_model):
    configs = []
    for config in (
        getattr(model, "config", None),
        getattr(getattr(model, "config", None), "text_config", None),
        getattr(base_model, "config", None),
    ):
        if config is not None and all(config is not existing for existing, _ in configs):
            configs.append((config, getattr(config, "_attn_implementation", None)))
    for config, _ in configs:
        config._attn_implementation = "eager"
    try:
        yield
    finally:
        for config, previous in configs:
            config._attn_implementation = previous


@torch.no_grad()
def collect_isolated_query_alpha(
    model,
    prompt: TokenizedPromptSplit,
    *,
    attention_layer_indices: Sequence[int],
    segments: Sequence[SegmentSpec],
) -> AlphaProbeResult:
    """Return query-to-context attention mass while discarding the private cache."""
    layer_indices = tuple(int(index) for index in attention_layer_indices)
    if not layer_indices or len(set(layer_indices)) != len(layer_indices):
        raise OracleContractError("alpha attention layer indices must be non-empty and unique")
    segment_tuple = tuple(segments)
    if not segment_tuple or segment_tuple[0].start != 0:
        raise OracleContractError("alpha segments must cover the context from position zero")
    cursor = 0
    for segment in segment_tuple:
        if segment.start != cursor or segment.end <= segment.start:
            raise OracleContractError("alpha segment boundaries are not contiguous")
        cursor = segment.end
    if cursor != prompt.context_tokens:
        raise OracleContractError("alpha segment catalog does not cover the prompt context")

    device = model.device
    context_ids = prompt.context_ids.to(device)
    query_ids = prompt.query_ids.to(device)
    base_model = getattr(model, "model", model)
    context_outputs = base_model(context_ids, use_cache=True, return_dict=True)
    private_cache = context_outputs.past_key_values
    for layer_index in layer_indices:
        layer = get_cache_layer(private_cache, layer_index)
        if not layer.has_kv() or layer.keys.shape[-2] != prompt.context_tokens:
            raise OracleContractError("isolated alpha cache does not contain Full-KV context")

    logical_positions = torch.arange(
        prompt.context_tokens,
        prompt.context_tokens + prompt.query_tokens,
        device=device,
        dtype=torch.long,
    ).unsqueeze(0)
    resident_positions = torch.arange(
        prompt.context_tokens,
        prompt.context_tokens + prompt.query_tokens,
        device=device,
        dtype=torch.long,
    )
    with _eager_attention(model, base_model):
        query_outputs = base_model(
            query_ids,
            past_key_values=private_cache,
            use_cache=True,
            output_attentions=True,
            position_ids=logical_positions,
            cache_position=resident_positions,
            return_dict=True,
        )

    attentions = getattr(query_outputs, "attentions", None)
    if attentions is None:
        raise OracleContractError("alpha probe produced no attention tensors")
    if len(attentions) == len(layer_indices):
        indexed_attentions = tuple(zip(layer_indices, attentions))
    elif max(layer_indices) < len(attentions):
        indexed_attentions = tuple(
            (layer_index, attentions[layer_index]) for layer_index in layer_indices
        )
    else:
        raise OracleContractError("alpha attention tuple cannot be mapped to model layers")
    token_mass_per_layer = []
    for layer_index, weights in indexed_attentions:
        if weights is None:
            raise OracleContractError(f"alpha attention layer {layer_index} is missing")
        if weights.ndim != 4 or weights.shape[-2] != prompt.query_tokens:
            raise OracleContractError("alpha attention tensor has an unexpected shape")
        if weights.shape[-1] < prompt.context_tokens:
            raise OracleContractError("alpha attention tensor omits context keys")
        context_weights = weights[..., : prompt.context_tokens].to(torch.float32)
        token_mass_per_layer.append(context_weights.mean(dim=(0, 1, 2)))
    token_mass = torch.stack(token_mass_per_layer, dim=0).mean(dim=0)
    if not torch.isfinite(token_mass).all() or torch.any(token_mass < 0):
        raise OracleContractError("alpha attention mass is invalid")

    segment_mass = tuple(
        float(token_mass[segment.start : segment.end].sum().item())
        for segment in segment_tuple
    )
    return AlphaProbeResult(
        context_tokens=prompt.context_tokens,
        query_tokens=prompt.query_tokens,
        attention_layer_indices=layer_indices,
        segment_ids=tuple(segment.segment_id for segment in segment_tuple),
        attention_mass=segment_mass,
    )
