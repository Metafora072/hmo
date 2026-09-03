"""Query-conditioned readout of segment contributions in Qwen3.5 DeltaNet state."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from experiments.phase2.e3_v2.alpha_probe import AlphaProbeResult, _eager_attention
from experiments.phase2.e3_v2.context_query import TokenizedPromptSplit
from experiments.phase2.e3_v2.oracle import OracleContractError, SegmentSpec
from experiments.phase2.e3_v2.recurrent_signals import (
    RecurrentSignalError,
    SurvivingSegmentContributions,
    _project_qwen35_context,
    qwen35_l2_normalize,
    surviving_segment_contributions_for_bounds,
)
from experiments.utils.cache_access import get_cache_layer


@dataclass(frozen=True)
class QueryAccessibilityResult:
    context_tokens: int
    query_tokens: int
    recurrent_layer_indices: tuple[int, ...]
    segment_ids: tuple[int, ...]
    read_norm: tuple[float, ...]
    read_share: tuple[float, ...]
    read_alignment: tuple[float, ...]

    def field_dict(self, name: str) -> dict[int, float]:
        if name not in {"read_norm", "read_share", "read_alignment"}:
            raise KeyError(name)
        return dict(zip(self.segment_ids, getattr(self, name)))


@dataclass(frozen=True)
class HybridQueryProbeResult:
    alpha: AlphaProbeResult
    accessibility: QueryAccessibilityResult


@dataclass(frozen=True)
class HybridQueryTokenProbeResult:
    alpha: AlphaProbeResult
    accessibility: QueryAccessibilityResult
    token_attention_mass: tuple[float, ...]


def segment_query_readout(
    normalized_query: torch.Tensor,
    contributions: torch.Tensor,
    cumulative_log_decay: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read each segment contribution with a real recurrent query.

    normalized_query is [batch, token=1, head, key_dim] and contributions is
    [segment, batch, head, key_dim, value_dim].
    """
    if normalized_query.ndim != 4 or normalized_query.shape[1] != 1:
        raise RecurrentSignalError("query readout requires [batch, 1, head, key] query")
    if contributions.ndim != 5:
        raise RecurrentSignalError("segment contributions must be rank five")
    segments, batch, heads, key_dim, _ = contributions.shape
    if (
        normalized_query.shape[0] != batch
        or normalized_query.shape[2] != heads
        or normalized_query.shape[3] != key_dim
        or cumulative_log_decay.shape != (batch, heads)
    ):
        raise RecurrentSignalError("query and contribution dimensions disagree")
    tensors = (normalized_query, contributions, cumulative_log_decay)
    if not all(torch.isfinite(value).all() for value in tensors):
        raise RecurrentSignalError("query readout contains non-finite inputs")

    decay = cumulative_log_decay.clamp(min=-80.0, max=0.0).exp()
    surviving = contributions * decay.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    reads = torch.einsum("bthk,sbhkv->stbhv", normalized_query, surviving)[:, 0]
    norms_by_head = torch.linalg.vector_norm(reads, dim=-1)
    read_norm = norms_by_head.mean(dim=(1, 2))
    read_share = (
        norms_by_head / norms_by_head.sum(dim=0, keepdim=True).clamp_min(eps)
    ).mean(dim=(1, 2))

    total_read = reads.sum(dim=0, keepdim=True)
    denominator = (
        torch.linalg.vector_norm(reads, dim=-1)
        * torch.linalg.vector_norm(total_read, dim=-1)
    ).clamp_min(eps)
    read_alignment = ((reads * total_read).sum(dim=-1) / denominator).mean(dim=(1, 2))
    outputs = (read_norm, read_share, read_alignment)
    if any(value.shape != (segments,) for value in outputs):
        raise RecurrentSignalError("query readout produced invalid segment shape")
    if not all(torch.isfinite(value).all() for value in outputs):
        raise RecurrentSignalError("query readout produced non-finite outputs")
    return outputs


def _project_cached_query(
    module,
    hidden_states: torch.Tensor,
    cache_params,
    *,
    layer_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if hidden_states.ndim != 3 or hidden_states.shape[1] != 1:
        raise RecurrentSignalError("cached query projection requires one token")
    if cache_params is None or not cache_params.has_previous_state(layer_idx):
        raise RecurrentSignalError("cached query projection requires prior recurrent state")
    layer_cache = cache_params.layers[layer_idx]
    conv_state = getattr(layer_cache, "conv_states", None)
    if not isinstance(conv_state, torch.Tensor):
        raise RecurrentSignalError("cached query projection requires convolution state")

    mixed_qkv = module.in_proj_qkv(hidden_states).transpose(1, 2)
    mixed_qkv = module.causal_conv1d_update(
        mixed_qkv,
        conv_state.clone(),
        module.conv1d.weight.squeeze(1),
        module.conv1d.bias,
        module.activation,
    ).transpose(1, 2)
    query, _, _ = torch.split(
        mixed_qkv,
        [module.key_dim, module.key_dim, module.value_dim],
        dim=-1,
    )
    batch = hidden_states.shape[0]
    query = query.reshape(batch, 1, module.num_k_heads, module.head_k_dim)
    if module.num_v_heads % module.num_k_heads != 0:
        raise RecurrentSignalError("value heads must be divisible by key heads")
    if module.num_v_heads > module.num_k_heads:
        query = query.repeat_interleave(module.num_v_heads // module.num_k_heads, dim=2)
    query = qwen35_l2_normalize(query).to(torch.float32)

    log_decay = -module.A_log.float().exp() * F.softplus(
        module.in_proj_a(hidden_states).float() + module.dt_bias
    )
    log_decay = log_decay[:, 0]
    if log_decay.shape != (batch, module.num_v_heads):
        raise RecurrentSignalError("query decay has unexpected head shape")
    return query, log_decay


class Qwen35QueryAccessibilityHookManager:
    """Capture context contributions, then read them during sequential query decode."""

    def __init__(
        self,
        model,
        recurrent_layer_indices: Sequence[int],
        *,
        segments: Sequence[SegmentSpec],
        segment_length: int,
    ):
        self.model = model
        self.recurrent_layer_indices = tuple(int(index) for index in recurrent_layer_indices)
        if (
            not self.recurrent_layer_indices
            or len(set(self.recurrent_layer_indices)) != len(self.recurrent_layer_indices)
        ):
            raise RecurrentSignalError("recurrent layer indices must be non-empty and unique")
        if segment_length <= 0:
            raise RecurrentSignalError("segment_length must be positive")
        self.segment_length = int(segment_length)
        self.segments = tuple(segments)
        if not self.segments:
            raise RecurrentSignalError("query accessibility segments must be non-empty")
        if self.segments[0].start != 0:
            raise RecurrentSignalError("query accessibility segments must start at zero")
        self._handles = []
        self._contributions: dict[int, SurvivingSegmentContributions] = {}
        self._cumulative_decay: dict[int, torch.Tensor] = {}
        self._readouts: dict[int, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {}

    def _get_module(self, layer_idx: int):
        return self.model.model.layers[layer_idx].linear_attn

    def attach(self) -> None:
        self.remove()
        for layer_idx in self.recurrent_layer_indices:
            handle = self._get_module(layer_idx).register_forward_pre_hook(
                _Qwen35QueryAccessibilityHook(self, layer_idx),
                with_kwargs=True,
            )
            self._handles.append(handle)

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def remove(self) -> None:
        self.detach()
        self._contributions.clear()
        self._cumulative_decay.clear()
        self._readouts.clear()

    def finalize(
        self,
        *,
        segments: Sequence[SegmentSpec],
        query_tokens: int,
    ) -> QueryAccessibilityResult:
        self.detach()
        expected = set(self.recurrent_layer_indices)
        if set(self._contributions) != expected or set(self._readouts) != expected:
            raise RecurrentSignalError("query accessibility capture is incomplete")
        segment_tuple = tuple(segments)
        expected_bounds = tuple((segment.start, segment.end) for segment in segment_tuple)
        layer_values = []
        for layer_idx in self.recurrent_layer_indices:
            contribution = self._contributions[layer_idx]
            bounds = tuple(zip(contribution.segment_starts, contribution.segment_ends))
            if bounds != expected_bounds:
                raise RecurrentSignalError("query accessibility segment boundaries disagree")
            readouts = self._readouts[layer_idx]
            if len(readouts) != query_tokens:
                raise RecurrentSignalError("query accessibility token count disagrees")
            layer_values.append(
                tuple(
                    torch.stack([item[field] for item in readouts]).mean(dim=0)
                    for field in range(3)
                )
            )
        aggregated = tuple(
            torch.stack([values[field] for values in layer_values]).mean(dim=0)
            for field in range(3)
        )
        return QueryAccessibilityResult(
            context_tokens=segment_tuple[-1].end,
            query_tokens=query_tokens,
            recurrent_layer_indices=self.recurrent_layer_indices,
            segment_ids=tuple(segment.segment_id for segment in segment_tuple),
            read_norm=tuple(float(value) for value in aggregated[0].tolist()),
            read_share=tuple(float(value) for value in aggregated[1].tolist()),
            read_alignment=tuple(float(value) for value in aggregated[2].tolist()),
        )

    def _capture_context(self, module, hidden_states: torch.Tensor, layer_idx: int) -> None:
        if layer_idx in self._contributions:
            raise RecurrentSignalError(f"recurrent layer {layer_idx} captured context twice")
        trace = _project_qwen35_context(module, hidden_states)
        self._contributions[layer_idx] = surviving_segment_contributions_for_bounds(
            trace,
            segment_starts=[segment.start for segment in self.segments],
            segment_ends=[segment.end for segment in self.segments],
            segment_length=self.segment_length,
        )
        self._readouts[layer_idx] = []

    def _capture_query(self, module, hidden_states, cache_params, layer_idx: int) -> None:
        if layer_idx not in self._contributions:
            raise RecurrentSignalError("query arrived before context contribution capture")
        query, log_decay = _project_cached_query(
            module,
            hidden_states,
            cache_params,
            layer_idx=layer_idx,
        )
        cumulative = self._cumulative_decay.get(layer_idx)
        cumulative = log_decay if cumulative is None else cumulative + log_decay
        self._cumulative_decay[layer_idx] = cumulative
        self._readouts[layer_idx].append(
            segment_query_readout(
                query,
                self._contributions[layer_idx].values,
                cumulative,
            )
        )


class _Qwen35QueryAccessibilityHook:
    def __init__(self, manager: Qwen35QueryAccessibilityHookManager, layer_idx: int):
        self.manager = manager
        self.layer_idx = layer_idx

    def __call__(self, module, args, kwargs):
        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None and args:
            hidden_states = args[0]
        if hidden_states is None or hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
            raise RecurrentSignalError("query accessibility requires one unpadded sample")
        cache = kwargs.get("cache_params")
        has_previous = (
            cache is not None
            and hasattr(cache, "has_previous_state")
            and cache.has_previous_state(self.layer_idx)
        )
        with torch.no_grad():
            if hidden_states.shape[1] == 1 and has_previous:
                self.manager._capture_query(module, hidden_states, cache, self.layer_idx)
            elif not has_previous and hidden_states.shape[1] > 1:
                self.manager._capture_context(module, hidden_states, self.layer_idx)
            else:
                raise RecurrentSignalError("unexpected DeltaNet context/query execution path")


@torch.no_grad()
def collect_hybrid_query_token_probe(
    model,
    prompt: TokenizedPromptSplit,
    *,
    attention_layer_indices: Sequence[int],
    recurrent_layer_indices: Sequence[int],
    segments: Sequence[SegmentSpec],
    segment_length: int,
) -> HybridQueryTokenProbeResult:
    """Collect corrected segment and token attention plus recurrent accessibility."""
    attention_indices = tuple(int(index) for index in attention_layer_indices)
    recurrent_indices = tuple(int(index) for index in recurrent_layer_indices)
    segment_tuple = tuple(segments)
    if not attention_indices or not recurrent_indices or not segment_tuple:
        raise OracleContractError("hybrid query probe indices and segments must be non-empty")
    if segment_tuple[0].start != 0 or segment_tuple[-1].end != prompt.context_tokens:
        raise OracleContractError("hybrid query probe segments do not cover context")

    device = model.device
    base_model = getattr(model, "model", model)
    manager = Qwen35QueryAccessibilityHookManager(
        model,
        recurrent_indices,
        segments=segment_tuple,
        segment_length=segment_length,
    )
    token_attention = []
    manager.attach()
    try:
        context_outputs = base_model(
            prompt.context_ids.to(device),
            use_cache=True,
            return_dict=True,
        )
        private_cache = context_outputs.past_key_values
        for layer_idx in attention_indices:
            layer = get_cache_layer(private_cache, layer_idx)
            if not layer.has_kv() or layer.keys.shape[-2] != prompt.context_tokens:
                raise OracleContractError("hybrid query probe lacks Full-KV context")

        with _eager_attention(model, base_model):
            query_ids = prompt.query_ids.to(device)
            for query_offset in range(prompt.query_tokens):
                logical_position = torch.tensor(
                    [[prompt.context_tokens + query_offset]],
                    device=device,
                    dtype=torch.long,
                )
                cache_position = torch.tensor(
                    [prompt.context_tokens + query_offset],
                    device=device,
                    dtype=torch.long,
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
                attentions = outputs.attentions
                if attentions is None:
                    raise OracleContractError("hybrid query probe produced no attention")
                indexed = (
                    tuple(zip(attention_indices, attentions))
                    if len(attentions) == len(attention_indices)
                    else tuple((idx, attentions[idx]) for idx in attention_indices)
                )
                for layer_idx, weights in indexed:
                    if weights is None or weights.ndim != 4 or weights.shape[-2] != 1:
                        raise OracleContractError(
                            f"hybrid query attention layer {layer_idx} is invalid"
                        )
                    token_attention.append(
                        weights[..., : prompt.context_tokens]
                        .to(torch.float32)
                        .mean(dim=(0, 1, 2))
                    )
        accessibility = manager.finalize(
            segments=segment_tuple,
            query_tokens=prompt.query_tokens,
        )
    finally:
        manager.detach()

    token_mass = torch.stack(token_attention).mean(dim=0)
    if (
        token_mass.shape != (prompt.context_tokens,)
        or not torch.isfinite(token_mass).all()
        or torch.any(token_mass < 0)
    ):
        raise OracleContractError("hybrid query token attention mass is invalid")
    segment_mass = tuple(
        float(token_mass[segment.start : segment.end].sum().item())
        for segment in segment_tuple
    )
    alpha = AlphaProbeResult(
        context_tokens=prompt.context_tokens,
        query_tokens=prompt.query_tokens,
        attention_layer_indices=attention_indices,
        segment_ids=tuple(segment.segment_id for segment in segment_tuple),
        attention_mass=segment_mass,
    )
    return HybridQueryTokenProbeResult(
        alpha=alpha,
        accessibility=accessibility,
        token_attention_mass=tuple(float(value) for value in token_mass.tolist()),
    )


@torch.no_grad()
def collect_hybrid_query_probe(
    model,
    prompt: TokenizedPromptSplit,
    *,
    attention_layer_indices: Sequence[int],
    recurrent_layer_indices: Sequence[int],
    segments: Sequence[SegmentSpec],
    segment_length: int,
) -> HybridQueryProbeResult:
    """Preserve the frozen segment-level probe API and behavior."""
    result = collect_hybrid_query_token_probe(
        model,
        prompt,
        attention_layer_indices=attention_layer_indices,
        recurrent_layer_indices=recurrent_layer_indices,
        segments=segments,
        segment_length=segment_length,
    )
    return HybridQueryProbeResult(
        alpha=result.alpha,
        accessibility=result.accessibility,
    )
