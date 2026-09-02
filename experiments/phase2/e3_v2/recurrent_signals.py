"""P0-C: exact Qwen3.5 DeltaNet traces and frozen recurrent candidates."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F


class RecurrentSignalError(RuntimeError):
    """Raised when recurrent instrumentation cannot satisfy the P0-C contract."""


@dataclass(frozen=True)
class DeltaRuleTrace:
    normalized_keys: torch.Tensor
    delta_residuals: torch.Tensor
    log_decay: torch.Tensor
    final_state: torch.Tensor


@dataclass(frozen=True)
class LayerRecurrentCandidates:
    layer_idx: int
    segment_starts: tuple[int, ...]
    segment_ends: tuple[int, ...]
    partial_segments: tuple[bool, ...]
    delta_update_rms: tuple[float, ...]
    log_survival: tuple[float, ...]
    decay_risk: tuple[float, ...]
    suffix_interference: tuple[float, ...]
    surviving_write_norm: tuple[float, ...]

    @property
    def n_segments(self) -> int:
        return len(self.segment_starts)


@dataclass(frozen=True)
class AggregatedRecurrentCandidates:
    layer_indices: tuple[int, ...]
    segment_starts: tuple[int, ...]
    segment_ends: tuple[int, ...]
    partial_segments: tuple[bool, ...]
    delta_update: tuple[float, ...]
    survival_retention: tuple[float, ...]
    decay_risk: tuple[float, ...]
    suffix_interference: tuple[float, ...]
    surviving_write_norm: tuple[float, ...]


@dataclass(frozen=True)
class SurvivingSegmentContributions:
    """Exact additive decomposition of the final recurrent state by segment."""

    segment_starts: tuple[int, ...]
    segment_ends: tuple[int, ...]
    partial_segments: tuple[bool, ...]
    values: torch.Tensor

    @property
    def n_segments(self) -> int:
        return len(self.segment_starts)


def qwen35_l2_normalize(value: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Match the Qwen3.5/FLA key normalization, including its additive epsilon."""
    inverse_norm = torch.rsqrt((value * value).sum(dim=-1, keepdim=True) + eps)
    return value * inverse_norm


def _validate_delta_inputs(
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    log_decay: torch.Tensor,
    initial_state: torch.Tensor | None,
) -> tuple[int, int, int, int, int]:
    if key.ndim != 4 or value.ndim != 4:
        raise RecurrentSignalError("key and value must have shape [batch, token, head, dim]")
    batch, tokens, heads, key_dim = key.shape
    if value.shape[:3] != (batch, tokens, heads):
        raise RecurrentSignalError("key and value batch/token/head dimensions must agree")
    value_dim = value.shape[-1]
    if beta.shape != (batch, tokens, heads) or log_decay.shape != beta.shape:
        raise RecurrentSignalError("beta and log_decay must have shape [batch, token, head]")
    if tokens == 0 or key_dim == 0 or value_dim == 0:
        raise RecurrentSignalError("delta-rule inputs must be non-empty")
    for name, tensor in (
        ("key", key),
        ("value", value),
        ("beta", beta),
        ("log_decay", log_decay),
    ):
        if not torch.isfinite(tensor).all():
            raise RecurrentSignalError(f"{name} contains non-finite values")
    if torch.any(beta < 0) or torch.any(beta > 1):
        raise RecurrentSignalError("beta must lie in [0, 1]")
    if torch.any(log_decay > 1e-6):
        raise RecurrentSignalError("Qwen3.5 log_decay g must be non-positive")
    if initial_state is not None and initial_state.shape != (
        batch,
        heads,
        key_dim,
        value_dim,
    ):
        raise RecurrentSignalError("initial_state has the wrong shape")
    return batch, tokens, heads, key_dim, value_dim


def sequential_gated_delta_trace(
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    log_decay: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
) -> DeltaRuleTrace:
    """Transparent token-wise reference matching Qwen3.5's recurrent fallback."""
    batch, tokens, heads, key_dim, value_dim = _validate_delta_inputs(
        key, value, beta, log_decay, initial_state
    )
    normalized_key = qwen35_l2_normalize(key).to(torch.float32)
    value_f = value.to(torch.float32)
    beta_f = beta.to(torch.float32)
    log_decay_f = log_decay.to(torch.float32)
    state = (
        torch.zeros(
            batch,
            heads,
            key_dim,
            value_dim,
            device=key.device,
            dtype=torch.float32,
        )
        if initial_state is None
        else initial_state.to(device=key.device, dtype=torch.float32).clone()
    )

    deltas = []
    for token_index in range(tokens):
        key_t = normalized_key[:, token_index]
        value_t = value_f[:, token_index]
        beta_t = beta_f[:, token_index].unsqueeze(-1)
        decay_t = log_decay_f[:, token_index].exp().unsqueeze(-1).unsqueeze(-1)
        state = state * decay_t
        prediction = (state * key_t.unsqueeze(-1)).sum(dim=-2)
        delta = (value_t - prediction) * beta_t
        state = state + key_t.unsqueeze(-1) * delta.unsqueeze(-2)
        deltas.append(delta)

    return DeltaRuleTrace(
        normalized_keys=normalized_key,
        delta_residuals=torch.stack(deltas, dim=1),
        log_decay=log_decay_f,
        final_state=state,
    )


def chunk_gated_delta_trace(
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    log_decay: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    chunk_size: int = 64,
) -> DeltaRuleTrace:
    """Recover actual delta residuals with Qwen3.5's chunk-WY recurrence."""
    if chunk_size <= 0:
        raise RecurrentSignalError("chunk_size must be positive")
    batch, tokens, heads, key_dim, value_dim = _validate_delta_inputs(
        key, value, beta, log_decay, initial_state
    )
    normalized_key = qwen35_l2_normalize(key).to(torch.float32)
    key_bh = normalized_key.transpose(1, 2).contiguous()
    value_bh = value.transpose(1, 2).contiguous().to(torch.float32)
    beta_bh = beta.transpose(1, 2).contiguous().to(torch.float32)
    decay_bh = log_decay.transpose(1, 2).contiguous().to(torch.float32)

    pad_size = (chunk_size - tokens % chunk_size) % chunk_size
    key_bh = F.pad(key_bh, (0, 0, 0, pad_size))
    value_bh = F.pad(value_bh, (0, 0, 0, pad_size))
    beta_bh = F.pad(beta_bh, (0, pad_size))
    decay_bh = F.pad(decay_bh, (0, pad_size))
    total_tokens = tokens + pad_size
    chunks = total_tokens // chunk_size

    value_beta = value_bh * beta_bh.unsqueeze(-1)
    key_beta = key_bh * beta_bh.unsqueeze(-1)
    key_chunks = key_bh.reshape(batch, heads, chunks, chunk_size, key_dim)
    value_beta = value_beta.reshape(batch, heads, chunks, chunk_size, value_dim)
    key_beta = key_beta.reshape(batch, heads, chunks, chunk_size, key_dim)
    cumulative_decay = decay_bh.reshape(batch, heads, chunks, chunk_size).cumsum(dim=-1)

    lower_decay = (
        cumulative_decay.unsqueeze(-1) - cumulative_decay.unsqueeze(-2)
    ).tril().exp().to(torch.float32).tril()
    diagonal_and_future = torch.triu(
        torch.ones(
            chunk_size,
            chunk_size,
            dtype=torch.bool,
            device=key.device,
        ),
        diagonal=0,
    )
    transform = -((key_beta @ key_chunks.transpose(-1, -2)) * lower_decay)
    transform = transform.masked_fill(diagonal_and_future, 0)
    for row_index in range(1, chunk_size):
        row = transform[..., row_index, :row_index].clone()
        previous = transform[..., :row_index, :row_index].clone()
        transform[..., row_index, :row_index] = row + (
            row.unsqueeze(-1) * previous
        ).sum(dim=-2)
    transform = transform + torch.eye(
        chunk_size,
        dtype=transform.dtype,
        device=transform.device,
    )

    effective_value = transform @ value_beta
    cumulative_key = transform @ (
        key_beta * cumulative_decay.exp().unsqueeze(-1)
    )
    state = (
        torch.zeros(
            batch,
            heads,
            key_dim,
            value_dim,
            device=key.device,
            dtype=torch.float32,
        )
        if initial_state is None
        else initial_state.to(device=key.device, dtype=torch.float32).clone()
    )

    delta_chunks = []
    for chunk_index in range(chunks):
        key_chunk = key_chunks[:, :, chunk_index]
        delta_chunk = effective_value[:, :, chunk_index] - (
            cumulative_key[:, :, chunk_index] @ state
        )
        delta_chunks.append(delta_chunk)
        chunk_decay = cumulative_decay[:, :, chunk_index]
        final_decay = chunk_decay[:, :, -1]
        decayed_key = key_chunk * (
            final_decay.unsqueeze(-1) - chunk_decay
        ).exp().unsqueeze(-1)
        state = state * final_decay.exp().unsqueeze(-1).unsqueeze(-1)
        state = state + decayed_key.transpose(-1, -2) @ delta_chunk

    delta = torch.stack(delta_chunks, dim=2)
    delta = delta.reshape(batch, heads, total_tokens, value_dim)
    delta = delta[:, :, :tokens].transpose(1, 2).contiguous()
    return DeltaRuleTrace(
        normalized_keys=normalized_key,
        delta_residuals=delta,
        log_decay=log_decay.to(torch.float32),
        final_state=state,
    )


def summarize_recurrent_trace(
    trace: DeltaRuleTrace,
    *,
    layer_idx: int,
    segment_length: int,
    eps: float = 1e-8,
) -> LayerRecurrentCandidates:
    """Aggregate one exact delta trace with the frozen P0-C segment policy."""
    if segment_length <= 0:
        raise RecurrentSignalError("segment_length must be positive")
    key = trace.normalized_keys
    delta = trace.delta_residuals
    log_decay = trace.log_decay
    if key.ndim != 4 or delta.ndim != 4 or log_decay.ndim != 3:
        raise RecurrentSignalError("trace tensors have invalid ranks")
    if key.shape[:3] != delta.shape[:3] or key.shape[:3] != log_decay.shape:
        raise RecurrentSignalError("trace batch/token/head dimensions do not agree")
    if not torch.isfinite(delta).all() or not torch.isfinite(log_decay).all():
        raise RecurrentSignalError("trace contains non-finite values")

    tokens = key.shape[1]
    minimum_tail = max(segment_length // 4, 1)
    starts = tuple(
        start
        for start in range(0, tokens, segment_length)
        if min(start + segment_length, tokens) - start >= minimum_tail
    )
    if not starts:
        raise RecurrentSignalError("context has no segment meeting the minimum tail length")
    ends = tuple(min(start + segment_length, tokens) for start in starts)
    partial = tuple(end - start < segment_length for start, end in zip(starts, ends))

    reverse_cumulative = torch.flip(
        torch.cumsum(torch.flip(log_decay, dims=(1,)), dim=1),
        dims=(1,),
    )
    log_survival_after_token = reverse_cumulative - log_decay
    survival_weight = log_survival_after_token.clamp(min=-80.0, max=0.0).exp()

    delta_rms = []
    log_survival = []
    decay_risk = []
    write_norm = []
    contributions = []
    for start, end in zip(starts, ends):
        segment_delta = delta[:, start:end]
        delta_rms.append(float(segment_delta.square().mean().sqrt().item()))
        suffix_log = log_decay[:, end:].sum(dim=1).mean()
        suffix_value = float(suffix_log.item())
        log_survival.append(suffix_value)
        decay_risk.append(-suffix_value)

        weighted_delta = segment_delta * survival_weight[:, start:end].unsqueeze(-1)
        contribution = torch.einsum(
            "bthk,bthv->bhkv",
            key[:, start:end],
            weighted_delta,
        )
        contributions.append(contribution)
        write_norm.append(
            float(contribution.square().sum(dim=(-2, -1)).sqrt().mean().item())
        )

    interference = [0.0] * len(contributions)
    later_contribution = torch.zeros_like(contributions[0])
    for segment_index in range(len(contributions) - 1, -1, -1):
        contribution = contributions[segment_index]
        norm_squared = contribution.square().sum(dim=(-2, -1))
        inner_product = (contribution * later_contribution).sum(dim=(-2, -1))
        score = torch.where(
            norm_squared > eps,
            -inner_product / norm_squared.clamp_min(eps),
            torch.zeros_like(inner_product),
        )
        interference[segment_index] = float(score.mean().item())
        later_contribution = later_contribution + contribution

    return LayerRecurrentCandidates(
        layer_idx=int(layer_idx),
        segment_starts=starts,
        segment_ends=ends,
        partial_segments=partial,
        delta_update_rms=tuple(delta_rms),
        log_survival=tuple(log_survival),
        decay_risk=tuple(decay_risk),
        suffix_interference=tuple(interference),
        surviving_write_norm=tuple(write_norm),
    )


def surviving_segment_contributions(
    trace: DeltaRuleTrace,
    *,
    segment_length: int,
) -> SurvivingSegmentContributions:
    """Decompose the final state into suffix-decayed segment contributions.

    The returned tensor has shape ``[segment, batch, head, key_dim, value_dim]``
    and sums to ``trace.final_state`` up to floating-point error.
    """
    if segment_length <= 0:
        raise RecurrentSignalError("segment_length must be positive")
    key = trace.normalized_keys
    delta = trace.delta_residuals
    log_decay = trace.log_decay
    if key.ndim != 4 or delta.ndim != 4 or log_decay.ndim != 3:
        raise RecurrentSignalError("trace tensors have invalid ranks")
    if key.shape[:3] != delta.shape[:3] or key.shape[:3] != log_decay.shape:
        raise RecurrentSignalError("trace batch/token/head dimensions do not agree")
    if not torch.isfinite(delta).all() or not torch.isfinite(log_decay).all():
        raise RecurrentSignalError("trace contains non-finite values")

    tokens = key.shape[1]
    minimum_tail = max(segment_length // 4, 1)
    starts = tuple(
        start
        for start in range(0, tokens, segment_length)
        if min(start + segment_length, tokens) - start >= minimum_tail
    )
    if not starts:
        raise RecurrentSignalError("context has no segment meeting the minimum tail length")
    ends = tuple(min(start + segment_length, tokens) for start in starts)
    partial = tuple(end - start < segment_length for start, end in zip(starts, ends))

    reverse_cumulative = torch.flip(
        torch.cumsum(torch.flip(log_decay, dims=(1,)), dim=1),
        dims=(1,),
    )
    log_survival_after_token = reverse_cumulative - log_decay
    survival_weight = log_survival_after_token.clamp(min=-80.0, max=0.0).exp()
    values = []
    for start, end in zip(starts, ends):
        weighted_delta = (
            delta[:, start:end] * survival_weight[:, start:end].unsqueeze(-1)
        )
        values.append(
            torch.einsum(
                "bthk,bthv->bhkv",
                key[:, start:end],
                weighted_delta,
            )
        )
    stacked = torch.stack(values, dim=0)
    if not torch.isfinite(stacked).all():
        raise RecurrentSignalError("segment contributions contain non-finite values")
    return SurvivingSegmentContributions(
        segment_starts=starts,
        segment_ends=ends,
        partial_segments=partial,
        values=stacked,
    )


def surviving_segment_contributions_for_bounds(
    trace: DeltaRuleTrace,
    *,
    segment_starts: Sequence[int],
    segment_ends: Sequence[int],
    segment_length: int,
) -> SurvivingSegmentContributions:
    """Decompose state using an externally frozen segment catalog."""
    starts = tuple(int(value) for value in segment_starts)
    ends = tuple(int(value) for value in segment_ends)
    if segment_length <= 0 or not starts or len(starts) != len(ends):
        raise RecurrentSignalError("explicit segment boundaries are invalid")
    if starts[0] != 0 or ends[-1] != trace.normalized_keys.shape[1]:
        raise RecurrentSignalError("explicit segments must cover the full context")
    if any(
        start != (0 if index == 0 else ends[index - 1])
        or end <= start
        or end - start > segment_length
        for index, (start, end) in enumerate(zip(starts, ends))
    ):
        raise RecurrentSignalError("explicit segment boundaries must be contiguous")

    key = trace.normalized_keys
    delta = trace.delta_residuals
    log_decay = trace.log_decay
    if key.ndim != 4 or delta.ndim != 4 or log_decay.ndim != 3:
        raise RecurrentSignalError("trace tensors have invalid ranks")
    if key.shape[:3] != delta.shape[:3] or key.shape[:3] != log_decay.shape:
        raise RecurrentSignalError("trace batch/token/head dimensions do not agree")
    if not torch.isfinite(delta).all() or not torch.isfinite(log_decay).all():
        raise RecurrentSignalError("trace contains non-finite values")

    reverse_cumulative = torch.flip(
        torch.cumsum(torch.flip(log_decay, dims=(1,)), dim=1),
        dims=(1,),
    )
    log_survival_after_token = reverse_cumulative - log_decay
    survival_weight = log_survival_after_token.clamp(min=-80.0, max=0.0).exp()
    values = []
    for start, end in zip(starts, ends):
        weighted_delta = (
            delta[:, start:end] * survival_weight[:, start:end].unsqueeze(-1)
        )
        values.append(
            torch.einsum(
                "bthk,bthv->bhkv",
                key[:, start:end],
                weighted_delta,
            )
        )
    stacked = torch.stack(values, dim=0)
    if not torch.isfinite(stacked).all():
        raise RecurrentSignalError("segment contributions contain non-finite values")
    return SurvivingSegmentContributions(
        segment_starts=starts,
        segment_ends=ends,
        partial_segments=tuple(
            end - start < segment_length for start, end in zip(starts, ends)
        ),
        values=stacked,
    )


def aggregate_recurrent_candidates(
    signals_per_layer: Mapping[int, LayerRecurrentCandidates],
) -> AggregatedRecurrentCandidates:
    """Mean-reduce frozen raw candidates across recurrent layers."""
    if not signals_per_layer:
        raise RecurrentSignalError("at least one recurrent layer is required")
    ordered_items = sorted(signals_per_layer.items())
    ordered = [signal for _, signal in ordered_items]
    reference = ordered[0]
    for layer_key, signal in ordered_items:
        if signal.layer_idx != layer_key:
            raise RecurrentSignalError("layer index key does not match signal.layer_idx")
        if (
            signal.segment_starts != reference.segment_starts
            or signal.segment_ends != reference.segment_ends
            or signal.partial_segments != reference.partial_segments
        ):
            raise RecurrentSignalError("recurrent layers disagree on segment boundaries")

    def mean_field(name: str) -> tuple[float, ...]:
        stacked = torch.tensor(
            [getattr(signal, name) for signal in ordered],
            dtype=torch.float64,
        )
        if not torch.isfinite(stacked).all():
            raise RecurrentSignalError(f"candidate field {name} contains non-finite values")
        return tuple(float(value) for value in stacked.mean(dim=0).tolist())

    return AggregatedRecurrentCandidates(
        layer_indices=tuple(signal.layer_idx for signal in ordered),
        segment_starts=reference.segment_starts,
        segment_ends=reference.segment_ends,
        partial_segments=reference.partial_segments,
        delta_update=mean_field("delta_update_rms"),
        survival_retention=mean_field("log_survival"),
        decay_risk=mean_field("decay_risk"),
        suffix_interference=mean_field("suffix_interference"),
        surviving_write_norm=mean_field("surviving_write_norm"),
    )


def _project_qwen35_context(module, hidden_states: torch.Tensor) -> DeltaRuleTrace:
    batch, tokens, _ = hidden_states.shape
    mixed_qkv = module.in_proj_qkv(hidden_states).transpose(1, 2)
    if module.causal_conv1d_fn is not None:
        mixed_qkv = module.causal_conv1d_fn(
            x=mixed_qkv,
            weight=module.conv1d.weight.squeeze(1),
            bias=module.conv1d.bias,
            activation=module.activation,
            seq_idx=None,
        )
    else:
        mixed_qkv = F.silu(module.conv1d(mixed_qkv)[:, :, :tokens])
    mixed_qkv = mixed_qkv.transpose(1, 2)
    _, key, value = torch.split(
        mixed_qkv,
        [module.key_dim, module.key_dim, module.value_dim],
        dim=-1,
    )
    key = key.reshape(batch, tokens, module.num_k_heads, module.head_k_dim)
    value = value.reshape(batch, tokens, module.num_v_heads, module.head_v_dim)
    if module.num_v_heads % module.num_k_heads != 0:
        raise RecurrentSignalError("value heads must be divisible by key heads")
    if module.num_v_heads > module.num_k_heads:
        key = key.repeat_interleave(module.num_v_heads // module.num_k_heads, dim=2)

    beta = module.in_proj_b(hidden_states).sigmoid()
    log_decay = -module.A_log.float().exp() * F.softplus(
        module.in_proj_a(hidden_states).float() + module.dt_bias
    )
    return chunk_gated_delta_trace(key, value, beta, log_decay)


class Qwen35RecurrentCandidateHookManager:
    """Collect exact P0-C candidates during one fresh context prefill."""

    def __init__(
        self,
        model,
        recurrent_layer_indices: Sequence[int],
        *,
        segment_length: int,
    ):
        if segment_length <= 0:
            raise RecurrentSignalError("segment_length must be positive")
        self.model = model
        self.recurrent_layer_indices = tuple(int(index) for index in recurrent_layer_indices)
        if not self.recurrent_layer_indices:
            raise RecurrentSignalError("at least one recurrent layer index is required")
        if len(set(self.recurrent_layer_indices)) != len(self.recurrent_layer_indices):
            raise RecurrentSignalError("recurrent layer indices must be unique")
        self.segment_length = int(segment_length)
        self._handles = []
        self._signals: dict[int, LayerRecurrentCandidates] = {}

    def _get_module(self, layer_idx: int):
        return self.model.model.layers[layer_idx].linear_attn

    def attach(self) -> None:
        self.detach()
        self._signals.clear()
        for layer_idx in self.recurrent_layer_indices:
            hook = _Qwen35CandidateHook(
                layer_idx,
                self.segment_length,
                self._signals,
            )
            handle = self._get_module(layer_idx).register_forward_pre_hook(
                hook,
                with_kwargs=True,
            )
            self._handles.append(handle)

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def clear(self) -> None:
        self._signals.clear()

    def remove(self) -> None:
        self.detach()
        self.clear()

    def finalize_context(self) -> Mapping[int, LayerRecurrentCandidates]:
        """Detach before query processing and validate one capture per layer."""
        self.detach()
        missing = sorted(set(self.recurrent_layer_indices) - set(self._signals))
        extra = sorted(set(self._signals) - set(self.recurrent_layer_indices))
        if missing or extra:
            raise RecurrentSignalError(
                f"incomplete context capture: missing={missing}, extra={extra}"
            )
        return dict(self._signals)

    def get_signals(self) -> Mapping[int, LayerRecurrentCandidates]:
        return dict(self._signals)


class _Qwen35CandidateHook:
    def __init__(self, layer_idx: int, segment_length: int, signals: dict):
        self.layer_idx = layer_idx
        self.segment_length = segment_length
        self.signals = signals

    def __call__(self, module, args, kwargs):
        if self.layer_idx in self.signals:
            raise RecurrentSignalError(
                f"recurrent layer {self.layer_idx} was captured more than once"
            )
        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None and args:
            hidden_states = args[0]
        if hidden_states is None or hidden_states.ndim != 3:
            raise RecurrentSignalError("Qwen3.5 hook requires [batch, token, hidden] states")
        if hidden_states.shape[0] != 1:
            raise RecurrentSignalError("P0-C requires one unpadded sample per context prefill")

        attention_mask = kwargs.get("attention_mask")
        if attention_mask is not None and not torch.all(attention_mask == 1):
            raise RecurrentSignalError("P0-C does not permit padded context instrumentation")
        cache = kwargs.get("cache_params")
        if (
            cache is not None
            and hasattr(cache, "has_previous_state")
            and cache.has_previous_state(self.layer_idx)
        ):
            raise RecurrentSignalError("P0-C hook must run on a fresh context prefill")

        with torch.no_grad():
            trace = _project_qwen35_context(module, hidden_states)
            signal = summarize_recurrent_trace(
                trace,
                layer_idx=self.layer_idx,
                segment_length=self.segment_length,
            )
        self.signals[self.layer_idx] = signal
