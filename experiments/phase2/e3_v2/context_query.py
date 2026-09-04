"""P0-B: exact context/query splitting and post-intervention scoring."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import torch

from experiments.utils.cache_access import get_cache_layer
from experiments.utils.eval_harness import PromptTextParts, build_prompt_parts
from .protocol import P0B_ANSWER_PREFIX, P0B_EXECUTION_EVENTS, P0B_PROTOCOL_VERSION


class ContextQueryProtocolError(RuntimeError):
    """Base class for fail-closed P0-B contract violations."""


class TokenBoundaryError(ContextQueryProtocolError):
    pass


class CacheContractError(ContextQueryProtocolError):
    pass


class RecurrentStateMutationError(ContextQueryProtocolError):
    pass


class ConsumedStateError(ContextQueryProtocolError):
    pass


@dataclass(frozen=True)
class TokenizedPromptSplit:
    text: PromptTextParts
    context_ids: torch.Tensor
    query_ids: torch.Tensor
    full_ids: torch.Tensor
    split_token_index: int

    @property
    def context_tokens(self) -> int:
        return int(self.context_ids.shape[1])

    @property
    def query_tokens(self) -> int:
        return int(self.query_ids.shape[1])


@dataclass(frozen=True)
class InterventionResult:
    name: str
    active_context_positions: torch.Tensor
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class PostInterventionState:
    cache: object
    first_answer_logits: torch.Tensor
    context_tokens: int
    query_tokens: int
    logical_position: int
    resident_kv_tokens: int
    active_context_positions: torch.Tensor
    intervention: InterventionResult
    events: tuple[str, ...]
    consumed_by: str | None = None


@dataclass(frozen=True)
class GoldAnswerScore:
    mean_logprob: float
    total_logprob: float
    n_tokens: int
    token_logprobs: tuple[float, ...]


@dataclass(frozen=True)
class GeneratedAnswer:
    token_ids: torch.Tensor
    text: str


Intervention = Callable[[object, torch.Tensor], InterventionResult]


def _offset_pairs(offset_mapping: Any) -> list[tuple[int, int]]:
    if isinstance(offset_mapping, torch.Tensor):
        offsets = offset_mapping.detach().cpu().tolist()
    else:
        offsets = offset_mapping
    if (
        len(offsets) == 1
        and offsets
        and offsets[0]
        and isinstance(offsets[0][0], (list, tuple))
    ):
        offsets = offsets[0]
    return [(int(start), int(end)) for start, end in offsets]


def tokenize_prompt_parts(tokenizer, text: PromptTextParts) -> TokenizedPromptSplit:
    """Tokenize once, then split only at a verified token boundary."""
    boundary = len(text.memory_context)
    encoded = tokenizer(
        text.full_prompt,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    full_ids = encoded["input_ids"]
    if full_ids.ndim != 2 or full_ids.shape[0] != 1:
        raise TokenBoundaryError("P0-B currently requires one unpadded sample per arm")

    offsets = _offset_pairs(encoded["offset_mapping"])
    if len(offsets) != full_ids.shape[1]:
        raise TokenBoundaryError("Tokenizer offsets do not align with input_ids")

    split_index = None
    query_started = False
    for index, (start, end) in enumerate(offsets):
        if start == end:
            if query_started and split_index is None:
                split_index = index
            continue
        if start < boundary < end:
            raise TokenBoundaryError(
                f"A tokenizer token crosses the context/query character boundary at {boundary}"
            )
        is_query = start >= boundary
        if is_query:
            query_started = True
            if split_index is None:
                split_index = index
        elif query_started:
            raise TokenBoundaryError("Tokenizer offsets are not monotonic around the boundary")

    if split_index is None:
        raise TokenBoundaryError("Query suffix produced no tokens")
    context_ids = full_ids[:, :split_index]
    query_ids = full_ids[:, split_index:]
    if context_ids.numel() == 0 or query_ids.numel() == 0:
        raise TokenBoundaryError("Both memory context and query suffix must contain tokens")
    if not torch.equal(torch.cat([context_ids, query_ids], dim=1), full_ids):
        raise TokenBoundaryError("Token split does not reconstruct the exact full prompt")

    return TokenizedPromptSplit(
        text=text,
        context_ids=context_ids,
        query_ids=query_ids,
        full_ids=full_ids,
        split_token_index=split_index,
    )


def tokenize_sample_prompt(sample, tokenizer) -> TokenizedPromptSplit:
    return tokenize_prompt_parts(tokenizer, build_prompt_parts(sample, tokenizer))


def tokenize_sample_prompt_aligned(sample, tokenizer) -> tuple[TokenizedPromptSplit, int]:
    """Move a boundary-crossing token wholly into context, then split exactly."""
    parts = build_prompt_parts(sample, tokenizer)
    nominal_boundary = len(parts.memory_context)
    encoded = tokenizer(
        parts.full_prompt,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    aligned_boundary = nominal_boundary
    for start, end in _offset_pairs(encoded["offset_mapping"]):
        if start < nominal_boundary < end:
            aligned_boundary = end
            break
    if aligned_boundary >= len(parts.full_prompt):
        raise TokenBoundaryError("Query vanished during token-boundary alignment")
    aligned = PromptTextParts(
        memory_context=parts.full_prompt[:aligned_boundary],
        query_suffix=parts.full_prompt[aligned_boundary:],
    )
    return tokenize_prompt_parts(tokenizer, aligned), aligned_boundary - nominal_boundary


def tokenize_answer_continuation(
    tokenizer,
    prompt: TokenizedPromptSplit,
    answer: str,
    answer_prefix: str = P0B_ANSWER_PREFIX,
) -> torch.Tensor:
    """Tokenize the gold answer as an exact continuation of the serialized prompt."""
    continuation = PromptTextParts(
        memory_context=prompt.text.full_prompt,
        query_suffix=answer_prefix + answer,
    )
    split = tokenize_prompt_parts(tokenizer, continuation)
    if not torch.equal(split.context_ids, prompt.full_ids):
        raise TokenBoundaryError("Gold continuation changed the already-tokenized prompt prefix")
    return split.query_ids


def _snapshot_recurrent_state(cache, layer_indices: Sequence[int]) -> dict:
    if not hasattr(cache, "layers"):
        raise CacheContractError("Recurrent-state validation requires cache.layers")
    snapshot = {}
    for layer_index in layer_indices:
        layer = cache.layers[layer_index]
        values = {}
        for name in (
            "conv_states",
            "recurrent_states",
            "has_previous_state",
            "is_conv_states_initialized",
            "is_recurrent_states_initialized",
        ):
            value = getattr(layer, name, None)
            values[name] = value.clone() if isinstance(value, torch.Tensor) else value
        snapshot[int(layer_index)] = values
    return snapshot


def _assert_recurrent_state_unchanged(cache, expected: Mapping[int, Mapping[str, Any]]) -> None:
    for layer_index, values in expected.items():
        layer = cache.layers[layer_index]
        for name, before in values.items():
            after = getattr(layer, name, None)
            if isinstance(before, torch.Tensor):
                unchanged = isinstance(after, torch.Tensor) and torch.equal(before, after)
            else:
                unchanged = before == after
            if not unchanged:
                raise RecurrentStateMutationError(
                    f"Attention-KV intervention mutated recurrent layer {layer_index}.{name}"
                )


def _attention_lengths(cache, layer_indices: Sequence[int]) -> dict[int, int]:
    lengths = {}
    for layer_index in layer_indices:
        layer = get_cache_layer(cache, int(layer_index))
        if not layer.has_kv():
            raise CacheContractError(f"Attention layer {layer_index} has no KV state")
        lengths[int(layer_index)] = int(layer.keys.shape[-2])
    if len(set(lengths.values())) > 1:
        raise CacheContractError(f"Attention layers have inconsistent KV lengths: {lengths}")
    return lengths


def _validate_active_positions(positions: torch.Tensor, context_tokens: int) -> torch.Tensor:
    positions = positions.detach().to(dtype=torch.long).reshape(-1)
    if positions.numel() == 0:
        raise CacheContractError("An oracle arm cannot drop every context position")
    if torch.any(positions < 0) or torch.any(positions >= context_tokens):
        raise CacheContractError("Active context positions fall outside the context boundary")
    if positions.numel() > 1 and torch.any(positions[1:] <= positions[:-1]):
        raise CacheContractError("Active context positions must be unique and increasing")
    return positions


def full_kv_intervention(cache, context_ids: torch.Tensor) -> InterventionResult:
    positions = torch.arange(context_ids.shape[1], device=context_ids.device)
    return InterventionResult(name="full_kv", active_context_positions=positions)


def _base_model(model):
    return getattr(model, "model", model)


def _last_logits(model, outputs) -> torch.Tensor:
    hidden = outputs.last_hidden_state[:, -1:, :]
    return model.lm_head(hidden)[:, -1, :]


@torch.no_grad()
def reference_full_prompt_logits(model, full_ids: torch.Tensor) -> torch.Tensor:
    """Run a cache-isolated Full-KV probe and expose no mutable probe state."""
    full_ids = full_ids.to(model.device)
    outputs = _base_model(model)(full_ids, use_cache=True, return_dict=True)
    return _last_logits(model, outputs).detach().clone()


@torch.no_grad()
def run_post_intervention_prompt(
    model,
    prompt: TokenizedPromptSplit,
    *,
    attention_layer_indices: Sequence[int],
    recurrent_layer_indices: Sequence[int],
    intervention: Intervention = full_kv_intervention,
) -> PostInterventionState:
    """Prefill context, intervene, then process query and expose first-answer logits."""
    if not attention_layer_indices or not recurrent_layer_indices:
        raise CacheContractError("P0-B requires both attention and recurrent cache layers")

    context_ids = prompt.context_ids.to(model.device)
    query_ids = prompt.query_ids.to(model.device)
    base_model = _base_model(model)
    events: list[str] = []

    context_outputs = base_model(context_ids, use_cache=True, return_dict=True)
    cache = context_outputs.past_key_values
    events.append("context_prefill_complete")
    pre_lengths = _attention_lengths(cache, attention_layer_indices)
    if next(iter(pre_lengths.values())) != prompt.context_tokens:
        raise CacheContractError(
            f"Context prefill KV length {pre_lengths} does not match {prompt.context_tokens} tokens"
        )

    recurrent_before = _snapshot_recurrent_state(cache, recurrent_layer_indices)
    intervention_result = intervention(cache, context_ids)
    if not isinstance(intervention_result, InterventionResult):
        raise CacheContractError("Intervention must return InterventionResult")
    active_positions = _validate_active_positions(
        intervention_result.active_context_positions,
        prompt.context_tokens,
    )
    _assert_recurrent_state_unchanged(cache, recurrent_before)
    post_lengths = _attention_lengths(cache, attention_layer_indices)
    if next(iter(post_lengths.values())) != active_positions.numel():
        raise CacheContractError(
            "Intervention active-position count does not match resident attention KV"
        )
    events.append("context_kv_intervention_complete")

    query_outputs = None
    for query_offset in range(prompt.query_tokens):
        query_token = query_ids[:, query_offset : query_offset + 1]
        logical_position = torch.tensor(
            [[prompt.context_tokens + query_offset]],
            device=query_ids.device,
            dtype=torch.long,
        )
        resident_position = torch.tensor(
            [active_positions.numel() + query_offset],
            device=query_ids.device,
            dtype=torch.long,
        )
        query_outputs = base_model(
            query_token,
            past_key_values=cache,
            use_cache=True,
            position_ids=logical_position,
            cache_position=resident_position,
            return_dict=True,
        )
        cache = query_outputs.past_key_values
    if query_outputs is None:
        raise CacheContractError("Query suffix unexpectedly contained no tokens")
    events.append("query_suffix_complete")
    query_lengths = _attention_lengths(cache, attention_layer_indices)
    expected_length = int(active_positions.numel()) + prompt.query_tokens
    if next(iter(query_lengths.values())) != expected_length:
        raise CacheContractError(
            f"Post-query KV length {query_lengths} does not match expected {expected_length}"
        )

    first_answer_logits = _last_logits(model, query_outputs)
    events.append("first_answer_logits_ready")
    if tuple(events) != P0B_EXECUTION_EVENTS:
        raise ContextQueryProtocolError(f"Unexpected P0-B event order: {events}")
    return PostInterventionState(
        cache=cache,
        first_answer_logits=first_answer_logits,
        context_tokens=prompt.context_tokens,
        query_tokens=prompt.query_tokens,
        logical_position=prompt.context_tokens + prompt.query_tokens,
        resident_kv_tokens=expected_length,
        active_context_positions=active_positions,
        intervention=intervention_result,
        events=tuple(events),
    )


def _consume(state: PostInterventionState, consumer: str) -> None:
    if state.consumed_by is not None:
        raise ConsumedStateError(
            f"Post-intervention cache already consumed by {state.consumed_by}; re-prefill the arm"
        )
    state.consumed_by = consumer


@torch.no_grad()
def score_gold_answer_logprob(
    model,
    state: PostInterventionState,
    answer_ids: torch.Tensor,
) -> GoldAnswerScore:
    """Teacher-force gold tokens entirely from post-intervention logits."""
    answer_ids = answer_ids.to(model.device)
    if answer_ids.ndim != 2 or answer_ids.shape[0] != 1 or answer_ids.shape[1] == 0:
        raise ContextQueryProtocolError("Gold scoring requires one non-empty answer")
    if state.first_answer_logits.ndim != 2 or state.first_answer_logits.shape[0] != 1:
        raise ContextQueryProtocolError("First-answer logits must have shape [1, vocab]")
    if torch.any(answer_ids < 0) or torch.any(answer_ids >= state.first_answer_logits.shape[-1]):
        raise ContextQueryProtocolError("Gold answer contains a token outside the model vocabulary")
    _consume(state, "gold_logprob")

    cache = state.cache
    next_logits = state.first_answer_logits
    token_logprobs = []
    base_model = _base_model(model)
    for step in range(answer_ids.shape[1]):
        target = answer_ids[:, step]
        value = torch.log_softmax(next_logits, dim=-1).gather(
            -1, target.unsqueeze(-1)
        ).squeeze(-1)
        token_logprobs.append(float(value.item()))
        if step + 1 == answer_ids.shape[1]:
            break

        target_input = target.unsqueeze(-1)
        position_ids = torch.full(
            target_input.shape,
            state.logical_position + step,
            device=target_input.device,
            dtype=torch.long,
        )
        cache_position = torch.tensor(
            [state.resident_kv_tokens + step],
            device=target_input.device,
            dtype=torch.long,
        )
        outputs = base_model(
            target_input,
            past_key_values=cache,
            use_cache=True,
            position_ids=position_ids,
            cache_position=cache_position,
            return_dict=True,
        )
        cache = outputs.past_key_values
        next_logits = _last_logits(model, outputs)

    total = float(sum(token_logprobs))
    return GoldAnswerScore(
        mean_logprob=total / len(token_logprobs),
        total_logprob=total,
        n_tokens=len(token_logprobs),
        token_logprobs=tuple(token_logprobs),
    )


@torch.no_grad()
def generate_greedy(
    model,
    tokenizer,
    state: PostInterventionState,
    max_new_tokens: int,
) -> GeneratedAnswer:
    """Greedy generation seeded only by post-intervention first-answer logits."""
    if max_new_tokens <= 0:
        raise ContextQueryProtocolError("max_new_tokens must be positive")
    if state.first_answer_logits.ndim != 2 or state.first_answer_logits.shape[0] != 1:
        raise ContextQueryProtocolError("First-answer logits must have shape [1, vocab]")
    _consume(state, "greedy_generation")

    cache = state.cache
    next_logits = state.first_answer_logits
    base_model = _base_model(model)
    generated = []
    eos_token_id = tokenizer.eos_token_id
    for step in range(max_new_tokens):
        next_token = next_logits.argmax(dim=-1, keepdim=True)
        generated.append(next_token)
        if eos_token_id is not None and int(next_token.item()) == int(eos_token_id):
            break
        if step + 1 == max_new_tokens:
            break

        position_ids = torch.full(
            next_token.shape,
            state.logical_position + step,
            device=next_token.device,
            dtype=torch.long,
        )
        cache_position = torch.tensor(
            [state.resident_kv_tokens + step],
            device=next_token.device,
            dtype=torch.long,
        )
        outputs = base_model(
            next_token,
            past_key_values=cache,
            use_cache=True,
            position_ids=position_ids,
            cache_position=cache_position,
            return_dict=True,
        )
        cache = outputs.past_key_values
        next_logits = _last_logits(model, outputs)

    token_ids = torch.cat(generated, dim=1)
    return GeneratedAnswer(
        token_ids=token_ids,
        text=tokenizer.decode(token_ids[0], skip_special_tokens=True),
    )
