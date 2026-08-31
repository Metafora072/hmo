"""
Helpers for running the current HMO prototype in experiment scripts.

These utilities keep V1-V4 aligned with the live controller semantics:
  - saturation is collected through the controller's hook path
  - interventions are applied by overriding controller actions
  - prompts/tokenization stay consistent with the shared eval harness
"""
from __future__ import annotations

import types
from contextlib import contextmanager

import numpy as np
import torch

from .eval_harness import build_prompt
from .saturation import compute_segment_saturation
from .cache_access import get_cache_layer
from .kv_ops import execute_refresh, extract_token_skeleton, drop_segment
# rts_runtime no longer needed — token-pruning RTS keeps tokens in-place


def build_input_ids(sample, tokenizer, device, max_length: int | None = None) -> torch.Tensor:
    """Tokenize one EvalSample into a single batch of input ids."""
    prompt = build_prompt(sample, tokenizer)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    return inputs["input_ids"].to(device)


@torch.no_grad()
def collect_sigma(controller, input_ids: torch.Tensor) -> np.ndarray:
    """Collect the controller's current per-segment saturation score."""
    controller.hook_mgr.clear()
    controller.hook_mgr.attach()
    try:
        # Mirror the live controller path as closely as possible.
        # Some hook signals only materialize on the normal prefill route.
        outputs, _, _ = controller._prefill_with_cache_last_logits(input_ids)
        signals = dict(controller.hook_mgr.get_signals())
    finally:
        controller.hook_mgr.remove()
    del outputs

    return compute_segment_saturation(
        signals,
        alpha_rho=controller.hmo.alpha_rho,
        alpha_c=controller.hmo.alpha_c,
        alpha_g=controller.hmo.alpha_g,
        segment_length=controller.hmo.segment_length,
        warmup_tokens=controller.hmo.warmup_tokens,
        input_ids=input_ids,
        repeat_ngram_size=controller.hmo.repeat_ngram_size,
        repeat_threshold=controller.hmo.repeat_threshold,
    )


def protected_segments(n_segments: int) -> set[int]:
    """Segments that the prototype always protects as sinks/recent context.
    Aligned with controller's 1+1 protection (first + last segment)."""
    protected = set(range(min(1, n_segments)))
    protected.update(range(max(0, n_segments - 1), n_segments))
    return protected


def middle_segments(n_segments: int) -> list[int]:
    """Return non-protected segment ids."""
    protected = protected_segments(n_segments)
    return [idx for idx in range(n_segments) if idx not in protected]


def select_triggered_refresh_segments(
    sigma: np.ndarray,
    budget: int,
    threshold: float,
) -> list[int]:
    """Select saturation-triggered refresh targets from middle segments only."""
    candidates = middle_segments(len(sigma))
    if budget <= 0 or not candidates:
        return []

    ranked = sorted(candidates, key=lambda idx: float(sigma[idx]), reverse=True)
    selected = [idx for idx in ranked[:budget] if float(sigma[idx]) > threshold]
    if not selected:
        selected = ranked[:1]
    return selected[:budget]


def select_periodic_segments(candidates: list[int], count: int) -> list[int]:
    """Pick evenly spaced candidates with the requested count."""
    if count <= 0 or not candidates:
        return []
    if count >= len(candidates):
        return list(candidates)

    positions = np.linspace(0, len(candidates) - 1, num=count, dtype=int)
    return [candidates[pos] for pos in positions.tolist()]


def all_kv_actions(n_segments: int) -> dict[int, str]:
    """Force the controller to keep every segment as full KV."""
    return {idx: "KV" for idx in range(n_segments)}


def refresh_only_actions(n_segments: int, refresh_segments: list[int]) -> dict[int, str]:
    """Legacy helper: keep all non-selected segments as KV and mark selected ones for refresh."""
    actions = all_kv_actions(n_segments)
    for idx in refresh_segments:
        if 0 <= idx < n_segments:
            actions[idx] = "refresh"
    return actions


def protected_kv_drop_rest_actions(n_segments: int) -> dict[int, str]:
    """Canonical refresh baseline: keep sinks/recent as KV and drop every other segment."""
    actions = {idx: "drop" for idx in range(n_segments)}
    for idx in protected_segments(n_segments):
        actions[idx] = "KV"
    return actions


def refresh_drop_rest_actions(n_segments: int, refresh_segments: list[int]) -> dict[int, str]:
    """
    Canonical refresh action map for V1/V2:
    - keep only protected sink/recent segments as KV
    - assign selected segments to refresh
    - drop all other middle segments
    """
    actions = protected_kv_drop_rest_actions(n_segments)
    for idx in refresh_segments:
        if 0 <= idx < n_segments and idx not in protected_segments(n_segments):
            actions[idx] = "refresh"
    return actions


def rts_only_actions(n_segments: int, sigma: np.ndarray, keep_ratio: float) -> dict[int, str]:
    """Keep a KV subset and turn the rest of the middle segments into RTS skeletons."""
    actions = {}
    protected = protected_segments(n_segments)
    for idx in protected:
        actions[idx] = "KV"

    middle = [idx for idx in range(n_segments) if idx not in protected]
    ranked = sorted(middle, key=lambda idx: float(sigma[idx]), reverse=True)
    n_keep = int(len(ranked) * keep_ratio)
    keep_set = set(ranked[:n_keep])

    for idx in ranked:
        actions[idx] = "KV" if idx in keep_set else "RTS"

    return actions


@contextmanager
def override_actions(controller, forced_actions: dict[int, str], budget_limit_bytes: int = 0, segment_costs_override: dict | None = None):
    """Temporarily replace the controller policy with a fixed action map."""
    original_decide = controller._decide_actions
    original_build_costs = controller._build_segment_costs

    def _fixed_actions(self, sigma, seq_len, segment_costs, input_ids, alpha=None):
        return dict(forced_actions), int(budget_limit_bytes)

    controller._decide_actions = types.MethodType(_fixed_actions, controller)

    if segment_costs_override is not None:
        def _fixed_costs(self, cache, input_ids, seq_len):
            return segment_costs_override
        controller._build_segment_costs = types.MethodType(_fixed_costs, controller)

    try:
        yield
    finally:
        controller._decide_actions = original_decide
        controller._build_segment_costs = original_build_costs


def run_forced_actions(
    controller,
    input_ids: torch.Tensor,
    forced_actions: dict[int, str],
    max_new_tokens: int,
    budget_limit_bytes: int = 0,
    segment_costs: dict | None = None,
):
    """Run the controller while forcing a fixed per-segment action map."""
    with override_actions(controller, forced_actions, budget_limit_bytes=budget_limit_bytes, segment_costs_override=segment_costs):
        return controller.run(input_ids, max_new_tokens=max_new_tokens)


@torch.no_grad()
def collect_sigma_and_segment_costs(controller, input_ids: torch.Tensor) -> tuple[np.ndarray, dict[int, dict]]:
    """Collect canonical sigma plus per-segment byte costs from one shared prefill pass."""
    controller.hook_mgr.clear()
    controller.hook_mgr.attach()
    try:
        outputs, cache, _ = controller._prefill_with_cache_last_logits(input_ids)
        signals = dict(controller.hook_mgr.get_signals())
    finally:
        controller.hook_mgr.remove()

    sigma = compute_segment_saturation(
        signals,
        alpha_rho=controller.hmo.alpha_rho,
        alpha_c=controller.hmo.alpha_c,
        alpha_g=controller.hmo.alpha_g,
        segment_length=controller.hmo.segment_length,
        warmup_tokens=controller.hmo.warmup_tokens,
        input_ids=input_ids,
        repeat_ngram_size=controller.hmo.repeat_ngram_size,
        repeat_threshold=controller.hmo.repeat_threshold,
    )
    segment_costs = controller._build_segment_costs(cache, input_ids, input_ids.shape[1])
    # Free the prefill cache and intermediates — callers will do their own prefill
    del cache, outputs
    import gc; gc.collect()
    torch.cuda.empty_cache()
    return sigma, segment_costs


@torch.no_grad()
def collect_segment_attention_scores(controller, input_ids: torch.Tensor) -> np.ndarray:
    """
    Collect a per-segment attention fragility score from full-attention layers.

    This delegates to the controller's theory-faithful α collector so that
    Phase-1 and Phase-2 experiments use the same attention fragility signal.
    """
    return controller.collect_segment_attention_scores(input_ids)


def estimate_action_bytes(segment_costs: dict[int, dict], actions: dict[int, str]) -> int:
    """Estimate tracked storage bytes for one fixed action map under the canonical accountant.
    Segments not in actions are treated as KV (matching controller behavior for tail segments)."""
    total = 0
    refresh_shared_counted = False
    for seg_idx in range(len(segment_costs)):
        action = actions.get(seg_idx, "KV")
        costs = segment_costs[seg_idx]
        if action == "KV":
            total += int(costs["kv_bytes"])
        elif action == "refresh":
            total += int(costs["refresh_segment_bytes"])
            if not refresh_shared_counted:
                total += int(costs["shared_refresh_bytes"])
                refresh_shared_counted = True
        elif action == "RTS":
            total += int(costs["rts_bytes"])
    return int(total)


def default_shared_budget_limit(segment_costs: dict[int, dict], keep_ratio: float, n_sigma: int | None = None) -> int:
    """Mirror the controller's default shared byte budget.
    Protected = first 1 + last 1 sigma-covered segment + any trailing segments beyond sigma.
    Middle = everything else within sigma coverage.
    Budget = protected_bytes + keep_ratio * middle_bytes."""
    n_total = len(segment_costs)
    n_sig = n_sigma if n_sigma is not None else n_total

    # Protected: first 1 + last 1 of sigma-covered segments
    protected = set()
    if n_sig >= 1:
        protected.add(0)
    if n_sig >= 2:
        protected.add(n_sig - 1)
    # Trailing segments beyond sigma → also protected as KV
    for i in range(n_sig, n_total):
        protected.add(i)

    protected_bytes = sum(int(segment_costs[idx]["kv_bytes"]) for idx in protected if idx in segment_costs)
    middle_bytes = sum(
        int(segment_costs[idx]["kv_bytes"])
        for idx in range(n_sig)
        if idx not in protected and idx in segment_costs
    )
    return int(protected_bytes + keep_ratio * middle_bytes)


def refresh_drop_rest_budgeted_actions(
    n_segments: int,
    refresh_candidates: list[int],
    segment_costs: dict[int, dict],
    budget_limit_bytes: int,
) -> dict[int, str]:
    """Add refresh segments greedily on top of the canonical protected/drop baseline while respecting a byte cap."""
    actions = protected_kv_drop_rest_actions(n_segments)
    for idx in refresh_candidates:
        if idx in protected_segments(n_segments):
            continue
        trial = dict(actions)
        trial[idx] = "refresh"
        if estimate_action_bytes(segment_costs, trial) <= budget_limit_bytes:
            actions = trial
    return actions


def hmo_periodic_actions(
    n_segments: int,
    segment_costs: dict[int, dict],
    budget_limit_bytes: int,
    refresh_budget: int,
    sigma: np.ndarray | None = None,
) -> dict[int, str]:
    """HMO-periodic: same structure as hmo_full, but refresh targets are
    evenly spaced across middle segments instead of phi-ranked.
    Non-refreshed middle segments get sigma-proportional RTS (mirrors _decide_actions)."""
    middle = middle_segments(n_segments)
    periodic_segs = select_periodic_segments(middle, refresh_budget)

    # Start with protected KV + drop rest
    actions = protected_kv_drop_rest_actions(n_segments)

    # Add periodic refresh segments with hmo_full-consistent budget charging:
    # max(kv_bytes, refresh_segment_bytes) per segment (same as _decide_actions)
    protected = protected_segments(n_segments)
    # Include trailing segments beyond n_segments in protected budget (mirrors _decide_actions)
    all_protected_indices = set(protected)
    for i in range(n_segments, len(segment_costs)):
        all_protected_indices.add(i)
    protected_bytes = sum(int(segment_costs[i]["kv_bytes"]) for i in all_protected_indices if i in segment_costs)
    tracked_bytes = protected_bytes
    refresh_shared_reserved = False

    for idx in periodic_segs:
        if idx in protected:
            continue
        costs = segment_costs[idx]
        refresh_increment = max(int(costs["kv_bytes"]), int(costs["refresh_segment_bytes"]))
        if not refresh_shared_reserved:
            refresh_increment = max(refresh_increment,
                int(costs["refresh_segment_bytes"]) + int(costs["shared_refresh_bytes"]))
        if tracked_bytes + refresh_increment <= budget_limit_bytes:
            actions[idx] = "refresh"
            tracked_bytes += refresh_increment
            refresh_shared_reserved = True

    # Remaining middle segments (not protected, not refreshed) → sigma-proportional RTS
    remaining = [i for i in middle if actions.get(i) == "drop"]
    if remaining and sigma is not None:
        remaining_budget = max(0, budget_limit_bytes - tracked_bytes)

        if remaining_budget > 0:
            first_mid = remaining[0]
            first_len = max(1, segment_costs[first_mid]["end"] - segment_costs[first_mid]["start"])
            bytes_per_token = segment_costs[first_mid]["kv_bytes"] / max(first_len, 1)
            total_affordable = int(remaining_budget / max(bytes_per_token, 1))

            if total_affordable > 0:
                sig_vals = np.array([max(float(sigma[idx]), 0.0) for idx in remaining], dtype=np.float32)
                sig_sum = float(sig_vals.sum())
                if sig_sum < 1e-8:
                    per_seg = [total_affordable // max(len(remaining), 1)] * len(remaining)
                else:
                    per_seg = [int(total_affordable * s / sig_sum) for s in sig_vals]

                seg_lens = [max(1, segment_costs[idx]["end"] - segment_costs[idx]["start"]) for idx in remaining]
                per_seg = [min(n, sl) for n, sl in zip(per_seg, seg_lens)]

                # Redistribute leftover tokens to highest-sigma segments (mirrors _decide_actions)
                used = sum(per_seg)
                leftover = max(0, total_affordable - used)
                ranked = sorted(range(len(remaining)), key=lambda j: float(sig_vals[j]), reverse=True)
                while leftover > 0:
                    progressed = False
                    for j in ranked:
                        if per_seg[j] < seg_lens[j]:
                            per_seg[j] += 1
                            leftover -= 1
                            progressed = True
                            if leftover == 0:
                                break
                    if not progressed:
                        break

                for idx, n_keep, seg_len_i in zip(remaining, per_seg, seg_lens):
                    if n_keep > 0:
                        actions[idx] = "RTS"
                        segment_costs[idx]["rts_n_keep"] = int(n_keep)
                        segment_costs[idx]["rts_bytes"] = int(
                            segment_costs[idx]["kv_bytes"] * n_keep / max(seg_len_i, 1)
                        )

    return actions

    return actions


def rts_budgeted_actions(
    n_segments: int,
    sigma: np.ndarray,
    segment_costs: dict[int, dict],
    budget_limit_bytes: int,
    rts_threshold: float,
) -> dict[int, str]:
    """Build an RTS-only action map under an explicit shared byte cap.

    For V3 validation: all middle segments are assigned RTS. The number of
    tokens kept per segment is computed to fill the budget evenly, so that
    the total token count matches what H2O would keep under the same budget.
    Protected segments (sinks/recent) stay as KV.
    """
    actions = protected_kv_drop_rest_actions(n_segments)
    middle = middle_segments(n_segments)

    if not middle:
        return actions

    # Calculate protected KV bytes
    protected_bytes = estimate_action_bytes(segment_costs, actions)
    remaining_budget = max(0, budget_limit_bytes - protected_bytes)

    if remaining_budget <= 0:
        return actions

    # Calculate bytes per token from the first middle segment
    first_mid = middle[0]
    seg_len = segment_costs[first_mid]["end"] - segment_costs[first_mid]["start"]
    seg_kv_bytes = segment_costs[first_mid]["kv_bytes"]
    bytes_per_token = seg_kv_bytes / max(seg_len, 1)

    # Total tokens we can afford across all middle segments
    total_affordable_tokens = int(remaining_budget / max(bytes_per_token, 1))

    # Distribute tokens across middle segments proportionally to sigma
    # (higher sigma segments get more tokens to preserve more information)
    sigma_vals = np.array([float(sigma[idx]) for idx in middle])
    sigma_sum = sigma_vals.sum()
    if sigma_sum < 1e-8:
        per_seg_tokens = [total_affordable_tokens // max(len(middle), 1)] * len(middle)
    else:
        per_seg_tokens = []
        for s in sigma_vals:
            n = int(total_affordable_tokens * s / sigma_sum)
            per_seg_tokens.append(n)

    # Cap each segment's tokens at its actual length
    for i, idx in enumerate(middle):
        seg_len_i = segment_costs[idx]["end"] - segment_costs[idx]["start"]
        per_seg_tokens[i] = min(per_seg_tokens[i], seg_len_i)

    # Renormalize: ensure total kept tokens does not exceed budget
    total_kept = sum(per_seg_tokens)
    if total_kept > total_affordable_tokens:
        # Scale down proportionally
        scale = total_affordable_tokens / max(total_kept, 1)
        per_seg_tokens = [max(0, int(t * scale)) for t in per_seg_tokens]

    # Assign all middle segments as RTS with their computed n_keep
    for i, idx in enumerate(middle):
        actions[idx] = "RTS"
        segment_costs[idx]["rts_n_keep"] = max(0, per_seg_tokens[i])

    return actions


@torch.no_grad()
def prepare_cache_with_actions(controller, input_ids: torch.Tensor, forced_actions: dict[int, str]):
    """Prepare a cache using the current HMO prototype semantics and a fixed action map."""
    device = input_ids.device
    seq_len = input_ids.shape[1]
    seg_len = controller.hmo.segment_length
    active_positions = torch.arange(seq_len, device=device, dtype=torch.long)

    controller.hook_mgr.clear()
    controller.hook_mgr.attach()
    try:
        outputs = controller.model(input_ids, use_cache=True)
        cache = outputs.past_key_values
        prefill_logits = outputs.logits
        signals = dict(controller.hook_mgr.get_signals())
    finally:
        controller.hook_mgr.remove()

    sigma = compute_segment_saturation(
        signals,
        alpha_rho=controller.hmo.alpha_rho,
        alpha_c=controller.hmo.alpha_c,
        alpha_g=controller.hmo.alpha_g,
        segment_length=controller.hmo.segment_length,
        warmup_tokens=controller.hmo.warmup_tokens,
        input_ids=input_ids,
        repeat_ngram_size=controller.hmo.repeat_ngram_size,
        repeat_threshold=controller.hmo.repeat_threshold,
    )

    replay_full_input_ids = input_ids.clone()
    refresh_store = {}
    rts_store = {}
    for seg_idx, action in forced_actions.items():
        if action == "refresh":
            start = seg_idx * seg_len
            end = min(start + seg_len, seq_len)
            refresh_store[seg_idx] = {
                "token_ids": input_ids[:, start:end].clone(),
                "position_ids": torch.arange(start, end, device=device).unsqueeze(0),
                "replay_full_input_ids": replay_full_input_ids,
            }

    for seg_idx in sorted(forced_actions.keys(), reverse=True):
        action = forced_actions[seg_idx]
        start = seg_idx * seg_len
        end = min(start + seg_len, seq_len)
        if action in {"drop", "refresh"}:
            drop_segment(cache, controller.attn_indices, start, end)
            keep_mask = (active_positions < start) | (active_positions >= end)
            active_positions = active_positions[keep_mask]
        elif action == "RTS":
            skeleton_result = extract_token_skeleton(
                cache,
                controller.attn_indices,
                start,
                end,
                n_keep=controller.hmo.skeleton_rank,
            )
            rts_store[seg_idx] = skeleton_result
            kept_set = set(skeleton_result.kept_positions)
            keep_mask = torch.tensor(
                [pos.item() in kept_set or pos.item() < start or pos.item() >= end
                 for pos in active_positions],
                dtype=torch.bool,
                device=device,
            )
            active_positions = active_positions[keep_mask]

    # Handle "evict" actions: per-token KV-norm eviction (matching controller semantics)
    evict_positions = set()
    protected_positions = set()
    for seg_idx, action in forced_actions.items():
        start = seg_idx * seg_len
        end = min(start + seg_len, seq_len)
        if action == "evict":
            for p in range(start, end):
                evict_positions.add(p)
        elif action == "KV":
            for p in range(start, end):
                protected_positions.add(p)

    if evict_positions:
        # Compute token importance from KV norms
        token_imp = torch.zeros(seq_len, device=device, dtype=torch.float32)
        for layer_idx in controller.attn_indices:
            layer = get_cache_layer(cache, layer_idx)
            if layer.has_kv():
                cache_len = layer.keys.shape[2]
                if cache_len > 0:
                    k_norms = layer.keys[0].float().norm(dim=-1).mean(dim=0)
                    v_norms = layer.values[0].float().norm(dim=-1).mean(dim=0)
                    for ci, pos in enumerate(active_positions):
                        if ci < len(k_norms) and pos.item() in evict_positions:
                            token_imp[pos.item()] += (k_norms[ci] + v_norms[ci]).item()

        # Budget: keep_ratio of evict region
        evict_pos_list = sorted(evict_positions)
        n_evict_keep = max(0, int(len(evict_pos_list) * controller.hmo.keep_ratio))
        evict_scores = torch.tensor([token_imp[p].item() for p in evict_pos_list], dtype=torch.float32)
        n_keep = min(n_evict_keep, len(evict_pos_list))
        if n_keep < len(evict_pos_list):
            _, top_idx = evict_scores.topk(max(1, n_keep))
            kept_evict = set(evict_pos_list[i] for i in top_idx.tolist())
        else:
            kept_evict = set(evict_pos_list)

        keep_mask = torch.ones(len(active_positions), dtype=torch.bool, device=device)
        for ci, pos in enumerate(active_positions):
            if pos.item() in evict_positions and pos.item() not in kept_evict:
                keep_mask[ci] = False
        from .kv_ops import evict_kv_tokens
        evict_kv_tokens(cache, controller.attn_indices, keep_mask)
        active_positions = active_positions[keep_mask]

    if refresh_store:
        for seg_idx in sorted(refresh_store):
            payload = refresh_store[seg_idx]
            active_positions = execute_refresh(
                controller.model,
                cache,
                controller.attn_indices,
                payload,
                active_positions,
            )

    # Token-pruning RTS keeps tokens in-place in cache, no separate store needed for decode
    return sigma, cache, prefill_logits, {}


def build_answer_ids(sample, tokenizer, device) -> torch.Tensor:
    """Tokenize the gold answer as a continuation of the chat prompt."""
    prompt = build_prompt(sample, tokenizer)

    full_text = prompt + " " + sample.answer
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
    full_ids = tokenizer(full_text, return_tensors="pt").input_ids
    answer_ids = full_ids[:, prompt_ids.shape[1]:]

    if answer_ids.numel() == 0:
        answer_ids = tokenizer(sample.answer, return_tensors="pt", add_special_tokens=False).input_ids
    return answer_ids.to(device)


@torch.no_grad()
def score_answer_logprob(controller, input_ids, answer_ids, cache, prefill_logits, rts_store_by_layer=None) -> float:
    """Score the gold answer under a prepared cache using teacher forcing."""
    if answer_ids.numel() == 0:
        return 0.0

    next_token_logits = prefill_logits[:, -1, :]
    logical_position = input_ids.shape[1]
    total_logprob = 0.0
    total_tokens = 0

    # Token-pruning RTS keeps tokens in-place, no patching needed
    for step in range(answer_ids.shape[1]):
        target = answer_ids[:, step]
        log_probs = torch.log_softmax(next_token_logits, dim=-1)
        token_logprob = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        total_logprob += float(token_logprob.sum().item())
        total_tokens += int(target.numel())

        target_input = target.unsqueeze(-1)
        position_ids = torch.full(
            target_input.shape,
            logical_position,
            device=target_input.device,
            dtype=torch.long,
        )
        outputs = controller.model(
            target_input,
            past_key_values=cache,
            use_cache=True,
            position_ids=position_ids,
        )
        cache = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        logical_position += 1

    return total_logprob / max(total_tokens, 1)
