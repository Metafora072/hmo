"""
HMO Controller — Hybrid Memory Orchestration for Qwen3.5
=========================================================
Training-free inference-time controller that coordinates two memory channels:
  1. DeltaNet recurrent state (fixed size, lossy)
  2. Attention KV cache (growing, exact)

The controller:
  - Monitors DeltaNet saturation during prefill via hooks
  - Decides per-segment actions: {KV, RTS, refresh, drop}
  - Directly manipulates the DynamicCache after prefill
  - Runs a custom decode loop with optional runtime refresh triggers

Usage:
    controller = HMOController(model, tokenizer, config)
    result = controller.run(input_ids, max_new_tokens=64)
"""
import torch
import numpy as np
from dataclasses import dataclass, field
from loguru import logger

from .cache_access import get_cache_layer
from .model_loader import get_linear_attention_indices, get_full_attention_indices
from .hooks import DeltaNetHookManager
from .saturation import compute_segment_saturation
from .kv_ops import (
    RTSSegment, TokenSkeletonResult,
    evict_kv_tokens, extract_rts_skeleton, extract_token_skeleton,
    drop_segment, execute_refresh, get_attention_kv_seq_len,
    evict_kv_tokens_per_layer,
)
from .memory_accounting import (
    HMOBudgetSnapshot,
    get_segment_kv_bytes,
    refresh_payload_nbytes,
    snapshot_hmo_budget,
    tensor_nbytes,
)
# rts_runtime no longer needed — token-pruning RTS keeps tokens in-place


@dataclass
class HMOConfig:
    """HMO controller hyperparameters."""
    segment_length: int = 512
    saturation_threshold: float = 0.7
    keep_ratio: float = 0.5          # fraction of KV tokens to keep as full KV
    skeleton_rank: int = 4            # SVD rank for RTS skeleton
    refresh_budget: int = 3           # max segments to mark for refresh
    alpha_rho: float = 0.4            # saturation signal weight: write pressure
    alpha_c: float = 0.3              # saturation signal weight: novelty collision
    alpha_g: float = 0.3              # saturation signal weight: decay pressure
    # Action thresholds (on normalized sigma)
    refresh_threshold: float = 0.8    # sigma > this → refresh candidate
    rts_threshold: float = 0.4        # sigma > this → RTS skeleton
    # drop: sigma <= rts_threshold and low attention score
    refresh_min_phi: float = 0.05     # do not refresh if all joint priorities are tiny
    refresh_alpha_mix: float = 0.0    # refresh ranking: 0=phi-only, 1=alpha-heavy
    rts_floor_tokens: int = 1         # best-effort per-segment RTS coverage floor
    rts_phi_mix: float = 0.5          # RTS allocation: 0=sigma-only, 1=phi-only
    kv_anchor_budget: int = 0         # V6.1 default: anchors are optional ablations
    kv_anchor_min_phi: float = 0.02   # avoid spending anchor budget on uninformative segments
    kv_anchor_diversity: float = 0.15 # greedy penalty for adjacent anchors
    warmup_tokens: int = 50
    repeat_ngram_size: int = 4
    repeat_threshold: float = 0.4


@dataclass
class HMOResult:
    """Result from a single HMO-controlled generation."""
    generated_text: str = ""
    generated_ids: torch.Tensor | None = None
    actions: dict = field(default_factory=dict)  # segment_idx -> action string
    sigma: np.ndarray | None = None
    peak_vram_mb: float = 0.0
    n_segments: int = 0
    n_kept_kv: int = 0
    n_skeleton: int = 0
    n_refresh: int = 0
    n_dropped: int = 0
    active_kv_bytes: int = 0
    refresh_storage_bytes: int = 0
    rts_storage_bytes: int = 0
    budget_charged_bytes: int = 0     # pre-replay: active_kv + refresh_storage + rts_storage
    decode_resident_bytes: int = 0    # post-replay: active_kv (with refreshed) + rts_storage
    total_tracked_bytes: int = 0      # alias for decode_resident_bytes (backward compat)
    budget_limit_bytes: int = 0


class HMOController:
    """
    Training-free inference-time memory orchestrator for hybrid-attention LLMs.
    """

    def __init__(
        self,
        model,
        tokenizer,
        config,
        hmo_config: HMOConfig | None = None,
        gpu_id: int = 0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.hmo = hmo_config or HMOConfig()
        self.gpu_id = gpu_id

        self.linear_indices = get_linear_attention_indices(config)
        self.attn_indices = get_full_attention_indices(config)
        self.hook_mgr = DeltaNetHookManager(
            model, self.linear_indices,
            segment_length=self.hmo.segment_length,
        )

        logger.info(
            f"HMOController: {len(self.linear_indices)} DeltaNet + "
            f"{len(self.attn_indices)} Attention layers, "
            f"seg_len={self.hmo.segment_length}"
        )

    @torch.no_grad()
    def _prefill_with_cache_last_logits(
        self,
        input_ids: torch.Tensor,
        output_attentions: bool = False,
    ):
        """
        Memory-efficient prefill helper.

        We only need:
          1. `past_key_values` for cache manipulation, and
          2. the last-token logits to seed decode.

        Calling the full CausalLM head over the entire sequence allocates a
        large [B, T, vocab] logits tensor, which becomes a major VRAM tax at
        32K+ contexts. Here we run the base model, then project only the final
        token hidden state through `lm_head`.
        """
        outputs = self.model.model(
            input_ids,
            use_cache=True,
            output_attentions=output_attentions,
            return_dict=True,
        )
        last_hidden = outputs.last_hidden_state[:, -1:, :]
        last_logits = self.model.lm_head(last_hidden)
        return outputs, outputs.past_key_values, last_logits

    @torch.no_grad()
    def run(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        do_sample: bool = False,
    ) -> HMOResult:
        """
        Full HMO-controlled generation pipeline.

        1. Prefill with hooks → saturation signals + cache
        2. Decide per-segment actions
        3. Execute cache operations
        4. Decode loop with modified cache
        """
        result = HMOResult()
        device = input_ids.device
        seq_len = input_ids.shape[1]
        seg_len = self.hmo.segment_length

        # ── Step 0: Collect attention fragility α from a separate probe pass ──
        # The frozen E1 theory contract defines φ = σ · α, where α is measured
        # from a real attention-side dependence signal (the first decode-step
        # attention distribution), not from a KV-norm proxy.
        alpha = self.collect_segment_attention_scores(input_ids)

        # ── Step 1: Prefill with DeltaNet hooks ──
        # Single prefill pass: hooks collect saturation signals, cache is kept for manipulation.
        self.hook_mgr.clear()
        self.hook_mgr.attach()
        try:
            outputs, cache, logits = self._prefill_with_cache_last_logits(input_ids)
        finally:
            signals = dict(self.hook_mgr.get_signals())
            self.hook_mgr.remove()
        del outputs
        import gc; gc.collect(); torch.cuda.empty_cache()

        # ── Step 2: Compute saturation scores ──
        sigma = compute_segment_saturation(
            signals,
            alpha_rho=self.hmo.alpha_rho,
            alpha_c=self.hmo.alpha_c,
            alpha_g=self.hmo.alpha_g,
            segment_length=self.hmo.segment_length,
            warmup_tokens=self.hmo.warmup_tokens,
            input_ids=input_ids,
            repeat_ngram_size=self.hmo.repeat_ngram_size,
            repeat_threshold=self.hmo.repeat_threshold,
        )
        result.sigma = sigma
        n_segs = len(sigma)
        result.n_segments = n_segs

        if n_segs == 0:
            # Sequence too short for segmentation, just decode normally
            return self._decode_only(input_ids, cache, logits, max_new_tokens, result, do_sample=do_sample)

        alpha = self._align_alpha_to_sigma(alpha, n_segs)

        # ── Step 3: Decide actions per segment using phi = sigma * alpha ──
        segment_costs = self._build_segment_costs(cache, input_ids, seq_len)
        actions, budget_limit_bytes = self._decide_actions(sigma, seq_len, segment_costs, input_ids, alpha=alpha)
        result.actions = actions
        result.n_segments = len(actions)
        result.budget_limit_bytes = int(budget_limit_bytes)

        # Count actions
        n_evict_segs = 0
        for a in actions.values():
            if a == "KV":
                result.n_kept_kv += 1
            elif a == "refresh":
                result.n_refresh += 1
            elif a == "evict":
                n_evict_segs += 1
            elif a == "RTS":
                result.n_skeleton += 1
            elif a == "drop":
                result.n_dropped += 1
        # n_dropped may be updated later after per-token eviction (actual token count)

        logger.info(
            f"HMO actions: KV={result.n_kept_kv}, refresh={result.n_refresh}, "
            f"evict_segs={n_evict_segs}, RTS={result.n_skeleton}, drop={result.n_dropped}"
        )

        # ── Step 4: Execute cache operations ──
        # Track the logical positions currently present in the attention KV cache.
        active_positions = torch.arange(seq_len, device=device, dtype=torch.long)

        # Store exact refresh replay payloads before mutating the active cache.
        replay_full_input_ids = input_ids.clone()
        refresh_store = {}
        rts_store: dict[int, TokenSkeletonResult] = {}
        for seg_idx, action in actions.items():
            if action == "refresh":
                start = seg_idx * seg_len
                end = min(start + seg_len, seq_len)
                refresh_store[seg_idx] = {
                    "token_ids": input_ids[:, start:end].clone(),
                    "position_ids": torch.arange(start, end, device=device).unsqueeze(0),
                    "replay_full_input_ids": replay_full_input_ids,
                }

        # Phase A+B: Execute legacy RTS, refresh, and drop in one reverse-order pass.
        # Processing from high to low segment index preserves absolute positions
        # for all operations (RTS prunes in-place, drop/refresh remove entire slices).
        for seg_idx in sorted(actions.keys(), reverse=True):
            action = actions[seg_idx]
            start = seg_idx * seg_len
            end = min(start + seg_len, seq_len)

            if action == "RTS":
                n_keep = segment_costs.get(seg_idx, {}).get("rts_n_keep", self.hmo.skeleton_rank)
                skeleton_result = extract_token_skeleton(
                    cache, self.attn_indices, start, end, n_keep=n_keep,
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
            elif action in {"refresh", "drop"}:
                drop_segment(cache, self.attn_indices, start, end)
                keep_mask = (active_positions < start) | (active_positions >= end)
                active_positions = active_positions[keep_mask]

        # Phase C: Per-token eviction on "evict" segments.
        # The theory-faithful E1 path should not emit "evict"; this branch is
        # kept only for backward compatibility with older approximate policies.
        token_importance = None
        evict_positions = set()
        protected_positions = set()
        for seg_idx, action in actions.items():
            start = seg_idx * seg_len
            end = min(start + seg_len, seq_len)
            if action == "evict":
                for p in range(start, end):
                    evict_positions.add(p)
            elif action == "KV":
                for p in range(start, end):
                    protected_positions.add(p)

        if evict_positions:
            if token_importance is None:
                token_importance = torch.zeros(seq_len, device=device, dtype=torch.float32)
                for layer_idx in self.attn_indices:
                    layer = get_cache_layer(cache, layer_idx)
                    if layer.has_kv():
                        k_norms = layer.keys[0].float().norm(dim=-1).mean(dim=0)
                        v_norms = layer.values[0].float().norm(dim=-1).mean(dim=0)
                        if k_norms.shape[0] == seq_len:
                            token_importance += k_norms + v_norms
            # token_importance was already computed from KV norms during prefill (Step 1)

            # Compute how many evict-region tokens to keep under budget
            # Sum across ALL attention layers (not just one)
            kv_bytes_per_token = 0
            for layer_idx in self.attn_indices:
                layer = get_cache_layer(cache, layer_idx)
                if layer.has_kv():
                    B, H, T, D = layer.keys.shape
                    kv_bytes_per_token += int(B * H * D * layer.keys.element_size() * 2)

            protected_bytes_actual = len(protected_positions) * kv_bytes_per_token
            # Refresh cost: per-segment token_ids + position_ids + shared replay prefix (once)
            refresh_bytes = 0
            if refresh_store:
                for s in refresh_store.values():
                    refresh_bytes += s["token_ids"].numel() * s["token_ids"].element_size()
                    refresh_bytes += s["position_ids"].numel() * s["position_ids"].element_size()
                # Shared replay prefix counted once
                first_payload = next(iter(refresh_store.values()))
                refresh_bytes += tensor_nbytes(first_payload["replay_full_input_ids"])
            remaining_budget = max(0, budget_limit_bytes - protected_bytes_actual - refresh_bytes)
            n_evict_keep = int(remaining_budget / max(kv_bytes_per_token, 1))

            # Select top-k tokens from evict regions by importance
            evict_pos_list = sorted(evict_positions)
            evict_scores = torch.tensor([token_importance[p].item() for p in evict_pos_list], dtype=torch.float32)
            n_keep = min(n_evict_keep, len(evict_pos_list))
            if n_keep < len(evict_pos_list):
                _, top_idx = evict_scores.topk(n_keep)
                kept_evict_positions = set(evict_pos_list[i] for i in top_idx.tolist())
            else:
                kept_evict_positions = set(evict_pos_list)

            # Build global keep mask for the current cache
            # active_positions maps cache index → original position
            keep_mask = torch.ones(len(active_positions), dtype=torch.bool, device=device)
            for ci, pos in enumerate(active_positions):
                p = pos.item()
                if p in evict_positions and p not in kept_evict_positions:
                    keep_mask[ci] = False

            # Apply eviction
            evict_kv_tokens(cache, self.attn_indices, keep_mask)
            active_positions = active_positions[keep_mask]

            result.n_dropped = len(evict_positions) - len(kept_evict_positions)

        # Budget snapshot BEFORE refresh replay — captures the "stored" state.
        # A second snapshot is taken AFTER replay inside _decode_loop to capture
        # the actual decode-time memory footprint (refresh KV re-enters active cache).
        pre_replay_snapshot = snapshot_hmo_budget(
            cache,
            self.attn_indices,
            refresh_store=refresh_store,
            rts_store=rts_store,
            budget_limit_bytes=budget_limit_bytes,
        )
        result.refresh_storage_bytes = pre_replay_snapshot.refresh_bytes
        result.rts_storage_bytes = pre_replay_snapshot.rts_bytes
        result.budget_charged_bytes = pre_replay_snapshot.total_bytes
        result.budget_limit_bytes = int(budget_limit_bytes)

        # ── Step 5: Decode loop ──
        return self._decode_loop(
            input_ids, cache, logits, max_new_tokens,
            refresh_store, rts_store, active_positions, result, do_sample,
        )

    def _estimate_rts_storage_bytes(self, cache, start: int, end: int) -> int:
        """Estimate token-pruning RTS cost: keeping skeleton_rank tokens per segment."""
        total = 0
        n_keep = self.hmo.skeleton_rank
        for layer_idx in self.attn_indices:
            layer = get_cache_layer(cache, layer_idx)
            if not layer.has_kv():
                continue
            B, H, _, D = layer.keys.shape
            # Each kept token stores K and V: B * H * D * element_size * 2 (K+V)
            total += int(B * H * n_keep * D * layer.keys.element_size() * 2)
        return total

    @torch.no_grad()
    def collect_segment_attention_scores(self, input_ids: torch.Tensor) -> np.ndarray:
        """
        Collect the theory-faithful attention fragility score α_j.

        This mirrors the V1/V2 validated protocol: perform a normal prefill,
        then use the first decode-step attention distribution to measure how
        much the generation query depends on each source segment.
        """
        seg_len = self.hmo.segment_length
        seq_len = input_ids.shape[1]
        n_segments = (seq_len + seg_len - 1) // seg_len
        if n_segments == 0:
            return np.array([], dtype=np.float32)

        prefill_outputs, cache, prefill_logits = self._prefill_with_cache_last_logits(input_ids)
        next_token = prefill_logits[:, -1, :].argmax(dim=-1, keepdim=True)

        orig_impl = getattr(self.model.config, "_attn_implementation", "sdpa")
        self.model.config._attn_implementation = "eager"
        if hasattr(self.model.config, "text_config"):
            self.model.config.text_config._attn_implementation = "eager"

        try:
            position_ids = torch.full(
                next_token.shape,
                seq_len,
                device=next_token.device,
                dtype=torch.long,
            )
            outputs = self.model.model(
                next_token,
                past_key_values=cache,
                use_cache=True,
                output_attentions=True,
                position_ids=position_ids,
                return_dict=True,
            )
        finally:
            self.model.config._attn_implementation = orig_impl
            if hasattr(self.model.config, "text_config"):
                self.model.config.text_config._attn_implementation = orig_impl

        token_importance = torch.zeros(seq_len, device=input_ids.device, dtype=torch.float32)
        if outputs.attentions is not None:
            for attn_w in outputs.attentions:
                if attn_w is None:
                    continue
                importance = attn_w[0].mean(dim=0).sum(dim=0).float().squeeze(0)
                if importance.shape[0] > seq_len:
                    importance = importance[:seq_len]
                elif importance.shape[0] < seq_len:
                    importance = torch.nn.functional.pad(importance, (0, seq_len - importance.shape[0]))
                token_importance += importance

        del outputs, cache, prefill_outputs, prefill_logits
        import gc; gc.collect(); torch.cuda.empty_cache()

        scores = []
        for seg_idx in range(n_segments):
            start = seg_idx * seg_len
            end = min(start + seg_len, seq_len)
            seg_score = token_importance[start:end].mean() if end > start else token_importance.new_tensor(0.0)
            scores.append(float(seg_score.item()))

        scores_arr = np.asarray(scores, dtype=np.float32)
        if scores_arr.size == 0:
            return scores_arr

        min_v = float(scores_arr.min())
        max_v = float(scores_arr.max())
        if max_v - min_v < 1e-8:
            return np.ones_like(scores_arr, dtype=np.float32)
        return (scores_arr - min_v) / (max_v - min_v)

    def _build_segment_costs(self, cache, input_ids: torch.Tensor, seq_len: int) -> dict[int, dict]:
        """Build per-segment byte costs for KV / refresh / RTS actions."""
        seg_len = self.hmo.segment_length
        costs = {}
        token_elem_size = input_ids.element_size()
        position_elem_size = torch.tensor(0, dtype=torch.long).element_size()
        replay_shared_bytes = tensor_nbytes(input_ids)

        for seg_idx in range((seq_len + seg_len - 1) // seg_len):
            start = seg_idx * seg_len
            end = min(start + seg_len, seq_len)
            refresh_segment_bytes = int((end - start) * (token_elem_size + position_elem_size))
            costs[seg_idx] = {
                "start": start,
                "end": end,
                "kv_bytes": get_segment_kv_bytes(cache, self.attn_indices, start, end),
                "refresh_segment_bytes": refresh_segment_bytes,
                "shared_refresh_bytes": replay_shared_bytes,
                "rts_bytes": self._estimate_rts_storage_bytes(cache, start, end),
            }
        return costs

    def _decide_actions(
        self,
        sigma: np.ndarray,
        seq_len: int,
        segment_costs: dict[int, dict],
        input_ids: torch.Tensor,
        alpha: np.ndarray | None = None,
    ) -> tuple[dict[int, str], int]:
        """
        Decide action for each segment using phi = sigma * alpha.

        Theory-faithful E1 strategy:
        1. Protect first 1 + last 1 segment as full KV (sinks + recent)
        2. Any trailing segments beyond sigma coverage → KV (included in budget)
        3. V6: reserve a few exact middle-segment KV anchors by diverse phi rank
        4. Select top remaining middle segments by phi rank for refresh
        5. Allocate the remaining budget to token-pruning RTS on the other middle segments
        6. Any middle segment that receives zero RTS tokens → drop
        """
        n_segs = len(sigma)
        n_total_segs = len(segment_costs)  # includes trailing segment if any
        actions = {}

        # Protect first 1 + last 1 sigma-covered segment as full KV
        n_protect_start = min(1, n_segs)
        n_protect_end = min(1, max(0, n_segs - n_protect_start))

        for i in range(n_protect_start):
            actions[i] = "KV"
        for i in range(max(0, n_segs - n_protect_end), n_segs):
            actions[i] = "KV"

        # Trailing segments beyond sigma coverage → KV (budgeted here, not post-hoc)
        for i in range(n_segs, n_total_segs):
            actions[i] = "KV"

        protected_bytes = sum(segment_costs[i]["kv_bytes"] for i in actions)
        middle_segs = [i for i in range(n_segs) if i not in actions]
        middle_full_kv_bytes = sum(segment_costs[i]["kv_bytes"] for i in middle_segs)
        budget_limit_bytes = int(protected_bytes + self.hmo.keep_ratio * middle_full_kv_bytes)
        tracked_bytes = protected_bytes

        if not middle_segs:
            return actions, budget_limit_bytes

        # Compute phi = sigma * alpha (dual-channel joint priority signal).
        # Alpha is already aligned in run(); keep a defensive pad/truncate here
        # for callers that invoke _decide_actions directly.
        alpha_arr = None
        if alpha is not None and len(alpha) > 0:
            alpha_arr = np.zeros(n_segs, dtype=np.float32)
            n_copy = min(n_segs, len(alpha))
            alpha_arr[:n_copy] = np.asarray(alpha[:n_copy], dtype=np.float32)
            if len(alpha) < n_segs:
                alpha_arr[n_copy:] = alpha_arr[n_copy - 1]
            phi = sigma * alpha_arr
        else:
            phi = sigma

        # V6 exact anchors: keep a small number of middle segments as full KV
        # before spending budget on refresh/RTS. This follows the recent
        # layer/budget and static+dynamic cache literature: a few exact anchors
        # can preserve global evidence better than representing every middle
        # segment only by sparse token skeletons.
        middle_for_budget = list(middle_segs)
        anchor_budget = max(0, int(self.hmo.kv_anchor_budget))
        if anchor_budget > 0:
            anchor_candidates = [
                i for i in middle_for_budget
                if float(phi[i]) >= float(self.hmo.kv_anchor_min_phi)
            ]
            selected_anchors: list[int] = []
            while anchor_candidates and len(selected_anchors) < anchor_budget:
                best_idx = None
                best_score = None
                for seg_idx in anchor_candidates:
                    score = float(phi[seg_idx])
                    if selected_anchors:
                        nearest = min(abs(seg_idx - s) for s in selected_anchors)
                        diversity_penalty = float(self.hmo.kv_anchor_diversity) / max(nearest, 1)
                        score -= diversity_penalty
                    if best_score is None or score > best_score:
                        best_idx = seg_idx
                        best_score = score
                if best_idx is None:
                    break
                anchor_bytes = int(segment_costs[best_idx]["kv_bytes"])
                if tracked_bytes + anchor_bytes > budget_limit_bytes:
                    break
                actions[best_idx] = "KV"
                tracked_bytes += anchor_bytes
                selected_anchors.append(best_idx)
                anchor_candidates.remove(best_idx)
                middle_for_budget.remove(best_idx)

        # Select refresh candidates by phi by default. V4.1 keeps the alpha/sigma
        # alignment fix but returns refresh/RTS scoring to the safer V3 defaults.
        if alpha_arr is not None:
            alpha_mix = min(max(float(self.hmo.refresh_alpha_mix), 0.0), 1.0)
            refresh_scores = (1.0 - alpha_mix) * phi + alpha_mix * alpha_arr
        else:
            refresh_scores = phi
        middle_refresh = [(i, refresh_scores[i], phi[i]) for i in middle_for_budget]
        middle_refresh.sort(key=lambda x: x[1], reverse=True)

        n_refresh = 0
        refresh_shared_reserved = False
        remaining_middle = []

        for seg_idx, refresh_score, phi_val in middle_refresh:
            costs = segment_costs[seg_idx]
            priority_gate = max(float(phi_val), float(refresh_score))
            if n_refresh < self.hmo.refresh_budget and priority_gate >= self.hmo.refresh_min_phi:
                # Try refresh for top-phi segments
                # Under the frozen pre-decode replay policy, a refresh segment
                # ultimately re-enters active KV before generation starts.
                # Formal E1+ budget matching therefore has to reserve the
                # larger of:
                #   (a) the exact replay payload storage, and
                #   (b) the resident KV bytes after replay.
                # In practice (b) dominates, and this is the correct budget to
                # compare against H2O / SnapKV in the formal experiments.
                refresh_increment = max(
                    int(costs["kv_bytes"]),
                    int(costs["refresh_segment_bytes"]),
                )
                if not refresh_shared_reserved:
                    refresh_increment = max(
                        refresh_increment,
                        int(costs["refresh_segment_bytes"]) + int(costs["shared_refresh_bytes"]),
                    )
                if tracked_bytes + refresh_increment <= budget_limit_bytes:
                    actions[seg_idx] = "refresh"
                    tracked_bytes += refresh_increment
                    n_refresh += 1
                    refresh_shared_reserved = True
                    continue

            remaining_middle.append(seg_idx)

        if not remaining_middle:
            return actions, budget_limit_bytes

        remaining_budget = max(0, budget_limit_bytes - tracked_bytes)
        if remaining_budget <= 0:
            for seg_idx in remaining_middle:
                actions[seg_idx] = "drop"
            return actions, budget_limit_bytes

        first_mid = remaining_middle[0]
        first_len = max(1, segment_costs[first_mid]["end"] - segment_costs[first_mid]["start"])
        bytes_per_token = segment_costs[first_mid]["kv_bytes"] / max(first_len, 1)
        total_affordable_tokens = int(remaining_budget / max(bytes_per_token, 1))

        if total_affordable_tokens <= 0:
            for seg_idx in remaining_middle:
                actions[seg_idx] = "drop"
            return actions, budget_limit_bytes

        sigma_vals = np.array([max(float(sigma[idx]), 0.0) for idx in remaining_middle], dtype=np.float32)
        if alpha_arr is not None:
            phi_vals = np.array([max(float(phi[idx]), 0.0) for idx in remaining_middle], dtype=np.float32)
            mix = min(max(float(self.hmo.rts_phi_mix), 0.0), 1.0)
            rts_priority = (1.0 - mix) * sigma_vals + mix * phi_vals
        else:
            rts_priority = sigma_vals

        seg_lens = [
            max(1, segment_costs[idx]["end"] - segment_costs[idx]["start"])
            for idx in remaining_middle
        ]
        per_seg_tokens = [0] * len(remaining_middle)

        # Best-effort coverage floor: avoid turning low-priority middle segments
        # into whole-segment drops when one token per segment is affordable.
        tokens_left = total_affordable_tokens
        floor = max(0, int(self.hmo.rts_floor_tokens))
        if floor > 0 and total_affordable_tokens > 0:
            floor_tokens = [min(floor, seg_len_i) for seg_len_i in seg_lens]
            floor_total = sum(floor_tokens)
            if floor_total <= total_affordable_tokens:
                per_seg_tokens = list(floor_tokens)
                tokens_left -= floor_total
            else:
                # Not enough budget for every segment. Give the floor to the
                # highest-priority segments rather than spreading zero-token RTS
                # across all segments.
                ranked_for_floor = sorted(
                    range(len(remaining_middle)),
                    key=lambda j: float(rts_priority[j]),
                    reverse=True,
                )
                for j in ranked_for_floor:
                    grant = min(floor_tokens[j], tokens_left)
                    per_seg_tokens[j] = grant
                    tokens_left -= grant
                    if tokens_left <= 0:
                        break
        if floor <= 0:
            tokens_left = total_affordable_tokens

        priority_sum = float(rts_priority.sum())
        if tokens_left > 0:
            if priority_sum < 1e-8:
                extra_tokens = [tokens_left // max(len(remaining_middle), 1)] * len(remaining_middle)
            else:
                extra_tokens = [
                    int(tokens_left * float(score) / priority_sum)
                    for score in rts_priority
                ]
            per_seg_tokens = [
                min(cur + extra, seg_len_i)
                for cur, extra, seg_len_i in zip(per_seg_tokens, extra_tokens, seg_lens)
            ]

        used_tokens = sum(per_seg_tokens)
        leftover = max(0, total_affordable_tokens - used_tokens)
        ranked_remaining = sorted(
            range(len(remaining_middle)),
            key=lambda j: float(rts_priority[j]),
            reverse=True,
        )
        while leftover > 0 and ranked_remaining:
            progressed = False
            for j in ranked_remaining:
                if per_seg_tokens[j] < seg_lens[j]:
                    per_seg_tokens[j] += 1
                    leftover -= 1
                    progressed = True
                    if leftover == 0:
                        break
            if not progressed:
                break

        for seg_idx, n_keep, seg_len_i in zip(remaining_middle, per_seg_tokens, seg_lens):
            if n_keep > 0:
                actions[seg_idx] = "RTS"
                segment_costs[seg_idx]["rts_n_keep"] = int(n_keep)
                segment_costs[seg_idx]["rts_bytes"] = int(
                    segment_costs[seg_idx]["kv_bytes"] * n_keep / max(seg_len_i, 1)
                )
            else:
                actions[seg_idx] = "drop"

        return actions, budget_limit_bytes

    def _decode_loop(
        self, input_ids, cache, prefill_logits, max_new_tokens,
        refresh_store, rts_store, active_positions, result, do_sample,
    ) -> HMOResult:
        """Custom decode loop using the modified cache."""
        if max_new_tokens <= 0:
            result.generated_ids = torch.tensor([], dtype=torch.long, device=input_ids.device).unsqueeze(0)
            result.generated_text = ""
            result.peak_vram_mb = torch.cuda.max_memory_allocated(self.gpu_id) / (1024 ** 2)
            return result

        # Get first token from prefill logits
        next_token_logits = prefill_logits[:, -1, :]
        if do_sample:
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)

        # Track the logical text position separately from the mutable KV cache length.
        # After KV eviction/drop/refresh, cache.get_seq_length() no longer matches the
        # original text position. Passing explicit position_ids keeps decode aligned with
        # the real sequence timeline instead of the shortened cache timeline.
        logical_position = input_ids.shape[1]

        # Replay refresh segments once before decode. This is still a prototype policy:
        # refresh is decided at prefill time and injected before autoregressive steps.
        # The first generated token still comes from prefill logits, so refresh affects
        # subsequent decode steps rather than the very first sampled token.
        if refresh_store:
            for seg_idx in sorted(refresh_store):
                payload = refresh_store[seg_idx]
                active_positions = execute_refresh(
                    self.model,
                    cache,
                    self.attn_indices,
                    payload,
                    active_positions,
                )
            logger.info(f"Executed eager refresh for {len(refresh_store)} segments before decode")

        # Post-replay budget snapshot: captures actual decode-time memory.
        # After refresh replay, refreshed KV is back in active cache.
        # Refresh storage cost is now zero (payloads consumed), active KV increased.
        # We report TWO separate accounting views:
        #   - budget_charged_bytes: what the allocator charged at decision time
        #     (pre-replay: active_kv + refresh_storage + rts_storage)
        #   - decode_resident_bytes: what's actually in memory during decode
        #     (post-replay: active_kv_with_refreshed + rts_storage, no refresh_storage)
        post_replay_snapshot = snapshot_hmo_budget(
            cache,
            self.attn_indices,
            refresh_store={},  # refresh payloads consumed after replay
            rts_store=rts_store,
            budget_limit_bytes=result.budget_limit_bytes,
        )
        result.active_kv_bytes = post_replay_snapshot.active_kv_bytes
        result.decode_resident_bytes = post_replay_snapshot.total_bytes
        result.total_tracked_bytes = post_replay_snapshot.total_bytes  # backward compat

        # Token-pruning RTS keeps skeleton tokens in-place in the cache,
        # so no runtime attention patching is needed. Just decode normally.
        generated_ids = [next_token]
        eos_token_id = self.tokenizer.eos_token_id

        for step in range(max_new_tokens - 1):
            if next_token.item() == eos_token_id:
                break

            # Forward one token with cache
            position_ids = torch.full(
                (next_token.shape[0], next_token.shape[1]),
                logical_position,
                device=next_token.device,
                dtype=torch.long,
            )
            outputs = self.model(
                next_token,
                past_key_values=cache,
                use_cache=True,
                position_ids=position_ids,
            )
            cache = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            logical_position += 1

            if do_sample:
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)

            generated_ids.append(next_token)

        # Assemble result
        all_ids = torch.cat(generated_ids, dim=-1)
        result.generated_ids = all_ids
        result.generated_text = self.tokenizer.decode(
            all_ids[0], skip_special_tokens=True
        )
        result.peak_vram_mb = torch.cuda.max_memory_allocated(self.gpu_id) / (1024 ** 2)

        return result

    def _align_alpha_to_sigma(self, alpha: np.ndarray | None, n_segs: int) -> np.ndarray | None:
        """
        Align attention fragility scores to saturation segments.

        V3 sometimes fell back to sigma-only when alpha had one extra tail
        segment. V4 keeps the dual-channel signal by truncating extra alpha
        scores and padding missing tail scores with the last observed value.
        """
        if alpha is None or n_segs <= 0:
            return None
        alpha_arr = np.asarray(alpha, dtype=np.float32).reshape(-1)
        if len(alpha_arr) == n_segs:
            return alpha_arr
        if len(alpha_arr) > n_segs:
            logger.warning(
                f"Attention fragility length mismatch (alpha={len(alpha_arr)}, sigma={n_segs}); "
                "truncating alpha instead of falling back to sigma-only."
            )
            return alpha_arr[:n_segs]
        if len(alpha_arr) == 0:
            return None
        logger.warning(
            f"Attention fragility length mismatch (alpha={len(alpha_arr)}, sigma={n_segs}); "
            "padding alpha tail instead of falling back to sigma-only."
        )
        padded = np.empty(n_segs, dtype=np.float32)
        padded[:len(alpha_arr)] = alpha_arr
        padded[len(alpha_arr):] = alpha_arr[-1]
        return padded

    def _decode_only(self, input_ids, cache, logits, max_new_tokens, result, do_sample=False):
        """Fallback: decode without any cache intervention."""
        return self._decode_loop(
            input_ids, cache, logits, max_new_tokens,
            refresh_store={}, rts_store={}, active_positions=torch.arange(input_ids.shape[1], device=input_ids.device, dtype=torch.long), result=result, do_sample=do_sample,
        )

    def run_baseline(
        self, input_ids: torch.Tensor, max_new_tokens: int = 64,
    ) -> HMOResult:
        """Run without any HMO intervention (full KV baseline)."""
        result = HMOResult()
        outputs, cache, logits = self._prefill_with_cache_last_logits(input_ids)
        del outputs
        import gc; gc.collect(); torch.cuda.empty_cache()
        # Populate memory accounting for the full-KV upper bound reference
        baseline_snapshot = snapshot_hmo_budget(
            cache, self.attn_indices,
            refresh_store={}, rts_store={}, budget_limit_bytes=0,
        )
        result.active_kv_bytes = baseline_snapshot.active_kv_bytes
        result.total_tracked_bytes = baseline_snapshot.total_bytes
        return self._decode_loop(
            input_ids, cache, logits,
            max_new_tokens, refresh_store={}, rts_store={}, active_positions=torch.arange(input_ids.shape[1], device=input_ids.device, dtype=torch.long), result=result, do_sample=False,
        )

    def run_h2o_baseline(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        budget_limit_bytes: int | None = None,
    ) -> HMOResult:
        """Run with H2O-style hard eviction (no skeleton, no refresh).

        Computes token importance from the last attention layer's QK product
        via a lightweight hook, avoiding output_attentions=True which stores
        all layers' full [B,H,T,T] matrices and OOMs at 16K+.
        """
        result = HMOResult()
        seq_len = input_ids.shape[1]

        # Hook the last attention layer to capture Q and K after RoPE,
        # then compute importance = sum of softmax(QK^T) column sums.
        last_attn_idx = self.attn_indices[-1]
        last_attn_module = self.model.model.layers[last_attn_idx].self_attn
        importance_result = {}

        def _importance_hook(module, args, kwargs, output):
            """Post-hook: compute token importance from QK without storing full T×T."""
            # We need to recompute Q, K from hidden_states to get importance.
            # But that's expensive. Instead, use a chunked approach:
            # For each query chunk, compute attention scores and accumulate column sums.
            hs = kwargs.get('hidden_states')
            if hs is None and len(args) > 0:
                hs = args[0]
            if hs is None:
                return

            with torch.no_grad():
                input_shape = hs.shape[:-1]
                hidden_shape = (*input_shape, -1, module.head_dim)

                from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb, repeat_kv

                query_states, _ = torch.chunk(
                    module.q_proj(hs).view(*input_shape, -1, module.head_dim * 2), 2, dim=-1
                )
                query_states = module.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
                key_states = module.k_norm(module.k_proj(hs).view(hidden_shape)).transpose(1, 2)

                cos, sin = kwargs.get('position_embeddings', (None, None))
                if cos is not None:
                    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

                key_states = repeat_kv(key_states, module.num_key_value_groups)

                # Chunked importance: process queries in chunks to avoid T×T allocation
                T = query_states.shape[2]
                chunk_size = min(1024, T)
                token_imp = torch.zeros(T, device=hs.device, dtype=torch.float32)

                for start in range(0, T, chunk_size):
                    end = min(start + chunk_size, T)
                    q_chunk = query_states[:, :, start:end, :]  # [B, H, chunk, D]
                    scores = torch.matmul(q_chunk, key_states.transpose(-1, -2)) * module.scaling  # [B, H, chunk, T]
                    # Causal mask
                    causal_mask = torch.arange(T, device=hs.device).unsqueeze(0) > torch.arange(start, end, device=hs.device).unsqueeze(1)
                    scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
                    weights = torch.softmax(scores, dim=-1, dtype=torch.float32)  # [B, H, chunk, T]
                    # Accumulate column sums (how much attention each token receives)
                    token_imp += weights[0].mean(dim=0).sum(dim=0)
                    del scores, weights, q_chunk

                importance_result['importance'] = token_imp
                del query_states, key_states

        handle = last_attn_module.register_forward_hook(_importance_hook, with_kwargs=True)
        try:
            outputs, cache, logits = self._prefill_with_cache_last_logits(input_ids)
        finally:
            handle.remove()

        token_importance = importance_result.get('importance', torch.zeros(seq_len, device=input_ids.device))
        del outputs, importance_result

        # H2O eviction: keep top-k + sinks + recent, optionally under a byte cap.
        keep_mask = torch.zeros(seq_len, dtype=torch.bool, device=input_ids.device)
        keep_mask[:4] = True   # sinks
        keep_mask[-32:] = True  # recent

        n_keep = self._compute_budget_n_keep(cache, seq_len, budget_limit_bytes) if budget_limit_bytes else int(seq_len * self.hmo.keep_ratio)
        n_middle_keep = max(0, n_keep - keep_mask.sum().item())
        middle_scores = token_importance[4:-32].clone()
        if n_middle_keep > 0 and len(middle_scores) > 0:
            _, top_idx = middle_scores.topk(min(n_middle_keep, len(middle_scores)))
            keep_mask[4 + top_idx] = True

        evict_kv_tokens(cache, self.attn_indices, keep_mask)
        return self._finalize_eviction_result(result, cache, keep_mask, input_ids, logits, max_new_tokens, budget_limit_bytes)

    def _compute_budget_n_keep(self, cache, seq_len, budget_limit_bytes):
        """Compute how many tokens to keep given a byte budget."""
        full_kv_bytes = 0
        for layer_idx in self.attn_indices:
            layer = get_cache_layer(cache, layer_idx)
            if layer.has_kv():
                full_kv_bytes += tensor_nbytes(layer.keys) + tensor_nbytes(layer.values)
        bytes_per_token = max(full_kv_bytes // max(seq_len, 1), 1)
        min_keep = min(seq_len, 4 + min(32, max(seq_len - 4, 0)))
        n_keep = max(min_keep, int(budget_limit_bytes // bytes_per_token))
        return min(seq_len, n_keep)

    def _finalize_eviction_result(self, result, cache, keep_mask, input_ids, logits,
                                  max_new_tokens, budget_limit_bytes):
        """Shared post-eviction logic: snapshot budget, run decode loop."""
        seq_len = input_ids.shape[1]
        result.n_kept_kv = keep_mask.sum().item()
        result.n_dropped = seq_len - result.n_kept_kv
        budget_snapshot = snapshot_hmo_budget(
            cache, self.attn_indices,
            refresh_store={}, rts_store={},
            budget_limit_bytes=int(budget_limit_bytes or 0),
        )
        result.active_kv_bytes = budget_snapshot.active_kv_bytes
        result.refresh_storage_bytes = budget_snapshot.refresh_bytes
        result.rts_storage_bytes = budget_snapshot.rts_bytes
        result.total_tracked_bytes = budget_snapshot.total_bytes
        result.budget_limit_bytes = budget_snapshot.budget_limit_bytes
        return self._decode_loop(
            input_ids, cache, logits, max_new_tokens,
            refresh_store={}, rts_store={},
            active_positions=torch.arange(seq_len, device=input_ids.device, dtype=torch.long)[keep_mask],
            result=result, do_sample=False,
        )

    def _kv_norm_token_scores(
        self,
        cache,
        seq_len: int,
        layer_indices: list[int] | None = None,
    ) -> torch.Tensor:
        """Approximate token importance by summed K/V norms over selected layers."""
        layer_indices = layer_indices or self.attn_indices
        device = None
        scores = None
        for layer_idx in layer_indices:
            layer = get_cache_layer(cache, layer_idx)
            if not layer.has_kv() or layer.keys.shape[-2] != seq_len:
                continue
            if scores is None:
                device = layer.keys.device
                scores = torch.zeros(seq_len, device=device, dtype=torch.float32)
            k_norms = layer.keys[0].float().norm(dim=-1).mean(dim=0)
            v_norms = layer.values[0].float().norm(dim=-1).mean(dim=0)
            scores += k_norms + v_norms
        if scores is None:
            device = torch.device(f"cuda:{self.gpu_id}") if torch.cuda.is_available() else torch.device("cpu")
            scores = torch.zeros(seq_len, device=device, dtype=torch.float32)
        return scores

    def _topk_keep_mask(
        self,
        scores: torch.Tensor,
        n_keep: int,
        n_sink: int = 4,
        n_recent: int = 32,
    ) -> torch.Tensor:
        """Build a global keep mask from token scores plus sink/recent protection."""
        seq_len = scores.shape[0]
        keep_mask = torch.zeros(seq_len, dtype=torch.bool, device=scores.device)
        keep_mask[:min(n_sink, seq_len)] = True
        if seq_len > n_sink and n_recent > 0:
            keep_mask[-min(n_recent, max(seq_len - n_sink, 0)):] = True

        n_middle_keep = max(0, min(n_keep, seq_len) - int(keep_mask.sum().item()))
        lo = min(n_sink, seq_len)
        hi = max(lo, seq_len - min(n_recent, max(seq_len - n_sink, 0)))
        middle_scores = scores[lo:hi].clone()
        if n_middle_keep > 0 and len(middle_scores) > 0:
            _, top_idx = middle_scores.topk(min(n_middle_keep, len(middle_scores)))
            keep_mask[lo + top_idx] = True
        return keep_mask

    def run_budgeted_recent_kv_baseline(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        budget_limit_bytes: int | None = None,
        n_sink: int = 4,
    ) -> HMOResult:
        """
        Budget-matched recent-KV baseline.

        This is a plain Full-KV subset baseline: keep a few sink tokens and use
        all remaining budget for the most recent tokens. It does not use
        importance scoring, refresh, or RTS.
        """
        result = HMOResult()
        seq_len = input_ids.shape[1]
        outputs, cache, logits = self._prefill_with_cache_last_logits(input_ids)
        del outputs

        n_keep = self._compute_budget_n_keep(cache, seq_len, budget_limit_bytes) if budget_limit_bytes else int(seq_len * self.hmo.keep_ratio)
        n_sink_eff = min(n_sink, seq_len)
        n_recent = max(0, min(seq_len - n_sink_eff, n_keep - n_sink_eff))

        keep_mask = torch.zeros(seq_len, dtype=torch.bool, device=input_ids.device)
        keep_mask[:n_sink_eff] = True
        if n_recent > 0:
            keep_mask[-n_recent:] = True

        evict_kv_tokens(cache, self.attn_indices, keep_mask)
        return self._finalize_eviction_result(result, cache, keep_mask, input_ids, logits, max_new_tokens, budget_limit_bytes)

    def run_budgeted_uniform_kv_baseline(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        budget_limit_bytes: int | None = None,
        n_sink: int = 4,
        n_recent: int = 32,
    ) -> HMOResult:
        """
        Budget-matched uniform-KV baseline.

        This plain Full-KV subset keeps sink/recent tokens and fills the rest of
        the budget with uniformly spaced middle tokens for global coverage.
        """
        result = HMOResult()
        seq_len = input_ids.shape[1]
        outputs, cache, logits = self._prefill_with_cache_last_logits(input_ids)
        del outputs

        n_keep = self._compute_budget_n_keep(cache, seq_len, budget_limit_bytes) if budget_limit_bytes else int(seq_len * self.hmo.keep_ratio)
        keep_mask = torch.zeros(seq_len, dtype=torch.bool, device=input_ids.device)

        n_sink_eff = min(n_sink, seq_len)
        keep_mask[:n_sink_eff] = True
        if seq_len > n_sink_eff and n_recent > 0:
            n_recent_eff = min(n_recent, max(seq_len - n_sink_eff, 0))
            keep_mask[-n_recent_eff:] = True

        n_middle_keep = max(0, min(n_keep, seq_len) - int(keep_mask.sum().item()))
        lo = n_sink_eff
        hi = seq_len - int(keep_mask[-min(n_recent, seq_len):].sum().item()) if n_recent > 0 else seq_len
        hi = max(lo, hi)
        middle_len = hi - lo
        if n_middle_keep > 0 and middle_len > 0:
            if n_middle_keep >= middle_len:
                keep_mask[lo:hi] = True
            else:
                uniform_idx = torch.linspace(
                    0, middle_len - 1,
                    steps=n_middle_keep,
                    device=input_ids.device,
                ).round().long().unique()
                # Rounding can collapse adjacent positions; fill any remaining
                # budget from left to right to keep the exact target count.
                keep_mask[lo + uniform_idx] = True
                remaining = min(n_keep, seq_len) - int(keep_mask.sum().item())
                if remaining > 0:
                    candidates = torch.arange(lo, hi, device=input_ids.device)
                    candidates = candidates[~keep_mask[candidates]]
                    keep_mask[candidates[:remaining]] = True

        evict_kv_tokens(cache, self.attn_indices, keep_mask)
        return self._finalize_eviction_result(result, cache, keep_mask, input_ids, logits, max_new_tokens, budget_limit_bytes)

    def run_sagekv_lite_baseline(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        budget_limit_bytes: int | None = None,
    ) -> HMOResult:
        """
        SAGE-KV-lite baseline.

        Faithful SAGE-KV uses self-attention-guided top-k at token and head
        granularity. This lite version keeps the post-prefill, one-shot eviction
        structure but uses global token scores from K/V norms so it remains
        compatible with dense cache tensors.
        """
        result = HMOResult()
        seq_len = input_ids.shape[1]
        outputs, cache, logits = self._prefill_with_cache_last_logits(input_ids)
        del outputs

        n_keep = self._compute_budget_n_keep(cache, seq_len, budget_limit_bytes) if budget_limit_bytes else int(seq_len * self.hmo.keep_ratio)
        scores = self._kv_norm_token_scores(cache, seq_len)
        keep_mask = self._topk_keep_mask(scores, n_keep)

        evict_kv_tokens(cache, self.attn_indices, keep_mask)
        return self._finalize_eviction_result(result, cache, keep_mask, input_ids, logits, max_new_tokens, budget_limit_bytes)

    def run_quest_lite_baseline(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        budget_limit_bytes: int | None = None,
        page_size: int = 32,
    ) -> HMOResult:
        """
        Quest-lite baseline.

        Faithful Quest selects KV pages dynamically for every decode query. This
        lite version uses the final prompt query as a static query proxy, scores
        KV pages once after prefill, then decodes with the selected pages.
        """
        result = HMOResult()
        seq_len = input_ids.shape[1]
        last_attn_idx = self.attn_indices[-1]
        last_attn_module = self.model.model.layers[last_attn_idx].self_attn
        page_scores_holder = {}

        def _quest_proxy_hook(module, args, kwargs, output):
            hs = kwargs.get("hidden_states")
            if hs is None and len(args) > 0:
                hs = args[0]
            if hs is None:
                return
            with torch.no_grad():
                input_shape = hs.shape[:-1]
                hidden_shape = (*input_shape, -1, module.head_dim)
                from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb, repeat_kv

                query_states, _ = torch.chunk(
                    module.q_proj(hs).view(*input_shape, -1, module.head_dim * 2), 2, dim=-1
                )
                query_states = module.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
                key_states = module.k_norm(module.k_proj(hs).view(hidden_shape)).transpose(1, 2)
                cos, sin = kwargs.get("position_embeddings", (None, None))
                if cos is not None:
                    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
                key_states = repeat_kv(key_states, module.num_key_value_groups)

                q_last = query_states[:, :, -1:, :]
                scores = torch.matmul(q_last, key_states.transpose(-1, -2)) * module.scaling
                token_scores = scores[0, :, 0, :].float().mean(dim=0)
                n_pages = (token_scores.shape[0] + page_size - 1) // page_size
                page_scores = torch.empty(n_pages, device=token_scores.device, dtype=torch.float32)
                for page_idx in range(n_pages):
                    start = page_idx * page_size
                    end = min(start + page_size, token_scores.shape[0])
                    page_scores[page_idx] = token_scores[start:end].max()
                page_scores_holder["scores"] = page_scores
                del query_states, key_states, scores

        handle = last_attn_module.register_forward_hook(_quest_proxy_hook, with_kwargs=True)
        try:
            outputs, cache, logits = self._prefill_with_cache_last_logits(input_ids)
        finally:
            handle.remove()
        del outputs

        n_keep = self._compute_budget_n_keep(cache, seq_len, budget_limit_bytes) if budget_limit_bytes else int(seq_len * self.hmo.keep_ratio)
        keep_mask = torch.zeros(seq_len, dtype=torch.bool, device=input_ids.device)
        keep_mask[:min(4, seq_len)] = True
        if seq_len > 4:
            keep_mask[-min(32, seq_len - 4):] = True

        page_scores = page_scores_holder.get("scores")
        if page_scores is None:
            scores = self._kv_norm_token_scores(cache, seq_len)
            keep_mask = self._topk_keep_mask(scores, n_keep)
        else:
            n_fixed = int(keep_mask.sum().item())
            n_page_tokens = max(0, n_keep - n_fixed)
            n_pages_keep = min(len(page_scores), max(0, (n_page_tokens + page_size - 1) // page_size))
            if n_pages_keep > 0:
                _, page_idx = page_scores.topk(n_pages_keep)
                for idx in page_idx.tolist():
                    start = idx * page_size
                    end = min(start + page_size, seq_len)
                    keep_mask[start:end] = True
            if int(keep_mask.sum().item()) > n_keep:
                scores = self._kv_norm_token_scores(cache, seq_len)
                keep_mask = self._topk_keep_mask(scores.masked_fill(~keep_mask, float("-inf")), n_keep)

        evict_kv_tokens(cache, self.attn_indices, keep_mask)
        return self._finalize_eviction_result(result, cache, keep_mask, input_ids, logits, max_new_tokens, budget_limit_bytes)

    def run_pyramidkv_lite_baseline(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        budget_limit_bytes: int | None = None,
        beta: float = 2.0,
    ) -> HMOResult:
        """
        PyramidKV-lite baseline.

        Faithful PyramidKV allocates different KV budgets across layers and
        selects tokens by attention scores. This lite version keeps the
        layer-wise pyramid budget and uses per-layer K/V norm scores.
        """
        result = HMOResult()
        seq_len = input_ids.shape[1]
        outputs, cache, logits = self._prefill_with_cache_last_logits(input_ids)
        del outputs

        if not self.attn_indices:
            return self._decode_only(input_ids, cache, logits, max_new_tokens, result)

        full_kv_bytes = 0
        bytes_per_token_layer = 0
        valid_layers = []
        for layer_idx in self.attn_indices:
            layer = get_cache_layer(cache, layer_idx)
            if not layer.has_kv() or layer.keys.shape[-2] != seq_len:
                continue
            valid_layers.append(layer_idx)
            layer_bytes = tensor_nbytes(layer.keys) + tensor_nbytes(layer.values)
            full_kv_bytes += layer_bytes
            bytes_per_token_layer = max(bytes_per_token_layer, layer_bytes // max(seq_len, 1))

        if not valid_layers or bytes_per_token_layer <= 0:
            return self._decode_only(input_ids, cache, logits, max_new_tokens, result)

        if budget_limit_bytes:
            total_token_budget = max(len(valid_layers) * 36, int(budget_limit_bytes // bytes_per_token_layer))
        else:
            total_token_budget = int(seq_len * len(valid_layers) * self.hmo.keep_ratio)

        m = len(valid_layers)
        avg = max(36.0, total_token_budget / max(m, 1))
        bottom = min(float(seq_len), 2.0 * avg)
        top = max(36.0, avg / max(beta, 1e-6))
        raw = torch.linspace(bottom, top, steps=m)
        raw = raw * (float(total_token_budget) / max(float(raw.sum().item()), 1.0))
        per_layer_keep = [int(max(36, min(seq_len, round(x)))) for x in raw.tolist()]

        # Adjust rounding so the total budget stays close to the target.
        diff = int(total_token_budget - sum(per_layer_keep))
        order = list(range(m))
        while diff != 0 and order:
            progressed = False
            for j in order:
                if diff > 0 and per_layer_keep[j] < seq_len:
                    per_layer_keep[j] += 1
                    diff -= 1
                    progressed = True
                elif diff < 0 and per_layer_keep[j] > 36:
                    per_layer_keep[j] -= 1
                    diff += 1
                    progressed = True
                if diff == 0:
                    break
            if not progressed:
                break

        layer_to_keep_mask = {}
        first_layer_mask = None
        for layer_idx, n_keep in zip(valid_layers, per_layer_keep):
            scores = self._kv_norm_token_scores(cache, seq_len, layer_indices=[layer_idx])
            keep_mask = self._topk_keep_mask(scores, n_keep)
            layer_to_keep_mask[layer_idx] = keep_mask
            if first_layer_mask is None:
                first_layer_mask = keep_mask

        evict_kv_tokens_per_layer(cache, layer_to_keep_mask)
        result.n_kept_kv = int(round(sum(per_layer_keep) / max(len(per_layer_keep), 1)))
        result.n_dropped = seq_len - result.n_kept_kv
        budget_snapshot = snapshot_hmo_budget(
            cache,
            self.attn_indices,
            refresh_store={},
            rts_store={},
            budget_limit_bytes=int(budget_limit_bytes or 0),
        )
        result.active_kv_bytes = budget_snapshot.active_kv_bytes
        result.total_tracked_bytes = budget_snapshot.total_bytes
        result.budget_limit_bytes = budget_snapshot.budget_limit_bytes
        active_positions = torch.arange(seq_len, device=input_ids.device, dtype=torch.long)
        if first_layer_mask is not None and first_layer_mask.shape[0] == seq_len:
            active_positions = active_positions[first_layer_mask]
        return self._decode_loop(
            input_ids, cache, logits, max_new_tokens,
            refresh_store={}, rts_store={}, active_positions=active_positions,
            result=result, do_sample=False,
        )

    def run_snapkv_baseline(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        budget_limit_bytes: int | None = None,
        observation_window: int = 32,
    ) -> HMOResult:
        """SnapKV baseline: use last `observation_window` queries' attention over
        all keys to score token importance. Hook the last 3 attention layers."""
        result = HMOResult()
        seq_len = input_ids.shape[1]

        # Hook last 3 (or fewer) attention layers to capture Q, K for scoring
        n_hook = min(3, len(self.attn_indices))
        hook_layers = self.attn_indices[-n_hook:]
        importance_accum = torch.zeros(seq_len, device=input_ids.device, dtype=torch.float32)
        handles = []

        def _make_snapkv_hook(layer_idx):
            def _hook(module, args, kwargs, output):
                hs = kwargs.get('hidden_states')
                if hs is None and len(args) > 0:
                    hs = args[0]
                if hs is None:
                    return
                with torch.no_grad():
                    input_shape = hs.shape[:-1]
                    hidden_shape = (*input_shape, -1, module.head_dim)
                    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb, repeat_kv
                    query_states, _ = torch.chunk(
                        module.q_proj(hs).view(*input_shape, -1, module.head_dim * 2), 2, dim=-1
                    )
                    query_states = module.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
                    key_states = module.k_norm(module.k_proj(hs).view(hidden_shape)).transpose(1, 2)
                    cos, sin = kwargs.get('position_embeddings', (None, None))
                    if cos is not None:
                        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
                    key_states = repeat_kv(key_states, module.num_key_value_groups)
                    # Only use last observation_window queries
                    T = query_states.shape[2]
                    obs_start = max(0, T - observation_window)
                    q_obs = query_states[:, :, obs_start:, :]
                    scores = torch.matmul(q_obs, key_states.transpose(-1, -2)) * module.scaling
                    # Causal mask for observation window queries
                    causal_mask = torch.arange(T, device=hs.device).unsqueeze(0) > torch.arange(obs_start, T, device=hs.device).unsqueeze(1)
                    scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
                    weights = torch.softmax(scores, dim=-1, dtype=torch.float32)
                    token_imp = weights[0].mean(dim=0).sum(dim=0)  # [T]
                    importance_accum.add_(token_imp)
                    del query_states, key_states, scores, weights
            return _hook

        for li in hook_layers:
            module = self.model.model.layers[li].self_attn
            h = module.register_forward_hook(_make_snapkv_hook(li), with_kwargs=True)
            handles.append(h)

        try:
            outputs, cache, logits = self._prefill_with_cache_last_logits(input_ids)
        finally:
            for h in handles:
                h.remove()
        del outputs

        n_keep = self._compute_budget_n_keep(cache, seq_len, budget_limit_bytes) if budget_limit_bytes else int(seq_len * self.hmo.keep_ratio)
        keep_mask = torch.zeros(seq_len, dtype=torch.bool, device=input_ids.device)
        keep_mask[:4] = True
        keep_mask[-32:] = True
        n_fixed = keep_mask.sum().item()
        n_middle_keep = max(0, n_keep - n_fixed)
        middle_scores = importance_accum[4:-32].clone()
        if n_middle_keep > 0 and len(middle_scores) > 0:
            _, top_idx = middle_scores.topk(min(n_middle_keep, len(middle_scores)))
            keep_mask[4 + top_idx] = True

        evict_kv_tokens(cache, self.attn_indices, keep_mask)
        return self._finalize_eviction_result(result, cache, keep_mask, input_ids, logits, max_new_tokens, budget_limit_bytes)

    def run_streamingllm_baseline(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        budget_limit_bytes: int | None = None,
        n_sink: int = 4,
    ) -> HMOResult:
        """StreamingLLM baseline: keep first n_sink tokens + last n_recent tokens,
        drop everything in between."""
        result = HMOResult()
        seq_len = input_ids.shape[1]

        outputs, cache, logits = self._prefill_with_cache_last_logits(input_ids)
        del outputs

        n_keep = self._compute_budget_n_keep(cache, seq_len, budget_limit_bytes) if budget_limit_bytes else int(seq_len * self.hmo.keep_ratio)
        n_recent = max(1, n_keep - n_sink)

        keep_mask = torch.zeros(seq_len, dtype=torch.bool, device=input_ids.device)
        keep_mask[:n_sink] = True
        keep_mask[-n_recent:] = True

        evict_kv_tokens(cache, self.attn_indices, keep_mask)
        return self._finalize_eviction_result(result, cache, keep_mask, input_ids, logits, max_new_tokens, budget_limit_bytes)

    def run_duoattention_baseline(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        budget_limit_bytes: int | None = None,
    ) -> HMOResult:
        """DuoAttention baseline: classify heads by attention entropy into
        retrieval (full context) vs streaming (sinks+recent). Use union-of-heads
        keep mask to avoid ragged tensors."""
        result = HMOResult()
        seq_len = input_ids.shape[1]

        # Collect per-head entropy from all attention layers
        head_entropies = {}  # (layer_idx, head_idx) -> entropy
        handles = []

        def _make_entropy_hook(layer_idx):
            def _hook(module, args, kwargs, output):
                hs = kwargs.get('hidden_states')
                if hs is None and len(args) > 0:
                    hs = args[0]
                if hs is None:
                    return
                with torch.no_grad():
                    input_shape = hs.shape[:-1]
                    hidden_shape = (*input_shape, -1, module.head_dim)
                    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb, repeat_kv
                    query_states, _ = torch.chunk(
                        module.q_proj(hs).view(*input_shape, -1, module.head_dim * 2), 2, dim=-1
                    )
                    query_states = module.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
                    key_states = module.k_norm(module.k_proj(hs).view(hidden_shape)).transpose(1, 2)
                    cos, sin = kwargs.get('position_embeddings', (None, None))
                    if cos is not None:
                        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
                    key_states = repeat_kv(key_states, module.num_key_value_groups)
                    # Use last query token for entropy computation
                    q_last = query_states[:, :, -1:, :]
                    scores = torch.matmul(q_last, key_states.transpose(-1, -2)) * module.scaling
                    probs = torch.softmax(scores, dim=-1, dtype=torch.float32)
                    # Per-head entropy: -sum(p * log(p))
                    log_probs = torch.log(probs + 1e-10)
                    entropy = -(probs * log_probs).sum(dim=-1)  # [B, H, 1]
                    for h_idx in range(entropy.shape[1]):
                        head_entropies[(layer_idx, h_idx)] = float(entropy[0, h_idx, 0].item())
                    del query_states, key_states, scores, probs
            return _hook

        for li in self.attn_indices:
            module = self.model.model.layers[li].self_attn
            h = module.register_forward_hook(_make_entropy_hook(li), with_kwargs=True)
            handles.append(h)

        try:
            outputs, cache, logits = self._prefill_with_cache_last_logits(input_ids)
        finally:
            for h in handles:
                h.remove()
        del outputs

        # Classify heads: entropy > median → retrieval, else streaming
        all_ent = list(head_entropies.values())
        median_ent = float(np.median(all_ent)) if all_ent else 0.0
        retrieval_heads = {k for k, v in head_entropies.items() if v > median_ent}

        n_keep = self._compute_budget_n_keep(cache, seq_len, budget_limit_bytes) if budget_limit_bytes else int(seq_len * self.hmo.keep_ratio)

        # DuoAttention policy:
        # - Streaming heads need only sinks + recent (cheap)
        # - Retrieval heads need broader context (expensive)
        # Union keep mask: sinks + recent (all heads) + top-k middle (retrieval heads)
        n_sink = 4
        n_recent = min(32, max(seq_len - n_sink, 0))
        n_retrieval_middle = max(0, n_keep - n_sink - n_recent)

        # Compute per-token importance from retrieval heads only.
        # Map query head indices to KV head indices (GQA: multiple query heads share one KV head).
        retrieval_importance = torch.zeros(seq_len, device=input_ids.device, dtype=torch.float32)
        retrieval_kv_heads_seen = set()
        for (li, hi) in retrieval_heads:
            layer = get_cache_layer(cache, li)
            if not layer.has_kv():
                continue
            n_kv_heads = layer.keys.shape[1]
            # Entropy was computed on expanded query heads; map back to KV head
            module = self.model.model.layers[li].self_attn
            n_groups = getattr(module, "num_key_value_groups", 1)
            kv_hi = hi // n_groups
            if kv_hi >= n_kv_heads or layer.keys.shape[2] != seq_len:
                continue
            # Avoid double-counting the same KV head from multiple query heads
            key = (li, kv_hi)
            if key in retrieval_kv_heads_seen:
                continue
            retrieval_kv_heads_seen.add(key)
            k_norms = layer.keys[0, kv_hi].float().norm(dim=-1)  # [T]
            retrieval_importance += k_norms

        keep_mask = torch.zeros(seq_len, dtype=torch.bool, device=input_ids.device)
        keep_mask[:n_sink] = True
        if n_recent > 0:
            keep_mask[-n_recent:] = True

        # Add top-scoring middle tokens for retrieval heads
        if n_retrieval_middle > 0 and seq_len > n_sink + n_recent:
            middle_scores = retrieval_importance[n_sink:-n_recent].clone() if n_recent > 0 else retrieval_importance[n_sink:].clone()
            actual_keep = min(n_retrieval_middle, len(middle_scores))
            if actual_keep > 0:
                _, top_idx = middle_scores.topk(actual_keep)
                keep_mask[n_sink + top_idx] = True

        evict_kv_tokens(cache, self.attn_indices, keep_mask)
        return self._finalize_eviction_result(result, cache, keep_mask, input_ids, logits, max_new_tokens, budget_limit_bytes)
