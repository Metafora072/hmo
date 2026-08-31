"""
HMO Research — KV Cache Operations
Direct manipulation of DynamicCache for Qwen3.5 hybrid-attention models.

Operations:
  - evict: remove token positions from attention KV cache
  - RTS: remove segment KV and store an explicit low-rank skeleton
  - refresh: recompute KV for stored tokens and splice into cache
  - drop: remove segment entirely from cache
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from loguru import logger

from .cache_access import get_cache_layer
from .memory_accounting import tensor_nbytes, tensors_nbytes


@dataclass
class RTSLayerFactors:
    """Low-rank RTS factors for one attention layer and one segment."""
    key_u: torch.Tensor
    key_s: torch.Tensor
    key_vt: torch.Tensor
    value_u: torch.Tensor
    value_s: torch.Tensor
    value_vt: torch.Tensor
    full_kv_bytes: int
    stored_bytes: int


@dataclass
class RTSSegment:
    """Explicit RTS storage for one logical segment across all attention layers."""
    start: int
    end: int
    rank: int
    layers: dict[int, RTSLayerFactors] = field(default_factory=dict)
    full_kv_bytes: int = 0
    stored_bytes: int = 0


def _factorize_low_rank(segment_tensor: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Factorize one [B, H, S, D] segment tensor into explicit low-rank factors."""
    B, H, S, D = segment_tensor.shape
    rank_eff = max(1, min(rank, S, D))
    flat = segment_tensor.reshape(B * H, S, D).float()
    u, sigma, vt = torch.linalg.svd(flat, full_matrices=False)
    u = u[:, :, :rank_eff].reshape(B, H, S, rank_eff).contiguous()
    sigma = sigma[:, :rank_eff].reshape(B, H, rank_eff).contiguous()
    vt = vt[:, :rank_eff, :].reshape(B, H, rank_eff, D).contiguous()
    return u, sigma, vt


def reconstruct_rts_segment(
    segment: RTSSegment,
    layer_idx: int,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct one RTS segment's approximate K/V for a given attention layer."""
    factors = segment.layers[layer_idx]

    key = (factors.key_u * factors.key_s.unsqueeze(-2)) @ factors.key_vt
    value = (factors.value_u * factors.value_s.unsqueeze(-2)) @ factors.value_vt

    if device is not None:
        key = key.to(device)
        value = value.to(device)
    if dtype is not None:
        key = key.to(dtype=dtype)
        value = value.to(dtype=dtype)
    return key, value


def get_attention_kv_seq_len(cache, attn_layer_idx: int) -> int:
    """Get current KV cache sequence length for an attention layer."""
    layer = get_cache_layer(cache, attn_layer_idx)
    if layer.has_kv():
        return layer.keys.shape[-2]
    return 0


def evict_kv_tokens(cache, attn_layer_indices: list[int], keep_mask: torch.Tensor):
    """
    Remove token positions from attention layers' KV cache.

    Args:
        cache: DynamicCache object
        attn_layer_indices: indices of full_attention layers
        keep_mask: [seq_len] bool tensor, True = keep this position
    """
    for layer_idx in attn_layer_indices:
        layer = get_cache_layer(cache, layer_idx)
        if not layer.has_kv():
            continue
        # keys/values shape: [B, num_heads, seq_len, head_dim]
        layer.keys = layer.keys[:, :, keep_mask, :]
        layer.values = layer.values[:, :, keep_mask, :]


def evict_kv_tokens_per_layer(cache, layer_to_keep_mask: dict[int, torch.Tensor]):
    """
    Remove token positions with a different keep mask for each attention layer.

    This is needed by PyramidKV-style baselines where lower layers keep more
    tokens and higher layers keep fewer. It should only be used for baselines
    that do not need exact global active-position bookkeeping during refresh.
    """
    for layer_idx, keep_mask in layer_to_keep_mask.items():
        layer = get_cache_layer(cache, layer_idx)
        if not layer.has_kv():
            continue
        layer.keys = layer.keys[:, :, keep_mask, :]
        layer.values = layer.values[:, :, keep_mask, :]


def extract_rts_skeleton(
    cache, attn_layer_indices: list[int],
    start: int, end: int, rank: int = 4,
 ) -> RTSSegment:
    """
    Remove a segment from active KV and return an explicit low-rank RTS store.

    The returned representation has a real physical footprint and can later be
    reconstructed on demand during decode.
    """
    segment = RTSSegment(start=start, end=end, rank=rank)
    for layer_idx in attn_layer_indices:
        layer = get_cache_layer(cache, layer_idx)
        if not layer.has_kv():
            continue

        seg_len = end - start
        if seg_len <= 0:
            continue

        seg_k = layer.keys[:, :, start:end, :]
        seg_v = layer.values[:, :, start:end, :]
        key_u, key_s, key_vt = _factorize_low_rank(seg_k, rank=rank)
        value_u, value_s, value_vt = _factorize_low_rank(seg_v, rank=rank)

        full_kv_bytes = tensors_nbytes(seg_k, seg_v)
        stored_bytes = tensors_nbytes(
            key_u, key_s, key_vt,
            value_u, value_s, value_vt,
        )

        segment.layers[layer_idx] = RTSLayerFactors(
            key_u=key_u,
            key_s=key_s,
            key_vt=key_vt,
            value_u=value_u,
            value_s=value_s,
            value_vt=value_vt,
            full_kv_bytes=full_kv_bytes,
            stored_bytes=stored_bytes,
        )
        segment.full_kv_bytes += full_kv_bytes
        segment.stored_bytes += stored_bytes

    drop_segment(cache, attn_layer_indices, start, end)
    return segment


def replace_with_skeleton(
    cache, attn_layer_indices: list[int],
    start: int, end: int, rank: int = 4,
):
    """
    Backward-compatible alias for the old name.

    The semantics are no longer in-place mutation: this now extracts an explicit
    RTS representation and removes the segment from active KV.
    """
    return extract_rts_skeleton(cache, attn_layer_indices, start, end, rank=rank)


def drop_segment(cache, attn_layer_indices: list[int], start: int, end: int):
    """
    Drop a segment entirely from attention KV cache.
    Removes positions [start, end) and shifts subsequent positions.

    Args:
        cache: DynamicCache object
        attn_layer_indices: indices of full_attention layers
        start, end: token position range to drop
    """
    for layer_idx in attn_layer_indices:
        layer = get_cache_layer(cache, layer_idx)
        if not layer.has_kv():
            continue
        seq_len = layer.keys.shape[-2]
        keep_before = layer.keys[:, :, :start, :]
        keep_after = layer.keys[:, :, end:, :]
        layer.keys = torch.cat([keep_before, keep_after], dim=-2)

        keep_before_v = layer.values[:, :, :start, :]
        keep_after_v = layer.values[:, :, end:, :]
        layer.values = torch.cat([keep_before_v, keep_after_v], dim=-2)


def insert_kv_segment(
    cache,
    attn_layer_indices: list[int],
    insert_at: int,
    segment_kv: dict[int, tuple[torch.Tensor, torch.Tensor]],
):
    """Insert one segment's exact KV back into the active cache without duplication."""
    for layer_idx in attn_layer_indices:
        layer = get_cache_layer(cache, layer_idx)
        if not layer.has_kv():
            continue

        seg_k, seg_v = segment_kv[layer_idx]
        before_k = layer.keys[:, :, :insert_at, :]
        after_k = layer.keys[:, :, insert_at:, :]
        layer.keys = torch.cat([before_k, seg_k, after_k], dim=-2)

        before_v = layer.values[:, :, :insert_at, :]
        after_v = layer.values[:, :, insert_at:, :]
        layer.values = torch.cat([before_v, seg_v, after_v], dim=-2)


@torch.no_grad()
def execute_refresh(
    model,
    cache,
    attn_layer_indices: list[int],
    refresh_payload: dict,
    active_positions: torch.Tensor,
) -> torch.Tensor:
    """
    Exact refresh by replaying the original prefix+segment tokens and splicing the
    reconstructed segment KV back into the active cache.

    The payload must contain:
      - `replay_prefix_token_ids`: original prefix token ids needed for exact replay
      - `token_ids`: original segment token ids
      - `position_ids`: original logical positions of the segment

    Returns:
        Updated active position vector after the refreshed segment has been reinserted.
    """
    refresh_token_ids = refresh_payload["token_ids"]
    position_ids = refresh_payload["position_ids"]
    if "replay_full_input_ids" in refresh_payload:
        replay_input_ids = refresh_payload["replay_full_input_ids"]
        seg_start = int(position_ids[0, 0].item())
        seg_end = int(position_ids[0, -1].item()) + 1
    else:
        prefix_ids = refresh_payload["replay_prefix_token_ids"]
        replay_input_ids = torch.cat([prefix_ids, refresh_token_ids], dim=1)
        seg_start = prefix_ids.shape[1]
        seg_end = seg_start + refresh_token_ids.shape[1]

    base_model = getattr(model, "model", model)
    replay_outputs = base_model(
        replay_input_ids,
        use_cache=True,
        return_dict=True,
    )
    replay_cache = replay_outputs.past_key_values

    segment_kv = {}
    for layer_idx in attn_layer_indices:
        replay_layer = get_cache_layer(replay_cache, layer_idx)
        segment_kv[layer_idx] = (
            replay_layer.keys[:, :, seg_start:seg_end, :].clone(),
            replay_layer.values[:, :, seg_start:seg_end, :].clone(),
        )

    insert_pos = int(position_ids[0, 0].item())
    insert_at = int(torch.searchsorted(active_positions, insert_pos).item())
    insert_kv_segment(cache, attn_layer_indices, insert_at, segment_kv)

    updated_positions = torch.cat(
        [
            active_positions[:insert_at],
            position_ids[0].to(active_positions.dtype),
            active_positions[insert_at:],
        ],
        dim=0,
    )

    del replay_outputs, replay_cache, segment_kv
    return updated_positions


# ── Token-Pruning RTS (v2) ──────────────────────────────────────────────────
# Replaces SVD-based RTS. Keeps top-r representative tokens per segment
# in-place in the cache. No runtime patching needed.


@dataclass
class TokenSkeletonResult:
    """Metadata from one token-pruning skeleton operation."""
    start: int
    end: int
    n_original: int
    n_kept: int
    kept_positions: list[int] = field(default_factory=list)
    original_kv_bytes: int = 0
    skeleton_kv_bytes: int = 0


def extract_token_skeleton(
    cache,
    attn_layer_indices: list[int],
    start: int,
    end: int,
    n_keep: int = 16,
) -> TokenSkeletonResult:
    """
    Token-pruning RTS: keep top-n_keep tokens per segment by KV norm,
    remove the rest from cache in-place.

    Unlike SVD-based RTS, the kept tokens stay in the cache with their
    original position encoding intact. No runtime attention patching needed.

    Selection criterion: sum of ||k_i|| + ||v_i|| across all attention layers
    and heads. This is a proxy for attention importance — tokens with larger
    KV norms contribute more to the softmax-weighted output.

    Args:
        cache: DynamicCache object
        attn_layer_indices: indices of full_attention layers
        start, end: token position range of the segment
        n_keep: number of tokens to retain per segment
    Returns:
        TokenSkeletonResult with metadata
    """
    seg_len = end - start
    if seg_len <= n_keep or seg_len <= 0:
        # Nothing to prune
        return TokenSkeletonResult(
            start=start, end=end, n_original=seg_len, n_kept=seg_len,
            kept_positions=list(range(start, end)),
        )

    # Compute per-token importance across all attention layers
    token_scores = torch.zeros(seg_len, device='cpu', dtype=torch.float32)
    original_kv_bytes = 0

    for layer_idx in attn_layer_indices:
        layer = get_cache_layer(cache, layer_idx)
        if not layer.has_kv():
            continue
        seg_k = layer.keys[:, :, start:end, :]  # [B, H, seg_len, D]
        seg_v = layer.values[:, :, start:end, :]
        original_kv_bytes += int(seg_k.numel() * seg_k.element_size() + seg_v.numel() * seg_v.element_size())
        # Sum of norms across batch and heads
        k_norm = seg_k.float().norm(dim=-1).sum(dim=(0, 1))  # [seg_len]
        v_norm = seg_v.float().norm(dim=-1).sum(dim=(0, 1))  # [seg_len]
        token_scores += (k_norm + v_norm).cpu()

    # Select top-n_keep positions within the segment
    _, top_indices = token_scores.topk(n_keep)
    top_indices_sorted = top_indices.sort().values  # keep original order
    keep_local = top_indices_sorted.tolist()
    keep_global = [start + i for i in keep_local]

    # Build a mask for the full cache sequence
    first_layer = get_cache_layer(cache, attn_layer_indices[0])
    seq_len_cache = first_layer.keys.shape[-2]
    cache_device = first_layer.keys.device
    keep_mask = torch.ones(seq_len_cache, dtype=torch.bool, device=cache_device)
    # Mark all positions in [start, end) as False, then re-enable the kept ones
    keep_mask[start:end] = False
    for pos in keep_global:
        keep_mask[pos] = True

    # Apply mask to all attention layers
    skeleton_kv_bytes = 0
    for layer_idx in attn_layer_indices:
        layer = get_cache_layer(cache, layer_idx)
        if not layer.has_kv():
            continue
        B = layer.keys.shape[0]
        layer.keys = layer.keys[:, :, keep_mask, :]
        layer.values = layer.values[:, :, keep_mask, :]
        skeleton_kv_bytes += int(
            B * n_keep * layer.keys.shape[1] * layer.keys.shape[-1] * layer.keys.element_size() * 2
        )

    return TokenSkeletonResult(
        start=start,
        end=end,
        n_original=seg_len,
        n_kept=n_keep,
        kept_positions=keep_global,
        original_kv_bytes=original_kv_bytes,
        skeleton_kv_bytes=skeleton_kv_bytes,
    )
