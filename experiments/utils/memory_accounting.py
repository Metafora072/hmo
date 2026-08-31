"""
HMO Research — Explicit memory accounting utilities.

These helpers make the controller's budget semantics concrete:
  - active attention KV bytes
  - refresh storage bytes
  - RTS storage bytes
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .cache_access import get_cache_layer


def tensor_nbytes(tensor: torch.Tensor | None) -> int:
    """Return the physical storage size of one tensor in bytes."""
    if tensor is None:
        return 0
    return int(tensor.numel() * tensor.element_size())


def tensors_nbytes(*tensors: torch.Tensor | None) -> int:
    """Return the combined physical storage size of multiple tensors in bytes."""
    return sum(tensor_nbytes(t) for t in tensors)


def get_active_kv_bytes(cache, attn_layer_indices: list[int]) -> int:
    """Count live attention KV storage currently resident in the DynamicCache."""
    total = 0
    for layer_idx in attn_layer_indices:
        layer = get_cache_layer(cache, layer_idx)
        if not layer.has_kv():
            continue
        total += tensors_nbytes(layer.keys, layer.values)
    return total


def get_segment_kv_bytes(cache, attn_layer_indices: list[int], start: int, end: int) -> int:
    """Count the KV bytes that correspond to one logical segment."""
    total = 0
    for layer_idx in attn_layer_indices:
        layer = get_cache_layer(cache, layer_idx)
        if not layer.has_kv():
            continue
        total += tensors_nbytes(
            layer.keys[:, :, start:end, :],
            layer.values[:, :, start:end, :],
        )
    return total


def refresh_payload_nbytes(payload: dict, counted_shared_tensors: set[int] | None = None) -> int:
    """
    Count refresh storage bytes.

    Shared replay metadata such as `replay_full_input_ids` is only counted once
    when `counted_shared_tensors` is provided.
    """
    total = tensors_nbytes(
        payload.get("token_ids"),
        payload.get("position_ids"),
    )

    shared = payload.get("replay_full_input_ids")
    if shared is not None:
        shared_id = id(shared)
        if counted_shared_tensors is None or shared_id not in counted_shared_tensors:
            total += tensor_nbytes(shared)
            if counted_shared_tensors is not None:
                counted_shared_tensors.add(shared_id)

    return total


@dataclass
class HMOBudgetSnapshot:
    """Tracked storage usage for one HMO cache state."""
    active_kv_bytes: int = 0
    refresh_bytes: int = 0
    rts_bytes: int = 0
    total_bytes: int = 0
    budget_limit_bytes: int = 0


def snapshot_hmo_budget(
    cache,
    attn_layer_indices: list[int],
    refresh_store: dict | None = None,
    rts_store: dict | None = None,
    budget_limit_bytes: int = 0,
) -> HMOBudgetSnapshot:
    """Materialize one explicit budget snapshot from the current cache + stores."""
    refresh_bytes = 0
    counted_shared_tensors: set[int] = set()
    if refresh_store:
        for payload in refresh_store.values():
            refresh_bytes += refresh_payload_nbytes(payload, counted_shared_tensors)

    rts_bytes = 0
    if rts_store:
        for segment in rts_store.values():
            # Old SVD-based RTS: stored_bytes = off-cache factor storage
            # New token-pruning RTS: skeleton_kv_bytes is already in active cache,
            # so it's counted by get_active_kv_bytes(). Don't double-count.
            if hasattr(segment, "stored_bytes") and segment.stored_bytes > 0:
                rts_bytes += int(segment.stored_bytes)

    active_kv_bytes = get_active_kv_bytes(cache, attn_layer_indices)
    total_bytes = active_kv_bytes + refresh_bytes + rts_bytes
    return HMOBudgetSnapshot(
        active_kv_bytes=active_kv_bytes,
        refresh_bytes=refresh_bytes,
        rts_bytes=rts_bytes,
        total_bytes=total_bytes,
        budget_limit_bytes=budget_limit_bytes,
    )
