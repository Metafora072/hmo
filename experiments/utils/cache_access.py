"""
Compatibility helpers for different Transformers cache layouts.

Qwen3.5 currently returns `Qwen3_5DynamicCache`, which stores attention KV in
`key_cache` / `value_cache` lists. Newer generic `DynamicCache` stores cache
layers in `.layers`, each with `.keys` / `.values`. The experiment code should
not depend on either private layout directly.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CacheLayerView:
    """Mutable view exposing `.keys` and `.values` for one cache layer."""
    cache: object
    layer_idx: int

    @property
    def keys(self) -> torch.Tensor | None:
        if hasattr(self.cache, "layers"):
            return getattr(self.cache.layers[self.layer_idx], "keys", None)
        if hasattr(self.cache, "key_cache"):
            return self.cache.key_cache[self.layer_idx]
        return None

    @keys.setter
    def keys(self, value: torch.Tensor | None) -> None:
        if hasattr(self.cache, "layers"):
            self.cache.layers[self.layer_idx].keys = value
        elif hasattr(self.cache, "key_cache"):
            self.cache.key_cache[self.layer_idx] = value
        else:
            raise AttributeError(f"Unsupported cache type: {type(self.cache).__name__}")

    @property
    def values(self) -> torch.Tensor | None:
        if hasattr(self.cache, "layers"):
            return getattr(self.cache.layers[self.layer_idx], "values", None)
        if hasattr(self.cache, "value_cache"):
            return self.cache.value_cache[self.layer_idx]
        return None

    @values.setter
    def values(self, value: torch.Tensor | None) -> None:
        if hasattr(self.cache, "layers"):
            self.cache.layers[self.layer_idx].values = value
        elif hasattr(self.cache, "value_cache"):
            self.cache.value_cache[self.layer_idx] = value
        else:
            raise AttributeError(f"Unsupported cache type: {type(self.cache).__name__}")

    def has_kv(self) -> bool:
        keys = self.keys
        values = self.values
        return (
            isinstance(keys, torch.Tensor)
            and isinstance(values, torch.Tensor)
            and keys.numel() > 0
            and values.numel() > 0
        )


def get_cache_layer(cache, layer_idx: int) -> CacheLayerView:
    """Return a mutable KV view for `layer_idx` across cache implementations."""
    return CacheLayerView(cache=cache, layer_idx=layer_idx)
