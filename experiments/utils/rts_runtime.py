"""
HMO Research — RTS runtime attention patching.

During decode, RTS segments live outside the active DynamicCache. This module
temporarily patches Qwen3.5 full-attention layers so they can attend to:
  - active KV in the cache
  - reconstructed RTS segments from the explicit low-rank store
"""
from __future__ import annotations

from contextlib import contextmanager
from types import MethodType

import torch
import torch.nn.functional as F
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb, repeat_kv

from .kv_ops import reconstruct_rts_segment


@contextmanager
def patch_qwen_attention_with_rts(model, attn_layer_indices: list[int], rts_store: dict[int, object] | None):
    """Temporarily augment Qwen3.5 attention layers with RTS memory during decode."""
    if not rts_store:
        yield
        return

    originals: list[tuple[object, object]] = []

    def make_forward(attn_module, original_forward):
        layer_idx = attn_module.layer_idx

        def _forward(this, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
            segments = rts_store.get(layer_idx, [])
            if not segments or hidden_states.shape[1] != 1:
                return original_forward(
                    hidden_states,
                    position_embeddings,
                    attention_mask,
                    past_key_values=past_key_values,
                    **kwargs,
                )

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, this.head_dim)

            query_states, gate = torch.chunk(
                this.q_proj(hidden_states).view(*input_shape, -1, this.head_dim * 2), 2, dim=-1
            )
            gate = gate.reshape(*input_shape, -1)

            query_states = this.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
            key_states = this.k_norm(this.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            value_states = this.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            if past_key_values is not None:
                key_states, value_states = past_key_values.update(key_states, value_states, this.layer_idx)

            all_keys = [repeat_kv(key_states, this.num_key_value_groups)]
            all_values = [repeat_kv(value_states, this.num_key_value_groups)]

            for segment in segments:
                seg_k, seg_v = reconstruct_rts_segment(
                    segment,
                    layer_idx=this.layer_idx,
                    dtype=key_states.dtype,
                    device=key_states.device,
                )
                all_keys.append(repeat_kv(seg_k, this.num_key_value_groups))
                all_values.append(repeat_kv(seg_v, this.num_key_value_groups))

            key_states = torch.cat(all_keys, dim=2)
            value_states = torch.cat(all_values, dim=2)

            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * this.scaling
            if attention_mask is not None:
                if attention_mask.shape[-1] != key_states.shape[-2]:
                    pad = key_states.shape[-2] - attention_mask.shape[-1]
                    if pad > 0:
                        attention_mask = F.pad(attention_mask, (0, pad), value=0.0)
                attn_weights = attn_weights + attention_mask

            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)
            attn_output = attn_output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
            attn_output = attn_output * torch.sigmoid(gate)
            attn_output = this.o_proj(attn_output)
            return attn_output, None

        return MethodType(_forward, attn_module)

    for layer_idx in attn_layer_indices:
        attn_module = model.model.layers[layer_idx].self_attn
        originals.append((attn_module, attn_module.forward))
        attn_module.forward = make_forward(attn_module, attn_module.forward)

    try:
        yield
    finally:
        for attn_module, original_forward in originals:
            attn_module.forward = original_forward
