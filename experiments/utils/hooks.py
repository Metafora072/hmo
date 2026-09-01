"""
HMO Research — legacy DeltaNet Layer Hooks (v2)
Memory-efficient: computes per-segment saturation signals inside the hook,
only stores scalar aggregates, NOT full [B,T,H,D] tensors.

This module preserves `sigma_current` as a historical baseline. Its rho/c/g
features are proxies and do not implement the E3-v2 P0-C candidates.

Historical implementation notes:
- Captures post-conv key (actual write direction) not pre-conv
- Computes rho from write magnitude vs positive decay magnitude
- Streams per-segment aggregates to avoid OOM at 128K
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class SegmentSignals:
    """Per-segment saturation signals aggregated across one DeltaNet layer."""
    layer_idx: int
    # Each list has length = num_segments, values are floats
    rho_max: list[float] = field(default_factory=list)    # max write-to-state ratio in segment
    c_max: list[float] = field(default_factory=list)      # max novelty collision in segment
    g_mag_min: list[float] = field(default_factory=list)  # min positive decay magnitude (-g)


class DeltaNetHookManager:
    """
    Attach forward hooks to DeltaNet layers. Computes per-segment saturation
    signals inside the hook to avoid storing full-sequence tensors.

    Usage:
        hook_mgr = DeltaNetHookManager(model, linear_indices, segment_length=512)
        hook_mgr.attach()
        with torch.no_grad():
            output = model(input_ids)  # prefill
        signals = hook_mgr.get_signals()  # dict[layer_idx -> SegmentSignals]
        hook_mgr.remove()
    """

    def __init__(self, model: nn.Module, linear_layer_indices: list[int],
                 segment_length: int = 512):
        self.model = model
        self.linear_layer_indices = linear_layer_indices
        self.segment_length = segment_length
        self._hooks = []
        self._signals: dict[int, SegmentSignals] = {}

    def _get_deltanet_module(self, layer_idx: int):
        return self.model.model.layers[layer_idx].linear_attn

    def attach(self):
        self.remove()
        for layer_idx in self.linear_layer_indices:
            module = self._get_deltanet_module(layer_idx)
            hook = _DeltaNetStreamHook(layer_idx, self._signals, self.segment_length)
            # Use forward_pre_hook with kwargs to capture hidden_states
            # (Qwen3.5 passes hidden_states as keyword arg)
            handle = module.register_forward_pre_hook(hook, with_kwargs=True)
            self._hooks.append(handle)
        logger.info(f"Attached streaming hooks to {len(self._hooks)} DeltaNet layers")

    def get_signals(self) -> dict[int, SegmentSignals]:
        return self._signals

    def clear(self):
        self._signals.clear()

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._signals.clear()


class _DeltaNetStreamHook:
    """
    Forward pre-hook that computes per-segment saturation signals
    without retaining full-sequence tensors.
    """

    def __init__(self, layer_idx: int, signals_dict: dict, segment_length: int):
        self.layer_idx = layer_idx
        self.signals_dict = signals_dict
        self.segment_length = segment_length

    def __call__(self, module, args, kwargs):
        # Pre-hook with kwargs: extract hidden_states
        hidden_states = kwargs.get('hidden_states')
        if hidden_states is None and len(args) > 0:
            hidden_states = args[0]
        if hidden_states is None:
            return
        B, T, _ = hidden_states.shape
        seg_len = self.segment_length

        with torch.no_grad():
            # --- Re-derive beta and g (cheap: just linear projections) ---
            b = module.in_proj_b(hidden_states)  # [B, T, num_heads]
            a = module.in_proj_a(hidden_states)  # [B, T, num_heads]
            beta = b.sigmoid()
            g = -module.A_log.float().exp() * F.softplus(a.float() + module.dt_bias)

            # --- Get post-conv key (actual write direction) ---
            mixed_qkv = module.in_proj_qkv(hidden_states)  # [B, T, conv_dim]
            mixed_qkv_t = mixed_qkv.transpose(1, 2)  # [B, conv_dim, T]

            # Apply causal conv1d (same as model forward)
            if module.causal_conv1d_fn is not None:
                mixed_qkv_conv = module.causal_conv1d_fn(
                    x=mixed_qkv_t,
                    weight=module.conv1d.weight.squeeze(1),
                    bias=module.conv1d.bias,
                    activation=module.activation,
                    seq_idx=None,
                )
            else:
                mixed_qkv_conv = F.silu(module.conv1d(mixed_qkv_t)[:, :, :T])

            mixed_qkv_conv = mixed_qkv_conv.transpose(1, 2)  # [B, T, conv_dim]

            key_dim = module.key_dim
            value_dim = module.value_dim
            _, key_post, value_post = torch.split(
                mixed_qkv_conv, [key_dim, key_dim, value_dim], dim=-1
            )
            # Reshape to heads
            key_post = key_post.reshape(B, T, module.num_k_heads, module.head_k_dim)
            # L2 normalize key (model uses use_qk_l2norm_in_kernel=True)
            key_norm = F.normalize(key_post, dim=-1)  # [B, T, H_k, D_k]

            # If num_v_heads > num_k_heads, repeat key to match
            if module.num_v_heads > module.num_k_heads:
                repeat_factor = module.num_v_heads // module.num_k_heads
                key_norm = key_norm.repeat_interleave(repeat_factor, dim=2)
            # key_norm: [B, T, H_v, D_k] — the actual write direction

            # --- Compute write magnitude: ||beta * key|| per token per head ---
            # beta: [B, T, H_v], key_norm: [B, T, H_v, D_k]
            write_mag = beta.unsqueeze(-1) * key_norm  # [B, T, H_v, D_k]
            write_norm = write_mag.norm(dim=-1)  # [B, T, H_v]

            # --- Historical positive decay-magnitude proxy tau = -g > 0 ---
            retention = (-g).clamp(min=1e-8)  # [B, T, H_v], positive

            # --- Per-segment aggregation (no full-seq storage) ---
            rho_max_list = []
            c_max_list = []
            g_mag_min_list = []

            n_segs = (T + seg_len - 1) // seg_len
            token_collision = torch.zeros(
                B, T, module.num_v_heads,
                device=hidden_states.device,
                dtype=key_norm.dtype,
            )
            max_lookback = min(8, T - 1)
            for offset in range(1, max_lookback + 1):
                cos = (key_norm[:, offset:, :, :] * key_norm[:, :-offset, :, :]).sum(dim=-1)
                token_collision[:, offset:, :] = torch.maximum(token_collision[:, offset:, :], cos)

            for s in range(n_segs):
                start = s * seg_len
                end = min(start + seg_len, T)
                if end - start < seg_len // 4:
                    break  # skip tiny trailing segment

                # rho: max write_norm / retention in this segment
                seg_write = write_norm[:, start:end, :]  # [B, seg, H]
                seg_retain = retention[:, start:end, :]
                seg_rho = (seg_write / seg_retain).amax(dim=(1, 2))  # [B]
                rho_max_list.append(seg_rho.mean().item())

                # g_mag_min: minimum retention in segment (= where decay pressure is strongest)
                seg_g_min = seg_retain.amin(dim=(1, 2))  # [B]
                g_mag_min_list.append(seg_g_min.mean().item())

                seg_c = token_collision[:, start:end, :].amax(dim=(1, 2))  # [B]
                c_max_list.append(seg_c.mean().item())

            # Free large intermediates
            del mixed_qkv, mixed_qkv_t, mixed_qkv_conv, key_post, key_norm
            del write_mag, write_norm, beta, g, retention, b, a, token_collision

        self.signals_dict[self.layer_idx] = SegmentSignals(
            layer_idx=self.layer_idx,
            rho_max=rho_max_list,
            c_max=c_max_list,
            g_mag_min=g_mag_min_list,
        )


class KDAHookManager:
    """
    Hook manager for Kimi-Linear's KDA (Key-Decay Attention) layers.
    Analogous to DeltaNetHookManager but adapted for Kimi's projection names.
    """

    def __init__(self, model, kda_layer_indices: list[int],
                 segment_length: int = 512):
        self.model = model
        self.kda_layer_indices = kda_layer_indices
        self.segment_length = segment_length
        self._hooks = []
        self._signals: dict[int, SegmentSignals] = {}

    def _get_kda_module(self, layer_idx: int):
        return self.model.model.layers[layer_idx].self_attn

    def attach(self):
        self.remove()
        for layer_idx in self.kda_layer_indices:
            module = self._get_kda_module(layer_idx)
            hook = _KDAStreamHook(layer_idx, self._signals, self.segment_length)
            handle = module.register_forward_pre_hook(hook, with_kwargs=True)
            self._hooks.append(handle)
        logger.info(f"Attached KDA hooks to {len(self._hooks)} layers")

    def get_signals(self) -> dict[int, SegmentSignals]:
        return self._signals

    def clear(self):
        self._signals.clear()

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._signals.clear()


class _KDAStreamHook:
    """Forward pre-hook for Kimi KDA layers. Same signal extraction as DeltaNet."""

    def __init__(self, layer_idx: int, signals_dict: dict, segment_length: int):
        self.layer_idx = layer_idx
        self.signals_dict = signals_dict
        self.segment_length = segment_length

    def __call__(self, module, args, kwargs):
        hidden_states = kwargs.get('hidden_states')
        if hidden_states is None and len(args) > 0:
            hidden_states = args[0]
        if hidden_states is None:
            return
        B, T, _ = hidden_states.shape
        seg_len = self.segment_length

        with torch.no_grad():
            beta = module.b_proj(hidden_states).float().sigmoid()  # [B, T, num_heads]

            g_raw = module.f_b_proj(module.f_a_proj(hidden_states))
            from fla.ops.kda.gate import fused_kda_gate
            g = fused_kda_gate(
                g_raw.reshape(B, T, module.num_k_heads, module.head_k_dim),
                module.A_log,
                dt_bias=module.dt_bias,
            )
            # g: [B, T, H, K], negative values (decay)

            key_pre = module.k_proj(hidden_states)  # [B, T, proj_k_size]
            key_post, _ = module.k_conv1d(
                x=key_pre,
                cache=None,
                output_final_state=False,
            )
            key_post = key_post.reshape(B, T, module.num_k_heads, module.head_k_dim)
            key_norm = F.normalize(key_post, dim=-1)

            # write_mag: beta [B,T,H] * key_norm [B,T,H,K] → [B,T,H,K]
            write_mag = beta.unsqueeze(-1) * key_norm
            write_norm = write_mag.norm(dim=-1)  # [B, T, H]

            # retention: mean across K dim → [B, T, H]
            retention = (-g).clamp(min=1e-8).mean(dim=-1)  # [B, T, H]

            rho_max_list = []
            c_max_list = []
            g_mag_min_list = []

            n_segs = (T + seg_len - 1) // seg_len
            token_collision = torch.zeros(
                B, T, module.num_k_heads,
                device=hidden_states.device, dtype=key_norm.dtype,
            )
            max_lookback = min(8, T - 1)
            for offset in range(1, max_lookback + 1):
                cos = (key_norm[:, offset:, :, :] * key_norm[:, :-offset, :, :]).sum(dim=-1)
                token_collision[:, offset:, :] = torch.maximum(token_collision[:, offset:, :], cos)

            for s in range(n_segs):
                start = s * seg_len
                end = min(start + seg_len, T)
                if end - start < seg_len // 4:
                    break

                seg_write = write_norm[:, start:end, :]
                seg_retain = retention[:, start:end, :]
                seg_rho = (seg_write / seg_retain).amax(dim=(1, 2))
                rho_max_list.append(seg_rho.mean().item())

                seg_g_min = seg_retain.amin(dim=(1, 2))
                g_mag_min_list.append(seg_g_min.mean().item())

                seg_c = token_collision[:, start:end, :].amax(dim=(1, 2))
                c_max_list.append(seg_c.mean().item())

            del key_pre, key_post, key_norm, write_mag, write_norm
            del beta, g, g_raw, retention, token_collision

        self.signals_dict[self.layer_idx] = SegmentSignals(
            layer_idx=self.layer_idx,
            rho_max=rho_max_list,
            c_max=c_max_list,
            g_mag_min=g_mag_min_list,
        )
