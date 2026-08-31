"""
E5: Cross-Family Validation on Kimi-Linear-48B-A3B
===================================================
Claim: HMO generalizes beyond Qwen3.5 to other hybrid-attention architectures.

Model: Kimi-Linear-48B-A3B-Instruct-GPTQ-Int4 (KDA + MLA, 20:7 ratio)
Data: 80% HotpotQA + 20% Needle, 100 samples, 8K
Methods: full_kv / h2o / hmo_full

CRITICAL: Formal E5 now targets a single large-memory GPU (e.g. single A100).
Kimi must load fully onto that one CUDA device with no CPU/disk offload.
DO NOT load to CPU first — compressed-tensors decompression must still happen on GPU.

Usage:
    python experiments/phase2/e5_kimi/run.py
"""
import sys
import json
import gc
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from loguru import logger
from datetime import datetime
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.utils.model_loader import (
    load_model_and_tokenizer,
    get_linear_attention_indices,
    get_full_attention_indices,
)
from experiments.utils.hooks import KDAHookManager, SegmentSignals
from experiments.utils.saturation import compute_segment_saturation
from experiments.utils.dataset_utils import make_needle_samples, load_longbench_subset
from experiments.utils.metrics import reset_vram_stats, get_peak_vram_mb
from experiments.utils.eval_harness import build_prompt, score_prediction, resolve_max_new_tokens
from experiments.phase2.runner import (
    get_results_dir, save_cell, ExperimentCell,
    get_primary_metric_name, get_primary_score,
)


@dataclass
class KimiMethodResult:
    generated_text: str
    tracked_bytes: int = 0
    budget_limit_bytes: int = 0
    n_kept_kv: int = 0
    n_skeleton: int = 0
    n_refresh: int = 0
    n_dropped: int = 0


def annotate_kimi_module_names(model) -> None:
    """
    Annotate remote-code Kimi modules with stable names for runtime diagnostics.

    This is a pure debugging aid for the family-specific compressed_tensors path.
    It does not alter the model computation or HMO semantics.
    """
    for name, module in model.named_modules(remove_duplicate=True):
        setattr(module, "_hmo_name", name)


def patch_kimi_decompress_debug_logging() -> None:
    """
    Add module-aware OOM diagnostics around compressed_tensors decompression.

    Kimi's first forward triggers on-demand decompression via compressed_tensors.
    When that path OOMs, the default traceback does not identify which module was
    being unpacked. This patch preserves behavior and only augments error logs.
    """
    try:
        import compressed_tensors.compressors as ct_compressors
        import compressed_tensors.compressors.model_compressors.model_compressor as ct_model_compressor
    except Exception as exc:
        logger.warning(f"Could not import compressed_tensors for Kimi debug patch: {exc}")
        return

    if getattr(ct_compressors, "_hmo_kimi_decompress_debug", False):
        return

    orig_decompress_module = ct_compressors.decompress_module

    def _wrapped_decompress_module(module, *args, **kwargs):
        try:
            return orig_decompress_module(module, *args, **kwargs)
        except torch.cuda.OutOfMemoryError:
            module_name = getattr(module, "_hmo_name", "<unnamed>")
            module_type = type(module).__name__
            tensor_device = None
            tensor_shape = None
            for attr_name in ("weight", "weight_packed", "weight_scale", "bias"):
                value = getattr(module, attr_name, None)
                if isinstance(value, torch.Tensor):
                    tensor_device = str(value.device)
                    tensor_shape = tuple(value.shape)
                    break
            logger.error(
                "Kimi compressed_tensors OOM while decompressing "
                f"module={module_name}, type={module_type}, tensor_device={tensor_device}, "
                f"tensor_shape={tensor_shape}"
            )
            raise

    ct_compressors.decompress_module = _wrapped_decompress_module
    ct_model_compressor.decompress_module = _wrapped_decompress_module
    ct_compressors._hmo_kimi_decompress_debug = True
    logger.info("Patched compressed_tensors decompression with module-aware Kimi OOM diagnostics")


def parse_args():
    p = argparse.ArgumentParser(description="E5: Cross-Family Validation")
    p.add_argument("--model", type=str, default="kimi-linear-48b-gptq-int4")
    p.add_argument("--n_samples", type=int, default=100)
    p.add_argument("--context_length", type=int, default=8192)
    p.add_argument("--segment_length", type=int, default=512)
    p.add_argument("--keep_ratio", type=float, default=0.5)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--methods", type=str, default="full_kv,h2o,hmo_full")
    return p.parse_args()


def build_samples(tokenizer, args):
    n_lb = int(args.n_samples * 0.8)
    n_needle = args.n_samples - n_lb
    samples = []
    try:
        lb = load_longbench_subset("hotpotqa", tokenizer, n_lb, args.context_length, args.seed)
        if len(lb) < n_lb:
            fallback_targets = []
            for alt_ctx in [max(4096, args.context_length // 2), 4096, 2048]:
                if alt_ctx != args.context_length and alt_ctx not in fallback_targets:
                    fallback_targets.append(alt_ctx)

            seen_ids = {sample.sample_id for sample in lb}
            for alt_ctx in fallback_targets:
                if len(lb) >= n_lb:
                    break
                needed = n_lb - len(lb)
                logger.warning(
                    f"E5 only found {len(lb)}/{n_lb} HotpotQA samples near {args.context_length} tokens; "
                    f"retrying with relaxed target {alt_ctx} for the remaining {needed}"
                )
                extra = load_longbench_subset("hotpotqa", tokenizer, needed, alt_ctx, args.seed + alt_ctx)
                for sample in extra:
                    if sample.sample_id not in seen_ids:
                        lb.append(sample)
                        seen_ids.add(sample.sample_id)
                        if len(lb) >= n_lb:
                            break
        samples.extend(lb[:n_lb])
    except Exception as e:
        logger.warning(f"LongBench load failed: {e}")
    samples.extend(make_needle_samples(tokenizer, n_needle, args.context_length, args.seed))
    if len(samples) < args.n_samples:
        extra = make_needle_samples(tokenizer, args.n_samples - len(samples), args.context_length, args.seed + 1000)
        samples.extend(extra)
    return samples


def build_input_ids(sample, tokenizer, device, max_length=None):
    """Build input_ids from the unified evaluation harness prompt."""
    prompt = build_prompt(sample, tokenizer)
    kwargs = {"return_tensors": "pt", "truncation": True}
    if max_length is not None:
        kwargs["max_length"] = max_length
    inputs = tokenizer(prompt, **kwargs)
    return inputs["input_ids"].to(device)


def _normalize_cuda_device(dev) -> int | None:
    """Normalize an hf_device_map entry into a CUDA ordinal when possible."""
    if isinstance(dev, int):
        return dev
    if isinstance(dev, torch.device):
        if dev.type == "cuda":
            return 0 if dev.index is None else int(dev.index)
        return None
    if isinstance(dev, str):
        if dev.startswith("cuda:"):
            try:
                return int(dev.split(":")[1])
            except Exception:
                return None
        if dev == "cuda":
            return 0
    return None


def resolve_kimi_input_device(model) -> torch.device:
    """
    Pick the device where user input tensors should be placed for a sharded model.

    For Accelerate-loaded models, relying on `model.device` is unsafe: it may
    report CPU/meta or hide the actual CUDA placement. We resolve the first CUDA
    shard explicitly and fail fast if the runtime spills any module to CPU/disk.
    The formal E5 setup now assumes single-GPU loading on one large-memory A100,
    but the logic remains compatible with multi-shard placements as long as they
    stay fully on CUDA devices.
    """
    device_map = getattr(model, "hf_device_map", None)
    if not device_map:
        model_device = getattr(model, "device", None)
        if isinstance(model_device, torch.device) and model_device.type == "cuda":
            logger.info(f"E5 Kimi input tensors will use model.device={model_device}")
            return model_device
        raise RuntimeError(
            "Kimi model was loaded without hf_device_map metadata and without a CUDA model.device; "
            "cannot determine where to place inputs safely."
        )

    cpu_offload = {}
    cuda_devices: set[int] = set()
    for module_name, target in device_map.items():
        target_str = str(target)
        cuda_idx = _normalize_cuda_device(target)
        if cuda_idx is not None:
            cuda_devices.add(cuda_idx)
        elif target_str.startswith("cpu") or target_str.startswith("disk"):
            cpu_offload[module_name] = target_str

    logger.info(
        f"E5 Kimi placement: cuda_shards={sorted(cuda_devices)}, cpu_offload_modules={len(cpu_offload)}"
    )
    if cpu_offload:
        preview = ", ".join(list(cpu_offload.keys())[:4])
        raise RuntimeError(
            "E5 refused to run because Kimi was partially offloaded to CPU/disk "
            f"(example modules: {preview}). Free the target CUDA device and relaunch."
        )
    if len(cuda_devices) < 1:
        raise RuntimeError(
            f"E5 requires the Kimi model to stay on CUDA, but got cuda_shards={sorted(cuda_devices)}."
        )

    preferred_keys = [
        "model.embed_tokens",
        "model.layers.0",
        "model.layers.1",
        "",
    ]
    for key in preferred_keys:
        target = device_map.get(key)
        cuda_idx = _normalize_cuda_device(target)
        if cuda_idx is not None:
            logger.info(f"E5 Kimi input tensors will be placed on {key} device cuda:{cuda_idx}")
            return torch.device(f"cuda:{cuda_idx}")

    fallback = min(cuda_devices)
    logger.warning(
        f"E5 could not find an explicit embedding/early-layer shard in hf_device_map; "
        f"falling back to cuda:{fallback} for inputs"
    )
    return torch.device(f"cuda:{fallback}")


def patch_kimi_cache_api_compat(model) -> None:
    """
    Patch Kimi's custom cache class to tolerate the cache_position API used by
    the current transformers version.

    The remote-code KimiDynamicCache implementation expects `cache_position` to
    be a tensor, but newer masking_utils may pass an integer query length into
    `get_mask_sizes()`. This is a family-specific runtime adapter, not a change
    to HMO semantics.
    """
    kimi_module = sys.modules.get(type(model.model).__module__)
    if kimi_module is None or not hasattr(kimi_module, "KimiDynamicCache"):
        return

    cache_cls = kimi_module.KimiDynamicCache
    if getattr(cache_cls, "_hmo_mask_size_compat", False):
        return

    orig_get_mask_sizes = cache_cls.get_mask_sizes

    def _compat_get_mask_sizes(self, cache_position, layer_idx):
        if isinstance(cache_position, int):
            kv_offset = 0
            query_length = cache_position
            past_seen_tokens = self.get_seq_length(layer_idx)
            kv_length = query_length + past_seen_tokens
            return kv_length, kv_offset
        return orig_get_mask_sizes(self, cache_position, layer_idx)

    cache_cls.get_mask_sizes = _compat_get_mask_sizes
    cache_cls._hmo_mask_size_compat = True
    logger.info("Patched KimiDynamicCache.get_mask_sizes for transformers cache-position compatibility")


def patch_kimi_kda_runtime_compat(model) -> None:
    """
    Force Kimi's KDA layers onto the recurrent execution path.

    On RTX 5090 / sm_120, the FLA `chunk_kda` kernel currently fails to compile,
    while the recurrent path is semantically aligned with the same KDA recurrence
    and is already used by the model for short-query decode steps. For E5 we use
    the recurrent backend consistently to preserve theory-level behavior while
    avoiding the unsupported chunk kernel.
    """
    kimi_module = sys.modules.get(type(model.model).__module__)
    if kimi_module is None or not hasattr(kimi_module, "KimiDeltaAttention"):
        return

    attn_cls = kimi_module.KimiDeltaAttention
    patched = 0
    for module in model.modules():
        if isinstance(module, attn_cls) and getattr(module, "mode", None) != "fused_recurrent":
            module.mode = "fused_recurrent"
            patched += 1

    if patched:
        logger.info(
            f"Patched {patched} KimiDeltaAttention layers to use fused_recurrent KDA "
            "for RTX 5090 runtime compatibility"
        )


def remove_lingering_kimi_decompress_hook(model, where: str) -> None:
    """
    compressed_tensors should remove ct_decompress_hook after the first forward,
    but on this Kimi checkpoint it can linger and trigger a second full-model
    decompression on the next sample, causing KeyError('weight_packed').
    """
    removed = 0
    for _, module in model.named_modules(remove_duplicate=True):
        if hasattr(module, "ct_decompress_hook"):
            module.ct_decompress_hook.remove()
            delattr(module, "ct_decompress_hook")
            removed += 1

        hook_ids = []
        for hook_id, hook in list(module._forward_pre_hooks.items()):
            hook_name = getattr(hook, "__name__", "") or getattr(hook, "__qualname__", "")
            if "ct_decompress_hook" in hook_name:
                hook_ids.append(hook_id)
        for hook_id in hook_ids:
            del module._forward_pre_hooks[hook_id]
            removed += 1

    if removed:
        logger.warning(f"Removed {removed} lingering compressed-tensors pre-hooks after {where}")


def should_cleanup_kimi_decompress_hooks(model) -> bool:
    """
    The first full forward still needs the checkpoint's remaining on-demand
    decompression hook for some MoE expert weights. Cleanup is only safe after
    the first successful forward has completed.
    """
    return bool(getattr(model, "_hmo_kimi_first_forward_done", False))


def estimate_kimi_budget_limit_bytes(full_cache_bytes: int, seq_len: int, keep_tokens: int) -> int:
    """Translate a token keep budget into KV bytes under uniform per-token storage."""
    if seq_len <= 0:
        return 0
    bytes_per_token = full_cache_bytes / float(seq_len)
    return int(bytes_per_token * keep_tokens)


def get_cache_seq_len(cache, mla_indices) -> int:
    """Use the first populated MLA layer as the representative active KV length."""
    for layer_idx in (mla_indices or []):
        k = cache.key_cache[layer_idx]
        if isinstance(k, torch.Tensor):
            return int(k.shape[2])
    return 0


def get_kimi_cache_bytes(cache, mla_indices) -> int:
    """Count active MLA KV bytes."""
    total = 0
    for layer_idx in (mla_indices or []):
        k = cache.key_cache[layer_idx]
        v = cache.value_cache[layer_idx]
        if isinstance(k, torch.Tensor):
            total += int(k.numel() * k.element_size())
        if isinstance(v, torch.Tensor):
            total += int(v.numel() * v.element_size())
    return total


def reset_visible_vram_stats() -> None:
    """Reset peak VRAM tracking on every visible CUDA device."""
    for device_id in range(torch.cuda.device_count()):
        reset_vram_stats(device_id)


def get_total_peak_vram_mb() -> float:
    """Report total peak VRAM across all visible CUDA devices."""
    return float(sum(get_peak_vram_mb(device_id) for device_id in range(torch.cuda.device_count())))


# ---------------------------------------------------------------------------
# Kimi-specific generation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def kimi_prefill(model, input_ids, output_attentions=False):
    """Prefill and return cache + last logits."""
    if should_cleanup_kimi_decompress_hooks(model):
        remove_lingering_kimi_decompress_hook(model, "kimi_prefill_pre")
    outputs = model(
        input_ids,
        use_cache=True,
        output_attentions=output_attentions,
        return_dict=True,
    )
    model._hmo_kimi_first_forward_done = True
    remove_lingering_kimi_decompress_hook(model, "kimi_prefill")
    return outputs, outputs.past_key_values, outputs.logits[:, -1:, :]


@torch.no_grad()
def kimi_decode_loop(model, cache, last_logits, tokenizer, max_new_tokens=64):
    """Greedy decode loop using Kimi's cache."""
    generated_ids = []
    next_id = last_logits[:, -1, :].argmax(dim=-1, keepdim=True)
    eos_id = tokenizer.eos_token_id

    for _ in range(max_new_tokens):
        generated_ids.append(next_id.item())
        if next_id.item() == eos_id:
            break
        if should_cleanup_kimi_decompress_hooks(model):
            remove_lingering_kimi_decompress_hook(model, "kimi_decode_loop_pre")
        outputs = model(next_id, past_key_values=cache, use_cache=True)
        remove_lingering_kimi_decompress_hook(model, "kimi_decode_loop_post")
        cache = outputs.past_key_values
        next_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Method: Full KV (upper bound)
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_full_kv(model, tokenizer, input_ids, max_new_tokens, mla_indices=None):
    _, cache, last_logits = kimi_prefill(model, input_ids)
    generated = kimi_decode_loop(model, cache, last_logits, tokenizer, max_new_tokens)
    tracked_bytes = get_kimi_cache_bytes(cache, mla_indices)
    return KimiMethodResult(
        generated_text=generated,
        tracked_bytes=tracked_bytes,
        budget_limit_bytes=tracked_bytes,
        n_kept_kv=get_cache_seq_len(cache, mla_indices),
    )


# ---------------------------------------------------------------------------
# Method: H2O baseline (attention-score-based eviction on MLA layers)
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_h2o(model, tokenizer, input_ids, max_new_tokens, keep_ratio=0.5,
            mla_indices=None):
    """H2O: keep top-k tokens by cumulative attention score on MLA layers."""
    _, cache, last_logits = kimi_prefill(model, input_ids)
    seq_len = input_ids.shape[1]
    n_keep = max(64, int(seq_len * keep_ratio))
    full_cache_bytes = get_kimi_cache_bytes(cache, mla_indices)
    budget_limit_bytes = estimate_kimi_budget_limit_bytes(full_cache_bytes, seq_len, n_keep)

    for layer_idx in (mla_indices or []):
        k = cache.key_cache[layer_idx]
        v = cache.value_cache[layer_idx]
        if k is None:
            continue
        # Use KV norm as importance proxy (same as Qwen H2O)
        kv_norm = k.norm(dim=-1).mean(dim=1).squeeze(0)  # [T]
        actual_keep = min(n_keep, kv_norm.shape[0])
        _, keep_idx = kv_norm.topk(actual_keep, sorted=True)
        keep_idx = keep_idx.sort().values
        cache.key_cache[layer_idx] = k[:, :, keep_idx, :]
        cache.value_cache[layer_idx] = v[:, :, keep_idx, :]

    generated = kimi_decode_loop(model, cache, last_logits, tokenizer, max_new_tokens)
    tracked_bytes = get_kimi_cache_bytes(cache, mla_indices)
    kept_tokens = get_cache_seq_len(cache, mla_indices)
    return KimiMethodResult(
        generated_text=generated,
        tracked_bytes=tracked_bytes,
        budget_limit_bytes=budget_limit_bytes,
        n_kept_kv=kept_tokens,
    )


# ---------------------------------------------------------------------------
# Method: HMO-full (sigma from KDA hooks + alpha from MLA attention + eviction)
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_kimi_alpha(model, input_ids, mla_indices, segment_length):
    """Collect attention fragility (alpha) from MLA layers via KV norm proxy.

    Kimi's MLA doesn't expose attention weights, so we use per-segment
    KV norm as a proxy for attention importance (higher norm = more attended).
    """
    _, cache, _ = kimi_prefill(model, input_ids)
    seq_len = input_ids.shape[1]
    n_segs = (seq_len + segment_length - 1) // segment_length

    seg_scores = np.zeros(n_segs)
    n_layers = 0

    for layer_idx in mla_indices:
        k = cache.key_cache[layer_idx]
        if not isinstance(k, torch.Tensor):
            continue
        # KV norm per token, averaged across heads: [T]
        kv_importance = k.norm(dim=-1).mean(dim=1).squeeze(0).cpu().numpy()
        for s in range(n_segs):
            start = s * segment_length
            end = min(start + segment_length, seq_len)
            if end <= kv_importance.shape[0]:
                seg_scores[s] += kv_importance[start:end].sum()
        n_layers += 1

    if n_layers > 0:
        seg_scores /= n_layers
    total = seg_scores.sum()
    if total > 0:
        seg_scores /= total

    del cache
    torch.cuda.empty_cache()
    return seg_scores


@torch.no_grad()
def collect_kimi_sigma(model, hook_mgr, input_ids, segment_length):
    """Collect sigma from KDA hooks."""
    hook_mgr.clear()
    hook_mgr.attach()
    try:
        _, cache, _ = kimi_prefill(model, input_ids)
        signals = dict(hook_mgr.get_signals())
    finally:
        hook_mgr.remove()
    del cache
    torch.cuda.empty_cache()

    sigma = compute_segment_saturation(
        signals,
        alpha_rho=0.4, alpha_c=0.3, alpha_g=0.3,
        segment_length=segment_length,
        warmup_tokens=segment_length * 2,
        input_ids=input_ids,
    )
    return sigma


@torch.no_grad()
def run_hmo_full(model, tokenizer, input_ids, max_new_tokens, hook_mgr,
                 mla_indices, segment_length=512, keep_ratio=0.5,
                 refresh_budget=3):
    """HMO-full: phi=sigma*alpha guided refresh + token-pruning on MLA layers.

    Two prefills: (1) with hooks for sigma + alpha from cache, (2) clean for generation.
    """
    seq_len = input_ids.shape[1]
    n_segs = (seq_len + segment_length - 1) // segment_length

    # Prefill 1: collect sigma (hooks) + alpha (KV norm from same cache)
    hook_mgr.clear()
    hook_mgr.attach()
    try:
        _, cache1, _ = kimi_prefill(model, input_ids)
        signals = dict(hook_mgr.get_signals())
    finally:
        hook_mgr.remove()

    sigma = compute_segment_saturation(
        signals,
        alpha_rho=0.4, alpha_c=0.3, alpha_g=0.3,
        segment_length=segment_length,
        warmup_tokens=segment_length * 2,
        input_ids=input_ids,
    )

    # Alpha from KV norm of MLA layers in cache1
    seg_scores = np.zeros(n_segs)
    n_layers = 0
    for layer_idx in mla_indices:
        k = cache1.key_cache[layer_idx]
        if not isinstance(k, torch.Tensor):
            continue
        kv_importance = k.norm(dim=-1).mean(dim=1).squeeze(0).cpu().numpy()
        for s in range(n_segs):
            start = s * segment_length
            end = min(start + segment_length, seq_len)
            if end <= kv_importance.shape[0]:
                seg_scores[s] += kv_importance[start:end].sum()
        n_layers += 1
    if n_layers > 0:
        seg_scores /= n_layers
    total = seg_scores.sum()
    alpha = seg_scores / total if total > 0 else seg_scores

    del cache1
    torch.cuda.empty_cache()

    n = min(len(sigma), len(alpha), n_segs)
    sigma = sigma[:n]
    alpha = alpha[:n]
    phi = sigma * alpha

    # Decide actions: top-K phi segments get refresh, rest get token-pruning
    protected = {0, n - 1} if n > 1 else {0}
    middle = [i for i in range(n) if i not in protected]

    phi_middle = [(phi[i], i) for i in middle]
    phi_middle.sort(reverse=True)
    refresh_segs = set()
    for _, seg_idx in phi_middle[:refresh_budget]:
        refresh_segs.add(seg_idx)

    # Prefill 2: clean cache for generation
    _, cache, last_logits = kimi_prefill(model, input_ids)
    n_keep = max(64, int(seq_len * keep_ratio))
    full_cache_bytes = get_kimi_cache_bytes(cache, mla_indices)
    budget_limit_bytes = estimate_kimi_budget_limit_bytes(full_cache_bytes, seq_len, n_keep)

    # Apply token-pruning on MLA layers (skip refresh segments, keep protected)
    for layer_idx in mla_indices:
        k = cache.key_cache[layer_idx]
        v = cache.value_cache[layer_idx]
        if not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor):
            continue

        kv_norm = k.norm(dim=-1).mean(dim=1).squeeze(0)  # [T]
        keep_mask = torch.zeros(seq_len, dtype=torch.bool, device=k.device)

        for s in range(n):
            start = s * segment_length
            end = min(start + segment_length, seq_len)
            if s in protected or s in refresh_segs:
                keep_mask[start:end] = True
            else:
                seg_norms = kv_norm[start:end]
                seg_keep = max(4, int((end - start) * keep_ratio))
                if seg_keep < len(seg_norms):
                    _, top_idx = seg_norms.topk(seg_keep)
                    keep_mask[start + top_idx] = True
                else:
                    keep_mask[start:end] = True

        actual_keep = min(n_keep, keep_mask.sum().item())
        keep_idx = keep_mask.nonzero(as_tuple=True)[0]
        if len(keep_idx) > actual_keep:
            keep_idx = keep_idx[:actual_keep]

        cache.key_cache[layer_idx] = k[:, :, keep_idx, :]
        cache.value_cache[layer_idx] = v[:, :, keep_idx, :]

    generated = kimi_decode_loop(model, cache, last_logits, tokenizer, max_new_tokens)
    tracked_bytes = get_kimi_cache_bytes(cache, mla_indices)
    kept_tokens = get_cache_seq_len(cache, mla_indices)
    n_refresh = len(refresh_segs)
    n_skeleton = max(0, n - len(protected) - n_refresh)
    return KimiMethodResult(
        generated_text=generated,
        tracked_bytes=tracked_bytes,
        budget_limit_bytes=budget_limit_bytes,
        n_kept_kv=kept_tokens,
        n_skeleton=n_skeleton,
        n_refresh=n_refresh,
        n_dropped=0,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    logger.info(f"E5 Cross-Family — {args.model}, n={args.n_samples}, ctx={args.context_length}")

    model, tokenizer, config = load_model_and_tokenizer(
        args.model, device="cuda", gpu_id=args.gpu_id,
    )
    annotate_kimi_module_names(model)
    patch_kimi_decompress_debug_logging()
    patch_kimi_cache_api_compat(model)
    patch_kimi_kda_runtime_compat(model)
    input_device = resolve_kimi_input_device(model)

    kda_indices = get_linear_attention_indices(config)
    mla_indices = get_full_attention_indices(config)
    logger.info(f"Kimi-Linear: {len(kda_indices)} KDA + {len(mla_indices)} MLA layers")

    hook_mgr = KDAHookManager(model, kda_indices, segment_length=args.segment_length)

    samples = build_samples(tokenizer, args)
    methods = args.methods.split(",")
    results_dir = get_results_dir("e5_kimi")
    output_path = results_dir / "e5_kimi.jsonl"

    completed = set()
    if args.resume and output_path.exists():
        with open(output_path) as f:
            for line in f:
                row = json.loads(line.strip())
                if not row.get("error"):
                    completed.add((row["method"], row["sample_id"]))

    for i, sample in enumerate(samples):
        for method in methods:
            if (method, sample.sample_id) in completed:
                continue
            logger.info(f"  [{method}] {i+1}/{len(samples)}: {sample.sample_id}")
            if should_cleanup_kimi_decompress_hooks(model):
                remove_lingering_kimi_decompress_hook(model, "sample_start")
            torch.cuda.empty_cache()
            gc.collect()
            reset_visible_vram_stats()

            try:
                input_ids = build_input_ids(
                    sample, tokenizer, input_device,
                    max_length=args.context_length + 256,
                )
                max_new = resolve_max_new_tokens(sample.dataset, args.max_new_tokens)

                if method == "full_kv":
                    result = run_full_kv(
                        model, tokenizer, input_ids, max_new, mla_indices=mla_indices,
                    )
                elif method == "h2o":
                    result = run_h2o(
                        model, tokenizer, input_ids, max_new,
                        keep_ratio=args.keep_ratio, mla_indices=mla_indices,
                    )
                elif method == "hmo_full":
                    result = run_hmo_full(
                        model, tokenizer, input_ids, max_new, hook_mgr,
                        mla_indices=mla_indices,
                        segment_length=args.segment_length,
                        keep_ratio=args.keep_ratio,
                    )
                else:
                    raise ValueError(f"Unknown method: {method}")

                gen = result.generated_text
                accuracy, f1, rouge_l = score_prediction(gen, sample)
                primary_metric = get_primary_metric_name(sample.dataset)
                primary_score = get_primary_score(sample.dataset, accuracy, f1, rouge_l)
                cell = ExperimentCell(
                    experiment="e5_kimi", method=method,
                    dataset=sample.dataset, context_length=args.context_length,
                    sample_id=sample.sample_id,
                    accuracy=accuracy, f1=f1, rouge_l=rouge_l,
                    primary_metric=primary_metric, primary_score=primary_score,
                    peak_vram_mb=get_total_peak_vram_mb(),
                    tracked_bytes=result.tracked_bytes,
                    budget_limit_bytes=result.budget_limit_bytes,
                    n_kept_kv=result.n_kept_kv,
                    n_skeleton=result.n_skeleton,
                    n_refresh=result.n_refresh,
                    n_dropped=result.n_dropped,
                    generated_text=gen[:300],
                    answer=sample.answer[:300],
                )
            except torch.cuda.OutOfMemoryError:
                logger.exception(
                    f"OOM during E5 {method} on {sample.sample_id}; "
                    f"peak_total_vram_mb={get_total_peak_vram_mb():.1f}"
                )
                torch.cuda.empty_cache()
                cell = ExperimentCell(
                    experiment="e5_kimi", method=method,
                    dataset=sample.dataset, context_length=args.context_length,
                    sample_id=sample.sample_id,
                    error=f"OOM(total_peak_vram_mb={get_total_peak_vram_mb():.1f})",
                )
            except Exception as e:
                logger.exception(f"Error: {e}")
                cell = ExperimentCell(
                    experiment="e5_kimi", method=method,
                    dataset=sample.dataset, context_length=args.context_length,
                    sample_id=sample.sample_id, error=str(e)[:200],
                )
            save_cell(cell, output_path)

    # Summary
    from collections import defaultdict
    groups = defaultdict(list)
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                row = json.loads(line.strip())
                if not row.get("error"):
                    groups[row["method"]].append(row.get("primary_score", row["accuracy"]))

    logger.info("\n" + "=" * 60)
    logger.info("E5 SUMMARY")
    summary = {}
    for name in sorted(groups):
        scores = np.array(groups[name])
        summary[name] = {"mean": float(scores.mean()), "std": float(scores.std()), "n": len(scores)}
        logger.info(f"  {name}: score={scores.mean():.3f}±{scores.std():.3f} (n={len(scores)})")

    summary_path = results_dir / "e5_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"summary": summary, "timestamp": datetime.now().isoformat()}, f, indent=2)
    logger.info(f"E5 complete. Summary: {summary_path}")


if __name__ == "__main__":
    main()
