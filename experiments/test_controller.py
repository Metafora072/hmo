"""
Quick smoke test for HMO Controller.
Loads Qwen3.5-0.8B, runs a short sequence through the controller,
verifies cache operations work correctly.

Usage:
    CUDA_VISIBLE_DEVICES=1 python experiments/test_controller.py
"""
import sys
import types
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.utils.model_loader import (
    load_model_and_tokenizer, get_linear_attention_indices, get_full_attention_indices,
)
from experiments.utils.hooks import SegmentSignals
from experiments.utils.hmo_controller import HMOController, HMOConfig, HMOResult
from experiments.utils.kv_ops import (
    get_attention_kv_seq_len,
    drop_segment,
    execute_refresh,
    extract_rts_skeleton,
)
from experiments.utils.saturation import compute_segment_saturation


def test_detector_contract_aggregation():
    """Canonical sigma aggregation must match the frozen detector formula."""
    print("=" * 60)
    print("Test 0: Detector contract (canonical aggregation)")
    print("=" * 60)

    signals = {
        0: SegmentSignals(
            layer_idx=0,
            rho_max=[1.0, 3.0, 2.0],
            c_max=[0.2, 0.9, 0.5],
            g_mag_min=[2.0, 1.0, 4.0],
        ),
        1: SegmentSignals(
            layer_idx=1,
            rho_max=[0.5, 1.5, 2.5],
            c_max=[0.1, 0.3, 0.9],
            g_mag_min=[1.5, 1.0, 0.5],
        ),
    }

    sigma = compute_segment_saturation(
        signals,
        alpha_rho=0.4,
        alpha_c=0.3,
        alpha_g=0.3,
        segment_length=4,
        warmup_tokens=0,
        repeat_mask=np.array([False, False, False]),
    )

    rho_all = np.array([signals[0].rho_max, signals[1].rho_max], dtype=np.float32)
    c_all = np.array([signals[0].c_max, signals[1].c_max], dtype=np.float32)
    g_all = np.array([signals[0].g_mag_min, signals[1].g_mag_min], dtype=np.float32)

    rho_mean = rho_all.mean(axis=1, keepdims=True)
    rho_std = rho_all.std(axis=1, keepdims=True)
    rho_std[rho_std < 1e-8] = 1.0
    rho_norm = 1.0 / (1.0 + np.exp(-((rho_all - rho_mean) / rho_std)))

    c_min = c_all.min(axis=1, keepdims=True)
    c_max = c_all.max(axis=1, keepdims=True)
    c_denom = c_max - c_min
    c_denom[c_denom < 1e-8] = 1.0
    c_norm = (c_all - c_min) / c_denom

    g_pressure = 1.0 / (g_all + 1e-8)
    g_min = g_pressure.min(axis=1, keepdims=True)
    g_max = g_pressure.max(axis=1, keepdims=True)
    g_denom = g_max - g_min
    g_denom[g_denom < 1e-8] = 1.0
    g_norm = (g_pressure - g_min) / g_denom

    expected = (0.4 * rho_norm + 0.3 * c_norm + 0.3 * g_norm).max(axis=0)
    assert np.allclose(sigma, expected, atol=1e-6), "Canonical detector aggregation drifted from the frozen formula"
    print("PASS\n")


def test_detector_contract_filters():
    """Warmup and repeat-text filtering must affect sigma as required by the contract."""
    print("=" * 60)
    print("Test 1: Detector contract (warmup + repeat filtering)")
    print("=" * 60)

    signals = {
        0: SegmentSignals(
            layer_idx=0,
            rho_max=[0.8, 0.8, 0.8],
            c_max=[0.9, 0.9, 0.1],
            g_mag_min=[1.0, 1.0, 1.0],
        ),
        1: SegmentSignals(
            layer_idx=1,
            rho_max=[0.7, 0.7, 0.7],
            c_max=[0.8, 0.8, 0.1],
            g_mag_min=[1.0, 1.0, 1.0],
        ),
    }
    input_ids = torch.tensor([[
        11, 12, 13, 14, 15, 16, 17, 18,
        99, 99, 99, 99, 99, 99, 99, 99,
        21, 22, 23, 24, 25, 26, 27, 28,
    ]])

    sigma = compute_segment_saturation(
        signals,
        alpha_rho=0.4,
        alpha_c=0.3,
        alpha_g=0.3,
        segment_length=8,
        warmup_tokens=8,
        input_ids=input_ids,
        repeat_ngram_size=2,
        repeat_threshold=0.4,
    )
    sigma_no_repeat = compute_segment_saturation(
        signals,
        alpha_rho=0.4,
        alpha_c=0.3,
        alpha_g=0.3,
        segment_length=8,
        warmup_tokens=8,
        repeat_mask=np.array([False, False, False]),
    )

    assert np.isclose(float(sigma[0]), 0.0), "Warmup segment should be suppressed"
    assert float(sigma[1]) < float(sigma_no_repeat[1]), "Repeat-text filtering did not suppress collision-driven saturation"
    print("PASS\n")


def test_live_detector_feature_collection(model, tokenizer, config):
    """A real Qwen3.5 prefill must expose finite rho/c/g_pressure signals on every DeltaNet layer."""
    print("=" * 60)
    print("Test 2: Detector contract (live rho / c / g_pressure collection)")
    print("=" * 60)

    hmo_cfg = HMOConfig(segment_length=64)
    controller = HMOController(model, tokenizer, config, hmo_config=hmo_cfg, gpu_id=0)

    prompt = "The secret code is ALPHA-42. Remember this. " * 12 + "\nWhat is the secret code?"
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)

    controller.hook_mgr.clear()
    controller.hook_mgr.attach()
    try:
        _ = model(input_ids, use_cache=True)
        signals = dict(controller.hook_mgr.get_signals())
    finally:
        controller.hook_mgr.remove()

    assert set(signals.keys()) == set(controller.linear_indices), "Not all DeltaNet layers reported detector signals"
    for layer_idx, sig in signals.items():
        n = len(sig.rho_max)
        assert n > 0, f"Layer {layer_idx} produced no segment signals"
        assert len(sig.c_max) == n and len(sig.g_mag_min) == n, f"Layer {layer_idx} detector feature lengths disagree"
        for name, values in {
            "rho": sig.rho_max,
            "c": sig.c_max,
            "g_pressure_base": sig.g_mag_min,
        }.items():
            arr = np.asarray(values, dtype=np.float32)
            assert np.all(np.isfinite(arr)), f"Layer {layer_idx} produced non-finite {name} values"
            assert np.all(arr >= 0.0), f"Layer {layer_idx} produced negative {name} values"

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
    assert len(sigma) > 0 and np.all(np.isfinite(sigma)), "Live detector aggregation produced invalid sigma"
    print("PASS\n")


def test_baseline_generation():
    """Test that baseline (no intervention) generates correctly."""
    print("=" * 60)
    print("Test 3: Baseline generation (no HMO)")
    print("=" * 60)

    model, tokenizer, config = load_model_and_tokenizer("qwen3.5-0.8b", device="cuda", gpu_id=0)
    hmo_cfg = HMOConfig(segment_length=64)  # small segments for testing
    controller = HMOController(model, tokenizer, config, hmo_config=hmo_cfg, gpu_id=0)

    prompt = "The secret code is ALPHA-42. Remember this. " * 20 + "\nWhat is the secret code?"
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    print(f"Input length: {input_ids.shape[1]} tokens")

    result = controller.run_baseline(input_ids, max_new_tokens=32)
    print(f"Baseline output: {result.generated_text[:200]}")
    print(f"Peak VRAM: {result.peak_vram_mb:.0f} MB")
    assert len(result.generated_text) > 0, "Baseline generated empty text!"
    print("PASS\n")
    return model, tokenizer, config


def test_hmo_generation(model, tokenizer, config):
    """Test HMO controller with cache intervention."""
    print("=" * 60)
    print("Test 4: HMO generation (with cache intervention)")
    print("=" * 60)

    hmo_cfg = HMOConfig(
        segment_length=64,
        keep_ratio=0.5,
        skeleton_rank=2,
        refresh_budget=2,
        saturation_threshold=0.5,
        refresh_threshold=0.7,
        rts_threshold=0.3,
    )
    controller = HMOController(model, tokenizer, config, hmo_config=hmo_cfg, gpu_id=0)

    prompt = "The secret code is ALPHA-42. Remember this. " * 20 + "\nWhat is the secret code?"
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)

    result = controller.run(input_ids, max_new_tokens=32)
    print(f"HMO output: {result.generated_text[:200]}")
    print(f"Actions: KV={result.n_kept_kv}, RTS={result.n_skeleton}, "
          f"refresh={result.n_refresh}, drop={result.n_dropped}")
    print(f"Segments: {result.n_segments}")
    print(f"Peak VRAM: {result.peak_vram_mb:.0f} MB")

    assert len(result.generated_text) > 0, "HMO generated empty text!"
    assert result.n_segments > 0, "No segments detected!"
    total_actions = result.n_kept_kv + result.n_skeleton + result.n_refresh + result.n_dropped
    assert total_actions == result.n_segments, f"Action count mismatch: {total_actions} != {result.n_segments}"
    print("PASS\n")


def test_h2o_baseline(model, tokenizer, config):
    """Test H2O baseline."""
    print("=" * 60)
    print("Test 5: H2O baseline (hard eviction)")
    print("=" * 60)

    hmo_cfg = HMOConfig(segment_length=64, keep_ratio=0.5)
    controller = HMOController(model, tokenizer, config, hmo_config=hmo_cfg, gpu_id=0)

    prompt = "The secret code is ALPHA-42. Remember this. " * 20 + "\nWhat is the secret code?"
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)

    result = controller.run_h2o_baseline(input_ids, max_new_tokens=32)
    print(f"H2O output: {result.generated_text[:200]}")
    print(f"Kept: {result.n_kept_kv}, Dropped: {result.n_dropped}")
    print(f"Peak VRAM: {result.peak_vram_mb:.0f} MB")

    assert len(result.generated_text) > 0, "H2O generated empty text!"
    print("PASS\n")


def test_forced_intervention_paths(model, tokenizer, config):
    """Force refresh/RTS/drop branches so the controller's main intervention paths are exercised."""
    print("=" * 60)
    print("Test 6: Forced intervention paths (refresh / RTS / drop)")
    print("=" * 60)

    hmo_cfg = HMOConfig(segment_length=64, keep_ratio=0.0, refresh_budget=2)
    controller = HMOController(model, tokenizer, config, hmo_config=hmo_cfg, gpu_id=0)

    prompt = ("The secret code is ALPHA-42. Remember this carefully. " * 80) + "\nWhat is the secret code?"
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)

    def forced_actions(self, sigma, seq_len, segment_costs, input_ids, alpha=None):
        n = len(sigma)
        actions = {}
        if n >= 1:
            actions[0] = "KV"
        if n >= 2:
            actions[1] = "KV"
        if n >= 3:
            actions[2] = "refresh"
        if n >= 4:
            actions[3] = "RTS"
        if n >= 5:
            actions[4] = "drop"
        if n >= 6:
            actions[n - 2] = "KV"
        if n >= 7:
            actions[n - 1] = "KV"
        for i in range(n):
            actions.setdefault(i, "RTS")
        return actions, 0

    controller._decide_actions = types.MethodType(forced_actions, controller)
    result = controller.run(input_ids, max_new_tokens=8)
    print(f"Forced-path output: {result.generated_text[:200]}")
    print(f"Actions: KV={result.n_kept_kv}, RTS={result.n_skeleton}, "
          f"refresh={result.n_refresh}, drop={result.n_dropped}")
    print(f"Segments: {result.n_segments}")
    print(f"Peak VRAM: {result.peak_vram_mb:.0f} MB")

    assert len(result.generated_text) > 0, "Forced-path run generated empty text!"
    assert result.n_refresh > 0, "Refresh branch was not exercised!"
    assert result.n_skeleton > 0, "RTS branch was not exercised!"
    assert result.n_dropped > 0, "Drop branch was not exercised!"
    print("PASS\n")


def test_refresh_contract_exact_reinsertion(model, tokenizer, config):
    """Refresh must be exclusive, non-duplicating, and exactly reconstruct the original KV."""
    print("=" * 60)
    print("Test 7: Refresh contract (exact reinsertion, no duplication)")
    print("=" * 60)

    torch.cuda.empty_cache()
    hmo_cfg = HMOConfig(segment_length=64, keep_ratio=0.0, refresh_budget=1)
    controller = HMOController(model, tokenizer, config, hmo_config=hmo_cfg, gpu_id=0)

    prompt = ("The secret code is ALPHA-42. Remember this carefully. " * 40) + "\nWhat is the secret code?"
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    seq_len = input_ids.shape[1]

    seg_idx = 2
    seg_len = hmo_cfg.segment_length
    start = seg_idx * seg_len
    end = min(start + seg_len, seq_len)
    assert end > start, "Chosen refresh segment is empty"

    original_outputs = model(input_ids, use_cache=True)
    original_cache = original_outputs.past_key_values

    working_outputs = model(input_ids, use_cache=True)
    working_cache = working_outputs.past_key_values

    original_kv_len = get_attention_kv_seq_len(original_cache, controller.attn_indices[0])
    print(f"Original KV length: {original_kv_len}")

    drop_segment(working_cache, controller.attn_indices, start, end)
    dropped_kv_len = get_attention_kv_seq_len(working_cache, controller.attn_indices[0])
    print(f"After refresh-segment drop: {dropped_kv_len}")
    assert dropped_kv_len == original_kv_len - (end - start), "Refresh segment did not leave active KV exclusively"

    payload = {
        "token_ids": input_ids[:, start:end].clone(),
        "position_ids": torch.arange(start, end, device=input_ids.device).unsqueeze(0),
        "replay_full_input_ids": input_ids.clone(),
    }
    active_positions = torch.cat(
        [
            torch.arange(0, start, device=input_ids.device, dtype=torch.long),
            torch.arange(end, seq_len, device=input_ids.device, dtype=torch.long),
        ],
        dim=0,
    )
    active_positions = execute_refresh(
        model,
        working_cache,
        controller.attn_indices,
        payload,
        active_positions,
    )

    refreshed_kv_len = get_attention_kv_seq_len(working_cache, controller.attn_indices[0])
    print(f"After exact refresh reinsertion: {refreshed_kv_len}")
    assert refreshed_kv_len == original_kv_len, "Refresh reinserted a duplicate KV copy instead of restoring exact length"
    assert active_positions.shape[0] == seq_len, "Active logical positions were not fully restored"

    for layer_idx in controller.attn_indices:
        original_layer = original_cache.layers[layer_idx]
        refreshed_layer = working_cache.layers[layer_idx]
        assert torch.allclose(
            refreshed_layer.keys, original_layer.keys, atol=1e-4, rtol=1e-3
        ), f"Layer {layer_idx} keys differ after exact refresh reinsertion"
        assert torch.allclose(
            refreshed_layer.values, original_layer.values, atol=1e-4, rtol=1e-3
        ), f"Layer {layer_idx} values differ after exact refresh reinsertion"

    print("PASS\n")


def test_rts_contract_storage_and_decode(model, tokenizer, config):
    """RTS must reduce active KV storage physically and remain decodable through the RTS runtime."""
    print("=" * 60)
    print("Test 8: RTS contract (physical storage + decode runtime)")
    print("=" * 60)

    torch.cuda.empty_cache()
    hmo_cfg = HMOConfig(segment_length=64, skeleton_rank=2, keep_ratio=0.0)
    controller = HMOController(model, tokenizer, config, hmo_config=hmo_cfg, gpu_id=0)

    prompt = ("The secret code is ALPHA-42. Remember this carefully. " * 40) + "\nWhat is the secret code?"
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    seq_len = input_ids.shape[1]

    outputs = model(input_ids, use_cache=True)
    cache = outputs.past_key_values

    seg_idx = 2
    start = seg_idx * hmo_cfg.segment_length
    end = min(start + hmo_cfg.segment_length, seq_len)
    assert end > start, "Chosen RTS segment is empty"

    original_kv_len = get_attention_kv_seq_len(cache, controller.attn_indices[0])
    rts_segment = extract_rts_skeleton(
        cache,
        controller.attn_indices,
        start,
        end,
        rank=hmo_cfg.skeleton_rank,
    )
    reduced_kv_len = get_attention_kv_seq_len(cache, controller.attn_indices[0])

    print(f"Original KV length: {original_kv_len}")
    print(f"After RTS extraction: {reduced_kv_len}")
    print(f"RTS stored bytes: {rts_segment.stored_bytes}")
    print(f"RTS full-KV bytes: {rts_segment.full_kv_bytes}")

    assert reduced_kv_len == original_kv_len - (end - start), "RTS did not physically remove its segment from active KV"
    assert rts_segment.stored_bytes < rts_segment.full_kv_bytes, "RTS store is not actually smaller than full KV"

    active_positions = torch.cat(
        [
            torch.arange(0, start, device=input_ids.device, dtype=torch.long),
            torch.arange(end, seq_len, device=input_ids.device, dtype=torch.long),
        ],
        dim=0,
    )
    result = controller._decode_loop(
        input_ids,
        cache,
        outputs.logits,
        max_new_tokens=8,
        refresh_store={},
        rts_store={seg_idx: rts_segment},
        active_positions=active_positions,
        result=HMOResult(),
        do_sample=False,
    )
    assert len(result.generated_text) > 0, "RTS runtime decode generated empty text!"
    print("PASS\n")


if __name__ == "__main__":
    test_detector_contract_aggregation()
    test_detector_contract_filters()
    model, tokenizer, config = test_baseline_generation()
    torch.cuda.empty_cache()
    test_live_detector_feature_collection(model, tokenizer, config)
    torch.cuda.empty_cache()
    test_hmo_generation(model, tokenizer, config)
    torch.cuda.empty_cache()
    test_h2o_baseline(model, tokenizer, config)
    torch.cuda.empty_cache()
    test_forced_intervention_paths(model, tokenizer, config)
    torch.cuda.empty_cache()
    test_refresh_contract_exact_reinsertion(model, tokenizer, config)
    torch.cuda.empty_cache()
    test_rts_contract_storage_and_decode(model, tokenizer, config)
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
