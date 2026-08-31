# HMO Research — Code Change Log

**Project**: Hybrid Memory Orchestration (HMO) for NeurIPS 2026

---

## 2026-04-14

### Initial Setup
- Created conda env `hmo_research` (Python 3.11, PyTorch 2.7.0+cu128)
- Installed all dependencies (see ENV_SETUP.md)
- Created directory structure: utils/, v1-v4, results/, logs/
- Created ENV_SETUP.md and CODE_LOG.md

### Shared Utils (v1)
- `model_loader.py`: Qwen3.5 model loading from /mnt/990evo/model/
- `hooks.py`: DeltaNet layer hooks (v1, stored full tensors)
- `saturation.py`: rho/c/srank computation (v1)
- `dataset_utils.py`: Needle + LongBench data loading
- `metrics.py`: accuracy, F1, correlation, latency
- `eval_harness.py`: unified generation + eval pipeline

### Codex Review Fixes (v2)
- `hooks.py` v2: streaming per-segment aggregation (no OOM at 128K), post-conv key capture
- `saturation.py` v2: proper signal normalization, g_pressure replaces broken srank
- `dataset_utils.py`: fixed Needle answer format (short extractive, not full sentence)
- `v1_saturation/run.py` v2: fixed oracle (segment removal, not masking with pad tokens), stratified sampling

### Experiment Scripts
- `v1_saturation/run.py`: saturation detection validation (kill question: can sigma predict failure?)
- `v2_refresh/run.py`: refresh validation (4 conditions: none/random/periodic/triggered)
- `v3_rts/run.py`: RTS skeleton validation (3 conditions: full_kv/h2o/rts)
- `v4_joint/run.py`: joint HMO validation (5 conditions: full_kv/h2o/rts/refresh/hmo_full)

### Model Matrix Update
- Kimi-Linear-48B-A3B: Q4_K_M → IQ4_XS (26.5GB, saves 3.5GB for inference headroom)

### HMO Controller Prototype
- `kv_ops.py`: KV cache 底层操作 — evict_kv_tokens, replace_with_skeleton (SVD), drop_segment, execute_refresh
- `hmo_controller.py`: 核心 controller — HMOController 类，包含：
  - `run()`: 完整 HMO pipeline (prefill + saturation detection + action decision + cache ops + decode)
  - `run_baseline()`: 无干预 baseline (full KV)
  - `run_h2o_baseline()`: H2O hard eviction baseline
  - `_decide_actions()`: 贪心动作分配 (protect sinks/recent, sigma-based refresh/RTS/drop)
  - `_decode_loop()`: 自定义 decode loop，使用修改后的 cache
- `hooks.py` 修复: register_forward_pre_hook with_kwargs=True (Qwen3.5 用 kwargs 传 hidden_states)
- `hmo_controller.py` 修复: get_signals() 返回 copy 防止 remove() 清空
- `hmo_controller.py` 修复: H2O baseline 用 eager attention + 跳过 None attentions
- `test_controller.py`: 3 个 smoke test 全部通过 (baseline / HMO / H2O)
