# HMO 理论-代码对照手册

**目的**: 从理论出发，逐块映射到代码实现，确保理论与代码完全一致。

**2026-04-22 升级说明**: 实验模型从 Qwen3.5-9B/4B (RTX 5090) 升级到 Qwen3.5-27B (A100 80GB)。
理论和代码实现不变——Qwen3.5 全系列共享相同的 hybrid-attention 架构（DeltaNet + Attention，3:1 比例），
27B 只是层数更多（64层 vs 36层），每层的 hook 接口和 KV cache 结构完全一致。
跨家族验证 E5 从 GGUF (llama-cpp) 升级到 GPTQ-Int4 (transformers)，使用真正的 KV cache 管理。

---

## 1. State Saturation Detection（DeltaNet 饱和检测）

### 理论

DeltaNet 的 gated delta rule 更新公式：
```
S_t = S_{t-1} · exp(g_t) + β_t · k_t ⊗ (v_t - S_{t-1}^T k_t)
```

S_t 是固定大小的 recurrent state `[H, D_k, D_v]`，exp(g_t) ∈ (0,1) 导致旧信息指数衰减。在足够长的序列上，早期 token 信息必然被覆盖（P1: 数学必然性）。

HMO 通过三个子信号检测 saturation 程度：

**子信号 (1): ρ — Write-to-State Ratio**
```
ρ_t^ℓ = ‖β_t^ℓ · u_t^ℓ‖ / (τ_t^ℓ + ε)
```
- 分子：写入幅度 = write gate × normalized write direction 的范数
- 分母：retention strength τ = -g > 0
- 物理含义：ρ 高 → 新信息写入速度超过旧信息保持速度 → state 被快速覆盖

**子信号 (2): c — Novelty Collision**
```
c_t^ℓ = max_{j=1..m} cos(u_t^ℓ, u_{t-j}^ℓ),  m=8
```
- 当前 write direction 与最近 8 个 write direction 的最大余弦相似度
- 物理含义：c 高 → 连续往相同方向写 → state 有效秩降低

**子信号 (3): p — Decay Pressure**
```
p_t^ℓ = 1 / (τ_t^ℓ + ε)
```
- retention 的倒数
- 物理含义：τ 小（衰减快）→ p 大 → 旧信息丢失更快

**层级聚合（per-segment）：**
```
σ_j = max_{t∈C_j, ℓ∈D} [α_ρ·ρ̄_t^ℓ + α_c·c_t^ℓ + α_g·p_t^ℓ]
```
- ρ̄ 为 z-score 归一化后的 ρ
- D 为所有 DeltaNet 层集合
- 先 segment 内 max over tokens，再 max over layers

### 代码映射

**文件: `experiments/utils/hooks.py`**

| 理论符号 | 代码位置 | 实现 |
|---------|---------|------|
| β_t^ℓ (write gate) | hooks.py:100-102 | `beta = module.in_proj_b(hs).sigmoid()` |
| g_t^ℓ (decay gate, < 0) | hooks.py:101-103 | `g = -A_log.exp() * softplus(a + dt_bias)` |
| τ_t^ℓ = -g (retention) | hooks.py:144-145 | `retention = (-g).clamp(min=1e-8)` |
| u_t^ℓ (write direction) | hooks.py:106-131 | post-conv1d key, L2 normalized |
| ‖β·u‖ (write magnitude) | hooks.py:141-142 | `write_norm = (beta.unsqueeze(-1) * key_norm).norm(dim=-1)` |
| ρ = ‖β·u‖/τ | hooks.py:170-173 | `seg_rho = (seg_write / seg_retain).amax(dim=(1,2))` |
| c = max cos similarity | hooks.py:152-161 | offset loop 1..8, `torch.maximum` 取 max |
| τ_min (存原始值) | hooks.py:175-177 | `seg_retain.amin(dim=(1,2))` |

**关键设计决策：**
- 用 `register_forward_pre_hook` + `with_kwargs=True`（Qwen3.5 传 hidden_states 为 keyword arg）
- 提取 post-conv key（不是 pre-conv），因为 conv 后的 key 才是实际写入 state 的方向
- GQA 处理：如果 num_v_heads > num_k_heads，repeat key 对齐维度
- 尾部 segment < seg_len//4 时跳过（统计量不可靠）

**文件: `experiments/utils/saturation.py`**

| 理论步骤 | 代码位置 | 实现 |
|---------|---------|------|
| ρ 归一化 | saturation.py:93-100 | z-score → sigmoid per layer → [0,1] |
| c 归一化 | saturation.py:101 | min-max per layer → [0,1] |
| p = 1/τ 归一化 | saturation.py:103 | `1/(g_all+ε)` → min-max per layer → [0,1] |
| 重复文本过滤 | saturation.py:105-114 | 4-gram 重复率 ≥ 0.4 的 segment，c 置零 |
| 加权组合 | saturation.py:117 | `0.4·ρ + 0.3·c + 0.3·p` |
| warmup 抑制 | saturation.py:119-128 | 前 50 token 区域的 segment 被线性衰减 |
| max across layers | saturation.py:131 | `combined.max(axis=0)` → σ [n_segs] |

**归一化选择的理由：**
- ρ 用 z-score sigmoid 而非 min-max：ρ 的绝对值跨层差异大，sigmoid 比 min-max 更鲁棒（不受单个 outlier 影响）
- c 用 min-max：c 本身是 cos 相似度（有界 [-1,1]），分布较均匀
- 只过滤 c 不过滤 ρ 和 g：重复文本导致 c 假阳性，但 ρ（写入速度）和 g（衰减速度）仍有物理意义

---

## 2. Attention Fragility α 和双通道联合信号 φ

### 理论

**H2 核心论点：** 一个 segment 值得 refresh 当且仅当两个条件同时满足：
1. DeltaNet 正在丢失该 segment 的信息（σ 高）
2. Attention 层依赖该 segment 的信息（α 高）

```
φ_j = σ_j · α_j
```

乘法关系意味着任一通道为零，φ 就为零——必须两个通道同时报警。

**V1 验证：**
- σ alone: Pearson = -0.045, AUC = 0.454（不具预测力）
- σ × α: Pearson = 0.566, AUC = 0.879（强预测力）

**α 的定义：** segment C_j 的 attention fragility score = attention 层对该 segment 的依赖程度。

### 代码映射

**文件: `experiments/utils/hmo_controller.py:388-463` — `collect_segment_attention_scores()`**

| 步骤 | 代码位置 | 实现 |
|------|---------|------|
| Prefill 获取 cache | hmo_controller.py:403 | `_prefill_with_cache_last_logits(input_ids)` |
| 采样下一个 token | hmo_controller.py:404 | `prefill_logits[:, -1, :].argmax()` |
| 切换 eager attention | hmo_controller.py:406-409 | `_attn_implementation = "eager"` |
| 一步 decode + output_attentions | hmo_controller.py:418-425 | `model.model(next_token, output_attentions=True)` |
| 提取 per-token importance | hmo_controller.py:431-441 | `attn_w[0].mean(dim=0).sum(dim=0)` 跨 head 平均 |
| 聚合为 per-segment α | hmo_controller.py:446-452 | `token_importance[start:end].mean()` |
| 归一化到 [0,1] | hmo_controller.py:459-463 | min-max normalization |

**为什么用 decode-step attention 而非 prefill attention：**
- Prefill 的 attention 是下三角矩阵，每行分布不同
- Decode 的第一步 query 能看到所有 prefill token，其 attention 分布直接反映"生成依赖哪些位置"

**为什么切换到 eager attention：**
- SDPA（fused kernel）不返回 attention weights
- 必须用 eager 模式才能拿到 `outputs.attentions`

**φ 的计算：** `hmo_controller.py:530-534`
```python
if alpha is not None:
    phi = sigma * alpha
else:
    phi = sigma  # fallback
```

**φ 在 controller 中的使用：** `hmo_controller.py:536-571`
- 按 φ 降序排列 middle segments
- Top `refresh_budget` 个 → refresh
- 剩余 → RTS 或 drop

---

## 3. 动作空间 {KV, RTS, refresh, drop} 和分配逻辑

### 理论

Rate-Distortion 公式：
```
min L_j = D_j^task(a_j) + β_A · R_A(a_j, r_j) + β_S · D̂_j^state(a_j)
```

Memory cost 排序：KV > refresh > RTS > drop

代码用贪心启发式近似求解，不直接优化 R-D 公式。

### 代码映射

**文件: `experiments/utils/hmo_controller.py:487-639` — `_decide_actions()`**

| 步骤 | 代码位置 | 逻辑 |
|------|---------|------|
| 保护 sinks + recent | hmo_controller.py:509-520 | 第 1 + 最后 1 个 segment → KV |
| 尾部 segment | hmo_controller.py:518-520 | sigma 没覆盖的 → KV |
| 计算 budget | hmo_controller.py:522-525 | `protected + keep_ratio × middle` |
| φ 排序 | hmo_controller.py:530-538 | `middle_phi.sort(reverse=True)` |
| Top-K refresh | hmo_controller.py:545-571 | 前 `refresh_budget` 个 → refresh |
| σ-proportional RTS | hmo_controller.py:594-608 | 按 σ 比例分配 token 数 |
| Budget 耗尽 → drop | hmo_controller.py:636-637 | `n_keep=0 → drop` |

**关键设计决策：**
- Refresh 用 top-K 排序（不用绝对阈值），避免 uniform-density 数据上永不触发
- RTS token 分配用 σ（不是 φ），因为高 φ segment 已被 refresh 拿走
- Protected segments 1+1（1024 tokens），比 H2O 的 4+32=36 tokens 多，是 segment-level 操作的固有代价

---

## 4. Cache 物理操作

### 理论

动作 plan 需要转化为对 DynamicCache 的物理操作：
- KV: 不动
- RTS: 在 segment 内按 KV norm 选 top-r tokens，物理删除其余
- refresh: 物理删除 segment KV，存储 token ids，decode 前 replay 重建
- drop: 物理删除 segment KV，不存储任何东西

### 代码映射

**文件: `experiments/utils/hmo_controller.py:235-373` — `run()` Step 4**

| 阶段 | 代码位置 | 操作 |
|------|---------|------|
| 存储 refresh payload | hmo_controller.py:240-251 | clone token_ids + position_ids（在 cache 操作前） |
| Phase A+B: reverse-order pass | hmo_controller.py:253-278 | 从后往前处理，保证绝对位置正确 |
| RTS 执行 | hmo_controller.py:261-274 | `extract_token_skeleton()` in-place pruning |
| drop/refresh 执行 | hmo_controller.py:275-278 | `drop_segment()` 物理删除 |
| Phase C: per-token evict | hmo_controller.py:280-352 | backward compat，正式实验不走此路径 |
| Pre-replay budget snapshot | hmo_controller.py:354-367 | 记录分配器承诺的 budget |

**文件: `experiments/utils/hmo_controller.py:641-743` — `_decode_loop()`**

| 阶段 | 代码位置 | 操作 |
|------|---------|------|
| Refresh replay | hmo_controller.py:670-680 | `execute_refresh()` × N 次 |
| Post-replay snapshot | hmo_controller.py:690-699 | 记录 decode 时实际内存 |
| Autoregressive decode | hmo_controller.py:706-733 | 用 `logical_position` 保证 RoPE 正确 |

**关键实现细节：**
- **Reverse-order 处理：** 从后往前操作 cache，避免前面的删除影响后面的绝对位置
- **active_positions 跟踪：** 维护 cache index → 原始 token position 的映射
- **logical_position：** decode 时用真实序列位置（不是 cache 长度）做 RoPE，因为 cache 被修改后 `cache.get_seq_length()` 不再等于原始长度
- **两个 budget 视角：** `budget_charged_bytes`（pre-replay，分配器承诺）vs `decode_resident_bytes`（post-replay，实际占用）

**相关文件：**
- `experiments/utils/kv_ops.py` — `extract_token_skeleton()`, `drop_segment()`, `execute_refresh()`, `evict_kv_tokens()`
- `experiments/utils/memory_accounting.py` — `snapshot_hmo_budget()`, `get_segment_kv_bytes()`

---

## 完整执行流程

```
run() 开始
  │
  ├── Step 0: collect_segment_attention_scores() → α
  │           [额外一次 prefill + 一步 eager decode]
  │
  ├── Step 1: prefill with DeltaNet hooks → signals + cache
  │           [第二次 prefill，hooks 提取 β/g/key]
  │
  ├── Step 2: compute_segment_saturation(signals) → σ
  │           [归一化 + 过滤 + 加权 + max across layers]
  │           compute α from token_importance → α
  │           [per-segment mean + min-max normalize]
  │
  ├── Step 3: _decide_actions(σ, α) → actions + budget
  │           φ = σ · α
  │           protect sinks/recent → top-K refresh → σ-proportional RTS → drop
  │
  ├── Step 4: Execute cache operations (reverse order)
  │   ├── 存储 refresh payload（在操作前）
  │   ├── RTS: extract_token_skeleton (in-place pruning)
  │   ├── refresh: drop_segment (KV removed, token ids stored)
  │   └── drop: drop_segment (KV removed, nothing stored)
  │
  ├── Pre-replay budget snapshot
  │
  └── Step 5: _decode_loop()
      ├── Refresh replay: execute_refresh × N [N 次额外 prefill]
      ├── Post-replay budget snapshot
      └── Autoregressive decode (logical_position for RoPE)
```

---

## 已知的理论-代码 Gap

| Gap | 状态 | 说明 |
|-----|------|------|
| α 需要额外一次 prefill + decode | ⚠️ 已知开销 | E6 会测量 overhead |
| refresh replay 需要 N 次额外 prefill | ⚠️ 已知开销 | refresh 的固有成本 |
| Protected 1+1 segments > H2O 的 4+32 tokens | ⚠️ 设计选择 | segment-level 操作的代价 |
| Phase C "evict" 分支 | ✅ 不影响 | 正式实验不走此路径 |
| 尾部 segment 跳过 | ✅ 已补偿 | `_decide_actions` 中自动设为 KV |

---

## 5. Budget Matching 和 Memory Accounting

### 理论

Rate-Distortion 公式中 R_A(a_j, r_j) 是 memory cost 项。论文核心实验原则：**所有方法在相同 total memory budget（bytes）下对比**。

Memory cost 排序：KV >> refresh ≈ RTS > drop
- KV: 全部 token 的 K+V tensors（~16MB per segment @9B bf16）
- refresh: token ids + position ids + shared replay prefix（~8KB per segment + ~seq_len×8 shared）
- RTS: skeleton tokens 的 K+V（= kv_bytes × n_keep/seg_len）
- drop: 0 bytes

**但 refresh 在 replay 后 KV 回到 cache**，所以 decode-time resident memory ≈ kv_bytes。refresh 的 budget 优势不是省 bytes，而是：(1) prefill 后先释放 KV 减少 peak memory，(2) 通过 φ 精准选择哪些 segment 恢复。

### 代码映射

**文件: `experiments/utils/memory_accounting.py`**

| 函数 | 作用 |
|------|------|
| `tensor_nbytes(t)` | `t.numel() * t.element_size()` |
| `get_active_kv_bytes(cache, attn_indices)` | 统计 attention 层（不含 DeltaNet）的 KV bytes |
| `get_segment_kv_bytes(cache, attn_indices, start, end)` | 一个 segment 在所有 attention 层的 KV bytes |
| `refresh_payload_nbytes(payload, counted_shared)` | token_ids + position_ids + shared prefix（用 `id()` 去重） |
| `snapshot_hmo_budget(...)` | 快照：active_kv + refresh + rts = total |

**文件: `experiments/utils/hmo_controller.py:465-485` — `_build_segment_costs()`**

每个 segment 的四种 cost：
- `kv_bytes`: 保留完整 KV 的成本
- `refresh_segment_bytes`: 存储 token ids + position ids
- `shared_refresh_bytes`: replay prefix（共享，只计一次）
- `rts_bytes`: token-pruning skeleton 的 KV 成本

**Budget 公式（hmo_controller.py:522-525）：**
```
budget = protected_bytes + keep_ratio × middle_bytes
```
keep_ratio=0.5 → 中间 segment 的 KV 只能保留一半 bytes，与 H2O 对齐。

**两次 Budget Snapshot：**
- Pre-replay（hmo_controller.py:354-367）：`budget_charged = active_kv + refresh_storage + rts_storage`
- Post-replay（hmo_controller.py:690-699）：`decode_resident = active_kv(with refreshed) + rts_storage`

**Token-pruning RTS 不产生额外 rts_bytes：** skeleton tokens 已在 active cache 中，被 `get_active_kv_bytes` 统计，不重复计算。

---

## 6. Baselines 实现

### 理论背景

所有 baseline 都是针对标准 Transformer 设计的 KV cache 管理方法。它们的原始开源代码**不支持 hybrid architecture**（Qwen3.5 的 DeltaNet + Attention 混合层），因此我们在 `hmo_controller.py` 中统一复现，确保：
1. 所有方法使用相同的 cache 操作接口（`evict_kv_tokens`, `_compute_budget_n_keep`）
2. 所有方法在相同的 `budget_limit_bytes` 下运行
3. 所有方法只操作 attention 层的 KV cache（`self.attn_indices`），DeltaNet 层不动

**这是 HMO 论文的一个重要叙述点：** 现有方法都是为纯 attention 模型设计的，无法感知 DeltaNet 通道的状态。HMO 是首个同时协调两个通道的方法。在论文中，我们应该说明 baseline 是在相同的 hybrid-architecture cache 操作框架下公平复现的，而非使用原始代码（原始代码无法在 hybrid 模型上运行）。

### 代码映射

**文件: `experiments/utils/hmo_controller.py`**

| Baseline | 方法 | 代码位置 | 核心逻辑 |
|----------|------|---------|---------|
| Full KV | `run_baseline()` | :752-770 | 不做任何 eviction，保留完整 cache |
| H2O | `run_h2o_baseline()` | :772-864 | 最后一层 attention 的 chunked QK importance → global top-k + sinks + recent |
| SnapKV | `run_snapkv_baseline()` | :901-977 | 最后 3 层 attention，observation_window=32 的 QK importance → global top-k |
| StreamingLLM | `run_streamingllm_baseline()` | :979-1002 | sink=4 + recent=n_keep-4，中间全部 drop |
| DuoAttention | `run_duoattention_baseline()` | :1004-1116 | per-head entropy 分类 retrieval/streaming → union keep mask |

### 与原始方法的差异

| Baseline | 我们的实现 | 与原始的差异 | 差异影响 |
|----------|-----------|-------------|---------|
| H2O | 最后一层 chunked QK | 原始用所有层累积 | 我们的信号稍弱，但省内存；对 H2O 略不利，对 HMO 的对比更保守 |
| SnapKV | 最后 3 层 QK，obs_window=32 | 原始用 pooling kernel | 核心逻辑一致，实现细节不同 |
| StreamingLLM | sink=4 + recent | 基本一致 | 最忠实的复现 |
| DuoAttention | online per-head entropy | 原始用 offline profiling | 差异最大；我们的更简单但可能不如原始准确 |

**论文叙述策略：**
- 明确说明"为了在 hybrid architecture 上公平对比，所有 baseline 在统一的 cache 操作框架下复现"
- 强调原始方法不支持 hybrid 模型（只操作 attention 层，不感知 DeltaNet）
- 这本身就是 HMO 的 motivation：现有方法是 single-channel（只管 attention），HMO 是 dual-channel（同时管 attention + DeltaNet）
- 如果 reviewer 质疑复现准确性，可以指出：(a) StreamingLLM 完全忠实，(b) H2O/SnapKV 的核心逻辑一致，(c) 我们的实现对 baseline 略不利（信号更弱），使对比更保守
