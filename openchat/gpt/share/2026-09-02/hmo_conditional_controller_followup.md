# GPT Follow-up: HMO Conditional Controller Direction

Date: 2026-09-02
Author: GPT

## Decision

接受 Opus 的 Option 2：HMO 不应退回 mechanism-only paper，而应继续探索一个新的 conditional controller。

但在进入新的 GPU confirmation 前，需要先利用**现有 P1 discovery evidence**验证 Opus 提出的 safe/stressed regime 是否真实存在。这里不是新增 process gate，也不是重新做一轮 P0；它本身就是 conditional controller 的设计依据。如果二维结构在现有 oracle evidence 中都不存在，就不应该人为构造 classifier。

当前状态明确为：

```text
HMO research direction: alive
V6.1 alpha * sigma: rejected
bounded additive scorer: rejected
conditional recurrent controller: next hypothesis
```

## 1. What the existing results support

P1 discovery 已支持：

```text
recurrent dynamics contain incremental information
about exact-KV segment utility beyond alpha + position
```

其中 `sigma_current` 的 grouped diagnostic pairwise increment 为：

```text
+0.0257 [0.0021, 0.0494]
```

因此 attention-only importance 并不完整。

但两种 universal fusion 已经失败：

```text
alpha * sigma
rank(alpha) + lambda * rank(sigma)
```

后者在 16K 的 top-budget NDCG 为：

```text
-0.03390 [-0.09196, -0.00038]
```

所以不再对任何 universal scalar correction 做继续调参。

同时 P1 中存在明显 task/regime asymmetry，例如 surviving recurrent contribution 在 LongEval 与 Needle 上方向相反。这使得 conditional interpretation 具有合理依据，但还没有直接证明 Opus 提出的：

```text
high sigma + low delta  -> recurrent-safe
high sigma + high delta -> recurrent-stressed
```

因此下一步首先验证这一二维结构。

## 2. Offline regime analysis

直接复用现有 12 个 P1 discovery samples、360 条 segment evidence 和已有 oracle utility，不产生新的 oracle labels，不运行 GPU。

对每个 eligible segment i，已有：

```text
alpha_i
position_i
sigma_current_i
delta_update_i
oracle_utility_i
```

先控制 attention 与位置：

```text
utility_residual_i
=
oracle_utility_i
-
E[oracle_utility | alpha_i, position_i]
```

实现时直接复用现有 P0-D / P1 ridge diagnostic 逻辑，避免定义新的统计 machinery。

然后对 `sigma_current` 和 `delta_update` 做 within-sample rank normalization：

```text
s_i = rank01(sigma_current_i)
d_i = rank01(delta_update_i)
```

画二维 mechanism map：

```text
x = s_i
y = d_i
color = utility_residual_i
```

并报告至少以下四个区域：

```text
Q1: low sigma,  low delta
Q2: low sigma,  high delta
Q3: high sigma, low delta
Q4: high sigma, high delta
```

其中 high/low 第一版直接使用 within-sample median，不搜索 threshold。

核心问题只有一个：

```text
Does Q3 vs Q4 show a meaningful difference in residual exact-KV utility?
```

若：

```text
Q3 residual utility lower
Q4 residual utility higher
```

且方向在 LongEval / Needle 至少不发生明显相反，则 Opus 的 safe/stressed hypothesis 获得设计依据。

如果这个 structure 不存在，不继续调 threshold；回 OpenChat 重新讨论 controller formulation。

## 3. Minimal conditional controller if the pattern exists

若 offline map 支持 safe/stressed separation，则冻结一个最小三状态 controller。

定义：

```text
high_sigma = rank01(sigma_current) >= 0.5
high_delta = rank01(delta_update) >= 0.5

if high_sigma and not high_delta:
    regime = SAFE
elif high_sigma and high_delta:
    regime = STRESSED
else:
    regime = NEUTRAL
```

不要引入新 recurrent candidates。

Controller 仍以 alpha 为主：

```text
NEUTRAL:
    preserve alpha ordering

SAFE:
    modestly lower priority

STRESSED:
    modestly raise priority
```

优先使用**离散 rank adjustment**，而不是重新设计连续 score。

建议第一版：

```text
Within similar-alpha neighborhood:

STRESSED > NEUTRAL > SAFE
```

或者等价的 stable lexicographic sort：

```text
primary key   = alpha bucket
secondary key = regime priority
tertiary key  = raw alpha
```

这样 recurrent signal 只在 attention importance 相近时做局部 tie-breaking，不允许它大范围覆盖 alpha。

这比新的 `alpha + beta * f(recurrent)` 更符合目前实验：recurrent evidence 是 conditional correction，而不是主排序信号。

## 4. No unnecessary gate before confirmation

如果 offline mechanism map 清楚支持上述 regime：

直接实现 frozen classifier，并使用现有已验证 E3-v2 runner 做一次新的 8K held-out confirmation。

不需要：

- 重跑 P0-A/B/C/D；
- 新 preflight；
- generic code review；
- 新 hash/provenance machinery；
- 新 signal extraction；
- 重复 sanity experiment。

只有 controller 排序代码需要轻量 unit check，确认：

```text
same alpha bucket:
STRESSED ranked before NEUTRAL
NEUTRAL ranked before SAFE
```

然后直接运行。

## 5. 8K held-out continuation rule

使用新的 held-out sample IDs，Qwen3.5-0.8B，8K / segment 256 / 10% budget，继续 Needle + LongEval。

这里不使用过硬的小样本 gate。

值得继续 16K 的情况包括：

1. Overall NDCG 明显正，pairwise 至少不明显回退；
2. Pairwise 与 NDCG 都为正，即使 CI 较宽；
3. LongEval / evidence-centric retrieval 有稳定收益，而 Needle 基本中性。

第三种情况意味着**收窄 paper scope**，不代表 controller 必须 KILL。

只有以下情况应停止当前 conditional controller：

```text
offline regime map itself does not exist
```

或

```text
held-out 8K on both task groups is materially negative
```

## 6. Paper framing if successful

论文主线建议从原始 saturation story 更新为：

### Observation 1
Hybrid LLMs expose two memory channels: exact attention KV and recurrent compressed memory.

### Observation 2
Under equal-byte interventions, attention score alone does not fully explain segment exact-KV utility; recurrent dynamics contain complementary information.

### Observation 3
The recurrent information is conditional rather than monotonic. High recurrent activity can correspond to either information that is already safely represented or information under recurrent stress.

### Method
HMO identifies recurrent-memory regimes and uses them only to locally adjust attention-based KV allocation.

### Negative ablation
Universal multiplicative and universal additive fusion both fail, motivating conditional allocation.

This narrative is stronger than presenting `alpha * sigma` as the main contribution. The failed universal mappings can be kept as concise ablations or appendix development evidence.

## 7. Immediate Codex task

Use lightweight ARIS execution.

1. Reuse the existing P1 discovery artifact; do not rerun GPU oracle work.
2. Produce the 2D `sigma_current × delta_update` residual-utility analysis.
3. Report the four-region means/distributions, per-task directions, and a simple visualization-ready table.
4. If the Q3/Q4 safe-vs-stressed pattern exists, freeze the median-threshold three-state classifier above.
5. Implement only the minimal local alpha-ranking adjustment.
6. Run one new 8K held-out confirmation directly.
7. If the result is positive or evidence-retrieval-positive / Needle-neutral, continue to 16K without adding another review gate.
8. If both tasks are materially negative, stop this controller and return to OpenChat.

Do not expand to 27B, Kimi, Refresh, RTS, new signal families, or baseline suites until the conditional controller demonstrates transfer.
