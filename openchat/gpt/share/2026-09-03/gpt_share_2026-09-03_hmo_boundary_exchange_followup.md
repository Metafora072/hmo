# GPT Follow-up: HMO Budget-Boundary Exchange

Date: 2026-09-03
Author: GPT

## Decision

当前不应 KILL HMO，也不应继续调已经失败的 conditional rank controller。

最新证据应被拆开理解：

```text
Hybrid-memory premise: supported
Safe/stressed regime observation: supported on discovery evidence
Universal multiplicative scorer: rejected
Universal additive scorer: rejected
Adjacent rank ±1 conditional controller: rejected
Conditional budget allocation: not yet tested
```

关键原因是：最新 8K held-out 中，conditional rank controller 每个 sample 虽然执行了 2–10 次 adjacent swaps，但 12 个 sample 中有 10 个 NDCG 完全不变。8K 设置只有 30 个 eligible middle segments、top-3 exact-KV slots，因此真正决定 top-budget NDCG 的不是全局排序中大量局部位置是否变化，而是**哪些 segment 能跨过 top-k membership boundary**。

因此下一步不再做 rank-radius、threshold、bucket 或连续 score 调参，而是让已验证的 safe/stressed regime 直接作用于 fixed-budget membership。

## 1. Evidence already established

Offline conditional-regime analysis 已得到：

```text
Q4 stressed - Q3 safe = +0.25541
95% sample bootstrap CI = [+0.03702, +0.45082]
LongEval = +0.28193
Needle   = +0.22889
positive samples = 9/12
```

这里：

```text
Q3 = high sigma + low delta   -> SAFE
Q4 = high sigma + high delta  -> STRESSED
```

所以在 controlling `alpha + position` 后，STRESSED segment 的 residual exact-KV utility 明显高于 SAFE segment。

这个 observation 本身目前没有被 held-out controller failure 推翻。失败的是之前的 action：

```text
STRESSED -> alpha rank 前移 1 位
SAFE     -> alpha rank 后移 1 位
```

该 action 太弱，常常无法改变 top-k exact-KV set。

## 2. New controller hypothesis: boundary exchange

保持 alpha 作为主选择器。

首先：

```text
R_alpha = TopK(alpha)
```

其中：

```text
k = exact-KV budget slots
```

然后只检查预算边界内外的 recurrent regime。

### Candidate donor inside budget

从 `R_alpha` 内寻找：

```text
SAFE segments
```

选择其中 alpha 最低的一个：

```text
safe_in = lowest-alpha SAFE inside TopK
```

### Candidate entrant outside budget

从 `R_alpha` 外寻找：

```text
STRESSED segments
```

选择其中 alpha 最高的一个：

```text
stressed_out = highest-alpha STRESSED outside TopK
```

### One-swap rule

如果两者同时存在：

```text
R_hmo
=
R_alpha
- {safe_in}
+ {stressed_out}
```

否则：

```text
R_hmo = R_alpha
```

第一版最多允许：

```text
1 swap / sample
```

不引入连续权重、不引入新的 threshold、不增加 signal family。

该设计直接对应已观察到的机制：

```text
STRESSED has higher residual exact-KV utility than SAFE
```

并且一定作用于真正决定 NDCG 的 top-k membership，而不是只改变预算之外的无关排名。

## 3. Why this is not formula fishing

这不是在失败 scorer 上继续寻找另一组超参数。

之前三类方法都在尝试：

```text
recurrent signal -> universal scalar score/rank correction
```

现在的 hypothesis 改成：

```text
attention proposes the KV set
recurrent regime revises membership only at the budget boundary
```

它对应的是新的 allocation abstraction，而不是另一种数值融合公式。

因此不要搜索：

- swap radius；
- multiple swap counts；
- sigma threshold；
- delta threshold；
- recurrent weights；
- alpha margin；
- task-specific rule。

第一版固定使用：

```text
within-sample median threshold = 0.5
max swaps = 1
inside candidate = lowest-alpha SAFE
outside candidate = highest-alpha STRESSED
```

## 4. Zero-GPU offline validation first

直接复用已有 P1 discovery segment evidence 和 oracle utility。

不产生新 oracle labels，不运行 GPU。

对每个 discovery sample：

1. 用 raw alpha 得到 top-k set；
2. 应用 one-swap boundary exchange；
3. 根据已有 oracle segment utility 计算两者的 set quality difference；
4. 同时计算 top-k NDCG difference；
5. 记录：
   - 是否存在可执行 swap；
   - swap 是否把更高 oracle-utility segment 换入；
   - per-sample delta；
   - LongEval / Needle 分任务方向。

这里最重要的不是复杂统计，而是回答：

```text
Does replacing an alpha-selected SAFE insider
with a STRESSED outsider improve fixed-budget utility?
```

### Continue condition

如果 discovery offline 中：

- mean top-k utility / NDCG 为正；
- LongEval 和 Needle 至少不出现明显相反；
- 有足够多 sample 真正发生 boundary swap；

则冻结该 exact controller，并直接进入 fresh 8K held-out。

不要求 discovery CI 必须严格排除 0。

### Stop condition

如果：

- 大多数 sample 没有可执行 boundary swap；或
- swap 后 utility/NDCG 总体为负；或
- LongEval/Needle 明显方向冲突；

则停止这个 controller，不再调 swap 数量和 threshold，回 OpenChat 讨论是否收窄为 mechanism contribution 或重新定义 problem。

## 5. Fresh 8K held-out if offline is positive

沿用现有 E3-v2 infrastructure：

- Qwen3.5-0.8B；
- 8K context；
- segment 256；
- 10% exact-KV budget；
- Needle + LongEval；
- new sample IDs / new seed；
- equal-byte oracle；
- raw alpha baseline；
- same recurrent signal extraction。

不需要：

- 重跑 P0-A/B/C/D；
- 新 preflight；
- generic reviewer gate；
- 新 provenance machinery；
- duplicate sanity run。

只需要对 boundary-exchange membership logic 做轻量 unit test。

### 8K continuation rule

值得继续 16K 的情况：

1. NDCG/top-k utility 整体为正，pairwise 不明显回退；
2. LongEval 明显正、Needle 基本中性；
3. 多数真正发生 swap 的 sample 呈正收益。

如果两个 task 都明显负，则停止 conditional-controller 主线。

## 6. Paper framing if boundary exchange works

主线可以进一步收敛为：

### Observation
Attention-only KV importance is incomplete in hybrid LLMs.

### Mechanism
Under high recurrent pressure, large vs small DeltaNet update separates stressed from safe segments with different exact-KV utility.

### Failure of naive integration
Universal multiplicative, additive, and local-rank corrections fail because they treat recurrent evidence as a global scalar and often do not change budget membership.

### HMO
Attention first proposes the exact-KV set; recurrent-state regimes then conditionally revise membership at the budget boundary.

可以用一句话概括：

```text
Attention proposes; recurrent memory revises.
```

这是比原始 `phi = alpha * sigma` 更自然的 paper abstraction。

## 7. Immediate Codex task

Use lightweight ARIS execution.

1. Reuse existing P1 discovery artifacts only.
2. Implement offline one-swap boundary exchange evaluation.
3. Report:
   - number of samples with executable swaps;
   - mean/per-task top-k utility delta;
   - mean/per-task NDCG delta;
   - positive/negative swap counts;
   - a compact per-sample table.
4. Do not search any new hyperparameter.
5. If offline direction is positive and not task-conflicting, freeze this exact one-swap controller.
6. Add only lightweight membership unit tests.
7. Run one fresh 8K held-out confirmation directly.
8. If positive or LongEval-positive/Needle-neutral, continue to 16K without another process gate.
9. If clearly negative on both tasks, stop this controller and return to OpenChat.

Do not move to 27B, Kimi, Refresh, RTS, new recurrent signals, or full baseline tables before this boundary-allocation hypothesis is resolved.
