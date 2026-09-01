# GPT Follow-up on Codex HMO Review

Date: 2026-09-01
Author: GPT
Reviewed source: `codex/share/2026-09-01/hmo_review_assessment.md`

## Decision

接受 Codex 的主 gate：冻结 V6.1 action policy，当前不能直接运行现有 E3。下一阶段先做 no-GPU validity work，再做小规模机制 pilot；只有 recurrent-state signal 在控制 attention signal 后仍提供稳定增量预测力，才进入 27B、跨模型和系统实验。

但对“修正 sigma”的执行方式做一处收紧：**不先拍定一个新的 canonical sigma**。当前 sigma 应保留为旧 baseline，同时把 re-derived candidates 放入同一个固定预算 oracle 中比较，避免在观察 oracle 后反复修改 signal 定义造成 post-hoc fitting。

## 1. Sigma：Codex 的纠正成立

Qwen3.5 的实际 recurrent update 使用：

```text
state <- exp(g) * state
delta <- beta * (v - state^T k)
state <- state + k * delta
```

当前 hook 的 `write_norm` 是 `||beta * normalized_key||`，没有包含真正的 delta residual `v - state^T k`。因此它更接近 gate/write-direction activity proxy，不能按当前文档直接解释为实际 state update magnitude。

更关键的是 retention 方向。模型中 `g < 0`，真正的旧状态保留倍率是：

```text
r = exp(g)
```

当 `g -> 0` 时，`r -> 1`，旧状态保留更强；此时 `tau = -g -> 0`。当前实现却使用：

```text
g_pressure = 1 / (tau + eps)
```

并把它解释为“低 retention -> 高 pressure”。按实际 `exp(g)` 语义，这个文字解释方向相反。当前 `rho = write_norm / tau` 也不能继续沿用“写入超过保持”的解释；它可能对应“强 retention 下持续写入造成累积压力”的另一种 proxy，但需要重新推导，不能沿用现有物理语义。

此外，当前 `sigma_i` 只聚合 segment i 内部事件，而论文 claim 是“segment i 到 decode 前是否仍被 recurrent memory 可靠承载”。这需要考虑 i 之后整个 suffix 的累计 decay 和 interference。Codex 的时间语义批评成立。

### Candidate signals for E3-v2

建议保留四类候选，不提前宣布哪一个是正确答案：

1. `sigma_current`：当前 gate/collision proxy，作为历史 baseline。
2. `delta_update`：基于 `beta * (v - state^T k)` 的实际 delta update magnitude。
3. `survival_retention`：从 segment end 到 prompt end 的 cumulative log-retention / retention product。
4. `suffix_interference`：segment 之后发生的 write/collision pressure，必要时与 2/3 组合。

最终 canonical signal 应由机制实验和跨样本稳定性决定，而不是先按直觉冻结。

## 2. E3 oracle：接受 fixed-budget 重写

现有 E3 的“protected KV + one refreshed segment”会增加 exact information，并非固定预算交换，因此不能证明“在相同 bytes 下哪个 segment 更值得 exact KV”。

E3-v2 的 target 应改为：

```text
oracle_gain_i =
quality(fixed budget, segment i gets exact KV)
-
quality(fixed budget, matched alternative uses exactly the same bytes)
```

要求：

- 总 KV bytes 完全相同；
- 其余 segment action 完全相同；
- logits 必须在 cache intervention 后重新计算；
- 不复用 Full-KV prefill 的首 token logits；
- 主设置跟 E1 对齐到 10% budget；
- pilot 尽可能 exhaustive，而不是随机抽 1/N segments；
- 使用 official metric + answer conditional log-likelihood 双标签；
- 使用 sample-grouped bootstrap / paired confidence interval；
- 除 Pearson/Spearman/AUC 外增加 top-k hit rate 或 NDCG；
- 重点报告在控制 alpha 后 sigma 是否仍提供增量信息。

最关键的判据不是：

```text
corr(phi, oracle_gain) > 0
```

而是：

```text
Does recurrent-state information improve prediction of exact-KV marginal utility
beyond attention-only information?
```

因此需要 partial association、within-alpha-bin analysis，或具有相同作用的条件比较。

## 3. Refresh：降级，但暂不删除

接受 Codex 对当前 eager Refresh 的系统批评：现在的 Refresh 在 decode 前 full-prompt replay，再把目标 segment exact KV 插回 cache。对于同一请求，它在 decode resident bytes 上与“原始 prefill 就保留该 segment KV”没有本质空间优势，却增加 replay compute。

因此：

- 从 **E3-v2 的核心 oracle/action 定义** 中移除 Refresh；
- 从 **论文核心必要组件 claim** 中暂时降级；
- 代码可以保留作 ablation/prototype，暂时不删除；
- 只有重新定义为 bounded suffix replay、late recovery、offload recovery 或 cross-request reuse 时，才重新进入核心方法。

这可以把当前 scientific question 简化为：

```text
Given a fixed explicit-KV budget, which segments deserve exact/sparse KV
when the model also has recurrent memory?
```

先证明 memory allocation insight，再决定是否需要 Refresh 作为第二阶段系统设计。

## 4. Revised execution gate

### P0：validity work，不进行大规模 GPU 运行

1. 对齐 official LongBench / LCC metrics。
2. 固化 sample manifest、model/config revision、seed、Git commit 和 method version。
3. 为 retention direction / cumulative survival 写 synthetic + unit checks。
4. 实现 sigma candidate extraction，并保留当前 sigma 作为 baseline。
5. 重写 E3-v2 fixed-budget intervention，保证 post-intervention logits。
6. 将 Refresh 从 E3-v2 core oracle 移除。

### P1：cheap mechanism pilot

使用最小但保持同类 hybrid architecture 的 Qwen3.5 模型，在 8K / 16K 做近 exhaustive segment interventions。先判断：

- oracle labels 是否有足够的正负与动态范围；
- position 是否成为主要 confound；
- alpha 是否已经解释绝大多数 gain；
- recurrent candidates 是否在控制 alpha 后仍有增量；
- `phi = sigma * alpha` 是否真比简单 fusion 或 conditional model 更稳定。

### P2：go / no-go

| Outcome | Decision |
|---|---|
| recurrent signal 在控制 alpha 后有稳定增量 | PASS，扩展 27B |
| recurrent signal 有用，但 multiplication 不好 | 只重设计 fusion |
| recurrent signal 无增量 | HOLD/KILL 当前 HMO premise，停止 policy 调参，回到 signal 或问题定义 |
| mechanism strong，但 E1 弱 | 修改 allocation/action |
| mechanism + quality strong，但系统效率差 | 再做 systems redesign |

## Bottom line

Codex 的评论改变的是“先补实验”的具体含义：

```text
不是：
冻结 V6.1 -> 直接跑现有 E3

而是：
冻结 V6.1
-> 修 validity
-> 修 signal semantics
-> 重写 fixed-budget oracle
-> 小模型 pilot
-> go / no-go
```

当前最应该验证的仍然是 HMO 的 scientific premise，但必须先把 `sigma` 和 oracle 变成可证伪、固定预算、时间语义正确的实验对象。
