# HMO Design Review: Independent Assessment And Next-Step Gate

Date: 2026-09-01
Author: Codex
Reviewed source: `gpt/share/2026-09-01/hmo_design_review.md`

## Executive Decision

GPT 的战略判断基本正确：当前 HMO 不应继续增加 action、anchor 或打分 heuristic，下一阶段必须围绕核心机制证据展开。

但不能直接运行现有 E3。推荐顺序是：

```text
冻结 V6.1 动作策略
  -> 修正评测口径和结果可追溯性
  -> 修正 sigma 的物理与时间语义
  -> 重写固定预算 E3 oracle
  -> 小模型 pilot
  -> go / no-go
  -> 27B 正式实验、跨模型与系统评测
```

这里“先补实验”指先补实验协议和可证伪机制，不是立即消耗 GPU 重跑现有脚本。

## Supported Parts Of The GPT Review

### 1. E1 does not establish the core claim

正式 E1 中，HMO macro mean 为 0.6018，略低于 SnapKV 的 0.6023 和 Quest-lite 的 0.6033。HMO 明显优于 budgeted recent/uniform KV，但这只能证明选择策略比朴素 KV 子集有效，不能证明 recurrent-aware signal 优于 attention-only signal。

Evidence: `../../../../experiments/results/e1_main/v6_1_27b_32k_keep010_1/result_tables.md`.

### 2. Metrics and baseline fidelity must be repaired before reruns

当前 QA F1 使用 token set intersection，而官方 LongBench 使用 multiset overlap；LCC 当前落到 substring-style accuracy，而官方指标是 code edit similarity。现有 H2O、SnapKV、DuoAttention 等实现也与原方法存在不同程度差异。

因此已有 E1 可作为研发信号，但不能直接作为投稿级主表。正式重跑前必须固定 evaluator、sample manifest、method version、model revision、seed 和完整配置。

### 3. System benefit is not established

E1 的 tracked attention KV 显著下降，但单请求 peak VRAM 没有下降。原始 E1 的 2948 条记录中，TTFT、decode latency 和 throughput 字段全部为零。

这不否定 resident KV saving，但意味着当前只能声称 attention-KV accounting benefit，不能声称端到端显存或吞吐收益。单请求全流程 peak VRAM 还会被 full prefill 支配，正式系统评测应增加 post-prefill resident memory、最大 batch/concurrency、TTFT、steady-state decode throughput 和吞吐-质量 Pareto。

### 4. Novelty must be narrow

HAM 已提出 RNN memory 与显式 KV 的互补结构；DASC 和 HYPIC 分别覆盖 recurrent checkpoint compression 与 hybrid prefix caching。HMO 仍可主张的窄定位是：

> Training-free, inference-time recurrent-state-aware allocation of existing full-attention KV in pretrained hybrid LLMs.

不能宽泛声称首次联合管理 recurrent memory 和 KV memory。

Primary sources:

- HAM: https://arxiv.org/abs/2603.22325
- DASC: https://arxiv.org/abs/2608.30386
- HYPIC: https://arxiv.org/abs/2607.01299
- LongBench evaluator: https://github.com/THUDM/LongBench/blob/main/LongBench/eval.py

## Corrections Missing From The GPT Review

### 1. Current sigma is not yet a valid memory-reliability estimator

The Qwen3.5 recurrent update is:

```text
state <- exp(g) * state
delta <- beta * (v - state^T k)
state <- state + k * delta
```

However, the current hook computes `||beta * normalized_key||`, which is approximately the beta gate magnitude. It does not compute the actual delta residual `v - state^T k` described in the review.

The current code also labels `-g` as retention and uses `1 / (-g)` as decay pressure. Since the actual retention multiplier is `exp(g)`, a value of `g` closer to zero means stronger retention; the documented pressure interpretation therefore requires re-derivation and a direction check.

Finally, `sigma_i` aggregates local events inside segment `i`. The target claim concerns whether segment `i` survives all later updates until decode. A final-memory reliability score must include downstream cumulative decay/interference from the end of segment `i` to the end of the prompt.

Evidence:

- `../../../../experiments/utils/hooks.py`
- `../../../../experiments/utils/saturation.py`
- `../../../../references/qwen3_5_source/modeling_qwen3_5.py`

### 2. Existing E3 oracle is not publication-valid

The current E3 compares protected first/last KV against protected KV plus one exact refreshed segment. This intervention adds information instead of exchanging equal bytes under a fixed budget.

It also reuses prefill logits computed before cache modification. The first generated token is therefore still a Full-KV prediction, which contaminates short-answer QA gains. Other weaknesses are a default keep ratio of 0.5 rather than the E1 value 0.1, sparse random segment testing, F1-only gains, and no sample-grouped uncertainty estimates.

Evidence: `../../../../experiments/phase2/e3_mechanism/run.py` and `../../../../experiments/utils/hmo_controller.py`.

### 3. Eager Refresh is dominated by KV retention

The current policy drops a selected segment, replays the full prompt before decode, and reinserts the exact KV. At decode time this occupies the same KV bytes as retaining that segment from the original prefill, while adding replay compute.

Refresh should therefore be removed from the core method unless it is redefined around a distinct systems constraint such as dynamic late recovery, CPU/NVMe offload, bounded suffix replay, or cross-request reuse. Until then it belongs in an ablation or future-work section.

### 4. Weakened baselines do not make comparison conservative

The project notes that some adapters are weaker than official methods, then describes this as conservative for HMO. The direction is reversed: weakening a comparator favors HMO. Unfaithful ports must either be validated against official implementations on a common supported model or named explicitly as proxies/lite variants.

## E3-v2 Protocol

### Target quantity

For each candidate segment `i`, estimate the marginal utility of assigning exact KV to `i` under a fixed byte budget:

```text
oracle_gain_i = quality(fixed budget, i gets exact KV)
              - quality(fixed budget, matched alternative gets the same bytes)
```

All other actions, bytes and decoding settings must remain identical. Logits used for evaluation must be computed after cache intervention.

### Labels and analyses

- Use official dataset metrics and gold-answer conditional log-likelihood as complementary targets.
- Compare alpha, sigma, phi, individual sigma components, position and random controls.
- Report partial or within-alpha-bin association so sigma must add information beyond alpha.
- Use sample-grouped bootstrap confidence intervals and paired tests.
- Report top-k hit rate or NDCG in addition to Pearson/Spearman/AUC.
- Run exhaustive or near-exhaustive segment interventions on a small pilot before sampling segments at 27B.

### Sigma candidates

- Actual delta update magnitude using `beta * (v - state^T k)` where feasible.
- Correct retention multiplier or cumulative log-retention.
- Survival-adjusted pressure from segment `i` through the remaining suffix.
- Collision/interference occurring after `i`, not only local key similarity inside `i`.

## Execution Order

### P0: no-GPU validity work

1. Vendor or call the official LongBench evaluator.
2. Add result manifests with model revision, config, seed, sample IDs and Git commit.
3. Re-derive sigma and add synthetic/unit checks for gate direction.
4. Rewrite E3 as a fixed-budget intervention with post-mutation logits.
5. Remove Refresh from the default core policy or explicitly redefine its systems use case.

### P1: cheap mechanism pilot

Use the smallest representative Qwen3.5 model at 8K/16K. Test enough segments per sample to inspect label density, variance, position confounds and whether sigma adds predictive value after conditioning on alpha.

### P2: go / no-go

| Outcome | Decision |
|---|---|
| `phi` consistently beats `alpha` with uncertainty bounds excluding zero | Continue HMO and scale E3 to 27B |
| `sigma` helps but multiplication hurts | Redesign fusion only |
| `sigma` adds no information beyond `alpha` | Stop policy tuning; redesign the signal or pivot |
| Mechanism is strong but E1 remains weak | Redesign allocation/actions |
| Mechanism and quality are strong but efficiency is poor | Simplify probes and remove/rebuild Refresh |

### P3: evidence expansion after the gate passes

Run faithful baselines, 5/10/20/30 percent budget curves, Qwen plus Kimi, paired significance analysis, and a corrected system experiment covering resident memory, TTFT, decode throughput and maximum serving concurrency.

## Claims Allowed Today

Supported:

> HMO is a promising hybrid-memory-aware KV allocation hypothesis that is competitive with attention-only compression proxies under an aggressive tracked-KV budget.

Not yet supported:

> Recurrent-state reliability predicts explicit KV demand better than attention-only signals.

> HMO reduces end-to-end GPU memory or improves inference throughput.

> The current four-action controller is necessary or Pareto-optimal.

The next milestone is not a larger E1 table. It is a valid causal/mechanistic answer to whether recurrent-state information provides incremental value for KV allocation.
