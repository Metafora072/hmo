# HMO Conditional Regime Offline Analysis

Date: 2026-09-03
Author: Codex
Scope: existing P1 discovery evidence only; no GPU and no new oracle labels

## Judgment

GPT 对 Opus 的修正是必要的：Opus 从 task-level asymmetry 直接推导 segment-level safe/stressed regime，原先缺少直接证据。固定协议的离线分析现在补上了这段证据。

在 12 个 Qwen3.5-0.8B、8K discovery samples 的 360 条 segment evidence 上，high-sigma/high-delta 的 residual exact-KV utility 高于 high-sigma/low-delta：

    Q4 stressed - Q3 safe = +0.25541
    sample bootstrap 95% CI = [+0.03702, +0.45082]
    positive samples = 9/12
    LongEval mean = +0.28193 (5/6 positive)
    Needle mean = +0.22889 (4/6 positive)

因此，Opus 提出的 safe/stressed 解释获得了直接的 discovery-level 设计依据。当前可以进入最小三状态 controller 的冻结与实现；这并不等于 controller 已经有效，方法效果仍需 fresh held-out confirmation。

## Frozen Analysis Protocol

1. 使用两个既有 P1 discovery runs，共 12 samples、360 segments。
2. 以 alpha + normalized_position 拟合 4-fold sample-grouped OOF ridge baseline，禁止同一 sample 跨 train/test。
3. 对 OOF utility residual 在 sample 内去均值，移除 sample offset。
4. 对 sigma_current、delta_update 分别做 within-sample average-rank normalization。
5. 固定 0.5 中位数边界，不搜索 threshold。
6. 每个 sample 分别计算 Q3、Q4 residual mean，再对 Q4-Q3 做 sample-level bootstrap。
7. 判据仅要求总体方向为正且 LongEval/Needle 不反转，不以 interim CI 作为额外流程 gate。

## Quadrant Evidence

| Regime | Segments | Mean centered residual | Median |
|---|---:|---:|---:|
| Q1 low sigma / low delta | 141 | -0.12495 | +0.04841 |
| Q2 low sigma / high delta | 39 | -0.12786 | +0.07538 |
| Q3 high sigma / low delta, safe | 39 | -0.06870 | +0.01841 |
| Q4 high sigma / high delta, stressed | 141 | +0.17931 | +0.07933 |

低 sigma 时 Q1 与 Q2 几乎相同，而高 sigma 时 Q4 明显高于 Q3。这比单独的 sigma correction 更符合 conditional interaction，而不是新的 universal scalar correction。

## Limits

- 这是 discovery evidence 的机制定位，不是 held-out controller result。
- sigma_current 与 delta_update 高度相关，导致 Q3 只有 39 段、Q4 有 141 段；部分 sample 的 Q3 仅 1 到 2 段，单样本 contrast 较噪。
- 三个 sample 的 contrast 为负，且第一轮两个 Needle samples 均为负，仍存在 seed/sample heterogeneity。
- 因此不应把该结果写成 controller 已解决 top-budget selection；它只支持进入一次冻结设计和判伪实验。

## Decision And Next Action

    Hybrid-memory premise: alive
    universal multiplicative/additive scorer: rejected
    safe/stressed segment-level hypothesis: supported on discovery evidence
    GPU status: not used

下一步建议冻结 Opus 的最小离散 +1/0/-1 rank controller，避免引入 alpha bucket 数量这一新超参数：

    STRESSED: base alpha rank 前移一位
    NEUTRAL:  base alpha rank 不变
    SAFE:     base alpha rank 后移一位

需定义稳定的 collision ordering 并做轻量 unit test，然后使用新 sample IDs 跑一次 8K held-out confirmation。当前尚未实现 controller，也未启动 GPU，等待 PZ 对该具体动作确认。

## Reproduction

Raw result:

    /mnt/nvme0/hmo/runs/p1_conditional_regime_offline_20260903/conditional_regime.json

Command:

    /home/pz/miniconda3/envs/hmo_research_v6/bin/python -m experiments.phase2.e3_v2.conditional_regime \
      --run-dir /mnt/nvme0/hmo/runs/p1_discovery_qwen08b_8k_20260902_192652 \
      --run-dir /mnt/nvme0/hmo/runs/p1_discovery_qwen08b_8k_seed20260903_20260902_194216 \
      --output /mnt/nvme0/hmo/runs/p1_conditional_regime_offline_20260903/conditional_regime.json \
      --folds 4 --ridge-lambda 0.001 --bootstrap-samples 5000 --seed 20260910

Verification: 3 new conditional-regime contract tests and all 80 CPU tests pass.
