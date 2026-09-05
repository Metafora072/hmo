# HMO 9B 六任务主表 Result-to-Claim

## 结论

- `claim_supported = partial`
- 原 broad native-QA superiority 主张：不支持。
- 收紧后的 residual-KV efficiency + controlled mechanism 主张：部分支持，
  medium confidence。

独立内部 reviewer 与 Codex 本地诊断一致。该 reviewer 是 Codex 调用的内部
评估，不代表 GPT 或 Opus 的发言。

## 决定性结果

Qwen3.5-9B、506 条未增广/未裁剪 `<=16K` 原生 LongBench QA：

| System | QA-F1 | Mean resident KV | Mean per-case Full fraction |
|---|---:|---:|---:|
| HMO | 0.464177 | 47,098,560 | 14.465% |
| ChunkKV | 0.479314 | 47,098,560 | 14.465% |
| Global Fixed | 0.476590 | 47,098,560 | 14.465% |
| Raw+Slack | 0.479974 | 47,098,560 | 14.465% |
| Full KV | 0.481461 | 338,657,215 | 100% |

HMO 对 ChunkKV 为 62W/372T/72L，均值差 -0.015136，配对 case-bootstrap
95% 区间 [-0.035097,+0.004213]。HMO 不是主表最佳方法，但在削减 85.535%
residual Full-Attention KV 后保留 Full 的 96.41% QA-F1。

## 诊断

HMO 平均保留 70.90% 的 layer-averaged query-attention mass，覆盖 38.62 个
中间 macro-regions；ChunkKV 分别为 79.02% 与 27.41。主表因此揭示的是
coverage/focus trade-off：HMO 为广覆盖付出的 attention-mass 成本在 broad
native QA 上没有平均转化成准确率优势。

`>14K` 后验 slice 中 HMO 对 ChunkKV 为 +0.0154，但任务内趋势不一致，不能
写成已验证的长度定律。MuSiQue 上 HMO 高于 ChunkKV、Global 和 Full，但低于
Raw+Slack；其余任务不形成普遍领先。

## Claim revision

可以写：HMO 以严格 resident-byte 口径把 Hybrid LLM 的 residual
Full-Attention KV 形式化为 recurrent global substrate 上的 stratified local
overlay；在 9B broad native QA 上，14.465% residual-KV footprint 保留 near-Full
质量，并由既有 equal-byte synthetic controls 支撑 locality 与工作区间机制。

不能写：HMO 在 broad native QA 上优于 ChunkKV/structured retention，或
free-start/stratification 普遍提升 downstream quality。

## Route

1. 不直接花 A100 成本复验当前 superiority 叙事。
2. 先在冻结的 9B Needle/LongEval 8K/16K 机制集加入 ChunkKV，仅跑中央 10%。
3. 若存在可信正向工作区间，再做预声明的小型 native 5%/20% sweep。
4. 若尝试 layer-wise 或 coverage/focus 新设计，将当前 506 条仅用于开发诊断，
   最终确认使用未按 observed outcomes 选择的记录。

完整数字、逐任务表、bootstrap 与 provenance 见仓库报告
`../experiments/results/NATIVE_LONGBENCH_SIX_TASK_9B_20260905.md`。
