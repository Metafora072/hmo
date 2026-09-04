# HMO 中文摘要草稿

## 工作标题

HMO: Stratified KV Overlays for Hybrid-Attention Language Models

## 摘要 V3

Hybrid-Attention Language Model 通过 recurrent state 压缩大部分全局上下文，
但其中保留的 Full-Attention 层仍需要随序列长度线性增长的 KV cache。现有
KV 压缩已从 token importance 推进到结构化单元保留，但仍未回答：当模型已有
全局压缩记忆时，稀缺的精确 KV 应如何分布在长上下文区域之间。本文提出
HMO，一种面向 Hybrid-Attention 模型的 stratified KV overlay。HMO 保持原生
recurrent state 不变，先在 macro-regions 间分配 locality-preserving
coverage，再在每个区域内放置 query-guided free-start window，并将剩余预算
用于可选 fidelity。我们证明，在固定 token 预算且证据是未知连续区间时，
连续保留最大化完整存活的候选 evidence spans；进一步的 coverage-floor
分析解释了 residual memory 在 global concentration、stratified coverage
和 saturation 之间的预算-长度转换。在 Qwen3.5-0.8B 与 9B 的全新 8K/16K
synthetic retrieval and line-reasoning suite 上，HMO 在逐样本严格等字节
条件下相对 scattered Top-token retention 提高 14.58 和 16.67 个百分点。
在平均仅使用约 13.38% Full-KV footprint 时，0.8B 上 HMO 达到 70.83%
answer containment，接近 Full KV 的 72.92%；9B 上二者均达到 95.83%。
5%/10%/20% Pareto 进一步显示，global chunk concentration 在 coverage floor
以下更强，而 stratified coverage 在更长上下文且预算越过该门槛后开始占优。
这些结果将 Hybrid residual-KV compression 从单元选择推进为预算与长度共同
决定的 memory organization 问题。

## 摘要写作边界

当前摘要不强调 recurrent-aware allocation，不声称复杂度从 `O(T)` 变为
次线性，也不把 Exact upgrade 写成已经独立验证的唯一创新。HotpotQA-32K-Aug
暂作为 transfer subsection 的 feasibility evidence，不承担摘要中的最优性
结论。

## 备选标题

1. From Concentration to Coverage: Stratified KV Overlays for Hybrid-Attention LLMs
2. Global State, Local Evidence: Stratified KV Overlays for Hybrid-Attention LLMs
3. HMO: Organizing Residual KV in Hybrid-Attention Language Models
