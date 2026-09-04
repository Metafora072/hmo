# HMO 中文摘要草稿

## 工作标题

HMO: A Locality-Preserving KV Overlay for Hybrid-Attention Language Models

## 摘要 V1

Hybrid-Attention Language Model 通过 recurrent state 压缩大部分全局上下文，
但其中保留的 Full-Attention 层仍需要随序列长度线性增长的 KV cache。现有
KV 压缩通常独立评估 token 重要性并保留最高分位置；这种方式可以维持较高
的 attention mass，却会把多 token 构成的局部关系拆成离散碎片。本文提出
HMO，一种面向 Hybrid-Attention 模型的 locality-preserving KV overlay。
HMO 保持模型原生 recurrent state 不变，将有限 KV 预算组织为两层动作：
首先通过 query-guided contiguous windows 建立广泛的局部证据覆盖，再将少量
高需求区域升级为 Exact KV 以补充 fidelity。我们证明，在固定 token 预算且
任务证据由未知连续区间构成时，连续保留最大化能够被完整覆盖的局部证据
区间数量。在 Qwen3.5-0.8B 上包含 48 个全新 8K/16K 样本的 synthetic
retrieval and line-reasoning suite 中，HMO 在逐样本严格等字节条件下比
scattered Top-token retention 提高 14.58 个百分点，并在平均仅使用 13.38%
Full-KV footprint 时达到 70.83% 的 answer containment，接近 Full KV 的
72.92%。这些结果表明，在当前 Hybrid 模型和任务范围内，residual KV 更适合
被组织为 recurrent global memory 之上的局部高保真 overlay，而不是独立
重要 token 的无结构集合。

## 摘要写作边界

当前摘要不强调 recurrent-aware allocation，不声称复杂度从 `O(T)` 变为
次线性，也不把 Exact upgrade 写成已经独立验证的唯一创新。后续 Pareto 和
HotpotQA 结果完成后，应替换或补充最后两句中的实证数字，但不改变核心故事。

## 备选标题

1. HMO: Preserving Local Evidence in the KV Cache of Hybrid Language Models
2. Global State, Local Evidence: A KV Overlay for Hybrid-Attention LLMs
3. Do Not Scatter the Evidence: Locality-Preserving KV Compression for Hybrid LLMs
