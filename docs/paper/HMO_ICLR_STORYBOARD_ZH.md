# HMO ICLR 中文故事板

## 叙事总纲

这篇论文不把 HMO 描述为一个新的 token 打分公式，而是提出一种面向
Hybrid-Attention Language Model 的 memory organization。Recurrent state
负责低成本的全局上下文基础，Full-Attention KV 负责局部高保真关系。现有
KV compression 将后者压缩为一组离散高分 token，忽略了关系证据的连续
结构。HMO 将 residual KV 重构为 locality-preserving overlay，并通过
coverage-fidelity hierarchy 在固定预算下实现该分工。

读者最终需要记住的不是 `width=16`，而是以下判断：

> Hybrid 模型已经拥有全局压缩记忆，因此有限 KV 的首要职责不是重复保存
> 更多孤立 token，而是保护 recurrent state 难以精确还原的局部关系。

## 从背景到方法的七步推进

### 第一步：Hybrid 并没有消除 KV 问题

Hybrid 模型用 DeltaNet 等 recurrent layers 将一部分长期记忆压缩为固定
状态，但为了保持精确检索和局部建模能力，模型仍周期性保留 Full-Attention
layers。这些层的 KV cache 依然随上下文长度线性增长。因此，Hybrid 架构
降低了 KV 压力，却没有消除 residual KV bottleneck。

### 第二步：两种记忆具有不同能力边界

Recurrent state 能以固定大小传播全局信息，但其更新过程会混合历史内容，
难以保证任意局部 token relation 的精确可恢复性。Full-Attention KV 保存
显式 token-level states，代价更高，却能支持高保真局部读取。Hybrid 推理
系统的关键问题不是简单地删除多少 KV，而是有限 KV 应该为 recurrent
memory 补充什么。

### 第三步：传统 importance compression 的隐含假设不合适

Top-token selection 将每个 token 的 utility 独立建模，并保留得分最大的
位置。该方法优化 singleton importance 或 attention mass，但许多问答证据
由连续实体、数字、句法结构和跨 token 关系共同构成。离散高分 token 可能
命中证据，却没有完整保留证据。

### 第四步：从 token importance 转向 relational completeness

HMO 的核心观察是，KV 的价值不仅取决于保留了哪些 token，还取决于这些
token 是否形成可被 attention 直接读取的完整局部单元。因此，在相同 token
预算下，连续窗口构成了一种简单而强的结构先验。

### 第五步：Coverage-Fidelity Overlay

HMO 先保护 prefix/suffix anchors，再将中间上下文划分为 segment。Coverage
阶段在 segment 内选择 query-attention mass 最大的连续窗口，使有限 KV
广泛覆盖局部关系。Fidelity 阶段利用剩余预算将高 query demand segment
升级为 Exact KV。DeltaNet recurrent state 始终保留，不被压缩或重置。
正式方法族将 coverage 定义为必选机制，将 fidelity upgrade 定义为可选动作，
并允许 `m=0`。

### 第六步：理论解释

固定保留 `k` 个位置时，一个长度为 `k` 的连续 run 完整包含的
`ell`-token 区间数量最多，为 `k-ell+1`。将位置拆成多个 runs 会减少能够
完整存活的局部证据区间。HMO 再在所有连续窗口中最大化 query-attention
mass，从而同时获得 locality class 内的关系完整性与 query relevance。

### 第七步：经验闭环

在 Qwen3.5-0.8B 的 48 个全新 8K/16K 样本上，Contiguous CF 与 Scattered
CF 使用逐样本完全相同的 resident KV bytes，前者得到 70.83%，后者得到
56.25%。Contiguous CF 以平均 13.38% Full-KV footprint 接近 Full KV 的
72.92%。该结果把问题观察、结构先验、理论命题和 end-task generation 串成
完整证据链。

## 章节故事板

## 第一章 Introduction

### 第一段：直接提出 residual KV bottleneck

不要从大模型取得广泛成功开始。首句直接指出 Hybrid-Attention 模型虽然用
recurrent state 压缩了多数层的历史信息，剩余 Full-Attention layers 仍使
KV memory 随上下文增长。这是部署 Hybrid 模型时尚未解决的结构性成本。

### 第二段：建立 dual-memory asymmetry

说明 recurrent state 和 attention KV 不是容量不同的同类存储，而是两个
能力不同的 memory channels。前者擅长全局压缩，后者擅长 token-level
fidelity。由此提出问题：在已有 recurrent global memory 时，有限 KV 最应
保留什么？

### 第三段：指出 scattered importance 的盲点

现有方法大多优化独立 token importance。用一个直观例子说明，答案由一段
连续 passkey 或多 token relation 构成时，保留若干高分碎片并不等于保留
完整证据。引出 relational completeness。

### 第四段：介绍 HMO

用一段话给出 overlay view、contiguous coverage 和 Exact fidelity 三个
部分。强调 HMO 不修改模型权重、不删除 recurrent state，并在真实 resident
bytes 下执行。

### 第五段：结果预告

报告三组最强数字：严格等字节相对 scattered 提升 14.58 pp；平均 footprint
为 Full KV 的 13.38%；质量为 70.83%，接近 Full KV 的 72.92%。Raw Exact
的 4.17 pp 点提升可作为补充，不承担开篇主结论。

### Contribution bullets

1. 提出 Hybrid memory overlay 视角，明确 recurrent global state 与 local
   high-fidelity KV 的职责分工。
2. 提出 HMO coverage-fidelity policy，以 query-guided contiguous windows
   替代无结构的 scattered token retention。
3. 证明连续保留在固定 token 数下最大化未知局部证据区间的完整覆盖，并给出
   KV retention coefficient。
4. 在真实 cache intervention 和逐样本字节核算下验证 HMO，以约 13.38%
   Full-KV footprint 达到接近 Full KV 的质量，并显著优于等字节 scattered
   retention。

## 第二章 Background And Related Work

### Hybrid sequence modeling

介绍 Full Attention 与 recurrent/linear attention 的状态和缓存语义，解释
Qwen3.5 的 Hybrid 层结构为什么产生 dual-memory system。这里只建立问题，
不把 Qwen3.5 写成唯一适用对象。

### KV cache compression

按照 eviction、heavy-hitter、recent-window、query-aware selection 和
head/layer allocation 分类相关工作。共同点不是它们全部选择 scattered
tokens，而是主流目标函数更偏向 singleton importance，通常没有显式优化
完整局部证据存活。

### Locality and span-structured evidence

讨论 local window、chunk retrieval 和 span-level reasoning 的结构先验。
定位 HMO 的区别：它不是固定 recent window，也不是检索完整外部 chunk，
而是在 Hybrid 模型原生 KV cache 内进行 query-guided local overlay。

## 第三章 Hybrid Memory Overlay

### System model

定义 Full-Attention layer 集合、recurrent layer 集合、KV resident bytes、
segment、protected anchors 和 query suffix。明确 recurrent state 在所有
系统间完全相同，因此比较只改变 attention KV overlay。

### Coverage action

每个被覆盖 segment 保留宽度为 `w` 的连续窗口。窗口位置由 token-level
query-attention mass 的 sliding sum 决定，tie 时选择最早起点，保证结果确定。

### Fidelity action

预算允许时，将高 query demand 的 segment 从 Sparse 升级为 Exact。正文将
其描述为通用 action hierarchy，实验中用 Sparse-only ablation 说明其当前
增益较小，因此不把它包装成唯一关键机制。

### Budget allocator

以真实 KV bytes 而非名义 token 比例计费。保护区、query KV、Sparse windows
和 Exact upgrades 分别核算。Raw Exact+Slack 后续将用于消除 whole-segment
rounding slack。

### Algorithm

主文给出 12 至 18 行伪代码，展示 query probe、segment scoring、coverage、
upgrade 和 cache intervention。实现细节、Qwen3.5 hook 和完整字节公式放入
附录。

## 第四章 Why Locality Matters

### Span survival proposition

给出 `N_ell(R)` 命题、run decomposition 和 merging proof。主文展示证明
核心，完整边界条件放入附录。

### Attention-guided corollary

将 max-mass selection 写成 locality-constrained optimum，不与 scattered
Top-token 的全局 singleton optimum 混淆。

### Retention analysis

给出 `rho_middle` 公式，强调降低 retention coefficient 而非改变 `O(T)`。
讨论 protected/query overhead 导致名义 10% middle cap 与整体 13.38%
footprint 不完全相同。

## 第五章 Experiments

### Research questions

- RQ1：连续局部保留是否优于同 allocator、同 bytes 的离散 token 保留？
- RQ2：HMO 能否在小比例 KV footprint 下接近 Full-KV end-task quality？
- RQ3：Coverage、Fidelity、budget 和 context length 分别如何影响结果？
- RQ4：结论能否迁移到真实长上下文任务和更大 Hybrid 模型？

### Main results

主表报告 Contiguous CF、Raw Exact+Slack、Raw Exact、Scattered CF、Sparse-only
和 Full KV。结果按 8K/16K 与 Needle/LongEval/HotpotQA 分组，同时报告 quality、
resident bytes、相对 Full 的逐样本平均比例和 decode overhead。

### Pareto

在 5%/10%/20% middle cap 下绘制 quality-memory curve。宽度与 allocator
规则保持固定，不根据测试标签调参。图中突出 HMO 是否在多个预算点占据更好
的局部 Pareto frontier。

### Ablations

最重要的 ablation 是 Contiguous versus Scattered。Sparse-only 用于分析
Exact upgrade；protected anchors 和 query guidance 可作为次级 ablation，
避免把正文变成组件穷举。

### Real-task and scale transfer

优先接入 32K HotpotQA 和官方 F1。大模型选择必须是实际可获得且属于目标
Hybrid architecture 的 Qwen3.5 checkpoint。模型规模是增强项，不是当前
故事成立的前置 Gate。

## 第六章 Discussion

### 与 V6.1 的概念关系

正文不讲迭代失败史。Discussion 可以说明 HMO 的核心是双记忆职责分工，
不要求显式预测 recurrent state 对每个 segment 的可靠性。Accessibility
signals 作为未来更精细 fidelity allocation 的方向。

### Generality

Contiguous overlay 可以应用到其他 Hybrid 或 Full-Attention 模型，但本文的
动机来自 Hybrid dual memory。使用 hybrid-oriented 而不是 hybrid-exclusive。

### Limitations

当前主要限制是模型规模较小、真实任务覆盖不足和 Exact upgrade 增益尚不
稳定。不要在正文每节重复这些限制，统一放在 Discussion 末尾。

## 第七章 Conclusion

重申 memory organization 而非某个超参数：当 recurrent state 已承担全局
压缩后，residual KV 应保护完整局部关系。HMO 证明这一简单分工能够显著
优于无结构的等字节保留，并以小比例 KV 接近 Full-KV quality。

## Figure 1 故事板

Figure 1 应横向包含三个部分。

左侧画 Hybrid 模型的 memory anatomy：多数 DeltaNet layers 对应固定大小
global recurrent state，少数 Full-Attention layers 对应随上下文增长的 KV
cache。用不同颜色明确两者职责，不画成两个完全独立模型。

中间画同一答案证据 span 的两种等字节保留。Scattered Top-token 命中多个
高 attention token，但答案 span 被打断；HMO contiguous window 完整覆盖
关系证据。视觉重点是相同 token 数、不同结构完整性。

右侧画结果小图：Contiguous 70.83、Scattered 56.25、Full KV 72.92，并标注
Contiguous 使用平均 13.38% Full-KV footprint。

建议 caption：HMO treats the residual Full-Attention cache as a local
high-fidelity overlay on top of recurrent global memory. At equal resident KV
bytes, contiguous coverage preserves complete relational evidence that
scattered importance retention fragments, closing most of the gap to Full KV.

## Reviewer 预期问题与回答

### 这是否只是 local window

不是固定 recent window。HMO 在每个 segment 内根据当前 query placement，
并在全局预算下组合 protected、Sparse 和 Exact actions。理论与 ablation 直接
解释为何 locality constraint 改善 scattered importance retention。

### 为什么必须是 Hybrid 模型

算法操作 attention KV，因此可推广到 Full-Attention 模型；但 overlay 的
memory-role 解释依赖 Hybrid 模型已有 recurrent global state。本文应表述为
designed for residual KV in Hybrid models，而不是声称只能用于 Hybrid，
也不声称已经实验证明 recurrent state 与 contiguous overlay 之间的特殊协同。

### 为什么不用更复杂的 span detector

连续 max-mass window 是参数少、确定性强、无需训练的结构先验。它在 fresh
实验中已经胜过 scattered selection。复杂 learned detector 不属于证明核心
命题所必需的机制。

### Exact upgrade 是否必要

Exact upgrade 是 coverage-fidelity action space 的自然上层，但当前实证中
Sparse-only 已较强。正文应让 locality-preserving coverage 承担主要结果，
将 fidelity 作为可选系统层和未来优化空间。

### 为什么主比较不是 Raw Exact

Contiguous versus Scattered 是单因素、严格等字节的机制比较。Raw Exact 是
不同粒度 action，其 whole-segment slack 需要 Raw Exact+Slack 和 Pareto
进一步公平化。论文最终应同时报告两类比较。

## 写作禁区

- 不声称将复杂度从 `O(T)` 降为次线性。
- 不声称 allocator 已经感知或量化 DeltaNet 遗忘。
- 不将历史 27B CSV 当作当前方法的大模型正式证据。
- 不把平均逐样本比例 13.38% 与 mean bytes 之比 13.03% 混用。
- 不在 Abstract 和 Introduction 中复述开发失败过程。
- 不让统计 caveat 淹没核心结果；集中放在实验分析与 Limitations。
