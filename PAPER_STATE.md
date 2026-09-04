# HMO Paper State

## 当前状态

论文处于故事冻结与证据扩展阶段。核心方法不再继续搜索新的 recurrent
打分公式，当前任务是围绕已经验证的 locality-preserving mechanism
建立完整论文，并用少量高收益实验补足 Pareto、真实任务和模型规模证据。

目标会议暂按 ICLR 规划，正文页数按 9 页控制。工作标题为：

> HMO: A Locality-Preserving KV Overlay for Hybrid-Attention Language Models

## 一句话贡献

HMO 将 Hybrid LLM 中仍然线性增长的 Full-Attention KV 重构为 recurrent
state 之上的局部高保真 overlay，并通过 query-guided contiguous coverage
在固定显存下避免离散 token 保留对完整关系证据的破坏。

## 论文靶子

### 系统靶子

Hybrid LLM 已通过 DeltaNet 等 recurrent state 压缩大部分全局上下文，但
剩余 Full-Attention 层的 KV cache 仍随上下文线性增长。HMO 的目标始终是
压缩这部分 residual KV，同时保留 recurrent state 和模型原生推理路径。

### 机制靶子

经典 KV compression 通常将 token utility 视为可独立排序的 modular
quantity。该处理能够保留较高的 attention mass，却可能把一个完整局部
关系拆成彼此孤立的 token。HMO 针对的是 importance retention 与 relational
completeness 之间的错配。

### V6.1 与当前版本的关系

V6.1 与当前版本拥有相同的系统对象、显存目标和双记忆视角。V6.1 试图用
`sigma * alpha` 显式估计 recurrent memory 对每个 segment 的可靠性，再决定
KV 分配；当前版本将这种不稳定的数值耦合改为结构化职责分工：recurrent
state 提供全局、有损、低成本的上下文基础，KV overlay 保留 query-relevant
的连续局部高保真证据。两者是同一研究靶子下的设计演化，不是方向切换。

## 方法状态

当前 HMO 包含以下动作层级：

1. 固定保护 prefix 与 suffix anchor。
2. 将中间上下文划分为长度为 256 token 的 segment。
3. 在覆盖阶段，为 segment 保留一个由 query attention 选择的连续窗口。
4. 在 fidelity 阶段，将少量高需求 segment 从 Sparse 升级为 Exact KV。
5. 在整个执行过程中保持 DeltaNet recurrent state 不变，并按真实 resident
   KV bytes 进行预算核算。

当前实验配置使用 16-token contiguous window 和 10% eligible-middle cap。
Exact upgrade 是框架中的可选 fidelity 层，不作为现阶段唯一或首要贡献。

正式定义上，`HMO family = mandatory locality-preserving coverage + optional
fidelity upgrades`。主算法允许 `m=0`，因此当前最强的 locality evidence 与
完整 action hierarchy 可以同时成立。Recurrent global memory 与 KV overlay
的分工属于 architecture-grounded design principle，不表述为已经由实验
单独证明的协同效应。

## 理论状态

### 完整局部证据覆盖

设一个 segment 含有 `n` 个有序 token，策略固定保留 `k` 个位置，任务证据
是长度为 `ell` 的未知连续区间，其中 `2 <= ell <= k`。若保留位置形成连续
run，长度分别为 `r_1, ..., r_m`，则能够被完整覆盖的 `ell` 长度区间数为：

$$
N_{\ell}(R)=\sum_{j=1}^{m}\max(r_j-\ell+1,0)\leq k-\ell+1.
$$

单个长度为 `k` 的连续窗口达到上界。该命题说明，在相同 token 数量下，
连续保留最大化完整局部证据的潜在存活数量；离散选择虽然可能获得更高的
singleton attention mass，却无法保证 relational completeness。

### Query-guided window placement

在宽度为 `k` 的所有连续窗口中，HMO 选择 query attention mass 最大者：

$$
W^*=\arg\max_{W:\lvert W\rvert=k}\sum_{t\in W}a_t.
$$

该选择在 locality class 内最大化保留的 query demand。它不声称超过全局
scattered Top-token 的 singleton mass，而是把局部完整性约束与 query
相关性结合起来。

### KV 保留比例

对长度为 `T` 的中间上下文、segment 长度 `L`、Sparse 宽度 `w` 和 `m` 个
Exact upgrade，保留 token 数满足：

$$
N_{\mathrm{keep}}\leq \left\lceil\frac{T}{L}\right\rceil w+m(L-w).
$$

忽略边界取整时，中间上下文的 KV 保留比例满足：

$$
\rho_{\mathrm{middle}}\lesssim\frac{w}{L}+\frac{m(L-w)}{T}.
$$

固定 `L`、`w` 和 upgrade 比例时，空间复杂度仍是 `O(T)`；贡献是显著降低
线性项系数，而不是改变渐近复杂度阶数。

## 当前核心证据

正式 fresh confirmation 使用 Qwen3.5-0.8B、8K/16K、Needle 与
LongEval-Lines，共 48 个未筛选样本。

| 系统 | Answer Containment | Mean Resident KV |
|---|---:|---:|
| Contiguous CF | 34/48，70.83% | 19,406,592 bytes |
| Scattered CF | 27/48，56.25% | 19,406,592 bytes |
| Raw Exact Top-K | 32/48，66.67% | 19,173,120 bytes |
| Contiguous Sparse-only | 32/48，66.67% | 19,406,592 bytes |
| Full KV | 35/48，72.92% | 148,934,400 bytes |

Contiguous CF 相对 Scattered CF 在逐样本严格等字节条件下提升 14.58 pp，
得到 7 wins、41 ties 和 0 losses；8K 与 16K 的方向均为正。Contiguous CF
的平均逐样本 KV footprint 为 Full KV 的 13.38%，质量与 Full KV 仅相差
2.08 pp。

## Claim Ladder

### 正文核心主张

Hybrid LLM 的 residual KV 应承担局部高保真 overlay，而不应继续采用纯
singleton importance 进行离散保留。Query-guided contiguous retention 在
相同 KV bytes 下显著改善当前模型上的长上下文生成质量。

### 系统主张

HMO 将 Full-Attention KV footprint 降至平均约 13.38%，同时在当前评测中
达到接近 Full KV 的 answer containment。

### 方向性主张

完整 coverage-fidelity policy 相对 Raw Exact Top-K 得到 4.17 pp 的正向
点估计，优势主要来自 16K LongEval-Lines。该结果用于支持 Pareto 趋势，
不单独承担全文贡献。

### 当前不需要承担的主张

- allocator 能够精确估计 DeltaNet 遗忘程度；
- recurrent accessibility 已被证明是有效分配信号；
- Exact upgrade 的独立收益已经充分建立；
- 方法在所有任务、模型尺寸和 Hybrid 架构上普遍有效；
- 理论能够保证最终生成答案正确。

## 论文资产

- Fresh confirmation 报告：
  `openchat/codex/share/2026-09-04/contiguous_cf_fresh_confirmation_report.md`
- 包装与理论草案：
  `openchat/codex/share/2026-09-04/optimistic_paper_story_and_theory_review_request.md`
- GPT 审阅跟进：
  `openchat/codex/share/2026-09-04/gpt_story_review_assessment.md`
- 冻结实验协议：`refine-logs/contiguous_cf_confirmation_protocol.json`
- 原始结果：
  `/mnt/nvme0/hmo/runs/contiguous_cf_confirmation_8k16k_s20261005_06_20260903_221800/`

## 下一阶段

### 已授权

- 固化 PAPER_STATE、中文故事板、论文计划和摘要草稿。
- 修正理论复杂度与显存统计口径。

### 待确认 GPU 工作

1. Raw Exact+Slack 严格等字节基线，并扩展 5%/10%/20% Pareto。
2. 32K HotpotQA 真实任务 transfer，先验证当前 0.8B 路径的 Full-KV
   solvability。
3. 根据磁盘和模型能力选择 Qwen3.5-4B/9B 的规模迁移；不默认下载缺失的
   27B/32B 权重。

## 执行原则

后续不使用统计显著性作为方法生死 Gate，也不要求每个组件独立显著。
实验的作用是增强证据面、构造清晰 Pareto 和回应预期 reviewer 问题。
必须继续满足结果真实性、同口径比较、真实字节核算、可复现 provenance
和不隐藏理论假设。
