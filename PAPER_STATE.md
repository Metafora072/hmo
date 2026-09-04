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

### Closest-work 后的新颖性边界

ChunkKV 已经提出用连续 fixed-boundary chunks 缓解离散 token compression
造成的语义破碎；SentenceKV、ProtoKV 和 Kara 也覆盖了 sentence、semantic
cluster 和 flexible chunk。因此 HMO 不再声称首次发现 locality 或首次使用
chunk。

当前差异收敛为 Hybrid residual-memory organization：HMO 在不改变 recurrent
state 的前提下，将 Full-Attention KV 组织成 stratified local overlay。
它先在 macro-segments 间执行 coverage-first 分配，再在 segment 内选择
query-guided free-start window，最后才执行可选 Exact fidelity。完整审计见
`docs/design/HMO_METHOD_AND_NOVELTY_DOSSIER_ZH.md`。

现有 equal-byte scattered 对照隔离了 retention geometry 的因果效应，但尚未
证明 HMO 优于已有 structured chunk 方法；下一轮应加入 Global Fixed-Chunk
Top-K。

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
3. 在覆盖阶段，为可负担的 segment 保留一个由 query attention 选择的连续
   窗口；预算足够时先覆盖全部 eligible segments，否则优先覆盖高需求
   segments。
4. 在 fidelity 阶段，将少量高需求 segment 从 Sparse 升级为 Exact KV。
5. 将不足以完成 upgrade 的逐 token slack 用于扩展 Sparse window。
6. 在整个执行过程中保持 DeltaNet recurrent state 不变，并按真实 resident
   KV bytes 进行预算核算。

当前实验配置使用 16-token contiguous window 和 10% eligible-middle cap。
Exact upgrade 是框架中的可选 fidelity 层，不作为现阶段唯一或首要贡献。
其中 16 是 base width；slack 扩展后单个 Sparse segment 可保留 17/18 token。

正式定义上，`HMO family = locality-preserving coverage actions + optional
fidelity upgrades`。mandatory 指被 coverage 的 segment 必须保留连续局部
结构，而不是所有预算下每个 segment 都必须覆盖。固定 `w=16,L=256` 时全段
coverage floor 约为 6.25%，所以 5% cap 必然只覆盖部分 segment。主算法允许
`m=0`。Recurrent global memory 与 KV overlay 的分工属于
architecture-grounded design principle，不表述为已经由实验单独证明的协同
效应。

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

对长度为 `T` 的中间上下文、segment 长度 `L`、base Sparse 宽度 `w`、
`c` 个被覆盖 segment、其中 `m` 个 Exact upgrade，以及 `s` 个 slack
extension token，保留 token 数为：

$$
N_{\mathrm{keep}}=cw+m(L-w)+s.
$$

相应的中间上下文 KV 保留比例为：

$$
\rho_{\mathrm{middle}}=\frac{cw+m(L-w)+s}{T}.
$$

只有当全部约 `T/L` 个 segment 被覆盖且 `s` 很小时，第一项才近似为
`w/L`。固定参数时空间复杂度仍是 `O(T)`；贡献是显著降低线性项系数，
而不是改变渐近复杂度阶数。

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

Qwen3.5-9B 的冻结 scale transfer 使用同样的 8K/16K、Needle 与
LongEval-Lines 配置，共 24 个 fresh 样本，并加入严格等字节的 Raw
Exact+Slack。

| 系统 | Answer Containment | Mean Resident KV |
|---|---:|---:|
| Contiguous CF | 23/24，95.83% | 51,757,056 bytes |
| Raw Exact+Slack | 23/24，95.83% | 51,757,056 bytes |
| Scattered CF | 19/24，79.17% | 51,757,056 bytes |
| Contiguous Sparse-only | 23/24，95.83% | 51,757,056 bytes |
| Raw Exact Top-K | 24/24，100.00% | 51,134,464 bytes |
| Full KV | 23/24，95.83% | 397,164,544 bytes |

9B 上 Contiguous 相对严格等字节 Scattered 提升 16.67 pp，4 wins、20
ties、0 losses；平均逐样本 footprint 为 13.38%，与 Full KV 的主指标逐例
完全一致。Contiguous 与 Raw Exact+Slack 在 24/24 个样本上生成 token
完全一致。Raw Exact 唯一的主指标优势来自一个格式敏感的 Needle 答案：
Raw 输出 `8:38 o'clock`，而 Contiguous、Raw+Slack 与 Full 均输出
语义相同但字符串规则未命中的 `8:38`。

统一的 post-hoc format-robust secondary analysis 保持主指标与原始结果不变，
只对 Needle clock answer 增加确定性格式 alias。0.8B 上 Contiguous 为 34/48、
Scattered 为 28/48，差值 `+12.50 pp`、6 wins/0 losses；9B 上分别为 24/24
和 20/24，差值仍为 `+16.67 pp`、4 wins/0 losses。9B 的 Contiguous、
Raw Exact+Slack、Raw Exact 和 Full 在该 secondary metric 下均为 24/24。

因此格式复算消除了 Raw Exact 的单例表面优势，同时也诚实扣除了 0.8B 上
Scattered 的一个格式假阴性；两种口径都支持跨规模 locality 机制。完整结果见
`experiments/results/FORMAT_ROBUST_SECONDARY_20260904.md`。

## Claim Ladder

### 正文核心主张

Hybrid LLM 的 residual KV 应承担局部高保真 overlay，而不应继续采用纯
singleton importance 进行离散保留。Query-guided contiguous retention 在
相同 KV bytes 下在 0.8B 和 9B 两个规模均改善长上下文生成质量。

### 系统主张

HMO 将 Full-Attention KV footprint 降至平均约 13.38%。在 0.8B 上质量
接近 Full KV；在 9B transfer 上 Contiguous 与 Full KV 均达到 23/24，
且二者的主指标逐例一致。

### 方向性主张

Raw Exact 不是当前贡献需要击败的主线：0.8B 上 Contiguous 相对 Raw
Exact 有 +4.17 pp 点估计但字节略多；9B 上 Contiguous 与严格等字节
Raw Exact+Slack 完全持平，而 Raw Exact 在一个格式敏感样本上多记一次
命中。Raw 系列保留为强公平基线，不包装为 HMO 的稳定优势。

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
- 9B scale-transfer 报告：
  `openchat/codex/share/2026-09-04/qwen35_9b_scale_transfer_report.md`
- 9B 冻结协议：`refine-logs/contiguous_cf_scale_transfer_9b_protocol.json`
- 9B 原始结果：
  `/mnt/nvme0/hmo/runs/contiguous_cf_scale9b_formal_c202236_20260904/`

- 方法与 closest-work dossier：
  `docs/design/HMO_METHOD_AND_NOVELTY_DOSSIER_ZH.md`
- Figure 1 storyboard：`docs/paper/HMO_FIGURE1_STORYBOARD_ZH.md`
- 格式鲁棒 secondary report：
  `experiments/results/FORMAT_ROBUST_SECONDARY_20260904.md`


## 下一阶段

### 已完成

- 固化 PAPER_STATE、中文故事板、论文计划和摘要草稿。
- 修正理论复杂度与显存统计口径。
- 实现 Raw Exact+Slack，并完成 Qwen3.5-9B 单卡 24 样本规模迁移。
- 完成 closest-work 审计、方法伪代码/理论假设、Figure 1 storyboard 和统一
  format-robust secondary analysis。

### 待确认 GPU 工作

1. 在 0.8B 的 48 样本集扩展 5%/10%/20% Pareto，并全程包含 Raw
   Exact+Slack；加入同 probe、同 resident bytes 的 Global Fixed-Chunk
   Top-K structured baseline。
2. 32K HotpotQA 真实任务 transfer，先验证当前 0.8B 路径的 Full-KV
   solvability。
3. 9B 在 16K 已达到 27.85 GiB PyTorch reserved 峰值；32K 优先使用
   0.8B/4B，不在当前单卡上直接尝试 9B BF16 或 27B/32B。


## 执行原则

后续不使用统计显著性作为方法生死 Gate，也不要求每个组件独立显著。
实验的作用是增强证据面、构造清晰 Pareto 和回应预期 reviewer 问题。
必须继续满足结果真实性、同口径比较、真实字节核算、可复现 provenance
和不隐藏理论假设。
