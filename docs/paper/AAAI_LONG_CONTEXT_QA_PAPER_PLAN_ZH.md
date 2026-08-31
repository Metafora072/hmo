# AAAI 论文规划：面向证据型长上下文 QA 的 HMO

这份文档是 `AAAI_LONG_CONTEXT_QA_PAPER_PLAN.md` 的中文版本，并根据当前 v6 方法和实验结果做了更适合写论文的中文化整理。建议不要把论文定位成“HMO 解决所有长上下文任务”，而应聚焦为：

> Hybrid-attention LLM 的长上下文压缩不能只做 KV cache pruning。对于证据型长上下文 QA 和检索任务，HMO 通过 hybrid-memory signals 判断每个上下文片段应该保留完整 KV、执行 refresh、压缩成 RTS skeleton，还是从活跃 KV 中移除。

## 推荐论文范围

### 主任务族

论文建议聚焦在：

```text
Evidence-centric long-context QA and retrieval
证据型长上下文问答与检索
```

这类任务的共同特点是：模型需要在很长的上下文中保留稀疏但关键的证据。当前结果中最适合放正文的任务是：

- Needle：隐藏证据检索。
- LongEval-Lines：行级精确位置检索。
- HotpotQA：长上下文多跳问答。
- NarrativeQA：长叙事文本理解与问答。

这些任务建议放进正文主实验。

以下内容建议放到 appendix 或 secondary analysis：

- GovReport：摘要任务需要广泛且连续的语义覆盖，HMO 当前不是最强。
- LCC：代码补全对 token 级精确性要求很高，而且当前样本量只有 18。
- StreamingLLM 和 DuoAttention：如果正文主表太宽，可以放附录。
- 64K no-refresh 实验：作为可扩展性或压力测试，必须清楚标注为 HMO w/o Refresh。
- v6 anchor、v7 uniform RTS / uniform KV、segment length ablation：作为消融或负面设计证据。

## 核心 Motivation

### 一句话 Motivation

现有 KV-cache 压缩方法通常把长上下文记忆视为只存在于 attention KV cache 中，但 hybrid-attention LLM 还会把信息存入 recurrent / linear-attention states，而这些隐式状态在长上下文下可能饱和或遗忘关键证据。

### 中文主线表述

现有长上下文压缩方法大多只管理 attention KV cache，但 Qwen3.5 这类 hybrid-attention LLM 同时具有显式 KV 记忆和隐式 recurrent memory。KV cache 是显式、精确、可按 token 操作的；recurrent memory 是固定容量、隐式、不可直接寻址的。在长上下文场景中，recurrent state 可能出现饱和或证据遗忘，因此仅根据 KV 重要性做压缩并不充分。HMO 的目标是根据 recurrent-state saturation 和 attention-side dependence，对上下文片段进行 KV、refresh、RTS 或 drop 的联合调度。

### Motivation 逻辑链

1. 长上下文 QA 需要模型在数万 token 中保留稀疏但关键的证据。
2. Hybrid-attention LLM 为长上下文推理提供了效率优势，但也引入了异构记忆结构。
3. 这类模型的记忆不是单一 KV cache：
   - Full-attention layers 存储精确 KV cache。
   - Linear / DeltaNet layers 存储压缩的 recurrent states。
4. KV cache 是显式、可删除、可保留的；recurrent memory 是隐式、固定容量的。
5. 在长上下文下，recurrent memory 可能饱和，导致某些证据不能被可靠承载。
6. 现有 KV compression 方法没有判断某个片段是否已经被 recurrent memory 安全表示，也不知道哪些片段必须保留 exact KV 或 refresh。
7. HMO 通过 hybrid-memory reliability signals 判断每个片段的记忆风险，并在固定预算下分配 segment-level actions。

## 主要 Claim

推荐使用这个 claim：

> HMO 在激进 KV 预算下，通过基于 hybrid-memory signals 的 exact KV、refresh 和 RTS skeleton 调度，保持 Full-KV 级别的检索能力，并在长上下文 QA 上取得有竞争力的表现。

不要使用这个 claim：

> HMO 在所有长上下文任务上全面超过所有 baseline。

当前实验结果不支持“全任务全面超越”的说法。更稳妥的定位是：

```text
机制创新 + 极低预算下性能保持 + 面向证据型 QA 的优势
```

## 方法设计原理

### 原理 1：把模型记忆建模为两条通道

HMO 面向 hybrid-attention models。它假设模型中存在两类性质不同的记忆：

| 记忆通道 | 特点 | 压缩风险 |
|---|---|---|
| Attention KV cache | 精确、可按 token 寻址、随上下文长度增长 | 显存开销大 |
| DeltaNet / recurrent state | 固定容量、隐式、高效 | 可能饱和或遗忘证据 |

因此，方法不应该只问：

```text
哪些 KV token 重要？
```

还应该问：

```text
哪些上下文区域已经不能被 recurrent memory 可靠表示？
哪些证据需要 exact KV 或 refresh 支持？
```

这是 HMO 区别于 KV-only baselines 的核心。

### 原理 2：用双信号估计片段可靠性

HMO 将 prompt 切分为固定长度的 segments，默认：

```text
segment_length = 512
```

对每个 segment，HMO 估计两个信号：

```text
sigma: DeltaNet / recurrent-state saturation
alpha: attention-side dependence 或 fragility
phi = sigma * alpha
```

直观解释：

- 高 sigma：该片段对 recurrent state 造成较大压力，可能存在饱和或遗忘风险。
- 高 alpha：生成过程在 attention 侧依赖该片段。
- 高 phi：该片段既有 recurrent memory 风险，又与生成结果相关，是高优先级证据区域。

因此，HMO 的核心优先级不是单纯 KV norm 或 attention score，而是：

```text
recurrent risk × attention dependence
```

### 原理 3：使用 segment-level action space

HMO 为每个 segment 分配一个动作：

| Action | 含义 | 作用 |
|---|---|---|
| KV | 保留完整 exact KV | 用于 sink / recent segments 或关键证据 |
| Refresh | 在 decode 前重放或重算高风险片段 | 恢复高 phi 片段的 exact memory |
| RTS | 保留稀疏 token skeleton | 用低成本覆盖大量中间片段 |
| Drop | 从活跃 KV 中移除 | 用于没有预算的低优先级片段 |

v6.1 默认策略可以概括为：

1. 始终保留第一个和最后一个 segment 为 exact KV。
2. 对中间 segments 按 hybrid priority 排序。
3. 若预算允许，将少量高优先级 segments 设为 refresh。
4. 将剩余预算分配给其他 segments 的 RTS skeleton。
5. 如果某个 segment 分不到任何 RTS token，则 drop。

对于 64K / 27B 实验，如果 refresh 导致 replay-induced OOM，必须把结果标注为：

```text
HMO w/o Refresh
```

不要把它写成完整 HMO。

### 原理 4：使用 budget-matched fairness

主实验应该使用一致的 KV-memory budget：

```text
budget = protected_bytes + keep_ratio * middle_bytes
```

当前最有说服力的设置是：

```text
Model: Qwen3.5-27B
Context length: 32768
keep_ratio: 0.10
```

这个设置下，HMO 与 SnapKV、Quest、SAGE-KV、H2O 和 naive budgeted KV subsets 的比较更公平。

### 原理 5：把 naive budgeted KV 作为诊断 baseline

Budgeted Recent KV 和 Budgeted Uniform KV 不是强 baseline，它们的作用是回答一个公平性问题：

> HMO 的效果是不是只是因为它保留了一小部分 KV？

从 Needle 和 LongEval-Lines 的结果看，答案是否定的：

- Naive recent / uniform KV 在 10% budget 下崩溃。
- HMO、SnapKV、Quest 和 SAGE-KV 仍能保持性能。

这说明：

```text
关键不是保留多少 KV，而是如何根据记忆状态选择和调度。
```

## 推荐主实验

### 正文主表

正文主表建议只使用这四个任务：

| Dataset | Metric | 放正文的理由 |
|---|---|---|
| Needle | Accuracy | 经典隐藏证据检索 |
| LongEval-Lines | Accuracy | 精确行级位置检索 |
| HotpotQA | F1 | 多跳长上下文 QA |
| NarrativeQA | F1 | 长叙事 QA，HMO 当前最强 |

正文主表建议包含这些方法：

| Method | 是否放正文 | 备注 |
|---|---|---|
| Full KV | 是 | 上界参考 |
| Budgeted Recent KV | 是 | 朴素同预算 baseline |
| Budgeted Uniform KV | 是 | 朴素同预算 baseline |
| H2O | 是 | 经典 KV eviction baseline |
| SnapKV | 是 | 强 KV compression baseline |
| Quest | 是 | 近期强 baseline；如果不是官方实现，标注 Quest-lite |
| SAGE-KV | 是 | 近期 baseline；如果不是官方实现，标注 SAGE-KV-lite |
| HMO | 是 | 主方法 |

可以放附录或补充表的方法：

- PyramidKV-lite
- StreamingLLM
- DuoAttention

### 当前主结果摘要

基于 `v6_1_27b_32k_keep010_1`：

| Dataset / Metric | Full KV | H2O | SnapKV | Quest | SAGE-KV | HMO |
|---|---:|---:|---:|---:|---:|---:|
| Needle Acc | 1.0000 | 0.7800 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| LongEval Acc | 1.0000 | 0.1200 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| HotpotQA F1 | 0.6795 | 0.6871 | 0.6787 | 0.6787 | 0.6268 | 0.6741 |
| NarrativeQA F1 | 0.3140 | 0.2899 | 0.3105 | 0.3118 | 0.3085 | 0.3162 |

主要解释：

- HMO 在 Needle 和 LongEval 上达到 Full KV / SnapKV / Quest 级别。
- HMO 在 HotpotQA 上略低于 Full KV、SnapKV、Quest 和 H2O，但差距较小，仍具竞争力。
- HMO 在 NarrativeQA 上取得最高分。
- HMO 在检索任务上显著优于 naive budgeted KV subsets。

### 附录表

GovReport 和 LCC 建议放附录：

| Dataset | 放附录的原因 |
|---|---|
| GovReport | 摘要任务更依赖广泛连续语义覆盖，HMO 当前较弱 |
| LCC | 代码补全需要 token 级精确性，且当前样本量较小 |

这不是隐藏负面结果，而是更准确地界定方法适用范围：

> HMO 面向证据型长上下文 QA / retrieval，而不是通用长上下文生成。

## 必要消融实验

AAAI 审稿人会关心每个设计到底有没有作用。建议至少放一个紧凑的 ablation table：

| Variant | 目的 |
|---|---|
| Full HMO | 主方法 |
| HMO w/o Refresh | 检查 refresh 的贡献和代价 |
| HMO w/o Alpha, sigma only | 检查 attention-side fragility 信号作用 |
| HMO w/o RTS | 检查 sparse skeleton coverage 作用 |
| Budgeted Recent KV | 检查朴素 recent-window retention |
| Budgeted Uniform KV | 检查朴素 global coverage |

如果时间有限，最小消融可以是：

```text
HMO
Budgeted Recent KV
Budgeted Uniform KV
HMO w/o Refresh
```

## 推荐图示

### Figure 1：Hybrid Memory Mismatch

画出 hybrid-attention model 的两类记忆：

- Full-attention KV cache：显式记忆。
- DeltaNet / recurrent state：隐式固定容量记忆。
- 长上下文 segments 同时流经两条 memory channels。
- recurrent state 饱和导致某些证据有遗忘风险。

### Figure 2：HMO Segment Policy

沿长上下文时间轴画出 segments，并用不同颜色表示：

```text
KV | Refresh | RTS | Drop
```

这张图要让读者一眼看到 HMO 不是简单 token pruning，而是在做 memory orchestration。

### Figure 3：Accuracy-Memory Tradeoff

画：

```text
x-axis: tracked KV memory
y-axis: primary score
```

使用 Needle、LongEval、HotpotQA、NarrativeQA。

重点突出：

- HMO 与 SnapKV / Quest 使用相近预算。
- HMO 的动作空间更丰富，能利用 hybrid-memory signals。

## AAAI 论文框架大纲

### 标题选项

最推荐：

```text
Beyond KV Cache: Hybrid Memory Orchestration for Long-Context Question Answering
```

备选：

```text
Hybrid Memory Orchestration for Evidence-Centric Long-Context QA
```

更技术化：

```text
Hybrid-Memory-Aware KV Compression for Hybrid-Attention Language Models
```

如果中文内部讨论，可以叫：

```text
超越 KV Cache：面向长上下文问答的混合记忆调度
```

### Abstract 写法

摘要结构建议：

1. 长上下文 LLM 越来越多使用 hybrid attention 来提高效率。
2. 现有 KV compression 方法忽略 recurrent memory。
3. HMO 估计 recurrent saturation 和 attention fragility。
4. HMO 对 segment 分配 KV、refresh、RTS、drop。
5. 在 Qwen3.5-27B、32K、10% KV budget 下，HMO 保持检索性能，并在 NarrativeQA 上取得最好结果。
6. 结果说明，证据型长上下文 QA 需要 hybrid-memory-aware compression。

英文摘要草稿：

> Long-context language models increasingly rely on hybrid attention architectures that combine exact full-attention layers with efficient recurrent or linear-attention layers. However, existing KV-cache compression methods largely treat the KV cache as the sole memory substrate, ignoring whether information is reliably preserved in the recurrent state. We introduce Hybrid Memory Orchestration (HMO), a training-free inference-time controller for hybrid-attention LLMs. HMO partitions the prompt into segments, estimates recurrent-state saturation and attention-side fragility, and assigns each segment to exact KV retention, refresh, sparse RTS skeleton retention, or dropping under a fixed memory budget. On Qwen3.5-27B with 32K contexts and a 10% KV budget, HMO matches Full-KV-level performance on retrieval tasks, achieves the best score on NarrativeQA, and substantially outperforms naive budget-matched KV subsets. These results suggest that long-context QA in hybrid-attention LLMs benefits from orchestrating both explicit KV cache and implicit recurrent memory.

中文摘要草稿：

> 长上下文语言模型越来越多地采用 hybrid attention 架构，将精确的 full-attention layers 与高效的 recurrent 或 linear-attention layers 结合。然而，现有 KV-cache compression 方法通常将 KV cache 视为唯一的记忆载体，忽略了信息是否已经被 recurrent state 可靠保存。本文提出 Hybrid Memory Orchestration (HMO)，一种无需训练的推理时记忆控制器。HMO 将长上下文划分为 segments，估计 recurrent-state saturation 和 attention-side fragility，并在固定记忆预算下将每个 segment 分配为 exact KV retention、refresh、sparse RTS skeleton retention 或 drop。在 Qwen3.5-27B、32K context 和 10% KV budget 下，HMO 在检索任务上保持 Full-KV 级别表现，在 NarrativeQA 上取得最佳结果，并显著优于朴素同预算 KV 子集方法。这表明，hybrid-attention LLM 的长上下文 QA 需要同时调度显式 KV cache 和隐式 recurrent memory。

### 1. Introduction

推荐段落逻辑：

1. 长上下文 QA 需要模型在数万 token 中保留稀疏证据。
2. Hybrid-attention LLM 提高了长上下文推理效率，但引入异构记忆。
3. KV compression 不充分，因为它忽略 recurrent-state saturation。
4. Evidence-centric QA 对证据遗失非常敏感，漏掉一个关键 segment 就可能答错。
5. HMO 联合监控 recurrent 和 attention signals，并选择 segment actions。
6. 总结贡献和实验结果。

贡献点可以写成：

- 我们指出 hybrid-attention LLM 中 KV-only compression 存在 hybrid-memory mismatch。
- 我们提出 HMO，一个无需训练的 inference-time memory controller，利用 recurrent saturation 和 attention fragility 进行调度。
- 我们定义 segment-level action space，包括 exact KV、refresh、RTS skeleton 和 drop。
- 我们在 Qwen3.5-27B、32K、10% KV budget 下验证 HMO，在证据型长上下文 QA / retrieval 中保持强性能。

### 2. Related Work

建议小节：

1. Long-context language models and hybrid attention。
2. KV cache compression and token eviction。
3. Retrieval-oriented long-context evaluation。
4. Dynamic inference-time memory management。

相关工作定位：

- H2O、StreamingLLM、SnapKV、PyramidKV、Quest、SAGE-KV 主要是 KV-centric。
- HMO 的不同点是把 recurrent-state reliability 也作为决策信号。

### 3. Problem Setup

需要定义：

- Prompt tokens 被切分为 segments。
- 模型包含 full-attention layers 和 DeltaNet / linear layers。
- KV budget。
- 目标是在 tracked KV memory budget 下最大化 QA / retrieval performance。

可用符号：

```text
x = [x_1, ..., x_T]
S_i = 第 i 个 segment
M_KV = 显式 KV cache
M_R = recurrent state
B = memory budget
```

### 4. Method: Hybrid Memory Orchestration

#### 4.1 Long Context Segmentation

默认 segment 长度：

```text
512 tokens
```

为什么用 segment-level：

- Token-level 决策开销大且噪声高。
- Whole-document compression 太粗。
- Segment-level 更符合长上下文证据粒度。

#### 4.2 Hybrid-Memory Signals

定义：

```text
sigma_i: recurrent saturation score
alpha_i: attention fragility / dependence score
phi_i = sigma_i * alpha_i
```

解释：

- `sigma_i` 衡量 recurrent state 对该 segment 的记忆压力。
- `alpha_i` 衡量 attention 侧对该 segment 的依赖。
- `phi_i` 衡量该 segment 是否既危险又重要。

#### 4.3 Segment Action Space

解释四个动作：

- KV
- Refresh
- RTS
- Drop

需要说明：

```text
完整 HMO 包含 refresh。
64K no-refresh 是工程可行性变体，不应和完整 HMO 混写。
```

#### 4.4 Budgeted Action Assignment

策略：

1. 保护 sink 和 recent segments。
2. 对中间 segments 按 phi 排序。
3. 若预算允许，将高风险 segments 分配为 refresh。
4. 将剩余预算分配给 RTS skeletons。
5. 没有分到预算的 segments 被 drop。

#### 4.5 Complexity And Memory Accounting

报告 tracked KV memory，而不只报告 peak VRAM。

需要诚实说明：

> 当前 prototype 仍然执行完整 prefill / probing，因此 peak VRAM 不一定按 tracked KV memory 等比例下降。

### 5. Experiments

#### 5.1 Setup

主设置：

```text
Model: Qwen3.5-27B
Context length: 32768
Budget: keep_ratio = 0.10
Samples: 50，LCC 在附录中说明样本数较少
```

主数据集：

- Needle
- LongEval-Lines
- HotpotQA
- NarrativeQA

Baselines：

- Full KV
- Budgeted Recent KV
- Budgeted Uniform KV
- H2O
- SnapKV
- Quest-lite
- SAGE-KV-lite
- HMO

#### 5.2 Main Results

使用 4-task 主表。

核心信息：

- 检索任务：HMO 与 Full KV 和强 baseline 持平。
- NarrativeQA：HMO 最优。
- HotpotQA：HMO 有竞争力，略低于最优方法。
- Naive budgeted KV 在精确检索任务上崩溃。

#### 5.3 Memory Budget Analysis

展示 tracked KV memory：

- Full KV 约 1.0-2.1 GB，取决于任务长度。
- HMO 在 10% budget 下约 0.15-0.27 GB。

不要过度声称 peak VRAM 降低，因为当前实现仍有 prefill/probe 峰值。

#### 5.4 Ablation Study

比较：

- HMO
- HMO w/o Refresh
- sigma-only
- no RTS
- budgeted recent
- budgeted uniform

#### 5.5 Appendix Results

包括：

- GovReport
- LCC
- StreamingLLM
- DuoAttention
- PyramidKV-lite
- 64K no-refresh stress test

### 6. Discussion

建议讨论点：

- 为什么 HMO 对 NarrativeQA 有帮助：长叙事 QA 需要跨上下文整合证据。
- 为什么 GovReport 较弱：摘要任务需要广泛连续覆盖，而不是稀疏证据保持。
- 为什么 LCC 困难：代码补全需要精确变量名、函数名和 token。
- 为什么 tracked memory 和 peak VRAM 在 prototype 中不同。

### 7. Limitations

必须明确写：

- 当前实现是 prototype，仍有完整 prefill / probe 的 peak memory。
- 如果 Quest / SAGE-KV / PyramidKV 是 lite implementation，需要明确说明，官方实现比较是未来工作。
- HMO 最适合证据型 QA / retrieval，不是通用 summarization / code generation 方法。
- Refresh 在 64K / 27B 下如果没有 chunked replay，会带来较高工程成本。

### 8. Conclusion

结尾可以写：

> Hybrid-attention LLMs require hybrid-memory-aware compression. HMO shows that orchestrating explicit KV and implicit recurrent memory can preserve long-context QA and retrieval performance under aggressive KV budgets.

中文版本：

> Hybrid-attention LLM 的长上下文压缩需要 hybrid-memory-aware 的方法。HMO 表明，在激进 KV 预算下，同时调度显式 KV cache 与隐式 recurrent memory，能够保持长上下文 QA 和检索任务的性能。

## 一段话版论文定位

这篇论文应围绕 evidence-centric long-context QA 展开，而不是围绕所有长上下文生成任务展开。核心观察是：hybrid-attention LLM 将记忆分布在 exact KV cache 和 implicit recurrent states 两类结构中，而现有 compression 方法大多只优化 KV 侧。HMO 通过估计 recurrent saturation 和 attention dependence 来弥合这一 mismatch，并在固定预算下将每个 segment 分配为 exact KV、refresh、RTS skeleton 或 drop。在 Qwen3.5-27B、32K context 和 10% KV budget 下，HMO 保持 Full-KV 级别的检索性能，在 NarrativeQA 上取得最佳结果，并显著优于朴素同预算 KV 子集方法，说明 hybrid-memory-aware orchestration 对长上下文 QA 是有价值的。

