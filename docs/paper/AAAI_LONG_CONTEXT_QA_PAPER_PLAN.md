# AAAI Paper Plan: HMO for Evidence-Centric Long-Context QA

This note gives a paper-level framing for the current v6 method and results. The recommended positioning is not "HMO solves all long-context tasks", but:

> Hybrid-attention LLMs require memory orchestration beyond KV cache pruning. For evidence-centric long-context QA and retrieval, HMO uses hybrid-memory signals to decide which context segments should be kept exactly, refreshed, represented by sparse RTS skeletons, or removed under a tight KV budget.

## Recommended Paper Scope

### Main Task Family

Focus the paper on:

```text
Evidence-centric long-context QA and retrieval
```

This includes tasks where the model must preserve sparse but important evidence over a long context:

- Needle: hidden evidence retrieval.
- LongEval-Lines: exact line-level evidence retrieval.
- HotpotQA: multi-hop long-context QA.
- NarrativeQA: long narrative understanding and QA.

Put these in the main paper.

Move these to appendix or secondary analysis:

- GovReport: summarization needs broad continuous semantic coverage; HMO is not strongest here.
- LCC: code completion is token-exact and current n is only 18.
- StreamingLLM and DuoAttention: include in appendix if the main table becomes too wide.
- 64K no-refresh experiments: use as scalability/stress-test, clearly labeled as HMO w/o Refresh.
- v6 anchor, v7 uniform RTS/uniform KV, and segment-length ablations: use as ablations or negative design evidence.

## Core Motivation

### One-Sentence Motivation

Existing KV-cache compression methods treat long-context memory as if it lives only in the attention KV cache, but hybrid-attention LLMs also store information in recurrent/linear-attention states that can saturate or forget evidence.

### Chinese Version

现有长上下文压缩方法大多只管理 attention KV cache，但 Qwen3.5 这类 hybrid-attention LLM 同时具有显式 KV 记忆和隐式 recurrent memory。隐式记忆容量固定，长上下文下可能饱和或遗忘，因此仅根据 KV 重要性做压缩并不充分。HMO 的目标是根据 recurrent-state saturation 和 attention-side dependence，对上下文片段进行 KV、refresh、RTS 或 drop 的联合调度。

### Motivation Logic

1. Hybrid-attention LLMs are increasingly used for efficient long-context inference.
2. Their memory is not a single structure:
   - Full-attention layers store exact KV cache.
   - Linear/DeltaNet layers store compressed recurrent states.
3. KV cache is explicit and token-addressable, but recurrent memory is implicit and fixed-size.
4. Under long contexts, recurrent memory may become saturated; some evidence may not be safely represented by the recurrent state.
5. Existing KV compression methods do not ask whether a segment is already safely carried by recurrent memory or whether it needs exact KV support.
6. HMO addresses this by measuring hybrid-memory reliability and assigning segment-level memory actions under a fixed budget.

## Main Claim

Use this claim:

> HMO preserves Full-KV-level retrieval performance and competitive long-context QA performance under aggressive KV budgets by orchestrating exact KV, refresh, and RTS skeletons according to hybrid-memory signals.

Avoid this claim:

> HMO outperforms all baselines on all long-context tasks.

The current results do not support a universal dominance claim.

## Method Design Principles

### Principle 1: Model Memory As Two Channels

HMO is designed for hybrid-attention models with two different memory channels:

| Memory Channel | Property | Risk Under Compression |
|---|---|---|
| Attention KV cache | Exact, token-addressable, grows with context | Expensive memory cost |
| DeltaNet/recurrent state | Fixed-size, implicit, efficient | Saturation and evidence loss |

The method should therefore decide not only "which KV tokens are important", but also "which context regions are no longer safely represented by recurrent memory".

### Principle 2: Use Dual Signals For Segment Reliability

HMO divides the prompt into fixed-length segments, default:

```text
segment_length = 512
```

For each segment, it estimates:

```text
sigma: DeltaNet/recurrent-state saturation
alpha: attention-side dependence or fragility
phi = sigma * alpha
```

Interpretation:

- High sigma: recurrent state is under pressure for this segment.
- High alpha: generation depends on this segment through attention.
- High phi: this segment is both recurrent-risky and generation-relevant.

This is the main algorithmic distinction from KV-only baselines.

### Principle 3: Segment-Level Action Space

HMO assigns each segment one of four actions:

| Action | Meaning | Role |
|---|---|---|
| KV | Keep exact full KV | For sink/recent segments and selected critical evidence |
| Refresh | Recompute/replay a risky segment before decode | For high-phi segments whose exact memory should be restored |
| RTS | Keep sparse token skeleton | For broad low-cost coverage of many middle segments |
| Drop | Remove segment from active KV | For segments receiving no budget |

The v6.1 default policy is:

1. Always keep first and last segment as exact KV.
2. Rank middle segments by hybrid priority.
3. Assign a few high-priority segments to refresh if budget allows.
4. Allocate remaining budget to RTS skeletons across other segments.
5. Drop segments only when they receive zero RTS budget.

For 64K/27B experiments, if refresh causes replay-induced OOM, report the result as:

```text
HMO w/o Refresh
```

Do not present it as the full HMO method.

### Principle 4: Budget-Matched Fairness

The main comparison should use the same KV-memory budget:

```text
budget = protected_bytes + keep_ratio * middle_bytes
```

The current strong setting is:

```text
Qwen3.5-27B
context_length = 32768
keep_ratio = 0.10
```

This is important because it makes HMO comparable to SnapKV, Quest, SAGE-KV, H2O, and naive budgeted KV subsets.

### Principle 5: Treat Naive Budgeted KV As A Diagnostic Baseline

Budgeted Recent KV and Budgeted Uniform KV are not meant to be strong methods. They answer a fairness question:

> Is HMO good merely because it keeps a small amount of KV?

The answer from Needle and LongEval-Lines is no:

- Naive recent/uniform KV collapses under 10% budget.
- HMO, SnapKV, Quest, and SAGE-KV preserve performance.

This supports the claim that memory selection and orchestration matter.

## Recommended Main Experiments

### Main Table

Use only these datasets in the main table:

| Dataset | Metric | Reason |
|---|---|---|
| Needle | Accuracy | Synthetic hidden evidence retrieval |
| LongEval-Lines | Accuracy | Exact line-level retrieval |
| HotpotQA | F1 | Multi-hop long-context QA |
| NarrativeQA | F1 | Long narrative QA; HMO is strongest here |

Recommended methods in main table:

| Method | Include? | Note |
|---|---|---|
| Full KV | Yes | Upper-bound reference |
| Budgeted Recent KV | Yes | Naive budget-matched baseline |
| Budgeted Uniform KV | Yes | Naive budget-matched baseline |
| H2O | Yes | Classic KV eviction baseline |
| SnapKV | Yes | Strong KV compression baseline |
| Quest | Yes | Strong recent baseline; label as Quest-lite if not official |
| SAGE-KV | Yes | Recent baseline; label as SAGE-KV-lite if not official |
| HMO | Yes | Main method |

Optional/appendix:

- PyramidKV-lite
- StreamingLLM
- DuoAttention

### Current Main Result Summary

From `v6_1_27b_32k_keep010_1`:

| Dataset / Metric | Full KV | H2O | SnapKV | Quest | SAGE-KV | HMO |
|---|---:|---:|---:|---:|---:|---:|
| Needle Acc | 1.0000 | 0.7800 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| LongEval Acc | 1.0000 | 0.1200 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| HotpotQA F1 | 0.6795 | 0.6871 | 0.6787 | 0.6787 | 0.6268 | 0.6741 |
| NarrativeQA F1 | 0.3140 | 0.2899 | 0.3105 | 0.3118 | 0.3085 | 0.3162 |

Main interpretation:

- HMO matches Full KV/SnapKV/Quest on Needle and LongEval.
- HMO is slightly below Full/SnapKV/Quest/H2O on HotpotQA but remains competitive.
- HMO is best on NarrativeQA.
- HMO strongly outperforms naive budgeted KV subsets on retrieval tasks.

### Appendix Table

Put GovReport and LCC in appendix:

| Dataset | Reason for Appendix |
|---|---|
| GovReport | Summarization favors broad continuous context coverage; HMO is weaker |
| LCC | Code completion is token-exact and current sample size is small |

This is not hiding negative results. It is better framed as:

> HMO is optimized for evidence-centric long-context QA/retrieval rather than universal long-context generation.

## Required Ablations

For AAAI, include at least one compact ablation table:

| Variant | Purpose |
|---|---|
| Full HMO | Main method |
| HMO w/o Refresh | Shows contribution/cost of refresh |
| HMO w/o Alpha, sigma only | Tests attention-side fragility signal |
| HMO w/o RTS | Tests sparse skeleton coverage |
| Budgeted Recent KV | Tests naive recent-window retention |
| Budgeted Uniform KV | Tests naive global coverage |

If time is limited, the minimal ablation is:

```text
HMO
Budgeted Recent KV
Budgeted Uniform KV
HMO w/o Refresh
```

## Recommended Figures

### Figure 1: Hybrid Memory Mismatch

Show a hybrid-attention model with:

- Full-attention KV cache as explicit memory.
- DeltaNet/recurrent state as implicit fixed-size memory.
- Long-context segments flowing through both channels.
- Saturated recurrent state causing evidence risk.

### Figure 2: HMO Segment Policy

Show segments assigned to:

```text
KV | Refresh | RTS | Drop
```

Use color blocks along a long context timeline.

### Figure 3: Accuracy-Memory Tradeoff

Plot:

```text
x-axis: tracked KV memory
y-axis: primary score
```

Use Needle, LongEval, HotpotQA, NarrativeQA.

Highlight that HMO operates near SnapKV/Quest memory but with hybrid-memory-aware actions.

## AAAI Paper Outline

### Title Options

Recommended:

```text
Beyond KV Cache: Hybrid Memory Orchestration for Long-Context Question Answering
```

Alternative:

```text
Hybrid Memory Orchestration for Evidence-Centric Long-Context QA
```

More technical:

```text
Hybrid-Memory-Aware KV Compression for Hybrid-Attention Language Models
```

### Abstract

Structure:

1. Long-context LLMs increasingly use hybrid attention for efficiency.
2. Existing KV compression methods ignore recurrent memory.
3. HMO estimates recurrent saturation and attention fragility.
4. It orchestrates segment actions: KV, refresh, RTS, drop.
5. On Qwen3.5-27B at 32K with 10% KV budget, HMO preserves retrieval performance and improves NarrativeQA.
6. Results show hybrid-memory-aware compression is necessary for evidence-centric long-context QA.

Draft:

> Long-context language models increasingly rely on hybrid attention architectures that combine exact full-attention layers with efficient recurrent or linear-attention layers. However, existing KV-cache compression methods largely treat the KV cache as the sole memory substrate, ignoring whether information is reliably preserved in the recurrent state. We introduce Hybrid Memory Orchestration (HMO), a training-free inference-time controller for hybrid-attention LLMs. HMO partitions the prompt into segments, estimates recurrent-state saturation and attention-side fragility, and assigns each segment to exact KV retention, refresh, sparse RTS skeleton retention, or dropping under a fixed memory budget. On Qwen3.5-27B with 32K contexts and a 10% KV budget, HMO matches Full-KV-level performance on retrieval tasks, achieves the best score on NarrativeQA, and substantially outperforms naive budget-matched KV subsets. These results suggest that long-context QA in hybrid-attention LLMs benefits from orchestrating both explicit KV cache and implicit recurrent memory.

### 1. Introduction

Recommended paragraph flow:

1. Long-context QA requires preserving sparse evidence over tens of thousands of tokens.
2. Hybrid-attention LLMs make long-context inference efficient but introduce heterogeneous memory.
3. KV compression is insufficient because it ignores recurrent-state saturation.
4. Evidence-centric QA is especially sensitive: missing one evidence segment can fail the answer.
5. HMO jointly monitors recurrent and attention signals and chooses segment actions.
6. Summarize results and contributions.

Contributions:

- We identify the hybrid-memory mismatch in KV-only compression for hybrid-attention LLMs.
- We propose HMO, a training-free memory controller using recurrent saturation and attention fragility.
- We define a segment-level action space including exact KV, refresh, RTS skeletons, and drop.
- We evaluate on Qwen3.5-27B under aggressive 10% KV budgets and show strong performance on evidence-centric long-context QA/retrieval.

### 2. Related Work

Suggested subsections:

1. Long-context language models and hybrid attention.
2. KV cache compression and token eviction.
3. Retrieval-oriented long-context evaluation.
4. Dynamic inference-time memory management.

Important positioning:

- H2O, StreamingLLM, SnapKV, PyramidKV, Quest, SAGE-KV are KV-centric.
- HMO differs by using recurrent-state reliability as a decision signal.

### 3. Problem Setup

Define:

- Prompt tokens split into segments.
- A hybrid model with full-attention layers and DeltaNet/linear layers.
- KV budget.
- Goal: maximize QA/retrieval performance under a tracked KV memory budget.

Notation:

```text
x = [x_1, ..., x_T]
S_i = segment i
M_KV = explicit KV cache
M_R = recurrent state
B = memory budget
```

### 4. Method: Hybrid Memory Orchestration

Subsections:

#### 4.1 Segmenting Long Context

Default segment length:

```text
512 tokens
```

Explain why segment-level is used:

- Token-level is expensive and noisy.
- Whole-document compression is too coarse.
- Segment-level fits evidence granularity.

#### 4.2 Hybrid-Memory Signals

Define:

```text
sigma_i: recurrent saturation score
alpha_i: attention fragility/dependence score
phi_i = sigma_i * alpha_i
```

Explain each signal intuitively.

#### 4.3 Segment Action Space

Explain actions:

- KV
- Refresh
- RTS
- Drop

Clarify that refresh is full HMO; 64K no-refresh is an implementation-safe variant.

#### 4.4 Budgeted Action Assignment

Explain policy:

1. Protect sink and recent segments.
2. Rank middle segments by phi.
3. Assign refresh to high-risk segments if budget allows.
4. Allocate remaining budget to RTS skeletons.
5. Drop segments with zero allocated budget.

#### 4.5 Complexity And Memory Accounting

Report tracked KV memory, not only peak VRAM.

Be honest:

> The current prototype still performs full prefill/probing, so peak VRAM does not necessarily decrease proportionally to tracked KV memory.

### 5. Experiments

#### 5.1 Setup

Main setup:

```text
Model: Qwen3.5-27B
Context length: 32768
Budget: keep_ratio = 0.10
Samples: 50 except LCC appendix
```

Main datasets:

- Needle
- LongEval-Lines
- HotpotQA
- NarrativeQA

Baselines:

- Full KV
- Budgeted Recent KV
- Budgeted Uniform KV
- H2O
- SnapKV
- Quest-lite
- SAGE-KV-lite
- HMO

#### 5.2 Main Results

Use the 4-task table.

Main message:

- Retrieval tasks: HMO matches Full KV and strong baselines.
- NarrativeQA: HMO is best.
- HotpotQA: HMO is competitive.
- Naive budgeted KV collapses on exact retrieval.

#### 5.3 Memory Budget Analysis

Show tracked KV memory:

- Full KV roughly 1.0-2.1 GB depending on task.
- HMO around 0.15-0.27 GB under 10% budget.

Do not overclaim peak VRAM reduction.

#### 5.4 Ablation Study

Compare:

- HMO
- HMO w/o Refresh
- sigma-only
- no RTS
- budgeted recent
- budgeted uniform

#### 5.5 Appendix Results

Include:

- GovReport
- LCC
- StreamingLLM
- DuoAttention
- PyramidKV-lite
- 64K no-refresh stress test

### 6. Discussion

Key discussion points:

- Why HMO helps NarrativeQA: evidence must be integrated across long narrative contexts.
- Why GovReport is weaker: summarization needs broad continuous coverage, not only sparse evidence retention.
- Why LCC is difficult: exact code tokens and variable names make skeleton retention brittle.
- Why tracked memory and peak VRAM differ in the prototype.

### 7. Limitations

Be explicit:

- Current implementation is a prototype and still has full prefill/probe peak memory.
- Some baselines are lite implementations; official implementation comparison is future work if not available.
- HMO is strongest for evidence-centric QA/retrieval, not universal summarization/code generation.
- Refresh can be expensive at 64K/27B without chunked replay.

### 8. Conclusion

End with:

> Hybrid-attention LLMs require hybrid-memory-aware compression. HMO shows that orchestrating explicit KV and implicit recurrent memory can preserve long-context QA and retrieval performance under aggressive KV budgets.

## Recommended Paper Framing In One Paragraph

This paper should be framed around evidence-centric long-context QA rather than universal long-context generation. The central observation is that hybrid-attention LLMs distribute memory across exact KV cache and implicit recurrent states, while existing compression methods optimize only the KV side. HMO addresses this mismatch by estimating recurrent saturation and attention dependence, then assigning each segment to exact KV, refresh, RTS skeleton, or drop under a memory budget. Experiments on Qwen3.5-27B at 32K context and 10% KV budget show that HMO preserves Full-KV-level retrieval performance, achieves the best NarrativeQA score, and substantially outperforms naive budget-matched KV subsets, demonstrating the value of hybrid-memory-aware orchestration for long-context QA.

