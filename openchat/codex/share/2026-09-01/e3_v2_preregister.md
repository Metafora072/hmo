# E3-v2 Preregistration: Incremental Value Of Recurrent Signals

Date: 2026-09-01
Author: Codex
Status: Protocol frozen for implementation review; no experiment has been run

## 1. Research Question

HMO 的核心问题不是某个 heuristic score 是否与最终分数偶然相关，而是：

> 在 pretrained hybrid-attention LLM 中，recurrent-state information 是否能在 attention-only information 之外，增量预测某个 context segment 获得 exact KV 后的边际价值？

Primary null hypothesis:

```text
H0: conditioning on alpha and position, recurrent features provide no
    incremental predictive value for fixed-budget exact-KV marginal utility.
```

Primary alternative:

```text
H1: at least one pre-registered recurrent feature provides stable,
    out-of-sample incremental value beyond alpha and position.
```

`phi = sigma * alpha` is a candidate fusion, not the hypothesis itself. A negative result for multiplication does not reject the broader recurrent-signal hypothesis.

## 2. Non-goals

E3-v2 does not attempt to establish:

- that the current four-action HMO policy is optimal;
- that Refresh is necessary;
- that HMO improves TTFT, throughput or peak VRAM;
- that a signal discovered on pilot data generalizes before confirmation;
- that HMO beats faithful end-to-end compression baselines.

Refresh is excluded from the E3-v2 core oracle. The code may remain as a separate prototype or ablation.

## 3. Unit Of Intervention

The intervention unit is one full, non-protected, non-partial context segment. First and last protected regions are identical in every arm.

For a middle segment `i`, exact-KV utility is evaluated through equal-byte swaps:

```text
Delta(i, j | R) = Q(R union {i}) - Q(R union {j})
```

where:

- `R` is the same background set of exact-KV middle segments in both arms;
- `i` and `j` have identical KV byte cost;
- the total exact-KV budget is identical;
- all non-exact middle segments receive the same action in both arms;
- `Q` is measured after the KV intervention.

At the primary 10% middle-KV setting, `|R union {i}|` is determined by the byte budget, not by a hard-coded segment count.

### Multiple backgrounds and donors

A segment must not be judged against only one arbitrary donor. The pilot uses multiple deterministic backgrounds and multiple equal-cost donors, balanced across position bins. Segment utility is aggregated over these pairwise interventions.

The manifest records every `(sample, R, i, j)` comparison. Donor or background changes after observing quality labels are prohibited.

## 4. Context-Query Boundary Protocol

This is a hard correctness requirement.

The current pipeline prefills the complete prompt before KV mutation, so the first answer-token logits already contain Full-KV information. E3-v2 instead separates:

```text
[memory context] [query / instruction suffix] [answer]
```

Execution for each oracle arm:

1. Tokenize context and query suffix separately while preserving their exact concatenated form.
2. Prefill only the memory context and collect recurrent signals/cache.
3. Apply the fixed-budget intervention to context attention KV.
4. Process the query suffix using the intervened cache and the unchanged recurrent state.
5. Obtain first-answer-token logits only after step 4.
6. Generate the answer or teacher-force the gold answer from those post-intervention logits.

Re-feeding the last prompt token into a cache that already contains the full prompt is invalid. Full-KV query probes used to compute `alpha` must use an isolated cache/state and must not leak into either oracle arm.

Correctness is prioritized over cache reuse in the pilot. Re-prefilling context per arm is acceptable until cache/state cloning is proven exact.

## 5. Quality Targets

### Primary continuous label

Mean gold-answer conditional log-likelihood per answer token, computed entirely from post-intervention logits:

```text
Q_logp = (1 / answer_tokens) * sum log p(y_t | context_arm, query, y_<t)
```

This target provides denser segment-level signal than generated-answer F1.

### Secondary end-task label

Generated output scored with the official dataset evaluator:

- HotpotQA and NarrativeQA: official LongBench QA F1;
- GovReport: official LongBench ROUGE-L;
- LCC: official LongBench code similarity;
- Needle and LongEval: their task-specific exact/retrieval score.

The official metric implementation and its upstream revision must be recorded in the run manifest.

## 6. Candidate Signals

Candidate definitions must be frozen in a signal manifest before any oracle quality labels are inspected.

### Controls

- `alpha`: query-aware full-attention dependence.
- absolute and normalized segment position.
- segment length and partial-segment flag.
- deterministic random score.

### Recurrent candidates

1. `sigma_current`: the existing gate/key-collision proxy, retained as a historical baseline without the old physical interpretation.
2. `delta_update`: an aggregation derived from the actual delta residual `beta * (v - state^T k)`.
3. `survival_retention`: suffix cumulative log-retention from the end of segment `i` to the end of the memory context.
4. `suffix_interference`: later write pressure aligned with or destructive to the representation written by segment `i`.

For Qwen3.5, retention must be derived from the actual multiplier `exp(g)`. Numerically stable implementations should operate in log space:

```text
log_survival_i = sum_{t > end(i)} g_t
decay_risk_i   = -log_survival_i
```

Layer/head/token aggregation, clipping and normalization choices belong in the frozen signal manifest. They may be selected on discovery data only and cannot change during confirmation.

## 7. Discovery And Confirmation Separation

### Discovery pilot

- smallest representative Qwen3.5 hybrid model;
- 8K and 16K contexts;
- near-exhaustive eligible segment interventions on a bounded sample set;
- used for implementation debugging, label-density inspection and candidate reduction.

### Confirmation

- disjoint sample IDs and frozen manifests;
- no signal formula, normalization, donor policy or threshold changes;
- first confirmation on the small model;
- 27B/32K expansion only after the small-model confirmation gate passes.

All exploratory variants remain labeled exploratory. A candidate selected on discovery data cannot be reported as confirmed on the same data.

## 8. Statistical Analysis

The main unit for uncertainty is the sample, not the individual segment pair.

Required analyses:

- within-sample pairwise ranking accuracy;
- NDCG at the budget-defined `k`;
- Spearman correlation with aggregated oracle utility;
- partial or residual association after controlling for `alpha` and position;
- within-`alpha`-bin comparisons;
- sample-grouped paired bootstrap confidence intervals;
- task-stratified results rather than only a heterogeneous macro average.

Primary comparison:

```text
attention-only predictor
vs
the same predictor plus a frozen recurrent candidate
```

Any learned diagnostic predictor is analysis-only and must use sample-grouped cross-validation. It does not turn the training-free HMO controller into a trained policy.

Multiple recurrent candidates require a declared correction or a discovery/confirmation selection rule.

## 9. Integrity Tests Before Pilot

The runner must refuse to produce scientific results unless all checks pass:

1. Equal-byte arms have exactly equal charged and decode-resident attention-KV bytes.
2. Query tokens are processed only after context KV intervention.
3. Full-KV equivalence reproduces the unmodified reference within numerical tolerance.
4. Repeated identical arms produce identical greedy logits and outputs.
5. A synthetic gate test confirms the direction of `exp(g)` retention and cumulative log-survival.
6. A controlled needle example demonstrates that context-KV intervention can change the first answer-token logits.
7. Alpha probing cannot mutate or leak cache/recurrent state into an oracle arm.
8. Segment IDs, positions, costs, donors and backgrounds are recoverable from the manifest.

Failure of any item is a BLOCK for pilot execution.

## 10. Go / No-go Gate

### PASS

On held-out confirmation data, a frozen recurrent candidate improves the primary attention-only ranking metric with:

- paired sample-bootstrap 95% confidence interval strictly above zero;
- median NDCG improvement of at least 0.02 or pairwise-ranking improvement of at least 0.03;
- the same direction across both context lengths and across at least two task groups;
- no explanation by position-only controls.

PASS authorizes 27B E3 and later allocation experiments. It does not by itself validate Refresh or system efficiency.

### HOLD

Use HOLD when the effect direction is positive but confidence or practical effect thresholds are not met, or when label density is insufficient. HOLD permits one pre-declared sample-size increase, not signal redesign on confirmation labels.

### KILL / PIVOT

Use KILL/PIVOT for the current HMO premise when recurrent candidates provide no incremental information beyond `alpha`, especially if the upper confidence bound excludes the minimum practical effect. Stop action-policy tuning and return to the signal or problem definition.

## 11. Approved Implementation Packages After Protocol Review

The protocol decomposes implementation into separately reviewable packages:

```text
P0-A  official metrics + immutable run manifest
P0-B  context/query split + post-intervention scoring
P0-C  recurrent candidate instrumentation + synthetic tests
P0-D  equal-byte multi-background oracle + statistical analysis
P1    small-model pilot, only after all integrity tests pass
```

Raw JSONL, logits and traces stay in `experiments/results/` or external storage. OpenChat receives only compact manifests, validation reports, decisions and result summaries.

## 12. Immediate Decision

No existing E3 or E1 command should be launched from this protocol alone. The next implementation target is P0-A, followed by the context/query correctness work in P0-B. GPU execution remains blocked until P0-A through P0-D pass their integrity checks.
