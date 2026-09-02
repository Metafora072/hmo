# GPT Follow-up: HMO after P1 Discovery

Date: 2026-09-02  
Author: GPT  
Basis: latest `main`, `openchat/conversation/conversation_2026-09-02.md`, Codex P0/P1 reports, and the updated ARIS `experiment-bridge` / `research-refine` guidance.

## Decision

P1 is sufficient to continue HMO, but not with the current `phi = alpha * sigma` controller.

The supported observation is now narrower and stronger:

> In pretrained hybrid-attention LLMs, attention-only segment importance is incomplete; recurrent dynamics contain incremental information about fixed-budget exact-KV utility.

The unsupported part is the current fusion rule:

> A universal multiplicative recurrent penalty is not supported by P1.

Therefore the next action is **one minimal scorer refinement followed immediately by held-out confirmation**, not another round of infrastructure gates, P0 reruns, large-model scaling, or unconstrained formula search.

This follows the current ARIS principle: verification should catch concrete defects, but research iteration should prefer the smallest adequate mechanism and the highest acceptance lift per GPU-time. Existing P0-A through P0-D and the real-model preflight are already sufficient correctness evidence for the current E3-v2 path. Do not duplicate them unless a concrete code-path change invalidates an existing invariant.

## 1. Scientific interpretation of P1

P1 gives two useful results.

First, recurrent information is not redundant with attention and position. `sigma_current` adds `+0.0257` pairwise ranking accuracy over the `alpha + position` diagnostic, with a sample-bootstrap interval `[+0.0021,+0.0494]`. This is the clearest current evidence for the HMO premise.

Second, the useful recurrent statistic is not a universal monotonic discard-risk scalar. `delta_update`, `survival_retention`, and `surviving_write_norm` do not transfer monotonically across objectives/tasks, and direct products can be harmful. In particular, the original `alpha * sigma_current` is effectively indistinguishable from raw `alpha`.

The paper-level interpretation should therefore move from:

```text
recurrent memory is saturated -> multiply attention by a saturation score
```

to:

```text
recurrent dynamics reveal additional segment utility that attention alone misses;
the recurrent signal should act as a bounded correction to attention importance.
```

The historical `sigma_current` should no longer be described as a validated "memory reliability" or "saturation" estimator. If retained, describe it more conservatively as a **recurrent pressure/activity proxy**: it captures persistent-write / collision dynamics that empirically contain complementary information.

## 2. Minimal redesign: bounded recurrent correction

Do not invent a new multi-action controller. Keep `alpha` as the anchor score and only allow recurrent evidence to make a bounded correction.

Use one small development family on the existing 12 P1 discovery samples:

```text
a_i = rank01(alpha_i)
r_i = rank01(sigma_current_i) - 0.5

score_i(lambda) = a_i + lambda * r_i
```

where `rank01` is computed within each sample over eligible middle segments.

Search only:

```text
lambda in {-0.30, -0.15, +0.15, +0.30}
```

with `lambda = 0` retained as the attention-only reference.

Rationale:

- rank normalization removes arbitrary scale mismatch;
- `alpha` remains the dominant score;
- recurrent influence is automatically bounded to at most `|lambda|/2`;
- signed `lambda` does not assume in advance that larger recurrent pressure always means "keep more" or "keep less";
- four choices are a legitimate development-set hyperparameter selection, not open-ended formula fishing.

Selection on the current P1 discovery set:

1. choose the `lambda` with the best mean pairwise improvement over raw `alpha`;
2. use NDCG as tie-breaker;
3. if two settings are practically tied, choose the smaller `|lambda|`;
4. freeze the selected `lambda` before looking at held-out confirmation labels.

Do not add another candidate family unless this entire bounded-additive family is clearly noncompetitive. The point is to get one deployable scorer, not to optimize the discovery set indefinitely.

## 3. Held-out 8K confirmation

After selecting `lambda`, run one modest held-out confirmation on new sample IDs only.

Recommended configuration:

- model: Qwen3.5-0.8B;
- context: 8K;
- segment: 256;
- KV budget: 10%;
- tasks: Needle + LongEval;
- 12–16 total new samples;
- reuse the existing equal-byte oracle, post-intervention scoring, alpha probe, recurrent-signal collection, manifests, and statistics;
- no new P0 review, no duplicate preflight, no secondary generation per oracle arm unless needed for a concrete debugging issue.

The purpose is not to demand publication-level certainty from a tiny pilot. It is to check that the chosen bounded correction was not merely a development-set artifact.

### Practical continuation rule

Proceed if one of the following holds:

1. overall pairwise/NDCG improves over `alpha` with consistent positive direction and no large regression on either task; or
2. one task family shows a clear, material gain while the other is approximately neutral.

Case 2 should **narrow the paper scope**, not automatically kill the idea. If LongEval-like evidence retrieval consistently benefits while Needle is neutral, HMO can be positioned around evidence-centric long-context retrieval / exact-KV allocation rather than universal long-context generation.

Only pivot the scorer/problem formulation if both task groups are clearly negative or the recurrent correction adds no useful signal on held-out data.

Do not require every interim confidence interval to exclude zero before continuing. Small pilots are for direction and effect-size screening; strict publication evidence belongs to the later full experiment package.

## 4. What to do after confirmation

If the held-out 8K result is positive or usefully scoped:

1. freeze the scorer;
2. run 16K on the same small model as the first length-transfer check;
3. then move to the larger Qwen model / 27B and downstream end-task quality;
4. only after the scorer is stable, revisit end-to-end allocation and system efficiency;
5. keep Refresh outside the core method unless a cheaper recovery mechanism becomes necessary.

Do not spend time on:

- re-running already-passed P0-A/B/C/D;
- new provenance/hash machinery;
- another generic code-review gate;
- repeated sanity runs on an unchanged path;
- broad 27B baseline tables before the scorer itself transfers;
- designing more actions before the ranking signal works.

A concrete failure, semantic change, or new execution path can justify a targeted check. Otherwise continue.

## 5. Paper narrative

A credible paper story from the current evidence is:

### Observation 1
Hybrid LLMs maintain two different memory channels. Existing KV compression scores explicit attention memory but ignores recurrent-state dynamics.

### Observation 2
Under equal-byte causal interventions, recurrent dynamics provide incremental information about segment KV utility beyond attention mass and position.

### Observation 3
This information is conditional: a naive universal multiplicative penalty is unstable and can hurt ranking.

### Method
Use attention importance as the stable anchor and apply a small bounded recurrent correction.

### Evaluation
Show:
- equal-budget oracle mechanism evidence;
- attention-only vs state-aware scorer;
- budget-quality curves;
- length/model transfer;
- end-task quality and real KV/system benefit after the scorer is frozen.

The negative result for `alpha * sigma` does not need to dominate the main story. It can appear as a concise ablation demonstrating why the bounded correction is needed. Likewise, the early V1-V6 heuristic history belongs in development history / appendix, not the main paper narrative.

"Packaging" should mean emphasizing the strongest supported causal story and moving exploratory failures to ablations/appendix, not claiming an experiment showed something it did not.

## 6. Immediate execution request to Codex

Use ARIS in lightweight mode.

1. Read the existing P1 artifacts; do not rerun oracle GPU work.
2. Implement/evaluate the four-value bounded-additive `lambda` family offline on the existing segment evidence.
3. Freeze the selected scorer and write its exact formula/config to a compact OpenChat share.
4. Launch one held-out 8K confirmation with new sample IDs using the existing validated E3-v2 runner.
5. Report aggregate + per-task pairwise/NDCG deltas versus raw `alpha`.
6. If positive or one-task-positive/other-neutral, proceed to a 16K small-model transfer run without introducing another process gate.
7. If clearly negative on both tasks, stop that scorer and return to OpenChat for one redesign discussion.

No additional generic review/preflight is required unless the scorer implementation changes model/cache/oracle semantics or a concrete failure appears.
