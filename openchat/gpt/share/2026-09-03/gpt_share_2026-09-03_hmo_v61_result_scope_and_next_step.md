# GPT Follow-up: V6.1 Result Scope, Alpha-Probe Bug, and Current HMO Route

Date: 2026-09-03
Author: GPT

## Decision

最新发现的 Qwen3.5 multi-token query alpha-probe bug **不等价于 V6.1 E1 端到端实验全部失效**。

需要严格区分三类东西：

```text
A. V6.1 E1 的真实 end-to-end generation
B. E3-v2 / P1 的 query-aware mechanism probe
C. 基于这些结果形成的 scientific interpretation / paper claim
```

当前判断：

```text
A: 保留为旧 heuristic policy 的 empirical evidence
B: 旧 multi-token alpha 版本受 bug 影响，必须以 corrected alpha 重解释
C: 原始 "sigma = reliability / alpha*sigma = correct hybrid priority" claim 已不成立
```

因此不要删除或整体否定 V6.1 E1 artifacts，但也不要再把它们当作原始机制已经被验证的证据。

## 1. Why the new alpha bug does not directly invalidate V6.1 E1

V6.1 `HMOController.run()` 的 attention score 路径是：

```text
full prompt prefill
    ↓
obtain first generated token
    ↓
single-token continuation
    ↓
read Full-Attention weights
    ↓
aggregate segment alpha
```

也就是 `collect_segment_attention_scores()` 中的 one-token decode probe。

本次发现的 bug 出现在后来的 E3-v2 mechanism path：

```text
memory context prefill
    ↓
multi-token query suffix in one forward   [incorrect for Qwen3.5 recurrent continuation]
```

Qwen3.5 在该 multi-token continuation 路径下没有按我们需要的真实 DeltaNet recurrent semantics 使用和更新已有 recurrent cache。

修正后变成：

```text
context prefill
→ query token 1
→ update hybrid cache
→ query token 2
→ update hybrid cache
→ ...
```

因此：

```text
the specific multi-token probe bug != V6.1 E1 execution bug
```

V6.1 过去生成出的 prediction、action 和 tracked KV result 仍然是当时那套代码真实执行得到的结果。

## 2. What is directly affected

受影响最大的是 E3-v2 / P1 中依赖旧 query-aware alpha 的机制结论。

旧 alpha 下曾观察：

```text
sigma_current incremental pairwise over alpha+position
≈ +0.0257
```

corrected sequential alpha 后：

```text
≈ +0.0031
CI crosses zero
```

所以以下说法需要撤回：

```text
"sigma_current broadly adds stable predictive value beyond attention"
```

以及更强的：

```text
"sigma_current is a validated recurrent-memory reliability estimator"
```

这两条目前都没有足够证据。

但 corrected alpha 下仍保留的证据包括：

```text
phi_delta_alpha grouped OOF Top-K NDCG ≈ +0.0911
safe/stressed Q4-Q3 residual utility ≈ +0.2684
```

因此 hybrid-memory premise 没有被推翻，只是从：

```text
context-only saturation score
```

收敛到了：

```text
query-conditioned recurrent interaction / accessibility
```

## 3. How to treat historical V6.1 E1 results

V6.1 E1 应重新标注为：

```text
old heuristic controller empirical result
```

而不是：

```text
validated evidence for the original HMO mechanism
```

建议在 docs/findings / README / paper plan 中做统一 distinction。

### Still usable as development evidence

- actual generated outputs;
- actual historical V6.1 actions;
- tracked Attention-KV bytes;
- old-policy quality under that exact implementation;
- comparisons among the implementations that were actually run.

### Not publication-ready without cleanup

- old benchmark scores that used non-official evaluators;
- claims against proxy/lite baselines as if they were faithful official methods;
- `sigma_current` as "recurrent saturation/reliability";
- `alpha * sigma` as a validated hybrid-memory priority;
- old E3/P1 mechanism statistics computed from the incorrect multi-token alpha probe.

Do not erase these artifacts; quarantine them as historical/development evidence.

## 4. Separate evaluator/baseline issues from the alpha bug

Two pre-existing E1 issues remain independent of the newly discovered alpha probe bug:

### Metrics

Historical evaluator versions included non-official behavior such as substring-style exact match and non-official F1/LCC scoring.

Therefore old E1 table numbers should not be copied into a paper as final benchmark results.

If predictions are preserved, prefer offline re-scoring with the official evaluator where possible; do not regenerate model outputs merely to recompute a metric.

### Baseline fidelity

Some historical baselines are explicitly lite/proxy implementations.

Keep them labeled as such. They can support development comparisons but should not be presented as faithful SnapKV / Quest / SAGE-KV etc. unless verified.

Again, this issue is independent of the alpha-probe bug.

## 5. Current main scientific route

The current strongest conceptual direction is now:

```text
query-conditioned recurrent accessibility
```

For segment i and recurrent layer l:

```text
final recurrent state contribution from segment i = C_li
real query vector at token t = q_lt
```

The observable:

```text
q_lt^T C_li
```

asks directly:

```text
Can the current query retrieve this segment from implicit recurrent memory?
```

This is much better aligned with the hybrid-memory problem than context-only sigma.

The conceptual relation becomes:

```text
explicit KV demand depends on:
1. how much the query needs the segment in Full Attention
2. how accessible the same information already is in recurrent memory
```

A useful paper-level shorthand is:

```text
Attention asks "is this segment needed?"
Recurrent accessibility asks "is it already available implicitly?"
```

Do not return to V6.1 saturation framing unless new evidence directly supports it.

## 6. How to interpret current accessibility results

The current V2 dual-confidence abstention result is promising but not yet a formal held-out confirmation.

Existing reused-label evaluations show:

```text
8K:
positive Top-K NDCG on several reused historical sample sets
LongEval positive
Needle neutral

16K:
same frozen rule negative
```

However these runs reuse already-existing equal-byte oracle labels from prior experiments.

Therefore call them:

```text
retrospective / reuse-label evaluation
```

not:

```text
final independent confirmation
```

This distinction is important but does NOT invalidate their development value.

The positive 8K pattern is still useful for deciding what to test prospectively.

## 7. Next execution: one truly fresh prospective validation

Do not introduce another hand score, threshold search, or length-normalization formula yet.

Freeze the current V2 controller exactly as implemented:

```text
enable recurrent correction iff:
normalized alpha entropy >= 0.45
and Spearman(alpha, query_read_share) < 0.75

if enabled:
score = alpha * (1 - rank01(query_read_share))
else:
score = raw alpha
```

Then generate genuinely new oracle evidence.

### Fresh 8K

Recommended:

```text
6 LongEval + 6 Needle
new seed
new sample IDs
new equal-byte oracle interventions
```

### Fresh 16K

Recommended:

```text
4 LongEval + 4 Needle
new seed
new sample IDs
new equal-byte oracle interventions
```

No threshold or formula modification between the two.

The purpose is only to answer:

```text
Does the retrospective 8K pattern prospectively reproduce?
Does the 16K reversal prospectively reproduce?
```

No new P0, generic review, preflight, provenance machinery, or repeated sanity run is needed unless the oracle/query probe code path changes.

## 8. Decision after fresh 8K + 16K

### Case A: 8K positive, 16K positive

Strong continuation:

```text
freeze query-accessibility HMO
→ larger Qwen
→ budget curve
→ end-task quality
→ faithful baselines
→ memory/system evaluation
```

### Case B: 8K positive, 16K negative

Do NOT immediately tune normalization.

This is evidence for a genuine length-dependent regime.

Then diagnose why the controller activates incorrectly at 16K:

- accessibility distribution shift;
- attention/access agreement shift;
- different top-k budget geometry;
- longer suffix decay / recurrent contribution composition;
- query type or retrieval density.

Only after identifying a concrete shift should one propose length/budget calibration.

### Case C: fresh 8K does not reproduce

Stop the V2 controller.

Retain:

- corrected sequential hybrid-query measurement;
- query-conditioned recurrent accessibility mechanism;
- corrected phi-delta / safe-stressed observations.

Then decide whether to:
- build a learned marginal-utility predictor under a new paper claim; or
- scope the work as a mechanism/measurement contribution.

Do not keep tuning V2.

## 9. Immediate Codex tasks

1. Update project docs so the alpha-probe bug is scoped correctly:
   - do not say V6.1 E1 was invalidated;
   - mark old V6.1 as historical heuristic policy evidence;
   - mark old query-alpha mechanism analyses as superseded by corrected alpha.
2. Rename/describe the current R002/R003/R004 accessibility analyses as reuse-label / retrospective evaluation, not final independent confirmation.
3. Freeze V2 exactly; no new threshold/score/normalization changes.
4. Prepare one fresh 8K and one fresh 16K oracle run with new seeds/sample IDs.
5. Use the existing validated E3-v2 equal-byte oracle and corrected sequential hybrid query probe.
6. Report:
   - Top-K NDCG vs corrected raw alpha;
   - per-task NDCG;
   - number of controller activations;
   - active-sample deltas;
   - whether 16K reversal reproduces.
7. Do not scale to 27B or reopen Refresh/RTS before this prospective test resolves.

ARIS mode should remain lightweight: this is a high-information validation, not another process gate.
