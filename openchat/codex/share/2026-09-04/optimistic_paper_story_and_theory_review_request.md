# Optimistic HMO Story And Theory Review Request

## Executive Direction

PZ explicitly asks the project to stop treating conservative statistical gates
as idea-death decisions. The operating objective is now to build the strongest
coherent and truthful paper from the observed mechanism, a defensible formal
model, and the available empirical evidence. Statistical uncertainty controls
wording and scope; it does not automatically terminate a promising direction.

The remaining hard constraints are result integrity, reproducible byte
accounting, no fabricated evidence, and no theorem whose assumptions are hidden.

## Proposed Paper Story

### Background

Hybrid LLMs replace many attention layers with constant-state recurrent layers,
but their remaining Full-Attention layers still retain a KV cache growing
linearly with context length. This residual cache becomes the memory bottleneck
in long-context inference.

### Observation

Classic importance-based KV compression treats token utility as modular: score
tokens independently and keep the largest scores. Long-context evidence is
often non-modular and span structured. A collection of individually important
but scattered tokens can preserve attention mass while destroying the complete
local relation needed to answer a query.

### Method

HMO uses a coverage-fidelity hierarchy for the Full-Attention cache while
leaving DeltaNet recurrent state active:

1. protect fixed prefix and suffix anchors;
2. establish broad middle-context coverage using one query-selected contiguous
   window per segment;
3. use remaining budget for Exact upgrades of high-demand segments;
4. execute all actions under measured resident-KV accounting.

The strongest current empirical component is locality-preserving coverage. The
Exact fidelity upgrade remains a reasonable hierarchical action but is not yet
independently established by the fresh result.

## Formal Core

### Proposition 1: Contiguous Retention Maximizes Complete Local-Span Coverage

Consider a segment with `n` ordered token positions. A policy retains a set
`R` of exactly `k` positions. Task evidence is an unknown contiguous interval
of length `ell`, where `2 <= ell <= k`.

Let `N_ell(R)` be the number of length-`ell` intervals fully contained in `R`.
If the retained positions form runs of consecutive lengths
`r_1, ..., r_m`, then

```text
N_ell(R) = sum_j max(r_j - ell + 1, 0) <= k - ell + 1.
```

The upper bound is attained by a single contiguous run of length `k`. For
`ell > 1`, splitting retained positions across multiple nontrivial runs weakly
reduces complete-span coverage and generally loses `(ell - 1)` opportunities
per additional run.

Under a uniform prior over the evidence-span start, the probability of fully
retaining the evidence is `N_ell(R) / (n - ell + 1)`. Therefore a contiguous
window maximizes complete local-evidence survival among all equal-cardinality
retention sets.

Proof sketch: each consecutive run of length `r_j` contains exactly
`max(r_j-ell+1,0)` complete length-`ell` subintervals. Merging two runs cannot
decrease this count. Repeated merging yields one run of total length `k`, whose
count is `k-ell+1`.

### Proposition 2: Query-Guided Placement Is Optimal Within The Locality Class

Let `a_t >= 0` be query-attention mass at token `t`. Among all contiguous
windows of width `k`, HMO selects

```text
W* = argmax_W sum_{t in W} a_t.
```

Thus HMO simultaneously attains the complete-span advantage of contiguous
retention and the maximum retained query-attention mass available within the
fixed-width locality class. This does not claim globally maximal singleton
mass; scattered Top-token selection has that property but lacks complete-span
coverage.

### Proposition 3: Attention-KV Retention Ratio

For middle-context length `T`, segment length `L`, Sparse width `w`, and `m`
Exact upgrades, the retained middle tokens are bounded by

```text
ceil(T / L) * w + m * (L - w).
```

The retained Full-Attention KV storage is therefore bounded by

```text
O(T * w / L + m * L),
```

plus fixed protected anchors and the unchanged recurrent state. This remains
`O(T)` when `L`, `w`, and the Exact-upgrade rate are fixed; the contribution is
a smaller retention coefficient rather than a new asymptotic complexity class.
Ignoring boundary rounding, the middle-context retention ratio is

```text
rho_middle <= w / L + m * (L - w) / T.
```

With `w=16` and `L=256`, the coverage floor retains 6.25% of each full middle
segment before Exact upgrades.

These propositions explain the inductive bias and memory behavior. They are not
presented as a theorem guaranteeing downstream generation accuracy.

## Empirical Alignment

The frozen fresh confirmation used 48 unfiltered Qwen3.5-0.8B samples across
8K/16K Needle and LongEval-Lines:

| system | containment | mean resident KV |
|---|---:|---:|
| contiguous CF | 34/48 (70.83%) | 19,406,592 bytes |
| scattered CF | 27/48 (56.25%) | 19,406,592 bytes |
| raw Exact Top-K | 32/48 (66.67%) | 19,173,120 bytes |
| contiguous Sparse-only | 32/48 (66.67%) | 19,406,592 bytes |
| Full KV | 35/48 (72.92%) | 148,934,400 bytes |

The exact equal-byte locality comparison is +14.58 pp, with 7 wins, 41 ties,
0 losses, a sample bootstrap interval of [+6.25,+25.00] pp, and exact sign-test
`p=0.0156`. The improvement is positive at both 8K and 16K.

At a mean per-sample Full-KV fraction of 13.38%, contiguous CF reaches 70.83%
versus Full KV at 72.92%. The ratio of mean resident bytes is 13.03%; these two
aggregation conventions must not be mixed. Against raw Exact it has a
directional +4.17 pp result, but this comparison is tie heavy and raw uses 1.22%
fewer bytes because of segment rounding slack. Exact upgrades over contiguous
Sparse-only add +4.17 pp but are not independently significant.

## Claim Ladder For A Non-Gate-Driven Paper

### Main claim

Locality-preserving contiguous KV retention protects complete relational
evidence that scattered token-importance compression can destroy, producing a
substantial equal-byte quality improvement on the tested Hybrid LLM.

### System claim

HMO retains roughly 13% of Full-Attention KV while approaching Full-KV quality
on the tested long-context suite.

### Directional claim

The complete coverage-fidelity policy shows a small positive point improvement
over raw Exact Top-K at measured bytes, especially at 16K.

### Claims not needed for the paper

- recurrent accessibility is a validated allocation signal;
- every component is separately statistically significant;
- universal gains across models and tasks;
- theoretical guarantees of end-task correctness.

## Revised Execution Attitude

- Do not use statistical significance as a binary continue/kill gate.
- Do not request generic external review after every small action.
- Preserve all real numbers, but organize the paper around the strongest true
  mechanism and use limitations to scope rather than erase it.
- Allow a component to remain as motivated system design when the complete
  method and central causal ablation form a coherent story.
- Prefer a small number of high-yield additions: a quality-memory Pareto view,
  one independent fresh package, and one more task or model if feasible.
- Stop only for invalid measurement, implementation error, hidden unfairness,
  or clearly wasteful computation.

## Questions For GPT Review

1. Is the background-observation-method-theory-evidence chain coherent enough
   for a paper centered on locality-preserving KV compression in Hybrid LLMs?
2. Is Proposition 1 a useful and defensible theoretical explanation, and which
   assumptions or weighted extension should be added without making it ornate?
3. Should Exact fidelity upgrades remain in the named method, or should the
   paper foreground contiguous coverage and present Exact upgrades as an
   optional system layer?
4. How should the +4.17 pp versus raw and near-Full result be framed positively
   without claiming statistical certainty that is not present?
5. Given limited time, which one or two additions most increase paper
   persuasiveness: exact-byte/Pareto comparison, a second seed package, a
   realistic task, latency measurement, or a larger model?
6. Propose a concise method name, paper title, and three-contribution list that
   match the evidence rather than the rejected recurrent-aware story.

Detailed fresh results are in
`codex/share/2026-09-04/contiguous_cf_fresh_confirmation_report.md`.
