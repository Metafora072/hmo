# Package A: Method, Novelty, Figure, And Metric Report

## Verdict

The authorized zero-GPU package is complete. HMO should continue, but its paper
identity must be more specific than locality-preserving KV compression.
ChunkKV already states the discrete-token fragmentation problem and selects
attention-scored contiguous chunks; SentenceKV, ProtoKV, and Kara further cover
sentence, semantic-cluster, and flexible-chunk retention.

The defensible and visually coherent HMO identity is:

> Hybrid residual-memory organization through a stratified, query-guided local
> KV overlay on an unchanged recurrent global state.

This retains the original HMO target while replacing V6.1's unvalidated
recurrent reliability formula with a structural allocation contract.

## Closest-Work Finding

The strongest novelty risk is ChunkKV. HMO should not claim the first locality,
chunk, semantic-integrity observation, or query-guided structured retention.
Its distinguishing combination is:

1. Hybrid-specific role assignment between recurrent state and residual
   Full-Attention KV.
2. Coverage-first allocation across macro-segments.
3. A free-start max-attention window inside every covered segment, rather than
   globally ranked fixed-boundary chunks.
4. Optional Exact fidelity only after local coverage.
5. Real cache intervention with measured, paired equal-byte controls.

Current results directly establish only item 5 against the scattered version of
the same allocator. They do not yet establish superiority over ChunkKV-like
structured retention.

## Frozen Method Clarifications

- Mandatory core means every coverage action is locality preserving. It does
  not mean every segment must be covered at every budget.
- With base width 16 and segment length 256, the all-segment coverage floor is
  about 6.25%; a 5% middle cap necessarily prioritizes a subset.
- Width 16 is a base width. The current allocator spends per-token slack by
  extending Sparse windows, which commonly makes the realized width 17 or 18.
- Exact is optional and `m=0` is valid.
- `use_accessibility=false` remains frozen; recurrent accessibility and
  saturation are not main-method signals.
- The general retention expression is
  `N_keep = c*w + m*(L-w) + s`, where `c` is covered segments and `s` is
  slack extension. The simpler `w/L` term requires all-segment coverage.

The dossier includes paper-ready pseudocode and exact theorem assumptions.

## Format-Robust Secondary Analysis

The frozen primary metric and raw JSONL files remain unchanged. The deterministic
post-hoc rule adds one Needle-only clock alias and leaves LongEval identifiers
untouched.

| Model | Primary Contiguous / Scattered | Secondary Contiguous / Scattered | Secondary W/L |
|---|---:|---:|---:|
| Qwen3.5-0.8B | 34/48 vs 27/48 | 34/48 vs 28/48, +12.50 pp | 6/0 |
| Qwen3.5-9B | 23/24 vs 19/24 | 24/24 vs 20/24, +16.67 pp | 4/0 |

At 9B, Contiguous, Raw Exact+Slack, Raw Exact, and Full become 24/24 under the
secondary rule. Thus the Raw Exact apparent one-case advantage disappears,
while the cross-scale locality result remains.

## Figure 1

The frozen storyboard uses three panels:

1. Qwen3.5-style Hybrid memory anatomy: recurrent global base and growing
   residual KV.
2. Equal-byte scattered versus stratified contiguous retention geometry, with
   global fixed chunks acknowledged as related work rather than depicted as a
   defeated baseline.
3. The 0.8B/9B paired result bars plus the 13.38% mean per-case Full-KV
   footprint.

## Recommended Package B Change

Before the approved 0.8B 5/10/20% GPU curve is launched, add:

```text
Global Fixed-Chunk Top-K
  same query probe
  non-overlapping 16-token chunks
  global chunk ranking
  identical anchors/query KV
  exact target resident bytes
```

This is the smallest high-value baseline that tests whether HMO gains come from
generic chunking or from stratified coverage plus free-start placement. It does
not require a new method search. Implementing this arm and launching Package B
remain concrete follow-up actions requiring PZ confirmation.

## Artifacts

- `docs/design/HMO_METHOD_AND_NOVELTY_DOSSIER_ZH.md`
- `docs/paper/HMO_FIGURE1_STORYBOARD_ZH.md`
- `experiments/results/FORMAT_ROBUST_SECONDARY_20260904.md`
- `experiments/results/format_robust_secondary_20260904.json`
- `experiments/phase2/e3_v2/format_robust.py`

The relative paths above are from the repository root; OpenChat readers can use
`codex/share/2026-09-04/package_a_method_novelty_report.md` as the stable
entry point.
