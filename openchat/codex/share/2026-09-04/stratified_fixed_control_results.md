# Stratified Fixed-Chunk Control Results

Date: 2026-09-04

## Executive Result

The approved P5 mechanism control is complete on GPU1. At the positive
Qwen3.5-0.8B 16K/10% slice:

| System | All | Needle | LongEval-Lines |
|---|---:|---:|---:|
| Global Fixed-Chunk | 16/24 | 10/12 | 6/12 |
| Stratified Fixed-Chunk | 17/24 | 10/12 | 7/12 |
| HMO stratified free-start | **18/24** | 10/12 | **8/12** |

HMO versus Stratified Fixed is `+4.17 pp`, with 2 wins, 21 ties, and 1 loss.
All three disagreements occur on LongEval. All 24 comparisons use exactly equal
measured resident KV bytes.

## Why This Is A Clean Control

Stratified Fixed reuses HMO's protected anchors, macro-segment allocation,
Exact upgrades, Sparse retained-token counts, 17/18-token slack extensions,
query probe, decoding settings, and byte target. It only restricts a Sparse
window's start to a segment-local 16-token boundary.

Before each new generation, the runner recomputed the parent HMO allocation and
retained positions and required an exact match with the SHA-pinned Package B
row. The new geometry is materially different: mean position Jaccard is 0.646,
and an average 54.54 of 59.5 Sparse segments change placement per sample.

## Interpretation

The result provides directional support that free-start placement contributes
part of HMO's 16K benefit. It is especially useful that the effect is isolated
to structured LongEval while Needle remains unchanged.

The `16 -> 17 -> 18` totals are not a nested additive sequence. Stratified
Fixed alone recovers `longeval_0008`; HMO instead recovers `longeval_0003` and
`longeval_0004`. Thus macro organization and free-start show complementary
sample-level benefits. The paper should present a two-level memory organization
story, not claim a strict additive decomposition or universal free-start win.

Independent result-to-claim review returns:

- `claim_supported`: `partial`
- `route`: `supplement`
- `confidence`: `medium`
- supported wording: under fixed stratified allocation, free-start gives a
  small directional 16K/10% gain concentrated on structured retrieval and
  explains part of the earlier HMO-versus-Global-Fixed gap.

## Artifacts

- Full report: `experiments/results/STRATIFIED_FIXED_CONTROL_20260904.md`
- Protocol: `refine-logs/stratified_fixed_chunk_control_protocol.json`
- Formal run:
  `/mnt/nvme0/hmo/runs/stratified_fixed_control_formal_20260904_1633/`
- Formal JSONL SHA256:
  `3a399d1012c4d4b82f3a2286ba4a8354562753257e923a0b9b09690faa64b058`
- Formal summary SHA256:
  `b831cd2be9dc30e4214fab6647d0fbfadc29a5b883a30c50f2cd49fc9eaeec24`

The formal run took 183.15 seconds and peaked at 4.35 GiB allocated / 5.18 GiB
reserved. GPU1 returned to 15 MiB.

## Recommended Route

The synthetic mechanism chain is now sufficiently developed for the current
paper story. The next high-value step is the already planned 32K HotpotQA
Full-KV solvability smoke, followed by a matched realistic-task comparison if
the backbone is capable. A fourth `global allocation + free-start` arm could
complete a 2x2 decomposition, but needs a principled non-overlap rule and is
lower priority than external validity. No next GPU job has been started.
