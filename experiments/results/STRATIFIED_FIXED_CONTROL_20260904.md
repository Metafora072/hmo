# HMO P5: Stratified Fixed-Chunk Mechanism Control

Date: 2026-09-04

## Question And Protocol

Does HMO's query-guided free-start window contribute beyond its stratified
macro-segment allocation at the positive 16K/10% slice?

The control holds fixed the Qwen3.5-0.8B model, 24 frozen 16K cases, 10%
middle-context cap, query probe, protected anchors, macro allocation, Exact
upgrades, per-segment retained-token counts, slack, generation settings, and
resident KV bytes. It changes only Sparse placement: window starts must lie on
a segment-local 16-token boundary. The resulting windows remain 17/18 tokens
where the parent allocation spent slack.

- Protocol: `refine-logs/stratified_fixed_chunk_control_protocol.json`
- Parent Package B JSONL SHA256:
  `5757ff898b921c1b0fcc6ed1e76d195667070999d56bf71189a752e38d49e1ab`
- Formal run:
  `/mnt/nvme0/hmo/runs/stratified_fixed_control_formal_20260904_1633/`
- Formal JSONL SHA256:
  `3a399d1012c4d4b82f3a2286ba4a8354562753257e923a0b9b09690faa64b058`
- Formal summary SHA256:
  `b831cd2be9dc30e4214fab6647d0fbfadc29a5b883a30c50f2cd49fc9eaeec24`
- Smoke summary SHA256:
  `05a8dbc93d471f65109714cdbb8be8e0edc2b9d6fdb363be2c61ee000a05923e`

The formal run completed 24/24 cases. The recomputed HMO allocation and
retained positions exactly matched every SHA-pinned parent row before the new
arm was generated. Measured resident KV bytes match in 24/24 cases.

## Results

| System | All | Needle | LongEval-Lines |
|---|---:|---:|---:|
| Global Fixed-Chunk Top-K | 16/24 | 10/12 | 6/12 |
| Stratified Fixed-Chunk | 17/24 | 10/12 | 7/12 |
| HMO stratified free-start | **18/24** | 10/12 | **8/12** |

The Global Fixed rows come from the same SHA-pinned Package B parent run. The
following paired comparisons use the same 24 sample IDs.

| Comparison | Delta | Wins / ties / losses | Sign p | Bootstrap 95% CI |
|---|---:|---:|---:|---:|
| Stratified Fixed vs Global Fixed | +4.17 pp | 1 / 23 / 0 | 1.000 | [0.00, 12.50] pp |
| HMO vs Stratified Fixed | +4.17 pp | 2 / 21 / 1 | 1.000 | [-8.33, 16.67] pp |
| HMO vs Global Fixed | +8.33 pp | 2 / 22 / 0 | 0.500 | [0.00, 20.83] pp |

HMO and the aligned control generate identical token IDs in 21/24 cases.
Their mean retained-position Jaccard is 0.646. On average, 54.54 of 59.5
Sparse segments per sample change position, with a range of 40 to 58, so the
control is not a near-identity perturbation. Runtime was 183.15 seconds on one
RTX 5090; peak CUDA memory was 4.35 GiB allocated and 5.18 GiB reserved.

## Disagreement Audit

All three HMO/aligned disagreements are LongEval cases.

| Sample | Global / aligned / HMO | Gold | HMO output | Aligned output |
|---|---:|---|---|---|
| `longeval_0003` | 0 / 0 / 1 | `CXOZD3` | `CXOZD3` | `26HHCR` |
| `longeval_0004` | 0 / 0 / 1 | `WRPK4H` | `WRPK4H` | `WRP` |
| `longeval_0008` | 0 / 1 / 0 | `4X8YOH` | `LMP35Y` | `4X8YOH` |

The total-count sequence `16 -> 17 -> 18` is not a nested, additive staircase.
Aligned recovers `0008`, whereas HMO recovers `0003` and `0004`. Token-position
inspection nevertheless confirms a placement mechanism: on `0004`, HMO keeps
all five answer tokens while the aligned window keeps only two; on `0008`, the
aligned window keeps 14/15 tokens from the target register line while HMO keeps
7/15. Direct answer-token retention alone does not explain every generation,
so this audit supports placement sensitivity rather than a deterministic
coverage-to-correctness rule.

## Claim Decision

Independent result-to-claim review returns `partial`, routes to `supplement`,
and assigns medium confidence. The supported paper claim is:

> Under HMO's 16K/10% stratified allocation, query-guided free-start windows
> provide a small directional gain over boundary-aligned local placement, with
> the effect concentrated on structured LongEval cases. Placement therefore
> explains part, but not all, of HMO's advantage over Global Fixed-Chunk.

This result strengthens the two-level design story but does not establish a
universal free-start advantage across tasks, budgets, lengths, or models. A
fourth `global allocation + free-start` corner would complete a 2x2 mechanism
decomposition, but it requires a new non-overlapping global-window definition.
Given the current synthetic evidence depth, realistic-task transfer now has
higher paper value than immediately expanding this control family.
