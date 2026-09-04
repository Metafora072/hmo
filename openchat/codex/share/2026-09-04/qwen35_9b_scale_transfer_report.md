# Qwen3.5-9B Scale-Transfer Report

## Decision Summary

The frozen single-GPU Qwen3.5-9B transfer completed all 24 formal cases.
The result strengthens the paper's locality mechanism across model scale:

- Contiguous CF: 23/24 answer containment.
- Scattered CF: 19/24 at exactly the same measured resident KV bytes.
- Paired Contiguous versus Scattered: +16.67 pp, 4 wins, 20 ties, 0
  losses. The exact sign-test p-value is 0.125 at this small transfer size.
- Contiguous CF and Full KV: both 23/24, with identical binary outcomes on
  all 24 cases.
- Contiguous CF and Raw Exact+Slack: identical generated token sequences on
  all 24 cases.
- Raw Exact Top-K: 24/24 while using slightly fewer realized bytes. Its one
  apparent win is a formatting-sensitive Needle case: Raw generated
  `8:38 o'clock`, while Contiguous, Raw+Slack, Scattered, and Full KV
  generated `8:38` for the reference `838 o'clock`.

The paper should lead with cross-scale contiguous-versus-scattered evidence and
near-Full quality at a small footprint. It should not claim that coverage-
fidelity consistently beats Raw Exact segment Top-K.

## Provenance

| Item | Value |
|---|---|
| Git commit | `7f51473` |
| Model | `Qwen/Qwen3.5-9B` |
| Model revision | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| Protocol | `refine-logs/contiguous_cf_scale_transfer_9b_protocol.json` |
| Protocol SHA-256 | `b58a69a6acc192f218c24e410ff188d9f1ab8eec7f0b04f7df5c366b3f1e09f8` |
| Formal manifest | `c03a4e4863bfe80688fb8062a9fea4986dd1eaa5a8dd705e5a4e9ac9bab4d7b7` |
| Formal cases | 24 fresh cases: 8K/16K x Needle/LongEval-Lines x 6 |
| GPU | One NVIDIA GeForce RTX 5090, physical GPU1 only |
| Raw result directory | `/mnt/nvme0/hmo/runs/contiguous_cf_scale9b_formal_c202236_20260904/` |
| Smoke result directory | `/mnt/nvme0/hmo/runs/contiguous_cf_scale9b_smoke_c202236_20260904_r2/` |

The method was transferred without retuning: 256-token segments, 16-token
max-attention contiguous windows, 10% eligible-middle KV cap, and protected
prefix/suffix anchors. The recurrent state remained unchanged.

## Raw Data Table

| System | Containment | Exact Match | Token F1 | Mean resident KV | Mean per-case Full ratio |
|---|---:|---:|---:|---:|---:|
| Contiguous CF | 23/24 (95.83%) | 22/24 (91.67%) | 97.22% | 51,757,056 B | 13.3849% |
| Raw Exact+Slack | 23/24 (95.83%) | 22/24 (91.67%) | 97.22% | 51,757,056 B | 13.3849% |
| Raw Exact Top-K | 24/24 (100.00%) | 23/24 (95.83%) | 98.61% | 51,134,464 B | 13.2674% |
| Scattered CF | 19/24 (79.17%) | 18/24 (75.00%) | 80.56% | 51,757,056 B | 13.3849% |
| Contiguous Sparse-only | 23/24 (95.83%) | 23/24 (95.83%) | 95.83% | 51,757,056 B | 13.3849% |
| Full KV | 23/24 (95.83%) | 21/24 (87.50%) | 95.83% | 397,164,544 B | 100.00% |

The ratio of mean bytes for Contiguous versus Full is 13.0316%; 13.3849% is
the mean of per-case ratios and remains the paper's primary footprint statistic.
All four intended equal-byte compressed arms matched measured resident bytes in
24/24 cases.

## Breakdown By Length And Task

Answer containment counts:

| Group | Contiguous | Raw+Slack | Raw Exact | Scattered | Sparse-only | Full |
|---|---:|---:|---:|---:|---:|---:|
| 8K Needle | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 |
| 8K LongEval | 6/6 | 6/6 | 6/6 | 4/6 | 5/6 | 6/6 |
| 16K Needle | 5/6 | 5/6 | 6/6 | 5/6 | 6/6 | 5/6 |
| 16K LongEval | 6/6 | 6/6 | 6/6 | 4/6 | 6/6 | 6/6 |

Contiguous beats Scattered in two 8K LongEval and two 16K LongEval cases,
with no reverse loss. This is the same mechanism direction observed in the
0.8B fresh confirmation (+14.58 pp, 7 wins, 0 losses), now reproduced at 9B
with a +16.67 pp point estimate.

## Paired Primary-Metric Comparisons

| Contiguous CF versus | Mean delta | Wins / ties / losses | Exact sign p |
|---|---:|---:|---:|
| Raw Exact+Slack | 0.00 pp | 0 / 24 / 0 | 1.000 |
| Raw Exact Top-K | -4.17 pp | 0 / 23 / 1 | 1.000 |
| Scattered CF | +16.67 pp | 4 / 20 / 0 | 0.125 |
| Contiguous Sparse-only | 0.00 pp | 1 / 22 / 1 | 1.000 |
| Full KV | 0.00 pp | 0 / 24 / 0 | 1.000 |

Statistical significance is not used as a continuation gate. The 24-case
transfer is evidence of direction and scale consistency, while the 48-case
0.8B confirmation remains the stronger inferential result for the locality
comparison.

## Raw Exact+Slack Audit

At 8K, the 10% budget happened to be exactly divisible by full segment cost,
so Raw Exact and Raw Exact+Slack had zero residual tokens. At 16K,
Raw Exact+Slack spent:

- 25 additional query-attention tokens per LongEval case;
- 51 additional query-attention tokens per Needle case.

Despite different retained positions, Raw Exact+Slack and Contiguous produced
identical token sequences in all 24 cases. This establishes strict-byte
fairness, but it does not separate the two methods at this high-solvability
operating point. A 5%/10%/20% Pareto sweep is still useful because the lower
budget is more likely to expose policy differences.

## Interpretation

### Observation

Contiguous and Scattered share the same allocator, action counts, query signal,
and measured bytes; only the within-segment retention shape differs.
Contiguous improves containment from 79.17% to 95.83%.

### Interpretation

The scale-transfer result is consistent with the complete-span survival
argument: local codes and relations tolerate a contiguous query-relevant
window better than the same number of globally scattered high-attention
tokens.

### Implication

The most defensible main contribution is locality-preserving residual KV
organization for Hybrid models. Exact upgrades remain optional; recurrent
accessibility is not needed for the supported claim.

### Next Step

1. Run the 0.8B 5%/10%/20% Pareto package with Raw Exact+Slack to reveal the
   quality-memory curve with adequate sample count.
2. Add a real 32K HotpotQA transfer. Qwen3.5-9B peaked at 23.62 GiB allocated
   and 27.85 GiB reserved at 16K, so 32K on this BF16 path is likely too close
   to a 32GB limit. Prefer 0.8B/4B for 32K or engineer memory reduction before
   attempting 9B.
3. Do not spend time on 27B/32B under the current single-card constraint.

## Runtime And Resource Closure

- Formal elapsed time: 721.13 seconds.
- Peak PyTorch allocated memory: 23.62 GiB.
- Peak PyTorch reserved memory: 27.85 GiB.
- The process exited normally.
- GPU1 returned to 15 MiB after completion.
- Persistent disk use: about 19GB model weights, 11MB formal output, and
  352KB smoke output.
