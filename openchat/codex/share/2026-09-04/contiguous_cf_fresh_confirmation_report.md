# Contiguous CF Fresh Confirmation Report

## Decision

The result-to-claim verdict is `partial` with medium confidence.

The fresh experiment supports one narrow mechanism claim: under the same
attention-led allocation and exactly equal resident KV bytes, retaining one
contiguous max-attention-mass window per Sparse segment is better than retaining
scattered top-attention tokens.

The broader method claim remains tentative. Contiguous coverage-fidelity has a
small positive point estimate over raw-alpha Exact Top-K, but the paired
interval crosses zero, only four cases differ, and the method uses 1.22% more
resident KV bytes because raw whole-segment selection leaves rounding slack.

## Frozen Protocol And Provenance

- Code commit: `c50e53df5aacef7bb2fc9b1db58ed27c3de282a2`.
- Protocol: `refine-logs/contiguous_cf_confirmation_protocol.json`.
- Protocol SHA-256:
  `fe89edab545a9907522f2103e4eea75ea854f06655e5ff0ae884d75cf8115f08`.
- Manifest ID:
  `203da24b57ca1aa93a56dccb49bc3f1284cfaddec2f797b5af2c73c30c0c3e54`.
- Scope: fresh confirmation without postselection.
- Model: Qwen3.5-0.8B revision
  `2fc06364715b967f1860aea9cf38778875588b17`.
- Cases: 48 new deterministic samples, with 12 Needle and 12 LongEval-Lines
  samples at each of 8K and 16K.
- Method: 10% eligible-middle cap, segment length 256, Sparse width 16,
  contiguous max-mass window, attention-led Exact upgrades, and one protected
  prefix/suffix segment.
- Raw summary:
  `/mnt/nvme0/hmo/runs/contiguous_cf_confirmation_8k16k_s20261005_06_20260903_221800/contiguous_cf_confirmation_summary.json`.
- Raw rows:
  `/mnt/nvme0/hmo/runs/contiguous_cf_confirmation_8k16k_s20261005_06_20260903_221800/contiguous_cf_confirmation_results.jsonl`.
- Log:
  `/mnt/nvme0/hmo/logs/contiguous_cf_confirmation_8k16k_s20261005_06_20260903_221800.log`.

## Raw Results

The containment and exact-match counts are identical for this synthetic suite.

| system | answer containment | token F1 |
|---|---:|---:|
| contiguous coverage-fidelity | 34/48 (70.83%) | 73.96% |
| raw-alpha Exact Top-K | 32/48 (66.67%) | 70.83% |
| scattered coverage-fidelity | 27/48 (56.25%) | 60.42% |
| contiguous Sparse-only | 32/48 (66.67%) | 69.79% |
| Full KV | 35/48 (72.92%) | 77.08% |

### Task And Length Breakdown

| group | contiguous CF | raw Exact | scattered CF | Sparse-only | Full KV |
|---|---:|---:|---:|---:|---:|
| 8K Needle | 11/12 | 10/12 | 10/12 | 11/12 | 10/12 |
| 8K LongEval-Lines | 5/12 | 6/12 | 3/12 | 6/12 | 9/12 |
| 16K Needle | 10/12 | 10/12 | 10/12 | 10/12 | 10/12 |
| 16K LongEval-Lines | 8/12 | 6/12 | 4/12 | 5/12 | 6/12 |

## Paired Analysis

The 95% intervals use 20,000 sample-level bootstrap resamples with seed
20260904. P-values are two-sided exact sign tests over discordant pairs.

| comparison | wins / ties / losses | delta | bootstrap 95% CI | exact p |
|---|---:|---:|---:|---:|
| contiguous CF vs raw Exact | 3 / 44 / 1 | +4.17 pp | [-4.17,+12.50] | 0.6250 |
| contiguous CF vs scattered CF | 7 / 41 / 0 | +14.58 pp | [+6.25,+25.00] | 0.0156 |
| contiguous CF vs Sparse-only | 6 / 38 / 4 | +4.17 pp | [-8.33,+16.67] | 0.7539 |
| contiguous CF vs Full KV | 3 / 41 / 4 | -2.08 pp | [-12.50,+8.33] | 1.0000 |

Against raw Exact, the containment delta is 0 pp at 8K and +8.33 pp at 16K.
Against scattered retention, it is +12.50 pp at 8K and +16.67 pp at 16K.

## Memory And Integrity

| system group | mean post-query resident KV | mean per-case fraction of Full |
|---|---:|---:|
| three coverage arms | 19,406,592 bytes | 13.383% |
| raw-alpha Exact Top-K | 19,173,120 bytes | 13.266% |
| Full KV | 148,934,400 bytes | 100% |

- The three coverage arms are exactly equal resident-byte in 48/48 cases.
- Contiguous CF uses 233,472 more bytes than raw on average, a 1.22% increase
  over raw caused by whole-segment rounding slack.
- Contiguous CF reduces measured resident KV by 86.97% versus Full KV.
- The ratio of mean contiguous/Full bytes is 13.03%; 13.38% is the mean of the
  per-case ratios. The paper should use the latter and label it explicitly.
- There were no protocol or byte-accounting failures.
- Runtime was 608.02 seconds on one RTX 5090.
- Peak allocated/reserved GPU memory was 4.35/5.05 GiB.
- GPU1 returned to 15 MiB after exit.

At 8K, contiguous CF allocates one Exact and 29 Sparse eligible segments; at
16K it allocates two Exact and 59-60 Sparse segments. Sparse-only makes every
eligible segment Sparse. The small, nonsignificant +2-case difference between
these systems does not establish that Exact fidelity upgrades are necessary.

## Key Findings

1. **Contiguous local coverage is confirmed in scope.** The only change between
   contiguous and scattered CF is the token-retention shape. The +7-case gain,
   zero losses, exact byte equality, positive deltas at both lengths, and exact
   p=0.0156 form a clean causal mechanism result.
2. **The method-versus-raw result is promising but weak.** The point estimate is
   +2/48, but there are only four discordant cases, the interval crosses zero,
   and raw is slightly smaller in measured bytes.
3. **The fidelity component is not separately validated.** Contiguous CF and
   contiguous Sparse-only differ by only +2/48 overall, with opposite 8K and
   16K directions and p=0.7539.
4. **The result is length- and task-structured.** The method-versus-raw gain is
   concentrated in 16K LongEval-Lines. Needle is near saturation and provides
   little discriminative evidence.
5. **This is not a recurrent-aware allocation result.** DeltaNet state remains
   active, but the winning allocator uses attention rather than the rejected
   recurrent accessibility multiplier. The evidence supports a KV policy for
   this Hybrid model, not a Hybrid-specific recurrent-memory allocation claim.

## Result-To-Claim Verdict

An independent internal reviewer returned `partial / medium`:

- supported: contiguous max-mass retention beats scattered top-token retention
  at equal bytes on the fresh Qwen3.5-0.8B 8K/16K synthetic suite;
- tentative: contiguous coverage-fidelity shows a small, mainly 16K advantage
  over raw Exact at measured bytes;
- unsupported: strong superiority over raw, necessity of Exact upgrades,
  recurrent-aware allocation, cross-task robustness, cross-model transfer, or
  larger-model generality.

## Recommended Route

Do not tune width, thresholds, or allocation scores on these confirmation
labels. The paper-safe working claim should center on contiguous local coverage,
with the method-versus-raw result reported as a small directional gain.

The next compact evidence package should be chosen before further GPU work:

1. remove the raw byte asymmetry through exact byte matching or a narrow
   quality-memory Pareto comparison;
2. add one independent fresh seed package to resolve the tie-heavy raw
   comparison;
3. add one less-synthetic long-context task if a broader end-task claim is
   required;
4. treat recurrent-aware allocation as a documented failed design branch, not
   as the contribution of the current method.

## Operational Audit

The first queued launch at 22:56:49 exited before any sample outcome because a
script-path entry point let the project-local `statistics.py` shadow Python's
standard library module. The result directory contained only its already-frozen
manifest. The launcher was changed to the equivalent module entry point; no
scientific code, protocol, sample, or parameter changed. The valid run started
at 00:30:26 and exited with status 0 at 00:40:42 using the same immutable
manifest.
