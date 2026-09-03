# Frozen V2 Equal-Byte End-Task Report

## Verdict

The preregistered end-task claim gate failed. Frozen V2 is not supported as a
deployable KV selector under the current claim. It remains a useful
query-conditioned recurrent-accessibility observable and a positive oracle
ranking result, but that ranking gain did not transfer reliably to generated
answer quality.

An independent internal Codex result-to-claim review returned
`claim_supported: no`, `confidence: high`. Supplying the complete exact-match
and token-F1 results did not change the verdict.

## Protocol

- Model: `Qwen/Qwen3.5-0.8B`, revision
  `2fc06364715b967f1860aea9cf38778875588b17`.
- Code commit: `d88756b`.
- Frozen V2 SHA-256:
  `01bcd3a9ea864ef2b9ab64c23c058badec70d7c88976168ff77100668b49a5f5`.
- Amended protocol SHA-256:
  `a085d5d27f55d1edadd467bd6d4ae5a0df3b53c58908b2b8d6c426bf417febe6`.
- Primary comparison: corrected raw-alpha Top-K versus frozen V2 Top-K.
- Primary metric: normalized answer containment; greedy decode, at most 32
  tokens.
- Exact budget: 10% of eligible middle attention KV in whole 256-token
  segments, with identical protected regions.
- Full-KV is a solvability reference, not a competing compressed baseline.

The initial 2K smoke failed before generation because its exact whole-segment
budget rounded to zero slots. The documented pre-outcome amendment changed only
the smoke context to 4K. Formal 8K/16K settings were unchanged.

## Raw Results

| Stage | Samples | Raw alpha | Frozen V2 | V2 - raw, 95% CI | Full-KV | Changed sets | W/T/L |
|---|---:|---:|---:|---:|---:|---:|---:|
| Smoke 4K | 2 | 50.00% | 100.00% | +50.00 pp [0,+100.00] | 50.00% | 1 | 1/1/0 |
| 8K | 24 | 70.83% | 75.00% | +4.17 pp [-8.33,+16.67] | 79.17% | 5 | 2/21/1 |
| 16K | 24 | 83.33% | 75.00% | -8.33 pp [-20.83,0] | 83.33% | 5 | 0/22/2 |
| Formal combined | 48 | 77.08% | 75.00% | -2.08 pp [-10.42,+6.25] | 81.25% | 10 | 2/43/3 |

The combined interval uses 10,000 sample-grouped bootstrap draws. The smoke is
execution evidence only and is excluded from the formal combined result.

| Stage/task | Raw alpha | Frozen V2 | Full-KV | V2 - raw |
|---|---:|---:|---:|---:|
| 8K Needle | 100.00% | 100.00% | 91.67% | 0 |
| 8K LongEval | 41.67% | 50.00% | 66.67% | +8.33 pp |
| 16K Needle | 91.67% | 91.67% | 91.67% | 0 |
| 16K LongEval | 75.00% | 58.33% | 75.00% | -16.67 pp |
| Combined Needle | 95.83% | 95.83% | 91.67% | 0 |
| Combined LongEval | 58.33% | 54.17% | 70.83% | -4.17 pp |

Strict normalized exact match follows the same paired deltas. Combined token F1
is 78.13% for raw alpha, 76.04% for V2, and 82.29% for Full-KV; the paired V2
delta is again -2.08 pp.

## Contract And Resource Checks

- All 48 formal raw/V2 pairs have exactly equal post-query resident attention-KV
  bytes; maximum pairwise byte difference is zero.
- 8K compressed bytes range from 14,204,928 to 14,475,264, versus Full-KV
  99,139,584 to 99,409,920.
- 16K compressed bytes range from 22,831,104 to 25,190,400, versus Full-KV
  198,168,576 to 199,028,736.
- No protocol errors occurred. All ten formal membership changes were on
  LongEval; Needle always abstained.
- 8K runtime: 211.40 seconds, peak allocated GPU memory 3.21 GB.
- 16K runtime: 310.34 seconds, peak allocated GPU memory 4.88 GB.
- GPU1 returned to 15 MiB after each run. GPU0 was not used.

Artifacts:

| Stage | Result path | SHA-256 |
|---|---|---|
| Smoke | `/mnt/nvme0/hmo/runs/p3_endtask_v2_smoke4k_s20261000_20260903_181347/end_task_summary.json` | `2cb415bd9f6f60f9f4bab2a45fcc226f00741bb4445ffdaa3f0f7dcb5f66626a` |
| 8K | `/mnt/nvme0/hmo/runs/p3_endtask_v2_8k_s20261001_20260903_181535/end_task_summary.json` | `f43a1a207b639a2e154d984c3c11e330410b548aee83bc507aa12427c39953b4` |
| 16K | `/mnt/nvme0/hmo/runs/p3_endtask_v2_16k_s20261002_20260903_182021/end_task_summary.json` | `a700a49e379fb1b49448413583bf10d1813cc2d8de4433db20d126665ba5c372` |

## Gate Decision

The 8K continuation gate passed: overall and Needle deltas were nonnegative,
five LongEval sets changed, and there were no errors. The final claim gate did
not pass:

- combined LongEval wins greater than losses: false, 2 wins versus 3 losses;
- Needle nonnegative at both lengths: true;
- equal-byte errors equal zero: true.

## Interpretation And Route

P2 established that V2 can rank marginal oracle segment utility better than raw
alpha on fresh samples. P3 shows that this does not imply a better jointly
retained set under greedy generation. Plausible explanations include
non-additive interactions among retained segments, mismatch between the
teacher-forced oracle utility and exact generated correctness, and
length-dependent noise in relative recurrent read-share. P3 does not distinguish
these explanations.

Stop this frozen V2 as a deployable selector and do not retune its thresholds,
formula, or rank normalization on P2/P3 labels. Preserve corrected sequential
alpha, exact query-to-recurrent-contribution measurement, the prospective P2
oracle result, and the P3 negative transfer result.

Before proposing another controller, the next approved work should diagnose the
oracle-to-generation mismatch on actual selected sets. A new selector should
only follow a prospectively frozen mechanism and should be evaluated with more
than one fresh seed plus a stronger compressed baseline.
