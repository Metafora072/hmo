# CF-HMO D1b Development Report

## Decision

Stop the current CF-HMO formula before fresh confirmation.

The new cache intervention is valid and byte-exact, but the proposed recurrent
fidelity rule is decisively negative on the ten declared development cases:

- CF-HMO: 0/10 at both Sparse width 16 and width 8.
- no-access: 3/10 at width 16 and 4/10 at width 8.
- Sparse-only: 0/10 at both widths.
- raw-alpha Exact Top-K: 5/10.
- Full-KV reference: 7/10.

This rejects `attention demand * accessibility deficit` as the Exact-upgrade
rule in the current coverage-fidelity design. It does not retract D0's finding
that hard segment Top-K fails through coverage loss.

## Implementation And Validation

- Main D1b implementation commit:
  `88fa640a99787d6cf646322ff87d363fdabdd674`.
- Eligible-signal boundary fix:
  `b75e81ee86559f7ee5f5d354d737b0f017645ff4`.
- New cache module:
  `experiments/phase2/e3_v2/coverage_fidelity_cache.py`.
- New independent runner:
  `experiments/phase2/e3_v2/run_coverage_fidelity.py`.
- Frozen `run_end_task.py` remained unchanged.
- Complete CPU suite after the fix: 120/120 passed.

The intervention materializes every allocator action as physical attention-KV
positions. It verifies planned context bytes, post-query realized bytes, stable
token selection, recurrent-state immutability through the existing P0-B path,
and one-use generation state. Each valid result row stores actions, positions,
cap, realized bytes, predictions, and token IDs.

Residual token budget now uses attention-priority round-robin assignment. This
prevents the Sparse-only ablation from accidentally creating a greedy Exact
upgrade while still consuming every affordable token slot.

## Run Audit

The first launch at commit `88fa640` failed before any generation outcome:

- Failed directory:
  `/mnt/nvme0/hmo/runs/d1b_cf_w16_b10_changed10_20260903_210951`.
- Cause: the hybrid probe exposed protected plus eligible segment signals,
  while the pure allocator correctly required eligible-only keys.
- Artifacts: immutable manifest only; no JSONL result rows.
- Resolution: add an explicit tested boundary adapter, commit `b75e81e`, and
  restart in new run directories. The failed directory remains for audit.

Valid width-16 run:

- Result:
  `/mnt/nvme0/hmo/runs/d1b_cf_w16_b10_changed10_20260903_211228/coverage_fidelity_summary.json`.
- Manifest:
  `9f6ea5951bcd9f9e9d64418e75f11efcda9ca8d230054ecb6a3f07771d9ef5e1`.
- Runtime: 147.00 seconds.
- Peak allocated/reserved: 4,640,177,152 / 5,532,286,976 bytes.

Valid width-8 run:

- Result:
  `/mnt/nvme0/hmo/runs/d1b_cf_w8_b10_changed10_20260903_211624/coverage_fidelity_summary.json`.
- Manifest:
  `ff8f51436a83116f95f8a302a4d13fac6a2eaea68dd7424c3e6bf6fe4cb52eb8`.
- Runtime: 153.58 seconds.
- Peak allocated/reserved: 4,640,177,152 / 5,532,286,976 bytes.

Both runs used physical GPU1. After each process exited, GPU1 returned to
15 MiB and its detached screen disappeared.

## Raw Quality Results

All values are normalized answer containment on the ten P3 membership-changed
LongEval development cases.

| width | CF-HMO | no-access | Sparse-only | raw Exact Top-K | Full-KV |
|---:|---:|---:|---:|---:|---:|
| 16 | 0/10 | 3/10 | 0/10 | 5/10 | 7/10 |
| 8 | 0/10 | 4/10 | 0/10 | 5/10 | 7/10 |

Primary paired comparisons:

| width | comparison | wins | ties | losses | mean delta |
|---:|---|---:|---:|---:|---:|
| 16 | CF vs no-access | 0 | 7 | 3 | -0.30 |
| 16 | CF vs Sparse-only | 0 | 10 | 0 | 0.00 |
| 16 | CF vs raw Exact | 0 | 5 | 5 | -0.50 |
| 8 | CF vs no-access | 0 | 6 | 4 | -0.40 |
| 8 | CF vs Sparse-only | 0 | 10 | 0 | 0.00 |
| 8 | CF vs raw Exact | 0 | 5 | 5 | -0.50 |

Width 16 allocated one Exact middle segment at 8K and two at 16K. Width 8
allocated two at 8K and four at 16K. Sparse-only allocated no Exact middle
segments in either run.

## Byte And Reproduction Checks

- CF, no-access, and Sparse-only had exactly equal post-query resident bytes in
  every case.
- Width 8 and width 16 consumed the same shared cap; the only change was its
  Sparse/Exact composition.
- Mean compressed post-query resident bytes: 19,850,035.2.
- Mean raw Exact post-query resident bytes: 19,696,435.2. Raw leaves segment
  granularity slack, as declared.
- Mean Full-KV post-query resident bytes: 148,671,283.2.
- Mean compressed resident fraction of Full-KV: 13.60%, including protected
  prefix/suffix and query tokens.
- Raw-alpha and Full-KV generated token IDs reproduced the frozen P3 artifacts
  in 10/10 cases for both valid runs.

The comparison is therefore not explained by budget drift or runner changes.

## Mechanism Findings

### 1. Non-contiguous Sparse retention is insufficient here

Sparse-only remained 0/10 even after deterministic residual spending increased
most segments beyond the nominal width. CF also tied Sparse-only in all twenty
width-by-case observations. Keeping scattered high-attention tokens did not
preserve the exact passkey relation needed for generation.

### 2. The recurrent deficit redirects Exact capacity away from the answer

CF never upgraded a complete answer-bearing segment at either width: 0/10.
At width 8 it repeatedly upgraded early segments such as 3-8, while answer
segments were often around 11-35. This is a direct failure mode of treating low
query read-share as positive need without a sufficiently strong relevance
constraint.

No-access width 8 upgraded the complete answer-bearing segment in 6/10 cases
and answered 4 correctly. The three width-16 no-access successes also exactly
coincided with complete answer-segment upgrades. More Exact slots help only
when attention relevance, rather than recurrent deficit, controls them.

### 3. Coverage remains the right diagnosed problem, but this solution is not

D0 established that all five P3 win/loss changes follow complete answer segment
coverage, with teacher-forced logprob agreeing 5/5. D1b adds that scattered
token coverage is not a sufficient substitute and that the proposed recurrent
upgrade score is actively harmful. The observation survives; the current
design does not.

## Stop Conditions And Next Choice

Do not run 5%, fresh 8K/16K confirmation, or a larger model for this formula.
Do not reverse or retune the accessibility score on these same labels.

There are two defensible next directions:

1. Minimal new research hypothesis: replace scattered top-token Sparse with one
   contiguous window per segment, selected by maximum query-attention mass.
   First perform a development-only survival diagnosis; proceed to actual cache
   generation only if it preserves complete local evidence. Keep Exact upgrades
   attention-led initially, and treat recurrent accessibility as an observable
   or bounded tie-break rather than a dominant multiplier.
2. Stop method iteration and frame the project around the validated measurement
   chain and negative systems result: recurrent accessibility predicts oracle
   ranking in some settings, but independent hard Top-K and deficit-driven
   coverage-fidelity both fail at end-task generation because exact local
   evidence dominates.

Direction 1 is the only compact implementation continuation supported by the
new failure anatomy. It is a new design decision and should be explicitly
authorized before execution.
