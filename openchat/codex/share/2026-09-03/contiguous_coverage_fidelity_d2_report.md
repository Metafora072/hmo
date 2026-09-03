# Contiguous Coverage-Fidelity D2 Report

## Decision

The contiguous coverage primitive is supported for one fresh confirmation.
The recurrent accessibility multiplier remains rejected.

The leading development configuration is:

- 10% eligible-middle KV cap;
- 16-token Sparse allocation per segment;
- one contiguous window maximizing token query-attention mass;
- attention-led Exact upgrades, equivalent to the current `no-access` arm;
- protected prefix and suffix remain Exact;
- Qwen3.5 DeltaNet recurrent state remains untouched and active.

`no-access` means only that the allocator does not consume the recurrent
accessibility observable. It does not remove DeltaNet memory from the model.
The resulting method is still a Hybrid-model KV policy, but it cannot support a
claim of recurrent-aware allocation.

## Code And Validation

- D2 implementation commit:
  `1b756972026fdb5f716148fd1052825f27294c90`.
- Selector: `max_mass_window`, with earliest-start deterministic tie-breaking.
- Existing `top_tokens` remains a separately selectable reproducibility path.
- D2 survival continuation condition was encoded before GPU outcomes:
  at least 5/10 complete-answer survival cases and a strict gain over scattered
  Top-token selection at the same width.
- Full CPU suite: 124/124 passed.

## D2 Survival Diagnosis

- Result:
  `/mnt/nvme0/hmo/runs/d2_window_survival_changed10_20260903_213011/contiguous_window_diagnosis_summary.json`.
- Manifest:
  `bf62e89cc4efc4b6aa43892863554ff64836f51145da223c6b260b8d3017618f`.
- Runtime: 39.83 seconds on physical GPU1.

| selector | width | complete answer | any answer token | mean retained fraction |
|---|---:|---:|---:|---:|
| scattered Top-token | 8 | 0/10 | 9/10 | 35.2% |
| scattered Top-token | 16 | 0/10 | 10/10 | 49.2% |
| contiguous max-mass window | 8 | 2/10 | 6/10 | 47.0% |
| contiguous max-mass window | 16 | 5/10 | 8/10 | 63.3% |

Width 16 passed the predeclared continuation condition exactly: 5/10 complete
survival and +5 cases over the same-width scattered selector.

Interpretation: scattered tokens maximize broad weak coverage, while a
contiguous window more often preserves the complete local key-value relation.
The lower any-hit count is a real tradeoff, not hidden by the selection rule.

## Actual Equal-Byte Generation

- Result:
  `/mnt/nvme0/hmo/runs/d2_window_w16_b10_changed10_20260903_213208/coverage_fidelity_summary.json`.
- Manifest:
  `8c79c9d057450f2d64e9e1016f12c5091d2c72a6756d99730f83653e8b032f7e`.
- Runtime: 149.53 seconds on physical GPU1.

All values are normalized answer containment on the ten selected P3 changed
LongEval development cases.

| system | correct |
|---|---:|
| recurrent-multiplier CF | 4/10 |
| attention-led contiguous CF (`no-access`) | 8/10 |
| contiguous Sparse-only | 2/10 |
| raw-alpha Exact Top-K | 5/10 |
| Full-KV reference | 7/10 |

Attention-led contiguous CF versus raw Exact:

- mean delta: +0.30;
- wins/ties/losses: 4/5/1;
- 8K mean delta: +0.20;
- 16K mean delta: +0.40.

Attention-led contiguous CF versus Full-KV:

- mean delta: +0.10;
- wins/ties/losses: 3/5/2.

The apparent Full-KV improvement is not a general claim. These ten cases were
selected because P3 segment membership changed, so this is a mechanism-focused
development set with high selection bias.

## Mechanism Separation

### Contiguous coverage is useful

Changing only the Sparse shape from scattered tokens to contiguous windows
improved:

- Sparse-only from 0/10 to 2/10;
- attention-led CF from 3/10 to 8/10 at width 16.

The result supports preserving local relational evidence instead of isolated
high-attention tokens.

### Attention-led Exact upgrades are currently necessary

The recurrent-multiplier arm remained below attention-led allocation: 4/10
versus 8/10, with 0 wins, 6 ties, and 4 losses. It still directed Exact slots
toward low-accessibility early segments rather than the strongest task-demand
segments.

Therefore:

- do not rename the recurrent-multiplier arm as the method;
- do not claim accessibility caused the D2 improvement;
- retain accessibility as a measured diagnostic and documented failed
  development ablation;
- define the candidate method by coverage/fidelity actions and attention-led
  allocation.

## Byte And Integrity Checks

- The three compressed arms were exactly equal in measured post-query resident
  bytes for all ten cases.
- Mean compressed post-query resident bytes: 19,850,035.2.
- Mean Full-KV resident bytes: 148,671,283.2.
- Mean compressed fraction including protected/query KV: 13.60%.
- Raw Exact leaves the declared segment-granularity slack.
- Raw-alpha and Full-KV generated tokens reproduced frozen P3 in 10/10 cases.
- GPU1 returned to 15 MiB after both runs.

## Evidence Boundary

This is positive development evidence, not confirmation:

- all ten cases are LongEval;
- all were selected after P3 because raw/V2 membership changed;
- width and primitive were chosen using these cases;
- there are no fresh Needle results for the new method;
- no cross-model or larger-model evidence exists.

## Recommended Freeze Bundle

Before further GPU execution, freeze one final protocol containing:

1. method: attention-led contiguous coverage-fidelity;
2. width 16, 10% middle cap, segment length 256, one protected prefix/suffix;
3. fresh 8K and 16K samples as one package, with new seeds and IDs;
4. primary comparison: selected method versus raw-alpha Exact Top-K at measured
   resident bytes;
5. ablations: scattered Top-token with the same allocator and contiguous
   Sparse-only;
6. Full-KV as solvability reference;
7. recurrent-multiplier development failure reported separately, not promoted
   to a confirmation claim;
8. claims limited to end-task quality/memory and the contiguous-local-coverage
   mechanism. No recurrent-aware allocation claim.

A fresh result must report all tasks, lengths, systems, predictions, and byte
measurements. No continuation gate between 8K and 16K, and no parameter changes
after observing either length.
