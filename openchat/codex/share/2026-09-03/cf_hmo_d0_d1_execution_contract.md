# CF-HMO D0/D1 Execution Contract

## Decision

GPT's follow-up resolves the main scientific disagreements. Proceed with D0 and
the implementation-independent part of D1 in parallel, while keeping the frozen
P3 runner unchanged.

D0 is not a claim gate. It chooses and validates the Sparse primitive. A failed
8-token K/V-norm skeleton does not reject coverage-fidelity, but it does prevent
that unvalidated primitive from being silently built into the new runner.

## D0: Focused Diagnosis

Use only the ten P3 changed-set LongEval samples as declared development cases.

Already complete without GPU:

- deterministic reconstruction of answer-bearing token segments;
- raw/V2 moved-in and moved-out segment mapping;
- observation that all five outcome changes track complete answer-segment
  coverage.

Small GPU diagnostic:

1. teacher-forced gold-answer log probability for the actual raw, V2, and
   Full-KV sets;
2. answer-token survival for sparse widths 8 and 16;
3. compare within-segment K/V-norm selection with corrected token-level
   sequential query-attention selection;
4. record exact resident bytes and per-case retained positions.

This is development-only evidence. It may select the Sparse primitive and width,
but must not be reported as fresh confirmation.

## D1a: Allocator And Tests

D1a can run in parallel with D0 because it depends only on action costs and
segment-level demand/accessibility:

- implement a pure deterministic allocator in
  `experiments/phase2/e3_v2/coverage_fidelity.py`;
- add unit tests for protected-region charging, Sparse-before-Exact precedence,
  deterministic ties, exact costs, no-access isolation, and residual budget;
- do not edit `run_end_task.py`.

The allocator has two explicit stages:

1. Sparse coverage floor. If every eligible segment fits at the chosen fixed
   width, cover all of them. Otherwise choose coverage by
   `a_i / b_i^S`.
2. Exact fidelity upgrades among covered segments by
   `a_i * (1-r_i) / (b_i^E-b_i^S)`.

No beta is used.

## Byte Semantics

The shared cap remains:

`protected exact bytes + fraction * eligible full-middle bytes`.

Because Exact upgrades have segment-sized granularity, the allocator must not
discard affordable residual token capacity. After whole Exact upgrades, assign
remaining Sparse token slots deterministically by Attention demand until no
common per-token KV unit fits.

Comparison roles:

- `cf_hmo` versus `cf_hmo_no_access` is the main equal-realized-byte causal
  comparison. The no-access arm changes only the recurrent deficit.
- `sparse_only` uses the same coverage primitive and byte cap but performs no
  Exact upgrades.
- `raw_alpha_exact_topk` remains the historical hard-segment baseline; report
  its realized bytes and quality honestly when segment granularity leaves cap
  slack.
- `full_kv_reference` remains a reference, not a compressed baseline.

No-access should set recurrent deficit to a constant while preserving coverage,
costs, tie-breaking, and residual allocation.

## D1b: Cache Integration

Begin only after D0 selects K/V norm versus token query-attention and width 8
versus 16:

- implement `run_coverage_fidelity.py`;
- reuse P3 cache isolation, corrected query processing, generation, metrics,
  and accounting primitives;
- add focused tiny-cache and one-sample real-model smoke tests;
- persist raw predictions, actions, selected token positions, and both cap and
  realized bytes.

Extending the alpha probe to return token mass must use a new result/API path so
the frozen segment-level alpha behavior remains reproducible.

## Development And Freeze

After D0/D1:

1. Run 10% 8K development first.
2. Add 5% for the leading primitive; carry at most two settings to 16K.
3. Compare CF-HMO, no-access, Sparse-only, raw-alpha Exact Top-K, and Full-KV.
4. Freeze method, sparse primitive, width, budgets, model revision, fresh seeds,
   primary metric, and ordered claim ladder once.
5. Run fresh 8K/16K confirmation as one package, preferably with two seeds.

The claim ladder must be ordered before confirmation. The recurrent-specific
claim depends primarily on CF-HMO versus no-access at equal realized bytes.
Pareto and raw-alpha comparisons are secondary and all cells remain reportable.

No P0/preflight replay is needed unless recurrence, query sequencing, cache
intervention, or byte-accounting semantics change. A 27B run remains contingent
on a healthy small-model confirmation.

## Immediate Authorized Bundle To Request

The next concrete bundle should be limited to:

1. implement the D0 diagnostic;
2. implement D1a allocator and CPU tests;
3. run D0 on GPU1;
4. append the diagnosis to OpenChat;
5. stop before D1b integration if the Sparse primitive remains ambiguous.

Expected D0 GPU use is several minutes on Qwen3.5-0.8B with less than the
approximately 6 GB already observed at 16K.
