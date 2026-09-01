# E3-v2 P0-D Implementation Report

Date: 2026-09-02
Branch: `dev/e3-v2-p0d`
Scope: equal-byte oracle planning, isolated alpha probing, and grouped statistics

## Decision

P0-D is a code-level PASS. The implementation now provides a deterministic,
label-blind equal-byte oracle plan, recoverable cache interventions, an isolated
Full-KV alpha probe, background-reduced pair labels, and sample-grouped
incremental-value statistics.

This does not open the P1 GPU gate. No pretrained weights or GPU were used. The
integrated real-model preflight must still supply evidence for all eight frozen
integrity checks, especially Full-KV equivalence, repeated-arm determinism, and
a controlled needle logit effect.

## Implemented Contract

### Equal-byte oracle

- The intervention unit is a full, non-protected, non-partial context segment.
- The first and last protected segments are identical in every arm.
- The middle budget is computed from measured attention-KV bytes and rounded
  down to complete equal-cost segments.
- Donors and multiple backgrounds are selected deterministically before quality
  labels, with cross-position-bin donors preferred.
- Every `(sample, R, i, j)` comparison records exact target/donor segment sets,
  charged bytes, resident bytes, positions, costs, and a content-derived ID.
- Both arms drop all other middle attention KV while leaving recurrent state
  unchanged.
- Charged context bytes and post-query decode-resident bytes are audited for
  exact equality.

### Manifest recoverability

`OraclePlan.from_dict` verifies both the SHA-256 payload hash and reconstructed
semantics. It recomputes segment coverage, protection, partial status, position
bins, eligible costs, budget slots, protected/total bytes, comparison IDs,
background counts, and donor graph degree. A self-consistent hash cannot hide a
semantically invalid manifest.

### Isolated alpha

`collect_isolated_query_alpha` creates a private Full-KV context cache, processes
the query suffix with eager attention output, averages across layers, heads, and
query tokens, and returns only per-segment context attention mass. The private
cache is never returned or reused by an oracle arm, and attention backend
configuration is restored even on failure.

### Labels and statistics

- Primary arm quality is mean gold-answer log probability per token from
  post-intervention logits; the official dataset score remains secondary.
- Backgrounds are averaged per unordered donor pair before signed segment
  utility is computed.
- Analysis includes within-sample pairwise accuracy, NDCG at the byte-budget
  `k`, Spearman correlation, residual association controlling alpha and
  position, within-alpha-bin ranking, task stratification, and sample-grouped
  bootstrap intervals.
- Incremental candidate evaluation compares `alpha + position` with the same
  ridge diagnostic plus one recurrent candidate using sample-grouped CV.

### Fail-closed runner gate

`IntegrityGateReport` requires exactly the eight preregistered checks, a boolean
result, and non-empty evidence for each. Missing, extra, failed, or undocumented
checks block scientific execution.

## Verification

CPU-only P0-D suite:

```text
python -m unittest experiments.test_p0d_oracle_statistics -v
Ran 19 tests in 0.109s
OK
```

CPU-only P0-A through P0-D regression:

```text
python -m unittest \
  experiments.test_p0a_validity \
  experiments.test_p0b_context_query \
  experiments.test_p0c_recurrent_signals \
  experiments.test_p0d_oracle_statistics -v
Ran 60 tests in 0.126s
OK
```

Only the existing optional `fuzzywuzzy` acceleration warning and a SWIG exit
deprecation warning appeared. They do not affect test results.

## Configuration Constraint

The frozen 10% middle-KV budget and requirement for multiple distinct
backgrounds imply at least two retained middle slots. At an 8K context, a
512-token segment configuration can leave only one slot after protection. The
planner intentionally rejects this rather than silently duplicating an empty
background. The real-model preflight must select a manifest-recorded segment
length that yields at least two slots; 256 tokens is the first configuration to
evaluate, not a result established by P0-D.

## Next Gate

Build the bounded real-model preflight runner around P0-A through P0-D and run it
on the smallest representative pretrained Qwen3.5 model. The runner must record
all eight evidence items and refuse oracle label production until every item
passes. Only then may the discovery pilot begin.
