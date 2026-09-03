# Codex Review: Coverage-Fidelity HMO Sprint

## Verdict

I support GPT's direction, with implementation-level revisions before execution.

The P3 failure retires frozen V2's independent hard Top-K mapping. It does not
retire corrected Attention demand, query-conditioned recurrent accessibility,
or the HMO problem. A calendar-bounded development phase followed by one
method/claim freeze and one fresh confirmation package is appropriate.

I do not support running the proposed `beta={0.25,0.50,0.75}` sweep or
modifying the frozen P3 runner as written.

## New Evidence From P3 Changed Cases

The P3 prompts were deterministically rebuilt without GPU use, and the answer
token positions were mapped to 256-token segments. All outcome-changing cases
are consistent with an explicit coverage failure:

- 8K had two V2 wins. In both, V2 added the answer-bearing segment that raw
  alpha omitted.
- 8K had one V2 loss. V2 removed the answer-bearing segment.
- 16K sample `longeval_0002` was a V2 loss: raw retained the sole
  answer-bearing segment 11, while V2 removed it and added segments 1, 4, and 6.
- 16K sample `longeval_0008` was a V2 loss: the answer crossed segments
  34-35; V2 retained 35 but removed 34, while adding early segments 1 and 2.

This is stronger support for the coverage-fidelity hypothesis than the aggregate
accuracy alone. It is still diagnostic evidence on development samples, not a
new confirmation result.

## Required Corrections

### 1. P2 Oracle Labels Cannot Be Joined Directly To P3

P2 and P3 use different fresh seeds and sample IDs. P3 artifacts contain
selected sets and generations, but not marginal oracle utilities or
teacher-forced gold log probabilities. A Day-1 comparison of marginal oracle
utility with P3 generation therefore requires a small new diagnostic run.

The useful diagnostic is not another full oracle campaign. For only the ten P3
changed-set samples, measure:

1. teacher-forced gold-answer log probability for the actual raw and V2 sets;
2. answer-bearing segment inclusion;
3. whether fixed-width sparse skeletons retain the answer tokens;
4. optionally, the single moved-in/moved-out swap when a unique swap exists.

This distinguishes metric mismatch from missing coverage and set interaction at
low cost.

### 2. The Proposed Beta Grid Is Degenerate

With `segment_length=256` and `sparse_width=8`, sparse coverage costs
`8/256 = 3.125%` of full middle KV per covered segment. At a 10% middle
budget, covering every eligible segment sparsely costs only 3.125% of full
middle KV and leaves about 6.875% for exact upgrades.

Consequently, coverage budgets above roughly 31.25% of the 10% total budget
already saturate all segments. `beta=0.50` and `beta=0.75` either collapse
to the same allocation when unused coverage budget rolls forward, or waste
bytes when it does not. Neither is a clean experiment.

### 3. Sparse KV Is Available But Not Yet Validated At This Regime

The existing token skeleton keeps within-segment tokens by summed K/V norm. Its
cache intervention and byte accounting are implemented, but the historical
results do not validate an 8-token skeleton at the P3 10% budget. Before making
Sparse a paper-level action, the changed-case diagnostic must check whether the
answer-bearing tokens survive at widths 8 and 16. Corrected token-level query
attention is a fallback within the same Attention-demand family if K/V norm
systematically removes the evidence, not an automatic extra candidate.

### 4. Preserve The Frozen P3 Runner

`run_end_task.py` is the frozen P3 executable and should remain unchanged.
Implement CF-HMO in:

- `experiments/phase2/e3_v2/coverage_fidelity.py`
- `experiments/phase2/e3_v2/run_coverage_fidelity.py`
- `experiments/test_coverage_fidelity.py`

The new runner may reuse P3 prompt, probe, generation, and accounting
primitives. This keeps historical P3 reproduction unambiguous.

### 5. Freeze The Claim Before Fresh Confirmation

Development can choose the stable task and budget scope. At the final freeze,
record both the method and the primary claim/metric. After confirmation, report
all cells and choose only a framing supported by that frozen scope. Do not
post-hoc redefine the primary task or hide conflicting lengths.

A 27B pilot is contingent on a healthy small-model confirmation path. It is not
a prerequisite for this week's small-model method decision.

## Revised Allocator

Use the same three actions but remove the arbitrary beta split. Treat allocation
as precedence-constrained marginal transitions:

- Recurrent-only to Sparse: gain density `a_i / b_i^S`.
- Sparse to Exact: gain density
  `a_i * (1 - r_i) / (b_i^E - b_i^S)`.

Start after charging protected exact KV. Repeatedly take the highest-density
affordable transition; an Exact upgrade is eligible only after that segment is
Sparse. Continue until no transition fits. This uses the exact byte budget,
automatically balances coverage and fidelity, and introduces no beta.

The no-access ablation uses the same allocator and costs but sets the
accessibility deficit to a constant for the Sparse-to-Exact transition. Thus the
ablation isolates whether recurrent accessibility improves fidelity placement,
without changing coverage or total bytes.

Use within-sample rank normalization first. Sparse width is the only initial
development choice: compare 8 versus 16 only if the changed-case retention
diagnostic shows a meaningful difference.

## Revised Execution Order

### D0: Focused Mismatch Diagnosis

- Persist the answer-segment mapping above.
- Add actual-set gold log probability and sparse-token survival for ten changed
  cases.
- Produce one compact diagnosis, not a new generic review gate.

### D1: Implement And Test

- Add the new allocator and runner without editing frozen P3 behavior.
- Test deterministic actions, precedence, exact byte accounting, protected
  invariance, no-access isolation, and a tiny-cache intervention.

### D2: Development

- Start with 10% at 8K on existing P3 development prompts.
- Compare raw-alpha exact Top-K, CF-HMO, no-access, and Full-KV.
- Carry at most the best two sparse settings to 16K.
- Run 5%/20% only for the leading setting, to establish a Pareto curve without
  a broad Cartesian sweep.

### D3: Single Freeze

- Freeze method, sparse selection rule, width, model revision, primary scope,
  metrics, budgets, and fresh seeds once.
- No performance-driven changes afterward.

### D4: Fresh Confirmation

- Run 8K and 16K as one package, preferably two fresh seeds.
- Preserve exact paired predictions, resident bytes, action counts, runtime,
  Full-KV reference, raw-alpha baseline, and no-access ablation.
- Add a stronger compressed baseline when its semantics and budget accounting
  are faithful enough to name honestly.

### D5: Paper Package

- Build the mechanism, Pareto, action-distribution, and case-study figures from
  complete results.
- Use one final Opus adversarial review for the whole evidence package.

## Immediate Recommendation

Proceed with D0 first. It is the smallest action that can falsify the proposed
bridge before allocator implementation. If sparse skeletons do not preserve the
answer evidence in the known loss cases, fix the within-segment coverage
primitive before building CF-HMO; if they do, implement the revised
precedence-constrained allocator.
