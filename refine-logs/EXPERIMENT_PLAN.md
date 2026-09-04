# Experiment Plan

**Problem**: prospective validation of query-conditioned recurrent-state-aware exact-KV allocation

**Method**: frozen dual-confidence abstention V2

**Protocol**: `query_accessibility_v2_prospective_protocol.json`

## Claim Map

| Claim | Required evidence |
|---|---|
| The retrospective 8K pattern reproduces | Fresh 6+6 oracle samples pass the preregistered continuation gate versus corrected raw alpha |
| The frozen controller transfers to 16K | Only after 8K passes: fresh 4+4 samples are positive overall and on LongEval, with Needle at least -0.005 |
| The method is generally length robust | Not claimable from these pilots alone; later larger-scale end-task and baseline evaluation is required |

## Run Order

| Run | Purpose | Gate | Approximate cost |
|---|---|---|---:|
| P2-8K-O | Fresh 8K equal-byte oracle acquisition | No candidate analysis | 33 min |
| P2-8K-Q | Frozen V2 query-accessibility evaluation | Apply fixed 8K continuation gate | 2-3 min |
| P2-16K-O | Fresh 16K oracle acquisition | Run only if P2-8K-Q passes | 35-45 min |
| P2-16K-Q | Frozen V2 query-accessibility evaluation | Diagnose transfer, do not tune | 2-3 min |

## Stop Rules

- No threshold, formula, task-conditioned rule, or length normalization changes.
- If fresh 8K fails its continuation gate, stop V2 and do not run fresh 16K.
- If 8K passes and 16K fails, preserve the result as a length-regime shift candidate and diagnose without fitting these outcomes.
- Bootstrap intervals report uncertainty; sparse abstention makes a strictly positive lower bound unsuitable as the sole pilot gate.

## Outcome

Both prospective stages passed their scoped oracle Top-K NDCG criteria. The
result-to-claim verdict is `partial`: proceed only under a new approved plan for
end-task quality and baseline evidence; do not tune V2 on P2 outcomes.

## P3 End-Task Validation

### Claims

- Primary: the frozen V2 membership decision improves equal-byte generated
  answer quality over corrected raw alpha on fresh LongEval-Lines samples.
- Anti-claim: the oracle NDCG gain is only a ranking artifact that disappears
  when the retained segment set is actually used for decoding.

### Systems and Metrics

- Equal-byte systems: corrected raw-alpha Top-K and frozen V2 Top-K.
- Reference only: Full-KV generation for task solvability.
- Primary metric: normalized answer containment.
- Secondary: strict normalized exact match, token F1, paired win/loss, active
  membership changes, and resident KV bytes.

### Run Order

| Run | Split | Gate | Priority |
|---|---|---|---|
| P3-S | 1+1 at 4K, seed 20261000 | complete; exact byte equality; no protocol errors | MUST |
| P3-8K | 12+12 at 8K, seed 20261001 | no overall/Needle regression and at least 2 LongEval membership changes | MUST |
| P3-16K | 12+12 at 16K, seed 20261002 | run only after P3-8K gate | MUST |

No controller, threshold, sample, or metric changes are allowed after smoke.
The original 2K smoke was amended to 4K before any generation outcome because
the exact 10% whole-segment budget contained zero eligible slots at 2K.

## P4 Package B: Structured-Baseline Pareto

**Protocol**: `contiguous_cf_pareto_protocol.json`

### Claims

- Map quality against measured resident KV at 5%, 10%, and 20% middle-context
  caps on the existing 48-case 0.8B confirmation suite.
- Separate generic chunk retention from HMO's macro-segment coverage and
  query-guided free-start placement.
- Recheck contiguous versus scattered geometry across all three budgets.

### Systems and Metrics

- Strictly equal-byte systems: Contiguous CF, Global Fixed-Chunk Top-K, Raw
  Exact+Slack, Scattered CF, and Contiguous Sparse-only.
- Reference: Full KV, generated once per sample and reused across budget rows.
- Primary metric: normalized answer containment.
- Secondary metrics: normalized exact match, token F1, paired win/tie/loss,
  measured resident KV bytes, and fraction of Full-KV bytes.

Global Fixed-Chunk partitions every eligible 256-token segment into aligned
16-token chunks, ranks all chunks globally by the same query-attention probe,
and retains chunks in rank order. A non-multiple byte target takes its final
tokens from the fixed-boundary prefix of the next ranked chunk; this
deterministic tail rule preserves exact byte equality without using a free-start
window.

### Run Order

| Run | Split | Gate | Priority |
|---|---|---|---|
| PB-S | 1 Needle sample at 8K, all three budgets | Operational correctness only | MUST |
| PB-P | Existing 12+12 at 8K and 12+12 at 16K, all budgets | No result gate | MUST |

The smoke must verify protocol integrity, nonempty plans, exact resident-byte
equality, parseable outputs, and successful generation. Its result does not
control whether the formal Pareto run proceeds.

## P5 Stratified Fixed-Chunk Mechanism Control

**Protocol**: `stratified_fixed_chunk_control_protocol.json`

### Claim

Isolate segment-internal free-start placement from macro-segment allocation.
The control reuses the 10% Contiguous CF allocation, including protected
segments, Exact upgrades, Sparse retained-token counts, and slack. Its only
change is to restrict each Sparse window start to a segment-local 16-token
boundary.

### Systems And Metrics

- Contiguous CF generation and measured bytes are reused from the SHA-pinned
  Package B parent run.
- Stratified Fixed-Chunk is newly generated and scored against dataset ground
  truth.
- Primary metric: normalized answer containment.
- Secondary: exact match, token F1, paired win/tie/loss, retained-position
  Jaccard, and measured resident KV bytes.

### Run Order

| Run | Split | Gate | Priority |
|---|---|---|---|
| P5-S | First frozen 16K Needle case at 10% | Operational correctness only | MUST |
| P5-P | Frozen 12+12 16K suite at 10% | No result gate | MUST |

The runner must recompute the parent HMO allocation and retained positions and
match them exactly against the frozen Package B row before generating the
control. The experiment is descriptive: either direction informs whether the
16K advantage comes from stratification or free-start placement.

### Outcome

P5 completed 24/24 exact equal-byte cases. HMO obtains 18/24 versus 17/24 for
Stratified Fixed-Chunk, with 2 wins, 21 ties, and 1 loss; all disagreements are
LongEval. Together with the parent Global Fixed result of 16/24, this provides
directional support for both macro organization and free-start placement, but
the sample-level gains are complementary rather than a nested additive ladder.
The result-to-claim verdict is `partial/supplement` with medium confidence.

## P6 HotpotQA-32K-Aug Full-KV Solvability

**Protocol**: `hotpotqa_32k_solvability_protocol.json`

### Purpose

Determine whether Qwen3.5-0.8B can answer any real HotpotQA questions at a 32K
memory-context length before spending GPU time on matched compressed systems.
This is a model/task routing check, not HMO evidence.

LongBench HotpotQA contains no native 32K records under the pinned Qwen3.5
tokenizer: its 200 contexts range from roughly 1.8K to 17.7K tokens. The frozen
`HotpotQA-32K-Aug` variant therefore keeps each selected base context, question,
and all gold answers unchanged, then appends a second real HotpotQA context as a
post-target distractor. Only the distractor tail may be truncated to fit 32,768
serialized memory-context tokens. Dataset/archive and record SHA-256 values are
frozen before generation outcomes.

### Run Order

| Run | Split | Interpretation | Priority |
|---|---|---|---|
| P6-S | first frozen case, Full KV only | operational smoke, excluded from claims | MUST |
| P6-F | four frozen cases, Full KV only | descriptive solvability/routing evidence | MUST |

The primary metric is the pinned official LongBench QA F1. Normalized exact
match and answer containment are secondary diagnostics. One or more nonzero-F1
cases is initial evidence that the path is viable; two or more is stronger
support for proposing a matched compressed pilot. These are descriptive routing
signals, not hard scientific gates. No compressed experiment starts
automatically after P6.

### Outcome

P6-S and P6-F completed successfully. On four exact-32K Full-KV cases, official
QA F1 is `0.3333/0/0.5926/0`, with mean `0.2315`, nonzero F1 on `2/4`, and
normalized gold containment on `2/4`. This supports the scoped routing judgment
that Qwen3.5-0.8B retains some usable 32K HotpotQA ability and that a paired
compressed pilot is reasonable. It does not yet provide HMO evidence.

## P7 HotpotQA-32K-Aug Equal-Byte Paired Pilot

**Protocol**: `hotpotqa_32k_paired_protocol.json`

### Claim

Test whether HMO's locality-preserving retention remains competitive on the
same four real-question 32K augmented cases that showed nonzero Full-KV
solvability. This is a small external-validity pilot, not a benchmark-level
estimate.

### Systems And Metrics

- Strictly equal measured compressed resident bytes: Contiguous CF, Global
  Fixed-Chunk Top-K, Raw Exact+Slack, and Scattered CF.
- Reference: the SHA-pinned P6 Full-KV generations, reconstructed and checked
  against the same context, query, answers, token counts, and byte accounting.
- Fixed budget: 10% middle-context cap, 256-token segments, 16-token base Sparse
  width, one protected prefix and suffix segment.
- Primary metric: official LongBench QA F1. Secondary metrics: normalized answer
  containment and normalized exact match.

### Run Order

| Run | Split | Gate | Priority |
|---|---|---|---|
| P7-S | first frozen case, four compressed arms | operational correctness only | MUST |
| P7-F | all four frozen cases | no result gate | MUST |

The smoke checks successful generation, exact compressed resident-byte equality,
parent Full-KV identity, and parseable outputs. Formal execution is descriptive;
no case filtering, tuning, or automatic follow-up is allowed.

### Outcome

P7 completed all four exact-byte cases. Mean official QA F1 for HMO / Fixed /
Raw+Slack / Scattered / Full is `0.3357 / 0.2315 / 0.3981 / 0.4038 / 0.2315`.
All systems contain gold on the same `2/4` cases. HMO beats Fixed in the formal
run by `+0.1042` F1 but trails Raw and Scattered by `0.0625/0.0682`; these
differences reflect phrasing on already-solved cases. Mean compressed footprint
is `11.556%` of Full and all four compressed arms are equal-byte in `4/4` cases.
An independent review returns `partial/supplement`, medium confidence. A
smoke/formal near-tie ranking difference weakens the Fixed comparison and should
be addressed before making a stronger real-task superiority claim.
