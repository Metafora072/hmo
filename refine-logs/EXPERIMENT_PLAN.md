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
| P3-S | 1+1 at 2K, seed 20261000 | complete; exact byte equality; no protocol errors | MUST |
| P3-8K | 12+12 at 8K, seed 20261001 | no overall/Needle regression and at least 2 LongEval membership changes | MUST |
| P3-16K | 12+12 at 16K, seed 20261002 | run only after P3-8K gate | MUST |

No controller, threshold, sample, or metric changes are allowed after smoke.
