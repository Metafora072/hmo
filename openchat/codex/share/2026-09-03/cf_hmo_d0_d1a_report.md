# CF-HMO D0 Diagnosis And D1a Allocator Report

## Status And Scope

- D1a pure allocator: implemented and CPU-tested.
- D0: complete on physical GPU1.
- Scope: development-only diagnosis on the ten membership-changed P3 LongEval
  cases. These results are not fresh confirmation evidence.
- Code commit: `d8ca4bd322c2e062b9519bb98b3f6f9c66594b60`.
- D0 manifest: `97b96eac58ffd506ff1d3c214634e19d84568a76272b0c6da3ec8263ea776aa8`.
- Result:
  `/mnt/nvme0/hmo/runs/d0_cf_diagnosis_s20261003_20260903_2045/cf_diagnosis_summary.json`.
- Log:
  `/mnt/nvme0/hmo/logs/d0_cf_diagnosis_s20261003_20260903_2045.log`.

## D1a Implementation

`experiments/phase2/e3_v2/coverage_fidelity.py` implements a deterministic,
model-independent byte allocator:

1. Protected boundary segments are Exact and charged separately.
2. Sparse coverage is allocated before any Exact upgrade. When the full floor
   does not fit, coverage uses rank-normalized attention demand per byte.
3. Covered segments are upgraded by attention demand times recurrent
   accessibility deficit per incremental byte.
4. No-access replaces only the deficit with a constant.
5. Remaining segment-granularity slack is spent as deterministic Sparse token
   slots until less than one common token cost remains.

The allocator records both cap and realized bytes and requires one eligible
segment size and one per-token KV cost. The frozen P3 runner was not edited.

Six allocator tests cover protection, coverage precedence, insufficient-floor
selection, no-access isolation with equal realized bytes, deterministic ties,
residual spending, and invalid inputs. Four D0 tests cover the shared K/V-norm
primitive, query-attention positions, survival semantics, and aggregation. The
complete `experiments/test_*.py` suite passed: 115/115.

## Raw D0 Table

`outcome` is the P3 generated-answer V2 minus raw result. `gold_delta` is V2
minus raw mean teacher-forced gold-answer log probability. Sparse cells show
retained answer tokens / total answer tokens.

| length | case | outcome | gold_delta | raw has full answer segment | V2 has full answer segment | KV-8 | KV-16 | Query-8 | Query-16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16K | 0000 | 0 | +0.0654 | yes | yes | 1/4 | 1/4 | 2/4 | 2/4 |
| 16K | 0002 | -1 | -5.6056 | yes | no | 1/5 | 2/5 | 1/5 | 2/5 |
| 16K | 0003 | 0 | +0.0155 | yes | yes | 0/4 | 0/4 | 1/4 | 2/4 |
| 16K | 0008 | -1 | -1.2457 | yes | no | 0/6 | 1/6 | 0/6 | 3/6 |
| 16K | 0009 | 0 | -0.0835 | yes | yes | 0/3 | 1/3 | 2/3 | 2/3 |
| 8K | 0001 | +1 | +4.0458 | no | yes | 1/5 | 2/5 | 2/5 | 2/5 |
| 8K | 0003 | 0 | -0.0655 | yes | yes | 0/5 | 0/5 | 2/5 | 2/5 |
| 8K | 0004 | +1 | +5.5312 | no | yes | 0/4 | 1/4 | 2/4 | 3/4 |
| 8K | 0007 | -1 | -4.1642 | yes | no | 0/5 | 1/5 | 2/5 | 3/5 |
| 8K | 0008 | 0 | -0.0197 | yes | yes | 0/5 | 1/5 | 1/5 | 1/5 |

All ten recomputed raw/V2 memberships, slot counts, byte limits, and gates
exactly matched the frozen P3 artifacts. Raw and V2 post-query resident bytes
were equal in every case.

## Findings

### 1. Coverage explains the outcome changes

The teacher-forced metric agrees in sign with all five non-tie generation
changes. The two V2 wins average `+4.7885` gold logprob; the three losses average
`-3.6718`; the five generated-answer ties average only `-0.0176`. This is much
stronger evidence for a coverage failure than the earlier binary answer check
alone.

Implication: P3 rejects independent hard Top-K selection because it can replace
an answer-bearing segment with a high-need but task-irrelevant segment. It does
not reject recurrent accessibility as a fidelity-upgrade signal.

### 2. Token query-attention dominates K/V norm for Sparse selection

| selector | width | cases retaining any answer token | cases retaining all answer tokens | mean answer-token retention |
|---|---:|---:|---:|---:|
| K/V norm | 8 | 3/10 | 0/10 | 6.5% |
| K/V norm | 16 | 8/10 | 0/10 | 22.0% |
| query attention | 8 | 9/10 | 0/10 | 35.2% |
| query attention | 16 | 10/10 | 0/10 | 49.2% |

Query-attention/16 strictly leads the tested candidates on both any-hit and
mean-retention diagnostics. Select token query-attention as the Sparse
primitive and use width 16 as the first D1b development setting. K/V norm is
rejected for this role.

### 3. Sparse survival alone is not sufficient evidence

No tested candidate retained every tokenizer piece of a six-character answer
in any case. Query-attention often retained the label neighborhood plus two or
three answer pieces, which may still complement the recurrent state, but D0
cannot establish end-task recovery from token overlap alone.

Implication: do not add a new hand-designed span heuristic from these ten
labels. The next evidence should be actual cache intervention and generation.

## Resource Record

- Runtime: 121.27 seconds.
- GPU: physical GPU1, NVIDIA GeForce RTX 5090.
- Peak allocated: 4,640,176,640 bytes.
- Peak reserved: 5,532,286,976 bytes.
- After exit: GPU1 returned to 15 MiB and the detached screen disappeared.

## Recommended Next Action

Proceed to D1b as a separate implementation bundle:

1. add a new coverage-fidelity cache intervention and runner without modifying
   frozen `run_end_task.py`;
2. use corrected token query-attention with width 16 first;
3. compare CF-HMO, no-access, Sparse-only, raw-alpha Exact Top-K, and Full-KV
   with measured resident bytes;
4. begin with the ten development changed cases, where outcome mechanisms are
   known, before freezing any fresh 8K/16K protocol;
5. retain width 8 only as the predeclared lower-cost alternative if the first
   actual-cache development run shows that width 16 leaves too few Exact
   upgrades.

D0 has selected the primitive, so another diagnostic gate or GPT/Opus review is
not needed before implementation. Final width and budget remain development
choices until actual cache quality is observed.
