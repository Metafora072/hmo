# C3 Qwen3.5-27B Execution Package

Date: 2026-09-05

## Decision

The C3 direction follows GPT's recommendation to stop method search and obtain
large-model/32K evidence, but the first paid package is reduced from roughly
1,080 generation cells to a 432-cell mandatory core. The larger matrix remains
an optional extension. This preserves the paper's target while making a
single-card paid run measurable and resumable.

## Frozen Matrix

| Block | Scope | Generation cells |
|---|---|---:|
| Preflight | one exact-32K Needle, HMO 10% + Full | 2 |
| Synthetic core | 12 Needle + 12 LongEval at exact 32K; four compressed systems x 5/10/20%; Full once/sample | 312 |
| Native core | frozen 12 HotpotQA + 12 NarrativeQA; four compressed systems at 10% + Full | 120 |

The preflight is a fit/runtime/cost calibration, not a result-quality gate. The
core has no continuation gate. Extensions require a later value/cost decision.

## Implementation

- `refine-logs/c3_27b_protocol.json` freezes the BF16 model revision, hardware
  floor, final systems/method, samples, budgets, counts, and no-filter policy.
- `experiments/phase2/e3_v2/c3_protocol.py` validates the master protocol and
  projects it into the existing final Pareto and native-QA runners.
- `run_pareto.py` remains compatible with the C2 protocol and now accepts
  protocol-defined stages, system subsets, and budgets. This makes the
  preflight exactly HMO+Full while the core uses all four compressed arms.
- `run_native_tasks.py` changes only the pinned model identity for C3 and reuses
  the exact C2 native records through the parent protocol SHA.
- Per-system, model-load, and sample-preparation times are recorded. The
  estimator `experiments/phase2/estimate_c3_cost.py` projects the mandatory
  package from the two-cell preflight and adds 25%.
- `experiments/phase2/run_c3_27b.sh` exposes only `validate`, `preflight`,
  `core-synthetic`, `core-native`, and `status`; GPU targets require one visible
  GPU, a clean commit, a model directory, and the pinned protocol.
- `experiments/C3_27B_ONE_SHOT_RUNBOOK.md` is the only current paid-run guide.
  The old V6.1 launcher is explicitly locked unless
  `HMO_ALLOW_LEGACY_PHASE2=1` is set.

## Verified Without GPU

- C3 protocol validates at SHA256
  `a5121b8d820ae49f8e584659894fee374244ce43d555a38b4f86fa13fa2097d4`.
- Both final runners accept the same master protocol and pin model revision
  `fc05daec18b0a78c049392ed2e771dde82bdf654`.
- Old C2 Pareto and native protocols still pass their original contract tests.
- Focused C2/C3 tests: 12/12 passed. Full CPU suite: 165/165 passed.

## Authority Boundary

No 27B weights were downloaded, no GPU was started, and no rental was created.
The next external action is a single 80GB-GPU preflight and requires PZ to
confirm the provider/resource and download/rental cost. After preflight, the
measured GPU-hour and price projection must be reported to PZ before the
432-cell core starts.
