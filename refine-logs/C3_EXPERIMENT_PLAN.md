# C3 Experiment Plan: Qwen3.5-27B One-Shot Validation

Date: 2026-09-05

This file supersedes the pre-C2 planning sections in `EXPERIMENT_PLAN.md` for
all future large-model execution. The machine-readable source of truth is
`c3_27b_protocol.json`.

## Claims

1. At exactly matched resident KV bytes, HMO's query-guided contiguous overlay
   transfers to Qwen3.5-27B at 32K and improves the quality-memory frontier over
   fixed, raw, and scattered retention.
2. Without task-specific tuning, the same frozen policy remains competitive on
   native HotpotQA and NarrativeQA at roughly one tenth of Full Attention KV.

The package does not claim universal superiority, cross-architecture transfer,
or that recurrent accessibility is a proven task-optimal allocation signal.

## Must Run

| Block | Cases | Systems and budgets | Generation cells | Purpose |
|---|---:|---|---:|---|
| PZ cost preflight | 1 Needle at exact 32K | HMO + Full, 10% | 2 | Measure fit, memory, runtime, and resumability |
| C3-S | 12 Needle + 12 LongEval at exact 32K | HMO, Fixed, Raw+Slack, Scattered at 5/10/20%; Full once | 312 | Large-model mechanism and Pareto evidence |
| C3-N | Frozen 12 HotpotQA + 12 NarrativeQA native cases | Same four compressed arms at 10% + Full | 120 | Task breadth and cross-scale external validity |

Mandatory core after preflight: **432 generation cells**. C3-S emits 72 result
rows because one Full generation is shared across the three budget rows. C3-N
emits 24 result rows.

The preflight is an operational and cost-calibration step, not a result-quality
gate. An OOM routes execution to another 80GB backend or an infrastructure fix;
it does not authorize changing the method, budget, samples, or BF16 target.

## Nice To Have

Only consider these after the mandatory core is complete and reviewed:

| Extension | Additional generation cells |
|---|---:|
| Increase synthetic tasks from 12 to 30 samples each | 468 |
| Add native 20% compressed arms, reusing Full | 96 |
| Four HotpotQA-32K-Aug central cases | 20 |
| Small 64K mechanism stress set | 40 |

No extension launches automatically. It requires a new value/cost decision.

## Fixed Contract

- Model: `Qwen/Qwen3.5-27B` BF16 at revision
  `fc05daec18b0a78c049392ed2e771dde82bdf654`.
- One visible 80GB GPU; at least 120 GiB persistent free space.
- Final HMO method, FP32 persistent query probe, exact measured resident-byte
  equality, greedy decoding, and all baseline definitions are unchanged.
- Native cases are reused exactly from `native_longbench_protocol.json`, SHA256
  `86ebfa5cfdff0613e559780811887b7537d0485cbd00534193c0aac433b49e2a`.
- No outcome-conditioned filtering, automatic continuation gate, or live method
  tuning during the paid run.

## Execution Decision

Run the two-cell preflight first. Feed its summary to
`experiments/phase2/estimate_c3_cost.py`, enter the provider's current hourly
price, and ask PZ to approve the projected cost before C3-S/C3-N. The estimator
uses measured model-load, sample-preparation, HMO-generation, and Full-generation
times, scales NarrativeQA's 128-token allowance conservatively, and adds 25%.

Operational commands, storage layout, detached execution, resumption, and
verification are defined only in `experiments/C3_27B_ONE_SHOT_RUNBOOK.md`.
