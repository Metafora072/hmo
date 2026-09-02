# HMO P1 Boundary-Exchange Offline Report

Date: 2026-09-03

## Decision

```text
exact boundary-exchange hypothesis: stop
fresh 8K GPU confirmation: not authorized by the frozen continue rule
```

The fixed `SAFE-in -> STRESSED-out` one-swap policy is too rarely actionable at
the actual Top-3 KV boundary. Only 2 of 12 discovery samples admitted an
exchange; one improved oracle utility and one reduced it. This fails the
pre-result requirement that strictly more than half of the samples have an
executable exchange. No threshold, exchange count, or candidate rule was
retuned after observing the result.

## Frozen Offline Protocol

- Evidence: the existing 12 Qwen3.5-0.8B 8K discovery samples, 360 segment rows.
- Baseline: raw-alpha TopK with `k=3` exact-KV slots.
- Regimes: within-sample average ranks, median threshold `0.5`.
- Donor: lowest-alpha SAFE segment inside raw-alpha TopK.
- Entrant: highest-alpha STRESSED segment outside raw-alpha TopK.
- Action: at most one equal-budget membership exchange per sample.
- Ranking after exchange: selected set first, preserving raw-alpha order within
  selected and unselected sets.
- No new GPU work, oracle labels, thresholds, weights, or signal candidates.

The continue rule was encoded before the data were evaluated:

1. strictly more than half the samples must admit an exchange;
2. overall TopK mean utility and NDCG deltas must be positive;
3. every task's TopK utility and NDCG directions must be nonnegative;
4. no confidence interval exclusion was required for this discovery screen.

## Results

| Measure | Result |
|---|---:|
| Executable exchanges | 2 / 12 |
| Positive / negative / zero exchanges | 1 / 1 / 0 |
| TopK mean utility delta | +0.000027 `[-0.000567,+0.000649]` |
| TopK sum utility delta | +0.000082 `[-0.001702,+0.001947]` |
| NDCG delta | +0.000033 `[-0.000129,+0.000228]` |
| Pairwise delta | -0.000575 `[-0.002299,+0.000575]` |

All six LongEval samples were no-ops. The two executable exchanges were both
Needle samples:

| Sample | SAFE out | STRESSED in | Oracle utility delta | NDCG delta |
|---|---:|---:|---:|---:|
| `run1:needle_0001` | 29 | 2 | -0.006809 | -0.000516 |
| `run1:needle_0003` | 29 | 26 | +0.007788 | +0.000912 |

The tiny positive pooled means arise from cancellation between these two
Needle exchanges and ten exact no-ops. They are not evidence for a deployable
controller.

## Interpretation

The previous adjacent-rank controller failed partly because most rank changes
did not cross the TopK boundary. This experiment tested the stronger action
directly and found a deeper limitation: with the fixed median regimes, the
required SAFE-inside/STRESSED-outside configuration is itself uncommon at the
budget boundary.

This result rejects the exact one-swap boundary policy. It does not erase the
discovery-level observation that high-sigma/high-delta and
high-sigma/low-delta segments have different residual utility. It shows that
the observation does not translate into useful TopK allocation through this
fixed training-free rule.

Per the stop condition, do not run fresh 8K or 16K, search thresholds, allow
multiple exchanges, or add alpha margins for this controller. The current
hand-written conditional-controller line returns to OpenChat for a scope or
problem-formulation decision.

## Reproduction

Code:

```text
experiments/phase2/e3_v2/boundary_exchange.py
experiments/test_boundary_exchange.py
```

Compact result artifact:

```text
/mnt/nvme0/hmo/runs/p1_boundary_exchange_offline_20260903/boundary_exchange.json
```

Command:

```bash
/home/pz/miniconda3/envs/hmo_research_v6/bin/python \
  -m experiments.phase2.e3_v2.boundary_exchange \
  --run-dir /mnt/nvme0/hmo/runs/p1_discovery_qwen08b_8k_20260902_192652 \
  --run-dir /mnt/nvme0/hmo/runs/p1_discovery_qwen08b_8k_seed20260903_20260902_194216 \
  --output /mnt/nvme0/hmo/runs/p1_boundary_exchange_offline_20260903/boundary_exchange.json \
  --bootstrap-samples 5000 \
  --seed 20260912
```

Focused verification: 12 tests passed before the screen. Full CPU regression
is recorded with the commit containing this report.
