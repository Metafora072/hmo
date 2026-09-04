# Package B Pareto Results And Claim Review

Date: 2026-09-04

## What Was Completed

Package B added Global Fixed-Chunk Top-K and ran the frozen Qwen3.5-0.8B
5%/10%/20% Pareto on the matched 48-case 8K/16K Needle and LongEval-Lines
suite. The formal run completed all 144 sample-budget cases. Contiguous CF,
Fixed Chunk, Raw Exact+Slack, Scattered CF, and Sparse-only used exactly equal
measured resident KV bytes in every case and at every budget.

Formal artifacts:

- Report: `experiments/results/PARETO_PACKAGE_B_20260904.md`
- Protocol: `refine-logs/contiguous_cf_pareto_protocol.json`
- Raw run:
  `/mnt/nvme0/hmo/runs/contiguous_cf_pareto_formal_20260904_1518/`
- JSONL SHA256:
  `5757ff898b921c1b0fcc6ed1e76d195667070999d56bf71189a752e38d49e1ab`
- Summary SHA256:
  `19a36c732649061cdfae28574992a9bf2df15a63aec2b4bc7e3c5ce009306853`

Runtime was 26.0 minutes on GPU1. Peak CUDA memory was 4.35 GiB allocated and
5.05 GiB reserved. GPU1 returned to 15 MiB after completion.

## Main Result

| Cap | Footprint | HMO | Fixed | Raw+Slack | Scattered | Sparse | Full |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5% | 8.57% | 30/48 | **36/48** | 30/48 | 21/48 | 30/48 | 35/48 |
| 10% | 13.38% | 34/48 | **36/48** | 32/48 | 27/48 | 32/48 | 35/48 |
| 20% | 23.01% | 35/48 | 35/48 | 35/48 | **36/48** | 33/48 | 35/48 |

HMO versus Scattered is the clean result: +18.75 pp at 5% with 9 wins and no
losses, and +14.58 pp at 10% with 7 wins and no losses. At 20%, the methods
largely saturate and Scattered has one additional correct case.

Global Fixed-Chunk is stronger than expected. It wins overall at 5% and 10%
and ties HMO at 20%. The interaction with length is informative: at 10%, HMO
loses to Fixed 16/24 versus 20/24 at 8K, but wins 18/24 versus 16/24 at 16K.
The 16K difference comes from LongEval, where HMO is 8/12 and Fixed is 6/12.

## Mechanism Interpretation

- At 5%, the budget is below the approximately 6.25% full-segment coverage
  floor. No Exact upgrades occur and HMO equals Sparse-only in 48/48 cases.
- At 10%, all eligible macro-segments can receive a local window. HMO begins
  separating from Raw/Sparse and beats Fixed on the 16K slice.
- At 20%, HMO, Fixed, Raw, and Full all obtain 35/48, indicating saturation on
  this suite.
- HMO and Fixed are genuinely different geometries: their mean retained-position
  Jaccard is 0.351, 0.336, and 0.485 at 5%, 10%, and 20%.

The resulting story is budget- and length-dependent memory organization. A
global fixed-chunk policy is effective for short contexts and extremely tight
budgets. Once a long context crosses the coverage floor, a stratified local
overlay becomes competitive and can better preserve relational evidence.

## Independent Result-To-Claim Review

Verdict: `partial`; route: `supplement`; confidence: high.

Supported:

- Contiguous structured retention is substantially safer than scattered
  singleton retention at tight and medium budgets.
- The coverage floor provides a coherent explanation for the transition from
  5% to 10%.
- HMO reaches near-Full quality at 13.38% footprint and ties Full at 23.01%.

Not supported:

- Unconditional superiority over Global Fixed-Chunk Top-K.
- A broad claim that HMO is especially effective at all long-context settings.
- A causal free-start advantage, because the current HMO/Fixed comparison
  changes both macro allocation and micro-window placement.

## Proposed Next Action

The highest-value next experiment is a low-cost 0.8B 16K/10% control named
`Stratified Fixed-Chunk`. It must reuse HMO's macro-segment allocation, Exact
upgrades, anchors, probe, and resident-byte target, while replacing each
free-start sparse window with the best aligned 16-token subchunk. This directly
isolates free-start placement from stratification.

No additional GPU job has been started. After this mechanism control, the next
external-validity step remains a realistic HotpotQA transfer. Both actions await
PZ confirmation.
