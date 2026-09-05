# Qwen3.5-9B Six-Task Native LongBench Result

## Run identity

- Code commit: `5348b87`
- Model: `Qwen/Qwen3.5-9B`
- Model revision: `c202236235762e1c871ad0ccb60c8ee5ba337b9a`
- Protocol: `refine-logs/native_longbench_six_task_9b_protocol.json`
- Protocol SHA256:
  `2b6c90154e42a543e1bc3ea534e5c81d84c4e7524a82192d8d59bbe29585464e`
- Raw run root:
  `/mnt/nvme0/hmo/runs/native_six_task_9b_5348b87/`
- Scope: 506 native, unaugmented, untruncated LongBench QA records with exact
  serialized memory context at or below 16K tokens.
- Tasks: NarrativeQA 61, Qasper 100, MultiFieldQA-en 100, HotpotQA 100,
  2WikiMultihopQA 100, MuSiQue 45.

The run completed all 506 unique sample IDs. All scores are finite. The four
compressed systems have exactly equal measured post-query resident KV bytes in
506/506 cases.

## Main results

| System | Official QA-F1 | Answer contains | Exact match | Mean resident KV | Mean per-case fraction of Full |
|---|---:|---:|---:|---:|---:|
| HMO / Contiguous-CF | 0.464177 | 0.328063 | 0.264822 | 47,098,560 | 14.465% |
| ChunkKV | 0.479314 | 0.335968 | 0.282609 | 47,098,560 | 14.465% |
| Global Fixed-Chunk Top-K | 0.476590 | 0.335968 | 0.282609 | 47,098,560 | 14.465% |
| Raw Attention Exact+Slack | 0.479974 | 0.335968 | 0.278656 | 47,098,560 | 14.465% |
| Full KV | 0.481461 | 0.339921 | 0.280632 | 338,657,215 | 100% |

HMO retains 96.41% of Full KV's official QA-F1 while reducing the measured
residual Full-Attention KV footprint by 85.535%. This is a residual-KV result,
not a claim that total process VRAM falls by the same fraction.

The prefix50 and prefix100 phases took 8,699.50 and 5,265.26 seconds,
respectively, for 3.88 GPU-hours in total. Peak PyTorch allocated/reserved
memory was 20.54/23.21 GiB. GPU1 returned to its idle 15 MiB state after exit.

## Per-task official QA-F1

| Dataset | N | HMO | ChunkKV | Global Fixed | Raw+Slack | Full KV |
|---|---:|---:|---:|---:|---:|---:|
| NarrativeQA | 61 | 0.306974 | 0.307254 | **0.324394** | 0.316424 | 0.313994 |
| Qasper | 100 | 0.423415 | 0.453607 | 0.450143 | 0.453685 | **0.459275** |
| MultiFieldQA-en | 100 | 0.508126 | 0.522817 | 0.518299 | 0.515311 | **0.527614** |
| HotpotQA | 100 | 0.587311 | **0.611382** | 0.602413 | 0.606438 | 0.597291 |
| 2WikiMultihopQA | 100 | 0.491699 | 0.501652 | 0.493366 | 0.475214 | **0.531866** |
| MuSiQue | 45 | 0.335406 | 0.329877 | 0.332099 | **0.411111** | 0.285802 |
| Task macro mean | 6 tasks | 0.442155 | 0.454431 | 0.453452 | **0.463031** | 0.452640 |

HMO leads ChunkKV, Global Fixed, and Full KV on MuSiQue, but not Raw+Slack. It
does not lead on the other five tasks. NarrativeQA is effectively tied with
ChunkKV, while Qasper is the largest consistent HMO deficit.

## Paired uncertainty

The following intervals resample the 506 paired cases 10,000 times with seed
20260905. They quantify case-level uncertainty only; the experiment has one
model revision and is not a multi-seed model-level study.

| Comparison | HMO mean delta | 95% bootstrap interval | HMO W/T/L | Exact sign p |
|---|---:|---:|---:|---:|
| HMO - ChunkKV | -0.015136 | [-0.035097, +0.004213] | 62/372/72 | 0.4370 |
| HMO - Global Fixed | -0.012413 | [-0.031949, +0.006708] | 52/382/72 | 0.0876 |
| HMO - Raw+Slack | -0.015796 | [-0.037553, +0.005536] | 65/371/70 | 0.7308 |
| HMO - Full KV | -0.017284 | [-0.036931, +0.001963] | 61/372/73 | 0.3420 |

The point estimates do not support algorithmic superiority. The paired
intervals also do not establish that HMO is reliably worse; most samples tie
and a minority of answer flips determine the means. This supports a
competitive-efficiency statement, not a best-method statement.

## Structural diagnosis

The persistent FP32 probe permits a no-GPU analysis of what the policies retain:

- HMO retains 70.90% of aggregate query-attention mass on average.
- ChunkKV retains 79.02% of its layer-local query-attention mass on average.
- Global Fixed and Raw+Slack retain 75.60% and 74.63% of aggregate mass.
- HMO touches 38.62 middle macro-segments on average; ChunkKV touches 27.41.
- Mean HMO/ChunkKV retained-position Jaccard is 0.408 across Full-Attention
  layers.
- HMO output tokens exactly equal ChunkKV on 315/506 cases and Full KV on
  312/506 cases. ChunkKV and Global Fixed match Full output on 337/506 and
  339/506 cases.

The observed trade-off is therefore concrete: HMO spends capacity on broad
macro-region coverage and gives up roughly eight points of retained observation
attention relative to layer-local ChunkKV. Broad native QA favors the more
concentrated policy on average. This does not refute the span-survival theorem;
it shows that the theorem is not a downstream optimality guarantee.

A post-hoc context split gives HMO - ChunkKV deltas of -0.0279 at <=8K,
-0.0067 at 8--12K, -0.0484 at 12--14K, and +0.0154 above 14K. The apparent
longest-context reversal is heterogeneous after conditioning on task, so it is
a hypothesis for a frozen follow-up, not a confirmed regime law.

## Result-to-claim verdict

- Verdict: `partial`, medium confidence.
- Original broad-native-superiority claim: `no`.
- Supported: Hybrid residual-memory formulation, exact byte accounting,
  operational feasibility, and near-Full quality at 14.465% residual-KV bytes.
- Still supported by earlier controlled evidence: locality versus scattered
  singleton retention and a limited budget/length working regime.
- Unsupported: HMO superiority over ChunkKV or simple structured retention on
  broad native QA, universal free-start benefit, cross-family generality, and
  total-VRAM or throughput gains.

The claim should be narrowed to a stratified residual-KV overlay with strong
byte-grounded compression and controlled mechanism evidence. The six-task
table must be reported as competitive/diagnostic evidence, not presented as a
win.

## Next decision

Do not spend A100 hours to repeat the current broad-superiority hypothesis at
27B. The minimum useful 5090 follow-up is:

1. add the now-validated ChunkKV adapter to the existing frozen 9B
   Needle/LongEval 8K/16K mechanism suite at 10%;
2. only if that preserves a positive mechanism slice, run a small predeclared
   native 5%/20% budget sweep to test whether 10% is the wrong operating point;
3. treat any layer-wise or coverage/focus redesign as development, then confirm
   it on records not selected by the observed outcomes.

This is a research routing decision, not a strict significance gate.
