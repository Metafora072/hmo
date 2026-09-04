# HMO Package B: Structured-Baseline Pareto

Date: 2026-09-04

## Protocol

- Model: Qwen3.5-0.8B, revision
  `2fc06364715b967f1860aea9cf38778875588b17`.
- Matched suite: 48 cases, with 12 Needle and 12 LongEval-Lines cases at each
  of 8K and 16K.
- Middle-context caps: 5%, 10%, and 20%.
- Frozen primary metric: normalized answer containment.
- Equal-byte systems: Contiguous CF, Global Fixed-Chunk Top-K, Raw
  Exact+Slack, Scattered CF, and Contiguous Sparse-only.
- Reference: Full KV, generated once per sample.
- Protocol:
  `refine-logs/contiguous_cf_pareto_protocol.json`.

Global Fixed-Chunk Top-K partitions every eligible 256-token segment into
aligned non-overlapping 16-token chunks and globally ranks their summed
query-attention mass. If the exact target has a remainder, the final tokens are
the fixed-boundary prefix of the next ranked chunk. It uses the same query probe,
anchors, query KV, and measured resident-byte target as HMO.

## Artifacts And Integrity

- Formal run:
  `/mnt/nvme0/hmo/runs/contiguous_cf_pareto_formal_20260904_1518/`
- Formal JSONL SHA256:
  `5757ff898b921c1b0fcc6ed1e76d195667070999d56bf71189a752e38d49e1ab`
- Formal summary SHA256:
  `19a36c732649061cdfae28574992a9bf2df15a63aec2b4bc7e3c5ce009306853`
- Smoke run:
  `/mnt/nvme0/hmo/runs/contiguous_cf_pareto_smoke_20260904_1515/`
- Smoke summary SHA256:
  `1d69898845705344578cc762f8d5be13b44721f393f4250d3119b7f737bbd9e5`

The smoke completed all three budget points. The formal run completed 144/144
budget cases. All five compressed systems had exactly equal measured resident
KV in 48/48 cases at every budget. The 10% Contiguous, Scattered, Sparse-only,
and Full rows reproduce the earlier frozen run's generated token IDs, primary
metrics, and resident bytes in 48/48 cases.

Runtime was 1561.99 seconds on one RTX 5090. Peak CUDA memory was 4.35 GiB
allocated and 5.05 GiB reserved. GPU1 returned to 15 MiB after completion.

## Primary Results

Values are correct cases out of 48. Footprint is the mean per-case measured
post-query resident KV fraction relative to Full KV.

| Middle cap | Footprint | HMO | Fixed chunk | Raw+Slack | Scattered | Sparse-only | Full |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5% | 8.57% | 30 | **36** | 30 | 21 | 30 | 35 |
| 10% | 13.38% | 34 | **36** | 32 | 27 | 32 | 35 |
| 20% | 23.01% | 35 | 35 | 35 | **36** | 33 | 35 |

### Paired HMO Comparisons

| Cap | Comparator | Delta | Wins / ties / losses | Two-sided exact sign p |
|---:|---|---:|---:|---:|
| 5% | Fixed chunk | -12.50 pp | 3 / 36 / 9 | 0.1460 |
| 5% | Raw+Slack | 0.00 pp | 6 / 36 / 6 | 1.0000 |
| 5% | Scattered | **+18.75 pp** | **9 / 39 / 0** | **0.0039** |
| 10% | Fixed chunk | -4.17 pp | 2 / 42 / 4 | 0.6875 |
| 10% | Raw+Slack | +4.17 pp | 3 / 44 / 1 | 0.6250 |
| 10% | Scattered | **+14.58 pp** | **7 / 41 / 0** | **0.0156** |
| 20% | Fixed chunk | 0.00 pp | 0 / 48 / 0 | 1.0000 |
| 20% | Raw+Slack | 0.00 pp | 0 / 48 / 0 | 1.0000 |
| 20% | Scattered | -2.08 pp | 0 / 47 / 1 | 1.0000 |

### Length Interaction

| Cap | 8K HMO / Fixed | 16K HMO / Fixed | HMO-Fixed delta at 8K / 16K |
|---:|---:|---:|---:|
| 5% | 15 / 20 | 15 / 16 | -20.83 / -4.17 pp |
| 10% | 16 / 20 | **18 / 16** | -16.67 / **+8.33 pp** |
| 20% | 19 / 19 | 16 / 16 | 0.00 / 0.00 pp |

At 10% on 16K LongEval-Lines, HMO obtains 8/12, compared with 6/12 for Fixed
chunk, Raw+Slack, and Full, 5/12 for Sparse-only, and 4/12 for Scattered. At 8K
LongEval-Lines, however, HMO is 5/12 while Fixed and Full are 9/12.

## Frozen Format-Robust Secondary

The existing deterministic clock-format alias changes only one underlying
Needle case. It does not alter the qualitative result.

| Middle cap | HMO | Fixed chunk | Raw+Slack | Scattered | Sparse-only | Full |
|---:|---:|---:|---:|---:|---:|---:|
| 5% | 30 | **36** | 31 | 22 | 30 | **36** |
| 10% | 34 | **36** | 33 | 28 | 32 | **36** |
| 20% | 36 | 36 | 36 | **37** | 34 | 36 |

## Findings

1. **Contiguous structure beats scattered singleton retention at tight and
   medium budgets.** HMO has zero losses against Scattered at 5% and 10%, with
   +18.75 and +14.58 pp primary gains under exact equal bytes.
2. **Generic fixed chunks are a genuinely strong baseline.** HMO does not
   dominate them: Fixed wins at 5%, remains slightly ahead overall at 10%, and
   ties HMO at 20%. Free-start locality must not be presented as universally
   better than fixed chunking.
3. **The 10% result has a meaningful length interaction.** HMO loses to Fixed
   at 8K but wins by +8.33 pp at 16K, entirely through LongEval. This supports a
   scoped long-context coverage interpretation, not a global superiority claim.
4. **There is a budget phase transition.** At 5%, below the approximately 6.25%
   `w/L` full-coverage floor, HMO and Sparse-only are identical in 48/48 cases
   because no Exact upgrades occur. At 10%, every eligible segment receives a
   local window and HMO starts to separate at 16K. At 20%, the structured
   methods largely saturate and HMO, Fixed, Raw, and Full tie on the primary
   metric.
5. **The retained geometries are not accidentally equivalent.** Mean active
   position Jaccard between HMO and Fixed is 0.351, 0.336, and 0.485 at
   5%, 10%, and 20%, respectively.

## Claim Decision

An independent result-to-claim review returned `partial` support, routed the
work to `supplement`, and assigned high confidence to that assessment.

The evidence strongly supports the narrow mechanism claim that contiguous
structured retention is safer than scattered token retention at 5-10% caps.
It partially supports the broader HMO claim: the combined stratified/free-start
overlay improves over global fixed chunks at 16K and 10%, but the reverse holds
at 8K and tighter budgets. The current experiment does not isolate free-start
placement from stratified allocation.

The cleanest paper story is therefore a **budget- and length-dependent memory
organization principle**: global chunk concentration is effective in
shorter or extremely tight regimes; once a long context can afford broad
coverage, HMO's stratified local overlay becomes competitive and can improve
relational retrieval. This is a stronger and more informative story than
claiming unconditional superiority.

## Highest-Value Follow-Up

Before a costly larger-model run, add one low-cost 16K/10% structured control:
keep HMO's identical macro-segment allocation and Exact upgrades, but restrict
each Sparse segment to its best aligned 16-token subchunk. This
`Stratified Fixed-Chunk` arm isolates free-start placement from stratification.
After that, the planned realistic-task transfer remains the main external
validity experiment.
