# Qwen3.5-9B ChunkKV Mechanism Transfer Result

## Run identity

- Code commit: `f465657`
- Model: `Qwen/Qwen3.5-9B`
- Model revision: `c202236235762e1c871ad0ccb60c8ee5ba337b9a`
- Protocol: `refine-logs/chunkkv_mechanism_transfer_9b_protocol.json`
- Protocol SHA256:
  `ce707bd4905bb5bbbae750c0023be00a263c20dd852833d8086f96edd6dbc7ee`
- Parent frozen scale-transfer protocol SHA256:
  `b58a69a6acc192f218c24e410ff188d9f1ab8eec7f0b04f7df5c366b3f1e09f8`
- Raw run root:
  `/mnt/nvme0/hmo/runs/chunkkv_mechanism_9b_f465657/`
- Scope: the same 24 frozen Qwen3.5-9B mechanism cases used by the earlier
  8K/16K Needle and LongEval-Lines scale transfer, at the central 10% budget.

The run completed all 24 cases and 72 generation cells with finite scores.
HMO and ChunkKV have exactly equal measured post-query resident KV bytes in
24/24 cases. There was no result-dependent continuation gate.

## Main results

| System | Answer contains | Exact match | Token F1 | Mean resident KV | Mean per-case fraction of Full |
|---|---:|---:|---:|---:|---:|
| HMO / Contiguous-CF | 0.958333 | 0.916667 | 0.972222 | 51,757,056 | 13.3849% |
| ChunkKV | 0.958333 | 0.916667 | 0.972222 | 51,757,056 | 13.3849% |
| Full KV | 0.958333 | 0.875000 | 0.958333 | 397,164,544 | 100% |

For every reported quality metric, HMO versus ChunkKV is 0 wins, 24 ties and
0 losses, with a mean delta of zero at both 8K and 16K. Generated token IDs are
identical in 23/24 cases. The only textual difference is a final period in
`335 grams of saffron.`; both outputs receive exactly the same scores.

HMO and ChunkKV reduce the residual Full-Attention KV footprint by 86.615% on
the mean per-case ratio. This is residual-KV accounting, not a claim that total
process VRAM or latency falls by the same percentage.

## Breakdown

| Stage / dataset | N | HMO contains | ChunkKV contains | Full contains | HMO / ChunkKV F1 |
|---|---:|---:|---:|---:|---:|
| 8K / Needle | 6 | 6/6 | 6/6 | 6/6 | 0.944444 / 0.944444 |
| 8K / LongEval-Lines | 6 | 6/6 | 6/6 | 6/6 | 1.000000 / 1.000000 |
| 16K / Needle | 6 | 5/6 | 5/6 | 5/6 | 0.944444 / 0.944444 |
| 16K / LongEval-Lines | 6 | 6/6 | 6/6 | 6/6 | 1.000000 / 1.000000 |

The one HMO/ChunkKV exact-match and F1 advantage over Full comes from answer
formatting on a single 16K Needle case. It is not evidence of substantive
quality superiority.

## Reproducibility and runtime

The sample IDs exactly match the historical frozen 9B scale-transfer run. New
HMO and Full generated-token sequences reproduce their historical outputs in
24/24 cases for each system. This isolates the new information to the added
ChunkKV comparison rather than run drift.

The formal run took 418.40 seconds (6.97 minutes) on one RTX 5090. Peak PyTorch
allocated/reserved memory was 22.01/25.12 GB. GPU1 returned to 15 MiB after
normal exit.

## Result-to-claim verdict

- Intended HMO-over-ChunkKV mechanism claim: `no`.
- Supported: HMO is behaviorally equivalent to ChunkKV on this controlled
  central-budget suite, under exact equal resident bytes; both preserve
  near-Full quality at 13.3849% of Full residual-KV bytes.
- Not supported: an incremental HMO advantage over public structured
  contiguous retention. The earlier HMO-over-Scattered locality result remains
  valid, but continuity/locality alone no longer differentiates HMO from
  ChunkKV.
- The single formatting-only HMO-over-Full difference must not be presented as
  a quality win.

An independent internal result-to-claim reviewer recommends against using a
5%/20% sweep to rescue the superiority claim: 5% is likely to favor
concentration, while 20% is likely to saturate. Such a sweep would be useful
only as an explicitly negative/equivalence curve, not as the next priority.

## Route

Do not escalate this claim to A100/27B. Narrow the paper contribution to the
Hybrid residual-KV formulation, exact byte accounting, competitive near-Full
retention, and the established locality benefit over scattered singleton
retention. The next useful work is a zero- or low-GPU failure/geometry analysis
or honest efficiency accounting; any redesigned policy needs a newly frozen
confirmation set.
