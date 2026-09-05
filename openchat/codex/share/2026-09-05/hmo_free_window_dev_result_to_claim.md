# HMO-FW development result and claim routing

## Run identity

- Code: `main@f2b7404`
- Protocol: `refine-logs/free_window_dev_protocol.json`, SHA256 `932ce1fdf8e1eb9d80c01dd13b37ba71d0dc8891efedc46cb5dd01ff39379dac`
- Results: `/mnt/nvme0/hmo/runs/free_window_9b_v1/formal/free_window_dev_results.jsonl`, SHA256 `f02e74681b48a47b7be0570f2ffeab93c45b552d367030c7532fe51bb1243a31`
- Summary: `/mnt/nvme0/hmo/runs/free_window_9b_v1/formal/free_window_dev_summary.json`, SHA256 `f8ffd882bb9a8d7daa08f3c9a114162bc7390004fed14f42be18846859274113`
- Manifest: SHA256 `68aa7ba7767877b9a2f6b194e79ba1ddc214356a71e35c6c3aed741183ca445d`
- Runtime: 120/120 cases, 1391.89 seconds on one RTX 5090; peak allocated/reserved GPU memory 22.05/24.93 GB; GPU1 released after exit.
- Integrity: 120/120 cases have exactly equal post-query resident bytes across old HMO, layer-local HMO, ChunkKV, and HMO-FW. Mean compressed resident fraction of Full is `14.2971%`.

## Raw development table

| System | QA-F1 | Answer contains | Exact match |
|---|---:|---:|---:|
| Old HMO | 0.43185 | 0.33333 | 0.27500 |
| Layer-local HMO | **0.44210** | 0.32500 | 0.27500 |
| ChunkKV | 0.43098 | 0.32500 | 0.27500 |
| HMO-FW | 0.43420 | 0.33333 | **0.28333** |
| Full KV | 0.44249 | 0.33333 | 0.27500 |

| Comparison | Mean F1 delta | Wins / ties / losses | Descriptive paired bootstrap 95% interval |
|---|---:|---:|---:|
| HMO-FW - ChunkKV | +0.00322 | 11 / 99 / 10 | [-0.01852, +0.02791] |
| HMO-FW - old HMO | +0.00236 | 17 / 84 / 19 | [-0.04146, +0.04670] |
| Layer-local - old HMO | +0.01026 | 14 / 97 / 9 | [-0.02565, +0.05051] |
| Layer-local - ChunkKV | +0.01112 | 21 / 87 / 12 | [-0.03199, +0.05460] |
| Layer-local - Full KV | -0.00039 | 17 / 88 / 15 | [-0.04443, +0.04478] |

The intervals are descriptive uncertainty summaries, not continuation gates.

## Task heterogeneity

| Dataset | HMO-FW - ChunkKV | Layer-local - ChunkKV |
|---|---:|---:|
| NarrativeQA | +0.00978 | +0.03425 |
| Qasper | +0.06710 | +0.06049 |
| MultiFieldQA | -0.02094 | +0.00699 |
| HotpotQA | +0.00456 | +0.01875 |
| 2WikiMQA | 0.00000 | -0.01500 |
| MuSiQue | -0.04115 | -0.03875 |

HMO-FW and ChunkKV generated identical token sequences on 88/120 cases. Layer-local and old HMO were also identical on 88/120; layer-local and ChunkKV were identical on 73/120.

## Diagnosis

1. The exact proxy result is real but not a sufficient end-task objective. On the selected cases, HMO-FW's proxy gain over ChunkKV is positive for every sample, yet its correlation with QA-F1 delta is Pearson `-0.218` and Spearman `-0.077`.
2. HMO-FW losses have slightly larger proxy gains than its wins (`0.01139` versus `0.01054`) and more macro-boundary crossings per layer (`4.41` versus `3.05`). This is consistent with the free relaxation selecting high-mass windows that do not preserve a task's distributed evidence structure.
3. Layer-local HMO is the stronger design candidate. It keeps V6.1's macro-region actions, coverage, Exact retention, and byte allocation, while allowing each Full-Attention layer to place its Sparse window independently. Thus it fixes the unfair shared-placement mismatch against a per-layer ChunkKV baseline without abandoning the original problem formulation.
4. HMO-FW remains valuable as a diagnostic relaxation and ablation. It has a stronger attention-mass guarantee but lower QA-F1 than layer-local by `0.00790`, supporting the paper-relevant statement that attention-mass maximization alone is insufficient and structured macro coverage can matter.

## Claim verdict

Independent internal result-to-claim review: `partial`, confidence `medium`.

Supported:

- exact per-layer equal-byte execution;
- exact non-degradation of the frozen retained-attention proxy relative to reconstructed ChunkKV;
- a small, task-heterogeneous HMO-FW development signal;
- a more promising layer-local correction that preserves the original HMO design line.

Not supported:

- general HMO-FW end-task superiority over ChunkKV or old HMO;
- treating retained query-attention mass as a monotone QA surrogate;
- spending A100 budget on the current HMO-FW candidate.

## Recommended route

Promote layer-local HMO, not HMO-FW, to the final candidate. The working story becomes: global orchestration decides *which semantic macro regions and fidelity levels* survive, while layer-local execution decides *where each Full-Attention layer reads within a Sparse region*. The old shared placement is a restricted feasible solution, so the same per-layer proxy non-degradation argument applies without discarding coverage.

Before further GPU use, PZ and GPT should review this candidate switch. If accepted, freeze it once and run a fresh-record confirmation on the four LongBench tasks that still have unused eligible source IDs. A practical package is 20 unseen records per task, with old HMO, layer-local HMO, ChunkKV, and Full KV generated under exact equal bytes. Based on this run, the estimate is roughly `0.6-0.9 RTX-5090 GPU-hours`. NarrativeQA and MuSiQue have no unused eligible IDs under the current <=16K protocol and should remain development evidence. No A100 action is recommended before that confirmation.
