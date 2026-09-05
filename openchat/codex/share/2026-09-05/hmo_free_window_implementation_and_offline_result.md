# HMO-FW implementation and zero-GPU result

## Judgment

GPT's proposal is technically actionable, but it is a method pivot rather than a small V6.1 patch. The implemented primary candidate is therefore named HMO-FW and kept separate from historical HMO results. Its precise claim is: under the frozen query-attention proxy and the same per-layer resident-byte target, exact free-start interval optimization contains the reconstructed ChunkKV plan as a feasible solution. This guarantees proxy non-degradation, not QA non-degradation.

## Implementation

- `experiments/phase2/e3_v2/free_window_allocator.py`: exact cardinality-constrained weighted interval DP, fixed short/partial ChunkKV fragment binding, per-layer HMO-FW plans, per-layer legacy-HMO placement control, and layer-specific Full-KV intervention.
- `experiments/phase2/e3_v2/analyze_free_window_offline.py`: reconstructs historical ChunkKV plans from persistent FP32 probes and compares plan geometry, retained attention mass, bytes, and selector time.
- `experiments/phase2/e3_v2/freeze_free_window_dev_protocol.py`: freezes a development subset without reading QA or proxy outcomes.
- `experiments/phase2/e3_v2/run_free_window_dev.py`: runs only the two new generation arms and reuses SHA-pinned old HMO, ChunkKV, and Full outputs.
- `experiments/phase2/run_free_window_dev.sh`: clean-commit, single-visible-GPU launcher with smoke, development, resume, and status modes.

The implementation preserves each layer's exact ChunkKV token and byte count. The recurrent state is untouched. Short final chunks and a partial next-ranked chunk are fixed; every full aligned ChunkKV chunk remains feasible as a width-10 free window. Numerical ties retain the baseline plan.

## Zero-GPU evidence

Source: `/mnt/nvme0/hmo/runs/native_six_task_9b_5348b87/formal/native_longbench_results.jsonl`, SHA256 `a84577f55911af85a8bb76b9049ce76da9482ef04edf6af314786889b94074ae`.

Output: `/mnt/nvme0/hmo/runs/free_window_dev_e9f7f35/offline_full.json`, SHA256 `04f364cc9c7dbd072deedbfb6ba6345eb2c79a9ff5356f47119fbca730a5099d`.

- 506/506 rows completed from 499 unique persistent probes in 86.52 seconds.
- Historical ChunkKV reconstructed exactly for 506/506 rows.
- Exact equal context resident bytes held for every layer and every row.
- Mean retained attention mass: old HMO `0.51312`, layer-local HMO `0.52323`, ChunkKV `0.57191`, HMO-FW `0.58244`.
- Mean proxy delta: layer-local minus old HMO `+0.01011`; HMO-FW minus ChunkKV `+0.01053`.
- HMO-FW minus ChunkKV was positive on all six dataset aggregates: NarrativeQA `+0.00939`, Qasper `+0.00854`, MultiFieldQA `+0.01092`, HotpotQA `+0.01120`, 2WikiMQA `+0.01124`, MuSiQue `+0.01252`.
- Every sample changed at least one layer; mean per-layer position Jaccard was `0.68972` against ChunkKV.
- HMO-FW touched `28.76` eligible macro regions per layer on average versus ChunkKV `27.41`, so removing the hard quota did not collapse selection to one local area in this corpus.
- Total HMO-FW selector time averaged `0.0682 s` per sample across all Full-Attention layers; this is algorithm time only, not end-to-end latency.

## Frozen GPU development package

`refine-logs/free_window_dev_protocol.json`, SHA256 `932ce1fdf8e1eb9d80c01dd13b37ba71d0dc8891efedc46cb5dd01ff39379dac`, was written before any new-method QA generation. It contains 120 development cases: 20 per task, four equal-rank context-length strata with five cases each, exact probe-identity deduplication, then seeded hash selection. The selector does not inspect old QA scores or offline proxy gains.

Only `hmo_layer_local` and `hmo_free_window` require new 9B inference. Old HMO, ChunkKV, and Full outputs are reused from the pinned 506-row result, while resident bytes are checked jointly. This is a development set, not independent confirmation. Any final claim requires one frozen method and unseen source record IDs.

## Validation and next action

All 185 repository tests pass, including brute-force checks of the exact DP, fixed-fragment feasibility, per-layer cache intervention, outcome-independent stratification, and paired summary accounting. GPU1 was idle at the pre-launch check. Next action is a two-case operational smoke on GPU1 after the clean Git commit, followed automatically by the 120-case development run if the smoke is operationally sound; no significance gate is imposed.
