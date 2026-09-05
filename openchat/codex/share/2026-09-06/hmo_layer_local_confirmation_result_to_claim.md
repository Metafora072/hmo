# Layer-local HMO fresh confirmation and claim routing

## Run identity

- Frozen code: `main@c21572c`
- Protocol: `refine-logs/layer_local_confirmation_protocol.json`, SHA256 `2076134355a3ca20941ca1504c68ddfb9e20494a418b810bb85ff1ef7b2cdd46`
- Formal results: `/mnt/nvme0/hmo/runs/layer_local_confirmation_9b_v1/formal/layer_local_confirmation_results.jsonl`, SHA256 `1ba609b86b0ae8267ed72b28a95bb1593dbab1e9a4130f9b23730ed2f816d2c1`
- Formal summary: `/mnt/nvme0/hmo/runs/layer_local_confirmation_9b_v1/formal/layer_local_confirmation_summary.json`, SHA256 `4b87dd185f79ce2d22c1dcc4351cb09d05e6fb81aa28208ac860636ef22d0b6d`
- Run manifest: SHA256 `35cefa8af77e5da3f4e1a4f1db6c8e0a56e9304c26af577a096e61cff318f75e`, manifest ID `ee9b5abe926cdfe9265759abb46f057ce273e0444b5c81a20ac850a2fde4fba7`
- Hardware/runtime: one RTX 5090 (GPU1), one-case smoke `123.89 s`, formal 80-case package `1489.41 s`; total `0.448 GPU-hours`. Formal peak allocated/reserved memory was `18.89/20.33 GiB`. GPU1 returned to 15 MiB after exit.
- Storage: complete smoke/formal logs, JSONL, summaries, manifests, and 80 identity-bound FP32 probes occupy about `16 MiB` under `/mnt/nvme0`.

## Frozen scope

This is a fresh-record, shorter-context transfer confirmation. It excludes every record ID in the previous 506-row six-task run, excludes prompt identities used by that run, and deduplicates the remaining identities. Qasper, MultiFieldQA-en, HotpotQA, and 2WikiMQA each contribute 20 cases, with five frozen cases from each of four equal-rank context-length strata. Selection uses neither QA outcomes nor attention-probe scores.

The remaining contexts are systematically shorter than the earlier longest-first set: selected cases span about `1.1K--9.3K` context tokens. NarrativeQA and MuSiQue have no unused eligible rows and are not represented as independent confirmation.

## Main results

| System | QA-F1 | Answer contains | Exact match |
|---|---:|---:|---:|
| Legacy HMO | 0.56956 | 0.48750 | 0.41250 |
| Layer-local HMO | 0.59052 | 0.50000 | 0.42500 |
| ChunkKV | **0.62022** | **0.53750** | 0.47500 |
| Full KV | 0.65280 | 0.56250 | **0.48750** |

| Comparison | Mean F1 delta | Wins / ties / losses | Paired bootstrap 95% interval |
|---|---:|---:|---:|
| Layer-local - legacy HMO | +0.02096 | 9 / 65 / 6 | [-0.03375, +0.07605] |
| Layer-local - ChunkKV | -0.02970 | 4 / 62 / 14 | [-0.06998, +0.00523] |
| Layer-local - Full KV | -0.06228 | 4 / 63 / 13 | [-0.11654, -0.01665] |

The intervals are fixed-seed descriptive uncertainty summaries, not post-hoc continuation gates.

## Task and length structure

| Dataset | Layer-local - legacy | Layer-local - ChunkKV |
|---|---:|---:|
| Qasper | +0.12359 | -0.04319 |
| MultiFieldQA-en | +0.00812 | +0.00454 |
| HotpotQA | +0.00833 | -0.04881 |
| 2WikiMQA | -0.05619 | -0.03133 |

| Length stratum | Mean context tokens | Layer-local - legacy | Layer-local - ChunkKV |
|---|---:|---:|---:|
| 0 | 2,588.9 | +0.05929 | -0.01050 |
| 1 | 4,162.0 | -0.03789 | -0.08573 |
| 2 | 4,982.4 | +0.05278 | -0.00787 |
| 3 | 6,028.5 | +0.00967 | -0.01470 |

There is no monotone length trend and no defensible positive task partition relative to ChunkKV except the near-tied MultiFieldQA slice.

## Contract and mechanism audit

- Legacy HMO, layer-local HMO, and ChunkKV have identical measured post-query total resident KV bytes in `80/80` cases.
- They also have identical per-Full-layer post-query byte counts and KV lengths in `80/80` cases. Thus the comparison does not hide a layerwise memory imbalance.
- All four systems have zero generation-cap hits.
- Mean compressed post-query residual-KV residency is `20.837%` of Full, including protected boundary and query tokens.
- Across `640` Full-layer/case pairs, layer-local retained query-attention mass is strictly higher than the legacy shared window in `636` and tied in `4`; minimum delta is zero. Mean absolute proxy gain is `0.00885`, and mean relative gain where defined is `10.59%`.
- Layer-local and legacy generated identical token sequences on `62/80` cases; layer-local and ChunkKV matched on `57/80`.
- Mean complete prompt/intervention/decode time is `3.09 s` for layer-local versus `3.11 s` for legacy and `3.13 s` for ChunkKV. Mean probe time is `2.95 s`; the single formal cache hit is the smoke case.

## Result-to-claim verdict

Independent internal review: `partial`, confidence `high`.

Supported:

1. Layer-local HMO is a clean correction to legacy HMO's cross-layer shared Sparse placement. It keeps the same global segment actions/counts and exact memory geometry while improving fresh-confirmation QA-F1 by `+0.02096` overall.
2. Under fixed actions and counts, independent per-layer max-mass window placement is proxy non-degrading versus the legacy shared window. Both the construction and the 640-pair audit support this bounded theorem.
3. The implementation is stable, exact-byte, and inexpensive on one 5090.

Not supported:

1. Layer-local HMO does not beat ChunkKV on this confirmation. The fresh result reverses the development package's `+0.01112` point estimate and ends at `-0.02970`.
2. Query-attention mass is not a monotone surrogate for QA. A proxy guarantee must not be rewritten as an answer-quality guarantee.
3. This run does not establish long-context, cross-budget, or cross-scale superiority, and it does not by itself justify an A100 run whose intended claim is HMO-over-ChunkKV.

## Recommended paper route

Freeze layer-local HMO v1 as the corrected HMO implementation and preserve HMO-FW as the diagnostic relaxation showing why pure attention-mass maximization is insufficient. The defensible story is a Hybrid-specific global/local residual-KV orchestration: global HMO chooses region coverage and fidelity; each Full-Attention layer chooses its local Sparse read window; DeltaNet recurrent state remains untouched. This story has a simple bounded theorem and two independent packages with positive overall layer-local-versus-legacy point estimates, but only a competitive, not superior, relationship to ChunkKV.

For the immediate decision, distinguish two goals:

- If the paper requires an explicit ChunkKV win, diagnose or pivot locally before paying for A100. Priority cases are the 14 ChunkKV wins and the 6 legacy wins, especially Qasper and 2WikiMQA; an optional 5%/20% 5090 sweep may map regimes but should not be called a rescue test.
- If the paper contribution is the Hybrid memory formulation, global/local decomposition, exact residual-KV accounting, and competitive quality, a later A100 run can be justified as scale/32K transfer rather than superiority validation. The C3 protocol and budget must first be updated to layer-local HMO and reviewed once by PZ/GPT; no paid resource has been started.
