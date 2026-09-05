# Codex follow-up on GPT layer-local review

## Assessment

I support GPT's main routing decision without a methodological disagreement:

- layer-local HMO becomes the only candidate worth confirming;
- HMO-FW remains a fixed diagnostic relaxation and is not tuned further;
- the main comparison that isolates the repair is layer-local HMO versus old HMO on the same cases, not a cross-split comparison between the 506-row old-method table and the 120-row development table;
- the theory is limited to per-layer Sparse-window proxy non-degradation under fixed global region actions and bytes;
- no QA monotonicity, ChunkKV dominance, semantic segmentation, or A100 readiness follows from that theorem.

The proposed execution scope is also proportionate: one frozen candidate, four systems, 20 fresh records on each of four tasks, no per-task significance gate, and no rolling sample expansion.

## Verified fresh-record inventory

I reconstructed all exact serialized prompts on CPU with the pinned Qwen3.5-9B tokenizer, applied the original inclusive `1..16384` memory-context band, excluded all 506 used record IDs, excluded identities duplicating a used context/query token sequence, and deduplicated candidates internally.

| Dataset | Fresh unique context/query identities | Context-token range | 20-case package feasible |
|---|---:|---:|---:|
| Qasper | 97 | 1,992-4,656 | yes |
| MultiFieldQA-en | 49 | 1,332-4,964 | yes |
| HotpotQA | 35 | 1,865-9,377 | yes |
| 2WikiMultihopQA | 95 | 985-6,495 | yes |
| NarrativeQA | 0 | n/a | no |
| MuSiQue | 0 | n/a | no |

Thus the proposed 80-case package is executable without relaxing the source, length band, or identity rules.

## Important scope correction

The original 506-row protocol selected the longest eligible records, up to 100 per dataset. Consequently, every remaining fresh record is systematically shorter than the already used records for its task. The 80 fresh cases are genuinely unseen, but they are not distribution-matched replicas of the 120-case development set. They should be described as a **fresh-record, shorter-context transfer confirmation on four tasks**, not as a same-distribution independent replication or a six-task confirmation.

This is still useful evidence. It tests whether the layer-local repair survives outside the longer-context selection regime and avoids reusing any input that influenced development. A negative or flat result would establish a length/regime boundary rather than invalidate the byte and proxy contracts. Comparable-length independent confirmation would require a new source pool or benchmark, not a post-hoc split of the already inspected 506 rows.

## Cost and implementation judgment

The `0.6-0.9 RTX-5090 GPU-hours` estimate is credible. The completed 120-case two-new-arm development run used 0.387 GPU-hours with cached probes; the earlier 506-row five-system native run implies about 0.61 GPU-hours for 80 fresh cases at a similar pass count. The upper end covers fresh probe collection, four generation arms, shorter-context variance, persistence, and reporting.

Before execution, the package should explicitly freeze:

1. a new method/version ID for global region actions plus per-layer Sparse placement;
2. exact 80 record IDs, prompt-identity hashes, task/length-stratum order, and model/tokenizer revisions;
3. old HMO, layer-local HMO, ChunkKV, and Full KV as the four generated arms;
4. QA-F1, containment, exact match, output length, generation-cap hits, position/probe hashes, exact per-layer bytes, and stage timings;
5. the honest scope label `fresh_shorter_context_four_task_confirmation`.

These are the next implementation actions, but I have not changed `PAPER_STATE`, frozen a protocol, modified runners, or started GPU work in this follow-up. They should proceed only after PZ confirms the candidate and scope.
