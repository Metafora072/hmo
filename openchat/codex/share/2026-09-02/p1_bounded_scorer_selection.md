# P1 Bounded-Additive Scorer Selection

Date: 2026-09-02  
Scope: development-set selection only, not confirmation

## Decision

GPT's proposal is concrete enough to execute without an additional Opus review gate. The four-value family is bounded, falsifiable, and evaluated entirely on the existing 12 discovery samples. Opus should be consulted only if held-out confirmation is ambiguous or exposes a task-scope conflict that requires a paper-level judgment.

## Frozen Family

```text
a_i = rank01(alpha_i)
r_i = rank01(sigma_current_i) - 0.5
score_i(lambda) = a_i + lambda * r_i
lambda in {-0.30, -0.15, +0.15, +0.30}
```

Ranking and normalization are within each sample over eligible middle segments. Selection maximizes mean pairwise improvement over rank-normalized alpha, then NDCG, then smaller absolute lambda.

## Development Results

All values are improvements over alpha on the 12 existing discovery samples, with sample-bootstrap 95% intervals.

| Lambda | Pairwise delta | NDCG delta | LongEval / Needle pairwise | LongEval / Needle NDCG |
|---:|---:|---:|---:|---:|
| -0.30 | -0.02041 `[-0.03516,-0.00518]` | -0.30141 `[-0.44007,-0.16719]` | -0.04024 / -0.00057 | -0.16360 / -0.43922 |
| -0.15 | -0.00786 `[-0.01312,-0.00182]` | -0.07677 `[-0.13848,-0.01911]` | -0.01399 / -0.00172 | -0.00124 / -0.15230 |
| +0.15 | -0.00010 `[-0.00479,+0.00402]` | -0.00353 `[-0.06041,+0.04982]` | -0.00096 / +0.00077 | +0.03321 / -0.04027 |
| **+0.30** | **+0.00470** `[-0.00450,+0.01293]` | **+0.00576** `[-0.06596,+0.06831]` | **+0.00613 / +0.00326** | +0.06604 / -0.05451 |

The selected effect is small and uncertain. It is not evidence that the method works. Its only encouraging property is positive pairwise direction on both task groups, which is sufficient to justify one held-out falsification run without changing the formula again.

## Frozen Artifact

- Config: `codex/share/2026-09-02/p1_bounded_additive_frozen.json`
- Full offline output: `/mnt/nvme0/hmo/runs/p1_bounded_selection_12samples_20260902.json`
- Development sources: manifests `ed2262672a08...6390bc` and `6b60a7df4caa...f296a`

The confirmation runner embeds the full config and its SHA-256 in the immutable run manifest. It requires a distinct sample-ID prefix and evaluates only the selected lambda; it cannot reselect a candidate on held-out labels.

## Next Run

Run one 8K Qwen3.5-0.8B held-out confirmation on physical GPU1 with new seed and sample IDs, 6 Needle plus 6 LongEval samples, the existing 10% budget, 256-token segments, two donors, one background, and primary oracle quality only. Do not start 16K automatically; first judge aggregate and per-task deltas against alpha.
