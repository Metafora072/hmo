# Research Findings

## 2026-09-02: Bounded Recurrent Correction at 8K

- Verdict: `partial` with medium confidence.
- Frozen method: `rank01(alpha) + 0.30 * (rank01(sigma_current) - 0.5)`.
- Held-out evidence: 12 new Qwen3.5-0.8B samples at 8K, 680 equal-byte oracle comparisons.
- Overall delta versus alpha: pairwise `+0.00183 [-0.00642,+0.01056]`; NDCG `+0.03515 [-0.03140,+0.10195]`.
- Task delta: LongEval pairwise/NDCG `+0.00902/+0.04368`; Needle `-0.00536/+0.02661`.
- Supported: a small directionally positive uplift survives freezing on fresh 8K samples, especially for LongEval.
- Unsupported: a stable, meaningful, cross-task ranking improvement.
- Constraint: do not retune lambda or introduce another formula on these confirmation labels.
- Next evidence: one 16K small-model transfer probe with the same scorer; consult Opus only if the length-transfer result remains mixed and a scope/pivot decision is needed.

