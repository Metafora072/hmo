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


## 2026-09-02: Bounded Recurrent Correction at 16K

- Updated verdict for the exact scorer: `no`, high confidence.
- Evidence: 6 new Qwen3.5-0.8B samples at 16K, 717 equal-byte oracle comparisons, unchanged frozen config.
- Overall delta versus alpha: pairwise `+0.00205 [-0.00212,+0.00714]`; NDCG `-0.03390 [-0.09196,-0.00038]`.
- Task delta: LongEval pairwise/NDCG `+0.00437/-0.00883`; Needle `-0.00027/-0.05897`.
- Interpretation: the scorer can slightly reorder broad pairs but does not improve, and at 16K harms, the top-budget selection objective.
- Stop condition: no more GPU runs or retuning for `rank01(alpha) + 0.30 * (rank01(sigma_current) - 0.5)`.
- Surviving claim: recurrent measurements may contain diagnostic information beyond attention, but this bounded additive rule does not convert it into reliable fixed-budget utility.
- Next decision: external Opus review should choose between a mechanism/diagnostic-only scope and a fresh method formulation under a new claim.


## 2026-09-03: Conditional Safe/Stressed Regime Evidence

- Verdict: the segment-level safe/stressed hypothesis is supported on existing discovery evidence; controller performance remains untested.
- Protocol: 12 discovery samples and 360 segments; sample-grouped OOF ridge residuals controlling alpha + position; within-sample sigma/delta rank median split; no threshold search.
- Main contrast: Q4 high-sigma/high-delta minus Q3 high-sigma/low-delta residual utility is +0.25541 [+0.03702,+0.45082], with 9/12 samples positive.
- Task contrast: LongEval +0.28193 (5/6 positive); Needle +0.22889 (4/6 positive); no aggregate task direction reversal.
- Interaction evidence: low-sigma Q1/Q2 means are nearly identical (-0.12495/-0.12786), while Q3/Q4 differ (-0.06870/+0.17931).
- Limitation: Q3 is sparse (39 segments versus 141 in Q4), and 3/12 sample contrasts are negative; this is a design basis, not a held-out method result.
- Next action pending confirmation: freeze a minimal discrete one-rank SAFE/NEUTRAL/STRESSED adjustment, unit-test it, then run one fresh 8K held-out confirmation on GPU1.
- Full report: openchat/codex/share/2026-09-03/conditional_regime_offline_report.md
