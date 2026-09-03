# Research Findings

## 2026-09-03: One-Swap Boundary Exchange Offline Screen

- Verdict for the exact policy: `no`; stop without a fresh GPU confirmation.
- Frozen action: replace the lowest-alpha SAFE segment inside TopK with the
  highest-alpha STRESSED segment outside TopK, at most once per sample.
- Existing discovery evidence: 12 Qwen3.5-0.8B 8K samples, Top-3 exact-KV
  budget, no new oracle labels or GPU work.
- Only 2/12 samples admitted an exchange, both on Needle; one exchange improved
  oracle utility and one reduced it. All six LongEval samples were no-ops.
- Overall deltas versus raw alpha: TopK mean utility `+0.000027
  [-0.000567,+0.000649]`; NDCG `+0.000033 [-0.000129,+0.000228]`; pairwise
  `-0.000575 [-0.002299,+0.000575]`.
- Interpretation: acting directly at the budget boundary does not rescue the
  fixed median safe/stressed controller because the required boundary
  configuration is itself rare.
- Stop condition: do not run fresh 8K/16K or tune thresholds, exchange count,
  alpha margins, or candidate rules for this exact controller.
- Full report: openchat/codex/share/2026-09-03/p1_boundary_exchange_offline_report.md

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


## 2026-09-03: Frozen Conditional Controller at 8K

- Result-to-claim verdict for the exact controller: no, high confidence.
- Evidence: 12 fresh Qwen3.5-0.8B samples at 8K, 683 equal-byte oracle comparisons, fixed top-3 budget, unchanged frozen SHA 183255763fb2bdfa7e29f9bc46e8eed88f8658183077b258dd0cab563e8f4a93.
- Overall delta versus raw alpha: pairwise -0.00019 [-0.00287,+0.00249]; NDCG +0.00751 [-0.00086,+0.02339].
- Task delta: LongEval pairwise/NDCG +0.00115/+0.01559; Needle -0.00153/-0.00057.
- Activity: 2 to 10 adjacent swaps per sample, but NDCG changed in only 2/12 samples; one LongEval gain of +0.09355 dominates the mean, one Needle sample changed by -0.00343, and ten were zero.
- Interpretation: discovery-level safe/stressed mechanism evidence survives, but the one-rank conditional mapping does not reliably improve held-out fixed-budget allocation.
- Stop condition: no 16K run, threshold search, larger swap radius, alpha buckets, or held-out retuning for this exact controller. Universal multiplicative/additive scorers remain rejected.
- Route: stop the current controller aspect and return to OpenChat for mechanism/diagnostic scope versus a genuinely new claim.
- Full report: openchat/codex/share/2026-09-03/p1_conditional_confirmation_8k_report.md

## 2026-09-03: Corrected Alpha and Query Accessibility Exploration

- The legacy multi-token alpha probe did not execute a true Qwen3.5 recurrent continuation. The corrected probe advances the query one token per forward on one private hybrid cache.
- Legacy versus corrected alpha across 12 discovery samples: mean Spearman 0.7835, mean Top-3 overlap 0.8333, unchanged argmax.
- Corrected sigma_current no longer shows stable broad incremental value; corrected phi_delta_alpha retains grouped OOF Top-K NDCG +0.0911 [+0.0370,+0.1523].
- Exact query-conditioned recurrent accessibility was implemented as the read norm/share of each segment's surviving state contribution under the real DeltaNet query.
- Direct access-deficit and access-excess mappings failed. A frozen dual-confidence abstention gate produced retrospective reuse-label 8K NDCG +0.0506 [0,+0.1068], LongEval +0.1012, Needle 0; all three active changes were positive.
- The same frozen controller failed 16K transfer: NDCG -0.0414 [-0.1845,+0.0602], LongEval -0.0828, Needle 0. A one-swap shrinkage also failed 16K.
- Verdict: the retrospective 8K pattern is promising but not prospectively confirmed, and the current controller is not length robust. Stop hand mapping changes and run one preregistered fresh 8K before any fresh 16K.
- Full report: openchat/codex/share/2026-09-03/query_accessibility_exploration_report.md

## 2026-09-03: V6.1 and Corrected-Alpha Scope Clarification

- The multi-token continuation bug belongs to the later E3-v2 query-aware
  alpha probe. V6.1 E1 uses full-prompt prefill followed by a single generated
  token continuation and is not invalidated by that specific bug.
- Preserve V6.1 outputs, actions, and tracked KV bytes as evidence for the old
  heuristic implementation. They do not validate sigma as recurrent-memory
  reliability or alpha-times-sigma as the HMO mechanism.
- Historical evaluator and lite/proxy baseline limitations remain separate
  publication-readiness issues.
- Current route: treat query-to-segment recurrent readout as an operational
  accessibility observable, then test the frozen V2 allocation prospectively.
