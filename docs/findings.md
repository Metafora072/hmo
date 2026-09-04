# Research Findings

## 2026-09-04: Fresh Contiguous Coverage-Fidelity Confirmation

- Result-to-claim verdict: `partial`, medium confidence.
- Evidence: 48 fresh, unfiltered Qwen3.5-0.8B samples across 8K/16K Needle and
  LongEval-Lines under a frozen 10% middle-KV protocol.
- Contiguous CF versus scattered CF at exactly equal resident bytes: 34/48
  versus 27/48, +14.58 pp [+6.25,+25.00], 7 wins/41 ties/0 losses, exact sign
  p=0.0156. The advantage is positive at both lengths.
- Contiguous CF versus raw-alpha Exact Top-K: 34/48 versus 32/48, +4.17 pp
  [-4.17,+12.50], 3 wins/44 ties/1 loss, p=0.6250. Raw uses 1.22% fewer
  resident bytes because whole-segment selection leaves rounding slack.
- Contiguous CF versus contiguous Sparse-only: 34/48 versus 32/48, +4.17 pp
  [-8.33,+16.67], p=0.7539. Exact upgrades are not separately validated.
- Memory: contiguous CF uses a mean per-case 13.383% of Full-KV resident bytes
  and scores 70.83%, versus Full KV at 72.92%. The ratio of mean bytes is
  13.030%; these aggregation conventions must remain explicitly separated.
- Supported claim: contiguous max-attention-mass windows preserve useful local
  evidence better than scattered top-attention tokens at equal bytes in this
  model and synthetic suite.
- Unsupported: a strong win over raw Exact, recurrent-aware allocation,
  cross-task/model generality, or necessity of the fidelity upgrade.
- Constraint: do not retune on these confirmation labels. Next evidence should
  address raw byte matching/Pareto, another fresh seed, and a less-synthetic
  task as needed.
- Full report:
  openchat/codex/share/2026-09-04/contiguous_cf_fresh_confirmation_report.md

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

## 2026-09-03: Frozen V2 Prospective 8K and 16K

- Result-to-claim verdict: `partial`, medium confidence.
- The method and sequential protocol were frozen before outcomes; both runs
  used new seeds, sample IDs, and equal-byte oracle interventions.
- Fresh 8K, 6+6 samples: NDCG delta versus corrected raw alpha `+0.09004
  [+0.02114,+0.16856]`; LongEval `+0.18007`; Needle `0`.
- Fresh 16K, 4+4 samples: NDCG delta `+0.05786
  [+0.01189,+0.11566]`; LongEval `+0.11572`; Needle `0`.
- Every nonzero NDCG change was positive, but all gains came from LongEval and
  the pairwise intervals crossed zero at both lengths.
- Supported claim: frozen query-conditioned accessibility improves this
  model's oracle Top-K allocation at 8K and 16K under the tested budget.
- Unsupported: cross-task benefit, end-task quality, semantic KV
  substitutability, larger-model generality, and system advantage.
- Full report: openchat/codex/share/2026-09-03/query_accessibility_v2_prospective_report.md

## 2026-09-03: Frozen V2 Equal-Byte End Task

- Result-to-claim verdict: `no`, high confidence; the preregistered claim gate
  failed.
- All 48 formal raw/V2 comparisons used exactly equal resident attention-KV
  bytes; maximum byte difference was zero.
- At 8K, raw/V2 containment was 70.83%/75.00%, a paired +4.17 pp
  [-8.33,+16.67]; LongEval had 2 wins and 1 loss, while Needle tied 12/12.
- At 16K, raw/V2 was 83.33%/75.00%, a paired -8.33 pp [-20.83,0];
  LongEval had 0 wins and 2 losses, while Needle tied 11/12.
- Combined formal result: raw 77.08%, V2 75.00%, paired -2.08 pp
  [-10.42,+6.25]. LongEval had 2 wins versus 3 losses, so the final gate failed.
- Interpretation: prospective oracle-ranking gains do not reliably transfer to
  the jointly retained KV set under greedy generation. Query-conditioned
  accessibility survives as an observable, not as a validated selector.
- Stop condition: do not retune this V2 on P2/P3 labels. Diagnose
  oracle-to-generation mismatch before defining a new prospective method.
- Full report:
  openchat/codex/share/2026-09-03/query_accessibility_v2_end_task_report.md

## 2026-09-04: Package B Structured-Baseline Pareto

- Result-to-claim verdict: `partial`, route `supplement`, high confidence.
- Evidence: Qwen3.5-0.8B, matched 48-case 8K/16K Needle and LongEval-Lines
  suite, 5%/10%/20% middle-context caps, and five compressed arms with exact
  measured resident-byte equality in 48/48 cases at every cap.
- HMO/Fixed/Raw+Slack/Scattered/Sparse-only/Full correct counts were
  `30/36/30/21/30/35` at 5%, `34/36/32/27/32/35` at 10%, and
  `35/35/35/36/33/35` at 20%.
- Supported: HMO beats Scattered at 5% by +18.75 pp (9 wins, 0 losses,
  sign p=.0039) and at 10% by +14.58 pp (7 wins, 0 losses, sign p=.0156).
- Partial support: at 10% and 16K, HMO beats Fixed 18/24 to 16/24; on 16K
  LongEval it is 8/12 versus 6/12 for Fixed, Raw, and Full.
- Unsupported: an unconditional HMO win over fixed chunks. Fixed wins overall
  at 5% (36/48 versus 30/48), remains ahead at 10% (36/48 versus 34/48), and
  ties at 20%.
- Budget interpretation: 5% is below the approximately 6.25% full-coverage
  floor and HMO equals Sparse-only in 48/48 cases; 10% activates full
  macro-segment coverage and shows a 16K benefit; 20% is largely saturated.
- The Fixed comparison changes both global/stratified allocation and
  fixed/free-start placement, so it does not isolate free-start alone.
- Next evidence: one low-cost 16K/10% Stratified Fixed-Chunk control under the
  identical HMO allocation and bytes, followed by realistic-task transfer.
- Full report: experiments/results/PARETO_PACKAGE_B_20260904.md

## 2026-09-04: P5 Stratified Fixed-Chunk Control

- Result-to-claim verdict: `partial`, route `supplement`, medium confidence.
- Evidence: frozen 24-case Qwen3.5-0.8B 16K/10% suite. The new aligned control
  reuses HMO's macro allocation, Exact upgrades, retained-token counts, slack,
  anchors, probe, and bytes; only Sparse-window start freedom is removed.
- HMO is 18/24 versus 17/24 for Stratified Fixed-Chunk, with 2 wins, 21 ties,
  and 1 loss. Needle ties 10/12; LongEval is 8/12 versus 7/12.
- The control is nontrivial: mean retained-position Jaccard is 0.646 and an
  average 54.54/59.5 Sparse segments change placement per sample.
- Combined with Package B, Global Fixed / Stratified Fixed / HMO totals are
  16/17/18 overall and 6/7/8 on LongEval. These are complementary sample-level
  gains, not a nested additive sequence.
- Supported: free-start placement contributes a small directional part of the
  positive 16K HMO result. Unsupported: universal superiority across regimes.
- Full report: experiments/results/STRATIFIED_FIXED_CONTROL_20260904.md

## 2026-09-04: P6 HotpotQA-32K-Aug Full-KV Solvability

- This is model/task routing evidence, not an HMO comparison or contribution
  claim.
- The pinned 200-record LongBench HotpotQA split has no native 32K examples;
  source contexts range from about 1.8K to 17.7K Qwen3.5 tokens.
- The frozen augmentation preserves each of the four longest base contexts,
  questions, and gold answers, appends another real HotpotQA context, and
  truncates only the distractor tail. Every serialized memory context is exactly
  32,768 tokens.
- Full-KV official QA F1 is `0.3333/0/0.5926/0`, mean `0.2315`; `2/4` outputs
  contain the normalized gold answer and no output is an exact match.
- The two successes are verbose answers containing the correct phrase. The two
  failures return a related entity rather than the requested attribute,
  indicating imperfect relation/answer-type precision rather than total context
  failure.
- Scoped decision: 0.8B shows enough 32K solvability to justify proposing a
  separately frozen paired compressed pilot. No compressed run started
  automatically.
- Full report: experiments/results/HOTPOTQA_32K_SOLVABILITY_20260904.md

## 2026-09-04: P7 HotpotQA-32K-Aug Equal-Byte Paired Pilot

- Independent result-to-claim verdict: `partial`, route `supplement`, medium
  confidence.
- Four compressed arms use exactly equal measured post-query resident bytes in
  `4/4` cases. Their mean is `46,611,456` bytes, `11.556%` of Full KV.
- Mean official QA F1 for HMO / Fixed / Raw+Slack / Scattered / Full is
  `0.3357 / 0.2315 / 0.3981 / 0.4038 / 0.2315`.
- HMO versus Fixed is `+0.1042` F1 (`2W/2T/0L`); versus Raw and Scattered it is
  `-0.0625` and `-0.0682`.
- Every system contains the answer on the same `2/4` cases. F1 differences come
  from verbosity and phrasing on already-solved cases, not additional solvability.
- Smoke/formal reproducibility audit: HMO positions/output are identical, but
  Fixed swaps 7 retained positions in each direction and changes from F1 1.0 to
  0.3333; Raw and Scattered also show small position changes without F1 changes.
  This is consistent with near-tied query-probe ranking sensitivity and weakens
  the formal HMO-versus-Fixed claim.
- Supported: at about 11.6% Full-KV footprint, HMO preserves the same real-task
  solvable set and remains a plausible equal-byte contender. Unsupported: HMO
  is the strongest real-task method or contiguous locality beats Scattered here.
- Full report: experiments/results/HOTPOTQA_32K_PAIRED_PILOT_20260904.md

## 2026-09-04: P8 Persistent FP32 Query-Probe Reproducibility

- Final runners now persist one identity-bound FP32 token-score artifact and
  derive all query-ranked arms from it; mismatched, incomplete, or corrupt
  artifacts fail closed.
- Two clean-commit repeats of the first 32K Hotpot case used the same probe ID
  and score SHA, with cache hits false then true.
- All four compressed retained-position hashes, all five generated token
  sequences, and all per-system resident-byte measurements are identical across
  repeats.
- Each compressed arm uses 46,657,536 resident KV bytes, 11.566% of Full on the
  case. R1/R2 completed in 42.01/37.92 seconds and GPU1 returned to 15 MiB.
- This resolves the observed P7 ranking drift through exact artifact reuse,
  without introducing tolerance bucketing or a new method hyperparameter.
- Full report: experiments/results/QUERY_PROBE_REPRO_20260904.md

## 2026-09-04: C2 Native LongBench QA

- Local result-to-claim verdict: `partial`, route `confirm C2 / proceed to C3
  planning`, medium confidence; independent external review is pending.
- Evidence: 12 native HotpotQA and 12 native NarrativeQA records, selected
  before outcomes by exact 8K--16K serialized context length; no augmentation,
  truncation, or outcome filtering.
- HMO / Fixed / Raw+Slack / Scattered / Full official QA F1 is
  `0.3086 / 0.3251 / 0.2873 / 0.3211 / 0.2602`.
- HMO versus Fixed / Raw / Scattered / Full paired deltas are
  `-0.0165 / +0.0213 / -0.0124 / +0.0485`.
- NarrativeQA is the positive slice: HMO is highest at `0.2934`, improving over
  Fixed, Raw, Scattered, and Full by `0.0504/0.0429/0.0474/0.0973`. HotpotQA
  favors Fixed and Scattered; HMO is effectively tied with Raw and Full.
- All compressed arms use exactly equal measured resident KV in 24/24 cases,
  averaging 12.80% of Full. No generation hits its token limit.
- Supported: native-task competitive efficiency and a task-conditioned benefit.
  Unsupported: universal native-QA superiority or full-split LongBench claims.
- Full report: experiments/results/NATIVE_LONGBENCH_C2_20260904.md
