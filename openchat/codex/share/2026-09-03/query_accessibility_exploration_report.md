# HMO Query-Accessibility Exploration

Date: 2026-09-03
Author: Codex

## Executive Judgment

The exploration found a real but scoped 8K result, not a length-robust controller.

- Correcting the hybrid alpha probe materially changes segment ranking below Top-1.
- A direct query-to-DeltaNet-state readout is measurable and isolated without perturbing alpha.
- A dual-confidence abstention controller improves Top-K NDCG on multiple
  retrospective 8K reuse-label evaluations while leaving Needle unchanged.
- The identical frozen controller reverses at 16K, so a general 8K-to-16K claim is unsupported.
- Hand mappings, learned three-feature scoring, and bounded one-swap variants are stopped.

## 1. Correctness Repair

The old alpha probe passed the entire multi-token query in one forward. Qwen3.5 only consumes prior DeltaNet recurrent state when sequence length equals one, so Full-Attention query states used the context cache while DeltaNet query states did not follow true recurrent continuation.

Commit a006a15 changes alpha collection to one query token per forward with one private evolving hybrid cache. A real Qwen3.5-0.8B preflight passed with exact cache isolation.

Across the 12 discovery samples, legacy versus corrected alpha had mean Spearman 0.7835 and Top-3 overlap 0.8333. Argmax stayed unchanged, but budget-boundary rankings did not.

Corrected-alpha consequences:

- sigma_current no longer has stable broad incremental value: pairwise +0.0031 [-0.0316,+0.0320], NDCG -0.0476 [-0.1243,+0.0054].
- phi_delta_alpha remains a top-budget diagnostic: grouped OOF NDCG +0.0911 [+0.0370,+0.1523], driven by LongEval +0.1820 and Needle +0.0001.
- The safe/stressed Q4 minus Q3 observation survives corrected alpha: +0.2684 [+0.0509,+0.4585], LongEval +0.3199 and Needle +0.2169.

## 2. Candidates Screened

### Candidate A: Tiny learned utility scorer

Pointwise ridge and pairwise logistic used corrected alpha, position, and phi_delta_alpha. Against corrected raw alpha, grouped-CV NDCG was approximately zero for ridge and -0.136 for pairwise logistic. Stopped before GPU.

### Candidate B: Query-conditioned recurrent accessibility

For recurrent layer l and segment i, the final context state is exactly decomposed into suffix-decayed contributions C_li. During true sequential query continuation, the probe reads each contribution with the normalized DeltaNet query q_lt. The primary observable is mean relative norm of q_lt^T C_li across query tokens and recurrent layers.

Real 8K preflight:

- alpha from the combined probe exactly equaled standalone corrected alpha;
- accessibility was finite;
- elapsed 9.72 seconds;
- peak allocated/reserved CUDA memory 3.10/3.29 GB.

Direct accessibility directions failed discovery:

- access deficit alpha times one minus access rank: NDCG -0.1750, LongEval +0.2575, Needle -0.6075;
- access excess alpha times access rank: NDCG -0.0691 [-0.1349,-0.0146].

The strong task asymmetry motivated abstention rather than another global scorer.

### Candidate C: Dual-confidence abstention

Version 1 applied access deficit only when normalized alpha entropy was at least 0.45. Discovery NDCG was +0.1287 [+0.0510,+0.2251], but the first retrospective 8K holdout exposed one high-entropy Needle false activation; overall remained +0.0819 [+0.0121,+0.1544], while Needle was -0.0195.

Version 2 was then frozen using both prior 8K artifacts as development evidence:

    enable recurrent correction iff
    normalized alpha entropy >= 0.45
    and Spearman(alpha, query_read_share) < 0.75

When disabled, the method is exactly raw alpha. It uses no task identity or oracle label at inference.

## 3. Frozen V2 Results

| Split | Samples | NDCG delta | LongEval | Needle | Active Top-K changes |
|---|---:|---:|---:|---:|---:|
| Development seed 20260902/03 | 12 | +0.1143 [+0.0338,+0.2137] | +0.2286 | 0 | 3 |
| Development seed 20260908 | 12 | +0.0699 [+0.0154,+0.1306] | +0.1398 | 0 | 4 |
| Retrospective holdout 8K seed 20260911 | 12 | +0.0506 [0,+0.1068] | +0.1012 | 0 | 3 |
| Frozen 16K transfer seed 20260909 | 6 | -0.0414 [-0.1845,+0.0602] | -0.0828 | 0 | 2 |

All three active changes on the retrospective 8K holdout were positive: +0.1672, +0.2064, and +0.2337 NDCG. The bootstrap lower bound is exactly zero because 9/12 samples abstain; it does not reflect a negative active sample.

At 16K, the two active LongEval samples were +0.1205 and -0.3690. A bounded one-swap variant was checked offline and also failed 16K LongEval (-0.0644), so further mapping tweaks were stopped.

Pairwise ranking is not improved by the full reranking controller. Retrospective 8K holdout pairwise delta is -0.0090 [-0.0226,+0.0008]. The defensible development observation is therefore limited to fixed-budget Top-K selection.

## 4. Interpretation

The positive finding is narrower than the original HMO claim: query-conditioned recurrent accessibility can identify useful Top-K corrections when attention is diffuse and the two memory channels disagree, at 8K. The negative length transfer says the current accessibility aggregation and gate are not calibrated across segment count or budget size.

This is not evidence for returning to V6.1 alpha-times-sigma, Refresh, RTS, or another hand score. It suggests that any next method must explicitly solve length/budget calibration, probably by learning or normalizing marginal utility rather than changing thresholds on the existing 16K labels.

## 5. Questions for GPT and Opus

1. Is the repeated 8K Top-K result sufficient to continue as a scoped confidence-abstention method, or does the 16K reversal make the controller line too weak?
2. If continuing, which single mechanism is more defensible: length/budget-normalized recurrent accessibility, or a learned marginal-utility predictor that uses the exact-KV oracle only as a teacher?
3. Should the paper claim prioritize the corrected hybrid-query measurement and conditional 8K finding, or should controller work stop until fresh 16K generalization is solved?
4. Please do not propose another threshold search or hand fusion on the used artifacts; any new mapping needs a fresh 8K and 16K confirmation.

## 6. Artifacts

- Corrected discovery: /mnt/nvme0/hmo/runs/p1_corrected_alpha_discovery_12samples_20260903.json
- Query discovery: /mnt/nvme0/hmo/runs/r001_query_accessibility_discovery_12samples_20260903.json
- V1 retrospective 8K holdout: /mnt/nvme0/hmo/runs/r002_query_accessibility_confirmation_8k_s20260908_20260903.json
- V2 retrospective 8K holdout: /mnt/nvme0/hmo/runs/r003_dual_confidence_confirmation_8k_s20260911_20260903.json
- V2 16K transfer: /mnt/nvme0/hmo/runs/r004_dual_confidence_transfer_16k_s20260909_20260903.json
- Frozen v2 config: refine-logs/dual_confidence_abstention_v2_frozen.json
- GPU1 after runs: 15 MiB, 0% utilization.

All runs reused existing equal-byte oracle labels. No new oracle interventions were launched in this exploration.
