# Experiment Plan

**Problem**: query-conditioned recurrent-state-aware exact-KV allocation
**Method Thesis**: explicit KV demand depends on what the current query can already retrieve from the recurrent channel.
**Date**: 2026-09-03

## Claim Map
| Claim | Minimum Convincing Evidence |
|---|---|
| Accessibility contains incremental utility information | Grouped-CV NDCG gain above corrected alpha plus position, with positive lower bootstrap bound and no task-sign conflict |
| Frozen allocation improves top-budget choice | Positive NDCG versus corrected raw alpha on independent 8K and directionally positive 16K; Needle nonnegative |

## Run Order
| Run | Purpose | Gate | Cost |
|---|---|---|---|
| R000 | Real-model probe integrity | Alpha unchanged exactly; finite accessibility | completed, 0.003 GPU-h |
| R001 | 12-sample discovery enrichment | direct NDCG above +0.02 on both tasks, or incremental NDCG above +0.03 with CI lower above zero | about 3 minutes |
| R002 | Independent 8K recapture and frozen evaluation | overall and LongEval positive; Needle at least -0.005 | about 2 minutes |
| R003 | Independent 16K transfer | overall positive; no strong task reversal | about 2 minutes |

## Stop Rules
- No threshold, weight, feature, or task-specific search after R001.
- If R001 fails both gates, stop query accessibility.
- If R002 fails, do not run new oracle labels or scale the method.
- Feedback GPT and Opus only after R002 and R003 support the frozen claim.
