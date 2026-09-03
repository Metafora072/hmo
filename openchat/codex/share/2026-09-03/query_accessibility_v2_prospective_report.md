# Query-Accessibility V2 Prospective Validation

Date: 2026-09-03  
Author: Codex  
Code: `main@bde0700`

## Verdict

`partial`, medium confidence.

The frozen V2 controller prospectively improves equal-byte oracle Top-K NDCG
over corrected sequential raw alpha on Qwen3.5-0.8B at both 8K and 16K. This
supports a scoped LongEval allocation claim, not a general cache-compression or
end-task quality claim.

## Frozen Protocol

- Method SHA-256: `01bcd3a9ea864ef2b9ab64c23c058badec70d7c88976168ff77100668b49a5f5`
- Protocol SHA-256: `c6de7c3c2208e68253731be45ce6cb447fd66817165933a6bff18673be295652`
- Model: Qwen3.5-0.8B revision `2fc06364715b967f1860aea9cf38778875588b17`
- Segment / middle-KV budget: 256 tokens / 10%
- Frozen rule: use `alpha*(1-rank01(query_read_share))` only when normalized
  alpha entropy is at least 0.45 and alpha/access Spearman is below 0.75;
  otherwise use corrected raw alpha.
- No task identity or oracle label is used at inference.
- Oracle acquisition used a dedicated prospective scope that performed no
  candidate analysis.

## Raw Results

| Length | Samples | Comparisons | Raw alpha NDCG | V2 NDCG | Delta, bootstrap 95% CI | LongEval delta | Needle delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8K | 6+6 | 686 | 0.82406 | 0.91409 | +0.09004 [+0.02114,+0.16856] | +0.18007 | 0 |
| 16K | 4+4 | 961 | 0.87832 | 0.93618 | +0.05786 [+0.01189,+0.11566] | +0.11572 | 0 |

| Length | Raw alpha pairwise | V2 pairwise | Delta, bootstrap 95% CI | Gate enabled | Top-K membership changed |
|---|---:|---:|---:|---:|---:|
| 8K | 0.58342 | 0.59549 | +0.01207 [-0.00498,+0.03295] | 4/12 | 3/12 |
| 16K | 0.52829 | 0.53650 | +0.00821 [-0.01480,+0.03381] | 3/8 | 3/8 |

All nonzero NDCG changes were positive: four LongEval samples at 8K and three
at 16K. One 8K sample improved NDCG by reordering the same Top-K membership,
which explains four nonzero NDCG deltas but only three membership changes.
Needle always abstained and was exactly unchanged.

The combined probe retained the corrected-alpha budget set: mean Top-K overlap
was 1.0 at both lengths; alpha Spearman was 0.99996 at 8K and 1.0 at 16K; no
sample changed alpha argmax.

## Interpretation

### Supported

The pre-frozen V2 rule can improve oracle fixed-budget segment selection over
corrected raw alpha for LongEval-Lines on this model and budget. The prior
retrospective 16K reversal did not reproduce prospectively.

### Not Supported

- broad cross-task benefit: Needle receives no gain because V2 always abstains;
- general pairwise ranking improvement: both pairwise intervals cross zero;
- end-task generation quality, latency, or memory benefit;
- semantic substitutability of recurrent state for explicit KV;
- robustness across model size, model family, budget, or realistic baselines.

The independent result-to-claim reviewer returned `partial` with medium
confidence and recommended the same scoped claim.

## Execution and Artifacts

| Stage | Runtime | Peak allocated / reserved VRAM |
|---|---:|---:|
| 8K oracle acquisition | 1986.3 s | 2.53 / 2.94 GB |
| 8K query evaluation | 28.9 s | 3.10 / 3.49 GB |
| 16K oracle acquisition | 3790.4 s | 3.52 / 4.49 GB |
| 16K query evaluation | 27.2 s | 4.66 / 5.30 GB |

- 8K run: `/mnt/nvme0/hmo/runs/p2_prospective_v2_8k_s20260921_20260903_085758`
- 8K result SHA-256: `c20ab083cb60d9982e8c2bfaafdd5a37f2d102a1cde4648fec2456bbaac1542d`
- 16K run: `/mnt/nvme0/hmo/runs/p2_prospective_v2_16k_s20260922_20260903_093357`
- 16K result SHA-256: `3534443e23d8fa814b67ee740c0b3bfa7f5168ba5c2505f3982b6e3d2500cabc`
- Combined artifact size: about 3.3 MB.
- GPU1 returned to 15 MiB after evaluation.

## Next Decision

Do not tune V2 on these outcomes. The highest-value next step is a separately
approved end-task, equal-byte generation validation that uses the frozen
selected KV sets, followed by faithful baselines and broader seeds/tasks if the
quality effect survives. Larger models and budget curves remain later evidence,
not an automatic continuation of this pilot.
