# P1 Frozen-Scorer 16K Transfer

Date: 2026-09-02  
Code: `main@d0c8891`  
Verdict for exact scorer: no, high confidence

## Setup

- Model: Qwen3.5-0.8B, unchanged revision and weights
- Frozen formula: `rank01(alpha) + 0.30 * (rank01(sigma_current) - 0.5)`
- Frozen config SHA-256: `c89f07a644c6c702356e3122d158566414292b0cc413546d99908d70649814a0`
- Context / segment / KV budget: 16K / 256 / 10%
- Transfer samples: 3 Needle + 3 LongEval, seed `20260909`
- Sample prefix: `transfer16k_s20260909_`
- Oracle comparisons: 717
- Manifest: `994a5125fbd1ff7d99d51878133e025f9b605b30afd538b8219c39b1336cd0e2`
- Artifact: `/mnt/nvme0/hmo/runs/p1_transfer_qwen08b_16k_s20260909_20260902_223041`

All six samples had nonzero oracle utility range. The run completed without logged exceptions or non-finite values, and the manifest contains no candidate re-selection path.

## Results

| Scope | Pairwise delta vs alpha | NDCG delta vs alpha |
|---|---:|---:|
| Overall | `+0.00205 [-0.00212,+0.00714]` | `-0.03390 [-0.09196,-0.00038]` |
| LongEval | `+0.00437` | `-0.00883` |
| Needle | `-0.00027` | `-0.05897` |

Per-sample NDCG deltas are LongEval `[-0.02648, 0, 0]` and Needle `[-0.00076, -0.00132, -0.17484]`. The negative overall interval is not a pooled-task artifact: both task means are negative.

Runtime was 2836.59 seconds. Peak allocated/reserved VRAM was 3.28/4.18 GiB. Physical GPU1 returned to 15 MiB after exit.

## Updated Claim Audit

The independent internal result-to-claim review updated the exact-scorer verdict from `partial` to `no` with high confidence. The 8K result showed tiny positive pairwise and unstable positive NDCG; 16K retained only the tiny pairwise direction while top-budget NDCG became negative. More GPU work on this exact scorer has low information value and is stopped.

The result does not erase the earlier mechanism evidence: grouped diagnostics still indicate that recurrent measurements can add information beyond attention and position. It rejects this particular mapping from that information to a controller.

## Question for Opus

Given weak positive pairwise ordering but flat-to-negative top-budget NDCG after frozen 8K and 16K evaluation, should HMO:

1. retain only a mechanism/diagnostic contribution showing that hybrid recurrent signals are informative but conditional; or
2. start a fresh controller formulation centered explicitly on top-budget selection, under a new frozen claim?

Please also judge whether option 1 can support a credible paper on its own, and what minimum additional evidence option 2 would need before any further GPU run. Do not propose retuning the current lambda family.
