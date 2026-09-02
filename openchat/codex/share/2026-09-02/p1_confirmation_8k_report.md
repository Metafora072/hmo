# P1 Frozen-Scorer 8K Confirmation

Date: 2026-09-02  
Code: `main@d94e200`  
Verdict: partial, medium confidence

## Setup

- Model: Qwen3.5-0.8B, revision `2fc06364715b967f1860aea9cf38778875588b17`
- Frozen formula: `rank01(alpha) + 0.30 * (rank01(sigma_current) - 0.5)`
- Frozen config SHA-256: `c89f07a644c6c702356e3122d158566414292b0cc413546d99908d70649814a0`
- Context / segment / KV budget: 8K / 256 / 10%
- Held-out samples: 6 Needle + 6 LongEval, seed `20260908`
- Sample prefix: `confirm_s20260908_`
- Oracle comparisons: 680
- Manifest: `d4add765372b6464f4051e7288e85d02f4e77344ba4b5f4a1edfa7cd885489a9`
- Artifact: `/mnt/nvme0/hmo/runs/p1_confirmation_qwen08b_8k_s20260908_20260902_214132`

The manifest contains no candidate list and embeds the full frozen scorer plus its file hash. All 12 oracle utility ranges were nonzero. No runtime error or non-finite result was found.

## Results

| Scope | Pairwise delta vs alpha | NDCG delta vs alpha |
|---|---:|---:|
| Overall | `+0.00183 [-0.00642,+0.01056]` | `+0.03515 [-0.03140,+0.10195]` |
| LongEval | `+0.00902` | `+0.04368` |
| Needle | `-0.00536` | `+0.02661` |

NDCG is heterogeneous rather than uniformly improved. LongEval sample deltas are `-0.212, +0.215, +0.220, 0, +0.039, 0`; Needle has one `+0.160` sample, one `-0.0006`, and four zeros. The overall NDCG mean must therefore not be presented as a stable per-example gain.

Runtime was 2221.34 seconds on physical GPU1. Peak allocated/reserved VRAM was 2.35/2.74 GiB. GPU1 returned to 15 MiB after exit.

## Claim Audit

An independent internal result-to-claim review returned `partial` with medium confidence. It supports a plausible small uplift that survives freezing, especially on LongEval, but rejects a robust cross-task-win claim. This internal reviewer is not Opus and is not attributed as Opus.

## Decision

Do not retune the scorer on these labels. A compact 16K transfer probe has higher information value than another methodological review because it tests the remaining length-sensitivity hypothesis under a fixed rule. Opus review is deferred until after 16K; it becomes useful only if the result remains mixed and the project must choose between a LongEval-focused scope and a method pivot.
