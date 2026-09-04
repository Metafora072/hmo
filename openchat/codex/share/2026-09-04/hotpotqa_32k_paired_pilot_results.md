# HotpotQA-32K-Aug Paired Pilot Results

Date: 2026-09-04

## Executive Result

The approved P7 four-case paired pilot completed on GPU1. At a frozen 10%
middle-context cap, the four compressed arms use exactly equal measured resident
attention-KV bytes in every case.

| System | Mean official QA F1 | Gold containment | Exact match |
|---|---:|---:|---:|
| HMO Contiguous CF | 0.3357 | 2/4 | 0/4 |
| Global Fixed-Chunk | 0.2315 | 2/4 | 0/4 |
| Raw Exact+Slack | 0.3981 | 2/4 | 1/4 |
| Scattered CF | **0.4038** | 2/4 | 1/4 |
| Full KV | 0.2315 | 2/4 | 0/4 |

HMO versus Fixed is `+0.1042` F1 with `2W/2T/0L`; versus Raw and Scattered it
is `-0.0625` and `-0.0682`. Every system solves the same two cases by answer
containment, so F1 differences come from output length and phrasing rather than
additional retrieved answers.

Mean compressed resident KV is `46,611,456` bytes, `11.556%` of Full KV. The
compressed systems are exact byte matches in `4/4` cases.

## Scientific Judgment

Independent result-to-claim review is `partial/supplement`, medium confidence.
This supports using the real-task result as an external-validity supplement:
HMO preserves Full-level solvability at about 11.6% residual-KV footprint and is
competitive with the structured baselines. It does not support claiming HMO is
best or that contiguous locality beats scattered retention on this pilot.

The operational smoke exposed a reproducibility caveat. On the shared first
case, HMO positions and output reproduced exactly, while Fixed swapped 7 of
3,737 retained positions in each direction and changed from F1 1.0 to 0.3333.
Raw swapped one position and Scattered 24, without output-score changes. This
near-tie ranking sensitivity weakens the formal HMO-versus-Fixed advantage and
should be addressed by deterministic ranking or repeated runs before a stronger
claim.

## Provenance

- Full report: `experiments/results/HOTPOTQA_32K_PAIRED_PILOT_20260904.md`
- Protocol: `refine-logs/hotpotqa_32k_paired_protocol.json`
- Formal run:
  `/mnt/nvme0/hmo/runs/hotpotqa_32k_paired_formal_20260904_174200/`
- Manifest code commit: `ddd189f36a4df0c6439b53445ec22e86b42bc7bc`
- JSONL SHA256:
  `b1ec7b520b9410d64a204770bc100f1676c55415519455f08ab17506e687e5cd`
- Summary SHA256:
  `a3b86be8062d95d119b0a8f2428f20cd485cf05ab7cbd1ecd61a5ecdffd73116`

Formal runtime was `147.48 s`; peak CUDA memory was `7.31/9.35 GiB`
allocated/reserved. GPU1 returned to `15 MiB`. No follow-up run was started.
