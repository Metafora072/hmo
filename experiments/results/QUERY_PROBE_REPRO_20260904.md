# HMO P8: Persistent FP32 Query-Probe Reproducibility

Date: 2026-09-04

## Purpose

P8 verifies the final deterministic query-ranking path before any broader 5090
matrix. It repeats the first frozen P7 HotpotQA-32K-Aug case in two fresh run
directories. R1 creates one identity-bound FP32 token-score artifact; R2 must
reuse that exact artifact. This is operational evidence and is excluded from
quality claims.

- Model: Qwen/Qwen3.5-0.8B
- Model revision: 2fc06364715b967f1860aea9cf38778875588b17
- Code commit: 348dff2632ccff6b4db9057ec6b33de7f8b5ae5c
- Parent paired protocol SHA256:
  1af7d542023c812b15f5e5f74f60dfab951a62becdacae37052f56a789c19268
- P8 protocol SHA256:
  6b38709c45f4c013e7842b5e905c9b6cbd3499057ee3f2d3b1292c12f15e7f90
- Sample: hotpotqa_32k_aug_b0029_d0168

## Required Invariants

| Check | Result |
|---|---|
| Both manifests bind the same clean commit | PASS |
| Probe ID equal across R1/R2 | PASS |
| Token-score SHA equal and matches the file | PASS |
| Cache-hit sequence is false, true | PASS |
| Four compressed position hashes equal across R1/R2 | PASS |
| Per-system resident bytes equal across R1/R2 | PASS |
| Four compressed arms are exactly equal-byte | PASS |

Probe ID:

fad4f015cbea80fd88ebdd2c808ef8b2a2713433a82e6025e9bfe54b90dd9339

FP32 score artifact SHA256:

68bf022c3ea4f23b371da7c0f7a60c6f72c7ff93c54a29dff295a9a2b180b745

Per-arm retained-position hashes:

| System | SHA256 |
|---|---|
| HMO Contiguous CF | 44e6ef034bdc9c29ee3d5fec28ade79af4357c41ebf561b2aa7a78088424d8e6 |
| Global Fixed-Chunk Top-K | 0614fa29893763e36ca9f5e6332a11550c5fba8348658495307ad3d45205735d |
| Raw Exact+Slack | e24164f29a66973a9ecf949e1194676bb0fe87c7b15ce6520fcd3bfa6d98e00d |
| Scattered CF | 5198dddd453648bd45479f52ca5e52401aa062a71c06a2fe9fcfc5e6ebd8cd73 |

Every compressed arm uses 46,657,536 post-query resident KV bytes, or 11.566%
of Full KV on this case.

## Supplementary Determinism

Generated token IDs and official QA F1 are identical for all five systems across
R1/R2. F1 is 0.7273 / 0.3333 / 1.0000 / 1.0000 / 0.3333 for HMO / Fixed /
Raw+Slack / Scattered / Full. These scores reproduce the final P7 first-case
values; they are not a new quality estimate.

## Runtime

| Run | Probe state | Time | Peak allocated | Peak reserved |
|---|---|---:|---:|---:|
| R1 | create | 42.01 s | 7.31 GiB | 7.49 GiB |
| R2 | reuse | 37.92 s | 4.81 GiB | 4.88 GiB |

R2 avoided probe recomputation and was 4.09 seconds faster in this single
operational pair. The memory difference is an observed runner property, not a
general latency or peak-memory claim.

GPU1 returned to 15 MiB after both processes exited. Two earlier launch attempts
were rejected before model execution because of environment/log-directory
packaging; they produced no probe or result data and are not P8 runs.

## Artifacts

- R1: /mnt/nvme0/hmo/runs/p8_probe_repro_348dff2_r1b/
- R2: /mnt/nvme0/hmo/runs/p8_probe_repro_348dff2_r2b/
- Probe cache: /mnt/nvme0/hmo/probes/query_fp32_v1/
- R1 JSONL SHA256:
  5df11b68df5867af0d349c976f8afc663be4697042c8bf7e96543e23eff95fb2
- R2 JSONL SHA256:
  28d6653368a17f8d7ad791abe48044c14dccc335dfdc057928bb829a095c3bd8
- R1 summary SHA256:
  6106e72b923e361409a9788248cc55027591db0ae4c326bfecc235b1220700af
- R2 summary SHA256:
  94a74c6f0823daf1568d041a5d892534d7867c91a533595a04345176724ec9b3
- R1 manifest SHA256:
  a61ad18b334cdaaa009f769d2ba8872492fb11f206fd22eedd91ed981fd0b908
- R2 manifest SHA256:
  46cbf003d6c2e0172f54dc5e6488a0a56df2528bebcd530fee54fa1a64957fcd

## Decision

The persistent-probe path is ready for Package C2. P8 resolves the specific P7
cross-run ranking-reproducibility problem by artifact reuse, without adding a
score-tolerance hyperparameter. It does not itself justify starting the 0.8B,
9B, or native-task matrices; those remain separately authorized work.
