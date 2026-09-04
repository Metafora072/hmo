# HotpotQA-32K-Aug Full-KV Solvability Results

Date: 2026-09-04

## Result

The approved P6 Full-KV routing package completed on Qwen3.5-0.8B. The four
frozen augmented samples all have exactly 32,768 serialized memory-context
tokens. Official LongBench QA F1 values are:

| Case | Gold | QA F1 | Gold contained |
|---|---|---:|---:|
| base 29 + donor 168 | University of Southern California | 0.3333 | yes |
| base 119 + donor 2 | Bassendean | 0.0000 | no |
| base 84 + donor 96 | due to the onset and progression of Alzheimer's disease | 0.5926 | yes |
| base 108 + donor 30 | American | 0.0000 | no |

Mean F1 is `0.2315`; `2/4` cases have nonzero F1 and contain the normalized
gold answer. This meets the frozen stronger descriptive routing signal for
considering a matched compressed pilot.

## Scope

This establishes only that the 0.8B Full-KV reference retains some usable
HotpotQA ability under the transparent 32K augmentation. It is not evidence
that HMO preserves that ability. The split has no native 32K HotpotQA records:
the protocol preserves one of the four longest base records and appends another
real HotpotQA context as a distractor, truncating only the distractor tail.

The two successful outputs contain the correct phrase inside a longer sentence,
which explains their partial F1. The failures return a nearby entity type. This
is better interpreted as imperfect relation/answer-type precision than complete
32K context failure.

## Provenance

- Full report: `experiments/results/HOTPOTQA_32K_SOLVABILITY_20260904.md`
- Protocol: `refine-logs/hotpotqa_32k_solvability_protocol.json`
- Formal run:
  `/mnt/nvme0/hmo/runs/hotpotqa_32k_solvability_formal_20260904_171905/`
- Code commit: `cfd0cb6a997e35c9a083385839c244152a9d0ae7`
- Manifest ID:
  `db61ee4020505f6d42732d27517f9c0b865832cd3c8cbc282c1a4bb0e7a4a5c5`
- JSONL SHA256:
  `65afcf742f79a816f712124ae24e5778805e665f356ed0be1ef3b2017945a6b0`
- Summary SHA256:
  `25fc2e203c79df03215f9467ead6c75ee3df1334f94001c7e8b6b14214c97229`

Formal runtime was `42.53 s`; peak CUDA memory was `4.81 GiB` allocated and
`6.35 GiB` reserved. GPU1 returned to `15 MiB`.

## Recommended Next Decision

The highest-value next GPU package is a small paired 32K pilot on frozen cases,
comparing HMO with Full KV and the strongest structured baselines under measured
resident-byte matching. It should be separately frozen and approved. No such
run has been started.
