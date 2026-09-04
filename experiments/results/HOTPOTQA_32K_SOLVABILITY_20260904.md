# HMO P6: HotpotQA-32K-Aug Full-KV Solvability

Date: 2026-09-04

## Purpose And Protocol

This run asks only whether Qwen3.5-0.8B retains usable HotpotQA ability with a
32K memory context before evaluating any compressed system. It is routing
evidence, not an HMO result.

The pinned LongBench HotpotQA split has no native 32K examples under the pinned
Qwen3.5 tokenizer; its longest source context has 17,719 tokens. The frozen
`HotpotQA-32K-Aug` protocol selects the four longest base records by source
token length, preserves each context, question, and all gold answers, and
appends another real HotpotQA context as a post-target distractor. Only the
distractor tail is truncated. All four serialized memory contexts contain
exactly 32,768 tokens.

- Model: `Qwen/Qwen3.5-0.8B`
- Model revision: `2fc06364715b967f1860aea9cf38778875588b17`
- Protocol: `refine-logs/hotpotqa_32k_solvability_protocol.json`
- Dataset revision: `zai-org/LongBench@5e628be450b7e67fb7ae6e201bd6d8f7056f7672`
- Dataset archive SHA256:
  `cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64`
- Metric: official LongBench `qa_f1_score`, revision
  `2e00731f8d0bff23dc4325161044d0ed8af94c1e`
- Formal run:
  `/mnt/nvme0/hmo/runs/hotpotqa_32k_solvability_formal_20260904_171905/`
- Manifest code commit: `cfd0cb6a997e35c9a083385839c244152a9d0ae7`

## Raw Results

| Base / donor | Gold | Full-KV prediction | QA F1 | Contains gold |
|---|---|---|---:|---:|
| 29 / 168 | University of Southern California | The 1958 Pro Bowl was played at the Los Angeles Memorial Coliseum, which is the home of the University of Southern California Trojans football | 0.3333 | 1 |
| 119 / 2 | Bassendean | Swan Districts Football Club | 0.0000 | 0 |
| 84 / 96 | due to the onset and progression of Alzheimer's disease | Patrick Bowlen stepped down as CEO of the Denver Broncos in 2014 due to the onset and progression of Alzheimer's disease. | 0.5926 | 1 |
| 108 / 30 | American | Merck & Co. | 0.0000 | 0 |

Aggregate results:

- Mean official QA F1: `0.2315`
- Median official QA F1: `0.1667`
- Nonzero-F1 cases: `2/4`
- Normalized answer-containment cases: `2/4`
- Normalized exact-match cases: `0/4`
- Mean post-query resident attention-KV: `403,344,384` bytes (`0.376 GiB`)

## Interpretation

The 0.8B backbone is not wholly incapable at 32K. In two cases it retrieves the
correct answer phrase but produces a longer explanatory sentence, which lowers
official F1. The two failures answer a nearby entity rather than the requested
attribute: a football club instead of its Perth suburb, and a company instead
of its nationality. This pattern suggests limited relation and answer-type
precision rather than total loss of long-context evidence.

The preregistered routing signal is therefore positive: `2/4` nonzero-F1 cases
meets the stronger descriptive signal for considering a paired compressed
pilot. The sample is deliberately tiny and augmented rather than native 32K,
so it does not establish benchmark-level HotpotQA quality, broad real-task
transfer, or any HMO advantage.

## Runtime And Integrity

The operational smoke completed with F1 `0.3333` before the formal package.
The formal run completed `4/4` cases in `42.53 s` on GPU1, with peak CUDA memory
of `4.81 GiB` allocated and `6.35 GiB` reserved. The process exited with status
zero and GPU1 returned to `15 MiB`.

- Formal JSONL SHA256:
  `65afcf742f79a816f712124ae24e5778805e665f356ed0be1ef3b2017945a6b0`
- Formal summary SHA256:
  `25fc2e203c79df03215f9467ead6c75ee3df1334f94001c7e8b6b14214c97229`
- Formal manifest SHA256:
  `05b3c896af4e8f7841724621ef124ea4df543764e7a4d87216a76a96385bff44`

No compressed arm was launched automatically. A next experiment requires a
separate frozen matched protocol and user confirmation.
