# C2 Native LongBench QA Result

Date: 2026-09-04

## Frozen Scope

This package evaluates Qwen3.5-0.8B on 24 native LongBench QA records: 12
HotpotQA and 12 NarrativeQA. Source contexts, questions, and answers are
unmodified; there is no augmentation or truncation. Cases were frozen before
outcomes by selecting the longest exact serialized memory contexts in the
inclusive 8,192--16,384 token band, with source index as the tie-breaker.

The systems are HMO Contiguous CF, Global Fixed-Chunk Top-K, Raw Exact+Slack,
Scattered CF, and Full KV. The four compressed systems share the same
persistent FP32 query probe and are matched by measured post-query resident KV
bytes. The official LongBench QA F1 implementation is pinned at revision
`2e00731f8d0bff23dc4325161044d0ed8af94c1e`.

## Result

| System | All 24 F1 | HotpotQA 12 | NarrativeQA 12 | Mean Full-KV footprint |
|---|---:|---:|---:|---:|
| HMO Contiguous CF | 0.3086 | 0.3239 | **0.2934** | 12.80% |
| Global Fixed-Chunk | **0.3251** | **0.4072** | 0.2430 | 12.80% |
| Raw Exact+Slack | 0.2873 | 0.3242 | 0.2504 | 12.80% |
| Scattered CF | 0.3211 | 0.3961 | 0.2460 | 12.80% |
| Full KV | 0.2602 | 0.3242 | 0.1961 | 100.00% |

Paired mean F1 deltas for HMO are:

| Comparator | All 24 | Wins / ties / losses | HotpotQA | NarrativeQA |
|---|---:|---:|---:|---:|
| Fixed | -0.0165 | 6 / 14 / 4 | -0.0833 | **+0.0504** |
| Raw+Slack | **+0.0213** | 4 / 16 / 4 | -0.0003 | **+0.0429** |
| Scattered | -0.0124 | 6 / 13 / 5 | -0.0722 | **+0.0474** |
| Full KV | **+0.0485** | 6 / 14 / 4 | -0.0003 | **+0.0973** |

HMO is therefore competitive overall rather than uniformly best. The positive
result is task-conditioned: on NarrativeQA, HMO has the highest mean F1 and
beats every comparator; on HotpotQA, fixed chunks and Scattered are stronger,
while HMO is essentially tied with Raw+Slack and Full. Compression can exceed
Full KV on some cases because removing distracting context changes generation;
this is an empirical outcome, not a monotonicity claim.

## Integrity

- Formal run: `/mnt/nvme0/hmo/runs/c2_native_formal_f585d41/`
- Formal manifest ID:
  `3c16f77deba28781984238137c17c18bfc10edc8c76a9a7de32edfb0ba5054ba`
- Formal results SHA256:
  `235ce1146f59610f8bf2ec5ce812b292a2111a264cc592a27abb28ccc2e3d628`
- Formal summary SHA256:
  `1cd067021f69abf52bd33648c99cfe28c8831542574b3fcdec9d8692bfb30b63`
- Formal manifest SHA256:
  `b6606d979c4fa749a1f597c98a8c031b491ab23f65d28432bef594f79bbf66ca`
- Clean code commit: `f585d419ca6b65d614db87b943c500bc53df5cf5`
- Protocol SHA256:
  `86ebfa5cfdff0613e559780811887b7537d0485cbd00534193c0aac433b49e2a`
- Dataset archive SHA256:
  `cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64`
- Complete rows: 24/24; exact equal-byte rows: 24/24
- Serialized context range: 13,701--16,302 tokens
- No generation reaches its max-token limit; observed maximum is 45 tokens
- Runtime: 546.13 s on one RTX 5090
- Peak CUDA memory: 4.36 GiB allocated / 5.65 GiB reserved
- GPU1 returned to 15 MiB after completion

The two-case smoke also completed before formal execution. It is operational
only and excluded from claims.

## Claim Decision

Local result-to-claim verdict: `partial`, route `confirm C2 / proceed to C3
planning`, confidence `medium`; independent external review is pending.

Supported: HMO remains a viable, strictly equal-byte residual-KV organization
on unmodified native QA; it improves over Raw+Slack and Full in aggregate and
is the strongest tested system on the frozen NarrativeQA slice at 12.8% of Full
KV. Together with the synthetic two-scale results, this broadens the evidence
from controlled retrieval to native QA.

Not supported: unconditional HMO superiority on native QA, a universal benefit
over fixed chunks or Scattered retention, an official full-split LongBench
score, or cross-architecture generalization. The paper should present this
package as task-breadth and competitive-efficiency evidence, with the primary
mechanism claim still carried by the stronger controlled equal-byte results.
