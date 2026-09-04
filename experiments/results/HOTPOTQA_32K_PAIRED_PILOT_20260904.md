# HMO P7: HotpotQA-32K-Aug Equal-Byte Paired Pilot

Date: 2026-09-04

## Question And Protocol

Does HMO's locality-preserving retention remain competitive with structured KV
retention baselines on the same transparently augmented 32K HotpotQA cases where
Qwen3.5-0.8B showed some Full-KV solvability?

The frozen pilot uses a 10% middle-context cap, 256-token segments, 16-token
base Sparse width, and one protected prefix and suffix segment. Contiguous CF,
Global Fixed-Chunk Top-K, Raw Exact+Slack, and Scattered CF use exactly equal
measured post-query resident attention-KV bytes. Full KV is reused from the
SHA-pinned P6 parent and is revalidated against each reconstructed context,
question, answer set, token count, byte count, and official score.

- Model: `Qwen/Qwen3.5-0.8B`
- Model revision: `2fc06364715b967f1860aea9cf38778875588b17`
- Protocol: `refine-logs/hotpotqa_32k_paired_protocol.json`
- Protocol SHA256:
  `1af7d542023c812b15f5e5f74f60dfab951a62becdacae37052f56a789c19268`
- Formal run:
  `/mnt/nvme0/hmo/runs/hotpotqa_32k_paired_formal_20260904_174200/`
- Manifest code commit: `ddd189f36a4df0c6439b53445ec22e86b42bc7bc`
- Primary metric: official LongBench QA F1

This remains a four-case pilot on augmented rather than native 32K examples. It
is not a benchmark-level estimate.

## Raw Results

| Case | HMO | Fixed | Raw+Slack | Scattered | Full KV |
|---|---:|---:|---:|---:|---:|
| base 29 + donor 168 | 0.7273 | 0.3333 | **1.0000** | **1.0000** | 0.3333 |
| base 119 + donor 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| base 84 + donor 96 | **0.6154** | 0.5926 | 0.5926 | **0.6154** | 0.5926 |
| base 108 + donor 30 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **Mean F1** | **0.3357** | 0.2315 | 0.3981 | **0.4038** | 0.2315 |

All five systems contain the normalized gold answer on the same `2/4` cases.
HMO, Fixed, and Full have `0/4` normalized exact matches; Raw+Slack and
Scattered have `1/4`. The F1 differences therefore reflect answer phrasing and
verbosity on already-solved cases, not access to additional answers.

## Paired Comparisons

| Comparison | Mean F1 delta | Wins / ties / losses | Containment delta |
|---|---:|---:|---:|
| HMO vs Fixed | +0.1042 | 2 / 2 / 0 | 0 |
| HMO vs Raw+Slack | -0.0625 | 1 / 2 / 1 | 0 |
| HMO vs Scattered | -0.0682 | 0 / 3 / 1 | 0 |
| HMO vs Full KV | +0.1042 | 2 / 2 / 0 | 0 |

Mean compressed resident attention-KV is `46,611,456` bytes (`0.0434 GiB`),
or `11.556%` of the mean Full-KV `403,344,384` bytes (`0.3756 GiB`). All four
compressed arms match exactly in `4/4` cases. Each HMO plan has five Exact and
121 Sparse eligible segments, with no recurrent-only eligible segment at this
budget.

## Reproducibility Audit

The operational smoke and formal run share the same protocol and first sample.
HMO reproduced exactly: its active positions, output token IDs, and F1 are
identical. The query-ranked baselines showed small numerical ranking changes:

| System | Removed / added positions | Position Jaccard | Smoke / formal F1 |
|---|---:|---:|---:|
| HMO | 0 / 0 | 1.00000 | 0.7273 / 0.7273 |
| Fixed | 7 / 7 | 0.99626 | 1.0000 / 0.3333 |
| Raw+Slack | 1 / 1 | 0.99946 | 1.0000 / 1.0000 |
| Scattered | 24 / 24 | 0.98724 | 1.0000 / 1.0000 |

The byte target and retained-token counts remain identical. This is consistent
with near-tied query-probe scores changing rank under GPU numerical variation.
Formal results remain the preregistered evidence, while the Fixed comparison
must be treated as directional rather than robust. A future confirmation should
make ranking deterministic or report repeated-run variance.

## Claim Decision

Independent result-to-claim review returns `partial`, routes to `supplement`,
and assigns medium confidence. The supported wording is:

> In a preregistered four-case transparently augmented 32K HotpotQA pilot on
> Qwen3.5-0.8B, HMO at exactly matched compressed resident bytes preserved the
> same solvable cases as structured baselines and Full KV. It beat Global Fixed
> in the formal run while trailing Raw Exact+Slack and Scattered by small mean-F1
> margins driven by phrasing on already-solved cases.

This is evidence that HMO is a plausible equal-byte real-task contender at
about 11.6% of Full-KV footprint. It does not show HMO is the strongest method,
that locality wins on this real-task pilot, or that the result generalizes
beyond this model, budget, and augmentation.

## Runtime And Artifacts

The formal run completed `4/4` cases in `147.48 s` on GPU1. Peak CUDA memory was
`7.31 GiB` allocated and `9.35 GiB` reserved. The process exited with status zero
and GPU1 returned to `15 MiB`.

- Formal JSONL SHA256:
  `b1ec7b520b9410d64a204770bc100f1676c55415519455f08ab17506e687e5cd`
- Formal summary SHA256:
  `a3b86be8062d95d119b0a8f2428f20cd485cf05ab7cbd1ecd61a5ecdffd73116`
- Formal manifest SHA256:
  `b30e4efdd76a9d5a112cae762be4431c06fd29b9a33f7dea14f39cab2b40fe08`

No repeat, expanded sample set, or additional budget was started automatically.
