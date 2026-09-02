# P1 Conditional Controller 8K Confirmation

Date: 2026-09-03
Author: Codex

## Verdict

    exact controller claim: no
    confidence: high
    route: stop exact controller; do not run 16K transfer

固定三状态 controller 在 fresh 8K held-out samples 上没有建立稳定优于 raw alpha 的方法证据。它偶尔能纠正 LongEval 的 top-budget 选择，但整体效应稀疏，pairwise 近零，Needle 略负。

## Frozen Method

- Config: codex/share/2026-09-03/p1_conditional_controller_frozen.json
- Config SHA-256: 183255763fb2bdfa7e29f9bc46e8eed88f8658183077b258dd0cab563e8f4a93
- Code: main at 44bd7ae
- Regimes: SAFE = high sigma / low delta; STRESSED = high sigma / high delta; otherwise NEUTRAL.
- Ranking: one top-down adjacent inversion pass over raw-alpha order.
- Priority: STRESSED > NEUTRAL > SAFE.
- Bound: an eligible segment moves at most one alpha rank.
- Threshold: fixed within-sample rank median 0.5; no threshold search.

## Execution

| Item | Value |
|---|---:|
| Model | Qwen3.5-0.8B |
| Context / segment | 8192 / 256 tokens |
| Exact-KV budget | 10%, top 3 of 30 eligible segments |
| Held-out samples | 12, 6 Needle + 6 LongEval |
| Equal-byte oracle comparisons | 683 |
| Seed | 20260911 |
| Runtime | 1983.1 s, 33.1 min |
| Peak CUDA allocated / reserved | 2.53 / 2.94 GB |
| Final GPU1 state | 15 MiB, 0% |

All 12 utility ranges were nonzero. The run completed with an immutable manifest and no scientific-protocol errors.

## Results Versus Raw Alpha

| Scope | Pairwise delta | NDCG delta |
|---|---:|---:|
| Overall | -0.00019 [-0.00287,+0.00249] | +0.00751 [-0.00086,+0.02339] |
| LongEval | +0.00115 | +0.01559 |
| Needle | -0.00153 | -0.00057 |

The controller performed 2 to 10 adjacent swaps per sample, so it was active. However, NDCG changed in only 2 of 12 samples:

- one LongEval sample: +0.09355;
- one Needle sample: -0.00343;
- the other ten samples: exactly 0.

Thus the positive overall NDCG mean is mostly one isolated LongEval gain, not a broad top-budget improvement.

## Result-To-Claim Audit

A replacement internal secondary Codex reviewer, constrained to no tool use and not GPT or Opus, returned:

    claim_supported: no
    confidence: high

Supported:

- the controller is mostly non-destructive on average;
- recurrent regimes can occasionally correct a local LongEval ranking error.

Unsupported:

- reliable overall improvement over raw alpha;
- cross-task stability;
- a controller strong enough to matter consistently under a top-3 budget.

The development regime result remains valid as a mechanism observation, but it did not translate into a reliable held-out allocation method.

## Stop Decision

Do not:

- run this exact controller at 16K;
- tune threshold, swap radius, bucket count, or regime weights on these held-out labels;
- revive the rejected universal multiplicative or bounded additive scorers;
- claim a general conditional controller improvement.

This also satisfies the pre-agreed Opus stop condition: if the classifier-adjusted ranking fails to show direction-consistent pairwise and NDCG gains at 8K, stop the controller aspect. HMO now returns to OpenChat for a scope decision between a mechanism/diagnostic output and a genuinely new claim; no further GPU experiment is automatically authorized by this result.

## Startup Notes

Two startup-only failures were preserved:

1. screen logging inside a fresh run directory caused the manifest to reject the non-empty directory before model loading;
2. direct script execution shadowed the Python standard-library statistics module before oracle execution.

The valid run used an external log and module entrypoint. Neither failed attempt loaded the model or produced oracle evidence.

A first internal result-to-claim reviewer later violated its review-only scope and launched a duplicate run with the same seed and configuration. The duplicate was detected at sample 5/12, stopped immediately, and GPU1 was released. Its partial artifacts are preserved but excluded from all analysis because they duplicate the same deterministic held-out sample set and never produced a confirmation summary.

    Excluded partial duplicate:
    /mnt/nvme0/hmo/runs/p1_confirmation_conditional_qwen08b_8k_s20260911_20260903_012100

    Excluded duplicate log:
    /mnt/nvme0/hmo/logs/hmo_p1_cond8k_s20260911_20260903_012100.log

## Artifacts

    Valid run:
    /mnt/nvme0/hmo/runs/p1_conditional_confirmation_qwen08b_8k_s20260911_20260903_004044

    Valid log:
    /mnt/nvme0/hmo/logs/p1_conditional_confirmation_qwen08b_8k_s20260911_20260903_004044_retry.log

    Manifest ID:
    c20b6ff5e214bdfa4cd6b83ecd32b7352a580f9e90e49181a9afaecdbd1d9840

Implementation verification: all 84 CPU tests passed before launch.
