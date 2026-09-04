# Format-Robust Secondary Analysis

## Protocol

Frozen `normalized_answer_contains` remains the primary metric and no raw result
was modified. This post-hoc secondary metric first applies the primary rule, then
adds exactly one task-aware alias: for a Needle truth written as `N o'clock`, a
colonized `H:MM` prediction with an optional `o'clock`, `AM`, or `PM` suffix is
treated as the same formatting of `N`. LongEval identifiers receive no additional
aliasing.

Implementation: `experiments/phase2/e3_v2/format_robust.py`
Machine-readable output: `experiments/results/format_robust_secondary_20260904.json`

## Results

### Qwen3.5-0.8B, 48 cases

| System | Frozen primary | Format-robust secondary | Changed cases |
|---|---:|---:|---:|
| Contiguous CF | 34/48, 70.83% | 34/48, 70.83% | 0 |
| Scattered CF | 27/48, 56.25% | 28/48, 58.33% | +1 |
| Raw Exact | 32/48, 66.67% | 33/48, 68.75% | +1 |
| Sparse-only | 32/48, 66.67% | 32/48, 66.67% | 0 |
| Full KV | 35/48, 72.92% | 36/48, 75.00% | +1 |

The equal-byte Contiguous-versus-Scattered gap changes from `+14.58 pp` to
`+12.50 pp`, with 6 wins, 42 ties, and 0 losses. The single changed sample has
truth `403 o'clock`; Full, Raw, and Scattered generated `4:03 PM`.

### Qwen3.5-9B, 24 cases

| System | Frozen primary | Format-robust secondary | Changed cases |
|---|---:|---:|---:|
| Contiguous CF | 23/24, 95.83% | 24/24, 100.00% | +1 |
| Scattered CF | 19/24, 79.17% | 20/24, 83.33% | +1 |
| Raw Exact+Slack | 23/24, 95.83% | 24/24, 100.00% | +1 |
| Raw Exact | 24/24, 100.00% | 24/24, 100.00% | 0 |
| Sparse-only | 23/24, 95.83% | 23/24, 95.83% | 0 |
| Full KV | 23/24, 95.83% | 24/24, 100.00% | +1 |

The equal-byte Contiguous-versus-Scattered gap remains `+16.67 pp`, with 4
wins, 20 ties, and 0 losses. The changed sample has truth `838 o'clock`;
Contiguous, Scattered, Raw+Slack, and Full generated `8:38`.

## Interpretation

The secondary view removes the known Raw Exact formatting advantage at 9B and
leaves the central locality result intact at both scales. At 0.8B it also shows
that one apparent Contiguous advantage over several baselines was formatting,
so the strict equal-byte mechanism effect is more accurately described as
`+12.50 pp` under this secondary rule. The paper should report the frozen
primary table first and this analysis beside it or in the appendix.
