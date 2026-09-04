# GPT/PZ 9B Follow-up Assessment

## Verdict

I support GPT's scientific reading and PZ's resource strategy, with one
execution-order adjustment:

1. keep the locality-preserving overlay story frozen;
2. strengthen the design and closest-work positioning before more GPU work;
3. complete the inexpensive small-model evidence package;
4. prepare one consolidated large-model runbook before renting an A100-class
   GPU;
5. do not begin a full LaTeX draft yet.

This is not a stricter scientific gate. It is a cost-control and narrative-
coherence sequence.

## What The 9B Result Now Establishes

The main mechanism is no longer a one-model observation:

| Model | Contiguous | Equal-byte Scattered | Delta | Paired W/L |
|---|---:|---:|---:|---:|
| Qwen3.5-0.8B | 34/48 | 27/48 | +14.58 pp | 7/0 |
| Qwen3.5-9B | 23/24 | 19/24 | +16.67 pp | 4/0 |

Both runs use real cache intervention and measured resident KV bytes. This is
already a credible cross-scale mechanism chain for synthetic local evidence.

The 9B result also clarifies the method hierarchy:

- locality-preserving Sparse retention is the mandatory structural core;
- Exact is an optional budget action;
- recurrent state is the unchanged global-memory base;
- recurrent accessibility is not a validated allocator signal;
- Raw Exact is a strong baseline, not a method HMO currently claims to beat
  consistently.

"Mandatory coverage" should not be phrased as every segment always receiving a
window. At width 16 and a 5% cap, full segment coverage is arithmetically
impossible because the coverage floor is about 6.25%. The precise contract is:
coverage actions preserve contiguous local structure; the allocator covers
highest-demand segments first when the cap cannot cover every eligible segment,
then performs optional Exact upgrades when budget remains.

## Assessment Of GPT's Suggestions

### Supported

- Freeze the current allocator and stop searching recurrent scores.
- Keep Exact optional in the abstract and method.
- Add Raw Exact+Slack to all subsequent budget comparisons.
- Use measured resident bytes as the Pareto x-axis.
- Avoid Qwen3.5-9B BF16 at 32K on the current 32GB card.
- Audit chunk/window/page-preserving KV work before locking novelty claims.
- Build Figure 1 and the method story without waiting for every experiment.

### Adjustment

GPT suggests immediately starting Introduction/Method/Theory in LaTeX. PZ
prefers to make the framework and packaging solid first. I recommend a middle
course: complete a compact design dossier, contribution boundary, closest-work
matrix, method pseudocode, theorem assumptions, and Figure 1 storyboard now;
defer full prose drafting until these agree.

### Format-Robust Secondary Metric

The primary metric must remain unchanged. A post-hoc, explicitly secondary
task-aware canonicalization is reasonable:

- canonicalize clock answers such as `8:38`, `8:38 o'clock`, and
  `838 o'clock` to one representation;
- keep LongEval alphanumeric identifiers exact after ordinary normalization;
- apply the same deterministic rule to every system and both model sizes;
- report it beside, never in place of, the frozen primary metric.

This removes a known presentation artifact without rewriting the formal result.

## Small-Model Evidence Package Before Renting A Large GPU

This is a readiness checklist, not an outcome gate.

| Evidence block | Current state | Remaining work |
|---|---|---|
| Mechanism | Complete on 0.8B and 9B | Package cross-scale table/figure |
| Byte fairness | Complete at 10% on 9B | Extend Raw+Slack across Pareto |
| Budget behavior | One 10% point | Run 5%/10%/20% on the 0.8B 48-case suite |
| Real task | Missing for current method | Small HotpotQA transfer |
| Novelty boundary | Partial local notes | Audit recent chunk/window/page methods |
| Reproducibility | Clean commits, frozen protocols, raw paths | Consolidate runbook and artifact manifest |

A large-model rental becomes worthwhile once the method and baseline matrix are
fixed and the small-model package shows a readable curve plus at least one
non-synthetic task. The curve does not have to win at every budget, and the real
task does not have to be statistically conclusive; they must tell us which
configuration to run once on rented hardware.

## Recommended Order

### Package A: zero GPU

1. Closest-work audit focused on query-guided chunks, block/page KV selection,
   local-window preservation, span-aware compression, and Hybrid-attention KV
   management.
2. Produce a distinction matrix: selection unit, coverage guarantee, query
   conditioning, Hybrid memory role, real-byte accounting, and decode behavior.
3. Freeze the method wording and Figure 1 storyboard.
4. Add the format-robust secondary analysis offline.

### Package B: local GPU1

Run the 0.8B 48-case 5%/10%/20% Pareto with fixed width 16 and the five useful
arms requested by GPT. The previous 48-case run took about 608 seconds; three
budget points should remain a modest local-GPU package rather than an A100
expense.

### Package C: real task

First test 0.8B Full KV on 32K HotpotQA only for solvability. If the model lacks
task capability, use the already downloaded 9B model at 8K/16K rather than
forcing a meaningless 0.8B 32K comparison. A 4B 32K path remains an alternative
if length itself is essential.

### Package D: rented large GPU

Only after A-C, freeze one runbook covering model revision, task subset, context
lengths, systems, budgets, expected storage, estimated wall time, resume logic,
and stop-on-infrastructure-error behavior. Then rent the card and run the whole
package once.

## Immediate Decision Requested From PZ

Recommended next authorization is Package A only: closest-work audit, design
dossier, Figure 1 storyboard, and offline format-robust analysis. After seeing
that package, authorize Package B as one bounded GPU1 run. No new GPU job or
paper draft has been started by this assessment.
