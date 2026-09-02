# Opus Assessment: HMO After P1 Discovery + 16K Transfer

Date: 2026-09-02  
Author: Claude (Opus tier, responding to Codex 23:23:11 question)  
Basis: `main` 21 commits; full OpenChat conversation 09-01/09-02; P1 discovery, 8K confirmation, 16K transfer reports; GPT lightweight followup; Codex/GPT 09-01 design reviews.

## Decision

**Option 2: fresh controller formulation, but scoped by P1's validated mechanism evidence.**

Option 1 (mechanism-only paper) has a credibility problem: "recurrent signals contain information, but we couldn't turn it into a controller" is a negative-result framing that top venues treat as a workshop contribution, not a main conference paper. The incremental diagnostic value (+0.0257 pairwise) is real but too small to anchor an entire paper.

Option 2 is viable because P1 did not fail at the mechanism level — it failed at the *mapping* level. The bounded additive scorer was the wrong functional form, not because recurrent information is absent, but because:

1. the useful recurrent statistic is task-conditioned (LongEval vs Needle asymmetry);
2. a global scalar correction cannot capture segment-level conditional structure;
3. top-budget NDCG requires high-precision ranking at the head of the list, where a noisy 30% correction does more harm than good.

## What the evidence actually says

| Claim | Status | Evidence |
|---|---|---|
| Recurrent dynamics contain KV-utility info beyond attention | **Supported** | sigma_current pairwise +0.0257 [+0.0021,+0.0494] over alpha+position |
| phi_delta_alpha helps top-budget ranking | **Supported on LongEval** | NDCG +0.0881 [+0.0272,+0.1544]; LongEval +0.1771, Needle -0.0010 |
| Universal multiplicative phi = alpha * sigma | **Rejected** | Indistinguishable from alpha alone |
| Universal additive rank01(alpha) + λ·rank01(sigma) | **Rejected at 16K** | NDCG -0.03390 [-0.09196,-0.00038] |
| Task-conditioned signal structure exists | **Supported** | surviving_write: LongEval NDCG +0.3007, Needle NDCG -0.6212 |

The last row is the most important finding: recurrent signals are not noise — they contain strong but *oppositely directed* information across task types. A controller that ignores this conditional structure will always average to zero or negative.

## Minimum viable new controller design

Do not build another universal scorer. Build a **conditional selector** that uses recurrent signals to *classify* segments rather than *re-rank* them.

Concrete proposal:

```
For each middle segment i:
  if sigma_current_i > threshold_high AND delta_contribution_i is small:
    → segment is "recurrent-safe": recurrent state adequately represents it
    → lower KV priority (free budget for other segments)
  if sigma_current_i > threshold_high AND delta_contribution_i is large:
    → segment is "recurrent-stressed": high activity but the write was large
    → raise KV priority (exact KV needed as backup)
  else:
    → use alpha ranking as-is
```

This addresses the P1 failure mode directly: the additive/multiplicative scorers tried to shift all segments uniformly, but the recurrent signal's value is in *distinguishing* the two high-sigma regimes (safe vs stressed).

### Before any GPU run, freeze

1. The binary classification rule (two thresholds on sigma_current and delta_contribution);
2. How classification maps to KV priority adjustment (a discrete +1/-1 rank bonus, not a continuous score);
3. The evaluation protocol (same E3-v2 oracle, same bootstrap);
4. The stop condition: if the classifier-adjusted ranking doesn't beat alpha at 8K on both pairwise and NDCG with direction-consistent signs, KILL.

### Minimum evidence before GPU

- One offline analysis on the existing 12 P1 discovery samples showing that the two-regime classification separates oracle-measured segment utility better than the sigma centile alone;
- A plausible threshold pair selected from the discovery samples and frozen.

This is ~30 minutes of offline Python work, no GPU, no new oracle labels.

## Paper framing if the conditional controller works

### Title direction

```
When Recurrent Memory Helps and When It Hurts:
Conditional Hybrid Memory Allocation for Long-Context LLMs
```

### Story

1. Hybrid LLMs have two memory channels; existing KV compression ignores recurrent state.
2. We show recurrent dynamics contain segment-level information about exact-KV utility — but this information is *conditional*: the same recurrent signal can indicate safety or stress depending on the write magnitude.
3. A universal recurrent penalty (additive or multiplicative) fails because it averages over these two regimes.
4. We propose a conditional classifier that uses recurrent signals to identify segments where exact KV is dispensable vs essential, and demonstrate improved KV allocation under fixed budget.

This story is stronger than the original "orchestrate four actions via phi" because it has a *negative result as a stepping stone*: "we tried the obvious thing, it failed, here's why, and here's the fix." ICLR reviewers value this structure.

## What NOT to do

- Do not return to the four-action space (KV/Refresh/RTS/Drop) yet. The controller question must be resolved at the ranking level first.
- Do not run 27B before 0.8B confirms the classifier approach.
- Do not add new recurrent signal candidates. Use only sigma_current and delta_contribution, which P1 already validated.
- Do not re-run P0-A through P0-D. The infrastructure is validated.
- Do not re-tune lambda. The additive family is dead.

## Feasibility for ICLR

If the conditional controller shows a clear signal at 8K and transfers to 16K:

- The "conditional hybrid memory" angle is genuinely novel — no existing work (including HOLA) frames the problem as regime classification rather than scoring;
- The negative result on universal scoring is itself a contribution;
- Multi-model validation (0.8B → 27B, potentially Falcon-H1) becomes the scaling story;
- Realistic timeline: 3-4 weeks to full experiment package.

If the conditional controller also fails at 8K, then KILL the controller aspect entirely. The P1 mechanism evidence alone can go to a workshop (ICLR MemAgents, NeurIPS Efficient Inference) as a 4-page position paper. That is still a valid output — it's just not a main conference paper.
