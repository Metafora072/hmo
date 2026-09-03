# Final Proposal: Query-Conditioned Recurrent Accessibility

## Problem Anchor
- Bottom-line problem: improve exact-KV segment allocation beyond corrected raw query attention in a pretrained hybrid-attention LLM under a fixed KV budget.
- Must-solve bottleneck: context-only recurrent proxies cannot say whether the current query can retrieve a segment from DeltaNet state.
- Non-goals: no multi-action policy or scaling before selection works.
- Constraints: Qwen3.5-0.8B, existing equal-byte oracle, GPU1, less than 8 pilot GPU-hours.
- Success condition: positive frozen Top-K NDCG on independent 8K and directional 16K transfer with Needle nonnegative.

## Frozen Pilot Method
1. Decompose each recurrent layer's final context state into suffix-decayed per-segment contributions C_li.
2. Process the complete query suffix one token per forward using the real recurrent cache.
3. Read each contribution with the normalized DeltaNet query q_lt and aggregate norm, relative share, and alignment.
4. Primary score: alpha_i multiplied by one minus within-sample rank of read share.
5. Falsification score: alpha_i multiplied by within-sample rank of read share.
6. No signal weights, thresholds, task labels, or oracle values are used at inference.

## Novelty Position
The method observes the actual query-to-recurrent-state interface of an already trained hybrid LLM. It differs from context-only saturation heuristics and from learned future-utility KV predictors: the score asks whether the implicit memory channel can currently answer the same query whose explicit KV is being budgeted.

## P3 End-Task Outcome

The frozen pilot passed prospective oracle Top-K validation but failed its
subsequent equal-byte generation claim gate. At 8K, generated containment moved
from 70.83% to 75.00%; at 16K it moved from 83.33% to 75.00%. Across 48 formal
samples, the paired delta was -2.08 pp [-10.42,+6.25], with combined LongEval
wins/losses of 2/3 and no Needle change.

Therefore, the query-conditioned readout remains a validated observable for
oracle ranking in this setup, but the frozen V2 mapping is not a validated
deployable KV selector. Do not retune it on the accumulated labels. The next
proposal must first explain the mismatch between marginal oracle rankings and
actual jointly retained-set generation.
