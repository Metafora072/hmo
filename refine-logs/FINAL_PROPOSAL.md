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
