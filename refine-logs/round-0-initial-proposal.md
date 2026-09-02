# Research Proposal: Query-Conditioned Recurrent Accessibility for HMO

## Problem Anchor
- Bottom-line problem: improve exact-KV segment allocation beyond corrected raw query attention in a pretrained hybrid-attention LLM under a fixed KV budget.
- Must-solve bottleneck: existing HMO recurrent signals are context-only proxies; they do not measure whether the current query can actually retrieve a segment from DeltaNet state.
- Non-goals: no Refresh, RTS, 27B scaling, broad benchmark table, or further hand-tuned sigma controller before segment selection works.
- Constraints: Qwen3.5-0.8B, 8K/16K contexts, GPU1, existing equal-byte oracle labels, at most 8 pilot GPU-hours.
- Success condition: a frozen method improves corrected-alpha Top-K NDCG on independent 8K and transfers directionally to 16K without harming Needle.

## Technical Gap
Corrected sequential alpha invalidates the earlier claim that sigma_current broadly adds ranking value. The remaining positive phi_delta diagnostic is query-conditioned only through alpha and does not show that recurrent memory itself is accessible to the query.

## Method Thesis
For each segment i and recurrent layer l, decompose the final context state into its surviving additive contribution C_li, then read it with every real query vector q_lt. The magnitude of q_lt^T C_li directly measures query-conditioned recurrent accessibility and can distinguish explicit-KV demand from recurrently available memory.

## Contribution Focus
- Dominant contribution: a mechanistic query-to-recurrent-memory readout for exact-KV allocation.
- Supporting contribution: corrected sequential-alpha methodology for hybrid models.
- Explicit non-contributions: no new backbone, no training of the LM, and no multi-action controller.

## Proposed Method
Context DeltaNet traces yield a segment-wise exact state decomposition whose sum reconstructs the final recurrent state. During true one-token query continuation, cloned convolution state recovers the model's normalized recurrent query without mutating inference. Segment read norm, relative read share, and alignment are averaged across query tokens and recurrent layers.

The primary training-free need score is alpha times one minus the within-sample accessibility rank. The opposite alpha-times-accessibility direction is retained only as a falsification control. A tiny learned mapping is allowed only if accessibility first shows strong out-of-fold incremental value.

## Claim-Driven Validation Sketch
- Claim 1: query accessibility adds Top-K utility information beyond corrected alpha. Test on the 12-sample discovery oracle with grouped folds.
- Claim 2: a frozen mapping improves allocation. Freeze on discovery, then evaluate without refitting on independent 8K and 16K evidence.

## Failure Modes
- If neither direct direction nor grouped incremental analysis is positive, kill this signal and stop controller tuning.
- If discovery is positive but independent 8K is not, document overfit and do not contact GPT/Opus with a proposed success.
