# Research Idea Report

**Direction**: recurrent-state-aware exact-KV allocation in hybrid-attention LLMs
**Generated**: 2026-09-03
**Ideas evaluated**: 3 generated, 3 piloted, 1 conditionally promising at 8K, 0 validated across 8K and 16K

## Ranked Ideas

### 1. Dual-confidence query accessibility
- Hypothesis: recurrent evidence should alter Top-K only when attention is diffuse and the recurrent readout disagrees with attention.
- Pilot: positive on two independent 8K seeds after freeze; final seed NDCG +0.0506, LongEval +0.1012, Needle 0.
- Limitation: frozen 16K transfer NDCG -0.0414; not length robust.
- Status: partial, discuss before more implementation.

### 2. Query-conditioned recurrent accessibility
- Hypothesis: q transpose C directly measures whether the query can retrieve a segment from DeltaNet state.
- Pilot: direct mapping failed overall but exposed strong LongEval/Needle asymmetry.
- Status: useful diagnostic representation; mapping unresolved.

### 3. Tiny learned utility mapping
- Hypothesis: a small ridge or pairwise model can map corrected alpha and recurrent write features to oracle utility.
- Pilot: no grouped-CV gain over corrected raw alpha; pairwise logistic harmed NDCG.
- Status: eliminated in current feature space.

## Key Discovery

The legacy alpha probe was not a true hybrid continuation. Correct sequential alpha weakens the old sigma claim but leaves phi_delta top-budget diagnostics and corrected safe/stressed contrast positive.

## Suggested Execution Order

1. Obtain GPT and Opus judgment on whether to continue the scoped 8K line.
2. If continuing, design one length-normalized or learned utility mechanism before generating fresh oracle labels.
3. Require fresh 8K plus fresh 16K confirmation; do not retune on current artifacts.
