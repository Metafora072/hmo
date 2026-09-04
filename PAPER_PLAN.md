# HMO Paper Plan

**Working title**: HMO: A Locality-Preserving KV Overlay for Hybrid-Attention Language Models  
**Target venue**: ICLR  
**Paper type**: Method + analysis + systems evaluation  
**Date**: 2026-09-04  
**Page budget**: 9 main-body pages, excluding references and appendix

## Claims-Evidence Matrix

| Claim | Current evidence | Status | Paper location |
|---|---|---|---|
| Hybrid residual KV is better organized as a local high-fidelity overlay on recurrent global state | Qwen3.5 hybrid execution keeps DeltaNet state fixed while intervening only on Full-Attention KV | Architecture-grounded formulation; synergy not causally established | Introduction, Method |
| Contiguous retention preserves complete span evidence better than scattered equal-cardinality retention | Span-survival proposition; fresh Contiguous vs Scattered 34/48 vs 27/48 at exact equal bytes | Strongly supported in current scope | Theory, Main Results |
| HMO approaches Full-KV quality at a small KV footprint | 70.83% at mean per-case 13.38% footprint vs Full KV 72.92% | Supported on current suite | Abstract, Main Results |
| Coverage-fidelity is better than raw Exact segment Top-K | 70.83% vs 66.67%, but current method uses 1.22% more bytes | Directional; needs Raw+Slack/Pareto | Main Results, Pareto |
| Exact upgrades improve contiguous coverage | 34/48 vs Sparse-only 32/48 | Optional component, not primary claim | Ablation |
| The result transfers to realistic tasks and larger Hybrid models | Historical V6.1 results do not test the current method | Needs experiment | Transfer section |

## Structure And Page Budget

### Section 1: Introduction, 1.20 pages

- Open with residual Full-Attention KV growth in Hybrid LLMs.
- Establish recurrent global memory versus token-level KV fidelity.
- Expose the scattered-importance failure mode.
- Present HMO as query-guided contiguous coverage plus optional Exact fidelity.
- Preview 14.58 pp equal-byte gain and near-Full quality at 13.38% footprint.
- End with four contribution bullets aligned with the matrix.

### Section 2: Background And Related Work, 0.75 page

- Hybrid/linear attention and recurrent state.
- KV eviction, heavy hitters, query-aware compression, and budget allocation.
- Local windows, chunks, and span-structured evidence.
- Position HMO as query-guided local overlay rather than recent-only window or
  singleton token ranking.

### Section 3: Hybrid Memory Overlay, 1.85 pages

- Formal system model and measured resident-byte objective.
- Define `HMO family = mandatory coverage + optional fidelity`, allowing `m=0`.
- Protected anchors and segment catalog.
- Query-guided contiguous coverage.
- Optional Exact fidelity upgrade.
- Budget accounting and deterministic intervention algorithm.
- State clearly that recurrent memory is preserved but not used as a validated
  per-segment reliability score.

### Section 4: Local Evidence Analysis, 0.90 page

- Proposition on complete `ell`-span coverage.
- Main proof sketch via consecutive-run decomposition and merging.
- Corollary for max-attention-mass placement within the locality class.
- Retention coefficient `w/L + m(L-w)/T`.
- Move full boundary cases and weighted-span extensions to appendix.

### Section 5: Experiments, 2.80 pages

- Setup, tasks, model, metrics, and real cache intervention.
- Main result table and exact byte accounting.
- Contiguous-versus-scattered causal ablation.
- 5%/10%/20% quality-memory Pareto with Raw Exact+Slack.
- 32K HotpotQA transfer and official F1, subject to Full-KV solvability.
- Optional larger Hybrid model result if a checkpoint becomes available.
- Latency and peak-memory table.

### Section 6: Discussion And Conclusion, 0.65 page

- Hybrid-oriented versus Hybrid-exclusive applicability.
- Exact fidelity as an optional layer.
- Model scale and benchmark breadth.
- Accessibility as future allocation signal rather than current contribution.
- Reassert memory-role organization and local relational completeness.
- Restate the strongest quality-memory result without introducing new claims.

Total planned main body: 8.15 pages, leaving about 0.85 page of formatting and
revision headroom.

## Figure And Table Plan

| ID | Type | Content | Data source | Priority |
|---|---|---|---|---|
| Figure 1 | Hero architecture | Hybrid recurrent global state + residual KV overlay; equal-byte scattered versus contiguous evidence | Manual, current method | Highest |
| Figure 2 | Quality-memory Pareto | HMO, Raw Exact+Slack, Scattered, Full at 5%/10%/20% | Planned Pareto run | Highest |
| Figure A1 | Appendix mechanism plot | Answer-span survival and per-case win/tie/loss across 8K/16K | Fresh confirmation JSON | Appendix |
| Table 1 | Main results | Quality, token F1, resident bytes, footprint, task/length groups | Fresh confirmation + planned transfer | Highest |
| Table A1 | Appendix ablation | Contiguous, Scattered, Sparse-only, optional Exact, anchors | Fresh confirmation and limited additions | Appendix |
| Table A2 | Appendix efficiency | Peak VRAM, TTFT, decode throughput, retained KV | Planned measurement | Appendix |
| Table 2 | Related-work matrix | Query-guided, contiguous evidence, per-segment coverage, Hybrid residual KV, real-byte accounting | Verified papers and method properties | High |

Figure 1 caption should state the comparison rather than only name components:
HMO uses recurrent state as global compressed memory and reserves residual KV
for query-relevant local relations; at equal bytes, contiguous coverage avoids
the evidence fragmentation caused by scattered importance retention.

## Abstract Contract

The abstract must contain the residual KV problem, dual-memory role mismatch,
contiguous overlay, span-survival proposition, 14.58 pp equal-byte result, and
13.38%/70.83% versus 72.92% Full-KV result. It must not claim sublinear memory,
recurrent-aware allocation, or current large-model generalization.

Draft source: `docs/paper/HMO_ABSTRACT_ZH.md`.

## Citation Plan

All formal metadata and BibTeX must be verified before use.

- Introduction: Hybrid-Attention motivation, Qwen3.5 technical report,
  DeltaNet/Gated DeltaNet, and representative KV cache compression surveys.
- Related Work: StreamingLLM, H2O, SnapKV, PyramidKV, Quest, SAGE-KV,
  DuoAttention, chunk retrieval, and local attention. `[VERIFY]`
- Method: official Qwen3.5 architecture and cache semantics. `[VERIFY]`
- Theory: prior span-level or structured-pruning analyses if a direct match
  exists; do not manufacture a related theorem. `[VERIFY]`
- Experiments: LongBench and official task metrics. `[VERIFY]`

## Appendix Plan

- Full proof and boundary cases for the span-survival proposition.
- Cache intervention and logical-position protocol.
- Full per-sample results, prompts, predictions, and byte accounting.
- Development history and rejected recurrent accessibility mappings.
- Additional budget/task/seed results that do not fit the main body.

## Reviewer Feedback Already Incorporated

- GPT accepted the locality-preserving overlay story and recommended making
  Exact fidelity optional.
- The complexity statement was corrected from an invalid asymptotic reduction
  to a retention-coefficient result.
- Mean per-case footprint 13.38% is separated from ratio-of-means 13.03%.
- A local audit found no currently available 27B Hybrid checkpoint, so large
  model transfer is an enhancement rather than an immediate assumption.
- Independent outline review scored logical flow 8.5/10 and accepted the broader
  HMO framework. It requested that recurrent/KV division remain a design
  principle rather than a proven synergy, that the abstract name the synthetic
  suite, that fidelity remain optional, and that the main-body plan leave
  formatting headroom.

## Next Steps

- [x] Freeze the paper story, claim ladder, theoretical core, and abstract V1.
- [ ] Cross-review this outline and apply only the minimum coherence fixes.
- [ ] Implement Raw Exact+Slack and the 5%/10%/20% Pareto package after PZ approval.
- [ ] Add 32K HotpotQA transfer after PZ approval.
- [ ] Generate Figure 1 and the current-results plots.
- [ ] Verify citations and construct the bibliography.
- [ ] Draft LaTeX section by section.
