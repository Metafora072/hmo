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
| Contiguous retention preserves complete span evidence better than scattered equal-cardinality retention | 0.8B: 34/48 vs 27/48; 9B: 23/24 vs 19/24, always exact equal bytes | Strong cross-scale synthetic evidence | Theory, Main Results |
| HMO approaches Full-KV quality at a small KV footprint | 0.8B: 70.83% vs 72.92%; 9B: both 95.83%, at mean per-case 13.38% footprint | Supported on two model sizes | Abstract, Main Results |
| HMO remains competitive with strong Raw Exact baselines | 9B Contiguous and Raw Exact+Slack produce identical tokens in 24/24; Raw Exact is 24/24 vs 23/24 on one formatting-sensitive case | Fairness baseline established; no superiority claim | Main Results, Pareto |
| Exact upgrades improve contiguous coverage | 0.8B: 34/48 vs 32/48; 9B: both 23/24 with opposite one-case differences | Optional component, not primary claim | Ablation |
| The result transfers to realistic tasks and larger Hybrid models | Current method transfers from Qwen3.5-0.8B to 9B; realistic 32K task remains pending | Model scale supported; task breadth pending | Transfer section |

## Structure And Page Budget

### Section 1: Introduction, 1.20 pages

- Open with residual Full-Attention KV growth in Hybrid LLMs.
- Establish recurrent global memory versus token-level KV fidelity.
- Expose the scattered-importance failure mode.
- Present HMO as query-guided contiguous coverage plus optional Exact fidelity.
- Preview consistent 14.58/16.67 pp equal-byte gains across 0.8B/9B and near-Full quality at 13.38% footprint.
- End with three contribution bullets aligned with the matrix.

### Section 2: Background And Related Work, 0.75 page

- Hybrid/linear attention and recurrent state.
- KV eviction, heavy hitters, query-aware compression, and budget allocation.
- Local windows, chunks, sentence/semantic units, and span-structured evidence.
- Treat ChunkKV as the closest method: do not claim locality or chunks as new.
- Position HMO through Hybrid residual-memory roles, coverage-first
  macro-segments, and query-guided free-start micro-windows.

### Section 3: Hybrid Memory Overlay, 1.85 pages

- Formal system model and measured resident-byte objective.
- Define `HMO family = locality-preserving coverage actions + optional
  fidelity`, allowing `m=0`; mandatory means covered segments use contiguous
  structure, not that every segment is covered under every budget.
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
- General retention coefficient `(cw + m(L-w) + s)/T`, with `w/L` only as
  the full-coverage approximation.
- Move full boundary cases and weighted-span extensions to appendix.

### Section 5: Experiments, 2.80 pages

- Setup, tasks, model, metrics, and real cache intervention.
- Main result table and exact byte accounting.
- Contiguous-versus-scattered causal ablation.
- 5%/10%/20% quality-memory Pareto with Raw Exact+Slack.
- 32K HotpotQA transfer and official F1, subject to Full-KV solvability.
- Qwen3.5-9B frozen transfer with Raw Exact+Slack and peak-memory reporting.
- Global Fixed-Chunk Top-K under the same query probe and exact target bytes,
  separating generic chunking from stratified coverage/free-start placement.
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
| Figure 1 | Hero architecture | Hybrid memory anatomy; equal-byte retention geometry; cross-scale result | `docs/paper/HMO_FIGURE1_STORYBOARD_ZH.md` | Highest |
| Figure 2 | Quality-memory Pareto | HMO, Global Fixed-Chunk, Raw Exact+Slack, Scattered, Full at 5%/10%/20% | Planned Pareto run | Highest |
| Figure A1 | Appendix mechanism plot | Answer-span survival and per-case win/tie/loss across 8K/16K | Fresh confirmation JSON | Appendix |
| Table 1 | Main results | Quality, token F1, resident bytes, footprint, task/length groups | 0.8B fresh confirmation + 9B transfer | Highest |
| Table A1 | Appendix ablation | Contiguous, Scattered, Sparse-only, optional Exact, anchors | Fresh confirmation and limited additions | Appendix |
| Table A2 | Appendix efficiency | Peak VRAM, TTFT, decode throughput, retained KV | Planned measurement | Appendix |
| Table 2 | Related-work matrix | Query-guided, contiguous evidence, per-segment coverage, Hybrid residual KV, real-byte accounting | Verified papers and method properties | High |

Figure 1 caption should state the comparison rather than only name components:
HMO uses recurrent state as global compressed memory and reserves residual KV
for query-relevant local relations; at equal bytes, contiguous coverage avoids
the evidence fragmentation caused by scattered importance retention.

## Abstract Contract

The abstract must contain the residual KV problem, dual-memory role mismatch,
contiguous overlay, span-survival proposition, the 14.58/16.67 pp cross-scale
equal-byte results, and the 13.38% footprint with near-Full quality. It must not
claim sublinear memory, recurrent-aware allocation, realistic-task transfer,
or superiority over Raw Exact.

Draft source: `docs/paper/HMO_ABSTRACT_ZH.md`.

## Citation Plan

All formal metadata and BibTeX must be verified before use.

- Introduction: Hybrid-Attention motivation, Qwen3.5 technical report,
  DeltaNet/Gated DeltaNet, and representative KV cache compression surveys.
- Related Work: H2O, SnapKV, PyramidKV, Quest, ChunkKV, SentenceKV, ProtoKV,
  Kara, Zoology, and Hypic. Initial metadata and method properties are verified
  in `docs/design/HMO_METHOD_AND_NOVELTY_DOSSIER_ZH.md`; BibTeX remains to be
  generated from primary records.
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
- Qwen3.5-9B BF16 was acquired and completed the frozen 24-case transfer on one
  RTX 5090; 16K peaked at 27.85 GiB PyTorch reserved memory.
- Independent outline review scored logical flow 8.5/10 and accepted the broader
  HMO framework. It requested that recurrent/KV division remain a design
  principle rather than a proven synergy, that the abstract name the synthetic
  suite, that fidelity remain optional, and that the main-body plan leave
  formatting headroom.

## Next Steps

- [x] Freeze the paper story, claim ladder, theoretical core, and abstract V1.
- [x] Audit closest work and freeze the revised contribution/method boundary.
- [x] Specify Figure 1 and add the post-hoc format-robust secondary analysis.
- [x] Implement Raw Exact+Slack and complete the frozen 10% Qwen3.5-9B transfer.
- [ ] Add Global Fixed-Chunk Top-K, then run the 5%/10%/20% Pareto package
  after PZ approval.
- [ ] Add 32K HotpotQA transfer after PZ approval.
- [ ] Generate Figure 1 and the current-results plots.
- [ ] Verify citations and construct the bibliography.
- [ ] Draft LaTeX section by section.
