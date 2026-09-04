# HMO Paper Plan

**Working title**: HMO: Stratified KV Overlays for Hybrid-Attention Language Models
**Target venue**: ICLR  
**Paper type**: Method + analysis + systems evaluation  
**Date**: 2026-09-04  
**Page budget**: 9 main-body pages, excluding references and appendix

## Claims-Evidence Matrix

| Claim | Current evidence | Status | Paper location |
|---|---|---|---|
| Hybrid residual KV is better organized as a local high-fidelity overlay on recurrent global state | Qwen3.5 hybrid execution keeps DeltaNet state fixed while intervening only on Full-Attention KV | Architecture-grounded formulation; synergy not causally established | Introduction, Method |
| Contiguous retention preserves complete span evidence better than scattered equal-cardinality retention | 0.8B Pareto: +18.75/+14.58 pp at 5%/10%, zero losses; 20% saturates. 9B 10%: +16.67 pp | Strong cross-scale, tight/mid-budget synthetic evidence | Theory, Main Results |
| HMO approaches Full-KV quality at a small KV footprint | 0.8B: 34/48 vs 35/48 at 13.38%, and both 35/48 at 23.01%; 9B: both 23/24 at 13.38% | Supported on two model sizes | Abstract, Main Results |
| HMO remains competitive with strong Raw Exact baselines | 0.8B HMO/Raw+Slack: 30/30, 34/32, 35/35 across 5%/10%/20%; 9B generated tokens identical in 24/24 | Fairness baseline established; no stable superiority claim | Main Results, Pareto |
| HMO improves over generic fixed chunks in the long-context coverage regime | At 10%/16K Global Fixed / Stratified Fixed / HMO are 16/17/18 overall and 6/7/8 on LongEval; HMO vs aligned is 2W/21T/1L at exact equal bytes | Partial, length-conditioned mechanism evidence | Main Results, Ablation |
| Exact upgrades improve contiguous coverage | 0.8B: 34/48 vs 32/48; 9B: both 23/24 with opposite one-case differences | Optional component, not primary claim | Ablation |
| The runner preserves real-task solvability at low residual-KV footprint | HotpotQA-32K-Aug: every system solves the same 2/4 cases; HMO uses 11.556% of Full-KV bytes | Four-case feasibility evidence; native-task breadth pending | Transfer section |

## Structure And Page Budget

### Section 1: Introduction, 1.20 pages

- Open with residual Full-Attention KV growth in Hybrid LLMs.
- Establish recurrent global memory versus token-level KV fidelity.
- Expose the scattered-importance failure mode.
- Present HMO as macro-region stratified coverage plus query-guided free-start
  windows and optional Exact fidelity.
- Introduce the budget-by-length transition around the regional coverage floor.
- Preview consistent 14.58/16.67 pp equal-byte gains across 0.8B/9B and near-Full quality at 13.38% footprint.
- End with three contribution bullets aligned with the matrix.

### Section 2: Background And Related Work, 0.75 page

- Hybrid/linear attention and recurrent state.
- KV eviction, heavy hitters, query-aware compression, and budget allocation.
- Local windows, chunks, sentence/semantic units, and span-structured evidence.
- Present token selection, structured retention, and HMO as a progression.
- Position HMO through Hybrid residual-memory roles, stratified macro coverage,
  and query-guided free-start micro-windows; do not claim locality or chunks as
  new.

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

### Section 4: Local Evidence And Allocation Regimes, 0.90 page

- Proposition on complete `ell`-span coverage.
- Main proof sketch via consecutive-run decomposition and merging.
- Corollary for max-attention-mass placement within the locality class.
- General retention coefficient `(cw + m(L-w) + s)/T`, with `w/L` only as
  the full-coverage approximation.
- Stylized separable-concave allocation model explaining when first-region
  coverage outranks repeated concentration.
- Move full boundary cases and weighted-span extensions to appendix.

### Section 5: Experiments, 2.80 pages

- Setup, tasks, model, metrics, and real cache intervention.
- Main result table and exact byte accounting.
- Contiguous-versus-scattered causal ablation.
- 5%/10%/20% quality-memory Pareto with Raw Exact+Slack and Global
  Fixed-Chunk Top-K.
- HotpotQA-32K-Aug feasibility and official F1, followed by a native task
  package on the final shared-probe runner.
- Qwen3.5-9B frozen transfer with Raw Exact+Slack and peak-memory reporting.
- Stratified Fixed-Chunk at 16K/10% to separate macro-segment allocation from
  free-start placement.
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
| Figure 2 | Quality-memory Pareto | HMO, Global Fixed-Chunk, Raw Exact+Slack, Scattered, Full at 5%/10%/20% | `experiments/results/PARETO_PACKAGE_B_20260904.md` | Highest |
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
stratified macro coverage, query-guided free-start windows, the span-survival
proposition, the budget-by-length transition, the 14.58/16.67 pp cross-scale
equal-byte results, and the 13.38% footprint with near-Full quality. It must not
claim sublinear memory, recurrent-aware allocation, or universal superiority
over Raw Exact or fixed-boundary chunks.

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
- Package B completed the frozen structured-baseline Pareto. Independent
  result-to-claim review returned `partial/supplement` with high confidence:
  contiguous versus scattered is supported at 5-10%, while fixed-chunk
  superiority is length-conditioned; the review requested one mechanism
  control, subsequently completed as P5.
- P5 completed that mechanism control. Independent result-to-claim review
  returned `partial/supplement` with medium confidence: under fixed stratified
  allocation, free-start is directionally +4.17 pp and all three disagreements
  occur on LongEval, but broader generalization remains unestablished.

## Next Steps

- [x] Freeze the paper story, claim ladder, theoretical core, and abstract V1.
- [x] Audit closest work and freeze the revised contribution/method boundary.
- [x] Specify Figure 1 and add the post-hoc format-robust secondary analysis.
- [x] Implement Raw Exact+Slack and complete the frozen 10% Qwen3.5-9B transfer.
- [x] Add Global Fixed-Chunk Top-K and complete the 5%/10%/20% Pareto package.
- [x] Add the 16K/10% Stratified Fixed-Chunk mechanism control.
- [x] Add the four-case HotpotQA-32K-Aug feasibility transfer.
- [x] Freeze the final method/theory contract and persistent shared-probe path.
- [ ] Complete the final deterministic 5090 validation package.
- [ ] Generate Figure 1 and the current-results plots.
- [ ] Verify citations and construct the bibliography.
- [ ] Draft LaTeX section by section.
