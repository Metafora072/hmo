# Codex Assessment: GPT Existing-Work-Plus And Pre-A100 Plan

Date: 2026-09-04

## Decision

I support the scientific direction and recommend executing a narrowed
pre-A100 convergence package. The method search should stop. HMO should be
positioned as a Hybrid residual-memory organization framework:

```text
recurrent global substrate
+
stratified macro-region coverage
+
query-guided free-start local windows
+
optional fidelity actions
```

The working title `HMO: Stratified KV Overlays for Hybrid-Attention Language
Models` is better aligned with the complete contribution than the current
locality-only title. Structured-retention prior work is a foundation, not a
reason to discard the project. The paper must still avoid claiming that HMO
invented chunk locality.

## What I Accept

1. Stop searching for another recurrent-accessibility score or allocator.
2. Make macro coverage plus free-start placement the method core; keep Exact
   upgrade optional.
3. Present Global Fixed's 5% advantage, HMO's 10%/16K advantage, and 20%
   saturation as a budget-by-length operating-regime result.
4. Treat HotpotQA-32K-Aug as runner and real-task feasibility evidence, not a
   method-ranking headline.
5. Finish one final method/theory contract and one resumable runner before
   renting a large GPU.

The official Qwen3.5 documentation confirms that the dense family uses a 3:1
Gated DeltaNet / Full-Attention stack, and the official Qwen3.5-27B model card
reports 27B parameters and native context beyond 32K. The architecture choice
therefore matches HMO's target rather than merely supplying a larger generic
LLM.

## Deterministic Ranking: Necessary Correction

GPT identified the right reproducibility problem, but score bucketing should
not be the default fix.

The current implementation already:

- computes one token-attention vector per sample and shares it across every arm
  inside a run;
- uses stable position/chunk-index tie breaks;
- casts collected attention weights to FP32 before aggregation.

The smoke/formal mismatch arises because separate runs recompute the BF16 model
probe. Near-tied raw scores can then differ before the already-deterministic
Python sorting step. Adding a tolerance or quantization grid would introduce a
new method hyperparameter and can itself change the selected positions.

The recommended contract is:

1. compute the probe once for a frozen model revision and prompt;
2. persist the token score vector as an exact FP32 `.npy` artifact with
   `allow_pickle=False`;
3. bind it to model revision, token-ID hash, query hash, attention-layer set,
   aggregation version, dtype, length, and SHA256;
4. derive segment masses and every system's ordering from that persisted
   vector;
5. use a total order `(-stored_score, position_or_chunk_index)`;
6. record probe identity and score hash in every result row;
7. test cache reuse, interrupted resume, corrupt/mismatched cache rejection,
   and identical retained-position hashes across repeated consumers.

Fresh recomputation can be retained as a diagnostic. Score bucketing should be
added only if cross-hardware recomputation, rather than artifact reuse, is a
required reproducibility target and its tolerance is frozen by a separate
sensitivity check.

## Theory Assessment

The complete-span survival proposition and max-mass-window corollary are clean
and directly tied to the scattered-versus-contiguous experiment. The coverage
floor `w/L` is useful but approximate because anchors, partial segments,
query-KV overhead, Exact upgrades, and slack change the realized total
footprint.

The proposed concentration-versus-coverage proposition is acceptable only as a
stylized allocation model. It should explicitly assume separable, monotone,
concave per-region utility and known marginal densities. HMO does not observe
the true downstream survival utility, so the proposition can explain the
organization principle but cannot prove that HMO's attention-based greedy
ordering is task-optimal.

## 5090 Package

A final local package is justified after the shared-probe implementation
passes CPU tests:

1. two-consumer plus resume reproducibility smoke on the same HotpotQA case;
2. one final 0.8B 5%/10%/20% matrix on the existing 48 frozen cases;
3. one final 9B central 10% scale point on the existing 24 frozen cases;
4. one native, non-augmented real-task package totaling 20--30 frozen
   HotpotQA/NarrativeQA cases at feasible natural lengths, with official
   metrics and no outcome-based sample filtering.

GPU1 is currently free, while GPU0 remains occupied. Qwen3.5-0.8B and 9B
weights are local. The 9B 16K path previously reached 23.62 GiB allocated and
27.85 GiB reserved on a 32 GiB RTX 5090, so the native package should not
silently expand to 9B/32K.

## A100 Plan: Feasible Direction, Unfrozen Scope

Qwen3.5-27B at 32K is a plausible target, not yet a costed executable matrix.
The historical V6.1 table reports roughly 59--63 GiB peak VRAM, but it does not
provide trustworthy nonzero latency data and does not exercise the final
reference probe/overlay runner. The current 27B weights are also absent
locally.

GPT's Tier-1 matrix contains approximately:

```text
mechanism: 2 tasks * 30 samples * (1 Full + 4 compressed * 3 budgets) = 780
real task: 2 tasks * 30 samples * (1 Full + 4 compressed at 10%) = 300
total = 1,080 generation cells, before retries and preflight
```

That may be publishable, but it is too large to label a one-shot rental without
measurement. A single A100/H100 80GB preflight must first measure:

- peak memory for Full and HMO under the final reference path;
- probe, prefill, and generation time separately;
- output storage per cell;
- resume after interruption;
- whether 80GB has sufficient operational headroom.

The runbook should price `1,080 * measured_seconds_per_cell`, then define a
smaller mandatory core and an extension tier. Rental or paid-resource launch
remains a PZ decision; a GPT plan cannot grant that authority.

## Recommended Authorized Sequence

Package C0, zero GPU:

- unify title, abstract, contribution list, storyboard, and PAPER_STATE;
- write `docs/design/HMO_FINAL_METHOD_AND_THEORY_ZH.md`;
- freeze the shared-probe artifact/schema and final validation protocols.

Package C1, implementation:

- implement FP32 probe persistence, identity validation, hashes, and shared
  consumption;
- add focused CPU tests and resume tests;
- run one repeated one-case operational smoke.

Package C2, GPU evidence:

- run the final 0.8B matrix;
- run the final 9B central point;
- run the small native real-task package.

Package C3, cost review:

- freeze one commit/config set;
- perform one rented-card 27B/32K preflight;
- report measured memory, seconds/cell, projected hours, storage, and price;
- obtain PZ confirmation for the final core/extension matrix.

No code change or GPU job was started as part of this assessment.
