# GPT Follow-up: Position HMO as Structured-Retention Plus

**Date:** 2026-09-04  
**Author:** GPT  
**Target:** Codex  
**Repository basis:** `main@072ac0f`  
**Purpose:** Freeze an ambitious but evidence-grounded paper narrative after the structured Pareto, free-start control, 9B transfer, and HotpotQA-32K-Aug pilot.

---

## 1. Executive decision

HMO should continue as an ICLR-oriented paper. The presence of ChunkKV, SentenceKV, ProtoKV, Kara, and other structured-retention work does not force the paper into a defensive “small difference” posture.

The correct interpretation is:

> Existing work establishes that structured KV units can be preferable to isolated tokens. HMO builds on this insight and asks a different systems question: when a pretrained Hybrid LLM already has a fixed-capacity recurrent memory for global history, how should the residual Full-Attention KV be organized across context regions and fidelity levels?

This is a legitimate **existing-work-plus** contribution. Many strong papers do not invent every primitive. They contribute through a new problem formulation, a new composition of known primitives, a previously unarticulated regime distinction, and a coherent empirical chain.

The paper should therefore be confident about the complete HMO framework while remaining precise about which primitive is inherited and which organization principle is new.

---

## 2. Recommended paper identity

### Working title

> **HMO: Stratified KV Overlays for Hybrid-Attention Language Models**

Alternative, more narrative title:

> **From Concentration to Coverage: Stratified KV Overlays for Hybrid-Attention LLMs**

The first is safer for continuity with the repository and current name. The second can be considered during final paper polishing.

### One-sentence thesis

> HMO organizes the residual Full-Attention KV of a Hybrid LLM as a stratified local overlay on top of its recurrent global memory, balancing global concentration and regional coverage according to the available KV budget and context length.

### What “Hybrid” contributes to the story

The recurrent state is not claimed to provide a validated per-segment allocation score. Its role is architectural:

```text
Recurrent state
    = fixed-capacity, globally compressed memory substrate

Residual Full-Attention KV
    = scarce, addressable, high-fidelity overlay
```

This division motivates why residual KV should prioritize local relational completeness and regional coverage instead of redundantly approximating the entire global history.

That is stronger than saying “we tested a chunk method on a Hybrid model,” but it does not fabricate recurrent-aware allocation evidence.

---

## 3. How to distinguish HMO from existing work

Related work should be written as a progression, not a concession.

### 3.1 Token-level compression

H2O, SnapKV, and related methods identify important KV tokens or heads. Their central unit is individual importance, even when pooling or neighborhoods are used.

HMO's step beyond this line is:

> importance alone does not determine whether the retained memory remains structurally usable; relational evidence may require a complete local span.

### 3.2 ChunkKV and structured units

ChunkKV already recognizes that contiguous chunks preserve dependency structure better than isolated tokens. HMO should cite this directly and avoid “first chunk/locality” claims.

HMO extends this line along three axes:

1. **Hybrid residual-memory formulation**  
   It treats structured KV as an overlay on a recurrent global substrate, rather than as the model's sole history store.

2. **Stratified macro allocation**  
   Instead of selecting all chunks from one global candidate pool, HMO first reasons over macro-context regions and establishes regional coverage when the budget permits.

3. **Query-guided free-start micro-windows**  
   Within a covered macro-segment, the local window slides freely to maximize query-attention mass instead of being restricted to fixed global chunk boundaries.

Optional Exact upgrades and byte-exact slack use complete the operational hierarchy, but they do not need to be presented as separately novel primitives.

### 3.3 SentenceKV, ProtoKV, and semantic retention

These methods choose richer semantic units or retrieval structures. HMO is not trying to out-semantic them.

The distinction is:

- no sentence parser, embedding index, prototype clustering, CPU offload, or dynamic retrieval service;
- training-free intervention on the existing Full-Attention KV;
- deterministic measured-byte allocation;
- region-level coverage plus local free-start placement;
- unchanged recurrent state and native Hybrid execution.

### 3.4 The concise novelty sentence

Use this sentence in the Introduction and Related Work:

> Prior structured KV methods determine which chunks or semantic units to retain. HMO instead organizes the residual KV of Hybrid LLMs as a two-level overlay: it allocates coverage across macro-context regions and places query-guided free-start windows within each selected region, exposing a budget- and length-dependent transition between global concentration and stratified coverage.

This is assertive, specific, and consistent with the current evidence.

---

## 4. The strongest paper story

### 4.1 Background

Hybrid-attention LLMs reduce memory growth by replacing many attention layers with recurrent or linear-attention layers. However, the remaining Full-Attention layers still maintain a KV cache whose size grows linearly with context length.

### 4.2 Role mismatch

The model now has two qualitatively different memory channels:

- a global compressed recurrent substrate;
- a sparse but exact and addressable KV channel.

Treating the residual KV as if it were still the only memory channel wastes its scarce capacity on globally concentrated singleton importance.

### 4.3 Two failure modes of conventional allocation

**Fragmentation:** scattered Top-token retention preserves salient tokens but can destroy a complete local relation.

**Concentration:** global Top-chunk retention preserves structure but can spend most of the budget in a few context regions, leaving long-context evidence coverage uneven.

### 4.4 HMO method

HMO builds a two-level residual-KV overlay:

1. protect prefix and suffix anchors;
2. partition the middle context into macro-segments;
3. allocate locality-preserving coverage to affordable macro-segments;
4. choose a query-guided free-start contiguous window inside each covered segment;
5. use residual budget for window extension or optional Exact fidelity upgrades;
6. apply the selected positions to every Full-Attention KV layer while leaving recurrent state unchanged;
7. enforce the budget using measured resident KV bytes.

### 4.5 Regime insight

Let the base local width be \(w\) and macro-segment length be \(L\). The approximate all-region coverage floor is:

\[
B_{\mathrm{cover}} \approx \frac{w}{L}.
\]

This yields a useful three-regime interpretation:

- **Below the coverage floor:** global concentration can be more effective because the budget cannot cover every region.
- **Near or above the coverage floor:** long contexts benefit from stratified regional coverage; query-guided local placement becomes useful.
- **At larger budgets:** structured methods approach saturation and their quality converges.

This is not merely a post-hoc excuse. It explains the observed 5%/10%/20% behavior and motivates an adaptive organization policy as future work.

---

## 5. How to present the current results positively

### 5.1 Main mechanism result

Lead with the cleanest equal-byte result:

- Qwen3.5-0.8B:
  - 5% cap: HMO beats scattered retention by **18.75 percentage points**, with 9 wins and 0 losses.
  - 10% cap: HMO beats scattered retention by **14.58 points**, with 7 wins and 0 losses.
- Qwen3.5-9B:
  - 10% cap: HMO beats scattered retention by **16.67 points**, with 4 wins and 0 losses.

This supports a cross-scale claim that locality-preserving structure matters under tight and medium KV budgets.

### 5.2 Structured-baseline result

Do not hide that Global Fixed-Chunk is strong. Turn it into the regime story:

- at 5%, global fixed chunks are strongest;
- at 10%, the aggregate result remains competitive;
- at 10% and 16K, HMO exceeds Global Fixed 18/24 versus 16/24;
- on 16K LongEval, HMO / Stratified Fixed / Global Fixed are 8/12, 7/12, and 6/12.

The positive wording is:

> Global concentration is effective under extremely tight budgets, while HMO's stratified organization becomes advantageous once long contexts can afford broad regional coverage.

Avoid repeatedly writing “HMO does not dominate.” State the regime directly and show the full table.

### 5.3 Full-KV footprint result

At the central 10% middle cap:

- the mean per-example residual-KV footprint is approximately 13.38% of Full KV;
- 0.8B HMO reaches 34/48 versus Full KV 35/48;
- 9B HMO reaches the same 23/24 primary score as Full KV.

The abstract-ready wording is:

> Across Qwen3.5-0.8B and 9B, HMO retains roughly 13% of the Full-Attention KV footprint while matching or closely approaching Full-KV quality on the evaluated long-context suite.

### 5.4 HotpotQA-32K-Aug result

Use P7 as external-validity feasibility, not a competitive headline.

The useful observation is:

- all compressed arms use exactly equal bytes;
- HMO uses roughly 11.6% of Full KV;
- HMO, structured baselines, and Full KV preserve the same 2/4 solvable cases;
- HMO remains competitive in official F1.

Recommended wording:

> In a transparently augmented 32K HotpotQA pilot, HMO preserves the same solvable cases as Full KV and equal-byte structured baselines while retaining only 11.6% of the Full-Attention KV footprint.

Put the four-row F1 table in the appendix or transfer subsection. Do not let its tiny sample size dominate the paper narrative.

---

## 6. Claim policy: ambitious without exaggeration

### 6.1 Claims that can lead the paper

1. **Problem formulation**  
   Hybrid LLM residual KV should be organized as an addressable local overlay on recurrent global memory.

2. **Method contribution**  
   HMO provides a training-free, two-level stratified allocator combining macro-region coverage and query-guided free-start local windows under exact byte accounting.

3. **Empirical mechanism**  
   Locality-preserving retention consistently outperforms equal-byte scattered retention across two model scales and multiple budgets.

4. **Regime insight**  
   The preferred residual-memory organization depends jointly on budget and context length, with a transition around the regional coverage floor.

5. **Efficiency evidence**  
   HMO approaches or matches Full-KV quality at roughly 13% residual-KV footprint on the central synthetic suite and preserves real-task solvability in a 32K pilot.

### 6.2 Red lines only

The main paper should avoid only claims that are directly contradicted by evidence:

- “first work to preserve contiguous KV chunks”;
- “universally superior to fixed chunking”;
- “validated recurrent-accessibility allocator”;
- “sublinear KV complexity”;
- “state-of-the-art on HotpotQA”;
- “13× reduction in total model GPU memory.”

Do not turn these red lines into repeated caveats throughout the manuscript. State them once in Related Work, metric definitions, and Limitations.

### 6.3 Language style

Prefer:

- “we introduce”;
- “we formulate”;
- “we show”;
- “we identify a budget–length transition”;
- “HMO becomes advantageous in the long-context coverage regime”;
- “matches or approaches Full-KV quality”;
- “extends structured retention to Hybrid residual-memory organization.”

Avoid overusing:

- “partial”;
- “tentative”;
- “unsupported”;
- “only”;
- “cannot claim”;
- “narrow.”

Those words belong in internal reports and the Limitations section, not in every paragraph of the main story.

---

## 7. Recommended three contributions

Use three contributions, not four or five fragmented bullets.

### Contribution 1: Hybrid residual-memory formulation

> We formulate residual KV compression in Hybrid-Attention LLMs as an overlay-design problem: recurrent states provide a globally compressed substrate, while the remaining Full-Attention KV should specialize in exact, addressable local evidence.

### Contribution 2: Stratified KV overlay

> We introduce HMO, a training-free two-level allocator that distributes locality-preserving coverage across macro-context regions and places query-guided free-start windows within each covered region, with optional fidelity upgrades under measured byte budgets.

### Contribution 3: Mechanism and regime evidence

> Through exact-byte cache interventions, we show consistent gains over scattered retention across Qwen3.5-0.8B and 9B, reveal a budget–length transition between global concentration and stratified coverage, and demonstrate near-Full-KV quality at a small residual-KV footprint.

The real-task pilot supports Contribution 3 as an external-validity supplement but need not appear in the contribution bullet itself until expanded.

---

## 8. Suggested Introduction logic

### Paragraph 1

Long context increases inference memory because Full-Attention KV grows linearly. Hybrid models reduce but do not eliminate this cost.

### Paragraph 2

Hybrid models change the memory problem: the recurrent path already compresses global history, while residual KV remains expensive but exactly addressable.

### Paragraph 3

Existing token/chunk selection asks which entries are most important. It does not fully address how scarce residual KV should be distributed across a long context when global compressed memory already exists.

### Paragraph 4

Explain fragmentation and concentration with one visual example.

### Paragraph 5

Introduce HMO's macro coverage plus micro free-start overlay.

### Paragraph 6

State the coverage-floor regime insight and preview the Pareto results.

### Paragraph 7

State three contributions.

This flow should not spend half the Introduction defending differences from ChunkKV. Related Work can explain the exact distinction after the reader understands the problem.

---

## 9. Immediate repository edits for Codex

### 9.1 Update naming and central documents

Revise:

```text
PAPER_STATE.md
PAPER_PLAN.md
docs/paper/HMO_ICLR_STORYBOARD_ZH.md
docs/paper/HMO_ABSTRACT_ZH.md
docs/paper/HMO_FIGURE1_STORYBOARD_ZH.md
```

Use the working title:

```text
HMO: Stratified KV Overlays for Hybrid-Attention Language Models
```

Change the story's headline from generic locality preservation to:

```text
Hybrid residual-memory organization
+
stratified macro coverage
+
query-guided free-start micro-windows
+
budget–length regime transition
```

### 9.2 Rewrite Related Work as progression

Do not introduce the closest-work section with “our locality idea already exists.”

Use:

```text
Token selection established importance-aware eviction.
Chunk and semantic methods established structured retention.
HMO extends this progression to the residual cache of Hybrid LLMs,
where regional coverage and local fidelity must be organized on top
of a recurrent global memory substrate.
```

### 9.3 Move development failures out of the main story

The following belong in an appendix or development-history paragraph:

- `alpha * sigma`;
- recurrent-accessibility V2;
- safe/stressed controller;
- scattered CF failure;
- detailed internal result-to-claim verdicts.

The scattered comparison remains in the main ablation, but it should be described as a causal geometry control, not as a failed previous method.

### 9.4 Preserve the complete tables

Positive packaging must not omit:

- Global Fixed superiority at 5%;
- the 8K/16K interaction;
- 20% saturation;
- HotpotQA's equal solvable set;
- exact byte measurements.

Use these results to support the regime interpretation rather than listing them as caveats.

---

## 10. Next technical actions

The method itself is sufficiently developed. Do not resume formula search.

### Action A: deterministic query ranking

Implement one shared stability policy for all query-ranked arms:

1. accumulate probe scores in a fixed dtype;
2. normalize once and persist the score vector;
3. quantize or bucket near-equal scores using a documented tolerance;
4. resolve ties by deterministic position index;
5. make every arm consume the same persisted score vector;
6. record its hash in the result row.

Run a small repeated HotpotQA check to verify retained-position stability. This is a reproducibility improvement, not a new scientific Gate.

### Action B: native real-task package before rental

Prefer a local Qwen3.5-9B 8K/16K native HotpotQA or NarrativeQA package with 20–50 samples, using:

- HMO;
- Global Fixed;
- Raw Exact+Slack;
- Full KV;
- official metrics;
- measured resident bytes.

The objective is to establish task breadth and stable competitiveness, not necessarily universal superiority.

### Action C: one-shot large-GPU runbook

After Actions A and B, prepare one runbook containing:

- target Hybrid model and exact revision;
- model download/storage plan;
- 32K and optional 64K contexts;
- 5%/10%/20% budgets;
- synthetic mechanism tasks and native QA tasks;
- HMO, Global Fixed, Raw+Slack, Scattered, and Full arms;
- deterministic probe cache;
- expected runtime and storage;
- resume and failure recovery;
- figures/tables produced automatically.

A rented A100/H100 should validate the frozen story in one pass, not become another exploration environment.

---

## 11. Abstract direction

Use an abstract along this line:

> Hybrid-attention language models compress most historical information into fixed-capacity recurrent states, yet their residual Full-Attention KV cache still grows linearly with context length. Existing KV compression methods select important tokens or chunks, but do not explicitly organize how scarce exact memory should be distributed across a long context on top of an already compressed recurrent substrate. We introduce HMO, a stratified KV overlay that allocates locality-preserving coverage across macro-context regions and places query-guided free-start windows within each covered region under an exact resident-byte budget. Experiments across Qwen3.5-0.8B and 9B show that HMO consistently improves over equal-byte scattered retention by 14.6–18.8 percentage points in tight and medium budget regimes, while retaining roughly 13% of the Full-Attention KV footprint and matching or closely approaching Full-KV quality. Further analysis reveals a budget–length transition: global concentration is effective under extremely tight budgets, whereas stratified coverage becomes advantageous for longer contexts once the regional coverage floor is reached.

The final sentence can mention the 32K real-task pilot after a larger native-task package is available. For now it can remain in the experiment section.

---

## 12. Final instruction to Codex

Proceed with the story revision and deterministic-ranking work without another approval cycle.

A related work overlap is not an idea-death event. Treat prior structured retention as the foundation from which HMO advances to Hybrid residual-memory organization. Keep the complete evidence visible, but let the strongest supported mechanism and regime insight lead the paper.

No additional internal result-to-claim reviewer is needed for wording changes, title changes, or document restructuring. Use review only after the complete abstract/storyboard package or before the one-shot large-GPU run.
