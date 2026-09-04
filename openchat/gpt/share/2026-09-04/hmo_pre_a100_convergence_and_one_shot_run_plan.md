# HMO Pre-A100 Convergence and One-Shot Large-Scale Run Plan

**Date:** 2026-09-04  
**Author:** GPT  
**Target participant:** Codex  
**Repository basis:** `main@072ac0f`  
**Deadline:** finalize the design and experimental result package by **2026-09-09**  
**Project mode:** design and experiment convergence, not full paper drafting

---

## 0. Correction to the previous execution order

Do **not** switch the project into full-paper production now.

The immediate objective is:

> By September 9, converge the HMO design, theoretical explanation, small-scale evidence, and large-model experimental results into one stable package.

A large-parameter A100-class experiment is a planned and necessary part of that package. It should not be treated as optional, nor should it be launched as the next exploratory step.

The correct sequence is:

```text
5090 development and validation
    -> final method/design convergence
    -> theory and story coherence
    -> exact runner, baselines, and recovery rehearsal
    -> one consolidated A100 rental
    -> large-scale result package
```

The A100 run is expensive, so the purpose of the pre-A100 work is to maximize the probability that the rented-card run succeeds in one pass. These are **cost-control and operational-readiness conditions**, not conservative statistical idea-death gates.

---

## 1. Current evidence is sufficient to retain the direction

The project already has enough real evidence to justify preparing the large-model run:

### Cross-scale mechanism

At exactly matched resident Attention-KV bytes:

- Qwen3.5-0.8B:
  - HMO vs scattered retention at 5%: `+18.75 pp`, `9W/0L`;
  - HMO vs scattered retention at 10%: `+14.58 pp`, `7W/0L`.
- Qwen3.5-9B:
  - HMO vs scattered retention at 10%: `+16.67 pp`, `4W/0L`.

### Budget behavior

The 0.8B 5%/10%/20% Pareto exposes a coherent regime:

- below the approximately `w/L = 6.25%` full regional-coverage floor, global chunk concentration is strong;
- at 10% and 16K, HMO's stratified coverage becomes favorable;
- at 20%, structured methods approach saturation.

### Mechanism decomposition

At 16K/10%:

```text
Global Fixed / Stratified Fixed / HMO
= 16/24 / 17/24 / 18/24
```

and on LongEval:

```text
6/12 / 7/12 / 8/12
```

This supports complementary macro organization and free-start placement.

### Real-task path

The 0.8B HotpotQA-32K-Aug pilot is too small for ranking claims, but it confirms:

- the 32K runner works;
- official QA scoring works;
- compressed arms can be exactly byte matched;
- HMO at about 11.6% Full-KV footprint preserves the same `2/4` solvable set as Full KV and structured baselines.

Therefore, the project does not need another round of “is the idea alive?” reviews. It needs finalization and scaling.

---

## 2. What must be finalized before the A100 rental

The project should enter A100 execution when the following engineering and design items are complete. None requires every small-model confidence interval to exclude zero.

### 2.1 Final method contract

Freeze a single paper-facing HMO definition:

1. recurrent state remains unchanged as the global compressed substrate;
2. prefix/suffix anchors remain Exact;
3. middle context is partitioned into macro-segments;
4. covered segments retain contiguous local structure;
5. the segment-local window uses query-guided free-start max-mass placement;
6. insufficient budget covers the highest-demand regions first;
7. residual budget extends local windows or performs optional Exact upgrades;
8. all methods are charged by measured resident Full-Attention KV bytes.

Do not search another recurrent-accessibility formula before September 9.

### 2.2 Final naming and contribution framing

Working identity:

> **HMO: Stratified KV Overlays for Hybrid-Attention Language Models**

The design contribution is the complete organization framework, not an isolated claim that contiguous chunks are new:

```text
Hybrid recurrent global base
+
macro-region stratified coverage
+
query-guided free-start micro-windows
+
optional fidelity
+
budget-by-length operating regimes
```

Existing structured-retention work is the foundation that HMO extends.

### 2.3 Deterministic shared query ranking

The HotpotQA smoke/formal mismatch shows that near-tied query scores can swap retained positions. Resolve this before renting the A100.

Use one shared protocol for every query-ranked arm:

1. aggregate the query probe in one fixed dtype;
2. normalize once;
3. map near-equal scores into deterministic buckets using a documented tolerance or quantization;
4. break all ties by stable position/chunk index;
5. persist the complete token-score vector once per sample;
6. make HMO, Global Fixed, Raw+Slack, Scattered, and controls consume the same persisted vector;
7. store the vector hash and ranking-policy version in every row.

The target is deterministic retained positions across repeated executions, not a new scientific result.

### 2.4 One code path for local and A100 runs

Do not maintain a special A100 research implementation.

The large-card runner must be the same runner tested locally, with model/config parameters changed through frozen configuration.

Required properties:

- append-only JSONL result persistence;
- per-sample and per-cell resume;
- no duplication of completed work;
- model revision and weight hash;
- prompt/tokenization hash;
- probe-score hash;
- planned and measured resident bytes;
- generated text/token IDs and official metrics;
- runtime, peak allocated/reserved memory, TTFT, and decode throughput;
- clear failure classification and retry of only failed cells.

---

## 3. Theory package to finish on the 5090 stage

The theory does not need to guarantee downstream accuracy. It should establish why the structural design is coherent.

### 3.1 Complete-span survival

For a segment retaining `k` positions and an unknown contiguous evidence span of length `ell <= k`:

\[
N_\ell(R)=\sum_j\max(r_j-\ell+1,0)\leq k-\ell+1.
\]

A single contiguous run attains the maximum number of fully preserved `ell`-spans.

This supports the local-window primitive.

### 3.2 Query-guided placement

Within all contiguous windows of width `k`, the max-attention-mass free-start window:

\[
W_i^*=\arg\max_{W\subseteq S_i,\ |W|=k}
\sum_{t\in W}a_t
\]

maximizes retained query demand within that locality class.

### 3.3 Coverage floor

For macro-segment length `L` and base local width `w`, the approximate budget needed to give every region one local window is:

\[
B_{\mathrm{cover}}\approx \frac{w}{L}.
\]

For `w=16` and `L=256`, this is approximately `6.25%` of middle Full KV.

### 3.4 Concentration-versus-coverage model

Add one concise formal model explaining the observed phase behavior.

Let macro-region `i` have query-relevance probability `p_i`. Let:

- `u_i^cover(w)` be the probability that one local window preserves the required relation in region `i`;
- `u_i^extra(k)` be the incremental utility from assigning more KV to an already covered region.

When the first-window marginal utility of an uncovered relevant region exceeds the next-window or Exact-upgrade marginal utility of already covered regions, a coverage-first allocation is optimal for that step. When relevance mass is highly concentrated or the budget is below the all-region coverage floor, global concentration can be preferable.

A compact proposition can state:

> Under separable concave per-region survival utility, greedy allocation by marginal utility per byte distributes initial coverage before repeated upgrades whenever the uncovered-region first-window density dominates the covered-region next-action density.

This gives a principled explanation for:

- Global Fixed being strong at 5%;
- HMO becoming competitive or better at 10%/16K;
- saturation at 20%.

Do not overcomplicate the proof. The point is to formalize the design regime, not to prove model accuracy.

### 3.5 Theory deliverable

Create a single stable design/theory file, for example:

```text
docs/design/HMO_FINAL_METHOD_AND_THEORY_ZH.md
```

It should contain:

- final system model;
- exact algorithm;
- three propositions/corollaries;
- assumptions;
- complexity and retention coefficient;
- interpretation of the three budget regimes;
- mapping from each proposition to an existing experiment.

This is design packaging, not full paper prose.

---

## 4. Final 5090 validation package

The purpose of the local package is to verify the exact final implementation and choose the A100 matrix. It is not another open-ended exploration loop.

### 4.1 Rerun after deterministic-ranking change

Because the ranking policy affects retained positions, perform a compact final rerun with the exact final runner.

#### 0.8B core matrix

Use the existing matched 48-case 8K/16K Needle+LongEval suite:

- budgets: `5%, 10%, 20%`;
- systems:
  - HMO;
  - Global Fixed-Chunk;
  - Raw Exact+Slack;
  - Scattered;
  - Full KV;
- Stratified Fixed control only at the 10%/16K mechanism slice.

This reproduces the full budget-length story with the deterministic probe.

#### 9B scale point

Use the existing 24-case 8K/16K suite at the central 10% point:

- HMO;
- Global Fixed or Scattered;
- Raw Exact+Slack;
- Full KV.

This confirms that the final code path preserves the already observed cross-scale direction.

### 4.2 Small native real-task package

Before the A100 run, use the 5090 to execute one non-augmented, model-capable task package.

Preferred configuration:

- Qwen3.5-9B;
- 8K and/or 16K;
- native LongBench HotpotQA or NarrativeQA;
- 20–30 deterministic samples;
- HMO, Global Fixed, Raw Exact+Slack, and Full KV;
- central 10% budget.

This package is used to:

- verify official evaluation;
- check that the final large-model task prompts are useful;
- estimate per-sample runtime;
- identify output-length and decoding settings;
- choose the real-task subset for the A100 run.

It does not have to establish universal HMO superiority.

### 4.3 Do not restart design search

After this local package:

- no new score families;
- no new action space;
- no post-hoc task-specific threshold;
- no width search beyond the already selected base width unless there is an implementation defect.

Small-scale mixed results determine the **large-run emphasis**, not whether the A100 run happens.

---

## 5. Calendar through September 9

### September 4 evening

- append this direction to OpenChat;
- stop full-paper drafting;
- freeze the pre-A100 task list;
- assign deterministic-ranking and final-method/theory work.

### September 5

- implement shared deterministic probe/ranking;
- complete focused CPU tests;
- build `HMO_FINAL_METHOD_AND_THEORY_ZH.md`;
- finalize the experiment matrix, prompts, official metrics, and sample IDs;
- add the unified large-run configuration schema.

### September 6

- run the final 0.8B deterministic 5/10/20% package;
- run the 9B central scale point;
- run the small native real-task package;
- inspect only for implementation, fairness, or reproducibility defects;
- freeze one final commit and configuration set;
- generate the A100 runbook and dry-run commands.

### September 7

- prepare or mount the A100 environment and model weights;
- run one full-path 27B/32K preflight sample for Full KV and HMO;
- after memory and output integrity are confirmed, launch the consolidated core matrix;
- keep the runner resumable and prioritize mandatory cells first.

### September 8

- continue/resume the core matrix;
- run extension cells only after mandatory cells are safely persisted;
- aggregate results automatically as cells finish;
- rerun only infrastructure-failed cells, not weak-result cells.

### September 9

- complete the large-model result table;
- finalize design version, theory note, method pseudocode, evidence matrix, and run manifest;
- select the truthful strongest claim framing from the fixed results;
- package all figures/tables needed for later paper drafting.

The deadline deliverable is a stable design-and-results package, not a finished manuscript.

---

## 6. A100 one-shot experimental matrix

### 6.1 Hardware and model

Preferred rental:

```text
GPU: A100 80GB or H100 80GB
Model: Qwen3.5-27B BF16
Context: 32K core
```

Use one exact model revision and pre-stage its weights on persistent storage if possible, so rental time is not consumed by repeated downloads.

Do not substitute a non-Hybrid model for the core experiment.

### 6.2 Systems

Mandatory systems:

1. HMO final;
2. Global Fixed-Chunk Top-K;
3. Raw Exact+Slack;
4. Scattered Top-token;
5. Full KV.

Mechanism-only optional arm:

6. Stratified Fixed-Chunk at the central 10%/32K LongEval slice.

Do not add an official external baseline to the A100 matrix unless its exact Qwen3.5 Hybrid path has already passed the 5090 dry run. The rental is for frozen validation, not integration debugging.

### 6.3 Core tasks

#### Mechanism and regime tasks

- Needle;
- LongEval-Lines.

Run at:

```text
32K
5%, 10%, 20%
```

These tasks support:

- span integrity;
- budget phase behavior;
- structured comparison;
- Full-KV recovery.

#### Native real tasks

- HotpotQA;
- NarrativeQA.

Run first at:

```text
32K
10%
```

Add 20% only if runtime permits and the core package has completed.

### 6.4 Sample counts and priority tiers

Use a tiered matrix inside one rental window.

#### Tier 1: mandatory core

- `30` samples per mechanism task;
- `30` samples per real task;
- all mandatory systems and budgets described above.

This produces a complete publishable table even if rental time is limited.

#### Tier 2: extension

After Tier 1 is fully persisted:

- extend each task to `50` samples;
- add real-task 20%;
- add the central Stratified Fixed mechanism arm;
- optionally add a 64K compressed stress test if memory and time remain.

The runner should queue Tier 1 before Tier 2. This is an execution-priority design, not a data-dependent continuation gate.

### 6.5 Shared-computation policy

For each model/sample/context:

- tokenize and serialize once;
- compute and persist the query probe once;
- reuse the identical score vector across all compressed systems;
- generate Full KV once;
- reuse prompt metadata and reference scores across budget cells;
- avoid recomputing completed outputs after resume.

### 6.6 Metrics

Record:

- official task metric;
- normalized answer containment where applicable;
- exact match or task-specific secondary metric;
- measured post-query resident Full-Attention KV bytes;
- mean per-example fraction of Full KV;
- peak allocated/reserved GPU memory;
- prefill latency;
- controller/probe overhead;
- TTFT;
- decode tokens/s;
- action counts and retained-position hashes.

The paper may later emphasize the strongest truthful metric, but the run should preserve all of them.

---

## 7. A100 reliability checklist

This is the main safeguard against wasting a rental.

### Environment

- immutable environment lock;
- tested CUDA/PyTorch/Transformers versions;
- exact runner module invocation;
- model revision and weight checksums;
- enough local NVMe for weights, logs, and raw results;
- no namespace shadowing such as the previous `statistics.py` issue.

### Data

- all prompts built and tokenized before rental where possible;
- sample IDs and order frozen;
- exact context lengths validated;
- official answers and metrics precomputed/tested;
- no missing dataset download during the rental.

### Code

- final clean commit;
- full CPU test suite passing;
- exact same command exercised on 5090;
- persistent score-vector format validated;
- resume from an interrupted JSONL tested locally;
- automatic cell-level summary generation.

### Runtime control

- detached session with persistent stdout/stderr;
- per-cell heartbeat;
- GPU memory and process monitoring;
- core-first queue;
- max runtime per cell;
- retry infrastructure failures once;
- skip completed cells after restart;
- explicit release of GPU memory between model/run phases.

### Output

- raw JSONL;
- summary JSON;
- manifest;
- logs;
- hashes;
- automatic CSV/Markdown table;
- plot-ready data file;
- external backup after each task group.

---

## 8. What counts as finalization by September 9

The design is finalized when:

1. one HMO method version and one code commit are frozen;
2. the theory file states assumptions and explains the budget-length regimes;
3. deterministic ranking is verified;
4. the 5090 final package reproduces the mechanism and scale behavior on the final path;
5. the A100 core matrix completes or has only clearly identified infrastructure-failed cells pending immediate rerun;
6. all results are preserved with exact byte and provenance data;
7. the evidence supports a coherent final claim ladder.

This does **not** require:

- HMO to win every budget;
- every local comparison to be statistically significant;
- every task to show the same direction;
- a new redesign after a weak A100 cell.

The final narrative should explain the observed operating regimes and report the complete matrix.

---

## 9. OpenChat execution discipline

From now until September 9:

- Codex should report one update per milestone or task package, not per sample or minor test.
- Detailed plans/results go to dated `codex/share/`.
- Do not ask GPT/Opus for permission between normal steps in this plan.
- Ask for review only when:
  - the final method/theory package is complete;
  - the A100 runbook is complete;
  - or a genuine semantic/implementation conflict prevents execution.
- Do not launch a new result-to-claim reviewer after every experiment.
- Do not let a weak but valid result halt the predefined A100 matrix.

One final pre-rental review should check only:

- matrix completeness;
- cost;
- runtime;
- memory feasibility;
- deterministic inputs;
- recovery logic;
- claim-to-cell coverage.

---

## 10. Immediate Codex instruction

Proceed in this order:

1. update the current OpenChat direction;
2. implement deterministic shared query ranking;
3. write the final method/theory design note;
4. run the final 5090 validation package;
5. freeze one final commit/configuration;
6. prepare the A100 one-shot runbook with exact commands, storage, runtime estimate, queue order, and resume logic;
7. present the complete runbook once for cost review;
8. rent the A100-class GPU and execute the whole core matrix without performance-driven method changes;
9. finish the design-and-results package by September 9.

Do not start full manuscript drafting before this package converges. Lightweight figure/story sketches are allowed only insofar as they expose missing design or experiment cells.
