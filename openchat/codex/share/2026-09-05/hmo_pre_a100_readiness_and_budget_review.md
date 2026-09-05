# HMO: 5090 Evidence Closure and One-Shot A100 Budget Review

Date: 2026-09-05  
Author: Codex  
Decision target: whether HMO is ready to spend one formal A100-80GB window on
Qwen3.5-27B, without a separate paid preflight.

> Implementation note: this is the review snapshot that preceded the approved
> changes. The executed state, validation evidence and final protocol hashes are
> recorded in `hmo_pre_a100_package_implementation_report.md`.

## 1. Executive verdict

**Current status: conditionally ready, but not yet fully closed for an algorithmic
A-conference submission.**

HMO already has a coherent problem, a frozen algorithm, an honest theory line,
strict resident-byte accounting, two-scale 5090 mechanism evidence, and a highly
reproducible execution package. These are enough to stop searching for a new
method and to preserve the current paper story.

The remaining weakness is mainly experimental coverage, not method invention:

1. the native-task evidence is only 24 examples from two LongBench datasets;
2. the final method has not yet been compared with a faithful public strong
   baseline such as SnapKV under the same measured resident-byte contract;
3. the current post-prefill implementation proves persistent Full-Attention KV
   reduction, but not proportional peak-VRAM or end-to-end throughput gains;
4. all final-model evidence is within the Qwen3.5 family, so architecture-wide
   generalization cannot yet be claimed.

The recommended path is therefore:

> finish one compact, high-value 5090 evidence round; freeze the paper claims and
> tables; then use a **single paid A100 window** for the 27B/32K scale-confirmation
> matrix. The first frozen formal case performs inline operational admission and
> immediately continues. It is not a separate preflight and not a result gate.

## 2. Readiness audit

| Dimension | Status | Evidence already present | Remaining issue |
|---|---|---|---|
| Problem and motivation | Satisfied | Hybrid models retain fixed-size recurrent state but residual Full-Attention KV still grows with context; HMO organizes that residual KV as a locality-preserving overlay. | Quantify whole-model memory benefit at batch/context regimes where residual KV is material. |
| Algorithm | Satisfied | Frozen protected prefix/suffix, segment demand, coverage-first allocation, free-start max-mass windows, optional Exact upgrades, and shared retained positions across Full-Attention layers. | Do not reopen allocator search before the scale run. |
| Theory | Satisfied within stated scope | Contiguous retention maximizes the number of fully surviving single-span placements at fixed cardinality; max-mass placement is optimal inside the contiguous-window class; separable concave regional utility supports greedy marginal allocation. | The theory does not prove attention score equals task utility or guarantee downstream correctness; state this explicitly. |
| Synthetic mechanism evidence | Strong | On Qwen3.5-0.8B, HMO beats equal-byte Scattered by +18.75 pp at 5% and +14.58 pp at 10%, with zero losses. On 9B/10%, HMO is 23/24 versus 19/24, +16.67 pp and 4W/0L. | Current final suites stop at 16K; 27B/32K is the planned scale anchor. |
| Quality-memory accounting | Strong | Every central comparison uses measured post-query resident bytes. Mean sample-normalized footprint is 13.38% of Full KV on the synthetic central suite and 12.80% on native C2. | Distinguish resident KV savings from process peak VRAM. |
| Native task evidence | Partial | Frozen, unaugmented HotpotQA and NarrativeQA: HMO/Fixed/Raw+Slack/Scattered/Full F1 = 0.3086/0.3251/0.2873/0.3211/0.2602. HMO is competitive and best on the NarrativeQA slice. | Only 12 examples per dataset; no broad benchmark claim is supportable yet. |
| Ablation logic | Mostly satisfied | Scattered isolates locality, Fixed isolates free-start placement, Raw+Slack isolates structural coverage versus pure importance, and 5/10/20% establishes the budget regime. | Present these as causal controls; do not claim every component wins on every task. |
| External baselines | Not yet sufficient | Fixed, Raw+Slack, Scattered and Full are clean internal controls. | Historical `snapkv`, `quest_lite`, and `sagekv_lite` are not faithful official baselines and must not be presented as such. |
| Efficiency evidence | Partial | Exact retained bytes and stage timers exist. Legacy 27B/32K V6.1 runs show about 59--63 GiB peak usage on an 80GB-class device, supporting capacity feasibility. | Legacy latency fields are zero. The current method creates Full prompt KV before compression, so it does not establish lower peak memory or serving throughput. |
| Reproducibility | Strong | Frozen protocol, revisions, seeds, persistent FP32 query probe and SHA, resume manifests, clean-commit launcher, official QA metric, 165/165 CPU tests. | Preserve this contract when revising run order. |
| A-conference idea position | Plausible, not sealed | The paper can center on a hybrid-specific residual-memory problem and a structured locality-preserving overlay with theory-to-ablation alignment. | SnapKV already clusters important KV positions; novelty needs a direct, fair comparison and precise differentiation from head-wise clustering/layer allocation. |

## 3. What the paper can claim now

The current evidence supports the following bounded story:

1. Hybrid recurrent/attention LLMs remove KV growth from recurrent layers but do
   not eliminate the residual Full-Attention KV bottleneck.
2. Under equal resident KV bytes, retaining isolated high-score tokens can destroy
   local relational structure. A query-guided contiguous overlay preserves that
   structure and gives a cleaner quality-memory frontier in the tested regime.
3. The effect transfers without retuning from Qwen3.5-0.8B to 9B and is strongest
   at constrained 5--10% middle-cache budgets.
4. HMO is a training-free memory-organization algorithm with
   `O(T + n log n + N_keep)` selection and
   linear retained KV in context length, but with a substantially smaller
   coefficient than Full KV.

It does **not** yet support these stronger claims:

- universal superiority over fixed chunks or raw attention Top-K;
- cross-architecture or cross-model-family generalization;
- proportional reduction in total process peak VRAM;
- production throughput/latency acceleration;
- a guarantee that the attention-derived utility matches task utility.

This is a viable algorithm-paper position if the title and abstract emphasize
**quality under a fixed residual-KV budget** and **local structure preservation**,
rather than presenting HMO as a finished serving system.

## 4. What recent top-venue papers actually do

The comparison below is about experimental structure, not a demand to reproduce
every paper's compute scale.

| Work | Main model/scale pattern | Quality evaluation | Baselines and efficiency | Lesson for HMO |
|---|---|---|---|---|
| [H2O, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/6ceefa7b15572587b78ecfcebb2827f8-Abstract-Conference.html) | OPT, LLaMA and GPT-NeoX from 6.7B upward; 30B used for systems results | Multiple standard tasks and generation workloads | Theoretical formulation plus accuracy, throughput and latency; 20% cache | A simple theory is acceptable when paired with model/task breadth and systems evidence. |
| [SnapKV, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/28ab418242603e0f7323e54185d19bde-Abstract-Conference.html) | 7B/MoE main experiments, with 35B and a 380K single-A100 stress result | 16 long-sequence datasets and needle retrieval | H2O comparison, memory and generation-speed measurements on A100-80GB | The broad 7B main table carries the paper; the larger model is an anchor, not the whole paper. |
| [Quest, ICML 2024](https://hanlab.mit.edu/projects/quest) | LongChat-7B and Yarn-Llama-2-7B | Six representative LongBench tasks plus 10K/100K passkey retrieval | Cache-budget sweeps and kernel-level latency on consumer/workstation GPUs | A focused six-task suite can be credible when the method and efficiency path are sharply demonstrated. |
| [DuoAttention, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/5c1ddd2e59df46fd2aa85c833b1b36ed-Paper-Conference.pdf) | Llama-2-7B, Llama-3-8B and Mistral-7B main; 70B is auxiliary | Needle, 14/21 LongBench tasks, and short-context quality checks | H2O, TOVA, FastGen, StreamingLLM; matched budgets; A100 memory/latency curves | Multiple 7--8B architectures and quality-preservation checks matter more than making 70B the main table. |
| [HeadKV, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/f649556471416b35e60ae0de7c1e3619-Paper-Conference.pdf) | Llama-3-8B and Mistral-7B | Six LongBench QA tasks, mostly 200 examples/task, plus four LooGLE tasks | SnapKV, PyramidKV and Ada-SnapKV at equal KV entries; 32K memory/latency averaged over three trials | HMO's 24 native examples are currently too thin; a six-task, 100+ example design is a useful minimum target. |
| [CAKE, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/dfae940651f3e690a12e19c874edad7c-Paper-Conference.pdf) | Five architectures from 7B to 70B; 7--8B models carry the dense sweeps | All 16 LongBench datasets and three NeedleBench subtasks | StreamingLLM, H2O, TOVA, SnapKV, PyramidKV across 64--2048 entries/layer; A100 efficiency | This is the high-compute end of the spectrum, not HMO's minimum bar, but it shows why public baselines and budget curves are expected. |
| [SAGE-KV, ICLR 2025 SLLM workshop](https://iclr.cc/virtual/2025/33493) | Three long-context 7--8B models from Llama and Qwen families | LongBench | Static StreamingLLM and dynamic Quest, with memory-efficiency comparison | Cross-family 7--8B evidence is a useful reference, but this is workshop evidence rather than an ICLR main-conference paper. |

### Model-selection implication

The recurring pattern is **7--8B breadth first, larger scale second**. Therefore:

- Qwen3.5-9B should carry HMO's broad benchmark, ablations and efficiency table;
- Qwen3.5-27B/32K should confirm scale transfer and long-context behavior;
- a 70B experiment is not required for the current story;
- because HMO is scoped to a hybrid architecture, one Qwen family can be
  acceptable if the paper explicitly avoids cross-family claims, but a second
  hybrid family would strengthen a later revision more than another Qwen size.

## 5. Recommended 5090 closure before buying A100 time

These are high-return paper tasks, not strict kill gates. A negative slice should
be reported and used to narrow claims, not to restart design search.

### 5.1 Must-do evidence package

| Package | Concrete experiment | Approximate 5090 use | Purpose |
|---|---|---:|---|
| Broad native quality | Qwen3.5-9B; NarrativeQA, Qasper, MultiFieldQA-en, HotpotQA, 2WikiMultihopQA and MuSiQue; freeze 100 examples/task; HMO, Fixed, Raw+Slack, Scattered, Full at central 10% | 5--8 h | Expands from 24 to 600 native examples and covers single-document plus multi-hop QA. |
| Faithful strong baseline | Add one verified SnapKV baseline on the same six-task subset and measured resident-byte budget; validate retained indices, per-layer/head accounting and prompts against its public algorithm | 1--3 h GPU, plus engineering | Establishes external position. If Qwen3.5 hybrid support requires adaptation, document it and call it an adaptation, not official SnapKV. |
| Efficiency and accounting | Qwen3.5-9B at 8K/16K, optionally the longest safe local context; HMO/SnapKV/Full; three repetitions; report post-query resident bytes, controller/probe time, TTFT, decode ms/token and CUDA peak | 1--2 h | Separates algorithm overhead, persistent cache saving and end-to-end behavior. |
| Hybrid memory decomposition | Compute recurrent-state bytes and Full-Attention KV bytes from each model config; plot total cache versus context length and batch for 0.8B/9B/27B | 0 GPU h | Shows exactly where a residual-KV overlay matters and avoids implying that 13% residual KV means 13% total VRAM. |
| Paper table freeze | Aggregate existing mechanism, budget, fixed/scattered/raw ablations and new native/efficiency evidence with bootstrap intervals and task-stratified results | 0 GPU h | Converts evidence into the final claim-to-table map before the paid run. |

Expected local GPU total: **7--13 RTX 5090 hours**. The estimate is based on the
observed 9B formal run (about 12 minutes for 24 synthetic cases across six
systems) and the 0.8B native run (about 9 minutes for 24 cases across five
systems), expanded conservatively for longer answers, a new baseline and repeated
timing. It is an engineering estimate, not a runtime guarantee.

### 5.2 Optional improvements

- Increase the six native tasks from 100 to 200 examples/task if local time is
  available. This moves closer to HeadKV's evaluation density.
- Add a second hybrid model family if an implementation-compatible checkpoint
  exists locally. This is more valuable for generalization than adding Qwen3.5-4B.
- Add RULER or a multi-needle reasoning task only if it tests relational locality;
  more single-key passkey retrieval has limited value after the existing suites.

## 6. Revised one-shot A100-80GB formal plan

### 6.1 No separate paid preflight

The currently frozen C3 protocol contains a separate two-cell preflight. This
should be superseded after review as follows:

1. Prepare the environment, data bundle, clean commit, model revision and output
   destination before the paid window whenever the provider permits it.
2. Start the paid session and load the model once.
3. Run the first frozen 32K formal sample as HMO plus Full. Those results count in
   the final table.
4. Check only operational invariants: process alive, finite outputs, byte logs,
   output persistence and no OOM. Do not inspect answer quality as a continuation
   condition.
5. Automatically continue the central 10% block, then side budgets, then native
   tasks. Resume manifests preserve every completed cell.

This spends only the unavoidable first few minutes on operational admission and
does not create a separate experiment, pause, or outcome gate.

### 6.2 Formal experiment matrix

| Order | Experiment | Generation cells | Why this order matters |
|---:|---|---:|---|
| 1 | 27B/32K synthetic central 10%: 12 Needle + 12 LongEval, four equal-byte compressed systems plus Full | 120 | Highest-value scale-transfer result is completed first; first sample includes inline admission. |
| 2 | Add 5% and 20% compressed arms, reusing Full outputs | 192 | Completes the quality-memory Pareto and tests the predicted constrained/saturation regimes. |
| 3 | 27B native central 10%: frozen 12 HotpotQA + 12 NarrativeQA, four compressed systems plus Full | 120 | Gives same-protocol cross-scale native confirmation; broad native evidence should already come from 9B/5090. |
|  | **Total** | **432** | No extra paid preflight cells. |

The systems remain HMO, Global Fixed-Chunk, Raw Exact+Slack, Scattered and Full.
All compressed systems use exact matched measured resident bytes. No method,
sample, prompt, budget or metric changes are allowed after outcomes are visible.

### 6.3 A100 GPU-hour budget

| Paid stage | Expected time | Conservative range | Output |
|---|---:|---:|---|
| Instance bootstrap, model load, first formal pair and invariant check | 0.35 h | 0.25--0.50 h | Loaded pinned 27B model and two retained formal cells |
| Central 10% synthetic block | 2.25 h | 1.5--3.0 h | Primary 27B/32K scale-transfer table |
| 5% and 20% synthetic side budgets | 3.0 h | 2.0--4.0 h | 27B quality-memory Pareto |
| Native 10% block | 2.25 h | 1.5--3.0 h | Cross-scale native confirmation |
| Integrity audit, summaries, checksums and retry allowance | 0.75 h | 0.5--1.0 h | Complete manifest and portable result bundle |
| **Total** | **8.6 h** | **5.75--11.5 h** | Full 432-cell formal package |

Recommended reservation: **12 A100-80GB GPU-hours**.  
Operational hard stop: **14 GPU-hours**, used only for transient retries or slow
provider I/O, not for adding experiments or tuning the method.

If the provider bills at `P` currency units per GPU-hour:

- expected planning cost: `8.6 * P`;
- recommended reserved cost: `12 * P`;
- absolute capped cost: `14 * P`.

This estimate is deliberately a range. It extrapolates from completed 5090 runs
and legacy 27B/32K capacity traces; it is not measured A100 latency. The largest
uncertainties are the provider's Qwen3.5 DeltaNet kernel, 32K prefill behavior,
storage bandwidth, and native-answer length. Legacy V6.1 establishes only that
27B/32K occupied roughly 59--63 GiB on an 80GB-class device; its timing fields are
zero and cannot calibrate speed.

## 7. Why this is enough without an A100 preflight

Before the formal run, the following risks can be removed on the 5090 or CPU:

- correctness of the allocator and exact-byte matcher;
- deterministic sample/probe generation and metric computation;
- resume, manifest, stage order and storage behavior;
- the 9B quality direction and public-baseline position;
- controller/probe overhead and the distinction between resident and peak memory;
- expected output count and disk footprint.

The only material uncertainty intentionally left to A100 is the actual 27B/32K
scale result and provider-specific runtime. That is exactly what the paid
experiment is supposed to measure.

## 8. Questions for GPT review

1. Is the scoped contribution strong enough for an algorithmic A conference once
   the six-task 9B native table and one faithful SnapKV comparison are added?
2. Is a broad Qwen3.5-9B main table plus a Qwen3.5-27B/32K scale anchor preferable
   to spending more A100 time on a wider 27B native table?
3. Is SnapKV the single highest-value public baseline, or should PyramidKV/CAKE
   replace it given HMO's structured allocation story?
4. Can the paper remain explicitly Qwen3.5-hybrid scoped, or is a second hybrid
   architecture necessary for the target venue?
5. Should the current C3 protocol be revised to central-first ordering and inline
   admission exactly as specified above before any paid run?

## 9. Recommended decision

**Do not purchase A100 time yet, but do not reopen the design.** Approve the
compact 5090 closure package, obtain GPT review on the claim/baseline scope, then
revise the frozen C3 run order once and launch the 12-hour reserved one-shot A100
window. This uses the local GPU for uncertainty reduction and the expensive GPU
only for evidence that cannot be obtained locally: the final 27B/32K result.
