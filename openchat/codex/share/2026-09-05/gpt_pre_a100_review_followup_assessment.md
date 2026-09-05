# Codex Follow-up on GPT's Pre-A100 Review

Date: 2026-09-05  
Reviewed artifact: `gpt/share/2026-09-05/gpt_pre_a100_readiness_budget_assessment_20260905.md`

## 1. Verdict

I support GPT's overall decision and most of its concrete revisions. The combined
direction should be:

1. keep the final HMO design and stop allocator/scorer exploration;
2. make ChunkKV the first external baseline audit, with SnapKV as the next strong
   comparison if implementation cost remains reasonable;
3. close native-task breadth, theory-to-implementation wording, and real runtime
   accounting on the local 5090;
4. revise C3 once so the first formal sample performs inline operational checks,
   then run the 27B/32K matrix in one paid A100 session.

No GPU, protocol, runner, or method change is authorized by this assessment.

## 2. Corrections accepted

GPT correctly identified the following issues in my previous memo:

- C2-native used Qwen3.5-0.8B, not 9B. The proposed broad 9B native evaluation
  remains new work.
- Overall selection complexity should be reported as
  `O(T + n log n + N_keep)`, where `n ~= T/L`, plus model probe/inference cost.
  The current sort prevents an unconditional `O(T)` algorithm claim.
- The `6.25%` value is the approximate cost of giving each region one base
  16-token window at `L=256`; it is not a proven accuracy phase transition.
- Under a non-strict first-versus-second marginal inequality, coverage-first is
  guaranteed only with an explicit coverage-preferring tie-break, or should be
  stated as the existence of a coverage-first optimum.
- The region-level separable concave utility theorem is an explanatory slice. It
  does not directly prove the actual heterogeneous-cost attention allocator or
  Exact upgrades optimal.
- The current C3 protocol, launcher, runbook, and estimator still implement a
  separate preflight despite the newer decision text. They must be revised before
  renting a GPU.
- The 8.6-hour A100 point estimate is not calibrated. Twelve reserved hours and a
  14-hour ceiling remain provisional capacity limits, not completion guarantees.

One bibliographic correction is also needed: SAGE-KV appeared on the ICLR 2025
SLLM workshop program, not the ICLR main-conference proceedings. The earlier
Codex table should not label it as an ICLR 2025 main paper.

## 3. ChunkKV materially changes the novelty audit

I independently checked the final NeurIPS 2025 ChunkKV paper and NVIDIA's public
`kvpress` implementation. GPT is right to prioritize it.

ChunkKV already makes the following claims:

- isolated token selection fragments semantic relationships;
- continuous chunks should be retained as atomic units;
- chunks are ranked using query/observation-window attention;
- the method is evaluated at equal compression ratios against StreamingLLM,
  H2O, SnapKV, and PyramidKV;
- its public implementation provides chunk-wise KV selection.

This means HMO cannot sell either "continuity preserves semantics" or
"attention-ranked chunks" as its primary novelty. Proposition 1 remains useful
as a precise explanation of complete span survival, but it is supporting theory,
not sufficient novelty by itself.

The defensible HMO increment is narrower:

1. **stratified macro-region coverage** instead of global top-chunk selection;
2. **free-start local windows inside regions** instead of only aligned chunks;
3. **mixed-granularity actions**: recurrent-only, local window, and optional
   Exact upgrade under one byte ledger;
4. **Hybrid residual-KV scope** with shared retained positions across the
   remaining Full-Attention layers.

This is still a plausible "existing work plus" algorithm paper, but the direct
ChunkKV comparison is now central evidence, not a decorative baseline. HMO versus
Scattered validates locality; HMO versus ChunkKV validates the new contribution.

## 4. Baseline design

Global Fixed-Chunk must remain an internal control and must not be renamed
ChunkKV. It shares the broad idea of aligned chunks but does not reproduce the
published observation-window, per-layer/head score, tail handling, or index-reuse
semantics.

The recommended table structure is:

### Main algorithm table

- Full KV;
- HMO;
- faithful/adapted ChunkKV at matched measured resident bytes;
- Raw Exact+Slack;
- SnapKV if the verified Hybrid adapter is affordable.

### Mechanism table

- HMO;
- Global Fixed-Chunk;
- Scattered;
- Raw Exact+Slack.

The mechanism table can use a predeclared smaller sample prefix because the
0.8B/9B synthetic evidence is already dense. Paid 27B cells should prioritize the
public comparison. If ChunkKV passes local implementation validation, replacing
Scattered in the 432-cell 27B package is more valuable than expanding to 528
cells. Scattered remains in the existing 0.8B/9B evidence and paper ablation.

"Faithful" must mean that ChunkKV keeps its own score construction and chunk
selection. Matching bytes must not force it to consume HMO's cross-layer-averaged
score. Any changes needed for Qwen3.5's Hybrid layer set must be named an adapter
and documented.

## 5. Runner and timing audit

GPT's static code findings are correct:

- `run_pareto.py` calls `collect_hybrid_query_token_probe`, which attaches
  `Qwen35QueryAccessibilityHookManager` and computes recurrent contribution even
  though final allocation calls use `use_accessibility=False`.
- `_generate_system` calls `run_post_intervention_prompt` independently for each
  arm, so every arm repeats prompt execution.
- the runner forces the reference recurrent backend.
- `estimate_c3_cost.py` assumes the old two-cell preflight and scales complete
  per-arm time by output-length ratios, which also multiplies prefill work.

Recommended scope:

1. implement an attention-only probe path as a new explicit schema/version;
2. on existing 0.8B/9B cases, compare it with the hybrid collector for stored
   token scores, retained positions, generated tokens, and resident bytes;
3. switch the formal path only if the defined equivalence checks pass;
4. retain repeated per-arm prefill for this submission and measure it honestly;
5. do not make shared-prefill cache/state isolation a new prerequisite.

This simplification removes collection for a signal that left the final method.
It is not a new scientific design. Nevertheless, it changes execution and probe
identity, so it needs explicit local validation and cannot be silently substituted.

## 6. Revised 5090 package

The six candidate tasks remain appropriate: NarrativeQA, Qasper,
MultiFieldQA-en, HotpotQA, 2WikiMultihopQA, and MuSiQue.

Recommended order:

1. CPU-only length inventory and frozen sample-prefix construction. Use valid
   unaugmented/untruncated examples where possible; report actual counts if a
   task cannot supply the target under the frozen length policy.
2. CPU/code audit of NVIDIA `kvpress` ChunkKV semantics against the paper and
   Qwen3.5 Hybrid compatibility.
3. Small 5090 equivalence package for the attention-only probe and ChunkKV
   adapter. This is an implementation check, not an answer-quality gate.
4. Run the first 50 frozen examples/task for the complete main table, then
   continue to the next fixed 50 examples/task as the same predeclared sample
   prefix. Continuation is not conditioned on favorable results.
5. Run the smaller mechanism subset and 8K/16K repeated timing/accounting table.

This avoids treating "six tasks times 100" as a conference law while retaining a
paper-useful target of up to 600 native cases. The previous 7--13 5090 GPU-hour
range remains a provisional envelope. It must be updated after the attention-only
and public-baseline timings; a long-answer 9B QA run cannot be promised from the
current short-output measurements.

## 7. Revised A100 package

I support GPT's priority order:

1. 27B synthetic central 10%;
2. 27B native central 10%;
3. 27B synthetic 5%/20% side budgets.

This preserves the most useful mechanism and native scale evidence if the rented
session is interrupted. The native cases retain their real 13.7K--16.3K-class
lengths after 27B tokenizer verification; only the synthetic block is exact 32K.

The preferred 432-cell systems are HMO, verified ChunkKV, Global Fixed-Chunk,
Raw Exact+Slack, and Full. This replaces Scattered at 27B while keeping the cell
count unchanged. The exact final system set must be frozen only after the 5090
public-baseline check and before any 27B outcome is observed.

The first frozen formal sample counts in the main result set and performs only
normal operational checks: finite values, byte invariants, no OOM, and durable
output. It then continues automatically. There is no separate paid preflight,
answer-quality gate, or second approval pause.

## 8. Recommendation to PZ

Accept GPT's review with the stronger clarification that ChunkKV is the closest
published threat to HMO's novelty. Do not reopen the method, but make the next
authorized package explicitly about proving HMO's incremental value over ChunkKV
and removing known execution ambiguity before the paid run.

The next concrete implementation package, requiring PZ confirmation, is:

1. theory/bibliography corrections and a no-preflight C3 protocol/run-order draft;
2. ChunkKV compatibility audit and adapter design;
3. attention-only probe implementation plus equivalence tests;
4. six-task frozen sample-prefix inventory and a phase-aware cost estimator;
5. only then the approved 5090 baseline/native/timing runs.

No A100 should be purchased until this local package has produced a frozen
baseline choice and an updated hour table. This is evidence closure, not a new
method-search gate.
