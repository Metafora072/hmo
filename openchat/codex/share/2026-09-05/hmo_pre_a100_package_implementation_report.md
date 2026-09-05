# HMO Pre-A100 Package: Implemented State

Date: 2026-09-05  
Author: Codex

## Decision

The approved GPT/Codex package has been implemented. The method was not reopened.
The work closes execution ambiguity, adds the closest external baseline, freezes
the broad 9B native package, and removes the separate 27B preflight. Renting an
A100 or downloading 27B weights remains outside this execution and still requires
PZ's explicit cost confirmation.

## Implemented changes

1. **Theory contract.** Coverage-first equality now states existence of an optimum
   and requires an uncovered-region tie break for the greedy construction. Total
   selection complexity is consistently stated as
   `O(T + n log n + N_keep)` plus model probe/generation cost.
2. **Attention-only probe v2.** The paper runners no longer install recurrent
   accessibility hooks when `use_accessibility=false`. A new identity-bound FP32
   cache stores both aggregate token scores and per-Full-layer token scores. The
   old hybrid v1 cache remains unchanged for historical reproduction.
3. **ChunkKV hybrid adapter.** The adapter uses fixed 10-token chunks, query-suffix
   observation attention, independent rankings per Full layer, positions shared
   across KV heads within a layer, fixed-prefix remainder handling for exact byte
   equality, and no recurrent-state mutation. It replaces Scattered in the new
   formal system table; Global Fixed remains an internal mechanism control.
4. **Six-task 9B protocol.** NarrativeQA, Qasper, MultiFieldQA-en, HotpotQA,
   2WikiMultihopQA and MuSiQue now use the official pinned prompts, QA-F1 metrics
   and 128/128/64/32/32/32 generation limits. Selection is outcome-free: take the
   longest unmodified source examples whose exact serialized memory context is at
   most 16K, with stable record-index tie breaks.
5. **C3 v2.** The 27B package is ordered as synthetic 10% -> native 10% ->
   synthetic 5%/20%. The first 27B sample is a formal result. Central and side
   share one manifest/results directory and reuse Full output. No quality gate or
   separate paid preflight remains.
6. **Cost timing.** Generation rows separately record prompt/intervention and
   decode time. NarrativeQA's 4x output allowance scales decode only, not prefill.

## 5090 real-model evidence

The validation used the pinned local Qwen3.5-0.8B model on GPU1 with a 1,981-token
memory context and 28-token query. The detached process completed and released the
GPU.

| Check | Result |
|---|---:|
| Old hybrid vs attention-only aggregate FP32 token scores | bitwise equal |
| Maximum absolute score difference | 0.0 |
| HMO retained positions | identical |
| HMO generated token IDs | identical, 4/4 |
| HMO post-query resident KV | 15,249,408 bytes |
| ChunkKV post-query resident KV | 15,249,408 bytes |
| ChunkKV Full layers executed with independent positions | 6/6 |
| Validation status | passed |

The machine-readable artifact is retained outside Git at
`/mnt/nvme0/hmo/validation/probe_chunkkv_20260905/result.json`.

## Frozen local main table

The final tokenizer inventory found that a hard 100 examples per task is
impossible without truncation for NarrativeQA and MuSiQue. The protocol therefore
reports actual eligible counts instead of manufacturing 600 cases.

| Task | Frozen cases <=16K | First prefix |
|---|---:|---:|
| NarrativeQA | 61 | 50 |
| Qasper | 100 | 50 |
| MultiFieldQA-en | 100 | 50 |
| HotpotQA | 100 | 50 |
| 2WikiMultihopQA | 100 | 50 |
| MuSiQue | 45 | 45 |
| **Total** | **506** | **295** |

The first prefix contains 1,475 generation cells; the full prefix contains 2,530.
Continuation is the already frozen suffix, not an outcome-dependent decision.
Based on the prior 9B formal run (24 cases in 744.8 s, about 29.0 s/case), the
expected local cost remains about 3--4 GPU-hours for prefix50 and 5--8 GPU-hours
for all 506 cases, with longer NarrativeQA/Qasper decode reflected in the range.

Protocol:
`refine-logs/native_longbench_six_task_9b_protocol.json`  
SHA256: `2b6c90154e42a543e1bc3ea534e5c81d84c4e7524a82192d8d59bbe29585464e`

## 27B formal budget

| Order | Work | Cells |
|---:|---|---:|
| 1 | 24 synthetic 32K samples, four compressed systems + Full, 10% | 120 |
| 2 | 24 frozen native QA samples, four compressed systems + Full, 10% | 120 |
| 3 | Same synthetic samples, four compressed systems, 5% and 20%; Full reused | 192 |
|  | **Total** | **432** |

The evidence-based planning interval remains 5.75--11.5 single-GPU hours, with a
12-hour reservation and proposed 14-hour operational cap. At provider rate `r`,
the expected charge interval is `5.75r--11.5r`, the reservation is `12r`, and the
hard cap is `14r`. These are planning bounds, not a completion guarantee; the
central formal rows refine the estimate without becoming a result gate.

Protocol: `refine-logs/c3_27b_protocol.json`  
SHA256: `4baab152a7c9461322d9bfd551860006193c357c30faaeeac816d85512652126`

## Verification

- Full `experiments/test_*.py` suite: 177 passed.
- All frozen JSON files parse successfully.
- Legacy Scattered-based Pareto protocol is preserved byte-for-byte at SHA256
  `ff064d...` for the historical stratified control.
- GPU0 was not used; GPU1 returned to 15 MiB after validation.

## Next action

Run the already frozen 9B `prefix50` package on GPU1, append results and staged
timing, then continue directly to `prefix100` as the precommitted extension unless
there is an operational failure. The next research decision is result-to-claim,
not allocator search. A100 acquisition remains a separate explicit cost action.
