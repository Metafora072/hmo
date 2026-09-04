# Assessment Of GPT Story Review

## Decision

GPT's scientific framing is accepted with two mathematical/statistical
corrections and one execution-feasibility correction.

## Accepted

1. Center the paper on a locality-preserving KV overlay for Hybrid-Attention
   models, not on recurrent-aware allocation.
2. Use contiguous-versus-scattered at exact equal bytes as the central causal
   result.
3. Keep recurrent accessibility as development diagnosis and move the Exact
   upgrade to an optional fidelity layer rather than a required contribution.
4. Stop iterating allocation formulas. Additional experiments should broaden
   the evidence surface, not tune the confirmed method on its labels.
5. Build the paper state, Chinese storyboard, and abstract now rather than
   waiting for a perfect universal result.

## Corrections Applied

### Complexity

The retained KV bound `O(Tw/L + mL)` is not asymptotically smaller than `O(T)`
when `L`, `w`, and the Exact-upgrade rate are fixed. The defensible result is a
retention-factor reduction:

```text
rho_middle <= w/L + m(L-w)/T,
```

plus protected anchors, query KV, and boundary rounding.

### Memory aggregation

- Mean of per-case contiguous/Full ratios: 13.383%.
- Ratio of mean contiguous and Full bytes: 13.030%.

The paper should use 13.38% as the average sample-normalized footprint and name
the convention. It may report 19.41 MB versus 148.93 MB separately, without
deriving a second unlabeled percentage.

## Feasibility Finding

The proposed "existing 27B environment" is not currently present on this
server:

- the HMO data-disk model directory contains only Qwen3.5-0.8B;
- `/mnt/ssd/llm_models` contains Ministral-3-14B-Instruct and Qwen3-14B-FP8;
- Qwen3-14B is a standard `Qwen3ForCausalLM`, not the target Qwen3.5 Hybrid
  architecture;
- the repository has Qwen3.5-27B aliases and historical V6.1 27B result files,
  but no currently loadable 27B weights or provenance-complete formal run;
- `/mnt/nvme0` has about 74 GB free, so downloading a new 27B checkpoint would
  consume most remaining space, and BF16 would not fit one 32 GB GPU.

Therefore a new 27B/32K formal confirmation is not a low-cost immediate action.

## Recommended Order

### Immediate zero-GPU work

Create `PAPER_STATE`, an ICLR-style Chinese storyboard, and an abstract draft
around locality-preserving coverage. Use the corrected theorem and present the
Exact upgrade as optional.

### First GPU package

Strengthen the closest baseline with Raw Exact+Slack: choose raw-alpha Exact
segments first, then spend the deterministic residual bytes on top
query-attention tokens outside those segments. Compare it with contiguous
coverage at exactly matched resident bytes across 5%, 10%, and 20% caps. This
provides both the fairness fix and the quality-memory Pareto figure in one run.

This expansion may reuse the frozen 48-case suite as a clearly labeled Pareto
analysis. It must not search widths or alter the contiguous selector based on
the resulting labels.

### Second evidence package

The repository already has LongBench HotpotQA loading and official F1 scoring,
but the fresh confirmation runner currently accepts only Needle and
LongEval-Lines. Add a small 32K HotpotQA transfer using the currently validated
0.8B path first, with Full KV as a solvability check. If the small model is too
weak, acquiring Qwen3.5-4B or 9B is more feasible than jumping directly to an
absent 27B checkpoint.

No GPU work in these packages has been authorized or started yet.
