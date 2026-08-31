# V6: Anchor-augmented HMO

V6 starts from the stable V4.1 policy and adds a small exact-KV anchor stage
before refresh/RTS allocation.

## Why V6

V4.1 is the strongest stable engineering version so far, but its performance is
not strong enough for a paper-level claim:

- It is stable on Needle, LongEval, HotpotQA, and LCC.
- It is still weaker than Quest-lite / Full KV on GovReport.
- It does not recover the stronger V3 NarrativeQA score.

The local research report `yxy/AI/deep-research-report.md` points to a broader
2025-2026 trend: strong KV-cache methods increasingly combine multiple
dimensions of control instead of relying on one eviction score. Relevant ideas:

- CAKE / SqueezeAttention: layer/budget-aware allocation.
- RocketKV: two-stage compression with coarse retention plus fine sparse use.
- HybridKV: static retained cache plus dynamic retrieved units.
- SCOPE: keep prefill-side information reliable before decode-time compression.

V6 adapts this trend to the current HMO codebase with a low-risk change:
reserve a few exact middle-context anchors before applying refresh/RTS.

## Method

V4.1:

```text
protected KV = first segment + last segment
phi_j = sigma_j * alpha_j
refresh = top phi segments under budget
RTS = remaining budget allocated over other middle segments
```

V6:

```text
protected KV = first segment + last segment
phi_j = sigma_j * alpha_j
exact anchors = diverse top-phi middle segments
refresh = top remaining phi segments under budget
RTS = remaining budget allocated over other middle segments
```

This creates a three-level cache:

```text
exact KV anchors      : high-fidelity evidence segments
refresh segments      : replayable dynamic exact segments
RTS skeleton segments : low-cost coverage for the rest
```

The goal is to avoid forcing all non-boundary middle context into sparse RTS,
which may be too lossy for GovReport and NarrativeQA.

## New Hyperparameters

```text
kv_anchor_budget = 2
kv_anchor_min_phi = 0.02
kv_anchor_diversity = 0.15
```

Set `kv_anchor_budget=0` to recover V4.1 behavior.

## Files Changed

- `experiments/utils/hmo_controller.py`
  - Adds anchor hyperparameters.
  - Adds diverse exact-KV middle anchor selection before refresh/RTS.
  - Adds budget-matched Full-KV subset baselines:
    `budgeted_recent_kv` and `budgeted_uniform_kv`.

- `experiments/phase2/e1_main/run.py`
  - Exposes anchor hyperparameters on the CLI.
  - Adds the two budgeted KV baselines to default E1 methods.

- `experiments/phase2/runner.py`
  - Dispatches the two new baseline method names.

## Budgeted Full-KV Subset Baselines

The new baselines are plain KV-retention controls. They use the same shared
budget as SnapKV / Quest-lite / SAGEKV-lite / HMO, but they do not use
importance scores, refresh, or RTS.

```text
budgeted_recent_kv:
    keep sink tokens + as many recent tokens as the budget allows

budgeted_uniform_kv:
    keep sink tokens + recent tokens + uniformly sampled middle tokens
```

They answer a fairness question: does HMO beat simple full-KV subsets under the
same memory budget, or is it only benefiting from keeping some complete KV?

`--resume` is already compatible with the new methods because completed cells
are keyed by `(method, dataset, context_length, sample_id)`. If a previous run
already contains old methods, adding these method names and resuming will skip
old successful cells and fill only missing cells for the new baselines.

## Suggested Smoke Test

```bash
cd /home/xinyi/yxy/AI/hmo-two-chanels-memory-LLM-v6

CUDA_VISIBLE_DEVICES=0 python experiments/phase2/e1_main/run.py \
  --model qwen3.5-0.8b \
  --gpu_id 0 \
  --n_samples 1 \
  --benchmarks needle \
  --context-lengths 1024 \
  --methods full_kv,budgeted_recent_kv,budgeted_uniform_kv,hmo_full \
  --max_new_tokens 8 \
  --run-name smoke_v6_budgeted_kv
```

## Suggested Small Comparison

```bash
cd /home/xinyi/yxy/AI/hmo-two-chanels-memory-LLM-v6

CUDA_VISIBLE_DEVICES=0 python experiments/phase2/e1_main/run.py \
  --model qwen3.5-0.8b \
  --gpu_id 0 \
  --n_samples 50 \
  --benchmarks needle,hotpotqa,narrativeqa,gov_report,lcc,longeval_lines \
  --context-lengths 8192 \
  --methods full_kv,budgeted_recent_kv,budgeted_uniform_kv,snapkv,pyramidkv_lite,quest_lite,sagekv_lite,hmo_full \
  --max_new_tokens 64 \
  --run-name v6_anchor_small_compare \
  --resume
```

## Ablations

Recover V4.1:

```bash
--kv_anchor_budget 0
```

Try more exact anchors:

```bash
--kv_anchor_budget 3
```

Force anchors only when priorities are stronger:

```bash
--kv_anchor_min_phi 0.05
```
