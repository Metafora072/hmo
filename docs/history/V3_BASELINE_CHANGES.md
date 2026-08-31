# V3 Baseline Changes

This version starts from `hmo-two-chanels-memory-LLM-v2` and adds three recent KV-cache baselines:

- `pyramidkv_lite`
- `quest_lite`
- `sagekv_lite`

They are intentionally named `*_lite` because they are repository-local approximations designed to run through the current dense-cache Qwen3.5 experiment harness.

## What Changed

### New Methods in E1

Default E1 methods now include:

```text
full_kv
h2o
snapkv
streamingllm
duoattention
pyramidkv_lite
quest_lite
sagekv_lite
hmo_full
```

You can still restrict methods with:

```bash
--methods full_kv,snapkv,pyramidkv_lite,quest_lite,sagekv_lite,hmo_full
```

### Controller Additions

Added to:

```text
experiments/utils/hmo_controller.py
```

New functions:

```python
run_pyramidkv_lite_baseline(...)
run_quest_lite_baseline(...)
run_sagekv_lite_baseline(...)
```

Helper functions:

```python
_kv_norm_token_scores(...)
_topk_keep_mask(...)
```

### KV Operation Additions

Added to:

```text
experiments/utils/kv_ops.py
```

New function:

```python
evict_kv_tokens_per_layer(cache, layer_to_keep_mask)
```

This supports PyramidKV-style layer-specific retention masks.

### Runner Dispatch

Added to:

```text
experiments/phase2/runner.py
```

The shared `run_method(...)` dispatcher now recognizes:

```text
pyramidkv_lite
quest_lite
sagekv_lite
```

## Method Semantics

### pyramidkv_lite

Implements the most important PyramidKV idea:

```text
lower attention layers keep more KV tokens
higher attention layers keep fewer KV tokens
```

Selection score:

```text
per-layer K/V norm score
```

This is not fully faithful PyramidKV because the original method uses attention-based token scores and a specific budget allocation formula. However, this version preserves the key layer-wise budget allocation idea.

### quest_lite

Implements a static approximation to Quest:

```text
last prompt query -> score KV pages -> keep top pages -> decode
```

Page size defaults to 32.

This is not faithful Quest because the original method performs query-aware page selection dynamically at each decode step. This version uses one query proxy from the prompt end so it can run without patching the attention kernel.

### sagekv_lite

Implements a post-prefill one-shot eviction baseline:

```text
prefill -> score tokens -> keep top-k -> decode
```

Selection score:

```text
global K/V norm score across attention layers
```

This is not faithful SAGE-KV because the original method uses self-attention-guided token/head-level selection. The current dense cache representation does not support different retained positions per head without deeper attention changes.

## Recommended Smoke Test

From the v3 project root:

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/phase2/e1_main/run.py \
  --model qwen3.5-0.8b \
  --gpu_id 0 \
  --n_samples 1 \
  --benchmarks needle \
  --context-lengths 1024 \
  --methods full_kv,pyramidkv_lite,quest_lite,sagekv_lite,hmo_full \
  --max_new_tokens 8 \
  --run-name smoke_v3_baselines
```

## Paper Caution

Use the names `PyramidKV-lite`, `Quest-lite`, and `SAGE-KV-lite` unless a faithful official implementation is integrated.

Do not claim these are exact reproductions of the original papers.
