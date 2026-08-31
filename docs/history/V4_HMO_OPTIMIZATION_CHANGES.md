# V4 HMO Optimization Changes

This is a compact experiment-focused copy derived from `hmo-two-chanels-memory-LLM-v3`.

The v4 directory intentionally excludes paper drafts, PDFs, research wiki files, and old result files. It keeps:

```text
experiments/
references/qwen3_5_source/
models -> ../hmo-two-chanels-memory-LLM-v2/models
docs/design/BASELINE_METHOD_DESIGNS.md
docs/history/V3_BASELINE_CHANGES.md
```

## Goal

V3 added three recent lightweight baselines:

```text
pyramidkv_lite
quest_lite
sagekv_lite
```

V4 keeps those baselines and improves HMO's own policy. The current v4.1 policy is deliberately conservative:

```text
V3 policy defaults + alpha/sigma alignment fix
```

An earlier v4 policy tried alpha-heavy refresh and stronger RTS floors, but the first small comparison reduced NarrativeQA performance. Those aggressive defaults were reverted.

## HMO Policy Changes

### 1. Alpha/Sigma Alignment

Previous behavior:

```text
if len(alpha) != len(sigma):
    alpha = None
```

This silently collapsed the method from:

```text
phi = sigma * alpha
```

to:

```text
phi = sigma
```

V4 behavior:

```text
alpha longer than sigma -> truncate alpha
alpha shorter than sigma -> pad alpha tail with the last alpha value
```

This keeps the dual-channel HMO signal active even when a short tail segment causes a one-segment mismatch.

### 2. Refresh Ranking

Refresh ranking is configurable:

```text
refresh_score = (1 - refresh_alpha_mix) * phi + refresh_alpha_mix * alpha
```

Default:

```text
refresh_alpha_mix = 0.0
```

Rationale:

The default is back to V3's phi-only refresh ranking. You can still test alpha-heavy refresh by passing `--refresh_alpha_mix`, but it is not the default.

### 3. RTS Allocation More Heavily Uses Phi

RTS token allocation now defaults to the V3 value:

```text
rts_phi_mix = 0.5
```

This avoids over-allocating RTS tokens based on noisy alpha/phi estimates.

### 4. RTS Coverage Floor Increased

Default:

```text
rts_floor_tokens = 1
```

The earlier floor of 2 reduced refresh usage and hurt NarrativeQA in the first v4 comparison.

### 5. Lower Refresh Priority Gate

Default:

```text
refresh_min_phi = 0.05
```

This restores the V3 threshold.

## New E1 CLI Arguments

The following arguments are now exposed in:

```text
experiments/phase2/e1_main/run.py
```

```bash
--refresh_budget
--refresh_min_phi
--refresh_alpha_mix
--rts_floor_tokens
--rts_phi_mix
```

Example:

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/phase2/e1_main/run.py \
  --model qwen3.5-0.8b \
  --gpu_id 0 \
  --n_samples 10 \
  --benchmarks narrativeqa,hotpotqa \
  --context-lengths 8192 \
  --methods snapkv,pyramidkv_lite,quest_lite,sagekv_lite,hmo_full \
  --refresh_alpha_mix 0.0 \
  --rts_phi_mix 0.5 \
  --rts_floor_tokens 1 \
  --run-name v4_policy_smoke
```

## Recommended Smoke Test

```bash
cd /home/xinyi/yxy/AI/hmo-two-chanels-memory-LLM-v4

CUDA_VISIBLE_DEVICES=0 python experiments/phase2/e1_main/run.py \
  --model qwen3.5-0.8b \
  --gpu_id 0 \
  --n_samples 1 \
  --benchmarks needle \
  --context-lengths 1024 \
  --methods full_kv,snapkv,pyramidkv_lite,quest_lite,sagekv_lite,hmo_full \
  --max_new_tokens 8 \
  --run-name smoke_v4_policy
```

## Caution

V4 is a policy-optimization branch. It should be compared against V3 on the same subset before using its results in a paper table.
