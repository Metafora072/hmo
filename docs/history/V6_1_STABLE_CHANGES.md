# V6.1: Stable HMO Mainline

V6.1 turns the current v6 directory into the paper-facing stable line.
The default `hmo_full` behavior is restored to the validated v4.1 policy,
while the v6 exact-KV anchor mechanism remains available as an explicit
ablation.

## Why change the default

The completed `v6_anchor_small_compare` run showed that exact middle-segment
KV anchors were neutral on easy retrieval tasks but hurt or failed to improve
global reasoning and summarization-style tasks:

- Needle and LongEval stayed tied with the best methods.
- HotpotQA was essentially tied but slightly lower than the best baselines.
- GovReport dropped below Full KV and Quest-lite.
- NarrativeQA dropped clearly below the v4.1/v5.1 HMO line.

The anchor stage spends budget on full middle-segment KV before refresh and
RTS allocation. In practice this reduced refresh/RTS coverage, which made the
policy less reliable for tasks needing broad semantic coverage.

## Default policy

The default is now:

```text
protected KV = first segment + last segment
phi_j = sigma_j * alpha_j
refresh = top remaining phi segments under budget
RTS = remaining budget allocated over other middle segments
drop = zero-token middle segments
```

This matches the stable v4.1 behavior:

```text
kv_anchor_budget = 0
```

The anchor code is still present. To reproduce the v6 anchor ablation, pass:

```bash
--kv_anchor_budget 2
```

## Baselines retained

V6.1 keeps the budget-matched Full-KV subset controls:

```text
budgeted_recent_kv
budgeted_uniform_kv
```

These baselines are important for the paper because they test whether HMO is
better than simply keeping a same-budget subset of exact KV tokens.

## Recommended stable comparison

```bash
cd /home/xinyi/yxy/AI/hmo-two-chanels-memory-LLM-v6

python experiments/phase2/e1_main/run.py \
  --model qwen3.5-0.8b \
  --gpu_id 0 \
  --n_samples 50 \
  --benchmarks needle,hotpotqa,narrativeqa,gov_report,lcc,longeval_lines \
  --context-lengths 8192 \
  --methods full_kv,budgeted_recent_kv,budgeted_uniform_kv,snapkv,pyramidkv_lite,quest_lite,sagekv_lite,hmo_full \
  --max_new_tokens 64 \
  --run-name v6_1_stable_small_compare \
  --resume
```

## Recommended anchor ablation

```bash
python experiments/phase2/e1_main/run.py \
  --model qwen3.5-0.8b \
  --gpu_id 0 \
  --n_samples 50 \
  --benchmarks needle,hotpotqa,narrativeqa,gov_report,lcc,longeval_lines \
  --context-lengths 8192 \
  --methods hmo_full \
  --kv_anchor_budget 2 \
  --max_new_tokens 64 \
  --run-name v6_anchor_ablation \
  --resume
```

## Smoke tests

Default stable path:

```bash
python experiments/phase2/e1_main/run.py \
  --model qwen3.5-0.8b \
  --gpu_id 0 \
  --n_samples 1 \
  --benchmarks needle \
  --context-lengths 1024 \
  --methods full_kv,budgeted_recent_kv,budgeted_uniform_kv,hmo_full \
  --max_new_tokens 8 \
  --run-name smoke_v6_1_stable
```

Anchor ablation path:

```bash
python experiments/phase2/e1_main/run.py \
  --model qwen3.5-0.8b \
  --gpu_id 0 \
  --n_samples 1 \
  --benchmarks needle \
  --context-lengths 8192 \
  --methods hmo_full \
  --kv_anchor_budget 2 \
  --max_new_tokens 8 \
  --run-name smoke_v6_anchor_ablation
```
