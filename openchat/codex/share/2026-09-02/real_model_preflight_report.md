# E3-v2 Real-Model Preflight Report

Date: 2026-09-02
Branch: `dev/e3-v2-preflight`
Final code commit: `77ca94b3a7ac4eacc1d2432abacdc95d89fe673c`
Scope: smallest-representative-model integrity validation before P1 discovery

## Decision

The integrated P0-A through P0-D real-model preflight is a PASS. The final
Qwen3.5-0.8B run passed all eight fail-closed integrity checks and bound the
result to a clean Git commit, an explicit Hugging Face revision, and a complete
SHA-256 of the local model weights.

This decision authorizes only a bounded P1 discovery pilot. It is not evidence
for the HMO scientific hypothesis, does not validate the planned ranking
signals, and does not authorize confirmation runs, Qwen3.5-27B experiments, or
paper claims.

## Frozen Configuration

- Model: `Qwen/Qwen3.5-0.8B`, BF16.
- Revision: `2fc06364715b967f1860aea9cf38778875588b17`.
- Weight SHA-256: `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`.
- Architecture: 24 layers, including 18 recurrent/linear-attention layers and
  6 Full-Attention layers at indices `3, 7, 11, 15, 19, 23`.
- Recurrent backend: `transformers_torch_reference`.
- Controlled context target: 2048 tokens; actual memory context: 1981 tokens;
  query suffix: 28 tokens.
- Segment length: 64; retained middle budget: 2 complete segments.
- Oracle: 31 segments and 237 deterministic equal-byte comparisons.
- Greedy suffix: 4 tokens; seed: 20260902.
- Physical device: GPU1, exposed to the process as `cuda:0` through
  `CUDA_VISIBLE_DEVICES=1`.

The torch reference backend is deliberate. The installed FLA/Triton 3.3 path
aborts on RTX 5090 compute capability 12.0. This preflight validates the
Transformers reference implementation, not the optimized Triton kernel.

## Iteration Evidence

### Run 1: environment failure preserved

Path: `/mnt/nvme0/hmo/runs/preflight_qwen08b_20260902_173849`

The first run failed before scientific gating because the FLA Triton kernel
reported unsupported compute capability 12.0. The runner was changed to use
Qwen3.5's exact Transformers torch implementations for causal convolution and
gated-delta recurrence. No approximation or proxy recurrence was introduced.

### Run 2: 6/8 BLOCK preserved

Path: `/mnt/nvme0/hmo/runs/preflight_qwen08b_20260902_174804`

Six checks passed. Two checks correctly blocked execution:

- Full-KV equivalence used raw BF16 allclose despite identical top-1 behavior;
  observed max/mean logit differences were 0.3125/0.0474.
- Qwen3.5 returns a compact tuple containing only the six Full-Attention
  tensors, while the alpha probe initially interpreted tuple positions as
  global layer IDs.

The audit also exposed a more important P0-B contract issue: Qwen3.5 consumes
prior recurrent state only for continuation forwards with sequence length one.
The query suffix is therefore processed one token per forward in
`p0b-context-query-v2`. The Full-KV gate was frozen as a distribution-level
contract: max logit difference <= 0.5, mean difference <= 0.1, JS divergence
<= 0.001, top-10 overlap >= 0.8, and identical top-1 token. The alpha probe now
maps compact attention outputs to the configured non-contiguous global layers.

### Run 3: numerical PASS, provenance incomplete

Path: `/mnt/nvme0/hmo/runs/preflight_qwen08b_20260902_175624`

All eight numerical checks passed, but copying the Hugging Face snapshot to the
data disk had removed the revision-bearing `snapshots/<revision>` path. The run
manifest therefore had no model revision and only filename/size identity for
the weight. This result was retained as useful numerical evidence but was not
accepted as the final provenance-complete run.

### Run 4: final provenance-complete PASS

Path: `/mnt/nvme0/hmo/runs/preflight_qwen08b_20260902_185407`
Run manifest ID: `25f0f0e52703525f88ec0db9676bc021f844da1d9008a12591e2cfe77a9dd244`
Oracle manifest ID: `63b0096d98d7639ac5cbc744be5b40775253fa8a72dd7b07effe376f6d3dfa96`

| Integrity check | Result | Decisive evidence |
| --- | --- | --- |
| Equal-byte arms | PASS | 1,572,864 charged middle bytes; 3,108,864 context-resident bytes; 3,452,928 post-query bytes |
| Query after intervention | PASS | Frozen four-event order; answer logit position 2009 |
| Full-KV equivalence | PASS | Same top-1 token 24; max/mean logit diff 0.3125/0.04740; JS 4.04e-6; top-10 overlap 1.0 |
| Repeated-arm determinism | PASS | Exact logits; identical tokens `[24, 18, 21, 7428]` |
| Recurrent gate direction | PASS | Observed norm ratio 6.04964770; expected 6.04964746 |
| Controlled needle effect | PASS | Dropping segment 18 changes top-1 24 to 760; max/mean diff 11.0625/1.37419; JS 0.67577 |
| Alpha isolation | PASS | Before/after logits exactly equal; 31 alpha segments; total context mass 0.534637 |
| Manifest recoverability | PASS | Exact round trip; 31 segments; 237 comparisons |

The final manifest records `dirty=false`, commit `77ca94b`, the pinned model
revision, config/index hashes, and the complete 1,746,942,600-byte weight hash.

## Verification And Resources

CPU contract and regression verification before the final run:

```text
python -m unittest experiments.test_real_model_preflight_contract -v
Ran 6 tests in 0.311s
OK

python -m unittest discover -s experiments -p 'test_*.py' -v
Ran 67 tests in 0.371s
OK
```

Final GPU runtime was 17.07 seconds. Peak allocated memory was 13,961,054,720
bytes and peak reserved memory was 14,013,169,664 bytes (about 13.05 GiB). GPU1
returned to 15 MiB after the process exited. Raw manifests and logs remain under
`/mnt/nvme0/hmo/runs` and `/mnt/nvme0/hmo/logs`; model weights and caches remain
on the data disk rather than the system disk.

## Next Gate

The next action is a bounded P1 discovery pilot on the 0.8B model using the
frozen runner contract. It should produce real equal-byte intervention labels
and evaluate whether recurrent candidates add sample-grouped ranking value over
`alpha + position`. Launch requires explicit user confirmation and a frozen
P1 run matrix; this report alone does not start it.
