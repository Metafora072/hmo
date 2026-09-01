# E3-v2 P0-B implementation report

Date: 2026-09-02
Branch: `dev/e3-v2-p0b`
Scope: context/query split and post-intervention answer scoring only

## Decision

P0-B is complete at the implementation and no-GPU contract-test level. The new
E3-v2 path no longer obtains answer logits from a Full-KV prefill of the whole
prompt. It performs context prefill, context attention-KV intervention, query
processing, and answer scoring/generation in that order.

This does not open the GPU gate. P0-C and P0-D remain required, and real-model
numerical equivalence plus controlled-needle effects must pass before P1.

## Implemented contract

- `build_prompt_parts` preserves the existing serialized prompt while exposing
  an exact memory-context/query-suffix character boundary.
- `tokenize_prompt_parts` tokenizes the full serialization once and splits only
  at a verified tokenizer offset. A token crossing the boundary fails closed.
- `run_post_intervention_prompt` prefills context only, validates attention KV
  length, snapshots recurrent state, applies one attention-KV intervention,
  verifies recurrent state is unchanged, and only then processes the query.
- Original logical `position_ids` are preserved for RoPE, while resident cache
  positions follow the compressed KV order used by the causal mask.
- First-answer-token logits are exposed only after the query suffix completes.
- Gold scoring reports mean conditional log probability per answer token;
  greedy generation and teacher forcing consume a fresh arm state exactly once.
- Full-KV probing creates a fresh cache and returns cloned logits only, exposing
  no mutable probe cache to an oracle arm.
- The protocol version, event order, position policy, answer prefix, and primary
  quality target are pinned in the immutable run manifest.
- Legacy `prototype_runner` behavior remains untouched for historical v1 runs;
  E3-v2 uses the isolated package under `experiments/phase2/e3_v2/`.

## Evidence

No-GPU tests:

```text
CUDA_VISIBLE_DEVICES='' .../hmo_research_v6/bin/python -m unittest \
  experiments.test_p0b_context_query experiments.test_p0a_validity -v

25 tests passed:
- 14 P0-B prompt/cache/scoring contract tests
- 11 P0-A metric/manifest regression tests
```

Additional checks:

- `python -m compileall -q experiments`: pass.
- `git diff --check`: pass.
- Local Qwen3.5-0.8B tokenizer only, no model weights loaded:
  - HotpotQA: context/query `42/42` tokens.
  - LCC: context/query `21/6` tokens.
  - Needle: context/query `13/26` tokens.
- Local Transformers source inspection confirms Qwen3.5 causal masking derives
  its query offset from resident DynamicCache KV length, while RoPE uses the
  supplied logical `position_ids`; DynamicLayer has no separate length metadata
  that must be repaired after KV slicing.

## Covered integrity checks

- Query runs only after context-KV intervention: covered by ordered trace test.
- Full-KV split equals unmodified reference: covered on deterministic fake
  Hybrid cache/model.
- Repeated identical arms match logits and greedy output: covered.
- Intervention cannot mutate recurrent cache state: covered and fail-closed.
- Alpha/reference probe cannot share mutable cache through this API: covered by
  fresh-prefill trace and logits-only return.
- Gold likelihood starts from post-intervention logits: covered.

## Remaining gates

- P0-C: instrument actual DeltaNet candidates and validate `exp(g)` retention
  direction with a synthetic gate test.
- P0-D: equal-byte multi-background oracle, real cache interventions, immutable
  segment/donor/background recovery, and alpha isolation in the formal runner.
- Pre-P1 real-model preflight: Full-KV numerical equivalence, repeated-arm
  determinism, and a controlled needle whose first-answer logits respond to KV
  intervention.

No GPU experiment or model inference was run for P0-B.
