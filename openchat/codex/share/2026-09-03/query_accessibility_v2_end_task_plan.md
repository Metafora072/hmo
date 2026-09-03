# Frozen V2 Equal-Byte End-Task Validation

## Question

Does the frozen query-accessibility V2 selector improve generated answer quality
when its selected segment set is actually retained for decoding, rather than only
improving oracle Top-K ranking?

## Frozen Contract

- Model: `Qwen/Qwen3.5-0.8B`, revision
  `2fc06364715b967f1860aea9cf38778875588b17`.
- Method: `refine-logs/query_accessibility_v2_method_frozen.json`, SHA-256
  `01bcd3a9ea864ef2b9ab64c23c058badec70d7c88976168ff77100668b49a5f5`.
- Protocol: `refine-logs/query_accessibility_v2_end_task_protocol.json`, SHA-256
  `8933b550b9176f9784d6482b1fe2766b710375c72557b32ebea1717936e0ae02`.
- Primary comparison: corrected raw-alpha Top-K versus frozen V2 Top-K.
- Budget: identical middle-segment slots and exact resident attention-KV bytes.
- Reference only: Full-KV generation, used to expose task solvability.
- Primary metric: normalized answer containment.
- Secondary metrics: strict normalized exact match, token F1, paired
  wins/ties/losses, membership changes, and resident KV bytes.

The selector never receives the dataset name, answer, or task label.

## Run Order

1. Smoke: fresh seed `20261000`, 1 Needle + 1 LongEval sample at 2K.
2. Main: fresh seed `20261001`, 12 + 12 samples at 8K.
3. Transfer: fresh seed `20261002`, 12 + 12 samples at 16K, only if the
   pre-registered 8K continuation gate passes.

The 8K continuation gate requires no overall or Needle primary regression, at
least two LongEval samples whose V2 membership differs from raw alpha, and zero
protocol errors. No controller, threshold, sample, or metric changes are allowed
after smoke.

## Implementation

`experiments/phase2/e3_v2/run_end_task.py` performs context prefill, exact
query-conditioned probing, frozen selection, per-arm cache-isolated generation,
and atomic result persistence. It fails closed on model/hash mismatch,
non-equal decode KV bytes, incomplete fresh sample sets, or divergent greedy
outputs when raw and V2 selected sets are identical.

The focused contract tests and the complete CPU suite pass: 105 tests total.
No GPU result was observed before this protocol was frozen.
