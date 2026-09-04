# E3-v2 execution path

This package is the preregistered replacement for the legacy full-prompt E3
prototype.

P0-B establishes the context/query contract:

1. serialize and tokenize the exact full prompt once;
2. prefill the memory context;
3. intervene on attention KV while recurrent state remains unchanged;
4. process the query suffix with logical positions from the original sequence;
5. expose answer logits only after the query suffix;
6. score or generate from a fresh, single-use post-intervention state.

P0-C adds exact Qwen3.5 recurrent instrumentation:

- recover `beta * (v - state^T k)` with the model's chunk-WY delta rule;
- compute cumulative log-survival from the actual `exp(g)` multiplier;
- measure later surviving writes that align with or cancel a segment write;
- freeze formulas and aggregation in the immutable run manifest.

`Qwen35RecurrentCandidateHookManager` must be attached immediately before the
fresh context prefill. Its `finalize_context()` method must run at the start of
the attention-KV intervention callback, which detaches every hook before query
processing. The legacy hook remains the source of `sigma_current` only.

The integrated real-model preflight is implemented in
`real_model_preflight.py`. It must pass before discovery is launched.

`run_discovery.py` is the bounded P1 GPU path. It collects equal-byte
gold-log-prob labels, query-aware alpha, `sigma_current`, and the P0-C recurrent
candidates, then runs sample-grouped incremental-value analysis. Pair results
are appended to JSONL as they finish so an interrupted run can resume without
repeating completed comparisons.

The default P1 configuration is deliberately lightweight: Qwen3.5-0.8B, two
Needle and two LongEval-Lines samples at 8K, 256-token segments, two donors per
segment, one background per pair, and no per-arm secondary generation. These
overrides are recorded in the run manifest and produce discovery evidence only;
they are not confirmation settings.

The same runner supports a held-out confirmation mode with --scope confirmation.
This mode requires both --frozen-scorer-config and a non-empty
--sample-id-prefix, embeds the frozen method SHA-256 and full configuration in
the immutable manifest, and writes confirmation_summary.json. Supported
schemas are dispatched without tuning on confirmation labels: the historical
bounded additive scorer remains reproducible, while the conditional controller
uses fixed median regimes and one top-down adjacent inversion pass in which each
segment moves at most one alpha rank.

For a controller evaluated by a downstream query probe, use
`--scope prospective_oracle` with a unique `--sample-id-prefix`. This scope
collects fresh equal-byte oracle labels but deliberately performs no candidate
analysis. `enrich_query_accessibility.py` then requires the frozen V2 config,
the prospective protocol, and its stage name; it verifies their hashes and
reports only frozen V2 versus corrected raw alpha.

## Qwen3.5-9B scale transfer

`run_scale_transfer.py` evaluates the frozen contiguous coverage-fidelity
method on Qwen3.5-9B without retuning. It adds Raw Exact+Slack, which first
selects raw-alpha Exact segments and then spends the segment-rounding residual
on globally top query-attention tokens outside those segments. This arm is
strictly resident-byte matched to the contiguous, scattered, and sparse-only
arms.

```bash
source /home/pz/miniconda3/etc/profile.d/conda.sh
conda activate hmo_research_v6
export CUDA_VISIBLE_DEVICES=1

python -m experiments.phase2.e3_v2.run_scale_transfer \
  --model-path /mnt/nvme0/hmo/models/Qwen3.5-9B \
  --model-revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --protocol refine-logs/contiguous_cf_scale_transfer_9b_protocol.json \
  --stage-set smoke \
  --run-dir /mnt/nvme0/hmo/runs/contiguous_cf_scale9b_smoke

python -m experiments.phase2.e3_v2.run_scale_transfer \
  --model-path /mnt/nvme0/hmo/models/Qwen3.5-9B \
  --model-revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --protocol refine-logs/contiguous_cf_scale_transfer_9b_protocol.json \
  --stage-set formal \
  --run-dir /mnt/nvme0/hmo/runs/contiguous_cf_scale9b_formal
```

## Qwen3.5-0.8B structured-baseline Pareto

`run_pareto.py` evaluates 5%, 10%, and 20% middle-context caps on the matched
48-case confirmation suite. Global Fixed-Chunk Top-K uses non-overlapping
16-token boundaries and the same query probe as HMO. Contiguous CF, fixed
chunks, Raw Exact+Slack, Scattered CF, and Sparse-only are exactly matched by
measured resident KV bytes; Full KV is generated once per sample.

```bash
source /home/pz/miniconda3/etc/profile.d/conda.sh
conda activate hmo_research_v6
export CUDA_VISIBLE_DEVICES=1

python -m experiments.phase2.e3_v2.run_pareto \
  --model-path /mnt/nvme0/hmo/models/Qwen3.5-0.8B \
  --model-revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --protocol refine-logs/contiguous_cf_pareto_protocol.json \
  --stage-set smoke \
  --run-dir /mnt/nvme0/hmo/runs/contiguous_cf_pareto_smoke

python -m experiments.phase2.e3_v2.run_pareto \
  --model-path /mnt/nvme0/hmo/models/Qwen3.5-0.8B \
  --model-revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --protocol refine-logs/contiguous_cf_pareto_protocol.json \
  --stage-set formal \
  --run-dir /mnt/nvme0/hmo/runs/contiguous_cf_pareto_formal
```

## Qwen3.5-0.8B Stratified Fixed-Chunk control

`run_stratified_fixed_control.py` reuses the SHA-pinned 16K/10% HMO rows from
Package B. It recomputes and exactly verifies the parent allocation and
retained positions, then generates one new equal-byte control whose Sparse
window starts are restricted to segment-local 16-token boundaries. Sparse
widths, including 17/18-token slack extensions, remain unchanged.

```bash
source /home/pz/miniconda3/etc/profile.d/conda.sh
conda activate hmo_research_v6
export CUDA_VISIBLE_DEVICES=1

COMMON_ARGS="--model-path /mnt/nvme0/hmo/models/Qwen3.5-0.8B \
  --model-revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --parent-protocol refine-logs/contiguous_cf_pareto_protocol.json \
  --parent-results /mnt/nvme0/hmo/runs/contiguous_cf_pareto_formal_20260904_1518/pareto_results.jsonl \
  --protocol refine-logs/stratified_fixed_chunk_control_protocol.json"

python -m experiments.phase2.e3_v2.run_stratified_fixed_control \
  $COMMON_ARGS --stage-set smoke \
  --run-dir /mnt/nvme0/hmo/runs/stratified_fixed_control_smoke

python -m experiments.phase2.e3_v2.run_stratified_fixed_control \
  $COMMON_ARGS --stage-set formal \
  --run-dir /mnt/nvme0/hmo/runs/stratified_fixed_control_formal
```

## Qwen3.5-0.8B HotpotQA-32K-Aug Full-KV solvability

The pinned LongBench HotpotQA split has no native 32K examples. This runner
preserves a frozen base record and appends a second real HotpotQA context as a
post-target distractor, truncating only the distractor tail to reach a 32,768
token serialized memory context. It runs Full KV only and uses the vendored
official LongBench QA F1; it does not automatically launch a compressed arm.

```bash
source /home/pz/miniconda3/etc/profile.d/conda.sh
conda activate hmo_research_v6
export CUDA_VISIBLE_DEVICES=1

COMMON_ARGS="--model-path /mnt/nvme0/hmo/models/Qwen3.5-0.8B \
  --model-revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --archive /mnt/nvme0/hmo/datasets/LongBench/5e628be450b7e67fb7ae6e201bd6d8f7056f7672/data.zip \
  --protocol refine-logs/hotpotqa_32k_solvability_protocol.json"

python -m experiments.phase2.e3_v2.run_hotpot_solvability \
  $COMMON_ARGS --stage-set smoke \
  --run-dir /mnt/nvme0/hmo/runs/hotpotqa_32k_solvability_smoke

python -m experiments.phase2.e3_v2.run_hotpot_solvability \
  $COMMON_ARGS --stage-set formal \
  --run-dir /mnt/nvme0/hmo/runs/hotpotqa_32k_solvability_formal
```
