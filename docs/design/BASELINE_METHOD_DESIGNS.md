# Recent KV-Cache Baseline Method Designs

This note summarizes three recent KV-cache compression / eviction methods that are relevant to HMO:

- PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling, 2024
- Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference, 2024
- SAGE-KV: Self-Attention Guided KV Cache Eviction, 2025

The purpose is not to implement them yet. The purpose is to make their design logic clear enough that we can decide whether to add them as baselines in `experiments/phase2/runner.py` later.

## Why These Baselines Matter

The current E1 baselines are:

| Method | Year | Core idea |
|---|---:|---|
| Full KV | standard | Keep all KV cache. |
| H2O | 2023 | Keep heavy-hitter tokens plus recent tokens. |
| StreamingLLM | 2023 | Keep attention sinks plus recent window. |
| SnapKV | 2024 | Use observation-window attention to select important KV tokens. |
| DuoAttention | 2024 | Separate retrieval heads and streaming heads. |

PyramidKV, Quest, and SAGE-KV are newer or stronger competitors because they improve the central weakness of fixed token-selection baselines:

1. PyramidKV asks: should every layer receive the same KV budget?
2. Quest asks: should KV importance depend on the current query?
3. SAGE-KV asks: should eviction happen at token and head granularity after prefill?

HMO asks a different question:

> In a hybrid-attention model, can recurrent-state saturation and attention fragility jointly decide whether a segment should remain full KV, become RTS, be refreshed, or be dropped?

So these methods are useful as baselines because they are strong KV-selection methods, but they do not explicitly use the recurrent memory channel.

## PyramidKV

### Paper

- Title: PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling
- Year: 2024
- Link: https://arxiv.org/abs/2406.02069

### Core Observation

PyramidKV observes that attention information flow changes across layers:

- Lower layers have more dispersed attention.
- Middle layers begin aggregating information into local regions.
- Higher layers concentrate attention on fewer critical tokens.

This motivates a pyramid-shaped KV budget:

> Keep more KV tokens in lower layers and fewer KV tokens in higher layers.

This differs from H2O, SnapKV, and StreamingLLM, which typically apply the same cache size to every layer.

### Design Logic

For a model with `m` attention layers, PyramidKV assigns each layer `l` its own cache budget:

```text
k_l = layer-specific KV budget
lower layers: larger k_l
higher layers: smaller k_l
```

The total budget is fixed, but it is distributed unevenly across layers.

The paper describes a decreasing arithmetic allocation from bottom to top layers. It also keeps a recent local window / instruction tokens across layers, then selects the remaining retained tokens using attention scores, following the same general spirit as SnapKV.

### Cache Operation

For each attention layer:

1. Keep recent local-window tokens.
2. Score old tokens using attention from the observation/instruction region.
3. Select top tokens according to that layer's budget.
4. Evict the rest from that layer's KV cache.

Unlike the current project code, this creates layer-specific cache lengths. That means implementation is more complex than a global keep mask.

### What It Competes With in HMO

PyramidKV competes with HMO's budget allocation strategy.

HMO currently assigns actions at segment level:

```text
segment -> KV / RTS / refresh / drop
```

PyramidKV assigns budgets at layer level:

```text
layer -> number of retained KV tokens
```

The two ideas are orthogonal:

- PyramidKV is layer-aware.
- HMO is segment-action-aware and recurrent-state-aware.

### How To Add as a Baseline

Likely implementation point:

```text
experiments/utils/hmo_controller.py
```

Add:

```python
run_pyramidkv_baseline(...)
```

Then add dispatch in:

```text
experiments/phase2/runner.py
```

Possible simplified implementation:

1. Run prefill and collect cache.
2. Compute per-layer budget.
3. For each full-attention layer, compute token importance by KV norm or attention-observation score.
4. Apply a layer-specific keep mask.

Important caveat:

The current `evict_kv_tokens(cache, attn_layer_indices, keep_mask)` applies the same keep mask to all layers. PyramidKV needs different masks per layer, so it needs a new function, e.g.:

```python
evict_kv_tokens_per_layer(cache, layer_to_keep_mask)
```

### Expected Strength

PyramidKV is a strong baseline, especially under tight KV budgets and long contexts. If added, it may be harder to beat than H2O and StreamingLLM. It is probably the most important new baseline to consider.

## Quest

### Paper

- Title: Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference
- Year: 2024
- Link: https://arxiv.org/abs/2406.10774
- Code noted by the paper: https://github.com/mit-han-lab/Quest

### Core Observation

Quest argues that token criticality is query-dependent:

> A token that is important for one decode query may be unimportant for another.

This differs from static prefill methods such as H2O or SnapKV, which decide what to keep before generation or using a fixed observation window.

### Design Logic

Quest groups KV cache into pages. For each page, it stores summary statistics of keys:

```text
page -> min key values
page -> max key values
```

At decode time, given the current query vector, it estimates which pages could produce high attention scores. Then it loads or attends only to the top-k critical pages.

So the decision is dynamic:

```text
current query q_t -> important KV pages for q_t
```

### Cache Operation

Quest does not simply delete KV once and for all. Conceptually, it keeps page metadata and dynamically selects which KV pages participate in attention.

One decode step looks like:

1. Compute current query.
2. Estimate page importance using query and page key min/max.
3. Select top-k pages.
4. Compute attention only over selected pages.

### What It Competes With in HMO

Quest competes with HMO's static post-prefill cache intervention.

Current HMO does:

```text
prefill -> compute sigma/alpha -> decide actions -> decode
```

Quest does:

```text
prefill -> build page summaries -> query-aware page selection at each decode step
```

So Quest is more dynamic during decode. HMO's refresh mechanism is dynamic in spirit, but in the current implementation refresh is mostly eager before decode.

### How To Add as a Baseline

Quest is harder to add faithfully because it requires modifying attention computation during decode.

There are two possible levels:

#### Faithful Implementation

Patch full-attention modules so that each decode query attends only to selected KV pages.

This likely touches:

```text
experiments/utils/rts_runtime.py
../../references/qwen3_5_source/modeling_qwen3_5.py
```

This is invasive and risky.

#### Approximate Implementation

Use a one-time query proxy from the last prompt token:

```text
last prompt query -> select top pages -> evict other pages -> decode normally
```

This is not fully faithful to Quest, but can be a practical "Quest-lite" baseline.

Possible function name:

```python
run_quest_lite_baseline(...)
```

Implementation idea:

1. Split KV positions into pages, e.g. page size 16 or 32 tokens.
2. For each page and layer, compute approximate max score using key vectors and the last prompt query.
3. Select top pages under budget.
4. Build a global keep mask from selected pages.
5. Reuse existing KV eviction path.

### Expected Strength

Quest should be strong when the relevant information changes depending on the generated query. It may be less directly comparable if implemented only as a one-time static approximation.

For paper fairness, if Quest is included, clearly state whether it is the official implementation or a project-local approximation.

## SAGE-KV

### Paper

- Title: LLMs Know What to Drop: Self-Attention Guided KV Cache Eviction for Efficient Long-Context Inference
- Year: 2025
- Link: https://arxiv.org/abs/2503.08879

### Core Observation

SAGE-KV observes that after prefill, the model's own self-attention reveals which tokens can be dropped. It uses this signal at both:

```text
token level
head level
```

The key claim is:

> The model implicitly knows which tokens are unnecessary after seeing the full prompt.

### Design Logic

SAGE-KV performs a one-time top-k selection after prefill.

Compared with H2O:

- H2O uses accumulated attention/heavy-hitter intuition.
- SAGE-KV uses self-attention-guided selection after prefill.

Compared with Quest:

- Quest is query-aware and dynamic during decode.
- SAGE-KV is post-prefill and one-time.

Compared with HMO:

- SAGE-KV is attention-only.
- HMO combines recurrent saturation and attention fragility.

### Cache Operation

After prefill:

1. Collect self-attention information.
2. Score token/head importance.
3. Select top-k tokens per head or per attention group.
4. Evict unselected KV entries.
5. Decode with the reduced cache.

The head-level part is important. A faithful implementation may need ragged per-head KV masks, which normal dense cache tensors do not handle easily.

### What It Competes With in HMO

SAGE-KV is a strong attention-guided eviction baseline.

It is particularly relevant because HMO also uses attention fragility `alpha`, but HMO multiplies it with recurrent saturation:

```text
phi = sigma * alpha
```

So SAGE-KV can test whether the recurrent channel adds value beyond attention-only eviction.

### How To Add as a Baseline

There are two implementation levels:

#### Faithful Implementation

Implement token/head-level top-k retention after prefill. This may require:

- storing per-head attention scores,
- selecting different tokens for different heads,
- modifying attention to support per-head sparse KV.

This is nontrivial in the current code because current KV tensors assume one sequence dimension shared across heads.

#### Practical Approximation

Use token-level self-attention scores averaged across heads and layers:

```text
token_score_i = mean attention received by token i during prefill
```

Then keep top-k tokens globally, using the existing global keep-mask logic.

Possible function name:

```python
run_sagekv_lite_baseline(...)
```

Implementation idea:

1. Register hooks on full-attention layers.
2. During prefill, collect attention weights or approximate attention scores.
3. Aggregate token importance across layers and heads.
4. Keep sink tokens, recent tokens, and top-k scored tokens.
5. Call `evict_kv_tokens(...)`.

If full attention weights are too expensive at 32K/65K, approximate with:

```text
KV norm
last-token attention proxy
observation-window attention proxy
```

But then it becomes closer to SnapKV, so the approximation should be named carefully.

### Expected Strength

SAGE-KV is likely stronger than StreamingLLM and may be competitive with SnapKV/Quest. It is a useful recent baseline because it is 2025 and directly targets KV eviction.

## Comparison With HMO

| Method | Year | Main signal | Granularity | Decode-time dynamic? | Uses recurrent state? | Implementation difficulty |
|---|---:|---|---|---|---|---|
| PyramidKV | 2024 | layer-wise attention sparsity | layer + token | no | no | medium |
| Quest | 2024 | current query vs KV page summaries | page | yes | no | high |
| SAGE-KV | 2025 | post-prefill self-attention | token + head | no | no | medium/high |
| HMO | current project | sigma * alpha | segment action | partly via refresh | yes | already implemented |

## Recommended Priority

### Priority 1: PyramidKV

Most suitable to add first.

Reasons:

- Strong and well-known.
- Directly comparable to H2O/SnapKV.
- Can be approximated without fully patching decode attention.
- Tests whether layer-wise KV allocation beats HMO's segment-wise orchestration.

### Priority 2: SAGE-KV Lite

Good recent 2025 baseline.

Reasons:

- Directly tests attention-only eviction.
- Useful for arguing that HMO's recurrent signal adds something beyond attention.
- Practical token-level approximation is feasible.

### Priority 3: Quest

Important but harder.

Reasons:

- Faithful implementation requires dynamic page selection inside attention.
- A lite version may be criticized as not true Quest.
- Better to cite in related work if implementation time is limited.

## Suggested Experiment Framing

If these are added, the paper can split baselines into groups:

### Static KV Eviction

- H2O
- StreamingLLM
- SnapKV
- SAGE-KV Lite

### Layer/Page-Aware KV Selection

- PyramidKV
- Quest or Quest-Lite

### Hybrid Memory Orchestration

- HMO

This framing makes HMO's difference clearer:

> Existing baselines decide which KV tokens/pages to retain. HMO decides which memory mode a segment should use in a hybrid-attention model.

## Practical Notes for This Repository

Current cache utilities:

```text
experiments/utils/kv_ops.py
experiments/utils/cache_access.py
experiments/utils/hmo_controller.py
experiments/phase2/runner.py
```

Current global-mask eviction:

```python
evict_kv_tokens(cache, attn_layer_indices, keep_mask)
```

This is sufficient for:

- H2O
- SnapKV
- StreamingLLM
- simple SAGE-KV Lite
- a simplified PyramidKV only if all layers share one mask

It is not sufficient for faithful:

- PyramidKV with per-layer budgets
- SAGE-KV with per-head masks
- Quest with dynamic page selection

Likely new helper functions:

```python
evict_kv_tokens_per_layer(cache, layer_to_keep_mask)
evict_kv_tokens_by_page(cache, attn_layer_indices, page_keep_mask, page_size)
collect_attention_token_scores(...)
collect_page_key_summaries(...)
```

## Recommendation for the Next Coding Step

Do not implement all three at once.

Suggested order:

1. Add `pyramidkv_lite`: layer-specific budgets, token scores by observation-window/KV norm.
2. Add `sagekv_lite`: post-prefill attention/KV-norm token scoring under same byte budget.
3. Keep Quest in related work unless a faithful attention patch is available.

This avoids turning the project into a baseline-engineering project before the HMO claim is stable.
