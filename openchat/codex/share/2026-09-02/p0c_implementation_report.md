# E3-v2 P0-C implementation report

Date: 2026-09-02
Branch: `dev/e3-v2-p0c`
Scope: Qwen3.5 recurrent candidate instrumentation and synthetic integrity gates

## Decision

P0-C is complete at the implementation and no-GPU contract-test level. The
E3-v2 path now derives candidates from the actual Qwen3.5 gated delta recurrence
instead of assigning physical meaning to the legacy `sigma_current` proxies.

This does not open the GPU gate. P0-D and integrated real-model preflight remain
required before P1.

## Corrected semantics

The local Transformers Qwen3.5 fallback defines:

```text
S'_t     = exp(g_t) * S_(t-1)
pred_t   = (S'_t)^T k_t
delta_t  = beta_t * (v_t - pred_t)
S_t      = S'_t + outer(k_t, delta_t)
```

Therefore:

- the physical retention multiplier is `exp(g)`, not `-g` or `1/(-g)`;
- actual update magnitude must use `delta_t`, including the state-dependent
  prediction residual;
- later retention is additive in log space and multiplicative in state space.

The old rho/c/g aggregation is unchanged in behavior and explicitly relabeled
as `sigma_current`, a historical baseline without physical interpretation.

## Implemented candidates

The immutable signal protocol is `p0c-recurrent-candidates-v1`.

- `delta_update`: RMS of the actual `delta_t` across batch, segment tokens,
  heads, and value dimensions, then mean across recurrent layers.
- `survival_retention`: mean head/layer suffix log-survival
  `sum(g_t)` after the segment boundary. `decay_risk` is its negation.
- `suffix_interference`: negative relative projection of later surviving writes
  onto the segment's surviving contribution. Positive values mean destructive
  cancellation; negative values mean alignment.
- `sigma_current`: existing legacy aggregation, retained only as a baseline.

Segment contributions use the exact final-context survival weight:

```text
C_i = sum_{t in segment i} exp(sum_{u > t} g_u) * outer(k_t, delta_t)
```

Raw candidates are not normalized or clipped at collection. Only the exponent
used to materialize a surviving contribution clamps log-survival to `[-80, 0]`.
Any later normalization must be fit on discovery data and frozen for
confirmation.

Partial tails use the same eligibility as `sigma_current`: a tail shorter than
`segment_length // 4` is excluded; eligible partial tails are retained and
flagged. This prevents silent segment-index drift in P0-D.

## Exact trace implementation

`chunk_gated_delta_trace` reproduces Qwen3.5's official chunk-WY algorithm and
exposes the effective per-token delta residuals. A transparent sequential
reference is retained solely for integrity testing.

The context hook:

- targets the loader's real path `model.model.layers[i].linear_attn`;
- reproduces Qwen3.5 projection, causal convolution, beta, and log-decay;
- accepts exactly one unpadded, fresh context prefill;
- refuses cached, duplicate, padded, missing, or multi-sample captures;
- must call `finalize_context()` at the start of the P0-B intervention callback,
  detaching all hooks before query processing.

The formulas, aggregation, clipping, normalization policy, and segment policy
are embedded in every immutable run manifest.

## Evidence

Numerical integrity:

- chunk-WY delta versus sequential reference maximum observed error: `4.8e-7`;
- final state versus sum of survival-weighted token writes error: `8.9e-8`;
- synthetic weak/strong gates confirm `exp(g)` retention direction;
- destructive and aligned suffix writes produce the preregistered score signs.

No-GPU regression:

```text
CUDA_VISIBLE_DEVICES='' .../hmo_research_v6/bin/python -m unittest \
  experiments.test_p0c_recurrent_signals \
  experiments.test_p0b_context_query \
  experiments.test_p0a_validity -v

41 tests passed:
- 16 P0-C recurrence/candidate/hook/manifest tests
- 14 P0-B context-query tests
- 11 P0-A metric/manifest tests
```

Additional checks:

- `python -m compileall -q experiments`: pass.
- `git diff --check`: pass.
- Randomly initialized real `Qwen3_5GatedDeltaNet` CPU smoke: pass, output
  `[1, 9, 1024]`, segment ends `(4, 8, 9)`, partial flags
  `(False, False, True)`.
- No pretrained weights or GPU were used.

## Residual risk and next gate

The local Transformers import reports that optional C++ extensions expect
Torch 2.11 or newer while this environment has Torch 2.7, so the real-layer CPU
smoke used the official PyTorch fallback. P0-D integrated preflight must record
and validate the actual GPU kernel/backend before scientific execution.

P0-D must now implement equal-byte multi-background oracle arms, isolated alpha
probing, recoverable intervention manifests, and integrated real-model integrity
checks. The formal GPU runner remains absent and blocked.
