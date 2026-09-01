"""Immutable protocol constants for the preregistered E3-v2 path."""

P0B_PROTOCOL_VERSION = "p0b-context-query-v1"
P0B_EXECUTION_EVENTS = (
    "context_prefill_complete",
    "context_kv_intervention_complete",
    "query_suffix_complete",
    "first_answer_logits_ready",
)
P0B_ANSWER_PREFIX = " "

P0C_PROTOCOL_VERSION = "p0c-recurrent-candidates-v1"


def recurrent_signal_protocol() -> dict:
    """Return a fresh, JSON-serializable copy of the frozen P0-C protocol."""
    return {
        "version": P0C_PROTOCOL_VERSION,
        "model_family": "qwen3.5_gated_deltanet",
        "state_update": "S_t=exp(g_t)*S_(t-1)+outer(k_t,delta_t)",
        "delta_residual": "delta_t=beta_t*(v_t-(exp(g_t)*S_(t-1))^T*k_t)",
        "key_normalization": "l2_eps_1e-6",
        "segment_policy": "fixed_width_partial_tail_min_quarter_segment",
        "candidates": {
            "sigma_current": {
                "source": "legacy_compute_segment_saturation",
                "role": "historical_baseline_without_physical_interpretation",
            },
            "delta_update": {
                "field": "delta_update_rms",
                "aggregation": "rms_batch_token_head_value_then_mean_layers",
            },
            "survival_retention": {
                "field": "log_survival",
                "formula": "mean_batch_head(sum_g_after_segment_end)",
                "companion": "decay_risk=-log_survival",
                "layer_aggregation": "mean",
            },
            "suffix_interference": {
                "field": "suffix_interference",
                "formula": "mean_batch_head(-dot(C_i,L_i)/(norm2(C_i)+1e-8))",
                "segment_contribution": (
                    "C_i=sum_t_in_i(exp(sum_g_after_t)*outer(k_t,delta_t))"
                ),
                "later_contribution": "L_i=sum_j_after_i(C_j)",
                "layer_aggregation": "mean",
            },
        },
        "clipping": {
            "log_survival": "none",
            "survival_exp_only": "clamp_log_to_minus_80_0",
            "candidate_values": "none",
        },
        "normalization": "none_at_collection_discovery_fit_only",
    }
