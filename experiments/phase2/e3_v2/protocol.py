"""Immutable protocol constants for the preregistered E3-v2 path."""

P0B_PROTOCOL_VERSION = "p0b-context-query-v2"
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


P0D_PROTOCOL_VERSION = "p0d-equal-byte-oracle-v1"


def oracle_protocol() -> dict:
    """Return the frozen P0-D oracle and analysis protocol."""
    return {
        "version": P0D_PROTOCOL_VERSION,
        "unit": "full_nonprotected_nonpartial_context_segment",
        "protected_prefix_segments": 1,
        "protected_suffix_segments": 1,
        "middle_kv_fraction": 0.10,
        "budget_rounding": "floor_complete_equal_cost_segments",
        "nonexact_action": "drop_attention_kv",
        "donors_per_segment": 3,
        "backgrounds_per_pair": 3,
        "position_bins": 4,
        "selection": "sha256_deterministic_cross_bin_preferred",
        "alpha": {
            "cache": "isolated_fresh_full_kv_context_then_query",
            "source": "full_attention_query_to_context_attention_mass",
            "aggregation": "mean_layers_heads_query_tokens_then_sum_segment_keys",
            "normalization": "none_at_collection",
        },
        "quality": {
            "primary": "mean_gold_answer_logprob_per_token",
            "secondary": "official_dataset_metric",
            "pair_delta": "Q(R_union_i)-Q(R_union_j)",
            "background_reduction": "mean_per_unordered_pair_before_ranking",
            "segment_utility": "mean_signed_pair_delta",
        },
        "statistics": {
            "uncertainty_unit": "sample",
            "pairwise_tie_credit": 0.5,
            "ndcg_relevance": "utility_minus_within_sample_minimum",
            "ndcg_k": "byte_budget_segment_slots",
            "partial_association_controls": ["alpha", "normalized_position"],
            "diagnostic_model": "train_fold_standardized_ridge",
            "ridge_lambda": 0.001,
            "grouped_folds": 5,
            "bootstrap_samples": 2000,
            "bootstrap_seed": 20260901,
            "candidate_selection": (
                "discovery_select_one_by_grouped_cv_pairwise_then_freeze_confirmation"
            ),
        },
        "integrity_gate": "all_eight_preregistered_checks_required",
    }
