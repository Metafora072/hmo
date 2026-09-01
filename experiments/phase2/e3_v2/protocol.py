"""Immutable protocol constants for the preregistered E3-v2 P0-B path."""

P0B_PROTOCOL_VERSION = "p0b-context-query-v1"
P0B_EXECUTION_EVENTS = (
    "context_prefill_complete",
    "context_kv_intervention_complete",
    "query_suffix_complete",
    "first_answer_logits_ready",
)
P0B_ANSWER_PREFIX = " "
