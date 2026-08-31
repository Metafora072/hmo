"""
Strict theory-faithfulness guardrails for the HMO prototype.

The user explicitly requested that prototype experiments must not run when the
implementation still contains approximations, substitutions, or semantic drift
relative to the canonical theory documents.
"""
from __future__ import annotations


_BLOCK_REASONS = {
    "v1fast": [
        "protocol mismatch: fast V1 uses approximate screening labels and thresholds instead of the canonical V1 oracle and pass criteria",
        "data mismatch: fast V1 uses Needle-only screening, but canonical V1 requires information-density-varying data",
    ],
    "v2fast": [
        "protocol mismatch: V2fast keeps the same non-budget-matched condition construction as V2",
        "screening mismatch: V2fast uses shortened contexts and screening thresholds rather than the canonical theory-validation protocol",
    ],
    "v3fast": [
        "protocol mismatch: V3fast still inherits the non-budget-matched V3 construction",
        "screening mismatch: V3fast is a convenience screen, not the canonical V3 protocol",
    ],
    "v4fast": [
        "protocol mismatch: V4fast still inherits the non-budget-matched V4 condition construction",
        "screening mismatch: V4fast is a convenience screen, not the canonical joint-validation protocol",
    ],
}


def require_theory_faithful(experiment_name: str) -> None:
    """
    Fail closed when an experiment still relies on approximate prototype paths.

    The current project state intentionally prefers a hard stop over silently
    producing misleading results.
    """
    key = experiment_name.lower()
    reasons = _BLOCK_REASONS.get(key)
    if not reasons:
        return

    bullet_list = "\n".join(f"- {reason}" for reason in reasons)
    raise RuntimeError(
        "Strict theory-faithfulness guard blocked this run.\n"
        f"Experiment: {experiment_name}\n"
        "Current blocking gaps:\n"
        f"{bullet_list}\n"
        "See refine-logs/CANONICAL_PROTOTYPE_CONTRACT.md and "
        "refine-logs/THEORY_CODE_ALIGNMENT_AUDIT.md before rerunning."
    )
