"""Fail-closed integrity gate for the E3-v2 scientific runner."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from experiments.phase2.e3_v2.oracle import OracleContractError


REQUIRED_INTEGRITY_CHECKS = (
    "equal_byte_arms",
    "query_after_intervention",
    "full_kv_equivalence",
    "repeated_arm_determinism",
    "recurrent_gate_direction",
    "controlled_needle_logit_effect",
    "alpha_isolation",
    "manifest_recoverability",
)


@dataclass(frozen=True)
class IntegrityCheck:
    passed: bool
    evidence: str

    def validate(self, name: str) -> None:
        if not isinstance(self.passed, bool) or not self.evidence.strip():
            raise OracleContractError(
                f"integrity check {name!r} requires a boolean result and evidence"
            )


@dataclass(frozen=True)
class IntegrityGateReport:
    checks: Mapping[str, IntegrityCheck]

    def require_pass(self) -> None:
        expected = set(REQUIRED_INTEGRITY_CHECKS)
        actual = set(self.checks)
        if actual != expected:
            raise OracleContractError(
                "integrity gate check set mismatch: "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )
        failures = []
        for name in REQUIRED_INTEGRITY_CHECKS:
            check = self.checks[name]
            if not isinstance(check, IntegrityCheck):
                raise OracleContractError(f"integrity check {name!r} has an invalid type")
            check.validate(name)
            if not check.passed:
                failures.append(name)
        if failures:
            raise OracleContractError(
                f"scientific execution blocked by integrity checks: {failures}"
            )

    def to_dict(self) -> dict:
        self.require_pass()
        return {
            "status": "pass",
            "checks": {
                name: {
                    "passed": self.checks[name].passed,
                    "evidence": self.checks[name].evidence,
                }
                for name in REQUIRED_INTEGRITY_CHECKS
            },
        }


def require_integrity_gate(checks: Mapping[str, IntegrityCheck]) -> IntegrityGateReport:
    """Return a validated report or block scientific execution."""
    report = IntegrityGateReport(checks=dict(checks))
    report.require_pass()
    return report
