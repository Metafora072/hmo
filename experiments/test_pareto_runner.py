"""CPU contract tests for the frozen HMO Pareto runner."""
from __future__ import annotations

import unittest

from experiments.phase2.e3_v2.run_pareto import (
    BUDGET_FRACTIONS,
    EQUAL_BYTE_SYSTEMS,
    PROJECT_ROOT,
    SYSTEMS,
    load_pareto_protocol,
    summarize_pareto_results,
)


class ParetoRunnerTests(unittest.TestCase):
    def test_frozen_protocol_matches_the_approved_package(self):
        protocol, protocol_sha = load_pareto_protocol(
            PROJECT_ROOT / "refine-logs/contiguous_cf_pareto_protocol.json"
        )
        self.assertEqual(len(protocol_sha), 64)
        self.assertEqual(tuple(protocol["systems"]), SYSTEMS)
        self.assertEqual(
            tuple(protocol["equal_byte_systems"]), EQUAL_BYTE_SYSTEMS
        )
        self.assertEqual(
            tuple(protocol["budget_fractions"]), BUDGET_FRACTIONS
        )
        self.assertEqual(protocol["execution"]["formal_sample_cases"], 48)
        self.assertEqual(protocol["execution"]["formal_budget_cases"], 144)
        self.assertFalse(protocol["execution"]["continuation_gate"])

    def test_summary_is_budget_stratified_and_checks_equal_bytes(self):
        rows = []
        for fraction in BUDGET_FRACTIONS:
            for sample_index in range(2):
                resident = int(fraction * 1000)
                systems = {}
                for system in SYSTEMS:
                    score = (
                        1.0
                        if system == "contiguous_cf"
                        or system == "full_kv_reference"
                        else 0.0
                    )
                    systems[system] = {
                        "normalized_answer_contains": score,
                        "normalized_exact_match": score,
                        "token_f1": score,
                        "post_query_resident_kv_bytes": (
                            1000
                            if system == "full_kv_reference"
                            else resident
                        ),
                    }
                rows.append(
                    {
                        "stage": "8k",
                        "sample_id": f"sample_{sample_index}",
                        "dataset": "needle",
                        "budget_fraction": fraction,
                        "systems": systems,
                    }
                )
        summary = summarize_pareto_results(rows)
        self.assertEqual(summary["budget_case_count"], 6)
        self.assertEqual(summary["sample_case_count"], 2)
        self.assertEqual(set(summary["by_budget"]), {"05pct", "10pct", "20pct"})
        for budget in summary["by_budget"].values():
            self.assertEqual(budget["equal_resident_byte_cases"], 2)
            comparison = budget["comparisons"][
                "contiguous_cf_vs_global_fixed_chunk_topk"
            ]["normalized_answer_contains"]
            self.assertEqual(comparison["wins"], 2)
            self.assertEqual(comparison["mean_delta"], 1.0)


if __name__ == "__main__":
    unittest.main()
