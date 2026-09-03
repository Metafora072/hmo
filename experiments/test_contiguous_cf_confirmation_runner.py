"""CPU contract tests for the contiguous CF confirmation runner."""
from __future__ import annotations

import unittest

from experiments.phase2.e3_v2.run_contiguous_cf_confirmation import (
    COMPRESSED_SYSTEMS,
    PROJECT_ROOT,
    SYSTEMS,
    load_confirmation_protocol,
    summarize_confirmation_results,
)


class ContiguousCFConfirmationRunnerTests(unittest.TestCase):
    def test_frozen_protocol_has_fresh_complete_sample_package(self):
        protocol, protocol_sha = load_confirmation_protocol(
            PROJECT_ROOT / "refine-logs/contiguous_cf_confirmation_protocol.json"
        )
        self.assertEqual(len(protocol_sha), 64)
        self.assertFalse(protocol["execution"]["continuation_gate"])
        self.assertEqual(protocol["execution"]["case_count"], 48)
        self.assertNotEqual(
            protocol["stages"]["8k"]["seed"],
            protocol["stages"]["16k"]["seed"],
        )
        self.assertEqual(tuple(protocol["systems"]), SYSTEMS)

    def test_summary_reports_primary_mechanism_and_task_groups(self):
        rows = []
        for stage in ("8k", "16k"):
            for dataset in ("needle", "longeval_lines"):
                values = {
                    "contiguous_cf": 1.0,
                    "scattered_cf": 0.0,
                    "contiguous_sparse_only": 0.0,
                    "raw_alpha_exact_topk": 0.5,
                    "full_kv_reference": 1.0,
                }
                rows.append(
                    {
                        "stage": stage,
                        "dataset": dataset,
                        "systems": {
                            system: {
                                "normalized_answer_contains": value,
                                "normalized_exact_match": value,
                                "token_f1": value,
                                "post_query_resident_kv_bytes": (
                                    100 if system in COMPRESSED_SYSTEMS else 1000
                                ),
                            }
                            for system, value in values.items()
                        },
                    }
                )
        summary = summarize_confirmation_results(rows)
        self.assertEqual(summary["case_count"], 4)
        self.assertEqual(len(summary["by_stage_dataset"]), 4)
        primary = summary["comparisons"][
            "contiguous_cf_vs_raw_alpha_exact_topk"
        ]["normalized_answer_contains"]
        self.assertEqual(primary["mean_delta"], 0.5)
        self.assertEqual(primary["wins"], 4)
        self.assertEqual(summary["equal_compressed_resident_byte_cases"], 4)


if __name__ == "__main__":
    unittest.main()
