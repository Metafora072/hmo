import json
import tempfile
import unittest
from pathlib import Path

from experiments.phase2.e3_v2.oracle import OracleContractError
from experiments.phase2.e3_v2.run_hotpot_solvability import (
    PROJECT_ROOT,
    choose_max_fitting_prefix,
    load_protocol,
    summarize_results,
)


class HotpotSolvabilityRunnerTest(unittest.TestCase):
    def test_frozen_protocol_loads(self):
        payload, digest = load_protocol(
            PROJECT_ROOT / "refine-logs/hotpotqa_32k_solvability_protocol.json"
        )
        self.assertEqual(len(payload["cases"]), 4)
        self.assertEqual(len(digest), 64)

    def test_protocol_rejects_outcome_conditioned_selection(self):
        source = PROJECT_ROOT / "refine-logs/hotpotqa_32k_solvability_protocol.json"
        payload = json.loads(source.read_text(encoding="utf-8"))
        payload["construction"]["outcome_conditioned_selection"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(OracleContractError):
                load_protocol(path)

    def test_prefix_search_returns_longest_fitting_prefix(self):
        count, measured = choose_max_fitting_prefix(
            20, 43, lambda prefix: 7 + 3 * prefix
        )
        self.assertEqual(count, 12)
        self.assertEqual(measured, 43)

    def test_prefix_search_rejects_oversized_base(self):
        with self.assertRaises(OracleContractError):
            choose_max_fitting_prefix(10, 5, lambda prefix: 6 + prefix)

    def test_summary_reports_descriptive_routing_signals(self):
        rows = [
            {
                "official_qa_f1": score,
                "normalized_exact_match": float(score == 1.0),
                "normalized_answer_contains": float(score > 0.0),
                "construction": {"memory_context_tokens": 32768},
                "post_query_resident_kv_bytes": 1000,
            }
            for score in (1.0, 0.5, 0.0, 0.0)
        ]
        summary = summarize_results(rows)
        self.assertEqual(summary["case_count"], 4)
        self.assertEqual(summary["nonzero_f1_cases"], 2)
        self.assertTrue(summary["initial_solvability_signal"])
        self.assertTrue(summary["stronger_compressed_pilot_signal"])


if __name__ == "__main__":
    unittest.main()
