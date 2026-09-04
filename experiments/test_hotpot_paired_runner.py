import json
import tempfile
import unittest
from pathlib import Path

from experiments.phase2.e3_v2.oracle import OracleContractError
from experiments.phase2.e3_v2.run_hotpot_paired import (
    PROJECT_ROOT,
    load_paired_protocol,
    summarize_results,
)


class HotpotPairedRunnerTest(unittest.TestCase):
    def test_frozen_protocol_loads(self):
        payload, digest = load_paired_protocol(
            PROJECT_ROOT / "refine-logs/hotpotqa_32k_paired_protocol.json"
        )
        self.assertEqual(payload["middle_kv_fraction"], 0.1)
        self.assertEqual(len(payload["equal_byte_systems"]), 4)
        self.assertEqual(len(digest), 64)

    def test_protocol_rejects_budget_change(self):
        source = PROJECT_ROOT / "refine-logs/hotpotqa_32k_paired_protocol.json"
        payload = json.loads(source.read_text(encoding="utf-8"))
        payload["middle_kv_fraction"] = 0.2
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(OracleContractError):
                load_paired_protocol(path)

    def test_summary_reports_equal_bytes_and_paired_deltas(self):
        names = (
            "contiguous_cf",
            "global_fixed_chunk_topk",
            "raw_alpha_exact_slack",
            "scattered_cf",
            "full_kv_reference",
        )
        rows = []
        for hmo, fixed in ((1.0, 0.5), (0.0, 0.0)):
            systems = {}
            for name in names:
                score = hmo if name == "contiguous_cf" else fixed
                systems[name] = {
                    "official_qa_f1": score,
                    "normalized_answer_contains": float(score > 0),
                    "normalized_exact_match": 0.0,
                    "post_query_resident_kv_bytes": (
                        1000 if name != "full_kv_reference" else 10000
                    ),
                }
            rows.append({"systems": systems})
        summary = summarize_results(rows)
        self.assertEqual(summary["equal_resident_byte_cases"], 2)
        self.assertEqual(summary["systems"]["contiguous_cf"]["official_qa_f1"], 0.5)
        comparison = summary["comparisons"][
            "contiguous_cf_vs_global_fixed_chunk_topk"
        ]["official_qa_f1"]
        self.assertEqual(comparison["mean_delta"], 0.25)
        self.assertEqual((comparison["wins"], comparison["ties"], comparison["losses"]), (1, 1, 0))


if __name__ == "__main__":
    unittest.main()
