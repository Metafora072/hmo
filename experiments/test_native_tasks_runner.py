"""CPU contract tests for the frozen native LongBench runner."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.phase2.e3_v2.oracle import OracleContractError
from experiments.phase2.e3_v2.run_native_tasks import (
    EQUAL_BYTE_SYSTEMS,
    PROJECT_ROOT,
    SYSTEMS,
    _selected_cases,
    load_native_protocol,
    select_longest_candidates,
    summarize_native_results,
)


class NativeLongBenchRunnerTest(unittest.TestCase):
    def test_frozen_protocol_loads(self):
        payload, digest = load_native_protocol(
            PROJECT_ROOT / "refine-logs/native_longbench_protocol.json"
        )
        self.assertEqual(sum(len(value["cases"]) for value in payload["datasets"].values()), 24)
        self.assertFalse(payload["selection"]["augmentation"])
        self.assertFalse(payload["selection"]["truncation"])
        self.assertEqual(len(digest), 64)

    def test_six_task_protocol_freezes_precommitted_prefixes(self):
        payload, digest = load_native_protocol(
            PROJECT_ROOT / "refine-logs/native_longbench_six_task_9b_protocol.json"
        )
        self.assertEqual(len(digest), 64)
        self.assertEqual(len(_selected_cases(payload, "prefix50")), 295)
        self.assertEqual(len(_selected_cases(payload, "prefix100")), 506)
        self.assertFalse(payload["execution"]["continuation_gate"])

    def test_protocol_rejects_outcome_filtering(self):
        source = PROJECT_ROOT / "refine-logs/native_longbench_protocol.json"
        payload = json.loads(source.read_text(encoding="utf-8"))
        payload["selection"]["outcome_conditioned_selection"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(OracleContractError):
                load_native_protocol(path)

    def test_candidate_selection_uses_length_then_index(self):
        metadata = [
            {"index": 4, "context_tokens": 9000},
            {"index": 1, "context_tokens": 9000},
            {"index": 2, "context_tokens": 17000},
            {"index": 3, "context_tokens": 8500},
        ]
        selected = select_longest_candidates(metadata, 8192, 16384, 3)
        self.assertEqual([item["index"] for item in selected], [1, 4, 3])

    def test_summary_reports_overall_and_dataset_slices(self):
        rows = []
        for dataset, hmo, baseline in (
            ("longbench_hotpotqa", 1.0, 0.0),
            ("longbench_narrativeqa", 0.5, 0.5),
        ):
            systems = {}
            for name in SYSTEMS:
                score = hmo if name == "contiguous_cf" else baseline
                systems[name] = {
                    "official_qa_f1": score,
                    "normalized_answer_contains": float(score > 0),
                    "normalized_exact_match": float(score == 1),
                    "post_query_resident_kv_bytes": 100 if name in EQUAL_BYTE_SYSTEMS else 1000,
                }
            rows.append({"dataset": dataset, "systems": systems})
        summary = summarize_native_results(rows)
        self.assertEqual(summary["case_count"], 2)
        self.assertEqual(summary["equal_resident_byte_cases"], 2)
        self.assertEqual(summary["by_dataset"]["hotpotqa"]["case_count"], 1)
        comparison = summary["comparisons"]["contiguous_cf_vs_chunkkv"]["official_qa_f1"]
        self.assertEqual((comparison["wins"], comparison["ties"], comparison["losses"]), (1, 1, 0))


if __name__ == "__main__":
    unittest.main()
