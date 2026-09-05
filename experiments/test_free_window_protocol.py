import copy
import unittest

from experiments.phase2.e3_v2.freeze_free_window_dev_protocol import (
    DATASET_ORDER,
    select_cases,
)
from experiments.phase2.e3_v2.run_free_window_dev import summarize


def _row(dataset: str, index: int) -> dict:
    return {
        "dataset": dataset,
        "sample_id": f"{dataset}_{index:04d}",
        "record_index": index,
        "record_sha256": f"record-{index}",
        "context_tokens": 1000 + index * 10,
        "query_tokens": 8,
        "query_probe": {"probe_id": f"{dataset}-probe-{index}"},
        "systems": {"unused": {"official_qa_f1": float(index)}},
    }


class FreeWindowProtocolTests(unittest.TestCase):
    def test_development_selection_is_stratified_and_outcome_independent(self):
        rows = [
            _row(dataset, index)
            for dataset in DATASET_ORDER
            for index in range(40)
        ]
        selected = select_cases(rows, per_dataset=20, strata=4, seed=17)
        changed = copy.deepcopy(rows)
        for row in changed:
            row["systems"]["unused"]["official_qa_f1"] *= -1000.0
        self.assertEqual(
            selected, select_cases(changed, per_dataset=20, strata=4, seed=17)
        )
        self.assertEqual(len(selected), 120)
        self.assertEqual(len({case["probe_id"] for case in selected}), 120)
        for dataset in DATASET_ORDER:
            members = [case for case in selected if case["dataset"] == dataset]
            self.assertEqual(len(members), 20)
            counts = {
                stratum: sum(
                    case["length_stratum"] == stratum for case in members
                )
                for stratum in range(4)
            }
            self.assertEqual(counts, {0: 5, 1: 5, 2: 5, 3: 5})

    def test_development_summary_reports_paired_deltas_and_equal_bytes(self):
        systems = (
            "hmo_legacy",
            "hmo_layer_local",
            "chunkkv",
            "hmo_free_window",
            "full_kv_reference",
        )
        rows = []
        for index, dataset in enumerate(DATASET_ORDER[:2]):
            payload = {}
            for offset, system in enumerate(systems):
                payload[system] = {
                    "official_qa_f1": float(index + offset),
                    "normalized_answer_contains": float(offset % 2),
                    "normalized_exact_match": 0.0,
                    "post_query_resident_kv_bytes": (
                        100 if system != "full_kv_reference" else 1000
                    ),
                }
            rows.append({"dataset": dataset, "systems": payload})
        analysis = summarize(
            rows,
            (
                ("hmo_free_window", "chunkkv"),
                ("hmo_layer_local", "hmo_legacy"),
            ),
        )
        overall = analysis["overall"]
        self.assertEqual(overall["case_count"], 2)
        self.assertEqual(overall["equal_resident_byte_cases"], 2)
        self.assertEqual(
            overall["comparisons"]["hmo_free_window_vs_chunkkv"]
            ["official_qa_f1"]["mean_delta"],
            1.0,
        )
        self.assertEqual(
            overall["comparisons"]["hmo_layer_local_vs_hmo_legacy"]
            ["official_qa_f1"]["wins"],
            2,
        )


if __name__ == "__main__":
    unittest.main()
