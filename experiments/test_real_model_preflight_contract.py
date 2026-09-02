"""CPU-only contract tests for the E3-v2 real-model preflight runner."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from experiments.phase2.e3_v2.context_query import TokenizedPromptSplit
from experiments.phase2.e3_v2.real_model_preflight import (
    _capture_check,
    _logit_difference,
    _needle_token_indices,
    _synthetic_gate_direction,
    model_provenance,
)
from experiments.utils.eval_harness import PromptTextParts


class CharacterTokenizer:
    def __call__(
        self,
        text,
        *,
        add_special_tokens,
        return_offsets_mapping,
        return_tensors,
    ):
        del add_special_tokens, return_offsets_mapping, return_tensors
        return {
            "input_ids": torch.arange(len(text), dtype=torch.long).unsqueeze(0),
            "offset_mapping": torch.tensor(
                [[(index, index + 1) for index in range(len(text))]],
                dtype=torch.long,
            ),
        }


class ProvenanceTests(unittest.TestCase):
    def test_model_snapshot_identity_is_path_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "snapshots" / "revision-123"
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text('{"model_type":"qwen"}\n')
            (snapshot / "model.safetensors.index.json").write_text(
                '{"weight_map":{}}\n'
            )
            (snapshot / "model.safetensors").write_bytes(b"weights")
            identity = model_provenance(
                snapshot, "Qwen/test", revision="pinned-revision"
            )
            self.assertEqual(identity["model_id"], "Qwen/test")
            self.assertEqual(identity["revision"], "pinned-revision")
            self.assertEqual(identity["weight_files"][0]["size_bytes"], 7)
            self.assertEqual(len(identity["weight_files"][0]["sha256"]), 64)
            self.assertNotIn(str(snapshot), str(identity))


class NeedleLocationTests(unittest.TestCase):
    def make_prompt(self, memory="abc NEEDLE xyz", query=" question"):
        full = memory + query
        ids = torch.arange(len(full), dtype=torch.long).unsqueeze(0)
        return TokenizedPromptSplit(
            text=PromptTextParts(memory_context=memory, query_suffix=query),
            context_ids=ids[:, : len(memory)],
            query_ids=ids[:, len(memory) :],
            full_ids=ids,
            split_token_index=len(memory),
        )

    def test_needle_offsets_are_limited_to_memory_context(self):
        prompt = self.make_prompt()
        self.assertEqual(
            _needle_token_indices(CharacterTokenizer(), prompt, "NEEDLE"),
            tuple(range(4, 10)),
        )

    def test_missing_or_duplicate_needle_fails_closed(self):
        with self.assertRaisesRegex(Exception, "exactly once"):
            _needle_token_indices(CharacterTokenizer(), self.make_prompt(), "absent")
        with self.assertRaisesRegex(Exception, "exactly once"):
            _needle_token_indices(
                CharacterTokenizer(), self.make_prompt(memory="NEEDLE NEEDLE"), "NEEDLE"
            )


class EvidenceTests(unittest.TestCase):
    def test_logit_difference_reports_exact_and_rank_changes(self):
        left = torch.tensor([[0.0, 2.0, 1.0]])
        right = torch.tensor([[0.0, 1.0, 3.0]])
        metrics = _logit_difference(left, right)
        self.assertEqual(metrics["max_abs"], 2.0)
        self.assertFalse(metrics["argmax_equal"])
        self.assertFalse(metrics["exact_equal"])
        self.assertGreater(metrics["js_divergence"], 0.0)
        self.assertGreater(metrics["max_probability_abs"], 0.0)
        self.assertEqual(metrics["top10_overlap"], 1.0)

    def test_capture_records_failure_as_nonempty_evidence(self):
        checks = {}

        def fail():
            raise RuntimeError("expected failure")

        _capture_check(checks, "probe", fail)
        self.assertFalse(checks["probe"].passed)
        self.assertIn("expected failure", checks["probe"].evidence)

    def test_synthetic_gate_direction_matches_exp_g(self):
        passed, metrics = _synthetic_gate_direction()
        self.assertTrue(passed)
        self.assertGreater(
            metrics["weak_final_state_norm"], metrics["strong_final_state_norm"]
        )


if __name__ == "__main__":
    unittest.main()
