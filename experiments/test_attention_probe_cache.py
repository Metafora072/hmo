"""Contracts for attention-only v2 probe artifacts."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from experiments.phase2.e3_v2.alpha_probe import AlphaProbeResult
from experiments.phase2.e3_v2.attention_probe import AttentionTokenProbeResult
from experiments.phase2.e3_v2.attention_probe_cache import (
    ATTENTION_PROBE_AGGREGATION,
    ATTENTION_PROBE_CACHE_SCHEMA,
    get_or_create_attention_probe,
)
from experiments.phase2.e3_v2.context_query import TokenizedPromptSplit
from experiments.phase2.e3_v2.oracle import OracleContractError, SegmentSpec
from experiments.utils.eval_harness import PromptTextParts


MODEL_IDENTITY = {
    "model_id": "Qwen/test",
    "revision": "a" * 40,
    "config_sha256": "b" * 64,
    "weight_index_sha256": "c" * 64,
}


def _segments():
    return tuple(
        SegmentSpec(
            segment_id=index,
            start=index * 2,
            end=(index + 1) * 2,
            token_count=2,
            kv_bytes=16,
            protected=index in {0, 2},
            partial=False,
            normalized_position=(index + 0.5) / 3,
            position_bin=index,
        )
        for index in range(3)
    )


def _prompt(query_ids=(7, 8)):
    context = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
    query = torch.tensor([list(query_ids)], dtype=torch.long)
    return TokenizedPromptSplit(
        text=PromptTextParts("context", "query"),
        context_ids=context,
        query_ids=query,
        full_ids=torch.cat([context, query], dim=1),
        split_token_index=6,
    )


def _result():
    aggregate = (0.15, 0.25, 0.35, 0.45, 0.55, 0.65)
    return AttentionTokenProbeResult(
        alpha=AlphaProbeResult(
            context_tokens=6,
            query_tokens=2,
            attention_layer_indices=(1, 3),
            segment_ids=(0, 1, 2),
            attention_mass=(0.4, 0.8, 1.2),
        ),
        token_attention_mass=aggregate,
        layer_token_attention_mass=(
            (0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
            (0.2, 0.3, 0.4, 0.5, 0.6, 0.7),
        ),
    )


class AttentionProbeCacheTests(unittest.TestCase):
    def _call(self, directory: Path, collector, prompt=None):
        return get_or_create_attention_probe(
            directory,
            model_identity=MODEL_IDENTITY,
            prompt=prompt or _prompt(),
            attention_layer_indices=(1, 3),
            segments=_segments(),
            segment_length=2,
            collector=collector,
        )

    def test_collects_once_and_reloads_both_arrays(self):
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            first = self._call(
                Path(directory), lambda: calls.append("collect") or _result()
            )
            second = self._call(
                Path(directory), lambda: self.fail("must not recollect")
            )
            self.assertEqual(calls, ["collect"])
            self.assertFalse(first.cache_hit)
            self.assertTrue(second.cache_hit)
            self.assertEqual(first.result, second.result)
            self.assertEqual(
                second.result.layer_scores()[3],
                second.result.layer_token_attention_mass[1],
            )
            self.assertEqual(
                first.provenance()["aggregation_version"],
                ATTENTION_PROBE_AGGREGATION,
            )

    def test_recurrent_identity_is_absent_and_query_changes_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            first = self._call(Path(directory), _result)
            metadata = json.loads(first.metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["schema_version"], ATTENTION_PROBE_CACHE_SCHEMA)
            self.assertNotIn("recurrent_layer_indices", metadata["identity"])
            second = self._call(Path(directory), _result, prompt=_prompt((7, 9)))
            self.assertNotEqual(first.probe_id, second.probe_id)

    def test_corrupt_layer_array_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            cached = self._call(Path(directory), _result)
            cached.layer_score_path.write_bytes(b"corrupt")
            with self.assertRaisesRegex(
                OracleContractError, "cannot load persistent attention probe"
            ):
                self._call(Path(directory), _result)

    def test_incomplete_artifact_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            cached = self._call(Path(directory), _result)
            cached.score_path.unlink()
            with self.assertRaisesRegex(OracleContractError, "incomplete"):
                self._call(Path(directory), _result)


if __name__ == "__main__":
    unittest.main()
