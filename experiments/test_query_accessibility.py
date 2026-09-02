"""Integrity tests for query-conditioned recurrent accessibility."""
from __future__ import annotations

import math
import unittest

import torch

from experiments.phase2.e3_v2.query_accessibility import segment_query_readout
from experiments.phase2.e3_v2.recurrent_signals import (
    RecurrentSignalError,
    chunk_gated_delta_trace,
    surviving_segment_contributions,
    surviving_segment_contributions_for_bounds,
)


class SegmentContributionTests(unittest.TestCase):
    def test_segment_contributions_reconstruct_final_state(self):
        torch.manual_seed(31)
        key = torch.randn(1, 9, 2, 4)
        value = torch.randn(1, 9, 2, 3)
        beta = torch.sigmoid(torch.randn(1, 9, 2))
        log_decay = -torch.rand(1, 9, 2) * 0.4
        trace = chunk_gated_delta_trace(
            key,
            value,
            beta,
            log_decay,
            chunk_size=4,
        )
        result = surviving_segment_contributions(trace, segment_length=4)
        self.assertEqual(result.segment_starts, (0, 4, 8))
        self.assertEqual(result.segment_ends, (4, 8, 9))
        torch.testing.assert_close(
            result.values.sum(dim=0),
            trace.final_state,
            rtol=2e-5,
            atol=2e-6,
        )

    def test_tiny_tail_matches_existing_segment_policy(self):
        trace = chunk_gated_delta_trace(
            torch.ones(1, 17, 1, 1),
            torch.ones(1, 17, 1, 1),
            torch.full((1, 17, 1), 0.5),
            torch.full((1, 17, 1), -0.1),
            chunk_size=8,
        )
        result = surviving_segment_contributions(trace, segment_length=8)
        self.assertEqual(result.segment_starts, (0, 8))
        self.assertEqual(result.segment_ends, (8, 16))
    def test_explicit_catalog_keeps_tiny_tail_and_reconstructs_state(self):
        trace = chunk_gated_delta_trace(
            torch.ones(1, 17, 1, 1),
            torch.ones(1, 17, 1, 1),
            torch.full((1, 17, 1), 0.5),
            torch.full((1, 17, 1), -0.1),
            chunk_size=8,
        )
        result = surviving_segment_contributions_for_bounds(
            trace,
            segment_starts=(0, 8, 16),
            segment_ends=(8, 16, 17),
            segment_length=8,
        )
        self.assertEqual(result.segment_ends, (8, 16, 17))
        torch.testing.assert_close(
            result.values.sum(dim=0),
            trace.final_state,
            rtol=2e-5,
            atol=2e-6,
        )



class QueryReadoutTests(unittest.TestCase):
    def test_query_selects_matching_segment_contribution(self):
        query = torch.tensor([[[[1.0, 0.0]]]])
        contributions = torch.tensor(
            [
                [[[[1.0], [0.0]]]],
                [[[[0.0], [2.0]]]],
            ]
        )
        norm, share, alignment = segment_query_readout(
            query,
            contributions,
            torch.zeros(1, 1),
        )
        torch.testing.assert_close(norm, torch.tensor([1.0, 0.0]))
        torch.testing.assert_close(share, torch.tensor([1.0, 0.0]))
        torch.testing.assert_close(alignment, torch.tensor([1.0, 0.0]))

    def test_query_decay_scales_norm_but_not_relative_share(self):
        query = torch.tensor([[[[1.0, 1.0]]]]) / math.sqrt(2)
        contributions = torch.tensor(
            [
                [[[[2.0], [0.0]]]],
                [[[[0.0], [1.0]]]],
            ]
        )
        base = segment_query_readout(query, contributions, torch.zeros(1, 1))
        decayed = segment_query_readout(
            query,
            contributions,
            torch.full((1, 1), -2.0),
        )
        torch.testing.assert_close(decayed[0], base[0] * math.exp(-2.0))
        torch.testing.assert_close(decayed[1], base[1])
        torch.testing.assert_close(decayed[2], base[2])

    def test_invalid_dimensions_fail_closed(self):
        with self.assertRaisesRegex(RecurrentSignalError, "rank five"):
            segment_query_readout(
                torch.ones(1, 1, 1, 1),
                torch.ones(1, 1, 1, 1),
                torch.zeros(1, 1),
            )


if __name__ == "__main__":
    unittest.main()
