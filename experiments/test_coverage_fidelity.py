"""Focused CPU tests for the coverage-fidelity allocator."""
from __future__ import annotations

import unittest

from experiments.phase2.e3_v2.coverage_fidelity import (
    CoverageFidelityError,
    allocate_coverage_fidelity,
)
from experiments.phase2.e3_v2.oracle import SegmentSpec


def _segments(count: int = 6, *, token_count: int = 10, kv_bytes: int = 100):
    return tuple(
        SegmentSpec(
            segment_id=index,
            start=index * token_count,
            end=(index + 1) * token_count,
            token_count=token_count,
            kv_bytes=kv_bytes,
            protected=index in {0, count - 1},
            partial=False,
            normalized_position=(index + 0.5) / count,
            position_bin=min(index, 3),
        )
        for index in range(count)
    )


def _signals():
    return (
        {1: 0.9, 2: 0.8, 3: 0.7, 4: 0.6},
        {1: 0.9, 2: 0.1, 3: 0.8, 4: 0.2},
    )


class CoverageFidelityAllocatorTests(unittest.TestCase):
    def test_protected_exact_and_sparse_coverage_precedes_upgrades(self):
        attention, accessibility = _signals()
        plan = allocate_coverage_fidelity(
            attention,
            accessibility,
            _segments(),
            middle_kv_fraction=0.4,
            sparse_width=2,
        )
        by_id = {item.segment_id: item for item in plan.allocations}
        self.assertEqual(plan.protected_kv_bytes, 200)
        self.assertEqual(plan.middle_budget_limit_bytes, 160)
        self.assertTrue(all(by_id[index].action == "exact" for index in (0, 5)))
        self.assertTrue(all(by_id[index].retained_tokens >= 2 for index in range(1, 5)))
        self.assertEqual(plan.total_charged_bytes, 360)

    def test_insufficient_floor_uses_attention_density(self):
        attention, accessibility = _signals()
        plan = allocate_coverage_fidelity(
            attention,
            accessibility,
            _segments(),
            middle_kv_fraction=0.15,
            sparse_width=2,
            enable_exact_upgrades=False,
        )
        covered = [
            item.segment_id for item in plan.allocations if item.action == "sparse"
        ]
        self.assertEqual(covered, [1, 2, 3])

    def test_accessibility_changes_only_fidelity_priority_at_equal_bytes(self):
        attention, accessibility = _signals()
        kwargs = dict(
            segments=_segments(), middle_kv_fraction=0.4, sparse_width=2
        )
        cf = allocate_coverage_fidelity(attention, accessibility, **kwargs)
        no_access = allocate_coverage_fidelity(
            attention, accessibility, use_accessibility=False, **kwargs
        )
        cf_tokens = {item.segment_id: item.retained_tokens for item in cf.allocations}
        no_access_tokens = {
            item.segment_id: item.retained_tokens for item in no_access.allocations
        }
        self.assertEqual(cf.total_charged_bytes, no_access.total_charged_bytes)
        self.assertEqual(
            {key for key, value in cf_tokens.items() if value > 0},
            {key for key, value in no_access_tokens.items() if value > 0},
        )
        self.assertNotEqual(cf_tokens, no_access_tokens)

    def test_disabled_accessibility_accepts_none_and_matches_legacy_call(self):
        attention, accessibility = _signals()
        kwargs = dict(
            segments=_segments(),
            middle_kv_fraction=0.4,
            sparse_width=2,
            use_accessibility=False,
        )
        legacy = allocate_coverage_fidelity(attention, accessibility, **kwargs)
        omitted = allocate_coverage_fidelity(attention, None, **kwargs)
        self.assertEqual(
            [item.retained_tokens for item in legacy.allocations],
            [item.retained_tokens for item in omitted.allocations],
        )
        self.assertTrue(
            all(item.accessibility_rank == 0.0 for item in omitted.allocations)
        )

    def test_enabled_accessibility_rejects_none(self):
        attention, _ = _signals()
        with self.assertRaisesRegex(CoverageFidelityError, "is required"):
            allocate_coverage_fidelity(
                attention,
                None,
                _segments(),
                middle_kv_fraction=0.4,
                sparse_width=2,
            )

    def test_ties_are_broken_by_segment_id(self):
        plan = allocate_coverage_fidelity(
            {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0},
            {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0},
            _segments(),
            middle_kv_fraction=0.10,
            sparse_width=2,
            enable_exact_upgrades=False,
        )
        by_id = {item.segment_id: item for item in plan.allocations}
        self.assertEqual(by_id[1].retained_tokens, 2)
        self.assertEqual(by_id[2].retained_tokens, 2)
        self.assertEqual(by_id[3].retained_tokens, 0)

    def test_residual_is_filled_to_less_than_one_token_cost(self):
        attention, accessibility = _signals()
        plan = allocate_coverage_fidelity(
            attention,
            accessibility,
            _segments(kv_bytes=120),
            middle_kv_fraction=0.43,
            sparse_width=2,
        )
        self.assertLess(plan.residual_middle_bytes, 12)
        self.assertEqual(plan.middle_charged_bytes, 204)

    def test_sparse_only_residual_does_not_create_greedy_exact_upgrade(self):
        attention, accessibility = _signals()
        plan = allocate_coverage_fidelity(
            attention,
            accessibility,
            _segments(),
            middle_kv_fraction=0.8,
            sparse_width=2,
            enable_exact_upgrades=False,
        )
        middle = [item for item in plan.allocations if item.segment_id in range(1, 5)]
        self.assertTrue(all(item.action == "sparse" for item in middle))
        self.assertLessEqual(
            max(item.retained_tokens for item in middle)
            - min(item.retained_tokens for item in middle),
            1,
        )

    def test_invalid_costs_and_signals_fail_closed(self):
        attention, accessibility = _signals()
        with self.assertRaisesRegex(CoverageFidelityError, "exactly match"):
            allocate_coverage_fidelity(
                {1: 1.0},
                accessibility,
                _segments(),
                middle_kv_fraction=0.1,
                sparse_width=2,
            )
        bad = list(_segments())
        bad[2] = SegmentSpec(**{**bad[2].__dict__, "kv_bytes": 101})
        with self.assertRaisesRegex(CoverageFidelityError, "token-divisible"):
            allocate_coverage_fidelity(
                attention,
                accessibility,
                bad,
                middle_kv_fraction=0.1,
                sparse_width=2,
            )


if __name__ == "__main__":
    unittest.main()
