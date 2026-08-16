"""Tests for the psychology system's pure logic (Phase 0: Panic)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from psychology import (  # noqa: E402
    leadership_test, panic_fail_outcome, unit_strength_total,
)


class LeadershipTestTests(unittest.TestCase):
    def test_roll_in_range(self):
        for _ in range(200):
            passed, roll = leadership_test(7)
            self.assertTrue(2 <= roll <= 12)
            self.assertEqual(passed, roll <= 7)

    def test_always_passes_on_ld12(self):
        for _ in range(50):
            passed, _roll = leadership_test(12)
            self.assertTrue(passed)

    def test_always_fails_on_ld1(self):
        for _ in range(50):
            passed, _roll = leadership_test(1)  # 2D6 minimum is 2
            self.assertFalse(passed)

    def test_modifier_applied(self):
        # Ld 5 with -2 modifier behaves like Ld 3.
        for _ in range(200):
            passed, roll = leadership_test(5, modifier=-2)
            self.assertEqual(passed, roll <= 3)


class PanicFailOutcomeTests(unittest.TestCase):
    def test_more_than_half_falls_back(self):
        self.assertEqual(panic_fail_outcome(6, 10), 'fall_back')

    def test_exactly_half_flees(self):
        self.assertEqual(panic_fail_outcome(5, 10), 'flee')

    def test_fewer_than_half_flees(self):
        self.assertEqual(panic_fail_outcome(4, 10), 'flee')

    def test_full_strength_falls_back(self):
        self.assertEqual(panic_fail_outcome(10, 10), 'fall_back')

    def test_zero_start_flees(self):
        self.assertEqual(panic_fail_outcome(0, 0), 'flee')


class UnitStrengthTotalTests(unittest.TestCase):
    class _FakeModel:
        def __init__(self, us):
            self._us = us

        def unit_strength(self):
            return self._us

    class _FakeUnit:
        def __init__(self, us, nmodels):
            self.model = UnitStrengthTotalTests._FakeModel(us)
            self.nmodels = nmodels

    class _FakeGraphics:
        def __init__(self, us, nmodels):
            self.unit = UnitStrengthTotalTests._FakeUnit(us, nmodels)

    def test_infantry(self):
        self.assertEqual(unit_strength_total(self._FakeGraphics(1, 20)), 20)

    def test_cavalry(self):
        self.assertEqual(unit_strength_total(self._FakeGraphics(2, 5)), 10)

    def test_no_models(self):
        self.assertEqual(unit_strength_total(self._FakeGraphics(1, 0)), 0)


if __name__ == "__main__":
    unittest.main()
