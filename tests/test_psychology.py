"""Tests for the psychology system's pure logic (Phase 0: Panic)."""

import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from psychology import (  # noqa: E402
    leadership_test, panic_fail_outcome, unit_strength_total, heavy_casualties,
    PANIC_US_THRESHOLD, PANIC_RADIUS, PsychologySystem, obb_distance,
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


class HeavyCasualtiesTests(unittest.TestCase):
    def test_more_than_quarter_lost(self):
        # 20 -> 14 is 6 lost (>25%).
        self.assertTrue(heavy_casualties(14, 20))

    def test_exactly_quarter_lost(self):
        # 20 -> 15 is 5 lost (exactly 25%, not more).
        self.assertFalse(heavy_casualties(15, 20))

    def test_no_losses(self):
        self.assertFalse(heavy_casualties(20, 20))

    def test_zero_start(self):
        self.assertFalse(heavy_casualties(0, 0))

    def test_small_unit_one_loss(self):
        # 3 -> 2 is 1 lost of 3 (>25%).
        self.assertTrue(heavy_casualties(2, 3))


class PanicConstantsTests(unittest.TestCase):
    def test_thresholds(self):
        self.assertEqual(PANIC_US_THRESHOLD, 5)
        self.assertEqual(PANIC_RADIUS, 6.0)


class ExemptionTests(unittest.TestCase):
    """No Need for Hysterics — a unit need not test when exempt."""

    @staticmethod
    def _unit(rules=None, state='Idle', tested=False, charging=False, combat=False):
        model = SimpleNamespace(special_rules=rules or [])
        return SimpleNamespace(
            state=state, unit=SimpleNamespace(model=model, name='U'),
            panicTestedThisPhase=tested, isChargingMove=charging, isInCombat=combat)

    def setUp(self):
        self.psy = PsychologySystem(None)

    def test_not_exempt(self):
        self.assertIsNone(self.psy.panic_exempt_reason(self._unit()))

    def test_already_tested(self):
        self.assertEqual(self.psy.panic_exempt_reason(self._unit(tested=True)),
                         "already tested this phase")

    def test_charging(self):
        self.assertEqual(self.psy.panic_exempt_reason(self._unit(charging=True)),
                         "making a charge move")

    def test_in_combat(self):
        self.assertEqual(self.psy.panic_exempt_reason(self._unit(combat=True)),
                         "engaged in combat")

    def test_fleeing(self):
        self.assertEqual(self.psy.panic_exempt_reason(self._unit(state='IsFleeing')),
                         "already fleeing")

    def test_unbreakable(self):
        u = self._unit(rules=[{'name': 'Unbreakable', 'Unbreakable': True}])
        self.assertEqual(self.psy.panic_exempt_reason(u), "Unbreakable")

    def test_ignore_panic(self):
        u = self._unit(rules=[{'name': 'Ignore Panic'}])
        self.assertEqual(self.psy.panic_exempt_reason(u), 'Ignore Panic')

    def test_immune_to_psychology(self):
        u = self._unit(rules=[{'name': 'Immune to Psychology'}])
        self.assertEqual(self.psy.panic_exempt_reason(u), 'Immune to Psychology')


class ObbDistanceTests(unittest.TestCase):
    def test_overlapping_boxes_zero(self):
        self.assertEqual(obb_distance((0, 0, 1, 1, 0), (0.5, 0, 1, 1, 0)), 0.0)

    def test_axis_aligned_gap(self):
        # Two 2x2 boxes (half 1) centred 5 apart on x -> 5 - 1 - 1 = 3 gap.
        self.assertAlmostEqual(
            obb_distance((0, 0, 1, 1, 0), (5, 0, 1, 1, 0)), 3.0, places=6)

    def test_symmetry(self):
        a, b = (0, 0, 1, 2, 30), (7, 1, 2, 1, -15)
        self.assertAlmostEqual(obb_distance(a, b), obb_distance(b, a), places=6)

    def test_quicksave_troopers_edge_to_edge(self):
        # From quicksave.json: centres 6.5" apart, footprints ~4.9x2.0 and
        # ~4.9x3.9 -> edge to edge well within 6" (missed by centre-to-centre).
        missile = (-0.4, 0.5, 4.9 / 2, 2.0 / 2, 180)
        troopers = (-6.3, -2.2, 4.9 / 2, 3.9 / 2, 0)
        self.assertLess(obb_distance(missile, troopers), 6.0)


if __name__ == "__main__":
    unittest.main()
