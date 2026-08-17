"""Tests for the psychology system's pure logic (Phase 0: Panic)."""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psychology  # noqa: E402

from psychology import (  # noqa: E402
    leadership_test, leadership_test_with_reroll, panic_fail_outcome,
    unit_strength_total, heavy_casualties, PANIC_US_THRESHOLD, PANIC_RADIUS,
    VENERABLE_RADIUS, PsychologySystem, obb_distance, is_skirmish_unit,
    fled_through_panics, is_venerable_unit,
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
        self.assertEqual(VENERABLE_RADIUS, PANIC_RADIUS)


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


class SkirmisherPanicTests(unittest.TestCase):
    """Skirmishers & Panic (Rulebook p. 185) — the fled-through guard."""

    @staticmethod
    def _unit(name, skirmisher=False, empty=False):
        model = SimpleNamespace(is_skirmisher=lambda: skirmisher,
                                special_rules=[])
        return SimpleNamespace(
            unitName=name, isSkirmisher=skirmisher,
            bodyNP=SimpleNamespace(isEmpty=lambda: empty),
            unit=SimpleNamespace(name=name, model=model))

    def _psy(self, friends, enemies):
        game = SimpleNamespace(player1Units=friends, player2Units=enemies)
        return PsychologySystem(game)

    def test_is_skirmish_unit(self):
        self.assertTrue(is_skirmish_unit(self._unit('Scouts', skirmisher=True)))
        self.assertFalse(is_skirmish_unit(self._unit('Spears')))
        self.assertFalse(is_skirmish_unit(None))

    def test_is_skirmish_unit_from_model_only(self):
        u = self._unit('Scouts', skirmisher=True)
        del u.isSkirmisher                       # only the model knows
        self.assertTrue(is_skirmish_unit(u))

    def test_predicate(self):
        self.assertFalse(fled_through_panics(True, False))   # skirmisher -> formed
        self.assertTrue(fled_through_panics(True, True))     # skirmisher -> skirmisher
        self.assertTrue(fled_through_panics(False, False))   # formed -> formed
        self.assertTrue(fled_through_panics(False, True))    # formed -> skirmisher

    def test_skirmishers_do_not_panic_formed_friends(self):
        fleer = self._unit('Scouts', skirmisher=True)
        formed = self._unit('Spearmen')
        psy = self._psy([fleer, formed], [])
        psy._after_unit_done(fleer, [formed], lambda: None)
        self.assertEqual(psy._panic_queue, [])

    def test_skirmishers_still_panic_skirmisher_friends(self):
        fleer = self._unit('Scouts', skirmisher=True)
        other = self._unit('Archers', skirmisher=True)
        psy = self._psy([fleer, other], [])
        psy._after_unit_done(fleer, [other], lambda: None)
        self.assertEqual([q[0] for q in psy._panic_queue], [other])

    def test_formed_unit_still_panics_formed_friends(self):
        fleer = self._unit('Knights')
        formed = self._unit('Spearmen')
        psy = self._psy([fleer, formed], [])
        psy._after_unit_done(fleer, [formed], lambda: None)
        self.assertEqual([q[0] for q in psy._panic_queue], [formed])

    def test_enemies_fled_through_do_not_test(self):
        fleer = self._unit('Knights')
        enemy = self._unit('Orcs')
        psy = self._psy([fleer], [enemy])
        psy._after_unit_done(fleer, [enemy], lambda: None)
        self.assertEqual(psy._panic_queue, [])

    def test_on_done_always_called(self):
        fleer = self._unit('Scouts', skirmisher=True)
        formed = self._unit('Spearmen')
        psy = self._psy([fleer, formed], [])
        calls = []
        psy._after_unit_done(fleer, [formed], lambda: calls.append(1))
        self.assertEqual(calls, [1])


class VenerableTests(unittest.TestCase):
    """Venerable — friendly units within 6" re-roll failed Panic tests."""

    @staticmethod
    def _unit(name, venerable=False, x=0.0, y=0.0, state='Idle', ld=7,
              nmodels=10, empty=False):
        model = SimpleNamespace(is_venerable=lambda: venerable,
                                is_skirmisher=lambda: False,
                                special_rules=[],
                                characteristics={'Ld': str(ld)})
        return SimpleNamespace(
            state=state, unitWidth=2.0, unitHeight=2.0,
            panicTestedThisPhase=False, isChargingMove=False, isInCombat=False,
            startOfBattleModels=nmodels,
            bodyNP=SimpleNamespace(isEmpty=lambda: empty,
                                   getPos=lambda: SimpleNamespace(x=x, y=y),
                                   getH=lambda: 0.0),
            unit=SimpleNamespace(name=name, model=model, nmodels=nmodels))

    @staticmethod
    def _psy(friends, enemies=()):
        game = SimpleNamespace(player1Units=list(friends),
                               player2Units=list(enemies))
        return PsychologySystem(game)

    # ─── rule detection ───────────────────────────────────────────────────

    def test_is_venerable_unit(self):
        self.assertTrue(is_venerable_unit(self._unit('Anvil', venerable=True)))
        self.assertFalse(is_venerable_unit(self._unit('Warriors')))
        self.assertFalse(is_venerable_unit(None))

    def test_is_venerable_unit_from_flag_only(self):
        u = self._unit('Anvil')
        u.isVenerable = True
        self.assertTrue(is_venerable_unit(u))

    def test_builder_registered(self):
        from special_rules import SPECIAL_RULE_BUILDERS
        self.assertIn('venerable', SPECIAL_RULE_BUILDERS)
        rule = SPECIAL_RULE_BUILDERS['venerable'](None, None, None)
        self.assertTrue(rule['venerable'])

    # ─── the 6" bubble (edge to edge, same as nearby-friend Panic) ────────

    def test_friend_within_six_inches(self):
        anvil = self._unit('Anvil', venerable=True, x=7.0)
        warriors = self._unit('Warriors')
        # 2x2 footprints centred 7" apart -> 5" edge to edge.
        psy = self._psy([anvil, warriors])
        self.assertIs(psy.venerable_source(warriors), anvil)

    def test_friend_beyond_six_inches(self):
        anvil = self._unit('Anvil', venerable=True, x=10.0)
        warriors = self._unit('Warriors')
        # centres 10" apart -> 8" edge to edge.
        psy = self._psy([anvil, warriors])
        self.assertIsNone(psy.venerable_source(warriors))

    def test_venerable_unit_benefits_itself(self):
        anvil = self._unit('Anvil', venerable=True)
        psy = self._psy([anvil])
        self.assertIs(psy.venerable_source(anvil), anvil)

    def test_enemy_venerable_does_not_help(self):
        anvil = self._unit('Anvil', venerable=True, x=2.0)
        warriors = self._unit('Warriors')
        psy = self._psy([warriors], [anvil])
        self.assertIsNone(psy.venerable_source(warriors))

    def test_fleeing_venerable_does_not_help(self):
        anvil = self._unit('Anvil', venerable=True, x=2.0, state='IsFleeing')
        warriors = self._unit('Warriors')
        psy = self._psy([anvil, warriors])
        self.assertIsNone(psy.venerable_source(warriors))

    def test_no_source_without_the_rule(self):
        friend = self._unit('Miners', x=2.0)
        warriors = self._unit('Warriors')
        psy = self._psy([friend, warriors])
        self.assertIsNone(psy.venerable_source(warriors))

    def test_removed_unit_has_no_source(self):
        anvil = self._unit('Anvil', venerable=True, x=2.0)
        warriors = self._unit('Warriors', empty=True)
        psy = self._psy([anvil, warriors])
        self.assertIsNone(psy.venerable_source(warriors))

    # ─── the re-roll itself ───────────────────────────────────────────────

    def test_reroll_only_on_failure(self):
        passed, rolls = leadership_test_with_reroll(12, reroll=True)
        self.assertTrue(passed)
        self.assertEqual(len(rolls), 1)

    def test_reroll_taken_when_failed_and_allowed(self):
        passed, rolls = leadership_test_with_reroll(1, reroll=True)
        self.assertFalse(passed)          # 2D6 minimum is 2 — both rolls fail
        self.assertEqual(len(rolls), 2)

    def test_no_reroll_when_not_allowed(self):
        passed, rolls = leadership_test_with_reroll(1, reroll=False)
        self.assertFalse(passed)
        self.assertEqual(len(rolls), 1)

    def test_only_one_reroll(self):
        for _ in range(50):
            _passed, rolls = leadership_test_with_reroll(7, reroll=True)
            self.assertLessEqual(len(rolls), 2)

    def test_reroll_uses_the_modifier(self):
        for _ in range(100):
            passed, rolls = leadership_test_with_reroll(5, modifier=-2, reroll=True)
            self.assertEqual(passed, rolls[-1] <= 3)

    # ─── the re-roll applied to a Panic test ──────────────────────────────

    def _resolve(self, unit, psy, dice):
        """Run a Panic test with scripted D6 results; returns (fled, done)."""
        fled, done = [], []
        psy._start_flee_move = lambda u, d, dist, outcome, cb: fled.append(u)
        with mock.patch.object(psychology.random, 'randint',
                               side_effect=list(dice)):
            psy._resolve_panic(unit, None, 'test', lambda: done.append(1))
        return fled, done

    def test_failed_panic_is_rerolled_near_venerable(self):
        anvil = self._unit('Anvil', venerable=True, x=2.0)
        warriors = self._unit('Warriors')
        psy = self._psy([anvil, warriors])
        # Ld 7: first test 6+6=12 fails, the Venerable re-roll 1+1=2 passes.
        fled, done = self._resolve(warriors, psy, [6, 6, 1, 1])
        self.assertEqual(fled, [])
        self.assertEqual(done, [1])

    def test_failed_reroll_still_flees(self):
        anvil = self._unit('Anvil', venerable=True, x=2.0)
        warriors = self._unit('Warriors')
        psy = self._psy([anvil, warriors])
        # Both tests fail (12, 12); the last two dice are the flee distance.
        fled, _done = self._resolve(warriors, psy, [6, 6, 6, 6, 3, 4])
        self.assertEqual(fled, [warriors])

    def test_no_reroll_without_venerable_friend(self):
        far = self._unit('Anvil', venerable=True, x=20.0)
        warriors = self._unit('Warriors')
        psy = self._psy([far, warriors])
        # Only one test is rolled (12 fails), then the flee distance.
        fled, _done = self._resolve(warriors, psy, [6, 6, 3, 4])
        self.assertEqual(fled, [warriors])

    def test_passed_panic_needs_no_reroll(self):
        anvil = self._unit('Anvil', venerable=True, x=2.0)
        warriors = self._unit('Warriors')
        psy = self._psy([anvil, warriors])
        fled, done = self._resolve(warriors, psy, [1, 1])
        self.assertEqual(fled, [])
        self.assertEqual(done, [1])


if __name__ == "__main__":
    unittest.main()
