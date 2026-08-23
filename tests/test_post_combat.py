"""Post-combat decisions (Rulebook p. 154-157)."""

import math
import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from post_combat import (  # noqa: E402
    GIVE_GROUND, catch_outcome, fall_back_roll, flee_direction, flee_roll,
    flees_from, give_ground_direction, may_pursue, pursuit_roll,
    restraint_test, winner_response,
)


class TestDistances(unittest.TestCase):

    def test_a_flee_roll_sums_its_dice(self):
        self.assertEqual(flee_roll([2, 5]), 7)

    def test_a_flee_roll_adds_the_swiftstride_die(self):
        self.assertEqual(flee_roll([2, 5, 6]), 13)

    def test_falling_back_discards_the_lowest(self):
        self.assertEqual(fall_back_roll([2, 5]), 5)

    def test_falling_back_adds_the_swiftstride_die_to_the_one_kept(self):
        self.assertEqual(fall_back_roll([2, 5, 4]), 9)

    def test_a_pursuit_roll_sums_its_dice(self):
        """No Movement is added and nothing is discarded, so a pursuit reaches
        further than the charge roll it is resolved through."""
        self.assertEqual(pursuit_roll([2, 5]), 7)

    def test_give_ground_is_two_inches(self):
        self.assertEqual(GIVE_GROUND, 2.0)


class TestTheLimitsOfEndurance(unittest.TestCase):
    """One flee move per phase; a second covers 0" and does not pivot
    (p. 133). A Fall Back moves exactly like a fleeing unit, so it counts."""

    def test_a_first_flee_rolls_normally(self):
        self.assertEqual(flee_roll([4, 5], already_fled=False), 9)

    def test_a_second_flee_covers_nothing(self):
        self.assertEqual(flee_roll([4, 5], already_fled=True), 0)

    def test_a_second_fall_back_covers_nothing(self):
        self.assertEqual(fall_back_roll([4, 5], already_fled=True), 0)

    def test_even_a_swiftstride_die_cannot_revive_it(self):
        self.assertEqual(flee_roll([6, 6, 6], already_fled=True), 0)


class TestWhoTheLoserFleesFrom(unittest.TestCase):
    """The Greater the Danger (p. 133): the highest Unit Strength, not the
    average of the enemies."""

    def test_the_only_enemy(self):
        self.assertEqual(flees_from([('a', 5)]), 'a')

    def test_the_strongest_enemy(self):
        self.assertEqual(flees_from([('a', 5), ('b', 12), ('c', 3)]), 'b')

    def test_equals_are_settled_at_random(self):
        class FixedChoice:
            def choice(self, seq):
                return seq[-1]
        self.assertEqual(
            flees_from([('a', 7), ('b', 7)], rng=FixedChoice()), 'b')

    def test_no_enemies_at_all(self):
        self.assertIsNone(flees_from([]))


class TestDirection(unittest.TestCase):

    def test_a_flee_runs_straight_away_from_one_enemy(self):
        self.assertEqual(flee_direction((0, 0), (0, -4)), (0.0, 1.0))

    def test_giving_ground_backs_away_from_two_enemies_diagonally(self):
        """Held in the front and a flank, the unit moves away from both."""
        dx, dy = give_ground_direction((0, 0), [(0, -1), (-1, 0)])
        self.assertAlmostEqual(dx, math.sqrt(0.5))
        self.assertAlmostEqual(dy, math.sqrt(0.5))

    def test_giving_ground_from_one_enemy_matches_a_flee(self):
        self.assertEqual(give_ground_direction((0, 0), [(0, -4)]),
                         flee_direction((0, 0), (0, -4)))

    def test_enemies_on_opposite_sides_cancel(self):
        """Surrounded: there is no direction that is away from both."""
        self.assertEqual(give_ground_direction((0, 0), [(0, -1), (0, 1)]),
                         (0.0, 0.0))

    def test_stacked_positions_do_not_divide_by_zero(self):
        self.assertEqual(flee_direction((3, 3), (3, 3)), (0.0, 0.0))


class TestTheWinnersChoice(unittest.TestCase):

    def test_giving_ground_is_answered_by_a_follow_up(self):
        self.assertEqual(winner_response('give_ground'), 'follow_up')

    def test_falling_back_and_breaking_are_answered_by_a_pursuit(self):
        self.assertEqual(winner_response('fall_back'), 'pursue')
        self.assertEqual(winner_response('break'), 'pursue')

    def test_restraint_is_a_leadership_test(self):
        self.assertTrue(restraint_test(8, [3, 4]))
        self.assertTrue(restraint_test(8, [4, 4]))   # equal to Ld passes
        self.assertFalse(restraint_test(8, [4, 5]))

    def test_a_unit_still_in_base_contact_cannot_pursue(self):
        self.assertFalse(may_pursue(still_in_base_contact=True))
        self.assertTrue(may_pursue(still_in_base_contact=False))


class TestCatching(unittest.TestCase):

    def test_a_fleeing_unit_is_run_down(self):
        self.assertEqual(catch_outcome('break'), 'destroyed')

    def test_a_unit_that_fell_back_is_only_re_engaged(self):
        self.assertEqual(catch_outcome('fall_back'), 'engaged')


class TestFleesFromOnTheBoard(unittest.TestCase):
    """`CombatResolver.fleesFrom` bridges the board to `flees_from`, and takes
    the unit wrapper rather than the profile inside it -- passing the wrong one
    raised at the table, not in the tests."""

    def _wrapper(self, name, unit_strength, nmodels):
        model = SimpleNamespace(unit_strength=lambda: unit_strength)
        return SimpleNamespace(
            unitName=name,
            unit=SimpleNamespace(name=name, model=model, nmodels=nmodels),
            bodyNP=SimpleNamespace(isEmpty=lambda: False),
            isInCombatWith=[])

    def _resolver(self):
        from combat_resolution import CombatResolver
        return CombatResolver.__new__(CombatResolver)

    def test_it_runs_from_the_strongest_of_its_enemies(self):
        loser = self._wrapper("Jade Warriors", 1, 9)
        weak = self._wrapper("Dire Wolves", 1, 5)
        strong = self._wrapper("Longbeards", 1, 20)
        loser.isInCombatWith = [weak, strong]
        self.assertIs(self._resolver().fleesFrom(loser), strong)

    def test_unit_strength_counts_models_not_units(self):
        """A handful of monstrous infantry can outweigh a horde of skeletons."""
        loser = self._wrapper("Jade Warriors", 1, 9)
        many = self._wrapper("Skeletons", 1, 10)
        heavy = self._wrapper("Trolls", 3, 4)
        loser.isInCombatWith = [many, heavy]
        self.assertIs(self._resolver().fleesFrom(loser), heavy)

    def test_a_destroyed_enemy_is_not_fled_from(self):
        loser = self._wrapper("Jade Warriors", 1, 9)
        gone = self._wrapper("Longbeards", 1, 20)
        gone.bodyNP = SimpleNamespace(isEmpty=lambda: True)
        alive = self._wrapper("Dire Wolves", 1, 5)
        loser.isInCombatWith = [gone, alive]
        self.assertIs(self._resolver().fleesFrom(loser), alive)


if __name__ == "__main__":
    unittest.main()
