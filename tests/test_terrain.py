"""Categories of terrain — Rulebook p. 269-270.

The rules a piece presents are separate from what it looks like: a wood may be
difficult, dangerous or impassable depending on its size and density, so a map
tags each piece with the going it presents.
"""

import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from panda3d.core import BitMask32, Point3  # noqa: E402

import terrain_system as ts  # noqa: E402
from battleFunctions import firing_rank_count  # noqa: E402
from collision_masks import CollisionMask as CM  # noqa: E402
from models import model  # noqa: E402
from movement_system import MovementSystem  # noqa: E402
from psychology import rank_bonus  # noqa: E402
from special_rules import charge_roll  # noqa: E402
from terrain_system import sees_over  # noqa: E402


class TestCategories(unittest.TestCase):

    def test_every_type_names_a_real_category(self):
        for name, rules in ts.TERRAIN_RULES.items():
            self.assertIn(rules['going'], ts.TERRAIN_CATEGORIES, name)

    def test_open_ground_does_nothing(self):
        cat = ts.TERRAIN_CATEGORIES['open']
        self.assertEqual(cat['movement_modifier'], 0)
        self.assertFalse(cat['dangerous'])
        self.assertFalse(cat['impassable'])
        self.assertFalse(cat['disrupts'])

    def test_difficult_terrain_costs_one_movement(self):
        self.assertEqual(ts.TERRAIN_CATEGORIES['difficult']['movement_modifier'], -1)
        self.assertTrue(ts.TERRAIN_CATEGORIES['difficult']['disrupts'])

    def test_dangerous_terrain_hinders_movement_like_difficult(self):
        difficult = ts.TERRAIN_CATEGORIES['difficult']
        dangerous = ts.TERRAIN_CATEGORIES['dangerous']
        self.assertEqual(dangerous['movement_modifier'],
                         difficult['movement_modifier'])
        self.assertEqual(dangerous['disrupts'], difficult['disrupts'])
        self.assertTrue(dangerous['dangerous'])

    def test_a_hill_is_open_ground(self):
        # Rulebook: "Hills in general are treated as open ground" — their
        # advantage is elevation, not the going.
        self.assertEqual(ts.TERRAIN_RULES['hill']['going'], 'open')

    def test_hills_and_woods_still_block_line_of_sight(self):
        self.assertTrue(ts.TERRAIN_RULES['hill']['blocks_line_of_sight'])
        self.assertTrue(ts.TERRAIN_RULES['forest']['blocks_line_of_sight'])
        self.assertFalse(ts.TERRAIN_RULES['river']['blocks_line_of_sight'])


class TestPieceCategory(unittest.TestCase):
    """TerrainPiece builds Panda3D geometry, so exercise the resolution logic
    the way __init__ does without instantiating one."""

    @staticmethod
    def _resolve(terrain_type, going=None):
        rules = ts.TERRAIN_RULES[terrain_type]
        resolved = going or rules['going']
        return ts.TERRAIN_CATEGORIES[resolved]

    def test_a_type_carries_a_default(self):
        self.assertEqual(self._resolve('forest')['movement_modifier'], -1)
        self.assertEqual(self._resolve('hill')['movement_modifier'], 0)

    def test_a_map_can_override_it(self):
        # The same wood, dense enough to be dangerous on this battlefield.
        self.assertTrue(self._resolve('forest', 'dangerous')['dangerous'])
        self.assertTrue(self._resolve('forest', 'impassable')['impassable'])

    def test_an_unknown_category_is_rejected(self):
        self.assertNotIn('swampy', ts.TERRAIN_CATEGORIES)


class TestMapFile(unittest.TestCase):

    MAP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'maps', 'sample_terrain.json')

    def test_the_sample_map_parses(self):
        with open(self.MAP) as f:
            data = json.load(f)
        for entry in data['terrain']:
            self.assertIn(entry['type'], ts.TERRAIN_RULES)
            if 'going' in entry:
                self.assertIn(entry['going'], ts.TERRAIN_CATEGORIES)

    def test_going_is_optional(self):
        # Maps written before the tag still load, falling back to the type.
        entry = {"type": "forest", "center": [0, 0, 0], "width": 10, "height": 10}
        going = entry.get('going') or ts.TERRAIN_RULES[entry['type']]['going']
        self.assertEqual(going, 'difficult')


class TestDangerousTerrainTests(unittest.TestCase):
    """Any model that begins in, passes through or ends in dangerous terrain
    tests; on a 1 it loses a Wound, once per separate feature."""

    def test_nothing_to_test(self):
        self.assertEqual(ts.dangerous_terrain_wounds(0, 20), 0)
        self.assertEqual(ts.dangerous_terrain_wounds(2, 0), 0)

    def test_every_model_tests(self):
        with mock.patch('terrain_system.random.randint', return_value=1):
            self.assertEqual(ts.dangerous_terrain_wounds(1, 20), 20)

    def test_a_test_per_feature(self):
        with mock.patch('terrain_system.random.randint', return_value=1):
            self.assertEqual(ts.dangerous_terrain_wounds(3, 10), 30)

    def test_a_two_is_safe(self):
        with mock.patch('terrain_system.random.randint', return_value=2):
            self.assertEqual(ts.dangerous_terrain_wounds(5, 20), 0)

    def test_the_damage_can_be_more_than_one_wound(self):
        # Iron Shod Wheels costs a chariot D3 Wounds instead of 1. Every D6
        # mishaps; every D3 rolls its maximum.
        def roll(low, high):
            return 1 if high == 6 else 3
        with mock.patch('terrain_system.random.randint', side_effect=roll):
            self.assertEqual(ts.dangerous_terrain_wounds(1, 2, 'D3'), 6)


class TestDisrupted(unittest.TestCase):
    """A unit with a quarter or more of its models in difficult terrain claims
    no Rank Bonus."""

    def test_a_quarter_is_enough(self):
        self.assertTrue(ts.is_disrupted(5, 20))
        self.assertFalse(ts.is_disrupted(4, 20))

    def test_it_rounds_against_the_unit(self):
        # 3 of 10 is 30%, over the quarter; 2 of 10 is 20% and is not.
        self.assertTrue(ts.is_disrupted(3, 10))
        self.assertFalse(ts.is_disrupted(2, 10))

    def test_an_empty_unit(self):
        self.assertFalse(ts.is_disrupted(0, 0))

    def test_all_of_them(self):
        self.assertTrue(ts.is_disrupted(20, 20))

    def test_it_costs_the_rank_bonus(self):
        unit = SimpleNamespace(model=model("Zombie", ""), nmodels=20,
                               files=5, ranks=4)
        self.assertEqual(rank_bonus(unit), 2)
        self.assertEqual(rank_bonus(unit, disrupted=True), 0)


class TestImpassableTerrain(unittest.TestCase):
    """Impassable terrain cannot be crossed; units must go around it."""

    def test_a_house_is_impassable(self):
        self.assertEqual(ts.TERRAIN_RULES['house']['going'], 'impassable')
        cat = ts.TERRAIN_CATEGORIES['impassable']
        self.assertTrue(cat['impassable'])
        # It stops movement outright rather than slowing it.
        self.assertEqual(cat['movement_modifier'], 0)
        self.assertFalse(cat['dangerous'])
        self.assertFalse(cat['disrupts'])

    def test_a_building_blocks_line_of_sight(self):
        self.assertTrue(ts.TERRAIN_RULES['house']['blocks_line_of_sight'])

    def test_it_has_its_own_collision_bit(self):
        self.assertEqual(ts._TERRAIN_COLLISION_MASK['house'],
                         CM.TERRAIN_IMPASSABLE)

    def test_movement_sweeps_hit_it(self):
        # The sweep used to test bit 9 only, so terrain never stopped anything.
        self.assertTrue((CM.MOVE_BLOCKERS & CM.TERRAIN_IMPASSABLE)
                        != BitMask32.allOff())
        self.assertTrue((CM.MOVE_BLOCKERS & CM.SWEEP_TARGET)
                        != BitMask32.allOff())

    def test_passable_terrain_does_not_block(self):
        for kind in ('forest', 'hill', 'river', 'marsh'):
            self.assertTrue((CM.MOVE_BLOCKERS & ts._TERRAIN_COLLISION_MASK[kind])
                            == BitMask32.allOff(), kind)


class TestVantagePoint(unittest.TestCase):
    """A unit entirely on a hill fires with one additional rank (p. 271)."""

    class _Movement:
        """MovementSystem.entirelyOnHill with the node walk stubbed out."""
        def __init__(self, inside, total):
            self._counts = (inside, total)

        def modelsInTerrain(self, unit, predicate):
            return self._counts

        entirelyOnHill = MovementSystem.entirelyOnHill

    def test_a_unit_wholly_on_the_hill_claims_it(self):
        self.assertTrue(self._Movement(20, 20).entirelyOnHill(None))

    def test_a_unit_partly_on_the_hill_claims_nothing(self):
        # Official FAQ 1.5.3: "To claim the benefits of being on a hill, a unit
        # must be entirely on the hill." Hills are organic shapes, so a unit on
        # the rim usually has a model hanging off.
        self.assertFalse(self._Movement(19, 20).entirelyOnHill(None))

    def test_a_unit_off_the_hill_claims_nothing(self):
        self.assertFalse(self._Movement(0, 20).entirelyOnHill(None))

    def test_a_unit_with_no_models(self):
        self.assertFalse(self._Movement(0, 0).entirelyOnHill(None))

    def test_only_the_front_rank_shoots_on_flat_ground(self):
        self.assertEqual(firing_rank_count(5, 20), 5)

    def test_a_hill_adds_a_whole_rank(self):
        self.assertEqual(firing_rank_count(5, 20, extra_ranks=1), 10)

    def test_volley_fire_adds_half_the_next_rank(self):
        self.assertEqual(firing_rank_count(5, 20, volley_fire=True), 8)

    def test_they_stack(self):
        # Front rank + the full second (Vantage Point) + half of the third.
        self.assertEqual(
            firing_rank_count(5, 20, extra_ranks=1, volley_fire=True), 13)

    def test_a_unit_cannot_field_more_shooters_than_it_has(self):
        self.assertEqual(firing_rank_count(5, 7, extra_ranks=1), 7)
        self.assertEqual(
            firing_rank_count(5, 7, extra_ranks=1, volley_fire=True), 7)
        self.assertEqual(firing_rank_count(5, 3, extra_ranks=1), 3)

    def test_nothing_to_shoot_with(self):
        self.assertEqual(firing_rank_count(5, 0), 0)
        self.assertEqual(firing_rank_count(0, 5), 0)


class TestSeeingOverUnitsOnTheSameHill(unittest.TestCase):
    """A unit closer to a hill's top sees over one closer to its bottom, and
    never the other way round (Official FAQ 1.5.3). The top is the centre."""

    TOP = (0.0, 0.0)

    def test_the_higher_unit_sees_over_the_lower(self):
        self.assertTrue(sees_over((2.0, 0.0), (6.0, 0.0), self.TOP))

    def test_the_lower_unit_does_not_see_over_the_higher(self):
        self.assertFalse(sees_over((6.0, 0.0), (2.0, 0.0), self.TOP))

    def test_units_at_the_same_height_block_each_other(self):
        # Neither is closer to the bottom, so no exception applies.
        self.assertFalse(sees_over((3.0, 0.0), (0.0, 3.0), self.TOP))
        self.assertFalse(sees_over((3.0, 0.0), (3.0, 0.0), self.TOP))

    def test_the_top_is_measured_from_the_hills_centre(self):
        top = (10.0, -4.0)
        self.assertTrue(sees_over((11.0, -4.0), (16.0, -4.0), top))
        self.assertFalse(sees_over((16.0, -4.0), (11.0, -4.0), top))

    def test_it_takes_any_indexable_position(self):
        self.assertTrue(sees_over(Point3(2, 0, 9), Point3(6, 0, 3), Point3(0, 0, 0)))


class TestChargingThroughDifficultTerrain(unittest.TestCase):
    """The Charge roll discards the highest result rather than the lowest."""

    def test_normally_the_lowest_is_discarded(self):
        self.assertEqual(charge_roll([2, 5]), 5)

    def test_through_difficult_terrain_the_highest_is(self):
        self.assertEqual(charge_roll([2, 5], difficult=True), 2)

    def test_the_swiftstride_die_is_still_added(self):
        # The bonus die was never one of the two being discarded between.
        self.assertEqual(charge_roll([2, 5, 6], difficult=True), 8)
        self.assertEqual(charge_roll([2, 5, 6]), 11)

    def test_a_double_is_unaffected(self):
        self.assertEqual(charge_roll([4, 4], difficult=True), 4)

    def test_no_dice_no_charge(self):
        self.assertEqual(charge_roll([], difficult=True), 0)


if __name__ == "__main__":
    unittest.main()
