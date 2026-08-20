"""Troop type properties — Rulebook, Troop Types in Detail (p. 194-195)."""

import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import troop_types  # noqa: E402
from battlescribe import get_catalogue  # noqa: E402
from models import model  # noqa: E402
from psychology import MAX_RANK_BONUS, rank_bonus  # noqa: E402


class TestNormalising(unittest.TestCase):
    """The catalogue spells troop types inconsistently."""

    def test_case_is_ignored(self):
        for raw in ("Heavy Chariot", "Heavy chariot", "heavy chariot"):
            self.assertEqual(troop_types.normalise(raw), "heavy chariot")

    def test_a_character_suffix_is_dropped(self):
        self.assertEqual(troop_types.normalise("Heavy chariot (named character)"),
                         "heavy chariot")

    def test_a_compound_type_takes_the_first(self):
        raw = "Regular infantry (skink handlers), war beasts (salamanders)"
        self.assertEqual(troop_types.normalise(raw), "regular infantry")

    def test_nothing_is_empty(self):
        self.assertEqual(troop_types.normalise(None), "")


class TestTheTable(unittest.TestCase):

    def test_heavy_chariots(self):
        self.assertEqual(troop_types.unit_strength("Heavy Chariot", 1), 5)
        self.assertEqual(troop_types.max_rank_bonus("Heavy Chariot", 2), 0)

    def test_light_chariots(self):
        self.assertEqual(troop_types.unit_strength("Light chariot", 1), 3)
        self.assertEqual(troop_types.max_rank_bonus("Light chariot", 2), 1)

    def test_an_unknown_type_keeps_the_default(self):
        self.assertEqual(troop_types.unit_strength("Regular infantry", 1), 1)
        self.assertEqual(troop_types.max_rank_bonus("Regular infantry", 2), 2)

    def test_the_rules_a_troop_type_grants(self):
        self.assertTrue(troop_types.has_rule("Heavy Chariot", "Scythed Wheels"))
        self.assertTrue(troop_types.has_rule("Heavy Chariot", "Lumbering"))
        self.assertTrue(troop_types.has_rule("Heavy Chariot", "Firing Platform"))
        self.assertTrue(troop_types.has_rule("Light Chariot", "Firing Platform"))
        # Scythed Wheels and Lumbering are heavy chariots only.
        self.assertFalse(troop_types.has_rule("Light Chariot", "Scythed Wheels"))
        self.assertFalse(troop_types.has_rule("Light Chariot", "Lumbering"))
        self.assertFalse(troop_types.has_rule("Regular infantry", "Firing Platform"))

    def test_split_profile_is_named_with_its_qualifier(self):
        self.assertTrue(troop_types.has_rule("Heavy Chariot", "Split Profile"))


class TestTheModelReportsThem(unittest.TestCase):

    def setUp(self):
        self.wagon = model("War Wagon", "")
        self.zombie = model("Zombie", "")

    def test_unit_strength(self):
        self.assertEqual(self.wagon.unit_strength(), 5)
        self.assertEqual(self.zombie.unit_strength(), 1)

    def test_a_mounted_model_is_still_us2(self):
        rider = model("Orc Boar Boy", "")
        rider.attach_mount(SimpleNamespace(model=model("War Boar", "")))
        self.assertEqual(rider.unit_strength(), 2)

    def test_scythed_wheels_armour_piercing(self):
        self.assertEqual(self.wagon.impact_hit_ap(), 2)
        self.assertEqual(self.zombie.impact_hit_ap(), 0)

    def test_firing_platform_all_round_vision(self):
        self.assertTrue(self.wagon.has_all_round_vision())
        self.assertFalse(self.zombie.has_all_round_vision())

    def test_skirmishers_keep_their_own_360_arc(self):
        skirmisher = model("Zombie", "")
        skirmisher.special_rules.append({'name': 'Skirmishers', 'skirmish': True})
        self.assertTrue(skirmisher.has_all_round_vision())


class TestRankBonus(unittest.TestCase):

    @staticmethod
    def _unit(m, nmodels, files, ranks):
        return SimpleNamespace(model=m, nmodels=nmodels, files=files, ranks=ranks)

    def test_one_per_rank_behind_the_first(self):
        z = model("Zombie", "")
        self.assertEqual(rank_bonus(self._unit(z, 10, 5, 2)), 1)
        self.assertEqual(rank_bonus(self._unit(z, 15, 5, 3)), 2)

    def test_a_single_rank_claims_none(self):
        self.assertEqual(rank_bonus(self._unit(model("Zombie", ""), 5, 5, 1)), 0)

    def test_an_incomplete_rank_does_not_count(self):
        # 12 models at 5 wide is two full ranks and a partial one.
        self.assertEqual(rank_bonus(self._unit(model("Zombie", ""), 12, 5, 3)), 1)

    def test_it_is_capped(self):
        self.assertEqual(rank_bonus(self._unit(model("Zombie", ""), 50, 5, 10)),
                         MAX_RANK_BONUS)

    def test_a_heavy_chariot_claims_none(self):
        wagon = model("War Wagon", "")
        self.assertEqual(rank_bonus(self._unit(wagon, 3, 3, 1)), 0)
        self.assertEqual(rank_bonus(self._unit(wagon, 6, 3, 2)), 0)

    def test_a_light_chariot_claims_at_most_one(self):
        chariot = model("Goblin Wolf Chariot", "")
        self.assertEqual(troop_types.normalise(chariot.troop_type()), "light chariot")
        self.assertEqual(rank_bonus(self._unit(chariot, 9, 3, 3)), 1)
        self.assertEqual(chariot.unit_strength(), 3)
        self.assertEqual(chariot.impact_hit_ap(), 0)   # Scythed Wheels is heavy only
        self.assertTrue(chariot.has_all_round_vision())

    def test_skirmishers_claim_none(self):
        m = model("Zombie", "")
        m.special_rules.append({'name': 'Skirmishers', 'skirmish': True})
        self.assertEqual(rank_bonus(self._unit(m, 20, 5, 4)), 0)


class TestRuleDescriptions(unittest.TestCase):
    """The rulebook's own descriptions live in the .gst, which uses a different
    XML namespace to the .cat files."""

    def test_core_rules_have_their_text(self):
        cat = get_catalogue()
        for name in ("Impact Hits", "Firing Platform", "Lumbering",
                     "Scythed Wheels", "Iron Shod Wheels", "Regeneration"):
            self.assertTrue(cat.rule_description(name), name)

    def test_army_specific_rules_still_load(self):
        self.assertTrue(get_catalogue().rule_description("Choppas"))


if __name__ == "__main__":
    unittest.main()
