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

    def test_singular_and_plural_reach_the_same_row(self):
        # The catalogue writes several of these both ways round.
        for a, b in (("War beast", "War beasts"),
                     ("Monstrous creature", "Monstrous Creatures"),
                     ("Behemoth", "Behemoths"),
                     ("War machine", "War Machines"),
                     ("Light chariot", "Light chariots")):
            self.assertEqual(troop_types.normalise(a), troop_types.normalise(b))
            self.assertIsNotNone(troop_types.properties(a), a)

    def test_nothing_is_empty(self):
        self.assertEqual(troop_types.normalise(None), "")


class TestEveryTroopTypeIsInTheTable(unittest.TestCase):
    """The catalogue never states a troop type's rules, so the table is the
    only source. A type missing from it silently falls back to the engine's
    old guesses -- US1, or US2 if mounted -- which is wrong for most of them."""

    def test_the_catalogue_resolves(self):
        cat = get_catalogue()
        missing = set()
        total = resolved = 0
        for rec in cat.by_slug.values():
            raw = (rec.get('Troop Type') or '').strip()
            if not raw:
                continue
            total += 1
            if troop_types.properties(raw) is None:
                # 'Special Feature' is a terrain entry, not a model.
                if troop_types.normalise(raw) != 'special feature':
                    missing.add(raw)
            else:
                resolved += 1
        self.assertEqual(missing, set())
        self.assertGreater(resolved, 500)

    def test_all_thirteen_types_are_present(self):
        self.assertEqual(len(troop_types.TROOP_TYPES), 13)


class TestTheTable(unittest.TestCase):

    def test_regular_infantry(self):
        self.assertEqual(troop_types.unit_strength("Regular infantry", 9), 1)
        self.assertEqual(troop_types.models_per_rank("Regular infantry", 9), 5)
        self.assertEqual(troop_types.max_rank_bonus("Regular infantry", 9), 2)

    def test_heavy_infantry_ranks_four_wide(self):
        self.assertEqual(troop_types.models_per_rank("Heavy infantry", 9), 4)
        self.assertEqual(troop_types.unit_strength("Heavy infantry", 9), 1)

    def test_monstrous_infantry_is_worth_three(self):
        self.assertEqual(troop_types.unit_strength("Monstrous infantry", 9), 3)
        self.assertEqual(troop_types.models_per_rank("Monstrous infantry", 9), 3)

    def test_cavalry(self):
        self.assertEqual(troop_types.unit_strength("Light cavalry", 9), 2)
        self.assertEqual(troop_types.unit_strength("Heavy cavalry", 9), 2)
        self.assertEqual(troop_types.unit_strength("Monstrous cavalry", 9), 3)
        for kind in ("Light cavalry", "Heavy cavalry", "Monstrous cavalry"):
            self.assertEqual(troop_types.max_rank_bonus(kind, 9), 1)

    def test_war_beasts_are_worth_one(self):
        self.assertEqual(troop_types.unit_strength("War beasts", 9), 1)
        self.assertEqual(troop_types.max_rank_bonus("War beasts", 9), 1)

    def test_swarms_cannot_form_ranks(self):
        self.assertEqual(troop_types.unit_strength("Swarms", 9), 3)
        self.assertEqual(troop_types.models_per_rank("Swarms", 9), 0)
        self.assertEqual(troop_types.max_rank_bonus("Swarms", 9), 0)

    def test_heavy_chariots(self):
        self.assertEqual(troop_types.unit_strength("Heavy Chariot", 1), 5)
        self.assertEqual(troop_types.max_rank_bonus("Heavy Chariot", 2), 0)

    def test_light_chariots(self):
        self.assertEqual(troop_types.unit_strength("Light chariot", 1), 3)
        self.assertEqual(troop_types.max_rank_bonus("Light chariot", 2), 1)

    def test_an_unknown_type_keeps_the_default(self):
        self.assertEqual(troop_types.unit_strength("Wandering Minstrel", 7), 7)
        self.assertEqual(troop_types.max_rank_bonus("Wandering Minstrel", 7), 7)
        self.assertEqual(troop_types.models_per_rank("Wandering Minstrel", 7), 7)

    def test_the_rules_a_troop_type_grants(self):
        self.assertTrue(troop_types.has_rule("Heavy Chariot", "Scythed Wheels"))
        self.assertTrue(troop_types.has_rule("Heavy Chariot", "Lumbering"))
        self.assertTrue(troop_types.has_rule("Heavy Chariot", "Firing Platform"))
        self.assertTrue(troop_types.has_rule("Light Chariot", "Firing Platform"))
        # Scythed Wheels and Lumbering are heavy chariots only.
        self.assertFalse(troop_types.has_rule("Light Chariot", "Scythed Wheels"))
        self.assertFalse(troop_types.has_rule("Light Chariot", "Lumbering"))
        self.assertFalse(troop_types.has_rule("Regular infantry", "Firing Platform"))

    def test_the_infantry_rules(self):
        for kind in ("Regular infantry", "Heavy infantry"):
            for rule in ("Press of Battle", "Massed Infantry", "Parry"):
                self.assertTrue(troop_types.has_rule(kind, rule), f"{kind}/{rule}")
        # Steady in the Ranks is heavy infantry only.
        self.assertTrue(troop_types.has_rule("Heavy infantry", "Steady in the Ranks"))
        self.assertFalse(troop_types.has_rule("Regular infantry", "Steady in the Ranks"))

    def test_the_big_and_clumsy_share_a_rule(self):
        self.assertTrue(troop_types.has_rule("Monstrous infantry", "Clumsy"))
        self.assertTrue(troop_types.has_rule("Monstrous cavalry", "Clumsy"))
        self.assertFalse(troop_types.has_rule("Heavy cavalry", "Clumsy"))

    def test_the_undisciplined(self):
        self.assertTrue(troop_types.has_rule("Swarms", "Undisciplined"))
        self.assertTrue(troop_types.has_rule("War beasts", "Undisciplined"))

    def test_monsters_and_war_machines(self):
        self.assertTrue(troop_types.has_rule("Behemoth", "Thunderstomp"))
        self.assertFalse(troop_types.has_rule("Monstrous creature", "Thunderstomp"))
        self.assertTrue(troop_types.has_rule("Behemoth", "Lumbering"))
        self.assertTrue(troop_types.has_rule("War machine", "Weapon of War"))
        self.assertTrue(troop_types.has_rule("War machine", "We're Not Paid to Fight"))

    def test_split_profile_is_named_with_its_qualifier(self):
        self.assertTrue(troop_types.has_rule("Heavy Chariot", "Split Profile"))
        self.assertTrue(troop_types.has_rule("Heavy cavalry", "Split Profile"))
        self.assertTrue(troop_types.has_rule("War machine", "Split Profile"))


class TestUnitStrengthFromWounds(unittest.TestCase):
    """Monsters and war machines are worth their *starting* Wounds."""

    def test_a_monster_is_worth_its_wounds(self):
        self.assertEqual(troop_types.unit_strength("Behemoth", 1, wounds=6), 6)
        self.assertEqual(troop_types.unit_strength("Monstrous creature", 1,
                                                   wounds=4), 4)
        self.assertEqual(troop_types.unit_strength("War machine", 1, wounds=3), 3)

    def test_without_a_profile_it_falls_back(self):
        self.assertEqual(troop_types.unit_strength("Behemoth", 7), 7)

    def test_a_fixed_row_ignores_wounds(self):
        self.assertEqual(troop_types.unit_strength("Heavy Chariot", 1, wounds=6), 5)

    def test_a_real_giant(self):
        giant = model("Giant", "")
        self.assertEqual(troop_types.normalise(giant.troop_type()), "behemoth")
        self.assertEqual(giant.starting_wounds(), 6)
        self.assertEqual(giant.unit_strength(), 6)

    def test_wounds_taken_do_not_change_it(self):
        giant = model("Giant", "")
        giant.characteristics['W'] = 1
        self.assertEqual(giant.unit_strength(), 6)


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

    def test_a_rank_needs_the_troop_types_models_per_rank(self):
        # Heavy infantry rank four wide, so a trailing rank of 4 counts for
        # them where it would not for regular infantry.
        orc = model("Orc Boy", "")           # regular infantry, 5 per rank
        black = model("Black Orc", "")       # heavy infantry, 4 per rank
        self.assertEqual(rank_bonus(self._unit(orc, 14, 5, 3)), 1)
        self.assertEqual(rank_bonus(self._unit(black, 12, 4, 3)), 2)

    def test_a_unit_narrower_than_its_models_per_rank_claims_none(self):
        # No rank holds five models, so no rank counts at all.
        orc = model("Orc Boy", "")
        self.assertEqual(rank_bonus(self._unit(orc, 12, 4, 3)), 0)

    def test_monstrous_infantry_rank_three_wide(self):
        troll = model("Troll", "")
        troll.characteristics['Troop Type'] = 'Monstrous infantry'
        self.assertEqual(rank_bonus(self._unit(troll, 9, 3, 3)), 2)
        self.assertEqual(rank_bonus(self._unit(troll, 8, 3, 3)), 1)

    def test_a_swarm_claims_none(self):
        m = model("Zombie", "")
        m.characteristics['Troop Type'] = 'Swarms'
        self.assertEqual(rank_bonus(self._unit(m, 20, 5, 4)), 0)

    def test_a_behemoth_claims_none(self):
        self.assertEqual(rank_bonus(self._unit(model("Giant", ""), 1, 1, 1)), 0)


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
