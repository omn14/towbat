"""Magic — Rulebook p. 106-108.

The catalogue carries every Lore of Magic as a shared infoGroup of Spell
profiles, so a Wizard's spells arrive with the roster rather than being coded
by hand.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from battlescribe import (SPELL_PHASES, _casting_value, _spell_range,  # noqa: E402
                          get_catalogue, spell_from_profile)
from models import model  # noqa: E402
from roster_importer import _wizard_level, import_roster  # noqa: E402
from spell_system import (CAST_FAILED, CAST_MISCAST, CAST_PERFECT,  # noqa: E402
                          CAST_SUCCESS, casting_outcome, casting_result,
                          dispel_result, is_dispelled, may_attempt,
                          miscast_result)

ROSTER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "strategy_armies", "Bm.json")


class TestParsingASpell(unittest.TestCase):

    FIREBALL = {"Number": "1", "Type": "Magic Missile", "Casting Value": "8+",
                "Range": '24"', "Effect": "The target enemy unit suffers 2D6…"}

    def test_a_whole_spell(self):
        s = spell_from_profile("Fireball", self.FIREBALL)
        self.assertEqual(s["name"], "Fireball")
        self.assertEqual(s["number"], 1)
        self.assertEqual(s["type"], "Magic Missile")
        self.assertEqual(s["casting_value"], 8)
        self.assertEqual(s["range"], 24)
        self.assertTrue(s["effect"])

    def test_casting_values(self):
        self.assertEqual(_casting_value("8+"), 8)
        self.assertEqual(_casting_value("10+"), 10)
        # A boosted spell writes both versions; the basic one is what we cast.
        self.assertEqual(_casting_value("8+/11+"), 8)
        self.assertIsNone(_casting_value("-"))
        self.assertIsNone(_casting_value(None))

    def test_ranges(self):
        self.assertEqual(_spell_range('24"'), 24)
        self.assertEqual(_spell_range("12"), 12)
        self.assertEqual(_spell_range('24\u2019\u2019'), 24)   # curly quotes in the data
        self.assertEqual(_spell_range("Self"), "Self")
        self.assertEqual(_spell_range("Combat"), "Combat")
        self.assertIsNone(_spell_range("-"))

    def test_the_signature_spell_has_no_number(self):
        s = spell_from_profile("Hammerhand", dict(self.FIREBALL, Number=""))
        self.assertIsNone(s["number"])

    def test_the_type_decides_the_phase(self):
        # Rulebook p. 108: when a spell may be cast depends on its type.
        self.assertEqual(SPELL_PHASES["enchantment"], "strategy")
        self.assertEqual(SPELL_PHASES["hex"], "strategy")
        self.assertEqual(SPELL_PHASES["conveyance"], "movement")
        self.assertEqual(SPELL_PHASES["magic missile"], "shooting")
        self.assertEqual(SPELL_PHASES["magical vortex"], "shooting")
        self.assertEqual(SPELL_PHASES["assailment"], "combat")

    def test_an_unknown_type_still_lands_somewhere(self):
        s = spell_from_profile("Odd", {"Type": "Whatever", "Casting Value": "7+"})
        self.assertEqual(s["phase"], "strategy")


class TestTheCatalogue(unittest.TestCase):

    def setUp(self):
        self.cat = get_catalogue()

    def test_the_lores_load(self):
        self.assertGreater(len(self.cat.lores), 30)
        self.assertIn("Battle Magic", self.cat.lores)
        self.assertIn("Necromancy", self.cat.lores)

    def test_a_full_lore_has_seven_spells(self):
        # Six numbered and a signature (Rulebook p. 106).
        spells = self.cat.lore("Battle Magic")
        self.assertEqual(len(spells), 7)
        self.assertEqual(sorted(s["number"] for s in spells
                                if s["number"] is not None), [1, 2, 3, 4, 5, 6])
        self.assertEqual(sum(1 for s in spells if s["number"] is None), 1)

    def test_a_spell_by_name(self):
        s = self.cat.spell("Fireball")
        self.assertEqual(s["casting_value"], 8)
        self.assertEqual(s["phase"], "shooting")

    def test_an_unknown_spell(self):
        self.assertIsNone(self.cat.spell("Not A Spell"))
        self.assertIsNone(self.cat.lore("Not A Lore"))

    def test_lookups_are_copies(self):
        s = self.cat.spell("Fireball")
        s["casting_value"] = 99
        self.assertEqual(self.cat.spell("Fireball")["casting_value"], 8)


class TestImportingAWizard(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.army = import_roster(ROSTER)
        cls.wizard = next((u for u in cls.army["units"] if u.get("spells")), None)

    def test_the_roster_carries_the_chosen_lore(self):
        self.assertIsNotNone(self.wizard, "no wizard in the sample roster")
        self.assertEqual(len(self.wizard["spells"]), 7)
        self.assertIn("Fireball", [s["name"] for s in self.wizard["spells"]])

    def test_the_level_of_wizardry(self):
        self.assertEqual(self.wizard["wizard_level"], 2)

    def test_units_without_magic_get_none(self):
        for u in self.army["units"]:
            if u is not self.wizard:
                self.assertEqual(u["spells"], [])
                self.assertIsNone(u["wizard_level"])

    def test_the_level_comes_from_the_upgrade(self):
        self.assertEqual(_wizard_level({"name": "Master Mage", "selections": [
            {"name": "Wizard Level 3"}]}), 3)
        self.assertEqual(_wizard_level({"name": "Mage", "selections": [
            {"name": "Battle Magic", "selections": [{"name": "Wizard Level 4"}]}]}), 4)
        self.assertIsNone(_wizard_level({"name": "State Trooper"}))


class TestTheModel(unittest.TestCase):

    def test_an_ordinary_model_knows_no_spells(self):
        m = model("Zombie", "")
        self.assertEqual(m.spells, {})
        self.assertFalse(m.is_wizard())
        self.assertEqual(m.wizard_level(0), 0)

    def test_a_wizard_reports_its_level(self):
        m = model("Zombie", "")
        m.special_rules.append({'name': 'Wizard Level 3', 'tag': 'wizard',
                                'wizard': True, 'wizard_level': 3})
        self.assertTrue(m.is_wizard())
        self.assertEqual(m.wizard_level(), 3)


class TestTheCastingRoll(unittest.TestCase):
    """2D6 plus the caster's Level of Wizardry divided by 2, rounding up."""

    def test_the_rulebooks_own_example(self):
        # A Level 2 Wizard rolling 1 and 6 has a casting result of 8.
        self.assertEqual(casting_result(7, 2), 8)

    def test_each_level(self):
        self.assertEqual(casting_result(7, 1), 8)   # 1/2 rounds up to 1
        self.assertEqual(casting_result(7, 2), 8)
        self.assertEqual(casting_result(7, 3), 9)   # 3/2 rounds up to 2
        self.assertEqual(casting_result(7, 4), 9)

    def test_no_level_adds_nothing(self):
        self.assertEqual(casting_result(7, 0), 7)

    def test_the_spell_is_cast_on_equalling_its_value(self):
        self.assertTrue(casting_result(6, 2) >= 7)
        self.assertFalse(casting_result(5, 2) >= 7)


class TestOutcomes(unittest.TestCase):
    """A natural double 6 is a perfect invocation, a natural double 1 a
    miscast, whatever the casting result would have been (p. 109)."""

    def test_a_plain_success(self):
        self.assertEqual(casting_outcome([4, 3], 2, 7), (CAST_SUCCESS, 8))

    def test_a_plain_failure(self):
        self.assertEqual(casting_outcome([2, 2], 2, 7), (CAST_FAILED, 5))

    def test_a_perfect_invocation(self):
        outcome, result = casting_outcome([6, 6], 1, 12)
        self.assertEqual(outcome, CAST_PERFECT)
        self.assertEqual(result, 13)

    def test_a_perfect_invocation_beats_any_casting_value(self):
        # "cast regardless of its casting value".
        self.assertEqual(casting_outcome([6, 6], 1, 99)[0], CAST_PERFECT)

    def test_a_miscast(self):
        self.assertEqual(casting_outcome([1, 1], 4, 3)[0], CAST_MISCAST)

    def test_a_miscast_even_when_the_result_would_pass(self):
        # Level 4 doubles 1 gives a result of 4, but a 2+ spell still miscasts.
        self.assertEqual(casting_outcome([1, 1], 4, 2)[0], CAST_MISCAST)

    def test_two_ones_that_are_not_a_double(self):
        self.assertEqual(casting_outcome([1, 2], 0, 3), (CAST_SUCCESS, 3))


class TestTheMiscastTable(unittest.TestCase):

    def test_every_roll_lands_on_a_row(self):
        for roll in range(2, 13):
            self.assertTrue(miscast_result(roll)['name'], roll)

    def test_an_impossible_roll(self):
        with self.assertRaises(ValueError):
            miscast_result(13)

    def test_the_low_rows_lose_the_spell(self):
        for roll in (2, 4, 5, 6, 7):
            self.assertFalse(miscast_result(roll)['cast'], roll)

    def test_dimensional_cascade(self):
        e = miscast_result(3)
        self.assertEqual((e['blast'], e['strength'], e['ap']), (5, 10, 4))

    def test_careless_conjuration_hits_only_the_wizard(self):
        e = miscast_result(7)
        self.assertEqual((e['blast'], e['strength'], e['ap']), (0, 4, 1))

    def test_barely_controlled_power_casts_at_the_casting_value(self):
        e = miscast_result(8)
        self.assertTrue(e['cast'])
        self.assertTrue(e['at_casting_value'])
        self.assertTrue(e['no_more_spells'])
        self.assertFalse(e['perfect'])

    def test_power_drain_is_a_perfect_invocation(self):
        e = miscast_result(11)
        self.assertTrue(e['cast'])
        self.assertTrue(e['perfect'])
        self.assertTrue(e['no_more_spells'])


class TestDispelling(unittest.TestCase):

    def test_a_fated_dispel_adds_nothing(self):
        self.assertEqual(dispel_result([4, 3]), 7)
        self.assertEqual(dispel_result([4, 3], wizard_level=4), 7)

    def test_a_wizardly_dispel_adds_half_the_level(self):
        self.assertEqual(dispel_result([4, 3], 2, wizardly=True), 8)
        self.assertEqual(dispel_result([4, 3], 3, wizardly=True), 9)

    def test_it_must_exceed_the_casting_result(self):
        self.assertTrue(is_dispelled(9, 8))
        self.assertFalse(is_dispelled(8, 8))    # a tie fails
        self.assertFalse(is_dispelled(7, 8))


class TestHowManySpellsATurn(unittest.TestCase):
    """As many spells a turn as the Level of Wizardry, each only once."""

    def test_a_level_two_wizard_gets_two(self):
        self.assertTrue(may_attempt([], 'Fireball', 2))
        self.assertTrue(may_attempt(['Fireball'], 'Hammerhand', 2))
        self.assertFalse(may_attempt(['Fireball', 'Hammerhand'], 'Tempest', 2))

    def test_no_spell_twice(self):
        self.assertFalse(may_attempt(['Fireball'], 'Fireball', 4))

    def test_a_level_one_wizard_gets_one(self):
        self.assertTrue(may_attempt([], 'Fireball', 1))
        self.assertFalse(may_attempt(['Fireball'], 'Tempest', 1))

    def test_a_spent_wizard_gets_none(self):
        self.assertFalse(may_attempt([], 'Fireball', 4, blocked=True))


if __name__ == "__main__":
    unittest.main()
