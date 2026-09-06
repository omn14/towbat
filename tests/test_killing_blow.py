"""Killing Blow — Rulebook p. 172."""

import os
import random
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import battleFunctions  # noqa: E402
import troop_types  # noqa: E402
from battleFunctions import (check_saves, killing_blow_struck,  # noqa: E402
                             simulate_battle, take_last_killing_blows)
from battlescribe import has_killing_blow  # noqa: E402
from models import model  # noqa: E402
from special_rules import build_special_rules  # noqa: E402


def _model(name="Warrior", troop_type="Regular infantry", killing_blow=False):
    m = model(name, "")
    m.characteristics['Troop Type'] = troop_type
    if killing_blow:
        m.characteristics['Special Rules'] = ['Killing Blow']
        for entry in build_special_rules(m):
            m.special_rules.append(entry)
    return m


class RuleNameTests(unittest.TestCase):

    def test_the_catalogue_spelling_is_recognised(self):
        self.assertTrue(has_killing_blow(['Killing Blow']))
        self.assertTrue(has_killing_blow(['killing blow']))

    def test_an_unrelated_rule_is_not(self):
        self.assertFalse(has_killing_blow(['Armour Bane (1)', 'Killing Spree']))

    def test_every_weapon_that_carries_it_is_flagged(self):
        from battlescribe import get_catalogue
        missed = [w.get('name') for w in get_catalogue().weapons_by_slug.values()
                  if not w.get('killing_blow')
                  and has_killing_blow(w.get('special_rules'))]
        self.assertEqual(missed, [], "Killing Blow weapons left unflagged")


class TroopTypeTests(unittest.TestCase):
    """Only infantry or cavalry can be felled (p. 172)."""

    def test_the_infantry_sub_categories_all_count(self):
        for tt in ('Regular infantry', 'Heavy infantry', 'Monstrous infantry',
                   'Swarms'):
            self.assertTrue(troop_types.is_infantry(tt), tt)

    def test_the_cavalry_sub_categories_all_count(self):
        for tt in ('Light cavalry', 'Heavy cavalry', 'Monstrous cavalry'):
            self.assertTrue(troop_types.is_cavalry(tt), tt)

    def test_the_troop_types_it_cannot_fell(self):
        for tt in ('Behemoth', 'Monstrous creature', 'War machine',
                   'Heavy chariot', 'Light chariot', 'War beasts'):
            self.assertFalse(troop_types.is_infantry(tt), tt)
            self.assertFalse(troop_types.is_cavalry(tt), tt)

    def test_a_model_answers_for_itself(self):
        self.assertTrue(_model(troop_type='Heavy infantry').is_infantry_or_cavalry())
        self.assertFalse(_model(troop_type='Behemoth').is_infantry_or_cavalry())


class WhenItIsStruckTests(unittest.TestCase):
    """The conditions on the natural 6 (p. 172 and the FAQ)."""

    def setUp(self):
        self.attacker = _model("Witch Hunter", killing_blow=True)
        self.victim = _model("Spearman", 'Regular infantry')

    def _struck(self, natural=6, target=4, wound=True, ranged=False):
        return killing_blow_struck(self.attacker, self.victim, natural, target,
                                   wound=wound, ranged=ranged)

    def test_a_natural_six_strikes_one(self):
        self.assertTrue(self._struck())

    def test_any_other_roll_does_not(self):
        for natural in (1, 2, 3, 4, 5):
            self.assertFalse(self._struck(natural=natural), natural)

    def test_a_six_that_did_not_wound_does_not(self):
        self.assertFalse(self._struck(wound=False))

    def test_a_missile_attack_cannot_strike_one(self):
        # "an attack made in combat" — shooting is not.
        self.assertFalse(self._struck(ranged=True))

    def test_a_model_without_the_rule_cannot(self):
        self.attacker = _model("Spearman")
        self.assertFalse(self._struck())

    def test_a_target_that_is_not_infantry_or_cavalry_is_safe(self):
        self.victim = _model("Giant", 'Behemoth')
        self.assertFalse(self._struck())

    def test_an_enemy_too_tough_to_wound_cannot_be_killed(self):
        # FAQ: "If a model cannot wound an enemy, it cannot kill it."
        self.assertFalse(self._struck(target=7))

    def test_a_weapon_can_carry_the_rule(self):
        from battlescribe import get_catalogue
        m = _model("Warrior")
        m.weapons['Man-catcher'] = dict(get_catalogue().weapon('Man-catcher'))
        m.equip_weapon('Man-catcher')
        self.assertTrue(m.has_killing_blow())

    def test_a_sheathed_weapon_does_not(self):
        from battlescribe import get_catalogue
        m = _model("Warrior")
        m.weapons['Man-catcher'] = dict(get_catalogue().weapon('Man-catcher'))
        m.equip_weapon('Hand Weapon')
        self.assertFalse(m.has_killing_blow())


class SavesItDeniesTests(unittest.TestCase):
    """No armour or Regeneration save; a Ward save is attempted as normal."""

    def _victim(self, armour=2, ward=None, regen=None):
        m = _model("Victim")
        m.armor_save = armour
        if ward:
            m.special_rules.append({'name': 'Ward', 'ward': ward})
        if regen:
            m.special_rules.append({'name': 'Regeneration', 'regen': regen})
        return m

    def test_an_ordinary_wound_is_stopped_by_armour(self):
        with mock.patch.object(random, 'randint', return_value=6):
            self.assertTrue(check_saves(self._victim(armour=2), 2, 0))

    def test_a_killing_blow_ignores_armour(self):
        with mock.patch.object(random, 'randint', return_value=6):
            self.assertFalse(check_saves(self._victim(armour=2), 2, 0,
                                         killing_blow=True))

    def test_a_killing_blow_ignores_regeneration(self):
        with mock.patch.object(random, 'randint', return_value=6):
            victim = self._victim(armour=7, regen=4)
            self.assertTrue(check_saves(victim, 7, 0))
            self.assertFalse(check_saves(victim, 7, 0, killing_blow=True))

    def test_a_ward_save_still_works(self):
        with mock.patch.object(random, 'randint', return_value=6):
            self.assertTrue(check_saves(self._victim(armour=7, ward=4), 7, 0,
                                        killing_blow=True))

    def test_a_failed_ward_save_does_not_stop_it(self):
        with mock.patch.object(random, 'randint', return_value=1):
            self.assertFalse(check_saves(self._victim(armour=7, ward=4), 7, 0,
                                         killing_blow=True))


class CountingThemTests(unittest.TestCase):
    """simulate_battle reports the blows out of band, then clears them."""

    def _unit(self, m, nmodels=1, files=1, name="Unit"):
        return SimpleNamespace(model=m, nmodels=nmodels, files=files, ranks=1,
                               name=name)

    def test_the_count_is_reported_and_then_cleared(self):
        attacker = _model("Witch Hunter", killing_blow=True)
        attacker.characteristics.update({'WS': '10', 'S': '10', 'A': '3'})
        victim = _model("Spearman")
        victim.characteristics.update({'WS': '1', 'T': '1', 'W': '1'})
        victim.armor_save = 7
        with mock.patch.object(random, 'randint', return_value=6):
            simulate_battle(self._unit(attacker, name="Hunters"),
                            self._unit(victim, name="Spearmen"), charge=False)
        self.assertEqual(take_last_killing_blows(), 3)
        self.assertEqual(take_last_killing_blows(), 0, "not cleared")

    def test_nothing_is_reported_without_the_rule(self):
        attacker = _model("Spearman")
        attacker.characteristics.update({'WS': '10', 'S': '10', 'A': '3'})
        victim = _model("Spearman")
        victim.characteristics.update({'WS': '1', 'T': '1'})
        victim.armor_save = 7
        with mock.patch.object(random, 'randint', return_value=6):
            simulate_battle(self._unit(attacker), self._unit(victim),
                            charge=False)
        self.assertEqual(take_last_killing_blows(), 0)


if __name__ == '__main__':
    unittest.main()
