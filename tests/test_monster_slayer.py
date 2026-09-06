"""Monster Slayer — Rulebook p. 173."""

import os
import random
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import troop_types  # noqa: E402
from battleFunctions import (simulate_battle, slaying_blow_struck,  # noqa: E402
                             slaying_rule_for, take_last_slaying_blows)
from battlescribe import get_catalogue, has_monster_slayer  # noqa: E402
from models import model  # noqa: E402
from special_rules import build_special_rules  # noqa: E402


def _model(name="Warrior", troop_type="Regular infantry", rules=()):
    m = model(name, "")
    m.characteristics['Troop Type'] = troop_type
    if rules:
        m.characteristics['Special Rules'] = list(rules)
        for entry in build_special_rules(m):
            m.special_rules.append(entry)
    return m


class RuleNameTests(unittest.TestCase):

    def test_the_catalogue_spelling_is_recognised(self):
        self.assertTrue(has_monster_slayer(['Monster Slayer']))
        self.assertTrue(has_monster_slayer(['monster slayer']))

    def test_an_unrelated_rule_is_not(self):
        self.assertFalse(has_monster_slayer(['Killing Blow', 'Monstrous Beast']))

    def test_every_weapon_that_carries_it_is_flagged(self):
        missed = [w.get('name') for w in get_catalogue().weapons_by_slug.values()
                  if not w.get('monster_slayer')
                  and has_monster_slayer(w.get('special_rules'))]
        self.assertEqual(missed, [], "Monster Slayer weapons left unflagged")

    def test_the_keyword_builds_a_rule(self):
        m = _model("Slayer", rules=['Monster Slayer'])
        self.assertTrue(any(r.get('monster_slayer') for r in m.special_rules))
        self.assertTrue(m.has_monster_slayer())


class TroopTypeTests(unittest.TestCase):
    """Only monsters can be felled: monstrous creatures and behemoths (p. 196)."""

    def test_both_monster_sub_categories_count(self):
        for tt in ('Monstrous creature', 'Behemoth', 'Behemoth (character)'):
            self.assertTrue(troop_types.is_monster(tt), tt)

    def test_monstrous_infantry_and_cavalry_are_not_monsters(self):
        # 'Monstrous' is not 'monster': these are infantry and cavalry.
        for tt in ('Monstrous infantry', 'Monstrous cavalry'):
            self.assertFalse(troop_types.is_monster(tt), tt)

    def test_the_ordinary_troop_types_are_not(self):
        for tt in ('Regular infantry', 'Heavy infantry', 'Heavy cavalry',
                   'War machine', 'Heavy chariot', 'War beasts', 'Swarms'):
            self.assertFalse(troop_types.is_monster(tt), tt)

    def test_a_model_answers_for_itself(self):
        self.assertTrue(_model(troop_type='Behemoth').is_monster())
        self.assertFalse(_model(troop_type='Monstrous infantry').is_monster())


class WhichBlowLandsTests(unittest.TestCase):
    """The two rules divide the troop types between them."""

    def setUp(self):
        self.slayer = _model("Warrior", 'Heavy infantry',
                             rules=['Monster Slayer'])
        self.giant = _model("Giant", 'Behemoth')

    def test_a_monster_slayer_fells_a_monster(self):
        self.assertEqual(slaying_rule_for(self.slayer, self.giant),
                         'Monster Slayer')

    def test_it_does_nothing_to_infantry(self):
        self.assertIsNone(slaying_rule_for(self.slayer,
                                           _model("Spearman", 'Regular infantry')))

    def test_killing_blow_does_nothing_to_a_monster(self):
        hunter = _model("Witch Hunter", 'Regular infantry',
                        rules=['Killing Blow'])
        self.assertIsNone(slaying_rule_for(hunter, self.giant))

    def test_a_weapon_with_both_picks_by_the_target(self):
        # Four weapons carry both rules; the target decides which one lands.
        m = _model("Champion", 'Heavy infantry')
        m.weapons['Decapitating Strike'] = dict(
            get_catalogue().weapon('Decapitating Strike'))
        m.equip_weapon('Decapitating Strike')
        self.assertTrue(m.has_monster_slayer())
        self.assertTrue(m.has_killing_blow())
        self.assertEqual(slaying_rule_for(m, self.giant), 'Monster Slayer')
        self.assertEqual(
            slaying_rule_for(m, _model("Spearman", 'Regular infantry')),
            'Killing Blow')

    def test_a_sheathed_weapon_carries_nothing(self):
        m = _model("Champion", 'Heavy infantry')
        m.weapons['Decapitating Strike'] = dict(
            get_catalogue().weapon('Decapitating Strike'))
        m.equip_weapon('Hand Weapon')
        self.assertFalse(m.has_monster_slayer())


class WhenItIsStruckTests(unittest.TestCase):
    """The conditions on the natural 6 (p. 173 and the FAQ)."""

    def setUp(self):
        self.attacker = _model("Warrior", 'Heavy infantry',
                               rules=['Monster Slayer'])
        self.victim = _model("Giant", 'Behemoth')

    def _struck(self, natural=6, target=4, wound=True, ranged=False):
        return slaying_blow_struck(self.attacker, self.victim, natural, target,
                                   wound=wound, ranged=ranged)

    def test_a_natural_six_strikes_one(self):
        self.assertEqual(self._struck(), 'Monster Slayer')

    def test_any_other_roll_does_not(self):
        for natural in (1, 2, 3, 4, 5):
            self.assertIsNone(self._struck(natural=natural), natural)

    def test_a_six_that_did_not_wound_does_not(self):
        self.assertIsNone(self._struck(wound=False))

    def test_a_missile_attack_cannot_strike_one(self):
        # "an attack made in combat" — shooting is not.
        self.assertIsNone(self._struck(ranged=True))

    def test_a_model_without_the_rule_cannot(self):
        self.attacker = _model("Spearman", 'Heavy infantry')
        self.assertIsNone(self._struck())

    def test_a_monstrous_infantry_target_is_safe(self):
        self.victim = _model("Ogre", 'Monstrous infantry')
        self.assertIsNone(self._struck())

    def test_an_enemy_too_tough_to_wound_cannot_be_killed(self):
        # FAQ: "If a model cannot wound an enemy, it cannot kill it."
        self.assertIsNone(self._struck(target=7))


class CountingThemTests(unittest.TestCase):
    """simulate_battle reports the blows out of band, then clears them."""

    def _unit(self, m, nmodels=1, name="Unit"):
        return SimpleNamespace(model=m, nmodels=nmodels, files=1, ranks=1,
                               name=name)

    def test_a_slain_monster_is_counted(self):
        attacker = _model("Warrior", 'Heavy infantry',
                          rules=['Monster Slayer'])
        attacker.characteristics.update({'WS': '10', 'S': '10', 'A': '2'})
        victim = _model("Giant", 'Behemoth')
        victim.characteristics.update({'WS': '1', 'T': '1', 'W': '6'})
        victim.armor_save = 3
        with mock.patch.object(random, 'randint', return_value=6):
            simulate_battle(self._unit(attacker, name="Slayers"),
                            self._unit(victim, name="Giant"), charge=False)
        self.assertEqual(take_last_slaying_blows(), 2)
        self.assertEqual(take_last_slaying_blows(), 0, "not cleared")

    def test_the_same_attacker_gets_nothing_from_infantry(self):
        attacker = _model("Warrior", 'Heavy infantry',
                          rules=['Monster Slayer'])
        attacker.characteristics.update({'WS': '10', 'S': '10', 'A': '2'})
        victim = _model("Spearman", 'Regular infantry')
        victim.characteristics.update({'WS': '1', 'T': '1'})
        victim.armor_save = 7
        with mock.patch.object(random, 'randint', return_value=6):
            simulate_battle(self._unit(attacker), self._unit(victim),
                            charge=False)
        self.assertEqual(take_last_slaying_blows(), 0)


if __name__ == '__main__':
    unittest.main()
