"""Tests for the Skirmishers special rule (Phase 0 + 1).

Covers the rule flag/state helpers and the enemy-fire -1 To Hit modifier.
Run:  python3 -m unittest tests.test_skirmishers
"""

import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import model  # noqa: E402
from special_rules import SPECIAL_RULE_BUILDERS  # noqa: E402
from toHitAndToWound import to_hit_ranged  # noqa: E402


SKIRMISH_RULE = {"name": "Skirmishers", "tag": "formation", "skirmish": True}


class SkirmisherFlagTests(unittest.TestCase):
    def test_builder_registered(self):
        self.assertIn("skirmishers", SPECIAL_RULE_BUILDERS)

    def test_is_skirmisher_true(self):
        m = model("State Trooper", "")
        m.special_rules.append(SKIRMISH_RULE)
        self.assertTrue(m.is_skirmisher())

    def test_is_skirmisher_false(self):
        m = model("State Trooper", "")
        self.assertFalse(m.is_skirmisher())

    def test_character_defaults_to_skirmish_on_foot_or_cavalry_mount(self):
        for mount_name in (None, 'Elven Steed'):
            with self.subTest(mount=mount_name):
                character = model('Noble', '')
                if mount_name:
                    character.attach_mount(SimpleNamespace(model=model(mount_name, '')))
                self.assertTrue(character.is_skirmisher())

    def test_explicit_character_formation_takes_precedence(self):
        for formation in ('Close Order', 'Open Order'):
            with self.subTest(formation=formation):
                character = model('Noble', '')
                character.special_rules.append({'name': formation})
                self.assertFalse(character.is_skirmisher())

    def test_monster_and_chariot_characters_use_mount_formation(self):
        for mount_name in ('Star Dragon', 'Tiranoc Chariot'):
            with self.subTest(mount=mount_name):
                character = model('Noble', '')
                mount = model(mount_name, '')
                character.attach_mount(SimpleNamespace(model=mount))
                character.special_rules.append(SKIRMISH_RULE)
                self.assertFalse(character.is_skirmisher())
                mount.special_rules.append(SKIRMISH_RULE)
                self.assertTrue(character.is_skirmisher())

    def test_joined_formation_overrides_then_restores_own_default(self):
        from characters import detach_character
        for formation in (None, 'Close Order'):
            with self.subTest(formation=formation):
                profile = model('Noble', '')
                if formation:
                    profile.special_rules.append({'name': formation})
                own_skirmish = profile.is_skirmisher()
                profile._joined_skirmish = not own_skirmish
                character = SimpleNamespace(unit=SimpleNamespace(model=profile, name='Noble'),
                                            isSkirmisher=not own_skirmish)
                host = SimpleNamespace(unit=SimpleNamespace(model=model('State Trooper', ''), name='Host'),
                                       joinedCharacter=character)
                character.hostUnit = host
                self.assertEqual(profile.is_skirmisher(), not own_skirmish)
                detach_character(host)
                self.assertEqual(profile.is_skirmisher(), own_skirmish)
                self.assertEqual(character.isSkirmisher, own_skirmish)

    def test_unit_strength_default(self):
        m = model("State Trooper", "")
        self.assertEqual(m.unit_strength(), 1)


class EnemyFireModifierTests(unittest.TestCase):
    @staticmethod
    def _shooter(bs, roll):
        return SimpleNamespace(characteristics={"BS": str(bs)},
                               attack_roll=roll, equipedWeapon={})

    def test_skirmisher_target_minus_one(self):
        # BS3 hits on 4+. A roll of 4 hits normally; -1 vs skirmishers needs 5+.
        self.assertTrue(to_hit_ranged(self._shooter(3, 4)))
        self.assertFalse(to_hit_ranged(self._shooter(3, 4), target_skirmisher=True))

    def test_non_skirmisher_unaffected(self):
        self.assertTrue(to_hit_ranged(self._shooter(3, 4), target_skirmisher=False))


if __name__ == "__main__":
    unittest.main()
