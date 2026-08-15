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
