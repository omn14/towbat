"""Tests for the Fly (X) special rule.

Covers the rule builder, the model flag helpers, and Fly Movement parsing.
Run:  python3 -m unittest tests.test_fly
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import model  # noqa: E402
from special_rules import SPECIAL_RULE_BUILDERS, parse_special_rule  # noqa: E402


class FlyRuleTests(unittest.TestCase):
    def test_builder_registered(self):
        self.assertIn("fly", SPECIAL_RULE_BUILDERS)

    def test_builder_parses_movement(self):
        display, param = parse_special_rule("Fly (9)")
        self.assertEqual(display, "Fly")
        entry = SPECIAL_RULE_BUILDERS["fly"](None, param, None)
        self.assertTrue(entry["fly"])
        self.assertEqual(entry["fly_movement"], 9)

    def test_builder_without_value(self):
        entry = SPECIAL_RULE_BUILDERS["fly"](None, None, None)
        self.assertTrue(entry["fly"])
        self.assertNotIn("fly_movement", entry)


class FlyFlagTests(unittest.TestCase):
    def test_is_flying_true(self):
        m = model("State Trooper", "")
        m.special_rules.append({"name": "Fly", "fly": True, "fly_movement": 8})
        self.assertTrue(m.is_flying())
        self.assertEqual(m.get_fly_movement(), 8)

    def test_is_flying_false(self):
        m = model("State Trooper", "")
        self.assertFalse(m.is_flying())
        self.assertEqual(m.get_fly_movement(default=0), 0)

    def test_fly_movement_default_without_value(self):
        m = model("State Trooper", "")
        m.special_rules.append({"name": "Fly", "fly": True})
        self.assertTrue(m.is_flying())
        self.assertEqual(m.get_fly_movement(default=4), 4)


if __name__ == "__main__":
    unittest.main()
