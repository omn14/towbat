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
    def test_movement_modifiers_affect_flight_but_never_make_it_negative(self):
        profile = model("State Trooper", "")
        profile.special_rules.append({"name": "Fly", "fly": True, "fly_movement": 8})
        initial = profile.get_movement()
        for modifier in (2, -1, -12):
            profile.characteristics['M'] = initial + modifier
            self.assertEqual(profile.get_fly_movement(), max(0, 8 + modifier))

    def test_chariot_flight_uses_modifiers_to_draught_beast_movement(self):
        profile = model("Lothern Skycutter", "")
        beast = profile.get_beasts()
        self.assertIsNotNone(beast)
        initial_ground, initial_flight = profile.get_movement(), profile.get_fly_movement()
        beast.characteristics['M'] = initial_ground + 2
        self.assertEqual(profile.get_movement(), initial_ground + 2)
        self.assertEqual(profile.get_fly_movement(), initial_flight + 2)

    def test_multiple_fly_values_use_the_best_in_either_order(self):
        profile = model("State Trooper", "")
        for values in ((8, 10), (10, 8)):
            profile.special_rules = [{"name": "Fly", "fly": True, "fly_movement": value}
                                     for value in values]
            self.assertEqual(profile.get_fly_movement(), 10)

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
