"""
Tests for the data-driven special-rule system (special_rules.py).

These are characterization tests for the first, non-breaking step of removing the
hardcoded model subclasses: they verify that `build_special_rules` reproduces the
engine-relevant rule hooks (charge / regen / Unbreakable) directly from the
catalogue keywords, without any subclass.

Run:  python -m unittest discover -s tests
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from special_rules import parse_special_rule, build_special_rules  # noqa: E402
from models import model  # noqa: E402


class TestParseSpecialRule(unittest.TestCase):
    def test_bare_name(self):
        self.assertEqual(parse_special_rule("Regeneration"), ("Regeneration", None))

    def test_with_param(self):
        self.assertEqual(parse_special_rule("Armour Bane (1)"), ("Armour Bane", "1"))

    def test_dice_param(self):
        self.assertEqual(parse_special_rule("Multiple Shots (D3)"), ("Multiple Shots", "D3"))

    def test_whitespace(self):
        self.assertEqual(parse_special_rule("  Fear  "), ("Fear", None))


def _by_name(rules, name):
    for r in rules:
        if r.get("name", "").lower() == name.lower():
            return r
    return None


class TestBuildSpecialRules(unittest.TestCase):
    def _rules_for(self, unit_name):
        m = model(unit_name, "")
        return build_special_rules(m)

    def test_troll_regeneration(self):
        # Common Troll lists "Regeneration" (no value) in the catalogue.
        rules = self._rules_for("Common Troll")
        regen = _by_name(rules, "Regeneration")
        self.assertIsNotNone(regen)
        self.assertIn("regen", regen)
        self.assertIsInstance(regen["regen"], int)

    def test_black_orc_furious_charge(self):
        rules = self._rules_for("Black Orc")
        fc = _by_name(rules, "Furious Charge")
        self.assertIsNotNone(fc)
        self.assertTrue(callable(fc.get("charge")))

    def test_unknown_rule_is_flag(self):
        # A rule with no coded builder still appears as a display flag.
        rules = self._rules_for("Common Troll")
        flag = _by_name(rules, "Motley Crew")
        self.assertIsNotNone(flag)
        self.assertNotIn("charge", flag)
        self.assertNotIn("regen", flag)

    def test_all_catalogue_rules_represented(self):
        m = model("Common Troll", "")
        raw = list(m.characteristics.get("Special Rules", []))
        built = build_special_rules(m)
        self.assertEqual(len(built), len(raw))


if __name__ == "__main__":
    unittest.main()
