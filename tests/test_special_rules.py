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
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from special_rules import (  # noqa: E402
    parse_special_rule, build_special_rules, SPECIAL_RULE_BUILDERS,
    unit_has_swiftstride, max_charge_range, max_pursuit_range, charge_roll,
    should_use_swiftstride, board_edge_distance, SWIFTSTRIDE_CHARGE_BONUS,
)
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


class TestSwiftstride(unittest.TestCase):
    """Swiftstride (Rulebook p. 178)."""

    @staticmethod
    def _unit(swiftstride=False, mount_swiftstride=False):
        mount = None
        if mount_swiftstride:
            mount = SimpleNamespace(is_swiftstride=lambda: True)
        rules = [{'name': 'Swiftstride', 'swiftstride': True}] if swiftstride else []
        m = SimpleNamespace(special_rules=rules,
                            get_mount=lambda: mount)
        # Reproduce models.model.is_swiftstride: own rules, else the mount's.
        m.is_swiftstride = lambda: (
            any(r.get('swiftstride') for r in m.special_rules)
            or bool(m.get_mount() is not None and m.get_mount().is_swiftstride()))
        return SimpleNamespace(unit=SimpleNamespace(model=m))

    def test_builder_registered(self):
        self.assertIn('swiftstride', SPECIAL_RULE_BUILDERS)
        rule = SPECIAL_RULE_BUILDERS['swiftstride'](None, None, None)
        self.assertTrue(rule['swiftstride'])

    # ─── "consists entirely of models with this special rule" ─────────────

    def test_unit_with_the_rule(self):
        self.assertTrue(unit_has_swiftstride(self._unit(swiftstride=True)))

    def test_unit_without_the_rule(self):
        self.assertFalse(unit_has_swiftstride(self._unit()))

    def test_rule_carried_by_the_mount(self):
        self.assertTrue(unit_has_swiftstride(self._unit(mount_swiftstride=True)))

    def test_joined_character_without_the_rule_breaks_it(self):
        cavalry = self._unit(swiftstride=True)
        cavalry.joinedCharacter = self._unit()      # a hero on foot
        self.assertFalse(unit_has_swiftstride(cavalry))

    def test_joined_character_with_the_rule_keeps_it(self):
        cavalry = self._unit(swiftstride=True)
        cavalry.joinedCharacter = self._unit(swiftstride=True)
        self.assertTrue(unit_has_swiftstride(cavalry))

    # ─── maximum possible charge range ────────────────────────────────────

    def test_max_charge_range(self):
        self.assertEqual(max_charge_range(8), 14)

    def test_max_charge_range_with_swiftstride(self):
        # +3", not the +6 the bonus die could theoretically add.
        self.assertEqual(max_charge_range(8, True), 14 + SWIFTSTRIDE_CHARGE_BONUS)
        self.assertEqual(SWIFTSTRIDE_CHARGE_BONUS, 3)

    def test_max_pursuit_range(self):
        # A Pursuit roll is 2D6 summed and adds no Movement.
        self.assertEqual(max_pursuit_range(), 12)

    def test_max_pursuit_range_with_swiftstride(self):
        # The bonus die applies in full here; the +3" is charge-only.
        self.assertEqual(max_pursuit_range(True), 18)

    # ─── the roll itself ──────────────────────────────────────────────────

    def test_charge_roll_discards_the_lowest(self):
        self.assertEqual(charge_roll([2, 5]), 5)

    def test_charge_roll_with_equal_dice(self):
        self.assertEqual(charge_roll([4, 4]), 4)

    def test_bonus_die_is_added_not_discarded(self):
        # 2D6 discard lowest = 5, plus the bonus die: never max(2, 5, 1).
        self.assertEqual(charge_roll([2, 5, 1]), 6)
        self.assertEqual(charge_roll([2, 5, 6]), 11)

    def test_charge_roll_of_nothing(self):
        self.assertEqual(charge_roll([]), 0)

    # ─── when to take the die ─────────────────────────────────────────────

    def test_always_taken_on_a_charge(self):
        self.assertTrue(should_use_swiftstride('charge'))

    def test_taken_when_fleeing_with_room(self):
        self.assertTrue(should_use_swiftstride('flee', distance_to_edge=20.0))

    def test_declined_when_fleeing_near_the_edge(self):
        # Fleeing off the battlefield destroys the unit.
        self.assertFalse(should_use_swiftstride('flee', distance_to_edge=4.0))

    def test_declined_when_falling_back_near_the_edge(self):
        self.assertFalse(should_use_swiftstride('fall back', distance_to_edge=4.0))

    def test_declined_when_pursuing_near_the_edge(self):
        self.assertFalse(should_use_swiftstride('pursuit', distance_to_edge=4.0))

    def test_taken_when_pursuing_with_room(self):
        self.assertTrue(should_use_swiftstride('pursuit', distance_to_edge=20.0))

    def test_taken_when_the_edge_is_unknown(self):
        self.assertTrue(should_use_swiftstride('flee'))

    def test_board_edge_distance(self):
        self.assertEqual(board_edge_distance(0.0, 0.0), 24.0)
        self.assertEqual(board_edge_distance(30.0, 0.0), 6.0)
        self.assertEqual(board_edge_distance(0.0, -22.0), 2.0)


if __name__ == "__main__":
    unittest.main()
