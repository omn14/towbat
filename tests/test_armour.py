"""Tests for armour-save derivation from roster equipment."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import armour_save_from_equipment, model  # noqa: E402


class ArmourSaveCalcTests(unittest.TestCase):
    def test_none(self):
        self.assertEqual(armour_save_from_equipment([]), 7)

    def test_shield_only(self):
        self.assertEqual(armour_save_from_equipment(["Shield"]), 6)

    def test_light_armour(self):
        self.assertEqual(armour_save_from_equipment(["Light Armour"]), 6)

    def test_heavy_plus_shield(self):
        self.assertEqual(armour_save_from_equipment(["Heavy Armour", "Shield"]), 4)

    def test_full_plate(self):
        self.assertEqual(armour_save_from_equipment(["Full Plate Armour"]), 4)

    def test_full_plate_shield_barding(self):
        self.assertEqual(
            armour_save_from_equipment(["Full Plate Armour", "Shield", "Barding"]), 2)

    def test_capped_at_two(self):
        # Multiple modifiers cannot push the save better than 2+.
        self.assertEqual(
            armour_save_from_equipment(
                ["Full Plate Armour", "Shield", "Barding", "Barding"]), 2)

    def test_case_insensitive(self):
        self.assertEqual(armour_save_from_equipment(["heavy armour", "SHIELD"]), 4)


class ModelSetArmourTests(unittest.TestCase):
    def test_set_armour_updates_save_and_list(self):
        m = model("State Trooper", "")
        m.set_armour(["Heavy Armour", "Shield"])
        self.assertEqual(m.armor_save, 4)
        self.assertEqual(m.armour, ["Heavy Armour", "Shield"])

    def test_default_save_no_armour(self):
        m = model("State Trooper", "")
        self.assertEqual(m.armor_save, 7)
        self.assertEqual(m.armour, [])


if __name__ == "__main__":
    unittest.main()
