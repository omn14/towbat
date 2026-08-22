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


class TwoHandedShieldTests(unittest.TestCase):
    @staticmethod
    def _greatweapon():
        return {"name": "Great Weapon", "tag": "combat",
                "special_rules": ["Requires Two Hands"]}

    def test_two_handed_removes_shield_in_melee(self):
        m = model("State Trooper", "")
        m.set_armour(["Heavy Armour", "Shield"])  # 4+ with shield
        self.assertEqual(m.armor_save, 4)
        m.weapons["Great Weapon"] = self._greatweapon()
        m.equip_weapon("Great Weapon")
        self.assertTrue(m.melee_weapon_requires_two_hands())
        # In melee the shield is disabled, so 4+ -> 5+.
        self.assertEqual(m.melee_armour_save(), 5)
        # The general (shooting) save is unchanged.
        self.assertEqual(m.armor_save, 4)

    def test_one_handed_keeps_shield_in_melee(self):
        m = model("State Trooper", "")
        m.set_armour(["Heavy Armour", "Shield"])
        m.equip_weapon("hand weapon")
        self.assertFalse(m.melee_weapon_requires_two_hands())
        # The shield stands, and Parry improves the 4+ to a 3+.
        self.assertEqual(m.melee_armour_save(), 3)

    def test_no_shield_unaffected(self):
        m = model("State Trooper", "")
        m.set_armour(["Heavy Armour"])  # 5+, no shield
        m.weapons["Great Weapon"] = self._greatweapon()
        m.equip_weapon("Great Weapon")
        self.assertEqual(m.melee_armour_save(), 5)


class ParryTests(unittest.TestCase):
    """Parry — Rulebook p. 190. Infantry fighting with a hand weapon and a
    shield improve their armour value by 1, to a maximum of 3+."""

    @staticmethod
    def _trooper(armour, weapon="hand weapon"):
        m = model("State Trooper", "")       # regular infantry
        m.set_armour(armour)
        m.equip_weapon(weapon)
        return m

    def test_a_hand_weapon_and_shield_improve_the_save(self):
        m = self._trooper(["Light Armour", "Shield"])   # 5+
        self.assertTrue(m.parry_applies())
        self.assertEqual(m.melee_armour_save(), 4)

    def test_it_stops_at_three_up(self):
        m = self._trooper(["Full Plate Armour", "Shield"])   # 3+
        self.assertEqual(m.melee_armour_save(), 3)

    def test_a_better_save_is_not_made_worse(self):
        m = self._trooper(["Full Plate Armour", "Shield", "Barding"])  # 2+
        self.assertEqual(m.melee_armour_save(), 2)

    def test_a_shield_alone_still_parries(self):
        m = self._trooper(["Shield"])        # 6+
        self.assertEqual(m.melee_armour_save(), 5)

    def test_no_shield_no_parry(self):
        m = self._trooper(["Heavy Armour"])
        self.assertFalse(m.parry_applies())
        self.assertEqual(m.melee_armour_save(), 5)

    def test_a_two_handed_weapon_cannot_parry(self):
        # It loses the shield's bonus outright, so there is nothing to improve.
        m = self._trooper(["Heavy Armour", "Shield"])
        m.weapons["Great Weapon"] = {"name": "Great Weapon", "tag": "combat",
                                     "special_rules": ["Requires Two Hands"]}
        m.equip_weapon("Great Weapon")
        self.assertFalse(m.parry_applies())
        self.assertEqual(m.melee_armour_save(), 5)

    def test_another_one_handed_weapon_does_not_parry(self):
        # The rule asks for a hand weapon specifically, not any one-hander.
        m = self._trooper(["Heavy Armour", "Shield"])
        m.weapons["Halberd"] = {"name": "Halberd", "tag": "combat"}
        m.equip_weapon("Halberd")
        self.assertFalse(m.parry_applies())
        self.assertEqual(m.melee_armour_save(), 4)

    def test_only_the_troop_types_that_have_it(self):
        # Heavy infantry parry; a light chariot does not.
        black_orc = model("Black Orc", "")
        black_orc.set_armour(["Heavy Armour", "Shield"])
        self.assertTrue(black_orc.parry_applies())
        chariot = model("Goblin Wolf Chariot", "")
        chariot.set_armour(["Heavy Armour", "Shield"])
        self.assertFalse(chariot.parry_applies())

    def test_shooting_is_unaffected(self):
        # Parry is a close combat rule; the stored save is what shooting uses.
        m = self._trooper(["Light Armour", "Shield"])
        self.assertEqual(m.armor_save, 5)
        self.assertEqual(m.melee_armour_save(), 4)


if __name__ == "__main__":
    unittest.main()
