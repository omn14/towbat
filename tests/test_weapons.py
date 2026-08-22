"""
Unit tests for the data-driven weapon and weapon-special-rule system.

Covers:
  * weapon_from_profile mapping (ranged/combat, S/AP, charge-only, notes)
  * catalogue weapon lookup by name
  * per-model weapon mechanics (melee strength/AP, armour bane)
  * combat integration (charge Strength/AP, Armour Bane on a natural 6)

Run:  python3 -m unittest discover -s tests
  or: python3 -m pytest tests/

Adding a new rule test: build a model, give_weapon(name) (or craft a weapon
dict), set model.charging as needed, then assert on the model helpers or on
simulate_attack()'s per-attack results.
"""

import contextlib
import io
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

# Make the repo root importable regardless of how the tests are launched.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from battlescribe import get_catalogue, weapon_from_profile  # noqa: E402
from models import model, armour_bane_x  # noqa: E402
from battleFunctions import simulate_attack, simulate_battle  # noqa: E402


@contextlib.contextmanager
def quiet():
    """Silence the verbose combat prints during a test."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def mk_unit(m, nmodels=5, files=5, ranks=1, name="u"):
    """Lightweight stand-in for units.unit (avoids the Panda3D dependency)."""
    return SimpleNamespace(model=m, name=name, nmodels=nmodels, files=files, ranks=ranks)


# Sample profiles mirroring the catalogue data, for mapper tests.
HANDGUN = {"R": '24"', "S": "4", "AP": "-1", "Special Rules": "Armour Bane (1), Ponderous"}
LANCE = {"R": "Combat", "S": "S+2", "AP": "-2", "Special Rules": "Armour Bane (1)",
         "Notes": "A lance can only be used during a turn in which the wielder charged."}
HALBERD = {"R": "Combat", "S": "S+1", "AP": "-1 (-2)",
           "Special Rules": "Armour Bane (1), Requires Two Hands",
           "Notes": "A halberd has an AP of -2 against enemy models the wielder charged this turn."}


class WeaponMapperTests(unittest.TestCase):
    def test_ranged_handgun(self):
        w = weapon_from_profile("Handgun", HANDGUN)
        self.assertEqual(w["tag"], "ranged")
        self.assertEqual(w["ranged_range"], 24)
        self.assertEqual(w["ranged_strength"], 4)
        self.assertEqual(w["ranged_AP"], 1)          # roster '-1' -> penetration 1
        self.assertEqual(w["ranged_shots"], 1)
        self.assertFalse(w["volley_fire"])
        self.assertIn("Armour Bane (1)", w["special_rules"])

    def test_lance_is_charge_only(self):
        w = weapon_from_profile("Lance", LANCE)
        self.assertEqual(w["tag"], "combat")
        self.assertTrue(w["charge_only"])
        self.assertEqual(w["strength_bonus"], 2)
        self.assertEqual(w["ap_penetration"], 2)
        self.assertNotIn("ap_penetration_charge", w)  # single AP value

    def test_halberd_conditional_ap_and_notes(self):
        w = weapon_from_profile("Halberd", HALBERD)
        self.assertNotIn("charge_only", w)            # S+1 is always-on
        self.assertEqual(w["strength_bonus"], 1)
        self.assertEqual(w["ap_penetration"], 1)      # base
        self.assertEqual(w["ap_penetration_charge"], 2)  # on the charge
        self.assertIn("notes", w)

    def test_volley_and_multiple_shots(self):
        w = weapon_from_profile("Blunderbuss", {
            "R": '12"', "S": "3", "AP": "-1",
            "Special Rules": "Multiple Shots (3), Volley Fire"})
        self.assertEqual(w["ranged_shots"], 3)
        self.assertTrue(w["volley_fire"])

    def test_multiple_shots_dice(self):
        w = weapon_from_profile("Doom Diver", {
            "R": '48"', "S": "3", "AP": "-", "Special Rules": "Multiple Shots (D3)"})
        self.assertEqual(w.get("ranged_shots_dice"), "D3")
        self.assertEqual(w["ranged_shots"], 1)  # single-shot fallback


class DiceExprTests(unittest.TestCase):
    def test_fixed(self):
        from models import roll_dice_expr
        self.assertEqual(roll_dice_expr("2"), 2)

    def test_dice_ranges(self):
        from models import roll_dice_expr
        for _ in range(200):
            self.assertIn(roll_dice_expr("D3"), (1, 2, 3))
            self.assertIn(roll_dice_expr("D6"), range(1, 7))
            self.assertIn(roll_dice_expr("D3+1"), (2, 3, 4))

    def test_roll_ranged_shots(self):
        m = model("State Missile Trooper", "")
        m.give_weapon("Handgun"); m.equip_weapon("Handgun")
        self.assertEqual(m.roll_ranged_shots(), 1)  # Handgun = single shot
        # A weapon with a dice count rolls in range.
        m.weapons["Dicey"] = {"name": "Dicey", "tag": "ranged", "ranged_shots_dice": "D3"}
        m.equip_weapon("Dicey")
        for _ in range(100):
            self.assertIn(m.roll_ranged_shots(), (1, 2, 3))


class MultipleShotsCombatTests(unittest.TestCase):
    def test_multiple_shots_apply_minus_one_to_hit(self):
        # Firing multiple shots imposes -1 To Hit on each shot.
        import toHitAndToWound
        m = model("State Trooper", "")
        m.weapons["Gun"] = {"name": "Gun", "tag": "ranged", "ranged_strength": 4,
                            "ranged_AP": 0, "ranged_shots": 2}
        m.equip_weapon("Gun")
        m.characteristics["BS"] = "3"; m.attack_roll = 4  # BS3 hits on 4+
        self.assertTrue(toHitAndToWound.to_hit_ranged(m, multiple_shots=False))
        # With -1, BS3 now needs 5+, so a roll of 4 misses.
        self.assertFalse(toHitAndToWound.to_hit_ranged(m, multiple_shots=True))

    def test_per_model_rolls_scale_with_files(self):
        # Fixed Multiple Shots (2): 5 front-rank models -> 10 shots.
        with quiet():
            m = model("State Trooper", "")
            m.weapons["Gun"] = {"name": "Gun", "tag": "ranged", "ranged_strength": 4,
                                "ranged_AP": 0, "ranged_shots": 2}
            m.equip_weapon("Gun")
            res = simulate_battle(mk_unit(m, 5, 5, 1), mk_unit(model("Zombie", ""), 10, 5, 2),
                                  charge=False)
        self.assertEqual(res[0], 10)  # attacks = shots(2) * front-rank models(5)


class ArmourBaneParseTests(unittest.TestCase):
    def test_values(self):
        self.assertEqual(armour_bane_x(["Armour Bane (1)"]), 1)
        self.assertEqual(armour_bane_x(["Armour Bane (2)", "X"]), 2)
        self.assertEqual(armour_bane_x(["Armour Bane(3)"]), 3)   # no space
        self.assertEqual(armour_bane_x(["Ponderous"]), 0)
        self.assertEqual(armour_bane_x([]), 0)


class CatalogueLookupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cat = get_catalogue()

    def test_weapon_by_name(self):
        self.assertIsNotNone(self.cat.weapon("Halberd"))
        self.assertIsNotNone(self.cat.weapon("Handgun"))

    def test_missing_weapon(self):
        self.assertIsNone(self.cat.weapon("Not A Real Weapon 123"))


class MeleeStrengthTests(unittest.TestCase):
    def test_halberd_strength_always_on(self):
        m = model("State Trooper", ""); m.give_weapon("Halberd")
        m.charging = False
        m.equip_best_melee()
        self.assertEqual(m.melee_strength_bonus(), 1)
        m.charging = True
        m.equip_best_melee()
        self.assertEqual(m.melee_strength_bonus(), 1)

    def test_lance_strength_charge_only(self):
        m = model("Demigryph Knight", ""); m.give_weapon("Lance")
        m.charging = False
        m.equip_best_melee()
        self.assertEqual(m.melee_strength_bonus(), 0)
        m.charging = True
        m.equip_best_melee()
        self.assertEqual(m.melee_strength_bonus(), 2)

    def test_a_weapon_that_is_carried_but_not_used_gives_nothing(self):
        # Owning a halberd is not wielding one. The bonuses used to be taken
        # from the best weapon a model owned, so a State Trooper told to fight
        # with its hand weapon still swung at the halberd's Strength and AP.
        m = model("State Trooper", ""); m.give_weapon("Halberd")
        m.charging = True
        m.equip_weapon("hand weapon")
        self.assertEqual(m.melee_strength_bonus(), 0)
        self.assertEqual(m.melee_ap(), 0)
        self.assertEqual(m.armour_bane_for_attack(), 0)

    def test_a_ranged_weapon_is_no_help_in_melee(self):
        m = model("State Missile Trooper", ""); m.give_weapon("Handgun")
        m.equip_weapon("Handgun")
        self.assertEqual(m.melee_strength_bonus(), 0)
        self.assertEqual(m.melee_ap(), 0)

    def test_apply_melee_strength(self):
        m = model("State Trooper", ""); m.give_weapon("Halberd")
        base = int(m.characteristics["S"])
        m.charging = False
        m.equip_best_melee()
        m.apply_melee_strength()
        self.assertEqual(int(m.characteristics["S"]), base + 1)


class MeleeApTests(unittest.TestCase):
    def test_halberd_ap_conditional(self):
        m = model("State Trooper", ""); m.give_weapon("Halberd")
        m.charging = False
        m.equip_best_melee()
        self.assertEqual(m.melee_ap(), 1)
        m.charging = True
        m.equip_best_melee()
        self.assertEqual(m.melee_ap(), 2)

    def test_lance_ap_charge_only(self):
        m = model("Demigryph Knight", ""); m.give_weapon("Lance")
        m.charging = False
        m.equip_best_melee()
        self.assertEqual(m.melee_ap(), 0)
        m.charging = True
        m.equip_best_melee()
        self.assertEqual(m.melee_ap(), 2)


class EquipBestMeleeTests(unittest.TestCase):
    def test_equips_halberd_over_hand_weapon(self):
        m = model("State Trooper", ""); m.give_weapon("Halberd")
        m.charging = False
        self.assertEqual(m.equip_best_melee(), "Halberd")
        self.assertEqual(m.equipedWeapon["name"], "Halberd")

    def test_lance_only_equipped_while_charging(self):
        m = model("Demigryph Knight", ""); m.give_weapon("Lance")
        m.charging = False
        self.assertEqual(m.equip_best_melee(), "hand weapon")
        m.charging = True
        self.assertEqual(m.equip_best_melee(), "Lance")

    def test_falls_back_to_hand_weapon(self):
        m = model("State Trooper", "")
        m.charging = False
        self.assertEqual(m.equip_best_melee(), "hand weapon")


class ArmourBaneCombatTests(unittest.TestCase):
    def test_bonus_on_natural_six(self):
        with mock.patch("battleFunctions.random.randint", return_value=6), quiet():
            m = model("State Trooper", ""); m.give_weapon("Halberd")
            m.equip_best_melee()
            simulate_attack(m, model("State Trooper", ""))
        # not charging: melee AP 1 + Armour Bane 1 on the natural 6 = 2
        self.assertEqual(m.attack_AP, 2)

    def test_no_bonus_on_non_six(self):
        with mock.patch("battleFunctions.random.randint", return_value=4), quiet():
            m = model("State Trooper", ""); m.give_weapon("Halberd")
            m.equip_best_melee()
            simulate_attack(m, model("State Trooper", ""))
        self.assertEqual(m.attack_AP, m.melee_ap())  # AP 1, no armour bane

    def test_hand_weapon_has_no_armour_bane(self):
        with mock.patch("battleFunctions.random.randint", return_value=6), quiet():
            m = model("State Trooper", "")
            simulate_attack(m, model("State Trooper", ""))
        self.assertEqual(m.attack_AP, 0)


class RangedCombatTests(unittest.TestCase):
    def test_handgun_equipped_ap(self):
        m = model("State Missile Trooper", ""); m.give_weapon("Handgun")
        m.equip_weapon("Handgun")
        self.assertEqual(m.equipedWeapon["tag"], "ranged")
        with mock.patch("battleFunctions.random.randint", return_value=6), quiet():
            simulate_attack(m, model("State Trooper", ""))
        # ranged AP 1 + Armour Bane 1 on the natural 6
        self.assertEqual(m.attack_AP, 2)


class BlunderbussTests(unittest.TestCase):
    def test_parsed_flags(self):
        w = get_catalogue().weapon("Blunderbuss")
        self.assertEqual(w.get("ranged_shots_dice"), "D3")
        self.assertTrue(w.get("volley_fire"))
        ignore = w.get("ignore_to_hit_penalties", [])
        self.assertIn("multiple_shots", ignore)
        self.assertIn("long_range", ignore)
        self.assertIn("stand_and_shoot", ignore)

    def test_ignores_multiple_shots_penalty(self):
        import toHitAndToWound
        m = model("State Trooper", "")
        m.weapons["Blunderbuss"] = get_catalogue().weapon("Blunderbuss")
        m.equip_weapon("Blunderbuss")
        m.characteristics["BS"] = "3"; m.attack_roll = 4  # BS3 hits on 4+
        # Blunderbuss ignores the -1 for multiple shots -> a 4 still hits.
        self.assertTrue(toHitAndToWound.to_hit_ranged(m, multiple_shots=True))

    def test_normal_gun_suffers_penalty(self):
        import toHitAndToWound
        m = model("State Trooper", "")
        m.weapons["Gun"] = {"name": "Gun", "tag": "ranged", "ranged_strength": 4,
                            "ranged_AP": 0, "ranged_shots": 2}
        m.equip_weapon("Gun")
        m.characteristics["BS"] = "3"; m.attack_roll = 4
        self.assertFalse(toHitAndToWound.to_hit_ranged(m, multiple_shots=True))


class LongRangeTests(unittest.TestCase):
    def test_long_range_penalty(self):
        import toHitAndToWound
        m = model("State Trooper", "")
        m.weapons["Gun"] = {"name": "Gun", "tag": "ranged", "ranged_strength": 4,
                            "ranged_AP": 0, "ranged_shots": 1}
        m.equip_weapon("Gun")
        m.characteristics["BS"] = "3"; m.attack_roll = 4  # BS3 hits on 4+
        self.assertTrue(toHitAndToWound.to_hit_ranged(m, long_range=False))
        # Long range -> -1, so BS3 needs 5+ and a 4 misses.
        self.assertFalse(toHitAndToWound.to_hit_ranged(m, long_range=True))

    def test_blunderbuss_ignores_long_range(self):
        import toHitAndToWound
        m = model("State Trooper", "")
        m.weapons["Blunderbuss"] = get_catalogue().weapon("Blunderbuss")
        m.equip_weapon("Blunderbuss")
        m.characteristics["BS"] = "3"; m.attack_roll = 4
        self.assertTrue(toHitAndToWound.to_hit_ranged(m, long_range=True))

    def test_simulate_attack_passes_long_range_flag(self):
        import battleFunctions
        m = model("State Missile Trooper", ""); m.give_weapon("Handgun")
        m.equip_weapon("Handgun")
        m.at_long_range = True
        captured = {}

        def fake(model1, **kw):
            captured.update(kw)
            return True

        with mock.patch.object(battleFunctions, "to_hit_ranged", fake), quiet():
            battleFunctions.simulate_attack(m, model("State Trooper", ""))
        self.assertTrue(captured.get("long_range"))


class ChargeCombatIntegrationTests(unittest.TestCase):
    def test_charge_applies_then_resets(self):
        atk = model("Demigryph Knight", ""); atk.give_weapon("Lance")
        base_s = int(atk.characteristics["S"])
        with quiet():
            simulate_battle(mk_unit(atk, 4, 4, 1), mk_unit(model("Black Orc", ""), 10, 5, 2),
                            charge=True)
        # Lance S+2/AP applied during combat, restored afterwards.
        self.assertEqual(int(atk.characteristics["S"]), base_s)
        self.assertEqual(atk.AP, 0)

    def test_defender_gets_always_on_strength(self):
        # Halberd S+1 applies even when defending (not charging).
        defender = model("State Trooper", ""); defender.give_weapon("Halberd")
        defender.equip_best_melee()
        seen = {}
        real_to_wound = simulate_battle.__globals__["to_wound"]

        def spy(m1, m2):
            seen["S"] = int(m1.characteristics["S"])
            return real_to_wound(m1, m2)

        base_s = int(defender.characteristics["S"])
        with quiet(), mock.patch.dict(simulate_battle.__globals__, {"to_wound": spy}):
            simulate_battle(mk_unit(defender, 5, 5, 1), mk_unit(model("Black Orc", ""), 10, 5, 2),
                            charge=False)
        self.assertEqual(seen.get("S"), base_s + 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
