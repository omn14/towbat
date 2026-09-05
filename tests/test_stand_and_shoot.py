"""Stand & Shoot (p. 120) and Standing and Shooting (p. 139)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from special_rules import can_stand_and_shoot
from toHitAndToWound import ranged_hit_requirement


def _shooter(weapon, bs="3"):
    from models import model
    m = model("Handgunner", "")
    m.give_weapon(weapon)
    m.equip_weapon(weapon)
    m.characteristics["BS"] = bs
    return m


class EligibilityTests(unittest.TestCase):
    def test_a_charger_inside_its_own_movement_is_too_close(self):
        self.assertFalse(can_stand_and_shoot(3.0, 6))
        self.assertFalse(can_stand_and_shoot(5.9, 6))

    def test_exactly_the_movement_is_far_enough(self):
        # The rule bars a distance *less than* the Movement, so the boundary
        # itself may shoot.
        self.assertTrue(can_stand_and_shoot(6.0, 6))

    def test_further_away_is_always_allowed(self):
        self.assertTrue(can_stand_and_shoot(20.0, 6))

    def test_quick_shot_ignores_the_distance(self):
        self.assertTrue(can_stand_and_shoot(1.0, 10, quick_shot=True))


class ReactionToHitTests(unittest.TestCase):
    def test_the_reaction_costs_one_to_hit(self):
        m = _shooter("Handgun")
        self.assertEqual(4, ranged_hit_requirement(m)[0])
        self.assertEqual(5, ranged_hit_requirement(m, stand_and_shoot=True)[0])

    def test_long_range_adds_nothing_to_a_stand_and_shoot(self):
        # "Models making a Stand & Shoot charge reaction do not suffer any
        # additional modifiers for Firing at Long Range" (p. 139).
        m = _shooter("Handgun")
        self.assertEqual(5, ranged_hit_requirement(m, long_range=True)[0])
        self.assertEqual(
            ranged_hit_requirement(m, stand_and_shoot=True)[0],
            ranged_hit_requirement(m, stand_and_shoot=True, long_range=True)[0])

    def test_other_modifiers_still_apply(self):
        m = _shooter("Handgun")
        self.assertEqual(6, ranged_hit_requirement(m, stand_and_shoot=True,
                                                   target_skirmisher=True)[0])


class MissileWeaponTests(unittest.TestCase):
    def test_a_sheathed_bow_still_counts_as_being_armed(self):
        # A unit expecting a fight may have equipped its melee weapon, but it
        # is still "armed with missile weapons" for the reaction.
        from models import model
        m = model("Handgunner", "")
        m.give_weapon("Handgun")
        m.give_weapon("Great Weapon")
        m.equip_weapon("Great Weapon")
        self.assertEqual("Handgun", m.missile_weapon().get("name"))

    def test_a_unit_with_no_missile_weapon_has_none(self):
        from models import model
        m = model("Swordsman", "")
        m.give_weapon("Great Weapon")
        m.equip_weapon("Great Weapon")
        self.assertEqual({}, m.missile_weapon())


class CombatResultTests(unittest.TestCase):
    """Wounds from a Stand & Shoot count towards the combat that follows."""

    def _source(self, name):
        path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), name)
        return open(path, encoding='utf-8').read()

    def test_the_banked_wounds_are_added_to_the_score(self):
        # The sum happens inside an async Panda3D method, so this holds the
        # wiring rather than the arithmetic: the tally must reach both scores.
        src = self._source('combat_resolution.py')
        self.assertIn("player1_score += p1_shot", src)
        self.assertIn("player2_score += p2_shot", src)

    def test_the_result_table_itemises_them(self):
        # A combat can be won by this alone, so it cannot be folded silently
        # into 'Wounds caused'.
        self.assertIn("'Stand & Shoot': (p1_shot, p2_shot)",
                      self._source('combat_resolution.py'))

    def test_the_reaction_banks_rather_than_spending_the_shooting_allowance(self):
        # hasAttackedThisTurn doubles as "this combat has been fought", so
        # setting it would bar the unit from the fight it just shot at.
        src = self._source('game.py')
        self.assertIn("attackerUnit.standAndShootWounds += total_wounds", src)

    def test_the_ai_takes_the_reaction_when_it_can(self):
        # It costs nothing a hold would have kept, so holding instead is
        # simply worse. A pursuit is the one case that still holds, having
        # never been declared as a charge.
        src = self._source('combat_resolution.py')
        self.assertIn('crchoice = "stand & shoot" if standShootWeapon else "hold"',
                      src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
