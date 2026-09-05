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
        self.assertIn('crchoice = "stand & shoot" if shootOption else "hold"',
                      src)


class FireAndFleeTests(unittest.TestCase):
    """Fire & Flee (p. 169): Stand & Shoot, then flee on a shortened roll."""

    def test_the_volley_costs_the_unit_ground(self):
        # Easy to get backwards: an ordinary Flee roll *sums* 2D6, so keeping
        # one die is the shorter run, not the longer one.
        from post_combat import flee_roll, fire_and_flee_roll
        self.assertEqual(7, flee_roll([2, 5]))
        self.assertEqual(5, fire_and_flee_roll([2, 5]))
        self.assertEqual(12, flee_roll([6, 6]))
        self.assertEqual(6, fire_and_flee_roll([6, 6]))

    def test_swiftstride_adds_its_die_rather_than_being_discarded(self):
        from post_combat import fire_and_flee_roll
        self.assertEqual(9, fire_and_flee_roll([3, 5, 4]))

    def test_a_second_flee_in_a_phase_covers_nothing(self):
        # The Limits of Endurance (p. 133) applies to this flee too.
        from post_combat import fire_and_flee_roll
        self.assertEqual(0, fire_and_flee_roll([6, 6], True))

    def test_the_roster_rule_reaches_the_model(self):
        from special_rules import build_special_rules
        from models import model
        m = model("Gyrocopter", "")
        m.characteristics["Special Rules"] = ['Fire & Flee']
        m.special_rules = build_special_rules(m)
        self.assertTrue(m.has_fire_and_flee())

    def test_a_unit_without_the_rule_does_not_have_it(self):
        from models import model
        m = model("Handgunner", "")
        self.assertFalse(m.has_fire_and_flee())

    def test_the_charge_reaction_flee_spends_its_allowance(self):
        # Both bugs this rule exposed were in that one line: a bool added to
        # the distance, and a flee that never consulted fledThisPhase.
        path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'combat_resolution.py')
        src = open(path, encoding='utf-8').read()
        self.assertNotIn('sum(fldice) + fleeBonus', src)
        self.assertIn('fleeingUnit.fledThisPhase = True', src)


class DeclarationPositionTests(unittest.TestCase):
    """The reaction is judged from where the charge was declared (p. 120).

    Found in a game log: every reaction was refused with "is 0.0" away",
    because `chargeAndChargeReaction` runs *after* the charger has been swept
    into contact, so its own transform always reads zero.
    """

    def _resolver(self, charger_movement=4):
        from types import SimpleNamespace
        from combat_resolution import CombatResolver
        from models import model

        def unit(name, weapon=None, x=0.0, y=0.0):
            m = model(name, "")
            if weapon:
                m.give_weapon(weapon)
                m.equip_weapon(weapon)
            m.characteristics['M'] = str(charger_movement)
            body = SimpleNamespace(getPos=lambda x=x, y=y: SimpleNamespace(x=x, y=y))
            return SimpleNamespace(unit=SimpleNamespace(name=name, model=m),
                                   state="Idle", bodyNP=body,
                                   unitWidth=2.0, unitHeight=2.0)

        boxes = {}

        class Psy:
            @staticmethod
            def _unit_box(u):
                return boxes[id(u)]

        game = SimpleNamespace(
            psychology=Psy(),
            terrain_manager=SimpleNamespace(los_block_point=lambda a, b: None))
        return CombatResolver(game), unit, boxes

    def test_the_contact_position_would_refuse_every_reaction(self):
        res, unit, boxes = self._resolver()
        d, ch = unit("Handgunners", "Handgun"), unit("Grave Guard")
        boxes[id(d)] = (0.0, 0.0, 1.0, 1.0, 0.0)
        boxes[id(ch)] = (0.0, 2.0, 1.0, 1.0, 0.0)   # nose to nose, as after the move
        self.assertIsNone(res.standAndShootOption(d, ch))

    def test_measured_from_where_the_charge_was_declared(self):
        from types import SimpleNamespace
        res, unit, boxes = self._resolver()
        d, ch = unit("Handgunners", "Handgun"), unit("Grave Guard")
        boxes[id(d)] = (0.0, 0.0, 1.0, 1.0, 0.0)
        boxes[id(ch)] = (0.0, 2.0, 1.0, 1.0, 0.0)
        declared = SimpleNamespace(x=0.0, y=9.0)     # 7" of clear ground away
        opt = res.standAndShootOption(d, ch, declared)
        self.assertIsNotNone(opt)
        self.assertAlmostEqual(7.0, opt.distance, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
