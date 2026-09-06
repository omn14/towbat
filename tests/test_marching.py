"""Marching (Rulebook p. 123).

A unit may double its Movement to march, but a unit that marched cannot shoot
that turn, nor cast a Magic Missile or Magical Vortex.

Only the arithmetic and the spell categories are covered here. Deciding the
band from the cursor, tinting the overlay and setting the flag on the click all
live inside Panda3D-bound methods; see the checklist for what that leaves
unverified.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from movement_system import is_march, MARCH_BARRED_SPELLS  # noqa: E402


class IsMarchTests(unittest.TestCase):
    def test_within_movement_is_an_ordinary_move(self):
        self.assertFalse(is_march(3.0, 4))

    def test_exactly_movement_is_not_a_march(self):
        # The first M is free; a march is what goes *beyond* it.
        self.assertFalse(is_march(4.0, 4))

    def test_a_hair_beyond_movement_is_a_march(self):
        self.assertTrue(is_march(4.01, 4))

    def test_double_movement_is_the_far_end_of_the_march(self):
        self.assertTrue(is_march(8.0, 4))

    def test_a_spent_manoeuvre_brings_the_march_forward(self):
        # Half the allowance already spent on a redress, so an ordinary move
        # now runs out at 2", not 4".
        self.assertFalse(is_march(2.0, 4, spent=2.0))
        self.assertTrue(is_march(2.5, 4, spent=2.0))

    def test_a_fully_spent_allowance_makes_any_move_a_march(self):
        self.assertTrue(is_march(0.5, 4, spent=4.0))


class BarredSpellTests(unittest.TestCase):
    """The barred set has to match what the catalogue actually writes."""

    def test_the_categories_exist_in_the_catalogue(self):
        from battlescribe import get_catalogue
        spells = get_catalogue().spells_by_slug
        types = {s.get('type') for s in spells.values()}
        for barred in MARCH_BARRED_SPELLS:
            self.assertIn(barred, types, f"no spell is typed {barred!r}")

    def test_an_offensive_spell_is_barred_and_a_buff_is_not(self):
        from battlescribe import get_catalogue
        spells = get_catalogue().spells_by_slug
        by_name = {s.get('name'): s for s in spells.values()}
        fireball = by_name.get('Fireball')
        oaken = by_name.get('Oaken Shield')
        self.assertIsNotNone(fireball, "Fireball missing from the catalogue")
        self.assertIsNotNone(oaken, "Oaken Shield missing from the catalogue")
        self.assertIn(fireball.get('type'), MARCH_BARRED_SPELLS)
        self.assertNotIn(oaken.get('type'), MARCH_BARRED_SPELLS)


class MoveAndShootTests(unittest.TestCase):
    """Move & Shoot (p. 174) — fires even after a march."""

    def _weapons(self):
        from battlescribe import get_catalogue
        return get_catalogue().weapons_by_slug.values()

    def _rule_names(self, weapon):
        return [str(r.get('name') if isinstance(r, dict) else r)
                for r in (weapon.get('special_rules') or [])]

    def test_move_or_shoot_is_not_mistaken_for_it(self):
        # The two rules are opposites and differ by a single word, so a match
        # on "move ... shoot" would flag 25 weapons that must not fire.
        wrong = [w.get('name') for w in self._weapons()
                 if w.get('move_and_shoot')
                 and any('or shoot' in n.lower() for n in self._rule_names(w))]
        self.assertEqual(wrong, [], "Move or Shoot weapons flagged as Move & Shoot")

    def test_every_move_and_shoot_weapon_is_flagged(self):
        missed = [w.get('name') for w in self._weapons()
                  if not w.get('move_and_shoot')
                  and any('& shoot' in n.lower() or 'and shoot' in n.lower()
                          for n in self._rule_names(w))]
        self.assertEqual(missed, [], "Move & Shoot weapons left unflagged")

    def test_the_rule_reaches_an_equipped_model(self):
        from models import model
        m = model("Gyrocopter", "")
        m.give_weapon("Clattergun"); m.equip_weapon("Clattergun")
        self.assertTrue(m.fires_after_marching())

    def test_an_ordinary_weapon_still_cannot_fire_after_a_march(self):
        from models import model
        m = model("State Missile Trooper", "")
        m.give_weapon("Handgun"); m.equip_weapon("Handgun")
        self.assertFalse(m.fires_after_marching())

    def test_a_save_written_before_the_flag_existed_still_works(self):
        # persistence stores the weapon dict, so a quicksave taken before
        # move_and_shoot was parsed has no such key — but it does keep the
        # rule names, which is enough to answer the question.
        from models import model
        m = model("Gyrocopter", "")
        m.weapons['Clattergun'] = {
            'name': 'Clattergun', 'tag': 'ranged', 'ranged_strength': 4,
            'ranged_AP': -1, 'special_rules': ['Armour Bane (1)', 'Move & Shoot'],
        }
        m.equip_weapon('Clattergun')
        self.assertNotIn('move_and_shoot', m.equipedWeapon)
        self.assertTrue(m.fires_after_marching())

    def test_the_fallback_does_not_free_a_move_or_shoot_weapon(self):
        from models import model
        m = model("Gyrocopter", "")
        m.weapons['Bombard'] = {
            'name': 'Bombard', 'tag': 'ranged', 'ranged_strength': 4,
            'ranged_AP': 0, 'special_rules': ['Move or Shoot'],
        }
        m.equip_weapon('Bombard')
        self.assertFalse(m.fires_after_marching())


class MoveOrShootTests(unittest.TestCase):
    """Move or Shoot (p. 174) — artillery cannot fire on the move."""

    def _weapons(self):
        from battlescribe import get_catalogue
        return get_catalogue().weapons_by_slug.values()

    def _rule_names(self, weapon):
        return [str(r.get('name') if isinstance(r, dict) else r)
                for r in (weapon.get('special_rules') or [])]

    def test_every_move_or_shoot_weapon_is_flagged(self):
        missed = [w.get('name') for w in self._weapons()
                  if not w.get('move_or_shoot')
                  and any('or shoot' in n.lower() for n in self._rule_names(w))]
        self.assertEqual(missed, [], "Move or Shoot weapons left unflagged")

    def test_move_and_shoot_is_not_mistaken_for_it(self):
        # The mirror of the trap above: matching "move ... shoot" here would
        # silence 16 weapons written to fire on the move.
        wrong = [w.get('name') for w in self._weapons()
                 if w.get('move_or_shoot')
                 and any('& shoot' in n.lower() or 'and shoot' in n.lower()
                         for n in self._rule_names(w))]
        self.assertEqual(wrong, [], "Move & Shoot weapons flagged as Move or Shoot")

    def test_the_spelling_with_a_capital_or_is_caught(self):
        from battlescribe import has_move_or_shoot
        for raw in ("Move or Shoot", "Move Or Shoot", "move or shoot"):
            self.assertTrue(has_move_or_shoot([raw]), raw)

    def test_move_and_shoot_does_not_match(self):
        from battlescribe import has_move_or_shoot
        for raw in ("Move & Shoot", "Move and Shoot"):
            self.assertFalse(has_move_or_shoot([raw]), raw)

    def test_the_rule_reaches_an_equipped_model(self):
        from models import model
        m = model("Great Cannon", "")
        m.give_weapon("Cannon"); m.equip_weapon("Cannon")
        self.assertTrue(m.cannot_shoot_after_moving())

    def test_an_ordinary_weapon_is_free_to_move(self):
        from models import model
        m = model("State Missile Trooper", "")
        m.give_weapon("Handgun"); m.equip_weapon("Handgun")
        self.assertFalse(m.cannot_shoot_after_moving())

    def test_a_save_written_before_the_flag_existed_still_works(self):
        from models import model
        m = model("Great Cannon", "")
        m.weapons['Cannon'] = {
            'name': 'Cannon', 'tag': 'ranged', 'ranged_strength': 10,
            'ranged_AP': -3, 'special_rules': ['Move or Shoot'],
        }
        m.equip_weapon('Cannon')
        self.assertNotIn('move_or_shoot', m.equipedWeapon)
        self.assertTrue(m.cannot_shoot_after_moving())

    def test_a_model_with_no_weapon_is_not_barred(self):
        from models import model
        m = model("State Missile Trooper", "")
        m.equipedWeapon = None
        self.assertFalse(m.cannot_shoot_after_moving())


if __name__ == "__main__":
    unittest.main(verbosity=2)
