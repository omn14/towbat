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


if __name__ == "__main__":
    unittest.main(verbosity=2)
