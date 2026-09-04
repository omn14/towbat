"""Moving and Shooting (p. 139) and Quick Shot (p. 175)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from battlescribe import has_quick_shot, get_catalogue
from toHitAndToWound import ranged_hit_requirement
from battleFunctions import _ranged_tohit_report


def _shooter(weapon, bs="3"):
    from models import model
    m = model("Gyrocopter", "")
    m.give_weapon(weapon)
    m.equip_weapon(weapon)
    m.characteristics["BS"] = bs
    return m


class QuickShotNameTests(unittest.TestCase):
    def test_both_catalogue_spellings_are_recognised(self):
        # The catalogue carries 32 'Quick Shot' and one 'Quick Shoot'; a match
        # on the exact printed name would silently miss the odd one out.
        self.assertTrue(has_quick_shot(["Quick Shot"]))
        self.assertTrue(has_quick_shot(["Quick Shoot"]))

    def test_an_unrelated_rule_is_not_quick_shot(self):
        self.assertFalse(has_quick_shot(["Multiple Shots (2)"]))
        self.assertFalse(has_quick_shot(["Move & Shoot"]))
        self.assertFalse(has_quick_shot([]))
        self.assertFalse(has_quick_shot(None))

    def test_every_quick_shot_weapon_in_the_catalogue_is_flagged(self):
        missed = []
        for w in get_catalogue().weapons_by_slug.values():
            names = [str(r.get('name') if isinstance(r, dict) else r)
                     for r in (w.get('special_rules') or [])]
            if has_quick_shot(names) and not w.get('quick_shot'):
                missed.append(w.get('name'))
        self.assertEqual([], missed)


class MovingAndShootingTests(unittest.TestCase):
    def test_moving_costs_one_to_hit(self):
        m = _shooter("Handgun")
        self.assertEqual(4, ranged_hit_requirement(m)[0])
        self.assertEqual(5, ranged_hit_requirement(m, moved=True)[0])

    def test_it_stacks_with_the_other_modifiers(self):
        m = _shooter("Handgun")
        self.assertEqual(6, ranged_hit_requirement(m, moved=True,
                                                   long_range=True)[0])


class QuickShotWaivesTheMovePenaltyTests(unittest.TestCase):
    def test_a_quick_shot_weapon_hits_the_same_having_moved(self):
        m = _shooter("Clattergun")
        self.assertTrue(m.equipedWeapon.get('quick_shot'))
        self.assertEqual(ranged_hit_requirement(m)[0],
                         ranged_hit_requirement(m, moved=True)[0])

    def test_it_waives_only_the_move_penalty(self):
        # Quick Shot says nothing about range or cover, so those still bite.
        m = _shooter("Clattergun")
        self.assertEqual(ranged_hit_requirement(m)[0] + 1,
                         ranged_hit_requirement(m, moved=True,
                                                long_range=True)[0])

    def test_a_save_written_before_the_flag_existed_still_works(self):
        # Same trap as Move & Shoot: persistence stores the weapon dict, and a
        # quicksave taken before quick_shot was parsed carries only the names.
        from models import model
        m = model("Gyrocopter", "")
        m.weapons['Clattergun'] = {
            'name': 'Clattergun', 'tag': 'ranged', 'ranged_strength': 4,
            'ranged_AP': -1, 'special_rules': ['Quick Shot'],
        }
        m.equip_weapon('Clattergun')
        m.characteristics["BS"] = "3"
        self.assertNotIn('quick_shot', m.equipedWeapon)
        self.assertEqual(ranged_hit_requirement(m)[0],
                         ranged_hit_requirement(m, moved=True)[0])


class ShotReportTests(unittest.TestCase):
    def test_the_report_agrees_with_the_dice(self):
        # The report used to keep its own copy of the To Hit ladder; if it
        # drifts again the numbers in the log stop describing the rolls.
        for weapon in ("Handgun", "Clattergun"):
            for bs in ("3", "6"):
                m = _shooter(weapon, bs)
                m.moved_this_turn = True
                m.at_long_range = True
                target, _mods = _ranged_tohit_report(m)
                self.assertEqual(
                    ranged_hit_requirement(m, moved=True, long_range=True)[0],
                    target, f"{weapon} BS{bs}")

    def test_a_waived_modifier_is_reported_as_waived(self):
        m = _shooter("Clattergun")
        m.moved_this_turn = True
        _target, mods = _ranged_tohit_report(m)
        self.assertIn('moved (waived)', mods)

    def test_an_applied_modifier_is_reported_as_applied(self):
        m = _shooter("Handgun")
        m.moved_this_turn = True
        _target, mods = _ranged_tohit_report(m)
        self.assertIn('moved -1', mods)


if __name__ == "__main__":
    unittest.main(verbosity=2)
