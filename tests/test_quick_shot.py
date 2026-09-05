"""Moving and Shooting (p. 139) and Quick Shot (p. 175)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from battlescribe import has_quick_shot, get_catalogue
from toHitAndToWound import ranged_hit_requirement
from battleFunctions import _ranged_tohit_report, _tohit_summary


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
        # An Asrai Longbow is plain: neither Ponderous nor Quick Shot. This
        # asserted -1 against a Handgun at first, which is Ponderous and takes
        # -2 — the test was describing a rule its weapon does not have.
        m = _shooter("Asrai Longbow")
        self.assertEqual(4, ranged_hit_requirement(m)[0])
        self.assertEqual(5, ranged_hit_requirement(m, moved=True)[0])

    def test_it_stacks_with_the_other_modifiers(self):
        m = _shooter("Asrai Longbow")
        self.assertEqual(6, ranged_hit_requirement(m, moved=True,
                                                   long_range=True)[0])


class PonderousTests(unittest.TestCase):
    def test_a_ponderous_weapon_costs_two(self):
        m = _shooter("Handgun")
        self.assertTrue(m.equipedWeapon.get('ponderous'))
        self.assertEqual(4, ranged_hit_requirement(m)[0])
        self.assertEqual(6, ranged_hit_requirement(m, moved=True)[0])

    def test_it_costs_nothing_standing_still(self):
        m = _shooter("Handgun")
        self.assertEqual(4, ranged_hit_requirement(m, moved=False)[0])

    def test_ponderous_and_quick_shot_cancel_to_the_usual_minus_one(self):
        # Naptha bombs carry both, so this is a real weapon rather than a
        # hypothetical: the FAQ has the two "effectively cancel one another
        # out, meaning the weapon would suffer a -1 To Hit modifier".
        m = _shooter("Naptha bombs")
        w = m.equipedWeapon
        self.assertTrue(w.get('ponderous'))
        self.assertTrue(w.get('quick_shot'))
        self.assertEqual(ranged_hit_requirement(m)[0] + 1,
                         ranged_hit_requirement(m, moved=True)[0])

    def test_a_save_written_before_the_flag_existed_still_works(self):
        from models import model
        m = model("Handgunner", "")
        m.weapons['Handgun'] = {
            'name': 'Handgun', 'tag': 'ranged', 'ranged_strength': 4,
            'ranged_AP': -1, 'special_rules': ['Armour Bane (1)', 'Ponderous'],
        }
        m.equip_weapon('Handgun')
        m.characteristics["BS"] = "3"
        self.assertNotIn('ponderous', m.equipedWeapon)
        self.assertEqual(6, ranged_hit_requirement(m, moved=True)[0])


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
                rep = _ranged_tohit_report(m)
                self.assertEqual(
                    ranged_hit_requirement(m, moved=True, long_range=True)[0],
                    rep['target'], f"{weapon} BS{bs}")

    def test_a_waived_modifier_says_which_rule_waived_it(self):
        m = _shooter("Clattergun")
        m.moved_this_turn = True
        self.assertIn('moved waived (Quick Shot)',
                      _ranged_tohit_report(m)['mods'])

    def test_an_applied_modifier_is_reported_as_applied(self):
        m = _shooter("Asrai Longbow")
        m.moved_this_turn = True
        self.assertIn('moved -1', _ranged_tohit_report(m)['mods'])

    def test_a_ponderous_penalty_is_reported_as_minus_two_and_named(self):
        # A game log showed `-1 To Hit (5+ -> 7+)`, which is a two-point change
        # described as one, and never named Ponderous as the cause.
        m = _shooter("Handgun")
        m.moved_this_turn = True
        self.assertIn('moved -2 (Ponderous)', _ranged_tohit_report(m)['mods'])

    def test_the_summary_shows_the_whole_sum(self):
        m = _shooter("Handgun")
        m.moved_this_turn = True
        m.at_long_range = True
        line = _tohit_summary(_ranged_tohit_report(m))
        self.assertIn('BS3 4+', line)
        self.assertIn('moved -2 (Ponderous)', line)
        self.assertIn('long range -1', line)
        self.assertIn('=  7+', line)
        self.assertIn('natural 6, then 4+', line)


if __name__ == "__main__":
    unittest.main(verbosity=2)
