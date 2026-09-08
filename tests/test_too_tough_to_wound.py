"""Too Tough to Wound (Rulebook p. 140; combat chart p. 149)."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest import mock

from battleFunctions import (format_combat_report, impact_hit_report,
                             resolve_impact_hits, resolve_magic_hits,
                             simulate_attack, simulate_battle,
                             take_last_combat_report, take_last_slaying_blows)
from models import model
from toHitAndToWound import to_wound


class TestToWoundChart(unittest.TestCase):
    def setUp(self):
        self.attacker = SimpleNamespace(characteristics={'S': 3})
        self.defender = SimpleNamespace(characteristics={'T': 3})

    def test_strength_shortfalls_two_through_five_still_wound_on_six(self):
        for toughness in range(5, 9):
            with self.subTest(toughness=toughness):
                self.defender.characteristics['T'] = toughness
                self.assertEqual(to_wound(self.attacker, self.defender), 6)

    def test_six_or_more_points_lower_cannot_wound(self):
        for toughness in (9, 10):
            with self.subTest(toughness=toughness):
                self.defender.characteristics['T'] = toughness
                self.assertEqual(to_wound(self.attacker, self.defender), 7)

    def test_cutoff_across_the_characteristic_range(self):
        for strength in range(1, 5):
            with self.subTest(strength=strength):
                self.attacker.characteristics['S'] = strength
                self.defender.characteristics['T'] = strength + 5
                self.assertEqual(to_wound(self.attacker, self.defender), 6)
                self.defender.characteristics['T'] = strength + 6
                self.assertEqual(to_wound(self.attacker, self.defender), 7)

    def test_other_chart_results_are_unchanged(self):
        for toughness, target in ((1, 2), (2, 3), (3, 4), (4, 5)):
            with self.subTest(toughness=toughness):
                self.defender.characteristics['T'] = toughness
                self.assertEqual(to_wound(self.attacker, self.defender), target)

    def test_explicit_attack_strength_decides_the_cutoff(self):
        self.defender.characteristics['T'] = 9
        self.assertEqual(to_wound(self.attacker, self.defender, strength=3), 7)
        self.assertEqual(to_wound(self.attacker, self.defender, strength=4), 6)

    def test_effective_defender_toughness_decides_the_cutoff(self):
        self.defender.get_toughness = lambda: 9
        self.assertEqual(to_wound(self.attacker, self.defender), 7)

    def test_absent_characteristics_remain_impossible(self):
        self.attacker.characteristics['S'] = '-'
        self.assertEqual(to_wound(self.attacker, self.defender), 7)
        self.attacker.characteristics['S'] = 3
        self.defender.characteristics['T'] = 0
        self.assertEqual(to_wound(self.attacker, self.defender), 7)


def _unit(profile):
    return SimpleNamespace(model=profile, nmodels=3, files=3, ranks=1,
                           name=profile.name)


class TestResolvedWounds(unittest.TestCase):
    def setUp(self):
        self.attacker = model('Goblin', '')
        self.defender = model('Goblin', '')
        for profile in (self.attacker, self.defender):
            profile.special_rules = []
            profile.equip_weapon('Hand Weapon')
            profile.characteristics.update({'WS': 3, 'BS': 3, 'S': 3, 'A': 3})
        self.defender.characteristics['T'] = 9
        patches = ExitStack()
        self.addCleanup(patches.close)
        self.fired = patches.enter_context(mock.patch('battleFunctions.rule_log'))
        self.skipped = patches.enter_context(mock.patch('battleFunctions.rule_skipped'))
        patches.enter_context(mock.patch('battleFunctions.random.randint', return_value=6))
        self.saves = patches.enter_context(
            mock.patch('battleFunctions.check_saves', return_value=False))

    def assert_block_logged(self, hits):
        calls = [call for call in self.fired.call_args_list
                 if call.args[0] == 'Too Tough to Wound']
        self.assertEqual(len(calls), 1)
        detail = calls[0].args[2]
        for expected in (f'{hits} hit(s)', 'S3', 'T9', '6 points lower',
                         'cannot wound', '0 wounds'):
            self.assertIn(expected, detail)

    def test_melee_sixes_cannot_wound_and_never_attempt_saves(self):
        result = simulate_battle(_unit(self.attacker), _unit(self.defender), False)
        attacks, hits, wounds, saves, unsaved = result
        self.assertGreater(hits, 1)
        self.assertEqual(attacks, hits)
        self.assertEqual((wounds, saves, unsaved), (0, 0, 0))
        self.saves.assert_not_called()
        self.assert_block_logged(hits)
        report = take_last_combat_report()
        self.assertIn('wound impossible', '\n'.join(format_combat_report(report)))

    def test_wound_bonus_cannot_bypass_impossibility_in_melee_or_shooting(self):
        self.attacker.special_rules.append({'to_wound': lambda roll, profile: roll + 2})
        for mode in ('melee', 'ranged'):
            with self.subTest(mode=mode):
                self.attacker.equipedWeapon = {'tag': mode, 'ranged_strength': 3}
                self.assertEqual(simulate_attack(self.attacker, self.defender),
                                 (True, False))
                self.assertEqual(self.attacker.wound_roll, 8)
        self.fired.assert_not_called()
        self.skipped.assert_not_called()

    def test_shooting_uses_weapon_strength_not_the_bearers(self):
        self.attacker.characteristics['S'] = 10
        self.attacker.equipedWeapon = {'name': 'Test bow', 'tag': 'ranged',
                                      'ranged_strength': 3, 'ranged_AP': 0}
        result = simulate_battle(_unit(self.attacker), _unit(self.defender), False)
        self.assertEqual(result[2:], (0, 0, 0))
        self.saves.assert_not_called()
        self.assert_block_logged(result[1])

    def test_magic_automatic_hits_do_not_automatically_wound(self):
        self.assertEqual(resolve_magic_hits(_unit(self.defender), 12, 3, 2),
                         (0, 0, 0))
        self.saves.assert_not_called()
        self.assert_block_logged(12)

    def test_impact_hits_use_unmodified_strength(self):
        self.attacker.special_rules.append({'impact_hits': '2'})
        self.attacker._base_characteristics['S'] = 3
        self.attacker.characteristics['S'] = 10
        attacker, defender = _unit(self.attacker), _unit(self.defender)
        self.assertEqual(resolve_impact_hits(attacker, defender), (6, 0, 0, 0))
        self.saves.assert_not_called()
        self.assert_block_logged(6)
        self.assertIn('wound impossible', '\n'.join(impact_hit_report(attacker, defender)))

    def test_five_point_gap_still_wounds_and_logs_why_rule_was_skipped(self):
        self.defender.characteristics['T'] = 8
        result = simulate_battle(_unit(self.attacker), _unit(self.defender), False)
        self.assertGreater(result[1], 0)
        self.assertEqual(result[1], result[2])
        calls = [call for call in self.skipped.call_args_list
                 if call.args[0] == 'Too Tough to Wound']
        self.assertEqual(len(calls), 1)
        self.assertIn('only 5 points lower', calls[0].args[2])
        self.assertIn('wounds on 6+', calls[0].args[2])

    def test_spell_strength_one_point_higher_can_wound(self):
        self.assertEqual(resolve_magic_hits(_unit(self.defender), 12, 4, 0),
                         (12, 0, 12))

    def test_no_hits_produces_no_rule_log(self):
        self.assertEqual(resolve_magic_hits(_unit(self.defender), 0, 3, 0),
                         (0, 0, 0))
        self.fired.assert_not_called()
        self.skipped.assert_not_called()

    def assert_slaying_barred(self, name, key, troop_type):
        self.attacker.special_rules.append({'name': name, key: True})
        self.defender.characteristics['Troop Type'] = troop_type
        result = simulate_battle(_unit(self.attacker), _unit(self.defender), False)
        self.assertEqual(result[2:], (0, 0, 0))
        self.assertEqual(take_last_slaying_blows(), 0)
        calls = [call for call in self.skipped.call_args_list
                 if call.args[0] == name]
        self.assertEqual(len(calls), 1)
        self.assertIn('cannot be wounded', calls[0].args[2])

    def test_no_killing_blow_on_an_unwoundable_target(self):
        self.assert_slaying_barred('Killing Blow', 'killing_blow', 'Regular infantry')

    def test_no_monster_slayer_on_an_unwoundable_target(self):
        self.assert_slaying_barred('Monster Slayer', 'monster_slayer', 'Behemoth')


if __name__ == '__main__':
    unittest.main()