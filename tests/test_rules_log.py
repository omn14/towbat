"""The rule log — every special rule that fires has to say so."""

import contextlib
import io
import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rules_log import PREFIX, rule_log, rule_skipped, subject_name  # noqa: E402


def _capture(fn, *args):
    with contextlib.redirect_stdout(io.StringIO()) as out:
        fn(*args)
    return out.getvalue().strip()


class TestNamingTheSubject(unittest.TestCase):
    """A rule can fire on a unit wrapper, a Unit or a bare model."""

    def test_a_unit_wrapper(self):
        wrapper = SimpleNamespace(unit=SimpleNamespace(name='Goblin Mob'))
        self.assertEqual(subject_name(wrapper), 'Goblin Mob')

    def test_a_unit(self):
        self.assertEqual(subject_name(SimpleNamespace(name='Goblin Mob')),
                         'Goblin Mob')

    def test_a_graphics_node_without_a_unit_name(self):
        self.assertEqual(subject_name(SimpleNamespace(unitName='Spearmen')),
                         'Spearmen')

    def test_a_plain_string(self):
        self.assertEqual(subject_name('War Wagon'), 'War Wagon')

    def test_nothing(self):
        self.assertEqual(subject_name(None), '-')

    def test_an_empty_name_falls_through(self):
        odd = SimpleNamespace(unit=SimpleNamespace(name=''), unitName='Spearmen')
        self.assertEqual(subject_name(odd), 'Spearmen')


class TestTheLogLine(unittest.TestCase):

    def setUp(self):
        self.unit = SimpleNamespace(unit=SimpleNamespace(name='War Wagon'))

    def test_it_names_the_rule_the_unit_and_what_changed(self):
        line = _capture(rule_log, 'Impact Hits (D6+1)', self.unit,
                        'charged 7", 3 models in contact -> 11 hits')
        self.assertIn('Impact Hits (D6+1)', line)
        self.assertIn('War Wagon', line)
        self.assertIn('11 hits', line)

    def test_every_line_carries_the_prefix(self):
        # One prefix means the whole rules trace is greppable.
        for fn in (rule_log, rule_skipped):
            self.assertTrue(_capture(fn, 'Parry', self.unit, 'x').startswith(PREFIX))

    def test_a_rule_that_declines_says_why(self):
        line = _capture(rule_skipped, 'Vantage Point', self.unit,
                        'only 4/10 models are on the hill')
        self.assertIn('Vantage Point', line)
        self.assertIn('not claimed', line)
        self.assertIn('4/10', line)

    def test_one_line_per_call(self):
        self.assertEqual(len(_capture(rule_log, 'Parry', self.unit,
                                      'armour 5+ -> 4+').splitlines()), 1)


class TestBattleJournal(unittest.TestCase):
    def test_armour_rolls_reach_battle_history_for_melee_and_shooting(self):
        from unittest.mock import patch
        from battleFunctions import simulate_battle
        from combat_resolution import CombatResolver
        from hud import HUD
        from models import model
        from rules_log import BattleJournal

        for mode in ('melee', 'ranged'):
            with self.subTest(mode=mode):
                attacker, defender = model('Goblin', ''), model('Goblin', '')
                for profile in (attacker, defender):
                    profile.special_rules = []
                    profile.characteristics.update({'WS': 3, 'BS': 3, 'S': 3, 'T': 3, 'A': 1})
                attacker.equipedWeapon = {'name': 'Test weapon', 'tag': mode,
                                         'ranged_strength': 3, 'ranged_AP': 0}
                attacker.AP = 0
                defender.armor_save = 5
                attacker_unit = SimpleNamespace(model=attacker, nmodels=2, files=2, ranks=1, name='Attackers')
                defender_unit = SimpleNamespace(model=defender, nmodels=2, files=2, ranks=1, name='Defenders')
                journal = BattleJournal()
                with patch('battleFunctions.random.randint', side_effect=[4, 4, 5, 4, 4, 2]) as dice, \
                        patch('battleFunctions.melee_attacks', return_value=2), \
                        patch.object(defender, 'melee_armour_save', return_value=5), \
                        patch.object(defender, 'effective_armour_save', return_value=5), \
                        patch('combat_resolution.battle_log', side_effect=journal.append):
                    results = simulate_battle(attacker_unit, defender_unit, False)
                    CombatResolver.printBattleResults(SimpleNamespace(),
                        SimpleNamespace(unit=attacker_unit), SimpleNamespace(unit=defender_unit), *results)
                self.assertEqual(results, (2, 2, 2, 1, 1))
                self.assertEqual(dice.call_count, 6)
                entry = journal.visible()[0]
                self.assertIn('Armour rolls (after modifiers): [5, 2]', entry.details)
                self.assertIn('2 wound(s) at 5+', entry.details)
                self.assertIn('Armour rolls (after modifiers): [5, 2]', journal.export())
                self.assertIn('Armour rolls (after modifiers): [5, 2]',
                              HUD.log_text(journal.visible(), details=True))
                self.assertNotIn('Armour rolls', HUD.log_text(journal.visible(), details=False))

    def test_automatic_hit_armour_rolls_are_reported_once_per_batch(self):
        from unittest.mock import patch
        from battleFunctions import resolve_impact_hits, resolve_magic_hits
        from models import model
        from rules_log import BattleJournal

        for source in ('Spell hits', 'Impact Hits'):
            with self.subTest(source=source):
                defender = model('Goblin', '')
                defender.special_rules = []
                defender.characteristics['T'] = 3
                target = SimpleNamespace(model=defender, name='Defenders')
                charger = model('Goblin', '')
                charger.special_rules = [{'name': 'Impact Hits (2)', 'impact_hits': '2'}]
                charger._base_characteristics['S'] = 3
                attacker = SimpleNamespace(model=charger, name='Attacker', nmodels=1, files=1)
                journal = BattleJournal()
                with patch('battleFunctions.random.randint', side_effect=[4, 4, 5, 2]) as dice, \
                        patch.object(defender, 'melee_armour_save', return_value=5), \
                        patch.object(charger, 'impact_hit_ap', return_value=0), \
                        patch('battleFunctions.battle_log', side_effect=journal.append):
                    if source == 'Spell hits':
                        result = resolve_magic_hits(target, 2, 3, 0)
                    else:
                        result = resolve_impact_hits(attacker, target)[1:]
                self.assertEqual(result, (2, 1, 1))
                self.assertEqual(dice.call_count, 4)
                self.assertEqual(len(journal.visible()), 1)
                entry = journal.visible()[0]
                self.assertIn(source, entry.text)
                self.assertEqual(entry.subject, 'Defenders')
                self.assertIn('Armour rolls (after modifiers): [5, 2]', entry.details)

    def test_war_machine_armour_rolls_are_reported_once_per_target(self):
        from types import MethodType
        from unittest.mock import Mock, patch
        from bombardment import Bombardment
        from cannon_fire import CannonFire
        from models import model
        from rules_log import BattleJournal

        for source in ('Cannon fire', 'Bombardment'):
            with self.subTest(source=source):
                defender = model('Goblin', '')
                defender.special_rules = []
                defender.characteristics['T'] = 3
                defender.armor_save = 5
                children = [Mock(), Mock()]
                target = Mock(unit=SimpleNamespace(model=defender, name='Defenders'))
                target.model.getChildren.return_value = children
                resolver = SimpleNamespace(game=Mock())
                resolver._models_under_template = Mock(return_value=[
                    (target, children[0], 0), (target, children[1], 1)])
                resolver._wound_unsaved = MethodType(Bombardment._wound_unsaved, resolver)
                journal = BattleJournal()
                with patch('battleFunctions.random.randint', side_effect=[4, 5, 4, 2]) as dice, \
                        patch('battleFunctions.battle_log', side_effect=journal.append):
                    if source == 'Cannon fire':
                        self.assertEqual(CannonFire._apply_wounds(resolver, target, 2, 3, 0), (1, 2, 1))
                    else:
                        Bombardment._resolve_damage(resolver, SimpleNamespace(), {}, None, 1)
                        resolver.game.removeModelsFromUnit.assert_called_once_with(target, 1)
                self.assertEqual(dice.call_count, 4)
                self.assertEqual(len(journal.visible()), 1)
                entry = journal.visible()[0]
                self.assertIn(source, entry.text)
                self.assertEqual(entry.subject, 'Defenders')
                self.assertIn('Armour rolls (after modifiers): [5, 2]', entry.details)

    def test_armour_roll_details_exclude_other_saves_and_armour_bypasses(self):
        from unittest.mock import patch
        from battleFunctions import check_saves
        from models import model

        cases = [
            ('armour', 5, 0, {}, [5], [5], [], True),
            ('ward', 5, 0, {}, [1, 5], [1], [5], True),
            ('regeneration', 5, 0, {}, [1, 1, 5], [1], [1], True),
            ('slaying', 5, 0, {'slaying_blow': True}, [1], [], [1], False),
            ('prohibited', 5, 0, {'allow_armour': False}, [5], [], [5], True),
            ('unarmoured', 7, 0, {}, [6, 5], [], [5], True),
            ('AP removes armour', 5, 2, {}, [6, 5], [], [5], True),
            ('modified armour roll', 5, 0, {}, [4], [5], [], True),
        ]
        for name, save, penetration, options, dice, expected_armour, expected_ward, expected_saved in cases:
            with self.subTest(name=name):
                defender = model('Goblin', '')
                defender.special_rules = [{'name': 'Ward', 'ward': 4}, {'name': 'Regeneration', 'regen': 4}]
                if name == 'modified armour roll':
                    defender.special_rules.append({'to_save': lambda rolled: rolled + 1})
                armour_rolls, ward_rolls = [], []
                with patch('battleFunctions.random.randint', side_effect=dice) as rolled:
                    saved = check_saves(defender, save, penetration, armour_rolls=armour_rolls,
                                        ward_rolls=ward_rolls, **options)
                self.assertEqual(saved, expected_saved)
                self.assertEqual(armour_rolls, expected_armour)
                self.assertEqual(ward_rolls, expected_ward)
                self.assertEqual(rolled.call_count, len(dice))

    def test_round_and_turn_headings_repeat_only_at_their_boundaries(self):
        from dataclasses import replace
        from rules_log import BattleJournal, LogEntry
        first = LogEntry(1, 'combat', 'First event', '', '',
                         dict(round=2, player=1, phase='MovementPhase'))
        phase = replace(first, sequence=2, context=dict(first.context, phase='CombatPhase'))
        turn = replace(first, sequence=3, context=dict(first.context, player=2))
        next_round = replace(turn, sequence=4, context=dict(turn.context, round=3))
        self.assertEqual(first.headings_since(), [
            ('round', 'Round 2'), ('turn', 'Player 1 Turn'), ('phase', 'Movement')])
        self.assertEqual(first.headings_since(first), [])
        self.assertEqual(phase.headings_since(first), [('phase', 'Combat')])
        self.assertEqual(turn.headings_since(phase), [
            ('turn', 'Player 2 Turn'), ('phase', 'Movement')])
        self.assertEqual(next_round.headings_since(turn), [
            ('round', 'Round 3'), ('turn', 'Player 2 Turn'), ('phase', 'Movement')])
        self.assertEqual(phase.headings_since(), [
            ('round', 'Round 2'), ('turn', 'Player 1 Turn'), ('phase', 'Combat')])
        journal = BattleJournal()
        journal.entries.extend([first, phase, turn, next_round])
        text = journal.export()
        self.assertEqual(text.count('Round 2'), 1)
        self.assertEqual(text.count('Player 1 Turn'), 1)
        self.assertEqual(text.count('Player 2 Turn'), 2)
        self.assertIn('Round 2\n\nPlayer 1 Turn\nMovement', text)
        self.assertNotIn('Round 2 / Player', text)

    def test_combat_heading_survives_initiative_changes_and_keeps_every_event(self):
        from dataclasses import replace
        from rules_log import BattleJournal, LogEntry
        first = LogEntry(1, 'combat', 'Combat: Dragon Princes, Chaos Knights, Silver Helms', '', '',
                         dict(round=4, player=1, phase='CombatPhase',
                              combat='Combat 1: Dragon Princes vs Chaos Knights vs Silver Helms'))
        riders = replace(first, sequence=2, text='Riders: 2 attacks -> 1 hit -> 0 wounds',
                         details='Hit rolls [3, 1]', context=dict(first.context, initiative=5))
        mounts = replace(riders, sequence=3, text='Mounts: 1 attack -> 1 hit -> 0 wounds',
                         details='Hit rolls [6]', context=dict(first.context, initiative=4))
        result = replace(first, sequence=4, text='Combat result: P1 0 - P2 0. Draw.')
        self.assertEqual(riders.headings_since(first), [('initiative', 'Initiative 5')])
        self.assertEqual(mounts.headings_since(riders), [('initiative', 'Initiative 4')])
        self.assertEqual(result.headings_since(mounts), [('initiative', 'Resolution')])
        self.assertEqual(mounts.headings_since(), [
            ('round', 'Round 4'), ('turn', 'Player 1 Turn'), ('phase', 'Combat'),
            ('combat', first.context['combat']), ('initiative', 'Initiative 4')])
        journal = BattleJournal()
        journal.entries.extend([first, riders, mounts, result])
        self.assertEqual(journal.export().count(first.context['combat']), 1)
        for entry in journal.entries:
            self.assertIn(entry.text, journal.export())
            self.assertIn(entry.details, journal.export())
        self.assertEqual(len(__import__('json').loads(journal.export(structured=True))), 4)

    def test_only_exact_combat_introductions_repeat_the_heading(self):
        from dataclasses import replace
        from rules_log import LogEntry
        entry = LogEntry(1, 'combat', 'Combat: Princes, Knights, Helms', '', '',
                         dict(combat='Princes vs Knights vs Helms'))
        self.assertTrue(entry.repeats_combat_heading)
        self.assertFalse(replace(entry, text=entry.text + ': choose weapons').repeats_combat_heading)
        self.assertFalse(replace(entry, context={}).repeats_combat_heading)
        self.assertFalse(replace(entry, category='warning').repeats_combat_heading)

    def test_new_combat_and_new_turn_restore_their_headings(self):
        from dataclasses import replace
        from rules_log import LogEntry
        first = LogEntry(1, 'combat', 'Attack', '', '',
                         dict(round=4, player=1, phase='CombatPhase', combat='Princes vs Knights', initiative=4))
        second = replace(first, sequence=2, context=dict(first.context, combat='Skycutter vs Horsemen'))
        self.assertEqual(second.headings_since(first), [
            ('combat', 'Skycutter vs Horsemen'), ('initiative', 'Initiative 4')])
        next_turn = replace(first, sequence=3, context=dict(first.context, player=2))
        self.assertEqual(next_turn.headings_since(second), [
            ('turn', 'Player 2 Turn'), ('phase', 'Combat'),
            ('combat', 'Princes vs Knights'), ('initiative', 'Initiative 4')])
        outside = replace(first, context=dict(first.context, combat=None, initiative=None))
        self.assertEqual(outside.headings_since(first), [('phase', 'Combat')])
        self.assertEqual(replace(first, context={}).headings_since(first), [('context', 'Battle')])

    def test_exported_event_clocks_are_stable_and_monotonic(self):
        from datetime import datetime, timedelta, timezone
        import json
        from unittest.mock import patch
        from rules_log import BattleJournal
        first = datetime(2026, 9, 12, 20, 0, 0, tzinfo=timezone.utc)
        journal = BattleJournal()
        with patch('rules_log.datetime') as clock, \
                patch('rules_log._session_started', 100), \
                patch('rules_log.monotonic', side_effect=[100.25, 103.75]):
            clock.now.side_effect = [first, first - timedelta(seconds=1)]
            journal.append('Search started', 'debug')
            journal.append('Search finished', 'debug')
        events = json.loads(journal.export(structured=True))
        self.assertEqual([entry['elapsed_seconds'] for entry in events], [.25, 3.75])
        self.assertEqual(events[0]['timestamp'], '2026-09-12T20:00:00.000+00:00')
        self.assertEqual(events[1]['timestamp'], '2026-09-12T19:59:59.000+00:00')
        self.assertIn('[2026-09-12T20:00:00.000+00:00 +0.250s] [debug] Search started', journal.export())
        self.assertIn('+3.750s] [debug] Search finished', journal.export())
        self.assertEqual(events, json.loads(journal.export(structured=True)))

    def test_filters_preserve_full_trace_and_warning(self):
        from rules_log import BattleJournal
        journal = BattleJournal()
        for category in ('combat', 'rule', 'skip', 'dice', 'debug', 'warning'):
            journal.append(category, category, 'Silver Helms')
        self.assertEqual([entry.category for entry in journal.visible()], ['combat', 'warning'])
        self.assertEqual(len(journal.visible('Rules')), 4)
        self.assertEqual(len(journal.visible('Debug')), 6)
        self.assertEqual(journal.visible(subject='Dragon Princes'), [])
        self.assertIn('[skip] skip', journal.export())

    def test_context_is_snapshotted_and_scopes_restore_after_errors(self):
        from rules_log import BattleJournal, log_scope
        journal = BattleJournal(limit=2)
        with log_scope(round=3, player=1, phase='CombatPhase', combat='Princes vs Knights'):
            try:
                with log_scope(initiative=9):
                    journal.append('9 attacks', 'combat', details='Fury +3 attacks')
                    raise ValueError('interrupted')
            except ValueError:
                pass
            journal.append('result', 'combat')
        self.assertEqual(journal.entries[0].context['initiative'], 9)
        self.assertNotIn('initiative', journal.entries[1].context)
        self.assertIn('Round 3\n\nPlayer 1 Turn', journal.export())
        self.assertIn('Fury +3 attacks', journal.export())
        self.assertEqual(len(__import__('json').loads(journal.export(structured=True))), 2)
        journal.append('new')
        self.assertEqual([entry.sequence for entry in journal.entries], [2, 3])


if __name__ == "__main__":
    unittest.main()
