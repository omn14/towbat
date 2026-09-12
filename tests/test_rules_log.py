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
        self.assertIn('Round 3 / Player 1', journal.export())
        self.assertIn('Fury +3 attacks', journal.export())
        self.assertEqual(len(__import__('json').loads(journal.export(structured=True))), 2)
        journal.append('new')
        self.assertEqual([entry.sequence for entry in journal.entries], [2, 3])


if __name__ == "__main__":
    unittest.main()
