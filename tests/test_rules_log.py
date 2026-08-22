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


if __name__ == "__main__":
    unittest.main()
