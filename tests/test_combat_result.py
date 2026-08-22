"""The combat result table.

A lost combat is the moment a player asks "why?", and the answer is always a
sum of small parts. The table shows every part, including the zeroes — the
question is usually which bonus the *other* side had.
"""

import contextlib
import io
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from combat_resolution import CombatResolver  # noqa: E402

ROWS = {'Wounds caused': (4, 2), 'Impact Hits': (0, 3), 'Flank / rear': (0, 1),
        'Rank Bonus': (2, 0), 'Battle Standard': (1, 0),
        'Massed Infantry': (1, 0)}


def _table(rows=None, totals=(8, 6), unit_strengths=(24, 15)):
    with contextlib.redirect_stdout(io.StringIO()) as out:
        CombatResolver.printCombatResult(rows if rows is not None else ROWS,
                                         totals, unit_strengths)
    return out.getvalue()


class TestTheCombatResultTable(unittest.TestCase):

    def test_every_contribution_has_a_line(self):
        table = _table()
        for label in ROWS:
            self.assertIn(label, table)

    def test_a_zero_is_still_shown(self):
        # Knowing the enemy had no rank bonus is worth as much as knowing yours.
        line = next(l for l in _table().splitlines() if 'Impact Hits' in l)
        self.assertIn('0', line)
        self.assertIn('3', line)

    def test_both_sides_are_totalled(self):
        line = next(l for l in _table().splitlines() if 'TOTAL' in l)
        self.assertIn('8', line)
        self.assertIn('6', line)

    def test_unit_strength_is_shown_because_massed_infantry_turns_on_it(self):
        line = next(l for l in _table().splitlines() if 'Unit Strength' in l)
        self.assertIn('24', line)
        self.assertIn('15', line)

    def test_it_says_who_won_and_by_how_much(self):
        self.assertIn('Player 1 wins by 2', _table())
        self.assertIn('Player 2 wins by 3', _table(totals=(5, 8)))

    def test_a_draw_says_so(self):
        self.assertIn('drawn combat', _table(totals=(4, 4)))

    def test_the_columns_line_up(self):
        # Every label is padded to the longest, so the numbers form columns and
        # each row comes out the same width.
        rows = [l for l in _table().splitlines() if l.strip() and '-' not in l]
        self.assertEqual(len({len(l) for l in rows}), 1, rows)


if __name__ == "__main__":
    unittest.main()
