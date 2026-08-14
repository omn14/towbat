"""
Unit tests for the character-joins-unit mechanic (characters.py) and the
combat building blocks it relies on.

The scene/physics parts of joining (reparenting nodes, Bullet bodies) need a
running Panda3D ShowBase and are exercised in-game, not here. These tests cover
the pure helpers and the deterministic front-rank attack maths that make a
joined character "replace one model" and fight/shoot with its own profile.

Run:  python3 -m pytest tests/test_characters.py
"""

import contextlib
import io
import os
import sys
import unittest
from types import SimpleNamespace

# Make the repo root importable regardless of how the tests are launched.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from characters import (  # noqa: E402
    is_character, same_player, has_joined_character, get_joined_character,
    on_host_removed, JOIN_TAG,
)
from models import model  # noqa: E402
from battleFunctions import simulate_battle  # noqa: E402


@contextlib.contextmanager
def quiet():
    """Silence the verbose combat prints during a test."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def mk_graphics(category=None, name="u"):
    """Lightweight stand-in for a unitGraphics (no Panda3D dependency)."""
    chars = {}
    if category is not None:
        chars['Category'] = category
    m = SimpleNamespace(characteristics=chars)
    return SimpleNamespace(unit=SimpleNamespace(model=m), unitName=name)


def mk_unit(m, nmodels=5, files=5, ranks=1, name="u"):
    """Lightweight stand-in for units.unit."""
    return SimpleNamespace(model=m, name=name, nmodels=nmodels, files=files, ranks=ranks)


class CharacterHelperTests(unittest.TestCase):
    def test_is_character_true_for_characters_category(self):
        self.assertTrue(is_character(mk_graphics("Characters")))

    def test_is_character_case_and_whitespace_insensitive(self):
        self.assertTrue(is_character(mk_graphics("  characters  ")))

    def test_is_character_false_for_core_unit(self):
        self.assertFalse(is_character(mk_graphics("Core")))

    def test_is_character_false_when_no_category(self):
        self.assertFalse(is_character(mk_graphics(None)))

    def test_same_player_true_within_player_one(self):
        a, b = mk_graphics(name="a"), mk_graphics(name="b")
        game = SimpleNamespace(player1Units=[a, b], player2Units=[])
        self.assertTrue(same_player(game, a, b))

    def test_same_player_false_across_sides(self):
        a, b = mk_graphics(name="a"), mk_graphics(name="b")
        game = SimpleNamespace(player1Units=[a], player2Units=[b])
        self.assertFalse(same_player(game, a, b))

    def test_joined_character_accessors(self):
        char = mk_graphics("Characters", name="lord")
        host = mk_graphics("Core", name="spears")
        self.assertFalse(has_joined_character(host))
        self.assertIsNone(get_joined_character(host))
        host.joinedCharacter = char
        self.assertTrue(has_joined_character(host))
        self.assertIs(get_joined_character(host), char)

    def test_on_host_removed_unlinks_and_drops_character(self):
        char = mk_graphics("Characters", name="lord")
        host = mk_graphics("Core", name="spears")
        host.joinedCharacter = char
        char.hostUnit = host
        game = SimpleNamespace(units=[host, char])
        on_host_removed(game, host)
        self.assertIsNone(host.joinedCharacter)
        self.assertIsNone(char.hostUnit)
        self.assertNotIn(char, game.units)

    def test_on_host_removed_noop_without_character(self):
        host = mk_graphics("Core", name="spears")
        game = SimpleNamespace(units=[host])
        on_host_removed(game, host)  # must not raise
        self.assertEqual(game.units, [host])


class CharacterCombatMathTests(unittest.TestCase):
    """Deterministic attack-count checks (simulate_battle returns the count)."""

    def _mk_model(self, attacks):
        m = model("Test", "")
        m.characteristics = {'A': str(attacks), 'WS': '3', 'S': '3',
                             'T': '3', 'W': '1', 'I': '3', 'Ld': '7'}
        return m

    def _attacks(self, atk_unit, charge):
        with quiet():
            attacks, *_ = simulate_battle(
                atk_unit, mk_unit(self._mk_model(1), 10, 5, 2), charge=charge)
        return attacks

    def test_single_model_character_makes_its_own_attacks(self):
        # A lone character (files=1) contributes exactly its A on the charge.
        char = mk_unit(self._mk_model(3), nmodels=1, files=1, ranks=1)
        self.assertEqual(self._attacks(char, charge=True), 3)

    def test_dropping_one_file_removes_one_front_rank_attack_on_charge(self):
        # Replacing one front-rank model = one fewer host attack on the charge.
        five = mk_unit(self._mk_model(1), nmodels=20, files=5, ranks=4)
        four = mk_unit(self._mk_model(1), nmodels=20, files=4, ranks=5)
        a5 = self._attacks(five, charge=True)
        a4 = self._attacks(four, charge=True)
        self.assertEqual(a5, 5)
        self.assertEqual(a4, 4)
        self.assertEqual(a5 - a4, 1)


if __name__ == "__main__":
    unittest.main()
