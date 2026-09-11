"""Amended standard-battle VP thresholds and non-stacking awards (p. 286)."""

import pytest

from victory_points import outcome, unit_award


@pytest.mark.parametrize('strength,wounds,fleeing,absent,expected', [
    (0, 0, False, False, 351),
    (20, 10, False, True, 351),
    (20, 10, True, False, 176),
    (5, 3, False, False, 176),
    (5, 3, True, False, 176),
    (6, 3, False, False, 0),
])
def test_destroyed_fleeing_and_quarter_strength(strength, wounds, fleeing, absent, expected):
    assert unit_award(351, 20, strength, 10, wounds, fleeing=fleeing, absent=absent)[0] == expected


def test_models_with_wounds_based_strength_use_remaining_wounds():
    assert unit_award(101, 8, 8, 8, 2)[0] == 51
    assert unit_award(101, 8, 8, 8, 3)[0] == 0
    assert unit_award(101, 5, 5, 4, 1)[0] == 0


@pytest.mark.parametrize('scores,expected', [
    ([0, 0], (None, 'Draw')),
    ([99, 0], (None, 'Draw')),
    ([100, 0], (1, 'Crushing victory')),
    ([250, 350], (2, 'Victory')),
    ([250, 500], (2, 'Crushing victory')),
])
def test_winning_margin_precedes_crushing_victory(scores, expected):
    assert outcome(scores) == expected