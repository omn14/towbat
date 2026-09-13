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


def test_battle_march_most_points_and_reduced_bonuses():
    from types import SimpleNamespace
    from battle_config import load_config
    from victory_points import battle_march_outcome, calculate
    assert battle_march_outcome([10, 0]) == (1, 'Victory')
    assert battle_march_outcome([25, 25]) == (None, 'Draw')
    assert battle_march_outcome([100, 101]) == (2, 'Victory')
    game = SimpleNamespace(battle_config=load_config(), units=[], victoryLedgerComplete=True,
        victoryRoster={'general': {'player': 2, 'name': 'General', 'points': 100, 'strength': 1,
                                  'wounds': 2, 'general': True, 'bsb': False},
                       'bsb': {'player': 2, 'name': 'BSB', 'points': 75, 'strength': 1,
                               'wounds': 2, 'general': False, 'bsb': True}},
        capturedStandards=[{'unit': 'Standard', 'captured_by': 1}],
        battle_awards=[{'player': 2, 'unit': 'Guard', 'points': 10, 'rule': 'Treasure Troves',
                        'objective': 'objective-1', 'turn': '1:0:0', 'reason': 'US 5, touching'}])
    result = calculate(game)
    assert result['scores'] == [275, 10]
    assert result['winner'] == 1
    assert result['rows'][-1]['rule'] == 'Treasure Troves'