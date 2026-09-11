"""Warband ranks do not make an ineligible Leadership source better (p. 180)."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from models import model
from special_rules import apply_rule_keywords
from warband import leadership_for_test, roll_charge


def member(rules=('Warband',), leadership=6):
    profile = model('Chaos Warrior', '')
    apply_rule_keywords(profile, list(rules), replace=True)
    profile.characteristics['Ld'] = leadership
    return SimpleNamespace(unit=SimpleNamespace(model=profile, nmodels=12, files=4, ranks=3, name='Warband'),
                           state='Idle', isInCombatWith=[])


@pytest.mark.parametrize('kind,expected', [('Fear', 8), ('Panic', 8), ('Break', 8), ('Restraint', 6), ('Impetuous', 6)])
def test_leadership_bonus_is_context_limited(kind, expected):
    unit = member()
    psychology = SimpleNamespace(leadership_of=lambda unit: (6, None))
    assert leadership_for_test(psychology, unit, kind)[0] == expected
    unit.state = 'IsFleeing'
    assert leadership_for_test(psychology, unit, kind)[0] == 6


def test_non_warband_character_leadership_is_not_modified():
    unit, general = member(), member(rules=(), leadership=9)
    psychology = SimpleNamespace(leadership_of=lambda unit: (9, general), general_of=lambda unit: general)
    assert leadership_for_test(psychology, unit, 'Fear')[0] == 9


def test_skirmishers_reroll_charge_dice_but_not_swiftstride_bonus():
    unit = member(rules=('Warband', 'Skirmishers'))
    psychology = SimpleNamespace(leadership_of=lambda unit: (6, None))
    assert leadership_for_test(psychology, unit, 'Fear')[0] == 6
    original = Mock()
    roll = AsyncMock(side_effect=[([original], [1, 2, 3]), ([], [4, 5])])
    game = SimpleNamespace(aiControls=lambda unit: False, makeChoiceNew=AsyncMock(return_value='Re-roll'), world=object())
    assert asyncio.run(roll_charge(game, unit, True, roll)) == ([], [4, 5, 3])
    roll.assert_awaited_with(2, False)
    assert roll.await_count == 2
    original.remove.assert_called_once_with(game.world)


def test_pursuit_never_gets_warband_charge_reroll():
    unit = member()
    unit.state = 'IsPursuing'
    roll = AsyncMock(return_value=([], [1, 1]))
    game = SimpleNamespace(makeChoiceNew=AsyncMock())
    assert asyncio.run(roll_charge(game, unit, False, roll)) == ([], [1, 1])
    game.makeChoiceNew.assert_not_awaited()


@pytest.mark.parametrize('dice,reroll', [([1, 2, 6], True), ([5, 2, 1], False)])
def test_ai_charge_reroll_ignores_separate_swiftstride_die(dice, reroll):
    unit = member(rules=('Warband', 'Swiftstride'))
    original = [Mock(), Mock(), Mock()]
    replacement = [Mock(), Mock()]
    roll = AsyncMock(side_effect=[(original, dice), (replacement, [4, 5])])
    game = SimpleNamespace(aiControls=lambda unit: True, makeChoiceNew=AsyncMock(), world=object())
    result = asyncio.run(roll_charge(game, unit, True, roll))
    assert result == (replacement + original[2:], [4, 5, dice[2]]) if reroll else result == (original, dice)
    assert roll.await_count == 1 + reroll
    original[2].remove.assert_not_called()
    for die in original[:2]:
        assert die.remove.call_count == int(reroll)
    game.makeChoiceNew.assert_not_awaited()