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