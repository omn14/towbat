"""Failed casting rerolls, never miscast rerolls (FoF p. 185; FAQ v1.5.3)."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from spell_system import Spell
from tests.test_magic_resistance import _UnitGraphics


def casting_case(*, ai=False, bound=False, choice='Re-roll', dice=None):
    caster = _UnitGraphics('Mage', "Lileath's Blessing", nmodels=1)
    target = _UnitGraphics('Target')
    game = SimpleNamespace(player1Units=[caster], player2Units=[target],
                           roundCounter=SimpleNamespace(current_player=1, currentRoundPlayer=[2, 2]),
                           aiControls=lambda member: ai, makeChoiceNew=AsyncMock(return_value=choice))
    spell = Spell('Test Spell', 8, wizard_level=2, game=game, caster=caster, bound=bound)
    spell._roll_casting_dice = AsyncMock(side_effect=dice or [(4, [2, 2]), (8, [4, 4])])
    return game, caster, target, spell


@pytest.mark.parametrize('ai', [False, True])
def test_failed_cast_rerolls_once_and_is_spent_for_turn(ai):
    game, caster, target, spell = casting_case(ai=ai)
    assert asyncio.run(spell._attempt(target))
    assert spell.casting == 9 and caster.lileathUsedTurn == [1, 2]
    spell._roll_casting_dice = AsyncMock(return_value=(4, [2, 2]))
    assert not asyncio.run(spell._attempt(target))
    spell._roll_casting_dice.assert_awaited_once()
    assert game.makeChoiceNew.await_count == (0 if ai else 1)


def test_decline_retains_use_and_next_turn_refreshes():
    game, caster, target, spell = casting_case(choice='Keep')
    assert not asyncio.run(spell._attempt(target))
    assert getattr(caster, 'lileathUsedTurn', None) is None
    caster.lileathUsedTurn = [1, 1]
    game.makeChoiceNew.return_value = 'Re-roll'
    spell._roll_casting_dice = AsyncMock(side_effect=[(4, [2, 2]), (8, [4, 4])])
    assert asyncio.run(spell._attempt(target))
    assert caster.lileathUsedTurn == [1, 2]


@pytest.mark.parametrize('dice', [[1, 1], [6, 6], [4, 4]])
def test_success_perfect_and_miscast_never_offer_blessing(dice):
    game, caster, target, spell = casting_case(dice=[(sum(dice), dice), (7, [3, 4])])
    asyncio.run(spell._attempt(target))
    game.makeChoiceNew.assert_not_awaited()
    assert getattr(caster, 'lileathUsedTurn', None) is None


def test_replacement_can_miscast_and_cannot_be_rerolled_again():
    game, caster, target, spell = casting_case(dice=[(4, [2, 2]), (2, [1, 1]), (7, [3, 4])])
    assert not asyncio.run(spell._attempt(target))
    assert spell._roll_casting_dice.await_count == 3
    game.makeChoiceNew.assert_awaited_once()
    assert caster.lileathUsedTurn == [1, 2]


def test_bound_spell_does_not_borrow_bearer_blessing():
    game, caster, target, spell = casting_case(bound=True)
    assert not asyncio.run(spell._attempt(target))
    game.makeChoiceNew.assert_not_awaited()
    assert getattr(caster, 'lileathUsedTurn', None) is None