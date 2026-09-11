"""Dispel options, natural dice and turn use (Rulebook pp. 110-111)."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from panda3d.core import NodePath

from dispelling import attempt, conjuration, wizard_reason
from spell_effects import register
from spell_system import Spell
from tests.test_spell_effects import effect_case


def dispel_case(*, wizard=False, ai=False, choice='Fated dispel'):
    game, caster, defender, spell = effect_case()
    game.player1Units = [caster]
    game.player2Units = [defender]
    game.aiControls = lambda member: ai
    game.makeChoiceNew = AsyncMock(return_value=choice)
    root = NodePath('world')
    for member in (caster, defender):
        member.bodyNP = root.attachNewNode(member.unitName)
        member.unitWidth = member.unitHeight = 1
        member.isDeployed = True
    defender.unit.model.is_wizard = lambda: wizard
    defender.unit.model.wizard_level = lambda default=0: 2 if wizard else default
    spell.casting = 8
    return game, caster, defender, spell


def test_fated_is_optional_once_per_player_turn_and_tie_fails():
    game, caster, defender, spell = dispel_case(choice='Pass')
    with patch.object(Spell, '_roll_casting_dice', AsyncMock(return_value=(8, [4, 4]))) as dice:
        assert not asyncio.run(attempt(game, spell, caster))
        dice.assert_not_awaited()
        game.makeChoiceNew.return_value = 'Fated dispel'
        assert not asyncio.run(attempt(game, spell, caster))
        assert not asyncio.run(attempt(game, spell, caster))
        assert dice.await_count == 1
        game.roundCounter.current_player = 2
        assert not asyncio.run(attempt(game, spell, caster))
        assert dice.await_count == 2


def test_natural_six_unbinds_but_immediate_perfect_never_offers():
    game, caster, defender, spell = dispel_case()
    spell.perfect = True
    spell.casting = 99
    with patch.object(Spell, '_roll_casting_dice', AsyncMock(return_value=(12, [6, 6]))):
        assert not asyncio.run(attempt(game, spell, caster))
        game.makeChoiceNew.assert_not_awaited()
        spell.perfect = False
        assert asyncio.run(attempt(game, spell, caster))


def test_wizard_range_fleeing_engaged_target_and_joined_membership():
    game, caster, defender, spell = dispel_case(wizard=True, ai=True)
    defender.bodyNP.setX(20)
    assert 'outside' in wizard_reason(game, defender, spell)
    defender.bodyNP.setX(10)
    defender.state = 'IsFleeing'
    assert 'fleeing' in wizard_reason(game, defender, spell)
    defender.state = 'Idle'
    host = SimpleNamespace(bodyNP=defender.bodyNP, joinedCharacter=defender,
                           isInCombat=True, state='Idle', unit=defender.unit, unitName='Host')
    defender.hostUnit = host
    game.player2Units = [host]
    assert 'engaged' in wizard_reason(game, defender, spell)
    spell.target = host
    assert wizard_reason(game, defender, spell) is None
    with patch.object(Spell, '_roll_casting_dice', AsyncMock(return_value=(8, [4, 4]))):
        assert asyncio.run(attempt(game, spell, caster))
    assert not hasattr(game, 'fatedDispelTurns')


def test_perfect_remains_uses_minimum_value_and_shares_fated_allowance():
    game, caster, defender, spell = dispel_case(ai=True)
    spell.casting = 19
    spell.casting_value = 7
    spell.perfect = True
    register(spell, duration='remains')
    game.roundCounter.current_player = 2
    with patch.object(Spell, '_roll_casting_dice', AsyncMock(return_value=(8, [4, 4]))) as dice:
        asyncio.run(conjuration(game))
        assert not game.remainsInPlay
        assert not asyncio.run(attempt(game, spell, caster, remains=True))
        assert dice.await_count == 1


def test_outclassed_rolls_table_and_blocks_later_wizard_dispels():
    game, caster, defender, spell = dispel_case(wizard=True, ai=True)
    with patch.object(Spell, '_roll_casting_dice', AsyncMock(side_effect=[(2, [1, 1]), (8, [4, 4])])):
        assert asyncio.run(attempt(game, spell, caster))
    assert 'Outclassed' in wizard_reason(game, defender, spell)
    assert not hasattr(game, 'fatedDispelTurns')