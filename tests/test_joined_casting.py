"""Joined Wizards retain their own spellbook and allowance (Rulebook pp. 108, 207)."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from battlescribe import get_catalogue, spell_key
from game import MyApp
from game_fsm import GamePhaseFSM
from spell_system import casting_units
from tests.test_bound_spells import app_stub, carrier, ring


def joined_game(host_spells=None, phase='shooting'):
    host = carrier(host_spells or [], level=2 if host_spells else 0)
    host.unit.name = 'Host'
    wizard = carrier([get_catalogue().spell('Fireball')], level=2)
    wizard.unitName = wizard.unit.name = 'Mage'
    host.joinedCharacter = wizard
    wizard.hostUnit = host
    wizard.bodyNP.getH.return_value = 0
    game = app_stub(host, phase)
    game.units.append(wizard)
    return game, host, wizard


def test_cast_key_discovers_joined_wizard_without_host_spellbook():
    game, host, wizard = joined_game()
    assert casting_units(game, host) == [wizard]
    MyApp.castSpell(game)
    game.fsm.request.assert_called_once_with('SpellPhase')
    assert not host.unit.model.spells and game.unitToMove is host


@pytest.mark.parametrize('blocked', ['fleeing', 'marching', 'spent', 'enemy'])
def test_joined_casting_respects_host_state_and_wizard_allowance(blocked):
    game, host, wizard = joined_game()
    if blocked == 'fleeing':
        host.state = 'IsFleeing'
    elif blocked == 'marching':
        host.marchedThisTurn = True
    elif blocked == 'spent':
        wizard.spellsCastThisTurn = ['Fireball']
    else:
        game.roundCounter.current_player = 2
    assert casting_units(game, host) == []
    MyApp.castSpell(game)
    game.fsm.request.assert_not_called()


def run_arc(game):
    tasks = Mock()
    tasks.add.side_effect = lambda awaitable, *args, **kwargs: awaitable
    with patch('game.taskMgr', tasks, create=True), patch('game.render', Mock(), create=True):
        asyncio.run(MyApp.taskMagicArcUpdate(game, SimpleNamespace(done='done')))


def test_arc_uses_joined_wizard_and_world_facing():
    game, host, wizard = joined_game()
    game.makeChoiceNew = AsyncMock(return_value='Fireball')
    run_arc(game)
    assert game.fsm.castingUnit is wizard
    assert game.fsm.spellInstanceToCast.caster is wizard
    assert game.unitToMove is wizard and wizard.hostUnit is host
    assert game.makeChoiceNew.call_args.kwargs['owner'] is wizard
    assert len(wizard.bodyNP.getH.call_args.args) == 1


def test_host_bound_item_and_joined_wizard_offer_distinct_casters():
    game, host, wizard = joined_game([ring()])
    game.makeChoiceNew = AsyncMock(side_effect=['2: Mage', 'Fireball'])
    run_arc(game)
    assert game.makeChoiceNew.call_args_list[0].args[0] == ['1: Host', '2: Mage']
    assert game.fsm.castingUnit is wizard
    assert spell_key(ring()) in host.unit.model.spells


def test_host_bound_caster_keeps_its_own_spell_identity_and_allowance():
    game, host, wizard = joined_game([ring()])
    host.bodyNP.getH.return_value = 0
    key = spell_key(ring())
    game.makeChoiceNew = AsyncMock(side_effect=['1: Host', key])
    run_arc(game)
    assert game.fsm.castingUnit is host
    assert game.fsm.spellInstanceToCast.bound
    assert game.fsm.spellInstanceToCast.selection_key == key
    assert wizard.spellsCastThisTurn == []


def test_cast_key_does_not_reenter_while_choosing_or_casting():
    game, _, _ = joined_game()
    game.awaitingChoice = True
    MyApp.castSpell(game)
    game.fsm.request.assert_not_called()
    game.awaitingChoice = False
    game.fsm.state = 'SpellPhase'
    MyApp.castSpell(game)
    game.fsm.request.assert_not_called()


def test_caster_choice_cancel_does_not_spend_or_change_selection():
    game, host, wizard = joined_game([ring()])
    game.makeChoiceNew = AsyncMock(return_value=None)
    run_arc(game)
    assert game.unitToMove is host
    assert not wizard.spellsCastThisTurn and not host.boundSpellPhases
    game.fsm.request.assert_called_once_with('ShootingPhase')


def test_spell_exit_restores_selected_host():
    game, host, wizard = joined_game()
    game.fsm.spellSelectionUnit = host
    game.fsm.game = game
    game.fsm._cleanup_phase = Mock()
    game.unitToMove = wizard
    with patch('game_fsm.taskMgr', Mock(), create=True):
        GamePhaseFSM.exitSpellPhase(game.fsm)
    assert game.unitToMove is host
    assert game.fsm.spellSelectionUnit is None


def test_retired_joined_wizard_cannot_cast_assailment():
    from spell_system import restore_spellbook

    game, host, wizard = joined_game(phase='combat')
    restore_spellbook(wizard.unit.model, [get_catalogue().spell('Hammerhand')], 2)
    host.isInCombat = True
    game.assailmentWindow = {'caster': wizard, 'targets': []}
    assert casting_units(game, host) == [wizard]
    wizard.retiredFromCombat = True
    assert casting_units(game, host) == []