"""Rallying Cry: early Rally without consuming the normal attempt (p. 175)."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import game as game_module
from game import MyApp
from models import model
from panda3d.core import NodePath
from special_rules import apply_rule_keywords
from psychology import leadership_passed, rally_leadership
from rallying_cry import (begin_command, choose_rallying_cry, command_radius,
                         finish_command, source_reason, target_distance, target_reason,
                         use_rallying_cry)


def make_character(name='Captain', position=(0, 0, 0)):
    profile = model('Captain of the Empire', '')
    apply_rule_keywords(profile, ['Rallying Cry'], replace=True)
    profile.characteristics['Ld'] = '7'
    body = NodePath(name)
    body.setPos(*position)
    return SimpleNamespace(
        unit=SimpleNamespace(model=profile, nmodels=1, name=name), unitName=name,
        bodyNP=body, modelWidth=1, modelHeight=1, unitWidth=1, unitHeight=1,
        state='Idle', isInCombat=False, isGeneral=False, isBSB=False,
        usedRallyingCry=False, hostUnit=None, joinedCharacter=None)


def context():
    actor = make_character()
    target = make_character('Fleeing', (8, 0, 0))
    root = NodePath('battlefield')
    actor.bodyNP.reparentTo(root)
    target.bodyNP.reparentTo(root)
    target.state = 'IsFleeing'
    game = SimpleNamespace(player1Units=[actor, target], player2Units=[],
                           units=[actor, target], unitToMove=actor,
                           fsm=SimpleNamespace(state='StrategyPhase'),
                           roundCounter=SimpleNamespace(current_player=1),
                           strategyCommandDone=False, refreshSelectedUnit=Mock(),
                           rallyUnit=AsyncMock(return_value=False),
                           aiControls=lambda unit: False,
                           makeChoiceNew=AsyncMock(return_value=None),
                           psychology=SimpleNamespace(leadership_of=lambda unit: (7, None)))
    return game, actor, target


def test_nomination_uses_base_edges_and_ordinary_command_range():
    game, actor, target = context()
    assert actor.unit.model.is_rallying_cry()
    assert command_radius(actor) == 7
    assert target_distance(actor, target) == 7
    assert target_reason(game, actor, target) is None
    target.bodyNP.setX(8.01)
    assert 'exceeds Command range 7' in target_reason(game, actor, target)


@pytest.mark.parametrize('tag', ['mount', 'crew', 'beasts'])
def test_split_profile_grants_rallying_cry(tag):
    game, actor, target = context()
    apply_rule_keywords(actor.unit.model, [], replace=True)
    part = make_character().unit.model
    if tag == 'mount':
        actor.unit.model.attach_mount(SimpleNamespace(model=part))
    else:
        actor.unit.model.special_rules.append(
            {'tag': tag, 'partUnit': SimpleNamespace(model=part), 'count': 1})
    assert source_reason(game, actor) is None
    if tag != 'mount':
        actor.unit.model.special_rules[-1]['count'] = 0
        assert 'requires a character with' in source_reason(game, actor)


def test_noncharacter_and_absent_models_cannot_nominate():
    game, actor, target = context()
    actor.unit.model.characteristics['Category'] = 'Core'
    assert 'requires a character' in source_reason(game, actor)
    actor.unit.model.characteristics['Category'] = 'Characters'
    actor.unit.nmodels = 0
    assert 'not on the battlefield' in source_reason(game, actor)
    actor.unit.nmodels = 1
    target.unit.nmodels = 0
    assert 'not on the battlefield' in target_reason(game, actor, target)


@pytest.mark.parametrize('role', ['isGeneral', 'isBSB'])
def test_command_roles_extend_range_and_large_mount(role):
    game, actor, target = context()
    setattr(actor, role, True)
    assert command_radius(actor) == 12
    mount = model('Barded Warhorse', '')
    apply_rule_keywords(mount, ['Large Target'], replace=True)
    actor.unit.model.attach_mount(SimpleNamespace(model=mount))
    assert command_radius(actor) == 18
    setattr(actor, role, False)
    assert command_radius(actor) == 7


@pytest.mark.parametrize('field, value, reason', [
    ('state', 'IsFleeing', 'fleeing'), ('isInCombat', True, 'engaged'),
    ('usedRallyingCry', True, 'already used'),
    ('retiredFromCombat', True, 'retired'),
])
def test_ineligible_sources(field, value, reason):
    game, actor, target = context()
    setattr(actor, field, value)
    assert reason in target_reason(game, actor, target)


def test_joined_character_uses_own_world_position_and_host_state():
    game, actor, target = context()
    host = make_character('Host', (20, 0, 0))
    host.bodyNP.reparentTo(actor.bodyNP.getTop())
    actor.hostUnit = host
    actor.bodyNP.reparentTo(host.bodyNP)
    actor.bodyNP.setX(-20)
    assert target_distance(actor, target) == 7
    assert source_reason(game, actor) is None
    host.state = 'IsFleeing'
    assert 'fleeing' in source_reason(game, actor)
    host.state = 'InCombat'
    assert 'engaged' in source_reason(game, actor)


def test_wrong_turn_phase_enemy_and_nonfleeing_targets():
    game, actor, target = context()
    game.roundCounter.current_player = 2
    assert 'turn' in source_reason(game, actor)
    game.roundCounter.current_player = 1
    game.strategyCommandDone = True
    assert 'Command sub-phase' in source_reason(game, actor)
    game.strategyCommandDone = False
    game.fsm.state = 'MovementPhase'
    assert 'Command sub-phase' in source_reason(game, actor)
    game.fsm.state = 'StrategyPhase'
    target.state = 'Idle'
    assert 'fleeing' in target_reason(game, actor, target)
    target.state = 'IsFleeing'
    game.player1Units.remove(target)
    game.player2Units.append(target)
    assert 'friendly' in target_reason(game, actor, target)


def test_nomination_spends_character_use_but_tests_target(capsys):
    game, actor, target = context()
    assert not asyncio.run(use_rallying_cry(game, actor, target))
    game.rallyUnit.assert_awaited_once_with(target, command=True)
    assert actor.usedRallyingCry
    assert game.unitToMove is actor
    assert not game.rallyingCryBusy
    asyncio.run(use_rallying_cry(game, actor, target))
    game.rallyUnit.assert_awaited_once()
    assert 'normal Rally' in capsys.readouterr().out


def test_invalid_nomination_does_not_spend_or_roll():
    game, actor, target = context()
    target.bodyNP.setX(20)
    assert not asyncio.run(use_rallying_cry(game, actor, target))
    assert not actor.usedRallyingCry
    game.rallyUnit.assert_not_awaited()


def test_human_can_cancel_without_spending():
    game, actor, target = context()
    asyncio.run(choose_rallying_cry(game, actor))
    assert game.makeChoiceNew.call_args.kwargs['owner'] is actor
    assert not actor.usedRallyingCry
    game.rallyUnit.assert_not_awaited()


def test_ai_nominates_without_human_choice():
    game, actor, target = context()
    game.aiControls = lambda unit: True
    asyncio.run(choose_rallying_cry(game, actor))
    game.makeChoiceNew.assert_not_awaited()
    game.rallyUnit.assert_awaited_once_with(target, command=True)


def test_begin_finish_command_and_busy_gate():
    game, actor, target = context()
    actor.usedRallyingCry = True
    begin_command(game)
    assert not actor.usedRallyingCry
    assert not game.strategyCommandDone
    game.rallyingCryBusy = True
    assert not finish_command(game)
    assert not game.strategyCommandDone
    game.rallyingCryBusy = False
    assert finish_command(game)
    assert 'Command sub-phase' in source_reason(game, actor)


@pytest.mark.parametrize('remaining, expected', [(10, 7), (5, 7), (4, 6), (3, 6), (2, 0)])
def test_rally_loss_thresholds(remaining, expected):
    unit = make_character()
    unit.startOfBattleModels = 10
    unit.unit.nmodels = remaining
    assert rally_leadership(unit, 7) == expected
    if expected == 0:
        assert leadership_passed(2, expected)
        assert not leadership_passed(3, expected)


@pytest.mark.parametrize('command, dice, rallied, attempted', [
    (True, [6, 6], False, False),
    (False, [6, 6], False, True),
    (True, [2, 3], True, True),
    (False, [2, 3], True, True),
])
def test_early_rally_only_preserves_normal_attempt_on_failure(
        command, dice, rallied, attempted, monkeypatch):
    unit = SimpleNamespace(
        unit=SimpleNamespace(model=SimpleNamespace(is_veteran=lambda: False),
                             nmodels=5, name='Fleeing Troops'),
        request=Mock(), spreadToSkirmish=Mock(), attemptedRallyThisTurn=False)
    game = SimpleNamespace(
        psychology=SimpleNamespace(leadership_of=lambda unit: (6, None),
                                   battle_standard_of=lambda unit: None),
        rollLeadershipDice=AsyncMock(return_value=dice),
        accept=Mock(), ignore=Mock(), giveSignal=Mock(), freeReformUnit=AsyncMock(),
        setActiveUnit=Mock(), taskLoopStrategy=Mock())
    monkeypatch.setattr(game_module, 'taskMgr',
                        SimpleNamespace(add=lambda function, *args, **kwargs: function()),
                        raising=False)
    assert asyncio.run(MyApp.rallyUnit(game, unit, command=command)) is rallied
    assert unit.attemptedRallyThisTurn is attempted
    assert unit.request.called is rallied
    assert unit.spreadToSkirmish.call_count == int(rallied)