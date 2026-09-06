"""Scouts, p. 177 and all three Official FAQ v1.5.3 qualifications."""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from panda3d.bullet import BulletRigidBodyNode
from panda3d.core import NodePath, Point3, Vec3

import deployPhase
import combat_resolution
import game as game_module
from game import MyApp
from game_fsm import GamePhaseFSM
from models import model
from scouts import (deployment_candidates, has_scouts, model_base_boxes,
                    nearest_enemy, placement_error, scout_charge_blocked,
                    scouts_block_vanguard)
from special_rules import apply_rule_keywords


def make_unit(root, name, x=0, y=0, *, scouts=True, deployed=False, offsets=((0, 0),)):
    profile = model('Dwarf Warrior', '')
    apply_rule_keywords(profile, ['Scouts'] if scouts else [], replace=True)
    body = root.attachNewNode(BulletRigidBodyNode('UnitCollision-' + name))
    body.setPos(x, y, 0)
    visual = body.attachNewNode('models')
    for i, (dx, dy) in enumerate(offsets):
        visual.attachNewNode(str(i)).setPos(dx, dy, 0)
    return NS(unitName=name, unit=NS(name=name, model=profile, nmodels=len(offsets)),
              bodyNP=body, model=visual, modelWidth=1, modelHeight=1,
              isDeployed=deployed, scoutDeploymentChoice=None, deployedAsScouts=False,
              color=(1, 1, 1, 1), state='Idle', joinedCharacter=None,
              isChargingMove=False, hasMovedThisTurn=False)


def make_game(p1, p2):
    game = NS(units=p1 + p2, player1Units=p1, player2Units=p2,
              deploymentStage='ordinary', scoutDeployFirst=None, firstFinishedDeploying=None,
              AIplayer2=NS(active=False, deployUnits=Mock()),
              boundary_np=NodePath('boundary'), fsm=NS(request=Mock()),
              accept=Mock(), ignore=Mock(), setActiveUnit=Mock(),
              setActiveUnitTask=Mock(), setActiveUnitTaskName='taskLoopDeploy',
              terrain_manager=NS(terrain_pieces=[]))
    counter = NS(current_player=1, currentRoundPlayer=[0, 0], update_round_display=Mock())
    counter.request = lambda state: setattr(counter, 'current_player', 1 if state == 'PlayerOne' else 2)
    game.roundCounter = counter
    return game


@pytest.fixture
def pair():
    root = NodePath('root')
    scout = make_unit(root, 'Scout')
    enemy = make_unit(root, 'Enemy', y=14, scouts=False, deployed=True)
    yield make_game([scout], [enemy]), scout, enemy
    root.removeNode()


def test_keyword_is_coded_and_joined_character_does_not_grant_it(pair):
    game, scout, enemy = pair
    assert has_scouts(scout)
    assert not has_scouts(enemy)
    enemy.joinedCharacter = scout
    assert not has_scouts(enemy)
    enemy.joinedCharacter = None
    scout.joinedCharacter = enemy
    assert not has_scouts(scout)


@pytest.mark.parametrize('gap,allowed', [(11.999, False), (12.0, False), (12.001, True)])
def test_strict_base_clearance(pair, gap, allowed):
    game, scout, enemy = pair
    enemy.bodyNP.setY(gap + 1)
    assert (placement_error(game, scout, scouting=True) is None) == allowed
    assert nearest_enemy(game, scout)[0] == pytest.approx(gap, abs=2e-6)


def test_own_zone_is_not_an_exception(pair):
    game, scout, enemy = pair
    scout.bodyNP.setY(-18)
    enemy.bodyNP.setY(-6)
    assert 'more than 12' in placement_error(game, scout, scouting=True)
    assert placement_error(game, scout, scouting=False) is None


def test_rotated_base_edges_not_centres(pair):
    game, scout, enemy = pair
    scout.modelWidth = 4
    scout.bodyNP.setH(90)
    assert nearest_enemy(game, scout)[0] == pytest.approx(11.5)
    assert placement_error(game, scout, scouting=True)


def test_empty_skirmish_blob_corner_is_not_a_model(pair):
    game, scout, enemy = pair
    # A loose group at (-10, 0) and (0, -10), not a filled 11x11 rectangle.
    scout.model.getChild(0).setPos(-10, 0, 0)
    scout.model.attachNewNode('second').setPos(0, -10, 0)
    scout.unit.nmodels = 2
    enemy.bodyNP.setPos(5, 5, 0)
    assert nearest_enemy(game, scout)[0] == pytest.approx((14**2 + 4**2)**0.5)
    assert placement_error(game, scout, scouting=True) is None


@pytest.mark.parametrize('x,y,heading', [(36, 0, 0), (0, 24, 0), (35.4, 0, 45)])
def test_entire_rotated_base_stays_on_board(pair, x, y, heading):
    game, scout, enemy = pair
    enemy.isDeployed = False
    scout.bodyNP.setPos(x, y, 0)
    scout.bodyNP.setH(heading)
    assert 'battlefield' in placement_error(game, scout, scouting=True)


def test_deep_outside_zone_not_mistaken_for_inside_boundary_walls(pair):
    game, scout, _ = pair
    scout.bodyNP.setPos(200, -18, 0)
    assert placement_error(game, scout)
    scout.bodyNP.setPos(0, -11.75, 0)
    assert 'zone' in placement_error(game, scout)
    scout.bodyNP.setY(-12.5)
    assert placement_error(game, scout) is None


def test_dead_undeployed_and_friendly_units_do_not_impose_twelve_inches(pair):
    game, scout, enemy = pair
    enemy.bodyNP.setY(2)
    enemy.isDeployed = False
    assert placement_error(game, scout, scouting=True) is None
    enemy.isDeployed = True
    enemy.unit.nmodels = 0
    assert placement_error(game, scout, scouting=True) is None
    enemy.unit.nmodels = 1
    game.player2Units.remove(enemy)
    game.player1Units.append(enemy)
    assert placement_error(game, scout, scouting=True) is None
    enemy.bodyNP.setY(0.5)
    assert 'overlap' in placement_error(game, scout, scouting=True)


def test_joined_enemy_character_base_is_measured(pair):
    game, scout, enemy = pair
    char = make_unit(enemy.bodyNP, 'Character', y=-3, deployed=True)
    enemy.joinedCharacter = char
    char.hostUnit = enemy
    game.units.append(char)
    assert nearest_enemy(game, scout)[0] == pytest.approx(10)


def test_impassable_blocks_but_difficult_and_hills_do_not(pair):
    game, scout, _ = pair
    piece = NS(is_impassable=True, center=Point3(0, 0, 0), width=2, height=2,
               terrain_type='house')
    game.terrain_manager.terrain_pieces.append(piece)
    assert 'impassable' in placement_error(game, scout, scouting=True)
    piece.is_impassable = False
    assert placement_error(game, scout, scouting=True) is None


@pytest.mark.parametrize('owner,rounds,blocked', [(1, [0, 0], True), (1, [1, 0], False),
                                               (2, [1, 0], True), (2, [1, 1], False)])
def test_first_turn_is_per_owner_not_current_player(pair, owner, rounds, blocked):
    game, scout, _ = pair
    if owner == 2:
        game.player1Units.remove(scout)
        game.player2Units.append(scout)
    game.roundCounter.current_player = 3 - owner
    game.roundCounter.currentRoundPlayer = rounds
    scout.deployedAsScouts = True
    assert scout_charge_blocked(game, scout) is blocked


def test_normal_deployment_keeps_charge_and_vanguard_available(pair):
    game, scout, _ = pair
    assert has_scouts(scout)
    assert not scout_charge_blocked(game, scout)
    assert not scouts_block_vanguard(scout)
    scout.deployedAsScouts = True
    assert scouts_block_vanguard(scout)


def test_late_character_cannot_bypass_charge_block_by_joining(pair):
    game, host, _ = pair
    char = make_unit(host.bodyNP, 'Character')
    char.deployedAsScouts = True
    host.joinedCharacter = char
    assert scout_charge_blocked(game, host)
    assert scouts_block_vanguard(host)


def test_reserving_is_not_a_drop_and_both_armies_must_finish_first(pair):
    game, scout, enemy = pair
    extra = make_unit(scout.bodyNP.getTop(), 'Ordinary', scouts=False)
    game.units.append(extra)
    game.player1Units.append(extra)
    enemy.isDeployed = False
    scout.scoutDeploymentChoice = 'scouts'
    deployPhase._advance_after_deploy(game, placed=False)
    assert game.roundCounter.current_player == 1
    assert deployment_candidates(game, 1) == [extra]
    extra.isDeployed = True
    deployPhase._advance_after_deploy(game)
    assert game.roundCounter.current_player == 2 and game.deploymentStage == 'ordinary'
    assert game.firstFinishedDeploying is None
    enemy.isDeployed = True
    deployPhase._advance_after_deploy(game)
    assert game.firstFinishedDeploying == 2
    assert game.deploymentStage == 'scouts' and game.roundCounter.current_player == 1
    assert deployment_candidates(game, 1) == [scout]


def test_rolloff_rerolls_ties_alternates_and_counts_last_scout(pair):
    game, first, second = pair
    apply_rule_keywords(second.unit.model, ['Scouts'], replace=True)
    second.isDeployed = False
    third = make_unit(first.bodyNP.getTop(), 'Third Scout')
    game.units.append(third)
    game.player2Units.append(third)
    for u in game.units:
        u.scoutDeploymentChoice = 'scouts'
    with patch.object(deployPhase.random, 'randint', side_effect=[3, 3, 2, 6]) as dice:
        deployPhase._advance_after_deploy(game, placed=False)
    assert dice.call_count == 4 and game.scoutDeployFirst == 2
    assert game.roundCounter.current_player == 2
    second.isDeployed = True
    deployPhase._advance_after_deploy(game)
    assert game.roundCounter.current_player == 1 and game.firstFinishedDeploying is None
    first.isDeployed = True
    deployPhase._advance_after_deploy(game)
    assert game.roundCounter.current_player == 2 and game.firstFinishedDeploying == 1
    third.isDeployed = True
    deployPhase._advance_after_deploy(game)
    game.fsm.request.assert_called_once_with('StrategyPhase')


def test_end_phase_cannot_skip_ordinary_or_scouts(pair):
    game, scout, _ = pair
    fsm = NS(state='DeployPhase', game=game, request=Mock())
    for step in ('ordinary', 'scouts'):
        game.deploymentStage = step
        GamePhaseFSM.nextPhase(fsm)
        fsm.request.assert_not_called()
    scout.isDeployed = True
    GamePhaseFSM.nextPhase(fsm)
    fsm.request.assert_called_once_with('StrategyPhase')


@pytest.mark.parametrize('choice', ['Deploy as Scouts later', 'Deploy normally'])
def test_actual_selection_offers_optional_scouts(pair, choice):
    game, scout, _ = pair
    game.unitToMove = scout
    game.makeChoiceNew = AsyncMock(return_value=choice)
    tasks = Mock()
    with patch.object(game_module, 'taskMgr', tasks, create=True):
        asyncio.run(MyApp.taskLoopDeploy(game, NS(done='done')))
    game.makeChoiceNew.assert_awaited_once()
    assert scout.scoutDeploymentChoice == ('scouts' if 'later' in choice else 'normal')
    if 'later' in choice:
        assert game.deploymentStage == 'scouts'
        tasks.add.assert_not_called()
    else:
        tasks.add.assert_called_once()


def test_invalid_drop_neither_deploys_nor_advances_and_logs_distance(pair, capsys):
    game, scout, enemy = pair
    game.deploymentStage = 'scouts'
    scout.scoutDeploymentChoice = 'scouts'
    game.unitToMove = scout
    game.checkUnitContactSmall = Mock(return_value=None)
    enemy.bodyNP.setY(13)
    with patch.object(deployPhase, 'taskMgr', Mock(), create=True):
        deployPhase.endMoveUnit(game, 'move')
    assert not scout.isDeployed and not scout.deployedAsScouts
    game.fsm.request.assert_not_called()
    assert '12.000"' in capsys.readouterr().out


def test_direct_auto_charge_refused_before_reactions_or_dice(pair):
    game, scout, _ = pair
    scout.deployedAsScouts = True
    game.autoCharge = True
    game.autoHold = True
    game.startTaskFunction = Mock()
    game.taskLoopPathTowardsMouse = Mock()
    resolver = combat_resolution.CombatResolver(game)
    # No contact object is required: the gate precedes even target lookup.
    result = asyncio.run(resolver.chargeAndChargeReaction(
        scout, None, Point3(0, -5, 0), Vec3(90, 0, 0), NS(done='done')))
    assert result == 'done' and scout.bodyNP.getY() == -5
    assert scout.bodyNP.getH() == 90 and not scout.hasMovedThisTurn
    assert not scout.isChargingMove and not game.autoCharge and not game.autoHold


@pytest.mark.parametrize('tag,key', [('mount', 'mountUnit'), ('crew', 'partUnit'), ('beasts', 'partUnit')])
def test_split_profile_scouts_grant_applies_to_whole_model(pair, tag, key):
    _, scout, non_scout = pair
    non_scout.unit.model.special_rules.append({'tag': tag, key: scout.unit, 'count': 1})
    assert has_scouts(non_scout)
    assert has_scouts(non_scout.unit)
    assert has_scouts(non_scout.unit.model)
    if tag != 'mount':
        non_scout.unit.model.special_rules[-1]['count'] = 0
        assert not has_scouts(non_scout)


def test_preview_queries_do_not_flood_rule_log(pair, capsys):
    game, scout, _ = pair
    scout.deployedAsScouts = True
    for _ in range(10):
        placement_error(game, scout, scouting=True)
        scout_charge_blocked(game, scout)
        scouts_block_vanguard(scout)
    assert capsys.readouterr().out == ''


def test_ai_exhausted_placement_search_can_really_be_taken_over(pair):
    game, scout, _ = pair
    game.roundCounter.current_player = 2
    game.AIplayer2.active = True
    task = NS(_deploy_attempts=200, done='done')
    assert deployPhase.taskMoveUnit(game, scout, task) == 'done'
    assert not game.AIplayer2.active
    game.accept.assert_called_once()


@pytest.mark.parametrize('module_name,class_name', [('ClassAI', 'ClassAI'), ('aiMinimaxIntegration', 'EnhancedAI')])
def test_both_ai_implementations_choose_current_stage_only(pair, module_name, class_name):
    import importlib
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    game, scout, enemy = pair
    enemy.isDeployed = False
    game.roundCounter.current_player = 2
    game.taskLoopDeploy = Mock()
    scout.scoutDeploymentChoice = 'scouts'
    game.player1Units.remove(scout)
    game.player2Units.insert(0, scout)
    ai = NS(game=game)
    with patch.object(module, 'taskMgr', Mock(), create=True):
        cls.deployUnits(ai)
        assert game.unitToMove is enemy
        game.deploymentStage = 'scouts'
        cls.deployUnits(ai)
        assert game.unitToMove is scout