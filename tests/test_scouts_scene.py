"""Real deployment, first-turn lifecycle and reload checks, rendered offscreen."""

import json
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest
from panda3d.core import AsyncTaskManager, getModelPath, loadPrcFileData, Vec2
from direct.task.Task import TaskManager

import aiMinimaxIntegration
import deployPhase
import game as game_module
import game_fsm
from game import MyApp
from persistence import load_game_state, save_game_state
from scouts import (deployment_candidates, has_scouts, model_base_boxes,
                    placement_error, scout_charge_blocked)


def drop(app, name, x, y):
    unit = next(u for u in app.units if u.unitName == name)
    app.unitToMove = unit
    unit.bodyNP.setPos(x, y, 0)
    deployPhase.endMoveUnit(app, 'test-deploy-move')
    assert unit.isDeployed, placement_error(app, unit, scouting=app.deploymentStage == 'scouts')
    return unit


def build_scenario():
    loadPrcFileData('', 'window-type offscreen\nwin-size 1280 720\naudio-library-name null')
    getModelPath().appendDirectory(str(ROOT))

    def add(app, player, name, profile, count):
        assert app._create_unit(dict(name=profile, nmodels=count, files=5,
                                     ranks=(count + 4) // 5), player, name) is not None

    def p1(app, _):
        add(app, 1, 'Normal Rangers', 'Ranger', 5)
        add(app, 1, 'P1 Scouts', 'Ranger', 5)

    def p2(app, _):
        add(app, 2, 'Warriors', 'Dwarf Warrior', 10)
        add(app, 2, 'P2 Scouts A', 'Ranger', 5)
        add(app, 2, 'P2 Scouts B', 'Ranger', 5)

    with patch.object(MyApp, 'load_player1_army', p1), patch.object(MyApp, 'load_player2_army', p2):
        app = MyApp()
    app.terrain_manager.clear()
    for u in app.units:
        u.scoutDeploymentChoice = 'scouts' if 'Scouts' in u.unitName else 'normal'
    drop(app, 'Normal Rangers', -8, -18)
    assert app.roundCounter.current_player == 2
    with patch.object(deployPhase.random, 'randint', side_effect=[6, 2]):
        drop(app, 'Warriors', 8, 18)
    assert app.fsm.state == 'DeployPhase' and app.deploymentStage == 'scouts'
    assert app.scoutDeployFirst == 1 and app.firstFinishedDeploying is None
    app.unitToMove = next(u for u in app.units if u.unitName == 'P1 Scouts')
    app.refreshSelectedUnit()
    return app


@pytest.fixture(scope='module')
def scene(tmp_path_factory):
    app = build_scenario()
    path = tmp_path_factory.mktemp('scouts-scene') / 'deployment.json'
    save_game_state(app, str(path))
    yield app, path
    app.destroy()


def finish_deployment(app):
    p1 = drop(app, 'P1 Scouts', -18, 0)
    assert app.roundCounter.current_player == 2 and app.firstFinishedDeploying == 1
    p2 = drop(app, 'P2 Scouts A', 18, 0)
    assert app.roundCounter.current_player == 2
    drop(app, 'P2 Scouts B', 18, -10)
    assert app.fsm.state == 'StrategyPhase' and app.roundCounter.current_player == 1
    return p1, p2


def test_reload_mid_scouts_restores_order_and_choices_without_reroll(scene, tmp_path):
    app, baseline = scene
    load_game_state(app, str(baseline))
    drop(app, 'P1 Scouts', -18, 0)
    halfway = tmp_path / 'halfway.json'
    save_game_state(app, str(halfway))
    drop(app, 'P2 Scouts A', 18, 0)
    drop(app, 'P2 Scouts B', 18, -10)
    with patch.object(deployPhase.random, 'randint', side_effect=AssertionError('re-rolled on load')):
        load_game_state(app, str(halfway))
    assert app.fsm.state == 'DeployPhase' and app.deploymentStage == 'scouts'
    assert app.roundCounter.current_player == 2 and app.scoutDeployFirst == 1
    assert app.firstFinishedDeploying == 1
    assert [u.unitName for u in deployment_candidates(app, 2)] == ['P2 Scouts A', 'P2 Scouts B']
    assert not deployment_candidates(app, 1)
    assert all(not u.deployedAsScouts for u in deployment_candidates(app, 2))
    app.fsm.nextPhase()
    assert app.fsm.state == 'DeployPhase'


def test_scout_charge_restriction_survives_strategy_and_reload_then_expires_per_owner(scene, tmp_path):
    app, baseline = scene
    load_game_state(app, str(baseline))
    p1, p2 = finish_deployment(app)
    normal = next(u for u in app.units if u.unitName == 'Normal Rangers')
    assert has_scouts(normal) and not scout_charge_blocked(app, normal)
    assert scout_charge_blocked(app, p1) and scout_charge_blocked(app, p2)
    first_turn = tmp_path / 'first-turn.json'
    save_game_state(app, str(first_turn))
    app.fsm.request('CombatPhase')
    app.fsm.request('StrategyPhase')
    assert app.roundCounter.currentRoundPlayer == [1, 0]
    assert not scout_charge_blocked(app, p1) and scout_charge_blocked(app, p2)
    app.fsm.request('CombatPhase')
    app.fsm.request('StrategyPhase')
    assert app.roundCounter.currentRoundPlayer == [1, 1]
    assert not scout_charge_blocked(app, p1) and not scout_charge_blocked(app, p2)
    for u in app.units:
        u.deployedAsScouts = False
    load_game_state(app, str(first_turn))
    assert scout_charge_blocked(app, p1) and scout_charge_blocked(app, p2)
    assert not scout_charge_blocked(app, normal)


def test_loading_old_save_clears_later_scout_history(scene, tmp_path):
    app, baseline = scene
    load_game_state(app, str(baseline))
    p1, p2 = finish_deployment(app)
    old = tmp_path / 'legacy.json'
    save_game_state(app, str(old))
    data = json.loads(old.read_text())
    for key in ('deployment_stage', 'scout_deploy_first', 'first_finished_deploying'):
        data.pop(key)
    for u in data['units']:
        u.pop('scoutDeploymentChoice')
        u.pop('deployedAsScouts')
    old.write_text(json.dumps(data))
    load_game_state(app, str(old))
    assert not scout_charge_blocked(app, p1) and not scout_charge_blocked(app, p2)
    assert all(u.scoutDeploymentChoice is None for u in app.units)
    assert app.scoutDeployFirst is None and app.firstFinishedDeploying is None


def test_same_phase_reload_cancels_old_placement_task(scene):
    app, baseline = scene
    load_game_state(app, str(baseline))
    app.taskMgr.add(lambda task: task.cont, 'taskMoveUnit')
    load_game_state(app, str(baseline))
    assert not app.taskMgr.hasTaskNamed('taskMoveUnit')


def test_actual_movement_into_enemy_refuses_charge_without_spending_move(scene, capsys):
    app, baseline = scene
    load_game_state(app, str(baseline))
    scout, enemy = finish_deployment(app)
    app.fsm.request('MovementPhase')
    app.unitToMove = scout
    origin = scout.bodyNP.getPos()
    heading = scout.bodyNP.getHpr()
    target = enemy.bodyNP.getPos()
    # The movement planner's normalised destination, which moveUnit commits.
    app.arcPoint = Vec2((target.x / 50 + 1) / 2, (target.y / 50 + 1) / 2)
    app.arcPointRotation = 0
    scout.wouldMarch = True
    app.autoCharge = True
    app.autoHold = True
    app.world.doPhysics(0.001)
    with patch.object(app, 'startTaskFunction') as restart, \
            patch.object(app, 'chargeAndChargeReaction') as charge:
        app.movement.moveUnit(scout)
    assert scout.bodyNP.getPos().almostEqual(origin)
    assert scout.bodyNP.getHpr().almostEqual(heading)
    assert not scout.hasMovedThisTurn and not scout.isChargingMove
    assert not scout.marchedThisTurn and not app.autoCharge and not app.autoHold
    assert 'Marching' not in capsys.readouterr().out
    restart.assert_called_once()
    charge.assert_not_called()


def test_loading_ai_scout_turn_restarts_deployment(scene, tmp_path):
    app, baseline = scene
    load_game_state(app, str(baseline))
    drop(app, 'P1 Scouts', -18, 0)
    app.AIplayer2.active = True
    path = tmp_path / 'ai-scouts.json'
    save_game_state(app, str(path))
    with patch.object(app.AIplayer2, 'deployUnits') as resume:
        load_game_state(app, str(path))
    resume.assert_called_once()
    assert app.roundCounter.current_player == 2 and app.deploymentStage == 'scouts'
    app.AIplayer2.active = False


@pytest.mark.parametrize('case', ['ordinary-deployed', 'second-own-turn', 'pursuit'])
def test_scout_charge_gate_does_not_block_allowed_contacts(scene, case):
    app, baseline = scene
    load_game_state(app, str(baseline))
    scout, enemy = finish_deployment(app)
    app.fsm.request('MovementPhase')
    if case == 'ordinary-deployed':
        scout.deployedAsScouts = False
    elif case == 'second-own-turn':
        app.roundCounter.currentRoundPlayer = [1, 0]
    else:
        scout.request('IsPursuing')
    target = enemy.bodyNP.getPos()
    app.arcPoint = Vec2((target.x / 50 + 1) / 2, (target.y / 50 + 1) / 2)
    app.arcPointRotation = 0
    scout.wouldMarch = False
    with patch.object(app.taskMgr, 'add') as schedule:
        app.movement.moveUnit(scout)
    assert scout.isChargingMove
    assert any(call.args[0] == app.chargeAndChargeReaction for call in schedule.call_args_list)


def test_first_turn_scout_can_still_make_ordinary_movement(scene):
    app, baseline = scene
    load_game_state(app, str(baseline))
    scout, _ = finish_deployment(app)
    app.fsm.request('MovementPhase')
    app.arcPoint = Vec2((-17 / 50 + 1) / 2, 0.5)
    app.arcPointRotation = 0
    scout.wouldMarch = False
    app.movement.moveUnit(scout)
    assert scout.bodyNP.getX() == pytest.approx(-17)
    assert scout.hasMovedThisTurn and not scout.isChargingMove


def test_panda_async_reservations_chain_to_first_ai_scout_drop(scene):
    app, baseline = scene
    load_game_state(app, str(baseline))
    app.deploymentStage = 'ordinary'
    app.scoutDeployFirst = None
    app.roundCounter.request('PlayerTwo')
    app.AIplayer2.active = True
    scouts = [u for u in app.player2Units if not u.isDeployed]
    for u in scouts:
        u.scoutDeploymentChoice = None
    # Only the deployment tasks run: the full offscreen app has mouse tasks
    # which require a real window and must not be stepped by this test.
    tasks = TaskManager()
    tasks.mgr = AsyncTaskManager('isolated-scout-deployment')
    with ExitStack() as stack:
        for module in (game_module, deployPhase, aiMinimaxIntegration, game_fsm):
            stack.enter_context(patch.object(module, 'taskMgr', tasks, create=True))
        choose = stack.enter_context(patch.object(app, 'makeChoiceNew', AsyncMock(
            return_value='Deploy as Scouts later')))
        stack.enter_context(patch.object(deployPhase.random, 'randint', side_effect=[2, 6]))
        stack.enter_context(patch.object(deployPhase.random, 'uniform', side_effect=[18, 0]))
        app.AIplayer2.deployUnits()
        for _ in range(12):
            tasks.step()
        assert choose.await_count == 2
        assert app.deploymentStage == 'scouts' and app.scoutDeployFirst == 2
        assert app.roundCounter.current_player == 1
        assert scouts[0].deployedAsScouts and not scouts[1].isDeployed
        assert not tasks.getTasks()
    app.AIplayer2.active = False


@pytest.mark.parametrize('y,allowed', [(-18, True), (-22.8, False)])
def test_joined_scout_character_validates_final_host_and_rolls_back(scene, y, allowed):
    app, baseline = scene
    load_game_state(app, str(baseline))
    host = app._create_unit(dict(name='Dwarf Warrior', nmodels=10, files=5, ranks=2), 1, 'Join Host')
    char = app._create_unit(dict(name='Captain of the Empire', nmodels=1, files=1,
                                 ranks=1, special_rules=['Scouts']), 1, 'Scout Character')
    host.isDeployed = True
    host.bodyNP.setPos(-22, y, 0)
    char.scoutDeploymentChoice = 'scouts'
    char.bodyNP.setPos(host.bodyNP.getPos())
    before = model_base_boxes(host)
    index = app.player1Units.index(char)
    app.unitToMove = char
    deployPhase.endMoveUnit(app, 'join-test')
    assert char.isDeployed is allowed
    assert char.deployedAsScouts is allowed
    if allowed:
        assert host.joinedCharacter is char and char.hostUnit is host
        assert char not in app.player1Units and char in app.units
        assert scout_charge_blocked(app, host)
    else:
        assert host.joinedCharacter is None and char.hostUnit is None
        assert app.player1Units.index(char) == index
        assert char.bodyNP.getParent() == host.bodyNP.getParent()
        assert model_base_boxes(host) == before
        assert char.bodyNP.node() in app.world.getRigidBodies()


if __name__ == '__main__':
    app = build_scenario()
    try:
        path = save_game_state(app, 'scouts.json')
        load_game_state(app, path)
        assert app.deploymentStage == 'scouts' and app.scoutDeployFirst == 1
        app.graphicsEngine.renderFrame()
        app.graphicsEngine.renderFrame()
        app.screenshot(str(ROOT / 'screenshots' / 'scouts.png'), defaultFilename=False)
    finally:
        app.destroy()