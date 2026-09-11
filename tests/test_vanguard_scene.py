"""Real Panda3D Vanguard movement and persistence regressions."""

import asyncio
import json
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from direct.task.Task import TaskManager
from panda3d.core import AsyncTaskManager, Vec2, Vec3

import aiMinimaxIntegration
import game as game_module
import game_fsm
import movement_system
from ClassAI import ClassAI
from characters import join_unit
from persistence import load_game_state, save_game_state
from scouts import model_base_boxes, nearest_enemy
from special_rules import apply_rule_keywords
from tests.test_scouts_scene import build_scenario, drop
from vanguard import (begin_vanguard, commit_vanguard_move, in_vanguard,
                      select_vanguard, skip_vanguard, vanguard_candidates,
                      vanguard_charge_blocked)


def build_vanguard_scenario():
    app = build_scenario()
    for unit in app.units:
        apply_rule_keywords(unit.unit.model, ['Vanguard'])
    drop(app, 'P1 Scouts', -18, 0)
    drop(app, 'P2 Scouts A', 18, 0)
    with patch('vanguard.random.randint', side_effect=[3, 3, 6, 2]):
        drop(app, 'P2 Scouts B', 18, -10)
    assert in_vanguard(app) and app.vanguardFirst == 1
    assert app.firstFinishedDeploying == 1
    return app


@pytest.fixture(scope='module')
def scene(tmp_path_factory):
    app = build_vanguard_scenario()
    path = tmp_path_factory.mktemp('vanguard') / 'baseline.json'
    save_game_state(app, str(path))
    yield app, path
    app.destroy()


def restore(scene):
    app, path = scene
    load_game_state(app, str(path))
    return app, next(unit for unit in app.units if unit.unitName == 'Normal Rangers')


def aim(app, unit, distance):
    app.unitToMove = unit
    app.vanguardActive = unit.unitName
    origin = unit.bodyNP.getPos()
    direction = unit.bodyNP.getQuat().getForward()
    target = origin + direction * distance
    if not unit.isSkirmisher:
        target += direction * unit.unitHeight * 0.5
    app.arcPoint = Vec2((target.x / 50 + 1) / 2, (target.y / 50 + 1) / 2)
    app.arcPointRotation = 0
    app.moveArceDistance = distance
    return origin


def test_only_normal_deployed_units_can_vanguard(scene):
    app, _ = restore(scene)
    assert [unit.unitName for unit in vanguard_candidates(app, 1)] == ['Normal Rangers']
    assert [unit.unitName for unit in vanguard_candidates(app, 2)] == ['Warriors']


def test_real_move_alternates_without_spending_first_turn(scene):
    app, unit = restore(scene)
    origin = aim(app, unit, 3)
    assert app.movement.moveUnit(unit)
    assert (unit.bodyNP.getPos() - origin).length() == pytest.approx(3)
    assert unit.madeVanguardMove and unit.vanguardDone
    assert not unit.hasMovedThisTurn and not unit.marchedThisTurn
    assert app.roundCounter.currentRoundPlayer == [0, 0]
    assert app.roundCounter.current_player == 2
    assert in_vanguard(app)
    skip_vanguard(app)
    assert app.fsm.state == 'StrategyPhase' and app.roundCounter.current_player == 1
    assert not unit.hasMovedThisTurn
    assert vanguard_charge_blocked(app, unit)


def test_skip_keeps_first_turn_charge_available(scene):
    app, unit = restore(scene)
    with patch.object(app, 'makeChoiceNew', AsyncMock(return_value='Skip')):
        asyncio.run(select_vanguard(app, unit, SimpleNamespace(done='done')))
    assert unit.vanguardDone and not unit.madeVanguardMove
    assert not vanguard_charge_blocked(app, unit)
    assert app.roundCounter.current_player == 2


def test_commit_rejects_march_distance_and_keeps_active_move(scene):
    app, unit = restore(scene)
    origin = aim(app, unit, 6)
    assert not commit_vanguard_move(app, unit)
    assert unit.bodyNP.getPos().almostEqual(origin)
    assert not unit.vanguardDone and not unit.madeVanguardMove
    assert app.vanguardActive == unit.unitName


def test_real_mouse_preview_caps_at_m_not_march_or_charge(scene):
    app, unit = restore(scene)
    app.unitToMove = unit
    app.vanguardActive = unit.unitName
    origin = unit.bodyNP.getPos()
    app.pathTowardsMouse(unit, origin.x, origin.y + 20)
    assert app.moveArceDistance == pytest.approx(3)
    assert not unit.wouldMarch
    assert app.movement.moveUnit(unit)
    assert (unit.bodyNP.getPos() - origin).length() == pytest.approx(3)


def test_reload_keeps_order_and_move_history_without_reroll(scene, tmp_path):
    app, unit = restore(scene)
    aim(app, unit, 2)
    assert commit_vanguard_move(app, unit)
    path = tmp_path / 'halfway.json'
    save_game_state(app, str(path))
    skip_vanguard(app)
    with patch('vanguard.random.randint', side_effect=AssertionError('rerolled')):
        load_game_state(app, str(path))
    assert in_vanguard(app) and app.vanguardFirst == 1
    assert app.roundCounter.current_player == 2
    assert unit.madeVanguardMove and unit.vanguardDone
    assert vanguard_charge_blocked(app, unit)


def test_skirmishers_leave_non_vanguard_character_at_original_position(scene):
    app, unit = restore(scene)
    character = app._create_unit(dict(name='Captain of the Empire', nmodels=1,
                                     files=1, ranks=1), 1, 'Non Vanguard Character')
    assert join_unit(app, character, unit)
    position = character.bodyNP.getPos(app.render)
    origin = aim(app, unit, 3)
    assert commit_vanguard_move(app, unit)
    assert character.bodyNP.getPos(app.render).almostEqual(position)
    assert (unit.bodyNP.getPos() - origin).length() == pytest.approx(3)
    assert character.hostUnit is None and unit.joinedCharacter is None
    assert character in app.player1Units and character in app.units
    assert character.bodyNP.node() in app.world.getRigidBodies()
    assert not character.madeVanguardMove and not character.hasMovedThisTurn


def test_first_own_turn_gate_then_expiry(scene):
    app, unit = restore(scene)
    aim(app, unit, 2)
    assert commit_vanguard_move(app, unit)
    skip_vanguard(app)
    origin = unit.bodyNP.getPos()
    unit.bodyNP.setY(0)
    with patch.object(app, 'startTaskFunction'):
        asyncio.run(app.combat.chargeAndChargeReaction(
            unit, None, origin, Vec3(0, 0, 0), SimpleNamespace(done='done')))
    assert unit.bodyNP.getPos().almostEqual(origin)
    assert not unit.isChargingMove and not unit.hasMovedThisTurn
    app.fsm.request('CombatPhase')
    app.fsm.request('StrategyPhase')
    assert not vanguard_charge_blocked(app, unit)


def test_redress_counts_as_vanguard_and_reload_keeps_spent_allowance(scene, tmp_path):
    app, _ = restore(scene)
    skip_vanguard(app)
    unit = vanguard_candidates(app, 2)[0]
    app.unitToMove = unit
    old_files = unit.unit.files
    app.redressRanks(1)
    assert unit.unit.files == old_files
    app.vanguardActive = unit.unitName
    app.redressRanks(1)
    assert unit.unit.files == old_files + 1
    assert unit.moveSpentThisTurn == pytest.approx(1.5)
    assert unit.madeVanguardMove and not unit.vanguardDone
    path = tmp_path / 'redress.json'
    save_game_state(app, str(path))
    load_game_state(app, str(path))
    assert app.vanguardActive == unit.unitName
    assert unit.moveSpentThisTurn == pytest.approx(1.5)
    assert unit.manoeuvreThisTurn == 'Redress the Ranks'
    assert app.taskMgr.hasTaskNamed('taskLoopPathTowardsMouse')
    aim(app, unit, 2)
    assert not commit_vanguard_move(app, unit)
    aim(app, unit, 1)
    assert commit_vanguard_move(app, unit)
    assert app.fsm.state == 'StrategyPhase'
    assert unit.moveSpentThisTurn == 0 and unit.manoeuvreThisTurn is None
    assert not unit.hasMovedThisTurn and vanguard_charge_blocked(app, unit)


def test_end_phase_after_redress_keeps_charge_ban(scene):
    app, _ = restore(scene)
    skip_vanguard(app)
    unit = vanguard_candidates(app, 2)[0]
    app.unitToMove = unit
    app.vanguardActive = unit.unitName
    app.redressRanks(1)
    app.fsm.nextPhase()
    assert app.fsm.state == 'StrategyPhase'
    assert vanguard_charge_blocked(app, unit)
    assert unit.moveSpentThisTurn == 0 and not unit.hasMovedThisTurn


def test_active_unit_cannot_be_switched_during_vanguard(scene):
    app, unit = restore(scene)
    app.unitToMove = unit
    app.vanguardActive = unit.unitName
    with patch.object(app, 'startTaskFunction') as start:
        asyncio.run(app.setActiveUnit(app.taskLoopDeploy, 'taskLoopDeploy'))
    start.assert_not_called()
    assert app.unitToMove is unit


def test_formed_backwards_preview_uses_half_movement(scene):
    app, _ = restore(scene)
    skip_vanguard(app)
    unit = vanguard_candidates(app, 2)[0]
    unit.bodyNP.setH(180)
    app.unitToMove = unit
    app.vanguardActive = unit.unitName
    origin = unit.bodyNP.getPos()
    app.pathTowardsMouse(unit, origin.x, origin.y + 5)
    assert app.movement.moveUnit(unit)
    assert (unit.bodyNP.getPos() - origin).length() == pytest.approx(1.5)
    assert unit.bodyNP.getH() == pytest.approx(180)


@pytest.mark.parametrize('offset,expected', [(Vec3(10, 0, 0), 1.5), (Vec3(0, -10, 0), 3)])
def test_formed_sideways_and_forward_caps(scene, offset, expected):
    app, _ = restore(scene)
    skip_vanguard(app)
    unit = vanguard_candidates(app, 2)[0]
    unit.bodyNP.setH(180)
    app.vanguardActive = unit.unitName
    origin = unit.bodyNP.getPos()
    target = origin + offset
    app.pathTowardsMouse(unit, target.x, target.y)
    assert app.movement.moveUnit(unit)
    assert (unit.bodyNP.getPos() - origin).length() == pytest.approx(expected, abs=1e-5)
    assert unit.bodyNP.getH() == pytest.approx(180)


@pytest.mark.parametrize('gap,allowed', [(0.5, False), (1.1, True)])
def test_enemy_clearance_is_one_inch_not_twelve(scene, gap, allowed):
    app, unit = restore(scene)
    enemy = next(unit for unit in app.units if unit.unitName == 'Warriors')
    front = max(box[1] + box[3] for box in model_base_boxes(unit))
    rear = min(box[1] - box[3] for box in model_base_boxes(enemy)) - enemy.bodyNP.getY()
    enemy.bodyNP.setPos(unit.bodyNP.getX(), front + 3 + gap - rear, 0)
    origin = aim(app, unit, 3)
    assert commit_vanguard_move(app, unit) is allowed
    if allowed:
        assert nearest_enemy(app, unit)[0] == pytest.approx(gap, abs=1e-5)
    else:
        assert unit.bodyNP.getPos().almostEqual(origin)
        assert not unit.madeVanguardMove and not unit.isChargingMove


def test_invalid_move_restores_joined_character(scene):
    app, unit = restore(scene)
    character = app._create_unit(dict(name='Captain of the Empire', nmodels=1,
                                     files=1, ranks=1), 1, 'Stationary Character')
    assert join_unit(app, character, unit)
    position = character.bodyNP.getPos(app.render)
    origin = aim(app, unit, 6)
    assert not commit_vanguard_move(app, unit)
    assert unit.bodyNP.getPos().almostEqual(origin)
    assert character.bodyNP.getPos(app.render).almostEqual(position)
    assert unit.joinedCharacter is character and character.hostUnit is unit
    assert character not in app.player1Units
    assert character.bodyNP.node() not in app.world.getRigidBodies()


def test_detached_character_stays_independent_after_reload(scene, tmp_path):
    app, unit = restore(scene)
    character = app._create_unit(dict(name='Captain of the Empire', nmodels=1,
                                     files=1, ranks=1), 1, 'Stationary Character')
    assert join_unit(app, character, unit)
    position = character.bodyNP.getPos(app.render)
    aim(app, unit, 3)
    assert commit_vanguard_move(app, unit)
    host_boxes = model_base_boxes(unit)
    path = tmp_path / 'detached.json'
    save_game_state(app, str(path))
    load_game_state(app, str(path))
    assert character.bodyNP.getPos(app.render).almostEqual(position)
    assert unit.joinedCharacter is None and character.hostUnit is None
    assert character in app.player1Units and not character.madeVanguardMove
    assert model_base_boxes(unit) == host_boxes


@pytest.mark.parametrize('ai_class', [ClassAI, aiMinimaxIntegration.EnhancedAI])
def test_ai_resolves_two_vanguards_sequentially_with_real_panda_tasks(scene, ai_class):
    app, _ = restore(scene)
    skip_vanguard(app)
    extra = app._create_unit(dict(name='Dwarf Warrior', nmodels=5, files=5,
                                 ranks=1, special_rules=['Vanguard']), 2, 'Extra Vanguard')
    extra.isDeployed = True
    extra.bodyNP.setPos(-8, 18, 0)
    candidates = vanguard_candidates(app, 2)
    for unit in candidates:
        unit.bodyNP.setH(180)
    origins = {unit.unitName: unit.bodyNP.getPos() for unit in candidates}
    tasks = TaskManager()
    tasks.mgr = AsyncTaskManager('isolated-vanguard')
    app.AIplayer2.active = True
    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(app, 'taskMgr', tasks))
            for module in (game_module, game_fsm, aiMinimaxIntegration, movement_system):
                stack.enter_context(patch.object(module, 'taskMgr', tasks, create=True))
            choose = stack.enter_context(patch.object(app, 'makeChoiceNew', AsyncMock(return_value='Move')))
            ai_class.deployUnits(app.AIplayer2)
            for _ in range(12):
                tasks.step()
            assert choose.await_count == 2
            assert app.fsm.state == 'StrategyPhase'
            assert not tasks.getTasks()
        for unit in candidates:
            assert unit.madeVanguardMove and unit.vanguardDone
            assert (unit.bodyNP.getPos() - origins[unit.unitName]).length() == pytest.approx(3, abs=1e-5)
    finally:
        app.AIplayer2.active = False
        tasks.removeTasksMatching('*')


def test_old_save_clears_stale_vanguard_history(scene, tmp_path):
    app, unit = restore(scene)
    aim(app, unit, 3)
    assert commit_vanguard_move(app, unit)
    skip_vanguard(app)
    path = tmp_path / 'old.json'
    save_game_state(app, str(path))
    data = json.loads(path.read_text())
    data.pop('vanguard_first')
    data.pop('vanguard_active')
    for saved in data['units']:
        saved.pop('madeVanguardMove')
        saved.pop('vanguardDone')
    path.write_text(json.dumps(data))
    load_game_state(app, str(path))
    assert not unit.madeVanguardMove and not unit.vanguardDone
    assert app.vanguardFirst is None and app.vanguardActive is None


def test_zero_move_does_not_detach_or_bar_character(scene):
    app, unit = restore(scene)
    character = app._create_unit(dict(name='Captain of the Empire', nmodels=1,
                                     files=1, ranks=1), 1, 'Stationary Character')
    assert join_unit(app, character, unit)
    aim(app, unit, 0)
    assert commit_vanguard_move(app, unit)
    assert unit.joinedCharacter is character and character.hostUnit is unit
    assert not unit.madeVanguardMove and not character.madeVanguardMove


def test_wheel_is_allowed_but_not_after_redress(scene):
    app, _ = restore(scene)
    skip_vanguard(app)
    unit = vanguard_candidates(app, 2)[0]
    unit.bodyNP.setH(180)
    app.vanguardActive = unit.unitName
    origin = unit.bodyNP.getPos()
    app.pathTowardsMouse(unit, origin.x + 6, origin.y - 6)
    assert abs(app.arcPointRotation) > 0
    assert app.movement.moveUnit(unit)
    assert unit.madeVanguardMove
    app, _ = restore(scene)
    skip_vanguard(app)
    unit = vanguard_candidates(app, 2)[0]
    unit.bodyNP.setH(180)
    app.unitToMove = unit
    app.vanguardActive = unit.unitName
    app.redressRanks(1)
    origin = unit.bodyNP.getPos()
    app.pathTowardsMouse(unit, origin.x + 6, origin.y - 6)
    assert not app.movement.moveUnit(unit)
    assert unit.bodyNP.getPos().almostEqual(origin)
    assert unit.moveSpentThisTurn == 1.5


def test_actual_first_turn_charge_contact_is_blocked(scene, capsys):
    app, unit = restore(scene)
    aim(app, unit, 3)
    assert commit_vanguard_move(app, unit)
    skip_vanguard(app)
    app.fsm.request('MovementPhase')
    origin = unit.bodyNP.getPos()
    enemy = next(unit for unit in app.units if unit.unitName == 'Warriors')
    target = enemy.bodyNP.getPos()
    app.arcPoint = Vec2((target.x / 50 + 1) / 2, (target.y / 50 + 1) / 2)
    app.arcPointRotation = 0
    unit.wouldMarch = True
    app.autoCharge = app.autoHold = True
    with patch.object(app, 'startTaskFunction') as restart, patch.object(app, 'chargeAndChargeReaction') as charge:
        app.movement.moveUnit(unit)
    assert unit.bodyNP.getPos().almostEqual(origin)
    assert not unit.hasMovedThisTurn and not unit.marchedThisTurn and not unit.isChargingMove
    assert not app.autoCharge and not app.autoHold
    restart.assert_called_once()
    charge.assert_not_called()
    output = capsys.readouterr().out
    assert 'Vanguard' in output and 'charge declaration refused' in output
    assert 'Marching' not in output


def test_loading_active_ai_vanguard_schedules_once(scene, tmp_path):
    app, _ = restore(scene)
    skip_vanguard(app)
    unit = vanguard_candidates(app, 2)[0]
    app.vanguardActive = unit.unitName
    app.AIplayer2.active = True
    path = tmp_path / 'ai-active.json'
    save_game_state(app, str(path))
    try:
        with patch('vanguard.random.randint', side_effect=AssertionError('rerolled')), \
                patch.object(app.AIplayer2, 'deployUnits') as duplicate:
            load_game_state(app, str(path))
        duplicate.assert_not_called()
        assert app.vanguardActive == unit.unitName
        assert len(app.taskMgr.getTasksNamed('taskLoopDeploy')) == 1
        assert app.unitToMove is unit
    finally:
        app.taskMgr.remove('taskLoopDeploy')
        app.AIplayer2.active = False


def test_end_phase_cannot_resolve_unit_while_choice_is_open(scene):
    app, unit = restore(scene)
    app.awaitingChoice = True
    try:
        app.fsm.nextPhase()
        assert in_vanguard(app) and not unit.vanguardDone
        assert app.roundCounter.current_player == 1
    finally:
        app.awaitingChoice = False


def test_formed_mixed_character_is_not_selectable_for_vanguard(scene):
    app, _ = restore(scene)
    skip_vanguard(app)
    unit = vanguard_candidates(app, 2)[0]
    character = app._create_unit(dict(name='Captain of the Empire', nmodels=1,
                                     files=1, ranks=1), 2, 'Formed Character')
    assert join_unit(app, character, unit)
    with patch.object(app, 'makeChoiceNew', AsyncMock()) as choose:
        asyncio.run(select_vanguard(app, unit, SimpleNamespace(done='done')))
    choose.assert_not_awaited()
    assert not unit.madeVanguardMove and not character.madeVanguardMove


def test_render_vanguard_selection_offscreen(scene, tmp_path):
    app, unit = restore(scene)
    app.unitToMove = unit
    app.refreshSelectedUnit()
    app.graphicsEngine.renderFrame()
    app.graphicsEngine.renderFrame()
    assert app.screenshot(str(tmp_path / 'vanguard.png'), defaultFilename=False)


def test_drilled_is_offered_once_when_vanguard_movement_begins(scene):
    app, unit = restore(scene)
    with patch.object(app, 'makeChoiceNew', AsyncMock(return_value='Move')), \
            patch('drilled.before_move', AsyncMock(return_value=None)) as redress:
        asyncio.run(select_vanguard(app, unit, SimpleNamespace(done='done')))
        asyncio.run(select_vanguard(app, unit, SimpleNamespace(done='done')))
    redress.assert_awaited_once_with(app, unit, 'Vanguard')


if __name__ == '__main__':
    app = build_vanguard_scenario()
    try:
        for unit in app.player2Units:
            unit.bodyNP.setH(180)
        app.AIplayer2.active = False
        skirmishers = next(unit for unit in app.units if unit.unitName == 'Normal Rangers')
        ranked = app._create_unit(dict(name='Dwarf Warrior', nmodels=10, files=5,
                                      ranks=2, special_rules=['Vanguard']),
                                  1, 'P1 Ranked Vanguard')
        blocked = app._create_unit(dict(name='Dwarf Warrior', nmodels=10, files=5,
                                       ranks=2, special_rules=['Vanguard']),
                                   1, 'P1 Blocked Vanguard')
        assert ranked is not None and blocked is not None
        for unit, position in ((ranked, (-24, -18, 0)), (blocked, (8, -18, 0))):
            unit.isDeployed = True
            unit.scoutDeploymentChoice = 'normal'
            unit.bodyNP.setPos(*position)
        for host, name in ((skirmishers, 'P1 Leave Behind Character'),
                           (blocked, 'P1 Blocking Character')):
            character = app._create_unit(dict(name='Captain of the Empire', nmodels=1,
                                             files=1, ranks=1), 1, name)
            assert character is not None and join_unit(app, character, host)
        path = save_game_state(app, 'vanguard.json')
        with patch('vanguard.random.randint', side_effect=AssertionError('rerolled on load')):
            load_game_state(app, path)
        assert in_vanguard(app) and app.vanguardFirst == 1
        assert app.roundCounter.current_player == 1
        assert app.roundCounter.currentRoundPlayer == [0, 0]
        assert not app.AIplayer2.active and app.vanguardActive is None
        assert {unit.unitName for unit in vanguard_candidates(app, 1)} == {
            'Normal Rangers', 'P1 Ranked Vanguard'}
        assert [unit.unitName for unit in vanguard_candidates(app, 2)] == ['Warriors']
        assert not any(unit.madeVanguardMove or unit.vanguardDone for unit in app.units)
        assert skirmishers.joinedCharacter.unitName == 'P1 Leave Behind Character'
        assert blocked.joinedCharacter.unitName == 'P1 Blocking Character'
        app.unitToMove = skirmishers
        app.refreshSelectedUnit()
        app.graphicsEngine.renderFrame()
        app.graphicsEngine.renderFrame()
        screenshot = Path(__file__).resolve().parents[1] / 'screenshots' / 'vanguard.png'
        assert app.screenshot(str(screenshot), defaultFilename=False)
        print(f'Verified Vanguard test save: {path}')
    finally:
        app.destroy()