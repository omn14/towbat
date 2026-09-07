"""Playable Rallying Cry Command scenario and real-game regressions (p. 175)."""

from contextlib import contextmanager
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from direct.task import Task
from panda3d.core import getModelPath, loadPrcFileData

from characters import join_unit
from choiceFunctions import Choice
import choiceFunctions
from game import MyApp
import game as game_module
import game_fsm
from persistence import load_game_state, save_game_state
from rallying_cry import source_reason, target_reason, use_rallying_cry
from special_rules import apply_rule_keywords
from tests.test_shieldwall_scene import combat_tasks

ROOT = Path(__file__).resolve().parents[1]


def named(app, name):
    return next(unit for unit in app.units if unit.unitName == name)


def build_scenario():
    loadPrcFileData('', 'window-type offscreen\nwin-size 1280 720\naudio-library-name null')
    getModelPath().appendDirectory(str(ROOT))

    def add(app, name, player=1, *, character=False, rules=()):
        count = 1 if character else 5
        unit = app._create_unit(dict(name='Captain of the Empire' if character else 'Dwarf Warrior',
                                    nmodels=count, files=count, ranks=1), player, name)
        assert unit is not None
        apply_rule_keywords(unit.unit.model, ['Close Order', *rules], replace=True)
        unit.unit.model.characteristics['Ld'] = '8' if character else '6'
        unit.unit.model._base_characteristics = dict(unit.unit.model.characteristics)
        return unit

    def player_one(app, _):
        add(app, 'Ready Caller', character=True, rules=['Rallying Cry'])
        add(app, 'Left Veterans', rules=['Veteran'])
        add(app, 'Left Ordinary')
        add(app, 'Veteran Guard', rules=['Veteran'])
        add(app, 'Centre Runners')
        add(app, 'Spent Caller', character=True, rules=['Rallying Cry'])
        add(app, 'Right Runners')
        add(app, 'Out Of Range')

    def player_two(app, _):
        add(app, 'Enemy Runners', player=2)

    with patch.object(MyApp, 'load_player1_army', player_one), \
            patch.object(MyApp, 'load_player2_army', player_two):
        app = MyApp()
    app.AIplayer2.active = False
    app.terrain_manager.clear()
    positions = {
        'Ready Caller': (-22, -10), 'Left Veterans': (-22, -3),
        'Left Ordinary': (-29, -4), 'Veteran Guard': (0, -10),
        'Centre Runners': (0, -3), 'Spent Caller': (22, -10),
        'Right Runners': (22, -3), 'Out Of Range': (0, 17),
        'Enemy Runners': (-10, 10),
    }
    for name, position in positions.items():
        unit = named(app, name)
        unit.bodyNP.setPos(*position, 0)
        unit.bodyNP.setH(0)
    caller = add(app, 'Joined Caller', character=True, rules=['Rallying Cry'])
    assert join_unit(app, caller, named(app, 'Veteran Guard'))
    for unit in app.units:
        unit.isDeployed = True
        unit.scoutDeploymentChoice = 'normal'
        unit.isGeneral = False
        unit.isBSB = False
        unit.request('Idle')
        unit.hasMovedThisTurn = False
        app.movement.alignModelsToHillNormal(unit)
    app.deploymentStage = 'ordinary'
    app.roundCounter.current_player = 1
    app.roundCounter.currentRoundPlayer = [0, 0]
    app.fsm.request('StrategyPhase')
    app.fsm.currentPhaseIndex = 0
    app.roundCounter.enterPlayerOne()
    for name in ('Left Veterans', 'Left Ordinary', 'Centre Runners', 'Right Runners',
                 'Out Of Range', 'Enemy Runners'):
        unit = named(app, name)
        unit.request('IsFleeing')
        unit.attemptedRallyThisTurn = False
        unit.bodyNP.setH(180)
    named(app, 'Spent Caller').usedRallyingCry = True
    app.unitToMove = named(app, 'Ready Caller')
    app.refreshSelectedUnit()
    return app


def verify_scenario(app):
    assert app.fsm.state == 'StrategyPhase'
    assert app.roundCounter.current_player == 1
    assert not app.AIplayer2.active
    assert not app.strategyCommandDone
    assert len(app.units) == 10
    caller = named(app, 'Ready Caller')
    assert source_reason(app, caller) is None
    assert target_reason(app, caller, named(app, 'Left Veterans')) is None
    assert target_reason(app, caller, named(app, 'Left Ordinary')) is None
    assert 'exceeds' in target_reason(app, caller, named(app, 'Out Of Range'))
    assert 'friendly' in target_reason(app, caller, named(app, 'Enemy Runners'))
    assert 'already used' in source_reason(app, named(app, 'Spent Caller'))
    joined = named(app, 'Joined Caller')
    assert named(app, 'Veteran Guard').joinedCharacter is joined
    assert source_reason(app, joined) is None
    assert target_reason(app, joined, named(app, 'Centre Runners')) is None
    assert not joined.unit.model.is_veteran()
    for unit in app.units:
        assert not unit.attemptedRallyThisTurn


@pytest.fixture(scope='module')
def scene(tmp_path_factory):
    app = build_scenario()
    path = save_game_state(app, str(tmp_path_factory.mktemp('rallying-cry') / 'baseline.json'))
    yield app, path
    app.destroy()


def restore(scene):
    app, path = scene
    load_game_state(app, path)
    app.unitToMove = named(app, 'Ready Caller')
    return app


@contextmanager
def strategy_tasks(app):
    with combat_tasks(app) as run, \
            patch.object(game_fsm, 'taskMgr', app.taskMgr, create=True), \
            patch.object(choiceFunctions, 'taskMgr', app.taskMgr, create=True):
        yield run


def capture_nomination(app, path):
    dialogs = []

    def create(*args, **kwargs):
        dialog = Choice(*args, **kwargs)
        dialogs.append(dialog)
        return dialog

    async def capture():
        from rallying_cry import choose_rallying_cry
        action = app.taskMgr.add(choose_rallying_cry(app, named(app, 'Ready Caller')),
                                 'rallyingCryTask')
        await Task.pause(0.1)
        dialog = dialogs[0]
        assert app.awaitingChoice
        app.fsm.nextPhase()
        assert app.fsm.state == 'StrategyPhase' and not app.strategyCommandDone
        dialog._showDetail('1. Left Veterans')
        bottom, _ = dialog.detail.getTightBounds(dialog.panel)
        assert bottom.z > max(button.getZ() + button['frameSize'][3] * button.getSz()
                              for button in dialog.buttons)
        app.graphicsEngine.renderFrame()
        app.graphicsEngine.renderFrame()
        assert app.screenshot(str(path), defaultFilename=False)
        dialog.onCancel()
        await action

    with strategy_tasks(app) as run, patch.object(game_module, 'Choice', side_effect=create):
        run(capture())
    assert not app.awaitingChoice
    assert not named(app, 'Ready Caller').usedRallyingCry


def test_real_nomination_popup_and_phase_lock(scene, tmp_path):
    app = restore(scene)
    capture_nomination(app, tmp_path / 'nomination.png')


def test_loaded_scene_and_render(scene, tmp_path):
    app = restore(scene)
    verify_scenario(app)
    app.graphicsEngine.renderFrame()
    app.graphicsEngine.renderFrame()
    assert app.screenshot(str(tmp_path / 'rallying-cry.png'), defaultFilename=False)


def test_real_selection_offers_only_valid_targets_and_target_veteran(scene):
    app = restore(scene)
    caller, target = named(app, 'Ready Caller'), named(app, 'Left Veterans')

    async def select():
        app.taskLoopStrategy(Task.Task())
        while app.taskMgr.hasTaskNamed('rallyingCryTask'):
            await Task.pause(0.1)

    with strategy_tasks(app) as run, \
            patch.object(app, 'makeChoiceNew', AsyncMock(side_effect=['1. Left Veterans', 'Re-roll'])) as choice, \
            patch.object(app, 'rollLeadershipDice', AsyncMock(side_effect=[[5, 6], [2, 3]])) as dice, \
            patch.object(app, 'freeReformUnit', lambda unit, task: task.done):
        run(select())
    assert choice.call_args_list[0].args[0] == ['1. Left Veterans', '2. Left Ordinary']
    assert choice.call_args_list[0].kwargs['owner'] is caller
    assert choice.call_args_list[1].kwargs['owner'] is target
    assert dice.await_count == 2
    assert caller.usedRallyingCry
    assert target.state == 'Idle'
    assert target.attemptedRallyThisTurn and target.cannotChargeThisTurn
    assert not target.hasMovedThisTurn and app.movedThisTurn(target)
    assert app.unitToMove is caller


def test_rallied_unit_can_move_but_counts_as_moved_for_shooting(scene, tmp_path):
    app = restore(scene)
    caller, target = named(app, 'Ready Caller'), named(app, 'Left Ordinary')
    with strategy_tasks(app) as run, \
            patch.object(app, 'rollLeadershipDice', AsyncMock(return_value=[2, 3])), \
            patch.object(app, 'freeReformUnit', lambda unit, task: task.done):
        run(use_rallying_cry(app, caller, target))
    path = save_game_state(app, str(tmp_path / 'rallied.json'))
    load_game_state(app, path)
    target = named(app, 'Left Ordinary')
    assert target.cannotChargeThisTurn and target.attemptedRallyThisTurn
    assert not target.hasMovedThisTurn and app.movedThisTurn(target)
    with patch.object(target.unit.model, 'cannot_shoot_after_moving', return_value=True):
        assert app.barredByMoveOrShoot(target, {'name': 'test weapon'})
    app.strategyCommandDone = True
    app.fsm.request('MovementPhase')
    app.unitToMove = target
    assert app.movement.redressRanks(target, -1)
    origin = target.bodyNP.getPos()
    with strategy_tasks(app):
        app.pathTowardsMouse(target, origin.x, origin.y - 1)
        app.movement.moveUnit(target)
    assert target.bodyNP.getPos() != origin
    assert target.hasMovedThisTurn
    app.fsm.request('StrategyPhase')
    assert not target.attemptedRallyThisTurn and not target.cannotChargeThisTurn
    assert not app.movedThisTurn(target)


def test_failure_then_fresh_normal_rally_after_command(scene):
    app = restore(scene)
    caller, target = named(app, 'Ready Caller'), named(app, 'Left Ordinary')
    with strategy_tasks(app) as run, \
            patch.object(app, 'rollLeadershipDice', AsyncMock(side_effect=[[5, 6], [2, 3]])) as dice, \
            patch.object(app, 'freeReformUnit', lambda unit, task: task.done):
        run(use_rallying_cry(app, caller, target))
        assert caller.usedRallyingCry
        assert target.state == 'IsFleeing' and not target.attemptedRallyThisTurn
        run(app.rallyUnit(target))
        assert dice.await_count == 1
        app.fsm.nextPhase()
        assert app.fsm.state == 'StrategyPhase' and app.strategyCommandDone
        run(app.rallyUnit(target))
    assert dice.await_count == 2
    assert target.state == 'Idle' and target.attemptedRallyThisTurn


def test_joined_caller_does_not_lend_hosts_veteran_to_target(scene):
    app = restore(scene)
    app.unitToMove = named(app, 'Veteran Guard')
    with strategy_tasks(app) as run, \
            patch.object(app, 'makeChoiceNew', AsyncMock(return_value='1. Centre Runners')) as choice, \
            patch.object(app, 'rollLeadershipDice', AsyncMock(return_value=[5, 6])) as dice:
        from rallying_cry import choose_rallying_cry
        run(choose_rallying_cry(app, app.unitToMove))
    assert named(app, 'Joined Caller').usedRallyingCry
    assert named(app, 'Centre Runners').state == 'IsFleeing'
    assert not named(app, 'Centre Runners').attemptedRallyThisTurn
    choice.assert_awaited_once()
    dice.assert_awaited_once()


def test_spent_and_command_window_survive_reload(scene, tmp_path):
    app = restore(scene)
    caller = named(app, 'Ready Caller')
    caller.usedRallyingCry = True
    app.strategyCommandDone = True
    path = save_game_state(app, str(tmp_path / 'spent.json'))
    caller.usedRallyingCry = False
    app.strategyCommandDone = False
    load_game_state(app, path)
    assert named(app, 'Ready Caller').usedRallyingCry
    assert app.strategyCommandDone
    load_game_state(app, scene[1])
    assert not named(app, 'Ready Caller').usedRallyingCry
    assert not app.strategyCommandDone


def test_legacy_save_does_not_reopen_command(scene, tmp_path):
    app = restore(scene)
    data = json.loads(Path(scene[1]).read_text())
    data.pop('strategy_command_done')
    for unit in data['units']:
        unit.pop('usedRallyingCry')
    path = tmp_path / 'legacy.json'
    path.write_text(json.dumps(data))
    load_game_state(app, str(path))
    assert app.strategyCommandDone
    assert not named(app, 'Spent Caller').usedRallyingCry


def test_spell_return_preserves_spent_use_and_command_state(scene):
    app = restore(scene)
    named(app, 'Ready Caller').usedRallyingCry = True
    app.strategyCommandDone = True
    app.fsm.request('SpellPhase')
    app.fsm.request('StrategyPhase')
    assert named(app, 'Ready Caller').usedRallyingCry
    assert app.strategyCommandDone


def test_next_strategy_resets_used_command_and_failed_rally(scene):
    app = restore(scene)
    named(app, 'Ready Caller').usedRallyingCry = True
    named(app, 'Left Ordinary').attemptedRallyThisTurn = True
    app.strategyCommandDone = True
    app.fsm.request('MovementPhase')
    from rallying_cry import begin_command
    with patch('rallying_cry.begin_command', wraps=begin_command) as begin:
        app.fsm.request('StrategyPhase')
    begin.assert_called_once_with(app)
    assert not named(app, 'Ready Caller').usedRallyingCry
    assert not named(app, 'Left Ordinary').attemptedRallyThisTurn
    assert not app.strategyCommandDone


def test_ai_end_command_resolves_without_mouse_reform(scene):
    app = restore(scene)
    caller, target = named(app, 'Ready Caller'), named(app, 'Enemy Runners')
    app.player1Units.remove(caller)
    app.player2Units.append(caller)
    caller._player = 2
    target.bodyNP.setPos(-22, -3, 0)
    app.roundCounter.current_player = 2
    app.AIplayer2.active = True

    async def advance():
        app.fsm.nextPhase()
        while app.taskMgr.hasTaskNamed('rallyingCryTask'):
            await Task.pause(0.1)

    with strategy_tasks(app) as run, \
            patch.object(app, 'makeChoiceNew', AsyncMock()) as choice, \
            patch.object(app, 'rollLeadershipDice', AsyncMock(return_value=[2, 3])) as dice, \
            patch.object(app, 'freeReformUnit') as reform:
        run(advance())
    assert caller.usedRallyingCry and app.strategyCommandDone
    assert target.state == 'Idle'
    dice.assert_awaited_once()
    choice.assert_not_awaited()
    reform.assert_not_called()


def test_joined_ai_character_is_not_treated_as_human(scene):
    app = restore(scene)
    caller, host = named(app, 'Joined Caller'), named(app, 'Veteran Guard')
    app.player1Units.remove(host)
    app.player2Units.append(host)
    host._player = caller._player = 2
    app.AIplayer2.active = True
    assert caller not in app.player2Units
    assert app.aiControls(caller)
    assert not app.aiControls(named(app, 'Ready Caller'))


if __name__ == '__main__':
    app = build_scenario()
    try:
        path = save_game_state(app, 'rallying_cry.json')
        load_game_state(app, path)
        verify_scenario(app)
        app.unitToMove = named(app, 'Ready Caller')
        app.refreshSelectedUnit()
        app.graphicsEngine.renderFrame()
        app.graphicsEngine.renderFrame()
        assert app.screenshot(str(ROOT / 'screenshots' / 'rallying_cry.png'), defaultFilename=False)
        capture_nomination(app, ROOT / 'screenshots' / 'rallying_cry_choice.png')
        print(f'Verified Rallying Cry save: {path}')
    finally:
        app.destroy()