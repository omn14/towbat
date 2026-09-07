"""Offscreen Veteran integrations and a playable Rally comparison (p. 180)."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from direct.task import Task
from panda3d.core import getModelPath, loadPrcFileData

from characters import join_unit
from game import MyApp
from persistence import load_game_state, save_game_state
from psychology import veteran_available, veteran_counts
from special_rules import apply_rule_keywords
from tests.test_shieldwall_scene import combat_tasks

ROOT = Path(__file__).resolve().parents[1]
CASES = ('Veteran Rally', 'Ordinary Rally', 'Veteran Escort', 'Tied Veterans', 'Veteran Captain')


def build_scenario():
    loadPrcFileData('', 'window-type offscreen\nwin-size 1280 720\naudio-library-name null')
    getModelPath().appendDirectory(str(ROOT))

    def add(app, name, count, veteran, player=1, character=False):
        unit = app._create_unit(dict(name='Captain of the Empire' if character else 'Dwarf Warrior',
                                    nmodels=count, files=min(count, 5), ranks=1), player, name)
        assert unit is not None
        apply_rule_keywords(unit.unit.model, ['Close Order'] + (['Veteran'] if veteran else []),
                            replace=True)
        unit.unit.model.characteristics['Ld'] = '6'
        unit.unit.model._base_characteristics = dict(unit.unit.model.characteristics)
        return unit

    def player_one(app, _):
        add(app, CASES[0], 5, True)
        add(app, CASES[1], 5, False)
        add(app, CASES[2], 5, True)
        add(app, CASES[3], 1, True)
        add(app, CASES[4], 1, True, character=True)

    def player_two(app, _):
        add(app, 'Enemy Line', 5, False, player=2)

    with patch.object(MyApp, 'load_player1_army', player_one), \
            patch.object(MyApp, 'load_player2_army', player_two):
        app = MyApp()
    app.AIplayer2.active = False
    app.terrain_manager.clear()
    for index, unit in enumerate(app.player1Units):
        unit.bodyNP.setPos(-24 + 12 * index, -8, 0)
        unit.bodyNP.setH(180)
    app.player2Units[0].bodyNP.setPos(0, 16, 0)
    app.player2Units[0].bodyNP.setH(180)
    for name in ('Veteran Escort', 'Tied Veterans'):
        host = next(unit for unit in app.units if unit.unitName == name)
        character = add(app, name + ' Ordinary Captain', 1, False, character=True)
        assert join_unit(app, character, host)
    for unit in app.units:
        unit.isDeployed = True
        unit.scoutDeploymentChoice = 'normal'
        unit.isGeneral = False
        unit.isBSB = False
        unit.request('Idle')
        unit.hasMovedThisTurn = False
        app.movement.alignModelsToHillNormal(unit)
    app.deploymentStage = 'ordinary'
    app.fsm.request('StrategyPhase')
    app.fsm.currentPhaseIndex = app.fsm.phases.index('StrategyPhase')
    app.roundCounter.current_player = 1
    app.roundCounter.currentRoundPlayer = [0, 0]
    app.roundCounter.enterPlayerOne()
    for unit in app.player1Units:
        unit.request('IsFleeing')
        unit.attemptedRallyThisTurn = False
    app.unitToMove = app.player1Units[0]
    app.refreshSelectedUnit()
    return app


def verify_scenario(app):
    assert app.fsm.state == 'StrategyPhase'
    assert app.roundCounter.current_player == 1
    assert not app.AIplayer2.active
    assert len(app.units) == 8
    assert [unit.unitName for unit in app.player1Units] == list(CASES)
    for unit, counts in zip(app.player1Units, [(5, 5), (0, 5), (5, 6), (1, 2), (1, 1)]):
        assert veteran_counts(unit) == counts
        assert unit.state == 'IsFleeing'
        assert not unit.attemptedRallyThisTurn
        assert app.psychology.leadership_of(unit) == (6, None)


@pytest.fixture(scope='module')
def scene(tmp_path_factory):
    app = build_scenario()
    path = save_game_state(app, str(tmp_path_factory.mktemp('veteran') / 'baseline.json'))
    yield app, path
    app.destroy()


def restore(scene, index=0):
    app, path = scene
    load_game_state(app, path)
    unit = app.player1Units[index]
    app.unitToMove = unit
    return app, unit


def test_save_restores_keywords_characters_and_rally_state(scene, tmp_path):
    app, unit = restore(scene)
    apply_rule_keywords(unit.unit.model, [], replace=True)
    unit.attemptedRallyThisTurn = True
    load_game_state(app, scene[1])
    verify_scenario(app)
    app.graphicsEngine.renderFrame()
    app.graphicsEngine.renderFrame()
    assert app.screenshot(str(tmp_path / 'veteran-loaded.png'), defaultFilename=False)


@pytest.mark.parametrize('index, eligible', [(0, True), (1, False), (2, True), (3, False), (4, True)])
def test_loaded_rally_uses_majority_and_optional_reroll(scene, index, eligible):
    app, unit = restore(scene, index)
    with combat_tasks(app) as run, \
            patch.object(app, 'rollLeadershipDice', AsyncMock(side_effect=[[5, 6], [2, 3]])) as dice, \
            patch.object(app, 'makeChoiceNew', AsyncMock(return_value='Re-roll')) as choice, \
            patch.object(app, 'freeReformUnit', lambda unit, task: task.done):
        run(app.rallyUnit(unit))
    assert unit.attemptedRallyThisTurn
    assert unit.state == ('Idle' if eligible else 'IsFleeing')
    assert dice.await_count == (2 if eligible else 1)
    assert choice.await_count == int(eligible)


def test_loaded_character_personal_test_cannot_borrow_veteran(scene):
    app, host = restore(scene, 2)
    character = host.joinedCharacter
    assert veteran_available(host)
    assert not veteran_available(character, personal=True)
    assert character.unit.model.characteristics['Ld'] == '6'


def test_casualties_recompute_majority_and_reload_restores_it(scene):
    app, host = restore(scene, 2)
    assert veteran_counts(host) == (5, 6)
    app.movement.removeModelsFromUnit(host, 4)
    assert veteran_counts(host) == (1, 2)
    assert not veteran_available(host)
    load_game_state(app, scene[1])
    assert veteran_available(app.player1Units[2])


def test_real_panic_queue_rerolls_without_moving(scene):
    app, unit = restore(scene)
    unit.request('Idle')
    origin = unit.bodyNP.getPos()

    async def resolve():
        app.psychology.panic_test(unit, cause='Veteran integration')
        while app.psychology._panic_active:
            await Task.pause(0.1)

    with combat_tasks(app) as run, \
            patch('psychology.random.randint', side_effect=[5, 6, 2, 3]) as dice, \
            patch.object(app, 'makeChoiceNew', AsyncMock(return_value='Re-roll')) as choice:
        run(resolve())
    assert dice.call_count == 4
    choice.assert_awaited_once()
    assert unit.panicTestedThisPhase
    assert unit.bodyNP.getPos() == origin
    assert unit.state == 'Idle'
    assert not app.psychology._panic_active


def test_loaded_restraint_failure_rerolls_once(scene):
    app, unit = restore(scene)
    unit.request('InCombat')
    outcome = []

    async def resolve():
        outcome.append(await app.combat.restrainChoice(unit, None, 'overrun'))

    with combat_tasks(app) as run, \
            patch.object(app.combat, 'rollBreakDice', AsyncMock(side_effect=[[5, 6], [2, 3]])) as dice, \
            patch.object(app, 'makeChoiceNew', AsyncMock(side_effect=['Restrain', 'Re-roll'])):
        run(resolve())
    assert outcome == ['restrain']
    assert dice.await_count == 2
    assert unit.state == 'Idle'


if __name__ == '__main__':
    app = build_scenario()
    try:
        path = save_game_state(app, 'veteran.json')
        load_game_state(app, path)
        verify_scenario(app)
        app.unitToMove = app.player1Units[0]
        app.refreshSelectedUnit()
        app.graphicsEngine.renderFrame()
        app.graphicsEngine.renderFrame()
        assert app.screenshot(str(ROOT / 'screenshots' / 'veteran.png'), defaultFilename=False)
        print(f'Verified Veteran save: {path}')
    finally:
        app.destroy()