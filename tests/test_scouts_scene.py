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


@pytest.mark.parametrize('width,depth', [(44, 30), (48, 36)])
def test_battle_march_geometry_reload_and_offscreen_lines(scene, tmp_path, width, depth):
    from panda3d.core import OrthographicLens, PNMImage, Point3
    from battle_config import load_config
    from battle_setup import resolve_setup, restore_battle
    from battle_terrain import footprint
    from shapely.geometry import box
    app, baseline = scene
    load_game_state(app, str(baseline))
    config = load_config()
    config['battlefield'].update(width=width, depth=depth)
    config['deployment'].update(map='close_encounter', mirror=True)
    setup = resolve_setup(config, 14)
    setup['player_zones'] = {'1': 2, '2': 1}
    saved = tmp_path / 'battle-march.json'
    lens, camera_transform = app.cam.node().getLens(), app.camera.getTransform()
    try:
        restore_battle(app, {'config': config, 'setup': setup})
        hill = app.terrain_manager.add_terrain('hill', Point3(-16, -10, 0), 6, 4)
        actual = footprint(hill)
        assert actual.area > 1
        assert actual.area < box(-19, -12, -13, -8).area
        save_game_state(app, str(saved))
        load_game_state(app, str(baseline))
        with patch('battle_setup.Random', side_effect=AssertionError('setup rerolled during load')):
            load_game_state(app, str(saved))
        assert app.battlefield.width == width and app.battlefield.depth == depth
        assert app.battle_setup == setup
        assert app.deploymentLine.isHidden()
        assert app.boundries.eastBoundry.getX() == width / 2 + 5
        assert app.boundary_np.getCollideMask().isZero()
        top_down = OrthographicLens()
        top_down.setFilmSize(80, 45)
        app.cam.node().setLens(top_down)
        app.camera.setPos(0, 0, 100)
        app.camera.lookAt(0, 0, 0)
        app.aspect2d.hide()
        app.graphicsEngine.renderFrame()
        app.graphicsEngine.renderFrame()
        image = PNMImage()
        assert app.win.getScreenshot(image)
        cyan = red = 0
        for horizontal in range(image.getXSize()):
            for vertical in range(image.getYSize()):
                color = image.getXel(horizontal, vertical)
                cyan += color.y > .65 and color.z > .8 and color.x < .3
                red += color.x > .8 and .2 < color.y < .5 and color.z < .4
        assert cyan > 100 and red > 100
        assert image.write(str(ROOT / '.pytest_cache' / f'battle_march_{width}x{depth}.png'))
    finally:
        app.cam.node().setLens(lens)
        app.camera.setTransform(camera_transform)
        app.aspect2d.show()
        load_game_state(app, str(baseline))


def test_battle_march_turn_awards_survive_real_reload(scene, tmp_path):
    import asyncio
    from battle_config import load_config
    from battle_objectives import finish_player_turn
    from battle_setup import resolve_setup, restore_battle
    app, baseline = scene
    load_game_state(app, str(baseline))
    config = load_config()
    config['objectives']['layout'] = 'two_troves'
    restore_battle(app, {'config': config, 'setup': resolve_setup(config, 12)})
    try:
        unit = next(member for member in app.units if member.unitName == 'Normal Rangers')
        unit.bodyNP.setPos(0, -7.5, 0)
        app.fsm.request('CombatPhase')
        app.roundCounter.enterPlayerTwo()
        with patch('spell_effects.end_turn') as expiry:
            asyncio.run(finish_player_turn(app.fsm, 'StrategyPhase'))
        expiry.assert_called_once_with(app)
        assert app.battle_awards[0]['player'] == 1
        assert app.battle_awards[0]['points'] == 10
        assert app.battle_scored_turns == ['2:0:0']
        assert app.roundCounter.currentRoundPlayer == [0, 1]
        saved = tmp_path / 'scored-turn.json'
        save_game_state(app, str(saved))
        load_game_state(app, str(saved))
        assert len(app.battle_awards) == 1
        assert app.battle_scored_turns == ['2:0:0']
    finally:
        load_game_state(app, str(baseline))


def test_frenzy_loss_and_follow_up_survive_reload(scene, tmp_path):
    from frenzy import has_frenzy, lose_frenzy
    app, baseline = scene
    load_game_state(app, str(baseline))
    unit = next(member for member in app.units if member.unitName == 'Normal Rangers')
    names = list(unit.unit.model.characteristics.get('Special Rules', []))
    app.applyDataRules(unit.unit.model, [*names, 'Frenzy'], replace=True)
    assert has_frenzy(unit.unit.model)
    lose_frenzy(unit)
    unit.frenzyFollowUpThisTurn = True
    path = tmp_path / 'frenzy-loss.json'
    try:
        save_game_state(app, str(path))
        load_game_state(app, str(path))
        assert not has_frenzy(unit.unit.model)
        assert unit.frenzyFollowUpThisTurn
        assert not unit.frenzyFollowUpNextTurn
    finally:
        load_game_state(app, str(baseline))


def test_frenzy_compulsory_charge_has_no_leadership_test(scene):
    import asyncio
    from impetuous import complete_declarations, legal_targets
    app, baseline = scene
    load_game_state(app, str(baseline))
    attacker = next(member for member in app.units if member.unitName == 'Normal Rangers')
    target = next(member for member in app.units if member.unitName == 'Warriors')
    try:
        attacker.unit.model.special_rules.append({'name': 'Frenzy'})
        attacker.bodyNP.setPos(0, -5, 0)
        target.bodyNP.setPos(0, 3, 0)
        app.fsm.request('MovementPhase')
        app.roundCounter.enterPlayerOne()
        attacker.request('Idle')
        attacker.bodyNP.setH(0)
        attacker.hasMovedThisTurn = False
        attacker.moveSpentThisTurn = 0
        attacker.cannotChargeThisTurn = False
        assert any(candidate[0] is target for candidate in legal_targets(app, attacker))
        with patch.object(app, 'rollLeadershipDice', AsyncMock(side_effect=AssertionError('Frenzy cannot test to avoid charging'))):
            asyncio.run(complete_declarations(app))
        assert len(app.chargeDeclarations) == 1
        assert app.chargeDeclarations[0].charger is attacker
        assert app.chargeDeclarations[0].defender is target
        assert app.chargeDeclarations[0].compulsory
    finally:
        load_game_state(app, str(baseline))


@pytest.mark.parametrize('property_name', ['magic_resistance', 'frenzy', 'stubborn'])
def test_landmark_grant_reload_and_next_turn_expiry(scene, tmp_path, property_name):
    from battle_config import load_config
    from battle_setup import resolve_setup, restore_battle
    from battle_objectives import score_turn
    from frenzy import lose_frenzy
    app, baseline = scene
    load_game_state(app, str(baseline))
    config = load_config()
    config['objectives'].update(layout='landmark', landmark_property=property_name)
    restore_battle(app, {'config': config, 'setup': resolve_setup(config, 12)})
    unit = next(member for member in app.units if member.unitName == 'Normal Rangers')
    unit.bodyNP.setPos(0, -3.5, 0)
    try:
        assert score_turn(app)[0]['points'] == 25
        grants = lambda: [rule for rule in unit.unit.model.special_rules
                          if rule.get('battle_march_source')]
        assert len(grants()) == 1
        if property_name == 'frenzy':
            lose_frenzy(unit)
            assert grants()[0]['frenzy_lost']
        saved = tmp_path / f'landmark-{property_name}.json'
        save_game_state(app, str(saved))
        load_game_state(app, str(saved))
        assert len(grants()) == 1
        pieces = [piece for piece in app.terrain_manager.terrain_pieces if getattr(piece, 'objective_id', None)]
        assert len(pieces) == 1
        landmark = pieces[0]
        from panda3d.core import Point3
        assert landmark.is_impassable and landmark.blocks_line_of_sight
        assert landmark.contains(Point3(0, 0, 0))
        assert not landmark.contains(Point3(1.8, 1.8, 0))
        assert landmark.ghost_np is not None
        assert '25 VP' in app.hud._objectives_text.getText()
        assert 'Landmark: P1' in app.hud._objectives_text.getText()
        assert len(app.hud.snapshot()['objectives'][0]) == 1
        assert app.terrain_manager.los_block_point(Point3(-6, 0, 20), Point3(6, 0, 20)) is not None
        assert app.terrain_manager.los_block_point(Point3(-4, 7, 20), Point3(7, -4, 20)) is None
        if property_name == 'frenzy':
            assert grants()[0]['frenzy_lost']
        assert score_turn(app) == []
        unit.bodyNP.setPos(0, -12, 0)
        app.roundCounter.enterPlayerTwo()
        assert score_turn(app) == []
        assert grants() == []
    finally:
        load_game_state(app, str(baseline))


@pytest.mark.parametrize('winner,choice,first', [(1, 'Take second turn', 2), (2, 'Take second turn', 1),
                                               (2, 'Take first turn', 2)])
def test_battle_march_first_turn_choice_is_separate_and_saved(scene, tmp_path, winner, choice, first):
    import asyncio
    from battle_config import load_config
    from battle_setup import resolve_setup, restore_battle, choose_first_turn
    app, baseline = scene
    load_game_state(app, str(baseline))
    config = load_config()
    restore_battle(app, {'config': config, 'setup': resolve_setup(config, 12)})
    rolls = [2, 2, 6, 1] if winner == 1 else [2, 2, 1, 6]
    try:
        with patch('battle_setup.random.randint', side_effect=rolls), \
                patch.object(app, 'aiControls', return_value=False), \
                patch.object(app, 'makeChoiceNew', AsyncMock(return_value=choice)):
            asyncio.run(choose_first_turn(app))
        assert app.fsm.state == 'StrategyPhase'
        assert app.roundCounter.current_player == first
        assert app.roundCounter.currentRoundPlayer == [0, 0]
        assert app.roundCounter.max_rounds == 5
        assert app.battle_setup['first_turn']['winner'] == winner
        saved = tmp_path / 'first-turn.json'
        save_game_state(app, str(saved))
        with patch('battle_setup.random.randint', side_effect=AssertionError('load must not reroll')):
            load_game_state(app, str(saved))
        assert app.roundCounter.current_player == first
        assert app.battle_setup['first_turn']['player'] == first
    finally:
        load_game_state(app, str(baseline))


def test_battle_march_pending_first_turn_choice_reloads_without_dice(scene, tmp_path):
    import asyncio
    from direct.task import Task
    from battle_config import load_config
    from battle_setup import resolve_setup, restore_battle, choose_first_turn
    from tests.test_shieldwall_scene import combat_tasks
    app, baseline = scene
    load_game_state(app, str(baseline))
    config = load_config()
    restore_battle(app, {'config': config, 'setup': resolve_setup(config, 12)})
    pending = tmp_path / 'pending-first-turn.json'

    async def save_choice(*args, **kwargs):
        assert save_game_state(app, str(pending)) == str(pending)
        return 'Take first turn'

    async def finish_pending():
        for unused in range(20):
            await Task.pause(.1)
            if app.fsm.state == 'StrategyPhase':
                break
        assert app.fsm.state == 'StrategyPhase'

    try:
        with patch('battle_setup.random.randint', side_effect=[6, 1]), \
                patch.object(app, 'aiControls', return_value=False), \
                patch.object(app, 'makeChoiceNew', side_effect=save_choice):
            asyncio.run(choose_first_turn(app))
        assert app.roundCounter.current_player == 1
        with combat_tasks(app) as run, \
                patch('battle_setup.random.randint', side_effect=AssertionError('pending choice rerolled')), \
                patch.object(app, 'aiControls', return_value=False), \
                patch.object(app, 'makeChoiceNew', AsyncMock(return_value='Take second turn')):
            load_game_state(app, str(pending))
            run(finish_pending())
        assert app.roundCounter.current_player == 2
        assert app.battle_setup['first_turn']['rolls'] == [[6, 1]]
    finally:
        load_game_state(app, str(baseline))


@pytest.mark.parametrize('first', [1, 2])
def test_battle_march_five_rounds_score_queued_boundaries_once(scene, tmp_path, first):
    from direct.task import Task
    from battle_config import load_config
    from battle_setup import resolve_setup, restore_battle
    from tests.test_shieldwall_scene import combat_tasks
    app, baseline = scene
    load_game_state(app, str(baseline))
    config = load_config()
    config['objectives']['layout'] = 'two_troves'
    setup = resolve_setup(config, 12)
    setup['first_turn'] = {'rolls': [[6, 1]], 'winner': 1, 'player': first}
    restore_battle(app, {'config': config, 'setup': setup})
    unit = next(member for member in app.units if member.unitName == 'Normal Rangers')
    unit.bodyNP.setPos(0, -7.5, 0)
    app.fsm.request('CombatPhase')
    app.roundCounter.request('PlayerOne' if first == 1 else 'PlayerTwo')

    async def advance():
        app.fsm.request('StrategyPhase')
        app.fsm.request('StrategyPhase')
        assert app.fsm.state == 'CombatPhase' and app.battleMarchBoundaryBusy
        for unused in range(20):
            await Task.pause(.1)
            if not app.battleMarchBoundaryBusy:
                break
        assert not app.battleMarchBoundaryBusy

    try:
        with patch('spell_effects.end_turn') as expiry, patch('rallying_cry.begin_command'):
            for turn in range(10):
                if turn:
                    app.fsm.request('CombatPhase')
                with combat_tasks(app) as run:
                    run(advance())
                assert len(app.battle_scored_turns) == turn + 1
                assert len(app.battle_awards) == turn + 1
                assert app.fsm.state == ('BattleEnded' if turn == 9 else 'StrategyPhase')
                if turn == 4:
                    saved = tmp_path / f'rounds-first-{first}.json'
                    save_game_state(app, str(saved))
                    load_game_state(app, str(saved))
            assert expiry.call_count == 10
        assert app.roundCounter.currentRoundPlayer == [5, 5]
        assert app.battleResult['scores'] == [100, 0]
        saved = tmp_path / f'ended-first-{first}.json'
        save_game_state(app, str(saved))
        load_game_state(app, str(saved))
        assert len(app.battle_awards) == 10
        assert app.roundCounter.finished
    finally:
        load_game_state(app, str(baseline))


@pytest.mark.parametrize('layout', ['three_troves', 'landmark'])
def test_objective_markers_and_hud_render_both_orientations(scene, layout):
    from panda3d.core import OrthographicLens, PNMImage
    from battle_config import load_config
    from battle_setup import resolve_setup, restore_battle
    from battle_objectives import score_turn
    app, baseline = scene
    load_game_state(app, str(baseline))
    config = load_config()
    config['objectives'].update(layout=layout, landmark_property='magic_resistance')
    restore_battle(app, {'config': config, 'setup': resolve_setup(config, 12)})
    unit = next(member for member in app.units if member.unitName == 'Normal Rangers')
    unit.bodyNP.setPos(0, -3.5, 0)
    old_lens = app.cam.node().getLens()
    old_camera = app.camera.getTransform()
    original_orientation = app.hud.orientation
    try:
        score_turn(app)
        lens = OrthographicLens()
        lens.setFilmSize(80, 45)
        app.cam.node().setLens(lens)
        app.camera.setPos(0, 0, 100)
        app.camera.lookAt(0, 0, 0)
        with patch.object(game_module, 'save_setting'):
            for unused in range(2):
                if app.hud._vertical:
                    app.hud.show_tab('objectives')
                app.hud._layout()
                app.eventMgr.doEvents()
                app.graphicsEngine.renderFrame()
                app.graphicsEngine.renderFrame()
                text = app.hud._objectives_text
                scale = text.getScale()[0]
                section = 'tabs' if app.hud._vertical else 'objectives'
                total = 2 if app.hud._vertical else 2 * app.getAspectRatio()
                assert text.textNode.getWidth() * scale <= app.hud._section_width(section, total) * .9
                image = PNMImage()
                assert app.win.getScreenshot(image)
                cyan_pixels = sum(image.getXel(horizontal, vertical).x < .4
                                  and image.getXel(horizontal, vertical).y > .55
                                  and image.getXel(horizontal, vertical).z > .7
                                  for horizontal in range(605, 675) for vertical in range(325, 395))
                assert cyan_pixels > 100
                assert image.write(str(ROOT / '.pytest_cache' / f'battle_march_{layout}_{app.hud.orientation}.png'))
                app.toggleHudLayout()
    finally:
        with patch.object(game_module, 'save_setting'):
            if app.hud.orientation != original_orientation:
                app.toggleHudLayout()
        app.cam.node().setLens(old_lens)
        app.camera.setTransform(old_camera)
        load_game_state(app, str(baseline))


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
    scout.bodyNP.setPos(target.x, target.y - 6, 0)
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
    app.chargeStage = 'remaining'
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