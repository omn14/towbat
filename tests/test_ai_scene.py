"""Fixture-owned battles with the real AI, commands, rules and task manager."""

import json
import math
from pathlib import Path
import random
from time import monotonic
from unittest.mock import patch

import pytest
from panda3d.core import ClockObject, Filename, Point3, getModelPath, loadPrcFileData

from game import MyApp
from gameStateTree import GameAction
from terrain_system import TerrainManager
from tests.test_faction_rules_scene import base_model
from units import unitGraphics


def scenario_config(scenario):
    from battle_config import REED_FENS_MAP
    if scenario == 'standard':
        return None
    reed = scenario == 'reed_fens'
    return {
        'schema_version': 1, 'ruleset': 'battle_march_generals_companion',
        'source': {'url': 'local:ai-acceptance', 'publication': 'AI acceptance fixture', 'reviewed': '2026-09-18'},
        'points_limit': 500,
        'battlefield': {'width': 30 if reed else 44, 'depth': 44 if reed else 30,
                        'show_boundary': True, 'show_deployment': True},
        'deployment': {'map': REED_FENS_MAP if reed else 'pitched_battle', 'mirror': False,
                       'method': 'alternating', 'first_turn': 'roll_off_choice'},
        'terrain': {'method': 'fixed' if reed else 'alternating', 'feature_count': 4 if reed else 0,
                    'recommended_max_span': 12, 'centre_clearance': 0,
                    'opponent_feature_clearance': 0, 'objective_clearance': 3},
        'objectives': {'layout': 'none' if reed else 'two_troves', 'trove_base_mm': 40,
                       'landmark_base_mm': 100, 'control_distance': 3,
                       'minimum_unit_strength': 5, 'landmark_property': 'stubborn'},
        'game': {'rounds': 6, 'time_limit_minutes': None},
        'scoring': {'trove_per_player_turn': 10, 'landmark_per_player_turn': 25,
                    'general': 50, 'captured_standard': 25, 'battle_standard_bearer': 25},
        'army': {'minimum_units': 2, 'maximum_unit_strength': 20,
                 'maximum_character_fraction': .25, 'maximum_core_fraction': .35,
                 'maximum_special_fraction': .3, 'maximum_rare_mercenary_fraction': .25,
                 'restricted_options_allowance': 1},
        'optional_rules': {'secondary_objectives': [], 'secret_objectives': False,
                           'random_happenings': [], 'battle_march_magic_items': False},
    }


@pytest.fixture
def scene_factory(tmp_path):
    loadPrcFileData('', 'window-type offscreen\nwin-size 640 360\naudio-library-name null')
    getModelPath().appendDirectory(Filename.fromOsSpecific(str(Path(__file__).resolve().parents[1])))
    rosters = {}
    for player in (1, 2):
        path = tmp_path / f'army{player}.json'
        path.write_text(json.dumps({'units': [
            {'name': 'Elven Archer', 'nmodels': 5, 'files': 5, 'ranks': 1,
             'weapons': [{'name': 'Longbow'}]},
            {'name': 'Chaos Warrior', 'nmodels': 5, 'files': 5, 'ranks': 1},
            {'name': 'Mage', 'nmodels': 1, 'files': 1, 'ranks': 1, 'wizard_level': 1,
             'spells': [{'name': 'Oaken Shield', 'casting_value': 7, 'range': 'Self',
                         'phase': 'strategy', 'type': 'Enchantment'}]},
        ]}))
        rosters[f'player{player}'] = str(path)
    import rules_log
    original_listeners = list(rules_log._listeners)
    bake = MyApp.bakeBattleMat
    app = None
    clock = ClockObject.getGlobalClock()
    old_mode, old_dt = clock.getMode(), clock.getDt()

    def close_scene():
        nonlocal app
        if app is not None:
            for task in list(app.taskMgr.getTasks()) + list(app.taskMgr.getDoLaters()):
                app.taskMgr.remove(task)
            app.destroy()
            app = None
        assert rules_log._listeners == original_listeners

    def create_scene(**options):
        nonlocal app
        close_scene()
        app = MyApp(rosters=rosters, **options)
        clock.setMode(ClockObject.MNonRealTime)
        clock.setDt(1 / 30)
        app.speedMultiplier = 100
        return app

    with patch.object(unitGraphics, 'loadFigureModel', base_model), \
            patch.object(MyApp, 'bakeBattleMat', lambda app, size=128: bake(app, size=128)), \
            patch.object(TerrainManager, 'load_from_json'), \
            patch('persistence.SAVE_DIR', str(tmp_path)):
        try:
            yield create_scene
        finally:
            close_scene()
            clock.setMode(old_mode)
            if old_dt > 0:
                clock.setDt(old_dt)


@pytest.fixture
def ai_scene(scene_factory, request):
    scenario, seed = getattr(request, 'param', ('standard', 41))
    random.seed(seed)
    app = scene_factory(battle_config=scenario_config(scenario), battle_seed=seed,
                        first_player=1 if seed == 41 else 2)
    app.roundCounter.max_rounds = 1
    return app


@pytest.mark.parametrize('scenario, backup', [('standard', False), ('battle_march', False),
                                            ('standard', True)])
def test_saved_battle_startup_restores_without_default_armies(scene_factory, tmp_path, scenario, backup):
    from battle_config import startup_options
    from persistence import save_game_state
    random.seed(41)
    app = scene_factory(battle_config=scenario_config(scenario), battle_seed=41)
    app.restoringBattle = True
    app.fsm.request('MovementPhase')
    app.restoringBattle = False
    app.chargeStage = 'remaining'
    app.roundCounter.currentRoundPlayer = [2, 1]
    app.roundCounter.current_player = 2
    app.first_player = 2
    for index, unit in enumerate(app.units):
        unit.isDeployed = True
        unit.bodyNP.setPos(index * 3, 7, 0)
        unit.hasMovedThisTurn = True
    expected = {unit.unitName: tuple(unit.bodyNP.getPos()) for unit in app.units}
    path = Path(save_game_state(app, str(tmp_path / 'saved battle.json')))
    if backup:
        path.rename(str(path) + '.bak')
    argument = path.name if backup else str(path)
    options = startup_options(['--load-save', argument])
    with patch.object(MyApp, 'load_player1_army', side_effect=AssertionError('default P1 army read')), \
            patch.object(MyApp, 'load_player2_army', side_effect=AssertionError('default P2 army read')):
        restored = scene_factory(load_save=options.load_save)
    assert restored.battle_config_screen is None
    assert restored.fsm.state == 'MovementPhase'
    assert restored.chargeStage == 'remaining'
    assert restored.roundCounter.currentRoundPlayer == [2, 1]
    assert restored.roundCounter.current_player == restored.first_player == 2
    assert restored.unitToMove in restored.player2Units
    assert {unit.unitName: tuple(unit.bodyNP.getPos()) for unit in restored.units} == expected
    assert all(unit.hasMovedThisTurn and unit.isDeployed for unit in restored.units)
    assert bool(getattr(restored, 'battle_config', None)) == (scenario != 'standard')
    restored.taskMgr.step()
    restored.graphicsEngine.renderFrame()


def test_saved_battle_startup_missing_file_fails_cleanly(scene_factory, tmp_path):
    with pytest.raises(ValueError, match='Could not load saved battle'):
        scene_factory(load_save=tmp_path / 'missing.json')
    assert scene_factory().fsm.state == 'DeployPhase'


def test_basic_movement_preview_checks_swept_route_without_mutation(ai_scene):
    app = ai_scene
    unit = app.player1Units[1]
    for other in app.units:
        other.isDeployed = other is unit
    unit.bodyNP.setPos(0, -12, 0)
    unit.bodyNP.setH(0)
    app.fsm.request('MovementPhase')
    app.chargeStage = 'remaining'
    app.terrain_manager.clear()
    app.terrain_manager.add_terrain('house', Point3(0, -7, 0), 4, 3)
    before = (tuple(unit.bodyNP.getPos()), unit.bodyNP.getH(), unit.hasMovedThisTurn, unit.moveSpentThisTurn)
    blocked = app.movement.previewBasicMove(unit, (0, -4, 0))
    assert blocked.error
    assert (tuple(unit.bodyNP.getPos()), unit.bodyNP.getH(), unit.hasMovedThisTurn, unit.moveSpentThisTurn) == before
    unit.bodyNP.setX(-8)
    clear = app.movement.previewBasicMove(unit, (-8, -4, 0))
    assert clear.error is None
    assert clear.marching and clear.distance == pytest.approx(8)
    assert app.movement.previewBasicMove(unit, (-8, -20, 0)).error
    assert not unit.hasMovedThisTurn and unit.moveSpentThisTurn == 0
    with patch.object(app.movement, 'movementAllowance', side_effect=[4, 2]):
        assert app.movement.previewBasicMove(unit, (-6, -12, 0)).error


@pytest.mark.parametrize('loose', [False, True])
@pytest.mark.parametrize('drilled', [False, True])
@pytest.mark.parametrize('dice, status', [([1, 1], 'completed'), ([6, 6], 'rejected')])
def test_ai_awaits_enemy_sighted_march(ai_scene, dice, status, drilled, loose):
    from unittest.mock import AsyncMock
    app = ai_scene
    app.AIplayer1.active = app.AIplayer2.active = True
    app.AIplayer1.automatic = app.AIplayer2.automatic = False
    mover, enemy = app.player1Units[2 if loose else 1], app.player2Units[1]
    travel = app.movement.movementAllowance(mover, features=[]) * 2
    if drilled:
        from special_rules import apply_rule_keywords
        apply_rule_keywords(mover.unit.model, ['Drilled'])
        status = 'completed'
    for other in app.units:
        other.isDeployed = other in (mover, enemy)
        if not other.isDeployed:
            other.bodyNP.setPos(40, 40, 0)
    app.terrain_manager.clear()
    mover.bodyNP.setPos(0, 0, 0)
    mover.bodyNP.setH(0)
    enemy.bodyNP.setPos(0, -7, 0)
    app.fsm.request('MovementPhase')
    app.chargeStage = 'remaining'
    outcomes = []

    async def roll_dice():
        from direct.task.Task import pause
        await pause(.2)
        assert app.AIplayer1._command_running
        assert not app.AIplayer1._can_take_turn()
        return dice

    async def execute():
        outcomes.append(await app.AIplayer1.execute_action(GameAction('move', mover.unitName,
                        {'target_x': 0, 'target_y': travel, 'heading': 0})))

    with patch.object(app, 'rollLeadershipDice', AsyncMock(side_effect=roll_dice)) as roll:
        app.taskMgr.add(execute(), 'ai-march-test')
        for frame in range(100):
            app.taskMgr.step()
            if outcomes:
                break
    assert outcomes and outcomes[0].status == status, (
        outcomes, mover.marchTestResult, tuple(mover.bodyNP.getPos()),
        [entry.text for entry in app.hud._journal.entries][-20:])
    assert mover.marchTestResult == (None if drilled else 'passed' if status == 'completed' else 'failed')
    assert roll.await_count == (0 if drilled else 1)
    assert mover.bodyNP.getY() == pytest.approx(travel if status == 'completed' else 0)
    assert not app.AIplayer1._command_running


def test_saved_movement_candidate_budget(scene_factory):
    from ai_movement import routes_towards
    app = scene_factory(load_save=Path(__file__).with_name('ai_movement_reed_fens.json'))
    for unit in app.units:
        if getattr(unit, 'hostUnit', None) is not None:
            continue
        started = monotonic()
        routes_towards(app, unit, (5, 13) if unit in app.player1Units else (-6, -12))
        assert monotonic() - started < 2, unit.unitName


def test_infantry_routes_around_house_with_useful_progress(ai_scene):
    from ai_movement import routes_towards
    from scouts import placement_error
    app = ai_scene
    app.AIplayer1.active = app.AIplayer2.active = True
    app.AIplayer1.automatic = app.AIplayer2.automatic = False
    unit = app.player1Units[1]
    for other in app.units:
        other.isDeployed = other is unit
        other.bodyNP.setPos(40, 40, 0)
    unit.bodyNP.setPos(-2, -14, 0)
    unit.bodyNP.setH(0)
    app.terrain_manager.clear()
    app.terrain_manager.add_terrain('house', Point3(-2, -7, 0), 4, 3)
    app.fsm.request('MovementPhase')
    app.chargeStage = 'remaining'
    positions = []
    for turn in range(4):
        unit.request('Idle')
        unit.hasMovedThisTurn = unit.marchedThisTurn = False
        unit.moveSpentThisTurn = 0
        unit.manoeuvreThisTurn = None
        routes = routes_towards(app, unit, (-2, 10))
        assert routes, (positions, unit.bodyNP.getH())
        preview, progress, waypoint = routes[0]
        before = tuple(unit.bodyNP.getPos())
        finish_action(app, GameAction('move', unit.unitName, {
            'target_x': preview.destination[0], 'target_y': preview.destination[1], 'heading': preview.heading}))
        assert tuple(unit.bodyNP.getPos()) == pytest.approx(preview.destination, abs=1e-4)
        assert placement_error(app, unit, deployment_zone=False) is None
        positions.append(tuple(unit.bodyNP.getPos()))
        assert math.dist(before, positions[-1]) > .5
        if unit.bodyNP.getY() > -3:
            break
    assert unit.bodyNP.getY() > -3, positions


def test_front_unit_clears_lane_before_rear_advances(ai_scene):
    from ai_policy import movement_candidates
    from scouts import placement_error
    app = ai_scene
    app.AIplayer1.active = app.AIplayer2.active = True
    app.AIplayer1.automatic = app.AIplayer2.automatic = False
    front, enemy = app.player1Units[1], app.player2Units[1]
    rear = app._create_unit({'name': 'Chaos Warrior', 'nmodels': 5, 'files': 5, 'ranks': 1}, 1, 'Rear regiment')
    for unit in app.units:
        unit.isDeployed = unit in (front, rear, enemy)
        unit.bodyNP.setPos(40, 40, 0)
    front.bodyNP.setPos(0, -4, 0)
    rear.bodyNP.setPos(0, -7, 0)
    enemy.bodyNP.setPos(0, 18, 0)
    front.bodyNP.setH(0)
    rear.bodyNP.setH(0)
    app.terrain_manager.clear()
    app.fsm.request('MovementPhase')
    app.chargeStage = 'remaining'
    options = movement_candidates(app, 1, [rear, front], [enemy])
    selected = max(options, key=lambda candidate: candidate.score)
    assert selected.action.unit_name == front.unitName
    finish_action(app, selected.action)
    options = movement_candidates(app, 1, [rear, front], [enemy])
    selected = max(options, key=lambda candidate: candidate.score)
    assert selected.action.unit_name == rear.unitName
    finish_action(app, selected.action)
    assert rear.bodyNP.getY() > -3
    assert placement_error(app, rear, deployment_zone=False) is None


def test_reed_fens_movement_reaches_combat(scene_factory, tmp_path):
    random.seed(42)
    app = scene_factory(load_save=Path(__file__).with_name('ai_movement_reed_fens.json'))
    app.AIplayer1.active = app.AIplayer2.active = True
    trace = []
    previous = None
    started = monotonic()
    for frame in range(9000):
        app.taskMgr.step()
        state = (app.fsm.state, app.roundCounter.current_player,
                 tuple(app.roundCounter.currentRoundPlayer))
        if state != previous:
            if app.fsm.state == 'ShootingPhase':
                from scouts import placement_error
                for unit in app.units:
                    if getattr(unit, 'hostUnit', None) is None and not unit.isInCombat:
                        assert placement_error(app, unit, deployment_zone=False) is None, unit.unitName
            trace.append({'window': state, 'units': [
                {'name': unit.unitName, 'position': tuple(unit.bodyNP.getPos(app.render)),
                 'heading': unit.bodyNP.getH(), 'state': unit.state,
                 'engaged': bool(unit.isInCombat), 'moved': unit.hasMovedThisTurn}
                for unit in app.units if getattr(unit, 'hostUnit', None) is None]})
            previous = state
        if any(unit.isInCombat for unit in app.units):
            break
        if (max(app.roundCounter.currentRoundPlayer) >= 3 or app.fsm.state == 'BattleEnded'
                or monotonic() - started > 60):
            break
        assert app.AIplayer1.active and app.AIplayer2.active, (
            app.AIplayer1.pause_reason, app.AIplayer2.pause_reason, trace)
    (tmp_path / 'movement-trace.json').write_text(json.dumps(trace, indent=2))
    actions = [entry.text for entry in app.hud._journal.entries if entry.text.startswith('AI P')]
    (tmp_path / 'movement-actions.json').write_text(json.dumps(actions, indent=2))
    (tmp_path / 'movement-summary.json').write_text(json.dumps({
        'seed': 42, 'seconds': monotonic() - started, 'frames': frame + 1,
        'player': app.roundCounter.current_player, 'rounds_completed': app.roundCounter.currentRoundPlayer,
        'contacts': {unit.unitName: [enemy.unitName for enemy in unit.isInCombatWith]
                     for unit in app.units if unit.isInCombat},
    }, indent=2))
    assert any(unit.isInCombat for unit in app.units), (trace, actions)
    assert max(app.roundCounter.currentRoundPlayer) <= 1


@pytest.mark.parametrize('ai_scene', [(scenario, seed) for scenario in
                    ('standard', 'battle_march', 'reed_fens') for seed in (41, 42)],
                    ids=[f'{scenario}-{seed}' for scenario in ('standard', 'battle_march', 'reed_fens')
                        for seed in (41, 42)], indirect=True)
def test_autonomous_battle_matrix(ai_scene, tmp_path, request, record_property):
    app = ai_scene
    assert len(app.units) == 6
    app.roundCounter.max_rounds = 6
    origins = {unit.unitName: tuple(unit.bodyNP.getPos()) for unit in app.units}
    app.AIplayer1.active = app.AIplayer2.active = True
    phases = {app.fsm.state}
    first_player = None
    started = monotonic()
    for frame in range(18000):
        app.taskMgr.step()
        phases.add(app.fsm.state)
        if app.fsm.state != 'DeployPhase' and first_player is None:
            first_player = app.roundCounter.current_player
        assert monotonic() - started < 45, (app.fsm.state, getattr(app, 'battle_setup', None),
            app.awaitingChoice, [task.getName() for task in app.taskMgr.getTasks()])
        if app.fsm.state == 'BattleEnded':
            break
        assert app.AIplayer1.active and app.AIplayer2.active, (
            app.fsm.state, app.roundCounter.current_player,
            getattr(app.AIplayer1, 'pause_reason', ''), getattr(app.AIplayer2, 'pause_reason', ''),
                        [(unit.unitName, unit.state, unit.hasMovedThisTurn, unit.marchTestResult,
                            getattr(unit, '_drilledMoveActive', False)) for unit in app.units],
                        [task.getName() for task in app.taskMgr.getTasks()],
                        '\n'.join(entry.text for entry in list(app.hud._journal.entries)[-25:]))
    assert app.fsm.state == 'BattleEnded', (app.fsm.state, phases)
    assert {'DeployPhase', 'MovementPhase', 'ShootingPhase', 'CombatPhase'} <= phases
    assert app.roundCounter.currentRoundPlayer == [6, 6]
    assert all(unit.isDeployed for unit in app.units)
    assert any(tuple(unit.bodyNP.getPos()) != origins[unit.unitName] for unit in app.units)
    assert app.battleResult is not None
    if not getattr(app, 'battle_config', None):
        assert first_player == app.first_player
    if getattr(app, 'battle_config', None):
        assert app.battle_setup['preparation']['stage'] == 'complete'
        assert bool(app.battle_objectives) == (app.battle_config['objectives']['layout'] != 'none')
    actions = [entry.text for entry in app.hud._journal.entries
               if entry.text.startswith('AI P') and ' -> ' in entry.text]
    (tmp_path / 'ai-journal.json').write_text(json.dumps([entry.text for entry in app.hud._journal.entries], indent=2))
    result = {
        'schema_version': 1, 'policy': 'live-utility-v1', 'setup': getattr(app, 'battle_setup', None),
        'case': request.node.callspec.id, 'rounds': app.roundCounter.currentRoundPlayer,
        'first_player': first_player,
        'frames': frame + 1, 'seconds': monotonic() - started, 'result': app.battleResult,
        'actions': actions, 'failure': None,
    }
    output = json.dumps(result, default=str, indent=2)
    (tmp_path / 'ai-match-result.json').write_text(output)
    record_property('ai_match', output)
    assert any(': cast(' in action and ' -> completed' in action for action in actions)
    assert (any((': shoot(' in action or ': attack(' in action) and ' -> completed' in action for action in actions)
            or any(entry.text.startswith('Shooting ') and ' eligible models ' in entry.text
                   and ' shots,' in entry.text and '-> 0 shots,' not in entry.text
                   for entry in app.hud._journal.entries)), '\n'.join(actions)


def test_ai_casts_through_live_spell_window(ai_scene):
    app = ai_scene
    wizard = app._create_unit({'name': 'Mage', 'nmodels': 1, 'files': 1, 'ranks': 1,
        'wizard_level': 1, 'spells': [{'name': 'Oaken Shield', 'casting_value': 7,
        'range': 'Self', 'phase': 'strategy', 'type': 'Enchantment'}]}, 1, 'AI Wizard')
    wizard.isDeployed = True
    app.AIplayer1.active = app.AIplayer2.active = True
    app.AIplayer1.automatic = app.AIplayer2.automatic = False
    app.fsm.request('StrategyPhase')
    app.strategyCommandDone = True
    completed = []

    async def cast():
        completed.append(await app.AIplayer1.execute_action(
            GameAction('cast', wizard.unitName, {'spell': 'Oaken Shield', 'target': wizard.unitName})))

    app.taskMgr.add(cast(), 'ai-casting-test')
    for frame in range(3000):
        app.taskMgr.step()
        if completed:
            break
    assert completed and completed[0].status == 'completed'
    assert wizard.spellsCastThisTurn == ['Oaken Shield']
    assert app.fsm.state == 'StrategyPhase'
    assert not app.castingSpell


def test_live_policy_targets_objectives_and_shooting(ai_scene):
    from ai_policy import movement_candidates, shooting_candidates
    app = ai_scene
    app.AIplayer1.active = app.AIplayer2.active = True
    app.AIplayer1.automatic = app.AIplayer2.automatic = False
    shooter, target = app.player1Units[0], app.player2Units[0]
    for member in app.units:
        member.isDeployed = member in (shooter, target)
        member.bodyNP.setPos(25, 20, 0)
    shooter.bodyNP.setPos(0, -12, 0)
    shooter.bodyNP.setH(0)
    target.bodyNP.setPos(0, 8, 0)
    target.bodyNP.setH(180)
    app.fsm.request('ShootingPhase')
    legal = shooting_candidates(app, [shooter], [target])
    assert legal and legal[0].action.parameters['weapon'] == 'Longbow'
    target.bodyNP.setY(40)
    assert not shooting_candidates(app, [shooter], [target])
    target.bodyNP.setY(8)
    target.isInCombat = True
    assert not shooting_candidates(app, [shooter], [target])
    target.isInCombat = False
    app.terrain_manager.add_terrain('house', Point3(0, 0, 0), 16, 4)
    assert not shooting_candidates(app, [shooter], [target])
    app.terrain_manager.clear()
    completed = []

    async def shoot():
        completed.append(await app.AIplayer1.execute_action(legal[0].action))

    app.taskMgr.add(shoot(), 'ai-live-volley')
    for frame in range(3000):
        app.taskMgr.step()
        if completed:
            break
    assert completed and completed[0].status == 'completed'
    assert shooter.hasAttackedThisTurn and not app.shootingInFlight
    app.battle_config = scenario_config('battle_march')
    app.battle_objectives = [{'id': 'objective-1', 'kind': 'trove', 'center': [0, 0],
        'diameter': 40 / 25.4, 'destroyed': False, 'controller': shooter.unitName,
        'player': 1, 'contested': False}]
    target.isDeployed = False
    app.fsm.request('MovementPhase')
    app.chargeStage = 'remaining'
    approach = movement_candidates(app, 1, [shooter], [])
    assert approach and all(candidate.reason.startswith('objective approach;') for candidate in approach)
    shooter.bodyNP.setPos(0, -2, 0)
    assert not movement_candidates(app, 1, [shooter], [])


def finish_action(app, action):
    outcomes = []

    async def execute():
        outcomes.append(await app.AIplayer1.execute_action(action))

    app.taskMgr.add(execute(), 'ai-command-test')
    for frame in range(3000):
        app.taskMgr.step()
        if outcomes:
            break
    assert outcomes and outcomes[0].status == 'completed', outcomes


def test_live_artillery_and_formation_commands(ai_scene):
    from ai_policy import shooting_candidates, formation_candidates
    app = ai_scene
    app.AIplayer1.active = app.AIplayer2.active = True
    app.AIplayer1.automatic = app.AIplayer2.automatic = False
    for index, member in enumerate(app.units):
        member.isDeployed = False
        member.bodyNP.setPos(25 + index * 3, 20, 0)
    cannon = app._create_unit({'name': 'Great Cannon', 'nmodels': 1, 'files': 1, 'ranks': 1}, 1, 'AI Cannon')
    target = app.player2Units[1]
    cannon.isDeployed = target.isDeployed = True
    cannon.bodyNP.setPos(0, -10, 0)
    target.bodyNP.setPos(0, 10, 0)
    cannon.bodyNP.setH(0)
    target.bodyNP.setH(180)
    app.fsm.request('ShootingPhase')
    options = shooting_candidates(app, [cannon], [target])
    assert options and options[0].action.action_type == 'cannon'
    finish_action(app, options[0].action)
    assert cannon.hasAttackedThisTurn
    column = app._create_unit({'name': 'Chaos Warrior', 'nmodels': 6, 'files': 1, 'ranks': 6}, 1, 'AI Column')
    column.isDeployed = True
    column.bodyNP.setPos(-10, 0, 0)
    app.fsm.request('MovementPhase')
    app.chargeStage = 'remaining'
    options = formation_candidates(app, [column])
    assert options
    finish_action(app, options[0].action)
    assert column.unit.files >= column.unit.ranks
    assert column.moveSpentThisTurn > 0


def test_live_character_join_command(ai_scene):
    from ai_policy import joining_candidates
    app = ai_scene
    app.AIplayer1.active = app.AIplayer2.active = True
    app.AIplayer1.automatic = app.AIplayer2.automatic = False
    for index, member in enumerate(app.units):
        member.isDeployed = False
        member.bodyNP.setPos(25 + index * 3, 20, 0)
    host, wizard = app.player1Units[0], app.player1Units[2]
    host.isDeployed = wizard.isDeployed = True
    host.bodyNP.setPos(0, 0, 0)
    wizard.bodyNP.setPos(0, -4, 0)
    host.bodyNP.setH(0)
    wizard.bodyNP.setH(0)
    app.fsm.request('MovementPhase')
    app.chargeStage = 'remaining'
    options = joining_candidates(app, [host, wizard])
    assert options
    finish_action(app, options[0].action)
    assert wizard.hostUnit is host and host.joinedMovementLocked


def test_ai_controls_start_disabled_and_toggle_both_sides(ai_scene):
    app = ai_scene
    assert not app.AIplayer1.active and not app.AIplayer2.active
    app.toggle_ai_player(1)
    app.toggle_ai_player2()
    assert app.aiControls(app.player1Units[0]) and app.aiControls(app.player2Units[0])
    app.toggle_ai_automatic()
    assert not app.AIplayer1.automatic
    assert app.AIplayer2.automatic
    app.toggle_ai_player(1)
    assert not app.aiControls(app.player1Units[0])


@pytest.mark.parametrize('drilled', [False, True])
def test_ai_reserve_move_preserves_earlier_budget(ai_scene, drilled):
    from ai_policy import movement_candidates
    from magic_items import current_turn
    from special_rules import apply_rule_keywords
    app = ai_scene
    app.AIplayer1.active = app.AIplayer2.active = True
    app.AIplayer1.automatic = app.AIplayer2.automatic = False
    mover, enemy = app.player1Units[1], app.player2Units[1]
    for index, member in enumerate(app.units):
        member.isDeployed = member in (mover, enemy)
        member.bodyNP.setPos(25 + index * 3, 20, 0)
    apply_rule_keywords(mover.unit.model, ['Reserve Move', 'Drilled'] if drilled else ['Reserve Move'])
    mover.bodyNP.setPos(0, 0, 0)
    mover.bodyNP.setH(0)
    enemy.bodyNP.setPos(0, 20, 0)
    mover.hasMovedThisTurn = True
    mover.moveSpentThisTurn = 1.5
    mover.request('Moved')
    app.fsm.request('ReserveMovePhase')
    before = tuple(mover.bodyNP.getPos())
    options = movement_candidates(app, 1, [mover], [enemy])
    assert options
    finish_action(app, options[0].action)
    assert mover.bodyNP.getX() == pytest.approx(options[0].action.parameters['target_x'], abs=1e-4)
    assert mover.bodyNP.getY() == pytest.approx(options[0].action.parameters['target_y'], abs=1e-4)
    assert tuple(mover.bodyNP.getPos()) != before
    assert mover.reserveDoneTurn == current_turn(app)
    assert mover.hasMovedThisTurn and mover.moveSpentThisTurn == 1.5
    assert not mover.marchedThisTurn