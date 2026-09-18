"""Fixture-owned battles with the real AI, commands, rules and task manager."""

import json
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
def ai_scene(tmp_path, request):
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
    scenario, seed = getattr(request, 'param', ('standard', 41))
    random.seed(seed)
    import rules_log
    original_listeners = list(rules_log._listeners)
    bake = MyApp.bakeBattleMat
    with patch.object(unitGraphics, 'loadFigureModel', base_model), \
            patch.object(MyApp, 'bakeBattleMat', lambda app, size=128: bake(app, size=128)), \
            patch.object(TerrainManager, 'load_from_json'), \
            patch('persistence.SAVE_DIR', str(tmp_path)):
        app = MyApp(rosters=rosters, battle_config=scenario_config(scenario), battle_seed=seed,
                first_player=1 if seed == 41 else 2)
        clock = ClockObject.getGlobalClock()
        old_mode, old_dt = clock.getMode(), clock.getDt()
        clock.setMode(ClockObject.MNonRealTime)
        clock.setDt(1 / 30)
        app.speedMultiplier = 100
        app.roundCounter.max_rounds = 1
        try:
            yield app
        finally:
            for task in list(app.taskMgr.getTasks()) + list(app.taskMgr.getDoLaters()):
                app.taskMgr.remove(task)
            app.destroy()
            assert rules_log._listeners == original_listeners
            clock.setMode(old_mode)
            clock.setDt(old_dt)


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
            [(unit.unitName, unit.state, unit.hasMovedThisTurn) for unit in app.units])
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
    assert any(': cast(' in action and ' -> completed' in action for action in actions)
    assert any(': shoot(' in action and ' -> completed' in action for action in actions)
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
    assert approach and all(candidate.reason == 'objective approach' for candidate in approach)
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


def test_ai_reserve_move_preserves_earlier_budget(ai_scene):
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
    apply_rule_keywords(mover.unit.model, ['Reserve Move'])
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
    assert tuple(mover.bodyNP.getPos()) != before
    assert mover.reserveDoneTurn == current_turn(app)
    assert mover.hasMovedThisTurn and mover.moveSpentThisTurn == 1.5
    assert not mover.marchedThisTurn