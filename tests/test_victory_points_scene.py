"""Original selected-army costs survive casualties, joining and reload (p. 286)."""

from math import ceil
from unittest.mock import patch

from panda3d.core import FrameBufferProperties, GraphicsPipe, WindowProperties

from command_groups import capture_standard
from characters import join_unit
from persistence import load_game_state, save_game_state
from tests.test_faction_rules_scene import members, scene as scene
from victory_points import calculate


def test_selected_rosters_retain_full_paid_costs_after_losses_and_reload(scene, tmp_path):
    app, baseline = scene
    load_game_state(app, baseline)
    assert len(app.victoryRoster) == 10
    assert [sum(record['points'] for record in app.victoryRoster.values() if record['player'] == player)
            for player in (1, 2)] == [500, 500]
    roster = members(app)
    general, knights, horsemen = (roster[name] for name in ('Aspiring Champion', 'Chaos Knight', 'Marauder Horsemen'))
    points = {unit.unitName: app.victoryRoster[unit.unitName]['points'] for unit in (general, knights, horsemen)}
    app.combat.removeUnitFromPlay(general)
    app.movement.removeModelsFromUnit(knights, knights.unit.nmodels - 1)
    horsemen.request('IsFleeing')
    capture_standard(app, roster['Chaos Warrior'], roster['Dragon Prince'])
    app.capturedStandards.append(dict(app.capturedStandards[0]))
    expected = points[general.unitName] + 100 + ceil(points[knights.unitName] / 2) + ceil(points[horsemen.unitName] / 2) + 50
    result = calculate(app)
    assert result['complete'] and result['scores'] == [expected, 0]
    path = save_game_state(app, str(tmp_path / 'victory-casualties.json'))
    for _ in range(2):
        load_game_state(app, baseline)
        load_game_state(app, path)
        assert calculate(app) == result


def test_joined_general_counts_separately_and_flees_with_host(scene):
    app, baseline = scene
    load_game_state(app, baseline)
    roster = members(app)
    mage, princes = roster['Mage'], roster['Dragon Prince']
    assert join_unit(app, mage, princes)
    assert calculate(app)['scores'] == [0, 0]
    princes.request('IsFleeing')
    expected = sum(ceil(app.victoryRoster[unit.unitName]['points'] / 2) for unit in (mage, princes)) + 100
    assert calculate(app)['scores'] == [0, expected]


def test_missing_legacy_roster_never_claims_a_certified_result(scene):
    app, baseline = scene
    load_game_state(app, baseline)
    app.victoryRoster = {}
    app.victoryLedgerComplete = False
    result = calculate(app)
    assert not result['complete'] and result['winner'] is None
    assert result['outcome'] == 'Incomplete scoring data'


def test_last_player_two_turn_ends_and_restores_result_without_extra_turn(scene, tmp_path):
    app, baseline = scene
    load_game_state(app, baseline)
    app.fsm.request('CombatPhase')
    app.roundCounter.max_rounds = 6
    app.roundCounter.currentRoundPlayer = [5, 5]
    app.roundCounter.request('PlayerOne')
    app.remainsInPlay = []
    with patch('rallying_cry.begin_command'), patch('chaos_gifts.begin_turn'):
        app.fsm.nextPhase()
    assert app.fsm.state == 'StrategyPhase'
    assert app.roundCounter.current_player == 2
    assert app.roundCounter.currentRoundPlayer == [6, 5]
    assert app.battleResult is None
    app.fsm.request('CombatPhase')
    for flag in ('resolvingCombat', 'awaitingChoice', '_reformActive'):
        setattr(app, flag, True)
        app.fsm.nextPhase()
        assert app.fsm.state == 'CombatPhase' and app.roundCounter.currentRoundPlayer == [6, 5]
        setattr(app, flag, False)
    app.fsm.nextPhase()
    assert app.fsm.state == 'BattleEnded'
    assert app.roundCounter.currentRoundPlayer == [6, 6]
    assert app.battleResult['complete'] and app.battleResult['outcome'] == 'Draw'
    app.fsm.nextPhase()
    assert app.fsm.state == 'BattleEnded' and app.roundCounter.currentRoundPlayer == [6, 6]
    app.graphicsEngine.renderFrame()
    app.graphicsEngine.renderFrame()
    app.screenshot('/tmp/towbat-victory-points.png', defaultFilename=False)
    path = save_game_state(app, str(tmp_path / 'ended-battle.json'))
    result = app.battleResult
    load_game_state(app, baseline)
    load_game_state(app, path)
    assert app.fsm.state == 'BattleEnded' and app.battleResult == result
    assert app.roundCounter.currentRoundPlayer == [6, 6]
    assert app.hud._end_btn['text'] == 'BATTLE\nRESULT'
    app.hud.set_battle_result(None)
    app.fsm.nextPhase()
    assert app.hud._battle_result_panel is not None


def test_result_panel_fits_smaller_viewport_with_long_outcome(scene):
    app, baseline = scene
    load_game_state(app, baseline)
    result = calculate(app)
    result.update(scores=[600, 0], winner=1, outcome='Crushing victory')
    app.hud.set_battle_result(result)
    properties = WindowProperties()
    properties.setSize(800, 600)
    framebuffer = FrameBufferProperties()
    framebuffer.setRgbColor(True)
    framebuffer.setDepthBits(24)
    buffer = app.graphicsEngine.makeOutput(app.pipe, 'victory-result', 0, framebuffer,
                                          properties, GraphicsPipe.BFRefuseWindow,
                                          app.win.getGsg(), app.win)
    assert buffer is not None
    try:
        app.setAspectRatio(800 / 600)
        buffer.makeDisplayRegion().setCamera(app.cam)
        overlay = buffer.makeDisplayRegion()
        overlay.setSort(20)
        overlay.setClearDepthActive(True)
        overlay.setCamera(app.cam2d)
        app.graphicsEngine.renderFrame()
        app.graphicsEngine.renderFrame()
        lower, upper = app.hud._battle_result_panel.getTightBounds()
        assert lower.x >= -.851 and upper.x <= .851
        assert lower.z >= -.431 and upper.z <= .731
        image = buffer.getScreenshot()
        assert image.getXSize() == 800 and image.getYSize() == 600
        assert image.write('/tmp/towbat-victory-points-800.png')
    finally:
        app.graphicsEngine.removeWindow(buffer)
        app.setAspectRatio(1280 / 720)
        app.hud.set_battle_result(None)