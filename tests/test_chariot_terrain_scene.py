"""Actual Skycutter flight and Iron Shod Wheels (pp. 170, 194)."""

from unittest.mock import AsyncMock, patch

import pytest

from panda3d.core import FrameBufferProperties, GraphicsPipe, Point3, TransformState, Vec3, WindowProperties

from persistence import load_game_state, save_game_state
from flight import grounded, set_mode
from tests.test_faction_rules_scene import members, scene as scene
from tests.test_shieldwall_scene import combat_tasks


def restore(scene):
    app, baseline = scene
    load_game_state(app, baseline)
    app.terrain_manager.clear()
    unit = members(app)['Lothern Skycutter']
    unit.bodyNP.setPos(0, -10, 0)
    unit.bodyNP.setH(0)
    return app, unit


def test_grounded_chariot_treats_difficult_as_dangerous_and_loses_d3(scene):
    app, unit = restore(scene)
    piece = app.terrain_manager.add_terrain('forest', Point3(0, -10, 0), 6, 6)
    with patch.object(unit.unit.model, 'is_flying', return_value=False), \
            patch('terrain_system.random.randint', side_effect=[1, 3]), \
            patch.object(app.movement, 'applyWounds') as damage:
        assert app.movement.dangerousTerrainTests(
            unit, Vec3(0, -14, 0), unit.bodyNP.getPos(), features=[piece]) == 3
    damage.assert_called_once_with(unit, 3)


def test_base_edge_crossing_is_found_without_explicit_features(scene):
    from scouts import model_base_boxes
    app, unit = restore(scene)
    unit.bodyNP.setPos(0, -2, 0)
    box, = model_base_boxes(unit)
    piece = app.terrain_manager.add_terrain('forest', Point3(box[0] + box[2], -10, 0), .2, .2)
    piece._field = None
    start, end = Vec3(0, -18, 0), unit.bodyNP.getPos()
    assert app.terrain_manager.get_terrain_between(start, end) == []
    with grounded(unit), patch('terrain_system.random.randint', side_effect=[1, 3]), \
            patch.object(app.movement, 'applyWounds') as damage:
        assert app.movement.dangerousTerrainTests(unit, start, end) == 3
    damage.assert_called_once_with(unit, 3)


def test_rectangular_terrain_tests_only_crossing_bases(scene):
    from scouts import model_base_boxes
    app, skycutter = restore(scene)
    unit = members(app)['Chaos Warrior']
    unit.unit.files = unit.unit.nmodels
    unit.layOutRanks()
    unit.bodyNP.setPos(0, -2, 0)
    unit.bodyNP.setH(0)
    box = max(model_base_boxes(unit), key=lambda box: box[0])
    piece = app.terrain_manager.add_terrain('forest', Point3(box[0] + box[2] - .05, -10, 0),
                                          .1, .1, going='dangerous')
    piece._field = None
    start, end = Vec3(0, -18, 0), unit.bodyNP.getPos()
    assert app.movement.movementAllowance(unit, start, end) == app.movement.movementAllowance(unit) - 1
    with patch('terrain_system.random.randint', return_value=2) as dice, \
            patch.object(app.movement, 'applyWounds'):
        assert app.movement.dangerousTerrainTests(unit, start, end) == 0
    dice.assert_called_once_with(1, 6)


def test_flyer_crosses_terrain_without_testing_but_tests_landing(scene):
    app, unit = restore(scene)
    piece = app.terrain_manager.add_terrain('forest', Point3(0, -10, 0), 4, 4)
    with patch('terrain_system.random.randint') as dice:
        assert app.movement.dangerousTerrainTests(
            unit, Vec3(0, -18, 0), Vec3(0, -2, 0), features=[piece]) == 0
    dice.assert_not_called()
    with patch('terrain_system.random.randint', side_effect=[1, 2]), \
            patch.object(app.movement, 'applyWounds') as damage:
        assert app.movement.dangerousTerrainTests(
            unit, Vec3(0, -18, 0), Vec3(0, -10, 0), features=[piece]) == 2
    damage.assert_called_once_with(unit, 2)


def test_ithilmar_reroll_precedes_iron_shod_wound_roll(scene):
    app, unit = restore(scene)
    piece = app.terrain_manager.add_terrain('forest', Point3(0, -10, 0), 6, 6)
    with patch.object(unit.unit.model, 'dangerous_terrain_reroll_sources', return_value=['Ithilmar Armour']), \
            patch('terrain_system.random.randint', side_effect=[1, 2]) as dice, \
            patch.object(app.movement, 'applyWounds'):
        assert app.movement.dangerousTerrainTests(
            unit, Vec3(0, -18, 0), unit.bodyNP.getPos(), features=[piece]) == 0
    assert dice.call_count == 2


def test_ground_override_applies_terrain_even_between_flight_endpoints(scene):
    app, unit = restore(scene)
    piece = app.terrain_manager.add_terrain('forest', Point3(0, -10, 0), 4, 4)
    assert unit.unit.model.can_fly() and unit.unit.model.is_flying()
    with grounded(unit), patch('terrain_system.random.randint', side_effect=[1, 2]), \
            patch.object(app.movement, 'applyWounds'):
        assert unit.unit.model.can_fly() and not unit.unit.model.is_flying()
        assert app.movement.dangerousTerrainTests(
            unit, Vec3(0, -18, 0), Vec3(0, -2, 0), features=[piece]) == 2
    assert unit.unit.model.is_flying()


def test_selected_flight_mode_controls_movement_and_survives_reload(scene, tmp_path):
    app, unit = restore(scene)
    app.unitToMove = unit
    assert set_mode(app, unit, 'ground')
    assert not unit.unit.model.is_flying() and unit.unit.model.can_fly()
    assert app.movement.movementAllowance(unit) == unit.unit.model.get_movement(0)
    path = save_game_state(app, str(tmp_path / 'grounded-skycutter.json'))
    assert set_mode(app, unit, 'fly')
    load_game_state(app, path)
    assert not unit.unit.model.is_flying()
    unit.hasMovedThisTurn = True
    assert not set_mode(app, unit, 'fly')


def test_linear_obstacle_blocks_ground_chariot_but_not_flight_and_roundtrips(scene):
    from scouts import placement_error
    app, unit = restore(scene)
    piece = app.terrain_manager.add_terrain('forest', Point3(0, -5, 0), 10, 1,
                                          linear_obstacle=True)
    app.world.doPhysics(1 / 60)
    transform = TransformState.makePosHpr(unit.bodyNP.getPos(), unit.bodyNP.getHpr())
    original_mask = piece.ghost_np.getCollideMask()
    assert app.movement.sweepTestDir(unit, transform, Vec3(0, 1, 0), 10)[0] == 1
    with grounded(unit):
        assert app.movement.sweepTestDir(unit, transform, Vec3(0, 1, 0), 10)[0] < 1
        assert app.movement.sweepTest(unit, Vec3(0, 1, 0), 10) < 1
    assert piece.ghost_np.getCollideMask() == original_mask
    unit.bodyNP.setPos(piece.center)
    assert 'impassable' in placement_error(app, unit, deployment_zone=False)
    records = app.terrain_manager.to_records()
    assert records[0]['linear_obstacle'] is True
    app.terrain_manager.clear()
    app.terrain_manager.load_records(records)
    assert app.terrain_manager.terrain_pieces[0].linear_obstacle


def test_live_pursuit_and_overrun_force_ground_then_restore_flight(scene):
    app, unit = restore(scene)
    target = members(app)['Chaos Warrior']

    def ground_plot(*args):
        assert not unit.unit.model.is_flying()

    with combat_tasks(app) as run, \
            patch.object(app, 'pathTowardsMouse', side_effect=ground_plot) as plot, \
            patch.object(app, 'moveUnit', return_value=None):
        run(app.combat.pursuitMove(unit, target, 'flee'))
    plot.assert_called_once()
    assert unit.unit.model.is_flying()

    def ground_sweep(*args):
        assert not unit.unit.model.is_flying()
        return 1

    with combat_tasks(app) as run, \
            patch.object(app.combat, 'swiftstrideChoice', AsyncMock(return_value=False)), \
            patch.object(app.combat, 'rollMoveDice', AsyncMock(return_value=[1, 2])), \
            patch.object(app, 'sweepTest', side_effect=ground_sweep), \
            patch.object(app.movement, 'dangerousTerrainTests') as terrain:
        run(app.combat.overrunMove(unit))
    assert unit.unit.model.is_flying()
    terrain.assert_called_once()


def test_live_follow_up_sweeps_and_tests_terrain_on_ground(scene):
    from direct.interval.IntervalGlobal import Sequence
    app, unit = restore(scene)
    app.attackSequence = Sequence()
    loser = members(app)['Chaos Warrior']
    loser.isInCombatWith = [unit]
    checked = []

    def ground_sweep(member, *args):
        if member is unit:
            assert not unit.unit.model.is_flying()
            checked.append('sweep')
        return 1

    def ground_terrain(member, *args):
        if member is unit:
            assert not unit.unit.model.is_flying()
            checked.append('terrain')

    with combat_tasks(app) as run, \
            patch.object(app, 'makeChoiceNew', AsyncMock(return_value='Keep formation')), \
            patch.object(app, 'sweepTest', side_effect=ground_sweep), \
            patch.object(app.combat, 'surrounded', return_value=False), \
            patch.object(app.movement, 'dangerousTerrainTests', side_effect=ground_terrain):
        run(app.combat.giveGroundMove(loser, [unit]))
    assert checked == ['sweep', 'terrain']
    assert unit.unit.model.is_flying()


@pytest.mark.parametrize('size', [(1280, 720), (800, 600)])
def test_flight_controls_render_and_select_ground(scene, tmp_path, size):
    app, unit = restore(scene)
    app.unitToMove = unit
    app.showSelectedUnit(unit)
    assert len(app.flightButtons) == 2
    assert all(not button.isHidden() for button in app.flightButtons)
    app.flightButtons[1]['command']()
    assert not unit.unit.model.is_flying()
    properties = WindowProperties()
    properties.setSize(*size)
    framebuffer = FrameBufferProperties()
    framebuffer.setRgbColor(True)
    framebuffer.setDepthBits(24)
    buffer = app.graphicsEngine.makeOutput(app.pipe, 'flight-controls', 0, framebuffer,
                                          properties, GraphicsPipe.BFRefuseWindow,
                                          app.win.getGsg(), app.win)
    assert buffer is not None
    try:
        app.setAspectRatio(size[0] / size[1])
        buffer.makeDisplayRegion().setCamera(app.cam)
        overlay = buffer.makeDisplayRegion()
        overlay.setSort(20)
        overlay.setClearDepthActive(True)
        overlay.setCamera(app.cam2d)
        app.graphicsEngine.renderFrame()
        app.graphicsEngine.renderFrame()
        image = buffer.getScreenshot()
        assert image.getXSize() == size[0] and image.getYSize() == size[1]
        assert image.write(str(tmp_path / f'flight-controls-{size[0]}.png'))
    finally:
        app.graphicsEngine.removeWindow(buffer)
        app.setAspectRatio(1280 / 720)