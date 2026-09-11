"""Formed charge approaches to loose Skirmishers (Rulebook p. 186)."""

import asyncio
import math
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from panda3d.core import Filename, PNMImage, Point2, Point3, Vec3

from scouts import model_base_boxes
from formed_skirmish_charge import footprint, preview_charge, route_to_model
from tests.test_skirmish_scene import restore, scene as scene


def approach_scene(scene):
    app, defender = restore(scene)
    app.fsm.request('MovementPhase')
    attacker = next(unit for unit in app.units if unit.unitName == 'Warriors')
    for index, other in enumerate(app.units):
        other.bodyNP.setPos(30, 15 + index * 3, 0)
    attacker.bodyNP.setPos(0, -8, 0)
    attacker.bodyNP.setH(0)
    defender.bodyNP.setPos(0, 0, 0)
    defender.bodyNP.setH(37)
    attacker.isDeployed = defender.isDeployed = True
    attacker.deployedAsScouts = defender.deployedAsScouts = False
    app.roundCounter.current_player = 2
    app.unitToMove = attacker
    app.world.doPhysics(1 / 60)
    return app, attacker, defender


def test_cursor_routes_to_actual_skirmisher_base(scene):
    from psychology import obb_distance
    app, attacker, defender = approach_scene(scene)
    original = attacker.bodyNP.getTransform()
    app.movement.pathTowardsMouse(attacker, 0, 0)
    preview = getattr(attacker, 'formedSkirmishPreview', None)
    assert preview is not None
    assert preview.target is defender and preview.error is None
    assert attacker.bodyNP.getTransform() == original
    route = preview.route
    assert min(obb_distance(box, model_base_boxes(defender)[route.target_index])
               for box in route.final_boxes) < 1e-5
    assert app.moveArceDistance == pytest.approx(route.distance)
    assert app.playerNP.getPos().almostEqual(Point3(*route.destination), 1e-5)


@pytest.mark.parametrize('heading', [0, 37, 90, 180, 270])
@pytest.mark.parametrize('offset', [-5, 0, 5])
def test_one_wheel_route_keeps_formed_shape_and_reaches_base(heading, offset):
    from formed_skirmish_charge import rotate
    from psychology import obb_distance
    attackers = [(*rotate((column, row - 6), (0, 0), heading), 0.5, 0.5, heading)
                 for row in (0, -1) for column in (-1, 0, 1)]
    target = (*rotate((offset, 2), (0, 0), heading), 0.5, 0.5, heading + 23)
    origin = (*rotate((0, -6.5), (0, 0), heading), 0)
    route = route_to_model(attackers, [target], 0, origin)
    assert route is not None
    assert abs(route.wheel) <= 90
    assert route.wheel_distance == pytest.approx(abs(math.radians(route.wheel)) * 3)
    assert route.distance == pytest.approx(route.lead + route.wheel_distance + route.advance)
    assert min(obb_distance(box, target) for box in route.final_boxes) < 1e-5
    assert route.pose(0)[0] == pytest.approx(origin)
    assert route.pose(0)[1] == pytest.approx(heading)
    assert footprint(route.final_boxes)[2:4] == pytest.approx((1.5, 1))
    for before, after in zip(attackers, route.final_boxes):
        assert math.dist(before[:2], attackers[0][:2]) == pytest.approx(math.dist(after[:2], route.final_boxes[0][:2]))
    if offset:
        assert abs(route.wheel) > 0
    else:
        assert route.wheel == 0


@pytest.mark.parametrize('dice,succeeds', [([6, 6], True), ([1, 1], False)])
@pytest.mark.parametrize('offset', [-3, 0, 3])
def test_real_route_resolution_uses_preview_wheel_and_roll(scene, dice, succeeds, offset):
    from direct.interval.IntervalGlobal import LerpPosHprInterval, Parallel
    from psychology import obb_distance
    from special_rules import apply_rule_keywords
    app, attacker, defender = approach_scene(scene)
    apply_rule_keywords(attacker.unit.model, ['First Charge'])
    defender.bodyNP.setX(offset)
    origin, facing = attacker.bodyNP.getPos(), attacker.bodyNP.getHpr()
    before = model_base_boxes(defender)
    preview = preview_charge(app, attacker, defender)
    assert preview.error is None
    app.playerNP.setPos(*preview.route.destination)
    app.moveArceDistance = preview.route.distance
    attacker.formedSkirmishCharge = preview
    app.autoRoll = False

    def finish(interval):
        interval.start()
        interval.finish()
        return iter(())

    with patch.object(LerpPosHprInterval, '__await__', finish), \
            patch.object(Parallel, '__await__', finish), \
            patch.object(app.combat, 'swiftstrideChargeChoice', AsyncMock(return_value=False)), \
            patch.object(app.combat, 'rullTerninger', AsyncMock(return_value=([], dice))), \
            patch.object(app.combat, 'alignToEnemy', AsyncMock()) as align, \
            patch.object(app.movement, 'dangerousTerrainTests'):
        asyncio.run(app.combat.chargeInterval(attacker, defender.bodyNP, 37, origin, facing, 'flank'))
    align.assert_not_awaited()
    travel = preview.route.distance if succeeds else max(dice)
    position, heading = preview.route.pose(travel)
    assert tuple(attacker.bodyNP.getPos()) == pytest.approx(position, abs=1e-5)
    assert (attacker.bodyNP.getH() - heading + 180) % 360 - 180 == pytest.approx(0, abs=1e-5)
    assert attacker.chargeAttempts == 1 and not attacker.chargeAttemptPending
    assert bool(getattr(defender, 'firstChargeDisruptedBy', [])) == succeeds
    if succeeds:
        assert attacker.state == defender.state == 'InCombat'
        assert attacker.isInCombatFlank == defender.isInCombatFlank == ['front']
        assert attacker.chargeDistance == pytest.approx(preview.route.distance)
        assert all(min(obb_distance(box, target) for target in model_base_boxes(attacker)) < 1e-5
                   for box in model_base_boxes(defender)[:defender.unit.files])
    else:
        assert attacker.state == 'Moved'
        assert model_base_boxes(defender) == before


def test_closest_visible_target_is_not_unit_center(scene):
    app, attacker, defender = approach_scene(scene)
    preview = preview_charge(app, attacker, defender)
    assert preview.error is None
    from psychology import obb_distance
    targets = model_base_boxes(defender)
    distances = [min(obb_distance(box, target) for box in model_base_boxes(attacker)) for target in targets]
    assert distances[preview.route.target_index] == pytest.approx(min(distances))


@pytest.mark.parametrize('position', [(0, -16), (20, -8), (-20, -8)])
def test_targets_outside_formed_front_arc_are_refused(scene, position):
    app, attacker, defender = approach_scene(scene)
    defender.bodyNP.setPos(*position, 0)
    preview = preview_charge(app, attacker, defender)
    assert preview.route is None and 'front arc' in preview.error


def test_blocked_declaration_restores_movement_before_choices(scene):
    app, attacker, defender = approach_scene(scene)
    defender.bodyNP.setY(25)
    origin, facing = attacker.bodyNP.getPos(), attacker.bodyNP.getHpr()
    with patch.object(app, 'makeChoiceNew', AsyncMock()) as choice, \
            patch.object(app.combat, 'standAndShootOption') as reaction, \
            patch('combat_resolution.rule_skipped') as log:
        asyncio.run(app.combat.chargeAndChargeReaction(
            attacker, None, origin, facing, SimpleNamespace(done='done'), defender=defender))
    choice.assert_not_awaited()
    reaction.assert_not_called()
    assert attacker.bodyNP.getPos() == origin and attacker.bodyNP.getHpr() == facing
    assert not attacker.hasMovedThisTurn and attacker.moveSpentThisTurn == 0
    assert any('exceeds maximum' in call.args[2] for call in log.call_args_list)


def test_obstructed_charge_does_not_pass_through_screen():
    attackers = [(column, row - 6, 0.5, 0.5, 0) for row in (0, -1) for column in (-1, 0, 1)]
    assert route_to_model(attackers, [(0, 2, 0.5, 0.5, 0)], 0, (0, -6.5, 0),
                          obstacles=[(0, -2, 10, 0.5, 0)]) is None


def test_one_wheel_can_be_delayed_until_rear_clears_obstacle():
    from formed_skirmish_charge import path_error
    attackers = [(column, row - 6, 0.5, 0.5, 0) for row in (0, -1, -2, -3) for column in (-1, 0, 1)]
    targets = [(5, 3, 0.5, 0.5, 0)]
    obstacle = (-2, -9, 0.4, 0.4, 0)
    route = route_to_model(attackers, targets, 0, (0, -7.5, 0), obstacles=[obstacle])
    assert route is not None and route.lead > 0
    assert route.wheel < 0
    assert path_error(route, targets, [obstacle]) is None


@pytest.mark.parametrize('destination', [(0, -6), (500, 500), (None, None)])
def test_leaving_target_clears_formed_charge_preview(scene, destination):
    app, attacker, defender = approach_scene(scene)
    app.movement.pathTowardsMouse(attacker, 0, 0)
    assert attacker.formedSkirmishPreview is not None
    app.movement.pathTowardsMouse(attacker, *destination)
    assert attacker.formedSkirmishPreview is None
    assert app.skirmMoveGhost is None


def test_removed_target_refused_before_reading_bases(scene):
    from panda3d.core import NodePath
    app, attacker, defender = approach_scene(scene)
    with patch.object(defender, 'bodyNP', NodePath()), \
            patch('formed_skirmish_charge.model_base_boxes') as boxes:
        preview = preview_charge(app, attacker, defender)
    boxes.assert_not_called()
    assert preview.route is None and 'no longer on the battlefield' in preview.error


def test_changed_target_revalidated_without_movement(scene):
    app, attacker, defender = approach_scene(scene)
    app.movement.pathTowardsMouse(attacker, 0, 0)
    defender.isDeployed = False
    original = attacker.bodyNP.getTransform()
    with patch.object(app, 'makeChoiceNew', AsyncMock()) as choice:
        asyncio.run(app.combat.chargeAndChargeReaction(
            attacker, None, attacker.bodyNP.getPos(), attacker.bodyNP.getHpr(),
            SimpleNamespace(done='done'), defender=defender))
    choice.assert_not_awaited()
    assert attacker.bodyNP.getTransform() == original
    assert not attacker.hasMovedThisTurn


@pytest.mark.parametrize('human', [False, True])
@pytest.mark.parametrize('accepts', [False, True])
def test_cursor_click_confirmation_and_ai_share_route(scene, human, accepts):
    import inspect
    from direct.interval.IntervalGlobal import LerpPosHprInterval, Parallel
    app, attacker, defender = approach_scene(scene)
    app.chargeStage = None
    app.autoCharge = app.autoHold = False
    app.movement.pathTowardsMouse(attacker, 0, 0)
    route = attacker.formedSkirmishPreview.route
    origin = attacker.bodyNP.getTransform()
    scheduled = []

    def schedule(function, *args, **kwargs):
        if inspect.isawaitable(function):
            return function
        scheduled.append((function, kwargs))

    def finish(interval):
        interval.start()
        interval.finish()
        return iter(())

    with patch.object(app.taskMgr, 'add', side_effect=schedule), \
            patch.object(app, 'startTaskFunction'), \
            patch.object(app, 'aiControls', return_value=not human), \
            patch.object(app, 'makeChoiceNew', AsyncMock(side_effect=['Yes' if accepts else 'No', 'hold'])), \
            patch.object(app.combat, 'standAndShootOption', return_value=None), \
            patch.object(app.combat, 'getFlankFromContact', side_effect=AssertionError('No footprint contact')):
        app.movement.moveUnit(attacker)
        declare, arguments = scheduled.pop()
        asyncio.run(declare(*arguments['extraArgs'], SimpleNamespace(done='done')))
        if human and not accepts:
            assert not scheduled and attacker.formedSkirmishCharge is None
            assert attacker.bodyNP.getTransform() == origin
            assert not attacker.hasMovedThisTurn
            return
        resolve, arguments = scheduled.pop()
        assert resolve == app.combat.chargeInterval
        assert attacker.formedSkirmishCharge.route.distance == pytest.approx(route.distance)
        assert attacker.hasMovedThisTurn
        app.autoRoll = False
        with patch.object(LerpPosHprInterval, '__await__', finish), \
                patch.object(Parallel, '__await__', finish), \
                patch.object(app.combat, 'swiftstrideChargeChoice', AsyncMock(return_value=False)), \
                patch.object(app.combat, 'rullTerninger', AsyncMock(return_value=([], [6, 6]))), \
                patch.object(app.movement, 'dangerousTerrainTests'):
            asyncio.run(resolve(*arguments['extraArgs']))
    assert attacker.state == defender.state == 'InCombat'


def test_screened_nearest_model_is_skipped_but_visible_nearest_is_not(scene):
    from formed_skirmish_charge import model_can_see
    app, attacker, defender = approach_scene(scene)
    screen = next(unit for unit in app.units if unit.unitName == 'P2 Scouts A')
    for other in app.units:
        other.isDeployed = other in (attacker, defender, screen)
    sources = [(column, -8, 0.5, 0.5, 0) for column in (-1, 0, 1)]
    targets = [(0, 0, 0.5, 0.5, 0), (2, 0.5, 0.5, 0.5, 0)]
    blockers = [(0, -1.5, 0.85, 0.25, 0)]
    assert not any(model_can_see(source, targets[:1], blockers, facing=0) for source in sources)
    mapping = {attacker: sources, defender: targets, screen: blockers}
    with patch('formed_skirmish_charge.model_base_boxes', side_effect=lambda unit: mapping[unit]):
        preview = preview_charge(app, attacker, defender)
    assert preview.error is None and preview.route.target_index == 1


@pytest.mark.parametrize('covered', [False, True])
def test_route_terrain_uses_full_formation_not_center_line(scene, covered, capsys):
    from formed_skirmish_charge import route_features, route_allowance
    app, attacker, defender = approach_scene(scene)
    route = preview_charge(app, attacker, defender).route
    piece = SimpleNamespace(center=Point3(2, -5, 0), width=0.3, height=0.3,
                            movement_modifier=-1, is_dangerous=True, is_impassable=False,
                            blocks_line_of_sight=False,
                            contains=lambda point: abs(point.x - 2) <= 0.15 and abs(point.y + 5) <= 0.15)
    with patch.object(app.terrain_manager, 'terrain_pieces', [piece]), \
            patch.object(attacker.unit.model, 'is_move_through_cover', return_value=covered):
        assert route_features(app, route) == [piece]
        assert app.terrain_manager.get_terrain_between(Point3(*route.origin), Point3(*route.destination)) == []
        assert route_allowance(app, attacker, route) == (3 if covered else 2)
        attacker.formedSkirmishCharge = SimpleNamespace(route=route)
        assert app.combat.chargeThroughDifficult(attacker, Point3(*route.origin)) is not covered
        capsys.readouterr()
        assert app.combat.chargeDistance(attacker, Point3(*route.origin), [2, 6]) == (9 if covered else 4)
        output = capsys.readouterr().out
        assert 'Charge Move' in output
        assert ('Move Through Cover' in output) is covered


def test_target_in_combat_since_preview_is_refused(scene):
    app, attacker, defender = approach_scene(scene)
    app.movement.pathTowardsMouse(attacker, 0, 0)
    defender.request('InCombat')
    with patch.object(app, 'makeChoiceNew', AsyncMock()) as choice:
        asyncio.run(app.combat.chargeAndChargeReaction(
            attacker, None, attacker.bodyNP.getPos(), attacker.bodyNP.getHpr(),
            SimpleNamespace(done='done'), defender=defender))
    choice.assert_not_awaited()
    assert not attacker.hasMovedThisTurn


def test_repeated_cursor_preview_cached_but_blocker_changes_invalidate(scene):
    app, attacker, defender = approach_scene(scene)
    with patch('formed_skirmish_charge.preview_charge', wraps=preview_charge) as plan:
        app.movement.pathTowardsMouse(attacker, 0, 0)
        app.movement.pathTowardsMouse(attacker, 0, 0)
        assert plan.call_count == 1
        blocker = next(unit for unit in app.units if unit.unitName == 'P2 Scouts A')
        blocker.isDeployed = True
        blocker.bodyNP.setPos(0, -4, 0)
        app.movement.pathTowardsMouse(attacker, 0, 0)
        assert plan.call_count == 2
        assert attacker.formedSkirmishPreview.error is not None


def test_loading_save_clears_unconfirmed_route(scene):
    from persistence import load_game_state
    app, attacker, defender = approach_scene(scene)
    app.movement.pathTowardsMouse(attacker, 0, 0)
    attacker.formedSkirmishCharge = attacker.formedSkirmishPreview
    load_game_state(app, str(scene[1]))
    assert attacker.formedSkirmishPreview is None
    assert attacker.formedSkirmishCharge is None
    assert attacker.formedSkirmishCache is None


def test_queued_loose_target_survives_reload_without_reselecting_sight(scene, tmp_path):
    from charge_declarations import resolve_declarations
    from persistence import load_game_state, save_game_state
    from tests.test_shieldwall_scene import combat_tasks
    app, attacker, defender = approach_scene(scene)
    origin, facing = attacker.bodyNP.getPos(), attacker.bodyNP.getHpr()
    with combat_tasks(app) as run, patch.object(app, 'aiControls', return_value=True):
        run(app.combat.chargeAndChargeReaction(attacker, None, origin, facing,
                                              SimpleNamespace(done=None), defender=defender))
    assert len(app.chargeDeclarations) == 1
    target_index = app.chargeDeclarations[0].target_index
    path = save_game_state(app, str(tmp_path / 'loose-declaration.json'))
    load_game_state(app, path)
    assert app.chargeDeclarations[0].target_index == target_index
    with combat_tasks(app) as run, \
            patch.object(app, 'aiControls', return_value=True), \
            patch('formed_skirmish_charge.model_can_see', side_effect=AssertionError('Sight is declaration-time')), \
            patch.object(app.combat, 'standAndShootOption', return_value=None), \
            patch.object(app.combat, 'swiftstrideChargeChoice', AsyncMock(return_value=False)), \
            patch.object(app.combat, 'rullTerninger', AsyncMock(return_value=([], [6, 6]))):
        run(resolve_declarations(app))
    assert attacker.state == defender.state == 'InCombat'
    assert app.chargeStage == 'remaining'


@pytest.mark.parametrize('destroyed', [False, True])
def test_stand_and_shoot_does_not_reselect_declared_target(scene, destroyed):
    from tests.test_shieldwall_scene import combat_tasks
    app, attacker, defender = approach_scene(scene)
    app.chargeStage = None
    expected_target = preview_charge(app, attacker, defender).route.target_index
    origin, facing = attacker.bodyNP.getPos(), attacker.bodyNP.getHpr()
    app.autoCharge, app.autoHold, app.autoRoll = True, False, False

    async def volley(*args, **kwargs):
        app.movement.removeModelsFromUnit(attacker, attacker.unit.nmodels if destroyed else 1)

    with patch.object(app, 'aiControls', return_value=True), \
            patch.object(app.combat, 'standAndShootOption', return_value=SimpleNamespace(weapon={}, distance=6)), \
            patch.object(app.combat, 'fireAndFleeOption', return_value=False), \
            patch.object(app.combat, 'standAndShoot', side_effect=volley), \
            patch('formed_skirmish_charge.preview_charge', wraps=preview_charge) as declared, \
            patch.object(app.taskMgr, 'add') as scheduled:
        asyncio.run(app.combat.chargeAndChargeReaction(
            attacker, None, origin, facing, SimpleNamespace(done='done'), defender=defender))
    assert declared.call_count == 1
    calls = [call for call in scheduled.call_args_list if call.args[0] == app.combat.chargeInterval]
    if destroyed:
        assert not calls and attacker.formedSkirmishCharge is None
        return
    assert attacker.formedSkirmishCharge.route.target_index == expected_target
    assert len(calls) == 1
    with combat_tasks(app) as run, \
            patch('formed_skirmish_charge.model_can_see', side_effect=AssertionError('Sight is declaration-time')), \
            patch.object(app.combat, 'swiftstrideChargeChoice', AsyncMock(return_value=False)), \
            patch.object(app.combat, 'rullTerninger', AsyncMock(return_value=([], [6, 6]))), \
            patch.object(app.movement, 'dangerousTerrainTests'):
        run(app.combat.chargeInterval(*calls[0].kwargs['extraArgs']))
    assert attacker.state == defender.state == 'InCombat'


@pytest.mark.parametrize('size', [(1280, 720), (800, 600)])
def test_route_preview_and_live_animation_render(scene, tmp_path, size):
    from tests.test_shieldwall_scene import combat_tasks
    app, attacker, defender = approach_scene(scene)
    defender.bodyNP.setX(-3)
    app.world.doPhysics(1 / 60)
    app.movement.pathTowardsMouse(attacker, -3, 0)
    preview = attacker.formedSkirmishPreview
    assert preview.error is None
    origin, facing = attacker.bodyNP.getPos(), attacker.bodyNP.getHpr()
    defender_before = model_base_boxes(defender)
    old_size = app.win.getXSize(), app.win.getYSize()
    camera_transform = app.camera.getTransform()
    window = app.openWindow(type='offscreen', size=size, makeCamera=False)
    for order, camera in enumerate((app.cam, app.cam2d, app.cam2dp)):
        region = window.makeDisplayRegion()
        region.setCamera(camera)
        region.setSort(order * 10)
    try:
        app.adjustWindowAspectRatio(size[0] / size[1])
        app.camera.setPos(-1, -29, 42)
        app.camera.lookAt(-1, -4, 0)
        status = app.skirmishMoveStatus
        assert 'Target model' in status['text'] and 'Wheel' in status['text']
        assert app.debugTextInfo.getText() == app.diceInfoText.getText() == ''
        left, right, bottom, top = status.getBounds()
        assert 0 <= status.getX() + left < status.getX() + right <= 2 * size[0] / size[1]
        assert -2 <= status.getZ() + bottom < status.getZ() + top <= 0
        app.graphicsEngine.renderFrame()
        app.graphicsEngine.renderFrame()
        image = PNMImage()
        assert window.getScreenshot(image)
        assert image.write(Filename.fromOsSpecific(str(tmp_path / f'route-{size[0]}x{size[1]}.png')))
        from psychology import _box_corners
        for box in preview.route.final_boxes:
            corner = _box_corners(*box)[0]
            screen = Point2()
            assert app.camLens.project(app.cam.getRelativePoint(app.render, Point3(*corner, 0.3)), screen)
            horizontal = round((screen.x + 1) * size[0] / 2)
            vertical = round((1 - screen.y) * size[1] / 2)
            assert any(image.getXel(column, row).z > 0.8 and image.getXel(column, row).y > 0.7
                       for column in range(horizontal - 3, horizontal + 4)
                       for row in range(vertical - 3, vertical + 4))
        from skirmish_ui import clear_plot_preview
        clear_plot_preview(app)
        attacker.formedSkirmishCharge = preview
        app.autoRoll = False
        samples = []

        def watch(task):
            samples.append((tuple(attacker.bodyNP.getPos()), model_base_boxes(defender)))
            return task.cont

        with combat_tasks(app) as run, \
                patch.object(app.combat, 'swiftstrideChargeChoice', AsyncMock(return_value=False)), \
                patch.object(app.combat, 'rullTerninger', AsyncMock(return_value=([], [6, 6]))), \
                patch.object(app.movement, 'dangerousTerrainTests'):
            app.taskMgr.add(watch)
            run(app.combat.chargeInterval(attacker, defender.bodyNP, 0, origin, facing, 'front'))
        assert attacker.state == defender.state == 'InCombat'
        assert any(math.dist(position, origin) > 0.1 and
                   math.dist(position, preview.route.destination) > 0.1 for position, boxes in samples)
        assert all(boxes == defender_before for position, boxes in samples
                   if math.dist(position, preview.route.destination) > 1e-4)
        app.graphicsEngine.renderFrame()
        app.graphicsEngine.renderFrame()
        assert window.getScreenshot(image)
        assert image.write(Filename.fromOsSpecific(str(tmp_path / f'contact-{size[0]}x{size[1]}.png')))
        attacker.model.hide()
        defender.model.hide()
        app.graphicsEngine.renderFrame()
        app.graphicsEngine.renderFrame()
        background = PNMImage()
        assert window.getScreenshot(background)
        for box in [*model_base_boxes(attacker), *model_base_boxes(defender)]:
            screen = Point2()
            assert app.camLens.project(app.cam.getRelativePoint(app.render, Point3(box[0], box[1], 0.2)), screen)
            assert abs(screen.x) < 0.8 and abs(screen.y) < 0.8
            horizontal = round((screen.x + 1) * size[0] / 2)
            vertical = round((1 - screen.y) * size[1] / 2)
            assert any((image.getXel(column, row) - background.getXel(column, row)).length() > 0.1
                       for column in range(horizontal - 8, horizontal + 9)
                       for row in range(vertical - 8, vertical + 9))
    finally:
        attacker.model.show()
        defender.model.show()
        app.camera.setTransform(camera_transform)
        app.closeWindow(window, keepCamera=True)
        app.adjustWindowAspectRatio(old_size[0] / old_size[1])