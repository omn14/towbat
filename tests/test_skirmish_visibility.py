"""Declaration-time Skirmisher sight and strict majority (pp. 103, 184, 186)."""

import asyncio
from math import cos, radians, sin
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from panda3d.core import Filename, PNMImage, Point3, Vec2

from characters import join_unit, slay_character
from scouts import model_base_boxes
from skirmish_movement import preview_action
from skirmish_visibility import ChargeVisibility, charge_visibility, model_can_see
from tests.test_skirmish_scene import restore, scene as scene


def column_scenario(scene):
    app, member = restore(scene)
    app.fsm.request('MovementPhase')
    enemy = next(unit for unit in app.units if unit.unitName == 'Warriors')
    for index, other in enumerate(app.units):
        other.bodyNP.setPos(25, 10 + index, 0)
    app.movement.removeModelsFromUnit(member, max(0, len(member.model.getChildren()) - 2))
    app.movement.removeModelsFromUnit(enemy, max(0, len(enemy.model.getChildren()) - 1))
    member.bodyNP.setPos(0, -6, 0)
    enemy.bodyNP.setPos(0, 0, 0)
    member.bodyNP.setH(0)
    enemy.bodyNP.setH(0)
    for record, vertical in zip(member.skirmishLayout, (0, -1.6)):
        record['x'], record['y'] = 0, vertical
    member.rebuildFootprint()
    return app, member, enemy


def test_exactly_half_visible_cannot_declare_charge(scene):
    app, member, enemy = column_scenario(scene)
    before = model_base_boxes(member)
    app.arcPoint = Vec2(0.5, 0.49)
    with patch('skirmish_movement.contact_target_at', return_value=enemy):
        preview = preview_action(app, member)
    assert preview.charge_target is enemy
    assert preview.error is not None and '1/2' in preview.error
    assert model_base_boxes(member) == before
    assert not member.hasMovedThisTurn and member.moveSpentThisTurn == 0


@pytest.mark.parametrize('visible,total,allowed', [(0, 0, False), (0, 1, False), (1, 1, True),
    (1, 2, False), (2, 3, True), (3, 6, False), (4, 6, True)])
def test_strict_majority(visible, total, allowed):
    result = ChargeVisibility((True,) * visible + (False,) * (total - visible))
    assert result.visible == visible and result.total == total
    assert result.allowed is allowed


def rotated(boxes, angle):
    cosine, sine = cos(radians(angle)), sin(radians(angle))
    return [(box[0] * cosine - box[1] * sine, box[0] * sine + box[1] * cosine,
             box[2], box[3], box[4] + angle) for box in boxes]


@pytest.mark.parametrize('angle', [0, 37, 90, 180, 270])
@pytest.mark.parametrize('gap,visible', [(0, False), (0.0001, True), (0.15, True)])
def test_rotated_slit_between_individual_bases(angle, gap, visible):
    observer = (0.173, -6, 0.5, 0.5, 0)
    target = (0, 4, 1, 0.5, 0)
    left = (-1 + 0.173 - gap / 2, 0, 1, 0.5, 0)
    right = (1 + 0.173 + gap / 2, 0, 1, 0.5, 0)
    observer, target, left, right = rotated([observer, target, left, right], angle)
    assert model_can_see(observer, [target], [left, right]) is visible


def test_visible_target_edge_is_enough_when_centre_is_blocked():
    observer = (0, -6, 0.5, 0.5, 0)
    target = (0, 4, 3, 0.5, 0)
    assert model_can_see(observer, [target], [(0, 0, 0.5, 0.5, 0)])


def test_blocker_behind_target_does_not_hide_it():
    assert model_can_see((0, -6, 0.5, 0.5, 0), [(0, 0, 0.5, 0.5, 0)], [(0, 4, 4, 1, 0)])


def test_empty_target_and_completely_screened_target():
    observer = (0, -6, 0.5, 0.5, 0)
    assert not model_can_see(observer, [])
    assert not model_can_see(observer, [(0, 4, 0.5, 0.5, 0)], [(0, 0, 4, 1, 0)])


@pytest.mark.parametrize('ai', [False, True])
def test_declaration_checks_original_position_before_choices_or_dice(scene, ai):
    app, member, enemy = column_scenario(scene)
    origin, facing = member.bodyNP.getPos(), member.bodyNP.getHpr()
    before = model_base_boxes(member)
    member.bodyNP.setPos(1.5, 0, 0)
    member.bodyNP.setH(180)
    assert charge_visibility(app, member, enemy).allowed
    app.autoCharge = app.autoHold = True
    contact = SimpleNamespace(getNode1=lambda: enemy.bodyNP.node())
    with patch.object(app, 'aiControls', return_value=ai), \
            patch.object(app, 'makeChoiceNew', AsyncMock()) as choices, \
            patch.object(app.combat, 'standAndShootOption') as reactions, \
            patch.object(app.combat, 'rollMoveDice', AsyncMock()) as dice, \
            patch.object(app, 'startTaskFunction'), \
            patch('combat_resolution.rule_skipped') as skipped:
        result = asyncio.run(app.combat.chargeAndChargeReaction(
            member, contact, origin, facing, SimpleNamespace(done='done')))
    assert result == 'done'
    choices.assert_not_called()
    reactions.assert_not_called()
    dice.assert_not_called()
    assert '1/2' in skipped.call_args.args[2]
    assert not app.autoCharge and not app.autoHold
    assert model_base_boxes(member) == before
    assert not member.hasMovedThisTurn and member.moveSpentThisTurn == 0


def test_original_position_query_does_not_mutate_live_contact(scene):
    app, member, enemy = column_scenario(scene)
    origin, facing = member.bodyNP.getPos(), member.bodyNP.getHpr()
    member.bodyNP.setPos(1.5, 0, 0)
    member.bodyNP.setH(180)
    before = model_base_boxes(member)
    result = charge_visibility(app, member, enemy, origin, facing)
    assert result.models == (True, False)
    assert model_base_boxes(member) == before


def add_visible_character(app, member):
    character = app._create_unit(dict(name='Captain of the Empire', nmodels=1, files=1, ranks=1),
                                 1, 'Sight Character')
    assert join_unit(app, character, member)
    member.skirmishCharacterPosition = [1.5, -0.8]
    member.placeCharacter()
    return character


def test_joined_character_counts_once_and_can_supply_majority(scene):
    app, member, enemy = column_scenario(scene)
    character = add_visible_character(app, member)
    before = model_base_boxes(member)
    sight = charge_visibility(app, member, enemy)
    assert sight.models == (True, False, True)
    assert sight.allowed and sight.total == 3
    assert character.hostUnit is member
    assert model_base_boxes(member) == before


@pytest.mark.parametrize('name', ['P1 Scouts', 'P2 Scouts A'])
def test_friendly_and_enemy_models_both_block_sight(scene, name):
    app, member, enemy = column_scenario(scene)
    blocker = next(unit for unit in app.units if unit.unitName == name)
    blocker.bodyNP.setPos(0, -3, 0)
    blocker.bodyNP.setH(0)
    blocker.isDeployed = True
    blocker.request('InCombat')
    assert charge_visibility(app, member, enemy).models == (False, False)
    blocker.isDeployed = False
    assert charge_visibility(app, member, enemy).models == (True, False)


@pytest.mark.parametrize('blocking,center_y,visible', [(True, -3, 0), (False, -3, 1), (True, 0, 1)])
def test_terrain_blocks_through_but_not_onto(scene, blocking, center_y, visible):
    app, member, enemy = column_scenario(scene)
    feature = SimpleNamespace(center=Point3(0, center_y, 0), width=4, height=1,
                              blocks_line_of_sight=blocking,
                              contains=lambda point: abs(point.x) <= 2 and abs(point.y - center_y) <= 0.5)
    with patch.object(app.terrain_manager, 'terrain_pieces', [feature]):
        assert charge_visibility(app, member, enemy).visible == visible


def test_preview_human_click_and_direct_ai_move_share_refusal(scene):
    app, member, enemy = column_scenario(scene)
    app.unitToMove = member
    app.world.doPhysics(1 / 60)
    before = model_base_boxes(member)
    with patch('rules_log.rule_log') as applied, patch('rules_log.rule_skipped') as skipped:
        app.movement._skirmishMovePreview(member, enemy.bodyNP.getPos())
        applied.assert_not_called()
        skipped.assert_not_called()
        assert '1/2' in app.skirmishMoveStatus['text']
        assert app.skirmMoveGhost is None
        app.startTaskFunction(app.taskLoopPathTowardsMouse, 'taskLoopPathTowardsMouse')
        app.onRightClick(member)
        assert '1/2' in skipped.call_args.args[2]
    with patch.object(app.taskMgr, 'add') as scheduled, \
            patch.object(app, 'startTaskFunction'), patch('movement_system.rule_skipped') as skipped:
        app.movement.moveUnit(member)
    assert not any(call.args[0] == app.chargeAndChargeReaction for call in scheduled.call_args_list)
    assert '1/2' in skipped.call_args.args[2]
    assert model_base_boxes(member) == before
    assert not member.hasMovedThisTurn and member.moveSpentThisTurn == 0
    assert getattr(app, 'skirmishEditor', None) is None


def test_valid_declaration_keeps_sight_decision_after_stand_and_shoot(scene):
    app, member, enemy = column_scenario(scene)
    character = add_visible_character(app, member)
    origin, facing = member.bodyNP.getPos(), member.bodyNP.getHpr()
    contact = SimpleNamespace(getNode1=lambda: enemy.bodyNP.node())
    app.autoCharge, app.autoHold = True, False

    async def volley(*args, **kwargs):
        slay_character(app, character)

    with patch.object(app, 'aiControls', return_value=True), \
            patch.object(app.combat, 'standAndShootOption', return_value=SimpleNamespace(weapon={}, distance=5)), \
            patch.object(app.combat, 'fireAndFleeOption', return_value=False), \
            patch.object(app.combat, 'standAndShoot', side_effect=volley), \
            patch.object(app.combat, 'getFlankFromContact', return_value=('front', 0)), \
            patch.object(app.taskMgr, 'add') as scheduled, \
            patch('skirmish_visibility.charge_visibility', wraps=charge_visibility) as checked, \
            patch('combat_resolution.rule_log') as logged:
        asyncio.run(app.combat.chargeAndChargeReaction(
            member, contact, origin, facing, SimpleNamespace(done='done')))
    assert checked.call_count == 1
    assert '2/3' in logged.call_args.args[2]
    assert member.hasMovedThisTurn
    assert any(call.args[0] == app.combat.chargeInterval for call in scheduled.call_args_list)
    assert charge_visibility(app, member, enemy).models == (True, False)


def test_human_confirmation_shows_count_and_cancel_retains_move(scene):
    app, member, enemy = column_scenario(scene)
    add_visible_character(app, member)
    origin, facing = member.bodyNP.getPos(), member.bodyNP.getHpr()
    before = model_base_boxes(member)
    app.autoCharge = app.autoHold = False
    contact = SimpleNamespace(getNode1=lambda: enemy.bodyNP.node())
    with patch.object(app, 'aiControls', return_value=False), \
            patch.object(app, 'makeChoiceNew', AsyncMock(return_value='No')) as choice, \
            patch.object(app.taskMgr, 'add', side_effect=lambda coroutine, *args, **kwargs: coroutine), \
            patch.object(app, 'startTaskFunction'), patch('combat_resolution.rule_log') as logged:
        asyncio.run(app.combat.chargeAndChargeReaction(
            member, contact, origin, facing, SimpleNamespace(done='done')))
    assert '2/3' in choice.call_args.kwargs['detail']
    logged.assert_not_called()
    assert model_base_boxes(member) == before
    assert not member.hasMovedThisTurn and member.moveSpentThisTurn == 0


def test_pursuit_contact_is_not_a_charge_declaration(scene):
    app, member, enemy = column_scenario(scene)
    member.request('IsPursuing')
    app.autoCharge = app.autoHold = True
    contact = SimpleNamespace(getNode1=lambda: enemy.bodyNP.node())
    with patch('skirmish_visibility.charge_visibility', side_effect=AssertionError('No declaration')), \
            patch.object(app.combat, 'standAndShootOption', return_value=None), \
            patch.object(app.combat, 'getFlankFromContact', return_value=('front', 0)), \
            patch.object(app.taskMgr, 'add') as scheduled:
        asyncio.run(app.combat.chargeAndChargeReaction(member, contact, member.bodyNP.getPos(),
            member.bodyNP.getHpr(), SimpleNamespace(done='done')))
    assert any(call.args[0] == app.combat.chargeInterval for call in scheduled.call_args_list)


@pytest.mark.parametrize('width,height', [(1280, 720), (800, 600)])
@pytest.mark.parametrize('allowed', [False, True])
def test_visibility_status_renders_without_overflow(scene, tmp_path, width, height, allowed):
    app, member, enemy = column_scenario(scene)
    if allowed:
        add_visible_character(app, member)
    app.unitToMove = member
    app.world.doPhysics(1 / 60)
    old_size = app.win.getXSize(), app.win.getYSize()
    camera_transform = app.camera.getTransform()
    window = app.openWindow(type='offscreen', size=(width, height), makeCamera=False)
    for order, camera in enumerate((app.cam, app.cam2d, app.cam2dp)):
        region = window.makeDisplayRegion()
        region.setCamera(camera)
        region.setSort(order * 10)
    app.adjustWindowAspectRatio(width / height)
    app.camera.setPos(0, -25, 45)
    app.camera.lookAt(0, -3, 0)
    try:
        app.movement._skirmishMovePreview(member, enemy.bodyNP.getPos())
        text = app.skirmishMoveStatus['text']
        assert ('2/3' if allowed else '1/2') in text
        assert text.startswith('CHARGE' if allowed else 'BLOCKED')
        assert (app.skirmMoveGhost is not None) is allowed
        left, right, bottom, top = app.skirmishMoveStatus.getBounds()
        assert 0 <= app.skirmishMoveStatus.getX() + left
        assert app.skirmishMoveStatus.getX() + right <= 2 * width / height
        assert -2 <= app.skirmishMoveStatus.getZ() + bottom
        assert app.skirmishMoveStatus.getZ() + top <= 0
        app.graphicsEngine.renderFrame()
        app.graphicsEngine.renderFrame()
        image = PNMImage()
        assert window.getScreenshot(image)
        colors = {tuple(image.getXel(column, row)) for column in range(0, width, 20)
                  for row in range(0, height, 20)}
        assert len(colors) > 30
        assert image.write(Filename.fromOsSpecific(str(
            tmp_path / f'charge-sight-{allowed}-{width}x{height}.png')))
    finally:
        app.setGroundOverlay(False)
        app.camera.setTransform(camera_transform)
        app.closeWindow(window, keepCamera=True)
        app.adjustWindowAspectRatio(old_size[0] / old_size[1])