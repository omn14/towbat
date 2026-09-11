"""Skirmish formation through the real application and save/load lifecycle."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from panda3d.core import Filename, NodePath, PNMImage, Point2, Point3, Vec2, Vec3, Vec4

from characters import join_unit, slay_character
from persistence import load_game_state, save_game_state
from scouts import model_base_boxes
from skirmish import coherency_error
from skirmish_movement import commit_move, current_positions, preview_action, preview_move
from skirmish_ui import clear_plot_preview
from tests.test_scouts_scene import build_scenario


def shader_vector(surface, name):
    """Copy the borrowed vector while its ShaderInput owner is still alive."""
    shader_input = surface.getShaderInput(name)
    return Vec4(shader_input.getVector())


@pytest.mark.parametrize('active', [False, True])
def test_clear_plot_preview_only_disables_active_overlay(active):
    game = SimpleNamespace(ground=NodePath('range-test'), setGroundOverlay=Mock())
    game.ground.setShaderInput('skirmishRangeActive', active)
    clear_plot_preview(game)
    if active:
        game.setGroundOverlay.assert_called_once_with(False)
    else:
        game.setGroundOverlay.assert_not_called()


@pytest.fixture(scope='module')
def scene(tmp_path_factory):
    app = build_scenario()
    baseline = tmp_path_factory.mktemp('skirmish') / 'baseline.json'
    save_game_state(app, str(baseline))
    yield app, baseline
    app.destroy()


def restore(scene):
    app, baseline = scene
    load_game_state(app, str(baseline))
    member = next(unit for unit in app.units if unit.unitName == 'Normal Rangers')
    assert member.isSkirmisher
    return app, member


@pytest.mark.parametrize('command', [False, True])
@pytest.mark.parametrize('human', [False, True])
@pytest.mark.parametrize('dice,rallied', [([6, 6], False), ([1, 1], True)])
def test_fleeing_compact_unit_waits_for_successful_rally(scene, tmp_path, command, human, dice, rallied):
    from tests.test_rallying_cry_scene import strategy_tasks
    app, member = restore(scene)
    app.AIplayer2.active = False
    app.fsm.request('CombatPhase')
    member.request('InCombat')
    member.request('IsFleeing')
    before = model_base_boxes(member)
    identities = [record['id'] for record in member.skirmishLayout]
    app.fsm.request('StrategyPhase')
    assert member.skirmishCombat and member.state == 'IsFleeing'
    assert model_base_boxes(member) == before
    saved = tmp_path / 'compact-fleeing.json'
    save_game_state(app, str(saved))
    load_game_state(app, str(saved))
    assert member.skirmishCombat and member.state == 'IsFleeing'
    for actual, previous in zip(model_base_boxes(member), before):
        assert actual == pytest.approx(previous, abs=1e-5)
    before = model_base_boxes(member)
    app.strategyCommandDone = True
    results = []

    async def rally():
        results.append(await app.rallyUnit(member, command=command))

    with strategy_tasks(app) as run, \
            patch.object(app, 'aiControls', return_value=not human), \
            patch.object(app, 'rollLeadershipDice', AsyncMock(return_value=dice)), \
            patch('psychology.reroll_leadership', AsyncMock(return_value=dice)), \
            patch.object(app, 'freeReformUnit', lambda unit, task: task.done):
        run(rally())
    assert results == [rallied]
    assert member.skirmishCombat is not rallied
    assert [record['id'] for record in member.skirmishLayout] == identities
    if rallied:
        assert member.state == 'Idle' and member.cannotChargeThisTurn
        assert coherency_error(model_base_boxes(member)) is None
        for actual, previous in zip(model_base_boxes(member), before):
            assert actual == pytest.approx(previous, abs=0.001)
    else:
        assert member.state == 'IsFleeing'
        assert model_base_boxes(member) == before


@pytest.mark.parametrize('reengaged', [False, True])
def test_nonfleeing_compact_unit_separates_only_at_combat_phase_end(scene, reengaged):
    app, member = restore(scene)
    app.fsm.request('CombatPhase')
    member.request('InCombat')
    member.request('Moved')
    before = model_base_boxes(member)
    assert member.skirmishCombat
    if reengaged:
        member.request('InCombat')
    with patch.object(app.fsm, '_spell_origin', 'CombatPhase', create=True):
        app.fsm.exitCombatPhase()
    assert member.skirmishCombat and model_base_boxes(member) == before
    app.fsm.request('StrategyPhase')
    assert member.skirmishCombat is reengaged
    if reengaged:
        assert model_base_boxes(member) == before
    else:
        assert coherency_error(model_base_boxes(member)) is None
        for actual, previous in zip(model_base_boxes(member), before):
            assert actual == pytest.approx(previous, abs=0.001)


def test_combat_flee_and_casualties_keep_compact_survivors_until_rally(scene):
    from tests.test_shieldwall_scene import combat_tasks
    app, member = restore(scene)
    app.terrain_manager.clear()
    app.fsm.request('CombatPhase')
    enemy = next(unit for unit in app.units if unit.unitName == 'Warriors')
    for index, other in enumerate(app.units):
        other.bodyNP.setPos(25, 10 + index, 0)
    member.bodyNP.setPos(0, -2, 0)
    enemy.bodyNP.setPos(0, 2, 0)
    for unit, target in ((member, enemy), (enemy, member)):
        unit.request('InCombat')
        unit.isInCombatWith = [target]
        unit.isInCombatFlank = ['front']
    before = [child.getPos(member.bodyNP) for child in member.model.getChildren()]
    identities = [record['id'] for record in member.skirmishLayout]
    app.world.doPhysics(1 / 60)
    with combat_tasks(app) as run, \
            patch.object(app.combat, 'swiftstrideChoice', AsyncMock(return_value=False)), \
            patch.object(app.combat, 'rollMoveDice', AsyncMock(return_value=[2, 3])):
        run(app.combat.fleeMove(member, 'break'))
    assert member.state == 'IsFleeing' and member.skirmishCombat
    assert member.bodyNP.getY() < -6.9
    assert [child.getPos(member.bodyNP) for child in member.model.getChildren()] == before
    app.movement.removeModelsFromUnit(member, 1)
    assert [record['id'] for record in member.skirmishLayout] == identities[:-1]
    survivors = model_base_boxes(member)
    app.fsm.request('StrategyPhase')
    assert member.state == 'IsFleeing' and member.skirmishCombat
    assert model_base_boxes(member) == survivors


def test_joined_character_separates_without_moving_ordinary_models_to_old_layout(scene):
    app, member = restore(scene)
    character = app._create_unit(dict(name='Captain of the Empire', nmodels=1,
                                     files=1, ranks=1), 1, 'Rally Character')
    assert join_unit(app, character, member)
    member.request('InCombat')
    before = model_base_boxes(member)
    member.request('Moved')
    member.spreadToSkirmish()
    assert not member.skirmishCombat
    assert coherency_error(model_base_boxes(member)) is None
    assert character.bodyNP.node() not in app.world.getRigidBodies()
    for actual, previous in zip(model_base_boxes(member), before):
        assert actual == pytest.approx(previous, abs=0.001)


def test_real_save_load_restores_casualties_and_exact_layout(scene, tmp_path):
    app, member = restore(scene)
    before = model_base_boxes(member)
    identity = member.savedSkirmishLayout()
    saved = tmp_path / 'loose.json'
    save_game_state(app, str(saved))
    app.movement.removeModelsFromUnit(member, 3)
    load_game_state(app, str(saved))
    assert model_base_boxes(member) == before
    assert member.savedSkirmishLayout() == identity


def test_legacy_save_rebuilds_loose_bases(scene, tmp_path):
    app, member = restore(scene)
    saved = tmp_path / 'legacy.json'
    save_game_state(app, str(saved))
    data = json.loads(saved.read_text())
    for record in data['units']:
        record.pop('skirmish_layout', None)
    saved.write_text(json.dumps(data))
    member.layOutRanks()
    load_game_state(app, str(saved))
    assert coherency_error(model_base_boxes(member)) is None


def test_join_keeps_ordinary_positions_and_restores_character(scene, tmp_path):
    app, member = restore(scene)
    before = model_base_boxes(member)
    character = app._create_unit(dict(name='Captain of the Empire', nmodels=1,
                                     files=1, ranks=1), 1, 'Loose Character')
    assert join_unit(app, character, member)
    assert model_base_boxes(member)[:member.unit.nmodels] == before
    assert coherency_error(model_base_boxes(member)) is None
    saved = tmp_path / 'joined.json'
    save_game_state(app, str(saved))
    positions = model_base_boxes(member)
    load_game_state(app, str(saved))
    assert model_base_boxes(member) == positions
    assert character.bodyNP.node() not in app.world.getRigidBodies()
    assert character.hostUnit is member


def test_slain_bridge_character_is_replaced_without_extra_casualties(scene):
    app, member = restore(scene)
    character = app._create_unit(dict(name='Captain of the Empire', nmodels=1,
                                     files=1, ranks=1), 1, 'Bridge Character')
    assert join_unit(app, character, member)
    member.unit.nmodels = 2
    member.restoreSkirmishLayout(dict(
        models=[dict(id=0, x=-1.5, y=0), dict(id=1, x=1.5, y=0)],
        combat=False, character=[0, 0]))
    assert coherency_error(model_base_boxes(member)) is None
    slay_character(app, character)
    assert member.unit.nmodels == 2
    assert coherency_error(model_base_boxes(member)) is None
    assert any(abs(record['x']) < 1e-6 for record in member.skirmishLayout)


def test_preview_is_read_only_and_measures_individual_travel(scene):
    app, member = restore(scene)
    before = model_base_boxes(member)
    proposed = [(position[0], position[1] + 1) for position in current_positions(member)]
    preview = preview_move(app, member, proposed)
    assert preview.error is None
    assert preview.distance == pytest.approx(1)
    assert model_base_boxes(member) == before
    assert not member.hasMovedThisTurn


def remaining_moves(app):
    import asyncio
    from charge_declarations import resolve_declarations
    asyncio.run(resolve_declarations(app))


def test_adjustment_cannot_grant_free_movement(scene):
    app, member = restore(scene)
    app.fsm.request('MovementPhase')
    remaining_moves(app)
    positions = [(position[0] + 20, position[1]) for position in current_positions(member)]
    before = model_base_boxes(member)
    assert not commit_move(app, member, positions)
    assert model_base_boxes(member) == before
    assert not member.hasMovedThisTurn


def test_adjustment_spends_movement_and_cannot_repeat(scene):
    app, member = restore(scene)
    app.fsm.request('MovementPhase')
    positions = [(position[0] + 1, position[1]) for position in current_positions(member)]
    assert not commit_move(app, member, positions)
    assert not member.hasMovedThisTurn
    remaining_moves(app)
    assert commit_move(app, member, positions)
    assert member.hasMovedThisTurn
    assert member.moveSpentThisTurn == pytest.approx(1)
    assert not member.marchedThisTurn
    assert not commit_move(app, member, positions)


def test_per_model_march_boundary(scene):
    app, member = restore(scene)
    movement = app.movement.movementAllowance(member)
    positions = current_positions(member)
    regular = [(position[0] + movement, position[1]) for position in positions]
    march = [(position[0] + movement + 0.01, position[1]) for position in positions]
    assert not preview_move(app, member, regular).marched
    assert preview_move(app, member, march).marched


@pytest.mark.parametrize('kind', ['move', 'charge', 'friendly', 'restricted'])
def test_preview_action_matches_contact_without_spending_move(scene, kind):
    app, member = restore(scene)
    app.fsm.request('MovementPhase')
    member.bodyNP.setPos(0, -6, 0)
    target = next(unit for unit in app.units if unit.unitName ==
                  ('P1 Scouts' if kind == 'friendly' else 'Warriors'))
    target.bodyNP.setPos(0, 0, 0)
    target.isDeployed = True
    destination = (Point3(0, -5, 0) if kind == 'move' else
                   Point3(0, -(member.unitHeight + target.unitHeight) / 2 + 0.01, 0))
    app.arcPoint = Vec2((destination.x / 50 + 1) / 2, (destination.y / 50 + 1) / 2)
    if kind == 'restricted':
        member.deployedAsScouts = True
        app.roundCounter.currentRoundPlayer[0] = 0
    before = model_base_boxes(member)
    flags = (member.hasMovedThisTurn, member.moveSpentThisTurn, member.marchedThisTurn)
    preview = preview_action(app, member)
    assert model_base_boxes(member) == before
    assert (member.hasMovedThisTurn, member.moveSpentThisTurn, member.marchedThisTurn) == flags
    assert preview.charge_target is (target if kind in ('charge', 'restricted') else None)
    assert bool(preview.error) == (kind in ('friendly', 'restricted'))
    if kind == 'charge':
        assert not preview.marched


def test_right_click_move_is_cancellable_before_any_state_change(scene):
    app, member = restore(scene)
    app.fsm.request('MovementPhase')
    app.unitToMove = member
    before = model_base_boxes(member)
    app.movement._skirmishMovePreview(member, member.bodyNP.getPos() + Vec3(0, 1, 0))
    assert 'MOVE (not a charge)' in app.skirmishMoveStatus['text']
    app.startTaskFunction(app.taskLoopPathTowardsMouse, 'taskLoopPathTowardsMouse')
    app.onRightClick(member)
    assert app.skirmishEditor is not None
    assert app.skirmishEditor.preview.distance == pytest.approx(1, abs=1e-5)
    assert model_base_boxes(member) == before
    assert member.moveSpentThisTurn == 0 and not member.hasMovedThisTurn
    app.onRightClick(member)
    assert app.skirmishEditor is None
    assert model_base_boxes(member) == before
    assert member.moveSpentThisTurn == 0 and not member.hasMovedThisTurn


def test_right_click_move_confirm_spends_move_once(scene):
    app, member = restore(scene)
    app.fsm.request('MovementPhase')
    remaining_moves(app)
    app.unitToMove = member
    app.movement._skirmishMovePreview(member, member.bodyNP.getPos() + Vec3(0, 1, 0))
    app.startTaskFunction(app.taskLoopPathTowardsMouse, 'taskLoopPathTowardsMouse')
    app.onRightClick(member)
    assert app.skirmishEditor.confirm()
    assert member.hasMovedThisTurn
    assert member.moveSpentThisTurn == pytest.approx(1, abs=1e-5)
    assert app.skirmishMoveStatus.isHidden()


def test_real_charge_preview_click_and_decline_preserve_move(scene):
    app, member = restore(scene)
    app.fsm.request('MovementPhase')
    app.unitToMove = member
    member.bodyNP.setPos(0, -6, 0)
    enemy = next(unit for unit in app.units if unit.unitName == 'Warriors')
    enemy.bodyNP.setPos(0, 0, 0)
    before = model_base_boxes(member)
    app.movement._skirmishMovePreview(member, enemy.bodyNP.getPos())
    assert preview_action(app, member).charge_target is enemy
    assert app.skirmishMoveStatus['text'].startswith('CHARGE: Warriors')
    assert not member.wouldMarch and not member.marchedThisTurn
    assert model_base_boxes(member) == before
    app.startTaskFunction(app.taskLoopPathTowardsMouse, 'taskLoopPathTowardsMouse')
    with patch.object(app.taskMgr, 'add') as scheduled:
        app.onRightClick(member)
    declaration = next(call for call in scheduled.call_args_list
                       if call.args[0] == app.chargeAndChargeReaction)
    assert getattr(app, 'skirmishEditor', None) is None
    assert not member.hasMovedThisTurn and not member.marchedThisTurn
    with patch.object(app, 'makeChoiceNew', AsyncMock(return_value='No')) as choose, \
            patch.object(app.taskMgr, 'add', side_effect=lambda task, **kwargs: task), \
            patch.object(app, 'startTaskFunction'):
        asyncio.run(app.combat.chargeAndChargeReaction(
            *declaration.kwargs['extraArgs'], SimpleNamespace(done='done')))
    assert choose.call_args.args[0] == ['Yes', 'No']
    assert model_base_boxes(member) == before
    assert member.moveSpentThisTurn == 0 and not member.hasMovedThisTurn
    assert not member.isChargingMove


@pytest.mark.parametrize('movement,swiftstride,gap,legal', [
    (3, False, 8, True), (3, False, 9, True), (3, False, 9.02, False),
    (4, False, 10, True), (8, False, 14.02, False), (3, True, 12, True)])
def test_charge_plot_uses_m_plus_six_not_double_m(scene, movement, swiftstride, gap, legal):
    app, member = restore(scene)
    app.fsm.request('MovementPhase')
    enemy = next(unit for unit in app.units if unit.unitName == 'Warriors')
    for index, other in enumerate(app.units):
        other.bodyNP.setPos(25, 12 + index, 0)
    member.bodyNP.setPos(0, -14, 0)
    enemy.bodyNP.setPos(0, -14 + gap + (member.unitHeight + enemy.unitHeight) / 2, 0)
    app.world.doPhysics(1 / 60)
    with patch.object(member.unit.model, 'get_movement', return_value=movement), \
            patch.object(member.unit.model, 'is_swiftstride', return_value=swiftstride):
        app.movement._skirmishMovePreview(member, enemy.bodyNP.getPos())
        preview = preview_action(app, member)
        assert (preview.charge_target is enemy and preview.error is None) == legal
        assert preview.charge_maximum == movement + 6 + (3 if swiftstride else 0)
        if legal:
            assert not preview.marched
            assert f'Roll needed: {round(gap - movement)} ' in app.combat.chargeRangeText(member, movement)
        with patch.object(app.taskMgr, 'add') as scheduled, patch.object(app, 'startTaskFunction'):
            app.movement.moveUnit(member)
        declarations = [call for call in scheduled.call_args_list
                        if call.args[0] == app.chargeAndChargeReaction]
        assert bool(declarations) == legal
        assert not member.marchedThisTurn
    assert not member.hasMovedThisTurn


@pytest.mark.parametrize('movement,swiftstride', [(3, False), (8, False), (3, True)])
def test_ground_ranges_use_remaining_march_and_independent_charge(scene, movement, swiftstride):
    app, member = restore(scene)
    app.fsm.request('MovementPhase')
    member.moveSpentThisTurn = 1
    with patch.object(member.unit.model, 'get_movement', return_value=movement), \
            patch.object(member.unit.model, 'is_swiftstride', return_value=swiftstride):
        app.movement._skirmishMovePreview(member, member.bodyNP.getPos() + Vec3(0, 1, 0))
    limits = shader_vector(app.ground, 'skirmishRangeLimits')
    assert tuple(limits)[:3] == pytest.approx((movement - 1, movement * 2 - 1,
                                              movement + 6 + (3 if swiftstride else 0)))
    assert shader_vector(app.ground, 'skirmishRangeActive').x
    app.setGroundOverlay(False)
    assert not shader_vector(app.ground, 'skirmishRangeActive').x


@pytest.mark.parametrize('restriction', ['rallied', 'scouts', 'vanguard', 'moved', 'state'])
def test_ground_charge_range_respects_restrictions(scene, restriction):
    app, member = restore(scene)
    app.fsm.request('MovementPhase')
    member.cannotChargeThisTurn = restriction == 'rallied'
    member.deployedAsScouts = restriction == 'scouts'
    app.roundCounter.currentRoundPlayer[0] = 0
    member.hasMovedThisTurn = restriction == 'moved'
    if restriction == 'state':
        member.request('Moved')
    with patch('vanguard.vanguard_charge_blocked', return_value=restriction == 'vanguard'):
        app.movement._skirmishMovePreview(member, member.bodyNP.getPos() + Vec3(0, 1, 0))
    if restriction in ('moved', 'state'):
        assert not shader_vector(app.ground, 'skirmishRangeActive').x
        assert app.skirmMoveGhost is None
    else:
        assert shader_vector(app.ground, 'skirmishRangeLimits').z == 0


def test_skirmish_ground_ranges_wrap_terrain_and_clear_together(scene):
    app, member = restore(scene)
    app.fsm.request('MovementPhase')
    member.bodyNP.setPos(0, 0, 0)
    hill = app.terrain_manager.add_terrain('hill', Point3(0, 4, 0), 3, 3)
    try:
        app.movement._skirmishMovePreview(member, Point3(1, 0, 0))
        assert shader_vector(hill.visual, 'skirmishRangeActive').x
        assert shader_vector(hill.visual, 'skirmishRangeLimits') == (
            shader_vector(app.ground, 'skirmishRangeLimits'))
        app.graphicsEngine.renderFrame()
        app.graphicsEngine.renderFrame()
        app.setGroundOverlay(False)
        assert not shader_vector(hill.visual, 'skirmishRangeActive').x
        assert not shader_vector(app.ground, 'skirmishRangeActive').x
    finally:
        app.terrain_manager.remove_terrain(hill)


@pytest.mark.parametrize('boundary', ['board', 'impassable', 'spent'])
def test_limited_destination_obeys_shared_legality(scene, boundary):
    from skirmish_movement import limited_move_preview
    app, member = restore(scene)
    app.fsm.request('MovementPhase')
    for index, other in enumerate(app.units):
        other.bodyNP.setPos(-25, 10 + index, 0)
    member.bodyNP.setPos(29 if boundary == 'board' else 0, 0, 0)
    member.moveSpentThisTurn = 2 if boundary == 'spent' else 0
    feature = SimpleNamespace(center=Vec3(4, 0, 0), width=2, height=2,
                              movement_modifier=0, is_dangerous=False, is_impassable=True)
    pieces = [feature] if boundary == 'impassable' else []
    before = model_base_boxes(member)
    target = member.bodyNP.getPos() + Vec3(12, 0, 0)
    with patch.object(app.terrain_manager, 'terrain_pieces', pieces):
        preview = limited_move_preview(app, member, target)
        assert preview.error is None
        assert preview_move(app, member, destination=preview.destination).error is None
    assert preview.distance <= 6 - member.moveSpentThisTurn + 1e-4
    assert preview.distance > 0
    assert model_base_boxes(member) == before
    assert not member.hasMovedThisTurn


def test_extra_charge_reach_cannot_be_spent_as_ordinary_move(scene):
    app, member = restore(scene)
    app.fsm.request('MovementPhase')
    member.bodyNP.setPos(0, -16, 0)
    app.movement._skirmishMovePreview(member, Point3(0, -8, 0))
    preview = preview_action(app, member)
    assert preview.charge_target is None and preview.error is None
    assert preview.distance == pytest.approx(6, abs=1e-4)
    assert app.skirmMoveGhost is not None
    assert not commit_move(app, member, destination=(0, -8, 0))
    assert not member.hasMovedThisTurn


def test_contact_rank_survives_combat_entry_and_save_load(scene, tmp_path):
    from skirmish_charge import apply_fighting_rank, plan_skirmish_charge
    from psychology import obb_distance
    app, member = restore(scene)
    enemy = next(unit for unit in app.units if unit.unitName == 'P2 Scouts A')
    member.bodyNP.setPos(0, -6, 0)
    member.bodyNP.setH(83)
    enemy.bodyNP.setPos(0, 0, 0)
    enemy.bodyNP.setH(25)
    formation = plan_skirmish_charge(model_base_boxes(member), model_base_boxes(enemy), 9, 3)
    assert formation is not None
    for participant, rank in ((member, formation.attacker), (enemy, formation.defender)):
        original_ids = [record['id'] for record in participant.skirmishLayout]
        apply_fighting_rank(app, participant, rank)
        actual = model_base_boxes(participant)
        for box, expected in zip(actual, rank.positions):
            assert box[:2] == pytest.approx(expected, abs=1e-5)
        assert [record['id'] for record in participant.skirmishLayout] == [original_ids[index] for index in rank.order]
        participant.request('InCombat')
        assert model_base_boxes(participant) == actual
    member.isInCombatWith = [enemy]
    enemy.isInCombatWith = [member]
    member.isInCombatFlank = enemy.isInCombatFlank = ['front']
    before = (model_base_boxes(member), model_base_boxes(enemy))
    assert min(obb_distance(attacker, defender) for attacker in before[0] for defender in before[1]) < 1e-5
    saved = tmp_path / 'aligned-skirmishers.json'
    save_game_state(app, str(saved))
    load_game_state(app, str(saved))
    for participant, expected in zip((member, enemy), before):
        assert len(model_base_boxes(participant)) == len(expected)
        for actual, saved_box in zip(model_base_boxes(participant), expected):
            assert actual == pytest.approx(saved_box, abs=1e-5)


def test_form_up_losses_remove_only_unreachable_models(scene):
    from skirmish_charge import apply_fighting_rank, first_contact, plan_skirmish_charge
    app, member = restore(scene)
    enemy = next(unit for unit in app.units if unit.unitName == 'P2 Scouts A')
    member.bodyNP.setPos(0, -8, 0)
    member.bodyNP.setH(0)
    depth = model_base_boxes(member)[0][3] * 2
    member.restoreSkirmishLayout(dict(
        models=[dict(id=record['id'], x=0, y=-index * (depth + 0.6))
                for index, record in enumerate(member.skirmishLayout)], combat=False))
    enemy.bodyNP.setPos(0, 0, 0)
    attackers, defenders = model_base_boxes(member), model_base_boxes(enemy)
    distance = first_contact(attackers, defenders)[2]
    plan = plan_skirmish_charge(attackers, defenders, distance, 3)
    assert plan.attacker.lost == list(range(1, len(attackers)))
    survivor_id = member.skirmishLayout[0]['id']
    apply_fighting_rank(app, member, plan.attacker)
    assert member.unit.nmodels == member.model.getNumChildren() == 1
    assert member.skirmishLayout[0]['id'] == survivor_id
    assert model_base_boxes(member)[0][:2] == pytest.approx(plan.attacker.positions[0], abs=1e-5)


@pytest.mark.parametrize('dice,succeeds', [([6, 6], True), ([1, 1], False)])
def test_resolved_two_skirmisher_charge_obeys_dice_and_alignment(scene, dice, succeeds):
    from direct.interval.IntervalGlobal import LerpPosHprInterval, Parallel
    from psychology import obb_distance
    from skirmish_charge import plan_skirmish_charge
    app, member = restore(scene)
    app.fsm.request('MovementPhase')
    enemy = next(unit for unit in app.units if unit.unitName == 'P2 Scouts A')
    for index, other in enumerate(app.units):
        other.bodyNP.setPos(25, 10 + index, 0)
    member.bodyNP.setPos(0, -8, 0)
    member.bodyNP.setH(35)
    enemy.bodyNP.setPos(0, 0, 0)
    enemy.bodyNP.setH(10)
    enemy.isDeployed = True
    origin, facing = member.bodyNP.getPos(), member.bodyNP.getHpr()
    original_enemy = model_base_boxes(enemy)
    expected = plan_skirmish_charge(model_base_boxes(member), original_enemy, 3 + max(dice), 3)
    app.movement._skirmishMovePreview(member, enemy.bodyNP.getPos())
    member.bodyNP.setPos(*preview_action(app, member).destination)
    app.autoRoll = False
    def finish_interval(interval):
        interval.start()
        interval.finish()
        return iter(())

    with patch.object(LerpPosHprInterval, '__await__', finish_interval), \
            patch.object(Parallel, '__await__', finish_interval), \
            patch.object(app.combat, 'swiftstrideChargeChoice', AsyncMock(return_value=False)), \
            patch.object(app.combat, 'rullTerninger', AsyncMock(return_value=([], dice))), \
            patch.object(app.movement, 'dangerousTerrainTests'):
        asyncio.run(app.combat.chargeInterval(member, enemy.bodyNP, 0, origin, facing, 'flank'))
    if succeeds:
        assert member.state == enemy.state == 'InCombat'
        assert (member.bodyNP.getH() - enemy.bodyNP.getH()) % 360 == pytest.approx(180, abs=1e-5)
        assert member.unit.files == expected.attacker.files
        assert min(obb_distance(attacker, defender) for attacker in model_base_boxes(member)
                   for defender in model_base_boxes(enemy)) < 1e-5
        assert enemy.isInCombatFlank == ['front']
    else:
        assert expected is None
        assert member.state == 'Moved' and not member.skirmishCombat
        assert (member.bodyNP.getPos() - origin).length() == pytest.approx(max(dice), abs=1e-5)
        assert model_base_boxes(enemy) == original_enemy


@pytest.mark.parametrize('size', [(1280, 720), (800, 600)])
def test_charged_skirmishers_form_wider_rank_at_corners(scene, tmp_path, size):
    from direct.interval.IntervalGlobal import LerpPosHprInterval, Parallel
    from psychology import obb_distance
    app, member = restore(scene)
    app.fsm.request('MovementPhase')
    enemy = next(unit for unit in app.units if unit.unitName == 'P2 Scouts A')
    for index, other in enumerate(app.units):
        other.bodyNP.setPos(25, 10 + index, 0)
    app.movement.removeModelsFromUnit(member, member.model.getNumChildren() - 1)
    member.restoreSkirmishLayout(dict(
        models=[dict(id=member.skirmishLayout[0]['id'], x=0, y=0)], combat=False))
    width, depth = (dimension * 2 for dimension in model_base_boxes(enemy)[0][2:4])
    positions = [(0, 0), (-width - 0.2, 0.3), (width + 0.2, 0.3),
                 (-0.8, depth + 0.9), (0.8, depth + 0.9)]
    enemy.restoreSkirmishLayout(dict(
        models=[dict(id=record['id'], x=position[0], y=position[1])
                for record, position in zip(enemy.skirmishLayout, positions)], combat=False))
    member.bodyNP.setPos(0, -6, 0)
    member.bodyNP.setH(0)
    enemy.bodyNP.setPos(0, 0, 0)
    enemy.bodyNP.setH(0)
    enemy.isDeployed = True
    original_enemy = dict(zip((record['id'] for record in enemy.skirmishLayout),
                             model_base_boxes(enemy)))
    origin, facing = member.bodyNP.getPos(), member.bodyNP.getHpr()
    app.world.doPhysics(1 / 60)
    app.movement._skirmishMovePreview(member, enemy.bodyNP.getPos())
    app.autoRoll = False

    def finish_interval(interval):
        interval.start()
        interval.finish()
        return iter(())

    with patch.object(LerpPosHprInterval, '__await__', finish_interval), \
            patch.object(Parallel, '__await__', finish_interval), \
            patch.object(app.combat, 'swiftstrideChargeChoice', AsyncMock(return_value=False)), \
            patch.object(app.combat, 'rullTerninger', AsyncMock(return_value=([], [6, 6]))), \
            patch.object(app.movement, 'dangerousTerrainTests'), \
            patch('combat_resolution.rule_log') as log:
        asyncio.run(app.combat.chargeInterval(member, enemy.bodyNP, 0, origin, facing, 'front'))
    assert member.state == enemy.state == 'InCombat'
    assert member.unit.files == 1 and enemy.unit.files == 3
    assert enemy.unit.nmodels == len(original_enemy) == 5
    target = model_base_boxes(member)[0]
    defending = model_base_boxes(enemy)
    assert all(obb_distance(box, target) < 1e-5 for box in defending[:3])
    allowance = app.movement.movementAllowance(enemy)
    for record, box in zip(enemy.skirmishLayout, defending):
        previous = original_enemy[record['id']]
        assert Vec2(box[0] - previous[0], box[1] - previous[1]).length() <= allowance + 1e-5
    assert any('1 chargers face 3 defenders' in call.args[2] for call in log.call_args_list)

    camera_transform = app.camera.getTransform()
    old_size = (app.win.getXSize(), app.win.getYSize())
    window = app.openWindow(type='offscreen', size=size, makeCamera=False)
    for order, camera in enumerate((app.cam, app.cam2d, app.cam2dp)):
        region = window.makeDisplayRegion()
        region.setCamera(camera)
        region.setSort(order * 10)
    try:
        app.adjustWindowAspectRatio(size[0] / size[1])
        app.setGroundOverlay(False)
        app.camera.setPos(0, -12, 20)
        app.camera.lookAt(0, 0, 0)
        app.graphicsEngine.renderFrame()
        app.graphicsEngine.renderFrame()
        image = PNMImage()
        assert window.getScreenshot(image)
        for box in [target, *defending[:3]]:
            screen = Point2()
            assert app.camLens.project(app.cam.getRelativePoint(app.render, Point3(box[0], box[1], 0.5)), screen)
            assert abs(screen.x) < 0.9 and abs(screen.y) < 0.9
            horizontal = round((screen.x + 1) * size[0] / 2)
            vertical = round((1 - screen.y) * size[1] / 2)
            pixels = [image.getXel(column, row)
                      for column in range(horizontal - 6, horizontal + 7)
                      for row in range(vertical - 6, vertical + 7)]
            assert any(max(pixel) - min(pixel) > 0.2 and
                       (pixel.x > pixel.y * 1.5 or pixel.z > pixel.y * 1.5) for pixel in pixels)
        assert image.write(Filename.fromOsSpecific(str(tmp_path / f'corner-contact-{size[0]}x{size[1]}.png')))
    finally:
        app.camera.setTransform(camera_transform)
        app.closeWindow(window, keepCamera=True)
        app.adjustWindowAspectRatio(old_size[0] / old_size[1])


@pytest.mark.parametrize('heading', [0, 37, 180])
def test_skirmisher_column_forms_against_formed_rear(scene, heading):
    from direct.interval.IntervalGlobal import LerpPosHprInterval, Parallel
    from psychology import obb_distance
    app, member = restore(scene)
    app.fsm.request('MovementPhase')
    enemy = next(unit for unit in app.units if unit.unitName == 'Warriors')
    for index, other in enumerate(app.units):
        other.bodyNP.setPos(25, 10 + index, 0)
    enemy.bodyNP.setPos(0, 0, 0)
    enemy.bodyNP.setH(heading)
    member.bodyNP.setPos(app.render.getRelativePoint(enemy.bodyNP, Point3(1.5, -4, 0)))
    member.bodyNP.setH(heading)
    depth = model_base_boxes(member)[0][3] * 2
    member.restoreSkirmishLayout(dict(
        models=[dict(id=record['id'], x=0, y=-index * (depth + 0.6))
                for index, record in enumerate(member.skirmishLayout)], combat=False))
    origin, facing = member.bodyNP.getPos(), member.bodyNP.getHpr()
    original_attackers = model_base_boxes(member)
    original_ids = [record['id'] for record in member.skirmishLayout]
    original_enemy = model_base_boxes(enemy)
    app.world.doPhysics(1 / 60)
    app.movement._skirmishMovePreview(member, enemy.bodyNP.getPos())
    assert preview_action(app, member).charge_target is enemy
    member.bodyNP.setPos(*preview_action(app, member).destination)
    app.autoRoll = False

    def finish_interval(interval):
        interval.start()
        interval.finish()
        return iter(())

    with patch.object(LerpPosHprInterval, '__await__', finish_interval), \
            patch.object(Parallel, '__await__', finish_interval), \
            patch.object(app.combat, 'swiftstrideChargeChoice', AsyncMock(return_value=False)), \
            patch.object(app.combat, 'rullTerninger', AsyncMock(return_value=([], [6, 6]))), \
            patch.object(app.movement, 'dangerousTerrainTests'):
        asyncio.run(app.combat.chargeInterval(member, enemy.bodyNP, 0, origin, facing, 'rear'))
    assert member.state == enemy.state == 'InCombat'
    assert model_base_boxes(enemy) == original_enemy
    assert enemy.isInCombatFlank == ['rear']
    assert (member.bodyNP.getH() - enemy.bodyNP.getH() + 180) % 360 - 180 == pytest.approx(0, abs=1e-5)
    assert all(min(obb_distance(box, target) for target in original_enemy) < 1e-5
               for box in model_base_boxes(member)[:member.unit.files])
    assert sorted(record['id'] for record in member.skirmishLayout) == sorted(original_ids)
    for record, box in zip(member.skirmishLayout, model_base_boxes(member)):
        previous = original_attackers[original_ids.index(record['id'])]
        assert Vec2(box[0] - previous[0], box[1] - previous[1]).length() <= 9 + 1e-5


@pytest.mark.parametrize('heading', [0, 37, 180])
@pytest.mark.parametrize('dice,succeeds', [([6, 6], True), ([1, 1], False)])
def test_formed_charger_does_not_align_against_loose_defender(scene, heading, dice, succeeds):
    import math
    from direct.interval.IntervalGlobal import LerpPosHprInterval, Parallel
    from psychology import _box_corners, obb_distance
    app, defender = restore(scene)
    app.fsm.request('MovementPhase')
    attacker = next(unit for unit in app.units if unit.unitName == 'Warriors')
    for index, other in enumerate(app.units):
        other.bodyNP.setPos(25, 10 + index, 0)
    forward = Vec3(-math.sin(math.radians(heading)), math.cos(math.radians(heading)), 0)
    attacker.bodyNP.setPos(forward * -8)
    attacker.bodyNP.setH(heading)
    defender.bodyNP.setPos(0, 0, 0)
    defender.bodyNP.setH(heading + 37)
    origin, facing = attacker.bodyNP.getPos(), attacker.bodyNP.getHpr()
    targets = model_base_boxes(defender)
    identities = [record['id'] for record in defender.skirmishLayout]
    front = max(corner[0] * forward.x + corner[1] * forward.y
                for box in model_base_boxes(attacker) for corner in _box_corners(*box))
    contact = min(corner[0] * forward.x + corner[1] * forward.y
                  for box in targets for corner in _box_corners(*box)) - front
    expected = origin + forward * contact
    attacker.bodyNP.setPos(expected)
    app.playerNP.setPos(expected)
    app.moveArceDistance = contact
    app.autoRoll = False

    def finish_interval(interval):
        interval.start()
        interval.finish()
        return iter(())

    with patch.object(LerpPosHprInterval, '__await__', finish_interval), \
            patch.object(Parallel, '__await__', finish_interval), \
            patch.object(app.combat, 'swiftstrideChargeChoice', AsyncMock(return_value=False)), \
            patch.object(app.combat, 'rullTerninger', AsyncMock(return_value=([], dice))), \
            patch.object(app.combat, 'alignToEnemy', AsyncMock()) as align, \
            patch.object(app.movement, 'dangerousTerrainTests'), \
            patch('combat_resolution.rule_log') as log:
        asyncio.run(app.combat.chargeInterval(attacker, defender.bodyNP, 37, origin, facing, 'flank'))
    align.assert_not_awaited()
    if not succeeds:
        assert attacker.state == 'Moved' and not defender.skirmishCombat
        assert model_base_boxes(defender) == targets
        return
    assert attacker.state == defender.state == 'InCombat'
    assert (attacker.bodyNP.getH() - facing.x + 180) % 360 - 180 == pytest.approx(0, abs=1e-5)
    assert attacker.bodyNP.getP() == facing.y and attacker.bodyNP.getR() == facing.z
    assert attacker.bodyNP.getPos().almostEqual(expected, 1e-5)
    assert defender.isInCombatFlank == ['front']
    assert all(min(obb_distance(box, target) for target in model_base_boxes(attacker)) < 1e-5
               for box in model_base_boxes(defender)[:defender.unit.files])
    for record, box in zip(defender.skirmishLayout, model_base_boxes(defender)):
        previous = targets[identities.index(record['id'])]
        assert math.dist(previous[:2], box[:2]) <= app.movement.movementAllowance(defender) + 1e-5
    assert any('without an alignment wheel' in call.args[2] for call in log.call_args_list)


def test_enemy_clearance_clamps_preview_without_spending_move(scene):
    from psychology import obb_distance
    app, member = restore(scene)
    app.fsm.request('MovementPhase')
    app.unitToMove = member
    member.bodyNP.setPos(0, -6, 0)
    enemy = next(unit for unit in app.units if unit.unitName == 'Warriors')
    enemy.bodyNP.setPos(0, 0, 0)
    before = model_base_boxes(member)
    app.movement._skirmishMovePreview(member, Point3(0, -3, 0))
    preview = preview_action(app, member)
    assert preview.error is None and preview.distance < 3
    assert all(obb_distance(box, target) >= 1 - 1e-5
               for box in preview.boxes for target in model_base_boxes(enemy))
    app.startTaskFunction(app.taskLoopPathTowardsMouse, 'taskLoopPathTowardsMouse')
    app.onRightClick(member)
    assert app.skirmishEditor is not None
    assert model_base_boxes(member) == before
    assert member.moveSpentThisTurn == 0 and not member.hasMovedThisTurn
    app.skirmishEditor.cancel()


@pytest.mark.parametrize('action', ['phase', 'selection', 'no-ground'])
def test_plot_status_clears_when_no_longer_applicable(scene, action):
    app, member = restore(scene)
    app.fsm.request('MovementPhase')
    app.unitToMove = member
    app.movement._skirmishMovePreview(member, member.bodyNP.getPos() + Vec3(0, 1, 0))
    assert not app.skirmishMoveStatus.isHidden()
    if action == 'phase':
        app.fsm.request('ShootingPhase')
    elif action == 'selection':
        app.showSelectedUnit(next(unit for unit in app.units if unit.unitName == 'Warriors'))
    else:
        world = SimpleNamespace(rayTestClosest=lambda *args: SimpleNamespace(hasHit=lambda: False))
        with patch.object(app, 'world', world):
            app.movement.pathTowardsMouse(member, x=100, y=100)
        assert app.arcPoint is None
    assert app.skirmishMoveStatus.isHidden()
    assert app.skirmMoveGhost is None
    assert not shader_vector(app.ground, 'skirmishRangeActive').x


def test_editor_cancel_keeps_live_state_and_phase_locked(scene):
    from skirmish_ui import open_editor
    app, member = restore(scene)
    app.fsm.request('MovementPhase')
    app.unitToMove = member
    before = model_base_boxes(member)
    editor = open_editor(app)
    editor.positions = [(point[0] + 1, point[1]) for point in editor.positions]
    editor.redraw()
    app.fsm.nextPhase()
    assert app.fsm.state == 'MovementPhase'
    assert model_base_boxes(member) == before
    editor.cancel()
    assert model_base_boxes(member) == before
    assert not member.hasMovedThisTurn and member.moveSpentThisTurn == 0
    assert app.skirmishEditor is None


def test_editor_confirm_and_load_cleanup(scene, tmp_path):
    from skirmish_ui import open_editor
    app, member = restore(scene)
    app.fsm.request('MovementPhase')
    remaining_moves(app)
    app.unitToMove = member
    saved = tmp_path / 'before-edit.json'
    save_game_state(app, str(saved))
    editor = open_editor(app)
    editor.positions = [(point[0] + 1, point[1]) for point in editor.positions]
    editor.redraw()
    assert editor.confirm()
    assert member.hasMovedThisTurn and app.skirmishEditor is None
    load_game_state(app, str(saved))
    app.unitToMove = member
    editor = open_editor(app)
    load_game_state(app, str(saved))
    assert app.skirmishEditor is None
    assert not app.taskMgr.hasTaskNamed('skirmishEditor')
    assert not member.hasMovedThisTurn


@pytest.mark.parametrize('distance', [1, 3])
def test_ordinary_move_uses_shared_gate_and_preserves_layout(scene, distance):
    app, member = restore(scene)
    app.fsm.request('MovementPhase')
    remaining_moves(app)
    app.unitToMove = member
    before = member.savedSkirmishLayout()
    origin = member.bodyNP.getPos()
    target = origin + Vec3(0, distance, 0)
    app.arcPoint = Vec2((target.x / 50 + 1) / 2, (target.y / 50 + 1) / 2)
    app.arcPointRotation = 0
    app.moveArceDistance = distance
    assert app.movement.moveUnit(member)
    assert member.bodyNP.getY() == pytest.approx(origin.y + distance, abs=1e-5)
    assert member.savedSkirmishLayout() == before
    assert member.moveSpentThisTurn == pytest.approx(distance, abs=1e-5)
    assert not member.marchedThisTurn


def test_last_ordinary_casualty_leaves_character_alive(scene):
    app, member = restore(scene)
    character = app._create_unit(dict(name='Captain of the Empire', nmodels=1,
                                     files=1, ranks=1), 1, 'Last Survivor')
    assert join_unit(app, character, member)
    position = character.bodyNP.getPos(app.render)
    app.movement.removeModelsFromUnit(member, member.unit.nmodels)
    assert character in app.units and character in app.player1Units
    assert character.hostUnit is None
    assert character.bodyNP.getPos(app.render).almostEqual(position)
    assert character.bodyNP.node() in app.world.getRigidBodies()


def test_only_models_crossing_dangerous_feature_test(scene):
    app, member = restore(scene)
    app.fsm.request('MovementPhase')
    remaining_moves(app)
    first = model_base_boxes(member)[0]
    feature = SimpleNamespace(center=Vec3(first[0], first[1], 0), width=0.1, height=0.1,
                              movement_modifier=-1, is_dangerous=True, is_impassable=False)
    positions = [(point[0] + 0.1, point[1]) for point in current_positions(member)]
    with patch.object(app.terrain_manager, 'terrain_pieces', [feature]):
        preview = preview_move(app, member, positions)
        assert sum(len(features) for features in preview.terrain) == 1
        with patch('skirmish_movement.dangerous_terrain_wounds', return_value=0) as wounds, \
                patch.object(app.movement, 'magicalVortexTests'), \
                patch.object(app.movement, 'updateDisrupted'), \
                patch.object(app.movement, 'alignModelsToHillNormal'):
            assert commit_move(app, member, positions)
            assert wounds.call_args.args == (1, 1)


def test_lethal_character_terrain_hit_does_not_leave_dead_host_link(scene):
    app, member = restore(scene)
    app.fsm.request('MovementPhase')
    remaining_moves(app)
    character = app._create_unit(dict(name='Captain of the Empire', nmodels=1,
                                     files=1, ranks=1), 1, 'Terrain Character')
    assert join_unit(app, character, member)
    final = model_base_boxes(member)[-1]
    feature = SimpleNamespace(center=Vec3(final[0], final[1], 0), width=0.1, height=0.1,
                              movement_modifier=-1, is_dangerous=True, is_impassable=False)
    positions = [(point[0], point[1] + 0.1) for point in current_positions(member)]
    with patch.object(app.terrain_manager, 'terrain_pieces', [feature]), \
            patch('skirmish_movement.dangerous_terrain_wounds', side_effect=[0, 99]), \
            patch.object(app.movement, 'magicalVortexTests'), \
            patch.object(app.movement, 'updateDisrupted'), \
            patch.object(app.movement, 'alignModelsToHillNormal'):
        assert commit_move(app, member, positions)
    assert member.joinedCharacter is None
    assert character not in app.units
    assert coherency_error(model_base_boxes(member)) is None


def test_flying_tests_landing_terrain_not_overflown_terrain(scene):
    app, member = restore(scene)
    app.fsm.request('MovementPhase')
    first = model_base_boxes(member)[0]
    feature = SimpleNamespace(center=Vec3(first[0], first[1], 0), width=0.1, height=0.1,
                              movement_modifier=-1, is_dangerous=True, is_impassable=False)
    origin = member.bodyNP.getPos()
    with patch.object(member.unit.model, 'is_flying', return_value=True), \
            patch.object(member.unit.model, 'get_fly_movement', return_value=10), \
            patch.object(app.terrain_manager, 'terrain_pieces', [feature]):
        leaving = preview_move(app, member, destination=tuple(origin + Vec3(0, 8, 0)))
        assert leaving.error is None
        assert sum(len(features) for features in leaving.terrain) == 0
        landing = preview_move(app, member, destination=tuple(origin + Vec3(0, 0.1, 0)))
        assert sum(len(features) for features in landing.terrain) == 1


def test_editor_drag_changes_ghost_only(scene):
    from skirmish_ui import open_editor
    app, member = restore(scene)
    app.fsm.request('MovementPhase')
    app.unitToMove = member
    before = model_base_boxes(member)
    editor = open_editor(app)
    try:
        editor.mode[0] = 'models'
        first = editor.preview.boxes[0]
        with patch.object(editor, 'pointer', return_value=Point3(first[0], first[1], 0)):
            editor.press()
        assert editor.dragging == 0
        with patch.object(editor, 'pointer', return_value=Point3(first[0] + 0.1, first[1], 0)):
            editor.update(SimpleNamespace(cont='cont'))
        editor.release()
        assert model_base_boxes(member) == before
        assert editor.preview.distance == pytest.approx(0.1, abs=1e-5)
    finally:
        editor.cancel()


def test_editor_render_and_slider_preview(scene, tmp_path):
    from skirmish_ui import open_editor
    app, member = restore(scene)
    app.fsm.request('MovementPhase')
    app.unitToMove = member
    editor = open_editor(app)
    try:
        for button in editor.mode_buttons:
            left, right, bottom, top = button.getBounds()
            assert 0 <= button.getX() + left * button.getSx()
            assert button.getX() + right * button.getSx() <= 0.70
        editor.sliders['width']['value'] = 4
        editor.reshape()
        assert editor.preview.error is None
        assert len(editor.preview.boxes) == member.unit.nmodels
        app.graphicsEngine.renderFrame()
        app.graphicsEngine.renderFrame()
        image = PNMImage()
        assert app.win.getScreenshot(image)
        assert image.getXSize() >= 640 and image.getYSize() >= 480
        colours = {tuple(image.getXel(horizontal, vertical))
                   for horizontal in range(0, image.getXSize(), 30)
                   for vertical in range(0, image.getYSize(), 30)}
        assert len(colours) > 30
        assert app.screenshot(Filename.fromOsSpecific(str(tmp_path / 'skirmish.png')).getFullpath(),
                              defaultFilename=False)
    finally:
        editor.cancel()


@pytest.mark.parametrize('width,height', [(1280, 720), (800, 600)])
def test_ground_range_bands_render_and_stop_at_maximum(scene, tmp_path, width, height):
    app, member = restore(scene)
    app.fsm.request('MovementPhase')
    app.terrain_manager.clear()
    for index, other in enumerate(app.units):
        other.bodyNP.setPos(-25, 10 + index, 0)
    member.bodyNP.setPos(0, 0, 0)
    old_size = app.win.getXSize(), app.win.getYSize()
    camera_transform = app.camera.getTransform()
    window = app.openWindow(type='offscreen', size=(width, height), makeCamera=False)
    for order, camera in enumerate((app.cam, app.cam2d, app.cam2dp)):
        region = window.makeDisplayRegion()
        region.setCamera(camera)
        region.setSort(order * 10)
    app.adjustWindowAspectRatio(width / height)
    app.camera.setPos(0, -25, 60)
    app.camera.lookAt(0, 0, 0)
    try:
        app.movement._skirmishMovePreview(member, Point3(1, 0, 0))
        app.graphicsEngine.renderFrame()
        app.graphicsEngine.renderFrame()
        shaded = PNMImage()
        assert window.getScreenshot(shaded)
        assert shaded.write(Filename.fromOsSpecific(str(
            tmp_path / f'skirmish-ranges-{width}x{height}.png')))
        app.setGroundOverlay(False)
        app.graphicsEngine.renderFrame()
        plain = PNMImage()
        assert window.getScreenshot(plain)
        deltas = []
        for distance in (2.5, 4.5, 7.5, 9.5):
            screen = Point2()
            assert app.camLens.project(app.cam.getRelativePoint(app.render, Point3(0, distance, 0)), screen)
            column = round((screen.x + 1) * width / 2)
            row = round((1 - screen.y) * height / 2)
            deltas.append(shaded.getXel(column, row) - plain.getXel(column, row))
        normal, march, charge, outside = deltas
        assert normal.y > 0.08 and normal.z > 0.02
        assert march.x > 0.15 and march.x > march.z + 0.1
        assert charge.z > 0.08 and charge.z > charge.x + 0.08
        assert tuple(outside) == pytest.approx((0, 0, 0), abs=0.005)
    finally:
        app.setGroundOverlay(False)
        app.camera.setTransform(camera_transform)
        app.closeWindow(window, keepCamera=True)
        app.adjustWindowAspectRatio(old_size[0] / old_size[1])


def build_skirmish_scenario():
    app = build_scenario()
    app.AIplayer2.active = False
    locations = [(-8, -6), (4, -6), (-12, 12), (0, 12), (12, 12)]
    for member, position in zip(app.units, locations):
        member.bodyNP.setPos(position[0], position[1], 0)
        member.isDeployed = True
        member.deployedAsScouts = False
        member.request('Idle')
    app.fsm.request('MovementPhase')
    app.unitToMove = next(member for member in app.units if member.unitName == 'Normal Rangers')
    app.camera.setPos(0, -37, 46)
    app.camera.lookAt(0, 0, 0)
    app.refreshSelectedUnit()
    return app


@pytest.mark.parametrize('mode', ['editor', 'ordinary', 'charge'])
def test_preview_squares_visible_over_board_and_models(scene, mode):
    from psychology import _box_corners
    from skirmish_ui import open_editor
    app, member = restore(scene)
    app.fsm.request('MovementPhase')
    app.unitToMove = member
    member.bodyNP.setPos(-4, -6, 0)
    app.camera.setPos(0, -37, 46)
    app.camera.lookAt(0, 0, 0)
    editor = None
    if mode == 'editor':
        editor = open_editor(app)
        preview = editor.preview
    elif mode == 'ordinary':
        destination = member.bodyNP.getPos() + Vec3(0, 0.2, 0)
        app.movement._skirmishMovePreview(member, destination)
        preview = preview_move(app, member, destination=tuple(app.unitHitPos))
    else:
        enemy = next(unit for unit in app.units if unit.unitName == 'Warriors')
        enemy.bodyNP.setPos(-4, 0, 0)
        app.movement._skirmishMovePreview(member, enemy.bodyNP.getPos())
        preview = preview_action(app, member)
        assert preview.charge_target is enemy
        assert app.skirmishMoveStatus['text'].startswith('CHARGE: Warriors')
    try:
        assert preview.error is None and not preview.marched
        app.graphicsEngine.renderFrame()
        app.graphicsEngine.renderFrame()
        image = PNMImage()
        assert app.win.getScreenshot(image)
        for model_index, box in enumerate(preview.boxes):
            corners = _box_corners(*box)
            for edge_index, (start, end) in enumerate(zip(corners, corners[1:] + corners[:1])):
                midpoint = Point3((start[0] + end[0]) / 2, (start[1] + end[1]) / 2, 0.3)
                screen = Point2()
                assert app.camLens.project(app.cam.getRelativePoint(app.render, midpoint), screen)
                horizontal = round((screen.x + 1) * image.getXSize() / 2)
                vertical = round((1 - screen.y) * image.getYSize() / 2)
                pixels = [image.getXel(column, row)
                          for column in range(max(0, horizontal - 3), min(image.getXSize(), horizontal + 4))
                          for row in range(max(0, vertical - 3), min(image.getYSize(), vertical + 4))]
                if mode == 'charge':
                    assert any(pixel.z > 0.9 and pixel.y > 0.7 and pixel.x < 0.4
                               for pixel in pixels), (model_index, edge_index)
                else:
                    assert any(pixel.y > 0.8 and pixel.y > pixel.x * 1.5
                               and pixel.y > pixel.z for pixel in pixels), (model_index, edge_index)
    finally:
        if editor is not None:
            editor.cancel()
        elif app.skirmMoveGhost is not None:
            app.skirmMoveGhost.removeNode()
            app.skirmMoveGhost = None
        app.setGroundOverlay(False)


if __name__ == '__main__':
    from skirmish_ui import open_editor
    application = build_skirmish_scenario()
    try:
        root = Path(__file__).resolve().parents[1]
        save = root / 'saves' / 'skirmishers.json'
        screenshot = root / 'screenshots' / 'skirmishers.png'
        screenshot.parent.mkdir(exist_ok=True)
        save_game_state(application, str(save))
        load_game_state(application, str(save))
        application.unitToMove = next(member for member in application.units
                                      if member.unitName == 'Normal Rangers')
        editor = open_editor(application)
        editor.sliders['width']['value'] = 4
        editor.reshape()
        application.graphicsEngine.renderFrame()
        application.graphicsEngine.renderFrame()
        assert application.screenshot(Filename.fromOsSpecific(str(screenshot)).getFullpath(),
                                      defaultFilename=False)
        editor.close()
        print(f'Scene: {save}\nScreenshot: {screenshot}')
    finally:
        application.destroy()