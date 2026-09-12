"""Counter Charge through the actual High Elf and Chaos reaction flow (p. 167)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from panda3d.core import Vec2, Vec3

from battleFunctions import charge_initiative_bonus
from magic_items import current_turn
from persistence import load_game_state, save_game_state
from psychology import obb_distance
from tests.test_faction_rules_scene import members, scene as scene
from tests.test_shieldwall_scene import combat_tasks


def declared_charge(scene, distance=10, *, cavalry='Silver Helm'):
    app, baseline = scene
    load_game_state(app, baseline)
    app.terrain_manager.clear()
    armies = members(app)
    charger, defender = armies[cavalry], armies['Chaos Knight']
    for index, member in enumerate(app.units):
        member.bodyNP.setPos(-35 + index * 7, 20, 0)
    defender.bodyNP.setPos(0, 0, 0)
    defender.bodyNP.setH(180)
    half_depth = (charger.unitHeight + defender.unitHeight) / 2
    origin = Vec3(0, -half_depth - distance, 0)
    facing = Vec3(0, 0, 0)
    charger.bodyNP.setPos(0, -half_depth + .01, 0)
    charger.bodyNP.setHpr(facing)
    for member in (charger, defender):
        member.request('Idle')
        member.isInCombat = False
        member.isInCombatWith = []
        member.isInCombatFlank = []
        member.scoutDeploymentChoice = 'normal'
        member.chargeAttempts = 0
    app.roundCounter.currentRoundPlayer = [2, 2]
    app.roundCounter.current_player = 1
    app.unitToMove = charger
    app.playerNP.setPos(charger.bodyNP.getPos())
    app.moveArceDistance = distance
    app.autoCharge = app.autoHold = False
    app.chargeStage = None
    contact = SimpleNamespace(getNode1=lambda: defender.bodyNP.node())
    return app, charger, defender, origin, facing, contact


@pytest.mark.parametrize('rough,expected,kept', [(False, 18, 'max(6, 2) = 6'),
                                                (True, 14, 'min(6, 2) = 2')])
def test_charge_report_separates_swiftstride_bonus(scene, rough, expected, kept, capsys):
    app, charger, _, origin, _, _ = declared_charge(scene)
    with patch.object(app.combat, 'chargeThroughDifficult', return_value=rough):
        assert app.combat.chargeDistance(charger, origin, [6, 2, 4]) == expected
    output = capsys.readouterr().out
    assert f'{kept} + Swiftstride 4 -> {expected}" range' in output


def test_chaos_knights_are_offered_countercharge_from_declaration_position(scene):
    app, charger, defender, origin, facing, contact = declared_charge(scene)
    choice = AsyncMock(side_effect=['Yes', 'hold'])
    with combat_tasks(app) as run, \
            patch.object(app, 'aiControls', return_value=False), \
            patch.object(app, 'makeChoiceNew', choice), \
            patch.object(app.combat, 'chargeInterval', AsyncMock(return_value=None)):
        run(app.combat.chargeAndChargeReaction(
            charger, contact, origin, facing, SimpleNamespace(done='done')))
    reaction = choice.call_args_list[1]
    assert reaction.kwargs['owner'] is defender
    assert 'counter charge' in reaction.args[0]
    assert not getattr(defender, 'counterChargeTurn', None)
    assert defender.chargeAttempts == 0


@pytest.mark.parametrize('ai, gap, reaction_die, charge_dice, contact_expected', [
    (False, 10, 5, [6, 6], True),
    (True, 10, 3, [5, 6], True),
    (False, 17, 1, [1, 1], False),
])
def test_countercharge_moves_and_resolves_both_charge_attempts(
        scene, ai, gap, reaction_die, charge_dice, contact_expected, capsys):
    app, charger, defender, origin, facing, contact = declared_charge(scene, gap)
    before = Vec3(defender.bodyNP.getPos())
    choices = AsyncMock(side_effect=['Yes', 'counter charge'])
    dice = AsyncMock(side_effect=[([], [reaction_die]), ([], charge_dice)])
    terrain_check = app.movement.dangerousTerrainTests

    def check_terrain(member, start, end):
        assert member.isChargingMove
        return terrain_check(member, start, end)

    with combat_tasks(app) as run, \
            patch.object(app, 'aiControls', return_value=ai), \
            patch.object(app, 'makeChoiceNew', choices), \
            patch.object(app.combat, 'rullTerninger', dice), \
            patch.object(app.movement, 'dangerousTerrainTests', side_effect=check_terrain), \
            patch.object(app.combat, 'swiftstrideChargeChoice', AsyncMock(return_value=False)) as swift:
        try:
            run(app.combat.chargeAndChargeReaction(
                charger, contact, origin, facing, SimpleNamespace(done='done')))
        except AssertionError as error:
            pytest.fail(f'{error}\n{capsys.readouterr().out}')
    expected_move = (reaction_die + 1) // 2 + 1
    assert (defender.bodyNP.getPos() - before).length() == pytest.approx(expected_move, abs=.05)
    assert defender.counterChargeTurn == current_turn(app)
    assert defender.chargedThisTurn and defender.chargeDistance == pytest.approx(expected_move, abs=.05)
    assert [call.args for call in dice.call_args_list] == [(1,), (2, False)]
    swift.assert_awaited_once_with(charger)
    assert charger.chargeAttempts == defender.chargeAttempts == 1
    assert not charger.chargeAttemptPending and not defender.chargeAttemptPending
    assert not charger.isChargingMove and not defender.isChargingMove
    assert not charger.marchedThisTurn
    assert bool(charger.firstChargeDisruptedBy) is contact_expected
    assert bool(defender.firstChargeDisruptedBy) is contact_expected
    if contact_expected:
        assert charger.isInCombatWith == [defender] and defender.isInCombatWith == [charger]
        assert charger.chargedThisTurn and charger.wasChargedThisTurn
        assert defender.wasChargedThisTurn
        assert obb_distance(app.psychology._unit_box(charger), app.psychology._unit_box(defender)) < .06
        assert charger.chargeDistance == pytest.approx(gap - expected_move, abs=.1)
        assert charge_initiative_bonus(defender.chargeDistance) > 0
    else:
        assert charger.state == 'Moved' and not defender.isInCombat
        assert (charger.bodyNP.getPos() - origin).length() == pytest.approx(max(charge_dice), abs=.05)
    if ai:
        choices.assert_not_called()
    output = capsys.readouterr().out
    assert 'Swiftstride does not modify this reaction move' in output
    assert 'incoming charge rolls are separate' in output
    assert 'both receive charging benefits' in output if contact_expected else 'without contact' in output


@pytest.mark.parametrize('offset', [-3, 3])
def test_countercharge_pivots_and_contacts_on_an_angled_approach(scene, offset, tmp_path):
    app, charger, defender, origin, facing, contact = declared_charge(scene)
    origin.x = offset
    initial_center = Vec3(defender.bodyNP.getPos())
    with combat_tasks(app) as run, \
            patch.object(app.combat, 'rullTerninger', AsyncMock(side_effect=[([], [3]), ([], [6, 6])])), \
            patch.object(app.combat, 'swiftstrideChargeChoice', AsyncMock(return_value=False)):
        run(app.combat.counterChargeInterval(charger, defender, origin, facing))
    assert (defender.bodyNP.getPos() - initial_center).length() == pytest.approx(3, abs=.05)
    assert abs((defender.bodyNP.getH() - 180 + 180) % 360 - 180) > 1
    assert charger.state == defender.state == 'InCombat'
    assert obb_distance(app.psychology._unit_box(charger), app.psychology._unit_box(defender)) < .06
    from combat_contacts import CombatContactSnapshot
    snapshot = CombatContactSnapshot([charger, defender])
    for member, target in ((charger, defender), (defender, charger)):
        positions = snapshot.positions(member, target)[1]
        assert any(position.contact for position in positions), [position.distance for position in positions]
    assert (charger.bodyNP.getH() - defender.bodyNP.getH()) % 360 == pytest.approx(180, abs=.05)
    app.camera.setPos(0, -28, 28)
    app.camera.lookAt(0, -5, 0)
    app.graphicsEngine.renderFrame()
    app.graphicsEngine.renderFrame()
    assert app.screenshot(str(tmp_path / f'counter-charge-{offset}.png'), defaultFilename=False)


def test_countercharge_usage_survives_reload_and_expires_next_turn(scene, tmp_path, capsys):
    app, charger, defender, origin, facing, contact = declared_charge(scene)
    defender.counterChargeTurn = current_turn(app)
    path = save_game_state(app, str(tmp_path / 'counter-charge-used.json'))
    defender.counterChargeTurn = None
    load_game_state(app, path)
    app.chargeStage = None
    assert app.combat.counterChargeOption(defender, charger, origin, facing) is None
    assert 'already used Counter Charge' in capsys.readouterr().out
    with combat_tasks(app) as run, \
            patch.object(app, 'aiControls', return_value=False), \
            patch.object(app, 'makeChoiceNew', AsyncMock(return_value='Yes')) as choice, \
            patch.object(app.combat, 'chargeInterval', AsyncMock(return_value=None)):
        run(app.combat.chargeAndChargeReaction(
            charger, contact, origin, facing, SimpleNamespace(done='done')))
    choice.assert_awaited_once()
    app.fsm.exitCombatPhase()
    assert app.combat.counterChargeOption(defender, charger, origin, facing) is not None


def test_short_starting_distance_logs_why_countercharge_is_unavailable(scene, capsys):
    app, charger, defender, origin, facing, contact = declared_charge(scene, 7)
    assert app.combat.counterChargeOption(defender, charger, origin, facing) is None
    assert '7.00" away, less than its Movement 8"' in capsys.readouterr().out


def test_flying_chariot_uses_its_fly_movement_for_reaction_distance(scene, capsys):
    app, charger, defender, origin, facing, contact = declared_charge(scene)
    skycutter = members(app)['Lothern Skycutter']
    assert skycutter.unit.model.is_flying()
    movement = skycutter.unit.model.get_fly_movement(0)
    half_depth = (skycutter.unitHeight + defender.unitHeight) / 2
    origin = Vec3(0, -half_depth - movement + 1, 0)
    assert app.combat.counterChargeOption(defender, skycutter, origin, facing) is None
    assert f'less than its Movement {movement}"' in capsys.readouterr().out
    origin.y -= 1
    assert app.combat.counterChargeOption(defender, skycutter, origin, facing) is not None


def test_enemy_contact_is_not_marked_or_logged_as_a_march(scene, capsys):
    app, charger, defender, origin, facing, contact = declared_charge(scene)
    charger.bodyNP.setPos(origin)
    charger.bodyNP.setHpr(facing)
    charger.wouldMarch = True
    charger.marchedThisTurn = charger.hasMovedThisTurn = False
    app.arcPoint = Vec2(.5, .45)
    app.arcPointRotation = 0
    with patch.object(app, 'checkUnitContactSmall', return_value=contact), \
            patch.object(app.taskMgr, 'add') as scheduled:
        app.movement.moveUnit(charger)
    assert scheduled.called and charger.isChargingMove
    assert not charger.marchedThisTurn
    assert '[Rule] Marching' not in capsys.readouterr().out


def test_ai_still_stands_and_shoots_without_countercharge(scene):
    app, charger, defender, origin, facing, contact = declared_charge(scene)
    option = SimpleNamespace(weapon={'name': 'Test bow'}, distance=10)
    with combat_tasks(app) as run, \
            patch.object(app, 'aiControls', return_value=True), \
            patch.object(app.combat, 'counterChargeOption', return_value=None), \
            patch.object(app.combat, 'standAndShootOption', return_value=option), \
            patch.object(app.combat, 'standAndShoot', AsyncMock(return_value=None)) as volley, \
            patch.object(app.combat, 'chargeInterval', AsyncMock(return_value=None)):
        run(app.combat.chargeAndChargeReaction(
            charger, contact, origin, facing, SimpleNamespace(done='done')))
    volley.assert_awaited_once_with(defender, charger, option.weapon, 10, target_boxes=None)
    assert not defender.counterChargeTurn


def test_column_after_drilled_holds_without_cancelling_incoming_charge(scene):
    app, charger, defender, origin, facing, contact = declared_charge(scene)
    option = SimpleNamespace(distance=10, movement=8)
    before = Vec3(defender.bodyNP.getPos())
    with combat_tasks(app) as run, \
            patch.object(app, 'aiControls', return_value=True), \
            patch.object(app.combat, 'counterChargeOption', return_value=option), \
            patch('drilled.before_move', AsyncMock(return_value=None)), \
            patch('drilled.marching_column', return_value=True), \
            patch.object(app.combat, 'chargeInterval', AsyncMock(return_value=None)) as charge:
        run(app.combat.chargeAndChargeReaction(
            charger, contact, origin, facing, SimpleNamespace(done='done')))
    charge.assert_awaited_once()
    assert defender.bodyNP.getPos() == before
    assert not defender.counterChargeTurn
    assert defender.chargeAttempts == 0