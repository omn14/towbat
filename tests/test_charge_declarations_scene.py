"""Declarations use real roster units and the live charge handler (pp. 119-121)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from charge_declarations import begin_declarations, queue_charge, resolve_declarations
from first_charge import begin_charge_attempt
from panda3d.core import Vec3
from tests.test_counter_charge_scene import declared_charge
from tests.test_faction_rules_scene import members, scene as scene
from tests.test_shieldwall_scene import combat_tasks
from persistence import load_game_state, save_game_state


def test_declaration_waits_for_reactions_and_movement(scene):
    app, charger, defender, origin, facing, contact = declared_charge(scene)
    app.fsm.request('MovementPhase')
    begin_declarations(app)
    choice = AsyncMock(side_effect=['Yes', 'hold'])
    move = AsyncMock(return_value=None)
    with combat_tasks(app) as run, \
            patch.object(app, 'aiControls', return_value=False), \
            patch.object(app, 'makeChoiceNew', choice), \
            patch.object(app.combat, 'chargeInterval', move):
        run(app.combat.chargeAndChargeReaction(charger, contact, origin, facing, SimpleNamespace(done=None)))
        assert charger.bodyNP.getPos().almostEqual(origin)
        assert len(app.chargeDeclarations) == 1
        assert charger.chargeAttempts == 1
        assert charger.chargeAttemptPending
        assert choice.await_count == 1
        move.assert_not_awaited()
        run(resolve_declarations(app))
    assert choice.await_count == 2
    move.assert_awaited_once()
    assert app.chargeDeclarations == []
    assert app.chargeStage == 'remaining'
    assert not charger.chargeAttemptPending


def test_countercharge_reaction_moves_before_the_incoming_charge_roll(scene):
    app, charger, defender, origin, facing, contact = declared_charge(scene)
    app.fsm.request('MovementPhase')
    before = defender.bodyNP.getPos()

    async def roll(count, bonus=False):
        if count == 1:
            assert charger.bodyNP.getPos().almostEqual(origin)
            return [], [3]
        assert (defender.bodyNP.getPos() - before).length() == pytest.approx(3, abs=.05)
        return [], [6, 6]

    with combat_tasks(app) as run, \
            patch.object(app, 'aiControls', return_value=True), \
            patch.object(app.combat, 'rullTerninger', AsyncMock(side_effect=roll)), \
            patch.object(app.combat, 'swiftstrideChargeChoice', AsyncMock(return_value=False)):
        run(app.combat.chargeAndChargeReaction(charger, contact, origin, facing, SimpleNamespace(done=None)))
        run(resolve_declarations(app))
    assert charger.state == defender.state == 'InCombat'
    assert charger.isInCombatWith == [defender]
    assert defender.isInCombatWith == [charger]
    assert charger.wasChargedThisTurn and defender.chargedThisTurn
    assert charger.firstChargeDisruptedBy and defender.firstChargeDisruptedBy
    assert charger.chargeAttempts == defender.chargeAttempts == 1

    import faulthandler
    faulthandler.dump_traceback_later(10, exit=True)
    try:
        for frame in range(3):
            app.eventMgr.doEvents()
            app.graphicsEngine.renderFrame()
    finally:
        faulthandler.cancel_dump_traceback_later()
    assert app.chargeStage == 'remaining'


def test_declarations_survive_reload_without_redeclaring_or_spending_another_attempt(scene, tmp_path):
    app, charger, defender, origin, facing, contact = declared_charge(scene)
    app.fsm.request('MovementPhase')
    with combat_tasks(app) as run, patch.object(app, 'aiControls', return_value=True):
        run(app.combat.chargeAndChargeReaction(charger, contact, origin, facing, SimpleNamespace(done=None)))
    filename = save_game_state(app, str(tmp_path / 'declared.json'))
    app.chargeDeclarations = []
    app.chargeStage = 'remaining'
    charger.bodyNP.setX(30)
    load_game_state(app, filename)
    assert app.chargeStage == 'declarations'
    assert len(app.chargeDeclarations) == 1
    pending = app.chargeDeclarations[0]
    assert pending.charger is charger and pending.defender is defender
    assert charger.bodyNP.getPos().almostEqual(origin)
    assert charger.chargeAttempts == 1 and charger.chargeAttemptPending
    with combat_tasks(app) as run, \
            patch.object(app, 'aiControls', return_value=False), \
            patch.object(app, 'makeChoiceNew', AsyncMock(return_value='hold')) as choice, \
            patch.object(app.combat, 'chargeInterval', AsyncMock(return_value=None)):
        run(resolve_declarations(app))
    assert choice.await_count == 1
    assert charger.chargeAttempts == 1 and not charger.chargeAttemptPending


def test_countercharge_target_does_not_force_charge_move_order(scene):
    app, charger, defender, origin, facing, contact = declared_charge(scene)
    second = members(app)['Dragon Prince']
    app.fsm.request('MovementPhase')
    before = Vec3(defender.bodyNP.getPos())
    moved = []

    async def choose(options, position, **kwargs):
        if options == ['Yes', 'No']:
            return 'Yes'
        if kwargs.get('owner') is defender:
            targets = [option for option in options if option.startswith('counter charge ')]
            assert len(targets) == 2
            assert charger.bodyNP.getPos().almostEqual(origin)
            return targets[1]
        return options[0]

    async def move(entry):
        assert (defender.bodyNP.getPos() - before).length() == pytest.approx(3, abs=.05)
        moved.append(entry.charger)

    with combat_tasks(app) as run, \
            patch.object(app, 'aiControls', return_value=False), \
            patch.object(app, 'rollLeadershipDice', AsyncMock(return_value=[1, 1])), \
            patch.object(app, 'makeChoiceNew', AsyncMock(side_effect=choose)), \
            patch.object(app.combat, 'rullTerninger', AsyncMock(return_value=([], [3]))) as dice, \
            patch.object(app.combat, 'resolveDeclaredCharge', AsyncMock(side_effect=move)):
        run(app.combat.chargeAndChargeReaction(charger, contact, origin, facing, SimpleNamespace(done=None)))
        second_origin = Vec3(5, origin.y, 0)
        second.bodyNP.setPos(second_origin)
        second.bodyNP.setHpr(facing)
        second.request('Idle')
        begin_charge_attempt(second)
        queue_charge(app, second, defender, second_origin, facing)
        run(resolve_declarations(app))
    assert moved == [charger, second]
    dice.assert_awaited_once_with(1)
    assert defender.chargeAttempts == 1
    assert not defender.chargeAttemptPending
    assert app.chargeStage == 'remaining'


def test_multiple_charges_recompute_contact_after_countercharge(scene):
    app, charger, defender, origin, facing, contact = declared_charge(scene)
    second = members(app)['Dragon Prince']
    app.fsm.request('MovementPhase')
    with combat_tasks(app) as run, \
            patch.object(app, 'aiControls', return_value=True), \
            patch.object(app, 'rollLeadershipDice', AsyncMock(return_value=[1, 1])), \
            patch.object(app.combat, 'rullTerninger', AsyncMock(
                side_effect=lambda count, bonus=False: ([], [3] if count == 1 else [6, 6]))), \
            patch.object(app.combat, 'swiftstrideChargeChoice', AsyncMock(return_value=False)):
        run(app.combat.chargeAndChargeReaction(charger, contact, origin, facing, SimpleNamespace(done=None)))
        second_origin = Vec3(12, -3, 0)
        second_facing = Vec3(90, 0, 0)
        second.bodyNP.setPos(second_origin)
        second.bodyNP.setHpr(second_facing)
        second.request('Idle')
        begin_charge_attempt(second)
        queue_charge(app, second, defender, second_origin, second_facing)
        run(resolve_declarations(app))
    assert charger.state == second.state == defender.state == 'InCombat'
    assert set(defender.isInCombatWith) == {charger, second}
    assert charger.isInCombatWith == second.isInCombatWith == [defender]
    assert set(defender.isInCombatFlank) == {'front', 'flank'}
    from psychology import obb_distance
    from combat_contacts import CombatContactSnapshot
    from combat_profiles import combat_profiles
    snapshot = CombatContactSnapshot([charger, second, defender])
    for member in (charger, second):
        assert obb_distance(app.psychology._unit_box(member), app.psychology._unit_box(defender)) < .06
        assert member.chargeAttempts == 1 and not member.chargeAttemptPending
        positions = snapshot.positions(member, defender)[1]
        assert any(position.contact for position in positions), [position.distance for position in positions]
        assert all(snapshot.attacks(part, member.unit.nmodels) > 0
                   for part in combat_profiles(member, defender))
    assert defender.chargeAttempts == 1 and not defender.chargeAttemptPending


def test_queued_failed_charge_moves_roll_only(scene):
    app, charger, defender, origin, facing, contact = declared_charge(scene, distance=17)
    app.fsm.request('MovementPhase')
    with combat_tasks(app) as run, \
            patch.object(app, 'aiControls', return_value=False), \
            patch.object(app, 'makeChoiceNew', AsyncMock(side_effect=['Yes', 'hold'])), \
            patch.object(app.combat, 'rullTerninger', AsyncMock(return_value=([], [1, 1]))), \
            patch.object(app.combat, 'swiftstrideChargeChoice', AsyncMock(return_value=False)):
        run(app.combat.chargeAndChargeReaction(charger, contact, origin, facing, SimpleNamespace(done=None)))
        run(resolve_declarations(app))
    assert (charger.bodyNP.getPos() - origin).length() == pytest.approx(1, abs=.01)
    assert charger.state == 'Moved' and not defender.isInCombat
    assert charger.chargeAttempts == 1 and not charger.chargeAttemptPending


def test_second_charge_uses_horsemen_combat_formation(scene):
    from scouts import model_base_boxes
    from psychology import obb_distance
    app, charger, knights, origin, facing, _ = declared_charge(scene)
    knights.bodyNP.setPos(25, 20, 0)
    defender = members(app)['Marauder Horsemen']
    defender.bodyNP.setPos(0, 0, 0)
    defender.bodyNP.setH(180)
    second = members(app)['Dragon Prince']
    second_origin, second_facing = Vec3(12, -3, 0), Vec3(90, 0, 0)
    second.bodyNP.setPos(second_origin)
    second.bodyNP.setHpr(second_facing)
    charger.bodyNP.setPos(origin)
    charger.bodyNP.setHpr(facing)
    begin_declarations(app)
    for member, start, heading in ((charger, origin, facing), (second, second_origin, second_facing)):
        begin_charge_attempt(member)
        entry = queue_charge(app, member, defender, start, heading)
        boxes = model_base_boxes(defender)
        entry.target_index = min(range(len(boxes)), key=lambda index:
                                 (boxes[index][0] - start.x) ** 2 + (boxes[index][1] - start.y) ** 2)
        entry.reaction = 'hold'
    with combat_tasks(app) as run, \
            patch.object(app, 'aiControls', return_value=True), \
            patch.object(app.combat, 'swiftstrideChargeChoice', AsyncMock(return_value=False)), \
            patch.object(app.combat, 'rullTerninger', AsyncMock(return_value=([], [6, 6]))):
        for entry in app.chargeDeclarations:
            run(app.combat.resolveDeclaredCharge(entry))
    assert defender.skirmishCombat
    assert set(defender.isInCombatWith) == {charger, second}
    for member in (charger, second):
        assert member.isInCombatWith == [defender]
        assert min(obb_distance(source, target) for source in model_base_boxes(member)
                   for target in model_base_boxes(defender)) < .06
        assert member.chargeAttempts == 1 and not member.chargeAttemptPending


def test_queued_charge_rebuilds_route_to_fleeing_horsemen(scene):
    from scouts import model_base_boxes
    app, charger, knights, origin, facing, _ = declared_charge(scene)
    knights.bodyNP.setPos(25, 20, 0)
    defender = members(app)['Marauder Horsemen']
    defender.bodyNP.setPos(0, -3, 0)
    defender.bodyNP.setH(0)
    defender.request('IsFleeing')
    defender.fledThisPhase = True
    charger.bodyNP.setPos(origin)
    charger.bodyNP.setHpr(facing)
    begin_declarations(app)
    begin_charge_attempt(charger)
    entry = queue_charge(app, charger, defender, origin, facing)
    targets = model_base_boxes(defender)
    entry.target_index = min(range(len(targets)), key=lambda index:
                             (targets[index][0] - origin.x) ** 2 + (targets[index][1] - origin.y) ** 2)
    entry.reaction = 'hold'
    with combat_tasks(app) as run, \
            patch.object(app.combat, 'chargeAndChargeReaction', AsyncMock()) as resolve:
        run(app.combat.resolveDeclaredCharge(entry))
    resolve.assert_awaited_once()
    assert entry.preview is not None and entry.preview.target is defender
    assert entry.preview.route.distance > 0
    assert not charger.chargeAttemptPending


@pytest.mark.parametrize('dice,caught', [([6, 6], True), ([1, 1], False)])
def test_horsemen_charger_runs_down_fleeing_elf_once(scene, dice, caught):
    app, baseline = scene
    load_game_state(app, baseline)
    charger, defender = members(app)['Marauder Horsemen'], members(app)['Elven Archer']
    for index, member in enumerate(app.units):
        member.bodyNP.setPos(-30 + index * 6, 20, 0)
    origin, facing = Vec3(0, -12, 0), Vec3(0, 0, 0)
    charger.bodyNP.setPos(origin)
    charger.bodyNP.setHpr(facing)
    defender.bodyNP.setPos(0, 2, 0)
    defender.bodyNP.setH(0)
    defender.request('IsFleeing')
    defender.fledThisPhase = True
    defender_origin = Vec3(defender.bodyNP.getPos())
    app.roundCounter.currentRoundPlayer = [2, 2]
    app.roundCounter.current_player = 2
    app.playerNP.setPos(0, -2, 0)
    app.moveArceDistance = 10
    begin_declarations(app)
    begin_charge_attempt(charger)
    entry = queue_charge(app, charger, defender, origin, facing)
    entry.reaction = 'hold'
    with combat_tasks(app) as run, \
            patch.object(app, 'aiControls', return_value=True), \
            patch.object(app.combat, 'swiftstrideChargeChoice', AsyncMock(return_value=False)), \
            patch.object(app.combat, 'rullTerninger', AsyncMock(return_value=([], dice))) as rolls:
        run(app.combat.resolveDeclaredCharge(entry))
    assert (defender not in app.units) is caught
    assert charger.state == 'Moved' and not charger.isInCombat
    assert not charger.skirmishCombat and not charger.chargeAttemptPending
    assert rolls.await_count == (1 if caught else 2)
    if not caught:
        assert defender.bodyNP.getPos() == defender_origin
        assert (charger.bodyNP.getPos() - origin).length() == pytest.approx(charger.unit.model.get_movement() + 1)


def test_dragon_princes_countercharge_incoming_horsemen(scene):
    app, baseline = scene
    load_game_state(app, baseline)
    charger, defender = members(app)['Marauder Horsemen'], members(app)['Dragon Prince']
    for index, member in enumerate(app.units):
        member.bodyNP.setPos(-30 + index * 6, 20, 0)
    origin, facing = Vec3(0, -15, 0), Vec3(0, 0, 0)
    charger.bodyNP.setPos(origin)
    charger.bodyNP.setHpr(facing)
    defender.bodyNP.setPos(0, 0, 0)
    defender.bodyNP.setH(180)
    app.roundCounter.currentRoundPlayer = [2, 2]
    app.roundCounter.current_player = 2
    app.playerNP.setPos(0, -2, 0)
    app.moveArceDistance = 13
    begin_declarations(app)
    begin_charge_attempt(charger)
    queue_charge(app, charger, defender, origin, facing)
    assert app.combat.counterChargeOption(defender, charger, origin, facing) is not None
    with combat_tasks(app) as run, \
            patch.object(app, 'aiControls', return_value=True), \
            patch.object(app.combat, 'swiftstrideChargeChoice', AsyncMock(return_value=False)), \
            patch.object(app.combat, 'rullTerninger', AsyncMock(side_effect=[([], [3]), ([], [6, 6])])):
        run(resolve_declarations(app))
    assert charger.state == defender.state == 'InCombat'
    assert charger.isInCombatWith == [defender] and defender.isInCombatWith == [charger]
    for member in (charger, defender):
        assert member.chargedThisTurn and member.wasChargedThisTurn
        assert member.chargeAttempts == 1 and not member.chargeAttemptPending


@pytest.mark.parametrize('dice,caught', [([6, 6], True), ([1, 1], False)])
@pytest.mark.parametrize('reform_dice', [None, [1, 1], [6, 6]])
def test_live_horsemen_chase_catches_or_moves_full_range(scene, dice, caught, reform_dice):
    from scouts import model_base_boxes
    app, charger, knights, origin, facing, _ = declared_charge(scene)
    knights.bodyNP.setPos(25, 20, 0)
    defender = members(app)['Marauder Horsemen']
    defender.bodyNP.setPos(0, 3, 0)
    defender.bodyNP.setH(0)
    defender.request('IsFleeing')
    defender.fledThisPhase = True
    charger.bodyNP.setPos(origin)
    charger.bodyNP.setHpr(facing)
    begin_declarations(app)
    begin_charge_attempt(charger)
    entry = queue_charge(app, charger, defender, origin, facing)
    targets = model_base_boxes(defender)
    entry.target_index = min(range(len(targets)), key=lambda index:
                             (targets[index][0] - origin.x) ** 2 + (targets[index][1] - origin.y) ** 2)
    entry.reaction = 'hold'
    with combat_tasks(app) as run, \
            patch.object(app, 'aiControls', return_value=reform_dice is None), \
            patch.object(app, 'makeChoiceNew', AsyncMock(return_value='Reform')), \
            patch.object(app, 'rollLeadershipDice', AsyncMock(return_value=reform_dice)) as leadership, \
            patch.object(app.combat, 'swiftstrideChargeChoice', AsyncMock(return_value=False)), \
            patch.object(app.combat, 'rullTerninger', AsyncMock(return_value=([], dice))), \
            patch.object(app.combat, 'freeReform', AsyncMock()) as reform, \
            patch.object(app.combat, '_formChargedSkirmishers', AsyncMock()) as form:
        run(app.combat.resolveDeclaredCharge(entry))
    assert (defender not in app.units) is caught
    assert charger.state == 'Moved' and not charger.isInCombat
    assert not charger.chargeAttemptPending
    assert reform.await_count == int(caught and reform_dice == [1, 1])
    assert leadership.await_count == int(caught and reform_dice is not None)
    form.assert_not_awaited()
    if not caught:
        position, heading = entry.preview.route.pose(charger.unit.model.get_movement(4) + 1)
        assert charger.bodyNP.getPos().almostEqual(Vec3(*position), .01)


@pytest.mark.parametrize('dice,redirected', [([1, 1], True), ([6, 6], False)])
@pytest.mark.parametrize('reaction', ['hold', 'flee'])
def test_redirect_leadership_and_hold_flee_only(scene, dice, redirected, reaction):
    from charge_declarations import redirect_charge, redirect_targets
    app, charger, defender, origin, facing, _ = declared_charge(scene)
    charger.bodyNP.setPos(origin)
    defender.request('IsFleeing')
    alternative = members(app)['Chaos Warrior']
    alternative.bodyNP.setPos(-7, -3, 0)
    alternative.bodyNP.setH(180)
    begin_declarations(app)
    begin_charge_attempt(charger)
    entry = queue_charge(app, charger, defender, origin, facing)
    assert alternative in redirect_targets(app, entry)
    choices = AsyncMock(side_effect=[f'Redirect: {alternative.unitName}', reaction])
    with combat_tasks(app) as run, \
            patch.object(app, 'aiControls', return_value=False), \
            patch.object(app, 'makeChoiceNew', choices), \
            patch('charge_declarations.flee_reaction', AsyncMock()) as flee, \
            patch.object(app, 'rollLeadershipDice', AsyncMock(return_value=dice)):
        run(redirect_charge(app, entry, [alternative]))
    assert flee.await_count == int(redirected and reaction == 'flee')
    assert entry.redirected is redirected
    assert entry.defender is (alternative if redirected else defender)
    assert charger.chargeAttempts == 1
    assert choices.await_count == (2 if redirected else 1)
    if redirected:
        assert choices.call_args.args[0] == ['hold', 'flee']
        assert choices.call_args.kwargs['owner'] is alternative
        alternative.request('IsFleeing')
        with combat_tasks(app) as run, patch.object(app, 'makeChoiceNew', choices):
            run(redirect_charge(app, entry, [defender]))
        assert choices.await_count == 2


def test_flee_reaction_uses_strongest_charger_once_before_charge_moves(scene, capsys):
    from charge_declarations import flee_reaction
    app, charger, defender, origin, facing, contact = declared_charge(scene)
    second = members(app)['Dragon Prince']
    app.fsm.request('MovementPhase')
    charger.bodyNP.setPos(origin)
    second.bodyNP.setPos(12, 0, 0)
    queue_charge(app, charger, defender, origin, facing)
    queue_charge(app, second, defender, second.bodyNP.getPos(), facing)
    moves = []

    def flee(member, direction, distance, cause, callback):
        moves.append((member, Vec3(direction), distance))
        callback()

    with combat_tasks(app) as run, \
            patch('psychology.unit_strength_total', side_effect=lambda member: 20 if member is second else 10), \
            patch.object(app.psychology, '_start_flee_move', side_effect=flee), \
            patch.object(app.combat, 'swiftstrideChoice', AsyncMock(return_value=False)), \
            patch.object(app.combat, 'rullTerninger', AsyncMock(return_value=([], [2, 2]))) as dice:
        try:
            run(flee_reaction(app, defender, app.chargeDeclarations))
            run(flee_reaction(app, defender, app.chargeDeclarations))
        except AssertionError as error:
            pytest.fail(f'{error}; moves={moves}; dice={dice.await_count}; {capsys.readouterr()}')
    assert len(moves) == 1 and moves[0][0] is defender
    assert moves[0][1].almostEqual(Vec3(-1, 0, 0)) and moves[0][2] == 4
    assert defender.state == 'IsFleeing' and defender.fledThisPhase
    dice.assert_awaited_once_with(2, False)


def test_ai_declares_resolves_then_moves_unreserved_units(scene):
    from ClassAI import ClassAI
    app, charger, defender, origin, facing, contact = declared_charge(scene)
    app.fsm.request('MovementPhase')
    second = members(app)['Dragon Prince']
    second.request('Idle')
    ai = ClassAI(app, [charger, second], [defender])
    visits = []

    def move(member):
        visits.append((app.chargeStage, member))
        if member is charger or app.chargeStage == 'remaining':
            member.hasMovedThisTurn = True
        ai.endLoopWaitForMoveComplete()

    with combat_tasks(app) as run, \
            patch('ClassAI.taskMgr', app.taskMgr, create=True), \
            patch.object(ai, 'moveTowardsClosestEnemy', side_effect=move):
        run(ai.takeMoveTurn())
    ai.helper1.ignoreAll()
    assert visits == [('declarations', charger), ('declarations', second), ('remaining', second)]
    assert app.chargeStage == 'remaining'


@pytest.mark.parametrize('stage', ['resolving', 'blocked'])
def test_saves_during_resolution_do_not_replace_existing_snapshot(scene, tmp_path, stage):
    app, charger, defender, origin, facing, contact = declared_charge(scene)
    path = tmp_path / 'stable.json'
    save_game_state(app, str(path))
    original = path.read_bytes()
    app.chargeStage = stage
    try:
        assert save_game_state(app, str(path)) is None
        assert path.read_bytes() == original
    finally:
        app.chargeStage = None


def test_enhanced_ai_closes_declarations_before_normal_decisions(scene):
    from gameStateTree import GameAction
    app, charger, defender, origin, facing, contact = declared_charge(scene)
    app.fsm.request('MovementPhase')
    ai = app.AIplayer2
    stages = []

    async def execute(action):
        stages.append((app.chargeStage, action.action_type))

    with combat_tasks(app) as run, \
            patch.object(ai, 'active', True), \
            patch.object(ai, 'player_units', [charger]), \
            patch.object(ai, 'enemy_units', [defender]), \
            patch.object(ai, 'execute_action', AsyncMock(side_effect=execute)), \
            patch.object(ai, 'make_decision', AsyncMock(return_value=GameAction('end_phase', 'system', {}))), \
            patch.object(app, 'save_game_state'), \
            patch.object(app.fsm, 'nextPhase') as next_phase, \
            patch('aiMinimaxIntegration.TreeVisualizer'):
        run(ai.take_turn())
    assert stages == [('declarations', 'move'), ('remaining', 'end_phase')]
    next_phase.assert_called_once()
    assert not ai._turn_running


def test_live_flee_move_finishes_before_charge_roll(scene):
    app, charger, defender, origin, facing, contact = declared_charge(scene)
    app.fsm.request('MovementPhase')
    before = Vec3(defender.bodyNP.getPos())
    rolls = [[2, 2], [1, 1]]

    async def roll(count, bonus=False):
        if len(rolls) == 1:
            assert defender.state == 'IsFleeing'
            assert (defender.bodyNP.getPos() - before).length() == pytest.approx(4, abs=.05)
        return [], rolls.pop(0)

    with combat_tasks(app) as run, \
            patch.object(app, 'aiControls', return_value=False), \
            patch.object(app, 'makeChoiceNew', AsyncMock(side_effect=['Yes', 'flee'])), \
            patch.object(app.combat, 'swiftstrideChoice', AsyncMock(return_value=False)), \
            patch.object(app.combat, 'rullTerninger', AsyncMock(side_effect=roll)):
        run(app.combat.chargeAndChargeReaction(charger, contact, origin, facing, SimpleNamespace(done=None)))
        run(resolve_declarations(app))
    assert not rolls
    assert defender.state == 'IsFleeing' and defender.fledThisPhase
    assert charger.state == 'Moved'
    assert (charger.bodyNP.getPos() - origin).length() == pytest.approx(
        charger.unit.model.get_movement(4) + 1, abs=.05)


@pytest.mark.parametrize('stage', ['resolving', 'blocked'])
def test_active_resolution_blocks_move_and_phase_input(scene, stage):
    app, charger, defender, origin, facing, contact = declared_charge(scene)
    app.fsm.request('MovementPhase')
    app.chargeStage = stage
    before = charger.bodyNP.getTransform()
    try:
        assert app.movement.moveUnit(charger) is False
        app.fsm.nextPhase()
        assert app.fsm.state == 'MovementPhase'
        assert charger.bodyNP.getTransform() == before
    finally:
        app.chargeStage = None


def test_phase_button_resolves_once_before_allowing_remaining_moves(scene, tmp_path):
    from direct.task import Task
    app, charger, defender, origin, facing, contact = declared_charge(scene)
    app.fsm.request('MovementPhase')
    assert app.hud._end_btn['text'] == 'RESOLVE\nCHARGES'
    app.graphicsEngine.renderFrame()
    app.graphicsEngine.renderFrame()
    assert app.screenshot(str(tmp_path / 'resolve-charges.png'), defaultFilename=False)

    async def close_declarations():
        app.fsm.nextPhase()
        app.fsm.nextPhase()
        assert len(app.taskMgr.getTasksNamed('resolveChargesTask')) == 1
        while app.chargeStage != 'remaining':
            await Task.pause(.1)

    with combat_tasks(app) as run, patch('game_fsm.taskMgr', app.taskMgr, create=True):
        run(close_declarations())
    assert app.fsm.state == 'MovementPhase'
    assert app.hud._end_btn['text'] == 'END\nPHASE'
    app.fsm.nextPhase()
    assert app.fsm.state == 'ShootingPhase'