"""Dragon Prince movement and compulsory charges (Rulebook pp. 167, 172)."""

import pytest
from unittest.mock import AsyncMock, patch
from panda3d.core import Vec3

from tests.test_counter_charge_scene import declared_charge
from tests.test_faction_rules_scene import members, scene as scene
from tests.test_shieldwall_scene import combat_tasks


def test_drilled_redress_is_free_and_preserves_front_rank(scene):
    app, charger, defender, origin, facing, contact = declared_charge(scene)
    prince = members(app)['Dragon Prince']
    prince.bodyNP.setPos(0, 12, 0)
    prince.bodyNP.setH(0)
    prince.request('Idle')
    prince.moveSpentThisTurn = 0
    prince.manoeuvreThisTurn = None
    prince.redressDelta = 0
    front = prince.bodyNP.getPos() + Vec3(0, prince.unitHeight / 2, 0)
    old_files = prince.unit.files
    assert app.movement.redressRanks(prince, -1, drilled=True)
    assert prince.unit.files == old_files - 1
    assert (prince.bodyNP.getPos() + Vec3(0, prince.unitHeight / 2, 0)).almostEqual(front)
    assert prince.moveSpentThisTurn == 0 and prince.manoeuvreThisTurn is None
    assert prince.redressDelta == 0 and not prince.hasMovedThisTurn
    assert not app.movement.redressRanks(charger, -1, drilled=True)


def test_drilled_choice_precedes_countercharge_movement(scene):
    app, charger, defender, origin, facing, contact = declared_charge(scene)
    prince = members(app)['Dragon Prince']
    defender.bodyNP.setPos(25, 20, 0)
    prince.bodyNP.setPos(0, 0, 0)
    prince.bodyNP.setH(180)
    before_files = prince.unit.files
    choice = AsyncMock(return_value=f'{before_files - 1} files')
    with combat_tasks(app) as run, \
            patch.object(app, 'aiControls', return_value=False), \
            patch.object(app, 'makeChoiceNew', choice), \
            patch.object(app.combat, 'rullTerninger', AsyncMock(return_value=([], [3]))):
        run(app.combat.counterChargeInterval(charger, prince, origin, facing, defer_charge=True))
    assert prince.unit.files == before_files - 1
    assert prince.chargeDistance == pytest.approx(3, abs=.05)
    assert choice.call_args.kwargs['owner'] is prince
    assert prince.moveSpentThisTurn == 0


@pytest.mark.parametrize('compulsory', [False, True])
def test_drilled_column_redresses_after_dice_and_charges_once(scene, compulsory):
    from charge_declarations import ChargeDeclaration
    from first_charge import begin_charge_attempt
    from drilled import marching_column
    app, silver, defender, origin, facing, contact = declared_charge(scene)
    prince = members(app)['Dragon Prince']
    silver.bodyNP.setPos(25, 20, 0)
    prince.bodyNP.setPos(origin)
    prince.bodyNP.setHpr(facing)
    assert app.movement.redressRanks(prince, 1 - prince.unit.files, drilled=True)
    assert marching_column(prince)
    origin = Vec3(prince.bodyNP.getPos())
    entry = ChargeDeclaration(prince, defender, tuple(origin), tuple(facing),
                              (0, -2, 0), tuple(facing), (0, -2, 0), 10,
                              reaction='hold', compulsory=compulsory)
    prince.hasMovedThisTurn = True
    begin_charge_attempt(prince)
    events = []

    async def roll(*args):
        events.append('dice')
        return [], [6, 6]

    async def choose(options, *args, **kwargs):
        events.append('redress')
        assert ('Keep formation' in options) is not compulsory
        return '3 files'

    with combat_tasks(app) as run, \
            patch.object(app, 'aiControls', return_value=False), \
            patch.object(app, 'makeChoiceNew', side_effect=choose), \
            patch.object(app.combat, 'swiftstrideChargeChoice', AsyncMock(return_value=False)), \
            patch.object(app.combat, 'rullTerninger', side_effect=roll):
        run(app.combat.resolveDeclaredCharge(entry))
    assert events == ['dice', 'redress']
    assert not marching_column(prince)
    assert prince.state == defender.state == 'InCombat'
    assert prince.isInCombatWith == [defender]
    assert prince.chargeAttempts == 1 and not prince.chargeAttemptPending


def test_non_drilled_column_fails_without_moving(scene):
    from charge_declarations import ChargeDeclaration
    from first_charge import begin_charge_attempt
    app, charger, defender, origin, facing, contact = declared_charge(scene)
    charger.bodyNP.setPos(origin)
    assert app.movement.redressRanks(charger, 1 - charger.unit.files)
    origin = Vec3(charger.bodyNP.getPos())
    begin_charge_attempt(charger)
    entry = ChargeDeclaration(charger, defender, tuple(origin), tuple(facing),
                              (0, -2, 0), tuple(facing), (0, -2, 0), 10, reaction='hold')
    with combat_tasks(app) as run, \
            patch.object(app.combat, 'swiftstrideChargeChoice', AsyncMock(return_value=False)), \
            patch.object(app.combat, 'rullTerninger', AsyncMock(return_value=([], [6, 6]))):
        run(app.combat.resolveDeclaredCharge(entry))
    assert charger.bodyNP.getPos().almostEqual(origin)
    assert charger.state == 'Moved' and not charger.isInCombat
    assert charger.chargeAttempts == 1 and not charger.chargeAttemptPending


def test_drilled_giving_ground_can_redress_while_engaged(scene):
    from direct.interval.IntervalGlobal import Sequence
    app, charger, defender, origin, facing, contact = declared_charge(scene)
    app.attackSequence = Sequence()
    prince = members(app)['Dragon Prince']
    charger.bodyNP.setPos(25, 20, 0)
    prince.bodyNP.setPos(0, -(prince.unitHeight + defender.unitHeight) / 2, 0)
    prince.bodyNP.setH(0)
    prince.isInCombatWith = [defender]
    prince.isInCombat = True
    prince.request('InCombat')
    front = prince.bodyNP.getY() + prince.unitHeight / 2
    files = prince.unit.files
    with combat_tasks(app) as run, \
            patch.object(app, 'aiControls', return_value=False), \
            patch.object(app, 'makeChoiceNew', AsyncMock(return_value=f'{files - 1} files')):
        run(app.combat.giveGroundMove(prince, []))
    assert prince.unit.files == files - 1
    assert prince.bodyNP.getY() + prince.unitHeight / 2 == pytest.approx(front - 2, abs=.05)


def test_drilled_remaining_move_redresses_before_replotting(scene):
    app, charger, defender, origin, facing, contact = declared_charge(scene)
    prince = members(app)['Dragon Prince']
    prince.bodyNP.setPos(15, 0, 0)
    prince.bodyNP.setH(0)
    app.fsm.request('MovementPhase')
    app.chargeStage = 'remaining'
    app.unitToMove = prince
    files = prince.unit.files
    app.pathTowardsMouse(prince, 15, 8)
    with combat_tasks(app) as run, \
            patch.object(app, 'aiControls', return_value=False), \
            patch.object(app, 'makeChoiceNew', AsyncMock(return_value=f'{files - 1} files')) as choice:
        async def move():
            await app.movement.moveUnit(prince)
        run(move())
    assert prince.unit.files == files - 1
    assert prince.state == 'Moved'
    assert prince.bodyNP.getY() > 2
    choice.assert_awaited_once()
    assert not prince._drilledMoveActive


def test_column_has_triple_march_and_no_rank_bonus(scene):
    from drilled import march_multiplier
    from psychology import rank_bonus
    app, charger, defender, origin, facing, contact = declared_charge(scene)
    prince = members(app)['Dragon Prince']
    prince.unit.files, prince.unit.ranks, prince.unit.nmodels = 5, 6, 30
    assert march_multiplier(prince) == 3
    assert rank_bonus(prince.unit) == 0
    prince.unit.files, prince.unit.ranks = 6, 5
    assert march_multiplier(prince) == 2
    assert rank_bonus(prince.unit) == 1


@pytest.mark.parametrize('roll, compulsory', [([1, 1], False), ([6, 6], True)])
def test_impetuous_tests_once_before_reactions_and_forces_failed_charge(scene, roll, compulsory):
    from charge_declarations import begin_declarations, resolve_declarations
    app, silver, defender, origin, facing, contact = declared_charge(scene)
    prince = members(app)['Dragon Prince']
    silver.bodyNP.setPos(25, 20, 0)
    prince.bodyNP.setPos(origin)
    prince.bodyNP.setHpr(facing)
    app.fsm.request('MovementPhase')
    begin_declarations(app)
    moves = []

    async def resolve(entry):
        assert entry.compulsory
        assert entry.charger is prince and entry.defender is defender
        moves.append(entry)

    with combat_tasks(app) as run, \
            patch.object(app, 'aiControls', return_value=True), \
            patch.object(app, 'rollLeadershipDice', AsyncMock(return_value=roll)) as dice, \
            patch.object(app.combat, 'counterChargeOption', return_value=None), \
            patch.object(app.combat, 'resolveDeclaredCharge', side_effect=resolve):
        run(resolve_declarations(app))
        run(resolve_declarations(app))
    assert bool(moves) is compulsory
    dice.assert_awaited_once()
    assert prince.hasMovedThisTurn is compulsory
    assert app.chargeStage == 'remaining'


@pytest.mark.parametrize('reason', ['rear', 'range', 'fleeing', 'restricted'])
def test_impetuous_does_not_test_without_legal_target(scene, reason):
    from charge_declarations import begin_declarations, resolve_declarations
    app, silver, defender, origin, facing, contact = declared_charge(scene)
    prince = members(app)['Dragon Prince']
    silver.bodyNP.setPos(25, 20, 0)
    prince.bodyNP.setPos(origin)
    prince.bodyNP.setH(180 if reason == 'rear' else 0)
    if reason == 'range':
        prince.bodyNP.setY(-40)
    elif reason == 'fleeing':
        prince.request('IsFleeing')
    elif reason == 'restricted':
        prince.cannotChargeThisTurn = True
    app.fsm.request('MovementPhase')
    begin_declarations(app)
    with combat_tasks(app) as run, \
            patch.object(app, 'rollLeadershipDice', AsyncMock(return_value=[6, 6])) as dice:
        run(resolve_declarations(app))
    dice.assert_not_awaited()
    assert not app.chargeDeclarations


def test_optional_drilled_column_can_stay_and_fail_charge(scene):
    from charge_declarations import ChargeDeclaration
    from first_charge import begin_charge_attempt
    app, silver, defender, origin, facing, contact = declared_charge(scene)
    prince = members(app)['Dragon Prince']
    silver.bodyNP.setPos(25, 20, 0)
    prince.bodyNP.setPos(origin)
    prince.bodyNP.setHpr(facing)
    assert app.movement.redressRanks(prince, 1 - prince.unit.files, drilled=True)
    origin = Vec3(prince.bodyNP.getPos())
    begin_charge_attempt(prince)
    entry = ChargeDeclaration(prince, defender, tuple(origin), tuple(facing),
                              (0, -2, 0), tuple(facing), (0, -2, 0), 10, reaction='hold')
    with combat_tasks(app) as run, \
            patch.object(app, 'aiControls', return_value=False), \
            patch.object(app, 'makeChoiceNew', AsyncMock(return_value='Keep formation')), \
            patch.object(app.combat, 'swiftstrideChargeChoice', AsyncMock(return_value=False)), \
            patch.object(app.combat, 'rullTerninger', AsyncMock(return_value=([], [6, 6]))):
        run(app.combat.resolveDeclaredCharge(entry))
    assert prince.unit.files == 1 and prince.state == 'Moved'
    assert prince.bodyNP.getPos().almostEqual(origin)
    assert prince.chargeAttempts == 1 and not prince.chargeAttemptPending


def test_compulsory_drilled_redress_cannot_overlap_neighbour(scene, capsys):
    from drilled import before_move, marching_column
    app, silver, defender, origin, facing, contact = declared_charge(scene)
    prince = members(app)['Dragon Prince']
    prince.bodyNP.setPos(15, 0, 0)
    prince.bodyNP.setH(0)
    assert app.movement.redressRanks(prince, 1 - prince.unit.files, drilled=True)
    silver.bodyNP.setPos(15 + (prince.unitWidth + silver.unitWidth) / 2 + .01,
                         prince.bodyNP.getY() + (prince.unitHeight - silver.unitHeight) / 2, 0)
    silver.bodyNP.setH(0)
    transform = prince.bodyNP.getTransform()
    with combat_tasks(app) as run, patch.object(app, 'makeChoiceNew', AsyncMock()) as choice:
        run(before_move(app, prince, 'charge move', compulsory=True))
    choice.assert_not_awaited()
    assert marching_column(prince) and prince.bodyNP.getTransform() == transform
    assert 'no legal Combat Order redress fits' in capsys.readouterr().out


def test_column_march_preview_and_commit_can_exceed_twice_movement(scene):
    app, charger, defender, origin, facing, contact = declared_charge(scene)
    prince = members(app)['Dragon Prince']
    prince.bodyNP.setPos(30, -18, 0)
    prince.bodyNP.setH(0)
    assert app.movement.redressRanks(prince, 1 - prince.unit.files, drilled=True)
    origin = Vec3(prince.bodyNP.getPos())
    app.fsm.request('MovementPhase')
    app.chargeStage = 'remaining'
    app.pathTowardsMouse(prince, 30, 6)
    with combat_tasks(app) as run, patch.object(app, 'aiControls', return_value=True):
        async def move():
            await app.movement.moveUnit(prince)
        run(move())
    assert prince.state == 'Moved' and prince.marchedThisTurn
    assert (prince.bodyNP.getPos() - origin).length() > 2 * app.movement.movementAllowance(prince)


def test_pending_drilled_choice_blocks_other_input_and_persistence(scene, tmp_path):
    from persistence import load_game_state, save_game_state
    app, charger, defender, origin, facing, contact = declared_charge(scene)
    prince = members(app)['Dragon Prince']
    app.fsm.request('MovementPhase')
    app.chargeStage = 'remaining'
    filename = save_game_state(app, str(tmp_path / 'before-choice.json'))
    prince._drilledMoveActive = True
    transform = charger.bodyNP.getTransform()
    try:
        assert app.movement.moveUnit(charger) is False
        app.fsm.nextPhase()
        assert app.fsm.state == 'MovementPhase'
        assert save_game_state(app, filename) is None
        assert load_game_state(app, filename) is None
        assert prince._drilledMoveActive
        assert charger.bodyNP.getTransform() == transform
    finally:
        prince._drilledMoveActive = False


def test_redress_cannot_spend_movement_to_evade_impetuous(scene):
    app, charger, defender, origin, facing, contact = declared_charge(scene)
    prince = members(app)['Dragon Prince']
    app.fsm.request('MovementPhase')
    before = prince.unit.files
    assert not app.movement.redressRanks(prince, -1)
    assert prince.unit.files == before and prince.moveSpentThisTurn == 0


@pytest.mark.parametrize('ai', [False, True])
@pytest.mark.parametrize('voluntary', [False, True])
def test_compulsory_column_charge_countercharge_and_first_charge_after_reload(
        scene, tmp_path, ai, voluntary, capsys):
    from charge_declarations import begin_declarations, queue_charge, resolve_declarations
    from first_charge import begin_charge_attempt
    from persistence import load_game_state, save_game_state
    from psychology import obb_distance
    app, silver, defender, origin, facing, contact = declared_charge(scene)
    prince = members(app)['Dragon Prince']
    silver.bodyNP.setPos(25, 20, 0)
    prince.bodyNP.setPos(origin)
    prince.bodyNP.setHpr(facing)
    assert app.movement.redressRanks(prince, 1 - prince.unit.files, drilled=True)
    app.fsm.request('MovementPhase')
    begin_declarations(app)
    if voluntary:
        begin_charge_attempt(prince)
        queue_charge(app, prince, defender, prince.bodyNP.getPos(), prince.bodyNP.getHpr())
    filename = save_game_state(app, str(tmp_path / 'column-declarations.json'))
    load_game_state(app, filename)
    events = []

    async def leadership():
        events.append('Impetuous')
        return [6, 6]

    async def dice(count, bonus=False):
        events.append('Counter Charge' if count == 1 else 'Charge roll')
        return [], [3] if count == 1 else [6, 6]

    async def choose(options, *args, **kwargs):
        if kwargs['owner'] is defender:
            return next(option for option in options if option.startswith('counter charge'))
        assert kwargs['owner'] is prince
        assert 'Keep formation' not in options
        return '3 files'

    with combat_tasks(app) as run, \
            patch.object(app, 'aiControls', return_value=ai), \
            patch.object(app, 'makeChoiceNew', side_effect=choose) as choices, \
            patch.object(app, 'rollLeadershipDice', side_effect=leadership) as leadership_roll, \
            patch.object(app.combat, 'rullTerninger', side_effect=dice), \
            patch.object(app.combat, 'swiftstrideChargeChoice', AsyncMock(return_value=False)):
        run(resolve_declarations(app))
    assert events == ['Impetuous', 'Counter Charge', 'Charge roll']
    leadership_roll.assert_awaited_once()
    if ai:
        choices.assert_not_called()
    assert prince.unit.files == (2 if ai else 3)
    assert prince.unit.ranks <= prince.unit.files
    assert prince.state == defender.state == 'InCombat'
    assert prince.isInCombatWith == [defender] and defender.isInCombatWith == [prince]
    assert obb_distance(app.psychology._unit_box(prince), app.psychology._unit_box(defender)) < .06
    assert prince.chargeAttempts == defender.chargeAttempts == 1
    assert not prince.chargeAttemptPending and not defender.chargeAttemptPending
    assert prince.firstChargeDisruptedBy and defender.firstChargeDisruptedBy
    assert prince.moveSpentThisTurn == 0 and not prince.marchedThisTurn
    assert app.chargeStage == 'remaining'
    output = capsys.readouterr().out
    assert '2D6=12 vs Ld' in output and 'must declare a charge' in output
    assert 'Drilled' in output and 'costs 0' in output


def test_live_enhanced_ai_cannot_end_phase_without_compulsory_charge(scene):
    from gameStateTree import GameAction
    app, silver, defender, origin, facing, contact = declared_charge(scene)
    prince = members(app)['Dragon Prince']
    silver.bodyNP.setPos(25, 20, 0)
    prince.bodyNP.setPos(origin)
    prince.bodyNP.setHpr(facing)
    app.fsm.request('MovementPhase')
    ai = app.AIplayer2
    with combat_tasks(app) as run, \
            patch.object(ai, 'active', True), \
            patch.object(ai, 'player_units', [prince]), \
            patch.object(ai, 'enemy_units', [defender]), \
            patch.object(ai, 'execute_action', AsyncMock(return_value=None)), \
            patch.object(ai, 'make_decision', AsyncMock(return_value=GameAction('end_phase', 'system', {}))), \
            patch.object(app, 'save_game_state'), \
            patch.object(app.fsm, 'nextPhase') as next_phase, \
            patch('aiMinimaxIntegration.TreeVisualizer'), \
            patch.object(app, 'aiControls', return_value=True), \
            patch.object(app, 'rollLeadershipDice', AsyncMock(return_value=[6, 6])), \
            patch.object(app.combat, 'rullTerninger', AsyncMock(side_effect=[([], [3]), ([], [6, 6])])), \
            patch.object(app.combat, 'swiftstrideChargeChoice', AsyncMock(return_value=False)):
        run(ai.take_turn())
    assert prince.state == defender.state == 'InCombat'
    assert app.chargeStage == 'remaining'
    next_phase.assert_called_once()


def test_impetuous_reuses_veteran_reroll_and_joined_model_scope(scene):
    from characters import join_unit
    from charge_declarations import resolve_declarations
    from impetuous import has_impetuous
    app, silver, defender, origin, facing, contact = declared_charge(scene)
    prince = members(app)['Dragon Prince']
    silver.bodyNP.setPos(origin)
    prince.bodyNP.setPos(-30, 20, 0)
    mage = members(app)['Mage']
    mage.unit.model.special_rules.append({'name': 'Impetuous'})
    assert join_unit(app, mage, silver)
    assert has_impetuous(silver)
    silver.unit.model.special_rules.append({'name': 'Veteran', 'veteran': True})
    app.fsm.request('MovementPhase')
    with combat_tasks(app) as run, \
            patch.object(app, 'aiControls', return_value=True), \
            patch.object(app, 'rollLeadershipDice', AsyncMock(side_effect=[[6, 6], [1, 1]])) as dice:
        run(resolve_declarations(app))
    assert dice.await_count == 2
    assert silver.state == 'Idle' and not silver.hasMovedThisTurn
    assert not app.chargeDeclarations