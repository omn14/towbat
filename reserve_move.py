"""Optional Basic Movement after Shooting (Rulebook pp. 118, 123-125, 177)."""

from panda3d.core import TransformState, Vec2

from characters import side_of
from magic_items import current_turn
from rules_log import rule_log, rule_skipped
from scouts import nearest_enemy, placement_error


def has_reserve_move(model):
    return any(rule.get('reserve_move') or rule.get('name', '').lower() == 'reserve move'
               for rule in model.special_rules if isinstance(rule, dict))


def has_majority(unit):
    count = unit.unit.nmodels
    enabled = count if has_reserve_move(unit.unit.model) else 0
    joined = getattr(unit, 'joinedCharacter', None)
    if joined is not None and joined.unit.nmodels > 0:
        count += 1
        enabled += int(has_reserve_move(joined.unit.model))
    return enabled > count / 2


def movement_restriction(game, unit):
    if getattr(unit, 'marchedThisTurn', False):
        return 'marched during Movement'
    if getattr(unit, 'chargedThisTurn', False) or any(
            entry.charger is unit for entry in getattr(game, 'chargeDeclarations', [])):
        return 'charged or attempted a charge during Movement'
    if getattr(unit, 'fledThisPhase', False) or unit.state == 'IsFleeing':
        return 'fled during Movement'
    return None


def record_movement(game):
    for unit in game.units:
        unit.reserveMovementTurn = current_turn(game)
        unit.reserveMovementBlocked = movement_restriction(game, unit)


def unavailable(game, unit):
    from chaos_gifts import succumbed
    if succumbed(unit):
        return 'succumbed to Stupidity; cannot move except to flee (p. 178)'
    if getattr(unit, 'hostUnit', None) is not None or not has_majority(unit):
        return 'a majority of models must have Reserve Move'
    if (not unit.isDeployed or unit.unit.nmodels <= 0 or unit.bodyNP.isEmpty()
            or getattr(unit, 'isInCombat', False) or unit.state not in ('Idle', 'Moved')):
        return 'not an unengaged, non-fleeing unit on the battlefield'
    if getattr(unit, 'attemptedRallyThisTurn', False):
        return 'rallied this turn'
    token = current_turn(game)
    if getattr(unit, 'reserveDoneTurn', None) == token:
        return 'Reserve Move already completed or declined this turn'
    if getattr(unit, 'reserveMovementTurn', None) == token:
        return getattr(unit, 'reserveMovementBlocked', None)
    return movement_restriction(game, unit)


def candidates(game):
    player = game.roundCounter.current_player
    return [unit for unit in game.units if side_of(game, unit, None) == player
            and unavailable(game, unit) is None]


def in_reserve(game):
    return getattr(getattr(game, 'fsm', None), 'state', None) == 'ReserveMovePhase'


def begin(game):
    player = game.roundCounter.current_player
    for unit in game.units:
        if side_of(game, unit, None) == player and has_majority(unit):
            reason = unavailable(game, unit)
            if reason:
                rule_skipped('Reserve Move', unit, reason)
    if not candidates(game):
        return False
    game.fsm.request('ReserveMovePhase')
    return True


async def ai_moves(game):
    for unit in candidates(game):
        rule_skipped('Reserve Move', unit, 'AI keeps its position; declines the optional move')
        finish_unit(game, unit)
    game.fsm.nextPhase()


def prepare(game):
    """Separate the extra move's budget from the earlier Movement phase (p. 177)."""
    for unit in candidates(game):
        if getattr(unit, 'reserveMoveOriginal', None) is not None:
            continue
        fields = ('hasMovedThisTurn', 'moveSpentThisTurn', 'manoeuvreThisTurn', 'redressDelta')
        unit.reserveMoveOriginal = {field: getattr(unit, field, None) for field in fields}
        unit.reserveMoveOriginal['state'] = unit.state
        unit.request('Idle')
        unit.hasMovedThisTurn = False
        unit.moveSpentThisTurn = 0.0
        unit.manoeuvreThisTurn = None
        unit.redressDelta = 0
        unit.wouldMarch = False


def finish_unit(game, unit, *, moved=False):
    original = getattr(unit, 'reserveMoveOriginal', None)
    if original is not None:
        unit.request('Moved' if moved else original['state'])
        for field, value in original.items():
            if field != 'state':
                setattr(unit, field, value)
        unit.hasMovedThisTurn = bool(original['hasMovedThisTurn'] or moved)
    unit.reserveMoveOriginal = None
    unit.reserveDoneTurn = current_turn(game)
    unit.wouldMarch = False


def finish_window(game):
    for unit in game.units:
        if getattr(unit, 'reserveMoveOriginal', None) is not None:
            moved = unit.moveSpentThisTurn > 0
            report = rule_log if moved else rule_skipped
            report('Reserve Move', unit, 'ends after manoeuvring' if moved else 'player declines the optional move')
            finish_unit(game, unit, moved=moved)
    game.taskMgr.remove('taskLoopPathTowardsMouse')
    game.setGroundOverlay(False)


def commit(game, unit, *, drilled_ready=False):
    """Validate an extra Basic Movement move; never declare a charge (p. 177)."""
    reason = unavailable(game, unit)
    if not in_reserve(game) or side_of(game, unit, None) != game.roundCounter.current_player:
        reason = 'not this player\'s Reserve Move window'
    if reason or game.arcPoint is None:
        rule_skipped('Reserve Move', unit, reason or 'no movement destination')
        return False
    from drilled import before_move, has_drilled
    if has_drilled(unit) and not drilled_ready:
        aim = getattr(unit, '_movementAim', None)
        if aim is None:
            aim = (game.arcPoint * 2 - Vec2(1, 1)) * 50
        unit._drilledMoveActive = True

        async def drilled_move():
            try:
                await before_move(game, unit, 'Reserve Move')
                game.pathTowardsMouse(unit, aim.x, aim.y)
                if game.arcPoint is not None:
                    commit(game, unit, drilled_ready=True)
            finally:
                unit._drilledMoveActive = False

        return game.taskMgr.add(drilled_move(), 'drilledReserveMove')
    origin, heading = unit.bodyNP.getPos(), unit.bodyNP.getHpr()
    forward, right = unit.bodyNP.getQuat().getForward(), unit.bodyNP.getQuat().getRight()
    target = (game.arcPoint * 2 - Vec2(1, 1)) * 50
    unit.bodyNP.setPos(target.x, target.y, 0)
    formed = not getattr(unit, 'isSkirmisher', False)
    if formed:
        unit.bodyNP.setH(heading.x + game.arcPointRotation)
        unit.bodyNP.setPos(unit.bodyNP.getPos() - unit.bodyNP.getQuat().getForward() * (unit.unitHeight / 2))
    destination = unit.bodyNP.getPos()
    displacement = destination - origin
    distance = max(displacement.length(), game.moveArceDistance)
    allowance = game.movement.movementAllowance(unit, origin, destination)
    error = placement_error(game, unit, deployment_zone=False)
    nearest, enemy = nearest_enemy(game, unit)
    if nearest < 1 - 1e-6:
        error = f'must remain 1" from {enemy.unit.name}; Reserve Move cannot charge'
    if distance + unit.moveSpentThisTurn > allowance + 1e-4:
        error = f'{distance:.3f}" plus manoeuvres exceeds M{allowance:g}; cannot march'
    if formed and abs(game.arcPointRotation) > 1e-4 and unit.manoeuvreThisTurn is not None:
        error = f'already performed {unit.manoeuvreThisTurn}; cannot also wheel'
    if formed and abs(game.arcPointRotation) <= 1e-4:
        sideways = abs(displacement.dot(right)) > 1e-4
        backwards = displacement.dot(forward) < -1e-4
        if sideways or backwards:
            if unit.manoeuvreThisTurn is not None or distance > allowance / 2 + 1e-4:
                error = 'sideways/backwards movement requires its own manoeuvre and is limited to half M'
            elif sideways and abs(displacement.dot(forward)) > 1e-4:
                error = 'formed units cannot translate diagonally without wheeling'
    unit.bodyNP.setPos(origin)
    unit.bodyNP.setHpr(heading)
    flying = all(member.unit.model.is_flying() for member in game.movement.movementParticipants(unit))
    if not flying and distance > 1e-6:
        fraction, _ = game.movement.sweepTestDir(unit, TransformState.makePosHpr(origin, heading),
                                                displacement.normalized(), displacement.length(), pass_over=False)
        if fraction < 1 - 1e-4:
            error = 'path crosses another unit or impassable terrain'
    if error:
        unit.bodyNP.node().setTransformDirty()
        rule_skipped('Reserve Move', unit, f'{error}; position restored (p. 177)')
        return False
    unit.bodyNP.setPos(destination)
    unit.bodyNP.setH(heading.x + (game.arcPointRotation if formed else 0))
    unit.bodyNP.node().setTransformDirty()
    from formed_skirmish_charge import movement_route, route_features
    route = movement_route(unit, origin, heading, destination, unit.bodyNP.getH())
    features = route_features(game, route) if route else None
    game.movement.movementAllowance(unit, origin, destination, log=True, features=features)
    game.movement.alignModelsToHillNormal(unit)
    game.movement.dangerousTerrainTests(unit, origin, destination, features=features, route=route)
    game.movement.updateDisrupted(unit)
    rule_log('Reserve Move', unit, f'moved {distance:.3f}" plus {unit.moveSpentThisTurn:g}" manoeuvres '
             f'of M{allowance:g}; no march or charge (p. 177)')
    finish_unit(game, unit, moved=distance > 1e-5 or unit.moveSpentThisTurn > 0)
    game.taskMgr.remove('taskLoopPathTowardsMouse')
    game.setGroundOverlay(False)
    from free_pivot import begin
    begin(game, unit, 'Reserve Move')
    return True