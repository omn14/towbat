"""Pre-game Basic Movement, Rulebook pp. 123-125, 180; Official FAQ v1.5.3."""

import random

from panda3d.core import BitMask32, TransformState, Vec2, Vec3

from characters import detach_character, side_of
from rules_log import battle_log, dice_roll, rule_log, rule_skipped
from scouts import (has_deployment_rule, nearest_enemy, placement_error,
                    scouts_block_vanguard)


def vanguard_unavailable(unit):
    """Quiet eligibility check; a character may be left by Skirmishers (FAQ)."""
    if not has_deployment_rule(unit, 'vanguard'):
        return 'does not have Vanguard'
    if not unit.isDeployed or unit.unit.nmodels <= 0:
        return 'not deployed or no surviving models'
    if scouts_block_vanguard(unit):
        return 'deployed using Scouts (Official FAQ)'
    character = getattr(unit, 'joinedCharacter', None)
    if (character is not None and not has_deployment_rule(character, 'vanguard')
            and not getattr(unit, 'isSkirmisher', False)):
        return f'{character.unit.name} lacks Vanguard and prevents its formed unit moving (Official FAQ)'
    return None


def vanguard_candidates(game, player):
    units = game.player1Units if player == 1 else game.player2Units
    return [unit for unit in units if not getattr(unit, 'vanguardDone', False)
            and vanguard_unavailable(unit) is None]


def _participating_profiles(unit):
    profiles = [unit.unit.model]
    character = getattr(unit, 'joinedCharacter', None)
    if character is not None and has_deployment_rule(character, 'vanguard'):
        profiles.append(character.unit.model)
    return profiles


def vanguard_flying(unit):
    return all(profile.is_flying() for profile in _participating_profiles(unit))


def vanguard_movement(unit):
    """Basic Movement uses the slowest participating model, never march (p. 123)."""
    profiles = _participating_profiles(unit)
    flying = vanguard_flying(unit)
    return min(profile.get_fly_movement(0) if flying else profile.get_movement(0)
               for profile in profiles)


def vanguard_charge_blocked(game, unit):
    """Only an actual Vanguard move bars the owner's first-turn declaration (p. 180)."""
    owner = side_of(game, unit, default=None)
    character = getattr(unit, 'joinedCharacter', None)
    return bool(owner is not None and game.roundCounter.currentRoundPlayer[owner - 1] == 0
                and (getattr(unit, 'madeVanguardMove', False)
                     or getattr(character, 'madeVanguardMove', False)))


def in_vanguard(game):
    return (getattr(getattr(game, 'fsm', None), 'state', None) == 'DeployPhase'
            and getattr(game, 'deploymentStage', None) == 'vanguard')


def record_vanguard_move(unit):
    unit.madeVanguardMove = True
    character = getattr(unit, 'joinedCharacter', None)
    if character is not None:
        character.madeVanguardMove = True


def vanguard_position_error(game, unit):
    error = placement_error(game, unit, deployment_zone=False)
    if error:
        return error
    distance, enemy = nearest_enemy(game, unit)
    if distance < 1.0 - 1e-6:
        return f'{distance:.3f}" from {enemy.unit.name}; a normal move must stay at least 1" from enemies (p. 118)'
    return None


def commit_vanguard_move(game, unit):
    """Commit only a legal Basic Movement move; never enter charge/turn-move code."""
    if (not in_vanguard(game) or unit.unitName != getattr(game, 'vanguardActive', None)
            or unit not in vanguard_candidates(game, game.roundCounter.current_player)):
        rule_skipped('Vanguard', unit, 'not the active unresolved Vanguard unit')
        return False
    if game.arcPoint is None:
        rule_skipped('Vanguard', unit, 'no valid movement destination')
        return False
    origin = unit.bodyNP.getPos()
    heading = unit.bodyNP.getHpr()
    forward = unit.bodyNP.getQuat().getForward()
    right = unit.bodyNP.getQuat().getRight()
    allowance = vanguard_movement(unit)
    flying = vanguard_flying(unit)
    character = getattr(unit, 'joinedCharacter', None)
    leaving = character is not None and not has_deployment_rule(character, 'vanguard')
    if leaving:
        character_transform = character.bodyNP.getTransform()
        character.bodyNP.wrtReparentTo(unit.bodyNP.getParent())
        character.hostUnit = None
        unit.joinedCharacter = None

    target = (game.arcPoint * 2 - Vec2(1, 1)) * 50
    unit.bodyNP.setPos(target.x, target.y, 0)
    if not getattr(unit, 'isSkirmisher', False):
        unit.bodyNP.setH(heading.x + game.arcPointRotation)
        unit.bodyNP.setPos(unit.bodyNP.getPos() - unit.bodyNP.getQuat().getForward() * (unit.unitHeight / 2))
    unit.bodyNP.node().setTransformDirty()
    destination = unit.bodyNP.getPos()
    displacement = destination - origin
    distance = max(displacement.length(), game.moveArceDistance)
    moved = displacement.length() > 1e-5 or abs(unit.bodyNP.getH() - heading.x) > 1e-5
    if not moved:
        unit.bodyNP.setPos(origin)
        unit.bodyNP.setHpr(heading)
        unit.bodyNP.node().setTransformDirty()
        if leaving:
            character.bodyNP.reparentTo(unit.bodyNP)
            character.bodyNP.setTransform(character_transform)
            character.hostUnit = unit
            unit.joinedCharacter = character
        if not unit.madeVanguardMove:
            rule_skipped('Vanguard', unit, 'no displacement; no new charge restriction or character separation')
        finish_vanguard_unit(game, unit)
        return True
    remaining = max(0.0, allowance - unit.moveSpentThisTurn)
    error = None
    if distance > remaining + 1e-4:
        error = f'{distance:.3f}" exceeds remaining Movement {remaining:g}"; Vanguard cannot march'
    if (not getattr(unit, 'isSkirmisher', False) and abs(game.arcPointRotation) > 1e-4
            and unit.manoeuvreThisTurn is not None):
        error = f'already performed {unit.manoeuvreThisTurn}; cannot also wheel (p. 124)'
    if not getattr(unit, 'isSkirmisher', False) and abs(game.arcPointRotation) <= 1e-4:
        sideways = abs(displacement.dot(right)) > 1e-4
        backwards = displacement.dot(forward) < -1e-4
        if sideways or backwards:
            if unit.manoeuvreThisTurn is not None:
                error = f'already performed {unit.manoeuvreThisTurn}; cannot move sideways or backwards (p. 124)'
            elif distance > allowance / 2 + 1e-4:
                error = f'sideways/backwards {distance:.3f}" exceeds half Movement {allowance / 2:g}" (p. 125)'
            elif sideways and abs(displacement.dot(forward)) > 1e-4:
                error = 'formed units cannot translate diagonally without a wheel (p. 123)'
    if not flying:
        allowance += game.movement.pathTerrainModifier(unit, origin, destination)
        if distance + unit.moveSpentThisTurn > max(1.0, allowance) + 1e-4:
            error = f'{distance:.3f}" plus manoeuvres exceeds terrain-adjusted Movement {max(1.0, allowance):g}"'
        displacement = destination - origin
        if displacement.length() > 1e-6:
            fraction, _ = game.movement.sweepTestDir(
                unit, TransformState.makePosHpr(origin, heading),
                displacement.normalized(), displacement.length(), pass_over=False)
            if fraction < 1.0:
                error = 'the Vanguard path crosses another unit or impassable terrain'
    error = error or vanguard_position_error(game, unit)
    if error:
        unit.bodyNP.setPos(origin)
        unit.bodyNP.setHpr(heading)
        unit.bodyNP.node().setTransformDirty()
        if leaving:
            character.bodyNP.reparentTo(unit.bodyNP)
            character.bodyNP.setTransform(character_transform)
            character.hostUnit = unit
            unit.joinedCharacter = character
        rule_skipped('Vanguard', unit, error)
        battle_log(error, 'info')
        return False
    if leaving:
        unit.joinedCharacter = character
        detach_character(unit)
        side = game.player1Units if side_of(game, unit) == 1 else game.player2Units
        side.append(character)
        game.world.attachRigidBody(character.bodyNP.node())
        rule_log('Vanguard', unit, f'Skirmishers leave {character.unit.name}, which lacks Vanguard, at its deployed position (FAQ)')
        game.roundCounter.apply_selection_masks()
    if moved:
        record_vanguard_move(unit)
        rule_log('Vanguard', unit, f'moved {distance:.3f}" of M{allowance:g}; no march, no charge declarations during its first own turn')
        game.movement.alignModelsToHillNormal(unit)
        game.movement.dangerousTerrainTests(unit, origin, destination)
        game.movement.updateDisrupted(unit)
    elif not unit.madeVanguardMove:
        rule_skipped('Vanguard', unit, 'no displacement; no additional charge restriction')
    finish_vanguard_unit(game, unit)
    return True


def begin_vanguard(game):
    """Enter after every ordinary and Scout drop; never re-roll on refresh/load."""
    if getattr(game, 'deploymentStage', None) == 'vanguard':
        return
    for unit in game.units:
        if getattr(unit, 'hostUnit', None) is None and has_deployment_rule(unit, 'vanguard'):
            reason = vanguard_unavailable(unit)
            if reason:
                rule_skipped('Vanguard', unit, reason)
    sides = [player for player in (1, 2) if vanguard_candidates(game, player)]
    if not sides:
        game.fsm.request('StrategyPhase')
        return
    player = sides[0]
    if len(sides) == 2:
        while True:
            rolls = [random.randint(1, 6), random.randint(1, 6)]
            dice_roll(rolls)
            if rolls[0] != rolls[1]:
                player = 1 if rolls[0] > rolls[1] else 2
                rule_log('Vanguard', 'deployment', f'roll-off P1={rolls[0]}, P2={rolls[1]} -> Player {player} moves first')
                break
            rule_log('Vanguard', 'deployment', f'roll-off {rolls[0]}-{rolls[1]} tied; re-roll')
    game.deploymentStage = 'vanguard'
    game.vanguardFirst = player
    game.vanguardActive = None
    game.roundCounter.request('PlayerOne' if player == 1 else 'PlayerTwo')
    refresh_vanguard(game)


def refresh_vanguard(game):
    game.boundary_np.setCollideMask(BitMask32.allOff())
    game.roundCounter.update_round_display()
    game.accept('mouse1', game.setActiveUnit,
                [game.setActiveUnitTask, game.setActiveUnitTaskName])
    battle_log(f'Player {game.roundCounter.current_player}: Vanguard.', 'info')
    if game.roundCounter.current_player == 2 and game.AIplayer2.active:
        candidates = vanguard_candidates(game, 2)
        if candidates:
            game.unitToMove = next((unit for unit in candidates
                                    if unit.unitName == getattr(game, 'vanguardActive', None)), candidates[0])
            game.taskMgr.add(game.taskLoopDeploy, 'taskLoopDeploy', appendTask=True)
    elif getattr(game, 'vanguardActive', None):
        unit = next((unit for unit in game.units if unit.unitName == game.vanguardActive), None)
        if unit is not None:
            game.unitToMove = unit
            game.accept('mouse3', game.onRightClick, [unit])
            game.startTaskFunction(game.taskLoopPathTowardsMouse, 'taskLoopPathTowardsMouse')


def finish_vanguard_unit(game, unit):
    unit.vanguardDone = True
    unit.moveSpentThisTurn = 0.0
    unit.manoeuvreThisTurn = None
    unit.redressDelta = 0
    unit.wouldMarch = False
    game.vanguardActive = None
    game.taskMgr.remove('taskLoopPathTowardsMouse')
    game.setGroundOverlay(False)
    if getattr(game, 'skirmMoveGhost', None):
        game.skirmMoveGhost.removeNode()
        game.skirmMoveGhost = None
    player = game.roundCounter.current_player
    other = 3 - player
    if vanguard_candidates(game, other):
        player = other
    elif not vanguard_candidates(game, player):
        game.fsm.request('StrategyPhase')
        return
    game.roundCounter.request('PlayerOne' if player == 1 else 'PlayerTwo')
    refresh_vanguard(game)


def skip_vanguard(game):
    """End Phase finishes an active manoeuvre or declines remaining optional moves."""
    candidates = vanguard_candidates(game, game.roundCounter.current_player)
    active = next((unit for unit in candidates
                   if unit.unitName == getattr(game, 'vanguardActive', None)), None)
    if active is not None:
        candidates = [active]
    for unit in candidates:
        if not getattr(unit, 'madeVanguardMove', False):
            rule_skipped('Vanguard', unit, 'player declines the optional move; first-turn charges remain available')
        unit.vanguardDone = True
    if candidates:
        finish_vanguard_unit(game, candidates[-1])
    else:
        other = 3 - game.roundCounter.current_player
        if vanguard_candidates(game, other):
            game.roundCounter.request('PlayerOne' if other == 1 else 'PlayerTwo')
            refresh_vanguard(game)
        else:
            game.fsm.request('StrategyPhase')


async def select_vanguard(game, unit, task):
    if unit not in vanguard_candidates(game, game.roundCounter.current_player):
        rule_skipped('Vanguard', unit, vanguard_unavailable(unit) or 'already resolved or not this player\'s unit')
        return task.done
    active = getattr(game, 'vanguardActive', None)
    if active and active != unit.unitName:
        battle_log('Finish the active Vanguard unit first.', 'info')
        return task.done
    if active is None:
        choice = await game.makeChoiceNew(
            ['Move', 'Skip'], Vec3(0, 0, 10), owner=unit,
            prompt=f'{unit.unit.name}: Vanguard',
            detail=f'Movement {vanguard_movement(unit):g}". No marching. A Vanguard move bars first-turn charges.')
        if choice != 'Move':
            rule_skipped('Vanguard', unit, 'optional move declined; first-turn charges remain available')
            finish_vanguard_unit(game, unit)
            return task.done
        game.vanguardActive = unit.unitName
    if game.aiControls(unit):
        origin = unit.bodyNP.getPos()
        direction = unit.bodyNP.getQuat().getForward()
        if not getattr(unit, 'isSkirmisher', False):
            origin += direction * (unit.unitHeight / 2)
        for fraction in (1.0, 0.75, 0.5, 0.25):
            target = origin + direction * vanguard_movement(unit) * fraction
            game.pathTowardsMouse(unit, target.x, target.y)
            if game.movement.moveUnit(unit):
                break
        else:
            rule_skipped('Vanguard', unit, 'AI found no legal forward move; holds position')
            finish_vanguard_unit(game, unit)
    else:
        game.startTaskFunction(game.taskLoopPathTowardsMouse, 'taskLoopPathTowardsMouse')
    return task.done