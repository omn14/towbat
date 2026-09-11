"""Optional post-move pivots: Quick Turn p. 183 and Lumbering p. 195."""

import math

from rules_log import rule_log, rule_skipped


def rule_for(game, unit, context):
    if getattr(unit, 'isSkirmisher', False):
        return None
    model = unit.unit.model
    if model.troop_type_rule('Lumbering'):
        return 'Lumbering'
    if (context == 'Remaining Moves'
            and getattr(game.fsm, 'state', '') == 'MovementPhase'
            and any(rule.get('name') == 'Open Order' for rule in model.special_rules)):
        return 'Quick Turn'
    return None


def pending(game):
    return any(getattr(unit, 'freePivot', None) is not None for unit in game.units)


def begin(game, unit, context):
    """One optional facing placement after a completed move (pp. 183, 195)."""
    if unit.bodyNP.isEmpty() or unit.unit.nmodels <= 0:
        return False
    name = rule_for(game, unit, context)
    if name is None:
        return False
    if (getattr(unit, 'marchedThisTurn', False) or getattr(unit, 'chargedThisTurn', False)
            or unit.state == 'IsFleeing' or getattr(unit, 'isInCombat', False)):
        rule_skipped(name, unit, f'{context}: marched, charged, fled or engaged; no free pivot')
        return False
    if game.aiControls(unit):
        rule_skipped(name, unit, f'{context}: AI keeps its facing')
        return False
    if getattr(unit, 'freePivot', None) is not None:
        return False
    unit.freePivot = {'rule': name, 'heading': unit.bodyNP.getH()}

    def finish():
        state = unit.freePivot
        if not unit.bodyNP.isEmpty():
            delta = (unit.bodyNP.getH() - state['heading'] + 180) % 360 - 180
            report = rule_log if abs(delta) > 1e-5 else rule_skipped
            report(name, unit, f'{context}: free pivot {delta:.1f} degrees; maximum 90')
            game.movement.alignModelsToHillNormal(unit)
            game.movement.updateDisrupted(unit)
        unit.freePivot = None

    game.startFreeReform(unit, on_done=finish)
    return True


def constrain(game, unit):
    """Clamp the live facing, then validate its final placement (pp. 125, 183)."""
    from scouts import model_base_boxes, nearest_enemy, placement_error
    from psychology import _box_corners

    state = unit.freePivot
    delta = (unit.bodyNP.getH() - state['heading'] + 180) % 360 - 180
    center = unit.bodyNP.getPos()
    radius = max((math.hypot(corner[0] - center.x, corner[1] - center.y)
                  for box in model_base_boxes(unit) for corner in _box_corners(*box)), default=0)
    movement = game.movement.movementAllowance(unit, center, center)
    limit = min(90, math.degrees(2 * math.asin(min(1, movement / radius)))) if radius else 90
    unit.bodyNP.setH(state['heading'] + max(-limit, min(limit, delta)))
    unit.bodyNP.node().setTransformDirty()
    error = placement_error(game, unit, deployment_zone=False)
    distance, enemy = nearest_enemy(game, unit)
    if distance < 1 - 1e-5:
        error = f'must remain 1 inch from {enemy.unit.name}'
    return error is None