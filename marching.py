"""Enemy Sighted and musicians' Quick Time (Rulebook pp. 123, 201)."""

from characters import enemy_units
from command_groups import musician_leadership
from psychology import leadership_passed, obb_distance, reroll_leadership
from rules_log import rule_log, rule_skipped
from scouts import model_base_boxes


def nearby_enemy(game, unit):
    own = model_base_boxes(unit)
    nearest, distance = None, float('inf')
    for enemy in enemy_units(game, unit):
        if enemy.state == 'IsFleeing' or enemy.unit.nmodels <= 0:
            continue
        clearance = min((obb_distance(first, second) for first in own
                         for second in model_base_boxes(enemy)), default=float('inf'))
        if clearance < distance:
            nearest, distance = enemy, clearance
    return (nearest, distance) if distance <= 8 else (None, distance)


async def enemy_sighted_test(game, unit, enemy, distance):
    from warband import leadership_for_test
    leadership, _ = leadership_for_test(game.psychology, unit, 'Enemy Sighted')
    leadership = musician_leadership(unit, leadership, 'march', log=True)
    dice = await game.rollLeadershipDice()
    dice = await reroll_leadership(game, unit, 'Enemy Sighted', dice, leadership, game.rollLeadershipDice)
    passed = leadership_passed(sum(dice), leadership)
    unit.marchTestResult = 'passed' if passed else 'failed'
    unit.marchedThisTurn = True
    rule_log('Enemy Sighted', unit,
             f'{enemy.unitName} at {distance:.2f}": 2D6={sum(dice)} vs Ld {leadership} -> '
             f'{"PASS; may march" if passed else "FAIL; normal movement only, counts as marched"} (p. 123)')
    return passed


def request_march(game, unit, on_pass):
    """Delay a committed march for its test; previews never roll (p. 123)."""
    profile = unit.unit.model
    joined = getattr(unit, 'joinedCharacter', None)
    flying = profile.is_flying() and (joined is None or joined.unit.model.is_flying())
    drilled = 'drilled' in {str(name).strip().lower() for name in profile.characteristics.get('Special Rules', [])}
    if flying or drilled:
        rule_log('Fly' if flying else 'Drilled', unit,
                 'committed march needs no Enemy Sighted test (pp. 167, 170)')
        return True
    state = getattr(unit, 'marchTestResult', None)
    if state == 'passed':
        return True
    if state in ('failed', 'pending'):
        rule_skipped('Enemy Sighted', unit, f'{state} test; march not committed')
        if state == 'failed' and hasattr(game, 'startTaskFunction'):
            game.startTaskFunction(game.taskLoopPathTowardsMouse, 'taskLoopPathTowardsMouse')
        return False
    enemy, distance = nearby_enemy(game, unit)
    if enemy is None:
        return True
    unit.marchTestResult = 'pending'

    async def resolve():
        try:
            if await enemy_sighted_test(game, unit, enemy, distance):
                on_pass()
            elif hasattr(game, 'startTaskFunction'):
                game.startTaskFunction(game.taskLoopPathTowardsMouse, 'taskLoopPathTowardsMouse')
        finally:
            if unit.marchTestResult == 'pending':
                unit.marchTestResult = None

    game.taskMgr.add(resolve())
    return False