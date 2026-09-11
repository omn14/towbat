"""Drilled free redress before moving (Rulebook pp. 125, 167)."""

from panda3d.core import Vec3

from rules_log import rule_skipped


def has_drilled(unit):
    host = getattr(unit, 'hostUnit', None) or unit
    return any(isinstance(rule, dict) and rule.get('name') == 'Drilled'
               for rule in host.unit.model.special_rules)


def marching_column(unit):
    """A formed unit deeper in models than it is wide (Rulebook p. 101)."""
    return (not getattr(unit, 'isSkirmisher', False)
            and unit.unit.ranks > unit.unit.files)


def march_multiplier(unit):
    """Marching Column triples Movement instead of doubling it (p. 101)."""
    return 3 if marching_column(unit) else 2


def move_pending(game):
    return any(getattr(unit, '_drilledMoveActive', False) is True
               for unit in getattr(game, 'units', []))


async def before_move(game, unit, context, *, compulsory=False):
    """One free redress immediately before a move (p. 167; FAQ v1.5.3).

    The FAQ includes Counter Charge and Giving Ground, and obliges a Drilled
    compulsory charger to leave Marching Column when a legal redress fits.
    """
    if not has_drilled(unit):
        return False
    if unit.state == 'IsFleeing':
        rule_skipped('Drilled', unit, f'{context}: fleeing; no free redress')
        return False
    files = unit.unit.files
    choices = {}
    for delta in range(-5, 6):
        if not delta or not 1 <= files + delta <= unit.unit.nmodels:
            continue
        if game.movement.redressRanks(unit, delta, drilled=True, preview=True):
            choices[f'{files + delta} files'] = delta
    required = compulsory and marching_column(unit)
    if required:
        choices = {label: delta for label, delta in choices.items()
                   if -(-unit.unit.nmodels // (files + delta)) <= files + delta}
    if not choices:
        rule_skipped('Drilled', unit, f'{context}: no legal '
                     f'{"Combat Order " if required else ""}redress fits from {files} files')
        return False
    options = list(choices) if required else ['Keep formation', *choices]
    if game.aiControls(unit):
        selected = min(choices, key=lambda label: abs(choices[label])) if required else options[0]
    else:
        selected = await game.makeChoiceNew(
            options, Vec3(0, 0, 10), owner=unit,
            prompt=f'{unit.unit.name}: Drilled before {context}',
            detail=f'{files} files; free redress of up to 5 models')
    if selected not in choices:
        if required:
            selected = min(choices, key=lambda label: abs(choices[label]))
        else:
            rule_skipped('Drilled', unit, f'{context}: keeps {files} files')
            return False
    return game.movement.redressRanks(unit, choices[selected], drilled=True)