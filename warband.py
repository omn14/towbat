"""Warband Leadership and Charge rolls (Rulebook p. 180, amended June 2025)."""

from panda3d.core import Vec3

from psychology import _stat_int, combat_rank_bonus
from rules_log import rule_log, rule_skipped


def has_warband(unit):
    return any(rule.get('name', '').casefold() == 'warband'
               for rule in getattr(unit.unit.model, 'special_rules', []))


def majority(unit):
    count = unit.unit.nmodels
    eligible = count if has_warband(unit) else 0
    joined = getattr(unit, 'joinedCharacter', None)
    if joined is not None and joined.unit.nmodels > 0:
        count += 1
        eligible += int(has_warband(joined))
    return eligible > count / 2


def leadership_for_test(psychology, unit, kind):
    original, general = psychology.leadership_of(unit)
    if getattr(unit.unit.model, 'troop_type_rule', lambda name: False)('Undisciplined'):
        rule_skipped('Undisciplined', unit,
                     f'{kind}: cannot use Inspiring Presence or Hold Your Ground; own Ld {original} (pp. 191, 193)')
    joined = getattr(unit, 'joinedCharacter', None)
    sources = [unit] + ([joined] if joined is not None else [])
    if not any(has_warband(source) for source in sources):
        return original, general
    general = getattr(psychology, 'general_of', lambda member: general)(unit)
    if general is not None and all(general is not source for source in sources):
        sources.append(general)
    values = []
    for source in sources:
        value = _stat_int(source.unit.model.characteristics, 'Ld', 7)
        host = getattr(source, 'hostUnit', None) or source
        bonus = 0
        if (has_warband(source) and has_warband(host) and getattr(host, 'state', '') != 'IsFleeing'
            and kind not in ('Restraint', 'Impetuous')):
            bonus = combat_rank_bonus(host)
        values.append(min(10, value + bonus))
        if has_warband(source):
            logger = rule_log if bonus else rule_skipped
            logger('Warband', source, f'{kind}: Ld {value} + rank bonus {bonus} -> {values[-1]} '
                   '(no rank modifier when fleeing, restraining or testing Impetuous; p. 180)')
    best = max(values)
    return best, general if general is not None and sources[values.index(best)] is general else None


async def roll_charge(game, unit, bonus, roll):
    models, dice = await roll(3 if bonus else 2, bonus)
    if getattr(unit, 'state', '') == 'IsPursuing' or not majority(unit):
        return models, dice
    if game.aiControls(unit):
        choice = 'Re-roll' if max(dice) < 4 else 'Keep'
    else:
        choice = await game.makeChoiceNew(['Re-roll', 'Keep'], Vec3(0, 0, 10), owner=unit,
                                         prompt=f'{unit.unit.name}: Warband Charge roll {dice}')
    if choice != 'Re-roll':
        rule_skipped('Warband', unit, f'keeps Charge roll {dice}; no re-roll (p. 180)')
        return models, dice
    for die in models[:2]:
        die.remove(game.world)
    replacement_models, replacement = await roll(2, False)
    result = replacement + dice[2:]
    rule_log('Warband', unit, f'Charge dice {dice[:2]} -> {replacement}; '
             f'Swiftstride bonus {dice[2:]} unchanged (p. 180; FAQ v1.5.3)')
    return replacement_models + models[2:], result