"""Fear, Terror and Flaming fear of war beasts (Rulebook pp. 168, 169, 171, 179)."""

import random
from contextlib import contextmanager
from types import SimpleNamespace

from magic_items import current_turn
from psychology import leadership_passed, reroll_leadership, unit_strength_total
from rules_log import rule_log, rule_skipped


def has_rule(model, name):
    return any(rule.get('name', '').casefold() == name.casefold()
               for rule in getattr(model, 'special_rules', []) if isinstance(rule, dict))


def causes(model, name):
    mount = getattr(model, 'get_mount', lambda: None)()
    return has_rule(model, name) or bool(mount and has_rule(mount, name))


def strength(unit):
    joined = getattr(unit, 'joinedCharacter', None)
    return unit_strength_total(unit) + (unit_strength_total(joined) if joined is not None else 0)


def identity(unit):
    host = getattr(unit, 'command_host', None) or getattr(unit, 'hostUnit', None) or unit
    return getattr(host, 'unitName', host.unit.name)


def immune(unit):
    count = unit.unit.nmodels
    protected = count if has_rule(unit.unit.model, 'Immune to Psychology') else 0
    joined = getattr(unit, 'joinedCharacter', None)
    if joined is not None and joined.unit.nmodels > 0:
        count += 1
        protected += int(has_rule(joined.unit.model, 'Immune to Psychology'))
    return protected > count / 2


def model_fears(unit, source):
    """FAQ v1.5.3: Flaming fear overrides War Beast/Swarm immunity."""
    model = unit.unit.model
    vulnerable = str(model.characteristics.get('Troop Type', '')).casefold() in ('war beasts', 'swarms')
    if vulnerable and causes(source, 'Flaming Attacks'):
        return True
    if immune(unit) or causes(model, 'Terror'):
        return False
    if causes(source, 'Terror'):
        return True
    if causes(model, 'Fear'):
        return False
    return causes(source, 'Fear')


def feared_strength(unit, enemy):
    """Count only Fear-causing models, not their ordinary companions (FAQ v1.5.3)."""
    result = unit_strength_total(enemy) if model_fears(unit, enemy.unit.model) else 0
    joined = getattr(enemy, 'joinedCharacter', None)
    if joined is not None and model_fears(unit, joined.unit.model):
        result += unit_strength_total(joined)
    return result


def fears(unit, enemy):
    return feared_strength(unit, enemy) > 0


def cannot_flee(unit):
    from chaos_gifts import succumbed
    return (immune(unit) or succumbed(unit) or getattr(unit, 'isInCombat', False)
            or any(rule.get('Unbreakable') for rule in unit.unit.model.special_rules))


async def terror_test(game, unit, charger):
    """A Terror declaration tests immediately, unless Flee is unavailable (p. 179)."""
    if not causes(charger.unit.model, 'Terror'):
        return True
    if (cannot_flee(unit) or any(causes(unit.unit.model, name) for name in ('Fear', 'Terror'))
            or unit.state == 'IsFleeing'):
        rule_skipped('Terror', unit, 'immune, already fleeing or cannot choose Flee; no test (p. 179)')
        return True
    from warband import leadership_for_test
    leadership = leadership_for_test(game.psychology, unit, 'Terror')[0]
    async def roll():
        return [random.randint(1, 6), random.randint(1, 6)]
    dice = await reroll_leadership(game, unit, 'Terror', await roll(), leadership, roll,
                                  cause=f'charged by {charger.unit.name}')
    passed = leadership_passed(sum(dice), leadership)
    rule_log('Terror', unit, f'{charger.unit.name} declares charge: 2D6={dice} vs Ld {leadership} '
             + ('passed; normal reaction' if passed else 'failed; must Flee (p. 179)'))
    return passed


async def test_fear(game, unit, enemies, context):
    own = strength(unit)
    threats = [enemy for enemy in enemies if feared_strength(unit, enemy) > own]
    if not threats:
        for enemy in enemies:
            if any(causes(enemy.unit.model, name) for name in ('Fear', 'Terror', 'Flaming Attacks')):
                rule_skipped('Fear', unit, f'{context}: immune or enemy {enemy.unit.name} '
                             f'Fear-causing US {feared_strength(unit, enemy)} is not greater than own US {own}')
        return True
    token = current_turn(game)
    if getattr(unit, 'fearTestTurn', None) == token:
        unit.fearTargets = list(dict.fromkeys([*getattr(unit, 'fearTargets', []),
                                               *(identity(enemy) for enemy in threats)]))
        return not getattr(unit, 'fearFailed', False)
    from warband import leadership_for_test
    leadership = leadership_for_test(game.psychology, unit, 'Fear')[0]
    async def roll():
        return [random.randint(1, 6), random.randint(1, 6)]
    dice = await reroll_leadership(game, unit, 'Fear', await roll(), leadership, roll, cause=context)
    unit.fearTestTurn = token
    unit.fearFailed = not leadership_passed(sum(dice), leadership)
    unit.fearTargets = [identity(enemy) for enemy in threats]
    names = ', '.join(f'{enemy.unit.name} Fear-causing US {feared_strength(unit, enemy)}' for enemy in threats)
    rule_log('Fear', unit, f'{context}: own US {own}, {names}; 2D6={dice} vs Ld {leadership}: '
             + ('failed; no charge / -1 To Hit feared enemies this turn' if unit.fearFailed else 'passed'))
    return not unit.fearFailed


test_fear.__test__ = False


@contextmanager
def attack_penalty(game, host, target, profile):
    """Apply -1 only to attacks directed at a feared enemy, including rerolls (p. 168)."""
    bearer = getattr(host, 'command_host', None) or getattr(host, 'hostUnit', None) or host
    attacker = SimpleNamespace(unit=SimpleNamespace(model=profile, nmodels=1))
    active = (getattr(bearer, 'fearFailed', False)
              and getattr(bearer, 'fearTestTurn', None) == current_turn(game)
              and identity(target) in getattr(bearer, 'fearTargets', [])
              and fears(attacker, target))
    previous = profile.special_rules
    if active:
        profile.special_rules = [*previous, {'name': 'Fear penalty', 'to_hit': lambda roll, model: roll - 1}]
        rule_log('Fear', bearer, f'{profile.name} directs attacks against {target.unit.name}: -1 To Hit (p. 168)')
    try:
        yield
    finally:
        profile.special_rules = previous