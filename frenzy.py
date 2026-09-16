"""Frenzy eligibility and loss (amended Rulebook p. 170)."""

from rules_log import rule_log
from characters import get_joined_characters


def has_frenzy(profile):
    return any(not rule.get('frenzy_lost') and
               (rule.get('frenzy') or rule.get('name', '').casefold() == 'frenzy')
               for rule in profile.special_rules if isinstance(rule, dict))


def model_frenzied(profile):
    if has_frenzy(profile):
        return True
    for tag in ('mount', 'crew', 'beasts'):
        part = getattr(profile, f'get_{tag}', lambda: None)()
        if part is not None and has_frenzy(part):
            return True
    return False


def counts(unit):
    unit = getattr(unit, 'hostUnit', None) or unit
    total = max(0, unit.unit.nmodels)
    frenzied = total if model_frenzied(unit.unit.model) else 0
    for joined in get_joined_characters(unit):
        total += max(0, joined.unit.nmodels)
        if model_frenzied(joined.unit.model):
            frenzied += max(0, joined.unit.nmodels)
    return frenzied, total


def majority(unit):
    frenzied, total = counts(unit)
    return frenzied > total / 2


def attack_bonus(part):
    """The bonus belongs to the fighting element specified on p. 170, not every part."""
    from troop_types import normalise
    host = part.host
    if not (getattr(host, 'chargedThisTurn', False) or getattr(host, 'frenzyFollowUpThisTurn', False)):
        return 0
    owner = ((getattr(part, 'character', None) or getattr(host, 'joinedCharacter', None))
             if part.role == 'character' else host)
    if owner is None or not model_frenzied(owner.unit.model):
        return 0
    profile = owner.unit.model
    mount = getattr(profile, 'get_mount', lambda: None)()
    troop = normalise((mount or profile).characteristics.get('Troop Type'))
    if troop in ('monstrous creature', 'behemoth'):
        return int(part.profile is (mount or profile))
    crew = getattr(profile, 'get_crew', lambda: None)()
    if troop in ('light chariot', 'heavy chariot'):
        return int(part.profile is (crew or profile))
    return int(part.profile is profile or (part.role == 'champion' and owner is host))


def lose_frenzy(unit):
    """Mark each source lost, retaining unrelated and permanent rules for persistence."""
    pending = [unit.unit.model, *getattr(unit.unit, 'command_models', {}).values()]
    for joined in get_joined_characters(unit):
        pending.append(joined.unit.model)
    changed, seen = [], set()
    while pending:
        profile = pending.pop()
        if id(profile) in seen:
            continue
        seen.add(id(profile))
        for tag in ('mount', 'crew', 'beasts'):
            part = getattr(profile, f'get_{tag}', lambda: None)()
            if part is not None:
                pending.append(part)
        for rule in profile.special_rules:
            if isinstance(rule, dict) and not rule.get('frenzy_lost') and (
                    rule.get('frenzy') or rule.get('name', '').casefold() == 'frenzy'):
                rule['frenzy_lost'] = True
                changed.append(getattr(profile, 'name', unit.unit.name))
    if changed:
        rule_log('Frenzy', unit, f'lost this combat round: {", ".join(changed)} lose Frenzy immediately (p. 170)')