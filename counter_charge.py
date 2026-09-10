"""Counter Charge eligibility and D3+1 movement (Rulebook p. 167; FAQ v1.5.3)."""


def has_counter_charge(unit):
    return any(isinstance(rule, dict) and rule.get('name') == 'Counter Charge'
               for rule in unit.unit.model.special_rules)


def unavailable_reason(defender, charger, *, distance, movement, flank, turn,
                       declared=True):
    """Use declaration-time distance/arc; the tentative contact is not the origin."""
    if not has_counter_charge(defender):
        return 'unit does not have Counter Charge'
    character = getattr(defender, 'joinedCharacter', None)
    if character is not None and not has_counter_charge(character):
        return f'joined {character.unit.name} does not have Counter Charge'
    if not declared:
        return 'pursuit is not a declared charge; no charge reaction'
    if defender.state == 'IsFleeing':
        return 'fleeing units cannot Counter Charge'
    if defender.state == 'InCombat' or getattr(defender, 'isInCombat', False):
        return 'already engaged in combat'
    if list(getattr(defender, 'counterChargeTurn', []) or []) == list(turn):
        return 'already used Counter Charge this turn; must Hold'
    troop_type = charger.unit.model.troop_type().lower()
    if not any(kind in troop_type for kind in ('cavalry', 'chariot', 'monster')):
        return f'charger troop type {charger.unit.model.troop_type()} is not cavalry, chariot or monster'
    if flank != 'front':
        return f'charge enters the {flank}, not the front arc'
    if distance + 1e-6 < movement:
        return f'charger is {distance:.2f}" away, less than its Movement {movement:g}"'
    return None


def counter_charge_distance(d6):
    """The D3+1 is not a Charge roll; no Swiftstride or charge rerolls (FAQ)."""
    if d6 not in range(1, 7):
        raise ValueError('Counter Charge requires one D6 result from 1 to 6')
    return (d6 + 1) // 2 + 1