"""First charge attempts and source-owned disruption (Rulebook pp. 101, 169)."""

from rules_log import rule_log, rule_skipped


def has_first_charge(unit):
    return any(rule.get('name') == 'First Charge' for rule in getattr(unit.unit.model, 'special_rules', [])
               if isinstance(rule, dict))


def begin_charge_attempt(unit):
    """A redirect continues the declared attempt; failure still spends it (p. 169)."""
    if getattr(unit, 'chargeAttemptPending', False):
        return
    unit.chargeAttempts = getattr(unit, 'chargeAttempts', 0) + 1
    unit.chargeAttemptPending = True
    unit.firstChargePending = unit.chargeAttempts == 1 and has_first_charge(unit)


def finish_charge_attempt(unit, target=None, *, next_turn=False):
    """Contact disrupts for the counted charge's turn (pp. 157, 169; FAQ v1.5.3)."""
    if not getattr(unit, 'chargeAttemptPending', False):
        return
    first = getattr(unit, 'firstChargePending', False)
    unit.chargeAttemptPending = unit.firstChargePending = False
    if not has_first_charge(unit):
        return
    if not first:
        rule_skipped('First Charge', unit,
                     f'charge attempt {unit.chargeAttempts}; the first attempt has already been spent')
    elif target is None:
        rule_skipped('First Charge', unit, 'first charge made no contact; benefit spent for the game')
    else:
        field = 'firstChargeDisruptedNextTurnBy' if next_turn else 'firstChargeDisruptedBy'
        sources = list(getattr(target, field, []))
        name = unit.unit.name
        if name not in sources:
            sources.append(name)
        setattr(target, field, sources)
        timing = "next turn's Combat phase" if next_turn else 'Combat phase'
        rule_log('First Charge', unit,
                 f'first charge contacted {target.unit.name}; target Disrupted '
                 f'{"next turn, " if next_turn else ""}until end of {timing} (p. 169)')


def count_as_charge(unit, target, *, next_turn=False):
    """A successful pursuit counts as charging when fought (p. 157; FAQ v1.5.3)."""
    begin_charge_attempt(unit)
    finish_charge_attempt(unit, target, next_turn=next_turn)


def expire_first_charge(unit):
    """Do not clear terrain or flank disruption with this source (pp. 101, 169)."""
    sources = getattr(unit, 'firstChargeDisruptedBy', [])
    if sources:
        rule_log('First Charge', unit,
                 f'end of Combat phase: disruption from {", ".join(sources)} expires; other sources unchanged')
    unit.firstChargeDisruptedBy = list(getattr(unit, 'firstChargeDisruptedNextTurnBy', []))
    unit.firstChargeDisruptedNextTurnBy = []
    if unit.firstChargeDisruptedBy:
        rule_log('First Charge', unit,
                 f'next turn begins: pursuit by {", ".join(unit.firstChargeDisruptedBy)} '
                 'counts as charging; Disrupted until the next Combat phase ends')