"""Standard battle Victory Points, not tactical assessment (amended Rulebook p. 286)."""

from math import ceil, isfinite


def unit_award(points, starting_strength, strength, starting_wounds, wounds, *, fleeing=False, absent=False):
    """Destroyed units give full points; fleeing/quarter strength gives half once."""
    if absent or strength <= 0:
        return points, 'destroyed or fled off the battlefield'
    if fleeing:
        return ceil(points / 2), 'fleeing at battle end'
    starting, remaining = ((starting_wounds, wounds) if starting_strength == starting_wounds
                           else (starting_strength, strength))
    if remaining * 4 <= starting:
        return ceil(points / 2), '25% or less of starting Wounds/Unit Strength'
    return 0, 'above 25% and not fleeing'


def outcome(scores):
    """A win needs a 100-point lead; double the opposing score is crushing (p. 286)."""
    if abs(scores[0] - scores[1]) < 100:
        return None, 'Draw'
    winner = 1 if scores[0] > scores[1] else 2
    crushing = scores[winner - 1] >= scores[2 - winner] * 2
    return winner, 'Crushing victory' if crushing else 'Victory'


def register_army(game, units, player):
    """Freeze paid costs and starting characteristics before casualties (p. 286)."""
    ledger = {name: record for name, record in getattr(game, 'victoryRoster', {}).items()
              if record['player'] != player}
    for unit in units:
        profile = unit.unit.model
        points = getattr(unit.unit, 'roster_metadata', {}).get('points_cost')
        points = float(points) if isinstance(points, (int, float)) and isfinite(points) and points >= 0 else None
        models = getattr(unit, 'startOfBattleModels', unit.unit.nmodels)
        ledger[unit.unitName] = {
            'player': player, 'name': unit.unit.name, 'points': points,
            'strength': models * profile.unit_strength(),
            'wounds': models * profile.starting_wounds(),
            'general': bool(getattr(unit, 'isGeneral', False)),
            'bsb': bool(getattr(unit, 'isBSB', False)),
        }
    game.victoryRoster = ledger
    game.victoryLedgerComplete = {record['player'] for record in ledger.values()} == {1, 2}
    game.battleResult = None


def calculate(game, *, log=False):
    """Score each original regiment/character once, including removed models (p. 286)."""
    from rules_log import rule_log, rule_skipped
    ledger = getattr(game, 'victoryRoster', {})
    live = {unit.unitName: unit for unit in game.units if not unit.bodyNP.isEmpty()}
    scores, rows, missing = [0, 0], [], []

    def award(player, name, rule, points, reason):
        rows.append({'player': player, 'unit': name, 'rule': rule, 'points': points, 'reason': reason})
        scores[player - 1] += points
        if log:
            logger = rule_log if points else rule_skipped
            logger(rule, name, f'{reason} -> Player {player} +{points:g} VP (p. 286)')

    for identity, record in ledger.items():
        member = live.get(identity)
        absent = member is None or member.unit.nmodels <= 0
        host = (getattr(member, 'hostUnit', None) or member) if member else None
        fleeing = not absent and getattr(host, 'state', '') == 'IsFleeing'
        strength = 0 if absent else member.unit.nmodels * member.unit.model.unit_strength()
        wounds = 0 if absent else member.unit.nmodels * member.unit.model.starting_wounds() - getattr(member, 'woundsOnModel', 0)
        player = 3 - record['player']
        if record['points'] is None:
            missing.append(identity)
        else:
            points, reason = unit_award(record['points'], record['strength'], strength,
                                        record['wounds'], wounds, fleeing=fleeing, absent=absent)
            award(player, record['name'], 'Dead or Fled', points, reason)
        if record['general']:
            award(player, record['name'], 'The King is Dead', 100 if absent or fleeing else 0,
                  'General destroyed/offboard/fleeing' if absent or fleeing else 'General still stands')
        if record['bsb']:
            award(player, record['name'], 'Trophies of War', 50 if absent or fleeing else 0,
                  'Battle Standard Bearer destroyed/offboard/fleeing' if absent or fleeing else 'Battle Standard Bearer still stands')
    seen = set()
    for trophy in getattr(game, 'capturedStandards', []):
        key = trophy['unit'], trophy.get('selection_ref'), trophy['captured_by']
        if key in seen or key[2] not in (1, 2):
            continue
        seen.add(key)
        award(key[2], key[0], 'Trophies of War', 50, 'captured enemy unit standard')
    complete = bool(ledger) and getattr(game, 'victoryLedgerComplete', False) and not missing
    winner, result = outcome(scores) if complete else (None, 'Incomplete scoring data')
    return {'scores': scores, 'winner': winner, 'outcome': result, 'rows': rows,
            'complete': complete, 'missing_costs': missing}