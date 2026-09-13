"""Battle March secondary objectives (General's Companion pp. 36-37)."""

from copy import deepcopy
import re

from battle_config import ConfigError, _keys, _number
from characters import side_of
from psychology import unit_strength_total
from rules_log import rule_log, rule_skipped
from scouts import model_base_boxes
from spell_templates import circle_distance


def enabled(game, name):
    config = getattr(game, 'battle_config', None)
    return bool(config and name in config['optional_rules']['secondary_objectives'])


def empty_state():
    return {'raid_attempts': [], 'awards': [], 'resolved_starts': []}


def validate_state(config, record, objectives):
    _keys(record, 'raid_attempts awards resolved_starts', 'battle_march.secondary')
    markers = {objective['id']: objective for objective in objectives}
    for field in record:
        if not isinstance(record[field], list):
            raise ConfigError(f'battle_march.secondary.{field}: expected a list')
    if (record['raid_attempts'] or record['awards']) and 'raid_and_burn' not in config['optional_rules']['secondary_objectives']:
        raise ConfigError('battle_march.secondary: Raid & Burn is not enabled')

    def turn_key(value):
        if not isinstance(value, str) or not re.fullmatch(r'[12]:[0-9]+:[0-9]+', value):
            raise ConfigError('battle_march.secondary: invalid player-turn key')

    seen = set()
    for attempt in record['raid_attempts']:
        _keys(attempt, 'unit objective player started', 'battle_march.secondary.raid_attempts')
        _number(attempt['player'], 'battle_march.secondary.player', 1, 2, integer=True)
        turn_key(attempt['started'])
        if not isinstance(attempt['unit'], str) or not attempt['unit']:
            raise ConfigError('battle_march.secondary.unit: expected a unit ID')
        marker = markers.get(attempt['objective'])
        key = attempt['unit'], attempt['objective']
        if marker is None or marker['kind'] != 'trove' or key in seen:
            raise ConfigError('battle_march.secondary: unknown or duplicate trove attempt')
        if int(attempt['started'][0]) != attempt['player']:
            raise ConfigError('battle_march.secondary: attempt began in another player turn')
        seen.add(key)
    seen.clear()
    for award in record['awards']:
        _keys(award, 'unit objective player turn points rule reason', 'battle_march.secondary.awards')
        _number(award['player'], 'battle_march.secondary.player', 1, 2, integer=True)
        turn_key(award['turn'])
        marker = markers.get(award['objective'])
        if (marker is None or not marker['destroyed'] or award['objective'] in seen
                or award['rule'] != 'Raid & Burn' or type(award['points']) is not int or award['points'] != 30):
            raise ConfigError('battle_march.secondary: invalid or duplicate destruction award')
        if any(not isinstance(award[field], str) or not award[field] for field in ('unit', 'reason')):
            raise ConfigError('battle_march.secondary: award requires a unit and reason')
        seen.add(award['objective'])
    for turn in record['resolved_starts']:
        turn_key(turn)
    if len(set(record['resolved_starts'])) != len(record['resolved_starts']):
        raise ConfigError('battle_march.secondary: duplicate resolved start')
    if any(award['turn'] not in record['resolved_starts'] for award in record['awards']):
        raise ConfigError('battle_march.secondary: destruction award has no resolved Start of Turn')
    return deepcopy(record)


def all_awards(game):
    return [*getattr(game, 'battle_awards', []), *getattr(game, 'battle_secondary', {}).get('awards', [])]


def in_contact(unit, objective):
    return any(circle_distance(objective['center'], box) <= objective['diameter'] / 2 + 1e-5
               for box in model_base_boxes(unit))


def raiding(game, unit):
    host = getattr(unit, 'hostUnit', None) or unit
    return enabled(game, 'raid_and_burn') and any(
        attempt['unit'] == host.unitName for attempt in getattr(game, 'battle_secondary', {}).get('raid_attempts', []))


def shooting_blocked(game, unit):
    if not raiding(game, unit):
        return False
    rule_log('Raid & Burn', unit, 'destroying a treasure trove -> cannot shoot (Companion p. 36)')
    return True


def spell_allowed(game, unit, spell_range, *, log=False):
    allowed = not raiding(game, unit) or str(spell_range).casefold() in ('self', 'combat')
    if not allowed and log:
        rule_log('Raid & Burn', unit, f'destroying a treasure trove -> spell range {spell_range!r} is not Self or Combat')
    return allowed


def after_move(game, unit):
    """Contact during Remaining Moves starts destruction (Companion p. 36)."""
    from battle_objectives import turn_key
    if (not enabled(game, 'raid_and_burn') or getattr(game, 'restoringBattle', False)
            or game.fsm.state != 'MovementPhase' or getattr(game, 'chargeStage', None) != 'remaining'
            or getattr(unit, 'hostUnit', None) is not None
            or side_of(game, unit) != game.roundCounter.current_player):
        return
    state = getattr(game, 'battle_secondary', None)
    if state is None:
        state = game.battle_secondary = empty_state()
    strength = unit_strength_total(unit)
    for objective in game.battle_objectives:
        if objective['kind'] != 'trove' or objective['destroyed'] or not in_contact(unit, objective):
            continue
        if strength < 5 or unit.state == 'IsFleeing' or getattr(unit, 'isInCombat', False):
            rule_skipped('Raid & Burn', unit, f'{objective["id"]}: US {strength}, state {unit.state}; '
                         'requires US 5, not fleeing or in combat')
            continue
        if any(attempt['unit'] == unit.unitName and attempt['objective'] == objective['id'] for attempt in state['raid_attempts']):
            continue
        state['raid_attempts'].append({'unit': unit.unitName, 'objective': objective['id'],
                                       'player': side_of(game, unit), 'started': turn_key(game)})
        rule_log('Raid & Burn', unit, f'{objective["id"]}: base contact, US {strength}; destruction begins, '
                 'shooting barred and spells restricted to Self/Combat until the next own Start of Turn')


def start_turn(game):
    """Resolve each attempted destruction once at its next own Start of Turn."""
    from battle_objectives import sync_markers, turn_key
    if not enabled(game, 'raid_and_burn') or getattr(game, 'restoringBattle', False):
        return
    state = getattr(game, 'battle_secondary', None)
    if state is None:
        state = game.battle_secondary = empty_state()
    current = turn_key(game)
    if current in state['resolved_starts']:
        return
    units = {unit.unitName: unit for unit in game.units}
    objectives = {objective['id']: objective for objective in game.battle_objectives}
    remaining = []
    for attempt in state['raid_attempts']:
        if attempt['player'] != game.roundCounter.current_player or attempt['started'] == current:
            remaining.append(attempt)
            continue
        unit = units.get(attempt['unit'])
        objective = objectives.get(attempt['objective'])
        strength = unit_strength_total(unit) if unit is not None else 0
        reason = ('unit removed' if unit is None else
                  'trove already destroyed' if objective is None or objective['destroyed'] else
                  'no longer in base contact' if not in_contact(unit, objective) else
                  f'Unit Strength {strength} below 5' if strength < 5 else
                  'fleeing' if unit.state == 'IsFleeing' else
                  'engaged in combat' if getattr(unit, 'isInCombat', False) else None)
        if reason:
            rule_skipped('Raid & Burn', attempt['unit'], f'{attempt["objective"]}: {reason}; attempt ends, 0 VP')
            continue
        objective.update(destroyed=True, controller=None, player=None, contested=False)
        award = {'unit': attempt['unit'], 'objective': objective['id'], 'player': attempt['player'],
                 'turn': current, 'points': 30, 'rule': 'Raid & Burn',
                 'reason': f'base contact retained, US {strength}, not fleeing or engaged at next own Start of Turn'}
        state['awards'].append(award)
        rule_log('Raid & Burn', unit, f'{objective["id"]}: {award["reason"]}; destroyed, Player {attempt["player"]} +30 VP')
    state['raid_attempts'] = [attempt for attempt in remaining
                              if not objectives.get(attempt['objective'], {}).get('destroyed', True)]
    state['resolved_starts'].append(current)
    sync_markers(game)