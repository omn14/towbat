"""Live, utility-ranked candidates; legality stays with the engine's queries."""

from dataclasses import dataclass
import math

from characters import side_of
from gameStateTree import GameAction
from models import stat_int


@dataclass(frozen=True)
class Candidate:
    action: GameAction
    score: float
    reason: str


def action_key(action):
    return action.action_type, action.unit_name, tuple(sorted(action.parameters.items()))


def living_units(game, player):
    return [unit for unit in game.units
            if side_of(game, unit, None) == player and unit.unit.nmodels > 0
            and not unit.bodyNP.isEmpty() and unit.isDeployed
            and getattr(unit, 'hostUnit', None) is None]


def material(unit):
    return unit.unit.nmodels * max(1, stat_int(unit.unit.model.characteristics, 'Points', 10))


def fighting_power(unit):
    profile = unit.unit.model.characteristics
    return (min(unit.unit.nmodels, max(1, unit.unit.files))
            * max(1, stat_int(profile, 'A', 1))
            * (stat_int(profile, 'WS', 3) + stat_int(profile, 'S', 3)))


def charge_candidates(game, friendlies):
    from impetuous import legal_targets
    options = [(unit, target, route) for unit in friendlies
               for target, route, target_index in legal_targets(game, unit)]
    result = []
    for unit, target, route in options:
        supporters = {id(ally): ally for ally, defender, other_route in options
                      if ally is not unit and defender is target}
        supporters.update({id(entry.charger): entry.charger
                           for entry in getattr(game, 'chargeDeclarations', [])
                           if entry.charger is not unit and entry.defender is target})
        supporters.update({id(ally): ally for ally in friendlies
                           if ally is not unit and target in ally.isInCombatWith})
        support = sum(fighting_power(ally) * 0.5 for ally in supporters.values())
        own = fighting_power(unit)
        enemy = max(1, fighting_power(target))
        advantage = (own + support) / enemy
        if advantage < 0.65:
            continue
        score = material(target) * min(2, advantage) - material(unit) / max(0.5, advantage) * 0.35
        score -= route.distance * 2
        if score > 0:
            result.append(Candidate(GameAction('charge', unit.unitName, {'target': target.unitName}),
                                    score, f'combat utility ratio {advantage:.2f}, route {route.distance:.1f}, support {support:.1f}'))
    return result


def shooting_candidates(game, friendlies, enemies):
    from shooting_geometry import shooting_solution
    from battle_secondary import shooting_blocked
    from chaos_gifts import succumbed
    result = []
    for unit in friendlies:
        if (unit.hasAttackedThisTurn or unit.isInCombat or unit.state == 'IsFleeing'
                or getattr(unit, 'chargedThisTurn', False) or succumbed(unit)
                or shooting_blocked(game, unit)):
            continue
        cannon = getattr(game, 'cannon', None)
        bombard = getattr(game, 'bombard', None)
        weapon = cannon.cannon_weapon(unit) if cannon else None
        action_type = 'cannon' if weapon else 'shoot'
        if weapon is None and bombard:
            weapon = bombard.bombardment_weapon(unit)
            if weapon:
                action_type = 'bombard'
        weapons = [(None, weapon)] if weapon else [(key, missile) for key, missile in unit.unit.model.weapons.items()
                                                   if missile.get('tag') == 'ranged']
        for weapon_key, missile in weapons:
            for target in enemies:
                if target.isInCombat:
                    continue
                solution = shooting_solution(game, unit, target, weapon=missile)
                if not solution.eligible:
                    continue
                distance = (unit.bodyNP.getPos() - target.bodyNP.getPos()).length()
                if weapon and not weapon.get('ranged_range_min', 0) <= distance <= weapon.get('ranged_range', 0):
                    continue
                score = len(solution.eligible) * math.sqrt(material(target)) / max(1, distance / 12)
                parameters = {'target': target.unitName}
                if weapon_key is not None:
                    parameters['weapon'] = weapon_key
                result.append(Candidate(GameAction(action_type, unit.unitName, parameters),
                                        score, f'{len(solution.eligible)} legal firing models at {distance:.1f}'))
    return result


def movement_candidates(game, player, friendlies, enemies):
    from chaos_gifts import succumbed
    from reserve_move import in_reserve, unavailable
    reserve = in_reserve(game)
    objectives = [objective for objective in getattr(game, 'battle_objectives', [])
                  if not objective.get('destroyed')]
    if objectives:
        from battle_objectives import control_snapshot, required_choices, resolve_control
        snapshots = control_snapshot(game)
        choices = {}
        while pending := required_choices(snapshots, choices):
            for controller, options in pending.items():
                choices[controller] = options[0]
        objectives = resolve_control(snapshots, choices)
    result = []
    for unit in friendlies:
        if (unit.hasMovedThisTurn or unit.isInCombat or unit.state != 'Idle'
                or getattr(unit, 'joinedMovementLocked', False) or succumbed(unit)
                or (reserve and unavailable(game, unit))):
            continue
        origin = unit.bodyNP.getPos()
        allowance = game.movement.movementAllowance(unit, features=[])
        if allowance <= 0:
            continue
        goals = []
        for objective in objectives:
            from psychology import unit_strength_total
            if unit_strength_total(unit) < game.battle_config['objectives']['minimum_unit_strength']:
                continue
            center = objective['center']
            distance = math.hypot(center[0] - origin.x, center[1] - origin.y)
            if objective.get('controller') == unit.unitName and not objective.get('contested'):
                goals = []
                break
            if distance > 1:
                scoring = game.battle_config['scoring']
                reward = scoring['landmark_per_player_turn' if objective['kind'] == 'landmark'
                                 else 'trove_per_player_turn']
                goals.append((center[0], center[1], (100 + reward * 5) / (1 + distance / allowance), 'objective approach'))
        else:
            if enemies:
                target = min(enemies, key=lambda enemy: (enemy.bodyNP.getPos() - origin).length())
                position = target.bodyNP.getPos()
                distance = (position - origin).length()
                ratio = fighting_power(unit) / max(1, fighting_power(target))
                shooting_position = bool(shooting_candidates(game, [unit], enemies))
                if not shooting_position and distance > 4:
                    direction = -1 if ratio < 0.5 and distance < allowance * 3 else 1
                    goals.append((origin.x + (position.x - origin.x) * direction,
                                  origin.y + (position.y - origin.y) * direction,
                                  20 if direction > 0 else 30, 'support approach' if direction > 0 else 'avoid isolated fight'))
        for target_x, target_y, score, reason in goals[:3]:
            delta_x, delta_y = target_x - origin.x, target_y - origin.y
            distance = math.hypot(delta_x, delta_y)
            if distance < 0.1:
                continue
            for fraction in (1.0, 0.5):
                travel = min(allowance * fraction, max(0, distance - 2))
                if travel < 0.1:
                    continue
                destination = {'target_x': origin.x + delta_x / distance * travel,
                               'target_y': origin.y + delta_y / distance * travel}
                result.append(Candidate(GameAction('move', unit.unitName, destination), score * fraction, reason))
    return result


def joining_candidates(game, friendlies):
    from characters import is_character, join_reason, remaining_move_reason
    from character_movement import preview
    result = []
    for character in friendlies:
        if not is_character(character) or remaining_move_reason(game, character):
            continue
        for host in friendlies:
            if join_reason(game, character, host, movement=True) is not None:
                continue
            movement = preview(game, character, host=host)
            if not movement.error and not movement.marched:
                result.append(Candidate(GameAction('join', character.unitName, {'target': host.unitName}),
                                        80 - movement.distance, 'legal escort contact move'))
    return result


def formation_candidates(game, friendlies):
    from drilled import marching_column
    result = []
    for unit in friendlies:
        if unit.hasMovedThisTurn or unit.isInCombat or unit.state != 'Idle' or not marching_column(unit):
            continue
        delta = min(5, math.ceil(math.sqrt(unit.unit.nmodels)) - unit.unit.files)
        if delta > 0 and game.movement.redressRanks(unit, delta, preview=True):
            result.append(Candidate(GameAction('redress', unit.unitName, {'delta': delta}),
                                    100, 'leave Marching Column using legal redress'))
    return result


def departure_candidates(game, player, enemies):
    from character_movement import preview
    from characters import leave_reason
    result = []
    for character in game.units:
        host = getattr(character, 'hostUnit', None)
        if (host is None or side_of(game, character, None) != player or host.unit.nmodels > 2
                or not enemies or leave_reason(game, character)):
            continue
        origin = character.bodyNP.getPos(game.render)
        threat = min(enemies, key=lambda enemy: (enemy.bodyNP.getPos(game.render) - origin).length())
        away = origin - threat.bodyNP.getPos(game.render)
        away.z = 0
        distance = away.length()
        if not 0 < distance < 12:
            continue
        destination = origin + away.normalized() * min(4, character.unit.model.get_movement(0))
        movement = preview(game, character, destination=tuple(destination))
        if not movement.error and not movement.marched:
            result.append(Candidate(GameAction('leave', character.unitName,
                {'target_x': destination.x, 'target_y': destination.y}), 90,
                'legal retreat from a depleted escort under threat'))
    return result


def use_scarce_magic(game, member, kind):
    """Save a single-use bonus for contested positions with a useful dice threshold."""
    spell = getattr(getattr(game, 'fsm', None), 'spellInstanceToCast', None)
    if spell is None:
        return False
    threshold = getattr(spell, 'casting_value' if kind == 'Casting' else 'casting', 0)
    needed = threshold - member.unit.model.wizard_level(0) + (kind == 'Dispel')
    if not 5 <= needed <= 11:
        return False
    host = getattr(member, 'hostUnit', None) or member
    if host.isInCombat:
        return True
    player = side_of(game, member, None)
    return any((enemy.bodyNP.getPos(game.render) - host.bodyNP.getPos(game.render)).length() <= 18
               for enemy in living_units(game, 3 - player))


def spell_candidates(game, player):
    from magic_items import casting_spellbook, item_target_protected
    from spell_system import build_spell
    from high_magic import visible_spell_target
    from psychology import obb_distance
    from panda3d.core import Point3
    available = getattr(game, 'castableSpells', None)
    if available is None:
        return []
    result = []
    members = [unit for unit in game.units if unit.isDeployed and unit.unit.nmodels > 0
               and not unit.bodyNP.isEmpty()]
    for caster in members:
        if side_of(game, caster, None) != player:
            continue
        book = casting_spellbook(caster)
        for key in available(caster):
            spell = build_spell(game, caster, key, allow_catalogue=False)
            if spell is None or book[key].get('type') == 'Assailment':
                continue
            friendly = book[key].get('type') in ('Enchantment', 'Conveyance')
            own = spell.targets_self or str(spell.spell_range).casefold() == 'self'
            targets = [caster] if own else [unit for unit in members
                       if (side_of(game, unit, None) == player) == friendly
                       and getattr(unit, 'hostUnit', None) is None]
            for target in targets:
                if item_target_protected(game, caster, target):
                    continue
                destination = Point3(target.bodyNP.getPos(game.render)) if spell.targets_ground else target
                if not own:
                    reach = spell.spell_range
                    if not isinstance(reach, (int, float)):
                        continue
                    distance = obb_distance(game.psychology._unit_box(caster), game.psychology._unit_box(target))
                    if distance > reach or (not friendly and target.isInCombat):
                        continue
                    if not visible_spell_target(game, caster, target):
                        continue
                if not spell.canTarget(destination):
                    continue
                value = material(target)
                if book[key].get('scroll_item_id') and value < material(caster) * 1.5:
                    continue
                score = 30 + math.sqrt(value) - spell.casting_value
                parameters = {'spell': key, 'target': target.unitName}
                if spell.targets_ground:
                    parameters.update(target_x=destination.x, target_y=destination.y)
                result.append(Candidate(GameAction('cast', caster.unitName, parameters),
                                        score, f'legal {key} target; casting value {spell.casting_value}'))
    return result


def candidates(game, player):
    friendlies = living_units(game, player)
    enemies = living_units(game, 3 - player)
    phase = game.fsm.state
    spells = spell_candidates(game, player) if phase in ('StrategyPhase', 'MovementPhase', 'ShootingPhase') else []
    if phase == 'MovementPhase' and getattr(game, 'chargeStage', None) == 'declarations':
        return charge_candidates(game, friendlies)
    if phase in ('MovementPhase', 'ReserveMovePhase'):
        joining = joining_candidates(game, friendlies) if phase == 'MovementPhase' else []
        manoeuvres = (formation_candidates(game, friendlies) + departure_candidates(game, player, enemies)
                      if phase == 'MovementPhase' else [])
        return spells + joining + manoeuvres + movement_candidates(game, player, friendlies, enemies)
    if phase == 'StrategyPhase' and getattr(game, 'strategyCommandDone', True):
        return spells + [Candidate(GameAction('rally', unit.unitName), material(unit), 'normal rally attempt')
                for unit in friendlies if unit.state == 'IsFleeing'
                and not getattr(unit, 'attemptedRallyThisTurn', False)]
    if phase == 'ShootingPhase':
        return spells + shooting_candidates(game, friendlies, enemies)
    if phase == 'CombatPhase':
        return [Candidate(GameAction('attack', unit.unitName, {'target': target.unitName}),
                          material(target), 'unresolved connected combat')
                for unit in friendlies if unit.isInCombat and not unit.hasAttackedThisTurn
                for target in unit.isInCombatWith if target in enemies]
    return []


def choose(game, player, rejected=()):
    ranked = [candidate for candidate in candidates(game, player)
              if action_key(candidate.action) not in rejected]
    if not ranked:
        return GameAction('end_phase', 'system')
    selected = max(ranked, key=lambda candidate: candidate.score)
    from rules_log import battle_log
    battle_log(f'AI P{player}: {selected.action}; utility {selected.score:.1f}; {selected.reason}', 'info')
    return selected.action