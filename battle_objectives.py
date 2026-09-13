"""Battle March control and turn awards (General's Companion pp. 25, 27).

The Matched Play Guide FAQ v1.5.3 limits each unit to one controlled object.
Control snapshots are read-only; scoring commits only after required choices.
"""

from collections import defaultdict
from copy import deepcopy

from characters import side_of
from chaos_gifts import succumbed
from psychology import unit_strength_total
from rules_log import rule_log, rule_skipped
from scouts import model_base_boxes
from spell_templates import circle_distance


TOLERANCE = 1e-5


def sync_markers(game):
    """Reconcile derived marker terrain without rerolls or duplicate pieces."""
    from panda3d.core import Point3, TextNode
    preparation = (getattr(game, 'battle_setup', None) or {}).get('preparation', {})
    objectives = ([] if preparation.get('stage') in ('armies', 'terrain', 'objectives')
                  else getattr(game, 'battle_objectives', []))
    hud = getattr(game, 'hud', None)
    if hud is not None:
        from battle_secondary import all_awards
        hud.set_objectives(objectives, all_awards(game))
    manager = getattr(game, 'terrain_manager', None)
    if manager is None:
        return
    existing = {}
    for piece in list(manager.terrain_pieces):
        identity = getattr(piece, 'objective_id', None)
        if identity is not None:
            if identity in existing:
                manager.remove_terrain(piece)
            else:
                existing[identity] = piece
    for objective in objectives:
        if objective['destroyed']:
            continue
        kind = 'landmark' if objective['kind'] == 'landmark' else 'treasure_trove'
        piece = existing.pop(objective['id'], None)
        center = Point3(*objective['center'], 0)
        if piece is not None and (piece.terrain_type != kind or piece.center != center
                                  or piece.width != objective['diameter']):
            manager.remove_terrain(piece)
            piece = None
        if piece is None:
            piece = manager.add_terrain(kind, center, objective['diameter'], objective['diameter'])
            piece.objective_id = objective['id']
        color = ((1, .6, .2, 1) if objective['contested'] else
                 (.3, .85, 1, 1) if objective['player'] == 1 else
                 (1, .35, .3, 1) if objective['player'] == 2 else
                 (1, .76, .15, 1) if kind == 'treasure_trove' else (1, 1, 1, 1))
        piece.visual.setColorScale(*color)
        if piece.visual.find('objective-number').isEmpty():
            label = TextNode('objective-number')
            label.setText(objective['id'].removeprefix('objective-'))
            label.setAlign(TextNode.ACenter)
            label.setTextColor(.05, .05, .05, 1)
            node = piece.visual.attachNewNode(label)
            node.setP(-90)
            node.setScale(.65 if kind == 'landmark' else .6)
            node.setPos(0, -.22, 3.42 if kind == 'landmark' else .18)
    for piece in existing.values():
        manager.remove_terrain(piece)


def landmark_profiles(unit):
    pending = [unit.unit.model, *getattr(unit.unit, 'command_models', {}).values()]
    joined = getattr(unit, 'joinedCharacter', None)
    if joined is not None:
        pending.append(joined.unit.model)
    seen = set()
    for profile in pending:
        if id(profile) in seen:
            continue
        seen.add(id(profile))
        yield profile
        for tag in ('mount', 'crew', 'beasts'):
            part = getattr(profile, f'get_{tag}', lambda: None)()
            if part is not None:
                pending.append(part)


def refresh_landmark_grants(game, results, turn):
    """Expire only source-owned grants at the next turn end (Companion p. 25)."""
    for unit in game.units:
        removed = set()
        for profile in landmark_profiles(unit):
            grants = [rule for rule in profile.special_rules if isinstance(rule, dict)
                      and rule.get('battle_march_source')]
            removed.update(rule['name'] for rule in grants)
            profile.special_rules = [rule for rule in profile.special_rules
                                     if not (isinstance(rule, dict) and rule.get('battle_march_source'))]
        if removed:
            rule_log('Strategic Landmarks', unit,
                     f'{", ".join(sorted(removed))} expires at turn {turn}; permanent sources retained')
    units = {unit.unitName: unit for unit in game.units}
    properties = {
        'magic_resistance': {'name': 'Magic Resistance (-2)', 'magic_resistance': -2},
        'frenzy': {'name': 'Frenzy', 'frenzy': True},
        'stubborn': {'name': 'Stubborn', 'stubborn': True},
    }
    for objective in results:
        if objective['kind'] != 'landmark' or objective['controller'] is None:
            continue
        unit = units[objective['controller']]
        grant = {**properties[objective['property']], 'battle_march_source': objective['id'],
                 'battle_march_turn': turn}
        for profile in landmark_profiles(unit):
            profile.special_rules.append(dict(grant))
        rule_log('Strategic Landmarks', unit,
                 f'{objective["id"]}: gains {grant["name"]} until the end of the next player turn')


def control_snapshot(game):
    config = game.battle_config['objectives']
    snapshots = []
    for objective in game.battle_objectives:
        contenders = []
        for unit in game.units:
            if getattr(unit, 'hostUnit', None) is not None:
                continue
            boxes = model_base_boxes(unit)
            if not boxes:
                continue
            joined = getattr(unit, 'joinedCharacter', None)
            strength = unit_strength_total(unit) + (unit_strength_total(joined) if joined else 0)
            distance = max(0, min(circle_distance(objective['center'], box) for box in boxes)
                           - objective['diameter'] / 2)
            reason = None
            if objective.get('destroyed'):
                reason = 'objective destroyed'
            elif not getattr(unit, 'isDeployed', False):
                reason = 'not deployed'
            elif unit.state == 'IsFleeing':
                reason = 'fleeing'
            elif succumbed(unit):
                reason = 'succumbed to Stupidity'
            elif strength < config['minimum_unit_strength']:
                reason = f'Unit Strength {strength} below {config["minimum_unit_strength"]}'
            elif distance > config['control_distance'] + TOLERANCE:
                reason = f'{distance:.3f}" exceeds {config["control_distance"]:g}" control distance'
            player = side_of(game, unit, default=None)
            if player not in (1, 2):
                reason = 'no owning player'
            contenders.append({'unit': unit.unitName, 'name': unit.unit.name, 'player': player,
                               'distance': distance, 'strength': strength, 'reason': reason})
        snapshots.append({'objective': deepcopy(objective), 'contenders': contenders})
    return snapshots


def required_choices(snapshots, choices=None):
    options = defaultdict(list)
    for result in _provisional_control(snapshots, choices or {}):
        if result['controller'] is not None:
            options[result['controller']].append(result['id'])
    return {unit: objectives for unit, objectives in options.items() if len(objectives) > 1}


def resolve_control(snapshots, choices):
    """Resolve distance, then US, then contested; no implicit multi-object choice."""
    selected = {}
    while pending := required_choices(snapshots, selected):
        unit, options = next(iter(pending.items()))
        if choices.get(unit) not in options:
            raise ValueError(f'{unit}: choose one objective from {options}')
        selected[unit] = choices[unit]
    if choices.keys() != selected.keys():
        raise ValueError('Objective choices may only resolve multiple controlled objectives')
    return _provisional_control(snapshots, selected)


def _provisional_control(snapshots, choices):
    results = []
    for snapshot in snapshots:
        objective = deepcopy(snapshot['objective'])
        eligible = [entry for entry in snapshot['contenders'] if entry['reason'] is None
                    and choices.get(entry['unit'], objective['id']) == objective['id']]
        closest = min((entry['distance'] for entry in eligible), default=float('inf'))
        nearest = [entry for entry in eligible if abs(entry['distance'] - closest) <= TOLERANCE]
        strength = max((entry['strength'] for entry in nearest), default=0)
        winners = [entry for entry in nearest if entry['strength'] == strength]
        controller = winners[0] if len(winners) == 1 else None
        objective.update(controller=controller['unit'] if controller else None,
                         player=controller['player'] if controller else None, contested=len(winners) > 1)
        results.append(objective)
    return results


def turn_key(game):
    counter = game.roundCounter
    return f'{counter.current_player}:{counter.currentRoundPlayer[0]}:{counter.currentRoundPlayer[1]}'


def score_turn(game, choices=None):
    """Commit a completed player-turn once, awarding either player's controllers."""
    key = turn_key(game)
    scored = getattr(game, 'battle_scored_turns', [])
    if key in scored:
        return []
    snapshots = control_snapshot(game)
    choices = {} if choices is None else choices
    results = resolve_control(snapshots, choices)
    refresh_landmark_grants(game, results, key)
    awards = []
    for snapshot, result in zip(snapshots, results):
        controller = result['controller']
        for contender in snapshot['contenders']:
            if contender['unit'] == controller:
                continue
            reason = (contender['reason'] or
                      ('controls another objective' if choices.get(contender['unit'], result['id']) != result['id']
                       else 'equal distance and Unit Strength: contested' if result['contested']
                       else 'another unit is closer or has greater Unit Strength at equal distance'))
            rule_skipped('Battle March objective', contender['name'],
                         f'{result["id"]}: {contender["distance"]:.3f}", US {contender["strength"]}; {reason}')
        if controller is not None:
            kind = result['kind']
            amount = game.battle_config['scoring'][f'{kind}_per_player_turn']
            entry = next(contender for contender in snapshot['contenders'] if contender['unit'] == controller)
            award = {'turn': key, 'objective': result['id'], 'player': result['player'],
                     'unit': controller, 'points': amount, 'rule': 'Strategic Landmarks' if kind == 'landmark' else 'Treasure Troves',
                     'reason': f'{entry["distance"]:.3f}" base distance, Unit Strength {entry["strength"]}'}
            awards.append(award)
            rule_log(award['rule'], entry['name'],
                     f'{result["id"]}: {award["reason"]} -> Player {result["player"]} +{amount} VP at turn {key}')
        else:
            rule_skipped('Battle March objective', result['id'],
                         'contested -> 0 VP' if result['contested'] else 'no eligible controlling unit -> 0 VP')
    game.battle_objectives = results
    game.battle_awards = [*getattr(game, 'battle_awards', []), *awards]
    game.battle_scored_turns = [*scored, key]
    sync_markers(game)
    return awards


async def finish_player_turn(fsm, request, args=()):
    """Resolve objective choices at the real turn boundary before advancing counters."""
    from panda3d.core import Point3
    from spell_effects import end_phase, end_turn
    game = fsm.game
    game.battleMarchBoundaryBusy = True
    game.magicBusy = True
    try:
        end_phase(game, 'CombatPhase')
        end_turn(game)
        snapshots = control_snapshot(game)
        choices = {}
        units = {unit.unitName: unit for unit in game.units}
        while pending := required_choices(snapshots, choices):
            identity, options = next(iter(pending.items()))
            unit = units[identity]
            if game.aiControls(unit):
                values = {entry['id']: game.battle_config['scoring'][f'{entry["kind"]}_per_player_turn']
                          for entry in game.battle_objectives}
                selected = max(options, key=lambda option: values[option])
            else:
                selected = await game.makeChoiceNew(options, Point3(0, 0, 10), owner=unit,
                    prompt=f'{unit.unit.name}: choose one objective to control',
                    detail='One unit can control one object (Matched Play Guide FAQ v1.5.3).')
            if selected not in options:
                raise ValueError(f'{identity}: invalid objective choice {selected!r}')
            choices[identity] = selected
        score_turn(game, choices)
        fsm._battle_march_boundary_ready = True
        game.magicBusy = False
        fsm.request(request, *args)
    finally:
        fsm._battle_march_boundary_ready = False
        game.battleMarchBoundaryBusy = False
        game.magicBusy = False