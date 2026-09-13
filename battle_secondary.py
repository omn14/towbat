"""Battle March secondary objectives (General's Companion pp. 36-37)."""

from copy import deepcopy
import math
import re

from battle_config import ConfigError, _keys, _number
from characters import side_of
from psychology import unit_strength_total
from rules_log import rule_log, rule_skipped
from scouts import model_base_boxes
from spell_templates import circle_distance


def cart_characteristics(name):
    """Supplement-only split profiles absent from the catalogues (Companion p. 37)."""
    profiles = {
        'Baggage Cart': {
            'M': '-', 'WS': '-', 'BS': '-', 'S': '4', 'T': '5', 'W': '4', 'I': '-', 'A': '-', 'Ld': '-',
            'Troop Type': 'Heavy Chariot', 'Base Size': '60x100', 'base_width_mm': 60, 'base_depth_mm': 100,
            'Special Rules': ['Close Order', 'Flee To Safety', 'Loner', 'Non-Combatant'],
            'Crew': [{'name': 'Baggage Cart Driver', 'count': 1}],
            'Beasts': [{'name': 'Baggage Cart Draft Animal', 'count': 2}],
        },
        'Baggage Cart Driver': {
            'M': '-', 'WS': '2', 'BS': '2', 'S': '3', 'T': '-', 'W': '-', 'I': '3', 'A': '1', 'Ld': '6',
            'Troop Type': 'Regular Infantry', 'Base Size': '25x25', 'Special Rules': [],
        },
        'Baggage Cart Draft Animal': {
            'M': '6', 'WS': '2', 'BS': '-', 'S': '3', 'T': '-', 'W': '-', 'I': '3', 'A': '1', 'Ld': '-',
            'Troop Type': 'War Beast', 'Base Size': '25x50', 'Special Rules': [],
        },
    }
    return deepcopy(profiles.get(name))


def cart_model(player_color):
    """A single 60 x 100mm cart miniature with load, driver and two draft animals."""
    from panda3d.core import (Geom, GeomNode, GeomTriangles, GeomVertexData,
                             GeomVertexFormat, GeomVertexWriter, NodePath)
    data = GeomVertexData('baggage-cart', GeomVertexFormat.getV3c4(), Geom.UHStatic)
    vertex = GeomVertexWriter(data, 'vertex')
    color = GeomVertexWriter(data, 'color')
    triangles = GeomTriangles(Geom.UHStatic)

    def polygon(points, tint):
        start = vertex.getWriteRow()
        for point in points:
            vertex.addData3(*point)
            color.addData4(*tint)
        for index in range(1, len(points) - 1):
            triangles.addVertices(start, start + index, start + index + 1)

    def cuboid(center, size, tint):
        corners = [(center[0] + horizontal * size[0] / 2, center[1] + vertical * size[1] / 2,
                    center[2] + height * size[2] / 2)
                   for height in (-1, 1) for vertical in (-1, 1) for horizontal in (-1, 1)]
        for indices in ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)):
            polygon([corners[index] for index in indices], tint)

    wood, iron, canvas = (.55, .32, .13, 1), (.12, .13, .13, 1), (.83, .78, .56, 1)
    cuboid((0, 0, .035), (60 / 25.4, 100 / 25.4, .07), player_color)
    cuboid((0, -.75, .60), (1.55, 1.65, .25), wood)
    for horizontal in (-.82, .82):
        cuboid((horizontal, -.75, 1.02), (.12, 1.7, .65), wood)
    for vertical in (-1.6, .10):
        cuboid((0, vertical, 1.02), (1.7, .12, .65), wood)
    cuboid((-.38, -.8, 1.03), (.65, 1.05, .58), canvas)
    cuboid((.38, -.95, 1.1), (.65, .7, .72), (.46, .61, .36, 1))
    for horizontal in (-1, 1):
        for vertical in (-1.3, -.2):
            rings = [[(horizontal + side * .09, vertical + math.cos(index * math.tau / 20) * .38,
                       .43 + math.sin(index * math.tau / 20) * .38) for index in range(20)] for side in (-1, 1)]
            polygon(rings[0], iron)
            polygon(rings[1], iron)
            for index in range(20):
                following = (index + 1) % 20
                polygon([rings[0][index], rings[0][following], rings[1][following], rings[1][index]], iron)
    cuboid((0, .30, .63), (.13, 1.35, .14), wood)
    for horizontal in (-.5, .5):
        tint = (.32, .29, .24, 1) if horizontal < 0 else (.58, .53, .44, 1)
        cuboid((horizontal, 1.02, .58), (.42, .85, .5), tint)
        cuboid((horizontal, 1.5, .9), (.28, .48, .30), tint)
        for lateral in (-.13, .13):
            for vertical in (.75, 1.32):
                cuboid((horizontal + lateral, vertical, .22), (.12, .14, .4), tint)
    cuboid((0, .0, 1.35), (.36, .3, .65), (.21, .42, .60, 1))
    cuboid((0, .0, 1.85), (.27, .27, .32), (.76, .60, .45, 1))
    triangles.closePrimitive()
    geometry = Geom(data)
    geometry.addPrimitive(triangles)
    node = GeomNode('baggage-cart-figure')
    node.addGeom(geometry)
    root = NodePath('baggage-cart-model')
    figure = root.attachNewNode(node)
    figure.setColorOff(1)
    figure.setLightOff()
    figure.setShaderOff()
    figure.setTwoSided(True)
    return root


def enabled(game, name):
    config = getattr(game, 'battle_config', None)
    return bool(config and name in config['optional_rules']['secondary_objectives'])


def non_combatant(unit):
    """A Non-Combatant cannot charge/chase or borrow command support (p. 37)."""
    profile = (getattr(unit, 'hostUnit', None) or unit).unit.model
    return ('Non-Combatant' in getattr(profile, 'characteristics', {}).get('Special Rules', [])
            or any(isinstance(rule, dict) and rule.get('name', '').casefold() == 'non-combatant'
                   for rule in getattr(profile, 'special_rules', [])))


def cart_escape_edge(field, enemy_zone, boxes, previous_boxes=None):
    """Any part may touch/cross an enemy-zone board edge (Companion p. 37)."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    from psychology import _box_corners
    boundary = Polygon(field.outline).boundary.intersection(Polygon(enemy_zone.vertices))
    footprint = unary_union([Polygon(_box_corners(*box)) for box in boxes])
    if previous_boxes:
        from battle_terrain import swept_footprint
        footprint = unary_union([
            swept_footprint(Polygon(_box_corners(*start)), end[0] - start[0], end[1] - start[1])
            for start, end in zip(previous_boxes, boxes)] + [footprint])
    return not boundary.is_empty and not footprint.is_empty and footprint.distance(boundary) <= 1e-5


def cart_awards(game):
    """Literal escaped-cart interpretation: owner +25, opponent +25 (p. 36)."""
    from battlefield import deployment_zone_for
    live = {unit.unitName: unit for unit in game.units if not unit.bodyNP.isEmpty() and unit.unit.nmodels > 0}
    awards = []
    for record in getattr(game, 'battle_secondary', {}).get('carts', []):
        owner = record['player']
        unit = live.get(record['unit'])
        alive = unit is not None and not record['escaped']
        awards.append({'player': owner if alive else 3 - owner, 'unit': record['unit'],
                       'points': 25, 'rule': 'Baggage Carts',
                       'reason': 'alive on the battlefield' if alive else 'not alive on the battlefield (including escape)'})
        zone = deployment_zone_for(game, 3 - owner)
        boxes = model_base_boxes(unit) if alive else []
        wholly = bool(boxes) and all(zone.contains_box(box) for box in boxes)
        if record['escaped'] or wholly:
            awards.append({'player': owner, 'unit': record['unit'], 'points': 25, 'rule': 'Baggage Carts',
                           'reason': 'escaped safely; not destroyed' if record['escaped'] else 'wholly within the enemy deployment zone'})
    return awards


def empty_state():
    return {'raid_attempts': [], 'awards': [], 'resolved_starts': [], 'cart_mode': None, 'carts': []}


def validate_state(config, record, objectives):
    _keys(record, 'raid_attempts awards resolved_starts'
          + (' cart_mode' if 'cart_mode' in record else '') + (' carts' if 'carts' in record else ''), 'battle_march.secondary')
    markers = {objective['id']: objective for objective in objectives}
    for field in ('raid_attempts', 'awards', 'resolved_starts'):
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
    record = deepcopy(record)
    record.setdefault('cart_mode', None)
    record.setdefault('carts', [])
    if record['cart_mode'] not in (None, 'player_1', 'player_2', 'both'):
        raise ConfigError('battle_march.secondary.cart_mode: invalid baggage assignment')
    if not isinstance(record['carts'], list):
        raise ConfigError('battle_march.secondary.carts: expected a list')
    if (record['carts'] or record['cart_mode']) and 'baggage_carts' not in config['optional_rules']['secondary_objectives']:
        raise ConfigError('battle_march.secondary: Baggage Carts is not enabled')
    seen.clear()
    owners = []
    for cart in record['carts']:
        _keys(cart, 'unit player escaped', 'battle_march.secondary.carts')
        _number(cart['player'], 'battle_march.secondary.cart.player', 1, 2, integer=True)
        if type(cart['escaped']) is not bool or not isinstance(cart['unit'], str) or not cart['unit'] or cart['unit'] in seen:
            raise ConfigError('battle_march.secondary.carts: invalid or duplicate cart')
        seen.add(cart['unit'])
        owners.append(cart['player'])
    expected = {None: [], 'player_1': [1], 'player_2': [2], 'both': [1, 2]}[record['cart_mode']]
    if sorted(owners) != expected:
        raise ConfigError('battle_march.secondary.carts: cart ownership does not match the setup choice')
    return record


async def prepare_carts(game):
    """Choose Attack the Baggage or Guard Duty, adding carts after mustering (p. 36)."""
    from battle_preparation import choose
    from deployPhase import stage_undeployed
    if not enabled(game, 'baggage_carts'):
        return
    state = game.battle_secondary
    if state['cart_mode'] is not None:
        return
    choices = {'Guard Duty: both players': ('both', (1, 2)),
               'Attack the Baggage: Player 1 defends': ('player_1', (1,)),
               'Attack the Baggage: Player 2 defends': ('player_2', (2,))}
    answer = await choose(game, 1, list(choices), 'Baggage Carts assignment')
    mode, owners = choices[answer]
    created = []
    try:
        for player in owners:
            identity = f'Baggage Cart P{player}'
            if any(unit.unitName == identity for unit in game.units):
                raise ConfigError(f'Baggage cart ID already in use: {identity}')
            unit = game._create_unit({'name': 'Baggage Cart', 'nmodels': 1, 'files': 1, 'ranks': 1,
                                      'points_cost': 0, 'category': 'Scenario'}, player, identity)
            if unit is None:
                raise ConfigError('Could not create baggage cart')
            created.append(unit)
        state['carts'] = [{'unit': unit.unitName, 'player': player, 'escaped': False}
                          for unit, player in zip(created, owners)]
        state['cart_mode'] = mode
        stage_undeployed(game)
        rule_log('Baggage Carts', 'setup', f'{answer}; {len(created)} carts added, one model on each 60 x 100mm base')
    except Exception:
        for unit in created:
            game.combat.removeUnitFromPlay(unit)
        raise


def escape_cart(game, unit, *, from_pos=None):
    """Remove a cart as escaped, never as destroyed, when its base reaches safety."""
    from battlefield import battlefield_for, deployment_zone_for
    if (not enabled(game, 'baggage_carts') or getattr(game, 'restoringBattle', False)
            or not getattr(unit, 'isDeployed', False) or unit.bodyNP.isEmpty() or unit.unit.nmodels <= 0):
        return False
    record = next((cart for cart in game.battle_secondary['carts'] if cart['unit'] == unit.unitName), None)
    if record is None or record['escaped']:
        return False
    boxes = model_base_boxes(unit)
    previous = None
    if from_pos is not None:
        position = unit.bodyNP.getPos(game.render)
        previous = [(box[0] - position.x + from_pos.x, box[1] - position.y + from_pos.y, *box[2:]) for box in boxes]
    if not cart_escape_edge(battlefield_for(game), deployment_zone_for(game, 3 - record['player']), boxes, previous):
        return False
    record['escaped'] = True
    rule_log('Flee To Safety', unit, f'base touches/crosses an enemy-zone board edge; escaped, not destroyed; '
             f'owner Player {record["player"]} +25 VP, opponent +25 VP at battle end')
    game.combat.removeUnitFromPlay(unit)
    return True


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