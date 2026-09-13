"""Resumable pre-deployment setup (General's Companion pp. 23-27; Rulebook p. 268)."""

from copy import deepcopy
import math
import random

from panda3d.core import Point3

from battle_config import ConfigError, _choice, _keys, _number, army_report
from battle_terrain import footprint, objective_clearance_shift, scatter_distance
from rules_log import dice_roll, rule_log, rule_skipped


TERRAIN_CHOICES = {'Hill': 'hill', 'Wood': 'forest', 'Building': 'house'}


def new_terrain_state():
    return {'selections': {'1': [], '2': []}, 'selection_complete': [],
            'rolls': [], 'winner': None, 'placed': []}


def roll_winner(rolls):
    if not isinstance(rolls, list):
        raise ConfigError('setup roll-off: expected a list')
    winner = None
    for pair in rolls:
        if winner is not None or not isinstance(pair, list) or len(pair) != 2:
            raise ConfigError('setup roll-off: expected ties followed by one deciding pair')
        for value in pair:
            _number(value, 'setup roll-off', 1, 6, integer=True)
        if pair[0] != pair[1]:
            winner = 1 if pair[0] > pair[1] else 2
    return winner


def roll_off(rolls, rule):
    winner = roll_winner(rolls)
    while winner is None:
        pair = [random.randint(1, 6), random.randint(1, 6)]
        rolls.append(pair)
        dice_roll(pair)
        winner = roll_winner(rolls)
        rule_log(rule, 'setup', f'P1={pair[0]}, P2={pair[1]} -> '
                 + (f'Player {winner}' if winner else 'tie; reroll'))
    return winner


def validate_terrain_state(config, state):
    _keys(state, 'selections selection_complete rolls winner placed'
          + (' scatter' if 'scatter' in state else ''), 'setup.terrain')
    _keys(state['selections'], '1 2', 'setup.terrain.selections')
    count = config['terrain']['feature_count']
    for selections in state['selections'].values():
        if not isinstance(selections, list) or len(selections) > count:
            raise ConfigError('setup.terrain.selections: too many features')
        for kind in selections:
            _choice(kind, tuple(TERRAIN_CHOICES.values()), 'setup.terrain.selections')
    complete = state['selection_complete']
    if (not isinstance(complete, list) or any(type(player) is not int or player not in (1, 2) for player in complete)
            or len(set(complete)) != len(complete)):
        raise ConfigError('setup.terrain.selection_complete: expected unique player numbers')
    winner = roll_winner(state['rolls'])
    if state['winner'] != winner or isinstance(state['winner'], bool):
        raise ConfigError('setup.terrain.winner: does not match recorded dice')
    placed = state['placed']
    if not isinstance(placed, list) or len(placed) > count:
        raise ConfigError('setup.terrain.placed: too many features')
    pool = state['selections']['1'] + state['selections']['2']
    used = set()
    for index, record in enumerate(placed):
        _keys(record, 'type width height center player pool_index', 'setup.terrain.placed')
        _number(record['pool_index'], 'setup.terrain.pool_index', 0, len(pool) - 1, integer=True)
        if record['pool_index'] in used or record['type'] != pool[record['pool_index']]:
            raise ConfigError('setup.terrain.placed: duplicated or mismatched pool feature')
        used.add(record['pool_index'])
        expected = winner if config['terrain']['method'] == 'scattered' or index % 2 == 0 else 3 - (winner or 0)
        if winner is None or len(complete) != 2 or type(record['player']) is not int or record['player'] != expected:
            raise ConfigError('setup.terrain.player: does not follow the placement roll-off')
        for dimension, maximum in (('width', config['battlefield']['width']), ('height', config['battlefield']['depth'])):
            _number(record[dimension], f'setup.terrain.{dimension}', .1, maximum)
        if not isinstance(record['center'], list) or len(record['center']) != 3:
            raise ConfigError('setup.terrain.center: expected three coordinates')
        for value in record['center']:
            _number(value, 'setup.terrain.center', -100, 100)
        if record['center'][2] != 0:
            raise ConfigError('setup.terrain.center: terrain must rest on the table')
    if 'scatter' in state:
        scatter = state['scatter']
        _keys(scatter, 'count_die selected results', 'setup.terrain.scatter')
        if config['terrain']['method'] != 'scattered':
            raise ConfigError('setup.terrain.scatter: only available with scattered placement')
        _number(scatter['count_die'], 'setup.terrain.scatter.count_die', 1, 6, integer=True)
        selected = scatter['selected']
        if not isinstance(selected, list) or len(selected) > min((scatter['count_die'] + 1) // 2, len(placed)):
            raise ConfigError('setup.terrain.scatter.selected: too many features')
        for index in selected:
            _number(index, 'setup.terrain.scatter.selected', 0, len(placed) - 1, integer=True)
        if len(selected) != len(set(selected)):
            raise ConfigError('setup.terrain.scatter.selected: repeated feature')
        if not isinstance(scatter['results'], list) or len(scatter['results']) > len(selected):
            raise ConfigError('setup.terrain.scatter.results: too many results')
        for index, result in enumerate(scatter['results']):
            _keys(result, 'feature dice face angle travel origin destination', 'setup.terrain.scatter.result')
            if type(result['feature']) is not int or result['feature'] != selected[index]:
                raise ConfigError('setup.terrain.scatter.result: feature differs from selected order')
            if not isinstance(result['dice'], list) or len(result['dice']) != 2:
                raise ConfigError('setup.terrain.scatter.dice: expected 2D6')
            for value in result['dice']:
                _number(value, 'setup.terrain.scatter.dice', 1, 6, integer=True)
            _number(result['face'], 'setup.terrain.scatter.face', 1, 6, integer=True)
            _number(result['travel'], 'setup.terrain.scatter.travel', 0, sum(result['dice']))
            for field in ('origin', 'destination'):
                if not isinstance(result[field], list) or len(result[field]) != 3:
                    raise ConfigError(f'setup.terrain.scatter.{field}: expected three coordinates')
                for value in result[field]:
                    _number(value, f'setup.terrain.scatter.{field}', -100, 100)
            if result['face'] <= 2:
                if result['angle'] is not None or result['travel'] != 0 or result['origin'] != result['destination']:
                    raise ConfigError('setup.terrain.scatter: Hit cannot move terrain')
            else:
                _number(result['angle'], 'setup.terrain.scatter.angle', 0, math.tau)
                expected = [result['origin'][0] + math.cos(result['angle']) * result['travel'],
                            result['origin'][1] + math.sin(result['angle']) * result['travel'], 0]
                if any(abs(first - second) > 1e-5 for first, second in zip(expected, result['destination'])):
                    raise ConfigError('setup.terrain.scatter: destination differs from the recorded move')
    return deepcopy(state)


def owner_for(game, player):
    units = game.player1Units if player == 1 else game.player2Units
    return next((unit for unit in units if getattr(unit, 'hostUnit', None) is None), None)


async def choose(game, player, options, prompt):
    owner = owner_for(game, player)
    if owner is None:
        raise ConfigError(f'Player {player}: cannot prepare an empty army')
    if game.aiControls(owner):
        return options[0]
    answer = await game.makeChoiceNew(options, Point3(0, 0, 10), owner=owner,
                                      prompt=f'Player {player}: {prompt}')
    if answer not in options:
        raise ConfigError('Setup choice cancelled; resume setup to continue')
    return answer


async def prepare_armies(game, preparation):
    game.battle_army_reports = {}
    for player in (1, 2):
        units = game.player1Units if player == 1 else game.player2Units
        if owner_for(game, player) is None:
            raise ConfigError(f'Player {player}: cannot prepare an empty army')
        report = army_report(game.battle_config, units)
        game.battle_army_reports[player] = report
        if player in preparation['army_acknowledged']:
            continue
        lines = [f'{report["points"]:g}/{game.battle_config["points_limit"]} points']
        lines += ['Violation: ' + text for text in report['violations']]
        lines += ['Unverified: ' + text for text in report['unverified']]
        rule_log('Battle March army report', f'Player {player}', '; '.join(lines))
        await choose(game, player, ['Acknowledge report'], '\n'.join(lines))
        preparation['army_acknowledged'].append(player)
    preparation['stage'] = 'terrain'


def rebuild_terrain(game, records):
    game.terrain_manager.clear()
    game.terrain_manager.load_records(records)


async def prepare_terrain(game, preparation):
    from battle_setup_ui import TerrainPlacement
    state = preparation.setdefault('terrain', new_terrain_state())
    count = game.battle_config['terrain']['feature_count']
    for player in (1, 2):
        if player in state['selection_complete']:
            continue
        selections = state['selections'][str(player)]
        while len(selections) < count:
            total = sum(len(entries) for entries in state['selections'].values())
            options = list(TERRAIN_CHOICES)
            if total >= count:
                options.append('Finish selection')
            answer = await choose(game, player, options, f'terrain pool ({len(selections)}/{count} selected)')
            if answer == 'Finish selection':
                break
            selections.append(TERRAIN_CHOICES[answer])
        state['selection_complete'].append(player)
    if state['winner'] is None:
        state['winner'] = roll_off(state['rolls'], 'Terrain placement order')
    rebuild_terrain(game, state['placed'])
    pool = state['selections']['1'] + state['selections']['2']
    while len(state['placed']) < count:
        player = (state['winner'] if game.battle_config['terrain']['method'] == 'scattered'
                  or len(state['placed']) % 2 == 0 else 3 - state['winner'])
        used = {record['pool_index'] for record in state['placed']}
        options = {f'{index + 1}: {kind}': index for index, kind in enumerate(pool) if index not in used}
        selected = await choose(game, player, list(options), f'place terrain {len(state["placed"]) + 1}/{count}')
        pool_index = options[selected]
        placed = [(footprint(piece), record['player'])
                  for piece, record in zip(game.terrain_manager.terrain_pieces, state['placed'])]
        editor = TerrainPlacement(game, {'type': pool[pool_index], 'width': 4, 'height': 4}, placed, player)
        record = await editor.choose(owner_for(game, player))
        if record is None:
            continue
        record['pool_index'] = pool_index
        state['placed'].append(record)
        game.terrain_manager.load_records([record])
    if game.battle_config['terrain']['method'] == 'scattered':
        await scatter_terrain(game, state)
    preparation['stage'] = 'objectives'


async def scatter_terrain(game, state):
    """Loser selects D3 features, then scatters each once (Storm Dragon p. 23)."""
    from dice import SCATTER_FACES
    if not state['placed']:
        rule_skipped('Scattered terrain', 'setup', '0 placed features; no D3 or scatter required')
        return
    if 'scatter' not in state:
        count_die = random.randint(1, 6)
        state['scatter'] = {'count_die': count_die, 'selected': [], 'results': []}
        dice_roll([count_die])
        rule_log('Scattered terrain', 'setup',
                 f'D6={count_die} -> D3={(count_die + 1) // 2}; {len(state["placed"])} available features')
    scatter = state['scatter']
    count = min((scatter['count_die'] + 1) // 2, len(state['placed']))
    player = 3 - state['winner']
    while len(scatter['selected']) < count:
        options = {f'{index + 1}: {record["type"]}': index for index, record in enumerate(state['placed'])
                   if index not in scatter['selected']}
        answer = await choose(game, player, list(options), f'scatter feature {len(scatter["selected"]) + 1}/{count}')
        scatter['selected'].append(options[answer])
    for index in scatter['selected'][len(scatter['results']):]:
        record = state['placed'][index]
        dice = [random.randint(1, 6), random.randint(1, 6)]
        face = random.randint(1, 6)
        dice_roll([*dice, face])
        angle = random.uniform(0, math.tau) if SCATTER_FACES[face] == 'Arrow' else None
        origin = list(record['center'])
        travel = 0.0
        destination = origin[:]
        if angle is not None:
            shapes = [footprint(piece) for piece in game.terrain_manager.terrain_pieces]
            direction = math.cos(angle), math.sin(angle)
            travel = scatter_distance(game.battlefield, shapes[index],
                                      [shape for other, shape in enumerate(shapes) if other != index],
                                      direction, sum(dice))
            destination = [origin[0] + direction[0] * travel, origin[1] + direction[1] * travel, 0]
        record['center'] = destination
        scatter['results'].append({'feature': index, 'dice': dice, 'face': face, 'angle': angle,
                                   'travel': travel, 'origin': origin, 'destination': destination})
        rule_log('Scattered terrain', f'Feature {index + 1}',
                 f'2D6={dice} -> {sum(dice)}", scatter={SCATTER_FACES[face]}; '
                 f'{travel:.2f}" moved from {origin[:2]} to {destination[:2]}'
                 + ('; stopped at first contact' if angle is not None and travel < sum(dice) - 1e-5 else ''))
        rebuild_terrain(game, state['placed'])


def place_objectives(game, preparation):
    from battle_objectives import sync_markers
    from shapely.affinity import translate
    from shapely.geometry import Point
    state = preparation['terrain']
    pieces = game.terrain_manager.terrain_pieces
    shapes = [footprint(piece) for piece in pieces]
    records = deepcopy(state['placed'])
    clearance = game.battle_config['terrain']['objective_clearance']
    for index, shape in enumerate(shapes):
        shift = objective_clearance_shift(game.battlefield, shape, game.battle_objectives, clearance)
        shifted = translate(shape, *shift)
        if any(shifted.intersects(other) for other_index, other in enumerate(shapes) if index != other_index):
            raise ConfigError(f'Objective clearance would overlap terrain {index + 1}; revise terrain placement')
        if any(shifted.distance(Point(*objective['center'])) < objective['diameter'] / 2 + clearance - 1e-5
               for objective in game.battle_objectives):
            raise ConfigError('Objective clearance could not be achieved; revise terrain placement')
        shapes[index] = shifted
        records[index]['center'][0] += shift[0]
        records[index]['center'][1] += shift[1]
    for original, moved in zip(state['placed'], records):
        if original['center'] != moved['center']:
            rule_log('Objective clearance', moved['type'], f'{original["center"][:2]} -> {moved["center"][:2]}; '
                     f'minimum displacement for {clearance:g}" clearance')
    state['placed'] = records
    rebuild_terrain(game, records)
    preparation['stage'] = 'zones'
    sync_markers(game)
    from battlefield import draw_battlefield
    draw_battlefield(game)


async def run_preparation(game):
    from battlefield import draw_battlefield
    from deployPhase import refresh_deployment
    from spell_generation import begin_spell_generation
    preparation = game.battle_setup['preparation']
    retry = False
    game.battleMarchSetupBusy = True
    game.magicBusy = True
    try:
        if preparation['stage'] == 'armies':
            await prepare_armies(game, preparation)
        if preparation['stage'] == 'terrain':
            await prepare_terrain(game, preparation)
        if preparation['stage'] == 'objectives':
            place_objectives(game, preparation)
        if preparation['stage'] == 'zones':
            player = 3 - preparation['map_player']
            choice = await choose(game, player, ['Zone 1', 'Zone 2'],
                                  f'{game.battle_setup["deployment_map"].replace("_", " ")}: choose deployment zone')
            zone = int(choice[-1])
            game.battle_setup['player_zones'] = {str(player): zone, str(3 - player): 3 - zone}
            rule_log('Deployment zones', 'setup', f'Player {player} chooses zone {zone}; '
                     f'Player {3 - player} takes zone {3 - zone}')
            draw_battlefield(game)
            preparation['stage'] = 'first_drop'
        if preparation['stage'] == 'first_drop':
            preparation['first_drop'] = roll_off(preparation['first_drop_rolls'], 'First deployment')
            preparation['stage'] = 'complete'
            game.roundCounter.request('PlayerOne' if preparation['first_drop'] == 1 else 'PlayerTwo')
        game.battleMarchSetupError = None
    except ConfigError as error:
        game.battleMarchSetupError = str(error)
        rule_skipped('Battle March setup', 'setup', str(error))
        player = preparation.get('terrain', {}).get('winner') or 1
        if owner_for(game, player) is not None:
            answer = await choose(game, player, ['Pause setup', 'Revise terrain'], str(error))
            if answer == 'Revise terrain':
                preparation['terrain']['placed'] = []
                preparation['terrain'].pop('scatter', None)
                preparation['stage'] = 'terrain'
                rebuild_terrain(game, [])
                retry = True
    finally:
        game.battleMarchSetupBusy = False
        game.magicBusy = False
    if preparation['stage'] == 'complete':
        refresh_deployment(game)
        begin_spell_generation(game)
    elif retry:
        begin_preparation(game)


def begin_preparation(game):
    preparation = (getattr(game, 'battle_setup', None) or {}).get('preparation')
    if not preparation or preparation['stage'] == 'complete':
        return False
    game.ignore('mouse1')
    if hasattr(game, 'boundary_np'):
        game.boundary_np.setCollideMask(0)
    if not getattr(game, 'restoringBattle', False) and not getattr(game, 'battleMarchSetupBusy', False):
        game.battleMarchSetupBusy = True
        game.magicBusy = True
        game.taskMgr.add(run_preparation(game), 'battleMarchPreparation')
    return True