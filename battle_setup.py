"""Resolved Battle March setup, separate from editable rules (Companion pp. 24-27)."""

from copy import deepcopy
from random import Random
import random
import re

from battle_config import (ConfigError, CUSTOM_DEPLOYMENT_MAPS, DEPLOYMENT_MAPS, LANDMARK_PROPERTIES,
                           MIRRORABLE_MAPS, OBJECTIVE_LAYOUTS, _boolean,
                           _choice, _keys, _number, validate_activation, validate_config)
from battlefield import Battlefield, STANDARD_BATTLEFIELD, draw_battlefield


def resolve_setup(config, seed):
    """Resolve setup dice once; previewing and restoring never call this function."""
    config = validate_config(config)
    _number(seed, 'setup.seed', 0, 2 ** 53 - 1, integer=True)
    rng = Random(seed)
    rolls = {}

    def resolve(value, choices, name, divisor=1):
        if value != 'random':
            return value
        rolls[name] = rng.randint(1, 6)
        return choices[(rolls[name] - 1) // divisor]

    layout = resolve(config['objectives']['layout'], OBJECTIVE_LAYOUTS, 'objectives', 2)
    property_name = (resolve(config['objectives']['landmark_property'], LANDMARK_PROPERTIES, 'landmark', 2)
                     if layout == 'landmark' else None)
    deployment = resolve(config['deployment']['map'], DEPLOYMENT_MAPS, 'deployment')
    return {
        'version': 1, 'seed': seed, 'rolls': rolls, 'deployment_map': deployment,
        'mirror': config['deployment']['mirror'] and deployment in MIRRORABLE_MAPS,
        'player_zones': {'1': 1, '2': 2}, 'objective_layout': layout,
        'landmark_property': property_name,
    }


def validate_setup(config, setup):
    config = validate_config(config)
    _keys(setup, 'version seed rolls deployment_map mirror player_zones objective_layout landmark_property'
          + (' first_turn' if 'first_turn' in setup else '')
          + (' preparation' if 'preparation' in setup else ''), 'setup')
    if type(setup['version']) is not int or setup['version'] != 1:
        raise ConfigError('setup.version: only version 1 is supported')
    _number(setup['seed'], 'setup.seed', 0, 2 ** 53 - 1, integer=True)
    _choice(setup['deployment_map'], (*DEPLOYMENT_MAPS, *CUSTOM_DEPLOYMENT_MAPS), 'setup.deployment_map')
    _choice(setup['objective_layout'], (*OBJECTIVE_LAYOUTS, 'none'), 'setup.objective_layout')
    if setup['objective_layout'] == 'landmark':
        _choice(setup['landmark_property'], LANDMARK_PROPERTIES, 'setup.landmark_property')
    elif setup['landmark_property'] is not None:
        raise ConfigError('setup.landmark_property: only landmarks have a property')
    _boolean(setup['mirror'], 'setup.mirror')
    if setup['mirror'] and setup['deployment_map'] not in MIRRORABLE_MAPS:
        raise ConfigError('setup.mirror: this map has no alternate deployment')
    _keys(setup['player_zones'], '1 2', 'setup.player_zones')
    for value in setup['player_zones'].values():
        _number(value, 'setup.player_zones', 1, 2, integer=True)
    if set(setup['player_zones'].values()) != {1, 2}:
        raise ConfigError('setup.player_zones: players must occupy opposite zones')
    resolutions = [
        ('objectives', config['objectives']['layout'], OBJECTIVE_LAYOUTS, 2, setup['objective_layout']),
        ('deployment', config['deployment']['map'], DEPLOYMENT_MAPS, 1, setup['deployment_map']),
    ]
    if setup['objective_layout'] == 'landmark':
        resolutions.append(('landmark', config['objectives']['landmark_property'],
                            LANDMARK_PROPERTIES, 2, setup['landmark_property']))
    _keys(setup['rolls'], ' '.join(name for name, value, *_ in resolutions if value == 'random'), 'setup.rolls')
    for name, value in setup['rolls'].items():
        _number(value, f'setup.rolls.{name}', 1, 6, integer=True)
    for name, configured, choices, divisor, resolved in resolutions:
        expected = choices[(setup['rolls'][name] - 1) // divisor] if configured == 'random' else configured
        if resolved != expected:
            raise ConfigError(f'setup.{name}: does not match the saved configuration and dice')
    if setup['mirror'] != (config['deployment']['mirror'] and setup['deployment_map'] in MIRRORABLE_MAPS):
        raise ConfigError('setup.mirror: does not match the saved configuration')
    if 'first_turn' in setup:
        record = setup['first_turn']
        _keys(record, 'rolls winner player' + (' glass' if 'glass' in record else ''), 'setup.first_turn')
        bonus = [0, 0]
        if 'glass' in record:
            from battle_preparation import roll_winner
            glass = record['glass']
            _keys(glass, 'players rolls winner', 'setup.first_turn.glass')
            if glass['players'] not in ([], [1], [2], [1, 2]) or any(type(player) is not int for player in glass['players']):
                raise ConfigError('setup.first_turn.glass.players: invalid owners')
            if glass['players'] and not config['optional_rules']['battle_march_magic_items']:
                raise ConfigError('setup.first_turn.glass: Battle March items are disabled')
            authentic = roll_winner(glass['rolls'])
            if len(glass['players']) < 2:
                if glass['rolls']:
                    raise ConfigError('setup.first_turn.glass: authenticity only rolls for two owners')
                authentic = next(iter(glass['players']), None)
            if glass['winner'] != authentic or isinstance(glass['winner'], bool) or (glass['players'] and authentic is None):
                raise ConfigError('setup.first_turn.glass.winner: does not match authenticity dice')
            if authentic is not None:
                bonus[authentic - 1] = 1
        if not isinstance(record['rolls'], list):
            raise ConfigError('setup.first_turn.rolls: expected a list')
        winner = None
        for pair in record['rolls']:
            if winner is not None or not isinstance(pair, list) or len(pair) != 2:
                raise ConfigError('setup.first_turn.rolls: expected ties followed by one deciding roll')
            for value in pair:
                _number(value, 'setup.first_turn.rolls', 1, 6, integer=True)
            totals = [pair[index] + bonus[index] for index in range(2)]
            if totals[0] != totals[1]:
                winner = 1 if totals[0] > totals[1] else 2
        if record['winner'] != winner or isinstance(record['winner'], bool):
            raise ConfigError('setup.first_turn.winner: does not match recorded dice')
        if record['player'] is not None:
            _number(record['player'], 'setup.first_turn.player', 1, 2, integer=True)
            if winner is None:
                raise ConfigError('setup.first_turn.player: cannot choose before the roll-off')
    if 'preparation' in setup:
        preparation = setup['preparation']
        _keys(preparation, 'stage army_acknowledged map_player first_drop_rolls first_drop'
              + (' terrain' if 'terrain' in preparation else ''), 'setup.preparation')
        _choice(preparation['stage'], ('armies', 'terrain', 'objectives', 'zones', 'first_drop', 'complete'),
                'setup.preparation.stage')
        if (not isinstance(preparation['army_acknowledged'], list)
                or any(type(player) is not int or player not in (1, 2) for player in preparation['army_acknowledged'])
                or len(preparation['army_acknowledged']) != len(set(preparation['army_acknowledged']))):
            raise ConfigError('setup.preparation.army_acknowledged: expected unique player numbers')
        _number(preparation['map_player'], 'setup.preparation.map_player', 1, 2, integer=True)
        if not isinstance(preparation['first_drop_rolls'], list):
            raise ConfigError('setup.preparation.first_drop_rolls: expected a list')
        winner = None
        for pair in preparation['first_drop_rolls']:
            if winner is not None or not isinstance(pair, list) or len(pair) != 2:
                raise ConfigError('setup.preparation.first_drop_rolls: expected ties then a deciding roll')
            for value in pair:
                _number(value, 'setup.preparation.first_drop_rolls', 1, 6, integer=True)
            if pair[0] != pair[1]:
                winner = 1 if pair[0] > pair[1] else 2
        if type(preparation['first_drop']) is bool or preparation['first_drop'] != winner:
            raise ConfigError('setup.preparation.first_drop: does not match recorded dice')
        if preparation['stage'] == 'complete' and (winner is None or len(preparation['army_acknowledged']) != 2):
            raise ConfigError('setup.preparation: complete setup requires both army reports and a first drop')
        if 'terrain' in preparation:
            from battle_preparation import validate_terrain_state
            validate_terrain_state(config, preparation['terrain'])
            if (preparation['stage'] in ('objectives', 'zones', 'first_drop', 'complete')
                    and len(preparation['terrain']['placed']) != config['terrain']['feature_count']):
                raise ConfigError('setup.preparation: terrain placement is incomplete')
        if preparation['stage'] != 'armies' and len(preparation['army_acknowledged']) != 2:
            raise ConfigError('setup.preparation: both army reports must be acknowledged')
    return deepcopy(setup)


def prepare_new_battle(game, config, seed):
    """Initialize a new setup without rewriting armies or scaling the visual table."""
    config = validate_activation(config)
    if any(getattr(unit, 'isDeployed', False) for unit in game.units):
        raise ConfigError('Battle March setup must start before any unit is deployed')
    setup = resolve_setup(config, seed)
    setup['preparation'] = {'stage': 'armies', 'army_acknowledged': [], 'map_player': 1,
                            'first_drop_rolls': [], 'first_drop': None}
    from battle_preparation import new_terrain_state
    setup['preparation']['terrain'] = new_terrain_state()
    game.terrain_manager.clear()
    restore_battle(game, {'config': config, 'setup': setup})
    return game.battle_setup


def first_turn_glass(game):
    """Only one authentic General's Ranger's Glass adds +1 (Companion p. 48)."""
    from characters import side_of
    from magic_items import EffectKind, effects_for, report_inactive_effects
    from battle_preparation import roll_off
    from rules_log import rule_log, rule_skipped
    players = []
    if game.battle_config['optional_rules']['battle_march_magic_items']:
        for unit in game.units:
            entries = effects_for(unit, EffectKind.FIRST_TURN)
            player = side_of(game, unit)
            if entries and getattr(unit, 'isGeneral', False) and player in (1, 2):
                players.append(player)
            elif entries:
                rule_skipped("The Ranger's Glass", unit, 'bearer is not the General; no first-turn modifier')
            report_inactive_effects(unit, EffectKind.FIRST_TURN, 'no first-turn modifier')
    record = {'players': sorted(set(players)), 'rolls': [], 'winner': None}
    if len(record['players']) == 2:
        record['winner'] = roll_off(record['rolls'], "Ranger's Glass authenticity")
        rule_log("The Ranger's Glass", 'setup', f'Player {record["winner"]} has the authentic item; '
                 f'Player {3 - record["winner"]} has a worthless copy with no +1 bonus')
    elif players:
        record['winner'] = players[0]
    return record


async def choose_first_turn(game):
    """Separate roll-off; the winner chooses first or second (Companion p. 27)."""
    from panda3d.core import Point3
    from rules_log import dice_roll, rule_log
    game.battleMarchSetupBusy = True
    game.magicBusy = True
    try:
        record = game.battle_setup.setdefault('first_turn', {'rolls': [], 'winner': None, 'player': None})
        if 'glass' not in record and not record['rolls']:
            record['glass'] = first_turn_glass(game)
        authentic = record.get('glass', {}).get('winner')
        while record['winner'] is None:
            pair = [random.randint(1, 6), random.randint(1, 6)]
            record['rolls'].append(pair)
            dice_roll(pair)
            totals = [pair[index] + (1 if authentic == index + 1 else 0) for index in range(2)]
            if authentic is not None:
                rule_log("The Ranger's Glass", 'setup', f'Player {authentic} first-turn die '
                         f'{pair[authentic - 1]} +1 -> {totals[authentic - 1]}')
            if totals[0] == totals[1]:
                rule_log('Battle March first turn', 'setup', f'P1={totals[0]}, P2={totals[1]} tied; reroll')
            else:
                record['winner'] = 1 if totals[0] > totals[1] else 2
                rule_log('Battle March first turn', 'setup',
                         f'P1={totals[0]}, P2={totals[1]} -> Player {record["winner"]} chooses; no first-finished bonus')
        if record['player'] is None:
            winner = record['winner']
            units = game.player1Units if winner == 1 else game.player2Units
            owner = next(unit for unit in units if getattr(unit, 'hostUnit', None) is None)
            choice = 'Take first turn' if game.aiControls(owner) else await game.makeChoiceNew(
                ['Take first turn', 'Take second turn'], Point3(0, 0, 10), owner=owner,
                prompt=f'Player {winner}: first-turn choice')
            if choice not in ('Take first turn', 'Take second turn'):
                raise ValueError('Invalid first-turn choice')
            record['player'] = winner if choice == 'Take first turn' else 3 - winner
            rule_log('Battle March first turn', 'setup',
                     f'Player {winner} chooses {choice.lower()}; Player {record["player"]} starts')
        game.magicBusy = False
        game.fsm.request('StrategyPhase')
    finally:
        game.battleMarchSetupBusy = False
        game.magicBusy = False


def saved_battle(game):
    config = getattr(game, 'battle_config', None)
    setup = getattr(game, 'battle_setup', None)
    if config is None and setup is None:
        return None
    record = {'config': validate_config(config), 'setup': validate_setup(config, setup),
              'runtime': {'objectives': deepcopy(getattr(game, 'battle_objectives', objective_records(config, setup))),
                          'awards': deepcopy(getattr(game, 'battle_awards', [])),
                          'scored_turns': list(getattr(game, 'battle_scored_turns', []))}}
    from battle_secondary import empty_state
    record['runtime']['secondary'] = deepcopy(getattr(game, 'battle_secondary', empty_state()))
    return validate_saved_battle(record)


def objective_records(config, setup):
    """Published marker centres and base sizes (General's Companion pp. 24-25).

    Two troves lie 7.5 inches north/south of centre; three lie on the east/west
    centreline at -11, 0 and +11. These offsets do not scale with the table.
    """
    setup = validate_setup(config, setup)
    landmark = setup['objective_layout'] == 'landmark'
    positions = {'none': (), 'two_troves': ((0, -7.5), (0, 7.5)),
                 'three_troves': ((-11, 0), (0, 0), (11, 0)), 'landmark': ((0, 0),)}
    diameter = config['objectives']['landmark_base_mm' if landmark else 'trove_base_mm'] / 25.4
    return [{'id': f'objective-{index + 1}', 'kind': 'landmark' if landmark else 'trove',
             'center': list(point), 'diameter': diameter, 'property': setup['landmark_property'],
             'controller': None, 'player': None, 'contested': False, 'destroyed': False}
            for index, point in enumerate(positions[setup['objective_layout']])]


def validate_saved_battle(record):
    if record is None:
        return None
    _keys(record, 'config setup' + (' runtime' if 'runtime' in record else ''), 'battle_march')
    config = validate_config(record['config'])
    setup = validate_setup(config, record['setup'])
    expected = objective_records(config, setup)
    runtime = record.get('runtime', {'objectives': expected, 'awards': [], 'scored_turns': []})
    _keys(runtime, 'objectives awards scored_turns' + (' secondary' if 'secondary' in runtime else ''), 'battle_march.runtime')
    if not isinstance(runtime['objectives'], list) or len(runtime['objectives']) != len(expected):
        raise ConfigError('battle_march.runtime.objectives: expected the resolved marker list')
    for objective, original in zip(runtime['objectives'], expected):
        _keys(objective, ' '.join(original), 'battle_march.runtime.objectives')
        for field in ('id', 'kind', 'center', 'diameter', 'property'):
            if objective[field] != original[field]:
                raise ConfigError(f'battle_march.runtime.objectives.{field}: differs from resolved setup')
        for field in ('contested', 'destroyed'):
            _boolean(objective[field], f'battle_march.runtime.objectives.{field}')
        controller = objective['controller']
        if controller is not None and (not isinstance(controller, str) or not controller):
            raise ConfigError('battle_march.runtime.objectives.controller: expected a unit ID or null')
        if controller is None:
            if objective['player'] is not None:
                raise ConfigError('battle_march.runtime.objectives.player: uncontrolled objective has no player')
        else:
            _number(objective['player'], 'battle_march.runtime.objectives.player', 1, 2, integer=True)
            if objective['contested'] or objective['destroyed']:
                raise ConfigError('battle_march.runtime.objectives: contested/destroyed objective cannot have a controller')
    turns = runtime['scored_turns']
    if (not isinstance(turns, list) or any(not isinstance(turn, str) or not re.fullmatch(r'[12]:[0-9]+:[0-9]+', turn)
                                         for turn in turns) or len(turns) != len(set(turns))):
        raise ConfigError('battle_march.runtime.scored_turns: expected unique player-turn keys')
    if not isinstance(runtime['awards'], list):
        raise ConfigError('battle_march.runtime.awards: expected a list')
    seen = set()
    for award in runtime['awards']:
        _keys(award, 'turn objective player unit points rule reason', 'battle_march.runtime.awards')
        for field in ('turn', 'objective', 'unit', 'rule', 'reason'):
            if not isinstance(award[field], str) or not award[field]:
                raise ConfigError(f'battle_march.runtime.awards.{field}: expected non-empty text')
        _number(award['player'], 'battle_march.runtime.awards.player', 1, 2, integer=True)
        _number(award['points'], 'battle_march.runtime.awards.points', 0, 10000, integer=True)
        key = award['turn'], award['objective']
        if key in seen or award['turn'] not in turns or award['objective'] not in {entry['id'] for entry in expected}:
            raise ConfigError('battle_march.runtime.awards: duplicate or unknown objective/turn')
        seen.add(key)
    from battle_secondary import empty_state, validate_state
    runtime = deepcopy(runtime)
    runtime['secondary'] = validate_state(config, runtime.get('secondary', empty_state()), runtime['objectives'])
    return {'config': config, 'setup': setup, 'runtime': runtime}


def restore_battle(game, record):
    """Rebuild derived geometry only; old saves clear any previously loaded preset."""
    record = validate_saved_battle(record)
    previous_field = getattr(game, 'battlefield', STANDARD_BATTLEFIELD)
    game.battle_config = record['config'] if record else None
    game.battle_setup = record['setup'] if record else None
    game.battle_objectives = record['runtime']['objectives'] if record else []
    game.battle_awards = record['runtime']['awards'] if record else []
    game.battle_scored_turns = record['runtime']['scored_turns'] if record else []
    from battle_secondary import empty_state
    game.battle_secondary = record['runtime']['secondary'] if record else empty_state()
    if record and hasattr(game, 'roundCounter'):
        game.roundCounter.max_rounds = record['config']['game']['rounds']
    game.battlefield = (Battlefield(record['config']['battlefield']['width'],
                                   record['config']['battlefield']['depth'])
                        if record else STANDARD_BATTLEFIELD)
    legacy_line = getattr(game, 'deploymentLine', None)
    if legacy_line is not None:
        legacy_line.hide() if record else legacy_line.show()
    if previous_field != game.battlefield and hasattr(game, 'boundries'):
        from ClassOutOfBounds import OutOfBounds
        game.boundries.destroy()
        game.boundries = OutOfBounds(game)
    overlay = getattr(game, 'battlefield_overlay', None)
    if record is not None:
        draw_battlefield(game)
    elif overlay is not None:
        overlay.removeNode()
        game.battlefield_overlay = None
    from battle_objectives import sync_markers
    sync_markers(game)