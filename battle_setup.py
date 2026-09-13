"""Resolved Battle March setup, separate from editable rules (Companion pp. 24-27)."""

from copy import deepcopy
from random import Random
import re

from battle_config import (ConfigError, DEPLOYMENT_MAPS, LANDMARK_PROPERTIES,
                           MIRRORABLE_MAPS, OBJECTIVE_LAYOUTS, _boolean,
                           _choice, _keys, _number, validate_config)
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
    _keys(setup, 'version seed rolls deployment_map mirror player_zones objective_layout landmark_property', 'setup')
    if type(setup['version']) is not int or setup['version'] != 1:
        raise ConfigError('setup.version: only version 1 is supported')
    _number(setup['seed'], 'setup.seed', 0, 2 ** 53 - 1, integer=True)
    _choice(setup['deployment_map'], DEPLOYMENT_MAPS, 'setup.deployment_map')
    _choice(setup['objective_layout'], OBJECTIVE_LAYOUTS, 'setup.objective_layout')
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
    return deepcopy(setup)


def saved_battle(game):
    config = getattr(game, 'battle_config', None)
    setup = getattr(game, 'battle_setup', None)
    if config is None and setup is None:
        return None
    record = {'config': validate_config(config), 'setup': validate_setup(config, setup),
              'runtime': {'objectives': deepcopy(getattr(game, 'battle_objectives', objective_records(config, setup))),
                          'awards': deepcopy(getattr(game, 'battle_awards', [])),
                          'scored_turns': list(getattr(game, 'battle_scored_turns', []))}}
    return validate_saved_battle(record)


def objective_records(config, setup):
    """Published marker centres and base sizes (General's Companion pp. 24-25).

    Two troves lie 7.5 inches north/south of centre; three lie on the east/west
    centreline at -11, 0 and +11. These offsets do not scale with the table.
    """
    setup = validate_setup(config, setup)
    landmark = setup['objective_layout'] == 'landmark'
    positions = {'two_troves': ((0, -7.5), (0, 7.5)),
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
    _keys(runtime, 'objectives awards scored_turns', 'battle_march.runtime')
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
    return {'config': config, 'setup': setup, 'runtime': deepcopy(runtime)}


def restore_battle(game, record):
    """Rebuild derived geometry only; old saves clear any previously loaded preset."""
    record = validate_saved_battle(record)
    previous_field = getattr(game, 'battlefield', STANDARD_BATTLEFIELD)
    game.battle_config = record['config'] if record else None
    game.battle_setup = record['setup'] if record else None
    game.battle_objectives = record['runtime']['objectives'] if record else []
    game.battle_awards = record['runtime']['awards'] if record else []
    game.battle_scored_turns = record['runtime']['scored_turns'] if record else []
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