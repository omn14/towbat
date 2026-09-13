"""Resolved Battle March setup, separate from editable rules (Companion pp. 24-27)."""

from copy import deepcopy
from random import Random

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
    return {'config': validate_config(config), 'setup': validate_setup(config, setup)}


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
             'controller': None, 'contested': False, 'destroyed': False}
            for index, point in enumerate(positions[setup['objective_layout']])]


def validate_saved_battle(record):
    if record is None:
        return None
    _keys(record, 'config setup', 'battle_march')
    config = validate_config(record['config'])
    return {'config': config, 'setup': validate_setup(config, record['setup'])}


def restore_battle(game, record):
    """Rebuild derived geometry only; old saves clear any previously loaded preset."""
    record = validate_saved_battle(record)
    previous_field = getattr(game, 'battlefield', STANDARD_BATTLEFIELD)
    game.battle_config = record['config'] if record else None
    game.battle_setup = record['setup'] if record else None
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