"""Battle March preset validation (General's Companion pp. 23-27, 36-46).

The visual board remains 72 x 48 inches; these dimensions describe play only.
Loading a preset does not enable its rules or mutate a running battle.
"""

from copy import deepcopy
import json
from math import isfinite
from pathlib import Path


DEFAULT_PRESET = Path(__file__).parent / 'config' / 'battle_march.json'
DEPLOYMENT_MAPS = ('pitched_battle', 'close_encounter', 'opposed_flanks',
                   'meeting_engagement', 'mountain_pass', 'outflank')
MIRRORABLE_MAPS = ('close_encounter', 'opposed_flanks', 'meeting_engagement', 'outflank')
OBJECTIVE_LAYOUTS = ('two_troves', 'three_troves', 'landmark')
LANDMARK_PROPERTIES = ('magic_resistance', 'frenzy', 'stubborn')


class ConfigError(ValueError):
    """Invalid or unsupported battle configuration, with a field path."""


def _keys(record, names, path):
    if not isinstance(record, dict):
        raise ConfigError(f'{path}: expected an object')
    expected = set(names.split())
    missing, unknown = expected - record.keys(), record.keys() - expected
    if missing:
        raise ConfigError(f'{path}: missing {", ".join(sorted(missing))}')
    if unknown:
        raise ConfigError(f'{path}: unknown {", ".join(sorted(unknown))}')


def _number(value, path, minimum, maximum, *, integer=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not isfinite(value) or not minimum <= value <= maximum
            or (integer and not isinstance(value, int))):
        kind = 'integer' if integer else 'number'
        raise ConfigError(f'{path}: expected a finite {kind} from {minimum} to {maximum}')


def _choice(value, choices, path):
    if not isinstance(value, str) or value not in choices:
        raise ConfigError(f'{path}: expected one of {", ".join(choices)}')


def _boolean(value, path):
    if not isinstance(value, bool):
        raise ConfigError(f'{path}: expected true or false')


def _options(value, choices, path):
    if not isinstance(value, list):
        raise ConfigError(f'{path}: expected a list')
    for entry in value:
        _choice(entry, choices, path)
    if len(set(value)) != len(value):
        raise ConfigError(f'{path}: duplicate options')


def validate_config(record):
    """Return an independent validated record; no coercion or silent defaults."""
    _keys(record, 'schema_version ruleset source points_limit battlefield deployment '
          'terrain objectives game scoring army optional_rules', 'config')
    if type(record['schema_version']) is not int or record['schema_version'] != 1:
        raise ConfigError('schema_version: only version 1 is supported')
    _choice(record['ruleset'], ('battle_march_generals_companion',), 'ruleset')
    _keys(record['source'], 'url publication reviewed', 'source')
    for field, value in record['source'].items():
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f'source.{field}: expected non-empty text')
    _number(record['points_limit'], 'points_limit', 400, 750, integer=True)

    battlefield = record['battlefield']
    _keys(battlefield, 'width depth show_boundary show_deployment', 'battlefield')
    _number(battlefield['width'], 'battlefield.width', 44, 48)
    _number(battlefield['depth'], 'battlefield.depth', 30, 36)
    for field in ('show_boundary', 'show_deployment'):
        _boolean(battlefield[field], f'battlefield.{field}')

    deployment = record['deployment']
    _keys(deployment, 'map mirror method first_turn', 'deployment')
    _choice(deployment['map'], ('random', *DEPLOYMENT_MAPS), 'deployment.map')
    _boolean(deployment['mirror'], 'deployment.mirror')
    if deployment['mirror'] and deployment['map'] not in ('random', *MIRRORABLE_MAPS):
        raise ConfigError('deployment.mirror: this map has no alternate deployment')
    _choice(deployment['method'], ('alternating',), 'deployment.method')
    _choice(deployment['first_turn'], ('roll_off_choice',), 'deployment.first_turn')

    terrain = record['terrain']
    _keys(terrain, 'method feature_count recommended_max_span centre_clearance '
          'opponent_feature_clearance objective_clearance', 'terrain')
    _choice(terrain['method'], ('alternating', 'scattered'), 'terrain.method')
    _number(terrain['feature_count'], 'terrain.feature_count', 0, 50, integer=True)
    for field in ('recommended_max_span', 'centre_clearance',
                  'opponent_feature_clearance', 'objective_clearance'):
        _number(terrain[field], f'terrain.{field}', 0, battlefield['width'])
    if terrain['recommended_max_span'] == 0:
        raise ConfigError('terrain.recommended_max_span: must be positive')

    objectives = record['objectives']
    _keys(objectives, 'layout trove_base_mm landmark_base_mm control_distance '
          'minimum_unit_strength landmark_property', 'objectives')
    _choice(objectives['layout'], ('random', *OBJECTIVE_LAYOUTS), 'objectives.layout')
    _choice(objectives['landmark_property'], ('random', *LANDMARK_PROPERTIES),
            'objectives.landmark_property')
    for field in ('trove_base_mm', 'landmark_base_mm'):
        _number(objectives[field], f'objectives.{field}', 1, 300)
    _number(objectives['control_distance'], 'objectives.control_distance', 0, 48)
    _number(objectives['minimum_unit_strength'], 'objectives.minimum_unit_strength',
            1, 100, integer=True)

    game = record['game']
    _keys(game, 'rounds time_limit_minutes', 'game')
    _number(game['rounds'], 'game.rounds', 1, 20, integer=True)
    if game['time_limit_minutes'] is not None:
        _number(game['time_limit_minutes'], 'game.time_limit_minutes', 1, 1440)
    _keys(record['scoring'], 'trove_per_player_turn landmark_per_player_turn general '
          'captured_standard battle_standard_bearer', 'scoring')
    for field, value in record['scoring'].items():
        _number(value, f'scoring.{field}', 0, 10000, integer=True)

    army = record['army']
    _keys(army, 'minimum_units maximum_unit_strength maximum_character_fraction '
          'maximum_core_fraction maximum_special_fraction maximum_rare_mercenary_fraction '
          'restricted_options_allowance', 'army')
    for field in ('minimum_units', 'maximum_unit_strength', 'restricted_options_allowance'):
        _number(army[field], f'army.{field}', 0, 100, integer=True)
    for field in ('maximum_character_fraction', 'maximum_core_fraction',
                  'maximum_special_fraction', 'maximum_rare_mercenary_fraction'):
        _number(army[field], f'army.{field}', 0, 1)

    optional = record['optional_rules']
    _keys(optional, 'secondary_objectives secret_objectives random_happenings '
          'battle_march_magic_items', 'optional_rules')
    _options(optional['secondary_objectives'], ('raid_and_burn', 'baggage_carts'),
             'optional_rules.secondary_objectives')
    _options(optional['random_happenings'], ('disruptive_weather', 'wilderness_terrain', 'chaos_of_war'),
             'optional_rules.random_happenings')
    for field in ('secret_objectives', 'battle_march_magic_items'):
        _boolean(optional[field], f'optional_rules.{field}')
    return deepcopy(record)


def _unique_object(pairs):
    record = {}
    for key, value in pairs:
        if key in record:
            raise ConfigError(f'duplicate JSON key: {key}')
        record[key] = value
    return record


def load_config(path=DEFAULT_PRESET):
    """Load JSON independently of the launch directory when using the default preset."""
    with open(path, encoding='utf-8') as stream:
        try:
            record = json.load(stream, object_pairs_hook=_unique_object)
        except json.JSONDecodeError as error:
            raise ConfigError(f'{path}: {error}') from error
    return validate_config(record)