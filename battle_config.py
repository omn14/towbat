"""Battle March preset validation (General's Companion pp. 23-27, 36-46).

The visual board remains 72 x 48 inches; these dimensions describe play only.
Loading a preset does not enable its rules or mutate a running battle.
"""

from copy import deepcopy
import json
from math import isfinite
from pathlib import Path


DEFAULT_PRESET = Path(__file__).parent / 'config' / 'battle_march.json'
REED_FENS_MAP = 'Grow Legue 2026 - Flank of the Reed Fens'
REED_FENS_PRESET = DEFAULT_PRESET.with_name('grow_legue_2026_reed_fens.json')
REED_FENS_TERRAIN = (
    {'type': 'marsh', 'center': [-8, -19, 0], 'width': 6, 'height': 4,
     'going': 'dangerous', 'footprint_shape': 'ellipse'},
    {'type': 'marsh', 'center': [8, 19, 0], 'width': 6, 'height': 4,
     'going': 'dangerous', 'footprint_shape': 'ellipse'},
    {'type': 'house', 'center': [-2, -7, 0], 'width': 4, 'height': 3, 'going': 'impassable'},
    {'type': 'house', 'center': [2, 7, 0], 'width': 4, 'height': 3, 'going': 'impassable'},
)
CUSTOM_DEPLOYMENT_MAPS = (REED_FENS_MAP,)
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
    roster_fields = ' rosters' if isinstance(record, dict) and 'rosters' in record else ''
    _keys(record, 'schema_version ruleset source points_limit battlefield deployment '
          'terrain objectives game scoring army optional_rules' + roster_fields, 'config')
    if roster_fields:
        _keys(record['rosters'], 'player1 player2', 'rosters')
        for player, path in record['rosters'].items():
            if path is not None and (not isinstance(path, str) or not path.strip()):
                raise ConfigError(f'rosters.{player}: expected a roster path or null')
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
    deployment = record['deployment']
    _keys(deployment, 'map mirror method first_turn', 'deployment')
    custom = deployment['map'] == REED_FENS_MAP
    _number(battlefield['width'], 'battlefield.width', 30 if custom else 44, 30 if custom else 48)
    _number(battlefield['depth'], 'battlefield.depth', 44 if custom else 30, 44 if custom else 36)
    for field in ('show_boundary', 'show_deployment'):
        _boolean(battlefield[field], f'battlefield.{field}')

    _choice(deployment['map'], ('random', *DEPLOYMENT_MAPS, *CUSTOM_DEPLOYMENT_MAPS), 'deployment.map')
    _boolean(deployment['mirror'], 'deployment.mirror')
    if deployment['mirror'] and deployment['map'] not in ('random', *MIRRORABLE_MAPS):
        raise ConfigError('deployment.mirror: this map has no alternate deployment')
    _choice(deployment['method'], ('alternating',), 'deployment.method')
    _choice(deployment['first_turn'], ('roll_off_choice',), 'deployment.first_turn')

    terrain = record['terrain']
    _keys(terrain, 'method feature_count recommended_max_span centre_clearance '
          'opponent_feature_clearance objective_clearance', 'terrain')
    _choice(terrain['method'], ('fixed',) if custom else ('alternating', 'scattered'), 'terrain.method')
    _number(terrain['feature_count'], 'terrain.feature_count', 0, 50, integer=True)
    if custom and terrain['feature_count'] != len(REED_FENS_TERRAIN):
        raise ConfigError('terrain.feature_count: Reed Fens has exactly four fixed features')
    for field in ('recommended_max_span', 'centre_clearance',
                  'opponent_feature_clearance', 'objective_clearance'):
        _number(terrain[field], f'terrain.{field}', 0, battlefield['width'])
    if terrain['recommended_max_span'] == 0:
        raise ConfigError('terrain.recommended_max_span: must be positive')

    objectives = record['objectives']
    _keys(objectives, 'layout trove_base_mm landmark_base_mm control_distance '
          'minimum_unit_strength landmark_property', 'objectives')
    _choice(objectives['layout'], ('none',) if custom else ('random', *OBJECTIVE_LAYOUTS), 'objectives.layout')
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


def validate_activation(record):
    """Reject configured features until their complete runtime handlers are available."""
    config = validate_config(record)
    unsupported = [f'optional_rules.{name}' for name, value in config['optional_rules'].items()
                   if value and name != 'secondary_objectives']
    if config['game']['time_limit_minutes'] is not None:
        unsupported.append('game.time_limit_minutes')
    if unsupported:
        raise ConfigError('Runtime support is not complete for: ' + ', '.join(unsupported))
    return config


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


def save_config(path, record):
    """Replace a preset atomically only after schema validation succeeds."""
    import os
    from tempfile import NamedTemporaryFile
    config = validate_config(record)
    destination = Path(path).expanduser().resolve()
    temporary = None
    try:
        with NamedTemporaryFile(mode='w', encoding='utf-8', dir=destination.parent,
                                prefix=f'.{destination.name}.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(config, stream, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return config


def startup_options(arguments=None):
    """Parse explicit opt-in without changing ordinary game startup."""
    import argparse
    parser = argparse.ArgumentParser(description='Warhammer: The Old World battle engine')
    parser.add_argument('--battle-config', nargs='?', const=str(DEFAULT_PRESET), metavar='PATH',
                        help='open Battle March configuration using the default preset or a JSON file')
    parser.add_argument('--battle-seed', type=int, help='repeatable Battle March setup seed')
    parser.add_argument('--debug', action='store_true', help='enable developer tools')
    options = parser.parse_args(arguments)
    if options.battle_seed is not None and options.battle_config is None:
        parser.error('--battle-seed requires --battle-config')
    options.battle_config_path = options.battle_config
    try:
        if options.battle_config is not None:
            options.battle_config = load_config(options.battle_config)
        if options.battle_seed is not None:
            _number(options.battle_seed, 'battle_seed', 0, 2 ** 53 - 1, integer=True)
    except (ConfigError, OSError) as error:
        parser.error(str(error))
    return options


def army_report(config, units, *, restricted_options=None, composition_verified=False):
    """Report mustering limits without mutating armies (General's Companion p. 23).

    Imported selections do not contain evaluated faction composition constraints.
    Callers must supply a verified restricted-option list, including upgrades,
    rather than assuming absence from an export means there are no restrictions.
    """
    from troop_types import normalise
    config = validate_config(config)
    limits, budget = config['army'], config['points_limit']
    violations, unverified = [], []
    total, qualifying, generals = 0, 0, 0
    fractions = {'characters': 'maximum_character_fraction', 'core': 'maximum_core_fraction',
                 'special': 'maximum_special_fraction', 'rare': 'maximum_rare_mercenary_fraction',
                 'mercenaries': 'maximum_rare_mercenary_fraction', 'mercenary': 'maximum_rare_mercenary_fraction'}
    members, seen = list(units), set()
    for member in members:
        joined = getattr(member, 'joinedCharacter', None)
        if joined is not None and all(joined is not entry for entry in members):
            members.append(joined)
    for member in members:
        if id(member) in seen:
            continue
        seen.add(id(member))
        group, name = member.unit, member.unit.name
        metadata = getattr(group, 'roster_metadata', {})
        profile = group.model
        category = str(metadata.get('category') or profile.characteristics.get('Category') or '').lower()
        character = category == 'characters'
        troop_type = normalise(profile.characteristics.get('Troop Type'))
        qualifying += int(not character and troop_type not in ('swarms', 'war beasts'))
        generals += int(character and getattr(member, 'isGeneral', False))
        if not troop_type:
            unverified.append(f'{name}: missing troop type for minimum-unit eligibility')
        points = metadata.get('points_cost')
        if isinstance(points, bool) or not isinstance(points, (int, float)) or not isfinite(points) or points < 0:
            unverified.append(f'{name}: missing or invalid paid points')
        else:
            total += points
            if category in fractions:
                maximum = budget * limits[fractions[category]]
                if points > maximum:
                    violations.append(f'{name}: {points:g} points exceeds the {category} single-selection cap {maximum:g}')
            else:
                unverified.append(f'{name}: unknown army category {category!r}')
        count = getattr(member, 'startOfBattleModels', group.nmodels)
        strength = count * profile.unit_strength()
        if strength > limits['maximum_unit_strength']:
            violations.append(f'{name}: starting Unit Strength {strength:g} exceeds {limits["maximum_unit_strength"]}')
    if total > budget:
        violations.append(f'Army: {total:g} points exceeds the agreed {budget}-point limit')
    if qualifying < limits['minimum_units']:
        violations.append(f'Army: {qualifying} qualifying units; at least {limits["minimum_units"]} required')
    if generals != 1:
        violations.append(f'Army: requires one designated General, found {generals}')
    if restricted_options is None:
        unverified.append('Per-1,000 restricted options: evaluated selected units, characters and upgrades required')
    elif len(restricted_options) > limits['restricted_options_allowance']:
        violations.append(f'Army: {len(restricted_options)} per-1,000 restricted options exceeds '
                          f'{limits["restricted_options_allowance"]}: {", ".join(restricted_options)}')
    if not composition_verified:
        unverified.append('Normal faction composition must still be verified against the selected army list')
    return {'points': total, 'qualifying_units': qualifying, 'violations': violations,
            'unverified': unverified, 'valid': not violations and not unverified}