"""Versioned General's Companion presets, not live battle state."""

import json

import pytest

from battle_config import ConfigError, DEPLOYMENT_MAPS, load_config, validate_config


def test_published_defaults_and_independent_loads():
    config = load_config()
    assert config['points_limit'] == 500
    assert config['battlefield']['width'] == 44
    assert config['battlefield']['depth'] == 30
    assert config['game']['rounds'] == 5
    assert config['scoring']['landmark_per_player_turn'] == 25
    config['battlefield']['width'] = 48
    assert load_config()['battlefield']['width'] == 44


@pytest.mark.parametrize('section,field,value', [
    ('battlefield', 'width', 72), ('battlefield', 'depth', 29),
    ('battlefield', 'width', float('nan')), ('battlefield', 'depth', True),
    ('battlefield', 'show_boundary', 'true'), ('deployment', 'map', 'pitched'),
    ('game', 'rounds', 5.5), ('game', 'rounds', 0),
    ('scoring', 'general', -1), ('army', 'maximum_core_fraction', 35),
    ('optional_rules', 'random_happenings', ['unknown']),
    ('optional_rules', 'secondary_objectives', ['raid_and_burn', 'raid_and_burn']),
])
def test_invalid_values_identify_field(section, field, value):
    config = load_config()
    config[section][field] = value
    with pytest.raises(ConfigError, match=section + r'\.' + field):
        validate_config(config)


def test_unknown_missing_and_duplicate_keys(tmp_path):
    config = load_config()
    config['battlefield']['widht'] = 44
    with pytest.raises(ConfigError, match='unknown widht'):
        validate_config(config)
    del config['battlefield']['widht']
    del config['battlefield']['width']
    with pytest.raises(ConfigError, match='missing width'):
        validate_config(config)
    path = tmp_path / 'duplicate.json'
    path.write_text('{"schema_version": 1, "schema_version": 2}')
    with pytest.raises(ConfigError, match='duplicate JSON key'):
        load_config(path)


def test_large_battle_and_json_round_trip(tmp_path):
    config = load_config()
    config['points_limit'] = 750
    config['battlefield'].update(width=48, depth=36)
    config['terrain']['recommended_max_span'] = 8
    path = tmp_path / 'large.json'
    path.write_text(json.dumps(config))
    assert load_config(path) == config


def test_unknown_version_and_non_mirrorable_map():
    config = load_config()
    config['schema_version'] = True
    with pytest.raises(ConfigError, match='schema_version'):
        validate_config(config)
    config['schema_version'] = 1
    config['deployment'].update(map='mountain_pass', mirror=True)
    with pytest.raises(ConfigError, match='deployment.mirror'):
        validate_config(config)


@pytest.mark.parametrize('width,depth', [(44, 30), (48, 36)])
@pytest.mark.parametrize('map_name', [
    'pitched_battle', 'close_encounter', 'opposed_flanks',
    'meeting_engagement', 'mountain_pass', 'outflank',
])
def test_maps_preserve_rotated_opposite_zones(width, depth, map_name):
    from battlefield import Battlefield
    field = Battlefield(width, depth)
    first = field.deployment_zone(map_name, 1)
    second = field.deployment_zone(map_name, 2)
    assert second.vertices == tuple((-point[0], -point[1]) for point in first.vertices)
    assert all(field.contains_point(point) for point in first.outline + second.outline)
    assert all(first.contains_point(point) for point in first.outline)
    assert all(second.contains_point(point) for point in second.outline)
    assert not first.contains_point((0, 0))
    assert not second.contains_point((0, 0))


@pytest.mark.parametrize('map_name', ['close_encounter', 'opposed_flanks', 'meeting_engagement', 'outflank'])
def test_mirrors_reflect_geometry_and_containment(map_name):
    from battlefield import Battlefield
    field = Battlefield(44, 30)
    original = field.deployment_zone(map_name)
    mirror = field.deployment_zone(map_name, mirror=True)
    assert mirror.outline == tuple((-point[0], point[1]) for point in original.outline)
    assert mirror.contains_box((-18, -13, 1, 1, -10)) == original.contains_box((18, -13, 1, 1, 10))


def test_fixed_measurements_on_larger_field():
    from battlefield import Battlefield
    field = Battlefield(48, 36)
    assert field.deployment_zone('pitched_battle').vertices[-1][1] == -7.5
    assert field.deployment_zone('meeting_engagement').vertices[0][0] == -24 + 11
    assert field.deployment_zone('opposed_flanks').vertices[-1][1] == 18 - 18
    assert field.deployment_zone('mountain_pass').vertices[0][0] == 11
    assert field.deployment_zone('outflank').vertices[0][0] == -24 + 22


def test_rotated_base_and_circle_edge_not_just_centre_or_corners():
    from battlefield import Battlefield
    field = Battlefield(44, 30)
    assert field.contains_box((21, 0, 1, 1, 0))
    assert not field.contains_box((21, 0, 1, 1, 45))
    band = field.deployment_zone('pitched_battle')
    assert band.contains_box((0, -8.5, 1, 1, 0))
    assert not band.contains_box((0, -8.5, 1, 1, 45))
    zone = field.deployment_zone('close_encounter')
    from psychology import _box_corners
    box = (6, -6, 4, 1, 45)
    assert all(zone.contains_point(point) for point in _box_corners(*box))
    assert not zone.contains_box(box)
    assert zone.contains_box((10, -10, 1, 1, 45))


def test_standard_bounds_remain_unchanged():
    from types import SimpleNamespace
    from battlefield import Battlefield, battlefield_for
    field = battlefield_for(SimpleNamespace())
    assert (field.width, field.depth) == (72, 48)
    assert field.deployment_zone().vertices == ((-36, -24), (36, -24), (36, -12), (-36, -12))
    assert field.edge_distance((30, 0)) == 6
    assert battlefield_for(SimpleNamespace(battlefield=Battlefield(44, 30))).edge_distance((21, 0)) == 1


def test_deployment_uses_resolved_side_and_every_base(monkeypatch):
    from types import SimpleNamespace
    from battlefield import Battlefield
    import scouts
    unit = SimpleNamespace()
    game = SimpleNamespace(battlefield=Battlefield(44, 30), units=[], battle_setup={
        'deployment_map': 'mountain_pass', 'mirror': False, 'player_zones': {'1': 2, '2': 1}})
    monkeypatch.setattr(scouts, 'side_of', lambda *args, **kwargs: 1)
    boxes = [(-15, 0, 1, 1, 0)]
    monkeypatch.setattr(scouts, 'model_base_boxes', lambda unit: boxes)
    assert scouts.placement_error(game, unit) is None
    boxes.append((-21.5, 0, 1, 1, 0))
    assert 'battlefield' in scouts.placement_error(game, unit)
    boxes[-1] = (-10.5, 0, 1, 1, 0)
    assert 'deployment zone' in scouts.placement_error(game, unit)
    assert scouts.placement_error(game, unit, deployment_zone=False) is None


def test_physics_walls_match_playable_edges(monkeypatch):
    from types import SimpleNamespace
    from battlefield import Battlefield
    from ClassOutOfBounds import OutOfBounds
    walls = []
    monkeypatch.setattr(OutOfBounds, 'boundry', lambda self, pos, shape: walls.append((pos, shape)))
    OutOfBounds(SimpleNamespace(battlefield=Battlefield(44, 30)))
    assert walls[0][0][1] - walls[0][1].y == 15
    assert walls[1][0][1] + walls[1][1].y == -15
    assert walls[2][0][0] + walls[2][1].x == -22
    assert walls[3][0][0] - walls[3][1].x == 22


def test_charge_route_checks_smaller_playable_field():
    from battlefield import Battlefield
    from formed_skirmish_charge import ChargeRoute, path_error
    route = ChargeRoute((21, 0, 0), 0, (21, 0), 0, 0, 0, 16, 0, [(21, 0, .5, .5, 0)])
    assert path_error(route, [], []) is None
    assert path_error(route, [], [], battlefield=Battlefield(44, 30)) == 'Charge would leave the battlefield'


@pytest.mark.parametrize('map_name', [
    'pitched_battle', 'close_encounter', 'opposed_flanks',
    'meeting_engagement', 'mountain_pass', 'outflank',
])
def test_ai_candidates_and_overlay_share_zones(map_name):
    from random import Random
    from types import SimpleNamespace
    from panda3d.core import NodePath, GeomVertexReader
    from battlefield import Battlefield, deployment_candidate, deployment_zone_for, draw_battlefield
    game = SimpleNamespace(render=NodePath('render'), battlefield=Battlefield(44, 30), battle_setup={
        'deployment_map': map_name, 'mirror': False, 'player_zones': {'1': 2, '2': 1}})
    for player in (1, 2):
        zone = deployment_zone_for(game, player)
        for seed in range(10):
            assert zone.contains_point(deployment_candidate(game, player, Random(seed)))
    root = draw_battlefield(game)
    assert root.getNumChildren() == 3
    for player in (1, 2):
        geometry = root.find(f'deployment-player-{player}').node().getGeom(0).getVertexData()
        reader = GeomVertexReader(geometry, 'vertex')
        zone = deployment_zone_for(game, player)
        while not reader.isAtEnd():
            vertex = reader.getData3()
            assert zone.contains_point((vertex.x, vertex.y), 1e-5)
    draw_battlefield(game)
    assert root.isEmpty()
    assert game.render.getNumChildren() == 1
    game.render.removeNode()


def test_saved_setup_is_independent_and_restores_legacy_bounds():
    from types import SimpleNamespace
    from panda3d.core import NodePath
    from battle_setup import resolve_setup, restore_battle, saved_battle
    config = load_config()
    game = SimpleNamespace(render=NodePath('render'))
    setup = resolve_setup(config, 42)
    restore_battle(game, {'config': config, 'setup': setup})
    saved = json.loads(json.dumps(saved_battle(game)))
    config['battlefield']['width'] = 48
    setup['player_zones']['1'] = 2
    assert game.battlefield.width == 44
    assert game.battle_setup['player_zones']['1'] == 1
    restore_battle(game, saved)
    assert saved_battle(game) == saved
    restore_battle(game, None)
    assert saved_battle(game) is None
    assert game.battlefield.width == 72
    assert game.battlefield_overlay is None
    game.render.removeNode()


def test_setup_rejects_tampered_rolls_and_duplicate_zones(monkeypatch):
    from battle_setup import resolve_setup, validate_setup
    config = load_config()
    setup = resolve_setup(config, 3)
    assert validate_setup(config, setup) == setup
    setup['rolls']['deployment'] = 7
    with pytest.raises(ConfigError, match='setup.rolls.deployment'):
        validate_setup(config, setup)
    setup = resolve_setup(config, 3)
    setup['player_zones'] = {'1': 2, '2': 2}
    with pytest.raises(ConfigError, match='opposite zones'):
        validate_setup(config, setup)
    setup = resolve_setup(config, 3)
    monkeypatch.setattr('battle_setup.Random', lambda *args: pytest.fail('load rolled setup dice'))
    assert validate_setup(config, setup) == setup


def test_army_report_checks_per_selection_limits_and_exclusions():
    from types import SimpleNamespace
    from battle_config import army_report

    def member(name, category, points, count=1, strength=1, troop='Regular infantry', general=False):
        profile = SimpleNamespace(characteristics={'Category': category, 'Troop Type': troop},
                                  unit_strength=lambda: strength)
        return SimpleNamespace(unit=SimpleNamespace(name=name, model=profile, nmodels=count,
                               roster_metadata={'points_cost': points}), isGeneral=general)

    general = member('General', 'Characters', 125, general=True)
    core = member('Core', 'Core', 175, count=20)
    machine = member('Machine', 'Special', 100, troop='War machine', strength=3)
    units = [general, core, machine]
    report = army_report(load_config(), units, restricted_options=['Machine'], composition_verified=True)
    assert report['valid'] and report['qualifying_units'] == 2 and report['points'] == 400
    general.unit.roster_metadata['points_cost'] = 126
    core.unit.nmodels = 21
    machine.unit.model.characteristics['Troop Type'] = 'War beasts'
    report = army_report(load_config(), units, restricted_options=['Machine', 'Core upgrade'], composition_verified=True)
    assert not report['valid']
    assert len(report['violations']) == 4
    assert any('125' in reason for reason in report['violations'])
    assert any('Unit Strength 21' in reason for reason in report['violations'])
    assert report['qualifying_units'] == 1


def test_army_report_does_not_claim_missing_metadata_is_valid():
    from battle_config import army_report
    report = army_report(load_config(), [])
    assert len(report['unverified']) == 2
    assert not report['valid']


@pytest.mark.parametrize('width,depth', [(44, 30), (48, 36)])
@pytest.mark.parametrize('layout,positions', [
    ('two_troves', [[0, -7.5], [0, 7.5]]),
    ('three_troves', [[-11, 0], [0, 0], [11, 0]]),
    ('landmark', [[0, 0]]),
])
def test_objective_markers_follow_published_diagrams(width, depth, layout, positions):
    from battle_setup import objective_records, resolve_setup
    config = load_config()
    config['battlefield'].update(width=width, depth=depth)
    config['objectives']['layout'] = layout
    records = objective_records(config, resolve_setup(config, 5))
    assert [record['center'] for record in records] == positions
    assert all(record['diameter'] == (100 if layout == 'landmark' else 40) / 25.4 for record in records)
    assert all(record['controller'] is None and not record['destroyed'] for record in records)


def test_terrain_spacing_recommendations_and_first_scatter_contact():
    from shapely.geometry import box
    from battlefield import Battlefield
    from battle_terrain import placement_report, scatter_distance
    field, config = Battlefield(44, 30), load_config()
    terrain = box(-21, -14, -17, -10)
    assert not placement_report(field, config, terrain, [], 1)['errors']
    assert placement_report(field, config, box(-2, -2, 2, 2), [], 1)['errors']
    assert placement_report(field, config, terrain, [(box(-15, -14, -13, -10), 2)], 1)['errors']
    assert not placement_report(field, config, terrain, [(box(-15, -14, -13, -10), 1)], 1)['errors']
    wide = placement_report(field, config, box(-21, -14, -8, -12), [], 1)
    assert wide['warnings']
    assert scatter_distance(field, terrain, [], (-1, 0), 12) == pytest.approx(1)
    obstacle = box(-10, -14, -9, -10)
    assert scatter_distance(field, terrain, [obstacle], (1, 0), 20) == pytest.approx(7)


def test_minimum_terrain_shift_clears_fixed_objective():
    from shapely.geometry import Point, box
    from shapely.affinity import translate
    from battlefield import Battlefield
    from battle_terrain import objective_clearance_shift
    objective = {'center': [0, 0], 'diameter': 2}
    terrain = box(2, -1, 4, 1)
    shift = objective_clearance_shift(Battlefield(44, 30), terrain, [objective], 3)
    assert shift[0] == pytest.approx(2, abs=1e-5) and shift[1] == pytest.approx(0)
    assert translate(terrain, *shift).distance(Point(0, 0)) >= 4
    assert objective['center'] == [0, 0]


def test_objective_control_ties_and_explicit_one_object_choice():
    from battle_objectives import required_choices, resolve_control
    first = {'unit': 'alpha', 'name': 'Alpha', 'player': 1, 'distance': 3, 'strength': 5, 'reason': None}
    second = dict(first, unit='beta', name='Beta', player=2)
    snapshots = [{'objective': {'id': 'first'}, 'contenders': [first, second]}]
    assert resolve_control(snapshots, {})[0]['contested']
    second['strength'] = 6
    assert resolve_control(snapshots, {})[0]['controller'] == 'beta'
    second['distance'] = 3.01
    assert resolve_control(snapshots, {})[0]['controller'] == 'alpha'
    snapshots.append({'objective': {'id': 'second'}, 'contenders': [first]})
    assert required_choices(snapshots) == {'alpha': ['first', 'second']}
    with pytest.raises(ValueError, match='choose one objective'):
        resolve_control(snapshots, {})
    results = resolve_control(snapshots, {'alpha': 'second'})
    assert [result['controller'] for result in results] == ['beta', 'alpha']
    second['distance'] = 2
    assert required_choices(snapshots) == {}
    with pytest.raises(ValueError, match='only resolve multiple'):
        resolve_control(snapshots, {'alpha': 'first'})


def test_objective_awards_both_sides_once_per_player_turn(monkeypatch):
    from types import SimpleNamespace
    import battle_objectives
    config = load_config()
    objectives = [{'id': 'first', 'kind': 'trove'},
                  {'id': 'second', 'kind': 'landmark', 'property': 'stubborn'}]
    units = [SimpleNamespace(unitName=f'unit-{player}',
                             unit=SimpleNamespace(name=f'Unit {player}',
                                                  model=SimpleNamespace(special_rules=[])))
             for player in (1, 2)]
    game = SimpleNamespace(battle_config=config, battle_objectives=objectives, units=units,
                           roundCounter=SimpleNamespace(current_player=2, currentRoundPlayer=[0, 0]))
    snapshots = [{'objective': objective, 'contenders': [
        {'unit': f'unit-{player}', 'name': f'Unit {player}', 'player': player,
         'distance': 0, 'strength': 5, 'reason': None}]} for player, objective in enumerate(objectives, 1)]
    monkeypatch.setattr(battle_objectives, 'control_snapshot', lambda game: snapshots)
    awards = battle_objectives.score_turn(game)
    assert [(entry['player'], entry['points']) for entry in awards] == [(1, 10), (2, 25)]
    assert battle_objectives.score_turn(game) == []
    assert len(game.battle_awards) == 2
    assert len(units[1].unit.model.special_rules) == 1
    game.roundCounter.currentRoundPlayer[1] = 1
    game.roundCounter.current_player = 1
    assert len(battle_objectives.score_turn(game)) == 2
    assert len(units[1].unit.model.special_rules) == 1


def test_objective_real_base_distance_and_joined_strength(monkeypatch):
    from types import SimpleNamespace
    import battle_objectives
    profile = SimpleNamespace(unit_strength=lambda: 1, special_rules=[])
    hero = SimpleNamespace(unit=SimpleNamespace(model=profile, nmodels=1), hostUnit=None)
    unit = SimpleNamespace(unit=SimpleNamespace(model=profile, nmodels=4, name='Guard'),
                           unitName='guard', joinedCharacter=hero, state='InCombat', isDeployed=True)
    hero.hostUnit = unit
    objective = {'id': 'first', 'kind': 'trove', 'center': [0, 0], 'diameter': 2}
    game = SimpleNamespace(battle_config=load_config(), units=[unit, hero], battle_objectives=[objective])
    monkeypatch.setattr(battle_objectives, 'model_base_boxes', lambda unit: [(4.5, 0, .5, .5, 0)])
    monkeypatch.setattr(battle_objectives, 'side_of', lambda *args, **kwargs: 1)
    contenders = battle_objectives.control_snapshot(game)[0]['contenders']
    assert len(contenders) == 1
    assert contenders[0]['distance'] == 3 and contenders[0]['strength'] == 5
    assert contenders[0]['reason'] is None
    unit.state = 'IsFleeing'
    assert battle_objectives.control_snapshot(game)[0]['contenders'][0]['reason'] == 'fleeing'
    unit.state = 'Idle'
    profile.special_rules = [{'name': 'Stupidity'}]
    unit.stupidityFailed = True
    assert battle_objectives.control_snapshot(game)[0]['contenders'][0]['reason'] == 'succumbed to Stupidity'


def test_persisted_objective_awards_reject_double_scoring():
    from battle_setup import resolve_setup, validate_saved_battle
    config = load_config()
    record = validate_saved_battle({'config': config, 'setup': resolve_setup(config, 4)})
    runtime = record['runtime']
    runtime['scored_turns'] = ['2:0:0']
    award = {'turn': '2:0:0', 'objective': 'objective-1', 'player': 1, 'unit': 'guard',
             'points': 10, 'rule': 'Treasure Troves', 'reason': 'US 5 at 1 inch'}
    runtime['awards'] = [award]
    assert validate_saved_battle(record) == record
    runtime['awards'].append(dict(award))
    with pytest.raises(ConfigError, match='duplicate or unknown'):
        validate_saved_battle(record)


def test_frenzy_majority_joined_character_and_loss():
    from types import SimpleNamespace
    from frenzy import counts, lose_frenzy, majority
    from fear import immune, cannot_flee
    from psychology import PsychologySystem
    permanent = {'name': 'Frenzy'}
    unrelated = {'name': 'Stubborn', 'stubborn': True}
    host = SimpleNamespace(unit=SimpleNamespace(name='Guard', nmodels=5,
        model=SimpleNamespace(name='Guard', special_rules=[unrelated])), state='Idle')
    hero = SimpleNamespace(unit=SimpleNamespace(name='Hero', nmodels=1,
        model=SimpleNamespace(name='Hero', special_rules=[permanent])))
    host.joinedCharacter = hero
    assert counts(host) == (1, 6) and not majority(host)
    assert not immune(host)
    host.unit.model.special_rules.append({'name': 'Frenzy', 'frenzy': True})
    assert majority(host) and immune(host) and cannot_flee(host)
    assert '6/6' in PsychologySystem(None).panic_exempt_reason(host)
    import asyncio
    from fear import test_fear
    assert asyncio.run(test_fear(None, host, [], 'charge'))
    lose_frenzy(host)
    assert not majority(host) and not immune(host)
    assert permanent['frenzy_lost']
    assert host.unit.model.special_rules[0] is unrelated


def test_frenzy_attacks_respect_turn_and_split_profile():
    from types import SimpleNamespace
    from frenzy import attack_bonus
    from battleFunctions import attack_characteristic
    rider = SimpleNamespace(special_rules=[{'name': 'Frenzy'}], characteristics={'A': '2', 'Troop Type': 'Heavy cavalry'})
    mount = SimpleNamespace(special_rules=[], characteristics={'A': '1', 'Troop Type': 'War beasts'})
    rider.get_mount = lambda: mount
    host = SimpleNamespace(unit=SimpleNamespace(model=rider), chargedThisTurn=True)
    main = SimpleNamespace(host=host, role='main', profile=rider)
    horse = SimpleNamespace(host=host, role='mount', profile=mount)
    assert attack_bonus(main) == 1 and attack_bonus(horse) == 0
    assert attack_characteristic(rider, frenzy_bonus=attack_bonus(main)) == 3
    host.chargedThisTurn = False
    assert attack_bonus(main) == 0
    host.frenzyFollowUpThisTurn = True
    assert attack_bonus(main) == 1
    mount.characteristics['Troop Type'] = 'Behemoth'
    assert attack_bonus(main) == 0 and attack_bonus(horse) == 1
    rider.special_rules[0]['frenzy_lost'] = True
    assert attack_bonus(horse) == 0


@pytest.mark.parametrize('property_name', ['magic_resistance', 'frenzy', 'stubborn'])
def test_landmark_grants_expire_without_removing_permanent_sources(property_name):
    from types import SimpleNamespace
    from battle_objectives import refresh_landmark_grants
    from frenzy import has_frenzy, lose_frenzy
    permanent = {'name': 'Frenzy', 'frenzy': True}
    profile = SimpleNamespace(name='Guard', special_rules=[permanent])
    unit = SimpleNamespace(unitName='guard', unit=SimpleNamespace(name='Guard', model=profile))
    game = SimpleNamespace(units=[unit])
    objective = {'id': 'landmark', 'kind': 'landmark', 'controller': 'guard', 'property': property_name}
    lose_frenzy(unit)
    refresh_landmark_grants(game, [objective], '1:0:0')
    assert permanent['frenzy_lost']
    assert len(profile.special_rules) == 2
    grant = profile.special_rules[-1]
    assert grant['battle_march_source'] == 'landmark'
    if property_name == 'frenzy':
        assert has_frenzy(profile)
        lose_frenzy(unit)
        assert not has_frenzy(profile)
    refresh_landmark_grants(game, [], '2:1:0')
    assert profile.special_rules == [permanent]
    assert permanent['frenzy_lost']


def test_round_landmark_base_distance_and_swept_contact():
    from psychology import CircularObstacle, obb_distance
    from skirmish import swept_base_overlaps
    obstacle = CircularObstacle((0, 0), 2)
    corner = (1.9, 1.9, .1, .1, 0)
    assert obb_distance(corner, obstacle) > .5
    assert not swept_base_overlaps(corner, (3, 1.9, .1, .1, 0), obstacle)
    assert obb_distance((2.1, 0, .1, .1, 0), obstacle) == pytest.approx(0)
    assert swept_base_overlaps((-4, 0, .1, .1, 0), (4, 0, .1, .1, 0), obstacle)


def test_landmark_sight_blocks_circle_not_empty_corners():
    from types import SimpleNamespace
    from panda3d.core import Point3
    from skirmish_visibility import model_can_see
    from shooting_geometry import model_shot
    piece = SimpleNamespace(terrain_type='landmark', center=Point3(0, 0, 0), width=4)
    observer = (-5, 0, .05, .05, 0)
    target = (5, 0, .05, .05, 0)
    assert not model_can_see(observer, [target], terrain=[piece])
    assert model_shot(observer, [target], [], 24, terrain=[piece])[2] == 'no line of sight'
    assert model_can_see((-4, 7, .05, .05, 0), [(7, -4, .05, .05, 0)], terrain=[piece])
    assert not model_can_see((-4, 6.7, .01, .01, 0), [(6.7, -4, .01, .01, 0)], terrain=[piece])


def test_activation_rejects_unfinished_optional_handlers():
    from battle_config import validate_activation
    config = load_config()
    assert validate_activation(config) == config
    config['optional_rules']['secondary_objectives'] = ['raid_and_burn', 'baggage_carts']
    assert validate_activation(config) == config
    config['optional_rules']['secret_objectives'] = True
    assert validate_config(config) == config
    with pytest.raises(ConfigError, match='optional_rules.secret_objectives'):
        validate_activation(config)
    config['optional_rules']['secret_objectives'] = False
    config['game']['time_limit_minutes'] = 120
    with pytest.raises(ConfigError, match='game.time_limit_minutes'):
        validate_activation(config)


def test_preparation_state_requires_recorded_first_drop_and_army_reports():
    from battle_setup import resolve_setup, validate_setup
    config = load_config()
    setup = resolve_setup(config, 12)
    preparation = {'stage': 'armies', 'army_acknowledged': [], 'map_player': 1,
                   'first_drop_rolls': [], 'first_drop': None}
    setup['preparation'] = preparation
    assert validate_setup(config, setup) == setup
    preparation.update(stage='complete', army_acknowledged=[1, 2], first_drop=2,
                       first_drop_rolls=[[3, 3], [1, 5]])
    assert validate_setup(config, setup) == setup
    preparation['first_drop'] = 1
    with pytest.raises(ConfigError, match='first_drop'):
        validate_setup(config, setup)


def test_terrain_preparation_records_validate_pool_ownership_and_rolls():
    from battle_preparation import new_terrain_state, validate_terrain_state
    config = load_config()
    state = new_terrain_state()
    assert validate_terrain_state(config, state) == state
    state.update(selections={'1': ['hill'], '2': ['house']}, selection_complete=[1, 2],
                 rolls=[[1, 1], [2, 6]], winner=2,
                 placed=[{'type': 'house', 'width': 4, 'height': 4, 'center': [17, -12, 0],
                          'player': 2, 'pool_index': 1}])
    assert validate_terrain_state(config, state) == state
    state['placed'][0]['player'] = 1
    with pytest.raises(ConfigError, match='player'):
        validate_terrain_state(config, state)


def test_startup_preset_is_explicit_and_validated_before_window_creation():
    from battle_config import startup_options
    assert startup_options([]).battle_config is None
    options = startup_options(['--battle-config', '--battle-seed', '19', '--debug'])
    assert options.battle_config == load_config()
    assert options.battle_seed == 19 and options.debug
    for arguments in (['--battle-seed', '19'], ['--battle-config', '--battle-seed', '-1']):
        with pytest.raises(SystemExit) as error:
            startup_options(arguments)
        assert error.value.code == 2


@pytest.mark.parametrize('failure', [None, 'contact', 'strength', 'combat', 'fleeing'])
def test_raid_and_burn_next_own_start_and_once_only(monkeypatch, failure):
    from types import SimpleNamespace
    import battle_secondary
    import battle_objectives
    config = load_config()
    config['optional_rules']['secondary_objectives'] = ['raid_and_burn']
    unit = SimpleNamespace(unitName='raiders', unit=SimpleNamespace(nmodels=5, name='Raiders'),
                           state='Moved', isInCombat=False, hostUnit=None)
    objective = {'id': 'objective-1', 'kind': 'trove', 'center': [0, 0], 'diameter': 2,
                 'destroyed': False, 'controller': None, 'player': None, 'contested': False}
    game = SimpleNamespace(battle_config=config, battle_objectives=[objective], units=[unit],
                           fsm=SimpleNamespace(state='MovementPhase'), chargeStage='remaining',
                           roundCounter=SimpleNamespace(current_player=1, currentRoundPlayer=[0, 0]))
    monkeypatch.setattr(battle_secondary, 'side_of', lambda *args: 1)
    monkeypatch.setattr(battle_secondary, 'unit_strength_total', lambda member: member.unit.nmodels)
    monkeypatch.setattr(battle_secondary, 'in_contact', lambda *args: True)
    monkeypatch.setattr(battle_objectives, 'sync_markers', lambda game: None)
    battle_secondary.after_move(game, unit)
    battle_secondary.after_move(game, unit)
    assert len(game.battle_secondary['raid_attempts']) == 1
    assert battle_secondary.shooting_blocked(game, unit)
    assert not battle_secondary.spell_allowed(game, unit, 24)
    assert battle_secondary.spell_allowed(game, unit, 'Combat')
    assert battle_secondary.spell_allowed(game, unit, 'Self')
    game.roundCounter.current_player = 2
    game.roundCounter.currentRoundPlayer = [1, 0]
    battle_secondary.start_turn(game)
    assert not objective['destroyed']
    if failure == 'contact':
        monkeypatch.setattr(battle_secondary, 'in_contact', lambda *args: False)
    elif failure == 'strength':
        unit.unit.nmodels = 4
    elif failure == 'combat':
        unit.isInCombat = True
    elif failure == 'fleeing':
        unit.state = 'IsFleeing'
    game.roundCounter.current_player = 1
    game.roundCounter.currentRoundPlayer = [1, 1]
    battle_secondary.start_turn(game)
    battle_secondary.start_turn(game)
    assert objective['destroyed'] is (failure is None)
    assert len(game.battle_secondary['awards']) == (1 if failure is None else 0)
    assert not game.battle_secondary['raid_attempts']
    assert not battle_secondary.raiding(game, unit)
    assert battle_secondary.validate_state(config, game.battle_secondary, [objective]) == game.battle_secondary


def test_baggage_cart_published_split_profiles():
    from models import model
    cart = model('Baggage Cart', '')
    assert cart.is_chariot()
    assert cart.characteristics['Troop Type'] == 'Heavy Chariot'
    assert cart.characteristics['Base Size'] == '60x100'
    assert cart.get_base_size() == (60, 100)
    assert cart.characteristics['S'] == '4'
    assert cart.characteristics['T'] == '5'
    assert cart.starting_wounds() == 4
    assert cart.unit_strength() == 5
    assert cart.part_count('crew') == 1 and cart.part_count('beasts') == 2
    assert cart.get_crew().characteristics['WS'] == '2'
    assert cart.get_crew().characteristics['Ld'] == '6'
    assert cart.get_beasts().characteristics['M'] == '6'
    assert cart.armor_save == 5
    assert cart.characteristics['Ld'] == '6'


def test_non_combatant_declines_general_bsb_and_compulsory_charges(monkeypatch):
    from types import SimpleNamespace
    from models import model
    from psychology import PsychologySystem
    from impetuous import legal_targets
    from battle_secondary import non_combatant
    cart = SimpleNamespace(unit=SimpleNamespace(model=model('Baggage Cart', '')))
    assert non_combatant(cart)
    system = PsychologySystem(SimpleNamespace())
    monkeypatch.setattr(system, '_command_source', lambda *args: pytest.fail('cart borrowed command support'))
    assert system.general_of(cart) is None
    assert system.battle_standard_of(cart) is None
    assert system.leadership_of(cart) == (6, None)
    game = SimpleNamespace(movement=SimpleNamespace(movementParticipants=lambda member: [member]))
    assert legal_targets(game, cart) == []


@pytest.mark.parametrize('width,depth', [(44, 30), (48, 36)])
@pytest.mark.parametrize('map_name', DEPLOYMENT_MAPS)
@pytest.mark.parametrize('mirror', [False, True])
def test_cart_escape_only_enemy_zone_board_edge(width, depth, map_name, mirror):
    from battle_config import MIRRORABLE_MAPS
    from battle_secondary import cart_escape_edge
    from battlefield import Battlefield
    field = Battlefield(width, depth)
    if mirror and map_name not in MIRRORABLE_MAPS:
        with pytest.raises(ValueError, match='no alternate deployment'):
            field.deployment_zone(map_name, 2, mirror=mirror)
        return
    enemy = field.deployment_zone(map_name, 2, mirror=mirror)
    for horizontal, vertical in field.outline:
        base = (horizontal, vertical, .4, .8, 37)
        assert cart_escape_edge(field, enemy, [base]) is enemy.contains_point((horizontal, vertical))
    assert not cart_escape_edge(field, enemy, [(0, 0, .4, .8, 37)])


def test_cart_escape_sweep_cannot_jump_over_qualifying_edge():
    from battle_secondary import cart_escape_edge
    from battlefield import Battlefield
    field = Battlefield(44, 30)
    enemy = field.deployment_zone('pitched_battle', 2)
    assert cart_escape_edge(field, enemy, [(0, 25, 1, 2, 0)], [(0, 10, 1, 2, 0)])
    assert not cart_escape_edge(field, enemy, [(0, -25, 1, 2, 0)], [(0, -10, 1, 2, 0)])


@pytest.mark.parametrize('corruption', ['duplicate', 'missing', 'owner', 'escaped', 'disabled'])
def test_baggage_saved_state_rejects_inconsistent_cart_records(corruption):
    from battle_secondary import empty_state, validate_state
    config = load_config()
    config['optional_rules']['secondary_objectives'] = ['baggage_carts']
    state = empty_state()
    state['cart_mode'] = 'both'
    state['carts'] = [{'unit': f'cart-{player}', 'player': player, 'escaped': False} for player in (1, 2)]
    assert validate_state(config, state, []) == state
    if corruption == 'duplicate':
        state['carts'][1]['unit'] = 'cart-1'
    elif corruption == 'missing':
        state['carts'].pop()
    elif corruption == 'owner':
        state['carts'][1]['player'] = 1
    elif corruption == 'escaped':
        state['carts'][0]['escaped'] = 'yes'
    else:
        config['optional_rules']['secondary_objectives'] = []
    with pytest.raises(ConfigError):
        validate_state(config, state, [])


def test_rangers_glass_authenticity_and_modified_rolls_validate():
    from battle_setup import resolve_setup, validate_setup
    config = load_config()
    config['optional_rules']['battle_march_magic_items'] = True
    setup = resolve_setup(config, 12)
    setup['first_turn'] = {'glass': {'players': [1, 2], 'rolls': [[3, 3], [5, 2]], 'winner': 1},
                           'rolls': [[2, 3], [4, 4]], 'winner': 1, 'player': 2}
    assert validate_setup(config, setup) == setup
    setup['first_turn']['glass']['winner'] = 2
    with pytest.raises(ConfigError, match='authenticity'):
        validate_setup(config, setup)
    setup['first_turn']['glass']['winner'] = 1
    setup['first_turn']['winner'] = 2
    with pytest.raises(ConfigError, match='recorded dice'):
        validate_setup(config, setup)