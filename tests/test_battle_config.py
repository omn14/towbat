"""Versioned General's Companion presets, not live battle state."""

import json

import pytest

from battle_config import ConfigError, load_config, validate_config


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