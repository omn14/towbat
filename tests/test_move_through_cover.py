"""Move Through Cover, Rulebook p. 174 and Official FAQ v1.5.3."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from panda3d.core import NodePath, Point3

from movement_system import MovementSystem
from combat_resolution import CombatResolver
from special_rules import apply_rule_keywords
from tests.test_scouts import make_unit


@pytest.fixture
def troops():
    root = NodePath('cover-tests')
    host = make_unit(root, 'Rangers', scouts=False, deployed=True)
    character = make_unit(root, 'Character', scouts=False, deployed=True)
    apply_rule_keywords(host.unit.model, ['Move Through Cover'], replace=True)
    apply_rule_keywords(character.unit.model, [], replace=True)
    host.unit.nmodels = 2
    character.unit.nmodels = 1
    host.joinedCharacter = character
    terrain = Mock(movement_modifier=-1, terrain_type='marsh')
    game = SimpleNamespace(terrain_manager=SimpleNamespace(
        get_terrain_between=lambda *args: [terrain],
        dangerous_between=lambda *args: [terrain]), remainsInPlay=[])
    movement = MovementSystem(game)
    movement.applyWounds = Mock()
    yield movement, host, character
    root.removeNode()


def test_rule_is_registered_and_not_shared_with_character(troops):
    _, host, character = troops
    assert host.unit.model.is_move_through_cover()
    assert not character.unit.model.is_move_through_cover()


def test_only_own_models_reroll_and_character_takes_own_wound(troops, capsys):
    movement, host, character = troops
    with patch('terrain_system.random.randint', side_effect=[1, 2, 1, 6, 1]):
        assert movement.dangerousTerrainTests(host, Point3(0), Point3(1)) == 1
    assert movement.applyWounds.call_args_list == [((host, 0),), ((character, 1),)]
    output = capsys.readouterr().out
    assert 'Move Through Cover' in output
    assert '2 mishap(s) avoided' in output


def test_character_rule_does_not_protect_rank_and_file(troops):
    movement, host, character = troops
    apply_rule_keywords(host.unit.model, [], replace=True)
    apply_rule_keywords(character.unit.model, ['Move Through Cover'], replace=True)
    with patch('terrain_system.random.randint', side_effect=[1, 1, 1, 6]):
        assert movement.dangerousTerrainTests(host, Point3(0), Point3(1)) == 2
    assert movement.applyWounds.call_args_list == [((host, 2),), ((character, 0),)]


@pytest.mark.parametrize('host_m,character_m,protected,expected', [
    (4, 4, False, 3), (3, 5, False, 3), (4, 3, False, 2),
    (4, 4, True, 4), (1, 1, False, 1),
])
def test_slowest_adjusted_model_faq(troops, host_m, character_m, protected, expected):
    movement, host, character = troops
    host.unit.model.characteristics['M'] = str(host_m)
    character.unit.model.characteristics['M'] = str(character_m)
    if protected:
        apply_rule_keywords(character.unit.model, ['Move Through Cover'])
    assert movement.movementAllowance(host, Point3(0), Point3(1)) == expected
    assert movement.movementAllowance(host) == min(host_m, character_m)


def test_all_protected_ignore_penalty_without_mutating_profile(troops):
    movement, host, _ = troops
    host.joinedCharacter = None
    host.unit.model.characteristics['M'] = '4'
    assert movement.pathTerrainModifier(host, Point3(0), Point3(1)) == 0
    assert host.unit.model.characteristics['M'] == '4'


@pytest.mark.parametrize('protected,expected', [(True, 9), (False, 5)])
def test_charge_faq_keeps_high_die_only_for_protected_unit(troops, protected, expected, capsys):
    movement, host, character = troops
    host.unit.model.characteristics['M'] = '4'
    character.unit.model.characteristics['M'] = '4'
    if protected:
        apply_rule_keywords(character.unit.model, ['Move Through Cover'])
    movement.game.terrain_manager.crosses_difficult = lambda *args: True
    movement.game.movement = movement
    movement.game.playerNP = NodePath('target')
    combat = CombatResolver(movement.game)
    assert combat.chargeDistance(host, Point3(0), [2, 5]) == expected
    assert 'Move Through Cover' in capsys.readouterr().out


@pytest.mark.parametrize('tag,count,expected', [
    ('mount', 1, True), ('crew', 1, True), ('beasts', 1, True),
    ('crew', 0, False), ('beasts', 0, False),
])
def test_split_profile_rules_only_from_live_parts(troops, tag, count, expected):
    _, host, character = troops
    apply_rule_keywords(host.unit.model, [], replace=True)
    apply_rule_keywords(character.unit.model, ['Move Through Cover'], replace=True)
    key = 'mountUnit' if tag == 'mount' else 'partUnit'
    host.unit.model.special_rules.append({'tag': tag, key: character.unit, 'count': count})
    assert host.unit.model.is_move_through_cover() is expected


def test_no_ones_logs_why_reroll_did_not_fire(troops, capsys):
    movement, host, _ = troops
    with patch('terrain_system.random.randint', return_value=2):
        assert movement.dangerousTerrainTests(host, Point3(0), Point3(1)) == 0
    assert 'no 1s to re-roll' in capsys.readouterr().out


def test_swiftstride_bonus_is_not_discarded_and_pursuit_is_unchanged(troops):
    movement, host, _ = troops
    host.joinedCharacter = None
    host.unit.model.characteristics['M'] = '4'
    movement.game.terrain_manager.crosses_difficult = lambda *args: True
    movement.game.movement = movement
    movement.game.playerNP = NodePath('target')
    combat = CombatResolver(movement.game)
    assert combat.chargeDistance(host, Point3(0), [2, 5, 6]) == 15
    host.state = 'IsPursuing'
    assert combat.chargeDistance(host, Point3(0), [2, 5, 6]) == 13


def test_open_and_flying_movement_have_no_terrain_penalty(troops):
    movement, host, character = troops
    host.unit.model.characteristics['M'] = '4'
    character.unit.model.characteristics['M'] = '4'
    movement.game.terrain_manager = None
    assert movement.movementAllowance(host, Point3(0), Point3(1)) == 4
    apply_rule_keywords(host.unit.model, ['Fly (9)'])
    apply_rule_keywords(character.unit.model, ['Fly (6)'])
    assert movement.movementAllowance(host, Point3(0), Point3(1)) == 6