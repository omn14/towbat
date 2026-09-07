"""Vanguard (Rulebook p. 180 and Official FAQ)."""

from types import SimpleNamespace

import pytest

from tests.test_scouts import make_game, make_unit
from panda3d.core import NodePath
from special_rules import apply_rule_keywords
from vanguard import (vanguard_candidates, vanguard_charge_blocked,
                      vanguard_flying, vanguard_movement, vanguard_unavailable)


def test_vanguard_keyword_is_distinct_from_scouts():
    profile = SimpleNamespace(characteristics={}, special_rules=[])
    apply_rule_keywords(profile, ['vanguard'], replace=True)
    assert any(rule.get('vanguard') for rule in profile.special_rules)
    assert not any(rule.get('scouts') for rule in profile.special_rules)


@pytest.fixture
def troops():
    root = NodePath('vanguard-tests')
    host = make_unit(root, 'Vanguard', scouts=False, deployed=True)
    character = make_unit(root, 'Character', scouts=False, deployed=True)
    apply_rule_keywords(host.unit.model, ['Vanguard'], replace=True)
    yield make_game([host, character], []), host, character
    root.removeNode()


def test_keyword_does_not_bar_charges(troops):
    game, host, _ = troops
    assert not vanguard_charge_blocked(game, host)
    assert vanguard_unavailable(host) is None


@pytest.mark.parametrize('skirmishing,allowed', [(False, False), (True, True)])
def test_non_vanguard_character_faq(troops, skirmishing, allowed):
    _, host, character = troops
    host.joinedCharacter = character
    host.isSkirmisher = skirmishing
    assert (vanguard_unavailable(host) is None) is allowed


def test_scouts_history_not_keyword_excludes_vanguard(troops):
    _, host, _ = troops
    apply_rule_keywords(host.unit.model, ['Vanguard', 'Scouts'], replace=True)
    assert vanguard_unavailable(host) is None
    host.deployedAsScouts = True
    assert 'Scouts' in vanguard_unavailable(host)


def test_slowest_vanguard_character_limits_move(troops):
    _, host, character = troops
    host.joinedCharacter = character
    apply_rule_keywords(character.unit.model, ['Vanguard'], replace=True)
    host.unit.model.characteristics['M'] = '7'
    character.unit.model.characteristics['M'] = '3'
    assert vanguard_movement(host) == 3


def test_skirmishers_leave_slow_non_vanguard_character(troops):
    _, host, character = troops
    host.joinedCharacter = character
    host.isSkirmisher = True
    host.unit.model.characteristics['M'] = '7'
    character.unit.model.characteristics['M'] = '3'
    assert vanguard_movement(host) == 7


@pytest.mark.parametrize('turns,blocked', [([0, 0], True), ([1, 0], True), ([1, 1], False)])
def test_charge_bar_uses_owners_completed_turns(troops, turns, blocked):
    game, host, _ = troops
    game.player1Units.remove(host)
    game.player2Units.append(host)
    host.madeVanguardMove = True
    game.roundCounter.currentRoundPlayer = turns
    assert vanguard_charge_blocked(game, host) is blocked


def test_candidates_exclude_completed_and_blocked_units(troops):
    game, host, character = troops
    assert vanguard_candidates(game, 1) == [host]
    host.vanguardDone = True
    assert vanguard_candidates(game, 1) == []
    assert character not in vanguard_candidates(game, 1)


def test_fly_requires_every_participating_model(troops):
    _, host, character = troops
    host.joinedCharacter = character
    apply_rule_keywords(host.unit.model, ['Vanguard', 'Fly (9)'], replace=True)
    apply_rule_keywords(character.unit.model, ['Vanguard'], replace=True)
    host.unit.model.characteristics['M'] = '7'
    character.unit.model.characteristics['M'] = '3'
    assert not vanguard_flying(host)
    assert vanguard_movement(host) == 3
    apply_rule_keywords(character.unit.model, ['Fly (6)'])
    assert vanguard_flying(host)
    assert vanguard_movement(host) == 6