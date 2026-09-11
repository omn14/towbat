"""Shared Miscast table damage and casting hooks (Rulebook pp. 95, 109-110)."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from panda3d.core import NodePath

from miscasts import miscast_targets, resolve_miscast_damage
from spell_system import Spell, miscast_result
from tests.test_battle_magic import _model, _unit


def damage_case():
    root = NodePath('world')
    members = [SimpleNamespace(unit=_unit(_model()), unitName=name, bodyNP=root.attachNewNode(name),
                               isDeployed=True) for name in ('Wizard', 'Friend', 'Enemy')]
    game = SimpleNamespace(units=members, player1Units=members[:2], player2Units=members[2:],
                           movement=Mock(), combat=Mock())
    return game, members


@pytest.mark.parametrize('roll,strength,ap', [(2, 10, 4), (4, 10, 4), (5, 6, 2), (6, 6, 2), (7, 4, 1)])
def test_table_damage_uses_its_own_strength_and_ap(roll, strength, ap):
    game, members = damage_case()
    wizard = members[0]
    with patch('miscasts.model_base_boxes', return_value=[(0, 0, .4, .4, 0)]), \
            patch('miscasts.resolve_magic_hits', return_value=(1, 0, 1)) as hits:
        result = resolve_miscast_damage(game, wizard, miscast_result(roll))
    expected = 1 if roll == 7 else 3
    assert len(result) == hits.call_count == expected
    assert all(call.args[1:] == (1, strength, ap) for call in hits.call_args_list)
    assert game.movement.applyWounds.call_count == expected


def test_blast_partial_rolls_central_hole_and_friendly_fire():
    game, members = damage_case()
    boxes = {id(members[0]): [(0, 0, 3, 3, 45)], id(members[1]): [(2, 0, .4, .4, 0)],
             id(members[2]): [(2.6, 0, .4, .4, 0)]}
    with patch('miscasts.model_base_boxes', side_effect=lambda member: boxes[id(member)]), \
            patch('miscasts.random.randint', return_value=3):
        assert miscast_targets(game, members[0], miscast_result(2), 'Miscast') == [(members[0], 1), (members[1], 1)]
    with patch('miscasts.model_base_boxes', side_effect=lambda member: boxes[id(member)]), \
            patch('miscasts.random.randint', return_value=4):
        assert len(miscast_targets(game, members[0], miscast_result(2), 'Miscast')) == 3


@pytest.mark.parametrize('roll', [8, 9, 10, 12])
def test_successful_table_rows_do_not_deal_damage(roll):
    game, members = damage_case()
    with patch('miscasts.resolve_magic_hits') as hits:
        assert resolve_miscast_damage(game, members[0], miscast_result(roll)) == []
    hits.assert_not_called()


def test_casting_miscast_calls_damage_once_and_never_applies_attempted_spell():
    spell = Spell('Test', 7)
    with patch.object(Spell, '_roll_casting_dice', AsyncMock(side_effect=[(2, [1, 1]), (7, [3, 4])])), \
            patch('miscasts.resolve_miscast_damage') as damage, \
            patch.object(spell, 'apply', AsyncMock()) as effect:
        asyncio.run(spell.spellFunction(None))
    damage.assert_called_once_with(None, None, miscast_result(7))
    effect.assert_not_awaited()


@pytest.mark.parametrize('dice,unsaved', [([6, 6], 0), ([6, 1, 5], 0),
                                        ([6, 1, 1, 4], 0), ([6, 1, 1, 1], 1), ([1], 0)])
def test_actual_armour_ward_regeneration_and_failed_wounding(dice, unsaved):
    game, members = damage_case()
    wizard = members[0]
    wizard.unit.model.characteristics['T'] = 3
    wizard.unit.model.armor_save = 4
    wizard.unit.model.special_rules = [{'name': 'Ward', 'ward': 5}, {'name': 'Regeneration', 'regen': 4}]
    with patch.object(wizard.unit.model, 'melee_armour_save', return_value=4), \
            patch('battleFunctions.random.randint', side_effect=dice):
        assert resolve_miscast_damage(game, wizard, miscast_result(7)) == [(wizard, unsaved)]
    assert game.movement.applyWounds.call_count == unsaved


def test_bound_double_one_never_enters_miscast_damage():
    spell = Spell('Test', 7)
    spell.bound = True
    with patch.object(Spell, '_roll_casting_dice', AsyncMock(return_value=(2, [1, 1]))) as dice, \
            patch('miscasts.resolve_miscast_damage') as damage:
        assert not asyncio.run(spell._attempt(None))
    assert dice.await_count == 1
    damage.assert_not_called()