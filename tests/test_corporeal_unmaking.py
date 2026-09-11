"""Corporeal Unmaking's Ward-only damage (Rulebook p. 329)."""

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from battleFunctions import resolve_magic_hits
from combat_resolution import CombatResolver
from high_magic import CorporealUnmakingSpell
from spell_system import spell_class
from tests.test_battle_magic import _model, _unit


@pytest.mark.parametrize('ward_roll, saved', [(4, 0), (5, 1)])
def test_ward_only_magic_never_rolls_armour_or_regeneration(ward_roll, saved):
    target = _unit(_model(ward=5, regen=2))
    target.model.armor_save = 2
    with patch('battleFunctions.random.randint', side_effect=[6, ward_roll]) as dice, \
            patch('battleFunctions.check_armor_save') as armour:
        result = resolve_magic_hits(target, 1, 5, 0,
                                    allow_armour=False, allow_regeneration=False)
    assert result == (1, saved, 1 - saved)
    assert dice.call_count == 2
    armour.assert_not_called()


def test_ward_only_magic_without_ward_suffers_each_wound():
    target = _unit(_model(regen=2))
    target.model.armor_save = 2
    with patch('battleFunctions.random.randint', side_effect=[6, 6, 6]) as dice:
        assert resolve_magic_hits(target, 3, 5, 0, allow_armour=False,
                                  allow_regeneration=False) == (3, 0, 3)
    assert dice.call_count == 3


def test_ordinary_magic_retains_armour_and_regeneration():
    target = _unit(_model(regen=2))
    with patch('battleFunctions.random.randint', return_value=6), \
            patch('battleFunctions.check_armor_save', side_effect=[False, True]) as armour:
        assert resolve_magic_hits(target, 1, 5, 0) == (1, 1, 0)
    assert armour.call_count == 2


def test_registry_and_fixed_spell_damage_do_not_trigger_shooting_panic(capsys):
    assert spell_class('Corporeal Unmaking') is CorporealUnmakingSpell
    target = SimpleNamespace(unit=_unit(_model(ward=5, regen=2)))
    caster = SimpleNamespace(unit=_unit(_model()))
    game = SimpleNamespace(movement=Mock(), psychology=Mock())
    spell = CorporealUnmakingSpell('Corporeal Unmaking', 8, game=game, caster=caster)
    with patch('high_magic.roll_dice_expr', return_value=3) as hits, \
            patch('battleFunctions.random.randint', side_effect=[6, 6, 1, 5, 4]):
        asyncio.run(spell.apply(target))
    hits.assert_called_once_with('D3')
    game.movement.applyWounds.assert_called_once_with(target, 1)
    game.psychology.check_heavy_casualties.assert_not_called()
    log = capsys.readouterr().out
    assert 'D3 -> 3 automatic magical S5 hits -> 2 wounds, 1 Ward saves, 1 unsaved' in log
    assert 'no armour or Regeneration' in log
    assert caster.assailmentWounds == 1


def test_combat_credit_caps_at_remaining_wounds_and_is_consumed_once():
    target = SimpleNamespace(unit=_unit(_model(), nmodels=1), woundsOnModel=0)
    caster = SimpleNamespace(unit=_unit(_model()), hostUnit=SimpleNamespace(unit=_unit(_model())))
    game = SimpleNamespace(movement=Mock())
    spell = CorporealUnmakingSpell('Corporeal Unmaking', 8, game=game, caster=caster)
    with patch('high_magic.roll_dice_expr', return_value=3), \
            patch('battleFunctions.random.randint', return_value=6):
        asyncio.run(spell.apply(target))
    assert caster.hostUnit.assailmentWounds == 1
    assert not hasattr(caster, 'assailmentWounds')
    assert CombatResolver.takeAssailmentWounds([caster.hostUnit]) == 1
    assert CombatResolver.takeAssailmentWounds([caster.hostUnit]) == 0


@pytest.mark.parametrize('roll, wounds', [(2, 0), (3, 1)])
def test_strength_five_wounds_toughness_four_on_three_and_bypasses_ethereal(roll, wounds, capsys):
    target = _unit(_model())
    target.model.characteristics['T'] = 4
    target.model.special_rules.append({'name': 'Ethereal'})
    with patch('battleFunctions.random.randint', return_value=roll) as dice:
        assert resolve_magic_hits(target, 1, 5, 0, allow_armour=False,
                                  allow_regeneration=False) == (wounds, 0, wounds)
    assert dice.call_count == 1
    assert 'magical hit(s) from spell can wound' in capsys.readouterr().out


def test_zero_wounds_does_not_bank_credit_or_remove_models(capsys):
    target = SimpleNamespace(unit=_unit(_model()))
    caster = SimpleNamespace(unit=_unit(_model()))
    game = SimpleNamespace(movement=Mock())
    spell = CorporealUnmakingSpell('Corporeal Unmaking', 8, game=game, caster=caster)
    with patch('high_magic.roll_dice_expr', return_value=1), \
            patch('battleFunctions.random.randint', return_value=1):
        asyncio.run(spell.apply(target))
    game.movement.applyWounds.assert_not_called()
    assert not hasattr(caster, 'assailmentWounds')
    assert '0 wounds, 0 Ward saves, 0 unsaved' in capsys.readouterr().out