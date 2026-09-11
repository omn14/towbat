"""High Elf spell effects and shared consumers (Rulebook pp. 328-329, FoF p. 186)."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
import pytest

from high_magic import CourageOfAenarionSpell, DrainMagicSpell, drained_casting_value
from models import model
from spell_system import Spell, spell_class
from spell_templates import scatter_template, template_coverage, swept_circle_distance
from panda3d.core import Point3


def test_courage_has_reversible_unbreakable_grant():
    profile = model('Mage', '')
    spell = CourageOfAenarionSpell('Courage of Aenarion', 10)
    before = spell.value(profile)
    grant = dict(spell.grant)
    profile.special_rules.append(grant)
    spell.grants = [(profile, grant)]
    assert spell.value(profile)
    spell.remove_effect()
    assert spell.value(profile) == before
    assert spell.duration == 'remains' and spell.allows_engaged
    assert spell_class(spell.name) is CourageOfAenarionSpell


def test_drain_changes_threshold_not_casting_result_or_stored_value():
    game = SimpleNamespace(fsm=SimpleNamespace(endOfTurnSpells=[]), remainsInPlay=[])
    aura = DrainMagicSpell('Drain Magic', 9, game=game)
    game.remainsInPlay = [aura, aura]
    spell = Spell('Test', 8, wizard_level=2, game=game)
    with patch.object(aura, 'affects', return_value=True), \
            patch.object(spell, '_roll_casting_dice', AsyncMock(return_value=(7, [3, 4]))):
        assert drained_casting_value(spell) == 10
        assert not asyncio.run(spell._attempt(None))
    assert spell.casting == 8 and spell.casting_value == 8
    with patch.object(aura, 'affects', return_value=False), \
            patch.object(spell, '_roll_casting_dice', AsyncMock(return_value=(7, [3, 4]))):
        assert asyncio.run(spell._attempt(None))


def test_template_full_partial_hole_and_rotated_bases():
    assert template_coverage((0, 0), 2.5, (0, 0, 4, 4, 45)) == 'automatic'
    assert template_coverage((0, 0), 2.5, (0, 1, .5, .5, 45)) == 'automatic'
    assert template_coverage((0, 0), 2.5, (0, 2.8, .5, .5, 0)) == 'partial'
    assert template_coverage((0, 0), 2.5, (0, 4, .5, .5, 0)) == 'miss'


def test_scatter_hit_stays_and_arrow_moves_distance():
    with patch('spell_templates.random.randint', return_value=1):
        center, detail = scatter_template(Point3(0, 0, 0), 4)
        assert center == Point3(0, 0, 0) and 'Hit' in detail
    with patch('spell_templates.random.randint', return_value=3), \
            patch('spell_templates.random.uniform', return_value=0):
        center, detail = scatter_template(Point3(0, 0, 0), 4)
        assert center == Point3(4, 0, 0) and 'arrow' in detail


def test_tempest_swept_bases_and_terrain_clipping_exclude_near_misses():
    before, after = (8.1, -2, .5, .5, 0), (8.1, 2, .5, .5, 0)
    assert swept_circle_distance((0, 0), before, after) > 7.5
    before, after = (0, 0, .5, .5, 0), (12, 0, .5, .5, 0)
    assert swept_circle_distance((0, 0), before, after) == 0
    assert swept_circle_distance((0, 0), before, after, (10, 0, 1, 1, 0)) == 9
    assert swept_circle_distance((0, 0), before, after, (5, 0, 1, 1, 0)) == 4


@pytest.mark.parametrize('name', ['Drain Magic', 'Walk Between Worlds', 'Fiery Convocation', 'Tempest',
    'Corporeal Unmaking', 'Fury of Khaine', 'Shield of Saphery', 'Hand of Khaine',
    'Courage of Aenarion', "Vaul's Unmaking"])
def test_all_high_elf_lore_entries_have_effect_implementations(name):
    from battlescribe import get_catalogue
    record = get_catalogue().spell(name)
    assert record['casting_value'] > 0
    assert spell_class(name) is not None


def test_hand_disallows_armour_but_counts_regenerated_wounds_for_combat():
    from high_magic import HandOfKhaineSpell
    from tests.test_battle_magic import _model, _unit
    target = SimpleNamespace(unit=_unit(_model(regen=2)))
    caster = SimpleNamespace(unit=_unit(_model()))
    damage = []
    game = SimpleNamespace(assailmentWindow={'caster': caster, 'damage': lambda *args: damage.append(args)})
    spell = HandOfKhaineSpell('Hand of Khaine', 7, game=game, caster=caster)
    with patch('battleFunctions.random.randint', return_value=6), \
            patch('battleFunctions.check_armor_save', return_value=True) as save:
        asyncio.run(spell.apply(target))
    save.assert_called_once_with(target.unit.model, 2, 0)
    assert damage == [(target, 0, 1)]