"""High Elf spells through actual roster state (Rulebook pp. 328-329; FoF p. 186)."""

import asyncio
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
import pytest

from characters import join_unit
from high_magic import CourageOfAenarionSpell, DrainMagicSpell, VaulsUnmakingSpell, FieryConvocationSpell
from magic_items import inventory
from persistence import load_game_state, save_game_state
from spell_effects import end_effect
from tests.test_faction_rules_scene import members, scene as scene
from tests.test_shieldwall_scene import combat_tasks
from tempest import TempestSpell
from panda3d.core import Point3


def test_vaul_disables_real_helm_and_survives_reload(scene, tmp_path):
    app, baseline = scene
    load_game_state(app, baseline)
    mage, enemy = members(app)['Mage'], members(app)['Aspiring Champion']
    mage.bodyNP.setPos(0, -15, 0)
    mage.bodyNP.setH(0)
    enemy.bodyNP.setPos(0, -8, 0)
    spell = VaulsUnmakingSpell("Vaul's Unmaking", 11, game=app, caster=mage)
    assert spell.canTarget(enemy)
    assert not spell.canTarget(members(app)['Chaos Warrior'])
    assert enemy.unit.model.effective_armour_save() == 4
    with combat_tasks(app) as run, patch.object(app, 'makeChoiceNew', AsyncMock(
            side_effect=lambda options, *args, **kwargs: next(iter(options)))) as choices:
        run(spell.apply(enemy))
    assert choices.call_args.kwargs['owner'] is mage
    assert enemy.unit.model.effective_armour_save() == 5
    assert inventory(enemy)[0].disabled_reason
    path = save_game_state(app, str(tmp_path / 'vaul.json'))
    load_game_state(app, path)
    assert enemy.unit.model.effective_armour_save() == 5
    assert inventory(enemy)[0].disabled_reason


def test_courage_join_reload_and_source_removal(scene, tmp_path):
    app, baseline = scene
    load_game_state(app, baseline)
    mage, host = members(app)['Mage'], members(app)['Silver Helm']
    assert join_unit(app, mage, host)
    spell = CourageOfAenarionSpell('Courage of Aenarion', 10, game=app, caster=mage)
    asyncio.run(spell.apply(host))
    assert spell.value(host.unit.model) and spell.value(mage.unit.model)
    path = save_game_state(app, str(tmp_path / 'courage.json'))
    load_game_state(app, path)
    restored = next(effect for effect in app.remainsInPlay if effect.name == spell.name)
    assert restored.value(host.unit.model)
    end_effect(restored, 'test dispel')
    assert not restored.value(host.unit.model)


def test_drain_measures_enemy_wizard_range_from_joined_model(scene):
    app, baseline = scene
    load_game_state(app, baseline)
    mage, enemy = members(app)['Mage'], members(app)['Aspiring Champion']
    mage.bodyNP.setPos(0, -15, 0)
    enemy.bodyNP.setPos(0, -5, 0)
    spell = DrainMagicSpell('Drain Magic', 9, game=app, caster=mage)
    asyncio.run(spell.apply(mage))
    assert not spell.affects(enemy)
    with patch.object(enemy.unit.model, 'is_wizard', return_value=True):
        assert spell.affects(enemy)
        enemy.bodyNP.setPos(0, 20, 0)
        assert not spell.affects(enemy)
    assert not spell.affects(mage)


def test_tempest_terrain_is_temporary_enemy_only_and_reloadable(scene, tmp_path):
    app, baseline = scene
    load_game_state(app, baseline)
    mage, enemy = members(app)['Mage'], members(app)['Chaos Warrior']
    mage.bodyNP.setPos(0, -15, 0)
    enemy.bodyNP.setPos(6, -10, 0)
    spell = TempestSpell('Tempest', 9, game=app, caster=mage)
    center = Point3(0, -10, 0)
    assert spell.canTarget(center)
    assert not spell.canTarget(mage.bodyNP.getPos())
    asyncio.run(spell.apply(center))
    start, end = Point3(6, -10, 0), Point3(6, -8, 0)
    ordinary = enemy.unit.model.get_movement(0)
    assert app.movement.movementAllowance(enemy, start, end) == ordinary - 1
    assert app.movement.movementAllowance(mage, start, end) == mage.unit.model.get_movement(0)
    spell.scatter(app)
    assert spell.piece.center.x == center.x and spell.piece.center.y == center.y
    path = save_game_state(app, str(tmp_path / 'tempest.json'))
    load_game_state(app, path)
    restored = next(effect for effect in app.remainsInPlay if effect.name == 'Tempest')
    assert app.movement.movementAllowance(enemy, start, end) == ordinary - 1
    end_effect(restored, 'dispel')
    assert app.movement.movementAllowance(enemy, start, end) == ordinary


def test_fiery_scatter_counts_enemy_bases_and_keeps_friends_safe(scene):
    app, baseline = scene
    load_game_state(app, baseline)
    mage, enemy, friendly = members(app)['Mage'], members(app)['Chaos Warrior'], members(app)['Silver Helm']
    mage.bodyNP.setPos(0, -15, 0)
    enemy.bodyNP.setPos(0, -8, 0)
    friendly.bodyNP.setPos(0, -8, 0)
    spell = FieryConvocationSpell('Fiery Convocation', 10, game=app, caster=mage)
    with patch('spell_templates.scatter_template', return_value=(Point3(0, -8, 0), 'Hit')), \
            patch('spell_templates.random.randint', return_value=6), \
            patch('battleFunctions.resolve_magic_hits', return_value=(1, 0, 1)) as hits, \
            patch.object(app.movement, 'applyWounds') as damage, \
            patch.object(app.psychology, 'check_heavy_casualties'):
        asyncio.run(spell.apply(enemy))
    assert hits.call_count == 1
    assert hits.call_args.args[0] is enemy.unit
    assert hits.call_args.args[1] == enemy.unit.nmodels
    assert hits.call_args.args[2:] == (4, 2)
    damage.assert_called_once_with(enemy, 1)


def test_tempest_difficult_upgrade_is_hashable_and_does_not_change_native_piece(scene):
    from tempest import tempest_features
    app, baseline = scene
    load_game_state(app, baseline)
    mage, enemy = members(app)['Mage'], members(app)['Chaos Warrior']
    enemy.bodyNP.setPos(6, -10, 0)
    spell = TempestSpell('Tempest', 9, game=app, caster=mage)
    spell.place(app, Point3(0, -10, 0))
    piece = app.terrain_manager.add_terrain('forest', Point3(6, -10, 0), 2, 2)
    start, end = Point3(6, -10, 0), Point3(6, -8, 0)
    features = tempest_features(app, enemy, start, end, [piece])
    assert any(feature.is_dangerous for feature in features)
    assert not piece.is_dangerous
    assert len(set(features)) == len(features)
    repeated = tempest_features(app, enemy, start, end, features)
    assert len(repeated) == len(features)
    app.movement.magicalVortexTests(enemy, start, end, features=features)
    assert app.movement.updateDisrupted(enemy)
    end_effect(spell, 'test cleanup')
    app.terrain_manager.remove_terrain(piece)


@pytest.mark.parametrize('failed,dispelled', [(True, False), (False, True), (False, False)])
def test_courage_replaces_only_after_success_and_reaches_panic_consumer(scene, failed, dispelled):
    from high_magic import FuryOfKhaineSpell
    app, baseline = scene
    load_game_state(app, baseline)
    mage, host = members(app)['Mage'], members(app)['Silver Helm']
    mage.bodyNP.setPos(0, -15, 0)
    mage.bodyNP.setH(0)
    host.bodyNP.setPos(0, -10, 0)
    previous = FuryOfKhaineSpell('Fury of Khaine', 9, game=app, caster=mage)
    asyncio.run(previous.apply(host))
    courage = CourageOfAenarionSpell('Courage of Aenarion', 10, game=app, caster=mage)
    with patch.object(courage, '_attempt', AsyncMock(return_value=not failed)), \
            patch.object(courage, '_dispelled', AsyncMock(return_value=dispelled)):
        asyncio.run(courage.spellFunction(host))
    assert previous.ended == (not failed and not dispelled)
    assert courage.value(host.unit.model) == previous.ended
    assert (app.psychology.panic_exempt_reason(host) == 'Unbreakable') == previous.ended


def test_drain_reload_recast_and_death_remove_aura(scene, tmp_path):
    from spell_effects import caster_removed
    app, baseline = scene
    load_game_state(app, baseline)
    mage = members(app)['Mage']
    spell = DrainMagicSpell('Drain Magic', 9, game=app, caster=mage)
    asyncio.run(spell.apply(mage))
    path = save_game_state(app, str(tmp_path / 'drain.json'))
    load_game_state(app, path)
    restored = next(effect for effect in app.remainsInPlay if effect.name == spell.name)
    replacement = DrainMagicSpell('Drain Magic', 9, game=app, caster=mage)
    with patch.object(replacement, '_attempt', AsyncMock(return_value=False)):
        asyncio.run(replacement.spellFunction(mage))
    assert restored.ended and not app.remainsInPlay
    asyncio.run(DrainMagicSpell('Drain Magic', 9, game=app, caster=mage).apply(mage))
    caster_removed(app, mage)
    assert not app.remainsInPlay


@pytest.mark.parametrize('protection', [1, 6])
def test_fiery_champion_look_out_sir_uses_correct_profile(scene, protection):
    from command_groups import champions, install_command
    from spell_templates import fiery_template
    app, baseline = scene
    load_game_state(app, baseline)
    mage, host = members(app)['Mage'], members(app)['Chaos Warrior']
    install_command(host, [{'role': 'champion', 'name': 'Champion', 'selection_ref': 'template/champion'}])
    host.unit.command_models = {'template/champion': deepcopy(host.unit.model)}
    champion = champions(host)[0]
    assert host.unit.nmodels >= 6
    spell = FieryConvocationSpell('Fiery Convocation', 10, game=app, caster=mage)
    def boxes(member):
        return [(0, 0, .5, .5, 0)] + [(50, 50, .5, .5, 0)] * (host.unit.nmodels - 1) if member is host else []
    with patch('scouts.model_base_boxes', side_effect=boxes), \
            patch.object(app, 'aiControls', return_value=True), \
            patch('spell_templates.random.randint', return_value=protection), \
            patch('battleFunctions.resolve_magic_hits', return_value=(0, 0, 0)) as damage:
        asyncio.run(fiery_template(spell, Point3(0, 0, 0)))
    assert damage.call_args.args[0] is (host.unit if protection == 6 else champion.unit)


@pytest.mark.parametrize('flammable', [False, True])
def test_fiery_only_suppresses_regeneration_on_flammable_targets(scene, flammable):
    from spell_templates import fiery_template
    app, baseline = scene
    load_game_state(app, baseline)
    mage, host = members(app)['Mage'], members(app)['Chaos Warrior']
    if flammable:
        host.unit.model.special_rules.append({'name': 'Flammable'})
    spell = FieryConvocationSpell('Fiery Convocation', 10, game=app, caster=mage)
    with patch('scouts.model_base_boxes', side_effect=lambda member: [(0, 0, .5, .5, 0)] if member is host else []), \
            patch('battleFunctions.resolve_magic_hits', return_value=(0, 0, 0)) as damage:
        asyncio.run(fiery_template(spell, Point3(0, 0, 0)))
    assert damage.call_args.kwargs['allow_regeneration'] is not flammable


def test_tempest_offscreen_render(scene):
    app, baseline = scene
    load_game_state(app, baseline)
    spell = TempestSpell('Tempest', 9, game=app, caster=members(app)['Mage'])
    spell.place(app, Point3(0, -8, 0))
    app.graphicsEngine.renderFrame()
    app.graphicsEngine.renderFrame()
    assert app.screenshot('/tmp/towbat-high-magic.png', defaultFilename=False)
    end_effect(spell, 'render complete')


def test_unmade_wand_removes_only_its_generated_slot_and_stays_removed(scene, tmp_path):
    from magic_items import disable_item
    app, baseline = scene
    load_game_state(app, baseline)
    mage = members(app)['Mage']
    before = set(mage.unit.model.spells)
    assert len(before) == 3
    wand = next(item for item in inventory(mage) if item.name == 'Silvery Wand')
    disable_item(mage, wand, "Vaul's Unmaking")
    remaining = set(mage.unit.model.spells)
    assert len(remaining) == 2 and remaining < before
    assert mage.unit.model.wizard_level() == 2
    path = save_game_state(app, str(tmp_path / 'unmade-wand.json'))
    load_game_state(app, path)
    assert set(mage.unit.model.spells) == remaining