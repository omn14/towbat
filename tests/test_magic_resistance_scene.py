"""Offscreen save/load and real casting-path checks for Magic Resistance."""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest
from panda3d.core import getModelPath, loadPrcFileData

from battlescribe import get_catalogue
from characters import join_unit, slay_character
from game import MyApp
from persistence import load_game_state, save_game_state
from roster_importer import _collect_spells
from special_rules import apply_rule_keywords, unit_magic_resistance
from spell_system import FireballSpell, Spell


def build_scenario():
    loadPrcFileData('', 'window-type offscreen\nwin-size 1280 720\naudio-library-name null')
    getModelPath().appendDirectory(str(ROOT))
    bound = []
    _collect_spells({'name': 'Ruby Ring of Ruin'}, bound)

    def add(game, side, name, profile, count=1, **extra):
        result = game._create_unit(dict(name=profile, nmodels=count,
                                       files=min(count, 5), ranks=(count + 4) // 5,
                                       **extra), side, name)
        assert result is not None

    def player_one(game, _filename):
        add(game, 1, 'MR Wizard Level 2', 'Master Mage',
            spells=[*get_catalogue().lore('Battle Magic'), *bound], wizard_level=2)
        add(game, 1, 'MR Ruby Ring non-Wizard', 'Captain of the Empire', spells=bound)
        add(game, 1, 'MR Friendly -1', 'Dwarf Warrior', 10)

    def player_two(game, _filename):
        add(game, 2, 'MR Enemy -1', 'Dwarf Warrior', 10)
        add(game, 2, 'MR Host with Runesmith -2', 'Dwarf Warrior', 10)
        add(game, 2, 'MR Runesmith -2', 'Runesmith')
        add(game, 2, 'MR Control no resistance', 'Zombie', 10)

    with patch.object(MyApp, 'load_player1_army', player_one), \
            patch.object(MyApp, 'load_player2_army', player_two):
        app = MyApp()
    app.fsm.request('ShootingPhase')
    positions = [(-8, -18), (0, -18), (-18, -5), (-12, -5), (-4, -5), (-3, -5), (4, -5)]
    for u, (x, y) in zip(app.units, positions):
        u.bodyNP.setPos(x, y, 0)
        u.bodyNP.setH(0 if u in app.player1Units else 180)
        u.isDeployed = True
        u.request('Idle')
        u.layOutRanks()
        u.rebuildFootprint()
    units = {u.unitName: u for u in app.units}
    assert join_unit(app, units['MR Runesmith -2'], units['MR Host with Runesmith -2'])
    app.unitToMove = units['MR Wizard Level 2']
    app.roundCounter.request('PlayerOne')
    app.refreshSelectedUnit()
    return app


@pytest.fixture(scope='module')
def scene(tmp_path_factory):
    app = build_scenario()
    path = tmp_path_factory.mktemp('mr-scene') / 'baseline.json'
    save_game_state(app, str(path))
    yield app, path
    app.destroy()


def test_loaded_scene_casts_bound_and_ordinary_with_strongest_resistance(scene, tmp_path):
    app, baseline = scene
    load_game_state(app, str(baseline))
    units = {u.unitName: u for u in app.units}
    wizard = units['MR Wizard Level 2']
    ring = units['MR Ruby Ring non-Wizard']
    host = units['MR Host with Runesmith -2']
    enemy = units['MR Enemy -1']
    assert unit_magic_resistance(host)[:2] == (-2, 'Runesmith')
    assert unit_magic_resistance(enemy)[0] == -1
    assert not ring.unit.model.is_wizard()
    for caster in (wizard, ring):
        for target in (enemy, host, units['MR Control no resistance']):
            origin, destination = caster.bodyNP.getPos(), target.bodyNP.getPos()
            assert (destination - origin).length() < 24
            assert app.terrain_manager.los_block_point(origin, destination) is None
    key = next(iter(ring.unit.model.spells))
    assert key in app.castableSpells(ring)
    assert {'Fireball', key} <= set(app.castableSpells(wizard))

    # 9 + 1 - 2 = 8 succeeds; a dispel of 9 beats that reduced result.
    app.unitToMove = ring
    app.fsm.request('SpellPhase')
    spell = FireballSpell('Fireball', 8, [], game=app, caster=ring,
                          bound=True, power_level=1, spell_range=24)
    spell.selection_key = key
    app.fsm.castingUnit = ring
    app.fsm.spellInstanceToCast = spell
    with patch.object(Spell, '_roll_casting_dice', AsyncMock(
            side_effect=[(9, [4, 5]), (9, [4, 5])])), \
            patch.object(spell, 'apply', AsyncMock()) as effect:
        asyncio.run(app.resolveSpell(host))
        effect.assert_not_awaited()
    assert spell.casting == 8 and app.fsm.state == 'ShootingPhase'
    assert ring.boundSpellPhases == ['shooting'] and ring.spellsCastThisTurn == []

    spent = tmp_path / 'spent.json'
    save_game_state(app, str(spent))
    ring.boundSpellPhases.clear()
    ring.unit.model.spells[key]['power_level'] = 99
    load_game_state(app, str(spent))
    assert ring.boundSpellPhases == ['shooting']
    assert ring.unit.model.spells[key]['power_level'] == 1
    assert app.castableSpells(ring) == [] and not ring.unit.model.is_wizard()

    # 8 + 1 - 2 = 7 fails before any damage or dispel.
    app.unitToMove = wizard
    app.fsm.request('SpellPhase')
    spell = FireballSpell('Fireball', 8, [], game=app, caster=wizard, wizard_level=2)
    app.fsm.castingUnit = wizard
    app.fsm.spellInstanceToCast = spell
    with patch.object(Spell, '_roll_casting_dice', AsyncMock(return_value=(8, [4, 4]))), \
            patch.object(app, 'dispelAttempt', AsyncMock()) as dispel, \
            patch.object(spell, 'apply', AsyncMock()) as effect:
        asyncio.run(app.resolveSpell(host))
        dispel.assert_not_awaited()
        effect.assert_not_awaited()
    assert spell.casting == 7 and wizard.spellsCastThisTurn == ['Fireball']
    assert key in app.castableSpells(wizard)


def test_dead_character_recreated_and_rule_replacements_restored(scene):
    app, baseline = scene
    load_game_state(app, str(baseline))
    units = {u.unitName: u for u in app.units}
    host = units['MR Host with Runesmith -2']
    character = units['MR Runesmith -2']
    slay_character(app, character)
    assert unit_magic_resistance(host)[0] == -1
    apply_rule_keywords(host.unit.model, ['Magic Resistance (-3)'], replace=True)
    assert unit_magic_resistance(host)[0] == -3
    load_game_state(app, str(baseline))
    assert host.unit.model.magic_resistance() == -1
    assert host.joinedCharacter is not None and host.joinedCharacter is not character
    assert unit_magic_resistance(host)[0] == -2


def test_reloading_unjoined_save_removes_a_later_character_grant(scene, tmp_path):
    app, baseline = scene
    load_game_state(app, str(baseline))
    host = next(u for u in app.units if u.unitName == 'MR Host with Runesmith -2')
    # A save before the join must not retain the current scene's host link.
    from characters import detach_character
    character = host.joinedCharacter
    character.bodyNP.wrtReparentTo(app.render)
    detach_character(host)
    app.world.attachRigidBody(character.bodyNP.node())
    app.player2Units.append(character)
    unjoined = tmp_path / 'unjoined.json'
    save_game_state(app, str(unjoined))
    assert join_unit(app, character, host)
    assert unit_magic_resistance(host)[0] == -2
    load_game_state(app, str(unjoined))
    assert host.joinedCharacter is None
    assert character.hostUnit is None and character in app.player2Units
    assert unit_magic_resistance(host)[0] == -1


def test_selected_mount_resistance_survives_reload_and_downgrade(scene, tmp_path):
    from models import model
    app, baseline = scene
    load_game_state(app, str(baseline))
    bearer = next(u for u in app.units if u.unitName == 'MR Ruby Ring non-Wizard')
    mount = model('Skeletal Steed', '')
    bearer.unit.model.attach_mount(mount)
    apply_rule_keywords(mount, ['Magic Resistance (-2)'], replace=True)
    path = tmp_path / 'mounted.json'
    save_game_state(app, str(path))
    apply_rule_keywords(mount, ['Magic Resistance (-3)'], replace=True)
    load_game_state(app, str(path))
    assert unit_magic_resistance(bearer)[0] == -2
    apply_rule_keywords(mount, [], replace=True)
    save_game_state(app, str(path))
    apply_rule_keywords(mount, ['Magic Resistance (-3)'], replace=True)
    load_game_state(app, str(path))
    assert unit_magic_resistance(bearer)[0] == 0
    apply_rule_keywords(mount, ['Magic Resistance (-3)'], replace=True)
    load_game_state(app, str(baseline))
    assert not bearer.unit.model.is_mounted()
    assert unit_magic_resistance(bearer)[0] == 0


if __name__ == '__main__':
    app = build_scenario()
    try:
        path = ROOT / 'saves' / 'magicresistance.json'
        save_game_state(app, str(path))
        load_game_state(app, str(path))
        assert unit_magic_resistance(next(u for u in app.units
                                         if u.unitName == 'MR Host with Runesmith -2'))[0] == -2
        app.graphicsEngine.renderFrame()
        app.graphicsEngine.renderFrame()
        app.screenshot(str(ROOT / 'screenshots' / 'magicresistance.png'), defaultFilename=False)
        print('Magic Resistance scenario saved, reloaded and rendered:', path)
    finally:
        app.destroy()