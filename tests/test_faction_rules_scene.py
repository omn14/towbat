"""Actual High Elf / Chaos roster effects and persistent profile ownership."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from panda3d.core import getModelPath, loadPrcFileData

from battleFunctions import strike_initiative, ward_save_value
from characters import join_unit
from combat_profiles import profile_strike_order
from game import MyApp
from magic_items import disable_item, inventory
from persistence import load_game_state, save_game_state
from psychology import reroll_leadership
from spell_generation import generate_spells, pending_wizards
from special_rules import apply_rule_keywords
from tests.test_shieldwall_scene import combat_tasks


@pytest.fixture(scope='module')
def scene(tmp_path_factory):
    loadPrcFileData('', 'window-type offscreen\nwin-size 1280 720\naudio-library-name null')
    getModelPath().appendDirectory(str(Path(__file__).resolve().parents[1]))
    with patch('spell_generation.begin_spell_generation'):
        app = MyApp()
        try:
            with patch.object(app, 'aiControls', return_value=True):
                for wizard in pending_wizards(app):
                    asyncio.run(generate_spells(app, wizard))
            app.AIplayer2.active = False
            app.fsm.request('MovementPhase')
            for index, member in enumerate(app.units):
                member.isDeployed = True
                member.bodyNP.setPos(-25 + index * 5, 0, 0)
                member.roundsFought = 1
            baseline = save_game_state(app, str(tmp_path_factory.mktemp('factions') / 'baseline.json'))
            yield app, baseline
        finally:
            app.destroy()


def members(app):
    return {member.unit.model.name: member for member in app.units}


def test_actual_roster_wards_and_profile_strike_order(scene):
    app, baseline = scene
    load_game_state(app, baseline)
    armies = members(app)
    assert len(app.units) == 10
    expected = {'Aspiring Champion': 5, 'Chaos Knight': 6, 'Chaos Warrior': 6,
                'Dragon Prince': 6, 'Mage': 0, 'Silver Helm': 0}
    for name, ward in expected.items():
        assert ward_save_value(armies[name].unit.model) == ward, name
    skycutter, enemy = armies['Lothern Skycutter'], armies['Chaos Knight']
    skycutter.roundsFought = 1
    order = profile_strike_order([skycutter], [enemy], lambda *_: 'front')
    assert {part.role: value for value, part in order} == {'crew': 5, 'beasts': 4}
    skycutter.roundsFought = 2
    assert {part.role: value for value, part in profile_strike_order(
        [skycutter], [enemy], lambda *_: 'front')} == {'crew': 4, 'beasts': 4}
    princes = armies['Dragon Prince']
    princes.roundsFought = 1
    order = profile_strike_order([princes], [enemy], lambda *_: 'front')
    assert {part.role: value for value, part in order} == {'main': 6, 'mount': 4}
    assert not princes.unit.model.get_mount().has_magical_attacks()


def test_actual_barding_does_not_protect_joined_mage_or_remove_movement_penalty(scene):
    from types import SimpleNamespace
    app, baseline = scene
    load_game_state(app, baseline)
    armies = members(app)
    host, mage = armies['Silver Helm'], armies['Mage']
    assert join_unit(app, mage, host)
    assert 'Ithilmar Barding' in host.unit.model.dangerous_terrain_reroll_sources()
    assert not mage.unit.model.dangerous_terrain_reroll_sources()
    assert not host.unit.model.is_move_through_cover()
    position = host.bodyNP.getPos(host.bodyNP.getTop())
    terrain = SimpleNamespace(is_dangerous=True, terrain_type='dangerous',
                              center=position, width=20, height=20)
    rolls = [1, 4] + [4] * (host.unit.nmodels - 1) + [1]
    with patch.object(app, 'terrain_manager', SimpleNamespace()), \
            patch.object(app.movement, 'magicalVortexTests'), \
            patch.object(app.movement, 'applyWounds') as wounds, \
            patch('terrain_system.random.randint', side_effect=rolls):
        assert app.movement.dangerousTerrainTests(host, tuple(position),
            (position.x + 1, position.y, position.z), features=[terrain]) == 1
    assert [(call.args[0], call.args[1]) for call in wounds.call_args_list] == [(host, 0), (mage, 1)]


def test_banner_veteran_and_mark_use_one_live_choice(scene, capsys):
    app, baseline = scene
    load_game_state(app, baseline)
    warriors = members(app)['Chaos Warrior']
    roll = AsyncMock(return_value=[6, 6])
    with combat_tasks(app) as run, patch.object(app, 'makeChoiceNew', AsyncMock(return_value='Re-roll')) as choice:
        run(reroll_leadership(app, warriors, 'Panic', [5, 6], 8, roll, cause='nearby friend destroyed'))
    roll.assert_awaited_once()
    choice.assert_awaited_once()
    output = capsys.readouterr().out
    assert 'The Banner Of The Bold' in output and 'Mark of Chaos Undivided' in output
    assert 'no further re-roll' in output


def test_reload_preserves_native_wards_but_not_disabled_item_armour(scene, tmp_path):
    app, baseline = scene
    load_game_state(app, baseline)
    armies = members(app)
    champion, prince = armies['Aspiring Champion'], armies['Dragon Prince']
    disable_item(champion, inventory(champion)[0], 'Test suppression')
    assert champion.unit.model.effective_armour_save() == 5
    assert ward_save_value(champion.unit.model) == 5
    prince.roundsFought = 2
    path = save_game_state(app, str(tmp_path / 'faction-state.json'))
    apply_rule_keywords(champion.unit.model, [], replace=True)
    apply_rule_keywords(prince.unit.model, [], replace=True)
    assert ward_save_value(champion.unit.model) == 0
    load_game_state(app, path)
    assert ward_save_value(champion.unit.model) == 5
    assert champion.unit.model.effective_armour_save() == 5
    assert ward_save_value(prince.unit.model) == 6 and prince.roundsFought == 2
    assert strike_initiative(prince.unit.model, first_round=False) == 5
    assert strike_initiative(prince.unit.model, first_round=True) == 6
    load_game_state(app, path)
    assert ward_save_value(champion.unit.model) == 5