"""Live selected items, spell generation and persistent bearer state."""

import json
from pathlib import Path

import pytest
from unittest.mock import AsyncMock, patch

from tests.test_shieldwall_scene import combat_tasks

from characters import detach_character, join_unit
from magic_items import (bearer_unavailable, disable_item, install_inventory, inventory,
                         inventory_lines, resolve_bearer, save_inventory)
from persistence import load_game_state, save_game_state
from tests.test_command_scene import scene as command_scene


def item_spec(model_name, item_name, category):
    return {'name': model_name, 'nmodels': 1, 'files': 1, 'ranks': 1,
            'roster_selections': [{'ref': 'owner', 'type': 'model', 'name': model_name}],
            'magic_items': [{'name': item_name, 'category': category, 'definition_id': 'export-id',
                             'owner_ref': 'owner', 'selection_ref': 'owner/item', 'number': 1,
                             'points_cost': 25}]}


@pytest.fixture(scope='module')
def item_scene(command_scene, tmp_path_factory):
    app, _ = command_scene
    mage = app._create_unit(dict(item_spec('Mage', 'Silvery Wand', 'Arcane Items'), wizard_level=2), 1, 'Mage')
    hero = app._create_unit(dict(item_spec('Aspiring Champion', 'Helm Of Courage', 'Magic Armour'),
                                 armour=['Heavy Armour']), 2, 'Hero')
    for index, member in enumerate((mage, hero)):
        member.isDeployed = True
        member.bodyNP.setPos(-12 + index * 24, -10, 0)
    knights = app.player1Units[0]
    install_inventory(knights, [{'name': 'The Banner Of The Bold', 'category': 'Magic Standards',
                                'owner_ref': 'knights/standard', 'selection_ref': 'knights/standard/banner',
                                'number': 1, 'points_cost': 10}])
    path = save_game_state(app, str(tmp_path_factory.mktemp('items') / 'baseline.json'))
    return app, path


def test_live_loader_resolves_bearers_and_selected_effects(item_scene):
    app, baseline = item_scene
    load_game_state(app, baseline)
    members = {member.unitName: member for member in app.units}
    for name in ('Mage', 'Hero', 'Knights'):
        member = members[name]
        item = inventory(member)[0]
        assert bearer_unavailable(member, item) is None
        assert 'supported' in inventory_lines(member)
        assert 'Item:' in '\n'.join(app.unitDetailLines(member))
    standard = resolve_bearer(members['Knights'], inventory(members['Knights'])[0])
    assert standard.command['role'] == 'standard_bearer'
    assert members['Mage'].unit.model.spells == {}
    assert members['Mage'].unit.model.wizard_level() == 2
    assert members['Hero'].unit.model.armor_save == 5
    assert members['Hero'].unit.model.effective_armour_save() == 4
    from psychology import veteran_available
    assert veteran_available(members['Knights'])


@pytest.mark.parametrize('recreate', [False, True])
def test_inventory_state_roundtrips_existing_and_recreated_bearer(item_scene, tmp_path, recreate):
    app, baseline = item_scene
    load_game_state(app, baseline)
    hero = next(member for member in app.units if member.unitName == 'Hero')
    item = inventory(hero)[0]
    item.uses['future_ability'] = {'count': 1, 'turn': None}
    disable_item(hero, item, 'Test item suppression')
    expected = save_inventory(hero)
    path = save_game_state(app, str(tmp_path / 'spent.json'))
    if recreate:
        data = json.loads(Path(path).read_text())
        data['units'] = [record for record in data['units'] if record['name'] != 'Hero']
        absent = tmp_path / 'absent.json'
        absent.write_text(json.dumps(data))
        load_game_state(app, str(absent))
    load_game_state(app, path)
    hero = next(member for member in app.units if member.unitName == 'Hero')
    assert save_inventory(hero) == expected
    assert resolve_bearer(hero, inventory(hero)[0]).profile is hero.unit.model
    load_game_state(app, path)
    assert save_inventory(hero) == expected


def test_old_save_does_not_infer_inventory_from_roster_metadata(item_scene, tmp_path):
    app, baseline = item_scene
    data = json.loads(Path(baseline).read_text())
    for record in data['units']:
        record.pop('magic_item_inventory', None)
    path = tmp_path / 'legacy.json'
    path.write_text(json.dumps(data))
    load_game_state(app, str(path))
    assert all(not inventory(member) for member in app.units)
    hero = next(member for member in app.units if member.unitName == 'Hero')
    assert hero.unit.roster_metadata['magic_items']


def test_joining_keeps_inventory_on_character_and_banner_loss_is_live(item_scene, tmp_path):
    app, baseline = item_scene
    load_game_state(app, baseline)
    mage = next(member for member in app.units if member.unitName == 'Mage')
    knights = app.player1Units[0]
    expected = save_inventory(mage)
    assert join_unit(app, mage, knights)
    assert save_inventory(mage) == expected
    assert 'Silvery Wand' in '\n'.join(app.unitDetailLines(knights))
    path = save_game_state(app, str(tmp_path / 'joined.json'))
    load_game_state(app, path)
    assert mage.hostUnit is knights and save_inventory(mage) == expected
    assert detach_character(knights) is mage
    assert save_inventory(mage) == expected
    load_game_state(app, baseline)
    banner = inventory(knights)[0]
    app.movement.removeModelsFromUnit(knights, 3)
    assert bearer_unavailable(knights, banner) == 'command bearer was lost'
    path = save_game_state(app, str(tmp_path / 'banner-lost.json'))
    load_game_state(app, baseline)
    load_game_state(app, path)
    assert bearer_unavailable(knights, inventory(knights)[0]) == 'command bearer was lost'


def test_item_details_render_in_existing_unit_card(item_scene, tmp_path):
    app, baseline = item_scene
    load_game_state(app, baseline)
    app.unitToMove = next(member for member in app.units if member.unitName == 'Hero')
    app.showSelectedUnit(app.unitToMove)
    assert all(len(line) <= app.CARD_LINE_CHARS for line in app.unitDetailLines(app.unitToMove))
    hud = app.hud
    assert not hud._detail_buttons[1].isHidden()
    hud.scroll_details(2)
    assert 'Helm Of Courage' in '\n'.join(label.getText() for label in hud._detail_labels)
    app.graphicsEngine.renderFrame()
    app.graphicsEngine.renderFrame()
    assert app.screenshot(str(tmp_path / 'items.png'), defaultFilename=False)
    app.showSelectedUnit(app.unitToMove)
    assert hud._detail_offset == 2
    hud.scroll_details(100)
    assert 'courage: ready' in '\n'.join(label.getText() for label in hud._detail_labels)
    assert hud._detail_buttons[1].isHidden()
    app.showSelectedUnit(app.player1Units[0])
    assert hud._detail_offset == 0
    hud.clear_unit()
    hud._layout()
    assert all(button.isHidden() for button in hud._detail_buttons)


@pytest.mark.parametrize('aspect', [16 / 9, 4 / 3])
def test_horizontal_detail_controls_fit_and_hide(item_scene, aspect):
    from hud import HUD
    from unittest.mock import patch

    app, _ = item_scene
    with patch.object(app, 'getAspectRatio', return_value=aspect):
        hud = HUD()
        try:
            hud.show_unit({'name': 'Test', 'details': ['Item: Helm Of Courage',
                                                     'Bearer: Aspiring Champion', 'unsupported']})
            hud._detail_buttons[1]['command'](*hud._detail_buttons[1]['extraArgs'])
            assert hud._detail_offset == 1
            assert hud._detail_labels[1].getText() == 'unsupported'
            width = hud._section_width('regiment', 2 * aspect) * 0.84
            for label in hud._detail_labels:
                assert label.textNode.getWidth() * label.getScale()[0] <= width + 0.001
            hud.clear_unit()
            hud._layout()
            assert all(button.isHidden() for button in hud._detail_buttons)
        finally:
            hud.destroy()


def test_live_break_path_spends_helm_once_and_reload_retains_armour(item_scene, tmp_path):
    from combat_resolution import CombatResolver

    app, baseline = item_scene
    load_game_state(app, baseline)
    hero = next(member for member in app.units if member.unitName == 'Hero')
    resolver = CombatResolver(app)
    resolver.isOverwhelmed = lambda *_: False
    resolver.rollBreakDice = AsyncMock(side_effect=[[6, 6], [1, 1], [6, 6]])
    resolver.notifyFleesCombat = lambda *_: None
    resolver.shieldwallOutcome = AsyncMock(side_effect=lambda member, outcome: outcome)

    async def check_break(expected):
        assert await resolver.breakTestPass([hero], 3) == [(hero, expected)]

    with patch.object(app.psychology, 'battle_standard_of', return_value=None), \
            patch.object(app, 'aiControls', return_value=True), combat_tasks(app) as run:
        run(check_break('give_ground'))
        assert inventory(hero)[0].uses['courage']['count'] == 1
        path = save_game_state(app, str(tmp_path / 'used-helm.json'))
        load_game_state(app, path)
        assert hero.unit.model.effective_armour_save() == 4
        run(check_break('break'))
        assert resolver.rollBreakDice.await_count == 3


def test_live_generation_and_save_keep_exact_spellbook(item_scene, tmp_path):
    from spell_generation import generate_spells
    from tests.test_spell_generation import mage_with_pool

    app, baseline = item_scene
    load_game_state(app, baseline)
    mage = next(member for member in app.units if member.unitName == 'Mage')
    mage.unit.roster_metadata.update(spell_pool=mage_with_pool().unit.roster_metadata['spell_pool'],
                                     spell_generation_pending=True)
    mage.unit.model.characteristics['Special Rules'] = ['Lore of Saphery']

    async def check_generation():
        assert await generate_spells(app, mage)

    with patch('spell_generation.random.randint', side_effect=[1, 1, 2, 6]), \
            patch.object(app, 'makeChoiceNew', AsyncMock(side_effect=['Drain Magic', 'Spell 2'])), \
            combat_tasks(app) as run:
        run(check_generation())
    expected = set(mage.unit.model.spells)
    path = save_game_state(app, str(tmp_path / 'generated.json'))
    load_game_state(app, baseline)
    with patch('spell_generation.random.randint', side_effect=AssertionError('no reload rerolls')):
        load_game_state(app, path)
    assert set(mage.unit.model.spells) == expected == {'Spell 1', 'Drain Magic', 'Spell 6'}
    assert mage.unit.model.wizard_level() == 2
    assert not mage.unit.roster_metadata['spell_generation_pending']


def test_deployment_waits_for_generation_choice(item_scene):
    from spell_generation import pending_wizards
    from tests.test_spell_generation import mage_with_pool

    app, baseline = item_scene
    load_game_state(app, baseline)
    mage = next(member for member in app.units if member.unitName == 'Mage')
    mage.unit.roster_metadata.update(spell_pool=mage_with_pool().unit.roster_metadata['spell_pool'],
                                     spell_generation_pending=True)
    with patch('spell_generation.random.randint', side_effect=[1, 2, 3]), \
            patch.object(app, 'aiControls', return_value=True), combat_tasks(app) as run:
        with patch.object(app.taskMgr, 'add') as schedule:
            app.fsm.request('DeployPhase')
        assert app.spellGenerationBusy
        app.fsm.nextPhase()
        assert app.fsm.state == 'DeployPhase'
        coroutine = next(call.args[0] for call in schedule.call_args_list
                         if len(call.args) > 1 and call.args[1] == 'spellGenerationTask')
        run(coroutine)
        assert not app.spellGenerationBusy and not pending_wizards(app)
        assert len(mage.unit.model.spells) == 3
    load_game_state(app, baseline)