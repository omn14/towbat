"""Live item inventory and save boundaries; selected effects remain unsupported."""

import json
from pathlib import Path

import pytest

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
    mage = app._create_unit(item_spec('Mage', 'Silvery Wand', 'Arcane Items'), 1, 'Mage')
    hero = app._create_unit(item_spec('Aspiring Champion', 'Helm Of Courage', 'Magic Armour'), 2, 'Hero')
    for index, member in enumerate((mage, hero)):
        member.isDeployed = True
        member.bodyNP.setPos(-12 + index * 24, -10, 0)
    knights = app.player1Units[0]
    install_inventory(knights, [{'name': 'The Banner Of The Bold', 'category': 'Magic Standards',
                                'owner_ref': 'knights/standard', 'selection_ref': 'knights/standard/banner',
                                'number': 1, 'points_cost': 10}])
    path = save_game_state(app, str(tmp_path_factory.mktemp('items') / 'baseline.json'))
    return app, path


def test_live_loader_resolves_bearers_but_does_not_enable_item_effects(item_scene):
    app, baseline = item_scene
    load_game_state(app, baseline)
    members = {member.unitName: member for member in app.units}
    for name in ('Mage', 'Hero', 'Knights'):
        member = members[name]
        item = inventory(member)[0]
        assert bearer_unavailable(member, item) is None
        assert 'unsupported' in inventory_lines(member)
        assert 'Item:' in '\n'.join(app.unitDetailLines(member))
    standard = resolve_bearer(members['Knights'], inventory(members['Knights'])[0])
    assert standard.command['role'] == 'standard_bearer'
    assert members['Mage'].unit.model.spells == {}


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
    assert 'unsupported' in '\n'.join(label.getText() for label in hud._detail_labels)
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