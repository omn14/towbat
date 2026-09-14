"""Native configuration controls without loading a battle or another GUI toolkit."""

from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from direct.gui.DirectGui import DGG, DirectCheckButton, DirectEntry, DirectOptionMenu
from direct.showbase.ShowBase import ShowBase
from panda3d.core import (Filename, FrameBufferProperties, GraphicsPipe, PNMImage, Point3,
                          WindowProperties, getModelPath, loadPrcFileData)

from battle_config import ConfigError, load_config, save_config, startup_options
from battle_config_ui import BattleConfigScreen, FIELDS


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def app():
    loadPrcFileData('', 'window-type offscreen\nwin-size 1280 720\naudio-library-name null')
    getModelPath().appendDirectory(Filename.fromOsSpecific(str(ROOT)))
    game = ShowBase()
    yield game
    game.destroy()


@pytest.fixture
def screen(app, tmp_path):
    editor = BattleConfigScreen(app, load_config(), tmp_path / 'battle.json', 19, Mock())
    yield editor
    editor.destroy()
    app.setAspectRatio(1280 / 720)


def click(screen, button):
    screen.game.messenger.send(DGG.B1CLICK + button.guiId, [None])
    screen.game.eventMgr.doEvents()


def test_form_saves_edits_across_tabs_and_starts_once(screen):
    initial = deepcopy(screen.config)
    screen.controls['points_limit'].enterText('700')
    screen.controls['battlefield.width'].enterText('48')
    screen.controls['battlefield.depth'].enterText('36')
    screen.controls['game.rounds'].enterText('7')
    screen.controls['seed'].enterText('23')
    screen.controls['deployment.map'].set('Close Encounter')
    click(screen, screen.controls['deployment.mirror'])
    click(screen, screen.tabs['Terrain'])
    screen.controls['terrain.feature_count'].enterText('2')
    screen.controls['terrain.method'].set('Scattered')
    click(screen, screen.tabs['Muster'])
    screen.controls['army.maximum_character_fraction'].enterText('30')
    click(screen, screen.tabs['Modules'])
    click(screen, screen.controls['optional_rules.secondary_objectives.raid_and_burn'])
    click(screen, screen.controls['optional_rules.secondary_objectives.baggage_carts'])
    click(screen, screen.save_button)
    saved = load_config(screen.path)
    assert saved['points_limit'] == 700
    assert saved['battlefield']['width'] == 48 and saved['battlefield']['depth'] == 36
    assert saved['deployment']['map'] == 'close_encounter' and saved['deployment']['mirror']
    assert saved['terrain']['feature_count'] == 2 and saved['terrain']['method'] == 'scattered'
    assert saved['army']['maximum_character_fraction'] == .30
    assert saved['optional_rules']['secondary_objectives'] == ['raid_and_burn', 'baggage_carts']
    assert saved['source'] == initial['source']
    screen.on_start.assert_not_called()
    click(screen, screen.start_button)
    click(screen, screen.start_button)
    screen.on_start.assert_called_once_with(saved, 23)


@pytest.mark.parametrize('invalid', ['abc', '500.5', '399', '751', 'nan'])
def test_invalid_edit_cannot_save_or_start(screen, invalid):
    save_config(screen.path, screen.config)
    original = screen.path.read_bytes()
    screen.controls['points_limit'].enterText(invalid)
    click(screen, screen.start_button)
    click(screen, screen.save_button)
    assert screen.path.read_bytes() == original
    assert screen.status.getText()
    screen.on_start.assert_not_called()
    assert not screen.started


def test_blank_seed_reload_invalid_file_and_io_failure(screen, tmp_path):
    screen.controls['seed'].enterText('')
    screen.controls['points_limit'].enterText('650')
    assert screen.save()
    screen.controls['points_limit'].enterText('700')
    click(screen, screen.load_button)
    assert screen.controls['points_limit'].get() == '650'
    assert screen.draft()[1] is None
    invalid = tmp_path / 'bad.json'
    invalid.write_text('{bad json', encoding='utf-8')
    screen.path_entry.enterText(str(invalid))
    click(screen, screen.load_button)
    assert screen.controls['points_limit'].get() == '650'
    assert screen.config['points_limit'] == 650
    screen.path_entry.enterText(str(tmp_path / 'missing' / 'battle.json'))
    click(screen, screen.start_button)
    screen.on_start.assert_not_called()
    assert not screen.started
    assert screen.status.getText()


def test_unsupported_loaded_options_can_be_disabled_but_not_started(app, tmp_path):
    config = load_config()
    config['optional_rules'].update(secret_objectives=True, random_happenings=['chaos_of_war'])
    config['game']['time_limit_minutes'] = 60
    path = tmp_path / 'unsupported.json'
    save_config(path, config)
    options = startup_options(['--battle-config', str(path)])
    assert options.battle_config_path == str(path)
    editor = BattleConfigScreen(app, options.battle_config, path, None, Mock())
    try:
        editor.start()
        editor.on_start.assert_not_called()
        assert path.read_text().find('60') >= 0
        click(editor, editor.tabs['Modules'])
        for key in ('optional_rules.secret_objectives', 'optional_rules.random_happenings.chaos_of_war'):
            click(editor, editor.controls[key])
            assert editor.controls[key]['state'] == DGG.DISABLED
        editor.controls['game.time_limit_minutes'].enterText('')
        editor.start()
        assert editor.started
        assert load_config(path)['game']['time_limit_minutes'] is None
        editor.on_start.assert_called_once()
    finally:
        editor.destroy()


def test_non_mirrorable_map_clears_mirror_and_exit_does_not_save(screen):
    screen.controls['deployment.map'].set('Close Encounter')
    click(screen, screen.controls['deployment.mirror'])
    screen.controls['deployment.map'].set('Pitched Battle')
    assert not screen.draft()[0]['deployment']['mirror']
    assert screen.controls['deployment.mirror']['state'] == DGG.DISABLED
    with patch.object(screen.game, 'userExit') as exit_game:
        click(screen, screen.exit_button)
        exit_game.assert_called_once()
    assert not screen.path.exists()
    assert screen.root.isEmpty()
    assert not screen.isAccepting('wheel_up')


def test_reed_fens_selection_locks_map_fields_and_restores_official_values(screen):
    from battle_config import REED_FENS_MAP
    screen.controls['battlefield.width'].enterText('48')
    screen.controls['points_limit'].enterText('650')
    screen.controls['deployment.map'].set(REED_FENS_MAP)
    config, seed = screen.draft()
    assert (config['battlefield']['width'], config['battlefield']['depth']) == (30, 44)
    assert config['terrain']['method'] == 'fixed'
    assert config['terrain']['feature_count'] == 4
    assert config['objectives']['layout'] == 'none'
    assert config['points_limit'] == 650
    assert screen.controls['battlefield.width']['state'] == DGG.DISABLED
    assert screen.controls['battlefield.depth']['state'] == DGG.DISABLED
    click(screen, screen.tabs['Terrain'])
    assert all(control['state'] == DGG.DISABLED for control in screen.controls.values())
    click(screen, screen.tabs['Objectives'])
    assert all(control['state'] == DGG.DISABLED for control in screen.controls.values())
    assert screen.save()
    assert load_config(screen.path) == config
    click(screen, screen.tabs['Battle'])
    screen.controls['deployment.map'].set('Pitched Battle')
    restored = screen.draft()[0]
    assert restored['battlefield']['width'] == 48
    assert restored['battlefield']['depth'] == 30
    assert restored['terrain']['method'] == 'alternating'
    assert restored['objectives']['layout'] == 'random'
    assert screen.controls['battlefield.width']['state'] == DGG.NORMAL
    click(screen, screen.load_button)
    assert screen.draft()[0] == config
    screen.controls['deployment.map'].set('Pitched Battle')
    assert screen.draft()[0]['battlefield']['width'] == 44


def test_atomic_save_keeps_original_on_failure(tmp_path):
    path = tmp_path / 'battle.json'
    config = load_config()
    save_config(path, config)
    original = path.read_bytes()
    config['points_limit'] = 700
    with patch('os.replace', side_effect=OSError('write failed')):
        with pytest.raises(OSError, match='write failed'):
            save_config(path, config)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]
    config['points_limit'] = -1
    with pytest.raises(ConfigError):
        save_config(path, config)
    assert path.read_bytes() == original


def test_ordinary_startup_does_not_open_editor():
    from game import MyApp
    with patch.object(ShowBase, '__init__', return_value=None), \
            patch.object(MyApp, 'setBackgroundColor'), \
            patch.object(MyApp, '_initialize_battle') as initialize, \
            patch('battle_config_ui.BattleConfigScreen') as editor:
        game = MyApp()
        assert game.battle_config_screen is None
        initialize.assert_called_once_with(None, None)
        editor.assert_not_called()


def test_large_seed_and_numeric_precision_survive_opening(app, tmp_path):
    config = load_config()
    config['battlefield']['width'] = 44.123456789
    seed = 2 ** 53 - 1
    editor = BattleConfigScreen(app, config, tmp_path / 'precise.json', seed, Mock())
    try:
        assert editor.draft() == (config, seed)
        assert editor.save()
        assert load_config(editor.path) == config
    finally:
        editor.destroy()


@pytest.mark.parametrize('width,height', [(1280, 720), (800, 600), (720, 960)])
def test_native_form_layout_popups_and_offscreen_pixels(screen, width, height):
    game = screen.game
    framebuffer = FrameBufferProperties()
    framebuffer.setRgbColor(True)
    properties = WindowProperties.size(width, height)
    buffer = game.graphicsEngine.makeOutput(game.pipe, 'config-layout', -2, framebuffer,
                                            properties, GraphicsPipe.BFRefuseWindow,
                                            game.win.getGsg(), game.win)
    region = buffer.makeDisplayRegion()
    region.setCamera(game.cam2d)
    try:
        game.setAspectRatio(width / height)
        for horizontal, vertical in ((-width / height, -1), (width / height, 1)):
            corner = game.render2d.getRelativePoint(screen.root, Point3(horizontal, 0, vertical))
            assert abs(corner.x) == pytest.approx(1)
            assert abs(corner.z) == pytest.approx(1)
        screen.controls['points_limit'].enterText('625')
        for tab in FIELDS:
            click(screen, screen.tabs[tab])
            game.eventMgr.doEvents()
            game.graphicsEngine.renderFrame()
            for label, control in zip(screen.labels, screen.controls.values()):
                lower, upper = label.getTightBounds(screen.root)
                assert lower.x >= -screen.width / 2
                assert upper.x < control.getX() - .01
                assert upper.z - lower.z < .14
            for control in screen.controls.values():
                bounds = control.guiItem.getFrame()
                for horizontal in (bounds[0], bounds[1]):
                    point = screen.root.getRelativePoint(control, Point3(horizontal, 0, 0))
                    assert -screen.width / 2 <= point.x <= screen.width / 2
                if isinstance(control, DirectOptionMenu):
                    control.showPopupMenu()
                    game.eventMgr.doEvents()
                    game.graphicsEngine.renderFrame()
                    item = control.component('item0')
                    assert item['text_scale'][0] * item.getScale(screen.root).x <= .05
                    control.hidePopupMenu()
            screen.scroll_by(1)
            game.eventMgr.doEvents()
            game.graphicsEngine.renderFrame()
            assert screen.scroll.verticalScroll['value'] == 1
        click(screen, screen.tabs['Battle'])
        assert screen.controls['points_limit'].get() == '625'
        from battle_config import REED_FENS_MAP
        screen.controls['deployment.map'].set(REED_FENS_MAP)
        menu = screen.controls['deployment.map']
        assert menu.get() == REED_FENS_MAP
        menu.showPopupMenu()
        game.eventMgr.doEvents()
        game.graphicsEngine.renderFrame()
        popup_bounds = menu.popupMenu.guiItem.getFrame()
        for horizontal in (popup_bounds[0], popup_bounds[1]):
            point = game.render2d.getRelativePoint(menu.popupMenu, Point3(horizontal, 0, 0))
            assert -1 <= point.x <= 1
        menu.hidePopupMenu()
        game.eventMgr.doEvents()
        game.graphicsEngine.renderFrame()
        game.graphicsEngine.renderFrame()
        image = PNMImage()
        assert buffer.getScreenshot(image)
        assert image.write(Filename.fromOsSpecific(str(ROOT / '.pytest_cache' / f'battle_config_ui_{width}x{height}.png')))
        colors = {tuple(image.getXel(horizontal, vertical))
                  for horizontal in range(0, width, 13) for vertical in range(0, height, 13)}
        assert len(colors) > 50
    finally:
        game.graphicsEngine.removeWindow(buffer)