"""Fresh-process Battle March startup, isolated from other Panda3D scenes."""

import asyncio
from copy import deepcopy
from pathlib import Path
from unittest.mock import AsyncMock, patch

from panda3d.core import (Filename, FrameBufferProperties, GraphicsPipe, OrthographicLens,
                          PNMImage, Point2, Point3, WindowProperties, getModelPath, loadPrcFileData)

from battle_config import load_config
from battle_preparation import run_preparation
from game import MyApp


def test_explicit_startup_preserves_visual_board_and_holds_deployment(tmp_path):
    root = Path(__file__).resolve().parents[1]
    loadPrcFileData('', 'window-type offscreen\nwin-size 1280 720\naudio-library-name null')
    getModelPath().appendDirectory(Filename.fromOsSpecific(str(root)))
    config = load_config()
    config['battlefield'].update(width=44, depth=30)
    config['deployment'].update(map='random', mirror=False)
    config['terrain'].update(method='alternating', feature_count=0)
    config['objectives']['layout'] = 'random'

    def add_army(game, player):
        for index in range(2):
            unit = game._create_unit(dict(name='Dwarf Warrior', nmodels=10, files=5, ranks=2),
                                     player, f'P{player} Unit {index + 1}')
            assert unit is not None

    with patch.object(MyApp, 'load_player1_army', lambda game, path: add_army(game, 1)), \
            patch.object(MyApp, 'load_player2_army', lambda game, path: add_army(game, 2)), \
            patch('battle_preparation.begin_preparation', return_value=True):
        app = MyApp(battle_config=config, battle_seed=19,
                    battle_config_path=tmp_path / 'chosen.json', configure_battle=True)
        assert not hasattr(app, 'units')
        assert not hasattr(app, 'fsm')
        assert not app.taskMgr.hasTaskNamed('mouseHoverUnit')
        screen = app.battle_config_screen
        window = app.win
        screen.controls['battlefield.width'].enterText('48')
        screen.controls['battlefield.depth'].enterText('36')
        screen.controls['game.rounds'].enterText('6')
        screen.controls['seed'].enterText('23')
        chosen = deepcopy(config)
        chosen['battlefield'].update(width=48, depth=36)
        chosen['game']['rounds'] = 6
        from direct.gui.DirectGui import DGG
        app.messenger.send(DGG.B1CLICK + screen.start_button.guiId, [None])
        app.eventMgr.doEvents()
        assert load_config(tmp_path / 'chosen.json') == chosen
        assert app.battle_config_screen is None
        assert app.win is window
        assert screen.root.isEmpty() and not screen.isAccepting('wheel_up')
    try:
        assert app.fsm.state == 'DeployPhase'
        assert app.roundCounter.max_rounds == 6
        assert app.battlefield.width == 48 and app.battlefield.depth == 36
        assert app.battle_setup['seed'] == 23
        assert app.deploymentLine.isHidden()
        assert not app.terrain_manager.terrain_pieces
        assert not any(unit.isDeployed for unit in app.units)
        assert app.battle_setup['preparation']['stage'] == 'armies'
        with patch.object(app, 'makeChoiceNew', AsyncMock(side_effect=lambda options, *args, **kwargs: options[0])), \
                patch('battle_preparation.random.randint', side_effect=[6, 1, 1, 5]):
            asyncio.run(run_preparation(app))
        assert app.battle_setup['preparation']['stage'] == 'complete'
        assert app.roundCounter.current_player == 2
        assert app.fsm.state == 'DeployPhase'
        app.eventMgr.doEvents()
        app.graphicsEngine.renderFrame()
        assert app.screenshot(Filename.fromOsSpecific(str(root / '.pytest_cache' / 'battle_march_startup.png')).getFullpath(), defaultFilename=False)
        check_reed_fens_scene(app, tmp_path, root)
    finally:
        app.destroy()


def check_reed_fens_scene(app, tmp_path, root):
    from battle_config import REED_FENS_MAP, REED_FENS_PRESET, REED_FENS_TERRAIN
    from battle_setup import prepare_new_battle
    from battlefield import deployment_candidate, deployment_zone_for
    from characters import side_of
    from persistence import load_game_state, save_game_state
    from scouts import model_base_boxes
    import random
    config = load_config(REED_FENS_PRESET)
    prepare_new_battle(app, config, 31)
    with patch.object(app, 'makeChoiceNew', AsyncMock(side_effect=lambda options, *args, **kwargs: options[0])), \
            patch('battle_preparation.random.randint', side_effect=[1, 5]) as dice:
        asyncio.run(run_preparation(app))
    assert dice.call_count == 2
    assert app.battle_setup['preparation']['stage'] == 'complete'
    assert app.battle_setup['deployment_map'] == REED_FENS_MAP
    assert app.battle_setup['rolls'] == {}
    assert not app.battle_objectives
    assert (app.battlefield.width, app.battlefield.depth) == (30, 44)
    assert app.terrain_manager.to_records() == list(REED_FENS_TERRAIN)
    assert [piece.is_dangerous for piece in app.terrain_manager.terrain_pieces] == [True, True, False, False]
    assert [piece.is_impassable for piece in app.terrain_manager.terrain_pieces] == [False, False, True, True]
    generator = random.Random(31)
    for unit in app.units:
        player = side_of(app, unit)
        zone = deployment_zone_for(app, player)
        for attempt in range(100):
            position = deployment_candidate(app, player, generator)
            assert position is not None
            unit.bodyNP.setPos(*position, 0)
            if all(zone.contains_box(bounds) for bounds in model_base_boxes(unit)):
                break
        assert all(zone.contains_box(bounds) for bounds in model_base_boxes(unit))
    saved = tmp_path / 'reed-fens.json'
    save_game_state(app, str(saved))
    expected_setup = deepcopy(app.battle_setup)
    with patch('battle_preparation.begin_preparation', return_value=True), \
            patch('spell_generation.begin_spell_generation', return_value=True), \
            patch('battle_setup.Random', side_effect=AssertionError('Reload must not resolve setup')), \
            patch('battle_preparation.objective_clearance_shift', side_effect=AssertionError('Fixed terrain moved')):
        for reload_index in range(2):
            load_game_state(app, str(saved))
            assert app.battle_setup == expected_setup
            assert app.terrain_manager.to_records() == list(REED_FENS_TERRAIN)
            assert len(app.terrain_manager.terrain_pieces) == 4
            assert not app.battle_objectives
    for width, height, camera_side in ((1280, 720, 0), (720, 960, 0),
                                       (1280, 720, -50), (720, 960, -50),
                                       (1280, 720, 50), (720, 960, 50)):
        lens = OrthographicLens()
        lens.setFilmSize(52 * width / height, 52)
        lens.setNearFar(.1, 200)
        app.cam.node().setLens(lens)
        app.camera.setPos(0, camera_side, 100)
        if camera_side:
            app.camera.lookAt(0, 0, 0)
        else:
            app.camera.setHpr(0, -90, 0)
        framebuffer = FrameBufferProperties()
        framebuffer.setRgbColor(True)
        framebuffer.setDepthBits(24)
        buffer = app.graphicsEngine.makeOutput(app.pipe, 'reed-fens-map', 0, framebuffer,
                                              WindowProperties.size(width, height), GraphicsPipe.BFRefuseWindow,
                                              app.win.getGsg(), app.win)
        assert buffer is not None
        try:
            buffer.makeDisplayRegion().setCamera(app.cam)
            for horizontal, vertical in app.battlefield.outline:
                projected = Point2()
                assert lens.project(app.cam.getRelativePoint(app.render, Point3(horizontal, vertical, 0)), projected)
                assert abs(projected.x) < .95 and abs(projected.y) < .95
            app.eventMgr.doEvents()
            app.graphicsEngine.renderFrame()
            app.graphicsEngine.renderFrame()
            image = PNMImage()
            assert buffer.getScreenshot(image)
            suffix = f'_{camera_side}' if camera_side else ''
            assert image.write(Filename.fromOsSpecific(str(root / '.pytest_cache' / f'reed_fens_{width}x{height}{suffix}.png')))
            for piece in app.terrain_manager.terrain_pieces:
                piece.visual.hide()
                app.graphicsEngine.renderFrame()
                background = PNMImage()
                assert buffer.getScreenshot(background)
                changed = sum((image.getXel(horizontal, vertical) - background.getXel(horizontal, vertical)).length() > .05
                              for horizontal in range(0, width, 3) for vertical in range(0, height, 3))
                assert changed > 30, (piece.terrain_type, width, height, changed)
                if piece.terrain_type == 'marsh':
                    visible = 0
                    for horizontal in range(-2, 3):
                        for vertical in range(-2, 3):
                            sample = Point3(piece.center.x + horizontal * piece.width / 10,
                                            piece.center.y + vertical * piece.height / 10, .05)
                            projected = Point2()
                            assert lens.project(app.cam.getRelativePoint(app.render, sample), projected)
                            pixel_x = int((projected.x + 1) * width / 2)
                            pixel_y = int((1 - projected.y) * height / 2)
                            visible += (image.getXel(pixel_x, pixel_y) - background.getXel(pixel_x, pixel_y)).length() > .05
                    if visible < 12:
                        background.write(Filename.fromOsSpecific(str(root / '.pytest_cache' / 'reed_fens_hidden_marsh.png')))
                    assert visible >= 12, (
                        f'center={tuple(piece.center)}, camera={camera_side}, size={width}x{height}, '
                        f'visible={visible}, film={tuple(lens.getFilmSize())}, pixel=({pixel_x},{pixel_y}), '
                        f'shown={tuple(image.getXel(pixel_x, pixel_y))}, hidden={tuple(background.getXel(pixel_x, pixel_y))}')
                piece.visual.show()
                app.graphicsEngine.renderFrame()
        finally:
            app.graphicsEngine.removeWindow(buffer)