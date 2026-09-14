"""Fresh-process Battle March startup, isolated from other Panda3D scenes."""

import asyncio
from copy import deepcopy
from pathlib import Path
from unittest.mock import AsyncMock, patch

from panda3d.core import Filename, getModelPath, loadPrcFileData

from battle_config import load_config
from battle_preparation import run_preparation
from game import MyApp


def test_explicit_startup_preserves_visual_board_and_holds_deployment(tmp_path):
    root = Path(__file__).resolve().parents[1]
    loadPrcFileData('', 'window-type offscreen\nwin-size 1280 720\naudio-library-name null')
    getModelPath().appendDirectory(Filename.fromOsSpecific(str(root)))
    config = load_config()
    config['terrain']['feature_count'] = 0

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
    finally:
        app.destroy()