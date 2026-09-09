"""Offscreen Move Through Cover regressions and reproducible playable save."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from panda3d.core import Point3, getModelPath, loadPrcFileData

from characters import join_unit
from game import MyApp
from persistence import load_game_state, save_game_state
from special_rules import apply_rule_keywords
from spell_system import PillarOfFireSpell

ROOT = Path(__file__).resolve().parents[1]


def build_scenario():
    loadPrcFileData('', 'window-type offscreen\nwin-size 1280 720\naudio-library-name null')
    getModelPath().appendDirectory(str(ROOT))
    placements = {}

    def add(app, player, name, xpos, protected=False):
        unit = app._create_unit(dict(name='Dwarf Warrior', nmodels=10, files=5, ranks=2),
                                player, name)
        assert unit is not None
        apply_rule_keywords(unit.unit.model, ['Move Through Cover'] if protected else [], replace=True)
        unit.unit.model.characteristics['M'] = '4'
        placements[name] = (xpos, -9 if player == 1 else -1, 0 if player == 1 else 180)
        return unit

    def player_one(app, _):
        add(app, 1, 'Cover Woods', -24, True)
        add(app, 1, 'Ordinary Woods', -12)
        add(app, 1, 'Cover Slow Escort', 0, True)
        add(app, 1, 'Ordinary Marsh', 12)
        add(app, 1, 'Cover Fast Escort', 24, True)

    def player_two(app, _):
        add(app, 2, 'Woods Target A', -24)
        add(app, 2, 'Woods Target B', -12)

    with patch.object(MyApp, 'load_player1_army', player_one), \
            patch.object(MyApp, 'load_player2_army', player_two):
        app = MyApp()
    app.AIplayer2.active = False
    for unit in app.units:
        xpos, ypos, heading = placements[unit.unitName]
        unit.bodyNP.setPos(xpos, ypos, 0)
        unit.bodyNP.setH(heading)
    app.terrain_manager.clear()
    app.terrain_manager.add_terrain('forest', Point3(-18, -7, 0), 24, 18, 'difficult')
    app.terrain_manager.add_terrain('marsh', Point3(12, -7, 0), 36, 18, 'dangerous')
    for name, movement in [('Cover Slow Escort', 4), ('Cover Fast Escort', 5)]:
        host = next(unit for unit in app.units if unit.unitName == name)
        character = app._create_unit(dict(name='Captain of the Empire', nmodels=1,
                                         files=1, ranks=1), 1, name + ' Captain')
        assert character is not None
        apply_rule_keywords(character.unit.model, [], replace=True)
        character.unit.model.characteristics['M'] = str(movement)
        if movement == 5:
            host.unit.model.characteristics['M'] = '3'
        assert join_unit(app, character, host)
    for unit in app.units:
        unit.isDeployed = True
        unit.scoutDeploymentChoice = 'normal'
        unit.deployedAsScouts = False
        unit.request('Idle')
        unit.hasMovedThisTurn = False
        app.movement.alignModelsToHillNormal(unit)
        app.movement.updateDisrupted(unit)
    app.deploymentStage = 'ordinary'
    app.fsm.request('MovementPhase')
    app.fsm.currentPhaseIndex = app.fsm.phases.index('MovementPhase')
    app.roundCounter.current_player = 1
    app.roundCounter.currentRoundPlayer = [0, 0]
    app.roundCounter.enterPlayerOne()
    app.unitToMove = app.player1Units[0]
    app.refreshSelectedUnit()
    return app


@pytest.fixture(scope='module')
def scene(tmp_path_factory):
    app = build_scenario()
    path = save_game_state(app, str(tmp_path_factory.mktemp('cover') / 'baseline.json'))
    yield app, path
    app.destroy()


def restore(scene, name):
    app, path = scene
    load_game_state(app, path)
    unit = next(unit for unit in app.units if unit.unitName == name)
    app.unitToMove = unit
    return app, unit


def verify_scenario(app):
    assert app.fsm.state == 'MovementPhase'
    assert app.roundCounter.current_player == 1
    assert not app.AIplayer2.active
    assert len(app.units) == 9
    assert [(piece.terrain_type, piece.going) for piece in app.terrain_manager.terrain_pieces] == [
        ('forest', 'difficult'), ('marsh', 'dangerous')]
    for name, expected in [('Cover Woods', 4), ('Ordinary Woods', 3),
                           ('Cover Slow Escort', 3), ('Ordinary Marsh', 3),
                           ('Cover Fast Escort', 3)]:
        unit = next(unit for unit in app.units if unit.unitName == name)
        start = unit.bodyNP.getPos()
        assert app.movement.movementAllowance(unit, start, start + Point3(0, 2, 0)) == expected
        assert not unit.hasMovedThisTurn
    assert all(unit.joinedCharacter is not None for unit in app.player1Units if 'Escort' in unit.unitName)


def test_save_restores_terrain_and_character_faq(scene, tmp_path):
    app, path = scene
    app.terrain_manager.clear()
    app.terrain_manager.add_terrain('hill', Point3(0), 10, 10)
    load_game_state(app, path)
    verify_scenario(app)
    app.graphicsEngine.renderFrame()
    app.graphicsEngine.renderFrame()
    assert app.screenshot(str(tmp_path / 'cover-loaded.png'), defaultFilename=False)


@pytest.mark.parametrize('name,maximum', [('Cover Woods', 8), ('Ordinary Woods', 6),
                                        ('Cover Slow Escort', 6), ('Cover Fast Escort', 6)])
def test_actual_ranked_preview_uses_terrain_allowance(scene, name, maximum):
    app, unit = restore(scene, name)
    for enemy in app.player2Units:
        enemy.bodyNP.setPos(30, 20, 0)
    origin = unit.bodyNP.getPos()
    app.pathTowardsMouse(unit, origin.x, origin.y + 20)
    assert app.moveArceDistance == pytest.approx(maximum, abs=0.06)


def test_dangerous_rerolls_apply_real_wounds_to_character_only(scene, capsys):
    app, unit = restore(scene, 'Cover Slow Escort')
    character = unit.joinedCharacter
    origin = unit.bodyNP.getPos()
    with patch('terrain_system.random.randint', side_effect=[1, 6] * 10 + [1]):
        wounds = app.movement.dangerousTerrainTests(unit, origin, origin + Point3(0, 2, 0))
    assert wounds == 1
    assert unit.unit.nmodels == 10 and character.unit.nmodels == 1
    assert character.woundsOnModel == 1
    assert '10 mishap(s) avoided' in capsys.readouterr().out


def test_protection_does_not_prevent_disruption(scene):
    app, unit = restore(scene, 'Cover Slow Escort')
    assert app.movement.updateDisrupted(unit)


@pytest.mark.parametrize('protected,maximum', [(True, 8), (False, 6)])
def test_skirmisher_preview_and_committed_move(scene, protected, maximum, capsys):
    app, ranked = restore(scene, 'Cover Woods' if protected else 'Ordinary Woods')
    for enemy in app.player2Units:
        enemy.bodyNP.setPos(30, 20, 0)
    origin = ranked.bodyNP.getPos()
    ranked.bodyNP.setPos(-30, 20, 0)
    unit = app._create_unit(dict(name='Ranger', nmodels=5, files=5, ranks=1,
                                 special_rules=['Skirmishers']), 1, 'Cover Skirmishers')
    assert unit is not None and unit.isSkirmisher
    apply_rule_keywords(unit.unit.model, ['Skirmishers'] +
                        (['Move Through Cover'] if protected else []), replace=True)
    unit.unit.model.characteristics['M'] = '4'
    unit.bodyNP.setPos(origin)
    unit.bodyNP.setH(0)
    unit.isDeployed = True
    unit.request('Idle')
    app.unitToMove = unit
    capsys.readouterr()
    app.pathTowardsMouse(unit, origin.x, origin.y + 20)
    assert app.moveArceDistance == pytest.approx(maximum)
    app.pathTowardsMouse(unit, origin.x, origin.y + 3.5)
    assert 'Move Through Cover' not in capsys.readouterr().out
    app.movement.moveUnit(unit)
    assert (unit.bodyNP.getPos() - origin).length() == pytest.approx(3.5)
    assert unit.hasMovedThisTurn
    assert unit.marchedThisTurn is (not protected)
    output = capsys.readouterr().out
    if protected:
        assert 'Move Through Cover' in output and 'terrain -1M' in output
        assert output.count('unit allowance 4"') == 1
    else:
        assert 'Move Through Cover' not in output


def test_old_save_preserves_existing_battlefield(scene, tmp_path):
    app, baseline = scene
    data = json.loads(Path(baseline).read_text())
    data.pop('terrain')
    path = tmp_path / 'old.json'
    path.write_text(json.dumps(data))
    app.terrain_manager.clear()
    hill = app.terrain_manager.add_terrain('hill', Point3(0), 10, 10)
    load_game_state(app, str(path))
    assert app.terrain_manager.terrain_pieces == [hill]


def test_saved_spell_terrain_is_not_duplicated(scene, tmp_path):
    app, caster = restore(scene, 'Cover Woods')
    spell = PillarOfFireSpell('Pillar of Fire', 9, app.fsm.endOfTurnSpells,
                              game=app, caster=caster)
    spell.place(app, Point3(0, 15, 0.1))
    path = save_game_state(app, str(tmp_path / 'vortex.json'))
    assert len(json.loads(Path(path).read_text())['terrain']) == 2
    for _ in range(2):
        load_game_state(app, path)
        assert len(app.terrain_manager.terrain_pieces) == 3
        assert len(app.remainsInPlay) == 1


if __name__ == '__main__':
    app = build_scenario()
    try:
        path = save_game_state(app, 'move_through_cover.json')
        app.terrain_manager.clear()
        load_game_state(app, path)
        verify_scenario(app)
        app.unitToMove = next(unit for unit in app.units if unit.unitName == 'Cover Slow Escort')
        app.refreshSelectedUnit()
        app.graphicsEngine.renderFrame()
        app.graphicsEngine.renderFrame()
        assert app.screenshot(str(ROOT / 'screenshots' / 'move_through_cover.png'),
                              defaultFilename=False)
        print(f'Verified Move Through Cover save: {path}')
    finally:
        app.destroy()