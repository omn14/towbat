"""Live command and split-profile regressions (Rulebook pp. 192-201, 210-211)."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from direct.interval.IntervalGlobal import Sequence
from panda3d.core import Filename, getModelPath, loadPrcFileData

from battleFunctions import melee_attacks
from challenges import Challenge, duellist
from command_groups import champions, has_command
from game import MyApp
from persistence import load_game_state, save_game_state
from tests.test_shieldwall_scene import combat_tasks

ROOT = Path(__file__).resolve().parents[1]


def knights_spec():
    return {'name': 'Chaos Knight', 'nmodels': 4, 'files': 4, 'ranks': 1,
            'mount': 'Chaos Steed', 'armour': ['Heavy Armour', 'Shield'],
            'weapons': [{'name': 'Hand Weapon'}, {'name': 'Lance'}],
            'command': [
                {'role': 'champion', 'name': 'Champion', 'selection_ref': 'knights/champion',
                 'profiles': [{'name': 'Champion', 'typeName': 'Model', 'characteristics': [
                     {'name': 'A', '$text': '2'}, {'name': 'I', '$text': '4'}]}]},
                {'role': 'standard_bearer', 'name': 'Standard Bearer', 'selection_ref': 'knights/standard'},
                {'role': 'musician', 'name': 'Musician', 'selection_ref': 'knights/musician'}]}


@pytest.fixture(scope='module')
def scene(tmp_path_factory):
    loadPrcFileData('', 'window-type offscreen\nwin-size 1280 720\naudio-library-name null')
    getModelPath().appendDirectory(Filename.fromOsSpecific(str(ROOT)))

    def first(app, _):
        app._create_unit(knights_spec(), 1, 'Knights')

    def second(app, _):
        app._create_unit({'name': 'Lothern Skycutter', 'nmodels': 1, 'files': 1, 'ranks': 1}, 2, 'Skycutter')

    with patch.object(MyApp, 'load_player1_army', first), patch.object(MyApp, 'load_player2_army', second):
        app = MyApp()
    app.AIplayer2.active = False
    app.fsm.request('MovementPhase')
    for index, member in enumerate(app.units):
        member.isDeployed = True
        member.bodyNP.setPos(0, index * 12, 0)
        member.bodyNP.setH(180 if index else 0)
    path = save_game_state(app, str(tmp_path_factory.mktemp('command') / 'baseline.json'))
    yield app, path
    app.destroy()


@pytest.mark.parametrize('size', [(1280, 720), (800, 600)])
def test_character_editor_render_bounds(scene, tmp_path, size):
    from character_movement import open_editor
    from characters import join_unit
    from panda3d.core import PNMImage
    app, baseline = scene
    load_game_state(app, baseline)
    app.chargeStage = 'remaining'
    host = app.player1Units[0]
    for index in range(2):
        member = app._create_unit(dict(name='Noble', nmodels=1, files=1, ranks=1), 1, f'Editor Noble {index}')
        assert join_unit(app, member, host)
    app.unitToMove = host
    editor = open_editor(app)
    assert editor is not None
    editor.menu.set(1)
    aspect = app.getAspectRatio()
    window = app.openWindow(type='offscreen', size=size, makeCamera=False)
    try:
        for order, camera in enumerate((app.cam, app.cam2d, app.cam2dp)):
            region = window.makeDisplayRegion()
            region.setCamera(camera)
            region.setSort(order * 10)
        app.adjustWindowAspectRatio(size[0] / size[1])
        app.graphicsEngine.renderFrame()
        app.graphicsEngine.renderFrame()
        image = PNMImage()
        assert window.getScreenshot(image)
        assert image.write(Filename.fromOsSpecific(str(tmp_path / f'character-editor-{size[0]}.png')))
        colors = {tuple(image.getXel(horizontal, vertical)) for horizontal in range(20, 190, 20)
                  for vertical in range(60, 180, 20)}
        assert len(colors) > 5
        bounds = editor.status.getTightBounds(editor.panel)
        assert bounds[0].z > editor.confirm_button.getZ() + .03
        assert editor.panel.getWidth() + .04 < 2 * size[0] / size[1]
    finally:
        editor.cancel()
        app.closeWindow(window)
        app.adjustWindowAspectRatio(aspect)


@pytest.mark.parametrize('size', [(1280, 720), (800, 600)])
def test_movement_readout_keeps_character_options_clear(scene, tmp_path, size):
    from characters import join_unit
    from panda3d.core import PNMImage
    from skirmish_ui import show_plot_status
    app, baseline = scene
    load_game_state(app, baseline)
    app.chargeStage = 'remaining'
    character = app._create_unit(dict(name='Noble', nmodels=1, files=1, ranks=1), 1, 'Movement Options Noble')
    character.isDeployed = True
    host = app.player1Units[0]
    aspect = app.getAspectRatio()
    window = app.openWindow(type='offscreen', size=size, makeCamera=False)
    try:
        for order, camera in enumerate((app.cam, app.cam2d, app.cam2dp)):
            region = window.makeDisplayRegion()
            region.setCamera(camera)
            region.setSort(order * 10)
        app.adjustWindowAspectRatio(size[0] / size[1])
        for joined in (False, True):
            if joined:
                assert join_unit(app, character, host)
            app.unitToMove = host if joined else character
            app.refreshSelectedUnit()
            button = app.characterMoveButton
            assert not button.isHidden()
            assert button['text'] == ('Leave unit' if joined else 'Join unit')
            for error in (None, 'The destination overlaps another unit; choose a different position.'):
                preview = SimpleNamespace(error=error, charge_target=None, marched=False, distance=2.68, allowance=5)
                show_plot_status(app, preview)
                status = app.skirmishMoveStatus
                top = status.getZ() + status.getBounds()[3]
                for control in (app.skirmishAdjustButton, button):
                    if not control.isHidden():
                        assert top <= control.getZ() + control.getBounds()[2] * control.getSz() - .02
                app.graphicsEngine.renderFrame()
                app.graphicsEngine.renderFrame()
                image = PNMImage()
                assert window.getScreenshot(image)
                assert image.write(Filename.fromOsSpecific(str(tmp_path / f'movement-options-{size[0]}-{joined}-{bool(error)}.png')))
            button['command']()
            assert app.characterMoveEditor is not None
            app.characterMoveEditor.cancel()
    finally:
        if getattr(app, 'characterMoveEditor', None) is not None:
            app.characterMoveEditor.cancel()
        app.closeWindow(window)
        app.adjustWindowAspectRatio(aspect)


def test_load_places_command_without_extra_bodies(scene, tmp_path):
    app, baseline = scene
    load_game_state(app, baseline)
    knights = app.player1Units[0]
    assert knights.unit.nmodels == len(knights.model.getChildren()) == 4
    children = {child.getPythonTag('command_role'): child for child in knights.model.getChildren()
                if child.hasPythonTag('command_role')}
    assert set(children) == {'champion', 'standard_bearer', 'musician'}
    assert children['standard_bearer'].getX() == pytest.approx(2 * knights.modelWidth)
    assert all(child.getY() == 0 for child in children.values())
    app.graphicsEngine.renderFrame()
    app.graphicsEngine.renderFrame()
    assert app.screenshot(Filename.fromOsSpecific(str(tmp_path / 'command.png')).getFullpath(), defaultFilename=False)


def test_live_casualties_preserve_then_remove_command_and_reload(scene, tmp_path):
    app, baseline = scene
    load_game_state(app, baseline)
    knights = app.player1Units[0]
    app.movement.removeModelsFromUnit(knights, 2)
    assert knights.unit.nmodels == 2
    assert has_command(knights, 'champion') and has_command(knights, 'standard_bearer')
    assert not has_command(knights, 'musician')
    path = save_game_state(app, str(tmp_path / 'casualties.json'))
    load_game_state(app, baseline)
    load_game_state(app, path)
    assert knights.unit.nmodels == 2 and not has_command(knights, 'musician')


def test_multiple_characters_mixed_bases_and_reload(scene, tmp_path):
    from characters import join_unit, get_joined_characters
    from scouts import model_base_boxes
    from psychology import _box_corners
    from shapely.geometry import Polygon
    app, baseline = scene
    load_game_state(app, baseline)
    host = app._create_unit(dict(name='Elven Spearman', nmodels=15, files=6, ranks=3), 1, 'Mixed Spears')
    for index, size in enumerate(((25, 25), (50, 50), (30, 60))):
        character = app._create_unit(dict(name='Noble', nmodels=1, files=1, ranks=1), 1, f'Mixed Noble {index}')
        character.modelWidth, character.modelHeight = (dimension / 25.4 for dimension in size)
        character.baseSize = size
        assert join_unit(app, character, host)
    assert len(get_joined_characters(host)) == 3
    assert len(host.model.getChildren()) == 15
    boxes = model_base_boxes(host)
    assert len(boxes) == 18
    polygons = [Polygon(_box_corners(*box)) for box in boxes]
    assert all(first.intersection(second).area < 1e-6 for index, first in enumerate(polygons)
               for second in polygons[index + 1:])
    assert len(host.characterPlacements['Mixed Noble 1']['cells']) == 4
    assert host.characterPlacements['Mixed Noble 2']['adjacent']
    from combat_profiles import combat_profiles
    from challenges import duellists
    enemy = app.player2Units[0]
    parts = [part for part in combat_profiles(host, enemy) if part.role == 'character']
    assert len(parts) == 3
    assert len({id(part.character) for part in parts}) == 3
    assert len(duellists(host)) == 3
    path = save_game_state(app, str(tmp_path / 'multiple-characters.json'))
    load_game_state(app, path)
    assert [member.unitName for member in get_joined_characters(host)] == [f'Mixed Noble {index}' for index in range(3)]
    assert all(member.bodyNP.node() not in app.world.getRigidBodies() for member in get_joined_characters(host))


@pytest.mark.parametrize('second_character', [False, True])
def test_remaining_moves_join_and_leave(scene, tmp_path, second_character):
    from panda3d.core import Point3
    from character_movement import preview, commit
    from characters import get_joined_characters, leave_reason
    app, baseline = scene
    load_game_state(app, baseline)
    app.terrain_manager.clear()
    app.fsm.request('MovementPhase')
    app.chargeStage = 'remaining'
    app.roundCounter.current_player = 1
    for other in app.units:
        other.isDeployed = False
    host = app._create_unit(dict(name='Elven Spearman', nmodels=10, files=5, ranks=2), 1, 'Join Host')
    character = app._create_unit(dict(name='Noble', nmodels=1, files=1, ranks=1), 1, 'Joining Noble')
    host.isDeployed = character.isDeployed = True
    host.bodyNP.setPos(0, 0, 0)
    character.bodyNP.setPos(0, -5, 0)
    before = host.bodyNP.getTransform(), character.bodyNP.getTransform()
    planned = preview(app, character, host=host)
    assert planned.error is None
    assert (host.bodyNP.getTransform(), character.bodyNP.getTransform()) == before
    assert commit(app, character, host=host)
    assert get_joined_characters(host) == [character]
    assert host.joinedMovementLocked and not host.hasMovedThisTurn
    first = character
    if second_character:
        character = app._create_unit(dict(name='Noble', nmodels=1, files=1, ranks=1), 1, 'Second Joining Noble')
        character.isDeployed = True
        character.bodyNP.setPos(4, 0, 0)
        assert commit(app, character, host=host)
        assert get_joined_characters(host) == [first, character]
    assert leave_reason(app, character)
    saved = save_game_state(app, str(tmp_path / 'join-lock.json'))
    load_game_state(app, saved)
    assert host.joinedMovementLocked and not host.hasMovedThisTurn
    assert app.movement.moveUnit(host) is None
    host.joinedMovementLocked = False
    character.hasMovedThisTurn = False
    character.moveSpentThisTurn = 0
    destination = tuple(character.bodyNP.getPos(app.render) + Point3(0, 4, 0))
    assert preview(app, character, destination=destination).error is None
    from character_movement import open_editor
    app.unitToMove = host
    editor = open_editor(app)
    assert editor is not None
    if second_character:
        editor.menu.set(1)
        assert editor.character is character
    original = character.bodyNP.getTransform(), host.bodyNP.getTransform()
    editor.destination = destination
    editor.redraw()
    app.graphicsEngine.renderFrame()
    app.graphicsEngine.renderFrame()
    assert app.screenshot(str(tmp_path / f'character-departure-{second_character}.png'), defaultFilename=False)
    status_bounds = editor.status.getTightBounds(editor.panel)
    assert status_bounds[0].z > editor.confirm_button.getZ() + .03
    assert editor.menu.getWidth() * editor.menu.getScale().x <= .651
    app.fsm.nextPhase()
    assert app.fsm.state == 'MovementPhase'
    editor.cancel()
    assert (character.bodyNP.getTransform(), host.bodyNP.getTransform()) == original
    editor = open_editor(app)
    if second_character:
        editor.menu.set(1)
    editor.destination = destination
    editor.redraw()
    assert editor.confirm()
    assert get_joined_characters(host) == ([first] if second_character else [])
    assert character.hostUnit is None and character in app.player1Units
    assert character.bodyNP.node() in app.world.getRigidBodies()
    assert not host.hasMovedThisTurn and character.hasMovedThisTurn


@pytest.mark.parametrize('fleeing', [False, True])
def test_catalogue_mounts_reload_and_multiple_survivors(scene, tmp_path, fleeing):
    from characters import get_joined_characters, join_unit
    from scouts import model_base_boxes
    from character_movement import polygon
    app, baseline = scene
    load_game_state(app, baseline)
    host = app._create_unit(dict(name='Elven Spearman', nmodels=15, files=6, ranks=3), 1, 'Mounted Escort')
    host.isDeployed = True
    members = []
    for index, (name, mount) in enumerate((('Glade Lord', 'Great Stag'), ('Noble', 'Elven Steed'))):
        character = app._create_unit(dict(name=name, mount=mount, nmodels=1, files=1, ranks=1), 1,
                                     f'Mounted Noble {index}')
        assert join_unit(app, character, host)
        members.append(character)
    assert len(host.characterPlacements[members[0].unitName]['cells']) == 4
    assert host.characterPlacements[members[1].unitName]['adjacent']
    before = model_base_boxes(host)
    if fleeing:
        host.request('IsFleeing')
    saved = save_game_state(app, str(tmp_path / 'mounted-escort.json'))
    for _ in range(2):
        load_game_state(app, saved)
        assert get_joined_characters(host) == members
        assert model_base_boxes(host) == pytest.approx(before)
    polygons = [polygon(box) for box in before]
    assert all(first.intersection(second).area < 1e-6 for index, first in enumerate(polygons)
               for second in polygons[index + 1:])
    with patch.object(app.psychology, 'on_unit_destroyed'):
        app.movement.removeModelsFromUnit(host, host.unit.nmodels)
    assert host not in app.units
    for member in members:
        assert member in app.units and member in app.player1Units and member.hostUnit is None
        assert member.unit.nmodels == 1
        assert member.bodyNP.node() in app.world.getRigidBodies()
        assert (member.state == 'IsFleeing') is fleeing


@pytest.mark.parametrize('heading', [0, 37, 180])
@pytest.mark.parametrize('with_command', [False, True])
@pytest.mark.parametrize('count_already_reduced', [False, True])
def test_casualties_keep_joined_character_in_front_rank(scene, heading, with_command, count_already_reduced):
    from characters import join_unit
    app, baseline = scene
    load_game_state(app, baseline)
    spec = dict(name='Elven Spearman', nmodels=14, files=5, ranks=3)
    if with_command:
        spec['command'] = knights_spec()['command']
    host = app._create_unit(spec, 1, 'Casualty Spears')
    character = app._create_unit(dict(name='Noble', nmodels=1, files=1, ranks=1), 1, 'Casualty Noble')
    host.bodyNP.setPos(7, -8, 0)
    host.bodyNP.setH(heading)
    assert join_unit(app, character, host)
    transform = host.bodyNP.getTransform()
    for casualties in (4, 1, 5, 1):
        if count_already_reduced:
            host.unit.nmodels -= casualties
        app.removeModelsFromUnit(host, casualties)
        position = character.bodyNP.getPos(host.model)
        assert position.y == pytest.approx(0, abs=1e-5)
        assert position.x == pytest.approx(host.characterSlot * host.modelWidth, abs=1e-5)
        assert host.characterSlot < host.unit.files
        assert character.hostUnit is host and character.bodyNP.getParent() == host.bodyNP
        assert host.bodyNP.getTransform() == transform
        assert len(host.model.getChildren()) == host.unit.nmodels
        assert all((child.getPos(app.render) - character.bodyNP.getPos(app.render)).length() > 1e-5
                   for child in host.model.getChildren())
    character.retiredFromCombat = True
    host.placeCharacter()
    app.removeModelsFromUnit(host, 1)
    position = character.bodyNP.getPos(host.model)
    assert position.y == pytest.approx(-host.modelHeight, abs=1e-5)
    assert character.retiredFromCombat and character.hostUnit is host


def test_live_scheduler_uses_later_mount_count_and_chariot_wounds(scene):
    app, baseline = scene
    load_game_state(app, baseline)
    knights, skycutter = app.player1Units[0], app.player2Units[0]
    knights.unit.model.characteristics['I'] = '6'
    champions(knights)[0].unit.model.characteristics['I'] = '6'
    knights.unit.model.get_mount().characteristics['I'] = '2'
    app.attackers = [knights, skycutter, knights]
    app.defenders = [skycutter, knights, skycutter]
    app.attackSequence = Sequence()
    resolver = app.combat
    resolver._combatStartModels = {id(member.unit): member.unit.nmodels for member in app.units}
    resolver._pendingWounds = {}
    calls = []

    def fight(attacker, target, charge, **kwargs):
        attacks = melee_attacks(attacker, charge)
        calls.append((attacker.model.name, attacks))
        wounds = 1 if attacker.model.name == 'Sea Guard Crew' else 0
        return attacks, wounds, wounds, 0, wounds

    removals = Sequence()
    with patch('combat_resolution.simulate_battle', side_effect=fight), \
            patch('combat_resolution.take_last_slaying_blows', return_value=0):
        assert resolver.resolveMeleeProfiles(None, removals) == [0, 1]
    assert calls == [('Chaos Knight', 3), ('Champion', 2), ('Sea Guard Crew', 3),
                     ('Swiftfeather Roc', 2), ('Chaos Steed', 3)]
    assert knights.unit.nmodels == 3
    removals.finish()
    assert len(knights.model.getChildren()) == knights.unit.nmodels == 3


def test_champion_challenge_roundtrip_and_no_wound_spill(scene, tmp_path):
    app, baseline = scene
    load_game_state(app, baseline)
    knights, other = app.player1Units[0], app.player2Units[0]
    rival = app._create_unit(knights_spec(), 2, 'Rival Knights')
    challenge = Challenge(duellist(knights), knights, duellist(rival), rival)
    app.challenges = [challenge]
    path = save_game_state(app, str(tmp_path / 'duel.json'))
    load_game_state(app, baseline)
    load_game_state(app, path)
    restored = app.challenges[0]
    assert restored.challenger is duellist(app.player1Units[0])
    assert restored.accepter.command_host.unitName == 'Rival Knights'
    before = restored.accepter.command_host.unit.nmodels
    assert app.combat.woundDuellist(restored.accepter, 10)
    assert restored.accepter.command_host.unit.nmodels == before - 1
    assert not has_command(restored.accepter.command_host, 'champion')


def test_live_rally_uses_musician_bonus(scene):
    app, baseline = scene
    load_game_state(app, baseline)
    knights = app.player1Units[0]
    knights.request('IsFleeing')
    with combat_tasks(app) as run, patch.object(app, 'rollLeadershipDice', AsyncMock(return_value=[4, 4])), \
            patch.object(app, 'aiControls', return_value=True), \
            patch.object(app.psychology, 'leadership_of', return_value=(7, None)):
        run(app.rallyUnit(knights))
    assert knights.state == 'Idle'


def test_skycutter_bows_fire_as_three_crew_not_one_hull(scene):
    app, baseline = scene
    load_game_state(app, baseline)
    knights, skycutter = app.player1Units[0], app.player2Units[0]
    profile = skycutter.unit.model
    for owner in (profile, profile.get_crew()):
        owner.give_weapon('Shortbow')
        owner.equip_weapon('Shortbow')
    with combat_tasks(app) as run, \
            patch('game.simulate_battle', return_value=(3, 0, 0, 0, 0)) as fire, \
            patch.object(app, 'shootingAnimation', AsyncMock()):
        run(app.shootAt(skycutter, knights))
    shooting_unit = fire.call_args.args[0]
    assert shooting_unit.model is profile.get_crew()
    assert shooting_unit.nmodels == shooting_unit.files == 3
    assert shooting_unit.model.firing_bs() == 4


@pytest.mark.parametrize('renown', [False, True])
def test_final_combat_tie_includes_stand_and_shoot_before_musician(scene, renown):
    app, baseline = scene
    load_game_state(app, baseline)
    knights, skycutter = app.player1Units[0], app.player2Units[0]
    knights.isInCombatWith = [skycutter]
    skycutter.isInCombatWith = [knights]
    knights.isInCombatFlank = skycutter.isInCombatFlank = ['front']
    skycutter.standAndShootWounds = 1
    if renown:
        from magic_items import install_inventory
        install_inventory(knights, [{'name': 'Banner of Renown', 'category': 'Magic Standards',
                                    'selection_ref': 'knights/banner', 'owner_ref': 'knights/standard'}])
    app.unitToMove = knights
    table = []

    class Scored(Exception):
        pass

    def capture(rows, scores, strength):
        table.append((rows, scores))
        raise Scored

    async def choose(options, *args, **kwargs):
        return 'No challenge' if 'No challenge' in options else next(iter(options))

    async def resolve():
        try:
            await app.combat._verySimpleBattleInner(SimpleNamespace(done='done'))
        except Scored:
            pass

    with combat_tasks(app) as run, patch.object(app, 'makeChoiceNew', AsyncMock(side_effect=choose)), \
            patch('combat_resolution.simulate_battle', return_value=(1, 0, 0, 0, 0)), \
            patch('combat_resolution.take_last_slaying_blows', return_value=0), \
            patch.object(app.combat, 'printCombatResult', side_effect=capture):
        run(resolve())
    rows, scores = table[0]
    assert rows['Standard Bearer'] == (1, 0)
    assert rows['Stand & Shoot'] == (0, 1)
    assert rows['Banner of Renown'] == ((1, 0) if renown else (0, 0))
    assert rows['Musician'] == ((0, 0) if renown else (1, 0))
    assert scores == (2, 1)


def test_failed_march_stays_at_origin_blocks_phase_and_survives_reload(scene, tmp_path):
    from direct.task import Task
    from marching import nearby_enemy, request_march

    app, baseline = scene
    load_game_state(app, baseline)
    knights, skycutter = app.player1Units[0], app.player2Units[0]
    skycutter.bodyNP.setPos(0, 5, 0)
    assert nearby_enemy(app, knights)[0] is skycutter
    origin = tuple(knights.bodyNP.getPos())
    completed = []

    async def attempt():
        assert not request_march(app, knights, lambda: completed.append(True))
        phase = app.fsm.state
        app.fsm.nextPhase()
        assert app.fsm.state == phase
        while knights.marchTestResult == 'pending':
            await Task.pause(0.01)

    with combat_tasks(app) as run, patch.object(app, 'rollLeadershipDice', AsyncMock(return_value=[6, 6])), \
            patch.object(app, 'startTaskFunction'), patch.object(app, 'aiControls', return_value=True):
        run(attempt())
    assert not completed
    assert tuple(knights.bodyNP.getPos()) == origin
    assert not knights.hasMovedThisTurn and knights.marchedThisTurn
    assert knights.marchTestResult == 'failed'
    path = save_game_state(app, str(tmp_path / 'failed-march.json'))
    load_game_state(app, baseline)
    load_game_state(app, path)
    assert knights.marchTestResult == 'failed' and knights.marchedThisTurn
    assert not knights.hasMovedThisTurn