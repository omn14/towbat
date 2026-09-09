"""Live command and split-profile regressions (Rulebook pp. 192-201, 210-211)."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from direct.interval.IntervalGlobal import Sequence
from panda3d.core import getModelPath, loadPrcFileData

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
    getModelPath().appendDirectory(str(ROOT))

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
    assert app.screenshot(str(tmp_path / 'command.png'), defaultFilename=False)


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


def test_final_combat_tie_includes_stand_and_shoot_before_musician(scene):
    app, baseline = scene
    load_game_state(app, baseline)
    knights, skycutter = app.player1Units[0], app.player2Units[0]
    knights.isInCombatWith = [skycutter]
    skycutter.isInCombatWith = [knights]
    knights.isInCombatFlank = skycutter.isInCombatFlank = ['front']
    skycutter.standAndShootWounds = 1
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
    assert rows['Musician'] == (1, 0)
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