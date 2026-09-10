"""Playable Shieldwall test save and offscreen integration (Rulebook p. 177)."""

import json
from contextlib import contextmanager, ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from direct.interval.IntervalGlobal import Sequence, ivalMgr
from direct.task.Task import TaskManager
from panda3d.core import AsyncTaskManager, ClockObject, Vec3, getModelPath, loadPrcFileData

import combat_resolution
import game as game_module
import movement_system
from game import MyApp
from persistence import load_game_state, save_game_state
from psychology import shieldwall_unavailable_reason
from special_rules import apply_rule_keywords

ROOT = Path(__file__).resolve().parents[1]
DEFENDERS = ('Shieldwall Ready', 'Shieldwall Spent', 'Shieldwall Weapon Choice')


def build_scenario():
    loadPrcFileData('', 'window-type offscreen\nwin-size 1280 720\naudio-library-name null')
    getModelPath().appendDirectory(str(ROOT))

    def add(app, player, name, count):
        unit = app._create_unit(dict(name='Dwarf Warrior', nmodels=count,
                                    files=5, ranks=count // 5), player, name)
        assert unit is not None
        rules = ['Close Order'] + (['Shieldwall', 'Stubborn'] if player == 1 else [])
        apply_rule_keywords(unit.unit.model, rules, replace=True)
        unit.unit.model.set_armour(['Heavy Armour', 'Shield'])
        unit.unit.model.characteristics['A'] = '0'
        unit.unit.model.characteristics['Ld'] = '10'
        unit.unit.model._base_characteristics = dict(unit.unit.model.characteristics)
        return unit

    def player_one(app, _):
        for name in DEFENDERS:
            unit = add(app, 1, name, 5)
            if name == 'Shieldwall Weapon Choice':
                unit.unit.model.give_weapon('Great Weapon')

    def player_two(app, _):
        for index in range(3):
            add(app, 2, f'Charger {index + 1}', 10)

    with patch.object(MyApp, 'load_player1_army', player_one), \
            patch.object(MyApp, 'load_player2_army', player_two):
        app = MyApp()
    app.AIplayer2.active = False
    app.terrain_manager.clear()
    for index, (defender, charger) in enumerate(zip(app.player1Units, app.player2Units)):
        xpos = -18 + index * 18
        depth = sum(unit.bodyNP.node().getShape(0).getHalfExtentsWithMargin().y
                    for unit in (defender, charger))
        defender.bodyNP.setPos(xpos, -6, 0)
        defender.bodyNP.setH(0)
        charger.bodyNP.setPos(xpos, -6 + depth - 0.01, 0)
        charger.bodyNP.setH(180)
        for unit, enemy in ((defender, charger), (charger, defender)):
            unit.isDeployed = True
            unit.scoutDeploymentChoice = 'normal'
            unit.request('InCombat')
            unit.isInCombat = True
            unit.isInCombatWith = [enemy]
            unit.isInCombatFlank = ['front']
            unit.hasMovedThisTurn = True
            unit.hasAttackedThisTurn = False
            app.movement.alignModelsToHillNormal(unit)
        defender.wasChargedThisTurn = True
        defender.usedShieldwall = index == 1
        charger.chargedThisTurn = True
        charger.chargeDistance = 4.0
    app.deploymentStage = 'ordinary'
    app.fsm.request('CombatPhase')
    app.fsm.currentPhaseIndex = app.fsm.phases.index('CombatPhase')
    app.roundCounter.current_player = 2
    app.roundCounter.currentRoundPlayer = [0, 0]
    app.roundCounter.enterPlayerTwo()
    app.unitToMove = app.player2Units[0]
    app.refreshSelectedUnit()
    return app


def verify_scenario(app):
    assert app.fsm.state == 'CombatPhase'
    assert app.roundCounter.current_player == 2
    assert not app.AIplayer2.active
    assert len(app.units) == 6
    for index, (defender, charger) in enumerate(zip(app.player1Units, app.player2Units)):
        assert defender.unitName == DEFENDERS[index]
        assert defender.wasChargedThisTurn and not defender.chargedThisTurn
        assert defender.usedShieldwall is (index == 1)
        assert not defender.usedStubborn
        assert defender.isInCombatWith == [charger]
        assert charger.isInCombatWith == [defender]
        assert not defender.hasAttackedThisTurn and not charger.hasAttackedThisTurn
        assert app.world.contactTestPair(defender.bodyNP.node(), charger.bodyNP.node()).getNumContacts()
    assert shieldwall_unavailable_reason(app.player1Units[0]) is None
    assert 'already used' in shieldwall_unavailable_reason(app.player1Units[1])


@pytest.fixture(scope='module')
def scene(tmp_path_factory):
    app = build_scenario()
    path = save_game_state(app, str(tmp_path_factory.mktemp('shieldwall') / 'baseline.json'))
    yield app, path
    app.destroy()


def test_save_reloads_ready_and_spent_units_and_renders(scene, tmp_path):
    app, path = scene
    for unit in app.units:
        unit.usedShieldwall = not unit.usedShieldwall
        unit.wasChargedThisTurn = False
    load_game_state(app, path)
    verify_scenario(app)
    app.graphicsEngine.renderFrame()
    app.graphicsEngine.renderFrame()
    assert app.screenshot(str(tmp_path / 'loaded.png'), defaultFilename=False)


def test_spent_rule_and_deferred_charge_survive_reload(scene, tmp_path):
    app, baseline = scene
    load_game_state(app, baseline)
    defender = app.player1Units[0]
    defender.usedShieldwall = True
    defender.countsAsChargeTargetNextTurn = True
    path = save_game_state(app, str(tmp_path / 'spent.json'))
    defender.usedShieldwall = False
    defender.countsAsChargeTargetNextTurn = False
    load_game_state(app, path)
    assert app.player1Units[0].usedShieldwall
    assert app.player1Units[0].countsAsChargeTargetNextTurn
    load_game_state(app, baseline)
    assert not app.player1Units[0].usedShieldwall
    assert not app.player1Units[0].countsAsChargeTargetNextTurn


def test_old_save_defaults_to_unused_and_no_incoming_charge(scene, tmp_path):
    app, baseline = scene
    records = json.loads(Path(baseline).read_text())
    for record in records['units']:
        for flag in ('usedShieldwall', 'wasChargedThisTurn', 'countsAsChargeTargetNextTurn'):
            record.pop(flag)
    path = tmp_path / 'legacy.json'
    path.write_text(json.dumps(records))
    load_game_state(app, str(path))
    assert all(not unit.usedShieldwall and not unit.wasChargedThisTurn
               and not unit.countsAsChargeTargetNextTurn for unit in app.units)


@contextmanager
def combat_tasks(app):
    tasks = TaskManager()
    tasks.mgr = AsyncTaskManager('isolated-shieldwall')
    clock = ClockObject.getGlobalClock()
    previous_mode, previous_dt = clock.getMode(), clock.getDt()
    clock.setMode(ClockObject.MNonRealTime)
    clock.setDt(0.1)
    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(app, 'taskMgr', tasks))
            for module in (combat_resolution, game_module, movement_system):
                stack.enter_context(patch.object(module, 'taskMgr', tasks, create=True))

            def run(coroutine):
                completed = []

                async def resolve():
                    await coroutine
                    completed.append(True)

                tasks.add(resolve())
                for _ in range(1000):
                    clock.tick()
                    ivalMgr.step()
                    tasks.step()
                    if completed:
                        break
                assert completed, 'combat did not finish within 100 simulated seconds'

            yield run
    finally:
        tasks.removeTasksMatching('*')
        clock.setDt(previous_dt)
        clock.setMode(previous_mode)


@pytest.mark.parametrize('index', [0, 2])
def test_full_combat_gives_ground_and_follows_up_without_panic(scene, index, capsys):
    app, baseline = scene
    load_game_state(app, baseline)
    defender, charger = app.player1Units[index], app.player2Units[index]
    origins = [Vec3(unit.bodyNP.getPos()) for unit in (defender, charger)]
    app.unitToMove = charger
    app.resolvingCombat = True

    async def choose(options, *args, **kwargs):
        for preferred in ('Hand weapon & shield', 'Stand Firm', 'Shieldwall', 'Follow up'):
            if preferred in options:
                return preferred
        return next(iter(options))

    with combat_tasks(app) as run, \
            patch.object(app, 'makeChoiceNew', AsyncMock(side_effect=choose)) as choices, \
            patch.object(app.psychology, 'on_unit_flees_combat') as panic:
        run(app.combat._verySimpleBattleInner(SimpleNamespace(done='done')))
        panic.assert_not_called()
    assert defender.usedStubborn and defender.usedShieldwall
    assert not defender.fledThisPhase
    assert not app.resolvingCombat
    assert any('Shieldwall' in call.args[0] for call in choices.call_args_list)
    for unit, origin in zip((defender, charger), origins):
        assert unit.bodyNP.getPos().y == pytest.approx(origin.y - 2, abs=0.02)
        assert unit.bodyNP.getH() == pytest.approx(0 if unit is defender else 180)
    assert defender.unit.nmodels == 5 and charger.unit.nmodels == 10
    assert 'Give Ground 2"' in capsys.readouterr().out


@pytest.mark.parametrize('handler', ['chargeInterval', '_skirmishChargeInterval'])
@pytest.mark.parametrize('distance, reached', [(4, True), (12, False)])
def test_charge_contact_records_target_only_when_reached(scene, handler, distance, reached):
    app, baseline = scene
    load_game_state(app, baseline)
    defender, charger = app.player1Units[0], app.player2Units[0]
    defender.request('Idle')
    charger.request('Idle')
    defender.wasChargedThisTurn = False
    charger.chargedThisTurn = False
    contact = Vec3(charger.bodyNP.getPos())
    origin = contact + Vec3(0, distance, 0)
    app.playerNP.setPos(contact)
    app.moveArceDistance = distance
    app.autoRoll = True
    app.attackSequence2 = Sequence()
    with combat_tasks(app) as run:
        if handler == 'chargeInterval':
            run(app.combat.chargeInterval(charger, defender.bodyNP, 0, origin,
                                         Vec3(180, 0, 0), 'front', chdice=[2, 2]))
        else:
            run(app.combat._skirmishChargeInterval(charger, defender.bodyNP, origin,
                                                 Vec3(180, 0, 0), 'front', chdice=[2, 2]))
    assert defender.wasChargedThisTurn is reached
    assert charger.chargedThisTurn is reached
    assert not defender.countsAsChargeTargetNextTurn


@pytest.mark.parametrize('handler', ['chargeInterval', '_skirmishChargeInterval'])
def test_caught_fallback_defers_charge_target_to_next_turn(scene, handler):
    app, baseline = scene
    load_game_state(app, baseline)
    defender, pursuer = app.player1Units[0], app.player2Units[0]
    apply_rule_keywords(pursuer.unit.model, ['First Charge'])
    pursuer.chargeAttempts = 0
    defender.request('Moved')
    pursuer.request('IsPursuing')
    defender.wasChargedThisTurn = False
    contact = Vec3(pursuer.bodyNP.getPos())
    app.playerNP.setPos(contact)
    app.moveArceDistance = 4
    app.autoRoll = True
    app.attackSequence2 = Sequence()
    with combat_tasks(app) as run:
        if handler == 'chargeInterval':
            run(app.combat.chargeInterval(pursuer, defender.bodyNP, 0, contact + Vec3(0, 4, 0),
                                         Vec3(180, 0, 0), 'front', chdice=[2, 2]))
        else:
            run(app.combat._skirmishChargeInterval(pursuer, defender.bodyNP, contact + Vec3(0, 4, 0),
                                                  Vec3(180, 0, 0), 'front', chdice=[2, 2]))
    assert defender.countsAsChargeTargetNextTurn
    assert pursuer.countsAsChargedNextTurn
    assert pursuer.chargeAttempts == 1 and not pursuer.chargeAttemptPending
    assert not defender.firstChargeDisruptedBy
    assert defender.firstChargeDisruptedNextTurnBy == [pursuer.unit.name]


@pytest.mark.parametrize('joins', [False, True])
def test_overrun_records_charge_target_and_deferred_combat(scene, joins):
    app, baseline = scene
    load_game_state(app, baseline)
    defender, overrunning = app.player1Units[0], app.player2Units[0]
    apply_rule_keywords(overrunning.unit.model, ['First Charge'])
    overrunning.chargeAttempts = 0
    defender.request('Idle')
    overrunning.request('Moved')
    defender.wasChargedThisTurn = False
    defender.startOfPhaseEngaged = False
    with combat_tasks(app) as run, \
            patch.object(app.combat, 'joinsCombatThisPhase', return_value=(joins, 'test combat timing')):
        run(app.combat.overrunContact(overrunning, 4))
    assert defender.wasChargedThisTurn
    assert defender.countsAsChargeTargetNextTurn is not joins
    assert overrunning.countsAsChargedNextTurn is not joins
    assert overrunning.chargeAttempts == 1
    assert bool(defender.firstChargeDisruptedBy) is joins
    assert bool(defender.firstChargeDisruptedNextTurnBy) is not joins


if __name__ == '__main__':
    app = build_scenario()
    try:
        path = save_game_state(app, 'shieldwall.json')
        load_game_state(app, path)
        verify_scenario(app)
        app.unitToMove = app.player2Units[0]
        app.refreshSelectedUnit()
        app.graphicsEngine.renderFrame()
        app.graphicsEngine.renderFrame()
        assert app.screenshot(str(ROOT / 'screenshots' / 'shieldwall.png'), defaultFilename=False)
        print(f'Verified Shieldwall save: {path}')
    finally:
        app.destroy()