"""Pursuit completion includes dice, movement and reform (Rulebook pp. 156-157)."""

from unittest.mock import AsyncMock, patch

import pytest
from direct.task import Task
from panda3d.core import Vec3

from tests.test_counter_charge_scene import declared_charge
from tests.test_faction_rules_scene import members, scene as scene
from tests.test_shieldwall_scene import combat_tasks


@pytest.mark.parametrize('fall_back_first', [True, False])
@pytest.mark.parametrize('departure', ['fall_back', 'break'])
def test_follow_up_drops_separated_opponent_from_both_combat_lists(
        scene, tmp_path, fall_back_first, departure):
    """Mixed Break results only retain actual Follow Up contact (pp. 154-156)."""
    from direct.interval.IntervalGlobal import Sequence
    from combat_contacts import engaged_units
    from persistence import load_game_state, save_game_state
    from psychology import obb_distance
    app, baseline = scene
    load_game_state(app, baseline)
    app.fsm.request('CombatPhase')
    app.attackSequence = Sequence()
    roster = members(app)
    warriors, princes, helms = (roster[name] for name in ('Chaos Warrior', 'Dragon Prince', 'Silver Helm'))
    for index, other in enumerate(app.units):
        other.bodyNP.setPos(35, 15 + index * 4, 0)
    warriors.bodyNP.setPos(0, 0, 0)
    warriors.bodyNP.setH(180)
    princes.bodyNP.setPos(-(helms.unitWidth + .1) / 2,
                          -(warriors.unitHeight + princes.unitHeight) / 2, 0)
    helms.bodyNP.setPos((princes.unitWidth + .1) / 2,
                       -(warriors.unitHeight + helms.unitHeight) / 2, 0)
    for loser in (princes, helms):
        loser.bodyNP.setH(0)
        loser.isInCombatWith = [warriors]
        loser.isInCombatFlank = ['front']
        loser.fledThisPhase = False
    warriors.isInCombatWith = [princes, helms]
    warriors.isInCombatFlank = ['left', 'front']
    for unit in (warriors, princes, helms):
        unit.request('InCombat')
        unit.isInCombat = True
    outcomes = [(princes, departure), (helms, 'give_ground')]
    if not fall_back_first:
        outcomes.reverse()
    responses = [dict(winner=warriors, target=helms, action='follow_up')]
    with combat_tasks(app) as run, \
            patch.object(app, 'resolvingCombat', True), \
            patch.object(app.combat, 'swiftstrideChoice', AsyncMock(return_value=False)), \
            patch.object(app.combat, 'rollMoveDice', AsyncMock(return_value=[1, 2])):
        run(app.combat.loserMovePass(outcomes, responses))
        run(app.combat.pursuitPass(outcomes, responses))
    box = app.psychology._unit_box
    assert obb_distance(box(warriors), box(princes)) > .1
    assert obb_distance(box(warriors), box(helms)) < .01
    assert warriors.isInCombatWith == [helms]
    assert warriors.isInCombatFlank == ['front']
    assert helms.isInCombatWith == [warriors]
    assert helms.isInCombat and warriors.isInCombat
    assert princes.isInCombatWith == princes.isInCombatFlank == []
    assert not princes.isInCombat
    assert princes.state == ('Moved' if departure == 'fall_back' else 'IsFleeing')
    assert set(engaged_units(warriors, helms)) == {warriors, helms}
    checkpoint = save_game_state(app, str(tmp_path / 'after-follow-up.json'))
    load_game_state(app, checkpoint)
    assert warriors.isInCombatWith == [helms]
    assert warriors.isInCombatFlank == ['front']
    assert princes.isInCombatWith == [] and not princes.isInCombat
    assert set(engaged_units(warriors, helms)) == {warriors, helms}


def test_pursuit_reestablishes_links_only_after_catching_fallen_back_quarry(scene):
    app, winner, target, origin, facing, contact = declared_charge(scene)
    app.fsm.request('CombatPhase')
    winner.bodyNP.setPos(0, -(winner.unitHeight + target.unitHeight) / 2, 0)
    winner.bodyNP.setH(0)
    target.bodyNP.setPos(0, 0, 0)
    target.bodyNP.setH(180)
    target.fledThisPhase = False
    for unit, enemy in ((winner, target), (target, winner)):
        unit.request('InCombat')
        unit.isInCombatWith = [enemy]
        unit.isInCombatFlank = ['front']
    responses = [dict(winner=winner, target=target, action='pursue')]
    with combat_tasks(app) as run, \
            patch.object(app, 'resolvingCombat', True), \
            patch.object(app, 'makeChoiceNew', AsyncMock(return_value='Keep formation')), \
            patch.object(app.combat, 'swiftstrideChoice', AsyncMock(return_value=False)), \
            patch.object(app.combat, 'rollMoveDice', AsyncMock(return_value=[1, 2])), \
            patch.object(app.combat, 'rullTerninger', AsyncMock(return_value=([], [6, 6]))):
        run(app.combat.loserMovePass([(target, 'fall_back')], responses))
        assert target.isInCombatWith == winner.isInCombatWith == []
        run(app.combat.pursuitPass([(target, 'fall_back')], responses))
    assert winner.isInCombatWith == [target]
    assert target.isInCombatWith == [winner]
    assert winner.isInCombat and target.isInCombat
    assert len(winner.isInCombatFlank) == len(target.isInCombatFlank) == 1


@pytest.mark.parametrize('advance', [0, 3])
def test_pursuit_waits_for_movement_and_reform_completion(scene, advance):
    app, winner, target, origin, facing, contact = declared_charge(scene)
    app.fsm.request('CombatPhase')
    winner.bodyNP.setPos(origin)
    events = []

    async def confirm(on_done):
        await Task.pause(12)
        winner.bodyNP.setH(90)
        events.append('reform-done')
        on_done()

    def reform(unit, on_done):
        assert unit is winner
        events.append('reform-start')
        app.taskMgr.add(confirm(on_done), 'confirm-test-reform')

    async def movement():
        await Task.pause(12)
        winner.bodyNP.setY(winner.bodyNP.getY() + advance)
        winner.request('Moved')
        await app.combat.freeReform(winner)
        events.append('move-done')

    app.hud.clear_log()
    with combat_tasks(app) as run, \
            patch.object(app, 'pathTowardsMouse'), \
            patch.object(app, 'moveUnit', side_effect=lambda unit, **kwargs: app.taskMgr.add(movement(), 'test-pursuit-move')), \
            patch.object(app, 'startFreeReform', side_effect=reform), \
            patch.object(app, 'aiControls', return_value=False):
        run(app.combat.pursuitMove(winner, target, 'flee'))
        assert events == ['reform-start', 'reform-done', 'move-done']
        assert winner.bodyNP.getH() == 90
        assert winner.pursuitQuarry is None
    summaries = [entry for entry in app.hud._journal.visible('Summary')
                 if entry.text.startswith('Pursuit:')]
    assert len(summaries) == 1
    assert winner.unit.name in summaries[0].text
    assert target.unit.name in summaries[0].text
    assert f'{advance:.2f}"' in summaries[0].text
    assert summaries[0].subject == winner.unit.name
    assert 'not caught' in summaries[0].text
    assert 'net ground displacement' in summaries[0].details
    assert app.hud._journal.visible('Summary', winner.unit.name) == summaries


@pytest.mark.parametrize('resolution,expected', [
    ('quarry', 'caught the quarry; locked in combat'),
    ('enemy', 'contacted Chaos Warrior Unit; quarry not caught'),
    ('removed', 'pursuer removed during the move'),
])
def test_pursuit_summary_reports_resolved_contact_or_loss(scene, resolution, expected):
    app, winner, target, origin, facing, contact = declared_charge(scene)
    app.fsm.request('CombatPhase')
    winner.bodyNP.setPos(origin)
    enemy = members(app)['Chaos Warrior']

    async def movement():
        winner.bodyNP.setY(winner.bodyNP.getY() + 2)
        if resolution == 'removed':
            app.combat.removeUnitFromPlay(winner)
        else:
            winner.request('InCombat')
            winner.isInCombat = True
            winner.isInCombatWith = [target if resolution == 'quarry' else enemy]

    app.hud.clear_log()
    with combat_tasks(app) as run, \
            patch.object(app, 'pathTowardsMouse'), \
            patch.object(app, 'makeChoiceNew', AsyncMock(return_value='Keep formation')), \
            patch.object(app, 'moveUnit', side_effect=lambda unit, **kwargs:
                         app.taskMgr.add(movement(), 'test-pursuit-outcome')):
        run(app.combat.pursuitMove(winner, target, 'fall_back'))
    summaries = [entry for entry in app.hud._journal.visible('Summary')
                 if entry.text.startswith('Pursuit:')]
    assert len(summaries) == 1
    assert expected in summaries[0].text


def test_next_pursuer_uses_confirmed_reform_even_after_quarry_removed(scene):
    app, first, target, origin, facing, contact = declared_charge(scene)
    app.fsm.request('CombatPhase')
    first.bodyNP.setPos(origin)
    second = members(app)['Dragon Prince']
    second.bodyNP.setPos(-10, -10, 0)
    destination = Vec3(target.bodyNP.getPos())
    events = []

    async def confirm(on_done):
        await Task.pause(12)
        first.bodyNP.setH(90)
        events.append('reform-done')
        on_done()

    def reform(unit, on_done):
        app.taskMgr.add(confirm(on_done), 'confirm-test-reform')

    def preview(unit, horizontal, vertical):
        if unit is second:
            assert events == ['first-move', 'reform-done']
            assert first.bodyNP.getH() == 90
            assert target.bodyNP.isEmpty()
            assert (horizontal, vertical) == (destination.x, destination.y)
        events.append('first-move' if unit is first else 'second-move')

    async def move(unit):
        unit.request('Moved')
        if unit is first:
            app.combat.removeUnitFromPlay(target)
            await app.combat.freeReform(unit)

    responses = [dict(winner=unit, target=target, action='pursue') for unit in (first, second)]
    app.hud.clear_log()
    with combat_tasks(app) as run, \
            patch.object(app.combat, 'stillEngaged', return_value=False), \
            patch.object(app, 'makeChoiceNew', AsyncMock(return_value='Keep formation')), \
            patch.object(app, 'pathTowardsMouse', side_effect=preview), \
            patch.object(app, 'moveUnit', side_effect=lambda unit, **kwargs: app.taskMgr.add(move(unit), 'test-pursuit-move')), \
            patch.object(app, 'startFreeReform', side_effect=reform), \
            patch.object(app, 'aiControls', return_value=False):
        run(app.combat.pursuitPass([(target, 'flee')], responses))
    assert events == ['first-move', 'reform-done', 'second-move']
    assert first.pursuitQuarry is second.pursuitQuarry is None
    summaries = [entry for entry in app.hud._journal.visible('Summary')
                 if entry.text.startswith('Pursuit:')]
    assert len(summaries) == 2
    assert 'caught and destroyed' in summaries[0].text
    assert 'quarry was already removed' in summaries[1].text


def test_live_capture_reform_finishes_before_second_path_and_dice(scene, capsys):
    app, second, target, origin, facing, contact = declared_charge(scene)
    first = members(app)['Dragon Prince']
    app.fsm.request('CombatPhase')
    target.request('IsFleeing')
    first.bodyNP.setPos(0, -8, 0)
    second.bodyNP.setPos(-10, -2, 0)
    for unit in (first, second):
        unit.request('Idle')
        unit.isInCombat = False
        unit.isInCombatWith = []
        unit.isInCombatFlank = []
    events = []
    confirmed = []
    path = app.pathTowardsMouse
    start_reform = app.startFreeReform

    def preview(unit, horizontal, vertical):
        if unit is second:
            assert events[-1] == 'reform-confirmed'
            assert not app._reformActive
            assert first.bodyNP.getH() == 90
            assert target.bodyNP.isEmpty()
        events.append('first-path' if unit is first else 'second-path')
        return path(unit, horizontal, vertical)

    async def roll(count, bonus=False):
        events.append('dice')
        await Task.pause(12)
        return [], [6, 6]

    async def confirm():
        await Task.pause(12)
        assert events == ['first-path', 'dice', 'reform-start']
        first.bodyNP.setH(90)
        confirmed.append(True)

    def reform(unit, on_done=None):
        assert unit is first
        events.append('reform-start')
        start_reform(unit, on_done)
        app.taskMgr.add(confirm(), 'confirm-live-reform')

    def reform_input(unit, task):
        if confirmed:
            events.append('reform-confirmed')
            return task.done
        return task.cont

    responses = [dict(winner=unit, target=target, action='pursue') for unit in (first, second)]
    app.hud.clear_log()
    with combat_tasks(app) as run, \
            patch.object(app, 'resolvingCombat', True), \
            patch.object(app, 'makeChoiceNew', AsyncMock(return_value='Keep formation')), \
            patch.object(app, 'pathTowardsMouse', side_effect=preview), \
            patch.object(app, 'startFreeReform', side_effect=reform), \
            patch.object(app, 'freeReformUnit', side_effect=reform_input), \
            patch.object(app, 'aiControls', return_value=False), \
            patch.object(app.combat, 'swiftstrideChoice', AsyncMock(return_value=False)), \
            patch.object(app.combat, 'rullTerninger', AsyncMock(side_effect=roll)):
        try:
            run(app.combat.pursuitPass([(target, 'flee')], responses))
        except AssertionError as error:
            raise AssertionError(f'{error}; events={events}; {capsys.readouterr()}') from error
    assert events == ['first-path', 'dice', 'reform-start', 'reform-confirmed', 'second-path', 'dice']
    assert not app._reformActive and not app._reformQueue
    assert first.pursuitQuarry is second.pursuitQuarry is None
    summaries = [entry for entry in app.hud._journal.visible('Summary')
                 if entry.text.startswith('Pursuit:')]
    assert [entry.subject for entry in summaries] == [first.unit.name, second.unit.name]
    assert 'caught and destroyed' in summaries[0].text
    assert 'quarry was already removed' in summaries[1].text


def test_game_move_callback_only_returns_task_when_requested(scene):
    app, unit, target, origin, facing, contact = declared_charge(scene)
    completion = object()
    with patch.object(app.movement, 'moveUnit', return_value=completion):
        assert app.moveUnit(unit) is None
        assert app.moveUnit(unit, wait_for_completion=True) is completion