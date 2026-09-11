"""Actual Chaos character gifts, Stupidity and reload (RH p. 116; Rulebook p. 178)."""

import asyncio
from unittest.mock import AsyncMock, patch

from chaos_gifts import apply_gift, begin_turn, start_and_command, succumbed
from magic_items import current_turn
from persistence import load_game_state, save_game_state
from tests.test_faction_rules_scene import members, scene as scene
from tests.test_shieldwall_scene import combat_tasks


def test_gifts_and_stupidity_survive_repeated_reload_and_expire_once(scene, tmp_path):
    app, baseline = scene
    load_game_state(app, baseline)
    champion = members(app)['Aspiring Champion']
    initiative = int(champion.unit.model.characteristics['I'])
    attacks = int(champion.unit.model.characteristics['A'])
    apply_gift(champion, 1)
    apply_gift(champion, 2)
    apply_gift(champion, 5)
    champion.stupidityFailed = True
    app.roundCounter.current_player = 2
    app.chaosCommandTurn = current_turn(app)
    saved = save_game_state(app, str(tmp_path / 'gifts.json'))
    for _ in range(2):
        load_game_state(app, saved)
        champion = members(app)['Aspiring Champion']
        assert succumbed(champion)
        assert int(champion.unit.model.characteristics['I']) == initiative + 1
        assert int(champion.unit.model.characteristics['A']) == attacks + 1
        with patch.object(app, 'makeChoiceNew', AsyncMock()) as choice:
            asyncio.run(start_and_command(app))
        choice.assert_not_awaited()
    app.roundCounter.currentRoundPlayer[1] += 1
    with patch.object(app, 'makeChoiceNew', AsyncMock(return_value='Decline')), \
            patch('chaos_gifts.random.randint', return_value=1):
        asyncio.run(start_and_command(app))
    assert not succumbed(champion)
    assert int(champion.unit.model.characteristics['I']) == initiative
    assert int(champion.unit.model.characteristics['A']) == attacks + 1


def test_restoration_does_not_schedule_command_choices(scene):
    app, baseline = scene
    load_game_state(app, baseline)
    app.roundCounter.current_player = 2
    app.chaosCommandTurn = None
    with patch.object(app, 'restoringBattle', True), patch.object(app.taskMgr, 'add') as add:
        begin_turn(app)
    add.assert_not_called()


def test_live_command_entry_blocks_phase_advance_until_gaze_choice(scene):
    import game_fsm
    app, baseline = scene
    load_game_state(app, baseline)
    app.roundCounter.current_player = 2
    champion = members(app)['Aspiring Champion']
    attacks = int(champion.unit.model.characteristics['A'])
    with combat_tasks(app) as run, patch.object(game_fsm, 'taskMgr', app.taskMgr, create=True), \
            patch.object(app, 'makeChoiceNew', AsyncMock(return_value='Roll')), \
            patch('chaos_gifts.random.randint', return_value=5):
        app.fsm.request('StrategyPhase')
        assert app.magicBusy
        app.fsm.nextPhase()
        assert app.fsm.state == 'StrategyPhase'
        async def finish_command_task():
            from direct.task import Task
            while app.chaosCommandBusy:
                await Task.pause(0)
        run(finish_command_task())
    assert not app.magicBusy
    assert int(champion.unit.model.characteristics['A']) == attacks + 1