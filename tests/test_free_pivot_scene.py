"""Bounded facing placement on the selected Warhounds and Skycutter."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from free_pivot import begin, constrain, pending, rule_for
from persistence import load_game_state, save_game_state
from tests.test_faction_rules_scene import members, scene as scene


def restore(scene, name):
    app, baseline = scene
    load_game_state(app, baseline)
    app.terrain_manager.clear()
    unit = members(app)[name]
    unit.bodyNP.setPos(0, -10, 0)
    unit.bodyNP.setH(0)
    unit.marchedThisTurn = unit.chargedThisTurn = unit.isInCombat = False
    unit.request('Moved')
    return app, unit


@pytest.mark.parametrize('name, rule', [('Chaos Warhound', 'Quick Turn'),
                                      ('Lothern Skycutter', 'Lumbering')])
def test_free_pivot_clamps_angle_and_does_not_translate(scene, name, rule):
    app, unit = restore(scene, name)
    origin = unit.bodyNP.getPos()
    with patch.object(app, 'aiControls', return_value=False), \
            patch.object(app, 'startFreeReform') as reform:
        assert begin(app, unit, 'Remaining Moves')
    assert unit.freePivot['rule'] == rule and pending(app)
    unit.bodyNP.setH(140)
    assert constrain(app, unit)
    assert unit.bodyNP.getH() == pytest.approx(90)
    assert unit.bodyNP.getPos() == origin
    reform.call_args.kwargs['on_done']()
    assert not pending(app)


def test_quick_turn_is_remaining_moves_only_and_never_after_marching(scene):
    app, unit = restore(scene, 'Chaos Warhound')
    assert rule_for(app, unit, 'Reserve Move') is None
    unit.marchedThisTurn = True
    assert not begin(app, unit, 'Remaining Moves')
    assert getattr(unit, 'freePivot', None) is None


def test_pivot_cannot_confirm_overlapping_friendly_or_crossing_board_edge(scene):
    app, unit = restore(scene, 'Lothern Skycutter')
    with patch.object(app, 'aiControls', return_value=False), \
            patch.object(app, 'startFreeReform') as reform:
        assert begin(app, unit, 'Remaining Moves')
    friend = members(app)['Silver Helm']
    friend.bodyNP.setPos(unit.bodyNP.getPos())
    assert not constrain(app, unit)
    friend.bodyNP.setPos(20, 10, 0)
    unit.bodyNP.setX(35.9)
    assert not constrain(app, unit)
    reform.call_args.kwargs['on_done']()


def test_pending_pivot_blocks_phase_move_save_and_load(scene, tmp_path):
    app, unit = restore(scene, 'Chaos Warhound')
    with patch.object(app, 'aiControls', return_value=False), \
            patch.object(app, 'startFreeReform') as reform:
        assert begin(app, unit, 'Remaining Moves')
    app.fsm.nextPhase()
    assert app.fsm.state == 'MovementPhase'
    assert app.movement.moveUnit(unit) is False
    assert save_game_state(app, str(tmp_path / 'pending-pivot.json')) is None
    load_game_state(app, scene[1])
    assert pending(app)
    reform.call_args.kwargs['on_done']()
    assert save_game_state(app, str(tmp_path / 'finished-pivot.json')) is not None


def test_remaining_move_completion_offers_free_pivot(scene):
    from panda3d.core import Vec2
    app, unit = restore(scene, 'Chaos Warhound')
    unit.request('Idle')
    app.chargeStage = 'remaining'
    app.arcPoint = Vec2(.5, .43)
    app.arcPointRotation = 0
    app.moveArceDistance = 1
    with patch.object(app, 'checkUnitContactSmall', return_value=None), \
            patch('free_pivot.begin') as pivot:
        app.movement.moveUnit(unit)
    pivot.assert_called_once_with(app, unit, 'Remaining Moves')


def test_mouse_facing_task_confirms_and_renders_bounded_pivot(scene, tmp_path):
    app, unit = restore(scene, 'Lothern Skycutter')
    task = SimpleNamespace(done='done', cont='cont')
    with patch.object(app, 'aiControls', return_value=False), \
            patch.object(app, 'startFreeReform') as reform:
        assert begin(app, unit, 'Remaining Moves')
    unit.bodyNP.setH(-120)
    app.signal = True
    with patch.object(app, 'mouseWatcherNode', SimpleNamespace(hasMouse=lambda: False)):
        assert app.freeReformUnit(unit, task) == task.done
    assert unit.bodyNP.getH() == pytest.approx(-90)
    reform.call_args.kwargs['on_done']()
    app.camera.setPos(0, -28, 25)
    app.camera.lookAt(0, -9, 0)
    app.graphicsEngine.renderFrame()
    app.graphicsEngine.renderFrame()
    assert app.screenshot(str(tmp_path / 'skycutter-free-pivot.png'), defaultFilename=False)