"""Loaded cavalry movement previews accept direct and legacy wrapped mounts."""

import math
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from panda3d.core import Vec3

from persistence import load_game_state
from tests.test_faction_rules_scene import members, scene as scene


@pytest.mark.parametrize('name, wrapped', [
    ('Silver Helm', False), ('Silver Helm', True),
    ('Dragon Prince', False), ('Dragon Prince', True),
    ('Chaos Knight', False), ('Chaos Knight', True),
    ('Mage', False),
])
def test_loaded_unit_movement_preview_preserves_mount_and_position(scene, name, wrapped, monkeypatch):
    app, baseline = scene
    load_game_state(app, baseline)
    app.terrain_manager.clear()
    unit = members(app)[name]
    profile = unit.unit.model
    mount = profile.get_mount()
    if name != 'Mage':
        assert mount is not None and not hasattr(mount, 'model')
        if wrapped:
            monkeypatch.setattr(profile, 'special_rules', list(profile.special_rules))
            profile.attach_mount(SimpleNamespace(model=mount))
    else:
        assert mount is None
    unit.request('Idle')
    unit.bodyNP.setPos(0, -15, 0)
    unit.bodyNP.setH(0)
    origin = Vec3(unit.bodyNP.getPos())
    facing = Vec3(unit.bodyNP.getHpr())
    movement = app.movement.movementAllowance(unit)
    scale = 2 * abs(app.ground.getTightBounds()[0][1])
    with patch.object(app.movement, 'pointArc', wraps=app.movement.pointArc) as arcs, \
            patch.object(app, 'setGroundOverlay', wraps=app.setGroundOverlay) as overlay:
        app.pathTowardsMouse(unit, 0, -12)
    assert len(arcs.call_args_list) == 2
    assert arcs.call_args_list[-1].kwargs['movedistance'] == pytest.approx(int(2 * movement) / scale)
    assert overlay.call_args.args[0] is True
    assert app.arcPoint is not None
    assert all(math.isfinite(value) for point in app.polygonpoints for value in point)
    assert unit.bodyNP.getPos().almostEqual(origin)
    assert unit.bodyNP.getHpr().almostEqual(facing)
    assert profile.get_mount() is mount