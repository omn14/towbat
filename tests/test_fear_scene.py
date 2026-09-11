"""Skycutter Fear against the actual Chaos roster (Rulebook p. 168)."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from panda3d.core import Vec3

from charge_declarations import begin_declarations
from fear import test_fear
from persistence import load_game_state, save_game_state
from tests.test_faction_rules_scene import members, scene as scene
from tests.test_shieldwall_scene import combat_tasks


def test_skycutter_fear_uses_current_strength_and_persists_result(scene, tmp_path):
    app, baseline = scene
    load_game_state(app, baseline)
    units = members(app)
    skycutter, warriors = units['Lothern Skycutter'], units['Chaos Warrior']
    with patch('fear.random.randint') as roll:
        assert asyncio.run(test_fear(app, warriors, [skycutter], 'combat chosen'))
    roll.assert_not_called()
    app.movement.applyWounds(warriors, 8)
    with patch.object(app, 'makeChoiceNew', AsyncMock(return_value='Keep')), \
            patch('fear.random.randint', return_value=6):
        assert not asyncio.run(test_fear(app, warriors, [skycutter], 'combat chosen'))
    saved = save_game_state(app, str(tmp_path / 'fear.json'))
    load_game_state(app, saved)
    with patch('fear.random.randint') as roll:
        assert not asyncio.run(test_fear(app, warriors, [skycutter], 'combat chosen'))
    roll.assert_not_called()


def test_failed_skycutter_fear_charge_stays_at_origin_without_reaction(scene):
    app, baseline = scene
    load_game_state(app, baseline)
    units = members(app)
    skycutter, knights = units['Lothern Skycutter'], units['Chaos Knight']
    app.movement.applyWounds(knights, 3)
    app.roundCounter.current_player = 2
    app.fsm.request('MovementPhase')
    begin_declarations(app)
    origin, facing = Vec3(0, -12, 0), Vec3(0, 0, 0)
    knights.bodyNP.setPos(origin)
    knights.bodyNP.setHpr(facing)
    skycutter.bodyNP.setPos(0, 0, 0)
    with combat_tasks(app) as run, patch.object(app, 'aiControls', return_value=True), \
            patch('fear.random.randint', return_value=6), \
            patch.object(app.combat, 'chargeInterval', AsyncMock()) as move:
        run(app.combat.chargeAndChargeReaction(knights, SimpleNamespace(), origin, facing,
                                              SimpleNamespace(done=None), defender=skycutter))
    assert knights.bodyNP.getPos().almostEqual(origin)
    assert knights.hasMovedThisTurn and knights.fearFailed
    assert knights.chargeAttempts == 1 and not knights.chargeAttemptPending
    assert app.chargeDeclarations == []
    move.assert_not_awaited()