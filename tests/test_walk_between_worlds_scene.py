"""Walk Between Worlds with the actual High Elf/Chaos roster scene (p. 329)."""

from panda3d.core import Point3, Vec2, Vec3
from unittest.mock import AsyncMock, patch
from types import SimpleNamespace
import asyncio
import pytest

from special_rules import apply_rule_keywords, is_ethereal
from tests.test_faction_rules_scene import members, scene as scene
from persistence import load_game_state, save_game_state
from characters import detach_character, join_unit, slay_character
from high_magic import WalkBetweenWorldsSpell
from reserve_move import has_reserve_move
from tests.test_shieldwall_scene import combat_tasks


def test_ethereal_crosses_terrain_not_units_and_cannot_finish_inside(scene):
    app, baseline = scene
    load_game_state(app, baseline)
    mage = members(app)['Mage']
    mage.bodyNP.setPos(0, -12, 0)
    mage.bodyNP.setH(0)
    app.terrain_manager.add_terrain('house', Point3(0, -8, 0), 2, 2)
    origin = mage.bodyNP.getTransform()
    ordinary, _ = app.movement.sweepTestDir(mage, origin, Vec3(0, 1, 0), 8, pass_over=False)
    assert ordinary < 1
    apply_rule_keywords(mage.unit.model, ['Ethereal'])
    spirit, _ = app.movement.sweepTestDir(mage, origin, Vec3(0, 1, 0), 8, pass_over=False)
    assert spirit == 1
    mage.bodyNP.setY(-8)
    assert app.movement.etherealDestinationBlocked(mage)
    mage.bodyNP.setY(-4)
    assert not app.movement.etherealDestinationBlocked(mage)
    mage.bodyNP.setY(-12)
    blocker = members(app)['Chaos Knight']
    blocker.bodyNP.setPos(0, -6, 0)
    blocker.bodyNP.node().setTransformDirty()
    mage.bodyNP.node().setTransformDirty()
    app.world.doPhysics(1 / 60)
    blocked, _ = app.movement.sweepTestDir(mage, origin, Vec3(0, 1, 0), 8, pass_over=False)
    assert blocked < 1


def test_reserve_phase_live_preview_move_reload_and_no_second_move(scene, tmp_path):
    from reserve_move import unavailable
    app, baseline = scene
    load_game_state(app, baseline)
    host = members(app)['Silver Helm']
    host.bodyNP.setPos(0, -15, 0)
    host.bodyNP.setH(0)
    apply_rule_keywords(host.unit.model, ['Reserve Move'])
    host.request('Moved')
    host.moveSpentThisTurn = 2
    app.fsm.request('ShootingPhase')
    with patch('game_fsm.taskMgr', app.taskMgr, create=True):
        app.fsm.nextPhase()
    assert app.fsm.state == 'ReserveMovePhase'
    assert host.moveSpentThisTurn == 0 and not host.hasMovedThisTurn
    app.unitToMove = host
    app.pathTowardsMouse(host, 0, -12)
    assert app.arcPoint is not None and not host.wouldMarch
    app.moveUnit(host)
    assert host.bodyNP.getY() > -15
    assert host.moveSpentThisTurn == 2
    assert unavailable(app, host) is not None
    path = save_game_state(app, str(tmp_path / 'reserve.json'))
    load_game_state(app, path)
    assert app.fsm.state == 'ReserveMovePhase'
    assert unavailable(app, host) is not None
    with patch('game_fsm.taskMgr', app.taskMgr, create=True):
        app.fsm.nextPhase()
    assert app.fsm.state == 'CombatPhase'


def test_reserve_commit_rejects_march_and_enemy_contact(scene):
    app, baseline = scene
    load_game_state(app, baseline)
    host, enemy = members(app)['Silver Helm'], members(app)['Chaos Knight']
    host.bodyNP.setPos(0, -15, 0)
    host.bodyNP.setH(0)
    apply_rule_keywords(host.unit.model, ['Reserve Move'])
    app.fsm.request('ReserveMovePhase')
    origin = host.bodyNP.getPos()
    app.arcPoint = Vec2(.5, .5)
    app.arcPointRotation = 0
    app.moveArceDistance = 15
    app.moveUnit(host)
    assert host.bodyNP.getPos().almostEqual(origin)
    enemy.bodyNP.setPos(0, -12, 0)
    app.arcPoint = Vec2(.5, (-12 + host.unitHeight / 2) / 100 + .5)
    app.moveArceDistance = 3
    app.moveUnit(host)
    assert host.bodyNP.getPos().almostEqual(origin)


def test_live_joined_walk_cast_reload_reserve_move_and_owner_expiry(scene, tmp_path):
    from battlescribe import get_catalogue
    from spell_system import Spell, restore_spellbook
    from spell_effects import end_turn, start_turn
    app, baseline = scene
    load_game_state(app, baseline)
    mage, host = members(app)['Mage'], members(app)['Silver Helm']
    host.bodyNP.setPos(0, -15, 0)
    host.bodyNP.setH(0)
    assert join_unit(app, mage, host)
    restore_spellbook(mage.unit.model, [get_catalogue().spell('Walk Between Worlds')], 2)
    app.fsm.request('MovementPhase')
    app.chargeStage = 'remaining'
    app.unitToMove = host
    with combat_tasks(app) as run, \
            patch.object(app, 'mouseWatcherNode', SimpleNamespace(hasMouse=lambda: False)), \
            patch.object(app, 'makeChoiceNew', AsyncMock(return_value='Walk Between Worlds')), \
            patch.object(Spell, '_roll_casting_dice', AsyncMock(return_value=(10, [5, 5]))), \
            patch.object(app, 'dispelAttempt', AsyncMock(return_value=False)):
        app.castSpell()
        app.taskMgr.remove('taskMagicArcUpdate')
        run(app.taskMagicArcUpdate(SimpleNamespace(done='done')))
    assert mage.spellsCastThisTurn == ['Walk Between Worlds']
    assert app.unitToMove is host and is_ethereal(host.unit.model)
    assert has_reserve_move(host.unit.model)
    path = save_game_state(app, str(tmp_path / 'walk.json'))
    load_game_state(app, path)
    assert mage.hostUnit is host and is_ethereal(host.unit.model)
    with patch('game_fsm.taskMgr', app.taskMgr, create=True):
        app.fsm.nextPhase()
        assert app.fsm.state == 'ShootingPhase'
        app.fsm.nextPhase()
    assert app.fsm.state == 'ReserveMovePhase'
    assert app.castableSpells(mage) == []
    app.graphicsEngine.renderFrame()
    app.graphicsEngine.renderFrame()
    app.screenshot(str(tmp_path / 'walk-reserve.png'), defaultFilename=False)
    phase_save = save_game_state(app, str(tmp_path / 'walk-reserve.json'))
    load_game_state(app, phase_save)
    assert host.reserveMoveOriginal is not None
    app.unitToMove = host
    app.pathTowardsMouse(host, 0, -12)
    app.moveUnit(host)
    assert host.bodyNP.getY() > -15
    assert host.reserveMoveOriginal is None
    end_turn(app)
    assert is_ethereal(host.unit.model)
    app.roundCounter.current_player = 2
    start_turn(app)
    assert is_ethereal(host.unit.model)
    app.roundCounter.current_player = 1
    app.roundCounter.currentRoundPlayer = [1, 1]
    start_turn(app)
    assert not is_ethereal(host.unit.model) and not has_reserve_move(host.unit.model)


def test_retirement_reload_return_to_rank_and_caster_death(scene, tmp_path):
    app, baseline = scene
    load_game_state(app, baseline)
    mage, host = members(app)['Mage'], members(app)['Dragon Prince']
    assert join_unit(app, mage, host)
    walk = WalkBetweenWorldsSpell('Walk Between Worlds', 10, game=app, caster=mage)
    asyncio.run(walk.apply(mage))
    host.request('InCombat')
    app.combat.retireFromCombat(mage, host)
    assert not is_ethereal(host.unit.model) and is_ethereal(mage.unit.model)
    path = save_game_state(app, str(tmp_path / 'retired-self.json'))
    load_game_state(app, path)
    assert not is_ethereal(host.unit.model) and is_ethereal(mage.unit.model)
    host.request('Idle')
    assert not mage.retiredFromCombat and is_ethereal(host.unit.model)
    slay_character(app, mage)
    assert not is_ethereal(host.unit.model)
    assert not has_reserve_move(host.unit.model)
    assert not app.fsm.endOfTurnSpells


def test_ethereal_join_restriction_and_detach_keeps_only_native_host_rule(scene):
    app, baseline = scene
    load_game_state(app, baseline)
    mage, host = members(app)['Mage'], members(app)['Silver Helm']
    walk = WalkBetweenWorldsSpell('Walk Between Worlds', 10, game=app, caster=mage)
    asyncio.run(walk.apply(mage))
    assert not join_unit(app, mage, host)
    apply_rule_keywords(host.unit.model, ['Ethereal'])
    assert join_unit(app, mage, host)
    assert has_reserve_move(host.unit.model)
    mage.bodyNP.wrtReparentTo(app.render)
    detach_character(host)
    app.player1Units.append(mage)
    app.world.attachRigidBody(mage.bodyNP.node())
    assert is_ethereal(host.unit.model) and not has_reserve_move(host.unit.model)
    assert is_ethereal(mage.unit.model) and has_reserve_move(mage.unit.model)


def test_reserve_ai_declines_without_spending_normal_movement(scene):
    from reserve_move import ai_moves, unavailable
    app, baseline = scene
    load_game_state(app, baseline)
    host = members(app)['Silver Helm']
    apply_rule_keywords(host.unit.model, ['Reserve Move'])
    host.moveSpentThisTurn = 2
    origin = host.bodyNP.getPos()
    app.fsm.request('ReserveMovePhase')
    with patch('game_fsm.taskMgr', app.taskMgr, create=True):
        asyncio.run(ai_moves(app))
    assert app.fsm.state == 'CombatPhase'
    assert host.bodyNP.getPos().almostEqual(origin)
    assert host.moveSpentThisTurn == 2 and unavailable(app, host)


def test_shooting_must_finish_before_reserve_window(scene):
    app, baseline = scene
    load_game_state(app, baseline)
    host, enemy = members(app)['Silver Helm'], members(app)['Chaos Knight']
    apply_rule_keywords(host.unit.model, ['Reserve Move'])
    app.fsm.request('ShootingPhase')

    async def volley(*args, **kwargs):
        assert app.shootingInFlight == 1
        app.fsm.nextPhase()
        assert app.fsm.state == 'ShootingPhase'

    with patch.object(app, '_shootAt', AsyncMock(side_effect=volley)), \
            patch('game_fsm.taskMgr', app.taskMgr, create=True):
        asyncio.run(app.shootAt(host, enemy))
        assert app.shootingInFlight == 0
        app.fsm.nextPhase()
    assert app.fsm.state == 'ReserveMovePhase'


def test_drilled_free_redress_precedes_reserve_commit(scene):
    app, baseline = scene
    load_game_state(app, baseline)
    host = members(app)['Dragon Prince']
    host.bodyNP.setPos(0, -15, 0)
    host.bodyNP.setH(0)
    apply_rule_keywords(host.unit.model, ['Reserve Move'])
    app.fsm.request('ReserveMovePhase')
    files = 2 if host.unit.files != 2 else 3
    app.unitToMove = host
    app.pathTowardsMouse(host, 0, -12)

    async def choose(*args, **kwargs):
        app.fsm.nextPhase()
        assert app.fsm.state == 'ReserveMovePhase'
        return f'{files} files'

    async def move():
        await app.movement.moveUnit(host)

    with combat_tasks(app) as run, patch.object(app, 'makeChoiceNew', AsyncMock(side_effect=choose)), \
            patch('game_fsm.taskMgr', app.taskMgr, create=True):
        run(move())
    assert host.unit.files == files
    assert host.bodyNP.getY() > -15
    assert not host._drilledMoveActive and host.reserveMoveOriginal is None
    assert host.moveSpentThisTurn == 0


@pytest.mark.parametrize('name', ['Mage', 'Silver Helm'])
def test_spell_granted_reserve_crosses_house_but_cannot_finish_inside(scene, name):
    app, baseline = scene
    load_game_state(app, baseline)
    mage, host = members(app)['Mage'], members(app)[name]
    host.bodyNP.setPos(0, -15, 0)
    host.bodyNP.setH(0)
    if host is not mage:
        assert join_unit(app, mage, host)
    walk = WalkBetweenWorldsSpell('Walk Between Worlds', 10, game=app, caster=mage)
    asyncio.run(walk.apply(mage))
    app.terrain_manager.add_terrain('house', Point3(0, -12.5, 0), 2, 1)
    app.fsm.request('ReserveMovePhase')
    app.arcPointRotation = 0
    offset = 0 if getattr(host, 'isSkirmisher', False) else host.unitHeight / 2
    app.arcPoint = Vec2(.5, (-12.5 + offset) / 100 + .5)
    app.moveArceDistance = 2.5
    app.moveUnit(host)
    assert host.bodyNP.getY() == -15
    assert host.reserveMoveOriginal is not None
    app.arcPoint = Vec2(.5, (-10 + offset) / 100 + .5)
    app.moveArceDistance = 5
    app.moveUnit(host)
    assert host.bodyNP.getY() == pytest.approx(-10)
    assert host.reserveMoveOriginal is None and is_ethereal(host.unit.model)