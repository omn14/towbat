"""Real joined-caster selection, aiming and allowance across spell detours."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from panda3d.core import Point3

from characters import join_unit
from persistence import load_game_state, save_game_state
from spell_system import HammerhandSpell, PillarOfFireSpell, Spell
from tests.test_magic_resistance_scene import scene
from tests.test_shieldwall_scene import combat_tasks


def joined_setup(app, baseline, phase='ShootingPhase'):
    load_game_state(app, str(baseline))
    wizard = next(unit for unit in app.units if unit.unitName == 'MR Wizard Level 2')
    host = next(unit for unit in app.units if unit.unitName == 'MR Friendly -1')
    assert join_unit(app, wizard, host)
    app.fsm.request(phase)
    app.strategyCommandDone = True
    app.unitToMove = host
    app.refreshSelectedUnit()
    return host, wizard


@pytest.mark.parametrize('cancel', [False, True])
def test_cast_from_selected_host_and_return_without_detaching(scene, tmp_path, cancel):
    app, baseline = scene
    host, wizard = joined_setup(app, baseline)
    assert wizard not in app.player1Units
    assert wizard.bodyNP.getParent() == host.bodyNP
    host.bodyNP.setH(73)
    with combat_tasks(app) as run, \
            patch.object(app, 'makeChoiceNew', AsyncMock(return_value=None if cancel else 'Fireball')) as choice:
        app.castSpell()
        assert app.fsm.state == 'SpellPhase'
        app.taskMgr.remove('taskMagicArcUpdate')
        run(app.taskMagicArcUpdate(SimpleNamespace(done='done')))
        assert choice.call_args.kwargs['owner'] is wizard
        if not cancel:
            assert app.fsm.castingUnit is wizard and app.unitToMove is wizard
            assert app.fsm.spellInstanceToCast.caster is wizard
            target = next(unit for unit in app.units if unit.unitName == 'MR Enemy -1')
            with patch.object(Spell, '_roll_casting_dice', AsyncMock(return_value=(2, [1, 1]))):
                run(app.resolveSpell(target))
    assert app.fsm.state == 'ShootingPhase' and app.unitToMove is host
    assert host.joinedCharacter is wizard and wizard.hostUnit is host
    assert wizard.spellsCastThisTurn == ([] if cancel else ['Fireball'])
    assert not host.spellsCastThisTurn and not wizard.boundSpellPhases
    if not cancel:
        path = save_game_state(app, str(tmp_path / 'joined-cast.json'))
        load_game_state(app, path)
        assert wizard.hostUnit is host and wizard.spellsCastThisTurn == ['Fireball']
        assert 'Fireball' not in app.castableSpells(wizard)


def test_joined_vortex_measures_world_position_and_assailment_uses_host_combat(scene):
    app, baseline = scene
    host, wizard = joined_setup(app, baseline)
    host.bodyNP.setPos(30, 20, 0)
    host.bodyNP.setH(65)
    origin = wizard.bodyNP.getPos(app.render)
    assert (origin - wizard.bodyNP.getPos()).length() > 20
    vortex = PillarOfFireSpell('Pillar of Fire', 9, [], game=app, caster=wizard)
    assert vortex.canTarget(origin + Point3(5, 0, 0))
    assert not vortex.canTarget(origin + Point3(13, 0, 0))
    app.unitToMove = wizard
    with patch.object(app, 'drawRangeRing') as ring:
        app.beginGroundTargeting(vortex)
        assert ring.call_args.args[0].almostEqual(origin)
    target = next(unit for unit in app.units if unit.unitName == 'MR Enemy -1')
    host.isInCombat = True
    host.isInCombatWith = [target]
    app.fsm.request('CombatPhase')
    hammer = HammerhandSpell('Hammerhand', 7, [], game=app, caster=wizard)
    assert not wizard.isInCombatWith
    assert 'Hammerhand' not in app.castableSpells(wizard)
    with patch.object(app, 'assailmentWindow', {'caster': wizard, 'targets': [target]}, create=True):
        assert 'Hammerhand' in app.castableSpells(wizard)
        assert hammer.canTarget(target)
        assert not hammer.canTarget(host)


def test_strategy_host_click_uses_joined_self_caster_and_restores_host(scene):
    app, baseline = scene
    host, wizard = joined_setup(app, baseline, 'StrategyPhase')
    with combat_tasks(app) as run, \
            patch.object(app, 'makeChoiceNew', AsyncMock(return_value='Oaken Shield')), \
            patch.object(Spell, '_roll_casting_dice', AsyncMock(return_value=(2, [1, 1]))):
        app.taskLoopStrategy(SimpleNamespace(done='done'))
        app.taskMgr.remove('taskMagicArcUpdate')
        assert app.fsm.state == 'SpellPhase'
        run(app.taskMagicArcUpdate(SimpleNamespace(done='done')))
    assert wizard.spellsCastThisTurn == ['Oaken Shield']
    assert app.unitToMove is host and app.fsm.state == 'StrategyPhase'
    assert not host.spellsCastThisTurn


def test_successful_joined_cast_applies_effect_with_wizard_ownership(scene):
    app, baseline = scene
    host, wizard = joined_setup(app, baseline)
    target = next(unit for unit in app.units if unit.unitName == 'MR Enemy -1')
    with combat_tasks(app) as run, \
            patch.object(app, 'makeChoiceNew', AsyncMock(return_value='Fireball')):
        app.castSpell()
        app.taskMgr.remove('taskMagicArcUpdate')
        run(app.taskMagicArcUpdate(SimpleNamespace(done='done')))
        spell = app.fsm.spellInstanceToCast
        with patch.object(Spell, '_roll_casting_dice', AsyncMock(return_value=(10, [5, 5]))), \
                patch.object(app, 'dispelAttempt', AsyncMock(return_value=False)) as dispel, \
                patch.object(spell, 'apply', AsyncMock(wraps=spell.apply)) as effect:
            run(app.resolveSpell(target))
            effect.assert_awaited_once_with(target)
            dispel.assert_awaited_once_with(spell, wizard)
    assert spell.caster is wizard and spell.wizard_level == 2
    assert wizard.spellsCastThisTurn == ['Fireball'] and not host.spellsCastThisTurn
    assert app.unitToMove is host and app.fsm.state == 'ShootingPhase'