"""Actual Mage/Chaos Assailment casting and wound credits (Rulebook pp. 151, 329)."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from panda3d.core import BitMask32

from battleFunctions import ward_save_value
from battlescribe import get_catalogue
from challenges import Challenge
from characters import join_unit
from high_magic import CorporealUnmakingSpell
from persistence import load_game_state, save_game_state
from spell_system import Spell, restore_spellbook
from tests.test_faction_rules_scene import members, scene as scene
from tests.test_shieldwall_scene import combat_tasks
from assailment import cast_at_initiative

NAME = 'Corporeal Unmaking'


def prepare_combat(app, baseline, *, joined=True, target_name='Chaos Warrior'):
    load_game_state(app, baseline)
    armies = members(app)
    mage, enemy = armies['Mage'], armies[target_name]
    host = armies['Silver Helm'] if joined else mage
    if joined:
        assert join_unit(app, mage, host)
    host.bodyNP.setPos(0, -14, 0)
    host.bodyNP.setH(0)
    enemy.bodyNP.setPos(0, -14 + (host.unitHeight + enemy.unitHeight) / 2, 0)
    enemy.bodyNP.setH(180)
    for unit, opponent in ((host, enemy), (enemy, host)):
        unit.request('InCombat')
        unit.isInCombat = True
        unit.isInCombatWith = [opponent]
        unit.isInCombatFlank = ['front']
        unit.hasAttackedThisTurn = False
    restore_spellbook(mage.unit.model, [get_catalogue().spell(NAME)], 2)
    app.fsm.request('CombatPhase')
    app.unitToMove = host
    return mage, host, enemy


@pytest.mark.parametrize('joined', [False, True])
def test_live_cast_uses_chaos_ward_and_preserves_casualty_credit_on_reload(scene, tmp_path, joined):
    app, baseline = scene
    mage, host, enemy = prepare_combat(app, baseline, joined=joined)
    original_count = enemy.unit.nmodels
    assert ward_save_value(enemy.unit.model) == 6
    credited = []
    def damage(target, wounds, regenerated=0):
        credited.append(wounds)
        app.movement.applyWounds(target, wounds)
    with combat_tasks(app) as run, \
            patch.object(app, 'makeChoiceNew', AsyncMock(return_value=NAME)), \
            patch.object(Spell, '_roll_casting_dice', AsyncMock(return_value=(8, [4, 4]))), \
            patch.object(app, 'dispelAttempt', AsyncMock(return_value=False)) as dispel, \
            patch('high_magic.roll_dice_expr', return_value=3), \
            patch('battleFunctions.random.randint', side_effect=[6, 6, 1, 6, 1]), \
            patch('battleFunctions.check_armor_save') as armour, \
            patch.object(app.psychology, 'check_heavy_casualties') as panic:
        assert NAME not in app.castableSpells(mage)
        run(cast_at_initiative(app, mage, [enemy], damage))
        spell = dispel.call_args.args[0]
        assert isinstance(spell, CorporealUnmakingSpell)
        assert spell.casting_value == 8 and spell.wizard_level == 2
        dispel.assert_awaited_once_with(spell, mage)
        armour.assert_not_called()
        panic.assert_not_called()
    assert enemy.unit.nmodels == original_count - 1
    assert len(enemy.model.getChildren()) == original_count - 1
    assert credited == [1]
    assert mage.spellsCastThisTurn == [NAME]
    assert app.unitToMove is host and app.fsm.state == 'CombatPhase'
    assert not app.fsm.endOfTurnSpells
    path = save_game_state(app, str(tmp_path / 'unmaking.json'))
    for _ in range(2):
        load_game_state(app, path)
        assert enemy.unit.nmodels == original_count - 1
        assert ward_save_value(enemy.unit.model) == 6
        assert mage.spellsCastThisTurn == [NAME]
        assert NAME not in app.castableSpells(mage)
        assert not app.fsm.endOfTurnSpells


@pytest.mark.parametrize('outcome', ['cancelled', 'failed', 'dispelled'])
def test_unsuccessful_cast_has_no_damage_or_combat_credit(scene, outcome):
    app, baseline = scene
    mage, host, enemy = prepare_combat(app, baseline)
    original_count = enemy.unit.nmodels
    roll = (3, [1, 2]) if outcome == 'failed' else (8, [4, 4])
    with combat_tasks(app) as run, \
            patch.object(app, 'makeChoiceNew', AsyncMock(return_value=None if outcome == 'cancelled' else NAME)), \
            patch.object(Spell, '_roll_casting_dice', AsyncMock(return_value=roll)), \
            patch.object(app, 'dispelAttempt', AsyncMock(return_value=outcome == 'dispelled')), \
            patch('high_magic.roll_dice_expr') as hits:
        run(cast_at_initiative(app, mage, [enemy], lambda *args: None))
        hits.assert_not_called()
    assert enemy.unit.nmodels == original_count
    assert getattr(host, 'assailmentWounds', 0) == 0
    assert mage.spellsCastThisTurn == ([] if outcome == 'cancelled' else [NAME])
    assert app.unitToMove is host and app.fsm.state == 'CombatPhase'


def test_target_guards_cover_joined_retired_fought_and_challenge_casters(scene, capsys):
    app, baseline = scene
    mage, host, enemy = prepare_combat(app, baseline)
    spell = CorporealUnmakingSpell(NAME, 8, game=app, caster=mage)
    assert not spell.canTarget(enemy)
    assert not mage.isInCombatWith
    assert not spell.canTarget(host)
    assert not spell.canTarget(members(app)['Chaos Knight'])
    mage.retiredFromCombat = True
    assert not spell.canTarget(enemy)
    assert NAME not in app.castableSpells(mage)
    mage.retiredFromCombat = False
    host.hasAttackedThisTurn = True
    assert not spell.canTarget(enemy)
    host.hasAttackedThisTurn = False
    with patch.object(app, 'resolvingCombat', True):
        assert not spell.canTarget(enemy)
    rival = members(app)['Aspiring Champion']
    with patch.object(app, 'challenges', [Challenge(mage, host, rival, enemy)]):
        assert not spell.canTarget(enemy)
        assert not spell.mark_targets(BitMask32.bit(5))
    with patch.object(app, 'challenges', [Challenge(members(app)['Dragon Prince'],
                                                   members(app)['Dragon Prince'], rival, enemy)]):
        assert not spell.canTarget(enemy)
    assert 'only available when the Wizard fights at Initiative' in capsys.readouterr().out


def test_multiwound_target_loses_one_wound_not_a_model(scene, tmp_path):
    app, baseline = scene
    mage, host, enemy = prepare_combat(app, baseline, target_name='Aspiring Champion')
    assert int(enemy.unit.model.characteristics['W']) > 1
    spell = CorporealUnmakingSpell(NAME, 8, game=app, caster=mage)
    with patch('high_magic.roll_dice_expr', return_value=1), \
            patch('battleFunctions.random.randint', side_effect=[6, 1]):
        asyncio.run(spell.apply(enemy))
    assert enemy.unit.nmodels == 1 and enemy.woundsOnModel == 1
    assert host.assailmentWounds == 1
    path = save_game_state(app, str(tmp_path / 'wounded-character.json'))
    load_game_state(app, path)
    assert enemy.unit.nmodels == 1 and enemy.woundsOnModel == 1


def test_live_combat_scores_banked_wounds_once_and_turn_end_clears_them(scene):
    app, baseline = scene
    mage, host, enemy = prepare_combat(app, baseline)
    spell = CorporealUnmakingSpell(NAME, 8, game=app, caster=mage)
    with patch('high_magic.roll_dice_expr', return_value=1), \
            patch('battleFunctions.random.randint', side_effect=[6, 1]):
        asyncio.run(spell.apply(enemy))
    with combat_tasks(app) as run, \
            patch.object(app, 'makeChoiceNew', AsyncMock(return_value=enemy.unitName)), \
            patch.object(app.combat, 'shieldwallWeaponChoice', AsyncMock()), \
            patch.object(app.combat, 'impactHits', return_value=(0, 0)), \
            patch.object(app.combat, 'challengeExchange', AsyncMock(return_value=None)), \
            patch.object(app.combat, 'resolveMeleeWithSpells', AsyncMock(return_value=(0, 0))), \
            patch.object(app.combat, 'overrunPass', AsyncMock()), \
            patch.object(app.combat, 'breakTestPass', AsyncMock(return_value=[])), \
            patch.object(app.combat, 'declarePass', AsyncMock(return_value=[])), \
            patch.object(app.combat, 'loserMovePass', AsyncMock()), \
            patch.object(app.combat, 'pursuitPass', AsyncMock()), \
            patch.object(app.combat, 'printCombatResult') as result:
        run(app.combat._verySimpleBattleInner(SimpleNamespace(done='done')))
    assert result.call_args.args[0]['Wounds caused'] == (1, 0)
    assert host.assailmentWounds == 0
    host.assailmentWounds = 2
    app.fsm.exitCombatPhase()
    assert host.assailmentWounds == 0