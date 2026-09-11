"""Assailment opportunities belong to Initiative, not the pre-fight menu (p. 108)."""

import asyncio
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from direct.interval.IntervalGlobal import Sequence

from assailment import cast_at_initiative
from battlescribe import get_catalogue
from challenges import Challenge
from command_groups import champions, install_command
from high_magic import CorporealUnmakingSpell
from spell_system import Spell, restore_spellbook
from tests.test_corporeal_unmaking_scene import prepare_combat
from tests.test_faction_rules_scene import scene as scene
from tests.test_shieldwall_scene import combat_tasks


@pytest.mark.parametrize('joined', [False, True])
def test_real_wizard_casts_in_scheduler_not_manual_menu(scene, joined):
    app, baseline = scene
    mage, host, enemy = prepare_combat(app, baseline, joined=joined)
    assert not app.castableSpells(mage)
    app.attackers, app.defenders = [host, enemy], [enemy, host]
    app.attackSequence = Sequence()
    app.combat._combatStartModels = {id(member.unit): member.unit.nmodels for member in (host, enemy)}
    app.combat._pendingWounds = {}
    removals = Sequence()
    original = enemy.unit.nmodels
    with patch.object(app, 'aiControls', return_value=True), \
            patch.object(Spell, '_attempt', AsyncMock(return_value=True)), \
            patch.object(Spell, '_dispelled', AsyncMock(return_value=False)), \
            patch('high_magic.roll_dice_expr', return_value=1), \
            patch('battleFunctions.random.randint', side_effect=[6, 1]), \
            patch('combat_resolution.simulate_battle', return_value=(0, 0, 0, 0, 0)):
        result = asyncio.run(app.combat.resolveMeleeWithSpells(None, removals))
    assert result == [1, 0]
    assert enemy.unit.nmodels == original - 1
    assert len(enemy.model.getChildren()) == original
    assert mage.spellsCastThisTurn == ['Corporeal Unmaking']
    assert not app.magicBusy and app.assailmentWindow is None
    removals.finish()
    assert len(enemy.model.getChildren()) == original - 1


def test_hand_selects_champion_and_defers_removal(scene):
    app, baseline = scene
    mage, host, enemy = prepare_combat(app, baseline)
    restore_spellbook(mage.unit.model, [get_catalogue().spell('Hand of Khaine')], 2)
    install_command(enemy, [{'role': 'champion', 'name': 'Champion', 'selection_ref': 'test/champion'}])
    enemy.unit.command_models = {'test/champion': deepcopy(enemy.unit.model)}
    champion = champions(enemy)[0]
    app.attackers, app.defenders = [host, enemy], [enemy, host]
    app.attackSequence = Sequence()
    app.combat._combatStartModels = {id(member.unit): member.unit.nmodels for member in (host, enemy)}
    app.combat._pendingWounds = {}
    removals = Sequence()
    async def choose(options, *args, **kwargs):
        assert kwargs['owner'] is mage
        return 'Hand of Khaine' if 'Hand of Khaine' in options else champion.unitName
    with patch.object(app, 'makeChoiceNew', choose), \
            patch.object(Spell, '_attempt', AsyncMock(return_value=True)), \
            patch.object(Spell, '_dispelled', AsyncMock(return_value=False)), \
            patch('battleFunctions.random.randint', side_effect=[6, 1]), \
            patch('combat_resolution.simulate_battle', return_value=(0, 0, 0, 0, 0)):
        result = asyncio.run(app.combat.resolveMeleeWithSpells(None, removals))
    assert result == [1, 0]
    assert not champion.command_entry['active']
    removals.finish()
    assert not champions(enemy)


@pytest.mark.parametrize('enemy_initiative,casts', [(10, 0), (5, 1), (1, 1)])
def test_challenge_higher_initiative_kill_blocks_spell_equal_does_not(scene, enemy_initiative, casts):
    app, baseline = scene
    mage, host, enemy = prepare_combat(app, baseline, joined=False, target_name='Aspiring Champion')
    mage.unit.model.special_rules = []
    enemy.unit.model.special_rules = []
    mage.unit.model.characteristics['I'] = 5
    enemy.unit.model.characteristics['I'] = enemy_initiative
    challenge = Challenge(mage, host, enemy, enemy)
    async def attempt(target):
        assert target is enemy and app.magicBusy
        return True
    def attacks(unit, target, **kwargs):
        return (1, 1, 4, 0, 4) if unit.model is enemy.unit.model else (0, 0, 0, 0, 0)
    with patch.object(mage.unit.model, 'is_wizard', return_value=True), \
            patch.object(app, 'aiControls', return_value=True), \
            patch('combat_resolution.strike_initiative', side_effect=lambda profile, **kwargs:
                5 if profile is mage.unit.model else enemy_initiative), \
            patch.object(CorporealUnmakingSpell, '_attempt', AsyncMock(side_effect=attempt)) as cast, \
            patch.object(Spell, '_dispelled', AsyncMock(return_value=False)), \
            patch.object(CorporealUnmakingSpell, 'apply', AsyncMock()), \
            patch('combat_resolution.simulate_battle', side_effect=attacks):
        asyncio.run(app.combat.resolveChallengeWithSpells(challenge))
    assert cast.await_count == casts


def test_spell_only_wipeout_keeps_combat_credit_and_overrun_context(scene):
    app, baseline = scene
    mage, host, enemy = prepare_combat(app, baseline, joined=False)
    app.movement.removeModelsFromUnit(enemy, enemy.unit.nmodels - 1)
    enemy.bodyNP.setPos(host.bodyNP.getX(), host.bodyNP.getY() + (host.unitHeight + enemy.unitHeight) / 2, 0)
    with combat_tasks(app) as run, \
            patch.object(app, 'aiControls', return_value=True), \
            patch.object(app, 'makeChoiceNew', AsyncMock(return_value=enemy.unitName)), \
            patch.object(Spell, '_attempt', AsyncMock(return_value=True)), \
            patch.object(Spell, '_dispelled', AsyncMock(return_value=False)), \
            patch('high_magic.roll_dice_expr', return_value=3), \
            patch('battleFunctions.random.randint', side_effect=[6] * 3 + [1] * 3), \
            patch('combat_resolution.simulate_battle', return_value=(0, 0, 0, 0, 0)), \
            patch.object(app.combat, 'shieldwallWeaponChoice', AsyncMock()), \
            patch.object(app.combat, 'impactHits', return_value=(0, 0)), \
            patch.object(app.combat, 'challengeExchange', AsyncMock(return_value=None)), \
            patch.object(app.combat, 'overrunPass', AsyncMock()) as overrun, \
            patch.object(app.combat, 'breakTestPass', AsyncMock(return_value=[])), \
            patch.object(app.combat, 'declarePass', AsyncMock(return_value=[])), \
            patch.object(app.combat, 'loserMovePass', AsyncMock()), \
            patch.object(app.combat, 'pursuitPass', AsyncMock()), \
            patch.object(app.combat, 'printCombatResult') as result:
        run(app.combat._verySimpleBattleInner(SimpleNamespace(done='done')))
    assert enemy not in app.units
    assert result.call_args.args[0]['Wounds caused'] == (1, 0)
    assert enemy in overrun.call_args.args[1][id(host)]
    assert mage.spellsCastThisTurn == ['Corporeal Unmaking']


def test_challenge_spell_wounds_and_overkill_are_not_ordinary_excess(scene):
    app, baseline = scene
    mage, host, enemy = prepare_combat(app, baseline, joined=False, target_name='Aspiring Champion')
    enemy.woundsOnModel = int(enemy.unit.model.characteristics['W']) - 1
    challenge = Challenge(mage, host, enemy, enemy)
    with patch.object(app, 'aiControls', return_value=True), \
            patch.object(Spell, '_attempt', AsyncMock(return_value=True)), \
            patch.object(Spell, '_dispelled', AsyncMock(return_value=False)), \
            patch('combat_resolution.strike_initiative', side_effect=lambda profile, **kwargs:
                  10 if profile is mage.unit.model else 1), \
            patch('high_magic.roll_dice_expr', return_value=3), \
            patch('battleFunctions.random.randint', side_effect=[6] * 3 + [1] * 3), \
            patch('combat_resolution.simulate_battle', return_value=(0, 0, 0, 0, 0)):
        result = asyncio.run(app.combat.resolveChallengeWithSpells(challenge))
    assert result == (1, 0, 2, 0)
    assert enemy not in app.units


def test_each_players_wizard_owns_their_challenge_spell_choice(scene):
    from high_magic import HandOfKhaineSpell
    app, baseline = scene
    mage, host, enemy = prepare_combat(app, baseline, joined=False, target_name='Aspiring Champion')
    for caster in (mage, enemy):
        restore_spellbook(caster.unit.model, [get_catalogue().spell('Hand of Khaine')], 1)
    owners = []
    async def choice(options, *args, **kwargs):
        owners.append(kwargs['owner'])
        return 'Hand of Khaine'
    with patch.object(app, 'aiControls', return_value=False), \
            patch.object(app, 'makeChoiceNew', choice), \
            patch.object(Spell, '_attempt', AsyncMock(return_value=True)), \
            patch.object(Spell, '_dispelled', AsyncMock(return_value=False)), \
            patch.object(HandOfKhaineSpell, 'apply', AsyncMock()), \
            patch('combat_resolution.simulate_battle', return_value=(0, 0, 0, 0, 0)):
        asyncio.run(app.combat.resolveChallengeWithSpells(Challenge(mage, host, enemy, enemy)))
    assert set(owners) == {mage, enemy}
    assert all(caster.spellsCastThisTurn == ['Hand of Khaine'] for caster in (mage, enemy))


def test_outside_unit_does_not_attack_an_opponent_removed_in_challenge(scene):
    from tests.test_faction_rules_scene import members
    app, baseline = scene
    mage, host, enemy = prepare_combat(app, baseline, joined=False, target_name='Aspiring Champion')
    outside = members(app)['Silver Helm']
    challenge = Challenge(mage, host, enemy, enemy)
    app.attackers, app.defenders = [outside], [enemy]
    app.attackSequence = Sequence()
    app.combat._combatStartModels = {id(outside.unit): outside.unit.nmodels}
    app.movement.removeModelsFromUnit(enemy, 1)
    with patch('combat_resolution.simulate_battle') as attacks:
        assert asyncio.run(app.combat.resolveMeleeWithSpells(challenge, Sequence())) == [0, 0]
    attacks.assert_not_called()