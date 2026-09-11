"""Real roster Miscast casualties, saves and magic lifecycle (pp. 109-110, 161)."""

import asyncio
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from panda3d.core import Point3

from characters import join_unit
from command_groups import champions, install_command
from challenges import Challenge
from battlescribe import get_catalogue
from direct.interval.IntervalGlobal import Sequence
from miscasts import miscast_targets, resolve_miscast_damage
from persistence import load_game_state, save_game_state
from scouts import model_base_boxes
from spell_system import PillarOfFireSpell, Spell, miscast_result, restore_spellbook
from tests.test_corporeal_unmaking_scene import prepare_combat
from tests.test_faction_rules_scene import members, scene as scene
from tests.test_shieldwall_scene import combat_tasks


@pytest.mark.parametrize('joined', [False, True])
def test_careless_kills_actual_wizard_and_ends_remains_in_play(scene, tmp_path, joined):
    app, baseline = scene
    load_game_state(app, baseline)
    mage, host = members(app)['Mage'], members(app)['Silver Helm']
    if joined:
        assert join_unit(app, mage, host)
    count = host.unit.nmodels
    mage.woundsOnModel = int(mage.unit.model.characteristics['W']) - 1
    vortex = PillarOfFireSpell('Pillar of Fire', 9, game=app, caster=mage)
    vortex.place(app, Point3(25, 20, 0))
    with patch('battleFunctions.random.randint', return_value=6):
        resolve_miscast_damage(app, mage, miscast_result(7))
    assert mage not in app.units and mage.bodyNP.isEmpty()
    assert host.unit.nmodels == count and getattr(host, 'joinedCharacter', None) is None
    assert not app.remainsInPlay
    path = save_game_state(app, str(tmp_path / 'dead-wizard.json'))
    load_game_state(app, path)
    assert mage not in app.units and not app.remainsInPlay


def test_joined_blast_is_centred_on_wizard_not_host_and_counts_each_base_once(scene):
    app, baseline = scene
    load_game_state(app, baseline)
    mage, host = members(app)['Mage'], members(app)['Silver Helm']
    assert join_unit(app, mage, host)
    host.bodyNP.setPos(10, 15, 0)
    host.bodyNP.setH(55)
    boxes = model_base_boxes(mage)
    observed = []
    def cover(center, radius, box):
        observed.append((center, radius))
        return 'automatic'
    with patch('miscasts.template_coverage', side_effect=cover):
        targets = miscast_targets(app, mage, miscast_result(5), 'Miscast')
    assert all(center == boxes[0][:2] and radius == 1.5 for center, radius in observed)
    assert sum(hits for member, hits in targets if member is mage) == 1
    assert sum(hits for member, hits in targets) == sum(member.unit.nmodels for member in app.units)


def test_friendly_casualties_panic_from_nearest_enemy_after_all_damage(scene):
    app, baseline = scene
    load_game_state(app, baseline)
    mage, host = members(app)['Mage'], members(app)['Silver Helm']
    assert join_unit(app, mage, host)
    host.startOfPhaseModels = host.unit.nmodels
    count = host.unit.nmodels
    with patch('miscasts.miscast_targets', return_value=[(host, 2)]), \
            patch('miscasts.resolve_magic_hits', return_value=(2, 0, 2)), \
            patch.object(app.psychology, 'panic_test') as panic:
        resolve_miscast_damage(app, mage, miscast_result(2))
    assert host.unit.nmodels == count - 2
    panic.assert_called_once_with(host, flee_from=None, cause='heavy casualties (movement)')


def test_outclassed_death_of_target_stops_spell_effect(scene):
    app, baseline = scene
    load_game_state(app, baseline)
    mage, enemy = members(app)['Mage'], members(app)['Aspiring Champion']
    enemy.bodyNP.setPos(mage.bodyNP.getPos() + Point3(0, 10, 0))
    spell = Spell('Test', 8, game=app, caster=enemy)
    mage.woundsOnModel = int(mage.unit.model.characteristics['W']) - 1
    with patch.object(app, 'aiControls', return_value=True), \
            patch.object(Spell, '_roll_casting_dice', AsyncMock(side_effect=[(9, [4, 5]), (2, [1, 1]), (7, [3, 4])])), \
            patch('battleFunctions.random.randint', return_value=6), \
            patch.object(spell, 'apply', AsyncMock()) as effect:
        asyncio.run(spell.spellFunction(mage))
    assert mage not in app.units
    effect.assert_not_awaited()


def test_casting_phase_returns_after_wizard_dies_and_attempt_is_spent(scene):
    app, baseline = scene
    load_game_state(app, baseline)
    mage = members(app)['Mage']
    key, record = next((key, record) for key, record in mage.unit.model.spells.items()
                       if record['phase'] != 'combat')
    app.fsm.request(record['phase'].title() + 'Phase')
    app.strategyCommandDone = True
    phase = app.fsm.state
    app.unitToMove = mage
    app.fsm.request('SpellPhase')
    app.fsm.castingUnit = mage
    spell = Spell(key, 8, game=app, caster=mage)
    spell.selection_key = key
    app.fsm.spellInstanceToCast = spell
    mage.woundsOnModel = int(mage.unit.model.characteristics['W']) - 1
    with combat_tasks(app) as run, \
            patch.object(Spell, '_roll_casting_dice', AsyncMock(side_effect=[(2, [1, 1]), (7, [3, 4])])), \
            patch('battleFunctions.random.randint', return_value=6):
        run(app.resolveSpell(mage))
    assert mage not in app.units and mage.spellsCastThisTurn == [key]
    assert app.fsm.state == phase and not app.magicBusy


@pytest.mark.parametrize('joined', [False, True])
def test_combat_miscast_death_has_one_removal_credit_and_no_second_spell(scene, joined):
    app, baseline = scene
    mage, host, enemy = prepare_combat(app, baseline, joined=joined)
    restore_spellbook(mage.unit.model, [get_catalogue().spell(name)
                       for name in ('Corporeal Unmaking', 'Hand of Khaine')], 2)
    mage.woundsOnModel = int(mage.unit.model.characteristics['W']) - 1
    app.attackers, app.defenders = [host, enemy], [enemy, host]
    app.attackSequence = Sequence()
    app.combat._combatStartModels = {id(member.unit): member.unit.nmodels for member in (host, enemy)}
    app.combat._pendingWounds = {}
    removals = Sequence()
    with patch.object(app, 'aiControls', return_value=True), \
            patch.object(Spell, '_roll_casting_dice', AsyncMock(side_effect=[(2, [1, 1]), (7, [3, 4])])) as rolls, \
            patch('battleFunctions.random.randint', return_value=6), \
            patch('combat_resolution.simulate_battle', return_value=(0, 0, 0, 0, 0)):
        result = asyncio.run(app.combat.resolveMeleeWithSpells(None, removals))
    assert result == [0, 1]
    assert mage.unit.nmodels == 0 and not mage.bodyNP.isEmpty()
    assert rolls.await_count == 2 and len(mage.spellsCastThisTurn) == 1
    assert not app.magicBusy and app.assailmentWindow is None
    removals.finish()
    assert mage not in app.units


def test_challenge_miscast_has_no_overkill_and_skips_lower_initiative_mount(scene):
    app, baseline = scene
    mage, host, enemy = prepare_combat(app, baseline, joined=False, target_name='Aspiring Champion')
    mage.woundsOnModel = int(mage.unit.model.characteristics['W']) - 1
    mage.unit.model.mount_model = deepcopy(mage.unit.model)
    mage.unit.model.mount_model.mount_model = None
    app.combat._pendingWounds = {}
    challenge = Challenge(mage, host, enemy, enemy)
    with patch.object(app, 'aiControls', return_value=True), \
            patch.object(Spell, '_roll_casting_dice', AsyncMock(side_effect=[(2, [1, 1]), (7, [3, 4])])), \
            patch('combat_resolution.strike_initiative', side_effect=lambda profile, **kwargs:
                  10 if profile is mage.unit.model else 1), \
            patch('battleFunctions.random.randint', return_value=6), \
            patch('combat_resolution.simulate_battle', return_value=(0, 0, 0, 0, 0)) as attack:
        result = asyncio.run(app.combat.resolveChallengeWithSpells(challenge))
    assert result == (0, 1, 0, 0)
    assert mage not in app.units
    assert attack.call_count == 1


def test_blast_and_earlier_melee_losses_do_not_resurrect_or_remove_twice(scene):
    app, baseline = scene
    mage, host, enemy = prepare_combat(app, baseline, joined=False)
    app.combat._pendingWounds = {}
    removals = Sequence()
    original = enemy.unit.nmodels
    app.combat.previewCombatWounds(enemy, 1)
    from direct.interval.IntervalGlobal import Func
    removals.append(Func(app.combat.applyCombatWounds, enemy, 1))
    app.assailmentWindow = {'miscast_damage': lambda member, wounds, regenerated=0:
                           app.combat.previewMiscastWounds(member, wounds, removals)}
    try:
        with patch('miscasts.miscast_targets', return_value=[(enemy, 2)]), \
                patch('miscasts.resolve_magic_hits', return_value=(2, 0, 2)), \
                patch.object(app.psychology, 'check_heavy_casualties') as panic:
            resolve_miscast_damage(app, mage, miscast_result(2))
        panic.assert_not_called()
        assert enemy.unit.nmodels == original - 3
        assert len(enemy.model.getChildren()) == original
        removals.finish()
        assert enemy.unit.nmodels == len(enemy.model.getChildren()) == original - 3
    finally:
        app.assailmentWindow = None


def test_challenge_blast_keeps_earlier_pending_wounds_and_casualties(scene):
    from direct.interval.IntervalGlobal import Func
    app, baseline = scene
    mage, host, enemy = prepare_combat(app, baseline, joined=False, target_name='Aspiring Champion')
    app.combat._pendingWounds = {}
    app.attackers, app.defenders = [mage, enemy], [enemy, mage]
    removals = Sequence()
    app.combat.previewCombatWounds(mage, 1)
    removals.append(Func(app.combat.applyCombatWounds, mage, 1))
    with patch.object(app, 'aiControls', return_value=True), \
            patch.object(Spell, '_roll_casting_dice', AsyncMock(side_effect=[(2, [1, 1]), (7, [3, 4])])), \
            patch('battleFunctions.random.randint', return_value=6), \
            patch('combat_resolution.strike_initiative', side_effect=lambda profile, **kwargs:
                  10 if profile is mage.unit.model else 1), \
            patch('combat_resolution.simulate_battle', return_value=(0, 0, 0, 0, 0)):
        result = asyncio.run(app.combat.resolveChallengeWithSpells(Challenge(mage, host, enemy, enemy), removals))
    assert result == (0, 1, 0, 0)
    assert mage.unit.nmodels == 0 and not mage.bodyNP.isEmpty()
    removals.finish()
    assert mage not in app.units


def test_template_champion_death_does_not_spill_into_host(scene):
    app, baseline = scene
    load_game_state(app, baseline)
    mage, host = members(app)['Mage'], members(app)['Chaos Warrior']
    install_command(host, [{'role': 'champion', 'name': 'Champion', 'selection_ref': 'test/champion'}])
    host.unit.command_models = {'test/champion': deepcopy(host.unit.model)}
    champion = champions(host)[0]
    count = host.unit.nmodels
    with patch('miscasts.miscast_targets', return_value=[(champion, 1)]), \
            patch('miscasts.resolve_magic_hits', return_value=(1, 0, 1)):
        resolve_miscast_damage(app, mage, miscast_result(2))
    assert host.unit.nmodels == count - 1
    assert not champions(host)


def test_regenerated_miscast_counts_for_combat_without_a_casualty(scene):
    app, baseline = scene
    mage, host, enemy = prepare_combat(app, baseline, joined=False)
    app.attackers, app.defenders = [host, enemy], [enemy, host]
    app.attackSequence = Sequence()
    app.combat._combatStartModels = {id(member.unit): member.unit.nmodels for member in (host, enemy)}
    app.combat._pendingWounds = {}
    removals = Sequence()
    with patch.object(mage.unit.model, 'special_rules',
                      [*mage.unit.model.special_rules, {'name': 'Regeneration', 'regen': 4}]), \
            patch.object(app, 'aiControls', return_value=True), \
            patch.object(Spell, '_roll_casting_dice', AsyncMock(side_effect=[(2, [1, 1]), (7, [3, 4])])), \
            patch('battleFunctions.random.randint', return_value=6), \
            patch('combat_resolution.simulate_battle', return_value=(0, 0, 0, 0, 0)):
        assert asyncio.run(app.combat.resolveMeleeWithSpells(None, removals)) == [0, 1]
    removals.finish()
    assert mage in app.units and mage.woundsOnModel == 0


def test_pending_challenge_death_blocks_outside_ordinary_attacks(scene):
    app, baseline = scene
    mage, host, enemy = prepare_combat(app, baseline, joined=False, target_name='Aspiring Champion')
    outside = members(app)['Silver Helm']
    app.attackers, app.defenders = [outside], [enemy]
    app.attackSequence = Sequence()
    app.combat._combatStartModels = {id(outside.unit): outside.unit.nmodels}
    app.combat._pendingWounds = {}
    removals = Sequence()
    app.combat.previewMiscastWounds(enemy, 10, removals)
    assert not enemy.bodyNP.isEmpty() and enemy.unit.nmodels == 0
    with patch('combat_resolution.simulate_battle') as attacks:
        assert asyncio.run(app.combat.resolveMeleeWithSpells(Challenge(mage, host, enemy, enemy), removals)) == [0, 0]
    attacks.assert_not_called()
    removals.finish()
    assert enemy not in app.units


@pytest.mark.parametrize('duel', [False, True])
def test_full_fight_finishes_after_fatal_miscast(scene, duel):
    app, baseline = scene
    mage, host, enemy = prepare_combat(app, baseline, joined=False, target_name='Aspiring Champion')
    mage.woundsOnModel = int(mage.unit.model.characteristics['W']) - 1
    challenge = Challenge(mage, host, enemy, enemy) if duel else None
    with combat_tasks(app) as run, \
            patch.object(app, 'aiControls', return_value=True), \
            patch.object(app, 'makeChoiceNew', AsyncMock(return_value=enemy.unitName)), \
            patch.object(Spell, '_roll_casting_dice', AsyncMock(side_effect=[(2, [1, 1]), (7, [3, 4])])), \
            patch('battleFunctions.random.randint', return_value=6), \
            patch('combat_resolution.simulate_battle', return_value=(0, 0, 0, 0, 0)), \
            patch.object(app.combat, 'shieldwallWeaponChoice', AsyncMock()), \
            patch.object(app.combat, 'impactHits', return_value=(0, 0)), \
            patch.object(app.combat, 'challengeExchange', AsyncMock(return_value=challenge)), \
            patch.object(app.combat, 'overrunPass', AsyncMock()), \
            patch.object(app.combat, 'breakTestPass', AsyncMock(return_value=[])), \
            patch.object(app.combat, 'declarePass', AsyncMock(return_value=[])), \
            patch.object(app.combat, 'loserMovePass', AsyncMock()), \
            patch.object(app.combat, 'pursuitPass', AsyncMock()), \
            patch.object(app.combat, 'printCombatResult') as result:
        run(app.combat._verySimpleBattleInner(SimpleNamespace(done='done')))
    assert mage not in app.units
    assert result.call_args.args[0]['Wounds caused'] == (0, 1)
    assert not app.magicBusy and app.assailmentWindow is None