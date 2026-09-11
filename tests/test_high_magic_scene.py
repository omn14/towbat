"""Selected High Magic in the actual High Elf/Chaos scene (Rulebook p. 329)."""

import asyncio
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from panda3d.core import BitMask32

from battleFunctions import attack_characteristic, melee_attacks, simulate_battle, ward_save_value
from battlescribe import get_catalogue
from characters import detach_character, join_unit
from combat_profiles import combat_profiles
from high_magic import FuryOfKhaineSpell, ShieldOfSapherySpell, profiles_for
from persistence import load_game_state, save_game_state
from spell_effects import end_turn
from spell_system import Spell, restore_spellbook
from tests.test_faction_rules_scene import members, scene as scene
from tests.test_shieldwall_scene import combat_tasks


@pytest.mark.parametrize('name', ['Silver Helm', 'Dragon Prince', 'Lothern Skycutter'])
def test_fury_reaches_real_split_profiles_and_survives_profile_resets(scene, name):
    app, baseline = scene
    load_game_state(app, baseline)
    mage, host = members(app)['Mage'], members(app)[name]
    parts = combat_profiles(host, members(app)['Chaos Knight'])
    before = [part.attacks(host.unit.nmodels, host.unit.nmodels) for part in parts]
    snapshots = [(profile, deepcopy(profile.characteristics)) for profile in profiles_for(host)]
    spell = FuryOfKhaineSpell('Fury of Khaine', 9, game=app, caster=mage)
    asyncio.run(spell.apply(host))
    after = [part.attacks(host.unit.nmodels, host.unit.nmodels) for part in parts]
    assert all(boosted > ordinary for boosted, ordinary in zip(after, before))
    for profile, snapshot in snapshots:
        assert profile.characteristics == snapshot
        profile.characteristics = deepcopy(snapshot)
    assert [part.attacks(host.unit.nmodels, host.unit.nmodels) for part in parts] == after
    end_turn(app)
    assert [part.attacks(host.unit.nmodels, host.unit.nmodels) for part in parts] == before


def test_live_spell_selection_target_masks_casting_and_replacement(scene):
    app, baseline = scene
    load_game_state(app, baseline)
    mage, host = members(app)['Mage'], members(app)['Silver Helm']
    mage.bodyNP.setPos(0, -15, 0)
    mage.bodyNP.setH(0)
    host.bodyNP.setPos(0, -5, 0)
    restore_spellbook(mage.unit.model, [get_catalogue().spell(name) for name in
                                      ('Fury of Khaine', 'Shield of Saphery')], 2)
    app.fsm.request('StrategyPhase')
    app.strategyCommandDone = True
    app.unitToMove = mage
    with combat_tasks(app) as run, \
            patch.object(app, 'mouseWatcherNode', SimpleNamespace(hasMouse=lambda: False)), \
            patch.object(app, 'makeChoiceNew', AsyncMock()) as choice, \
            patch.object(Spell, '_roll_casting_dice', AsyncMock(return_value=(10, [5, 5]))), \
            patch.object(app, 'dispelAttempt', AsyncMock(return_value=False)):
        for name, cls in [('Fury of Khaine', FuryOfKhaineSpell), ('Shield of Saphery', ShieldOfSapherySpell)]:
            choice.return_value = name
            app.castSpell()
            app.taskMgr.remove('taskMagicArcUpdate')
            run(app.taskMagicArcUpdate(SimpleNamespace(done='done')))
            assert isinstance(app.fsm.spellInstanceToCast, cls)
            assert host.bodyNP.getCollideMask() == BitMask32.bit(5)
            run(app.resolveSpell(host))
    assert mage.spellsCastThisTurn == ['Fury of Khaine', 'Shield of Saphery']
    assert [spell.name for spell in app.fsm.endOfTurnSpells] == ['Shield of Saphery']
    assert ward_save_value(host.unit.model) == 5


def test_join_propagation_reload_and_departed_recipient_replacement(scene, tmp_path):
    app, baseline = scene
    load_game_state(app, baseline)
    mage, host = members(app)['Mage'], members(app)['Silver Helm']
    original = attack_characteristic(mage.unit.model)
    fury = FuryOfKhaineSpell('Fury of Khaine', 9, game=app, caster=mage)
    asyncio.run(fury.apply(mage))
    assert join_unit(app, mage, host)
    assert len(fury.affected_members) == 2
    assert any(rule.get('extra_attacks') for rule in host.unit.model.special_rules)
    mage.bodyNP.wrtReparentTo(app.render)
    detach_character(host)
    app.player1Units.append(mage)
    app.world.attachRigidBody(mage.bodyNP.node())
    assert attack_characteristic(mage.unit.model) == original + 1
    shield = ShieldOfSapherySpell('Shield of Saphery', 8, game=app, caster=mage)
    asyncio.run(shield.apply(host))
    assert attack_characteristic(mage.unit.model) == original + 1
    assert not any(rule.get('extra_attacks') for rule in host.unit.model.special_rules)
    path = save_game_state(app, str(tmp_path / 'spells.json'))
    for _ in range(2):
        load_game_state(app, path)
        assert attack_characteristic(mage.unit.model) == original + 1
        assert ward_save_value(host.unit.model) == 5
        assert len(app.fsm.endOfTurnSpells) == 2
    app.fsm.exitCombatPhase()
    assert attack_characteristic(mage.unit.model) == original
    assert ward_save_value(host.unit.model) == 0
    assert not app.fsm.endOfTurnSpells


def test_joining_buffed_unit_receives_both_grants_and_existing_ward_survives(scene, tmp_path):
    app, baseline = scene
    load_game_state(app, baseline)
    mage, host = members(app)['Mage'], members(app)['Dragon Prince']
    original = attack_characteristic(mage.unit.model)
    shield = ShieldOfSapherySpell('Shield of Saphery', 8, game=app, caster=mage)
    asyncio.run(shield.apply(host))
    fury = FuryOfKhaineSpell('Fury of Khaine', 9, game=app, caster=mage)
    asyncio.run(fury.apply(host))
    assert join_unit(app, mage, host)
    assert ward_save_value(mage.unit.model) == 5
    assert any(rule.get('extra_attacks') for rule in mage.unit.model.special_rules)
    path = save_game_state(app, str(tmp_path / 'joined-effects.json'))
    for _ in range(2):
        load_game_state(app, path)
        assert mage.hostUnit is host
        assert ward_save_value(mage.unit.model) == 5
        assert attack_characteristic(mage.unit.model) == original + 1
        assert len(app.fsm.endOfTurnSpells) == 2
    end_turn(app)
    assert ward_save_value(host.unit.model) == 6
    assert ward_save_value(mage.unit.model) == 0
    assert attack_characteristic(mage.unit.model) == original


def test_joined_caster_marks_own_and_screened_units_in_world_arc(scene):
    app, baseline = scene
    load_game_state(app, baseline)
    mage, host = members(app)['Mage'], members(app)['Silver Helm']
    target, screen = members(app)['Dragon Prince'], members(app)['Lothern Skycutter']
    host.bodyNP.setPos(0, -15, 0)
    host.bodyNP.setH(90)
    target.bodyNP.setPos(-10, -15, 0)
    screen.bodyNP.setPos(-5, -15, 0)
    assert join_unit(app, mage, host)
    fury = FuryOfKhaineSpell('Fury of Khaine', 9, game=app, caster=mage)
    assert fury.canTarget(host) and fury.canTarget(target)
    app.fsm.request('SpellPhase')
    app.fsm.spellInstanceToCast = fury
    assert app.checkArrows(BitMask32.bit(5))
    assert host.bodyNP.getCollideMask() == BitMask32.bit(5)
    assert target.bodyNP.getCollideMask() == BitMask32.bit(5)
    target.bodyNP.setPos(10, -15, 0)
    assert not fury.canTarget(target)


def test_survivor_keeps_spell_after_original_recipient_death_and_reload(scene, tmp_path):
    from characters import slay_character
    app, baseline = scene
    load_game_state(app, baseline)
    mage, host = members(app)['Mage'], members(app)['Silver Helm']
    fury = FuryOfKhaineSpell('Fury of Khaine', 9, game=app, caster=mage)
    asyncio.run(fury.apply(mage))
    assert join_unit(app, mage, host)
    slay_character(app, mage)
    expected = melee_attacks(host.unit, charge=True)
    path = save_game_state(app, str(tmp_path / 'surviving-effect.json'))
    load_game_state(app, path)
    assert len(app.fsm.endOfTurnSpells) == 1
    assert app.fsm.endOfTurnSpells[0].caster is None
    assert melee_attacks(host.unit, charge=True) == expected
    end_turn(app)
    assert melee_attacks(host.unit, charge=True) < expected


def test_live_combat_attack_count_and_ward_outcomes(scene, capsys):
    app, baseline = scene
    load_game_state(app, baseline)
    mage, host, enemy = members(app)['Mage'], members(app)['Silver Helm'], members(app)['Chaos Knight']
    original = melee_attacks(host.unit, charge=True)
    shield = ShieldOfSapherySpell('Shield of Saphery', 8, game=app, caster=mage)
    asyncio.run(shield.apply(host))
    fury = FuryOfKhaineSpell('Fury of Khaine', 9, game=app, caster=mage)
    asyncio.run(fury.apply(host))
    with patch('battleFunctions.random.randint', return_value=5), \
            patch('battleFunctions.check_armor_save', return_value=False):
        attacks = simulate_battle(host.unit, enemy.unit, charge=True)
        protected = simulate_battle(enemy.unit, host.unit, charge=True)
        assert attacks[0] == original + min(host.unit.files, host.unit.nmodels)
        assert protected[2] > 0 and protected[4] == 0
        end_turn(app)
        exposed = simulate_battle(enemy.unit, host.unit, charge=True)
        assert exposed[4] > 0
    assert melee_attacks(host.unit, charge=True) == original
    log = capsys.readouterr().out
    assert 'Shield of Saphery' in log and 'Fury of Khaine' in log