"""Magic state in the actual High Elf/Chaos scene (FoF p. 185; Rulebook p. 111)."""

from unittest.mock import AsyncMock, patch

import pytest

from magic_items import current_turn
from persistence import load_game_state, save_game_state
from spell_system import OakenShieldSpell, PillarOfFireSpell, Spell
from tests.test_faction_rules_scene import members, scene as scene
from tests.test_shieldwall_scene import combat_tasks


@pytest.mark.parametrize('name', ['Fury of Khaine', 'Shield of Saphery', 'Walk Between Worlds',
                                 'Courage of Aenarion', 'Drain Magic', 'Tempest'])
@pytest.mark.parametrize('ending', ['boundary', 'removed', 'dispelled'])
def test_selected_effect_lifecycle_matrix(scene, tmp_path, name, ending):
    from high_magic import HIGH_MAGIC, profiles_for
    from panda3d.core import Point3
    from spell_effects import active_spells, caster_removed, end_effect, end_phase, end_turn, start_turn
    app, baseline = scene
    load_game_state(app, baseline)
    mage, host = members(app)['Mage'], members(app)['Silver Helm']
    target = mage if name in ('Walk Between Worlds', 'Drain Magic') else host
    spell = HIGH_MAGIC[name](name, 9, game=app, caster=mage)
    if name == 'Tempest':
        spell.place(app, Point3(20, 0, 0))
    else:
        spell.attach(target, 1)
    lifecycle = dict(spell.lifecycle)
    path = save_game_state(app, str(tmp_path / 'active-effect.json'))
    assert path is not None
    for _ in range(2):
        load_game_state(app, path)
        matching = [effect for effect in active_spells(app) if effect.name == name]
        assert len(matching) == 1
        spell = matching[0]
        assert spell.lifecycle == lifecycle
        for member in app.units:
            for profile in profiles_for(member):
                assert sum(rule.get('name') == name for rule in profile.special_rules) <= 1
    if ending == 'dispelled':
        end_effect(spell, 'audit dispel')
        end_effect(spell, 'repeated audit cleanup')
    elif ending == 'removed':
        caster_removed(app, mage)
        assert spell.ended is (lifecycle['duration'] == 'remains' or name == 'Walk Between Worlds')
        end_turn(app)
    else:
        end_phase(app, 'Movement')
        assert not spell.ended
        end_turn(app)
        assert spell.ended is (lifecycle['duration'] == 'end_turn')
        owner = lifecycle['owner']
        app.roundCounter.current_player = 3 - owner
        start_turn(app)
        if lifecycle['duration'] == 'next_start':
            assert not spell.ended
        app.roundCounter.current_player = owner
        app.roundCounter.currentRoundPlayer[owner - 1] += 1
        start_turn(app)
        assert spell.ended is (lifecycle['duration'] != 'remains')
        if not spell.ended:
            end_effect(spell, 'audit voluntary ending')
    assert spell not in active_spells(app)
    for member in app.units:
        for profile in profiles_for(member):
            assert all(rule.get('name') != name for rule in profile.special_rules)
    if name == 'Tempest':
        assert not any(piece.terrain_type == 'pillar_of_fire' for piece in app.terrain_manager.terrain_pieces)
    ended = save_game_state(app, str(tmp_path / 'ended-effect.json'))
    load_game_state(app, ended)
    assert not any(effect.name == name for effect in active_spells(app))


@pytest.mark.parametrize('name', ['Silvery Wand', 'Helm Of Courage', 'The Banner Of The Bold'])
def test_selected_item_disable_and_reload_matrix(scene, tmp_path, name):
    from magic_items import EffectKind, activate_ability, disable_item, effects_for, inventory, save_inventory
    app, baseline = scene
    load_game_state(app, baseline)
    carrier, item = next((member, item) for member in app.units for item in inventory(member) if item.name == name)
    profile = carrier.unit.model
    base = dict(profile.characteristics)
    before_spells = len(profile.spells)
    if name == 'Helm Of Courage':
        assert activate_ability(app, carrier, item, 'courage', 'Break', confirmed=True)
        assert not activate_ability(app, carrier, item, 'courage', 'Break', confirmed=True)
        assert effects_for(carrier, EffectKind.ARMOUR)
    assert disable_item(carrier, item, 'audit item unmade')
    assert not disable_item(carrier, item, 'audit repeated suppression')
    expected = save_inventory(carrier)
    path = save_game_state(app, str(tmp_path / 'disabled-item.json'))
    for _ in range(2):
        load_game_state(app, path)
        assert save_inventory(carrier) == expected
        assert profile.characteristics == base
        assert not any(effect.item.name == name for kind in EffectKind for effect in effects_for(carrier, kind))
        if name == 'Silvery Wand':
            assert len(profile.spells) == before_spells - 1
            assert profile.wizard_level() == 2
        elif name == 'Helm Of Courage':
            assert profile.effective_armour_save() == profile.armor_save
            assert inventory(carrier)[0].uses['courage']['count'] == 1


def test_lileath_and_effect_expiry_survive_repeated_reload(scene, tmp_path):
    app, baseline = scene
    load_game_state(app, baseline)
    mage = members(app)['Mage']
    mage.lileathUsedTurn = current_turn(app)
    effect = OakenShieldSpell('Oaken Shield', 7, app.fsm.endOfTurnSpells, game=app, caster=mage)
    effect.attach(mage, 2)
    token = dict(effect.lifecycle)
    path = save_game_state(app, str(tmp_path / 'magic-active.json'))
    for _ in range(2):
        load_game_state(app, path)
        assert mage.lileathUsedTurn == current_turn(app)
        assert len(app.fsm.endOfTurnSpells) == 1
        assert app.fsm.endOfTurnSpells[0].lifecycle == token
        assert sum(rule.get('name') == 'Oaken Shield' for rule in mage.unit.model.special_rules) == 1
    app.fsm.exitCombatPhase()
    with patch('chaos_gifts.begin_turn'):
        app.fsm.enterStrategyPhase()
    assert len(app.fsm.endOfTurnSpells) == 1
    app.fsm.exitCombatPhase()
    with patch('chaos_gifts.begin_turn'):
        app.fsm.enterStrategyPhase()
    assert not app.fsm.endOfTurnSpells
    assert not any(rule.get('name') == 'Oaken Shield' for rule in mage.unit.model.special_rules)


def test_actual_mage_rerolls_without_an_extra_attempt(scene):
    from battlescribe import get_catalogue
    from spell_system import restore_spellbook
    app, baseline = scene
    load_game_state(app, baseline)
    mage = members(app)['Mage']
    restore_spellbook(mage.unit.model, [get_catalogue().spell('Fireball')], 2)
    key = next(iter(mage.unit.model.spells))
    phase = mage.unit.model.spells[key]['phase']
    app.fsm.request(phase.title() + 'Phase')
    app.strategyCommandDone = True
    if phase == 'combat':
        mage.isInCombat = True
    assert key in app.castableSpells(mage)
    app.unitToMove = mage
    app.fsm.request('SpellPhase')
    spell = Spell(key, 8, game=app, caster=mage, wizard_level=2)
    spell.selection_key = key
    app.fsm.castingUnit = mage
    app.fsm.spellInstanceToCast = spell
    with combat_tasks(app) as run, \
            patch.object(app, 'aiControls', return_value=True), \
            patch.object(app, 'dispelAttempt', AsyncMock(return_value=False)), \
            patch.object(spell, 'apply', AsyncMock(return_value=None)) as apply, \
            patch.object(spell, '_roll_casting_dice', AsyncMock(side_effect=[(4, [2, 2]), (8, [4, 4])])):
        run(app.resolveSpell(mage))
    assert mage.spellsCastThisTurn == [key]
    assert mage.lileathUsedTurn == current_turn(app)
    apply.assert_awaited_once()


def test_chaos_fated_choice_is_optional_and_spent_state_survives_reload(scene, tmp_path):
    app, baseline = scene
    load_game_state(app, baseline)
    mage = members(app)['Mage']
    spell = Spell('Test', 8, game=app, caster=mage)
    spell.casting = 8
    path = str(tmp_path / 'fated.json')
    with combat_tasks(app) as run, \
            patch.object(app, 'makeChoiceNew', AsyncMock(return_value='Pass')) as choice, \
            patch.object(Spell, '_roll_casting_dice', AsyncMock(return_value=(9, [4, 5]))) as dice:
        run(app.dispelAttempt(spell, mage))
        dice.assert_not_awaited()
        assert choice.call_args.kwargs['owner'] in app.player2Units
        assert choice.call_args.args[0] == ['Fated dispel', 'Pass']
        choice.return_value = 'Fated dispel'
        run(app.dispelAttempt(spell, mage))
        save_game_state(app, path)
        load_game_state(app, path)
        run(app.dispelAttempt(spell, mage))
        assert dice.await_count == 1
    assert app.fatedDispelTurns['2'] == current_turn(app)


def test_live_conjuration_dispels_restored_vortex_and_guards_phase(scene, tmp_path):
    from panda3d.core import Point3
    app, baseline = scene
    load_game_state(app, baseline)
    mage = members(app)['Mage']
    spell = PillarOfFireSpell('Pillar of Fire', 9, game=app, caster=mage)
    spell.place(app, Point3(25, 20, 0))
    spell.perfect = True
    spell.casting = 20
    app.roundCounter.request('PlayerTwo')
    with patch.object(app, 'restoringBattle', True):
        app.fsm.request('StrategyPhase')
    app.strategyCommandDone = True
    path = save_game_state(app, str(tmp_path / 'vortex.json'))
    assert path is not None
    load_game_state(app, path)
    assert len(app.remainsInPlay) == 1
    restored_piece = app.remainsInPlay[0].piece
    with combat_tasks(app) as run, \
            patch('game_fsm.taskMgr', app.taskMgr, create=True), \
            patch.object(app, 'makeChoiceNew', AsyncMock(return_value='Fated dispel')), \
            patch.object(Spell, '_roll_casting_dice', AsyncMock(return_value=(10, [5, 5]))):
        async def advance():
            app.fsm.nextPhase()
            assert app.magicBusy
            assert save_game_state(app, str(tmp_path / 'inflight.json')) is None
            app.fsm.nextPhase()
            assert app.fsm.state == 'StrategyPhase'
            await app.taskMgr.getTasksNamed('conjurationTask')[0]
        run(advance())
    assert not app.remainsInPlay
    assert restored_piece not in app.terrain_manager.terrain_pieces
    assert app.conjurationDoneTurn == current_turn(app)


def test_failed_recast_ends_only_old_vortex_before_dice(scene):
    from panda3d.core import Point3
    app, baseline = scene
    load_game_state(app, baseline)
    mage = members(app)['Mage']
    old = PillarOfFireSpell('Pillar of Fire', 9, game=app, caster=mage)
    old.place(app, Point3(25, 20, 0))
    old_piece = old.piece
    replacement = PillarOfFireSpell('Pillar of Fire', 9, game=app, caster=mage)
    point = mage.bodyNP.getPos(app.render) + Point3(0, 5, 0)

    async def roll():
        assert not app.remainsInPlay
        assert old_piece not in app.terrain_manager.terrain_pieces
        return 4, [2, 2]

    with combat_tasks(app) as run, \
            patch.object(app, 'makeChoiceNew', AsyncMock(return_value='Keep')), \
            patch.object(replacement, '_roll_casting_dice', side_effect=roll), \
            patch.object(replacement, 'apply', AsyncMock(return_value=None)) as effect:
        run(replacement.spellFunction(point))
    effect.assert_not_awaited()


def test_joined_caster_death_ends_vortex_without_erasing_timed_ward(scene):
    from characters import join_unit, slay_character
    from panda3d.core import Point3
    app, baseline = scene
    load_game_state(app, baseline)
    mage, host = members(app)['Mage'], members(app)['Silver Helm']
    assert join_unit(app, mage, host)
    vortex = PillarOfFireSpell('Pillar of Fire', 9, game=app, caster=mage)
    vortex.place(app, Point3(25, 20, 0))
    ward = OakenShieldSpell('Oaken Shield', 7, app.fsm.endOfTurnSpells, game=app, caster=mage)
    ward.attach(host, 2)
    slay_character(app, mage)
    assert not app.remainsInPlay
    assert ward in app.fsm.endOfTurnSpells
    assert ward.rule in host.unit.model.special_rules


def test_voluntary_end_choice_waits_before_advancing_phase(scene):
    from panda3d.core import Point3
    app, baseline = scene
    load_game_state(app, baseline)
    mage = members(app)['Mage']
    spell = PillarOfFireSpell('Pillar of Fire', 9, game=app, caster=mage)
    spell.place(app, Point3(25, 20, 0))
    app.fsm.request('ShootingPhase')
    with combat_tasks(app) as run, \
            patch('game_fsm.taskMgr', app.taskMgr, create=True), \
            patch.object(app, 'makeChoiceNew', AsyncMock(return_value='End spell')) as choice:
        async def advance():
            app.fsm.nextPhase()
            assert app.fsm.state == 'ShootingPhase' and app.magicBusy
            await app.taskMgr.getTasksNamed('endSpellChoiceTask')[0]
        run(advance())
    assert choice.call_args.kwargs['owner'] is mage
    assert not app.remainsInPlay
    assert app.fsm.state == 'CombatPhase'