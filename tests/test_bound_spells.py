"""Bound casting, roster identity and phase detours — Rulebook pp. 109, 342."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from battlescribe import get_catalogue, spell_key
from game import MyApp
from game_fsm import GamePhaseFSM
from models import model
from roster_importer import _collect_spells, import_roster
from spell_system import FireballSpell, restore_spellbook, spell_readout


def ring():
    out = []
    _collect_spells({'name': 'Ruby Ring of Ruin', 'type': 'upgrade'}, out)
    return out[0]


def carrier(spells=None, level=0):
    m = model('Zombie', '')
    restore_spellbook(m, spells if spells is not None else [ring()], level)
    return SimpleNamespace(
        unitName='Bearer', unit=SimpleNamespace(model=m, nmodels=1, name='Bearer'),
        state='Idle', isInCombat=False, hasAttackedThisTurn=False,
        marchedThisTurn=False, spellsCastThisTurn=[], boundSpellPhases=[],
        cannotCastThisTurn=False, color=(1, 1, 1, 1), model=Mock(),
        updateTextNode=Mock(), bodyNP=Mock(), roundsFought=0,
        countsAsChargedNextTurn=False, isInCombatWith=[], hostUnit=None)


def app_stub(unit, phase='shooting'):
    game = Mock()
    game.unitToMove = unit
    game.units = [unit]
    game.player1Units = [unit]
    game.player2Units = []
    game.fsm = SimpleNamespace(endOfTurnSpells=[], request=Mock(),
                               phaseBeforeSpell=phase.title() + 'Phase')
    game.castingPhase = lambda: phase
    game.castableSpells = lambda u: MyApp.castableSpells(game, u)
    game.resolveSpell = lambda target: MyApp.resolveSpell(game, target)
    game.roundCounter.current_player = 1
    return game


def test_selected_ring_imports_real_fireball_metadata():
    spell = ring()
    assert spell['name'] == 'Fireball'
    assert spell['bound'] and spell['power_level'] == 1
    assert spell['type'] == 'Magic Missile'
    assert spell['phase'] == 'shooting'
    assert spell['range'] == 24 and spell['casting_value'] == 8
    assert 'Power Level 1' in spell_readout(spell_key(spell), spell)


def test_roster_keeps_bound_and_ordinary_fireball_separate(tmp_path):
    ordinary = {'name': 'Fireball', 'typeName': 'Spell', 'characteristics': [
        {'name': 'Type', '$text': 'Magic Missile'},
        {'name': 'Casting Value', '$text': '8+'}, {'name': 'Range', '$text': '24"'}]}
    selection = {'type': 'unit', 'name': 'Captain', 'selections': [
        {'type': 'model', 'name': 'Captain', 'number': 1, 'selections': [
            {'name': 'Ruby Ring of Ruin', 'type': 'upgrade'}]}]}
    path = tmp_path / 'roster.json'

    def imported():
        path.write_text(json.dumps({'roster': {'forces': [{'catalogueName': 'Empire',
                                                          'selections': [selection]}]}}))
        return import_roster(str(path))['units'][0]

    only_bound = imported()
    assert only_bound['wizard_level'] is None
    assert len(only_bound['spells']) == 1
    selection['profiles'] = [ordinary]
    mixed = imported()
    assert mixed['wizard_level'] == 1
    assert {spell_key(s) for s in mixed['spells']} == {'Fireball', spell_key(ring())}


def test_available_but_unselected_items_grant_nothing():
    spells = []
    _collect_spells({'name': 'Captain', 'selectionEntries': [
        {'name': 'Ruby Ring of Ruin'}]}, spells)
    assert spells == []


def test_spellbook_reload_replaces_metadata_but_keeps_coded_class():
    m = carrier().unit.model
    key = spell_key(ring())
    m.spells[key].update(power_level=99, obsolete=True, **{'class': FireballSpell})
    m.spells['Stale spell'] = {'name': 'Stale spell'}
    restore_spellbook(m, [ring()])
    assert set(m.spells) == {key}
    assert m.spells[key]['power_level'] == 1
    assert 'obsolete' not in m.spells[key]
    assert m.spells[key]['class'] is FireballSpell
    assert not m.is_wizard()
    restore_spellbook(m, [get_catalogue().spell('Fireball')], 3)
    assert m.is_wizard() and m.wizard_level() == 3
    restore_spellbook(m, [ring()], 0)
    assert not m.is_wizard()
    assert set(m.spells) == {key}
    restore_spellbook(m, [])
    assert m.spells == {}


def test_nonwizard_can_select_bound_cast_without_wizard_slots():
    unit = carrier()
    game = app_stub(unit)
    unit.spellsCastThisTurn = ['Fireball']
    assert not unit.unit.model.is_wizard()
    assert game.castableSpells(unit) == [spell_key(ring())]
    MyApp.castSpell(game)
    game.fsm.request.assert_called_once_with('SpellPhase')


def test_no_more_spells_restriction_also_blocks_bound_spells():
    unit = carrier([ring(), get_catalogue().spell('Fireball')], 2)
    unit.cannotCastThisTurn = True
    assert app_stub(unit).castableSpells(unit) == []


@pytest.mark.parametrize('state,engaged,marched,expected', [
    ('Idle', False, False, True), ('IsFleeing', False, False, False),
    ('InCombat', True, False, False), ('Idle', False, True, False)])
def test_bound_fireball_obeys_normal_casting_restrictions(state, engaged, marched, expected):
    unit = carrier()
    unit.state, unit.isInCombat, unit.marchedThisTurn = state, engaged, marched
    assert bool(app_stub(unit).castableSpells(unit)) is expected


def test_one_bound_attempt_per_phase_across_all_items_not_one_per_spell():
    second = dict(ring(), source='Another item')
    unit = carrier([ring(), second])
    game = app_stub(unit)
    assert len(game.castableSpells(unit)) == 2
    unit.boundSpellPhases.append('shooting')
    assert game.castableSpells(unit) == []
    unit.unit.model.spells[spell_key(second)]['phase'] = 'strategy'
    assert len(app_stub(unit, 'strategy').castableSpells(unit)) == 1


def test_bound_attempt_failure_spends_phase_not_ordinary_spell_slot():
    unit = carrier([ring(), get_catalogue().spell('Fireball')], 2)
    game = app_stub(unit)
    spell = FireballSpell('Fireball', 8, [], bound=True, power_level=1,
                          wizard_level=2, game=game, caster=unit)
    spell.selection_key = spell_key(ring())
    game.fsm.spellInstanceToCast = spell
    game.fsm.castingUnit = unit
    with patch.object(spell, '_roll_casting_dice', AsyncMock(return_value=(2, [1, 1]))) as dice:
        asyncio.run(MyApp.resolveSpell(game, unit))
        assert spell.casting == 3 and not spell.perfect
        assert unit.boundSpellPhases == ['shooting']
        assert unit.spellsCastThisTurn == [] and not unit.cannotCastThisTurn
        assert game.castableSpells(unit) == ['Fireball']
        game.fsm.request.assert_called_once_with('ShootingPhase')
        game.fsm.request.reset_mock()
        asyncio.run(MyApp.resolveSpell(game, unit))
        dice.assert_awaited_once()
    game.fsm.request.assert_called_once_with('ShootingPhase')


def test_bound_is_selectable_through_actual_arc_task():
    unit = carrier()
    unit.bodyNP.getH.return_value = 0
    game = app_stub(unit)
    key = spell_key(ring())
    game.makeChoiceNew = AsyncMock(return_value=key)
    tasks = Mock()
    tasks.add.side_effect = lambda awaitable, *args, **kwargs: awaitable
    with patch('game.taskMgr', tasks, create=True), patch('game.render', Mock(), create=True):
        asyncio.run(MyApp.taskMagicArcUpdate(game, SimpleNamespace(done='done')))
    selected = game.fsm.spellInstanceToCast
    assert isinstance(selected, FireballSpell)
    assert selected.name == 'Fireball'
    assert selected.selection_key == key and selected.bound
    assert selected.power_level == 1 and selected.spell_range == 24
    assert not unit.unit.model.is_wizard()


def test_self_and_assailment_are_not_blocked_by_combat_ui_gate():
    unit = carrier([get_catalogue().spell('Oaken Shield'),
                    get_catalogue().spell('Hammerhand')], 2)
    unit.isInCombat = True
    assert app_stub(unit, 'strategy').castableSpells(unit) == ['Oaken Shield']
    assert app_stub(unit, 'combat').castableSpells(unit) == ['Hammerhand']
    unit.hasAttackedThisTurn = True
    assert app_stub(unit, 'combat').castableSpells(unit) == []


@pytest.mark.parametrize('phase', GamePhaseFSM.PHASES)
def test_real_fsm_casting_detour_does_not_reset_phase_or_advance_turn(phase):
    unit = carrier()
    unit.state = 'InCombat'
    unit.isInCombat = True
    game = app_stub(unit)
    game.unitCopies = []
    game.remainsInPlay = []
    game.rangeRing = None
    game.trajectoryLine = None
    game.cannon = game.bombard = None
    with patch('game_fsm.taskMgr', Mock(), create=True), \
            patch('game_fsm.messenger', Mock(), create=True):
        fsm = GamePhaseFSM(game)
        game.fsm = fsm
        if phase != 'StrategyPhase':
            fsm.request(phase)
        before_rounds = unit.roundsFought
        unit.spellsCastThisTurn = ['Fireball']
        unit.boundSpellPhases = ['shooting']
        unit.panicTestedThisPhase = True
        unit.hasAttackedThisTurn = True
        fsm.request('SpellPhase')
        fsm.nextPhase()
        assert fsm.state == 'SpellPhase'
        assert fsm.phaseBeforeSpell == phase
        fsm.request(phase)
        assert unit.boundSpellPhases == ['shooting']
        assert unit.spellsCastThisTurn == ['Fireball']
        assert unit.roundsFought == before_rounds
        assert unit.panicTestedThisPhase and unit.hasAttackedThisTurn
        game.roundCounter.next_turn.assert_not_called()
        if fsm.state != 'CombatPhase':
            fsm.request('CombatPhase')
        fsm.nextPhase()
        assert fsm.state == 'StrategyPhase'
        game.roundCounter.next_turn.assert_called_once()
        assert unit.boundSpellPhases == [] and unit.spellsCastThisTurn == []