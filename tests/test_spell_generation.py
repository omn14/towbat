"""Rulebook p. 319 generation and Forces of Fantasy p. 186 Saphery substitutions."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from magic_items import install_inventory
from spell_generation import generate_spells, generation_reference, spell_tables, start_generation
from spell_system import restore_spellbook
from tests.test_magic_items import live_member


def mage_with_pool():
    mage = live_member('Mage')
    restore_spellbook(mage.unit.model, [], 2)
    mage.unit.model.characteristics['Special Rules'] = ['Lore of Saphery']
    pool = [{'name': f'Spell {number}', 'number': number, 'type': 'Enchantment',
             'casting_value': 8, 'range': 12, 'phase': 'strategy'} for number in range(1, 7)]
    pool += [{'name': 'Drain Magic', 'number': None}, {'name': "Vaul's Unmaking", 'number': 3},
             {'name': 'Courage of Aenarion', 'number': 1}, {'name': 'Hand of Khaine', 'number': 2}]
    mage.unit.roster_metadata.update(spell_pool=pool, spell_generation_pending=True)
    install_inventory(mage, [{'name': 'Silvery Wand', 'category': 'Arcane Items',
                             'owner_ref': 'bearer', 'selection_ref': 'bearer/wand'}])
    return mage


def test_saphery_numbers_do_not_pollute_main_d6_table():
    mage = mage_with_pool()
    table, signatures = spell_tables(mage)
    assert len(table) == 6 and len(signatures) == 4
    assert table[3]['name'] == 'Spell 3'
    mage.unit.model.characteristics['Special Rules'] = []
    assert [entry['name'] for entry in spell_tables(mage)[1]] == ['Drain Magic']


@pytest.mark.parametrize('signature', ['Drain Magic', "Vaul's Unmaking", 'Hand of Khaine', 'Courage of Aenarion'])
def test_duplicate_rerolls_and_one_signature_swap(signature):
    mage = mage_with_pool()
    game = SimpleNamespace(aiControls=lambda member: False,
                           makeChoiceNew=AsyncMock(side_effect=[signature, 'Spell 2']))
    with patch('spell_generation.random.randint', side_effect=[1, 1, 2, 6]) as dice:
        assert asyncio.run(generate_spells(game, mage))
    assert dice.call_count == 4
    assert set(mage.unit.model.spells) == {'Spell 1', signature, 'Spell 6'}
    assert mage.unit.model.wizard_level() == 2
    assert mage.unit.roster_metadata['spell_generation']['complete']
    assert not mage.unit.roster_metadata['spell_generation_pending']
    game.makeChoiceNew.assert_awaited()
    first, replacement = game.makeChoiceNew.call_args_list
    assert len(first.kwargs['reference']) == len(replacement.kwargs['reference']) == 10
    assert any(entry['name'] == signature and entry['status'] == 'Selected signature'
               for entry in replacement.kwargs['reference'])
    with patch('spell_generation.random.randint', side_effect=AssertionError('must not reroll')):
        assert asyncio.run(generate_spells(game, mage))


def test_banked_rolls_resume_after_json_roundtrip_without_refunding_choices():
    mage = mage_with_pool()
    with patch('spell_generation.random.randint', side_effect=[2, 4, 6]):
        state = start_generation(mage)
    assert state['rolls'] == [2, 4, 6]
    mage.unit.roster_metadata = json.loads(json.dumps(mage.unit.roster_metadata))
    game = SimpleNamespace(aiControls=lambda member: True)
    with patch('spell_generation.random.randint', side_effect=AssertionError('must not reroll')):
        assert asyncio.run(generate_spells(game, mage))
    assert set(mage.unit.model.spells) == {'Spell 2', 'Spell 4', 'Spell 6'}


def test_incomplete_or_ambiguous_pool_stays_pending_without_rolling():
    mage = mage_with_pool()
    mage.unit.roster_metadata['spell_pool'].pop(0)
    game = SimpleNamespace(aiControls=lambda member: True)
    with patch('spell_generation.random.randint') as dice:
        assert not asyncio.run(generate_spells(game, mage))
    dice.assert_not_called()
    assert mage.unit.roster_metadata['spell_generation_pending']
    assert not mage.unit.model.spells


def test_explicit_known_and_bound_spells_are_preserved():
    mage = mage_with_pool()
    restore_spellbook(mage.unit.model, [mage.unit.roster_metadata['spell_pool'][0],
                                       {'name': 'Fireball', 'bound': True, 'source': 'Ruby Ring of Ruin'}], 2)
    game = SimpleNamespace(aiControls=lambda member: True)
    with patch('spell_generation.random.randint', side_effect=[1, 2, 3]):
        assert asyncio.run(generate_spells(game, mage))
    assert set(mage.unit.model.spells) == {'Spell 1', 'Spell 2', 'Spell 3', 'Fireball [Bound: Ruby Ring of Ruin]'}


def test_runtime_spell_classes_do_not_enter_saved_generation_state():
    mage = mage_with_pool()
    selected = dict(mage.unit.roster_metadata['spell_pool'][0], **{'class': SimpleNamespace})
    restore_spellbook(mage.unit.model, [selected], 2)
    with patch('spell_generation.random.randint', side_effect=[2, 3]):
        start_generation(mage)
    saved = json.loads(json.dumps(mage.unit.roster_metadata))
    assert 'class' not in saved['spell_generation']['known'][0]


def test_resume_signature_replacement_never_offers_second_swap_or_rerolls():
    mage = mage_with_pool()
    game = SimpleNamespace(aiControls=lambda member: False,
                           makeChoiceNew=AsyncMock(side_effect=['Drain Magic', None]))
    with patch('spell_generation.random.randint', side_effect=[2, 4, 6]):
        assert not asyncio.run(generate_spells(game, mage))
    mage.unit.roster_metadata = json.loads(json.dumps(mage.unit.roster_metadata))
    assert mage.unit.roster_metadata['spell_generation']['stage'] == 'replace'
    game.makeChoiceNew = AsyncMock(return_value='Spell 4')
    with patch('spell_generation.random.randint', side_effect=AssertionError('no new dice')):
        assert asyncio.run(generate_spells(game, mage))
    game.makeChoiceNew.assert_awaited_once()
    assert game.makeChoiceNew.call_args.args[0] == ['Spell 2', 'Spell 4', 'Spell 6']
    assert any(entry['name'] == 'Drain Magic' and entry['status'] == 'Selected signature'
               for entry in game.makeChoiceNew.call_args.kwargs['reference'])
    assert set(mage.unit.model.spells) == {'Spell 2', 'Drain Magic', 'Spell 6'}


def test_each_player_can_choose_wizard_order_and_share_spells():
    from spell_generation import prepare_spellbooks

    first, second = mage_with_pool(), mage_with_pool()
    first.unitName, second.unitName = 'First', 'Second'
    game = SimpleNamespace(units=[first, second], player1Units=[first, second], player2Units=[],
                           aiControls=lambda member: False,
                           makeChoiceNew=AsyncMock(side_effect=['Second', 'Keep spells', 'Keep spells']))
    with patch('spell_generation.random.randint', side_effect=[1, 2, 3, 1, 2, 3]):
        assert asyncio.run(prepare_spellbooks(game))
    assert game.makeChoiceNew.call_args_list[1].kwargs['owner'] is second
    assert game.makeChoiceNew.call_args_list[2].kwargs['owner'] is first
    assert set(first.unit.model.spells) == set(second.unit.model.spells)
    assert not game.spellGenerationBusy


def test_generation_reference_covers_unrolled_spells_and_full_profiles_without_mutation():
    mage = mage_with_pool()
    mage.unit.roster_metadata['spell_pool'][0]['effect'] = 'First paragraph.\n\nSecond paragraph.'
    with patch('spell_generation.random.randint', side_effect=[1, 2, 6]):
        state = start_generation(mage)
    before = json.dumps(mage.unit.roster_metadata)
    with patch('spell_generation.random.randint', side_effect=AssertionError('inspection must not roll')):
        reference = generation_reference(mage, state)
    assert len(reference) == 10
    assert [entry['name'] for entry in reference if entry['status'] == 'Generated'] == [
        'Spell 1', 'Spell 2', 'Spell 6']
    assert [entry['name'] for entry in reference if entry['status'] == 'Not generated'] == [
        'Spell 3', 'Spell 4', 'Spell 5']
    assert len([entry for entry in reference if entry['status'] == 'Signature option']) == 4
    detail = reference[0]['detail']
    for text in ('Type: Enchantment', 'Casting value: 8+', 'Range: 12"', 'Phase: Strategy',
                 'First paragraph.\n\nSecond paragraph.', 'Engine effect: not implemented'):
        assert text in detail
    assert json.dumps(mage.unit.roster_metadata) == before
    assert not mage.unit.model.spells


def test_generation_reference_distinguishes_known_and_pending_signature_on_resume():
    mage = mage_with_pool()
    restore_spellbook(mage.unit.model, [mage.unit.roster_metadata['spell_pool'][0]], 2)
    with patch('spell_generation.random.randint', side_effect=[2, 6]):
        state = start_generation(mage)
    state.update(stage='replace', signature=state['signatures'][0])
    mage.unit.roster_metadata = json.loads(json.dumps(mage.unit.roster_metadata))
    reference = generation_reference(mage, mage.unit.roster_metadata['spell_generation'])
    assert reference[0]['status'] == 'Already known'
    selected = next(entry for entry in reference if entry['name'] == state['signature']['name'])
    assert selected['status'] == 'Selected signature'
    assert 'Effect text not recorded' in selected['detail']