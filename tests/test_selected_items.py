"""Selected roster item effects (Battle March p. 47; Forces of Fantasy p. 183)."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest

from command_groups import install_command
from magic_items import disable_item, install_inventory
from psychology import reroll_leadership, veteran_available, veteran_counts
from tests.test_magic_items import live_member


def banner_unit():
    host = live_member('Warriors', 10)
    install_command(host, [{'role': 'standard_bearer', 'selection_ref': 'standard'}])
    banner = install_inventory(host, [{'name': 'The Banner Of The Bold', 'category': 'Magic Standards',
                                      'selection_ref': 'standard/banner', 'owner_ref': 'standard'}])[0]
    return host, banner


def test_banner_grants_actual_veteran_reroll_and_logs_source(capsys):
    host, _ = banner_unit()
    assert not host.unit.model.is_veteran()
    assert veteran_counts(host) == (10, 10)
    roller = AsyncMock(return_value=[1, 2])
    game = SimpleNamespace(aiControls=lambda member: True)
    assert asyncio.run(reroll_leadership(game, host, 'Rally', [6, 6], 8, roller)) == [1, 2]
    roller.assert_awaited_once()
    output = capsys.readouterr().out
    assert 'The Banner Of The Bold' in output and '10/10' in output and '2D6=12 vs Ld 8' in output


def test_banner_applies_to_joined_characters_personal_tests_only_while_joined():
    host, banner = banner_unit()
    character = live_member('Hero')
    host.joinedCharacter = character
    character.hostUnit = host
    assert veteran_counts(host) == (11, 11)
    assert veteran_available(character, personal=True)
    character.hostUnit = None
    host.joinedCharacter = None
    assert not veteran_available(character, personal=True)
    disable_item(host, banner, 'Test suppression')
    assert not veteran_available(host)
    host.unit.model.special_rules.append({'veteran': True})
    assert veteran_available(host)


def test_lost_banner_and_break_tests_do_not_offer_veteran():
    host, _ = banner_unit()
    game = SimpleNamespace(aiControls=lambda member: True)
    roller = AsyncMock(return_value=[1, 1])
    assert asyncio.run(reroll_leadership(game, host, 'Break', [6, 6], 8, roller)) == [6, 6]
    host.unit.command[0]['active'] = False
    assert not veteran_available(host)
    assert asyncio.run(reroll_leadership(game, host, 'Rally', [6, 6], 8, roller)) == [6, 6]
    roller.assert_not_awaited()


def helm_bearer():
    bearer = live_member('Champion')
    bearer.unit.model.set_armour(['Heavy Armour'])
    helm = install_inventory(bearer, [{'name': 'Helm Of Courage', 'category': 'Magic Armour',
                                     'selection_ref': 'bearer/helm', 'owner_ref': 'bearer'}])[0]
    return bearer, helm


def test_helm_armour_changes_real_saves_not_baseline_and_passive_survives_use():
    from battleFunctions import resolve_magic_hits
    from magic_items import spend_ability

    bearer, helm = helm_bearer()
    profile = bearer.unit.model
    assert profile.armor_save == 5 and profile.effective_armour_save() == 4
    with patch('battleFunctions.random.randint', side_effect=[6, 4]):
        assert resolve_magic_hits(bearer.unit, 1, 4, 0) == (1, 1, 0)
    assert spend_ability(helm, 'courage', 'Break', confirmed=True)
    profile.reset_characteristics()
    assert profile.melee_armour_save() == 4 and profile.armor_save == 5
    disable_item(bearer, helm, 'Test suppression')
    assert profile.melee_armour_save() == 5


@pytest.mark.parametrize('accept', [False, True])
def test_helm_break_choice_is_optional_and_persists_one_use(accept):
    from magic_items import reroll_break_test, restore_inventory, save_inventory

    bearer, helm = helm_bearer()
    game = SimpleNamespace(units=[bearer], aiControls=lambda member: False,
                           makeChoiceNew=AsyncMock(return_value='Re-roll\n(break)' if accept else 'Keep'))
    roller = AsyncMock(return_value=[1, 1])
    result = asyncio.run(reroll_break_test(game, bearer, [6, 6], 8, 3, False, roller))
    assert result == ([1, 1] if accept else [6, 6])
    assert bool(helm.uses) is accept
    restore_inventory(bearer, save_inventory(bearer))
    if accept:
        result = asyncio.run(reroll_break_test(game, bearer, [6, 6], 8, 3, False, roller))
        assert result == [6, 6]
        roller.assert_awaited_once()
    assert bearer.unit.model.effective_armour_save() == 4


def test_helm_bsb_priority_and_joined_scope_share_one_item():
    from magic_items import reroll_break_test

    bearer, helm = helm_bearer()
    host = live_member('Host', 10)
    host.joinedCharacter = bearer
    bearer.hostUnit = host
    game = SimpleNamespace(units=[host, bearer], aiControls=lambda member: True)
    roller = AsyncMock(return_value=[6, 6])
    result = asyncio.run(reroll_break_test(game, host, [6, 6], 8, 3, False, roller, bsb=bearer))
    assert result == [6, 6] and not helm.uses
    roller.assert_awaited_once()
    asyncio.run(reroll_break_test(game, host, [6, 6], 8, 3, False, roller))
    assert helm.uses['courage']['count'] == 1
    assert host.unit.model.effective_armour_save() == host.unit.model.armor_save
    assert bearer.unit.model.effective_armour_save() == 4


def test_wand_adds_one_known_spell_without_changing_wizard_level_or_casting_limit():
    from magic_items import known_spell_count
    from spell_system import may_attempt, restore_spellbook

    mage = live_member('Mage')
    restore_spellbook(mage.unit.model, [], 2)
    wand = install_inventory(mage, [{'name': 'Silvery Wand', 'category': 'Arcane Items',
                                    'owner_ref': 'bearer', 'selection_ref': 'bearer/wand'}])[0]
    assert known_spell_count(mage) == 3
    assert mage.unit.model.wizard_level() == 2
    assert not may_attempt(['First', 'Second'], 'Third', mage.unit.model.wizard_level())
    disable_item(mage, wand, 'Test suppression')
    assert known_spell_count(mage) == 2


def test_declined_lost_banner_and_spent_helm_explain_nonapplication(capsys):
    from magic_items import reroll_break_test, spend_ability

    host, _ = banner_unit()
    game = SimpleNamespace(aiControls=lambda member: False, makeChoiceNew=AsyncMock(return_value='Keep'))
    roller = AsyncMock(return_value=[1, 1])
    asyncio.run(reroll_leadership(game, host, 'Rally', [6, 6], 8, roller))
    output = capsys.readouterr().out
    assert 'declines granted Veteran' in output and 'The Banner Of The Bold' in output
    host.unit.command[0]['active'] = False
    asyncio.run(reroll_leadership(game, host, 'Rally', [6, 6], 8, roller))
    assert 'command bearer was lost' in capsys.readouterr().out
    bearer, helm = helm_bearer()
    spend_ability(helm, 'courage', 'Break', confirmed=True)
    asyncio.run(reroll_break_test(game, bearer, [6, 6], 8, 3, False, roller))
    output = capsys.readouterr().out
    assert 'Helm Of Courage' in output and 'ability spent' in output and '2D6=12' in output
    roller.assert_not_awaited()