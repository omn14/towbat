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


def test_warding_talisman_real_save_log_suppression_and_better_native_ward(capsys):
    from battleFunctions import check_saves, report_ward_saves, ward_save_value
    from magic_items import restore_inventory, save_inventory
    bearer = live_member('Warded Champion')
    profile = bearer.unit.model
    profile.special_rules = []
    item = install_inventory(bearer, [{'name': 'The Warding Talisman', 'category': 'Talismans',
                                     'selection_ref': 'bearer/talisman', 'owner_ref': 'bearer'}])[0]
    assert ward_save_value(profile) == 6
    rolls = []
    with patch('battleFunctions.random.randint', return_value=6):
        assert check_saves(profile, 7, 4, slaying_blow=True, ward_rolls=rolls)
    report_ward_saves(bearer.unit, 1, rolls)
    assert 'The Warding Talisman' in capsys.readouterr().out
    profile.special_rules.append({'name': 'Native Ward', 'ward': 5})
    assert ward_save_value(profile) == 5
    report_ward_saves(bearer.unit, 1, [5])
    assert 'superseded by 5+' in capsys.readouterr().out
    disable_item(bearer, item, 'Test suppression')
    restore_inventory(bearer, save_inventory(bearer))
    assert ward_save_value(profile) == 5
    profile.special_rules.clear()
    assert ward_save_value(profile) == 0


@pytest.mark.parametrize('penetration,allowed,roll,saved', [(0, True, 3, True), (1, True, 3, False),
                                                         (1, True, 5, True), (0, False, 6, False)])
def test_padded_hauberk_heavy_armour_and_ap_zero_only(penetration, allowed, roll, saved, capsys):
    from battleFunctions import check_saves
    from magic_items import report_ap_armour
    bearer = live_member('Padded Champion')
    profile = bearer.unit.model
    profile.special_rules = []
    profile.set_armour(['Shield'])
    item = install_inventory(bearer, [{'name': 'Padded Hauberk', 'category': 'Magic Armour',
                                     'selection_ref': 'bearer/padding', 'owner_ref': 'bearer'}])[0]
    assert profile.armor_save == 6 and profile.effective_armour_save() == 4
    history = []
    with patch('battleFunctions.random.randint', return_value=roll):
        assert check_saves(profile, profile.effective_armour_save(), penetration, allow_armour=allowed,
                           armour_modifiers=history) is saved
    report_ap_armour(profile, history)
    output = capsys.readouterr().out
    assert 'Padded Hauberk' in output
    assert ('4+ -> 3+' in output) is (penetration == 0 and allowed)
    profile.reset_characteristics()
    assert profile.armor_save == 6
    disable_item(bearer, item, 'Test suppression')
    assert profile.effective_armour_save() == 6


def test_padded_hauberk_magic_batch_reports_once_after_saves(capsys):
    from battleFunctions import resolve_magic_hits
    bearer = live_member('Padded Champion')
    bearer.unit.model.special_rules = []
    install_inventory(bearer, [{'name': 'Padded Hauberk', 'category': 'Magic Armour',
                              'selection_ref': 'bearer/padding', 'owner_ref': 'bearer'}])
    with patch('battleFunctions.random.randint', side_effect=[6, 6, 4, 4]):
        assert resolve_magic_hits(bearer.unit, 2, 4, 0) == (2, 2, 0)
    output = capsys.readouterr().out
    assert output.count('2 wound(s) at AP 0: armour 5+ -> 4+') == 1


@pytest.mark.parametrize('accept', [False, True])
def test_wyrdstone_shard_declared_before_cast_and_shared_use_persists(accept):
    from magic_items import magic_roll_bonus, restore_inventory, save_inventory
    from spell_system import Spell
    bearer = live_member('Wizard')
    item = install_inventory(bearer, [{'name': 'Wyrdstone Shard', 'category': 'Arcane Items',
                                     'selection_ref': 'bearer/shard', 'owner_ref': 'bearer'}])[0]
    game = SimpleNamespace(units=[bearer], player1Units=[bearer], player2Units=[],
                           aiControls=lambda member: False,
                           makeChoiceNew=AsyncMock(return_value='Use shard' if accept else 'Keep shard'))
    bearer.game = game
    spell = Spell('Test casting', 8, [], game=game, caster=bearer)
    spell.wizard_level = 2
    async def roll():
        assert bool(item.uses) is accept
        return 6, [3, 3]
    with patch.object(spell, '_roll_casting_dice', side_effect=roll):
        assert asyncio.run(spell._attempt(bearer)) is accept
    assert spell.casting == (8 if accept else 7)
    restore_inventory(bearer, save_inventory(bearer))
    game.makeChoiceNew.reset_mock()
    if accept:
        assert asyncio.run(magic_roll_bonus(game, bearer, 'Dispel')) == 0
        game.makeChoiceNew.assert_not_called()


@pytest.mark.parametrize('accept', [False, True])
def test_banner_of_renown_optional_use_survives_reload(accept):
    from magic_items import combat_result_bonus, restore_inventory, save_inventory
    host, _ = banner_unit()
    item = install_inventory(host, [{'name': 'Banner of Renown', 'category': 'Magic Standards',
                                    'selection_ref': 'standard/renown', 'owner_ref': 'standard'}])[0]
    game = SimpleNamespace(units=[host], aiControls=lambda member: False,
                           makeChoiceNew=AsyncMock(return_value='Raise banner' if accept else 'Keep banner'))
    host.game = game
    assert asyncio.run(combat_result_bonus(game, [host], 3, 3)) == int(accept)
    assert bool(item.uses) is accept
    restore_inventory(host, save_inventory(host))
    if accept:
        game.makeChoiceNew.reset_mock()
        assert asyncio.run(combat_result_bonus(game, [host], 3, 3)) == 0
        game.makeChoiceNew.assert_not_called()


def test_diestros_blade_owned_profile_equips_and_disables_with_item():
    from battleFunctions import strike_initiative
    from combat_weapons import choose_profile
    from magic_items import inventory, restore_inventory, save_inventory
    bearer = live_member('Swordsman')
    profile = bearer.unit.model
    profile.give_weapon('Great Weapon')
    item = install_inventory(bearer, [{'name': "Diestro's Blade", 'category': 'Magic Weapons',
                                     'selection_ref': 'bearer/blade', 'owner_ref': 'bearer'}])[0]
    game = SimpleNamespace(units=[bearer], aiControls=lambda member: True)
    bearer.game = game
    assert asyncio.run(choose_profile(game, bearer, profile, False)) == "Diestro's Blade"
    baseline = int(profile.characteristics['I'])
    assert strike_initiative(profile) == min(10, baseline + 1)
    assert profile.melee_ap() == 1
    assert profile.has_magical_attacks()
    saved = save_inventory(bearer)
    restore_inventory(bearer, saved)
    assert strike_initiative(profile) == min(10, baseline + 1)
    item = inventory(bearer)[0]
    disable_item(bearer, item, 'Test suppression')
    assert not any(weapon.get('item_source') for weapon in profile.weapons.values())
    assert profile.equipedWeapon['name'].casefold() == 'hand weapon'
    assert strike_initiative(profile) == baseline


@pytest.mark.parametrize('arc,reroll', [('front', False), ('flank', True), ('rear', True)])
def test_skirmishers_blade_extra_attack_and_target_specific_wound_reroll(arc, reroll):
    from battleFunctions import attack_characteristic, simulate_attack
    from combat_weapons import weapon_target
    bearer, target = live_member('Swordsman'), live_member('Enemy')
    profile = bearer.unit.model
    install_inventory(bearer, [{'name': "Skirmisher's Blade", 'category': 'Magic Weapons',
                              'selection_ref': 'bearer/blade', 'owner_ref': 'bearer'}])
    profile.equip_weapon("Skirmisher's Blade")
    assert attack_characteristic(profile) == int(profile.characteristics['A']) + 1
    target.isInCombatWith = [bearer]
    target.isInCombatFlank = [arc]
    with weapon_target(profile, bearer, target), patch('battleFunctions.random.randint', side_effect=[6, 1, 6]) as dice:
        assert simulate_attack(profile, target.unit.model) == (True, reroll)
    assert dice.call_count == (3 if reroll else 2)
    assert not hasattr(profile, '_weapon_wound_reroll')
    assert profile.attack_AP == 1


def test_thornspitter_two_profiles_share_ownership_and_use_normal_shooting():
    from battleFunctions import simulate_attack
    from combat_weapons import available_weapons
    from magic_items import inventory, restore_inventory, save_inventory
    bearer, target = live_member('Archer'), live_member('Enemy')
    profile = bearer.unit.model
    item = install_inventory(bearer, [{'name': 'Thornspitter Stave', 'category': 'Magic Weapons',
                                     'selection_ref': 'bearer/stave', 'owner_ref': 'bearer'}])[0]
    assert list(available_weapons(profile, False)) == ['Thornspitter Stave']
    profile.equip_weapon('Thornspitter Stave')
    assert profile.melee_strength_bonus() == 1
    assert profile.melee_ap() == 0 and profile.armour_bane_for_attack() == 1
    assert profile.has_magical_attacks()
    ranged = profile.missile_weapon()
    assert ranged['ranged_range'] == 24 and ranged['item_source'] == item.instance_id
    profile.equip_weapon(ranged['name'])
    profile.characteristics['BS'] = '3'
    for moved, expected in [(False, (True, True)), (True, (False, False))]:
        profile.moved_this_turn = moved
        with patch('battleFunctions.random.randint', side_effect=[4, 6]):
            assert simulate_attack(profile, target.unit.model) == expected
        assert int(profile.characteristics['S']) == 4
        assert profile.attack_AP == (0 if moved else 1)
    restore_inventory(bearer, save_inventory(bearer))
    assert profile.equipedWeapon['name'] == 'Thornspitter Stave (ranged)'
    disable_item(bearer, inventory(bearer)[0], 'Test suppression')
    assert not profile.missile_weapon()
    assert not any(weapon.get('item_source') for weapon in profile.weapons.values())


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