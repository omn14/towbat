"""Actual selected weapon choices and independent character equipment (p. 213)."""

from unittest.mock import AsyncMock, patch

import pytest

from characters import join_unit
from combat_weapons import available_weapons, choose_unit_weapons
from command_groups import champions
from combat_profiles import combat_profiles
from persistence import load_game_state
from tests.test_faction_rules_scene import members, scene as scene
from tests.test_shieldwall_scene import combat_tasks


@pytest.mark.parametrize('weapon', ['Hand Weapon', 'Halberd'])
def test_warriors_choose_defence_or_halberd_without_changing_equipment(scene, weapon):
    app, baseline = scene
    load_game_state(app, baseline)
    warriors = members(app)['Chaos Warrior']
    profile = warriors.unit.model
    equipment = list(profile.armour)
    with combat_tasks(app) as run, patch.object(app, 'aiControls', return_value=False), \
            patch.object(app, 'makeChoiceNew', AsyncMock(return_value=weapon)) as choice:
        run(choose_unit_weapons(app, warriors, set()))
    choice.assert_awaited_once()
    assert profile.equipedWeapon['name'] == weapon
    assert profile.armour == equipment
    assert profile.melee_weapon_requires_two_hands() is (weapon == 'Halberd')
    assert profile.faction_hand_weapon('ensorcelled_weapons') is (weapon == 'Hand Weapon')
    assert profile.parry_applies() is (weapon == 'Hand Weapon')
    assert profile.melee_armour_save() == (5 if weapon == 'Halberd' else 3)


def test_lance_available_on_short_charge_and_champion_follows_unit(scene):
    app, baseline = scene
    load_game_state(app, baseline)
    knights = members(app)['Chaos Knight']
    knights.chargedThisTurn = True
    knights.chargeDistance = 1
    with combat_tasks(app) as run, patch.object(app, 'aiControls', return_value=False), \
            patch.object(app, 'makeChoiceNew', AsyncMock(return_value='Lance')) as choice:
        run(choose_unit_weapons(app, knights, set()))
    choice.assert_awaited_once()
    for profile in (knights.unit.model, champions(knights)[0].unit.model):
        assert profile.equipedWeapon['name'] == 'Lance'
        assert profile.melee_strength_bonus() == 2 and profile.melee_ap() == 2
        assert 'Lance' not in available_weapons(profile, False)
    assert knights.unit.model.get_mount().equipedWeapon['name'] == 'Hand Weapon'


def test_horsemen_throwing_spears_support_riders_only_on_short_charge(scene):
    app, baseline = scene
    load_game_state(app, baseline)
    horsemen, enemy = members(app)['Marauder Horsemen'], members(app)['Silver Helm']
    horsemen.unit.files = 3
    horsemen.unit.ranks = 2
    horsemen.chargedThisTurn = True
    horsemen.chargeDistance = 1
    with combat_tasks(app) as run, patch.object(app, 'aiControls', return_value=False), \
            patch.object(app, 'makeChoiceNew', AsyncMock(return_value='Throwing Spear')):
        run(choose_unit_weapons(app, horsemen, set()))
    horsemen.unit.model.reset_characteristics()
    parts = combat_profiles(horsemen, enemy)
    assert {part.role: part.attacks(5, 5) for part in parts} == {'main': 5, 'mount': 3}
    assert not horsemen.unit.model.missile_weapon()
    horsemen.chargedThisTurn = False
    assert {part.role: part.attacks(5, 5) for part in parts} == {'main': 3, 'mount': 3}


def test_joined_character_chooses_independently_and_duel_keeps_that_choice(scene):
    from challenges import Challenge
    app, baseline = scene
    load_game_state(app, baseline)
    units = members(app)
    champion, warriors = units['Aspiring Champion'], units['Chaos Warrior']
    assert join_unit(app, champion, warriors)
    selected = set()
    async def choose(options, *args, owner, **kwargs):
        return 'Great Weapon' if owner is champion else 'Hand Weapon'
    with combat_tasks(app) as run, patch.object(app, 'aiControls', return_value=False), \
            patch.object(app, 'makeChoiceNew', side_effect=choose) as choice:
        run(choose_unit_weapons(app, warriors, selected))
        app.combat._combatArmedProfiles = selected
        run(app.combat.armDuellists(Challenge(champion, warriors)))
    assert choice.await_count == 2
    assert champion.unit.model.equipedWeapon['name'] == 'Great Weapon'
    assert warriors.unit.model.equipedWeapon['name'] == 'Hand Weapon'
    app.combat._combatArmedProfiles = set()


def test_target_specific_lances_survive_reload_and_resolve_champion_host(scene, tmp_path, capsys):
    from battleFunctions import simulate_battle
    from combat_weapons import weapon_target
    from first_charge import finish_charge_attempt
    from persistence import save_game_state

    app, baseline = scene
    load_game_state(app, baseline)
    units = members(app)
    helms, knights = units['Silver Helm'], units['Chaos Knight']
    helms.chargedThisTurn = True
    helms.chargeDistance = 1
    finish_charge_attempt(helms, knights)
    save_path = tmp_path / 'charge-targets.json'
    assert save_game_state(app, str(save_path)) is not None
    load_game_state(app, str(save_path))
    units = members(app)
    helms, knights, warriors = units['Silver Helm'], units['Chaos Knight'], units['Chaos Warrior']
    profile = helms.unit.model
    profile.equip_weapon('Lance')
    base_strength = int(profile.characteristics['S'])
    observed = []
    def attack(striker, defender):
        observed.append((int(striker.characteristics['S']), striker.melee_ap()))
        striker.ithilmar_rerolled = False
        return False, False
    for target, expected in [(knights, (base_strength + 2, 2)),
                             (warriors, (base_strength, 0)),
                             (champions(knights)[0], (base_strength + 2, 2))]:
        observed.clear()
        with weapon_target(profile, helms, target), patch('battleFunctions.simulate_attack', side_effect=attack):
            simulate_battle(helms.unit, target.unit, True, charge_distance=1)
        assert observed and set(observed) == {expected}
        assert not hasattr(profile, '_charged_target')
    output = capsys.readouterr().out
    assert 'was not charged; weapon S+0, AP-0' in output
    assert 'was charged; weapon S+2, AP-2' in output