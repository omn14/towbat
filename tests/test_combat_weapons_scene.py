"""Actual selected weapon choices and independent character equipment (p. 213)."""

from unittest.mock import AsyncMock, patch

import pytest

from characters import join_unit
from combat_weapons import available_weapons, choose_unit_weapons
from command_groups import champions
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