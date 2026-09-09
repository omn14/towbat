"""Command promotions, scoring and attrition (Rulebook pp. 198-201)."""

from types import SimpleNamespace

from command_groups import (has_command, install_command, living_command,
                            musician_bonus, musician_leadership,
                            remove_command_casualties, standard_bonus)


def member(*roles, count=4, files=4):
    group = SimpleNamespace(nmodels=count, files=files)
    result = SimpleNamespace(unit=group, unitName='Knights')
    install_command(result, [{'role': role, 'name': role} for role in roles])
    return result


def test_standards_do_not_stack_or_act_as_bsb():
    first = member('standard_bearer')
    second = member('standard_bearer')
    assert standard_bonus([first, second]) == 1
    assert not getattr(first, 'isBSB', False)
    first.unit.nmodels = second.unit.nmodels = 0
    assert standard_bonus([first, second]) == 0


def test_musician_only_breaks_a_final_tie():
    music = member('musician')
    ordinary = member()
    assert musician_bonus([music], [ordinary], 4, 4) == (1, 0)
    assert musician_bonus([ordinary], [music], 4, 4) == (0, 1)
    assert musician_bonus([music], [music], 4, 4) == (0, 0)
    assert musician_bonus([music], [ordinary], 3, 4) == (0, 0)


def test_musician_must_be_in_front_rank_for_combat():
    command = member('champion', 'standard_bearer', 'musician', files=2)
    assert has_command(command, 'musician')
    assert not has_command(command, 'musician', front=True)


def test_casualty_priority_and_no_resurrection():
    command = member('musician', 'champion', 'standard_bearer')
    assert command.unit.nmodels == 4
    command.unit.nmodels = 2
    remove_command_casualties(command, log=False)
    assert [entry['role'] for entry in living_command(command)] == ['champion', 'standard_bearer']
    command.unit.nmodels = 1
    remove_command_casualties(command, log=False)
    assert has_command(command, 'champion')
    assert not has_command(command, 'standard_bearer')
    command.unit.nmodels = 4
    assert not has_command(command, 'musician')


def test_musician_only_modifies_rally_and_march():
    command = member('musician')
    assert musician_leadership(command, 8, 'rally') == 9
    assert musician_leadership(command, 9, 'march') == 10
    assert musician_leadership(command, 10, 'rally') == 10
    assert musician_leadership(command, 8, 'break') == 8
    assert musician_leadership(command, 8, 'panic') == 8


def test_roster_champion_is_a_promotion_with_separate_profile():
    from models import model
    from roster_runtime import apply_roster_ownership

    group = SimpleNamespace(model=model('Chaos Knight', ''), nmodels=4, files=4)
    apply_roster_ownership(group, {'command': [
        {'selection_ref': 'champ', 'role': 'champion', 'name': 'Champion',
         'profiles': [{'name': 'Champion', 'typeName': 'Model',
                       'characteristics': [{'name': 'A', '$text': '2'}]}]}]})
    assert group.nmodels == 4
    assert group.model.characteristics['A'] == '1'
    assert group.command_models['champ'].characteristics['A'] == '2'
    group.command_models['champ'].reset_characteristics()
    assert group.command_models['champ'].characteristics['A'] == '2'


def test_split_equipment_does_not_leak_between_profiles():
    from models import model
    from roster_runtime import apply_roster_ownership
    from persistence import _save_profile_state, _restore_profile_state

    profile = model('Lothern Skycutter', '')
    group = SimpleNamespace(model=profile, nmodels=1, files=1)
    records = [{'ref': 'hull', 'name': 'Lothern Skycutter'}, {'ref': 'roc', 'name': 'Swiftfeather Roc'}]
    equipment = [{'category': 'Weapon', 'owner_ref': owner,
                  'profile': {'name': name, 'characteristics': []}} for owner, name in
                 [('hull', 'Cavalry Spear'), ('hull', 'Shortbow'), ('roc', 'Wicked Claws')]]
    apply_roster_ownership(group, {'roster_selections': records, 'equipment': equipment})
    assert profile.get_crew().weapon_slot('Cavalry Spear') is not None
    assert profile.get_crew().weapon_slot('Wicked Claws') is None
    assert profile.get_beasts().weapon_slot('Wicked Claws') is not None
    assert profile.get_beasts().weapon_slot('Cavalry Spear') is None
    restored = model('Lothern Skycutter', '')
    _restore_profile_state(restored, _save_profile_state(profile))
    assert restored.get_crew().weapon_slot('Cavalry Spear') is not None
    assert restored.get_beasts().weapon_slot('Cavalry Spear') is None


def test_knights_have_three_riders_one_champion_and_four_mounts():
    from combat_profiles import combat_profiles
    from models import model
    from roster_runtime import apply_roster_ownership

    host = member('champion', count=4)
    host.unit.model = model('Chaos Knight', '')
    host.unit.model.attach_mount(model('Chaos Steed', ''))
    host.unit.name = 'Knights'
    apply_roster_ownership(host.unit, {'command': [{'role': 'champion', 'profiles': [
        {'name': 'Doom Knight', 'typeName': 'Model',
         'characteristics': [{'name': 'A', '$text': '2'}]}]}]})
    parts = combat_profiles(host, None)
    assert {part.role: part.attacks(4, 4) for part in parts} == {
        'main': 3, 'champion': 2, 'mount': 4}
    assert {part.role: part.attacks(2, 4) for part in parts} == {
        'main': 1, 'champion': 2, 'mount': 2}


def test_parts_use_their_own_initiative_and_do_not_repeat():
    from combat_profiles import profile_strike_order
    from models import model

    host = member(count=1, files=1)
    host.unit.model = model('Lothern Skycutter', '')
    host.unit.model.get_beasts().characteristics['I'] = '5'
    host.hasAttackedThisTurn = False
    order = profile_strike_order([host, host], [None, None], lambda *_: 'front')
    assert len(order) == 2
    assert {part.role: initiative for initiative, part in order} == {
        'crew': 4, 'beasts': 5}
    assert {part.role: part.attacks(1, 1) for _, part in order} == {'crew': 3, 'beasts': 2}
    assert all(part.attacks(0, 1) == 0 for _, part in order)


def test_champion_can_challenge_and_keeps_personal_state():
    from command_groups import champions
    from challenges import duellist, refusal_barred, wounds_remaining
    from models import model
    from roster_runtime import apply_roster_ownership

    host = member(count=4)
    host.unit.model = model('Chaos Knight', '')
    apply_roster_ownership(host.unit, {'command': [{'role': 'champion'}]})
    champion = duellist(host)
    assert champion is champions(host)[0]
    assert refusal_barred(champion, host) is None
    assert wounds_remaining(champion) == 1
    champion.retiredFromCombat = True
    assert duellist(host) is None
    assert host.unit.command[0]['retired']
    champion.woundsOnModel = 1
    assert host.unit.command[0]['wounds'] == 1


def test_combat_overflow_leaves_champion_until_last_model():
    from combat_resolution import CombatResolver
    from models import model
    from roster_runtime import apply_roster_ownership

    target = member(count=4)
    target.unit.model = model('Chaos Knight', '')
    apply_roster_ownership(target.unit, {'command': [{'role': 'champion'}]})
    resolver = object.__new__(CombatResolver)
    resolver._pendingWounds = {}
    assert resolver.commandWoundLimit(target, 10) == (3, 0)
    resolver.previewCombatWounds(target, 3)
    assert target.unit.nmodels == 1
    assert resolver.commandWoundLimit(target, 10) == (1, 0)


def test_multi_wound_body_survives_until_later_profile_step():
    from combat_resolution import CombatResolver
    from models import model

    target = member(count=1)
    target.unit.model = model('Lothern Skycutter', '')
    resolver = object.__new__(CombatResolver)
    resolver.previewCombatWounds(target, 1)
    assert target.unit.nmodels == 1
    resolver.previewCombatWounds(target, 2)
    assert target.unit.nmodels == 1
    resolver.previewCombatWounds(target, 1)
    assert target.unit.nmodels == 0


def test_standard_takes_centre_and_command_overflows_to_rear():
    from command_groups import command_positions

    host = member('champion', 'standard_bearer', 'musician', count=4, files=4)
    slots = command_positions(host)
    assert slots[id(host.unit.command[1])] == 2
    assert all(slot < 4 for slot in slots.values())
    host.unit.files = 2
    assert not has_command(host, 'musician', front=True)
    assert has_command(host, 'champion', front=True)
    assert has_command(host, 'standard_bearer', front=True)


def test_standard_trophy_is_permanent_and_does_not_duplicate():
    from command_groups import capture_standard

    loser = member('standard_bearer')
    winner = member(count=5)
    game = SimpleNamespace(player1Units=[loser], player2Units=[winner])
    capture_standard(game, loser, winner)
    capture_standard(game, loser, winner)
    assert not has_command(loser, 'standard_bearer')
    assert len(game.capturedStandards) == 1
    assert game.capturedStandards[0]['victory_points'] == 50
    assert game.capturedStandards[0]['captured_by'] == 2


def test_musician_changes_enemy_sighted_result_and_failure_counts_as_march():
    import asyncio
    from unittest.mock import AsyncMock, patch
    from marching import enemy_sighted_test

    host = member('musician')
    enemy = member()
    game = SimpleNamespace(psychology=SimpleNamespace(leadership_of=lambda _: (7, None)),
                           rollLeadershipDice=AsyncMock(return_value=[4, 4]))
    with patch('marching.reroll_leadership', AsyncMock(return_value=[4, 4])):
        assert asyncio.run(enemy_sighted_test(game, host, enemy, 8))
        assert host.marchTestResult == 'passed'
        host.unit.command.clear()
        assert not asyncio.run(enemy_sighted_test(game, host, enemy, 8))
        assert host.marchTestResult == 'failed'
        assert host.marchedThisTurn


def test_drilled_and_flying_marches_do_not_test():
    from unittest.mock import Mock
    from marching import request_march
    from models import model
    from special_rules import apply_rule_keywords

    for rule in ('Drilled', 'Fly (10)'):
        host = member()
        host.unit.model = model('Chaos Knight', '')
        apply_rule_keywords(host.unit.model, [rule])
        game = SimpleNamespace(taskMgr=Mock())
        assert request_march(game, host, lambda: None)
        game.taskMgr.add.assert_not_called()


def test_split_attack_counts_include_existing_charge_hooks():
    from unittest.mock import patch
    from battleFunctions import simulate_battle
    from combat_profiles import combat_profiles
    from models import model
    from special_rules import apply_rule_keywords

    host = member(count=4)
    host.unit.name = 'Knights'
    host.unit.model = model('Chaos Knight', '')
    host.unit.ranks = 1
    host.chargedThisTurn = True
    apply_rule_keywords(host.unit.model, ['Furious Charge'])
    target = SimpleNamespace(name='Enemy', model=model('Chaos Warrior', ''), nmodels=4, files=4, ranks=1)
    part = combat_profiles(host, None)[0]
    with patch('battleFunctions.simulate_attack', return_value=(False, False)):
        result = simulate_battle(part.unit(lambda: part.attacks(4, 4)), target, charge=True)
    assert result[0] == 8
    assert host.unit.model.characteristics['A'] == '1'