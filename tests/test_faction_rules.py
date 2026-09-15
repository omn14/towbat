"""Frequent High Elf and Chaos effects, with explicit profile and cause boundaries."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from battleFunctions import check_saves, report_ward_saves, resolve_magic_hits, strike_initiative, ward_save_value
from battleFunctions import simulate_attack, simulate_battle
from models import model
from special_rules import build_special_rules


def profile(*keywords):
    fighter = model('State Trooper', '')
    fighter.characteristics['Special Rules'] = list(keywords)
    fighter.special_rules = build_special_rules(fighter)
    return fighter


@pytest.mark.parametrize('round_number,expected', [(1, 5), (2, 4)])
def test_martial_prowess_changes_attack_and_defence_without_persisting(round_number, expected):
    from special_rules import martial_prowess
    from toHitAndToWound import to_hit
    fighter, enemy = profile('Martial Prowess'), profile()
    fighter.characteristics['WS'] = enemy.characteristics['WS'] = '4'
    host = SimpleNamespace(roundsFought=round_number, unit=SimpleNamespace(name='Spearmen'))
    with martial_prowess((fighter, host), (fighter, host), (enemy, host)):
        assert int(fighter.characteristics['WS']) == expected
        assert to_hit(fighter, enemy) == (3 if round_number == 1 else 4)
        assert to_hit(enemy, fighter) == 4
        assert enemy.characteristics['WS'] == '4'
    assert fighter.characteristics['WS'] == '4'


@pytest.mark.parametrize('round_number,joined,native,expected', [
    (1, True, False, 7), (2, True, False, 6), (1, False, False, 6), (1, True, True, 7)])
def test_joined_character_receives_unit_martial_prowess_once(round_number, joined, native, expected):
    from special_rules import martial_prowess, martial_prowess_applies
    from toHitAndToWound import to_hit
    regiment_profile = profile('Martial Prowess')
    character_profile = profile(*(['Martial Prowess'] if native else []))
    character_profile.characteristics['WS'] = '6'
    character = SimpleNamespace(unit=SimpleNamespace(model=character_profile, nmodels=1))
    host = SimpleNamespace(unit=SimpleNamespace(model=regiment_profile, name='Spearmen'),
                           roundsFought=round_number, joinedCharacter=character if joined else None)
    character.hostUnit = host if joined else None
    character.roundsFought = round_number
    enemy = profile()
    enemy.characteristics['WS'] = '3'
    assert martial_prowess_applies(character_profile, host) == (joined or native)
    with martial_prowess((character_profile, character), (character_profile, host)):
        assert int(character_profile.characteristics['WS']) == expected
        assert to_hit(enemy, character_profile) == (5 if expected == 7 else 4)
    assert character_profile.characteristics['WS'] == '6'
    host.joinedCharacter = None
    character.hostUnit = None
    with martial_prowess((character_profile, character)):
        assert int(character_profile.characteristics['WS']) == (7 if native and round_number == 1 else 6)


def test_dragon_armour_is_separate_ward_not_body_armour():
    fighter = profile('Dragon Armour')
    fighter.set_armour(['Full Plate Armour', 'Shield', 'Barding'])
    assert fighter.armor_save == 2 and ward_save_value(fighter) == 6
    with patch('battleFunctions.random.randint', side_effect=[1, 6]):
        assert check_saves(fighter, 2, 10)
    with patch('battleFunctions.random.randint', side_effect=[5]):
        assert not check_saves(fighter, 2, 10, slaying_blow=True)
    fighter.special_rules.append({'name': 'Better Ward', 'ward': 4})
    with patch('battleFunctions.random.randint', return_value=4) as dice:
        assert check_saves(fighter, 2, 10, slaying_blow=True)
    assert dice.call_count == 1 and ward_save_value(fighter) == 4


@pytest.mark.parametrize('shooting,magical,expected', [(True, False, 6), (True, True, 0), (False, False, 0)])
def test_deflect_shots_only_protects_nonmagical_shooting(shooting, magical, expected):
    fighter = profile('Deflect Shots')
    attack = {'shooting': shooting, 'magical': magical}
    assert ward_save_value(fighter, attack=attack) == expected
    with patch('battleFunctions.random.randint', return_value=6):
        assert check_saves(fighter, 7, 0, allow_armour=False, attack=attack) == bool(expected)


@pytest.mark.parametrize('shooting,magical,permitted,expected', [
    (True, False, True, 4), (True, True, True, 5), (False, False, True, 5), (True, False, False, 5)])
def test_lion_cloak_is_conditional_armour(shooting, magical, permitted, expected):
    from battleFunctions import conditional_armour_save
    fighter = profile('Lion Cloak')
    attack = {'shooting': shooting, 'magical': magical}
    assert conditional_armour_save(fighter, 5, attack, permitted=permitted) == expected
    assert conditional_armour_save(fighter, 2, attack, permitted=permitted) == 2
    with patch('battleFunctions.random.randint', return_value=4):
        assert check_saves(fighter, 5, 0, attack=attack, allow_armour=permitted) == (expected == 4)


def test_warden_grants_terrain_reroll_and_killing_blow_without_extra_exported_rules():
    fighter = profile('Warden of Saphery')
    assert fighter.has_killing_blow()
    assert fighter.dangerous_terrain_reroll_sources() == ['Ithilmar Armour']
    assert ward_save_value(fighter, attack={'shooting': True}) == 6


@pytest.mark.parametrize('magical', [False, True])
def test_deflect_shots_reaches_shooting_and_cannon_batches(magical, capsys):
    from cannon_fire import CannonFire
    attacker, defender = profile(), profile('Deflect Shots')
    attacker.give_weapon('Shortbow')
    attacker.equip_weapon('Shortbow')
    attacker.equipedWeapon['magical'] = magical
    firing = SimpleNamespace(model=attacker, name='Archer', nmodels=1, files=1, ranks=1)
    target = SimpleNamespace(model=defender, name='Noble', nmodels=1, files=1, ranks=1)
    with patch('battleFunctions.random.randint', return_value=6):
        assert simulate_battle(firing, target, charge=False)[-1] == int(magical)
    cannon_target = SimpleNamespace(unit=target, model=SimpleNamespace(getChildren=lambda: [object()]))
    with patch('battleFunctions.random.randint', return_value=6):
        assert CannonFire.__new__(CannonFire)._apply_wounds(cannon_target, 1, 10, 3,
                                                          magical=magical)[0] == int(magical)
    assert 'Deflect Shots' in capsys.readouterr().out


def test_lion_cloak_reaches_shooting_save_without_changing_melee_armour(capsys):
    attacker, defender = profile(), profile('Lion Cloak')
    defender.set_armour(['Heavy Armour'])
    attacker.give_weapon('Shortbow')
    attacker.equip_weapon('Shortbow')
    firing = SimpleNamespace(model=attacker, name='Archer', nmodels=1, files=1, ranks=1)
    target = SimpleNamespace(model=defender, name='Lion', nmodels=1, files=1, ranks=1)
    with patch('battleFunctions.random.randint', side_effect=[6, 6, 4]):
        assert simulate_battle(firing, target, charge=False)[-1] == 0
    assert defender.armor_save == 5 and defender.melee_armour_save() == 5
    assert '5+ -> 4+' in capsys.readouterr().out


@pytest.mark.parametrize('keyword,value', [('Chaos Armour (5+)', 5), ('Chaos Armour (6+)', 6),
                                         ('Chaos Armour', 0), ('Chaos Armour (X+)', 0)])
def test_chaos_armour_uses_only_explicit_valid_ward(keyword, value):
    fighter = profile(keyword)
    assert ward_save_value(fighter) == value and fighter.armor_save == 7


def test_faction_ward_magic_batch_logs_deciding_rolls_once(capsys):
    target = SimpleNamespace(model=profile('Chaos Armour (5+)'), name='Champion')
    with patch('battleFunctions.random.randint', side_effect=[6, 6, 1, 5, 1, 4]):
        assert resolve_magic_hits(target, 2, 4, 0) == (2, 1, 1)
    output = capsys.readouterr().out
    assert output.count('Chaos Armour') == 1
    assert 'saves 1/2' in output and '[5, 4]' in output
    target.model.special_rules.append({'name': 'Spell Ward', 'ward': 4})
    report_ward_saves(target, 2, [4, 4])
    assert 'superseded by 4+' in capsys.readouterr().out
    target.model.special_rules.pop()
    report_ward_saves(target, 0, [])
    assert 'none reached the 5+ Ward' in capsys.readouterr().out


def test_war_machine_wounds_allow_faction_ward(capsys):
    from cannon_fire import CannonFire
    from bombardment import Bombardment

    fighter = profile('Chaos Armour (5+)')
    target = SimpleNamespace(unit=SimpleNamespace(model=fighter, name='Champion'),
                             model=SimpleNamespace(getChildren=lambda: [object()]))
    cannon = CannonFire.__new__(CannonFire)
    bombardment = Bombardment.__new__(Bombardment)
    with patch('battleFunctions.random.randint', side_effect=[6, 1, 5]):
        assert cannon._apply_wounds(target, 1, 10, 3) == (0, 1, 1)
    assert 'saves 1/1' in capsys.readouterr().out
    rolls = []
    with patch('battleFunctions.random.randint', side_effect=[6, 1, 5]):
        assert not bombardment._wound_unsaved(fighter, 10, 3, ward_rolls=rolls)
    assert rolls == [5]


def test_template_ward_report_is_once_per_target(capsys):
    from bombardment import Bombardment

    children = [object(), object()]
    target = Mock(unit=SimpleNamespace(model=profile('Chaos Armour (5+)'), name='Warriors'))
    target.model.getChildren.return_value = children
    bombardment = Bombardment.__new__(Bombardment)
    bombardment.game = Mock()
    under = [(target, child, index) for index, child in enumerate(children)]
    with patch.object(bombardment, '_models_under_template', return_value=under), \
            patch('battleFunctions.random.randint', side_effect=[6, 1, 5, 6, 1, 6]):
        bombardment._resolve_damage(object(), {'ranged_strength': 10}, None, 1)
    bombardment.game.removeModelsFromUnit.assert_not_called()
    output = capsys.readouterr().out
    assert output.count('Chaos Armour') == 1
    assert 'saves 2/2' in output and '[5, 6]' in output


def test_scoped_reflexes_survives_catalogue_merge_and_is_revoked():
    from special_rules import apply_rule_keywords

    hull = model('Lothern Skycutter', '')
    crew, beast = hull.get_crew(), hull.get_beasts()
    keywords = list(hull.characteristics['Special Rules'])
    original = strike_initiative(crew, first_round=True)
    apply_rule_keywords(hull, ['Elven Reflexes (Sea Guard Crew only)'])
    assert strike_initiative(crew, first_round=True) == original + 1
    assert strike_initiative(crew, first_round=False) == original
    assert strike_initiative(beast, first_round=True) == strike_initiative(beast)
    apply_rule_keywords(hull, keywords, replace=True)
    assert strike_initiative(crew, first_round=True) == original
    apply_rule_keywords(hull, ['Elven Reflexes (Sea Guard Crew only)'])
    hull.attach_crew(crew, 3)
    assert strike_initiative(crew, first_round=True) == original + 1
    assert sum(bool(rule.get('parent_crew_reflexes')) for rule in crew.special_rules) == 1


def test_charge_log_does_not_count_reflexes_as_charge_bonus(capsys):
    from tests.test_initiative import _engage, _fighter, _order
    from special_rules import apply_rule_keywords

    charger = _fighter('Silver Helm', 4, charged=True, distance=4)
    defender = _fighter('Chaos Knight', 3)
    apply_rule_keywords(charger.unit.model, ['Elven Reflexes'])
    charger.roundsFought = 1
    _engage(charger, defender)
    assert _order((charger, defender)) == [(8, 0)]
    output = capsys.readouterr().out
    assert 'I4 -> I5' in output and '+3 Initiative (I5 -> I8' in output


@pytest.mark.parametrize('first,charged,initial,expected', [
    (True, False, 4, 5), (False, False, 4, 4), (True, True, 4, 8),
    (True, False, 10, 10), (False, True, 4, 7)])
def test_elven_reflexes_first_round_and_cap(first, charged, initial, expected, capsys):
    fighter = profile('Elven Reflexes')
    fighter.characteristics['I'] = str(initial)
    assert strike_initiative(fighter, charged, 4, first_round=first, log=True) == expected
    assert fighter.characteristics['I'] == str(initial)
    assert 'Elven Reflexes' in capsys.readouterr().out
    mount = profile()
    mount.characteristics['I'] = '3'
    assert strike_initiative(mount, first_round=first) == 3


@pytest.mark.parametrize('name,extra,eligible', [('Hand Weapon', {}, True), ('Lance', {}, False),
    ('Great Weapon', {}, False), ('Two Hand Weapons', {}, False),
    ('Hand Weapon', {'magical': True}, False),
    ('Hand Weapon', {'special_rules': ['Magical Attacks']}, False),
    ('Hand Weapon', {'tag': 'ranged'}, False)])
def test_faction_weapon_scope(name, extra, eligible):
    fighter = profile('Ithilmar Weapons', 'Ensorcelled Weapons')
    fighter.equipedWeapon = dict(name=name, tag='combat', ap_penetration=0)
    fighter.equipedWeapon.update(extra)
    assert fighter.faction_hand_weapon('ithilmar_weapons') == eligible
    assert fighter.melee_ap() == int(eligible)
    if eligible:
        assert fighter.has_magical_attacks()
    mount = profile()
    assert not mount.faction_hand_weapon('ensorcelled_weapons') and not mount.has_magical_attacks()


@pytest.mark.parametrize('hatred', [False, True])
def test_ithilmar_shares_one_reroll_with_hatred(hatred):
    fighter, victim = profile('Ithilmar Weapons'), profile()
    fighter.hatred_rerolls = hatred
    with patch('battleFunctions.random.randint', side_effect=[1, 6, 1]) as dice:
        assert simulate_attack(fighter, victim) == (False, False)
    assert dice.call_count == 3
    assert fighter.ithilmar_rerolled != hatred


def test_weapon_batch_logs_and_does_not_mutate_stored_weapon(capsys):
    fighter = profile('Ithilmar Weapons', 'Ensorcelled Weapons')
    attacker = SimpleNamespace(name='Elf', model=fighter, nmodels=1, files=1, _attack_count=1)
    defender = SimpleNamespace(name='Enemy', model=profile(), nmodels=1, files=1)
    with patch('battleFunctions.random.randint', side_effect=[1, 6, 6, 1]):
        assert simulate_battle(attacker, defender, charge=False)[-1] == 1
    output = capsys.readouterr().out
    assert '1 natural 1(s)' in output and 'AP0 -> AP-1' in output
    assert fighter.attack_magical and fighter.equipedWeapon.get('ap_penetration', 0) == 0


@pytest.mark.parametrize('keyword,kind,cause,allowed', [
    ('Valour of Ages', 'Panic', 'heavy casualties (shooting)', True),
    ('Valour of Ages', 'Panic', 'fled through', True),
    ('Valour of Ages', 'Panic', 'nearby friend destroyed', False),
    ('Valour of Ages', 'Panic', 'nearby friend flees combat', False),
    ('Valour of Ages', 'Fear', '', False), ('Valour of Ages', 'Rally', '', False),
    ('Mark of Chaos Undivided', 'Panic', 'nearby friend destroyed', True),
    ('Mark of Chaos Undivided', 'Fear', '', True), ('Mark of Chaos Undivided', 'Terror', '', True),
    ('Mark of Chaos Undivided', 'Break', '', False), ('Mark of Chaos Undivided', 'Rally', '', False)])
def test_faction_leadership_reroll_gates(keyword, kind, cause, allowed, capsys):
    from psychology import reroll_leadership
    unit = SimpleNamespace(unit=SimpleNamespace(model=profile(keyword), nmodels=5, name='Faction unit'))
    game = SimpleNamespace(aiControls=lambda unit: True)
    dice = AsyncMock(return_value=[6, 6])
    result = asyncio.run(reroll_leadership(game, unit, kind, [5, 6], 7, dice, cause=cause))
    assert result == ([6, 6] if allowed else [5, 6])
    assert dice.await_count == int(allowed)
    assert keyword in capsys.readouterr().out


@pytest.mark.parametrize('answer', ['Re-roll', 'Keep'])
def test_mark_and_veteran_share_optional_single_roll(answer):
    from psychology import reroll_leadership
    unit = SimpleNamespace(unit=SimpleNamespace(model=profile('Veteran', 'Mark of Chaos Undivided'),
                                               nmodels=5, name='Warriors'))
    game = SimpleNamespace(aiControls=lambda unit: False, makeChoiceNew=AsyncMock(return_value=answer))
    dice = AsyncMock(return_value=[6, 6])
    result = asyncio.run(reroll_leadership(game, unit, 'Panic', [5, 6], 7, dice))
    assert result == ([6, 6] if answer == 'Re-roll' else [5, 6])
    assert dice.await_count == int(answer == 'Re-roll')
    game.makeChoiceNew.assert_awaited_once()


def test_valour_uses_live_panic_cause_and_queue():
    from psychology import PsychologySystem
    unit = SimpleNamespace(unit=SimpleNamespace(model=profile('Valour of Ages'), nmodels=5, name='Elves'),
                           bodyNP=SimpleNamespace(isEmpty=lambda: False))
    queued = []
    game = SimpleNamespace(aiControls=lambda unit: False, makeChoiceNew=AsyncMock(return_value='Re-roll'),
                           taskMgr=SimpleNamespace(add=queued.append))
    system = PsychologySystem(game)
    system.panic_exempt_reason = lambda unit: None
    system.leadership_of = lambda unit: (7, None)
    system.venerable_source = system.battle_standard_of = lambda unit: None
    system._panic_result = Mock()
    with patch('psychology.random.randint', side_effect=[6, 6, 1, 2]):
        system._resolve_panic(unit, None, 'fled through', Mock())
        assert len(queued) == 1
        system._panic_result.assert_not_called()
        asyncio.run(queued.pop())
    assert system._panic_result.call_args.args[-2:] == (True, 3)


@pytest.mark.parametrize('cover', [False, True])
def test_ithilmar_barding_rerolls_once_without_lending_movement_immunity(cover, capsys):
    from terrain_system import dangerous_terrain_wounds
    fighter = profile('Ithilmar Barding', *(['Move Through Cover'] if cover else []))
    assert fighter.is_move_through_cover() == cover
    sources = fighter.dangerous_terrain_reroll_sources()
    assert 'Ithilmar Barding' in sources
    with patch('terrain_system.random.randint', side_effect=[1, 1, 1, 4]) as dice:
        assert dangerous_terrain_wounds(1, 2, reroll_sources=sources, subject=fighter) == 1
    assert dice.call_count == 4
    output = capsys.readouterr().out
    assert 'Ithilmar Barding' in output and '1 mishap(s) avoided' in output
    with patch('terrain_system.random.randint', return_value=4):
        assert dangerous_terrain_wounds(1, 1, reroll_sources=sources, subject=fighter) == 0
    assert 'no 1s' in capsys.readouterr().out