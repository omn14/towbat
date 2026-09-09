import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import combat_resolution
from combat_resolution import CombatResolver
from game_fsm import GamePhaseFSM
from models import model
from psychology import shieldwall_unavailable_reason
from special_rules import apply_rule_keywords


def test_shieldwall_keyword_is_active():
    profile = model('Dwarf Warrior', '')
    apply_rule_keywords(profile, ['Shieldwall'])
    assert profile.is_shieldwall()
    assert any(rule.get('shieldwall') for rule in profile.special_rules)


@pytest.fixture
def defender():
    profile = model('Dwarf Warrior', '')
    profile.special_rules = []
    profile.characteristics['Special Rules'] = []
    apply_rule_keywords(profile, ['Shieldwall', 'Close Order'])
    profile.set_armour(['Heavy Armour', 'Shield'])
    profile.equip_weapon('hand weapon')
    return SimpleNamespace(
        unit=SimpleNamespace(model=profile, name='Shieldwall defenders', nmodels=10),
        bodyNP=SimpleNamespace(isEmpty=lambda: False),
        wasChargedThisTurn=True, usedShieldwall=False, usedStubborn=False,
        isSkirmisher=False, isInCombatWith=[], spreadToSkirmish=Mock())


@pytest.fixture
def resolver(monkeypatch):
    combat = CombatResolver.__new__(CombatResolver)
    combat.game = SimpleNamespace(aiControls=lambda unit: True, psychology=None,
                                  makeChoiceNew=AsyncMock(return_value='Shieldwall'))
    combat.rollBreakDice = AsyncMock(return_value=[3, 4])
    combat.isOverwhelmed = Mock(return_value=False)
    combat.notifyFleesCombat = Mock()
    monkeypatch.setattr(combat_resolution, 'taskMgr', SimpleNamespace(add=lambda coro: coro),
                        raising=False)
    return combat


def test_converts_fallback_once_without_panic(defender, resolver, capsys):
    defender.unit.model.characteristics['Ld'] = '9'
    assert asyncio.run(resolver.breakTestPass([defender], 3)) == [(defender, 'give_ground')]
    assert defender.usedShieldwall
    resolver.notifyFleesCombat.assert_not_called()
    assert 'Give Ground 2"' in capsys.readouterr().out
    assert asyncio.run(resolver.breakTestPass([defender], 3)) == [(defender, 'fall_back')]
    resolver.notifyFleesCombat.assert_called_once_with(defender)
    assert 'already used' in capsys.readouterr().out


@pytest.mark.parametrize('deferred', [False, True])
def test_turn_end_updates_charge_target_but_not_spent_rule(defender, deferred):
    defender.usedShieldwall = True
    defender.countsAsChargedNextTurn = False
    defender.countsAsChargeTargetNextTurn = deferred
    game = SimpleNamespace(ignore=Mock(), units=[defender], unitCopies=[],
                           roundCounter=Mock())
    phase = SimpleNamespace(game=game, end_of_turn_spells=[])
    GamePhaseFSM.exitCombatPhase(phase)
    assert defender.wasChargedThisTurn is deferred
    assert not defender.countsAsChargeTargetNextTurn
    assert defender.usedShieldwall
    defender.spreadToSkirmish.assert_called_once_with()
    GamePhaseFSM.exitCombatPhase(phase)
    assert not defender.wasChargedThisTurn
    assert defender.usedShieldwall
    assert defender.spreadToSkirmish.call_count == 2


@pytest.mark.parametrize('change, reason', [
    ({'wasChargedThisTurn': False}, 'was not charged'),
    ({'usedShieldwall': True}, 'already used'),
    ({'isSkirmisher': True}, 'Close Order'),
])
def test_ineligible_state_keeps_fallback(defender, resolver, change, reason, capsys):
    for name, value in change.items():
        setattr(defender, name, value)
    assert reason in shieldwall_unavailable_reason(defender)
    assert asyncio.run(resolver.shieldwallOutcome(defender, 'fall_back')) == 'fall_back'
    assert reason in capsys.readouterr().out


@pytest.mark.parametrize('equipment, weapon, reason', [
    (['Heavy Armour'], 'hand weapon', 'not equipped'),
    (['Shield'], 'Great Weapon', 'Requires Two Hands'),
    (['Shield'], 'Additional Hand Weapon', 'Requires Two Hands'),
])
def test_must_be_using_shields(defender, resolver, equipment, weapon, reason):
    profile = defender.unit.model
    profile.set_armour(equipment)
    profile.give_weapon(weapon)
    profile.equip_weapon(profile.weapon_slot(weapon))
    assert reason in shieldwall_unavailable_reason(defender)
    assert asyncio.run(resolver.shieldwallOutcome(defender, 'fall_back')) == 'fall_back'
    assert not defender.usedShieldwall


def test_open_order_is_not_close_order(defender):
    defender.unit.model.characteristics['Special Rules'] = ['Shieldwall', 'Open Order']
    defender.unit.model.special_rules = [
        rule for rule in defender.unit.model.special_rules if rule.get('name') != 'Close Order']
    apply_rule_keywords(defender.unit.model, ['Open Order'])
    assert 'Close Order' in shieldwall_unavailable_reason(defender)


def test_single_survivor_and_disruption_do_not_remove_close_order(defender):
    defender.unit.nmodels = 1
    defender.isDisrupted = True
    assert shieldwall_unavailable_reason(defender) is None


def test_character_rule_does_not_grant_shieldwall(defender, resolver):
    defender.joinedCharacter = SimpleNamespace(unit=defender.unit)
    defender.unit = SimpleNamespace(model=model('Captain', ''), name='Ordinary host')
    assert asyncio.run(resolver.shieldwallOutcome(defender, 'fall_back')) == 'fall_back'
    assert not defender.usedShieldwall


@pytest.mark.parametrize('outcome', ['break', 'give_ground'])
def test_other_results_do_not_spend_shieldwall(defender, resolver, outcome):
    assert asyncio.run(resolver.shieldwallOutcome(defender, outcome)) == outcome
    assert not defender.usedShieldwall


@pytest.mark.parametrize('answer, outcome, spent', [
    ('Shieldwall', 'give_ground', True),
    ('Fall Back in Good Order', 'fall_back', False),
])
def test_human_choice_belongs_to_defender(defender, resolver, answer, outcome, spent):
    resolver.game.aiControls = lambda unit: False
    resolver.game.makeChoiceNew.return_value = answer
    assert asyncio.run(resolver.shieldwallOutcome(defender, 'fall_back')) == outcome
    assert defender.usedShieldwall is spent
    assert resolver.game.makeChoiceNew.call_args.kwargs['owner'] is defender


def test_stubborn_then_shieldwall_even_when_overwhelmed(defender, resolver):
    apply_rule_keywords(defender.unit.model, ['Stubborn'])
    resolver.isOverwhelmed.return_value = True
    assert asyncio.run(resolver.breakTestPass([defender], 8)) == [(defender, 'give_ground')]
    assert defender.usedStubborn and defender.usedShieldwall
    resolver.rollBreakDice.assert_not_called()
    resolver.notifyFleesCombat.assert_not_called()


def test_overwhelmed_break_cannot_be_replaced(defender, resolver):
    defender.unit.model.characteristics['Ld'] = '9'
    resolver.isOverwhelmed.return_value = True
    assert asyncio.run(resolver.breakTestPass([defender], 3)) == [(defender, 'break')]
    assert not defender.usedShieldwall


def test_shieldwall_uses_final_bsb_reroll(defender, resolver):
    defender.unit.model.characteristics['Ld'] = '9'
    resolver.game.psychology = SimpleNamespace(
        leadership_of=lambda unit: (9, None), battle_standard_of=lambda unit: defender)
    resolver.rollBreakDice.side_effect = [[6, 6], [3, 4]]
    assert asyncio.run(resolver.breakTestPass([defender], 3)) == [(defender, 'give_ground')]
    assert resolver.rollBreakDice.await_count == 2
    assert defender.usedShieldwall
    resolver.notifyFleesCombat.assert_not_called()


@pytest.mark.parametrize('answer, using_shields', [
    ('Hand weapon & shield', True), ('Great Weapon', False),
])
def test_shield_choice_precedes_attacks_and_does_not_spend_rule(
        defender, resolver, answer, using_shields):
    profile = defender.unit.model
    profile.give_weapon('Great Weapon')
    profile.equip_best_melee()
    resolver.game.aiControls = lambda unit: False
    resolver.game.makeChoiceNew.return_value = answer
    assert profile.melee_weapon_requires_two_hands()
    asyncio.run(resolver.shieldwallWeaponChoice(defender))
    assert (not profile.melee_weapon_requires_two_hands()) is using_shields
    assert not defender.usedShieldwall
    assert resolver.game.makeChoiceNew.call_args.kwargs['owner'] is defender


def test_ai_chooses_shields_before_combat(defender, resolver):
    profile = defender.unit.model
    profile.give_weapon('Great Weapon')
    profile.equip_best_melee()
    asyncio.run(resolver.shieldwallWeaponChoice(defender))
    assert profile.uses_hand_weapon()
    assert not defender.usedShieldwall
    resolver.game.makeChoiceNew.assert_not_called()