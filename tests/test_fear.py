"""Fear tests and immunities (Rulebook pp. 168, 169, 171, 179)."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fear import attack_penalty, fears, immune, terror_test, test_fear
from models import model
from special_rules import apply_rule_keywords


def member(name='Chaos Warriors', count=2, rules=()):
    profile = model('Chaos Warrior', '')
    apply_rule_keywords(profile, list(rules), replace=True)
    return SimpleNamespace(unit=SimpleNamespace(model=profile, nmodels=count, name=name),
                           unitName=name, state='Idle')


def test_fear_immunity_and_terror_escalation():
    ordinary = member()
    scary = member(rules=['Fear'])
    terror = member(rules=['Terror'])
    assert fears(ordinary, scary) and not fears(scary, scary)
    assert fears(scary, terror) and not fears(terror, terror)
    ordinary.joinedCharacter = member(count=1, rules=['Fear'])
    assert fears(ordinary, scary)


def test_flaming_model_causes_fear_in_war_beasts_not_ordinary_infantry():
    flames = member(rules=['Flaming Attacks'])
    target = member()
    assert not fears(target, flames)
    target.unit.model.characteristics['Troop Type'] = 'War Beasts'
    assert fears(target, flames)
    flames.unit.model.special_rules = []
    flames.unit.model.equipedWeapon = {'special_rules': ['Flaming Attacks']}
    assert not fears(target, flames)


def test_immune_to_psychology_requires_majority():
    target = member(count=1, rules=['Immune to Psychology'])
    target.joinedCharacter = member(count=1)
    assert not immune(target)
    target.unit.nmodels = 2
    assert immune(target)


def test_relative_strength_and_one_failed_test_per_turn():
    target, enemy = member(), member('Scary', count=3, rules=['Fear'])
    game = SimpleNamespace(roundCounter=SimpleNamespace(current_player=1, currentRoundPlayer=[1, 1]),
                           psychology=SimpleNamespace(leadership_of=lambda unit: (7, None)),
                           aiControls=lambda unit: True, makeChoiceNew=AsyncMock())
    with patch('fear.random.randint', return_value=6) as roll:
        assert not asyncio.run(test_fear(game, target, [enemy], 'charge'))
        assert not asyncio.run(test_fear(game, target, [enemy], 'combat'))
    assert roll.call_count == 2
    enemy.unit.nmodels = 2
    assert asyncio.run(test_fear(game, target, [enemy], 'charge'))


def test_mark_of_undivided_rerolls_failed_fear_once():
    target = member(rules=['Mark of Chaos Undivided'])
    enemy = member('Scary', count=3, rules=['Fear'])
    game = SimpleNamespace(roundCounter=SimpleNamespace(current_player=1, currentRoundPlayer=[1, 1]),
                           psychology=SimpleNamespace(leadership_of=lambda unit: (7, None)),
                           aiControls=lambda unit: True)
    with patch('fear.random.randint', side_effect=[6, 6, 1, 1]) as roll:
        assert asyncio.run(test_fear(game, target, [enemy], 'charge'))
    assert roll.call_count == 4


def test_combat_penalty_is_targeted_and_does_not_leak_to_other_attacks():
    from battleFunctions import _apply_to_hit_modifiers
    from magic_items import current_turn
    target, enemy = member(), member('Scary', rules=['Fear'])
    game = SimpleNamespace(roundCounter=SimpleNamespace(current_player=1, currentRoundPlayer=[1, 1]))
    target.fearFailed = True
    target.fearTestTurn = current_turn(game)
    target.fearTargets = [enemy.unitName]
    profile = target.unit.model
    with attack_penalty(game, target, enemy, profile):
        assert _apply_to_hit_modifiers(profile, 4) == 3
    assert _apply_to_hit_modifiers(profile, 4) == 4
    with attack_penalty(game, target, member(), profile):
        assert _apply_to_hit_modifiers(profile, 4) == 4
    with attack_penalty(game, target, member('Other scary unit', rules=['Fear']), profile):
        assert _apply_to_hit_modifiers(profile, 4) == 4


def test_terror_uses_mark_reroll_but_does_not_test_units_that_cannot_flee():
    target, enemy = member(rules=['Mark of Chaos Undivided']), member(rules=['Terror'])
    game = SimpleNamespace(psychology=SimpleNamespace(leadership_of=lambda unit: (7, None)),
                           aiControls=lambda unit: True)
    with patch('fear.random.randint', side_effect=[6, 6, 1, 1]) as roll:
        assert asyncio.run(terror_test(game, target, enemy))
    assert roll.call_count == 4
    target.isInCombat = True
    with patch('fear.random.randint') as roll:
        assert asyncio.run(terror_test(game, target, enemy))
    roll.assert_not_called()