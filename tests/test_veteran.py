import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from models import model
from game import MyApp
import game as game_module
import combat_resolution
from combat_resolution import CombatResolver
from psychology import veteran_available, veteran_counts
from psychology import leadership_passed
from psychology import reroll_leadership
from psychology import PsychologySystem
from post_combat import restraint_test
from special_rules import apply_rule_keywords


def make_unit(count=5, veteran=True):
    profile = model('Dwarf Warrior', '')
    apply_rule_keywords(profile, ['Veteran'] if veteran else [], replace=True)
    return SimpleNamespace(unit=SimpleNamespace(model=profile, nmodels=count, name='Veterans'),
                           joinedCharacter=None)


def test_keyword_is_coded():
    assert make_unit().unit.model.is_veteran()
    assert not make_unit(veteran=False).unit.model.is_veteran()


@pytest.mark.parametrize('count, veteran, char_veteran, expected', [
    (5, True, False, True), (1, True, False, False),
    (5, False, True, False), (1, False, True, False),
    (1, True, True, True), (0, False, True, True),
])
def test_strict_model_majority_with_character(count, veteran, char_veteran, expected):
    unit = make_unit(count, veteran)
    unit.joinedCharacter = make_unit(1, char_veteran)
    assert veteran_available(unit) is expected
    assert veteran_counts(unit) == (count * veteran + int(char_veteran), count + 1)


def test_faq_personal_character_test_does_not_borrow_host_rule():
    host = make_unit()
    character = make_unit(1, False)
    host.joinedCharacter = character
    character.hostUnit = host
    assert veteran_available(host)
    assert not veteran_available(character, personal=True)
    apply_rule_keywords(character.unit.model, ['Veteran'])
    assert veteran_available(character, personal=True)


def test_empty_unit_has_no_majority():
    assert not veteran_available(make_unit(0))


@pytest.mark.parametrize('dice, ld, modifier, passed', [
    ([1, 1], 1, -5, True), ([6, 6], 12, 5, False),
    ([3, 4], 8, -1, True), ([4, 4], 8, -1, False),
])
def test_leadership_extremes_and_modifiers(dice, ld, modifier, passed):
    assert leadership_passed(sum(dice), ld, modifier) is passed
    assert restraint_test(ld + modifier, dice) is passed


def test_retired_character_cannot_supply_majority():
    host = make_unit(1, True)
    host.joinedCharacter = make_unit(1, True)
    host.joinedCharacter.retiredFromCombat = True
    assert veteran_counts(host) == (1, 2)
    assert not veteran_available(host)


@pytest.mark.parametrize('kind', ['Rally', 'Restraint', 'Panic'])
def test_failed_test_rerolled_once(kind, capsys):
    unit = make_unit()
    game = SimpleNamespace(aiControls=lambda unit: True)
    roll = AsyncMock(return_value=[2, 3])
    result = asyncio.run(reroll_leadership(game, unit, kind, [5, 6], 7, roll))
    assert result == [2, 3]
    roll.assert_awaited_once()
    output = capsys.readouterr().out
    assert '5/5 Veteran models' in output and '2D6=11 vs Ld 7' in output and 'PASS' in output


@pytest.mark.parametrize('answer, expected', [('Re-roll', [6, 6]), ('Keep', [4, 5])])
def test_optional_reroll_and_second_failure_stands(answer, expected):
    unit = make_unit()
    game = SimpleNamespace(aiControls=lambda unit: False,
                           makeChoiceNew=AsyncMock(return_value=answer))
    roll = AsyncMock(return_value=[6, 6])
    assert asyncio.run(reroll_leadership(game, unit, 'Rally', [4, 5], 7, roll)) == expected
    assert roll.await_count == (answer == 'Re-roll')
    assert game.makeChoiceNew.call_args.kwargs['owner'] is unit


@pytest.mark.parametrize('dice, kind, veteran, expected', [
    ([1, 1], 'Rally', True, False), ([3, 4], 'Rally', True, False),
    ([6, 6], 'Break', True, False), ([6, 6], 'Rally', False, False),
    ([6, 6], 'Rally', True, True),
])
def test_only_failed_leadership_tests_get_rerolls(dice, kind, veteran, expected):
    unit = make_unit(veteran=veteran)
    game = SimpleNamespace(aiControls=lambda unit: True)
    roll = AsyncMock(return_value=[2, 3])
    asyncio.run(reroll_leadership(game, unit, kind, dice, 7, roll))
    assert roll.await_count == int(expected)


def test_veteran_and_other_sources_never_stack():
    game = SimpleNamespace(aiControls=lambda unit: True)
    roll = AsyncMock(return_value=[6, 6])
    assert asyncio.run(reroll_leadership(game, make_unit(), 'Panic', [5, 6], 7, roll,
                                       other_rule='Venerable and Hold Your Ground')) == [6, 6]
    roll.assert_awaited_once()


def test_personal_character_cannot_reroll_via_host(capsys):
    game = SimpleNamespace(aiControls=lambda unit: True)
    host, character = make_unit(), make_unit(1, False)
    host.joinedCharacter = character
    character.hostUnit = host
    roll = AsyncMock(return_value=[1, 1])
    assert asyncio.run(reroll_leadership(game, character, 'Rallying Cry', [5, 5], 7, roll,
                                       personal=True)) == [5, 5]
    roll.assert_not_awaited()
    assert 'cannot borrow' in capsys.readouterr().out


def test_panic_queue_waits_for_veteran_choice():
    unit = make_unit()
    unit.bodyNP = SimpleNamespace(isEmpty=lambda: False)
    queued = []
    game = SimpleNamespace(aiControls=lambda unit: False,
                           makeChoiceNew=AsyncMock(return_value='Re-roll'),
                           taskMgr=SimpleNamespace(add=queued.append))
    psychology = PsychologySystem(game)
    psychology.panic_exempt_reason = lambda unit: None
    psychology.leadership_of = lambda unit: (7, None)
    psychology.venerable_source = lambda unit: None
    psychology.battle_standard_of = lambda unit: None
    done = Mock()
    with patch('psychology.random.randint', side_effect=[6, 6, 1, 2]) as dice:
        psychology._resolve_panic(unit, None, 'test', done)
        assert len(queued) == 1
        done.assert_not_called()
        assert dice.call_count == 2
        asyncio.run(queued.pop())
    done.assert_called_once()
    game.makeChoiceNew.assert_awaited_once()


def test_panic_veteran_and_venerable_do_not_add_third_roll():
    unit = make_unit()
    unit.bodyNP = SimpleNamespace(isEmpty=lambda: False)
    game = SimpleNamespace(aiControls=lambda unit: True,
                           taskMgr=SimpleNamespace(add=lambda coro: asyncio.run(coro)))
    psychology = PsychologySystem(game)
    psychology.panic_exempt_reason = lambda unit: None
    psychology.leadership_of = lambda unit: (7, None)
    psychology.venerable_source = lambda unit: unit
    psychology.battle_standard_of = lambda unit: unit
    psychology._panic_result = Mock()
    with patch('psychology.random.randint', side_effect=[6, 6, 5, 6]) as dice:
        psychology._resolve_panic(unit, None, 'test', Mock())
        assert dice.call_count == 4
    assert psychology._panic_result.call_args.args[-2:] == (False, 11)


@pytest.mark.parametrize('veteran, answer, rallied, roll_count', [
    (True, 'Re-roll', True, 2), (True, 'Keep', False, 1),
    (False, 'Re-roll', False, 1),
])
def test_real_rally_path(veteran, answer, rallied, roll_count, monkeypatch):
    unit = make_unit(veteran=veteran)
    unit.request = Mock()
    unit.spreadToSkirmish = Mock()
    game = SimpleNamespace(
        psychology=SimpleNamespace(leadership_of=lambda unit: (7, None),
                                   battle_standard_of=lambda unit: None),
        rollLeadershipDice=AsyncMock(side_effect=[[6, 6], [2, 3]]),
        aiControls=lambda unit: False, makeChoiceNew=AsyncMock(return_value=answer),
        accept=Mock(), ignore=Mock(), giveSignal=Mock(), freeReformUnit=AsyncMock(),
        setActiveUnit=Mock(), taskLoopStrategy=Mock())
    monkeypatch.setattr(game_module, 'taskMgr',
                        SimpleNamespace(add=lambda function, *args, **kwargs: function()),
                        raising=False)
    asyncio.run(MyApp.rallyUnit(game, unit))
    assert game.rollLeadershipDice.await_count == roll_count
    assert unit.attemptedRallyThisTurn
    if rallied:
        unit.request.assert_called_once_with('Idle')
        unit.spreadToSkirmish.assert_called_once_with()
    else:
        unit.request.assert_not_called()
        unit.spreadToSkirmish.assert_not_called()


def test_real_restraint_path_rerolls_then_holds(monkeypatch):
    unit = make_unit()
    unit.request = Mock()
    resolver = CombatResolver.__new__(CombatResolver)
    resolver.game = SimpleNamespace(
        aiControls=lambda unit: False, psychology=None,
        makeChoiceNew=AsyncMock(side_effect=['Restrain', 'Re-roll']))
    resolver.rollBreakDice = AsyncMock(side_effect=[[6, 6], [2, 3]])
    monkeypatch.setattr(combat_resolution, 'taskMgr',
                        SimpleNamespace(add=lambda coroutine: coroutine), raising=False)
    assert asyncio.run(resolver.restrainChoice(unit, None, 'overrun')) == 'restrain'
    assert resolver.rollBreakDice.await_count == 2
    unit.request.assert_called_once_with('Idle')


@pytest.mark.parametrize('tag', ['mount', 'crew', 'beasts'])
def test_shared_split_profile_counts_once(tag):
    unit = make_unit(3, False)
    part = make_unit().unit.model
    if tag == 'mount':
        unit.unit.model.attach_mount(SimpleNamespace(model=part))
    else:
        unit.unit.model.special_rules.append(
            {'tag': tag, 'partUnit': SimpleNamespace(model=part), 'count': 6})
    assert unit.unit.model.is_veteran()
    assert veteran_counts(unit) == (3, 3)


@pytest.mark.parametrize('tag', ['crew', 'beasts'])
def test_absent_split_parts_cannot_grant_veteran(tag):
    unit = make_unit(3, False)
    unit.unit.model.special_rules.append(
        {'tag': tag, 'partUnit': SimpleNamespace(model=make_unit().unit.model), 'count': 0})
    assert not veteran_available(unit)


def test_break_resolution_does_not_offer_veteran(monkeypatch, capsys):
    unit = make_unit()
    unit.bodyNP = SimpleNamespace(isEmpty=lambda: False)
    unit.isInCombatWith = []
    resolver = CombatResolver.__new__(CombatResolver)
    resolver.game = SimpleNamespace(psychology=None, aiControls=lambda unit: False,
                                    makeChoiceNew=AsyncMock())
    resolver.rollBreakDice = AsyncMock(return_value=[6, 6])
    resolver.isOverwhelmed = Mock(return_value=False)
    resolver.notifyFleesCombat = Mock()
    assert asyncio.run(resolver.breakTestPass([unit], 2)) == [(unit, 'break')]
    resolver.rollBreakDice.assert_awaited_once()
    resolver.game.makeChoiceNew.assert_not_called()
    assert 'a Break test is not a Leadership test' in capsys.readouterr().out