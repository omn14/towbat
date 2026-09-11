"""Character-only Gaze table, caps, expiry and contagious Stupidity (RH p. 116)."""

from types import SimpleNamespace
import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from chaos_gifts import apply_gift, expire_gifts, has_stupidity, succumbed
from chaos_gifts import begin_turn, start_and_command


def champion():
    profile = SimpleNamespace(characteristics=dict(I=4, T=4, WS=5, A=3, S=4, Ld=8),
                              special_rules=[], is_veteran=lambda: False)
    return SimpleNamespace(unit=SimpleNamespace(model=profile, name='Champion'), gazeState={})


@pytest.mark.parametrize('roll,stats', [(2, ['I']), (3, ['T']), (4, ['WS']), (5, ['A']), (6, ['S', 'Ld'])])
def test_gift_changes_only_named_stats_and_expires_only_temporary(roll, stats):
    member = champion()
    before = dict(member.unit.model.characteristics)
    apply_gift(member, roll)
    assert member.unit.model.characteristics == {key: value + (key in stats) for key, value in before.items()}
    expire_gifts(member)
    assert member.unit.model.characteristics == {key: value + (key in stats and roll >= 4) for key, value in before.items()}


def test_first_damnation_grants_stupidity_then_reduces_leadership_to_two():
    member = champion()
    apply_gift(member, 1)
    assert has_stupidity(member) and member.unit.model.characteristics['Ld'] == 8
    for _ in range(10):
        apply_gift(member, 1)
    assert member.unit.model.characteristics['Ld'] == 2


def test_caps_do_not_lose_stats_at_expiry():
    member = champion()
    member.unit.model.characteristics['I'] = 10
    apply_gift(member, 2)
    expire_gifts(member)
    assert member.unit.model.characteristics['I'] == 10


def test_stupidity_follows_joined_character_and_leaves_with_them():
    host, member = champion(), champion()
    apply_gift(member, 1)
    host.joinedCharacter = member
    host.stupidityFailed = True
    member.hostUnit = host
    assert succumbed(host) and succumbed(member)
    host.joinedCharacter = None
    assert not succumbed(host)


def turn_case():
    member = champion()
    member.unit.nmodels = 1
    member.bodyNP = Mock(isEmpty=Mock(return_value=False))
    member.state = 'Idle'
    member.unit.model.special_rules = [{'name': 'Gaze of the Gods'}]
    game = SimpleNamespace(units=[member], player1Units=[member], player2Units=[],
                           roundCounter=SimpleNamespace(current_player=1, currentRoundPlayer=[1, 1]),
                           aiControls=lambda unit: False, makeChoiceNew=AsyncMock(return_value='Roll'),
                           psychology=SimpleNamespace(leadership_of=lambda unit: (8, None)))
    return game, member


def test_turn_gaze_is_optional_owner_selected_and_not_repeated():
    game, member = turn_case()
    with patch('chaos_gifts.random.randint', return_value=5) as roll:
        asyncio.run(start_and_command(game))
        asyncio.run(start_and_command(game))
    assert member.unit.model.characteristics['A'] == 4
    roll.assert_called_once()
    assert game.makeChoiceNew.call_args.kwargs['owner'] is member


def test_begin_turn_locks_input_and_unlocks_after_choice():
    game, member = turn_case()
    pending = []
    game.taskMgr = SimpleNamespace(add=lambda coroutine, name: pending.append(coroutine))
    begin_turn(game)
    begin_turn(game)
    assert len(pending) == 1 and game.magicBusy and game.chaosCommandBusy
    with patch('chaos_gifts.random.randint', return_value=2):
        asyncio.run(pending[0])
    assert not game.magicBusy and not game.chaosCommandBusy
    assert member.unit.model.characteristics['I'] == 5


def test_failed_stupidity_blocks_spell_selection_and_wizardly_dispel():
    from game import MyApp
    from dispelling import wizard_reason
    game, member = turn_case()
    apply_gift(member, 1)
    member.stupidityFailed = True
    assert MyApp.castableSpells(game, member) == []
    assert 'Stupidity' in wizard_reason(game, member, None)


def test_failed_stupidity_prevents_assailment_without_spending_attempt():
    from assailment import cast_at_initiative
    game, member = turn_case()
    apply_gift(member, 1)
    member.stupidityFailed = True
    asyncio.run(cast_at_initiative(game, member, [object()], Mock()))
    assert not hasattr(member, 'spellsCastThisTurn')


def test_failed_stupidity_blocks_movement_shooting_reserve_and_declaration():
    from game import MyApp
    from movement_system import MovementSystem
    from charge_declarations import queue_charge
    from reserve_move import unavailable
    game, member = turn_case()
    apply_gift(member, 1)
    member.stupidityFailed = True
    assert MovementSystem.moveUnit(SimpleNamespace(game=game), member) is False
    assert MovementSystem.redressRanks(SimpleNamespace(game=game), member, 1) is False
    assert queue_charge(game, member, None, None, None) is None
    assert 'Stupidity' in unavailable(game, member)
    asyncio.run(MyApp._shootAt(game, member, None))
    completed = Mock()
    MyApp.startFreeReform(game, member, on_done=completed)
    completed.assert_called_once()


def test_failed_stupidity_blocks_post_combat_movement():
    from combat_resolution import CombatResolver
    game, member = turn_case()
    apply_gift(member, 1)
    member.stupidityFailed = True
    combat = SimpleNamespace(game=game)
    asyncio.run(CombatResolver.overrunMove(combat, member))
    asyncio.run(CombatResolver.pursuitMove(combat, member, None, None))
    asyncio.run(CombatResolver.giveGroundMove(combat, member, []))


def test_failed_stupidity_charge_reaction_holds_without_offering_choice():
    from charge_declarations import choose_reactions
    game, member = turn_case()
    apply_gift(member, 1)
    member.stupidityFailed = True
    entry = SimpleNamespace(charger=object(), defender=member)
    asyncio.run(choose_reactions(game, [entry]))
    assert entry.reaction == 'hold'
    game.makeChoiceNew.assert_not_awaited()


@pytest.mark.parametrize('state,engaged', [('IsFleeing', False), ('InCombat', True)])
def test_fleeing_or_engaged_skips_stupidity_and_clears_previous_failure(state, engaged):
    game, member = turn_case()
    apply_gift(member, 1)
    member.stupidityFailed = True
    member.state, member.isInCombat = state, engaged
    game.makeChoiceNew.return_value = 'Decline'
    with patch('chaos_gifts.random.randint') as roll:
        asyncio.run(start_and_command(game))
    roll.assert_not_called()
    assert not succumbed(member)


def test_new_stupidity_waits_until_next_own_turn_and_failed_test_persists():
    game, member = turn_case()
    with patch('chaos_gifts.random.randint', return_value=1) as roll:
        asyncio.run(start_and_command(game))
    roll.assert_called_once()
    assert has_stupidity(member) and not succumbed(member)
    game.roundCounter.currentRoundPlayer[0] += 1
    game.makeChoiceNew.return_value = 'Decline'
    with patch('chaos_gifts.random.randint', return_value=6) as roll:
        asyncio.run(start_and_command(game))
    assert roll.call_count == 2 and succumbed(member)
    game.roundCounter.current_player = 2
    asyncio.run(start_and_command(game))
    assert succumbed(member)