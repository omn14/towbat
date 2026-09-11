"""Both players receive every configured turn before the battle ends."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from ClassRoundCounter import RoundCounter


@pytest.mark.parametrize('rounds', [1, 6])
def test_last_round_includes_second_players_turn(rounds):
    game = SimpleNamespace(player1Units=[], player2Units=[])
    with patch('ClassRoundCounter.messenger', Mock(), create=True):
        counter = RoundCounter(game, rounds)
        for number in range(rounds):
            assert counter.current_player == 1
            counter.next_turn()
            assert counter.current_player == 2
            assert counter.currentRoundPlayer == [number + 1, number]
            counter.next_turn()
        assert counter.currentRoundPlayer == [rounds, rounds]
        counter.next_turn()
        assert counter.currentRoundPlayer == [rounds, rounds]