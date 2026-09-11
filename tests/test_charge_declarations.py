from types import SimpleNamespace
from unittest.mock import Mock

from charge_declarations import begin_declarations, queue_charge


def unit_at(position):
    return SimpleNamespace(bodyNP=Mock(getPos=Mock(return_value=position),
                                       getHpr=Mock(return_value=(0, 0, 0))),
                           chargeAttempts=1, chargeAttemptPending=True)


def test_declaration_reserves_charger_without_moving_or_consuming_attempt():
    charger = unit_at((0, -2, 0))
    defender = unit_at((0, 0, 0))
    game = SimpleNamespace(playerNP=defender.bodyNP, moveArceDistance=10)
    begin_declarations(game)
    entry = queue_charge(game, charger, defender, (0, -12, 0), (0, 0, 0))
    assert game.chargeStage == 'declarations'
    assert game.chargeDeclarations == [entry]
    assert entry.contact_position == (0, -2, 0)
    assert entry.origin == (0, -12, 0)
    charger.bodyNP.setPos.assert_called_once_with(0, -12, 0)
    assert charger.hasMovedThisTurn and not charger.isChargingMove
    assert charger.chargeAttempts == 1 and charger.chargeAttemptPending
    assert queue_charge(game, charger, defender, (0, -12, 0), (0, 0, 0)) is None
    assert len(game.chargeDeclarations) == 1


def test_multiple_declarations_keep_independent_route_snapshots():
    defender = unit_at((0, 0, 0))
    game = SimpleNamespace(playerNP=defender.bodyNP, moveArceDistance=10)
    begin_declarations(game)
    first = queue_charge(game, unit_at((-2, -2, 0)), defender, (-2, -12, 0), (0, 0, 0))
    game.moveArceDistance = 14
    second = queue_charge(game, unit_at((2, -2, 0)), defender, (2, -16, 0), (0, 0, 0))
    assert [entry.distance for entry in game.chargeDeclarations] == [10, 14]
    assert first.defender is second.defender
    begin_declarations(game)
    assert game.chargeDeclarations == []