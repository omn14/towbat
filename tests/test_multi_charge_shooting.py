"""FAQ v1.5.3: one too-close charger prevents every Stand & Shoot reaction."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from charge_declarations import choose_reactions
from models import model


@pytest.mark.parametrize('quick', [False, True])
def test_close_charger_blocks_other_shooting_unless_quick_shot(quick):
    defender_model = model('Elven Archer', '')
    weapon = {'name': 'Test bow', 'quick_shot': quick}
    defender_model.missile_weapon = lambda: weapon
    defender = SimpleNamespace(unit=SimpleNamespace(model=defender_model, nmodels=10, name='Archers'),
                               bodyNP=Mock(isEmpty=Mock(return_value=False)), state='Idle',
                               isInCombat=False)
    entries = [SimpleNamespace(defender=defender,
                               charger=SimpleNamespace(unit=SimpleNamespace(name=name)),
                               origin=(0, distance, 0), facing=(0, 0, 0))
               for name, distance in [('Near cavalry', 2), ('Far cavalry', 12)]]
    game = SimpleNamespace(
        roundCounter=SimpleNamespace(current_player=1, currentRoundPlayer=[1, 1]),
        aiControls=lambda unit: False, makeChoiceNew=AsyncMock(return_value='hold'),
        combat=SimpleNamespace(counterChargeOption=Mock(return_value=None),
                               chargeReactionMeasure=Mock(side_effect=[(2, 8), (12, 8)]),
                               standAndShootOption=Mock(return_value=object()),
                               fireAndFleeOption=Mock(return_value=False)))
    asyncio.run(choose_reactions(game, entries))
    options = game.makeChoiceNew.call_args.args[0]
    assert any(option.startswith('stand & shoot') for option in options) is quick
    if not quick:
        game.combat.standAndShootOption.assert_not_called()