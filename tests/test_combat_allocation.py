"""Contact/closest routing and mandatory per-model attack choices (p. 147)."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from combat_allocation import AttackAllocation, nearest_targets
from combat_initiative import InitiativeStep, resolve_steps


def test_contact_targets_override_nearby_targets_and_ties_remain_choices():
    assert nearest_targets([('contact', 0), ('close', .1)]) == ['contact']
    assert nearest_targets([('left', 2), ('right', 2), ('far', 3)]) == ['left', 'right']
    assert nearest_targets([]) == []


def test_only_ambiguous_models_choose_and_all_attacks_are_allocated():
    first = SimpleNamespace(unitName='Left')
    second = SimpleNamespace(unitName='Right')
    owner = SimpleNamespace(unit=SimpleNamespace(name='Knights'))
    game = SimpleNamespace(aiControls=lambda unit: False,
                           makeChoiceNew=AsyncMock(side_effect=['Left', 'Right', None]))
    allocation = AttackAllocation(owner, SimpleNamespace(name='Rider'),
                                  [(0, 2, [first]), (1, 3, [first, second]), (2, 0, [second])])
    asyncio.run(allocation.resolve(game))
    assert allocation.attacks == [(first, 4), (second, 1)]
    assert game.makeChoiceNew.await_count == 3
    assert all(call.kwargs['owner'] is owner for call in game.makeChoiceNew.call_args_list)


def test_ai_chooses_a_legal_target_without_a_prompt():
    target = SimpleNamespace(unitName='Enemy')
    owner = SimpleNamespace(unit=SimpleNamespace(name='Knights'))
    game = SimpleNamespace(aiControls=lambda unit: True, makeChoiceNew=AsyncMock())
    allocation = AttackAllocation(owner, SimpleNamespace(name='Mount'),
                                  [(0, 2, [target, SimpleNamespace(unitName='Other enemy')])])
    asyncio.run(allocation.resolve(game))
    assert allocation.attacks == [(target, 2)]
    game.makeChoiceNew.assert_not_awaited()


def test_equal_initiative_choices_finish_before_either_side_rolls():
    events = []
    game = SimpleNamespace()

    async def choose(label):
        events.append(f'choose {label}')

    def stream(label):
        decision = SimpleNamespace(resolve=lambda game: choose(label))
        def prepare():
            events.append(f'freeze {label}')
            return [decision]
        yield InitiativeStep(5, prepare)
        events.append(f'roll {label}')
        return label

    assert asyncio.run(resolve_steps(game, [stream('first'), stream('second')], None)) == ['first', 'second']
    assert events == ['freeze first', 'freeze second', 'choose first', 'choose second', 'roll first', 'roll second']