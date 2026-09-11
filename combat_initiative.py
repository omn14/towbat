"""Share Initiative boundaries between duel and ordinary attacks (pp. 146, 211)."""

from dataclasses import dataclass
from typing import Callable


@dataclass
class InitiativeStep:
    initiative: int
    prepare: Callable[[], None]


async def resolve_steps(game, streams, challenge):
    """Freeze all equal-Initiative groups before any of them changes casualties."""
    from assailment import cast_at_initiative

    pending = {}
    results = [None] * len(streams)

    def advance(index):
        try:
            pending[index] = next(streams[index])
        except StopIteration as finished:
            results[index] = finished.value
            pending.pop(index, None)

    for index in range(len(streams)):
        advance(index)
    while pending:
        initiative = max(event.initiative for event in pending.values())
        selected = [index for index, event in pending.items() if event.initiative == initiative]
        for index in selected:
            pending[index].prepare()
        for index in selected:
            advance(index)
            while index in pending and not isinstance(pending[index], InitiativeStep):
                caster, targets, damage, miscast_damage = pending[index]
                await cast_at_initiative(game, caster, targets, damage, challenge=challenge,
                                         miscast_damage=miscast_damage)
                advance(index)
    return results