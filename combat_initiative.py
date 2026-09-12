"""Share Initiative boundaries between duel and ordinary attacks (pp. 146, 211)."""

from dataclasses import dataclass
from typing import Callable


@dataclass
class InitiativeStep:
    initiative: int
    prepare: Callable[[], object]


async def resolve_steps(game, streams, challenge):
    """Freeze all equal-Initiative groups before any of them changes casualties."""
    from assailment import cast_at_initiative
    from combat_allocation import AttackAllocation
    from rules_log import log_scope

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
    previous = getattr(game, 'assailmentInitiativeSurvivors', None)
    try:
        while pending:
            initiative = max(event.initiative for event in pending.values())
            selected = [index for index, event in pending.items() if event.initiative == initiative]
            if game is not None:
                candidates = list(getattr(game, 'units', []))
                if challenge is not None:
                    candidates.extend(challenge.participants())
                game.assailmentInitiativeSurvivors = {id(member) for member in candidates if member.unit.nmodels > 0}
            with log_scope(initiative=initiative):
                decisions = [pending[index].prepare() for index in selected]
                for prepared in decisions:
                    for allocation in prepared or ():
                        await allocation.resolve(game)
                for index in selected:
                    advance(index)
                    while index in pending and not isinstance(pending[index], InitiativeStep):
                        if isinstance(pending[index], AttackAllocation):
                            await pending[index].resolve(game)
                            advance(index)
                            continue
                        caster, targets, damage, miscast_damage = pending[index]
                        await cast_at_initiative(game, caster, targets, damage, challenge=challenge,
                                                 miscast_damage=miscast_damage)
                        advance(index)
    finally:
        if game is not None:
            game.assailmentInitiativeSurvivors = previous
    return results