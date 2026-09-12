"""Challenge and ordinary attacks share an Initiative clock (pp. 146, 211)."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from combat_initiative import InitiativeStep, resolve_steps


def test_groups_interleave_and_snapshot_equal_initiative_before_casualties():
    remaining = {'duel': 1, 'ordinary': 2}
    attacks = []

    def group(name, initiatives, victim):
        frozen = {}
        def prepare():
            frozen['models'] = remaining[name]
        for initiative in initiatives:
            yield InitiativeStep(initiative, prepare)
            attacks.append((name, initiative, frozen['models']))
            if initiative == 5:
                remaining[victim] = 0
        return name

    results = asyncio.run(resolve_steps(None, [group('duel', [9, 5, 1], 'ordinary'),
                                               group('ordinary', [8, 5, 2], 'duel')], None))
    assert results == ['duel', 'ordinary']
    assert attacks == [('duel', 9, 1), ('ordinary', 8, 2), ('duel', 5, 1),
                       ('ordinary', 5, 2), ('ordinary', 2, 0), ('duel', 1, 0)]


def test_spell_window_finishes_before_lower_initiative_prepares():
    events = []
    def caster():
        yield InitiativeStep(7, lambda: events.append('wizard snapshot'))
        yield 'wizard', ['enemy'], None, None
        events.append('wizard attacks')
        return (1, 0)
    def ordinary():
        yield InitiativeStep(3, lambda: events.append('lower snapshot'))
        events.append('lower attacks')
        return (0, 1)
    with patch('assailment.cast_at_initiative', AsyncMock(side_effect=lambda *args, **kwargs:
                                                        events.append('cast'))) as cast:
        assert asyncio.run(resolve_steps(None, [caster(), ordinary()], None)) == [(1, 0), (0, 1)]
    cast.assert_awaited_once()
    assert events == ['wizard snapshot', 'cast', 'wizard attacks', 'lower snapshot', 'lower attacks']


def test_caster_snapshot_is_restored_when_resolution_is_interrupted():
    survivor = SimpleNamespace(unit=SimpleNamespace(nmodels=1))
    previous = {42}
    game = SimpleNamespace(units=[survivor], assailmentInitiativeSurvivors=previous)

    def interrupted():
        yield InitiativeStep(5, lambda: None)
        assert game.assailmentInitiativeSurvivors == {id(survivor)}
        raise RuntimeError('interrupted combat')

    with pytest.raises(RuntimeError, match='interrupted combat'):
        asyncio.run(resolve_steps(game, [interrupted()], None))
    assert game.assailmentInitiativeSurvivors is previous


def test_log_initiative_tracks_interleaved_streams_and_preparation():
    from rules_log import BattleJournal, log_scope
    journal = BattleJournal()

    def stream(initiatives):
        for initiative in initiatives:
            yield InitiativeStep(initiative, lambda: journal.append('prepare') and None)
            journal.append('attack')

    with log_scope(combat='Princes vs Knights'):
        asyncio.run(resolve_steps(None, [stream([9, 5]), stream([7, 5])], None))
    assert [entry.context['initiative'] for entry in journal.entries] == [9, 9, 7, 7, 5, 5, 5, 5]
    journal.append('after')
    assert 'combat' not in journal.entries[-1].context