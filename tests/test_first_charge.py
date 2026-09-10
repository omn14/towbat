"""First Charge is a game-limited attempt, not the first combat round (p. 169)."""

from types import SimpleNamespace
import asyncio
from unittest.mock import AsyncMock

import pytest

from first_charge import begin_charge_attempt, count_as_charge, expire_first_charge, finish_charge_attempt


def fighter(name='Knights', first_charge=True):
    profile = SimpleNamespace(special_rules=[{'name': 'First Charge'}] if first_charge else [])
    return SimpleNamespace(unit=SimpleNamespace(name=name, model=profile), isDisrupted=False)


def test_success_disrupts_target_and_never_refreshes(capsys):
    charger, target = fighter(), fighter('Warriors', False)
    begin_charge_attempt(charger)
    finish_charge_attempt(charger, target)
    assert charger.chargeAttempts == 1 and not charger.chargeAttemptPending
    assert target.firstChargeDisruptedBy == ['Knights']
    target.isDisrupted = True
    expire_first_charge(target)
    assert target.firstChargeDisruptedBy == [] and target.isDisrupted
    begin_charge_attempt(charger)
    finish_charge_attempt(charger, target)
    assert charger.chargeAttempts == 2 and target.firstChargeDisruptedBy == []
    output = capsys.readouterr().out
    assert 'contacted Warriors' in output and 'other sources unchanged' in output
    assert 'first attempt has already been spent' in output


def test_failed_first_charge_spends_the_benefit(capsys):
    charger, target = fighter(), fighter('Warriors', False)
    begin_charge_attempt(charger)
    finish_charge_attempt(charger)
    begin_charge_attempt(charger)
    finish_charge_attempt(charger, target)
    assert not getattr(target, 'firstChargeDisruptedBy', [])
    assert 'first charge made no contact' in capsys.readouterr().out


def test_redirect_and_repeated_finish_do_not_spend_twice():
    charger, target = fighter(), fighter('Redirected target', False)
    begin_charge_attempt(charger)
    begin_charge_attempt(charger)
    finish_charge_attempt(charger, target)
    finish_charge_attempt(charger, target)
    assert charger.chargeAttempts == 1
    assert target.firstChargeDisruptedBy == ['Knights']


def test_ordinary_charge_does_not_grant_disruption(capsys):
    charger, target = fighter(first_charge=False), fighter('Warriors', False)
    begin_charge_attempt(charger)
    finish_charge_attempt(charger, target)
    assert charger.chargeAttempts == 1
    assert not getattr(target, 'firstChargeDisruptedBy', [])
    assert not capsys.readouterr().out


def test_disruption_removes_rank_points_without_mutating_terrain(capsys):
    from models import model
    from psychology import combat_rank_bonus

    charger = fighter()
    target = SimpleNamespace(unit=SimpleNamespace(name='Warriors', model=model('Chaos Warrior', ''),
                                                 nmodels=15, files=5, ranks=3),
                             isDisrupted=False, isSkirmisher=False)
    assert combat_rank_bonus(target) == 2
    begin_charge_attempt(charger)
    finish_charge_attempt(charger, target)
    assert combat_rank_bonus(target, log=True) == 0
    assert not target.isDisrupted
    assert 'rank bonus +2 -> +0' in capsys.readouterr().out
    expire_first_charge(target)
    assert combat_rank_bonus(target) == 2


@pytest.mark.parametrize('cause', ['terrain', 'flank', 'skirmish', 'ranks'])
def test_already_zero_rank_bonus_reports_no_additional_loss(cause, capsys):
    from models import model
    from psychology import combat_rank_bonus

    target = SimpleNamespace(unit=SimpleNamespace(name='Warriors', model=model('Chaos Warrior', ''),
                                                 nmodels=5 if cause == 'ranks' else 15,
                                                 files=5, ranks=1 if cause == 'ranks' else 3),
                             isDisrupted=cause == 'terrain', isSkirmisher=cause == 'skirmish',
                             firstChargeDisruptedBy=['Knights'])
    if cause == 'flank':
        enemy = SimpleNamespace(unit=SimpleNamespace(model=model('Chaos Warrior', ''), nmodels=5),
                                bodyNP=SimpleNamespace(isEmpty=lambda: False))
        target.isInCombatWith = [enemy]
        target.isInCombatFlank = ['flank']
    capsys.readouterr()
    assert combat_rank_bonus(target) == 0
    assert not capsys.readouterr().out
    assert combat_rank_bonus(target, log=True) == 0
    output = capsys.readouterr().out
    first_charge_lines = [line for line in output.splitlines() if 'First Charge' in line]
    assert len(first_charge_lines) == 1
    assert 'already' in first_charge_lines[0] and '+0 rank bonus' in first_charge_lines[0]


@pytest.mark.parametrize('fleeing', [False, True])
def test_public_charge_wrappers_finalize_failed_attempts(fleeing):
    from combat_resolution import CombatResolver

    charger = fighter()
    charger.state = 'Idle'
    resolver = CombatResolver.__new__(CombatResolver)
    resolver._resolveChargeInterval = AsyncMock()
    resolver._resolveFleeInterval = AsyncMock()
    begin_charge_attempt(charger)
    if fleeing:
        asyncio.run(resolver.fleeInterval(charger, None, 0, None, None))
    else:
        asyncio.run(resolver.chargeInterval(charger, None, 0, None, None, 'front'))
    assert charger.chargeAttempts == 1
    assert not charger.chargeAttemptPending and not charger.firstChargePending


def test_pursuit_without_contact_does_not_spend_a_charge_attempt():
    from combat_resolution import CombatResolver

    charger = fighter()
    charger.state = 'IsPursuing'
    resolver = CombatResolver.__new__(CombatResolver)
    resolver._resolveChargeInterval = AsyncMock()
    asyncio.run(resolver.chargeInterval(charger, None, 0, None, None, 'front'))
    assert getattr(charger, 'chargeAttempts', 0) == 0


@pytest.mark.parametrize('next_turn', [False, True])
def test_pursuit_contact_counts_as_charge_when_combat_is_fought(next_turn):
    charger, target = fighter(), fighter('Warriors', False)
    count_as_charge(charger, target, next_turn=next_turn)
    assert charger.chargeAttempts == 1
    assert bool(getattr(target, 'firstChargeDisruptedBy', [])) != next_turn
    if next_turn:
        assert target.firstChargeDisruptedNextTurnBy == ['Knights']
        expire_first_charge(target)
        assert target.firstChargeDisruptedBy == ['Knights']
    expire_first_charge(target)
    assert not target.firstChargeDisruptedBy
    count_as_charge(charger, target)
    assert charger.chargeAttempts == 2 and not target.firstChargeDisruptedBy