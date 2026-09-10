"""Counter Charge's reaction gates are independent of charge-roll bonuses."""

from types import SimpleNamespace

import pytest

from counter_charge import counter_charge_distance, has_counter_charge, unavailable_reason
from models import model


def fighter(name):
    return SimpleNamespace(unit=SimpleNamespace(name=name, model=model(name, '')),
                           state='Idle', isInCombat=False)


def reason(defender, charger, *, distance=8, movement=8, flank='front',
           turn=(1, 2), declared=True):
    return unavailable_reason(defender, charger, distance=distance, movement=movement,
                              flank=flank, turn=turn, declared=declared)


def test_real_roster_rule_owners_and_minimum_distance():
    charger = fighter('Silver Helm')
    assert not has_counter_charge(charger)
    for name in ('Chaos Knight', 'Dragon Prince'):
        defender = fighter(name)
        assert has_counter_charge(defender)
        assert reason(defender, charger) is None
        assert 'less than' in reason(defender, charger, distance=7.99)
        assert reason(defender, charger, distance=12) is None


@pytest.mark.parametrize('flank', ['flank', 'rear'])
def test_nonfrontal_charge_cannot_be_countercharged(flank):
    assert 'not the front' in reason(fighter('Chaos Knight'), fighter('Silver Helm'), flank=flank)


def test_engaged_fleeing_and_pursuit_cannot_react():
    defender, charger = fighter('Chaos Knight'), fighter('Silver Helm')
    assert 'no charge reaction' in reason(defender, charger, declared=False)
    defender.state = 'IsFleeing'
    assert 'fleeing' in reason(defender, charger)
    defender.state = 'InCombat'
    assert 'engaged' in reason(defender, charger)
    defender.state = 'Idle'
    defender.isInCombat = True
    assert 'engaged' in reason(defender, charger)


def test_joined_character_must_have_rule_and_does_not_change_charger_type():
    defender, charger = fighter('Chaos Knight'), fighter('Silver Helm')
    defender.joinedCharacter = fighter('Mage')
    assert 'joined Mage' in reason(defender, charger)
    defender.joinedCharacter = fighter('Dragon Prince')
    assert reason(defender, charger) is None
    infantry = fighter('Chaos Warrior')
    infantry.joinedCharacter = charger
    assert 'troop type' in reason(defender, infantry)


def test_once_per_game_turn_not_once_per_own_turn():
    defender, charger = fighter('Chaos Knight'), fighter('Silver Helm')
    defender.counterChargeTurn = [1, 2]
    assert 'already used' in reason(defender, charger)
    assert reason(defender, charger, turn=[2, 2]) is None
    assert reason(defender, charger, turn=[1, 3]) is None


@pytest.mark.parametrize('d6, distance', [(1, 2), (2, 2), (3, 3), (4, 3), (5, 4), (6, 4)])
def test_countercharge_uses_only_d3_plus_one(d6, distance):
    assert counter_charge_distance(d6) == distance