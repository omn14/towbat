"""Full A only in contact; non-contact fighting/support use ground M (pp. 145-146)."""

import pytest

from combat_contacts import engaged_units, fighting_positions


def formation(files=5, ranks=2):
    return [(column, -row, 0.5, 0.5, 0) for row in range(ranks) for column in range(files)]


def test_wide_rank_full_contact_single_closing_attacks_and_movement_limit():
    positions = fighting_positions(formation(), [(0, 1, 0.5, 0.5, 0)], 5)
    assert [place.attacks(3, 1) for place in positions] == [3, 3, 1, 0, 0, 0, 0, 0, 0, 0]
    assert positions[4].attacks(3, 10) == 1
    assert positions[5].attacks(3, 1, support=True) == 1


def test_split_parts_are_independently_limited_outside_contact():
    positions = fighting_positions(formation(), [(0, 1, 0.5, 0.5, 0)], 5)
    assert positions[0].attacks(2, 4, count=3) == 6
    assert positions[2].attacks(2, 4, count=3) == 3
    assert positions[5].attacks(2, 4) == 0
    assert positions[5].attacks(2, 4, support=True) == 1


def test_press_of_battle_moves_support_to_third_rank():
    positions = fighting_positions(formation(ranks=3), [(2, 1, 2.5, 0.5, 0)], 5, press=True)
    assert [place.attacks(2, 4, support=True) for place in positions] == [2] * 5 + [1] * 10
    assert all(not place.supporting for place in positions[:10])


@pytest.mark.parametrize('facing,enemy', [('left', (-1, -1, .5, 1.5, 0)),
                                         ('right', (5, -1, .5, 1.5, 0)),
                                         ('rear', (2, -3, 2.5, .5, 0))])
def test_flank_and_rear_fighting_never_grant_support(facing, enemy):
    positions = fighting_positions(formation(ranks=3), [enemy], 5, facing=facing, press=True)
    assert any(place.fighting and not place.contact for place in positions)
    assert not any(place.supporting for place in positions)


def test_no_contact_means_no_fighting_rank_even_within_movement():
    positions = fighting_positions(formation(), [(2, 3, 2.5, .5, 0)], 5)
    assert all(place.attacks(3, 10, support=True) == 0 for place in positions)


def test_reserved_character_slot_does_not_shift_ordinary_rank_membership():
    boxes = formation(files=3, ranks=2)
    slots = [0, 2, 3, 4, 5]
    positions = fighting_positions([boxes[slot] for slot in slots], [(0, 1, .5, .5, 0)], 3, slots=slots)
    assert [place.fighting for place in positions] == [True, True, False, False, False]


def test_multiple_combat_follows_long_chains_and_cycles_without_dead_branches():
    from types import SimpleNamespace
    hosts = [SimpleNamespace(unit=SimpleNamespace(nmodels=1), isInCombatWith=[]) for _ in range(7)]
    for first, second in zip(hosts, hosts[1:]):
        first.isInCombatWith.append(second)
        second.isInCombatWith.append(first)
    hosts[4].isInCombatWith.append(hosts[1])
    hosts[5].unit.nmodels = 0
    assert [id(host) for host in engaged_units(hosts[0], hosts[1])] == [id(host) for host in hosts[:5]]