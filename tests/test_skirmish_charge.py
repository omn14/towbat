"""Rulebook p. 187 fighting-rank and individual movement regressions."""

import math

import pytest

from psychology import obb_distance
from skirmish import layout_positions
from skirmish_charge import formed_contact, plan_formed_charge, plan_skirmish_charge


def boxes(positions, heading=0):
    return [(horizontal, vertical, 0.5, 0.5, heading) for horizontal, vertical in positions]


def test_two_units_face_each_other_and_share_contact_line():
    attackers = boxes([(-1.6, -6), (0, -6), (1.6, -6), (-0.8, -7.6), (0.8, -7.6)], 90)
    defenders = boxes([(-1.6, 0), (0, 0), (1.6, 0), (-0.8, 1.6), (0.8, 1.6)], 0)
    plan = plan_skirmish_charge(attackers, defenders, 9, 3)
    assert plan is not None
    assert (plan.attacker.heading - plan.defender.heading) % 360 == pytest.approx(180)
    assert abs(plan.attacker.order.index(plan.first_attacker) - (plan.attacker.files - 1) / 2) <= 0.5
    for formation, original, limit in ((plan.attacker, attackers, 9), (plan.defender, defenders, 3)):
        for index, position in zip(formation.order, formation.positions):
            assert math.dist(original[index][:2], position) <= limit + 1e-5
    attacking = [(*position, 0.5, 0.5, plan.attacker.heading) for position in plan.attacker.positions]
    defending = [(*position, 0.5, 0.5, plan.defender.heading) for position in plan.defender.positions]
    assert all(min(obb_distance(box, other) for other in attacking[:plan.attacker.files]) < 1e-5
               for box in defending[:plan.defender.files])


def test_unreachable_models_do_not_teleport_into_fighting_rank():
    attackers = boxes([(0, -5), (0, -6.6), (0, -8.2)])
    defenders = boxes([(0, 0), (1.6, 0), (3.2, 0)])
    plan = plan_skirmish_charge(attackers, defenders, 4, 3)
    assert plan.attacker.files == 1
    assert plan.attacker.lost == [1, 2]
    assert plan_skirmish_charge(attackers, defenders, 3.99, 3) is None


def test_skirmishers_do_not_keep_preset_five_file_frontage():
    attackers = boxes([(index * 1.5, -4) for index in range(7)])
    defenders = boxes([(index * 1.5, 0) for index in range(7)])
    plan = plan_skirmish_charge(attackers, defenders, 9, 4)
    assert plan.attacker.files == 7


@pytest.mark.parametrize('heading', [0, 37, 90, 180, 270])
def test_defenders_use_corner_contact_against_one_charging_file(heading):
    angle = math.radians(heading)

    def rotate(position):
        horizontal, vertical = position
        return (horizontal * math.cos(angle) - vertical * math.sin(angle),
                horizontal * math.sin(angle) + vertical * math.cos(angle))

    attackers = boxes([rotate((0, -4))], heading)
    defenders = boxes([rotate(position) for position in
                       [(0, 0), (-1.2, 0.3), (1.2, 0.3),
                        (0, 1.6), (-1.6, 1.9), (1.6, 1.9)]], heading)
    plan = plan_skirmish_charge(attackers, defenders, 3, 4)
    assert plan is not None
    assert plan.attacker.files == 1
    assert plan.defender.files == 3
    assert plan.defender.lost == []
    assert len(plan.defender.positions) == len(defenders)
    target = (*plan.attacker.positions[0], 0.5, 0.5, plan.attacker.heading)
    assert all(obb_distance((*position, 0.5, 0.5, plan.defender.heading), target) < 1e-5
               for position in plan.defender.positions[:3])
    assert all(math.dist(defenders[index][:2], position) <= 4 + 1e-5
               for index, position in zip(plan.defender.order, plan.defender.positions))


@pytest.mark.parametrize('movement,files', [(0.2, 1), (0.4, 3), ([4, 0.2, 0.4], 2)])
def test_corner_contact_still_requires_each_defenders_movement(movement, files):
    attackers = boxes([(0, -4)])
    defenders = boxes([(0, 0), (-1.2, 0.3), (1.2, 0.3)])
    plan = plan_skirmish_charge(attackers, defenders, 3, movement)
    assert plan.defender.files == files
    limits = movement if isinstance(movement, list) else [movement] * len(defenders)
    assert all(math.dist(defenders[index][:2], position) <= limits[index] + 1e-5
               for index, position in zip(plan.defender.order, plan.defender.positions))


@pytest.mark.parametrize('attacking_count', range(2, 10))
@pytest.mark.parametrize('defending_count', range(2, 10))
@pytest.mark.parametrize('defender_width', [0.8, 1.0, 2.0])
def test_uneven_frontages_keep_defenders_in_contact(attacking_count, defending_count, defender_width):
    attackers = boxes([(horizontal, vertical - 7)
                       for horizontal, vertical in layout_positions(attacking_count, 1, 1)])
    defenders = [(horizontal, vertical, defender_width / 2, 0.5, 0)
                 for horizontal, vertical in layout_positions(defending_count, defender_width, 1)]
    plan = plan_skirmish_charge(attackers, defenders, 9, 3)
    attacking = [(*position, 0.5, 0.5, plan.attacker.heading)
                 for position in plan.attacker.positions[:plan.attacker.files]]
    defending = [(*position, defender_width / 2, 0.5, plan.defender.heading)
                 for position in plan.defender.positions[:plan.defender.files]]
    assert all(min(obb_distance(box, other) for other in attacking) < 1e-5 for box in defending)
    for rank, original, allowance in ((plan.attacker, attackers, 9), (plan.defender, defenders, 3)):
        assert sorted(rank.order + rank.lost) == list(range(len(original)))
        assert all(math.dist(original[index][:2], position) <= allowance + 1e-5
                   for index, position in zip(rank.order, rank.positions))


@pytest.mark.parametrize('heading', [0, 37, 180])
@pytest.mark.parametrize('side', ['front', 'rear', 'left', 'right'])
def test_formed_face_frontage_and_rear_ranks(side, heading):
    radians = math.radians(heading)

    def rotate(position):
        horizontal, vertical = position
        return (horizontal * math.cos(radians) - vertical * math.sin(radians),
                horizontal * math.sin(radians) + vertical * math.cos(radians))

    offsets = [(index * 1.6, 6 if side == 'front' else -6) for index in range(-4, 5)]
    if side in ('left', 'right'):
        offsets = [(6 if side == 'right' else -6, index * 1.6) for index in range(-4, 5)]
    attackers = boxes([rotate(position) for position in offsets], heading + 73)
    defenders = boxes([rotate((column, row)) for row in range(-1, 2)
                       for column in range(-2, 3)], heading)
    before = list(defenders)
    plan = plan_formed_charge(attackers, defenders, 9)
    assert plan is not None and plan.defender is None
    assert plan.flank == (side if side in ('front', 'rear') else 'flank')
    assert plan.first_attacker == (4 if side == 'right' else 3)
    assert plan.attacker.files == {'front': 6, 'rear': 6, 'left': 4, 'right': 5}[side]
    assert plan.attacker.lost == []
    assert len(plan.attacker.positions) == len(attackers)
    assert defenders == before
    for position in plan.attacker.positions[:plan.attacker.files]:
        box = (*position, 0.5, 0.5, plan.attacker.heading)
        assert min(obb_distance(box, target) for target in defenders) < 1e-5
    for index, position in zip(plan.attacker.order, plan.attacker.positions):
        assert math.dist(attackers[index][:2], position) <= 9 + 1e-5
    distance = formed_contact(attackers, defenders)[2]
    assert plan.distance == pytest.approx(distance)
    assert plan_formed_charge(attackers, defenders, distance - 0.01) is None