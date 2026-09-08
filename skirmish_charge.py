"""Contact-anchored Skirmisher fighting ranks (Rulebook pp. 186-187)."""

from dataclasses import dataclass
import math

from psychology import _box_corners, obb_distance
from skirmish import EPSILON


@dataclass
class FightingRank:
    positions: list
    order: list
    files: int
    heading: float
    lost: list


@dataclass
class SkirmishCharge:
    attacker: FightingRank
    defender: FightingRank | None
    first_attacker: int
    first_defender: int
    distance: float
    flank: str = 'front'


def _heading(direction):
    return math.degrees(math.atan2(-direction[0], direction[1]))


def _distance(start, end):
    return math.hypot(start[0] - end[0], start[1] - end[1])


def _assign(boxes, slots, limits, fixed):
    assigned = dict(fixed)
    locked = set(fixed)

    def place(slot, visited):
        for model_index in sorted(range(len(boxes)), key=lambda index: _distance(boxes[index], slots[slot])):
            if model_index in visited or _distance(boxes[model_index], slots[slot]) > limits[model_index] + EPSILON:
                continue
            visited.add(model_index)
            occupied = next((key for key, value in assigned.items() if value == model_index), None)
            if occupied is None or (occupied not in locked and place(occupied, visited)):
                assigned[slot] = model_index
                return True
        return False

    for slot in range(len(slots)):
        if slot not in assigned:
            place(slot, set(fixed.values()))
    return assigned


def _rank(boxes, limits, anchor_index, anchor, direction, max_files=None, front_targets=None):
    width, depth = boxes[0][2] * 2, boxes[0][3] * 2
    right = (direction[1], -direction[0])
    count = len(boxes)
    for files in range(min(count, max_files or count), 0, -1):
        candidates = []
        anchor_slots = range(files) if front_targets is not None else [(files - 1) // 2]
        for anchor_slot in anchor_slots:
            slots = []
            for slot in range(count):
                row, column = divmod(slot, files)
                lateral = (column - anchor_slot) * width
                slots.append((anchor[0] + right[0] * lateral - direction[0] * row * depth,
                              anchor[1] + right[1] * lateral - direction[1] * row * depth))
            if front_targets is not None and any(
                    min(obb_distance((*position, width / 2, depth / 2, _heading(direction)), target)
                        for target in front_targets) > EPSILON for position in slots[:files]):
                continue
            assigned = _assign(boxes, slots, limits, {anchor_slot: anchor_index})
            if any(slot not in assigned for slot in range(files)):
                continue
            occupied = 0
            while occupied in assigned:
                occupied += 1
            order = [assigned[slot] for slot in range(occupied)]
            candidates.append(FightingRank(
                slots[:occupied], order, files, _heading(direction),
                [index for index in range(count) if index not in order]))
        if candidates:
            return min(candidates, key=lambda rank: (
                len(rank.lost), sum(_distance(boxes[index], position)
                                    for index, position in zip(rank.order, rank.positions))))
    raise ValueError('The first model cannot form a fighting rank')


def formed_contact(attackers, defenders, origin=None):
    """Select the declaration arc and nearest contact on its base edge (p. 186)."""
    heading = math.radians(defenders[0][4])
    right, forward = (math.cos(heading), math.sin(heading)), (-math.sin(heading), math.cos(heading))

    def project(point, axis):
        return point[0] * axis[0] + point[1] * axis[1]

    corners = [_box_corners(*box) for box in defenders]
    horizontal = [project(corner, right) for points in corners for corner in points]
    vertical = [project(corner, forward) for points in corners for corner in points]
    width, depth = max(horizontal) - min(horizontal), max(vertical) - min(vertical)
    if origin is None:
        origin = tuple(sum(box[axis] for box in attackers) / len(attackers) for axis in (0, 1))
    lateral = project(origin, right) - (min(horizontal) + max(horizontal)) / 2
    longitudinal = project(origin, forward) - (min(vertical) + max(vertical)) / 2
    if abs(longitudinal) * width >= abs(lateral) * depth:
        sign = 1 if longitudinal >= 0 else -1
        outward = (forward[0] * sign, forward[1] * sign)
        flank = 'front' if sign == 1 else 'rear'
    else:
        sign = 1 if lateral >= 0 else -1
        outward = (right[0] * sign, right[1] * sign)
        flank = 'flank'
    direction = (-outward[0], -outward[1])
    tangent = (direction[1], -direction[0])
    edge = max(project(corner, outward) for points in corners for corner in points)
    distances = [min(obb_distance(source, target) for target in defenders) for source in attackers]
    nearest = min(distances)
    first_attacker = next(index for index, distance in enumerate(distances) if distance <= nearest + EPSILON)
    source = attackers[first_attacker]
    candidates = []
    for index, points in enumerate(corners):
        if max(project(corner, outward) for corner in points) < edge - EPSILON:
            continue
        lower = min(project(corner, tangent) for corner in points) - source[2]
        upper = max(project(corner, tangent) for corner in points) + source[2]
        contact = max(lower, min(upper, project(source, tangent)))
        anchor = (tangent[0] * contact + outward[0] * (edge + source[3]),
                  tangent[1] * contact + outward[1] * (edge + source[3]))
        candidates.append((_distance(source, anchor), index, anchor))
    distance, first_defender, anchor = min(candidates)
    return first_attacker, first_defender, distance, anchor, direction, flank


def plan_formed_charge(attackers, defenders, charge_distance, origin=None):
    """Form only the chargers against the stationary charged face (p. 186).

    Every fighting model must touch an enemy base; models unable to reach it
    form behind, with unassigned models lost to coherency (FAQ v1.5.3).
    """
    first_attacker, first_defender, distance, anchor, direction, flank = formed_contact(
        attackers, defenders, origin)
    if distance > charge_distance + EPSILON:
        return None
    attack = _rank(attackers, [charge_distance] * len(attackers), first_attacker,
                   anchor, direction, front_targets=defenders)
    return SkirmishCharge(attack, None, first_attacker, first_defender, distance, flank)


def first_contact(attackers, defenders):
    """Closest starting bases and shortest translation to a target edge (p. 187)."""
    first_attacker, first_defender = min(
        ((attacker, defender) for attacker in range(len(attackers)) for defender in range(len(defenders))),
        key=lambda pair: (obb_distance(attackers[pair[0]], defenders[pair[1]]), pair))
    source, target = attackers[first_attacker], defenders[first_defender]
    corners = _box_corners(*target)
    candidates = []
    for start, end in zip(corners, corners[1:] + corners[:1]):
        edge = (end[0] - start[0], end[1] - start[1])
        length = math.hypot(*edge)
        right = (edge[0] / length, edge[1] / length)
        outward = (right[1], -right[0])
        midpoint = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        if (midpoint[0] - target[0]) * outward[0] + (midpoint[1] - target[1]) * outward[1] < 0:
            outward = (-outward[0], -outward[1])
        lateral = (source[0] - midpoint[0]) * right[0] + (source[1] - midpoint[1]) * right[1]
        lateral = max(-length / 2, min(length / 2, lateral))
        contact = (midpoint[0] + lateral * right[0], midpoint[1] + lateral * right[1])
        anchor = (contact[0] + outward[0] * source[3], contact[1] + outward[1] * source[3])
        candidates.append((_distance(source, anchor), anchor, contact, (-outward[0], -outward[1])))
    distance, anchor, contact, direction = min(candidates, key=lambda candidate: candidate[0])
    return first_attacker, first_defender, distance, anchor, contact, direction


def plan_skirmish_charge(attackers, defenders, charge_distance, defender_movement):
    """First contact, centred charging rank, then defenders within M (p. 187).

    Bases within each unit must share dimensions. Models unable to form any
    contiguous rank are identified as coherency losses (Official FAQ v1.5.3).
    """
    if not attackers or not defenders:
        raise ValueError('Both units need living models')
    for boxes in (attackers, defenders):
        if any(abs(box[2] - boxes[0][2]) > EPSILON or abs(box[3] - boxes[0][3]) > EPSILON
               for box in boxes):
            raise ValueError('Mixed base sizes need individual fighting-rank slots')
    first_attacker, first_defender, distance, anchor, contact, direction = first_contact(attackers, defenders)
    source, target = attackers[first_attacker], defenders[first_defender]
    if distance > charge_distance + EPSILON:
        return None
    attack = _rank(attackers, [charge_distance] * len(attackers), first_attacker, anchor, direction)
    defender_anchor = (contact[0] + direction[0] * target[3], contact[1] + direction[1] * target[3])
    limits = ([defender_movement] * len(defenders) if isinstance(defender_movement, (int, float))
              else list(defender_movement))
    if _distance(target, defender_anchor) > limits[first_defender] + EPSILON:
        raise ValueError('The contacted defender cannot align within its Movement')
    max_files = max(1, math.ceil(attack.files * source[2] / target[2]))
    defend = _rank(defenders, limits, first_defender, defender_anchor,
                   (-direction[0], -direction[1]), max_files)
    return SkirmishCharge(attack, defend, first_attacker, first_defender, distance)


def supported_pair(attacker, defender):
    return (getattr(attacker, 'isSkirmisher', False) and getattr(defender, 'isSkirmisher', False)
            and not attacker.skirmishCombat and not defender.skirmishCombat
            and defender.state != 'IsFleeing' and attacker.state != 'IsPursuing'
            and getattr(attacker, 'joinedCharacter', None) is None
            and getattr(defender, 'joinedCharacter', None) is None)


def supported_formed_target(attacker, defender):
    return (getattr(attacker, 'isSkirmisher', False) and not attacker.skirmishCombat
            and not getattr(defender, 'isSkirmisher', False)
            and defender.state not in ('IsFleeing', 'InCombat') and attacker.state != 'IsPursuing'
            and getattr(attacker, 'joinedCharacter', None) is None
            and getattr(defender, 'joinedCharacter', None) is None)


def apply_fighting_rank(game, unit, formation):
    """Keep planned model identities and contact while resizing the body (pp. 186-187)."""
    from panda3d.core import Point3
    from rules_log import rule_log
    children = list(unit.model.getChildren())
    records = list(unit.skirmishLayout)
    order = formation.order + formation.lost
    for slot, index in enumerate(order):
        children[index].reparentTo(unit.model, slot)
        children[index].setHpr(0, 0, 0)
    unit.skirmishLayout = [records[index] for index in order]
    unit.skirmishCombat = True
    unit.unit.files = formation.files
    if formation.lost:
        game.movement.removeModelsFromUnit(unit, len(formation.lost))
        rule_log('Skirmishers', unit,
                 f'{len(formation.lost)} models cannot form up within movement; '
                 'removed as coherency casualties (p. 184, FAQ v1.5.3)')
    unit.unit.ranks = math.ceil(unit.unit.nmodels / formation.files)
    unit.bodyNP.setHpr(game.render, formation.heading, 0, 0)
    unit.layOutRanks()
    unit.rebuildFootprint()
    first = unit.model.getChild(0)
    current = first.getPos(game.render)
    desired = Point3(*formation.positions[0], current.z)
    unit.bodyNP.setPos(game.render, unit.bodyNP.getPos(game.render) + desired - current)
    unit.bodyNP.node().setTransformDirty()