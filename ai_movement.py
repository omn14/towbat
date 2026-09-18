"""Bounded movement candidates: coarse detours, authoritative formation previews."""

import heapq
import math

from formed_skirmish_charge import ChargeRoute, footprint
from scouts import model_base_boxes


def segment_clear(start, end, obstacles):
    for left, bottom, right, top in obstacles:
        near, far = 0.0, 1.0
        for origin, delta, lower, upper in ((start[0], end[0] - start[0], left, right),
                                            (start[1], end[1] - start[1], bottom, top)):
            if abs(delta) < 1e-9:
                if origin <= lower or origin >= upper:
                    far = -1
                    break
            else:
                entry, exit = sorted(((lower - origin) / delta, (upper - origin) / delta))
                near, far = max(near, entry), min(far, exit)
        if near < far and far > 0 and near < 1:
            return False
    return True


def navigation_field(game, unit, goal, body):
    from battlefield import battlefield_for
    from chariot_terrain import linear_impassable
    from special_rules import unit_is_ethereal
    obstacles = []
    flying = all(member.unit.model.is_flying() for member in game.movement.movementParticipants(unit))
    radius = math.hypot(body[2], body[3]) + 0.15
    if not flying and not unit_is_ethereal(unit):
        for piece in game.terrain_manager.terrain_pieces:
            if piece.is_impassable or linear_impassable(piece, unit):
                obstacles.append((piece.center.x - piece.width / 2 - radius,
                                  piece.center.y - piece.height / 2 - radius,
                                  piece.center.x + piece.width / 2 + radius,
                                  piece.center.y + piece.height / 2 + radius))
    field = battlefield_for(game)
    nodes = [tuple(goal)]
    for left, bottom, right, top in obstacles:
        for point in ((left - .05, bottom - .05), (right + .05, bottom - .05),
                      (left - .05, top + .05), (right + .05, top + .05)):
            if field.contains_box((*point, body[2], body[3], body[4])):
                nodes.append(point)
    distances = [0.0] + [float('inf')] * (len(nodes) - 1)
    queue = [(0.0, 0)]
    while queue:
        distance, index = heapq.heappop(queue)
        if distance != distances[index]:
            continue
        for other, point in enumerate(nodes):
            if index == other or not segment_clear(nodes[index], point, obstacles):
                continue
            candidate = distance + math.dist(nodes[index], point)
            if candidate < distances[other]:
                distances[other] = candidate
                heapq.heappush(queue, (candidate, other))

    def remaining(position, *, allow_escape=False):
        options = [(math.dist(position, point) + distances[index], index)
                   for index, point in enumerate(nodes) if segment_clear(position, point, obstacles)]
        if not options and allow_escape:
            outside = [box for box in obstacles
                       if not (box[0] < position[0] < box[2] and box[1] < position[1] < box[3])]
            options = [(math.dist(position, point) + distances[index], index)
                       for index, point in enumerate(nodes) if index > 0
                       and segment_clear(position, point, outside)]
        if not options:
            return float('inf'), tuple(goal)
        distance, index = min(options)
        return distance, nodes[index]

    return remaining


def routes_towards(game, unit, goal):
    """One wheel/advance or loose translation, within the engine's legal budget."""
    from drilled import march_multiplier
    from reserve_move import in_reserve
    boxes = model_base_boxes(unit)
    if not boxes:
        return []
    body = footprint(boxes)
    origin = tuple(unit.bodyNP.getPos(game.render))
    heading = unit.bodyNP.getH(game.render)
    remaining = navigation_field(game, unit, goal, body)
    initial, waypoint = remaining(origin[:2], allow_escape=True)
    if not math.isfinite(initial):
        return []
    bearing = math.degrees(math.atan2(-(waypoint[0] - origin[0]), waypoint[1] - origin[1]))
    desired = (bearing - heading + 180) % 360 - 180
    allowance = game.movement.movementAllowance(unit, features=[])
    maximum = allowance * (1 if in_reserve(game) else march_multiplier(unit)) - unit.moveSpentThisTurn
    if getattr(unit, 'marchTestResult', None) == 'failed':
        maximum = allowance - unit.moveSpentThisTurn
    loose = getattr(unit, 'isSkirmisher', False)
    angles = (desired, 0, -30, 30, -60, 60, -90, 90)
    results, seen = [], set()

    def consider(destination, final_heading):
        key = (*[round(value, 3) for value in destination], round(final_heading, 3))
        if key in seen:
            return
        seen.add(key)
        preview = game.movement.previewBasicMove(unit, destination, final_heading)
        if preview.error:
            return
        rest, next_waypoint = remaining(destination[:2])
        if not math.isfinite(rest):
            return
        progress = initial - rest
        target_heading = math.degrees(math.atan2(-(next_waypoint[0] - destination[0]),
                                                 next_waypoint[1] - destination[1]))
        old_error = abs((target_heading - heading + 180) % 360 - 180)
        new_error = abs((target_heading - final_heading + 180) % 360 - 180)
        facing_gain = (old_error - new_error) / 90
        if progress < .25 and facing_gain < .25:
            return
        results.append((preview, progress + facing_gain, waypoint))

    for angle in angles:
        if not loose and abs(angle) > 90:
            continue
        turn_cost = 0 if loose else abs(math.radians(angle)) * body[2] * 2
        travel = maximum - turn_cost
        if travel < 0:
            continue
        for advance in (travel, travel * .5, min(travel, math.dist(origin[:2], waypoint)), 0):
            if loose:
                direction = math.radians(heading + angle)
                destination = (origin[0] - math.sin(direction) * advance,
                               origin[1] + math.cos(direction) * advance, origin[2])
                final_heading = heading
            else:
                radians = math.radians(heading)
                forward, right = (-math.sin(radians), math.cos(radians)), (math.cos(radians), math.sin(radians))
                pivot = (body[0] + forward[0] * body[3] - math.copysign(body[2], angle) * right[0],
                         body[1] + forward[1] * body[3] - math.copysign(body[2], angle) * right[1])
                route = ChargeRoute(origin, heading, pivot, 0, angle, turn_cost, advance, 0, boxes)
                destination, final_heading = route.destination, heading + angle
            consider(destination, final_heading)
    if not loose:
        travel = max(0, allowance / 2 - unit.moveSpentThisTurn)
        for angle in (-90, 90, 180):
            radians = math.radians(heading + angle)
            consider((origin[0] - math.sin(radians) * travel,
                      origin[1] + math.cos(radians) * travel, origin[2]), heading)
    return sorted(results, key=lambda result: result[1], reverse=True)[:4]