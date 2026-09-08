"""Shared formed-to-loose charge routes (Rulebook pp. 103, 121, 124, 126, 186)."""

from dataclasses import dataclass
import math

from panda3d.core import Point3, Vec2

from psychology import _box_corners, obb_distance
from scouts import BOARD_HALF_DEPTH, BOARD_HALF_WIDTH, model_base_boxes
from skirmish import EPSILON, swept_base_overlaps
from skirmish_charge import plan_skirmish_defence, supported_skirmish_defender
from skirmish_visibility import model_can_see


def rotate(point, pivot, angle):
    cosine, sine = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    horizontal, vertical = point[0] - pivot[0], point[1] - pivot[1]
    return (pivot[0] + horizontal * cosine - vertical * sine,
            pivot[1] + horizontal * sine + vertical * cosine)


def translated(box, direction, distance):
    return (box[0] + direction[0] * distance, box[1] + direction[1] * distance, *box[2:])


def entry_distance(box, target, direction):
    """First touching distance of a translating OBB, by separating axes."""
    near, far = 0.0, float('inf')
    for heading in (box[4], target[4]):
        cosine, sine = math.cos(math.radians(heading)), math.sin(math.radians(heading))
        for axis in ((cosine, sine), (-sine, cosine)):
            source = [corner[0] * axis[0] + corner[1] * axis[1] for corner in _box_corners(*box)]
            dest = [corner[0] * axis[0] + corner[1] * axis[1] for corner in _box_corners(*target)]
            speed = direction[0] * axis[0] + direction[1] * axis[1]
            if abs(speed) < 1e-12:
                if max(source) < min(dest) - 1e-12 or max(dest) < min(source) - 1e-12:
                    return None
                continue
            first, last = sorted(((min(dest) - max(source)) / speed, (max(dest) - min(source)) / speed))
            near, far = max(near, first), min(far, last)
            if near > far + 1e-12:
                return None
    return near


def footprint(boxes):
    heading = boxes[0][4]
    cosine, sine = math.cos(math.radians(heading)), math.sin(math.radians(heading))
    corners = [corner for box in boxes for corner in _box_corners(*box)]
    lateral = [point[0] * cosine + point[1] * sine for point in corners]
    forward = [-point[0] * sine + point[1] * cosine for point in corners]
    center_x, center_y = (min(lateral) + max(lateral)) / 2, (min(forward) + max(forward)) / 2
    return (center_x * cosine - center_y * sine, center_x * sine + center_y * cosine,
            (max(lateral) - min(lateral)) / 2, (max(forward) - min(forward)) / 2, heading)


@dataclass
class ChargeRoute:
    origin: tuple
    heading: float
    pivot: tuple
    lead: float
    wheel: float
    wheel_distance: float
    advance: float
    target_index: int
    original_boxes: list

    @property
    def distance(self):
        return self.lead + self.wheel_distance + self.advance

    def pose(self, distance):
        forward = (-math.sin(math.radians(self.heading)), math.cos(math.radians(self.heading)))
        lead = min(max(0, distance), self.lead)
        origin = (self.origin[0] + forward[0] * lead, self.origin[1] + forward[1] * lead)
        fraction = min(1, max(0, distance - self.lead) / self.wheel_distance) if self.wheel_distance else 1
        angle = self.wheel * fraction
        position = rotate(origin, self.pivot, angle)
        heading = self.heading + angle
        advance = min(self.advance, max(0, distance - self.lead - self.wheel_distance))
        return ((position[0] - math.sin(math.radians(heading)) * advance,
                 position[1] + math.cos(math.radians(heading)) * advance, self.origin[2]), heading)

    @property
    def destination(self):
        return self.pose(self.distance)[0]

    def boxes_at(self, distance):
        position, heading = self.pose(distance)
        return [(*rotate((position[0] + box[0] - self.origin[0],
                          position[1] + box[1] - self.origin[1]), position, heading - self.heading),
                 box[2], box[3], box[4] + heading - self.heading) for box in self.original_boxes]

    @property
    def final_boxes(self):
        return self.boxes_at(self.distance)


def route_to_model(boxes, targets, target_index, origin, obstacles=()):
    """One front-corner wheel and straight approach to the selected model (p. 186)."""
    body = footprint(boxes)
    heading = body[4]
    forward = (-math.sin(math.radians(heading)), math.cos(math.radians(heading)))
    right = (forward[1], -forward[0])
    target = targets[target_index]

    def candidate(angle, lead=0):
        pivot = (body[0] + forward[0] * (body[3] + lead) - math.copysign(body[2], angle or 1) * right[0],
                 body[1] + forward[1] * (body[3] + lead) - math.copysign(body[2], angle or 1) * right[1])
        wheel_cost = abs(math.radians(angle)) * body[2] * 2
        route = ChargeRoute(tuple(origin), heading, pivot, lead, angle, wheel_cost, 0, target_index, boxes)
        turned = route.boxes_at(lead + wheel_cost)
        direction = (-math.sin(math.radians(heading + angle)), math.cos(math.radians(heading + angle)))
        distances = [entry_distance(box, target, direction) for box in turned]
        distances = [distance for distance in distances if distance is not None]
        if not distances:
            return None
        route.advance = min(distances)
        if plan_skirmish_defence(route.final_boxes, [target], float('inf')) is None:
            return None
        return route

    def routes(lead):
        candidates = []
        straight = candidate(0, lead)
        if straight is not None:
            candidates.append(straight)
        for sign in (-1, 1):
            found = False
            for angle in range(1, 91):
                route = candidate(sign * angle, lead)
                if route is None:
                    continue
                candidates.append(route)
                if not found:
                    previous, current = angle - 1.0, float(angle)
                    for iteration in range(20):
                        middle = (previous + current) / 2
                        refined = candidate(sign * middle, lead)
                        if refined is None:
                            previous = middle
                        else:
                            current = middle
                            route = refined
                    candidates.append(route)
                    found = True
        return sorted(candidates, key=lambda item: (item.distance, abs(item.wheel)))

    direct = routes(0)
    for route in direct:
        if path_error(route, targets, obstacles) is None:
            return route
    remaining = max(0, (target[0] - body[0]) * forward[0] +
                    (target[1] - body[1]) * forward[1] - body[3] - math.hypot(target[2], target[3]))
    for step in range(1, min(100, math.ceil(remaining * 2))):
        lead = step / 2
        if any(swept_base_overlaps(body, translated(body, forward, lead), obstacle)
               for obstacle in [*obstacles, *targets]):
            break
        for route in routes(lead):
            if path_error(route, targets, obstacles) is None:
                return route
    return None


def path_error(route, targets, obstacles):
    """Conservative wheel sweeps; exact SAT for the straight segments."""
    stops = [0, route.lead]
    steps = max(1, math.ceil(abs(route.wheel) / 0.5))
    stops.extend(route.lead + route.wheel_distance * index / steps for index in range(1, steps + 1))
    stops.append(route.distance)
    before = footprint(route.boxes_at(0))
    for distance in stops[1:]:
        after = footprint(route.boxes_at(distance))
        if any(abs(corner[0]) > BOARD_HALF_WIDTH + EPSILON or
               abs(corner[1]) > BOARD_HALF_DEPTH + EPSILON for corner in _box_corners(*after)):
            return 'Charge would leave the battlefield'
        change = abs(math.radians(after[4] - before[4]))
        padding = math.hypot(before[2], before[3]) * change
        swept = (*before[:2], before[2] + padding, before[3] + padding, before[4])
        end = (*after[:2], *swept[2:])
        if any(swept_base_overlaps(swept, end, obstacle) for obstacle in obstacles):
            return 'Charge path is blocked'
        if change and any(swept_base_overlaps(swept, end, target) for target in targets):
            return 'Wheel contacts the target before the approach is complete'
        if not change:
            length = math.dist(before[:2], after[:2])
            if length > EPSILON:
                direction = ((after[0] - before[0]) / length, (after[1] - before[1]) / length)
                if any((entry := entry_distance(before, target, direction)) is not None and
                       entry < length - EPSILON for target in targets):
                    return 'Another target model is contacted first'
        before = after
    return None


def route_features(game, route, travel=None):
    pieces = getattr(getattr(game, 'terrain_manager', None), 'terrain_pieces', [])
    travel = route.distance if travel is None else travel
    stops = sorted({0, min(travel, route.lead), travel,
                    *(min(travel, route.lead + route.wheel_distance * index / max(1, math.ceil(abs(route.wheel))))
                      for index in range(max(1, math.ceil(abs(route.wheel))) + 1))})
    found = []
    for piece in pieces:
        obstacle = (piece.center.x, piece.center.y, piece.width / 2, piece.height / 2, 0)
        for start, end in zip(stops, stops[1:]):
            before, after = footprint(route.boxes_at(start)), footprint(route.boxes_at(end))
            padding = math.hypot(before[2], before[3]) * abs(math.radians(after[4] - before[4]))
            padded = (*before[:2], before[2] + padding, before[3] + padding, before[4])
            if swept_base_overlaps(padded, (*after[:2], *padded[2:]), obstacle):
                found.append(piece)
                break
    return found


def route_allowance(game, unit, route):
    profiles = [member.unit.model for member in game.movement.movementParticipants(unit)]
    flying = all(profile.is_flying() for profile in profiles)
    modifier = min([0, *(piece.movement_modifier for piece in route_features(game, route))])
    return min(profile.get_fly_movement(0) if flying else
               profile.get_movement(0) if profile.is_move_through_cover() or modifier == 0 else
               max(1, profile.get_movement(0) + modifier) for profile in profiles)


@dataclass
class FormedChargePreview:
    target: object
    route: ChargeRoute | None
    maximum: float
    error: str | None = None


def starting_boxes(unit, origin=None, facing=None):
    boxes = model_base_boxes(unit)
    if origin is None:
        return boxes
    current = unit.bodyNP.getPos(unit.bodyNP.getTop())
    change = facing[0] - unit.bodyNP.getH() if facing is not None else 0
    return [(*rotate((origin[0] + box[0] - current.x, origin[1] + box[1] - current.y), origin, change),
             box[2], box[3], box[4] + change) for box in boxes]


def preview_charge(game, unit, target, origin=None, facing=None):
    from characters import side_of
    from scouts import scout_charge_blocked
    from special_rules import max_charge_range, unit_has_swiftstride
    from vanguard import in_vanguard, vanguard_charge_blocked
    maximum = max_charge_range(game.movement.movementAllowance(unit), unit_has_swiftstride(unit))
    result = FormedChargePreview(target, None, maximum)
    if target not in game.units or not target.isDeployed or target.unit.nmodels <= 0 or target.bodyNP.isEmpty():
        result.error = 'Charge target is no longer on the battlefield'
        return result
    if not supported_skirmish_defender(unit, target) or side_of(game, unit) == side_of(game, target):
        result.error = 'Unsupported charge pairing'
        return result
    if unit.state != 'Idle' or unit.hasMovedThisTurn or unit.moveSpentThisTurn or unit.cannotChargeThisTurn:
        result.error = 'Cannot charge after moving or while charge-restricted'
        return result
    if in_vanguard(game) or scout_charge_blocked(game, unit) or vanguard_charge_blocked(game, unit):
        result.error = 'Cannot charge after Scouts or Vanguard this turn'
        return result
    boxes = starting_boxes(unit, origin, facing)
    origin = tuple(unit.bodyNP.getPos(game.render) if origin is None else origin)
    targets = model_base_boxes(target)
    others = [box for other in game.units if other not in (unit, target)
              and getattr(other, 'hostUnit', None) is None and other.isDeployed and other.unit.nmodels > 0
              for box in model_base_boxes(other)]
    pieces = getattr(getattr(game, 'terrain_manager', None), 'terrain_pieces', [])
    visible = []
    for index, destination in enumerate(targets):
        for source_index, observer in enumerate(boxes):
            terrain = [(piece.center.x, piece.center.y, piece.width / 2, piece.height / 2, 0)
                       for piece in pieces if piece.blocks_line_of_sight
                       and not piece.contains(Point3(*observer[:2], 0))
                       and not piece.contains(Point3(*destination[:2], 0))]
            if model_can_see(observer, [destination],
                             [*boxes[:source_index], *boxes[source_index + 1:], *others,
                              *targets[:index], *targets[index + 1:], *terrain], facing=observer[4]):
                visible.append(index)
                break
    if not visible:
        result.error = 'No visible Skirmisher in the front arc'
        return result
    nearest = min(visible, key=lambda index: (min(obb_distance(box, targets[index]) for box in boxes), index))
    obstacles = [*others, *((piece.center.x, piece.center.y, piece.width / 2, piece.height / 2, 0)
                            for piece in pieces if piece.is_impassable)]
    result.route = route_to_model(boxes, targets, nearest, origin, obstacles)
    if result.route is None:
        result.error = f'No clear one-wheel route to closest visible model {nearest + 1}'
    else:
        result.maximum = max_charge_range(route_allowance(game, unit, result.route), unit_has_swiftstride(unit))
        if result.route.distance > result.maximum + EPSILON:
            result.error = f'Charge {result.route.distance:.2f}" exceeds maximum {result.maximum:g}"'
        elif plan_skirmish_defence(result.route.final_boxes, targets, game.movement.movementAllowance(target)) is None:
            result.error = 'Contacted defender cannot align within its Movement'
    return result


def plot_charge(game, unit, point):
    from characters import side_of
    from skirmish_movement import MovePreview, draw_preview
    from skirmish_ui import clear_plot_preview, show_plot_status
    previous = getattr(unit, 'formedSkirmishPreview', None)
    unit.formedSkirmishPreview = None
    targets = [other for other in game.units if supported_skirmish_defender(unit, other)
               and other.isDeployed and side_of(game, other) != side_of(game, unit)
               and obb_distance((point.x, point.y, 0, 0, 0), footprint(model_base_boxes(other))) <= EPSILON]
    if not targets:
        if previous is not None:
            clear_plot_preview(game)
        return False
    target = min(targets, key=lambda other: (other.bodyNP.getPos(game.render) - point).length())
    from scouts import scout_charge_blocked
    from special_rules import unit_has_swiftstride
    from vanguard import in_vanguard, vanguard_charge_blocked
    key = (target, tuple((other, other.isDeployed, other.state, getattr(other, 'hostUnit', None),
                          tuple(model_base_boxes(other))) for other in game.units),
           tuple((piece, tuple(piece.center), piece.width, piece.height, piece.movement_modifier,
                  piece.blocks_line_of_sight, piece.is_impassable)
                 for piece in game.terrain_manager.terrain_pieces),
           unit.state, unit.hasMovedThisTurn, unit.cannotChargeThisTurn, unit.moveSpentThisTurn,
           game.movement.movementAllowance(unit), game.movement.movementAllowance(target),
           unit.unit.model.is_move_through_cover(), unit.unit.model.is_flying(), unit_has_swiftstride(unit),
           in_vanguard(game), scout_charge_blocked(game, unit), vanguard_charge_blocked(game, unit))
    cached = getattr(unit, 'formedSkirmishCache', None)
    if cached is not None and cached[0] == key:
        preview = cached[1]
    else:
        preview = preview_charge(game, unit, target)
        unit.formedSkirmishCache = (key, preview)
    unit.formedSkirmishPreview = preview
    if previous is preview and not game.skirmishMoveStatus.isHidden():
        return True
    clear_plot_preview(game)
    game.setGroundOverlay(False)
    route = preview.route
    destination = route.destination if route is not None else tuple(unit.bodyNP.getPos())
    distance = route.distance if route is not None else 0
    game.moveArceDistance = distance
    game.debugTextInfo.setText('')
    game.diceInfoText.setText('')
    game.playerNP.setPos(*destination)
    game.unitHitPos = Point3(*destination)
    game.arcPoint = Vec2(destination[0] / 100 + 0.5, destination[1] / 100 + 0.5)
    game.arcPointRotation = route.wheel if route is not None else 0
    unit.wouldMarch = False
    display = MovePreview([], destination, route.final_boxes if route else [], [distance], 0, False, [],
                          error=preview.error, charge_target=target, charge_maximum=preview.maximum)
    show_plot_status(game, display)
    if route is not None and not preview.error:
        game.skirmishMoveStatus['text'] += (f'\nTarget model {route.target_index + 1}'
                                            f'\nWheel {abs(route.wheel):.1f} deg / {route.wheel_distance:.2f}"')
        game.skirmishMoveStatus.resetFrameSize()
    if not preview.error:
        game.skirmMoveGhost = draw_preview(game.render, display)
        from panda3d.core import LineSegs
        lines = LineSegs('formed-charge-route')
        lines.setThickness(2)
        lines.setColor(0.15, 0.85, 1, 1)
        for index in range(41):
            position, heading = route.pose(route.distance * index / 40)
            if index == 0:
                lines.moveTo(position[0], position[1], 0.3)
            else:
                lines.drawTo(position[0], position[1], 0.3)
        game.skirmMoveGhost.attachNewNode(lines.create())
    return True