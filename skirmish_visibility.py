"""Planar per-model charge sight (Rulebook pp. 103, 184, 186; FAQ v1.5.3)."""

from dataclasses import dataclass
from math import asin, atan2, cos, hypot, radians, sin, sqrt, tau

from panda3d.core import Point3

from psychology import _box_corners
from scouts import model_base_boxes
from skirmish import EPSILON


@dataclass(frozen=True)
class ChargeVisibility:
    models: tuple[bool, ...]

    @property
    def visible(self):
        return sum(self.models)

    @property
    def total(self):
        return len(self.models)

    @property
    def allowed(self):
        return self.visible * 2 > self.total

    def detail(self, target):
        return (f'{self.visible}/{self.total} models can see {target}; '
                f'need {self.total // 2 + 1}/{self.total} (more than 50%)')


def _ray_entry(origin, direction, box):
    angle = radians(box[4])
    cosine, sine = cos(angle), sin(angle)
    offset_x, offset_y = origin[0] - box[0], origin[1] - box[1]
    local_origin = (offset_x * cosine + offset_y * sine, -offset_x * sine + offset_y * cosine)
    local_direction = (direction[0] * cosine + direction[1] * sine,
                       -direction[0] * sine + direction[1] * cosine)
    near, far = 0.0, float('inf')
    for position, component, extent in zip(local_origin, local_direction, box[2:4]):
        if abs(component) < 1e-12:
            if abs(position) > extent:
                return None
            continue
        first, last = sorted(((-extent - position) / component, (extent - position) / component))
        near, far = max(near, first), min(far, last)
        if far < near:
            return None
    return near


def _ray_segment_entry(origin, direction, edge):
    first, last = edge
    segment = (last[0] - first[0], last[1] - first[1])
    offset = (first[0] - origin[0], first[1] - origin[1])
    denominator = direction[0] * segment[1] - direction[1] * segment[0]
    offset_cross = offset[0] * direction[1] - offset[1] * direction[0]
    if abs(denominator) < 1e-12:
        if abs(offset_cross) > 1e-12:
            return None
        distances = [(point[0] - origin[0]) * direction[0]
                     + (point[1] - origin[1]) * direction[1] for point in edge]
        return max(0.0, min(distances)) if max(distances) >= 0 else None
    distance = (offset[0] * segment[1] - offset[1] * segment[0]) / denominator
    fraction = offset_cross / denominator
    return distance if distance >= 0 and 0 <= fraction <= 1 else None


def _ray_circle_entry(origin, direction, center, radius):
    offset = (origin[0] - center[0], origin[1] - center[1])
    projection = offset[0] * direction[0] + offset[1] * direction[1]
    discriminant = projection ** 2 - (offset[0] ** 2 + offset[1] ** 2 - radius ** 2)
    if discriminant < 0:
        return None
    near, far = -projection - sqrt(discriminant), -projection + sqrt(discriminant)
    return max(0, near) if far >= 0 else None


def model_can_see(observer, targets, blockers=(), facing=None, *, terrain=()):
    return _model_sight(observer, targets, blockers, facing, terrain=terrain)


def model_visible_fraction(observer, target, blockers=(), *, terrain=()):
    """Exposed angular fraction of one XY base silhouette for cover (p. 139)."""
    return _model_sight(observer, [target], blockers, terrain=terrain, measure=True)


def _model_sight(observer, targets, blockers=(), facing=None, *, terrain=(), measure=False):
    """Sight from a base centre to any exposed target edge, not a centre ray.

    Between successive vertex angles, disjoint opaque footprints keep their
    front-to-back order. Testing those open angular intervals detects even a
    narrow slit without allowing sight along the seam of touching bases.
    Terrain uses its rim, with rectangular fallback, and the engine's
    see-onto/not-through convention (p. 271). This is an XY base abstraction,
    not sculpt/eye-height visibility (p. 103).
    """
    if not targets:
        return 0.0 if measure else False
    if terrain and len(targets) > 1:
        return any(model_can_see(observer, [target], blockers, facing, terrain=terrain)
                   for target in targets)
    origin = observer[:2]
    reach = max(hypot(corner[0] - origin[0], corner[1] - origin[1])
                for target in targets for corner in _box_corners(*target))
    blockers = list(blockers)
    edges = []
    circles = []
    for piece in terrain:
        if getattr(piece, 'terrain_type', None) == 'landmark':
            circles.append(((piece.center.x, piece.center.y), piece.width / 2))
            continue
        if (not piece.blocks_line_of_sight
                or piece.contains(Point3(*origin, 0))
                or piece.contains(Point3(*targets[0][:2], 0))
                or hypot(piece.center.x - origin[0], piece.center.y - origin[1])
                - hypot(piece.width / 2, piece.height / 2) > reach):
            continue
        outline = getattr(piece, 'sight_edges', None)
        if outline is None:
            blockers.append((piece.center.x, piece.center.y, piece.width / 2, piece.height / 2, 0))
        else:
            edges.extend(outline)
    blockers = [box for box in blockers
                if hypot(box[0] - origin[0], box[1] - origin[1]) - hypot(box[2], box[3]) <= reach]

    def clear(direction):
        if facing is not None:
            angle = radians(facing)
            forward = -sin(angle) * direction[0] + cos(angle) * direction[1]
            lateral = cos(angle) * direction[0] + sin(angle) * direction[1]
            if forward < abs(lateral) - 1e-12:
                return False
        hits = [_ray_entry(origin, direction, target) for target in targets]
        distance = min((hit for hit in hits if hit is not None), default=None)
        if distance is None:
            return False
        return (all((hit := _ray_entry(origin, direction, box)) is None or hit > distance + EPSILON
                for box in blockers)
            and all((hit := _ray_segment_entry(origin, direction, edge)) is None
                or hit > distance + EPSILON for edge in edges)
            and all((hit := _ray_circle_entry(origin, direction, center, radius)) is None
                or hit > distance + EPSILON for center, radius in circles))

    for target in targets:
        length = hypot(target[0] - origin[0], target[1] - origin[1])
        if length <= EPSILON:
            return 1.0 if measure else True
        if not measure and clear(((target[0] - origin[0]) / length, (target[1] - origin[1]) / length)):
            return True
    corners = [corner for box in [*targets, *blockers] for corner in _box_corners(*box)]
    corners.extend(point for edge in edges for point in edge)
    angles = sorted({atan2(corner[1] - origin[1], corner[0] - origin[0]) % tau for corner in corners})
    for center, radius in circles:
        separation = hypot(center[0] - origin[0], center[1] - origin[1])
        if separation <= radius:
            return 0.0 if measure else False
        direction = atan2(center[1] - origin[1], center[0] - origin[0])
        half_angle = asin(radius / separation)
        angles = sorted({*angles, (direction - half_angle) % tau, (direction + half_angle) % tau})
    if facing is not None:
        angles = sorted({*angles, radians(facing + 45) % tau, radians(facing + 135) % tau})
    total_angle = visible_angle = 0.0
    for first, last in zip(angles, angles[1:] + [angles[0] + tau]):
        if last - first <= 1e-10:
            continue
        middle = (first + last) / 2
        direction = (cos(middle), sin(middle))
        if measure:
            if any(_ray_entry(origin, direction, target) is not None for target in targets):
                total_angle += last - first
                if clear(direction):
                    visible_angle += last - first
        elif clear(direction):
            return True
    return visible_angle / total_angle if measure and total_angle else 0.0 if measure else False


def charge_visibility(game, unit, target, from_pos=None, from_hpr=None):
    """Count all live bases, including attached characters, at declaration.

    Friendly models also block sight; Skirmishers have a 360-degree vision arc
    (pp. 103, 184). Terrain retains the engine's see-onto/not-through convention,
    using the shaped rim. No state, physics bodies or dice are changed.
    """
    observers = model_base_boxes(unit)
    if from_pos is not None:
        root = unit.bodyNP.getTop()
        current = unit.bodyNP.getPos(root)
        original = root.getRelativePoint(unit.bodyNP.getParent(), Point3(*from_pos))
        change = 0 if from_hpr is None else from_hpr[0] - unit.bodyNP.getH()
        cosine, sine = cos(radians(change)), sin(radians(change))
        observers = [(original.x + (box[0] - current.x) * cosine - (box[1] - current.y) * sine,
                      original.y + (box[0] - current.x) * sine + (box[1] - current.y) * cosine,
                      box[2], box[3], box[4] + change) for box in observers]
    targets = model_base_boxes(target)
    blockers = [box for other in game.units
                if other is not unit and other is not target
                and getattr(other, 'hostUnit', None) is None
                and other.isDeployed and other.unit.nmodels > 0
                for box in model_base_boxes(other)]
    terrain = getattr(getattr(game, 'terrain_manager', None), 'terrain_pieces', [])
    visible = []
    for index, observer in enumerate(observers):
        own = observers[:index] + observers[index + 1:]
        seen = False
        for destination in targets:
            if model_can_see(observer, [destination], [*own, *blockers], terrain=terrain):
                seen = True
                break
        visible.append(seen)
    return ChargeVisibility(tuple(visible))