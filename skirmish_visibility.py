"""Planar per-model charge sight (Rulebook pp. 103, 184, 186; FAQ v1.5.3)."""

from dataclasses import dataclass
from math import atan2, cos, hypot, radians, sin, tau

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


def model_can_see(observer, targets, blockers=()):
    """Sight from a base centre to any exposed target edge, not a centre ray.

    Between successive vertex angles, disjoint opaque rectangles keep their
    front-to-back order. Testing those open angular intervals detects even a
    narrow slit without allowing sight along the seam of touching bases.
    This is an XY base abstraction, not sculpt/eye-height visibility (p. 103).
    """
    if not targets:
        return False
    origin = observer[:2]
    reach = max(hypot(corner[0] - origin[0], corner[1] - origin[1])
                for target in targets for corner in _box_corners(*target))
    blockers = [box for box in blockers
                if hypot(box[0] - origin[0], box[1] - origin[1]) - hypot(box[2], box[3]) <= reach]

    def clear(direction):
        hits = [_ray_entry(origin, direction, target) for target in targets]
        distance = min((hit for hit in hits if hit is not None), default=None)
        if distance is None:
            return False
        return all((hit := _ray_entry(origin, direction, box)) is None or hit > distance + EPSILON
                   for box in blockers)

    for target in targets:
        length = hypot(target[0] - origin[0], target[1] - origin[1])
        if length <= EPSILON:
            return True
        if clear(((target[0] - origin[0]) / length, (target[1] - origin[1]) / length)):
            return True
    angles = sorted({atan2(corner[1] - origin[1], corner[0] - origin[0]) % tau
                     for box in [*targets, *blockers] for corner in _box_corners(*box)})
    for first, last in zip(angles, angles[1:] + [angles[0] + tau]):
        if last - first <= 1e-10:
            continue
        middle = (first + last) / 2
        if clear((cos(middle), sin(middle))):
            return True
    return False


def charge_visibility(game, unit, target, from_pos=None, from_hpr=None):
    """Count all live bases, including attached characters, at declaration.

    Friendly models also block sight; Skirmishers have a 360-degree vision arc
    (pp. 103, 184). Terrain retains the engine's see-onto/not-through convention,
    using conservative rectangles. No state, physics bodies or dice are changed.
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
            terrain_boxes = [(piece.center.x, piece.center.y, piece.width / 2, piece.height / 2, 0)
                             for piece in terrain if piece.blocks_line_of_sight
                             and not piece.contains(Point3(*observer[:2], 0))
                             and not piece.contains(Point3(*destination[:2], 0))]
            if model_can_see(observer, [destination], [*own, *blockers, *terrain_boxes]):
                seen = True
                break
        visible.append(seen)
    return ChargeVisibility(tuple(visible))