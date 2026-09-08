"""Base geometry for Skirmish formation (Rulebook pp. 184-185, FAQ v1.5.3)."""

import math

from psychology import _box_corners, obb_distance


COHERENCY = 1.0
EPSILON = 1e-5


def layout_positions(count, width, depth, columns=None, gap=0.6):
    """A reproducible connected layout, centred on its base extents (p. 184)."""
    if count <= 0:
        return []
    if width <= 0 or depth <= 0 or not EPSILON < gap <= COHERENCY:
        raise ValueError('Base dimensions must be positive and gap must be in (0, 1].')
    columns = max(1, min(count, int(columns or math.ceil(math.sqrt(count)))))
    positions = [((index % columns) * (width + gap),
                  -(index // columns) * (depth + gap)) for index in range(count)]
    centre_x = (min(point[0] for point in positions) + max(point[0] for point in positions)) / 2
    centre_y = (min(point[1] for point in positions) + max(point[1] for point in positions)) / 2
    return [(point[0] - centre_x, point[1] - centre_y) for point in positions]


def coherent_groups(boxes):
    """Connected components, not merely a neighbour for every model (p. 184)."""
    neighbours = [set() for _ in boxes]
    for first, box in enumerate(boxes):
        for second in range(first + 1, len(boxes)):
            if obb_distance(box, boxes[second]) <= COHERENCY + EPSILON:
                neighbours[first].add(second)
                neighbours[second].add(first)
    remaining = set(range(len(boxes)))
    groups = []
    while remaining:
        pending = [min(remaining)]
        group = set()
        while pending:
            current = pending.pop()
            if current not in remaining:
                continue
            remaining.remove(current)
            group.add(current)
            pending.extend(neighbours[current] & remaining)
        groups.append(group)
    return groups


def separated_positions(boxes):
    """Open axis-aligned compact bases without rebuilding their ranks (p. 185).

    A 0.0001-inch gap stands in for the smallest separation, safely above the
    float32 contact tolerance. Minimum uniform expansion about the base-extents
    centre retains the existing layout rather than granting a free reform.
    """
    if not boxes:
        return []
    error = coherency_error(boxes, loose=False)
    if error is not None:
        raise ValueError(error)
    expansion = 1.0
    gap = EPSILON * 10
    for first, box in enumerate(boxes):
        for other in boxes[first + 1:]:
            delta_x, delta_y = abs(box[0] - other[0]), abs(box[1] - other[1])
            width, depth = box[2] + other[2], box[3] + other[3]
            if delta_x < width - EPSILON and delta_y < depth - EPSILON:
                raise ValueError('Existing compact bases overlap')
            if obb_distance(box, other) > EPSILON:
                continue
            factors = [extent / delta for extent, delta in
                       ((width + gap, delta_x), (depth + gap, delta_y)) if delta > EPSILON]
            expansion = max(expansion, min(factors))
    if expansion == 1.0:
        return [box[:2] for box in boxes]
    centre_x = (min(box[0] - box[2] for box in boxes) + max(box[0] + box[2] for box in boxes)) / 2
    centre_y = (min(box[1] - box[3] for box in boxes) + max(box[1] + box[3] for box in boxes)) / 2
    positions = [(centre_x + (box[0] - centre_x) * expansion,
                  centre_y + (box[1] - centre_y) * expansion) for box in boxes]
    error = coherency_error([(*position, *box[2:]) for position, box in zip(positions, boxes)])
    if error is not None:
        raise ValueError(error)
    return positions


def coherency_error(boxes, *, loose=True):
    """One survivor is coherent; loose bases must not touch (Rulebook p. 184)."""
    if any(not all(math.isfinite(value) for value in box) or
           box[2] <= 0 or box[3] <= 0 for box in boxes):
        return 'Invalid base dimensions or position'
    if loose:
        for first, box in enumerate(boxes):
            for second in range(first + 1, len(boxes)):
                if obb_distance(box, boxes[second]) <= EPSILON:
                    return f'Models {first + 1} and {second + 1} touch or overlap'
    groups = coherent_groups(boxes)
    if len(groups) > 1:
        return f'{len(groups)} separate groups; every model must connect within 1 inch'
    return None


def casualty_indices(boxes, count, *, protected=()):
    """Remove ordinary casualties without breaking the surviving group (p. 184)."""
    protected = set(protected)
    remaining = list(range(len(boxes)))
    removed = []
    for _ in range(min(max(0, count), len(boxes) - len(protected))):
        for candidate in reversed(remaining):
            if candidate in protected:
                continue
            survivors = [index for index in remaining if index != candidate]
            if len(coherent_groups([boxes[index] for index in survivors])) <= 1:
                removed.append(candidate)
                remaining.remove(candidate)
                break
        else:
            raise ValueError('No ordinary casualty can preserve coherency')
    return removed


def targeted_replacement(boxes, victim, eligible):
    """Replace a targeted casualty with a removable ordinary model (FAQ v1.5.3)."""
    if len(coherent_groups([box for index, box in enumerate(boxes) if index != victim])) <= 1:
        return None
    for candidate in reversed(list(eligible)):
        if candidate == victim:
            continue
        replacement = (*boxes[victim][:2], *boxes[candidate][2:])
        survivors = [replacement if index == victim else box
                     for index, box in enumerate(boxes) if index != candidate]
        if coherency_error(survivors) is None:
            return candidate
    raise ValueError('No replacement can preserve coherency')


def swept_base_overlaps(start, end, obstacle):
    """SAT for a translated base's swept convex polygon against an OBB."""
    start_corners = _box_corners(*start)
    end_corners = _box_corners(*end)
    obstacle_corners = _box_corners(*obstacle)
    axes = [(start[1] - end[1], end[0] - start[0])]
    for polygon in (start_corners, obstacle_corners):
        for index, point in enumerate(polygon):
            following = polygon[(index + 1) % len(polygon)]
            axes.append((point[1] - following[1], following[0] - point[0]))
    for axis_x, axis_y in axes:
        if abs(axis_x) + abs(axis_y) < EPSILON:
            continue
        moving = [point[0] * axis_x + point[1] * axis_y
                  for point in start_corners + end_corners]
        fixed = [point[0] * axis_x + point[1] * axis_y for point in obstacle_corners]
        if max(moving) < min(fixed) - EPSILON or max(fixed) < min(moving) - EPSILON:
            return False
    return True