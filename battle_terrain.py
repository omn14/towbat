"""Battle March terrain setup (Rulebook p. 268; Companion pp. 24-25).

Scattering follows Dawn of the Storm Dragon p. 23. Calculations are read-only;
the setup controller commits accepted positions through TerrainManager.
"""

from math import hypot

from shapely.affinity import scale, translate
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import nearest_points, polygonize, unary_union

from battle_config import ConfigError


TOLERANCE = 1e-6


def footprint(piece):
    """Use the rendered natural-feature rim, not its bounding rectangle."""
    edges = piece.sight_edges
    if edges is not None:
        geometry = unary_union(list(polygonize([LineString(edge) for edge in edges])))
        if geometry.is_empty or not geometry.is_valid:
            raise ConfigError(f'{piece.terrain_type}: terrain outline is not a closed valid polygon')
        return geometry
    if piece.terrain_type in ('river', 'pillar_of_fire'):
        raise ConfigError(f'{piece.terrain_type}: setup footprint support is not implemented')
    return box(piece.center.x - piece.width / 2, piece.center.y - piece.height / 2,
               piece.center.x + piece.width / 2, piece.center.y + piece.height / 2)


def placement_report(field, config, geometry, placed, player, *, special=False, scenario=False):
    """Return mandatory violations separately from size recommendations."""
    rules = config['terrain']
    errors, warnings = [], []
    board = Polygon(field.outline)
    if not board.buffer(TOLERANCE).covers(geometry):
        errors.append('Terrain must be wholly on the playable battlefield')
    min_x, min_y, max_x, max_y = geometry.bounds
    span = max(hypot(first[0] - second[0], first[1] - second[1])
               for first in geometry.convex_hull.exterior.coords
               for second in geometry.convex_hull.exterior.coords)
    if span > rules['recommended_max_span'] + TOLERANCE:
        warnings.append(f'Terrain span {span:.2f}" exceeds the recommended {rules["recommended_max_span"]:g}"')
    centre_distance = geometry.distance(Point(0, 0))
    if not scenario:
        if special and centre_distance > rules['centre_clearance'] + TOLERANCE:
            errors.append(f'Special terrain must be within {rules["centre_clearance"]:g}" of centre')
        elif not special and centre_distance < rules['centre_clearance'] - TOLERANCE:
            errors.append(f'Terrain is {centre_distance:.2f}" from centre; requires {rules["centre_clearance"]:g}"')
    for index, (other, owner) in enumerate(placed):
        distance = geometry.distance(other)
        if geometry.intersects(other):
            errors.append(f'Terrain overlaps or touches feature {index + 1}')
        elif (rules['method'] == 'alternating' and owner != player
              and distance < rules['opponent_feature_clearance'] - TOLERANCE):
            errors.append(f'Opponent feature {index + 1} is {distance:.2f}" away; '
                          f'requires {rules["opponent_feature_clearance"]:g}"')
    return {'errors': errors, 'warnings': warnings}


def swept_footprint(geometry, horizontal, vertical):
    """Union boundary-edge sweeps, preserving concavities rather than a convex hull."""
    pieces = [geometry, translate(geometry, horizontal, vertical)]
    polygons = [geometry] if geometry.geom_type == 'Polygon' else list(geometry.geoms)
    for polygon in polygons:
        for ring in [polygon.exterior, *polygon.interiors]:
            points = list(ring.coords)
            for start, end in zip(points, points[1:]):
                quad = Polygon([start, end, (end[0] + horizontal, end[1] + vertical),
                                (start[0] + horizontal, start[1] + vertical)])
                if quad.area > 0:
                    pieces.append(quad)
    return unary_union(pieces)


def scatter_distance(field, geometry, obstacles, direction, distance):
    """Stop at the first touched feature or edge, never jump through one (p. 23)."""
    length = hypot(*direction)
    if distance <= 0 or length == 0:
        return 0.0
    direction = (direction[0] / length, direction[1] / length)
    board = Polygon(field.outline)
    if not board.buffer(TOLERANCE).covers(geometry):
        raise ConfigError('Cannot scatter terrain that starts outside the battlefield')
    if any(geometry.intersects(other) for other in obstacles):
        return 0.0

    def clear(travel):
        swept = swept_footprint(geometry, direction[0] * travel, direction[1] * travel)
        return board.covers(swept) and not any(swept.intersects(other) for other in obstacles)

    if clear(distance):
        return float(distance)
    lower, upper = 0.0, float(distance)
    for iteration in range(40):
        middle = (lower + upper) / 2
        if clear(middle):
            lower = middle
        else:
            upper = middle
    return lower


def objective_clearance_shift(field, geometry, objectives, clearance):
    """Shortest translation clearing all fixed marker bases without leaving the board.

    Reflecting the terrain gives the forbidden translation region around each
    round marker. Its boundary supplies the minimum displacement, including when
    the marker is inside the terrain. Other terrain must be revalidated afterward.
    """
    reflected = scale(geometry, xfact=-1, yfact=-1, origin=(0, 0))
    forbidden = unary_union([
        translate(reflected, *objective['center']).buffer(
            objective['diameter'] / 2 + clearance + TOLERANCE, quad_segs=128)
        for objective in objectives if not objective.get('destroyed')])
    origin = Point(0, 0)
    if not forbidden.contains(origin):
        return (0.0, 0.0)
    min_x, min_y, max_x, max_y = geometry.bounds
    allowed = box(-field.width / 2 - min_x, -field.depth / 2 - min_y,
                  field.width / 2 - max_x, field.depth / 2 - max_y).difference(forbidden)
    if allowed.is_empty:
        raise ConfigError('No on-board terrain position clears the fixed objectives; choose a smaller feature')
    nearest = nearest_points(origin, allowed)[1]
    return (nearest.x, nearest.y)