"""Playable bounds and deployment geometry (General's Companion pp. 24, 26-27).

All coordinates are inches about the board centre. Side 1 is the lower/right
zone in the printed diagrams; choosing sides is separate from player identity.
"""

from dataclasses import dataclass
from math import atan2, cos, degrees, hypot, pi, sin

from battle_config import DEPLOYMENT_MAPS, MIRRORABLE_MAPS
from psychology import _box_corners, obb_distance


EPSILON = 1e-7


@dataclass(frozen=True)
class DeploymentZone:
    vertices: tuple
    outline: tuple
    excluded_radius: float = 0

    def contains_point(self, point, tolerance=EPSILON):
        cross_products = []
        for start, finish in zip(self.vertices, self.vertices[1:] + self.vertices[:1]):
            edge_x, edge_y = finish[0] - start[0], finish[1] - start[1]
            cross_products.append((edge_x * (point[1] - start[1])
                                   - edge_y * (point[0] - start[0])) / hypot(edge_x, edge_y))
        inside = (min(cross_products) >= -tolerance or max(cross_products) <= tolerance)
        return inside and hypot(*point) + tolerance >= self.excluded_radius

    def contains_box(self, box, tolerance=EPSILON):
        """A base edge can cross the excluded circle even if its corners do not."""
        return (all(self.contains_point(corner, tolerance) for corner in _box_corners(*box))
                and (not self.excluded_radius or obb_distance(box, (0, 0, 0, 0, 0))
                     + tolerance >= self.excluded_radius))


@dataclass(frozen=True)
class Battlefield:
    width: float = 72
    depth: float = 48

    def __post_init__(self):
        from math import isfinite
        for value in (self.width, self.depth):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not isfinite(value) or value <= 0):
                raise ValueError('Battlefield dimensions must be finite and positive')

    @property
    def outline(self):
        half_width, half_depth = self.width / 2, self.depth / 2
        return ((-half_width, -half_depth), (half_width, -half_depth),
                (half_width, half_depth), (-half_width, half_depth))

    def contains_point(self, point, tolerance=EPSILON):
        return (abs(point[0]) <= self.width / 2 + tolerance
                and abs(point[1]) <= self.depth / 2 + tolerance)

    def contains_box(self, box, tolerance=EPSILON):
        return all(self.contains_point(corner, tolerance) for corner in _box_corners(*box))

    def edge_distance(self, point):
        return max(0, min(self.width / 2 - abs(point[0]), self.depth / 2 - abs(point[1])))

    def deployment_zone(self, map_name='standard', side=1, *, mirror=False):
        """Printed distances stay fixed on both supported Battle March sizes."""
        if side not in (1, 2):
            raise ValueError('Deployment side must be 1 or 2')
        if map_name not in ('standard', *DEPLOYMENT_MAPS):
            raise ValueError(f'Unknown or unresolved deployment map: {map_name}')
        if mirror and map_name not in MIRRORABLE_MAPS:
            raise ValueError(f'{map_name} has no alternate deployment')
        if map_name != 'standard' and not (44 <= self.width <= 48 and 30 <= self.depth <= 36):
            raise ValueError('Battle March maps require a 44-48 by 30-36 inch battlefield')
        half_width, half_depth = self.width / 2, self.depth / 2
        radius = 0
        if map_name in ('standard', 'pitched_battle', 'meeting_engagement'):
            front = -half_depth + 12 if map_name == 'standard' else -7.5
            left = -half_width + 11 if map_name == 'meeting_engagement' else -half_width
            vertices = ((left, -half_depth), (half_width, -half_depth),
                        (half_width, front), (left, front))
        elif map_name == 'close_encounter':
            vertices = ((0, -half_depth), (half_width, -half_depth), (half_width, 0), (0, 0))
            radius = 7.5
        elif map_name == 'opposed_flanks':
            vertices = ((-half_width, -half_depth), (half_width, -half_depth),
                        (half_width, half_depth - 18))
        elif map_name == 'mountain_pass':
            vertices = ((11, -half_depth), (half_width, -half_depth),
                        (half_width, half_depth), (11, half_depth))
        else:
            vertices = ((-half_width + 22, -half_depth), (half_width, -half_depth),
                        (half_width, half_depth))
        outline = vertices
        if radius:
            arc = tuple((radius * cos(-pi * index / 128), radius * sin(-pi * index / 128))
                        for index in range(65))
            outline = vertices[:3] + arc
        direction = 1 if side == 1 else -1

        def transform(points):
            return tuple((point[0] * direction * (-1 if mirror else 1), point[1] * direction)
                         for point in points)

        return DeploymentZone(transform(vertices), transform(outline), radius)


STANDARD_BATTLEFIELD = Battlefield()


def battlefield_for(game):
    """Legacy games and lightweight test fixtures retain their 72 x 48 bounds."""
    field = getattr(game, 'battlefield', None)
    return field if isinstance(field, Battlefield) else STANDARD_BATTLEFIELD


def deployment_zone_for(game, player):
    """Use resolved, saved side choices; never roll during a placement preview."""
    setup = getattr(game, 'battle_setup', None)
    if setup is None:
        return battlefield_for(game).deployment_zone(side=player)
    return battlefield_for(game).deployment_zone(
        setup['deployment_map'], setup['player_zones'][str(player)], mirror=setup['mirror'])


def deployment_candidate(game, player, rng):
    """A bounded candidate, not a legal drop; the caller validates every base."""
    zone = deployment_zone_for(game, player)
    horizontal, vertical = zip(*zone.vertices)
    for attempt in range(100):
        point = (rng.uniform(min(horizontal), max(horizontal)), rng.uniform(min(vertical), max(vertical)))
        if zone.contains_point(point):
            return point
    return None


def deployment_heading(game, player):
    vertices = deployment_zone_for(game, player).vertices
    centre = tuple(sum(point[axis] for point in vertices) / len(vertices) for axis in (0, 1))
    return degrees(atan2(centre[0], -centre[1]))


def draw_battlefield(game):
    """Thin overlays leave the 72 x 48 visual table and model scale intact."""
    from panda3d.core import LineSegs
    previous = getattr(game, 'battlefield_overlay', None)
    if previous is not None:
        previous.removeNode()
    root = game.render.attachNewNode('battlefield-overlay')
    game.battlefield_overlay = root
    root.setDepthTest(False)
    root.setDepthWrite(False)
    root.setBin('fixed', 20)
    settings = (getattr(game, 'battle_config', None) or {}).get('battlefield', {})

    def line(name, points, color):
        geometry = LineSegs(name)
        geometry.setThickness(2)
        geometry.setColor(*color)
        geometry.moveTo(*points[0], .12)
        for point in (*points[1:], points[0]):
            geometry.drawTo(*point, .12)
        root.attachNewNode(geometry.create())

    if settings.get('show_boundary', True):
        line('playable-boundary', battlefield_for(game).outline, (1, 1, 1, 1))
    preparation = (getattr(game, 'battle_setup', None) or {}).get('preparation', {})
    if settings.get('show_deployment', True) and preparation.get('stage') not in ('armies', 'terrain', 'objectives'):
        for player, color in ((1, (.15, .85, 1, 1)), (2, (1, .35, .25, 1))):
            line(f'deployment-player-{player}', deployment_zone_for(game, player).outline, color)
    return root