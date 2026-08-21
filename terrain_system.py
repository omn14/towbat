"""Terrain system — terrain pieces, placement, and rule queries.

Each ``TerrainPiece`` represents a rectangular area on the battlefield with
associated game-play rules (movement penalties, line-of-sight blocking,
combat modifiers, etc.).  The ``TerrainManager`` owns every piece and
exposes simple query helpers consumed by MovementSystem, CombatResolver,
and the AI subsystems.
"""

import json
import math
import random

from panda3d.core import (
    Point3, Vec2, Vec3, Vec4, BitMask32,
    TransparencyAttrib, NodePath, LineSegs, Shader,
    GeomVertexFormat, GeomVertexData, GeomVertexWriter,
    Geom, GeomTriangles, GeomNode,
)
from panda3d.bullet import (
    BulletGhostNode, BulletBoxShape,
    BulletTriangleMesh, BulletTriangleMeshShape, BulletRigidBodyNode,
)

from collision_masks import CollisionMask as CM


# ── Terrain rule definitions ──────────────────────────────────────────────────

# The going a piece presents is separate from what it looks like: the rulebook
# is explicit that a wood "might be classed as difficult, dangerous or even
# impassable terrain, based upon its size and density" (p. 270). A map may say
# so per piece; each terrain type carries the category it presents by default.
TERRAIN_CATEGORIES = {
    'open': {
        'movement_modifier': 0,
        'dangerous': False,
        'impassable': False,
        'disrupts': False,
        'description': 'Open ground',
    },
    'difficult': {
        'movement_modifier': -1,
        'dangerous': False,
        'impassable': False,
        'disrupts': True,
        'description': 'Difficult terrain — -1 Movement, charge discards the highest die',
    },
    'dangerous': {
        # Dangerous terrain "hinders movement just like difficult terrain".
        'movement_modifier': -1,
        'dangerous': True,
        'impassable': False,
        'disrupts': True,
        'description': 'Dangerous terrain — as difficult, and a D6 test or lose a Wound',
    },
    'impassable': {
        'movement_modifier': 0,
        'dangerous': False,
        'impassable': True,
        'disrupts': False,
        'description': 'Impassable terrain — cannot be crossed',
    },
}

TERRAIN_RULES = {
    'forest': {
        'going': 'difficult',
        'blocks_line_of_sight': True,
    },
    'hill': {
        # Hills are open ground; their advantage is elevation, not going.
        'going': 'open',
        'blocks_line_of_sight': True,
    },
    'river': {
        'going': 'dangerous',
        'blocks_line_of_sight': False,
    },
    'marsh': {
        'going': 'dangerous',
        'blocks_line_of_sight': False,
    },
    'house': {
        # "most buildings" are impassable terrain (Rulebook p. 270).
        'going': 'impassable',
        'blocks_line_of_sight': True,
    },
    'pillar_of_fire': {
        # A Magical Vortex, conjured mid-battle and gone when the spell ends.
        'going': 'difficult',
        'blocks_line_of_sight': False,
    },
}

# Colours used to tint the terrain overlay (RGBA, alpha < 1 → translucent)
_TERRAIN_COLORS = {
    'forest': Vec4(0.15, 0.55, 0.15, 0.50),
    'hill':   Vec4(0.50, 0.40, 0.20, 0.35),
    'river':  Vec4(0.10, 0.20, 0.60, 0.45),
    'marsh':  Vec4(0.30, 0.30, 0.10, 0.40),
    'house':  Vec4(0.45, 0.20, 0.15, 0.60),
    'pillar_of_fire': Vec4(0.95, 0.35, 0.05, 0.65),
}

_TERRAIN_COLLISION_MASK = {
    'forest': BitMask32.bit(20),
    'hill':   BitMask32.bit(21),
    'river':  BitMask32.bit(22),
    'marsh':  BitMask32.bit(23),
    'house':  CM.TERRAIN_IMPASSABLE,
    'pillar_of_fire': BitMask32.bit(23),
}

_TERRAIN_MODEL_PATHS = {
    'forest': "models/hills.bam",
    'hill':   "models/hills.bam",
    'river':  "models/hills.bam",
    'marsh':  "models/hills.bam",
    'pillar_of_fire': "models/hills.bam",
}

# Integer id handed to the terrain shader (see shaders/terrain.frag).
_TERRAIN_TYPE_ID = {'forest': 0, 'hill': 1, 'river': 2, 'marsh': 3,
                    'pillar_of_fire': 3}

# Water is drawn as a flat animated surface instead of a raised mesh.
_WATER_TYPES = {'river', 'marsh'}

# Pieces that build their own coloured geometry and want no terrain shader.
_BUILT_TYPES = {'house'}

# Small lift so raised terrain doesn't z-fight the ground plane.
_HILL_LIFT = 0.02

# Vertical scale per type — hills rise, forests sit low, water is nearly flat.
_TERRAIN_HEIGHT = {'forest': 2.0, 'hill': 4.0, 'river': 0.3, 'marsh': 0.4,
                   'house': 4.5, 'pillar_of_fire': 2.5}

# Cached, shared GLSL shader instance (loaded once on first use).
_TERRAIN_SHADER = None


# ── Dangerous Terrain tests (Rulebook p. 269) ─────────────────────────────────

# A unit with a quarter or more of its models in difficult terrain is Disrupted.
DISRUPT_FRACTION = 4


def is_disrupted(models_in_terrain: int, models: int) -> bool:
    """True if enough of a unit's models stand in difficult terrain to cost it
    its Rank Bonus (Rulebook p. 269)."""
    if models <= 0:
        return False
    return models_in_terrain * DISRUPT_FRACTION >= models


def sees_over(shooter_pos, blocker_pos, hill_center) -> bool:
    """True if a unit on a hill can see over another unit on the same hill.

    Only a unit closer to the bottom can be seen over; the hill's top is its
    centre (Official FAQ 1.5.3). Positions need only be indexable as (x, y).
    """
    def to_top(p):
        return math.hypot(p[0] - hill_center[0], p[1] - hill_center[1])
    return to_top(blocker_pos) > to_top(shooter_pos)


def dangerous_terrain_wounds(features: int, models: int, damage='1') -> int:
    """Wounds a unit suffers crossing *features* dangerous terrain features.

    Every model that begins in, passes through or ends in dangerous terrain
    tests, once per separate feature, and loses a Wound on a roll of 1.
    *damage* is a dice expression so Iron Shod Wheels can cost a chariot D3.
    """
    if features <= 0 or models <= 0:
        return 0
    from models import roll_dice_expr
    wounds = 0
    for _ in range(features * models):
        if random.randint(1, 6) == 1:
            wounds += roll_dice_expr(damage)
    return wounds


def _get_terrain_shader():
    """Load the procedural terrain shader once and reuse it for every piece."""
    global _TERRAIN_SHADER
    if _TERRAIN_SHADER is None:
        try:
            _TERRAIN_SHADER = Shader.load(
                Shader.SL_GLSL,
                vertex="shaders/terrain.vert",
                fragment="shaders/terrain.frag",
            )
        except Exception as exc:  # pragma: no cover — shader is optional eye-candy
            print(f"[Terrain] Could not load terrain shader: {exc}")
            _TERRAIN_SHADER = False  # sentinel: don't retry
    return _TERRAIN_SHADER or None


# Cached, shared low-poly fir-tree model used to populate forests.
_TREE_MODEL = None


def _cone_geom(radius, height, base_z, segments, color):
    """Return a Geom for a single cone (apex up) with per-vertex colour."""
    fmt = GeomVertexFormat.getV3n3c4()
    vdata = GeomVertexData('cone', fmt, Geom.UHStatic)
    vw = GeomVertexWriter(vdata, 'vertex')
    nw = GeomVertexWriter(vdata, 'normal')
    cw = GeomVertexWriter(vdata, 'color')

    # Apex vertex (index 0).
    vw.addData3(0.0, 0.0, base_z + height)
    nw.addData3(0.0, 0.0, 1.0)
    cw.addData4(*color)

    # Base ring — the extra vertex closes the loop.
    for i in range(segments + 1):
        ang = 2.0 * math.pi * i / segments
        cx, cy = math.cos(ang), math.sin(ang)
        vw.addData3(cx * radius, cy * radius, base_z)
        n = Vec3(cx * height, cy * height, radius)
        n.normalize()
        nw.addData3(n.x, n.y, n.z)
        cw.addData4(*color)

    tris = GeomTriangles(Geom.UHStatic)
    for i in range(1, segments + 1):
        tris.addVertices(0, i, i + 1)
    tris.closePrimitive()

    geom = Geom(vdata)
    geom.addPrimitive(tris)
    return geom


def _get_tree_model():
    """Build a small fir tree (brown trunk + stacked green cones) once."""
    global _TREE_MODEL
    if _TREE_MODEL is None:
        try:
            node = GeomNode('tree')
            trunk = (0.35, 0.22, 0.08, 1.0)
            lower = (0.10, 0.35, 0.10, 1.0)
            upper = (0.16, 0.45, 0.14, 1.0)
            node.addGeom(_cone_geom(0.15, 0.6, 0.0, 6, trunk))
            node.addGeom(_cone_geom(0.9, 1.4, 0.4, 7, lower))
            node.addGeom(_cone_geom(0.6, 1.2, 1.1, 7, upper))
            _TREE_MODEL = NodePath(node)
        except Exception as exc:  # pragma: no cover — trees are optional eye-candy
            print(f"[Terrain] Could not build tree model: {exc}")
            _TREE_MODEL = False  # sentinel: don't retry
    return _TREE_MODEL or None


# ── Simple coloured-mesh builder (houses) ──────────────────────────────────

class _MeshBuilder:
    """Accumulates flat-shaded coloured triangles into one Geom."""

    def __init__(self, name='mesh'):
        fmt = GeomVertexFormat.getV3n3c4()
        self._vdata = GeomVertexData(name, fmt, Geom.UHStatic)
        self._vw = GeomVertexWriter(self._vdata, 'vertex')
        self._nw = GeomVertexWriter(self._vdata, 'normal')
        self._cw = GeomVertexWriter(self._vdata, 'color')
        self._tris = GeomTriangles(Geom.UHStatic)
        self._n = 0

    def tri(self, a, b, c, color):
        n = Vec3(*b) - Vec3(*a)
        n = n.cross(Vec3(*c) - Vec3(*a))
        if n.length() > 1e-9:
            n.normalize()
        else:
            n = Vec3(0, 0, 1)
        for p in (a, b, c):
            self._vw.addData3(*p)
            self._nw.addData3(n.x, n.y, n.z)
            self._cw.addData4(*color)
        self._tris.addVertices(self._n, self._n + 1, self._n + 2)
        self._n += 3

    def quad(self, a, b, c, d, color):
        self.tri(a, b, c, color)
        self.tri(a, c, d, color)

    def box(self, x0, x1, y0, y1, z0, z1, color, top_color=None):
        self.quad((x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1), color)
        self.quad((x1, y1, z0), (x0, y1, z0), (x0, y1, z1), (x1, y1, z1), color)
        self.quad((x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1), color)
        self.quad((x0, y1, z0), (x0, y0, z0), (x0, y0, z1), (x0, y1, z1), color)
        self.quad((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
                  top_color or color)

    def geom(self):
        g = Geom(self._vdata)
        self._tris.closePrimitive()
        g.addPrimitive(self._tris)
        return g


# Medieval timber-framed house palette.
_HOUSE_PLASTER = (0.86, 0.82, 0.71, 1.0)
_HOUSE_TIMBER = (0.26, 0.17, 0.10, 1.0)
_HOUSE_ROOF = (0.42, 0.22, 0.16, 1.0)
_HOUSE_ROOF_DARK = (0.32, 0.16, 0.12, 1.0)
_HOUSE_DOOR = (0.30, 0.19, 0.11, 1.0)
_HOUSE_WINDOW = (0.14, 0.12, 0.10, 1.0)
_HOUSE_STONE = (0.55, 0.52, 0.47, 1.0)


def _gable_face(b, x, sx, points, color):
    """A flat polygon on the gable end at *x*, given as (y, z) points wound
    counter-clockwise seen from +x, and re-wound so its normal faces outward.

    Without this the far gable is lit as though it faced into the house.
    """
    order = list(points) if sx > 0 else list(reversed(points))
    verts = [(x, y, z) for y, z in order]
    (b.tri if len(verts) == 3 else b.quad)(*verts, color)


def _gable_brace(b, x, sx, start, end, half_width):
    """A diagonal timber on the gable face at *x*, from (y, z) to (y, z)."""
    (ay, az), (by, bz) = start, end
    dy, dz = by - ay, bz - az
    length = math.hypot(dy, dz) or 1.0
    ny, nz = -dz / length * half_width, dy / length * half_width
    _gable_face(b, x, sx, [(ay + ny, az + nz), (by + ny, bz + nz),
                           (by - ny, bz - nz), (ay - ny, az - nz)],
                _HOUSE_TIMBER)


def _spans_between(lo, hi, openings):
    """What is left of the span *lo*..*hi* after cutting out *openings*.

    A timber rail runs the length of a wall but stops at each door and window
    rather than crossing it.
    """
    spans = [(lo, hi)]
    for o_lo, o_hi in sorted(openings):
        kept = []
        for s_lo, s_hi in spans:
            if o_hi <= s_lo or o_lo >= s_hi:
                kept.append((s_lo, s_hi))
                continue
            if s_lo < o_lo:
                kept.append((s_lo, o_lo))
            if o_hi < s_hi:
                kept.append((o_hi, s_hi))
        spans = kept
    return [(a, z) for a, z in spans if z - a > 1e-6]


def _house_geom(length, breadth, height):
    """A timber-framed house with a gabled roof, centred on the origin.

    The ridge runs along X, so *length* is the ridge axis and the roof slopes
    down towards ±Y. The caller turns the piece if its footprint is deeper
    than it is wide.
    """
    b = _MeshBuilder('house')
    # Leave a margin so the building sits inside its footprint rather than
    # exactly on the edge that blocks movement.
    hw, hd = length * 0.42, breadth * 0.42
    wall = height * 0.55
    ridge = height
    eave = min(hw, hd) * 0.16              # roof overhang
    sill = wall * 0.06                     # stone footing
    post = min(hw, hd) * 0.09
    prox = 0.03                            # sit trim just proud of the plaster

    b.box(-hw, hw, -hd, hd, 0.0, sill, _HOUSE_STONE)
    b.box(-hw, hw, -hd, hd, sill, wall, _HOUSE_PLASTER)

    # Corner posts, the timber frame that dates the building.
    for sx in (-1, 1):
        for sy in (-1, 1):
            b.box(sx * hw - post, sx * hw + post, sy * hd - post, sy * hd + post,
                  sill, wall, _HOUSE_TIMBER)

    # Door and shuttered windows along the front.
    dw, dh = hw * 0.16, wall * 0.66
    ww = hw * 0.13
    win_lo, win_hi = wall * 0.42, wall * 0.74
    win_x = [sx * hw * 0.55 for sx in (-1, 1)]
    b.quad((-dw, -hd - prox, 0.0), (dw, -hd - prox, 0.0),
           (dw, -hd - prox, dh), (-dw, -hd - prox, dh), _HOUSE_DOOR)
    for cx in win_x:
        b.quad((cx - ww, -hd - prox, win_lo), (cx + ww, -hd - prox, win_lo),
               (cx + ww, -hd - prox, win_hi), (cx - ww, -hd - prox, win_hi),
               _HOUSE_WINDOW)

    # Mid rail, half way from the ground to the window sills, stopping either
    # side of every opening it would otherwise cross.
    rail = win_lo * 0.5
    openings = [(-dw, dw)] + [(cx - ww, cx + ww) for cx in win_x]
    for sy in (-1, 1):
        spans = _spans_between(-hw, hw, openings) if sy < 0 else [(-hw, hw)]
        for x_lo, x_hi in spans:
            b.box(x_lo, x_hi, sy * hd - 0.02, sy * hd + 0.02,
                  rail - post * 0.6, rail + post * 0.6, _HOUSE_TIMBER)

    # Gable ends: the wall triangle between the eaves and the ridge, framed
    # with a tie beam, braces and a king post, and lit by a small window.
    # Each layer stands a little further proud so no two share a plane.
    for sx in (-1, 1):
        x = sx * hw
        gable = ridge - wall
        _gable_face(b, x, sx, [(-hd, wall), (hd, wall), (0.0, ridge)],
                    _HOUSE_PLASTER)

        xg = x + sx * prox * 0.5
        _gable_face(b, xg, sx, [(-ww * 0.9, wall + gable * 0.12),
                                (ww * 0.9, wall + gable * 0.12),
                                (ww * 0.9, wall + gable * 0.44),
                                (-ww * 0.9, wall + gable * 0.44)], _HOUSE_WINDOW)
        _gable_face(b, xg, sx, [(-ww, win_lo), (ww, win_lo),
                                (ww, win_hi), (-ww, win_hi)], _HOUSE_WINDOW)

        xt = x + sx * prox
        _gable_face(b, xt, sx, [(-hd, wall - post * 0.6), (hd, wall - post * 0.6),
                                (hd, wall + post * 0.6), (-hd, wall + post * 0.6)],
                    _HOUSE_TIMBER)                              # tie beam
        for sy in (-1, 1):                                      # corner studs
            _gable_face(b, xt, sx, [(sy * hd - post, sill), (sy * hd + post, sill),
                                    (sy * hd + post, wall), (sy * hd - post, wall)],
                        _HOUSE_TIMBER)

        xb = x + sx * prox * 1.5
        for sy in (-1, 1):
            _gable_brace(b, xb, sx, (sy * hd * 0.74, wall),
                         (sy * post * 0.4, wall + gable * 0.60), post * 0.5)

        _gable_face(b, x + sx * prox * 2.0, sx,
                    [(-post * 0.5, wall + gable * 0.55),
                     (post * 0.5, wall + gable * 0.55),
                     (post * 0.5, ridge), (-post * 0.5, ridge)],
                    _HOUSE_TIMBER)                              # king post

    # Roof: two slopes meeting at the ridge, overhanging all four sides.
    x0, x1 = -hw - eave, hw + eave
    y0, y1 = -hd - eave, hd + eave
    b.quad((x0, y0, wall), (x1, y0, wall), (x1, 0.0, ridge), (x0, 0.0, ridge),
           _HOUSE_ROOF)
    b.quad((x1, y1, wall), (x0, y1, wall), (x0, 0.0, ridge), (x1, 0.0, ridge),
           _HOUSE_ROOF)
    # Undersides, dropped by the roof's thickness: coplanar with the slopes
    # they would z-fight and show through as dark patches.
    t = ridge * 0.03
    b.quad((x1, y0, wall - t), (x0, y0, wall - t), (x0, 0.0, ridge - t),
           (x1, 0.0, ridge - t), _HOUSE_ROOF_DARK)
    b.quad((x0, y1, wall - t), (x1, y1, wall - t), (x1, 0.0, ridge - t),
           (x0, 0.0, ridge - t), _HOUSE_ROOF_DARK)

    # Chimney, rising against one gable end.
    cw = min(hw, hd) * 0.15
    cx = hw * 0.72
    b.box(cx - cw, cx + cw, -cw, cw, wall * 0.5, ridge * 1.2, _HOUSE_STONE)
    return b.geom()


# ── Deterministic value noise for procedural terrain shapes ────────────────
def _hash2(x, y, seed):
    n = math.sin(x * 127.1 + y * 311.7 + seed * 74.7) * 43758.5453
    return n - math.floor(n)


def _vnoise(x, y, seed):
    x0, y0 = math.floor(x), math.floor(y)
    fx, fy = x - x0, y - y0
    a = _hash2(x0, y0, seed)
    b = _hash2(x0 + 1, y0, seed)
    c = _hash2(x0, y0 + 1, seed)
    d = _hash2(x0 + 1, y0 + 1, seed)
    ux = fx * fx * (3.0 - 2.0 * fx)
    uy = fy * fy * (3.0 - 2.0 * fy)
    return (a * (1.0 - ux) + b * ux) * (1.0 - uy) + (c * (1.0 - ux) + d * ux) * uy


def _fbm2(x, y, seed, octaves=4):
    v, amp, f = 0.0, 0.5, 1.0
    for _ in range(octaves):
        v += amp * _vnoise(x * f, y * f, seed)
        f *= 2.0
        amp *= 0.5
    return v


# ── TerrainPiece ──────────────────────────────────────────────────────────────

class TerrainPiece:
    """A single rectangular terrain feature on the battlefield."""

    def __init__(self, terrain_type: str, center: Point3,
                 width: float, height: float, game, going: str = None):
        if terrain_type not in TERRAIN_RULES:
            raise ValueError(
                f"Unknown terrain type '{terrain_type}'. "
                f"Valid types: {list(TERRAIN_RULES.keys())}"
            )
        self.terrain_type = terrain_type
        self.rules = TERRAIN_RULES[terrain_type]
        self.going = going or self.rules['going']
        if self.going not in TERRAIN_CATEGORIES:
            raise ValueError(
                f"Unknown terrain category '{self.going}'. "
                f"Valid categories: {list(TERRAIN_CATEGORIES)}"
            )
        self.category = TERRAIN_CATEGORIES[self.going]
        self.center = center
        self.width = width
        self.height = height
        self.game = game

        self.river_centerline = None   # populated for river pieces
        self.debug_np = None
        self._hf = None                # surface height function (forest/hill)
        self._field = None             # footprint field (forest/hill rim test)
        self._field_edge = 0.0

        self._create_visual()
        # Water and field-shaped pieces read as natural shapes; a rectangle
        # outline would misrepresent them.
        self.outline = None
        if self.terrain_type not in _WATER_TYPES and self.terrain_type not in ('hill', 'forest'):
            self._create_outline()
        #self._create_collision()
        self._create_collision_from_mesh()
        self.trees_np = None
        if self.terrain_type == 'forest':
            self._create_trees()

    def _create_visual(self):
        if self.terrain_type == 'river':
            self._create_river_visual()
        elif self.terrain_type in _WATER_TYPES:
            self._create_water_visual()
        elif self.terrain_type == 'house':
            self._create_house_visual()
        elif self.terrain_type == 'hill':
            self._create_heightfield_visual(
                peak=min(self.width, self.height) * 0.11, seed=11.0)
        elif self.terrain_type == 'forest':
            self._create_heightfield_visual(
                peak=min(self.width, self.height) * 0.03, seed=23.0)
        else:
            self._create_mesh_visual()
        if self.terrain_type not in _BUILT_TYPES:
            self._apply_shader()

    def _create_house_visual(self):
        """A medieval timber-framed house filling the piece's footprint."""
        long_side = max(self.width, self.height)
        short_side = min(self.width, self.height)
        node = GeomNode(f"house_{id(self)}")
        node.addGeom(_house_geom(long_side, short_side,
                                 _TERRAIN_HEIGHT.get('house', 4.0)))
        self.visual = render.attachNewNode(node)
        self.visual.setPos(self.center.x, self.center.y, _HILL_LIFT)
        # The trim is flat decals on the walls; two-sided saves winding each
        # one per face just to keep it from being culled.
        self.visual.setTwoSided(True)
        # The ridge is built along X; turn it if the footprint is the deeper
        # way round. Buildings sit square to the board.
        self.visual.setH(90.0 if self.height > self.width else 0.0)
        self.visual.flattenStrong()

    def _create_mesh_visual(self):
        model_path = _TERRAIN_MODEL_PATHS.get(self.terrain_type, "models/hills.bam")
        self.visual = loader.loadModel(model_path)
        self.visual.reparentTo(render)

        # The source mesh spans roughly -1..1, so a scale of width/2 makes its
        # footprint match the outline rectangle (full width × height).  The
        # vertical scale is per-type so hills rise while forests/water stay low.
        z_scale = _TERRAIN_HEIGHT.get(self.terrain_type, 3.0)
        self.visual.setScale(self.width / 2.0, self.height / 2.0, z_scale)
        self.visual.setPos(self.center.x, self.center.y, 0.0)
        self.visual.flattenStrong()

    def _create_heightfield_visual(self, peak, seed):
        """Organic kidney-shaped mound with a flat top for units to stand on.

        A smooth metaball field defines the footprint (clean rim, no edge
        spikes); the height plateaus in the centre and slopes to the rim.
        The mesh is clipped to the rim and ``contains()`` uses the same field,
        so both the silhouette and collision follow the visible shape.
        """
        hw = self.width / 2.0
        hh = self.height / 2.0
        N = 40
        f_edge = 0.35      # rim iso-level (lower = wider, gentler slope toe)
        f_top = 0.95       # plateau (flat top) begins here

        def field(nx, ny):
            g1 = math.exp(-((nx + 0.30) ** 2 + (ny - 0.05) ** 2) / (2 * 0.34 ** 2))
            g2 = math.exp(-((nx - 0.30) ** 2 + (ny + 0.20) ** 2) / (2 * 0.34 ** 2))
            g3 = math.exp(-((nx + 0.02) ** 2 + (ny - 0.28) ** 2) / (2 * 0.22 ** 2))
            f = g1 + g2 + 0.6 * g3
            # Low-frequency wobble only — keeps the rim organic but smooth.
            f += 0.10 * (_fbm2(nx * 1.2 + seed, ny * 1.2 + seed, seed) - 0.5)
            return f

        def sstep(a, b, x):
            t = max(0.0, min(1.0, (x - a) / (b - a)))
            return t * t * (3.0 - 2.0 * t)

        # Field-based containment (rim) + surface height for tree placement.
        self._field = field
        self._field_edge = f_edge
        self._hf = lambda lx, ly: peak * sstep(f_edge, f_top, field(lx / hw, ly / hh))

        dx = (2.0 * hw) / N
        dy = (2.0 * hh) / N
        F = [[0.0] * (N + 1) for _ in range(N + 1)]
        H = [[0.0] * (N + 1) for _ in range(N + 1)]
        for j in range(N + 1):
            for i in range(N + 1):
                fv = field((-hw + i * dx) / hw, (-hh + j * dy) / hh)
                F[j][i] = fv
                H[j][i] = peak * sstep(f_edge, f_top, fv)

        fmt = GeomVertexFormat.getV3n3t2()
        vdata = GeomVertexData(self.terrain_type, fmt, Geom.UHStatic)
        vw = GeomVertexWriter(vdata, 'vertex')
        nw = GeomVertexWriter(vdata, 'normal')
        tw = GeomVertexWriter(vdata, 'texcoord')

        for j in range(N + 1):
            for i in range(N + 1):
                vw.addData3(-hw + i * dx, -hh + j * dy, H[j][i])
                il, ir = max(i - 1, 0), min(i + 1, N)
                jd, ju = max(j - 1, 0), min(j + 1, N)
                dhx = (H[j][ir] - H[j][il]) / ((ir - il) * dx)
                dhy = (H[ju][i] - H[jd][i]) / ((ju - jd) * dy)
                n = Vec3(-dhx, -dhy, 1.0)
                n.normalize()
                nw.addData3(n.x, n.y, n.z)
                # Store the field value; the shader discards fragments below the
                # rim, giving a smooth per-pixel edge (no stair-stepping).
                tw.addData2(F[j][i], i / N)

        tris = GeomTriangles(Geom.UHStatic)
        row = N + 1
        for j in range(N):
            for i in range(N):
                v00 = j * row + i
                v10 = v00 + 1
                v01 = v00 + row
                v11 = v01 + 1
                tris.addVertices(v00, v10, v11)
                tris.addVertices(v00, v11, v01)
        tris.closePrimitive()

        geom = Geom(vdata)
        geom.addPrimitive(tris)
        node = GeomNode(f"terrain_{self.terrain_type}")
        node.addGeom(geom)

        self.visual = render.attachNewNode(node)
        # Lift a hair so the rim doesn't z-fight the ground plane (no depth
        # offset — that would let the hill draw over units standing in front).
        self.visual.setPos(self.center.x, self.center.y, _HILL_LIFT)
        self.visual.flattenStrong()

        self._build_heightfield_debug(F, f_edge, f_top, dx, dy, N)

    def _build_heightfield_debug(self, F, f_edge, f_top, dx, dy, N):
        """Draw the collision rim (magenta) and plateau edge (yellow), hidden
        until toggled with F7 — same idea as the river debug band."""
        hw = self.width / 2.0
        hh = self.height / 2.0
        z = 0.25
        ls = LineSegs("hill_debug")
        ls.setThickness(2.5)

        def wpos(i, j):
            return (self.center.x - hw + i * dx, self.center.y - hh + j * dy)

        def add_contour(iso, color):
            ls.setColor(*color)
            for j in range(N):
                for i in range(N):
                    corners = [
                        (F[j][i], wpos(i, j)),
                        (F[j][i + 1], wpos(i + 1, j)),
                        (F[j + 1][i + 1], wpos(i + 1, j + 1)),
                        (F[j + 1][i], wpos(i, j + 1)),
                    ]
                    pts = []
                    for k in range(4):
                        fa, pa = corners[k]
                        fb, pb = corners[(k + 1) % 4]
                        if (fa < iso) != (fb < iso):
                            t = (iso - fa) / (fb - fa)
                            pts.append((pa[0] + (pb[0] - pa[0]) * t,
                                        pa[1] + (pb[1] - pa[1]) * t))
                    for s in range(0, len(pts) - 1, 2):
                        ls.moveTo(pts[s][0], pts[s][1], z)
                        ls.drawTo(pts[s + 1][0], pts[s + 1][1], z)

        add_contour(f_edge, (1.0, 0.0, 1.0, 1.0))   # rim = collision boundary
        add_contour(f_top, (1.0, 1.0, 0.0, 1.0))     # plateau / flat-top edge

        self.debug_np = render.attachNewNode(ls.create())
        self.debug_np.hide()

    def _create_water_visual(self):
        """Flat quad (texcoords 0..1) for marsh; the shader paints an
        irregular, soft-edged bog so it doesn't read as a box."""
        fmt = GeomVertexFormat.getV3n3t2()
        vdata = GeomVertexData('water', fmt, Geom.UHStatic)
        vw = GeomVertexWriter(vdata, 'vertex')
        nw = GeomVertexWriter(vdata, 'normal')
        tw = GeomVertexWriter(vdata, 'texcoord')

        hw, hh = self.width / 2.0, self.height / 2.0
        corners = [(-hw, -hh, 0.0, 0.0), (hw, -hh, 1.0, 0.0),
                   (hw, hh, 1.0, 1.0), (-hw, hh, 0.0, 1.0)]
        for x, y, u, v in corners:
            vw.addData3(x, y, 0.0)
            nw.addData3(0.0, 0.0, 1.0)
            tw.addData2(u, v)

        tris = GeomTriangles(Geom.UHStatic)
        tris.addVertices(0, 1, 2)
        tris.addVertices(0, 2, 3)
        tris.closePrimitive()

        geom = Geom(vdata)
        geom.addPrimitive(tris)
        node = GeomNode(f"water_{self.terrain_type}")
        node.addGeom(geom)

        self.visual = render.attachNewNode(node)
        # Sit just above the ground plane to avoid z-fighting.
        self.visual.setPos(self.center.x, self.center.y, 0.05)
        self.visual.setTransparency(TransparencyAttrib.MAlpha)
        self.visual.setDepthWrite(False)

    def _create_river_visual(self):
        """Build a meandering ribbon mesh that snakes across the piece.

        Only the water ribbon is drawn (everything else is transparent),
        so the river reads as a natural winding channel rather than a box.
        Texcoords: u runs along the flow, v runs bank-to-bank (0..1).
        """
        tau = 2.0 * math.pi
        long_dim = max(self.width, self.height)
        band = min(self.width, self.height)
        flow_x = self.width >= self.height
        half_long = long_dim / 2.0
        half_w = band * 0.18            # water half-width
        bank_scale = 1.5                # extra ribbon width for the banks
        amp = band * 0.28               # meander amplitude (stays inside band)
        segs = max(24, int(long_dim * 1.2))

        fmt = GeomVertexFormat.getV3n3t2()
        vdata = GeomVertexData('river', fmt, Geom.UHStatic)
        vw = GeomVertexWriter(vdata, 'vertex')
        nw = GeomVertexWriter(vdata, 'normal')
        tw = GeomVertexWriter(vdata, 'texcoord')

        # Sinuous centreline in piece-local coordinates.
        centre = []
        for i in range(segs + 1):
            s = i / segs
            l = -half_long + s * (2.0 * half_long)
            # Taper the meander to zero over a straight approach at both ends
            # so the river runs square into the board edges, winding in between.
            t0 = max(0.0, min(1.0, s / 0.22))
            t1 = max(0.0, min(1.0, (1.0 - s) / 0.22))
            taper = (t0 * t0 * (3.0 - 2.0 * t0)) * (t1 * t1 * (3.0 - 2.0 * t1))
            off = amp * taper * (math.sin(s * tau * 1.5 + 0.7) * 0.6 +
                                 math.sin(s * tau * 3.1 + 2.3) * 0.4)
            cx, cy = (l, off) if flow_x else (off, l)
            centre.append((s, cx, cy))

        # World-space centreline (x, y, half_width) drives `contains()` so the
        # gameplay water region matches the visible ribbon.
        self.river_centerline = []
        left_pts, right_pts = [], []
        for i, (s, cx, cy) in enumerate(centre):
            # Perpendicular from the smoothed tangent for a clean ribbon width.
            _, ax, ay = centre[max(0, i - 1)]
            _, bx, by = centre[min(segs, i + 1)]
            tx, ty = bx - ax, by - ay
            tl = math.hypot(tx, ty) or 1.0
            px, py = -ty / tl, tx / tl
            # Natural width variation so the banks aren't perfectly parallel.
            w = half_w * (0.7 + 0.5 * (0.5 + 0.5 * math.sin(s * tau * 2.3 + 1.1)))
            wf = w * bank_scale          # full ribbon incl. banks (mesh width)
            vw.addData3(cx + px * wf, cy + py * wf, 0.0)   # left edge (v=0)
            nw.addData3(0.0, 0.0, 1.0)
            tw.addData2(s, 0.0)
            vw.addData3(cx - px * wf, cy - py * wf, 0.0)   # right edge (v=1)
            nw.addData3(0.0, 0.0, 1.0)
            tw.addData2(s, 1.0)

            # Store the WATER half-width so gameplay/collision = water only.
            wx, wy = self.center.x + cx, self.center.y + cy
            self.river_centerline.append((wx, wy, w))
            left_pts.append((self.center.x + cx + px * w, self.center.y + cy + py * w))
            right_pts.append((self.center.x + cx - px * w, self.center.y + cy - py * w))

        tris = GeomTriangles(Geom.UHStatic)
        for i in range(segs):
            l0, r0 = 2 * i, 2 * i + 1
            l1, r1 = 2 * (i + 1), 2 * (i + 1) + 1
            tris.addVertices(l0, r0, l1)
            tris.addVertices(r0, r1, l1)
        tris.closePrimitive()

        geom = Geom(vdata)
        geom.addPrimitive(tris)
        node = GeomNode('river')
        node.addGeom(geom)

        self.visual = render.attachNewNode(node)
        self.visual.setPos(self.center.x, self.center.y, 0.05)
        self.visual.setTransparency(TransparencyAttrib.MAlpha)
        self.visual.setDepthWrite(False)

        self._build_river_debug(left_pts, right_pts)

    def _build_river_debug(self, left_pts, right_pts):
        """Draw the water-detection band (banks + centreline spine), hidden
        until toggled, so the gameplay water region can be seen on the board."""
        z = 0.2
        ls = LineSegs("river_debug")
        ls.setThickness(2.5)

        # Detection boundary = the ribbon banks (magenta).
        ls.setColor(1.0, 0.0, 1.0, 1.0)
        ls.moveTo(left_pts[0][0], left_pts[0][1], z)
        for x, y in left_pts[1:]:
            ls.drawTo(x, y, z)
        for x, y in reversed(right_pts):
            ls.drawTo(x, y, z)
        ls.drawTo(left_pts[0][0], left_pts[0][1], z)

        # Centreline spine (yellow).
        ls.setColor(1.0, 1.0, 0.0, 1.0)
        cl = self.river_centerline
        ls.moveTo(cl[0][0], cl[0][1], z)
        for x, y, _w in cl[1:]:
            ls.drawTo(x, y, z)

        self.debug_np = render.attachNewNode(ls.create())
        self.debug_np.hide()

    def _apply_shader(self):
        """Apply the procedural terrain shader, colouring the mesh by type."""
        shader = _get_terrain_shader()
        if shader is None:
            return
        self.visual.setShader(shader)
        self.visual.setShaderInput(
            "terrainType", _TERRAIN_TYPE_ID.get(self.terrain_type, 1)
        )
        base = _TERRAIN_COLORS.get(self.terrain_type, Vec4(0.5, 0.5, 0.5, 1.0))
        self.visual.setShaderInput("baseColor", base)
        self.visual.setShaderInput("pieceSize", Vec2(self.width, self.height))
        # Rim level for the per-pixel edge discard (hill/forest); 0 = no cut.
        self.visual.setShaderInput(
            "edgeLevel", self._field_edge if self._field is not None else 0.0)
        # Movement/shooting range overlay defaults (updated by TerrainManager).
        self.visual.setShaderInput("moveActive", False)
        self.visual.setShaderInput("movePoints", [Vec2(0, 0)])

    # ── Scattered trees for forests ───────────────────────────────────

    def _create_trees(self):
        """Instance low-poly fir trees across the forest footprint."""
        tree_model = _get_tree_model()
        if tree_model is None:
            return
        self.trees_np = render.attachNewNode("forest_trees")

        # Deterministic layout so trees don't jump around on reload.
        seed = int(self.center.x * 131 + self.center.y * 17 +
                   self.width * 7 + self.height * 3)
        rng = random.Random(seed)

        margin = 1.0
        hw = max(self.width / 2.0 - margin, 0.5)
        hh = max(self.height / 2.0 - margin, 0.5)
        spacing = 2.2
        count = int((self.width * self.height) / (spacing * spacing))
        count = max(4, min(count, 60))  # keep the tree budget sane

        for _ in range(count):
            x = self.center.x + rng.uniform(-hw, hw)
            y = self.center.y + rng.uniform(-hh, hh)
            # Keep trees within the organic forest rim, not the rectangle.
            if self._field is not None:
                nx = (x - self.center.x) / (self.width / 2.0)
                ny = (y - self.center.y) / (self.height / 2.0)
                if self._field(nx, ny) < self._field_edge:
                    continue
            z = self._hf(x - self.center.x, y - self.center.y) if self._hf else 0.0
            placeholder = self.trees_np.attachNewNode("tree")
            tree_model.instanceTo(placeholder)
            placeholder.setPos(x, y, z)
            placeholder.setH(rng.uniform(0.0, 360.0))
            placeholder.setScale(rng.uniform(0.8, 1.5))

    # ── Outline rectangle drawn with LineSegs ─────────────────────────

    def _create_outline(self):
        ls = LineSegs(f"terrain_outline_{self.terrain_type}")
        base_color = _TERRAIN_COLORS.get(
            self.terrain_type, Vec4(0.5, 0.5, 0.5, 1.0)
        )
        ls.setColor(base_color.x, base_color.y, base_color.z, 1.0)
        ls.setThickness(2.0)

        hw, hh = self.width / 2, self.height / 2
        cx, cy, cz = self.center.x, self.center.y, 0.1

        corners = [
            Point3(cx - hw, cy - hh, cz),
            Point3(cx + hw, cy - hh, cz),
            Point3(cx + hw, cy + hh, cz),
            Point3(cx - hw, cy + hh, cz),
        ]
        ls.moveTo(corners[0])
        for c in corners[1:]:
            ls.drawTo(c)
        ls.drawTo(corners[0])

        self.outline = render.attachNewNode(ls.create())

    # ── Bullet ghost for overlap / sensor queries ─────────────────────

    def _create_collision(self):
        shape = BulletBoxShape(Vec3(self.width / 2, self.height / 2, 1.0))
        self.ghost = BulletGhostNode(f"terrain_{self.terrain_type}")
        self.ghost.addShape(shape)
        self.ghost_np = render.attachNewNode(self.ghost)
        self.ghost_np.setPos(self.center)
        # Use bit 20 for terrain — the movement sweep tests only query bit 9,
        # so terrain ghosts won't block unit pathfinding / sweep collisions.
        mask = _TERRAIN_COLLISION_MASK.get(self.terrain_type, BitMask32.bit(20))
        self.ghost_np.setCollideMask(mask)
        self.game.world.attachGhost(self.ghost)

    # ── Bullet ghost for overlap / sensor queries ─────────────────────

    def _create_collision_from_mesh(self):
        """Create a static rigid body whose collision shape matches the
        visual mesh.  Uses BulletTriangleMeshShape (static, mass-0) so
        the shape can be concave / non-box."""

        # Work on a *copy* so flattenStrong doesn't destroy the visual.
        collision_copy = self.visual.copyTo(render)
        collision_copy.flattenStrong()

        mesh = BulletTriangleMesh()
        # The visual IS a GeomNode, and '**' matches only descendants, so the
        # copy itself has to be included or the mesh comes out empty.
        geom_nps = list(collision_copy.findAllMatches('**/+GeomNode'))
        if isinstance(collision_copy.node(), GeomNode):
            geom_nps.append(collision_copy)
        for geom_np in geom_nps:
            geom_node = geom_np.node()
            ts = geom_np.getTransform(render)   # world space: keeps the piece's position
            for geom in geom_node.getGeoms():
                mesh.addGeom(geom, ts=ts)

        collision_copy.removeNode()  # no longer needed

        shape = BulletTriangleMeshShape(mesh, dynamic=False)

        body = BulletRigidBodyNode(f"terrain_body_{self.terrain_type}")
        body.addShape(shape)
        body.setMass(0)  # static — won't move

        self.ghost_np = render.attachNewNode(body)
        # No extra setPos needed — geometry is already in world space
        # after flattenStrong().

        mask = _TERRAIN_COLLISION_MASK.get(self.terrain_type, BitMask32.bit(20))
        self.ghost_np.setCollideMask(mask)
        self.game.world.attachRigidBody(body)

    # ── Queries ───────────────────────────────────────────────────────

    def contains(self, pos) -> bool:
        """Return *True* if world-space *pos* lies inside this piece.

        Rivers test distance to the meandering centreline; hills/forests test
        the metaball footprint field; everything else uses its AABB — so the
        gameplay region matches the visible shape.
        """
        if self.terrain_type == 'river' and self.river_centerline:
            return self._river_contains(pos)
        if self._field is not None:
            nx = (pos.x - self.center.x) / (self.width / 2.0)
            ny = (pos.y - self.center.y) / (self.height / 2.0)
            return self._field(nx, ny) >= self._field_edge
        return (abs(pos.x - self.center.x) <= self.width / 2 and
                abs(pos.y - self.center.y) <= self.height / 2)

    def _river_contains(self, pos) -> bool:
        px, py = pos.x, pos.y
        pts = self.river_centerline
        best = float('inf')
        best_w = 0.0
        for i in range(len(pts) - 1):
            ax, ay, aw = pts[i]
            bx, by, bw = pts[i + 1]
            dx, dy = bx - ax, by - ay
            seg2 = dx * dx + dy * dy
            t = 0.0 if seg2 < 1e-9 else ((px - ax) * dx + (py - ay) * dy) / seg2
            t = max(0.0, min(1.0, t))
            cxp, cyp = ax + dx * t, ay + dy * t
            d = math.hypot(px - cxp, py - cyp)
            if d < best:
                best = d
                best_w = aw + (bw - aw) * t
        return best <= best_w

    @property
    def movement_modifier(self) -> int:
        return self.category['movement_modifier']

    @property
    def blocks_line_of_sight(self) -> bool:
        return self.rules['blocks_line_of_sight']

    @property
    def is_dangerous(self) -> bool:
        return self.category['dangerous']

    @property
    def is_impassable(self) -> bool:
        return self.category['impassable']

    @property
    def disrupts(self) -> bool:
        """True if standing in this terrain costs a unit its Rank Bonus."""
        return self.category['disrupts']

    @property
    def description(self) -> str:
        return self.category['description']

    # ── Cleanup ───────────────────────────────────────────────────────

    def destroy(self):
        self.visual.removeNode()
        if self.outline is not None:
            self.outline.removeNode()
        if self.debug_np is not None:
            self.debug_np.removeNode()
            self.debug_np = None
        if getattr(self, 'trees_np', None) is not None:
            self.trees_np.removeNode()
            self.trees_np = None
        body = self.ghost_np.node()
        if hasattr(body, 'getMass'):
            # It's a rigid body (from _create_collision_from_mesh)
            self.game.world.removeRigidBody(body)
        else:
            # It's a ghost (from _create_collision)
            self.game.world.removeGhost(body)
        self.ghost_np.removeNode()


# ── TerrainManager ────────────────────────────────────────────────────────────

class TerrainManager:
    """Owns every terrain piece on the current battlefield."""

    def __init__(self, game):
        self.game = game
        self.terrain_pieces: list[TerrainPiece] = []

    # ── Add / remove ──────────────────────────────────────────────────

    def add_terrain(self, terrain_type: str, center: Point3,
                    width: float, height: float,
                    going: str = None) -> TerrainPiece:
        piece = TerrainPiece(terrain_type, center, width, height, self.game, going)
        self.terrain_pieces.append(piece)
        print(f"[Terrain] Added {terrain_type} at ({center.x:.0f}, "
              f"{center.y:.0f}) size {width:.0f}×{height:.0f}  "
              f"— {piece.description}")
        return piece

    def remove_terrain(self, piece):
        """Take a single piece off the board again, for terrain conjured
        mid-battle such as a Magical Vortex."""
        if piece in self.terrain_pieces:
            self.terrain_pieces.remove(piece)
            piece.destroy()

    def clear(self):
        for piece in self.terrain_pieces:
            piece.destroy()
        self.terrain_pieces.clear()

    def set_move_overlay(self, active, points=None):
        """Broadcast the movement/shooting range polygon to every terrain
        piece so the indicator wraps over hills/forests/water, not just the
        flat ground card."""
        for piece in self.terrain_pieces:
            if points is not None:
                piece.visual.setShaderInput("movePoints", points)
            piece.visual.setShaderInput("moveActive", active)

    # ── Debug ─────────────────────────────────────────────────────────

    def toggle_debug(self):
        """Show/hide the river water-detection band and print a report."""
        self.debug_visible = not getattr(self, 'debug_visible', False)
        for piece in self.terrain_pieces:
            if piece.debug_np is not None:
                (piece.debug_np.show if self.debug_visible else piece.debug_np.hide)()
        print(f"[Terrain] debug band {'ON' if self.debug_visible else 'OFF'} "
              f"({len(self.terrain_pieces)} pieces)")

    def debug_point(self, pos, label="") -> None:
        """Print which terrain a world-space point falls in (in-water check)."""
        hits = self.get_all_terrain_at(pos)
        types = ", ".join(t.terrain_type for t in hits) if hits else "open ground"
        in_water = any(t.terrain_type in ('river', 'marsh') for t in hits)
        print(f"[Terrain] {label} ({pos.x:.1f}, {pos.y:.1f}) -> {types} "
              f"| in_water={in_water} | M{self.get_movement_modifier(pos):+d}")

    # ── Queries ───────────────────────────────────────────────────────

    def get_terrain_at(self, pos) -> TerrainPiece | None:
        """Return the first terrain piece that contains *pos*, or ``None``."""
        for t in self.terrain_pieces:
            if t.contains(pos):
                return t
        return None

    def get_all_terrain_at(self, pos) -> list[TerrainPiece]:
        """Return every terrain piece overlapping *pos*."""
        return [t for t in self.terrain_pieces if t.contains(pos)]

    def get_movement_modifier(self, pos) -> int:
        """Movement characteristic modifier at *pos*.

        If the position overlaps several pieces the most restrictive (lowest)
        modifier wins; the modifiers do not stack.
        """
        terrains = self.get_all_terrain_at(pos)
        if not terrains:
            return 0
        return min(t.movement_modifier for t in terrains)

    def los_block_point(self, from_pos, to_pos):
        """Return the world point where line of sight is blocked along
        *from_pos*→*to_pos*, or None if unobstructed.

        Forests and hills block sight: you can see *onto* them (units within a
        wood, or on a hill's near slope/crest) but not through/over them to the
        dead ground behind.  A piece the shooter stands in/on never blocks.
        """
        dx = to_pos.x - from_pos.x
        dy = to_pos.y - from_pos.y
        length = math.hypot(dx, dy)
        steps = max(int(length / 0.5), 2)   # sample every ~0.5 world-units

        # A wood/hill the shooter occupies doesn't block its own line of sight.
        ignore = {id(p) for p in self.terrain_pieces
                  if p.blocks_line_of_sight and p.contains(from_pos)}

        entered = None        # the first blocking piece the sight line enters
        exit_point = None     # last sample still inside that piece (its far edge)
        for i in range(steps + 1):
            t = i / steps
            sample = Point3(from_pos.x + dx * t, from_pos.y + dy * t, 0)
            inside = False
            for p in self.terrain_pieces:
                if id(p) in ignore or not p.blocks_line_of_sight:
                    continue
                if p.contains(sample):
                    if entered is None:
                        entered = p
                    if p is entered:
                        inside = True
            if entered is not None:
                if inside:
                    exit_point = sample          # advance through the piece
                else:
                    return exit_point            # left it → block here
        # Sight line ended inside the piece (target within it) → not blocked.
        return None

    def get_surface_height(self, pos) -> float:
        """World-space Z of the raised terrain surface (hill/forest) at *pos*,
        or 0.0 on flat ground.  Used to sit unit models on the topography."""
        best = 0.0
        for t in self.terrain_pieces:
            if t._hf is not None and t.contains(pos):
                h = t._hf(pos.x - t.center.x, pos.y - t.center.y) + _HILL_LIFT
                if h > best:
                    best = h
        return best

    def get_surface_normal(self, pos) -> Vec3:
        """Surface normal of the raised terrain at *pos* (up on flat ground).
        Horizontal gradient is damped so models stay closer to upright for
        rank visual coherence."""
        tilt = 0.5   # <1 biases the normal toward vertical
        for t in self.terrain_pieces:
            if t._hf is not None and t.contains(pos):
                e = 0.3
                lx, ly = pos.x - t.center.x, pos.y - t.center.y
                dhx = t._hf(lx + e, ly) - t._hf(lx - e, ly)
                dhy = t._hf(lx, ly + e) - t._hf(lx, ly - e)
                n = Vec3(-dhx / (2.0 * e) * tilt, -dhy / (2.0 * e) * tilt, 1.0)
                n.normalize()
                return n
        return Vec3(0, 0, 1)

    def get_terrain_between(self, pos_a, pos_b) -> list[TerrainPiece]:
        """Return terrain pieces whose AABB is crossed by the line
        *pos_a* → *pos_b* (sampled at small intervals)."""
        hit: set[int] = set()
        result: list[TerrainPiece] = []
        dx = pos_b.x - pos_a.x
        dy = pos_b.y - pos_a.y
        length = math.sqrt(dx * dx + dy * dy)
        steps = max(int(length / 1.0), 4)  # sample every ~1 world-unit
        for i in range(steps + 1):
            t = i / steps
            sample = Point3(pos_a.x + dx * t, pos_a.y + dy * t, 0)
            for idx, terrain in enumerate(self.terrain_pieces):
                if idx not in hit and terrain.contains(sample):
                    hit.add(idx)
                    result.append(terrain)
        return result

    def dangerous_between(self, pos_a, pos_b) -> list[TerrainPiece]:
        """Dangerous pieces on the path. A model tests once per feature it
        meets, so each piece is listed once (Rulebook p. 269)."""
        return [t for t in self.get_terrain_between(pos_a, pos_b) if t.is_dangerous]

    def crosses_difficult(self, pos_a, pos_b) -> bool:
        """True if the path meets terrain that hinders movement. Dangerous
        terrain hinders it just like difficult terrain."""
        return any(t.movement_modifier < 0
                   for t in self.get_terrain_between(pos_a, pos_b))

    def is_charge_allowed(self, from_pos, to_pos) -> bool:
        """Return *False* if impassable terrain lies on the charge path."""
        return not any(t.is_impassable
                       for t in self.get_terrain_between(from_pos, to_pos))

    # ── Serialisation ─────────────────────────────────────────────────

    def load_from_json(self, filepath: str):
        """Load terrain layout from a JSON file.

        ``going`` is optional and overrides the category the terrain type
        presents by default, so the same wood can be difficult on one map and
        dangerous on another::

            {
                "terrain": [
                    {"type": "forest", "center": [0, 5, 0], "width": 15,
                     "height": 10, "going": "dangerous"},
                    ...
                ]
            }
        """
        with open(filepath) as f:
            data = json.load(f)
        for entry in data['terrain']:
            self.add_terrain(
                entry['type'],
                Point3(*entry['center']),
                entry['width'],
                entry['height'],
                entry.get('going'),
            )

    def save_to_json(self, filepath: str):
        data = {
            'terrain': [
                {
                    'type': t.terrain_type,
                    'center': [t.center.x, t.center.y, t.center.z],
                    'width': t.width,
                    'height': t.height,
                    'going': t.going,
                }
                for t in self.terrain_pieces
            ]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
