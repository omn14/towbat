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


# ── Terrain rule definitions ──────────────────────────────────────────────────

TERRAIN_RULES = {
    'forest': {
        'movement_multiplier': 0.5,     # half movement
        'blocks_line_of_sight': True,
        'combat_modifier': 0,
        'charge_allowed': True,
        'formation_break': True,        # units lose rank bonuses
        'description': 'Difficult terrain — halves movement rate',
    },
    'hill': {
        'movement_multiplier': 1.0,     # no penalty
        'blocks_line_of_sight': False,   # but gives LoS advantage
        'combat_modifier': 1,           # higher-ground advantage
        'charge_allowed': True,
        'formation_break': False,
        'description': 'Open terrain — +1 combat res when defending',
    },
    'river': {
        'movement_multiplier': 0.5,
        'blocks_line_of_sight': False,
        'combat_modifier': -1,
        'charge_allowed': False,
        'formation_break': True,
        'description': 'Water — halves movement, no charges across',
    },
    'marsh': {
        'movement_multiplier': 0.25,
        'blocks_line_of_sight': False,
        'combat_modifier': -1,
        'charge_allowed': False,
        'formation_break': True,
        'description': 'Very difficult terrain — quarter movement rate',
    },
}

# Colours used to tint the terrain overlay (RGBA, alpha < 1 → translucent)
_TERRAIN_COLORS = {
    'forest': Vec4(0.15, 0.55, 0.15, 0.50),
    'hill':   Vec4(0.50, 0.40, 0.20, 0.35),
    'river':  Vec4(0.10, 0.20, 0.60, 0.45),
    'marsh':  Vec4(0.30, 0.30, 0.10, 0.40),
}

_TERRAIN_COLLISION_MASK = {
    'forest': BitMask32.bit(20),
    'hill':   BitMask32.bit(21),
    'river':  BitMask32.bit(22),
    'marsh':  BitMask32.bit(23),
}

_TERRAIN_MODEL_PATHS = {
    'forest': "models/hills.bam",
    'hill':   "models/hills.bam",
    'river':  "models/hills.bam",
    'marsh':  "models/hills.bam",
}

# Integer id handed to the terrain shader (see shaders/terrain.frag).
_TERRAIN_TYPE_ID = {'forest': 0, 'hill': 1, 'river': 2, 'marsh': 3}

# Water is drawn as a flat animated surface instead of a raised mesh.
_WATER_TYPES = {'river', 'marsh'}

# Vertical scale per type — hills rise, forests sit low, water is nearly flat.
_TERRAIN_HEIGHT = {'forest': 2.0, 'hill': 4.0, 'river': 0.3, 'marsh': 0.4}

# Cached, shared GLSL shader instance (loaded once on first use).
_TERRAIN_SHADER = None


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


# ── TerrainPiece ──────────────────────────────────────────────────────────────

class TerrainPiece:
    """A single rectangular terrain feature on the battlefield."""

    def __init__(self, terrain_type: str, center: Point3,
                 width: float, height: float, game):
        if terrain_type not in TERRAIN_RULES:
            raise ValueError(
                f"Unknown terrain type '{terrain_type}'. "
                f"Valid types: {list(TERRAIN_RULES.keys())}"
            )
        self.terrain_type = terrain_type
        self.rules = TERRAIN_RULES[terrain_type]
        self.center = center
        self.width = width
        self.height = height
        self.game = game

        self.river_centerline = None   # populated for river pieces
        self.debug_np = None

        self._create_visual()
        # Water reads as a natural shape; a rectangle outline would box it in.
        self.outline = None
        if self.terrain_type not in _WATER_TYPES:
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
        else:
            self._create_mesh_visual()
        self._apply_shader()

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
            off = amp * (math.sin(s * tau * 1.5 + 0.7) * 0.6 +
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
            vw.addData3(cx + px * w, cy + py * w, 0.0)   # left bank (v=0)
            nw.addData3(0.0, 0.0, 1.0)
            tw.addData2(s, 0.0)
            vw.addData3(cx - px * w, cy - py * w, 0.0)   # right bank (v=1)
            nw.addData3(0.0, 0.0, 1.0)
            tw.addData2(s, 1.0)

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
            placeholder = self.trees_np.attachNewNode("tree")
            tree_model.instanceTo(placeholder)
            placeholder.setPos(x, y, 0.0)
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
        for geom_np in collision_copy.findAllMatches('**/+GeomNode'):
            geom_node = geom_np.node()
            ts = geom_np.getTransform(collision_copy)
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

        Rivers test the distance to the meandering centreline so the water
        region matches the visible ribbon; everything else uses its AABB.
        """
        if self.terrain_type == 'river' and self.river_centerline:
            return self._river_contains(pos)
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
    def movement_multiplier(self) -> float:
        return self.rules['movement_multiplier']

    @property
    def blocks_line_of_sight(self) -> bool:
        return self.rules['blocks_line_of_sight']

    @property
    def combat_modifier(self) -> int:
        return self.rules['combat_modifier']

    @property
    def charge_allowed(self) -> bool:
        return self.rules['charge_allowed']

    @property
    def formation_break(self) -> bool:
        return self.rules['formation_break']

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
                    width: float, height: float) -> TerrainPiece:
        piece = TerrainPiece(terrain_type, center, width, height, self.game)
        self.terrain_pieces.append(piece)
        print(f"[Terrain] Added {terrain_type} at ({center.x:.0f}, "
              f"{center.y:.0f}) size {width:.0f}×{height:.0f}  "
              f"— {piece.rules['description']}")
        return piece

    def clear(self):
        for piece in self.terrain_pieces:
            piece.destroy()
        self.terrain_pieces.clear()

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
              f"| in_water={in_water} | move×{self.get_movement_multiplier(pos):.2f}")

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

    def get_movement_multiplier(self, pos) -> float:
        """Return the effective movement multiplier at *pos*.

        If the position overlaps multiple terrain pieces the most
        restrictive (lowest) multiplier wins.
        """
        terrains = self.get_all_terrain_at(pos)
        if not terrains:
            return 1.0
        return min(t.movement_multiplier for t in terrains)

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

    def is_charge_allowed(self, from_pos, to_pos) -> bool:
        """Return *False* if any terrain on the charge path blocks charges."""
        for t in self.get_terrain_between(from_pos, to_pos):
            if not t.charge_allowed:
                return False
        return True

    # ── Serialisation ─────────────────────────────────────────────────

    def load_from_json(self, filepath: str):
        """Load terrain layout from a JSON file.

        Expected format::

            {
                "terrain": [
                    {"type": "forest", "center": [0, 5, 0], "width": 15, "height": 10},
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
            )

    def save_to_json(self, filepath: str):
        data = {
            'terrain': [
                {
                    'type': t.terrain_type,
                    'center': [t.center.x, t.center.y, t.center.z],
                    'width': t.width,
                    'height': t.height,
                }
                for t in self.terrain_pieces
            ]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
