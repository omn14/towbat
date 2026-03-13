"""Terrain system — terrain pieces, placement, and rule queries.

Each ``TerrainPiece`` represents a rectangular area on the battlefield with
associated game-play rules (movement penalties, line-of-sight blocking,
combat modifiers, etc.).  The ``TerrainManager`` owns every piece and
exposes simple query helpers consumed by MovementSystem, CombatResolver,
and the AI subsystems.
"""

import json
import math

from panda3d.core import (
    Point3, Vec3, Vec4, BitMask32,
    TransparencyAttrib, NodePath, LineSegs,
)
from panda3d.bullet import BulletGhostNode, BulletBoxShape


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

        #self._create_visual()
        self._create_outline()
        self._create_collision()

    # ── Visual: translucent box on the ground ───────────────────────

    def _create_visual(self):
        box_height = 1.0  # how tall the terrain box appears
        self.visual = loader.loadModel("models/box")
        self.visual.reparentTo(render)
        # The built-in box is a 2×2×2 cube centred at origin.
        # Scale it to match terrain width/height and desired box_height.
        self.visual.setScale(self.width , self.height , box_height )
        self.visual.setPos(self.center.x-self.width/2, self.center.y-self.height/2, box_height / 2)
        color = _TERRAIN_COLORS.get(
            self.terrain_type, Vec4(0.5, 0.5, 0.5, 0.4)
        )
        self.visual.setColor(color)
        self.visual.setTransparency(TransparencyAttrib.MAlpha)
        self.visual.setDepthWrite(False)
        self.visual.setBin('fixed', 0)  # render after opaque units but before UI

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

    # ── Queries ───────────────────────────────────────────────────────

    def contains(self, pos) -> bool:
        """Return *True* if world-space *pos* lies inside this piece (AABB)."""
        return (abs(pos.x - self.center.x) <= self.width / 2 and
                abs(pos.y - self.center.y) <= self.height / 2)

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
        self.outline.removeNode()
        self.game.world.removeGhost(self.ghost)
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
