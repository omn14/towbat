"""
Centralised Bullet / Panda3D collision-mask constants.

Every BitMask32 used in the project is defined here so nothing is a
magic number.  Import the masks you need and pass them straight to
setCollideMask(), rayTestClosest(), sweepTestClosest(), etc.

Usage
-----
    from collision_masks import CollisionMask as CM

    # Setting what a node can be *hit by* (its "into" mask)
    unit.bodyNP.setCollideMask(CM.UNIT_DEFAULT)

    # Querying – which masks to *look for* (the "from" mask)
    result = world.rayTestClosest(pFrom, pTo, CM.UNIT_DEFAULT)

Combining masks
---------------
    # A unit that is both selectable AND sweep-detectable:
    unit.bodyNP.setCollideMask(CM.UNIT_DEFAULT | CM.SWEEP_TARGET)

    # A ray that should hit units OR the board:
    result = world.rayTestClosest(p1, p2, CM.UNIT_DEFAULT | CM.BOARD)
"""

from panda3d.core import BitMask32


class CollisionMask:
    """Named collision-mask constants.

    Panda3D's BitMask32 gives us bits 0-31.  Each gameplay system
    gets its own bit so masks never accidentally overlap.

    The *into* mask lives on the node  (setCollideMask).
    The *from* mask is passed to the query (rayTest / sweepTest).

    A collision is detected when  (from_mask & into_mask) != 0.
    """

    # ── unit / selection ────────────────────────────────────────
    UNIT_DEFAULT     = BitMask32.bit(1)   # units at rest – picked by mouse rays
    DEPLOY_ZONE      = BitMask32.bit(2)   # free: was the floating phase cube
    ARROW_DEFAULT    = BitMask32.bit(3)   # projectile / arrow collision
    COMBAT_ENGAGED   = BitMask32.bit(4)   # units locked in melee
    ARROW_ALT        = BitMask32.bit(5)   # alternate arrow check pass
    OPPONENT_UNIT    = BitMask32.bit(7)   # the non-active player's units

    # ── movement / sweep tests ──────────────────────────────────
    SWEEP_TARGET     = BitMask32.bit(9)   # "other units" during a sweep test
    SWEEP_SELF       = BitMask32.bit(30)  # current unit excluded from sweep

    # ── menus ───────────────────────────────────────────────────
    MENU_CHOICE      = BitMask32.bit(29)  # the floating choice-menu cubes

    # ── environment ─────────────────────────────────────────────
    BOUNDARY         = BitMask32.bit(11)  # table-edge boundary walls
    OUT_OF_BOUNDS    = BitMask32.bit(31)  # the four board-edge ghost volumes
    TERRAIN_FOREST          = BitMask32.bit(20)  # terrain ghost nodes (forest, hill…)
    TERRAIN_HILL            = BitMask32.bit(21)  # terrain ghost nodes (hill)
    TERRAIN_RIVER           = BitMask32.bit(22)  # terrain ghost nodes (river)
    TERRAIN_MARSH           = BitMask32.bit(23)  # terrain ghost nodes (marsh)
    TERRAIN_IMPASSABLE      = BitMask32.bit(24)  # terrain that stops movement (houses…)
    BOARD            = BitMask32.bit(1)   # the game board (shares bit 1 today)

    # ── convenience combos ──────────────────────────────────────
    NOTHING          = BitMask32.allOff()
    EVERYTHING       = BitMask32.allOn()

    # What a *unit at rest* should be hittable by:
    #   mouse-pick rays (UNIT_DEFAULT) + sweep tests (SWEEP_TARGET)
    UNIT_INTO        = UNIT_DEFAULT | SWEEP_TARGET          # bits 1 + 9

    # What a *unit in combat* should be hittable by:
    COMBAT_INTO      = COMBAT_ENGAGED                       # bit 4 only

    # What a mouse-hover pick ray should hit: any unit (active, opponent or in
    # combat) but NOT deploy-zone cubes, boundary or terrain.
    HOVER_PICK       = UNIT_DEFAULT | COMBAT_ENGAGED | OPPONENT_UNIT  # bits 1 + 4 + 7

    # What a *terrain piece* should be hittable by (terrain queries only):
    TERRAIN_INTO     = (TERRAIN_FOREST | TERRAIN_HILL | TERRAIN_RIVER
                        | TERRAIN_MARSH | TERRAIN_IMPASSABLE)      # bits 20-24

    # What stops a moving unit: other units, and terrain it cannot cross.
    MOVE_BLOCKERS    = SWEEP_TARGET | TERRAIN_IMPASSABLE     # bits 9 + 24

    # ── helpers ─────────────────────────────────────────────────
    @staticmethod
    def combine(*masks: BitMask32) -> BitMask32:
        """OR together an arbitrary number of masks.

        >>> CM.combine(CM.UNIT_DEFAULT, CM.SWEEP_TARGET, CM.BOUNDARY)
        """
        result = BitMask32.allOff()
        for m in masks:
            result = result | m
        return result

    @staticmethod
    def has(mask: BitMask32, flag: BitMask32) -> bool:
        """Test whether *flag* is present in *mask*.

        >>> CM.has(unit.bodyNP.getCollideMask(), CM.SWEEP_TARGET)
        True
        """
        return (mask & flag) != BitMask32.allOff()

    @staticmethod
    def describe(mask: BitMask32) -> str:
        """Return a human-readable list of active bits for debugging.

        >>> CM.describe(unit.bodyNP.getCollideMask())
        'UNIT_DEFAULT | SWEEP_TARGET'
        """
        names = {
            1:  "UNIT_DEFAULT / BOARD",
            2:  "DEPLOY_ZONE",
            3:  "ARROW_DEFAULT",
            4:  "COMBAT_ENGAGED",
            5:  "ARROW_ALT",
            9:  "SWEEP_TARGET",
            11: "BOUNDARY",
            20: "TERRAIN_FOREST",
            21: "TERRAIN_HILL",
            22: "TERRAIN_RIVER",
            23: "TERRAIN_MARSH",
            30: "SWEEP_SELF",
        }
        active = []
        for bit in range(32):
            if mask.getBit(bit):
                active.append(names.get(bit, f"bit({bit})"))
        return " | ".join(active) if active else "NOTHING"
