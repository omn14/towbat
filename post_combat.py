"""What happens once the Break tests are made (Rulebook p. 154-157).

The Break test itself lives in `psychology.break_test_outcome`; this module
turns its three outcomes into moves, and answers the questions the winners have
to answer before the losers move.

Positions arrive as plain ``(x, y)`` pairs and dice as plain lists, so none of
this needs a window or a physics world to test. Anything that has to touch a
NodePath belongs in `combat_resolution`.
"""

import math
import random

from psychology import GIVE_GROUND
from special_rules import charge_roll

__all__ = [
    'GIVE_GROUND', 'flee_roll', 'fall_back_roll', 'pursuit_roll',
    'flees_from', 'flee_direction', 'give_ground_direction', 'facing_vector',
    'restraint_test', 'winner_response', 'catch_outcome', 'may_pursue',
    'nearest_corner',
]


def nearest_corner(corners, half_x: float, half_y: float) -> int:
    """Index of the corner closest to a base of half extents *half_x* by
    *half_y*, everything in that base's own frame.

    A unit that runs in at an angle meets the enemy corner-first, and that
    corner is the point it pivots about. Pivoting about anything else -- the
    middle of the edge, the point under their centre -- swings the corner off
    the enemy and opens a gap.
    """
    def distance(p):
        dx, dy = abs(p[0]) - half_x, abs(p[1]) - half_y
        if dx > 0 or dy > 0:
            return math.hypot(max(dx, 0.0), max(dy, 0.0))
        return max(dx, dy)

    return min(range(len(corners)), key=lambda i: distance(corners[i]))


# ─── Distances ────────────────────────────────────────────────────────────

def flee_roll(dice, already_fled: bool = False) -> int:
    """A Flee roll is 2D6 summed (p. 132).

    Swiftstride's bonus die is one of *dice* and is added like the rest -- a
    Flee roll discards nothing.

    The Limits of Endurance (p. 133): a unit only ever makes one flee move in a
    phase, so a second one covers 0" and does not pivot.
    """
    return 0 if already_fled else sum(dice)


def fall_back_roll(dice, already_fled: bool = False) -> int:
    """Fall Back in Good Order rolls 2D6 and discards the lowest (p. 134).

    That is the Charge roll's arithmetic, Swiftstride's added die included, so
    it is the same function rather than a second copy of it. A Fall Back moves
    "exactly like a fleeing unit", so it is a flee move for the purposes of The
    Limits of Endurance and it is spent by one.
    """
    return 0 if already_fled else charge_roll(dice)


def pursuit_roll(dice) -> int:
    """A Pursuit roll is 2D6 summed, with no Movement added (p. 156)."""
    return sum(dice)


# ─── Direction ────────────────────────────────────────────────────────────

def _away(from_xy, to_xy):
    """Unit vector pointing from *to_xy* towards *from_xy*."""
    dx = from_xy[0] - to_xy[0]
    dy = from_xy[1] - to_xy[1]
    length = math.hypot(dx, dy)
    if length == 0:
        return (0.0, 0.0)
    return (dx / length, dy / length)


def flees_from(candidates, rng=None):
    """The single enemy a Break or Fall Back move runs directly away from.

    Not the average of the enemies engaging it: the unit flees from the one
    with the highest Unit Strength, chosen at random between equals (The
    Greater the Danger, p. 133; Loser Breaks & Flees, p. 154).

    *candidates* is a sequence of ``(unit, unit_strength)``.
    """
    if not candidates:
        return None
    best = max(us for _, us in candidates)
    tied = [unit for unit, us in candidates if us == best]
    if len(tied) == 1:
        return tied[0]
    return (rng or random).choice(tied)


def flee_direction(loser_xy, winner_xy):
    """Directly away from the one winner that broke the unit (p. 154)."""
    return _away(loser_xy, winner_xy)


def facing_vector(heading_deg: float):
    """The way a unit is pointing, as a unit vector.

    Panda3D's heading turns anticlockwise from +Y, which is the convention the
    charge code uses to walk a unit forward. An Overrun moves along this and
    nothing else -- it may not pivot (p. 156).
    """
    h = math.radians(heading_deg)
    return (-math.sin(h), math.cos(h))


def give_ground_direction(loser_xy, winner_xys):
    """As directly as possible away from *every* unit engaging it (p. 155).

    Deliberately a different rule from a flee direction: a unit held in two
    arcs gives ground diagonally, away from both at once, rather than turning
    its back on the biggest of them.
    """
    total_x = total_y = 0.0
    for winner_xy in winner_xys:
        dx, dy = _away(loser_xy, winner_xy)
        total_x += dx
        total_y += dy
    length = math.hypot(total_x, total_y)
    if length == 0:
        return (0.0, 0.0)
    return (total_x / length, total_y / length)


# ─── The winners' choice ──────────────────────────────────────────────────

def winner_response(loser_outcome: str) -> str:
    """Which move answers each Break test result (p. 156).

    A Follow Up answers a unit that Gives Ground; a Pursuit answers one that
    Falls Back in Good Order or Breaks.
    """
    return 'follow_up' if loser_outcome == 'give_ground' else 'pursue'


def restraint_test(ld: int, dice) -> bool:
    """Restrain & Reform (p. 156): a Leadership test, not a free choice.

    Passing holds the unit where it is and earns a free reform; failing forces
    the follow up or the pursuit whether the player wants it or not.
    """
    return sum(dice) <= ld


def may_pursue(still_in_base_contact: bool) -> bool:
    """Still Engaged (p. 156): a unit still in base contact with an enemy can
    neither follow up nor pursue."""
    return not still_in_base_contact


def catch_outcome(loser_outcome: str) -> str:
    """What catching the pursued unit means (Catching the Curs!, p. 157).

    A fleeing unit is run down and removed; one that Fell Back in Good Order is
    merely caught, and the two are locked together with the pursuer counting as
    having charged next turn.
    """
    return 'destroyed' if loser_outcome == 'break' else 'engaged'
