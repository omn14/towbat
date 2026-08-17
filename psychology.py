"""Psychology subsystem — Panic tests (Phase 0).

The Old World psychology rules (Rulebook p. 160-161) all resolve into a
Leadership test that, on failure, makes a unit Fall Back in Good Order or Flee.
This module provides the reusable Panic test and its pure helpers; the causes
(heavy casualties, nearby friend destroyed/flees, fled through) are wired in
later phases and simply call ``PsychologySystem.panic_test``.

Pure functions (``leadership_test``, ``leadership_test_with_reroll``,
``panic_fail_outcome``, ``unit_strength_total``) are free of Panda3D state so
they can be unit-tested.
"""

from __future__ import annotations

import random
import math

from panda3d.core import Vec3, Point3
from direct.interval.LerpInterval import LerpPosInterval
from direct.interval.IntervalGlobal import Sequence
from direct.interval.FunctionInterval import Func


# ── Oriented-box (footprint) geometry ──────────────────────────────────────
# A unit box is (cx, cy, half_width, half_depth, heading_degrees).

def _box_corners(cx, cy, hx, hy, ang_deg):
    a = math.radians(ang_deg)
    ca, sa = math.cos(a), math.sin(a)
    corners = []
    for sx, sy in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)):
        corners.append((cx + sx * ca - sy * sa, cy + sx * sa + sy * ca))
    return corners


def _polys_overlap(a, b):
    """Separating-Axis test for two convex polygons (True if they overlap)."""
    for poly in (a, b):
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            nx, ny = -(y2 - y1), (x2 - x1)   # edge normal (need not be unit)
            amin = min(px * nx + py * ny for px, py in a)
            amax = max(px * nx + py * ny for px, py in a)
            bmin = min(px * nx + py * ny for px, py in b)
            bmax = max(px * nx + py * ny for px, py in b)
            if amax < bmin or bmax < amin:
                return False                  # found a separating axis
    return True


def _seg_point_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    if l2 == 0.0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / l2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _poly_edge_dist(a, b):
    best = float('inf')
    for (px, py) in a:
        n = len(b)
        for i in range(n):
            bx1, by1 = b[i]
            bx2, by2 = b[(i + 1) % n]
            best = min(best, _seg_point_dist(px, py, bx1, by1, bx2, by2))
    return best


def obb_distance(box_a, box_b) -> float:
    """Closest distance between two oriented footprint boxes (0 if overlapping).
    Each box is (cx, cy, half_width, half_depth, heading_degrees)."""
    a = _box_corners(*box_a)
    b = _box_corners(*box_b)
    if _polys_overlap(a, b):
        return 0.0
    return min(_poly_edge_dist(a, b), _poly_edge_dist(b, a))


def _stat_int(characteristics: dict, key: str, default: int = 0) -> int:
    try:
        return int(characteristics.get(key))
    except (KeyError, TypeError, ValueError):
        return default


def leadership_test(ld: int, modifier: int = 0):
    """Roll 2D6 against Leadership (+modifier). Returns ``(passed, roll)``."""
    roll = random.randint(1, 6) + random.randint(1, 6)
    return roll <= (ld + modifier), roll


def leadership_test_with_reroll(ld: int, modifier: int = 0, reroll: bool = False):
    """Leadership test that may re-roll a failure (e.g. Venerable).

    Returns ``(passed, rolls)`` where *rolls* holds every 2D6 result made, so
    the caller can report the re-roll.  Only one re-roll is ever made.
    """
    passed, roll = leadership_test(ld, modifier)
    rolls = [roll]
    if not passed and reroll:
        passed, roll = leadership_test(ld, modifier)
        rolls.append(roll)
    return passed, rolls


def panic_fail_outcome(remaining: int, start_of_battle: int) -> str:
    """Outcome of a *failed* Panic test.

    More than 50% of the start-of-battle models remain -> 'fall_back',
    otherwise (50% or fewer remain) -> 'flee'.
    """
    if start_of_battle <= 0:
        return 'flee'
    return 'fall_back' if remaining * 2 > start_of_battle else 'flee'


def unit_strength_total(unit) -> int:
    """Total Unit Strength: per-model US x current models."""
    return unit.unit.model.unit_strength() * max(0, unit.unit.nmodels)


def heavy_casualties(remaining: int, start_of_phase: int) -> bool:
    """True if more than 25% of the start-of-phase models were lost."""
    if start_of_phase <= 0:
        return False
    return (start_of_phase - remaining) * 4 > start_of_phase


def is_skirmish_unit(unit) -> bool:
    """True if *unit* fights in a Skirmish formation."""
    if unit is None:
        return False
    if getattr(unit, 'isSkirmisher', False):
        return True
    model = getattr(getattr(unit, 'unit', None), 'model', None)
    check = getattr(model, 'is_skirmisher', None)
    return bool(check()) if callable(check) else False


def fled_through_panics(fleer_skirmish: bool, target_skirmish: bool) -> bool:
    """Whether a unit fled through must take a Panic test (Rulebook p. 185).

    Skirmishers do not cause Panic in *formed* friendly units they flee through;
    they still panic friendly Skirmishers (and still cause Panic as normal when
    annihilated or when they Break and flee).
    """
    return (not fleer_skirmish) or target_skirmish


def is_venerable_unit(unit) -> bool:
    """True if *unit* carries the Venerable special rule."""
    if unit is None:
        return False
    if getattr(unit, 'isVenerable', False):
        return True
    model = getattr(getattr(unit, 'unit', None), 'model', None)
    check = getattr(model, 'is_venerable', None)
    return bool(check()) if callable(check) else False


# Units of this Unit Strength or more cause nearby-friend Panic when destroyed
# or when they flee/fall back from combat.
PANIC_US_THRESHOLD = 5
# Friendly units within this range (inches) must test.
PANIC_RADIUS = 6.0
# Venerable grants its Panic re-roll to friendly units within the same 6" bubble.
VENERABLE_RADIUS = PANIC_RADIUS

# Special-rule names that make a unit exempt from Panic (full exemption).
_FULL_PANIC_IMMUNE = ('ignore panic', 'immune to psychology')


class PsychologySystem:
    """Panic tests and (later) the rest of the psychology rules."""

    def __init__(self, game):
        self.game = game
        # Panic tests are resolved one unit at a time: a unit must finish its
        # move (and rally/reform) before the next queued test starts.
        self._panic_queue = []
        self._panic_active = False
        # While held (e.g. during combat resolution) tests queue but do not run
        # until released, so their moves/reforms don't clash with combat choices.
        self._panic_hold = False

    # ─── Exemptions (No Need for Hysterics) ───────────────────────────────

    def panic_exempt_reason(self, unit):
        """Return why *unit* need not take a Panic test, or None if it must.
        A unit is exempt if it has already tested this phase, is making a charge
        move, is fleeing or engaged in combat, or is rule-immune to Panic."""
        if getattr(unit, 'panicTestedThisPhase', False):
            return "already tested this phase"
        if getattr(unit, 'isChargingMove', False):
            return "making a charge move"
        if getattr(unit, 'isInCombat', False):
            return "engaged in combat"
        if unit.state == 'IsFleeing':
            return "already fleeing"
        for r in unit.unit.model.special_rules:
            if not isinstance(r, dict):
                continue
            if r.get('Unbreakable'):
                return "Unbreakable"
            name = str(r.get('name', '')).lower()
            if any(k in name for k in _FULL_PANIC_IMMUNE):
                return r.get('name')
        return None

    def is_panic_exempt(self, unit) -> bool:
        return self.panic_exempt_reason(unit) is not None

    # ─── Panic test (queued, sequential) ──────────────────────────────────

    def panic_test(self, unit, flee_from=None, cause: str = ""):
        """Queue a Panic test for *unit*.  Tests resolve one at a time: each
        unit completes its move (and rally/reform) before the next starts, so
        a fled-through / nearby unit only begins after the first unit rallies."""
        if unit is None or unit.bodyNP.isEmpty():
            return
        self._panic_queue.append((unit, flee_from, cause))
        if not self._panic_active and not self._panic_hold:
            self._run_next_panic()

    def hold_panic(self):
        """Queue Panic tests without resolving them (used during combat)."""
        self._panic_hold = True

    def release_panic(self):
        """Resume resolving queued Panic tests (after combat has finished)."""
        self._panic_hold = False
        if not self._panic_active and self._panic_queue:
            self._run_next_panic()

    def _run_next_panic(self):
        if not self._panic_queue:
            self._panic_active = False
            return
        self._panic_active = True
        unit, flee_from, cause = self._panic_queue.pop(0)
        self._resolve_panic(unit, flee_from, cause, self._run_next_panic)

    def _resolve_panic(self, unit, flee_from, cause, on_done):
        """Test *unit*; on failure move it (flee / fall back) and call *on_done*
        only once the whole move (and rally/reform) has finished."""
        if unit is None or unit.bodyNP.isEmpty():
            on_done()
            return
        reason = self.panic_exempt_reason(unit)
        if reason is not None:
            print(f"[Panic] {unit.unit.name} is exempt from Panic ({cause}): {reason}.")
            on_done()
            return

        unit.panicTestedThisPhase = True
        ld = _stat_int(unit.unit.model.characteristics, 'Ld', 7)
        venerable = self.venerable_source(unit)
        passed, rolls = leadership_test_with_reroll(ld, reroll=venerable is not None)
        roll = rolls[-1]
        if len(rolls) > 1:
            print(f"[Panic] {unit.unit.name} re-rolls a failed Panic test "
                  f"(Venerable: {venerable.unit.name} within "
                  f"{self.VENERABLE_RADIUS:.0f}\"): 2D6={rolls[0]} -> {rolls[-1]}")
        remaining = unit.unit.nmodels
        start = getattr(unit, 'startOfBattleModels', remaining) or remaining
        pct = 100.0 * remaining / max(1, start)
        print(f"[Panic] {unit.unit.name}: Ld {ld} vs 2D6={roll} -> "
              f"{'PASS' if passed else 'FAIL'}  ({cause}; {remaining}/{start} "
              f"models, {pct:.0f}% remain)")
        if passed:
            on_done()
            return

        enemy = flee_from or self.nearest_non_fleeing_enemy(unit)
        direction = self._flee_direction(unit, enemy)
        outcome = panic_fail_outcome(remaining, start)
        enemy_name = enemy.unit.name if enemy is not None else "board edge"
        up = unit.bodyNP.getPos()
        ep = enemy.bodyNP.getPos() if enemy is not None else None
        eps = f"({ep.x:.0f},{ep.y:.0f})" if ep is not None else "n/a"
        print(f"[Panic] {unit.unit.name} fails -> {outcome} away from {enemy_name} | "
              f"unit=({up.x:.0f},{up.y:.0f}) enemy={eps} "
              f"dir=({direction.x:.2f},{direction.y:.2f})")

        # A fleeing unit's Flee roll is 2D6; Fall Back in Good Order discards
        # the lowest (Rulebook p. 134).
        d1, d2 = random.randint(1, 6), random.randint(1, 6)
        distance = max(d1, d2) if outcome == 'fall_back' else d1 + d2
        self._start_flee_move(unit, direction, distance, outcome, on_done)

    def _start_flee_move(self, unit, direction: Vec3, distance: float, outcome, on_done):
        """Animate the flee/fall-back move, then chain the fled-through tests
        (queued) and, for a fall back, the rally + free reform — calling
        *on_done* only when this unit's whole sequence is finished."""
        start = unit.bodyNP.getPos()
        final, passed = self._flee_until_clear(unit, start, direction, distance)
        final.z = start.z
        # Face the flee direction (heading only — keep the body upright).
        unit.bodyNP.lookAt(Point3(start.x + direction.x, start.y + direction.y, start.z))
        unit.bodyNP.setP(0)
        unit.bodyNP.setR(0)
        label = "falls back" if outcome == 'fall_back' else "flees"
        print(f"[Panic] {unit.unit.name} {label} {distance:.0f}\" from "
              f"({start.x:.0f},{start.y:.0f}) -> ({final.x:.0f},{final.y:.0f})")
        if passed:
            print(f"[Panic] {unit.unit.name} fled through: "
                  f"{', '.join(p.unit.name for p in passed)}")
        unit.hasMovedThisTurn = True
        if outcome == 'flee':
            unit.request("IsFleeing")

        def after_move(task=None):
            unit.updateTextNode()
            if outcome == 'fall_back':
                # Auto-rally: regain composure, cannot charge this turn.
                unit.request("Idle")
                unit.cannotChargeThisTurn = True
                unit.attemptedRallyThisTurn = True
                unit.updateTextNode()
                print(f"[Panic] {unit.unit.name} Falls Back in Good Order and "
                      f"rallies — free reform (cannot charge this turn).")
                self.game.startFreeReform(unit, on_done=lambda: self._after_unit_done(unit, passed, on_done))
            else:
                print(f"[Panic] {unit.unit.name} Flees!")
                self._after_unit_done(unit, passed, on_done)

        interval = LerpPosInterval(unit.bodyNP, 1.0, final, blendType='easeInOut')
        Sequence(interval, Func(after_move)).start()

    def _after_unit_done(self, unit, passed, on_done):
        """Queue the fled-through units' Panic tests (they run after this unit,
        in order), then hand back to the queue.  Only friendly units test, and
        Skirmishers do not panic formed friendlies they flee through."""
        fleer_skirmish = is_skirmish_unit(unit)
        friends = self._friendlies_of(unit)
        for other in passed:
            if other is unit or other.bodyNP.isEmpty():
                continue
            if other not in friends:
                print(f"[Panic] {unit.unit.name} fled through enemy "
                      f"{other.unit.name} — no Panic test (friendly units only).")
                continue
            if not fled_through_panics(fleer_skirmish, is_skirmish_unit(other)):
                print(f"[Panic] Skirmishers {unit.unit.name} fled through formed "
                      f"{other.unit.name} — no Panic test (Skirmishers & Panic).")
                continue
            self._panic_queue.append((other, None, "fled through"))
        on_done()

    # ─── Common causes of Panic ───────────────────────────────────────────

    def _friendlies_of(self, unit):
        return (self.game.player1Units if unit in self.game.player1Units
                else self.game.player2Units)

    @staticmethod
    def _unit_box(unit):
        """Footprint box (cx, cy, half_w, half_d, heading) of a live unit."""
        p = unit.bodyNP.getPos()
        hx = getattr(unit, 'unitWidth', 2.0) / 2.0
        hy = getattr(unit, 'unitHeight', 2.0) / 2.0
        return (p.x, p.y, hx, hy, unit.bodyNP.getH())

    def units_within(self, src_box, radius, side_units):
        """Units from *side_units* whose footprint is within *radius* (edge to
        edge, oriented boxes) of *src_box*."""
        out = []
        for u in side_units:
            if u.bodyNP.isEmpty():
                continue
            if obb_distance(src_box, self._unit_box(u)) <= radius:
                out.append(u)
        return out

    def panic_nearby_friends(self, src_box, side_units, cause, exclude=None):
        """Every friendly unit within PANIC_RADIUS (edge to edge) of *src_box*
        makes a Panic test.  (The controlling player picks the order; we use
        list order.)"""
        near = [u for u in self.units_within(src_box, self.PANIC_RADIUS, side_units)
                if u is not exclude]
        print(f"[Panic] {cause}: {len(near)} friendly unit(s) within "
              f"{self.PANIC_RADIUS:.0f}\": {', '.join(u.unit.name for u in near) or 'none'}")
        for u in near:
            self.panic_test(u, cause=cause)

    def venerable_source(self, unit):
        """The friendly Venerable unit whose 6" bubble covers *unit*, else None.

        Venerable lets friendly units within 6" (edge to edge, same measurement
        as the nearby-friend Panic bubble) re-roll failed Panic tests.  A unit
        carrying the rule benefits from it itself; a fleeing Venerable unit
        inspires nobody.
        """
        if unit is None or unit.bodyNP.isEmpty():
            return None
        box = self._unit_box(unit)
        for u in self.units_within(box, self.VENERABLE_RADIUS,
                                   self._friendlies_of(unit)):
            if not is_venerable_unit(u):
                continue
            if u is not unit and getattr(u, 'state', None) == 'IsFleeing':
                continue
            return u
        return None

    def check_heavy_casualties(self, unit, phase, attacker=None):
        """>25% of start-of-phase models lost in a non-Combat phase -> Panic
        test, fleeing from *attacker* (or the nearest non-fleeing enemy)."""
        if unit is None or unit.bodyNP.isEmpty() or phase == 'combat':
            return
        start = getattr(unit, 'startOfPhaseModels', unit.unit.nmodels)
        if heavy_casualties(unit.unit.nmodels, start):
            self.panic_test(unit, flee_from=attacker,
                            cause=f"heavy casualties ({phase})")

    def on_unit_destroyed(self, src_box, side_units, unit_strength):
        """A friendly unit of US>=5 was destroyed: friendlies within 6" (edge to
        edge) test.  *src_box* is the destroyed unit's footprint (measure point)."""
        if unit_strength < self.PANIC_US_THRESHOLD:
            print(f"[Panic] destroyed unit US {unit_strength} < "
                  f"{self.PANIC_US_THRESHOLD} — no nearby-friend Panic.")
            return
        print(f"[Panic] a US {unit_strength} unit was destroyed — friends within "
              f"{self.PANIC_RADIUS:.0f}\" test.")
        self.panic_nearby_friends(src_box, side_units, cause="nearby friend destroyed")

    def on_unit_flees_combat(self, unit, unit_strength=None):
        """A unit that lost combat and broke/fell back panics friends within 6"
        if its Unit Strength was >= 5 *at the start of that combat* (pass it as
        *unit_strength*; falls back to the current US)."""
        if unit is None or unit.bodyNP.isEmpty():
            return
        us = unit_strength if unit_strength is not None else unit_strength_total(unit)
        if us < self.PANIC_US_THRESHOLD:
            print(f"[Panic] {unit.unit.name} flees/FBIG but combat-start US {us} "
                  f"< {self.PANIC_US_THRESHOLD} — no nearby-friend Panic.")
            return
        print(f"[Panic] {unit.unit.name} (combat-start US {us}) flees combat — "
              f"friends within {self.PANIC_RADIUS:.0f}\" test.")
        self.panic_nearby_friends(self._unit_box(unit), self._friendlies_of(unit),
                                  cause="nearby friend flees combat", exclude=unit)

    PANIC_US_THRESHOLD = PANIC_US_THRESHOLD
    PANIC_RADIUS = PANIC_RADIUS
    VENERABLE_RADIUS = VENERABLE_RADIUS

    # ─── Direction & movement helpers ─────────────────────────────────────

    def nearest_non_fleeing_enemy(self, unit):
        """Closest enemy unit that is not itself fleeing, or None."""
        enemies = (self.game.player2Units if unit in self.game.player1Units
                   else self.game.player1Units)
        upos = unit.bodyNP.getPos()
        best, best_d = None, float('inf')
        for e in enemies:
            if e.bodyNP.isEmpty() or e.state == 'IsFleeing':
                continue
            d = (e.bodyNP.getPos() - upos).lengthSquared()
            if d < best_d:
                best, best_d = e, d
        return best

    def _flee_direction(self, unit, enemy) -> Vec3:
        """Unit-length vector pointing directly away from *enemy* (or toward
        the unit's own board edge when there is no enemy to flee from)."""
        own_edge = Vec3(0, -1 if unit in self.game.player1Units else 1, 0)
        if enemy is None or enemy.bodyNP.isEmpty():
            return own_edge
        d = unit.bodyNP.getPos() - enemy.bodyNP.getPos()
        d.z = 0
        if d.length() < 1e-4:
            return own_edge
        return d.normalized()

    def _flee_until_clear(self, unit, start, direction: Vec3, distance: float,
                          step: float = 0.5, max_steps: int = 600):
        """Landing point straight along *direction*: sample the whole path from
        *start*, collecting every unit crossed (Fled Through), and land at the
        first spot clear of other units at or beyond the flee *distance* (the
        unit can pass through but not settle inside one).  Returns
        (final_pos, units_passed_through)."""
        body = unit.bodyNP
        passed = []
        seen = set()
        final = Point3(start + direction * distance)
        final.z = start.z
        d = step
        for _ in range(max_steps):
            pos = start + direction * d
            pos.z = start.z
            body.setPos(pos)
            for u in self.game.units:
                if not u.bodyNP.isEmpty():
                    u.bodyNP.node().setTransformDirty()
            contact = self.game.checkUnitContactSmall(unit)
            if contact is not None:
                other = self.game.getSelectedUnit(contact.getNode1())
                if other is not None and id(other) not in seen:
                    seen.add(id(other))
                    passed.append(other)
            elif d >= distance:
                final = Point3(pos)   # first clear spot at/beyond the flee move
                break
            d += step
        body.setPos(start)
        return final, passed
