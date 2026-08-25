"""Psychology subsystem — Panic tests (Phase 0).

The Old World psychology rules (Rulebook p. 160-161) all resolve into a
Leadership test that, on failure, makes a unit Fall Back in Good Order or Flee.
This module provides the reusable Panic test and its pure helpers; the causes
(heavy casualties, nearby friend destroyed/flees, fled through) are wired in
later phases and simply call ``PsychologySystem.panic_test``.

Pure functions (``leadership_test``, ``leadership_test_with_reroll``,
``panic_fail_outcome``, ``unit_strength_total``, ``break_test_outcome``,
``overwhelmed``) are free of Panda3D state so they can be unit-tested.
"""

from __future__ import annotations

import random
import math

from panda3d.core import Vec3, Point3
from direct.interval.LerpInterval import LerpPosInterval
from direct.interval.IntervalGlobal import Sequence
from direct.interval.FunctionInterval import Func

from special_rules import (board_edge_distance, should_use_swiftstride,
                           unit_has_swiftstride)


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


def is_stubborn_unit(unit) -> bool:
    """True if *unit* carries the Stubborn special rule."""
    if unit is None:
        return False
    if getattr(unit, 'isStubborn', False):
        return True
    model = getattr(getattr(unit, 'unit', None), 'model', None)
    check = getattr(model, 'is_stubborn', None)
    return bool(check()) if callable(check) else False


def is_battle_standard_unit(unit) -> bool:
    """True if *unit* carries the army's Battle Standard."""
    if unit is None:
        return False
    if getattr(unit, 'isBSB', False):
        return True
    model = getattr(getattr(unit, 'unit', None), 'model', None)
    check = getattr(model, 'is_battle_standard', None)
    return bool(check()) if callable(check) else False


def is_character_unit(unit) -> bool:
    """True if *unit* is a character model (catalogue Category)."""
    model = getattr(getattr(unit, 'unit', None), 'model', None)
    chars = getattr(model, 'characteristics', None) or {}
    return str(chars.get('Category', '')).strip().lower() == 'characters'


def is_large_target(unit) -> bool:
    """True if *unit* has the Large Target special rule (or is mounted on one)."""
    model = getattr(getattr(unit, 'unit', None), 'model', None)
    rules = getattr(model, 'special_rules', None) or []
    return any(isinstance(r, dict) and r.get('large_target') for r in rules)


def command_range(general) -> float:
    """The General's Command range in inches.

    The General and Battle Standard Bearer have a flat 12" Command range
    regardless of their Leadership (Rulebook p. 202), extended to 18" by Large
    Target.
    """
    return (LARGE_TARGET_COMMAND_RANGE if is_large_target(general)
            else COMMAND_RANGE)


def effective_leadership(own_ld: int, general_ld=None) -> int:
    """Leadership to test on, given an inspiring General's Ld (or None).

    Inspiring Presence lets a unit *use* the General's Leadership, so a lower
    value is simply never taken.
    """
    return own_ld if general_ld is None else max(own_ld, general_ld)


def select_general(units):
    """Nominate the army General from *units* and return it (or None).

    The General is the character with the highest Leadership; a character the
    army list explicitly flags as the General wins outright (Rulebook p. 203).
    Ties fall to list order, where the rules would let the player choose.
    """
    for u in units:
        setattr(u, 'isGeneral', False)
    characters = [u for u in units if is_character_unit(u)]
    if not characters:
        return None
    model_flag = [u for u in characters
                  if any(isinstance(r, dict) and r.get('general')
                         for r in (u.unit.model.special_rules or []))]
    pool = model_flag or characters
    # The Battle Standard Bearer cannot be the General (Rulebook p. 203) unless
    # it is the only character left to lead the army.
    eligible = [u for u in pool if not _carries_battle_standard(u)]
    if not eligible:
        print("Only the Battle Standard Bearer can lead — making it the General.")
        eligible = pool
    general = max(eligible, key=lambda u: _stat_int(u.unit.model.characteristics, 'Ld', 0))
    general.isGeneral = True
    return general


def _carries_battle_standard(unit) -> bool:
    """Whether the unit's profile carries the Battle Standard keyword (as opposed
    to the ``isBSB`` nomination, which is what selection is about to set)."""
    return any(isinstance(r, dict) and r.get('battle_standard')
               for r in (unit.unit.model.special_rules or []))


def select_battle_standard(units):
    """Nominate the army Battle Standard Bearer from *units* and return it.

    The bearer is a character the army list equips with the Battle Standard; the
    banner is lost with them and cannot be picked up by anybody else.
    """
    for u in units:
        setattr(u, 'isBSB', False)
    bearers = [u for u in units
               if is_character_unit(u) and _carries_battle_standard(u)]
    if not bearers:
        return None
    bearers[0].isBSB = True
    return bearers[0]


def side_unit_strength(units_on_side) -> int:
    """Total Unit Strength of every live unit fighting on this side."""
    return sum(unit_strength_total(u) for u in units_on_side
               if not u.bodyNP.isEmpty())


def massed_infantry_bonus(units_on_side, own_us: int, enemy_us: int) -> int:
    """Combat result point for weight of numbers (Rulebook p. 190).

    The side needs both the higher Unit Strength *and* a unit with the rule;
    numbers alone are not enough, and neither is having the infantry.
    """
    if own_us <= enemy_us:
        return 0
    for u in units_on_side:
        if u.unit.model.troop_type_rule('Massed Infantry'):
            return 1
    return 0


def battle_standard_bonus(units_on_side) -> int:
    """Combat result points from a Battle Standard fighting on this side.

    The Battle Standard is worth +1 even when an ordinary standard is also
    present, but two Battle Standards on the same side still only count once
    (Rulebook p. 203).
    """
    for u in units_on_side:
        if is_battle_standard_unit(u):
            return 1
        joined = getattr(u, 'joinedCharacter', None)
        if joined is not None and is_battle_standard_unit(joined):
            return 1
    return 0


def stubborn_available(unit) -> bool:
    """True if *unit* may still use Stubborn to skip a Break test.

    Only the unit's own profile is inspected: a unit is not Stubborn because a
    Stubborn character joined it, and a Stubborn character cannot use the rule
    while part of a unit that is not Stubborn (Rulebook p. 178).
    """
    return is_stubborn_unit(unit) and not getattr(unit, 'usedStubborn', False)


MAX_RANK_BONUS = 2
# The rules are written around regular infantry, so that is the fallback for a
# troop type the table does not know.
MODELS_PER_RANK = 5


def rank_bonus(unit, disrupted: bool = False) -> int:
    """Combat result points a formed unit claims for its ranks.

    One per rank behind the first. A rank only counts if it holds at least the
    troop type's models per rank, which is also why a unit narrower than that
    claims nothing at all. Skirmishers claim none, a Disrupted unit claims
    none, and each troop type caps what it can claim -- a heavy chariot cannot
    form ranks at all.
    """
    model = unit.model
    if disrupted or model.is_skirmisher():
        return 0
    cap = model.max_rank_bonus(MAX_RANK_BONUS)
    if cap <= 0:
        return 0
    per_rank = model.models_per_rank(MODELS_PER_RANK)
    if per_rank <= 0 or unit.files < per_rank:
        return 0
    bonus = unit.ranks - 1
    remainder = unit.nmodels % unit.files if unit.files else 0
    if 0 < remainder < per_rank:
        bonus -= 1
    return max(0, min(bonus, cap))


def overwhelmed(winner_us: int, loser_us: int) -> bool:
    """True if the winning side's Unit Strength is *more than twice* the loser's.

    A losing unit that rolls the Fall Back in Good Order result Breaks instead
    (Rulebook p. 154). Unit Strength is the total of every unit on a side,
    worked out at the end of the Combat phase.
    """
    return winner_us > 2 * loser_us


def break_test_outcome(dice, ld: int, diff: int, overwhelm: bool = False) -> str:
    """Resolve a Break test to 'break', 'fall_back' or 'give_ground'.

    *dice* are the two D6 results, *diff* the winner's combat result score minus
    the loser's. Natural roll above Ld -> Breaks; natural at or below Ld but the
    modified roll above Ld -> Falls Back in Good Order (Breaks instead when the
    loser is overwhelmed); otherwise, or on a natural double 1, Gives Ground.
    """
    natural = sum(dice)
    if len(dice) == 2 and dice[0] == 1 and dice[1] == 1:
        return 'give_ground'
    if natural > ld:
        return 'break'
    if natural + diff > ld:
        return 'break' if overwhelm else 'fall_back'
    return 'give_ground'


def should_use_stubborn(ld: int, diff: int, overwhelm: bool) -> bool:
    """AI policy for spending a unit's one Stubborn refusal.

    Stubborn is not free: it trades the chance of Giving Ground (staying in the
    fight) for a guaranteed Fall Back. Spend it when the Break test would more
    likely than not end in a Break -- which an overwhelmed unit also suffers on
    what would otherwise be a Fall Back result.
    """
    hold_on = ld - diff if overwhelm else ld
    return (1.0 - _p_at_most(hold_on)) >= 0.5


def should_reroll_break(outcome: str, ld: int, diff: int,
                        overwhelm: bool = False) -> bool:
    """AI policy for a Battle Standard's Break test re-roll.

    The second roll has to be accepted even if it is worse, so only re-roll when
    the odds beat what was rolled: always after a Break, and after a Fall Back
    only when Giving Ground is likelier than Breaking.
    """
    if outcome == 'give_ground':
        return False
    if outcome == 'break':
        return True
    p_give_ground = _p_at_most(ld - diff)
    p_break = 1.0 - _p_at_most(ld)
    return p_give_ground > p_break


def _p_at_most(target: int) -> float:
    """Probability that 2D6 rolls *target* or less."""
    if target < 2:
        return 0.0
    if target >= 12:
        return 1.0
    counts = {2: 1, 3: 3, 4: 6, 5: 10, 6: 15, 7: 21,
              8: 26, 9: 30, 10: 33, 11: 35, 12: 36}
    return counts[target] / 36.0


# A unit that Gives Ground backs off this far, in inches (Rulebook p. 134).
GIVE_GROUND = 2.0
# Units of this Unit Strength or more cause nearby-friend Panic when destroyed
# or when they flee/fall back from combat.
PANIC_US_THRESHOLD = 5
# Friendly units within this range (inches) must test.
PANIC_RADIUS = 6.0
# Venerable grants its Panic re-roll to friendly units within the same 6" bubble.
VENERABLE_RADIUS = PANIC_RADIUS
# The General's Command range, in inches (Inspiring Presence reaches this far).
COMMAND_RANGE = 12.0
LARGE_TARGET_COMMAND_RANGE = 18.0

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

    def panic_test(self, unit, flee_from=None, cause: str = "",
                   compulsory: bool = False):
        """Queue a Panic test for *unit*.  Tests resolve one at a time: each
        unit completes its move (and rally/reform) before the next starts, so
        a fled-through / nearby unit only begins after the first unit rallies.

        A *compulsory* test is one the unit must take even when it would pass
        automatically; failing that one costs it ground rather than its nerve.
        """
        if unit is None or unit.bodyNP.isEmpty():
            return
        self._panic_queue.append((unit, flee_from, cause, compulsory))
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
        unit, flee_from, cause, compulsory = self._panic_queue.pop(0)
        self._resolve_panic(unit, flee_from, cause, self._run_next_panic,
                            compulsory)

    def _resolve_panic(self, unit, flee_from, cause, on_done, compulsory=False):
        """Test *unit*; on failure move it (flee / fall back) and call *on_done*
        only once the whole move (and rally/reform) has finished."""
        if unit is None or unit.bodyNP.isEmpty():
            on_done()
            return
        reason = self.panic_exempt_reason(unit)
        if reason is not None and not compulsory:
            print(f"[Panic] {unit.unit.name} is exempt from Panic ({cause}): {reason}.")
            on_done()
            return
        forced = reason is not None
        if forced:
            print(f"[Panic] {unit.unit.name} would be exempt ({reason}) but "
                  f"{cause} compels the test — a failure costs it ground.")

        unit.panicTestedThisPhase = True
        ld, general = self.leadership_of(unit)
        if general is not None:
            print(f"[Panic] {unit.unit.name} uses the General's Leadership "
                  f"({general.unit.name}, Ld {ld}) — Inspiring Presence.")
        venerable = self.venerable_source(unit)
        bsb = self.battle_standard_of(unit)
        reroll_source = venerable or bsb
        passed, rolls = leadership_test_with_reroll(ld, reroll=reroll_source is not None)
        roll = rolls[-1]
        if len(rolls) > 1:
            why = ("Venerable" if venerable is not None
                   else "Hold Your Ground")
            print(f"[Panic] {unit.unit.name} re-rolls a failed Panic test "
                  f"({why}: {reroll_source.unit.name}): "
                  f"2D6={rolls[0]} -> {rolls[-1]}")
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
        outcome = ('give_ground' if forced
                   else panic_fail_outcome(remaining, start))
        enemy_name = enemy.unit.name if enemy is not None else "board edge"
        up = unit.bodyNP.getPos()
        ep = enemy.bodyNP.getPos() if enemy is not None else None
        eps = f"({ep.x:.0f},{ep.y:.0f})" if ep is not None else "n/a"
        print(f"[Panic] {unit.unit.name} fails -> {outcome} away from {enemy_name} | "
              f"unit=({up.x:.0f},{up.y:.0f}) enemy={eps} "
              f"dir=({direction.x:.2f},{direction.y:.2f})")

        # A fleeing unit's Flee roll is 2D6; Fall Back in Good Order discards
        # the lowest (Rulebook p. 134). Giving Ground is a flat 2".
        d1, d2 = random.randint(1, 6), random.randint(1, 6)
        spent = outcome != 'give_ground' and getattr(unit, 'fledThisPhase', False)
        if outcome == 'give_ground':
            distance = GIVE_GROUND
        elif outcome == 'fall_back':
            distance = max(d1, d2)
        else:
            distance = d1 + d2
        # Panic flees resolve without a prompt, so Swiftstride's optional die is
        # taken on the same policy the AI uses.
        if not spent and outcome != 'give_ground' and unit_has_swiftstride(unit) and should_use_swiftstride(
                'flee', board_edge_distance(up.x, up.y)):
            bonus = random.randint(1, 6)
            distance += bonus
            print(f"[Panic] {unit.unit.name} adds Swiftstride +{bonus} to its "
                  f"Flee roll.")
        if spent:
            # The Limits of Endurance (p. 133): one flee move per phase, and a
            # second covers nothing.
            print(f"[Panic] {unit.unit.name} has already fled this phase — "
                  f"The Limits of Endurance, so it moves 0\" instead of "
                  f"{distance}\".")
            distance = 0
        if outcome != 'give_ground':
            unit.fledThisPhase = True
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
        label = {'fall_back': "falls back",
                 'give_ground': "gives ground"}.get(outcome, "flees")
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
            if outcome != 'give_ground':
                # Fleeing through an enemy is perilous (p. 133), and a Fall
                # Back moves exactly like a flee (p. 134). A Give Ground is 2"
                # backwards and runs through nobody.
                resolver = getattr(self.game, 'combat', None)
                if resolver is not None:
                    resolver.perilTests(unit, Vec3(final - start))
            gone = unit.bodyNP.isEmpty()
            if outcome == 'give_ground':
                # The unit never lost its nerve, so there is nothing to rally
                # from and nobody it can be said to have fled through.
                print(f"[Panic] {unit.unit.name} Gives Ground.")
                on_done()
            elif outcome == 'fall_back' and not gone:
                # Auto-rally: regain composure, cannot charge this turn.
                unit.request("Idle")
                unit.cannotChargeThisTurn = True
                unit.attemptedRallyThisTurn = True
                unit.updateTextNode()
                print(f"[Panic] {unit.unit.name} Falls Back in Good Order and "
                      f"rallies — free reform (cannot charge this turn).")
                self.game.startFreeReform(unit, on_done=lambda: self._after_unit_done(unit, passed, on_done))
            else:
                if not gone:
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
            self._panic_queue.append((other, None, "fled through", False))
        on_done()

    # ─── Common causes of Panic ───────────────────────────────────────────

    def _friendlies_of(self, unit):
        return (self.game.player1Units if unit in self.game.player1Units
                else self.game.player2Units)

    @staticmethod
    def _unit_box(unit):
        """Footprint box (cx, cy, half_w, half_d, heading) of a live unit, in
        world space -- a joined character's body is parented to its host, so its
        own transform is host-relative and has to be resolved against the root."""
        body = unit.bodyNP
        top = body.getTop()
        p = body.getPos(top)
        hx = getattr(unit, 'unitWidth', 2.0) / 2.0
        hy = getattr(unit, 'unitHeight', 2.0) / 2.0
        return (p.x, p.y, hx, hy, body.getH(top))

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

    # ─── Command range (Inspiring Presence / Hold Your Ground) ────────────

    def command_models_on_side(self, unit, is_source):
        """Friendly units matching *is_source*, including one riding along
        inside a host unit. A joined character is dropped from the player lists,
        so hosts have to be searched too."""
        out = []
        for u in self._friendlies_of(unit):
            if is_source(u):
                out.append(u)
            joined = getattr(u, 'joinedCharacter', None)
            if joined is not None and is_source(joined):
                out.append(joined)
        return out

    def generals_on_side(self, unit):
        return self.command_models_on_side(
            unit, lambda u: getattr(u, 'isGeneral', False))

    @staticmethod
    def _is_fleeing(unit):
        """True if *unit* is fleeing — for a joined character, that is whatever
        its host is doing."""
        host = getattr(unit, 'hostUnit', None)
        return getattr(host or unit, 'state', None) == 'IsFleeing'

    def _command_source(self, unit, is_source):
        """The friendly command model whose Command range covers *unit*.

        Measured edge to edge from the model's own base — one that has joined a
        unit leads from wherever it stands in the host's ranks, not from the
        host's centre. A fleeing model inspires nobody.
        """
        if unit is None or unit.bodyNP.isEmpty():
            return None
        box = self._unit_box(unit)
        for src in self.command_models_on_side(unit, is_source):
            if src.bodyNP.isEmpty() or self._is_fleeing(src):
                continue
            if obb_distance(self._unit_box(src), box) <= command_range(src):
                return src
        return None

    def general_of(self, unit):
        """The friendly General whose Command range covers *unit*, else None."""
        return self._command_source(unit, lambda u: getattr(u, 'isGeneral', False))

    def battle_standard_of(self, unit):
        """The friendly Battle Standard Bearer whose Command range covers *unit*.

        Hold Your Ground lets those units re-roll failed Panic and Rally tests,
        and re-roll the 2D6 of a Break test (Rulebook p. 203).
        """
        return self._command_source(unit, is_battle_standard_unit)

    def leadership_of(self, unit):
        """Leadership *unit* tests on, plus the inspiring General (or None).

        Inspiring Presence lets any friendly unit within the General's Command
        range use the General's Leadership instead of its own.
        """
        own = _stat_int(unit.unit.model.characteristics, 'Ld', 7)
        general = self.general_of(unit)
        if general is None or general is unit:
            return own, None
        gen_ld = _stat_int(general.unit.model.characteristics, 'Ld', 7)
        ld = effective_leadership(own, gen_ld)
        return ld, (general if ld > own else None)

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
