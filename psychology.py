"""Psychology subsystem — Panic tests (Phase 0).

The Old World psychology rules (Rulebook p. 160-161) all resolve into a
Leadership test that, on failure, makes a unit Fall Back in Good Order or Flee.
This module provides the reusable Panic test and its pure helpers; the causes
(heavy casualties, nearby friend destroyed/flees, fled through) are wired in
later phases and simply call ``PsychologySystem.panic_test``.

Pure functions (``leadership_test``, ``panic_fail_outcome``,
``unit_strength_total``) are free of Panda3D state so they can be unit-tested.
"""

from __future__ import annotations

import random

from panda3d.core import Vec3


def _stat_int(characteristics: dict, key: str, default: int = 0) -> int:
    try:
        return int(characteristics.get(key))
    except (KeyError, TypeError, ValueError):
        return default


def leadership_test(ld: int, modifier: int = 0):
    """Roll 2D6 against Leadership (+modifier). Returns ``(passed, roll)``."""
    roll = random.randint(1, 6) + random.randint(1, 6)
    return roll <= (ld + modifier), roll


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


# Special-rule names that make a unit exempt from Panic (full exemption).
_FULL_PANIC_IMMUNE = ('ignore panic', 'immune to psychology')


class PsychologySystem:
    """Panic tests and (later) the rest of the psychology rules."""

    def __init__(self, game):
        self.game = game

    # ─── Exemptions (No Need for Hysterics) ───────────────────────────────

    def panic_exempt_reason(self, unit):
        """Return why *unit* need not take a Panic test, or None if it must.
        A unit is exempt if it has already tested this phase, is fleeing or
        engaged in combat, or is rule-immune to Panic."""
        if getattr(unit, 'panicTestedThisPhase', False):
            return "already tested this phase"
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

    # ─── Panic test ───────────────────────────────────────────────────────

    def panic_test(self, unit, flee_from=None, cause: str = "") -> str | None:
        """Make a Panic test for *unit*.  On failure the unit Falls Back in Good
        Order (>50% of its start-of-battle models remain) or Flees, away from
        *flee_from* (or the nearest non-fleeing enemy).  Returns the outcome
        ('pass' / 'fall_back' / 'flee' / 'exempt')."""
        if unit is None or unit.bodyNP.isEmpty():
            return None
        reason = self.panic_exempt_reason(unit)
        if reason is not None:
            print(f"[Panic] {unit.unit.name} is exempt from Panic ({cause}): {reason}.")
            return 'exempt'

        unit.panicTestedThisPhase = True
        ld = _stat_int(unit.unit.model.characteristics, 'Ld', 7)
        passed, roll = leadership_test(ld)
        remaining = unit.unit.nmodels
        start = getattr(unit, 'startOfBattleModels', remaining) or remaining
        pct = 100.0 * remaining / max(1, start)
        print(f"[Panic] {unit.unit.name}: Ld {ld} vs 2D6={roll} -> "
              f"{'PASS' if passed else 'FAIL'}  ({cause}; {remaining}/{start} "
              f"models, {pct:.0f}% remain)")
        if passed:
            return 'pass'

        enemy = flee_from or self.nearest_non_fleeing_enemy(unit)
        direction = self._flee_direction(unit, enemy)
        outcome = panic_fail_outcome(remaining, start)
        if outcome == 'fall_back':
            self._fall_back_in_good_order(unit, direction)
            print(f"[Panic] {unit.unit.name} Falls Back in Good Order.")
        else:
            self._flee(unit, direction)
            print(f"[Panic] {unit.unit.name} Flees!")
        return outcome

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

    def _flee(self, unit, direction: Vec3):
        distance = random.randint(1, 6) + random.randint(1, 6)
        unit.request("IsFleeing")
        unit.hasMovedThisTurn = True
        self.game.movement.fallBack(unit.bodyNP, direction, length=distance, flee=True)
        unit.updateTextNode()

    def _fall_back_in_good_order(self, unit, direction: Vec3):
        move = _stat_int(unit.unit.model.characteristics, 'M', 4)
        self.game.movement.fallBack(unit.bodyNP, direction, length=move, GG=True)
        unit.hasMovedThisTurn = True
        unit.updateTextNode()
