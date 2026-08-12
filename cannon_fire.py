"""
Cannon Fire war-machine shooting (Warhammer: the Old World, first pass).

Implements the Cannon Fire special rule: choose a target point, roll an
Artillery Dice for the strike distance, bounce with a second Artillery Dice,
then hit every enemy model whose base lies under the strike point or the
bounce path. Misfires end the shot (the Misfire table is not yet modelled).

Deferred (per design): the Black Powder Misfire table, Multiple Wounds (X),
the rank/file hit caps and the 'crunch' stops (monstrous/Behemoth, terrain).
"""

import random

from panda3d.core import Vec3, Point3, LineSegs

from battleFunctions import check_armor_save
from dice import ArtilleryDice, checkDice


def wound_target(strength: int, toughness: int) -> int:
    """To Wound target number on a d6 for the given Strength vs Toughness."""
    diff = strength - toughness
    if diff >= 2:
        return 2
    if diff == 1:
        return 3
    if diff == 0:
        return 4
    if diff == -1:
        return 5
    return 6


class CannonFire:
    """Resolves Cannon Fire for war machines. One instance per game."""

    def __init__(self, game):
        self.game = game
        self.marker = None
        self.pathLine = None

    # ─── Detection ──────────────────────────────────────────────────

    def cannon_weapon(self, unit):
        """Return the unit's Cannon Fire weapon dict, or None."""
        if not (unit and unit.unit and unit.unit.model):
            return None
        for w in unit.unit.model.weapons.values():
            rules = [str(r).lower() for r in (w.get('special_rules') or [])]
            if any('cannon fire' in r for r in rules):
                return w
        return None

    def is_cannon(self, unit) -> bool:
        return self.cannon_weapon(unit) is not None

    # ─── Targeting ──────────────────────────────────────────────────

    def begin_targeting(self, cannonUnit):
        """Enter Cannon Fire mode: show max range and wait for a ground click."""
        weapon = self.cannon_weapon(cannonUnit)
        max_range = weapon.get('ranged_range', 0)
        self.game.drawRangeRing(cannonUnit.bodyNP.getPos(), max_range, color=(1, 0.5, 0, 0.8))
        self.game.debugTextInfo.setText(
            f"Cannon Fire: click a target point (max range {max_range}\")")
        self.game.ignore('mouse1')
        self.game.accept('mouse1', self._on_fire_click, [cannonUnit])

    def _on_fire_click(self, cannonUnit):
        if self.game.awaitingChoice:
            return
        target = Point3(self.game.mousePosOnGround)
        self.game.ignore('mouse1')
        taskMgr.add(self._fire_task(cannonUnit, target))

    async def _fire_task(self, cannonUnit, target):
        try:
            await self.fire(cannonUnit, target)
        finally:
            # Return control to unit selection for the next shot.
            self.game.accept(
                'mouse1', self.game.setActiveUnit,
                [self.game.taskShootingArcUpdate, "taskShootingArcUpdate"])

    # ─── Artillery Dice ─────────────────────────────────────────────

    async def roll_artillery(self, position):
        """Roll one Artillery Dice and return its value ('Misfire' or int)."""
        die = ArtilleryDice(self.game.world, position=position, size=1.0,
                            color=(0.15, 0.15, 0.15, 1))
        die.roll()
        await taskMgr.add(checkDice, "checkArtilleryDice",
                          extraArgs=[[die]], appendTask=True)
        value = die.artillery_value()
        die.remove(self.game.world)
        return value

    # ─── Fire sequence ──────────────────────────────────────────────

    async def fire(self, cannonUnit, target):
        weapon = self.cannon_weapon(cannonUnit)
        cannon_pos = cannonUnit.bodyNP.getPos()

        direction = Vec3(target - cannon_pos)
        direction.z = 0
        if direction.length() < 1e-4:
            self.game.debugTextInfo.setText("Cannon Fire: invalid target point")
            return
        dir_n = direction.normalized()

        max_range = weapon.get('ranged_range', 0)
        if direction.length() > max_range:
            self.game.debugTextInfo.setText("Cannon Fire: target out of range")
            return

        self._place_marker(target)

        # Step 2 — strike distance.
        first = await self.roll_artillery(cannon_pos + Vec3(0, 0, 12))
        if first == 'Misfire':
            self.game.diceInfoText.setText("Artillery Dice: MISFIRE! The shot fails.")
            self._draw_path(cannon_pos, target, misfire=True)
            return
        strike_point = Point3(target + dir_n * first)
        self._place_marker(strike_point)

        # Step 3 — bounce distance ('Misfire' buries the ball at the strike point).
        second = await self.roll_artillery(cannon_pos + Vec3(1, 0, 12))
        bounce = 0 if second == 'Misfire' else second
        bounce_end = Point3(strike_point + dir_n * bounce)

        self.game.diceInfoText.setText(
            f"Cannon Fire: strike {first}\", bounce "
            f"{'buried' if second == 'Misfire' else str(second) + chr(34)}")
        self._draw_path(strike_point, bounce_end)

        # Step 4 — determine hits (first pass: base centre within the corridor).
        hits = self._models_under_path(cannonUnit, strike_point, dir_n, bounce)

        # Step 5 — wound and remove casualties (S/AP from the cannon weapon).
        strength = int(weapon.get('ranged_strength') or 10)
        ap = int(weapon.get('ranged_AP', 0))
        total_hit = total_cas = 0
        for unit, count in hits:
            total_hit += count
            casualties = self._apply_wounds(unit, count, strength, ap)
            total_cas += casualties
            if casualties:
                self.game.removeModelsFromUnit(unit, casualties)

        self.game.debugText.setText(
            f"Cannon Fire hit {total_hit} model(s), {total_cas} slain.")
        cannonUnit.hasAttackedThisTurn = True

    # ─── Hit determination ──────────────────────────────────────────

    def _models_under_path(self, cannonUnit, strike_point, dir_n, bounce):
        """Return [(unit, hit_count)] for enemy models under the ball's path."""
        enemies = (self.game.player2Units
                   if cannonUnit in self.game.player1Units
                   else self.game.player1Units)
        results = []
        for unit in list(enemies):
            if unit.model.isEmpty():
                continue
            half_w = max(getattr(unit, 'modelWidth', 1.0) / 2.0, 0.5)
            count = 0
            for child in unit.model.getChildren():
                rel = Vec3(child.getPos(render) - strike_point)
                rel.z = 0
                proj = rel.dot(dir_n)          # distance along the path
                perp = (rel - dir_n * proj).length()
                if -0.5 <= proj <= bounce and perp <= half_w:
                    count += 1
            if count:
                results.append((unit, count))
        return results

    def _apply_wounds(self, unit, hits, strength, ap):
        """Roll To Wound then armour save for each hit; return casualties."""
        model = unit.unit.model
        toughness = model.get_toughness() if hasattr(model, 'get_toughness') else 4
        target = wound_target(strength, toughness)
        casualties = 0
        for _ in range(hits):
            if random.randint(1, 6) < target:
                continue  # failed to wound
            if not check_armor_save(model, model.armor_save, ap):
                casualties += 1
        return min(casualties, unit.unit.nmodels)

    # ─── Visuals ────────────────────────────────────────────────────

    def _place_marker(self, point):
        if self.marker:
            self.marker.removeNode()
        ls = LineSegs()
        ls.setColor(1, 0.2, 0.2, 1)
        ls.setThickness(2.0)
        r = 1.0
        for d in ((-r, 0), (r, 0)):
            ls.moveTo(point.x + d[0], point.y - r, point.z + 0.2)
            ls.drawTo(point.x + d[0], point.y + r, point.z + 0.2)
        ls.moveTo(point.x - r, point.y, point.z + 0.2)
        ls.drawTo(point.x + r, point.y, point.z + 0.2)
        self.marker = render.attachNewNode(ls.create())
        self.marker.setName("CannonTargetMarker")

    def _draw_path(self, start, end, misfire=False):
        if self.pathLine:
            self.pathLine.removeNode()
        ls = LineSegs()
        ls.setColor((0.6, 0.6, 0.6, 1) if misfire else (1, 0.3, 0.1, 1))
        ls.setThickness(3.0)
        ls.moveTo(start.x, start.y, start.z + 0.3)
        ls.drawTo(end.x, end.y, end.z + 0.3)
        self.pathLine = render.attachNewNode(ls.create())
        self.pathLine.setName("CannonPath")

    def cleanup(self):
        """Remove Cannon Fire markers/path (called on shooting phase exit)."""
        for attr in ('marker', 'pathLine'):
            node = getattr(self, attr, None)
            if node:
                node.removeNode()
                setattr(self, attr, None)
