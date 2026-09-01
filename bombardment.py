"""
Bombardment war-machine shooting (Warhammer: the Old World, first pass).

Implements the Bombardment special rule (Mortar and similar). Place a blast
template over the target unit's centre, scatter it with an Artillery Dice and a
Scatter Dice, then hit every enemy model under the template: the single model
under the central hole takes the higher (bracketed) Strength/AP and Multiple
Wounds; all others take the weapon's normal Strength/AP.

Misfires end the shot (the Black Powder Misfire table is not yet modelled).
"""

import math
import random

from panda3d.core import Vec3, Point3, LineSegs

from battleFunctions import check_armor_save
from cannon_fire import wound_target
from models import roll_dice_expr, stat_int
from dice import ArtilleryDice, ScatterDice, checkDice
from rules_log import battle_log


class Bombardment:
    """Resolves Bombardment shooting for war machines. One instance per game."""

    def __init__(self, game):
        self.game = game
        self.circles = []

    # ─── Detection ──────────────────────────────────────────────────

    def bombardment_weapon(self, unit):
        """Return the unit's Bombardment weapon dict, or None."""
        if not (unit and unit.unit and unit.unit.model):
            return None
        for w in unit.unit.model.weapons.values():
            rules = [str(r).lower() for r in (w.get('special_rules') or [])]
            if any('bombardment' in r for r in rules):
                return w
        return None

    def is_bombardment(self, unit) -> bool:
        return self.bombardment_weapon(unit) is not None

    # ─── Targeting ──────────────────────────────────────────────────

    def begin_targeting(self, unit):
        """Enter Bombardment mode: show min/max range and wait for a target unit."""
        weapon = self.bombardment_weapon(unit)
        min_r = weapon.get('ranged_range_min', 0)
        max_r = weapon.get('ranged_range', 0)
        self._clear_circles()
        pos = unit.bodyNP.getPos()
        if min_r:
            self._draw_circle(pos, min_r, (1, 0.3, 0.3, 0.7))
        self._draw_circle(pos, max_r, (1, 0.6, 0, 0.8))
        self.game.debugTextInfo.setText(
            f"Bombardment: click an enemy unit ({min_r}-{max_r}\")")
        self.game.ignore('mouse1')
        self.game.accept('mouse1', self._on_fire_click, [unit])

    def _on_fire_click(self, unit):
        if self.game.awaitingChoice:
            return
        enemies = (self.game.player2Units
                   if unit in self.game.player1Units
                   else self.game.player1Units)
        if not enemies:
            return
        point = Point3(self.game.mousePosOnGround)
        target = min(enemies, key=lambda u: (u.bodyNP.getPos() - point).length())
        self.game.ignore('mouse1')
        taskMgr.add(self._fire_task(unit, target))

    async def _fire_task(self, unit, target):
        try:
            await self.fire(unit, target)
        finally:
            self.game.accept(
                'mouse1', self.game.setActiveUnit,
                [self.game.taskShootingArcUpdate, "taskShootingArcUpdate"])

    # ─── Dice ───────────────────────────────────────────────────────

    async def roll_scatter_dice(self, position):
        """Roll an Artillery Dice and a Scatter Dice together; return (art, scat)."""
        art = ArtilleryDice(self.game.world, position=position, size=1.0,
                            color=(0.15, 0.15, 0.15, 1))
        scat = ScatterDice(self.game.world, position=position + Vec3(1.5, 0, 0),
                           size=1.0, color=(0.1, 0.1, 0.45, 1))
        art.roll()
        scat.roll()
        await taskMgr.add(checkDice, "checkBombardDice",
                          extraArgs=[[art, scat]], appendTask=True)
        art_val = art.artillery_value()
        scat_val = scat.scatter_value()
        art.remove(self.game.world)
        scat.remove(self.game.world)
        return art_val, scat_val

    # ─── Fire sequence ──────────────────────────────────────────────

    async def fire(self, unit, target):
        weapon = self.bombardment_weapon(unit)
        min_r = weapon.get('ranged_range_min', 0)
        max_r = weapon.get('ranged_range', 0)
        origin = unit.bodyNP.getPos()
        centre = Point3(target.bodyNP.getPos())

        flat = Vec3(centre - origin)
        flat.z = 0
        dist = flat.length()
        if dist < min_r or dist > max_r:
            self.game.debugTextInfo.setText(
                f"Bombardment: target out of range ({min_r}-{max_r}\")")
            return

        radius = weapon.get('blast_diameter', 3) / 2.0
        self._place_template(centre, radius, (1, 1, 0, 0.8))

        # Scatter: Artillery Dice (distance) + Scatter Dice (hit/direction).
        art, scat = await self.roll_scatter_dice(origin + Vec3(0, 0, 12))
        if art == 'Misfire':
            self.game.diceInfoText.setText("Artillery Dice: MISFIRE! The shot fails.")
            return
        if scat == 'Hit!':
            strike = centre
            note = "on target (Hit!)"
        else:
            angle = random.uniform(0, 2 * math.pi)
            strike = Point3(centre + Vec3(math.cos(angle), math.sin(angle), 0) * art)
            note = f"scattered {art}\""
        self.game.diceInfoText.setText(f"Bombardment: {note}")
        self._place_template(strike, radius, (1, 0.3, 0.1, 0.9))

        self._resolve_damage(unit, weapon, strike, radius)
        unit.hasAttackedThisTurn = True

    # ─── Damage ─────────────────────────────────────────────────────

    def _models_under_template(self, unit, centre, radius):
        """Return [(enemy_unit, child_np, dist_to_centre)] under the template."""
        enemies = (self.game.player2Units
                   if unit in self.game.player1Units
                   else self.game.player1Units)
        out = []
        for enemy in list(enemies):
            if enemy.model.isEmpty():
                continue
            for child in enemy.model.getChildren():
                rel = Vec3(child.getPos(render) - centre)
                rel.z = 0
                d = rel.length()
                if d <= radius:
                    out.append((enemy, child, d))
        return out

    def _resolve_damage(self, unit, weapon, centre, radius):
        under = self._models_under_template(unit, centre, radius)
        if not under:
            self.game.debugText.setText("Bombardment: no models under the template.")
            return

        strength = int(weapon.get('ranged_strength') or 3)
        ap = int(weapon.get('ranged_AP', 0))
        s_central = int(weapon.get('ranged_strength_central', strength))
        ap_central = int(weapon.get('ranged_AP_central', ap))
        mw = weapon.get('multiple_wounds')

        # The single model nearest the centre lies under the central hole.
        central_enemy, central_child, _ = min(under, key=lambda t: t[2])

        by_unit = {}
        for enemy, child, _ in under:
            by_unit.setdefault(enemy, []).append(child)

        total_hit = total_cas = 0
        for enemy, children in by_unit.items():
            model = enemy.unit.model
            cas = 0
            for child in children:
                total_hit += 1
                if enemy is central_enemy and child is central_child:
                    if self._wound_unsaved(model, s_central, ap_central):
                        wounds = roll_dice_expr(mw) if mw else 1
                        if wounds >= stat_int(model.characteristics, 'W', 1):
                            cas += 1
                else:
                    if self._wound_unsaved(model, strength, ap):
                        cas += 1
            cas = min(cas, len(enemy.model.getChildren()))
            total_cas += cas
            if cas:
                self.game.removeModelsFromUnit(enemy, cas)
                if getattr(self.game, 'psychology', None):
                    self.game.psychology.check_heavy_casualties(enemy, 'shooting')

        summary = (f"Bombardment: {total_hit} under template, {total_cas} slain "
                   f"(centre S{s_central} AP-{ap_central}, rest S{strength} AP-{ap})")
        print(summary)
        self.game.debugText.setText(summary)
        battle_log(summary, 'good' if total_cas else 'combat')

    def _wound_unsaved(self, model, strength, ap):
        """Roll To Wound then armour save; True if a model is slain."""
        toughness = model.get_toughness() if hasattr(model, 'get_toughness') else 4
        if random.randint(1, 6) < wound_target(strength, toughness):
            return False
        return not check_armor_save(model, model.armor_save, ap)

    # ─── Visuals ────────────────────────────────────────────────────

    def _draw_circle(self, centre, radius, color, segments=48):
        ls = LineSegs()
        ls.setColor(*color)
        ls.setThickness(2.0)
        for i in range(segments + 1):
            a = 2 * math.pi * i / segments
            x = centre.x + radius * math.cos(a)
            y = centre.y + radius * math.sin(a)
            if i == 0:
                ls.moveTo(x, y, centre.z + 0.2)
            else:
                ls.drawTo(x, y, centre.z + 0.2)
        np = render.attachNewNode(ls.create())
        np.setName("BombardCircle")
        self.circles.append(np)
        return np

    def _place_template(self, centre, radius, color):
        self._draw_circle(centre, radius, color)

    def _clear_circles(self):
        for np in self.circles:
            np.removeNode()
        self.circles = []

    def cleanup(self):
        """Remove Bombardment templates/rings (called on shooting phase exit)."""
        self._clear_circles()
