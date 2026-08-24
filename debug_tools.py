"""Developer debug mode: free unit movement and special-rule test helpers.

Only active when the ``WH_DEBUG`` environment variable is set or ``--debug`` is
passed on the command line, so nothing is bound or drawn in a normal game.

    $env:WH_DEBUG = "1"; python game.py

Press F12 in game to toggle the tools on; F12 again restores normal play.
"""

from __future__ import annotations

import os
import math
import random
import sys

from direct.showbase.DirectObject import DirectObject
from direct.task.Task import Task
from panda3d.core import LineSegs, Point3, TextNode

import gui_theme
from collision_masks import CollisionMask as CM
from special_rules import SPECIAL_RULE_BUILDERS
from psychology import command_range

NUDGE_COARSE = 0.5
NUDGE_FINE = 0.1
ROT_COARSE = 15.0
ROT_FINE = 1.0

# Keywords that have a coded engine effect, cycled with shift-b / shift-n.
TESTABLE_RULES = sorted(SPECIAL_RULE_BUILDERS)

# Params that make a keyword's coded effect measurable (the catalogue omits them).
RULE_TEST_PARAMS = {"regeneration": "5+", "fly": "9"}

HELP_TEXT = """
── DEBUG MODE ──────────────────────────────────────────────
  F12          toggle debug mode off
  h            print this help

  Movement (bypasses allowance, terrain, sweeps, collision)
  g            grab / drop the unit under the cursor
  arrows       nudge 0.5"   (+shift = 0.1")
  z / x        rotate 15deg (+shift = 1deg)
  b / n        select previous / next unit

  Turn & phase
  r            reset all turn flags on every unit
  1 2 3 4      jump to Strategy / Movement / Shooting / Combat

  Combat
  e            engage selected unit with the nearest enemy
  shift-e      disengage selected unit
  k            remove 1 model  (shift-k removes 5)

  Special rules
  shift-b / n  cycle the rule under test
  shift-g      grant it to the selected unit
  shift-d      revoke it from the selected unit
  i            dump selected unit   (shift-i dumps all)

  Dice & state
  y            loaded d6: off -> all 1s -> all 6s -> off
  F8           snapshot to debug_snapshot.json (shift-F8 restores)

  Always on while debug mode is active: a ring around each General showing
  its Command range (the Inspiring Presence Leadership bubble).
────────────────────────────────────────────────────────────
"""


def debug_enabled() -> bool:
    """True when the developer asked for debug mode at launch."""
    return bool(os.environ.get("WH_DEBUG")) or "--debug" in sys.argv


class DebugTools(DirectObject):
    """Free-movement and rule-testing tools, toggled with F12."""

    SNAPSHOT = "debug_snapshot.json"

    def __init__(self, game):
        super().__init__()
        self.game = game
        self.enabled = False
        self.grabbed = None
        self.rule_index = 0
        self.loaded_dice = None
        self._orig_randint = None
        self.overlay = None
        self._command_rings = []
        self.accept("f12", self.toggle)
        print("[debug] debug mode available - press F12")

    # ─── Activation ───────────────────────────────────────────────────────

    def toggle(self):
        self.disable() if self.enabled else self.enable()

    def enable(self):
        if self.enabled:
            return
        self.enabled = True
        self._bind()
        if self.overlay is None:
            self.overlay = gui_theme.styled_text(
                text="", pos=(0.30, 0.92), scale=0.038,
                fg=gui_theme.GOLD, align=TextNode.ALeft,
            )
        self.overlay.show()
        self.addTask(self._overlay_task, "debugOverlay")
        print(HELP_TEXT)

    def disable(self):
        if not self.enabled:
            return
        self.drop()
        self.set_loaded_dice(None)
        self._unbind()
        self.removeTask("debugOverlay")
        self._clear_command_rings()
        if self.overlay:
            self.overlay.hide()
        self.enabled = False
        print("[debug] debug mode off")

    def _bind(self):
        self.accept("h", lambda: print(HELP_TEXT))
        self.accept("g", self.grab)
        self.accept("arrow_up", self.nudge, [0, NUDGE_COARSE])
        self.accept("arrow_down", self.nudge, [0, -NUDGE_COARSE])
        self.accept("arrow_left", self.nudge, [-NUDGE_COARSE, 0])
        self.accept("arrow_right", self.nudge, [NUDGE_COARSE, 0])
        self.accept("shift-arrow_up", self.nudge, [0, NUDGE_FINE])
        self.accept("shift-arrow_down", self.nudge, [0, -NUDGE_FINE])
        self.accept("shift-arrow_left", self.nudge, [-NUDGE_FINE, 0])
        self.accept("shift-arrow_right", self.nudge, [NUDGE_FINE, 0])
        self.accept("z", self.rotate, [ROT_COARSE])
        self.accept("x", self.rotate, [-ROT_COARSE])
        self.accept("shift-z", self.rotate, [ROT_FINE])
        self.accept("shift-x", self.rotate, [-ROT_FINE])
        self.accept("b", self.cycle_unit, [-1])
        self.accept("n", self.cycle_unit, [1])
        self.accept("r", self.reset_flags)
        for i, phase in enumerate(self.game.fsm.PHASES):
            self.accept(str(i + 1), self.goto_phase, [phase])
        self.accept("e", self.engage_nearest)
        self.accept("shift-e", self.disengage)
        self.accept("k", self.kill_models, [1])
        self.accept("shift-k", self.kill_models, [5])
        self.accept("shift-b", self.cycle_rule, [-1])
        self.accept("shift-n", self.cycle_rule, [1])
        self.accept("shift-g", self.grant_rule)
        self.accept("shift-d", self.revoke_rule)
        self.accept("i", self.dump)
        self.accept("shift-i", self.dump_all)
        self.accept("y", self.cycle_loaded_dice)
        self.accept("f8", self.snapshot)
        self.accept("shift-f8", self.restore)

    def _unbind(self):
        for event in ("h", "g", "z", "x", "shift-z", "shift-x",
                      "b", "n", "shift-b", "shift-n", "r", "e", "shift-e", "k",
                      "shift-k", "shift-g", "shift-d", "i", "shift-i", "y",
                      "f8", "shift-f8"):
            self.ignore(event)
        for arrow in ("arrow_up", "arrow_down", "arrow_left", "arrow_right"):
            self.ignore(arrow)
            self.ignore("shift-" + arrow)
        for i in range(len(self.game.fsm.PHASES)):
            self.ignore(str(i + 1))

    # ─── Selection ────────────────────────────────────────────────────────

    @property
    def selected(self):
        unit = getattr(self.game, "unitToMove", None)
        if unit is None or unit not in self.game.units:
            unit = self.game.units[0] if self.game.units else None
            self.game.unitToMove = unit
        return unit

    def select(self, unit):
        self.game.unitToMove = unit
        return unit

    def cycle_unit(self, step):
        if not self.game.units:
            return
        try:
            index = self.game.units.index(self.selected)
        except ValueError:
            index = 0
        unit = self.game.units[(index + step) % len(self.game.units)]
        self.select(unit)
        print(f"[debug] selected {unit.unitName}")

    def unit_under_cursor(self):
        if not base.mouseWatcherNode.hasMouse():
            return None
        p_mouse = base.mouseWatcherNode.getMouse()
        p_from, p_to = Point3(), Point3()
        base.camLens.extrude(p_mouse, p_from, p_to)
        p_from = render.getRelativePoint(base.cam, p_from)
        p_to = render.getRelativePoint(base.cam, p_to)
        result = self.game.world.rayTestClosest(p_from, p_to, CM.HOVER_PICK)
        if not result.hasHit():
            return None
        name = result.getNode().getName()
        if not name.startswith("UnitCollision-"):
            return None
        name = name.replace("UnitCollision-", "")
        for unit in self.game.units:
            if unit.unitName == name:
                return unit
        return None

    # ─── Free movement ────────────────────────────────────────────────────

    def _ground_point(self):
        """World point where the mouse ray crosses the z=0 plane."""
        if not base.mouseWatcherNode.hasMouse():
            return None
        p_mouse = base.mouseWatcherNode.getMouse()
        p_from, p_to = Point3(), Point3()
        base.camLens.extrude(p_mouse, p_from, p_to)
        p_from = render.getRelativePoint(base.cam, p_from)
        p_to = render.getRelativePoint(base.cam, p_to)
        dz = p_to.z - p_from.z
        if abs(dz) < 1e-6:
            return None
        t = -p_from.z / dz
        return Point3(p_from.x + (p_to.x - p_from.x) * t,
                      p_from.y + (p_to.y - p_from.y) * t, 0)

    def grab(self):
        if self.grabbed is not None:
            self.drop()
            return
        unit = self.unit_under_cursor() or self.selected
        if unit is None:
            return
        self.select(unit)
        self.grabbed = unit
        self.addTask(self._grab_task, "debugGrab")
        print(f"[debug] grabbed {unit.unitName} - press g to drop")

    def drop(self):
        self.removeTask("debugGrab")
        if self.grabbed is None:
            return
        unit, self.grabbed = self.grabbed, None
        self._settle(unit)
        pos = unit.bodyNP.getPos()
        print(f"[debug] dropped {unit.unitName} at "
              f"({pos.x:.1f}, {pos.y:.1f}) H={unit.bodyNP.getH():.0f}")

    def _grab_task(self, task):
        unit = self.grabbed
        if unit is None or unit not in self.game.units:
            self.grabbed = None
            return Task.done
        point = self._ground_point()
        if point is not None:
            unit.bodyNP.setPos(point.x, point.y, 0)
        return Task.cont

    def _settle(self, unit):
        """Re-seat the unit's models on the terrain after a free move."""
        if unit is None or unit.model.isEmpty():
            return
        self.game.movement.alignModelsToHillNormal(unit)

    def nudge(self, dx, dy):
        unit = self.selected
        if unit is None:
            return
        pos = unit.bodyNP.getPos()
        unit.bodyNP.setPos(pos.x + dx, pos.y + dy, pos.z)
        self._settle(unit)

    def rotate(self, degrees):
        unit = self.selected
        if unit is None:
            return
        unit.bodyNP.setH(unit.bodyNP.getH() + degrees)
        self._settle(unit)

    # ─── Turn & phase control ─────────────────────────────────────────────

    def reset_flags(self):
        for unit in self.game.units:
            unit.hasMovedThisTurn = False
            unit.hasAttackedThisTurn = False
            unit.attemptedRallyThisTurn = False
            unit.chargedThisTurn = False
            unit.countsAsChargedNextTurn = False
            unit.cannotChargeThisTurn = False
            unit.isChargingMove = False
            unit.panicTestedThisPhase = False
            unit.fledThisPhase = False
            unit.usedStubborn = False
            unit.madePursuitChoice = False
            unit.startOfPhaseModels = unit.unit.nmodels
            if unit.state not in ("InCombat", "IsFleeing"):
                unit.request("Idle")
            unit.updateTextNode()
        print("[debug] turn flags reset on all units")

    def goto_phase(self, phase):
        self.game.fsm.currentPhaseIndex = self.game.fsm.PHASES.index(phase)
        self.game.fsm.request(phase)
        print(f"[debug] forced {phase}")

    # ─── Combat setup ─────────────────────────────────────────────────────

    def _enemies_of(self, unit):
        return (self.game.player2Units if unit in self.game.player1Units
                else self.game.player1Units)

    def _flank_of(self, attacker, defender):
        """Which facing of *defender* the attacker sits in: front/flank/rear."""
        local = attacker.bodyNP.getPos(defender.bodyNP)
        width = max(getattr(defender, "unitWidth", 1.0), 0.01)
        height = max(getattr(defender, "unitHeight", 1.0), 0.01)
        if abs(local.x) * height > abs(local.y) * width:
            return "flank"
        return "front" if local.y > 0 else "rear"

    def engage_nearest(self):
        unit = self.selected
        if unit is None:
            return
        enemies = [u for u in self._enemies_of(unit) if u is not unit]
        if not enemies:
            print("[debug] no enemy units to engage")
            return
        pos = unit.bodyNP.getPos()
        target = min(enemies, key=lambda u: (u.bodyNP.getPos() - pos).length())
        flank = self._flank_of(unit, target)

        unit.request("InCombat")
        unit.isInCombat = True
        if target.state != "InCombat":
            target.request("InCombat")
        target.isInCombat = True
        if target not in unit.isInCombatWith:
            unit.isInCombatWith.append(target)
            unit.isInCombatFlank.append("front")
        if unit not in target.isInCombatWith:
            target.isInCombatWith.append(unit)
            target.isInCombatFlank.append(flank)
        unit.updateTextNode()
        target.updateTextNode()
        # Rules that ask whether a combat predates the phase (Pursuit into a
        # New Combat) would otherwise read a hand-made engagement as brand new.
        unit.startOfPhaseEngaged = True
        target.startOfPhaseEngaged = True
        print(f"[debug] {unit.unitName} engaged {target.unitName} ({flank})")

    def disengage(self):
        unit = self.selected
        if unit is None:
            return
        for other in list(unit.isInCombatWith):
            if unit in other.isInCombatWith:
                index = other.isInCombatWith.index(unit)
                other.isInCombatWith.pop(index)
                if index < len(other.isInCombatFlank):
                    other.isInCombatFlank.pop(index)
            if not other.isInCombatWith:
                other.request("Idle")
                other.startOfPhaseEngaged = False
            other.updateTextNode()
        unit.isInCombatWith = []
        unit.isInCombatFlank = []
        unit.request("Idle")
        unit.startOfPhaseEngaged = False
        unit.updateTextNode()
        print(f"[debug] {unit.unitName} disengaged")

    def kill_models(self, count):
        unit = self.selected
        if unit is None:
            return
        name = unit.unitName
        self.game.movement.removeModelsFromUnit(unit, count)
        print(f"[debug] removed {count} model(s) from {name}")

    # ─── Special rules ────────────────────────────────────────────────────

    @property
    def rule_keyword(self):
        return TESTABLE_RULES[self.rule_index % len(TESTABLE_RULES)]

    def cycle_rule(self, step):
        self.rule_index = (self.rule_index + step) % len(TESTABLE_RULES)
        print(f"[debug] rule under test: {self.rule_keyword}")

    def grant_rule(self, keyword=None, param=None):
        unit = self.selected
        if unit is None:
            return
        keyword = (keyword or self.rule_keyword).lower()
        builder = SPECIAL_RULE_BUILDERS.get(keyword)
        if builder is None:
            print(f"[debug] no coded builder for '{keyword}'")
            return
        model = unit.unit.model
        entry = builder(model, param or RULE_TEST_PARAMS.get(keyword), None)
        rules = [r for r in model.special_rules
                 if r.get("name", "").lower() != entry["name"].lower()]
        rules.append(entry)
        model.special_rules = rules
        unit.updateTextNode()
        print(f"[debug] granted {entry['name']} to {unit.unitName}: {entry}")

    def revoke_rule(self, keyword=None):
        unit = self.selected
        if unit is None:
            return
        keyword = (keyword or self.rule_keyword).lower()
        model = unit.unit.model
        before = len(model.special_rules)
        model.special_rules = [r for r in model.special_rules
                               if r.get("name", "").lower() != keyword]
        unit.updateTextNode()
        removed = before - len(model.special_rules)
        print(f"[debug] revoked '{keyword}' from {unit.unitName} ({removed} removed)")

    @staticmethod
    def _rule_hooks(rule):
        """Keys carrying an engine effect, as opposed to display metadata."""
        return sorted(k for k in rule if k not in ("name", "description", "tag"))

    def _rule_summary(self, unit):
        coded, flags = [], []
        for rule in unit.unit.model.special_rules:
            hooks = self._rule_hooks(rule)
            (coded if hooks else flags).append(rule.get("name", "?"))
        return coded, flags

    # ─── Inspection ───────────────────────────────────────────────────────

    def dump(self, unit=None):
        unit = unit or self.selected
        if unit is None:
            return
        pos = unit.bodyNP.getPos()
        player = 1 if unit in self.game.player1Units else 2
        print(f"\n── {unit.unitName} (player {player}) ─────────────────────")
        print(f"  state    : {unit.state}   pos ({pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f})"
              f"  H {unit.bodyNP.getH():.1f}")
        print(f"  models   : {unit.unit.nmodels} "
              f"(battle start {unit.startOfBattleModels}, "
              f"phase start {unit.startOfPhaseModels})")
        print(f"  flags    : moved={unit.hasMovedThisTurn} "
              f"attacked={unit.hasAttackedThisTurn} charged={unit.chargedThisTurn} "
              f"noCharge={unit.cannotChargeThisTurn} charging={unit.isChargingMove}")
        print(f"           : panicked={unit.panicTestedThisPhase} "
              f"stubbornUsed={unit.usedStubborn} rallied={unit.attemptedRallyThisTurn} "
              f"skirmisher={unit.isSkirmisher}")
        psy = getattr(self.game, "psychology", None)
        if psy is not None:
            ld, general = psy.leadership_of(unit)
            source = "General" if getattr(unit, "isGeneral", False) else (
                general.unitName if general is not None else "own")
            bsb = psy.battle_standard_of(unit)
            standard = ("self" if getattr(unit, "isBSB", False)
                        else bsb.unitName if bsb is not None else "none")
            print(f"  command  : Ld {ld} ({source})   re-rolls: {standard}")
        if unit.isInCombatWith:
            pairs = ", ".join(
                f"{u.unitName}({f})" for u, f in
                zip(unit.isInCombatWith, unit.isInCombatFlank + ["?"] * len(unit.isInCombatWith))
            )
            print(f"  combat   : {pairs}")
        for rule in unit.unit.model.special_rules:
            hooks = self._rule_hooks(rule)
            mark = "CODED" if hooks else "flag "
            detail = f" -> {', '.join(hooks)}" if hooks else ""
            print(f"  [{mark}] {rule.get('name', '?')}{detail}")

    def dump_all(self):
        for unit in list(self.game.units):
            self.dump(unit)

    # ─── Dice ─────────────────────────────────────────────────────────────

    def cycle_loaded_dice(self):
        nxt = {None: 1, 1: 6, 6: None}[self.loaded_dice]
        self.set_loaded_dice(nxt)
        print(f"[debug] loaded d6: {self.loaded_dice or 'off'} "
              f"(logic rolls only, physics dice unaffected)")

    def set_loaded_dice(self, value):
        """Force every random.randint(1, 6) to *value*; None restores normal rolls."""
        if value is None:
            if self._orig_randint is not None:
                random.randint = self._orig_randint
                self._orig_randint = None
        else:
            if self._orig_randint is None:
                self._orig_randint = random.randint
            original = self._orig_randint

            def loaded(a, b, _value=value, _original=original):
                if (a, b) == (1, 6):
                    return _value
                return _original(a, b)

            random.randint = loaded
        self.loaded_dice = value

    # ─── Snapshots ────────────────────────────────────────────────────────

    def snapshot(self):
        self.drop()
        self.game.save_game_state(self.SNAPSHOT)
        print(f"[debug] snapshot written to {self.SNAPSHOT}")

    def restore(self):
        if not os.path.exists(self.SNAPSHOT):
            print(f"[debug] no {self.SNAPSHOT} to restore")
            return
        self.drop()
        self.game.load_game_state(self.SNAPSHOT)
        print(f"[debug] restored {self.SNAPSHOT}")

    # ─── Command range rings ──────────────────────────────────────────────

    def _clear_command_rings(self):
        for node in self._command_rings:
            node.removeNode()
        self._command_rings = []

    def _side_of(self, unit):
        if unit in self.game.player1Units:
            return 1
        if unit in self.game.player2Units:
            return 2
        return getattr(unit, "_player", 2)   # a joined character leaves both lists

    def _update_command_rings(self):
        """Ring each General's and Battle Standard's Command range."""
        self._clear_command_rings()
        for unit in self.game.units:
            if unit.bodyNP.isEmpty():
                continue
            general = getattr(unit, "isGeneral", False)
            if not general and not getattr(unit, "isBSB", False):
                continue
            body = unit.bodyNP
            centre = body.getPos(body.getTop())   # a joined leader sits under its host
            if general:
                colour = ((0.3, 1.0, 0.4, 0.8) if self._side_of(unit) == 1
                          else (1.0, 0.4, 0.3, 0.8))
            else:
                colour = ((0.4, 0.7, 1.0, 0.8) if self._side_of(unit) == 1
                          else (1.0, 0.8, 0.3, 0.8))
            self._command_rings.append(
                self._draw_ring(centre, command_range(unit), colour))

    @staticmethod
    def _draw_ring(centre, radius, colour, segments=64):
        ls = LineSegs()
        ls.setColor(*colour)
        ls.setThickness(2.0)
        for i in range(segments + 1):
            a = 2 * math.pi * i / segments
            x = centre.x + radius * math.cos(a)
            y = centre.y + radius * math.sin(a)
            if i == 0:
                ls.moveTo(x, y, centre.z + 0.2)
            else:
                ls.drawTo(x, y, centre.z + 0.2)
        node = render.attachNewNode(ls.create())
        node.setName("DebugCommandRing")
        return node

    # ─── Overlay ──────────────────────────────────────────────────────────

    def _overlay_task(self, task):
        unit = self.selected
        lines = [f"DEBUG  phase {self.game.fsm.state}  round "
                 f"{self.game.roundCounter.currentRoundPlayer}"
                 f" P{self.game.roundCounter.current_player}"]
        if unit is None:
            lines.append("no units")
        else:
            pos = unit.bodyNP.getPos()
            lines.append(f"{unit.unitName}  [{unit.state}]"
                         f"{'  GRABBED' if self.grabbed is unit else ''}")
            lines.append(f"  ({pos.x:6.2f},{pos.y:6.2f})  H {unit.bodyNP.getH():6.1f}"
                         f"  x{unit.unit.nmodels}")
            lines.append(f"  moved {int(unit.hasMovedThisTurn)}"
                         f"  atk {int(unit.hasAttackedThisTurn)}"
                         f"  chg {int(unit.chargedThisTurn)}"
                         f"  combat {int(unit.isInCombat)}")
            coded, flags = self._rule_summary(unit)
            lines.append(f"  coded: {', '.join(coded) or '-'}")
            lines.append(f"  flags: {', '.join(flags) or '-'}")
        lines.append(f"rule under test: {self.rule_keyword}"
                     f"   dice: {self.loaded_dice or 'random'}")
        self.overlay.setText("\n".join(lines))
        self._update_command_rings()
        return Task.cont
