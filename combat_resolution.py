"""
Combat resolution subsystem.

Handles all combat-related logic including:
- Contact detection (checkUnitContactSmall)
- Charge/flee interval animations
- Flank detection
- Battle resolution (attack sequences, wound calculation, morale)
- Post-combat outcomes (give ground, fall back in good order, flee)

All methods operate on the game instance passed during construction.
"""

import math
from types import SimpleNamespace

from panda3d.core import Vec2, Vec3, NodePath


def _stat_int(characteristics: dict, key: str, default: int = 4) -> int:
    """Safely read a numeric stat from a characteristics dict.
    Returns *default* if the value is missing or non-numeric (e.g. '-')."""
    try:
        return int(characteristics[key])
    except (KeyError, ValueError, TypeError):
        return default

from panda3d.bullet import BulletBoxShape
from panda3d.core import LRotationf
from direct.interval.LerpInterval import LerpPosHprInterval, LerpPosInterval
from direct.interval.IntervalGlobal import Sequence, Parallel, Wait
from direct.interval.FunctionInterval import Func
from direct.task.Task import Task

from dice import Dice, checkDice
from battleFunctions import (MIN_IMPACT_HIT_CHARGE, impact_hit_report,
                             resolve_impact_hits, simulate_battle)
from characters import JOIN_TAG
from special_rules import (board_edge_distance, charge_roll, max_charge_range,
                           max_pursuit_range, should_use_swiftstride,
                           unit_has_swiftstride)
from psychology import (MAX_RANK_BONUS, battle_standard_bonus, break_test_outcome,
                       massed_infantry_bonus, overwhelmed, rank_bonus,
                       should_reroll_break, should_use_stubborn,
                       side_unit_strength, stubborn_available,
                       unit_strength_total)
from post_combat import (GIVE_GROUND, fall_back_roll, flee_direction, flee_roll,
                         flees_from, give_ground_direction, restraint_test,
                         winner_response)
from rules_log import rule_log, rule_skipped

# The Swiftstride die is thrown in its own colour so it is never mistaken for
# one of the dice a Charge or Fall Back roll discards between.
SWIFTSTRIDE_DIE_COLOR = (0.85, 0.05, 0.05, 1)


class CombatResolver:
    """Encapsulates all combat resolution logic for the game."""

    def __init__(self, game):
        self.game = game
        # Dice state used during charge/flee intervals
        self.terningerCharge = []
        self.terningerFlee = []

    # ─── Contact Detection ────────────────────────────────────────────────

    def checkUnitContactSmall(self, unit):
        contacts = self.game.world.contactTest(unit.bodyNP.node())
        for contact in contacts.getContacts():
            mpoint = contact.getManifoldPoint()
            if 'UnitCollision-' in contact.getNode1().getName():
                return contact
        return None

    # ─── Charge & Charge Reaction ─────────────────────────────────────────

    async def chargeAndChargeReaction(self, unit, c, oposUnit, orotUnit, task):
        chargeYesNo = ["Yes", "No"]
        if self.game.autoCharge or (self.game.roundCounter.current_player in [1, 2] and self.game.AIplayer2.active):
            cynchoice = "Yes"
        else:
            cynchoice = await taskMgr.add(self.game.makeChoiceNew(chargeYesNo, Vec3(-20, 0, 10)))

        if cynchoice == "Yes":
            print("Charging into combat...")

            chargeReaction = ["hold", "flee"]
            if self.game.autoHold or (self.game.roundCounter.current_player in [1, 2] and self.game.AIplayer2.active):
                crchoice = "hold"
            else:
                crchoice = await taskMgr.add(self.game.makeChoiceNew(chargeReaction, Vec3(20, 0, 10)))
            defenderNP = render.find(f"**/{c.getNode1().getName()}")
            if crchoice == "hold":
                print("Defender holds position.")

                flank, angleToRotate = self.getFlankFromContact(unit, c)

                unit.hasMovedThisTurn = True
                unit.updateTextNode()
                taskMgr.add(self.chargeInterval, "chargeIntervalTask",
                            extraArgs=[unit, defenderNP, angleToRotate, oposUnit, orotUnit, flank],
                            appendTask=False)

            elif crchoice == "flee":
                flank, angleToRotate = self.getFlankFromContact(unit, c)
                print("Defender flees!")
                loserUnit = self.game.getSelectedUnit(defenderNP)
                loserUnit.request("IsFleeing")
                taskMgr.add(self.fleeInterval, "fleeIntervalTask",
                            extraArgs=[unit, defenderNP, angleToRotate, oposUnit, orotUnit],
                            appendTask=False)
                fleeDirection = defenderNP.getPos() - unit.bodyNP.getPos()
                storeRotation = defenderNP.getHpr()
                defenderNP.lookAt(defenderNP.getPos() + fleeDirection)
                fleeRotation = defenderNP.getHpr()
                defenderNP.setHpr(storeRotation)

        else:
            print("Charge cancelled.")
            unit.bodyNP.setPos(oposUnit)
            unit.bodyNP.setHpr(orotUnit)
            self.game.startTaskFunction(self.game.taskLoopPathTowardsMouse, "taskLoopPathTowardsMouse")
            self.game.autoCharge = False
            self.game.autoHold = False

        return task.done

    # ─── Flee Interval ────────────────────────────────────────────────────

    async def fleeInterval(self, unit, defenderNP, angleToRotate, oposUnit, orotUnit):
        fleeingUnit = self.game.getSelectedUnit(defenderNP.node())
        fleePos = defenderNP.getPos()
        chargeBonus = await self.swiftstrideChargeChoice(unit)
        fleeBonus = await self.swiftstrideChoice(
            fleeingUnit, 'flee',
            distance_to_edge=board_edge_distance(fleePos.x, fleePos.y))

        self.terningerCharge = []
        for i in range(3 if chargeBonus else 2):
            swift = chargeBonus and i == 2
            terning = Dice(self.game.world, position=Vec3(-20 + i * 2, 0, 10), size=1.0,
                           body_color=SWIFTSTRIDE_DIE_COLOR if swift else None)
            self.terningerCharge.append(terning)
        for terning in self.terningerCharge:
            terning.roll()
        chtask = taskMgr.add(checkDice, "checkDiceTaskCharge",
                             extraArgs=[self.terningerCharge], appendTask=True)

        self.terningerFlee = []
        for i in range(3 if fleeBonus else 2):
            swift = fleeBonus and i == 2
            terning = Dice(self.game.world, position=Vec3(20 + i * 2, 0, 10), size=1.0,
                           color=(1, 0, 0, 1),
                           body_color=SWIFTSTRIDE_DIE_COLOR if swift else None)
            self.terningerFlee.append(terning)
        for terning in self.terningerFlee:
            terning.roll()
        await taskMgr.add(checkDice, "checkDiceTaskFlee",
                          extraArgs=[self.terningerFlee], appendTask=True)
        await chtask
        chdice = []
        for terning in self.terningerCharge:
            chdice.append(terning.currentValue)
        print("Charge dice results:", chdice)
        fldice = []
        for terning in self.terningerFlee:
            fldice.append(terning.currentValue)
        print("Flee dice results:", fldice)
        contactPos = unit.bodyNP.getPos()
        contactRot = unit.bodyNP.getHpr()

        shape = unit.bodyNP.node().getShape(0)
        if isinstance(shape, BulletBoxShape):
            half_extents = shape.getHalfExtentsWithMargin()
            width = half_extents.x
            height = half_extents.y
        parent = unit.bodyNP.getParent()
        newnode = render.attachNewNode(f"Temp-{unit.unitName}")
        unit.bodyNP.setPos(oposUnit)
        unit.bodyNP.setHpr(orotUnit)

        newnode.reparentTo(unit.bodyNP)

        rot = LRotationf()
        rot.setHpr(unit.bodyNP.getHpr())
        rgt = rot.getRight()
        dire = contactPos - oposUnit
        angle_between = rgt.dot(dire.normalized())
        if angle_between >= 0:
            sign = 1
        else:
            sign = -1

        newnode.setPos(Vec3(width * sign / unit.bodyNP.getScale().x,
                            height / unit.bodyNP.getScale().y, 0))
        newnode.wrtReparentTo(render)

        unit.bodyNP.setHpr(orotUnit)
        unit.bodyNP.wrtReparentTo(newnode)
        # Rotate the new node smoothly to align with defender
        newnode_hpr = newnode.getHpr()
        positive_h = newnode_hpr.x % 360
        positive_p = newnode_hpr.y % 360
        positive_r = newnode_hpr.z % 360

        if positive_h > 180:
            positive_h -= 360

        if positive_h - contactRot.x > 180:
            contactRot = Vec3(contactRot.x + 360, contactRot.y, contactRot.z)
        newnode.setHpr(positive_h, positive_p, positive_r)

        wheel1Angle = contactRot.x - orotUnit.x
        newnode.setHpr(contactRot)
        wheel1Pos = newnode.getPos(render)

        if 1:
            direction = self.game.playerNP.getPos() - wheel1Pos
            wdistance = abs(math.radians(wheel1Angle) * width * 2)
            cdistance = self.game.moveArceDistance - wdistance

        rough = self.chargeThroughDifficult(unit, oposUnit)
        chdist = _stat_int(unit.unit.model.characteristics, 'M') + charge_roll(chdice, rough)
        for rule in unit.unit.model.special_rules:
            if rule.get('mountUnit'):
                chdist = _stat_int(rule['mountUnit'].model.characteristics, 'M') + charge_roll(chdice, rough)
        fldist = sum(fldice) + fleeBonus
        if chdist < wdistance:
            angle = math.degrees(chdist / width)
            contactRot = Vec3(orotUnit.x + angle, contactRot.y, contactRot.z) * wheel1Angle / abs(wheel1Angle)

        newnode.setHpr(positive_h, positive_p, positive_r)

        rotation_interval = LerpPosHprInterval(
            newnode,
            duration=1.5,
            pos=newnode.getPos(),
            hpr=contactRot,
            blendType='easeInOut'
        )
        await rotation_interval
        if chdist < wdistance:
            unit.bodyNP.wrtReparentTo(parent)

        ocdistance = cdistance
        if chdist < wdistance + cdistance:
            cdistance = chdist - wdistance
            cdistance = max(cdistance, 0)

        angle = contactRot.x
        vector = Vec2(-math.sin(math.radians(angle)), math.cos(math.radians(angle)))
        cmove = chdist - wdistance
        pos_interval = LerpPosHprInterval(
            newnode,
            duration=1.5,
            pos=wheel1Pos + Vec3(vector.x, vector.y, 0) * cmove,
            hpr=contactRot,
            blendType='easeInOut'
        )

        rotation_interval = LerpPosHprInterval(
            defenderNP,
            duration=0.5,
            pos=defenderNP.getPos(),
            hpr=contactRot,
            blendType='easeInOut'
        )
        await rotation_interval

        angle = contactRot.x
        vector = Vec2(-math.sin(math.radians(angle)), math.cos(math.radians(angle)))
        pos_interval2 = LerpPosHprInterval(
            defenderNP,
            duration=1.5,
            pos=defenderNP.getPos() + Vec3(vector.x, vector.y, 0) * fldist,
            hpr=contactRot,
            blendType='easeInOut'
        )

        par = Parallel(
            pos_interval,
            pos_interval2
        )
        defenderUnit = self.game.getSelectedUnit(defenderNP.node())
        taskMgr.add(self.game.checkFleeCaught, "checkFleeCaughtTask",
                     extraArgs=[defenderUnit, unit], appendTask=True)
        await par
        if taskMgr.hasTaskNamed("checkFleeCaughtTask"):
            taskMgr.remove("checkFleeCaughtTask")
        for terning in self.terningerCharge:
            terning.remove(self.game.world)
        for terning in self.terningerFlee:
            terning.remove(self.game.world)
        unit.bodyNP.wrtReparentTo(parent)

        unit.request("Moved")

        return

    # ─── Dice Rolling ─────────────────────────────────────────────────────

    async def rullTerninger(self, antall, bonus=False):
        """Roll *antall* dice together. With *bonus*, the last one is the
        Swiftstride die and is thrown in its own colour."""
        terninger = []
        for i in range(antall):
            swift = bonus and i == antall - 1
            terning = Dice(self.game.world, position=Vec3(0 + i * 4, 0, 10), size=1.0,
                           body_color=SWIFTSTRIDE_DIE_COLOR if swift else None)
            terninger.append(terning)
        for terning in terninger:
            terning.roll()
        await taskMgr.add(checkDice, "checkDiceTask", extraArgs=[terninger], appendTask=True)

        chdice = []
        for terning in terninger:
            chdice.append(terning.currentValue)
        return terninger, chdice

    async def swiftstrideChoice(self, unit, kind, distance_to_edge=None):
        """Whether *unit* spends Swiftstride's bonus die on this roll.

        Asked *before* the dice are thrown, as the rule requires, so the choice
        cannot be made knowing the result.
        """
        if not unit_has_swiftstride(unit):
            return False
        if self.game.roundCounter.current_player in [1, 2] and self.game.AIplayer2.active:
            use = should_use_swiftstride(kind, distance_to_edge)
        else:
            choice = [unit.unitName + '\nSwiftstride +D6',
                      unit.unitName + '\nNo bonus']
            selected = await taskMgr.add(
                self.game.makeChoiceNew(choice, Vec3(0, 0, 10)))
            use = selected == choice[0]
        print(f"{unit.unit.name} {'takes' if use else 'declines'} its Swiftstride "
              f"{kind} bonus.")
        return use

    async def swiftstrideChargeChoice(self, unit):
        """Swiftstride choice for a charge, or for a pursuit — pursuit moves are
        resolved through the charge machinery with the roll summed instead of
        discarded."""
        if unit.state == "IsPursuing":
            p = unit.bodyNP.getPos()
            return await self.swiftstrideChoice(
                unit, 'pursuit', distance_to_edge=board_edge_distance(p.x, p.y))
        return await self.swiftstrideChoice(unit, 'charge')

    def chargeRangeText(self, unit, maxmove):
        """The roll-needed readout. A pursuit adds no Movement and sums 2D6, so
        it reaches further than a charge's discard-lowest roll."""
        swiftstride = unit_has_swiftstride(unit)
        needed = math.ceil(self.game.moveArceDistance)
        if unit.state == "IsPursuing":
            return (f"Roll needed: {needed:.0f}"
                    f"   (max pursuit {max_pursuit_range(swiftstride)}\")")
        return (f"Roll needed: {needed - int(maxmove):.0f}"
                f"   (max charge {max_charge_range(int(maxmove), swiftstride)}\")")

    def chargeThroughDifficult(self, unit, from_pos) -> bool:
        """True if the charge path meets terrain that hinders movement, which
        makes the Charge roll discard the highest die instead of the lowest
        (Rulebook p. 269)."""
        tm = getattr(self.game, 'terrain_manager', None)
        if tm is None:
            return False
        return tm.crosses_difficult(from_pos, self.game.playerNP.getPos())

    # ─── Charge Interval ──────────────────────────────────────────────────

    async def chargeInterval(self, unit, defenderNP, angleToRotate, oposUnit, orotUnit, flank, chdice=None):
        # Skirmishers charge straight in — no wheel, no flank-align pivot — but
        # the charge roll is still made and must reach the target to connect.
        if getattr(unit, 'isSkirmisher', False):
            await self._skirmishChargeInterval(unit, defenderNP, oposUnit, orotUnit, flank, chdice)
            return
        maxmove = _stat_int(unit.unit.model.characteristics, 'M')
        durIntConst = 1.0
        for rule in unit.unit.model.special_rules:
            if rule.get('mountUnit'):
                maxmove = _stat_int(rule['mountUnit'].model.characteristics, 'M')
        if unit.state == "IsPursuing":
            maxmove = 0
        self.game.diceInfoText.setText(self.chargeRangeText(unit, maxmove))
        if not self.game.autoRoll:
            bonus = await self.swiftstrideChargeChoice(unit)
            terninger, chdice = await self.rullTerninger(3 if bonus else 2, bonus)

        else:
            while self.game.attackSequence2.isPlaying():
                await Task.pause(0.5)
            await Task.pause(0.5)
            if chdice is None:
                chdice = [6, 6]
            terninger = []
        self.game.autoCharge = False
        self.game.autoHold = False
        print("Charge dice results:", chdice)
        contactPos = unit.bodyNP.getPos()
        contactRot = unit.bodyNP.getHpr()

        shape = unit.bodyNP.node().getShape(0)
        if isinstance(shape, BulletBoxShape):
            half_extents = shape.getHalfExtentsWithMargin()
            width = half_extents.x
            height = half_extents.y
        parent = unit.bodyNP.getParent()
        newnode = render.attachNewNode(f"Temp-{unit.unitName}")
        unit.bodyNP.setPos(oposUnit)
        unit.bodyNP.setHpr(orotUnit)

        newnode.reparentTo(unit.bodyNP)

        rot = LRotationf()
        rot.setHpr(unit.bodyNP.getHpr())
        rgt = rot.getRight()
        dire = (contactPos + Vec3(-math.sin(math.radians(contactRot.x)),
                                   math.cos(math.radians(contactRot.x)), 0) * height) - \
               (oposUnit + Vec3(-math.sin(math.radians(orotUnit.x)),
                                 math.cos(math.radians(orotUnit.x)), 0) * height)

        angle_between = rgt.dot(dire.normalized())
        if angle_between >= 0:
            sign = 1
        else:
            sign = -1

        newnode.setPos(Vec3(width * sign / unit.bodyNP.getScale().x,
                            height / unit.bodyNP.getScale().y, 0))
        newnode.wrtReparentTo(render)

        unit.bodyNP.setHpr(orotUnit)
        unit.bodyNP.wrtReparentTo(newnode)
        newnode_hpr = newnode.getHpr()
        positive_h = newnode_hpr.x % 360
        positive_p = newnode_hpr.y % 360
        positive_r = newnode_hpr.z % 360

        if positive_h > 180:
            positive_h -= 360

        if positive_h < 0:
            positive_h += 360

        if positive_h - contactRot.x > 180:
            contactRot = Vec3(contactRot.x + 360, contactRot.y, contactRot.z)

        newnode.setHpr(positive_h, positive_p, positive_r)

        wheel1Angle = contactRot.x - orotUnit.x
        if wheel1Angle > 180:
            wheel1Angle -= 360
        newnode.setHpr(contactRot)
        wheel1Pos = newnode.getPos(render)

        if 1:
            direction = self.game.playerNP.getPos() - wheel1Pos
            wdistance = abs(math.radians(wheel1Angle) * width * 2)
            cdistance = self.game.moveArceDistance - wdistance

        rough = self.chargeThroughDifficult(unit, oposUnit)
        if rough:
            print("Charging through difficult terrain \u2014 the Charge roll "
                  "discards the highest die.")
        chdist = _stat_int(unit.unit.model.characteristics, 'M') + charge_roll(chdice, rough)
        for rule in unit.unit.model.special_rules:
            if rule.get('mountUnit'):
                chdist = _stat_int(rule['mountUnit'].model.characteristics, 'M') + charge_roll(chdice, rough)
        if unit.state == "IsPursuing":
            chdist = sum(chdice)
        if chdist < wdistance:
            angle = math.degrees(chdist / (width * 2))
            contactRot = Vec3(orotUnit.x + angle, contactRot.y, contactRot.z) * wheel1Angle / abs(wheel1Angle)

        newnode.setHpr(positive_h, positive_p, positive_r)

        rotation_interval = LerpPosHprInterval(
            newnode,
            duration=0.5 * durIntConst,
            pos=newnode.getPos(),
            hpr=contactRot,
            blendType='easeInOut'
        )
        await rotation_interval
        if chdist < wdistance:
            unit.bodyNP.wrtReparentTo(parent)
            if terninger:
                for terning in terninger:
                    terning.remove(self.game.world)
            unit.request("Moved")
            return
        ocdistance = cdistance
        if chdist < wdistance + cdistance:
            cdistance = chdist - wdistance

        angle = contactRot.x
        vector = Vec2(-math.sin(math.radians(angle)), math.cos(math.radians(angle)))
        cmove = min(chdist, cdistance)
        pos_interval = LerpPosHprInterval(
            newnode,
            duration=0.5 * durIntConst,
            pos=wheel1Pos + Vec3(vector.x, vector.y, 0) * cmove,
            hpr=contactRot,
            blendType='easeInOut'
        )

        await pos_interval
        unit.bodyNP.wrtReparentTo(parent)
        if chdist < self.game.moveArceDistance:
            for terning in terninger:
                terning.remove(self.game.world)
            if unit.state == "IsPursuing":
                # Not a failure: a pursuit that does not reach still moves its
                # full roll, and the quarry simply gets away (p. 156).
                print(f"Pursuit of {chdist}\" did not catch the enemy "
                      f"({self.game.moveArceDistance:.1f}\" away) — "
                      f"the unit advances and halts.")
            else:
                print("Charge fell short \u2014 unit did not reach the enemy.")
            self.game.movement.dangerousTerrainTests(unit, oposUnit,
                                                     unit.bodyNP.getPos())
            unit.request("Moved")
            return

        defenderUnit = self.game.getSelectedUnit(defenderNP.node())

        if defenderUnit in self.game.player1Units:
            if unit in self.game.player1Units:
                print("Both units belong to Player 1, cannot enter combat.")
                direction = unit.bodyNP.getPos() - defenderNP.getPos()
                direction.normalize()
                self.game.fallBackContactTest(unit.bodyNP, direction * .3)
                for terning in terninger:
                    terning.remove(self.game.world)
                del terninger
                unit.request("Moved")
                return
        if defenderUnit in self.game.player2Units:
            if unit in self.game.player2Units:
                print("Both units belong to Player 2, cannot enter combat.")
                direction = unit.bodyNP.getPos() - defenderNP.getPos()
                direction.normalize()
                self.game.fallBackContactTest(unit.bodyNP, direction * .3)
                for terning in terninger:
                    terning.remove(self.game.world)
                del terninger
                unit.request("Moved")
                return

        parent = unit.bodyNP.getParent()
        newnode = render.attachNewNode(f"Temp-{unit.unitName}")
        newnode.setPos(self.game.playerNP.getPos())
        newnode.setHpr(unit.bodyNP.getHpr())
        unit.bodyNP.wrtReparentTo(newnode)

        finalHpr = (newnode.getH() + angleToRotate, newnode.getP(), newnode.getR())
        rotation_interval = LerpPosHprInterval(
            newnode,
            duration=0.5 * durIntConst,
            pos=newnode.getPos(),
            hpr=finalHpr,
            blendType='easeInOut'
        )

        await rotation_interval
        unit.bodyNP.wrtReparentTo(parent)
        newnode.removeNode()

        if defenderUnit.state == "IsFleeing":
            print("Contact detected between fleeing unit and pursuer!")
            self.game.world.removeRigidBody(defenderUnit.bodyNP.node())
            defenderUnit.model.removeNode()
            defenderUnit.bodyNP.removeNode()
            self.game.units.remove(defenderUnit)
            if defenderUnit in self.game.player1Units:
                self.game.player1Units.remove(defenderUnit)
            if defenderUnit in self.game.player2Units:
                self.game.player2Units.remove(defenderUnit)
            unit.request("Moved")
            for terning in terninger:
                terning.remove(self.game.world)
            return

        unit.request("InCombat")
        unit.isInCombat = True
        unit.chargedThisTurn = True
        # Impact Hits need to know the charge covered 3" or more (p. 172).
        unit.chargeDistance = float(self.game.moveArceDistance)
        self.game.movement.dangerousTerrainTests(unit, oposUnit,
                                                 unit.bodyNP.getPos())

        if defenderUnit.state != "InCombat":
            defenderUnit.request("InCombat")
        unit.isInCombatWith.append(defenderUnit)
        unit.isInCombatFlank.append("front")
        defenderUnit.isInCombatWith.append(unit)
        defenderUnit.isInCombat = True

        defenderUnit.isInCombatFlank.append(flank)
        unit.updateTextNode()
        defenderUnit.updateTextNode()
        if terninger:
            for terning in terninger:
                terning.remove(self.game.world)
            del terninger
        return

    async def _skirmishChargeInterval(self, unit, defenderNP, oposUnit, orotUnit, flank, chdice=None):
        """Charge move for Skirmishers: straight in, keeping facing, no wheel or
        flank-align pivot.  The charge roll is still made; if it falls short the
        unit advances only the rolled distance and does not reach combat."""
        model = unit.unit.model
        maxmove = model.get_fly_movement(0) if model.is_flying() else model.get_movement(0)

        self.game.diceInfoText.setText(self.chargeRangeText(unit, maxmove))
        if not self.game.autoRoll:
            bonus = await self.swiftstrideChargeChoice(unit)
            terninger, chdice = await self.rullTerninger(3 if bonus else 2, bonus)
        else:
            while self.game.attackSequence2.isPlaying():
                await Task.pause(0.5)
            await Task.pause(0.5)
            if chdice is None:
                chdice = [6, 6]
            terninger = []
        self.game.autoCharge = False
        self.game.autoHold = False
        print("Charge dice results:", chdice)

        chdist = maxmove + charge_roll(chdice, self.chargeThroughDifficult(unit, oposUnit))
        if unit.state == "IsPursuing":
            chdist = sum(chdice)

        target = self.game.playerNP.getPos()
        d = target - oposUnit
        dist = d.length()
        dirn = d / dist if dist > 1e-6 else Vec3(0, 1, 0)
        reached = chdist >= self.game.moveArceDistance
        travel = dist if reached else min(chdist, dist)
        endp = oposUnit + dirn * travel

        unit.bodyNP.setPos(oposUnit)
        unit.bodyNP.setHpr(orotUnit)
        await LerpPosHprInterval(unit.bodyNP, duration=0.5,
                                 pos=endp, hpr=orotUnit, blendType='easeInOut')

        for terning in terninger:
            terning.remove(self.game.world)

        if not reached:
            print("Charge fell short \u2014 skirmishers did not reach the enemy.")
            self.game.movement.dangerousTerrainTests(unit, oposUnit, endp)
            unit.request("Moved")
            return

        defenderUnit = self.game.getSelectedUnit(defenderNP.node())

        # Two units of the same player can never charge each other.
        for player_units in (self.game.player1Units, self.game.player2Units):
            if defenderUnit in player_units and unit in player_units:
                print("Both units belong to the same player, cannot enter combat.")
                push = unit.bodyNP.getPos() - defenderNP.getPos()
                push.normalize()
                self.game.fallBackContactTest(unit.bodyNP, push * .3)
                unit.request("Moved")
                return

        if defenderUnit.state == "IsFleeing":
            print("Contact detected between fleeing unit and pursuer!")
            self.game.world.removeRigidBody(defenderUnit.bodyNP.node())
            defenderUnit.model.removeNode()
            defenderUnit.bodyNP.removeNode()
            self.game.units.remove(defenderUnit)
            if defenderUnit in self.game.player1Units:
                self.game.player1Units.remove(defenderUnit)
            if defenderUnit in self.game.player2Units:
                self.game.player2Units.remove(defenderUnit)
            unit.request("Moved")
            return

        unit.request("InCombat")
        unit.isInCombat = True
        unit.chargedThisTurn = True
        unit.chargeDistance = float(travel)
        self.game.movement.dangerousTerrainTests(unit, oposUnit, endp)
        if defenderUnit.state != "InCombat":
            defenderUnit.request("InCombat")
        unit.isInCombatWith.append(defenderUnit)
        unit.isInCombatFlank.append("front")
        defenderUnit.isInCombatWith.append(unit)
        defenderUnit.isInCombat = True
        defenderUnit.isInCombatFlank.append(flank)
        unit.updateTextNode()
        defenderUnit.updateTextNode()

    # ─── Flank Detection ──────────────────────────────────────────────────

    def getFlankFromContact(self, unit, contact):
        flank = "front"
        angleAttacker = unit.bodyNP.getH()
        defenderNP = render.find(f"**/{contact.getNode1().getName()}")
        angleDefender = defenderNP.getH()
        hitloc = self.game.playerNP.getPos(defenderNP)

        shape = contact.getNode1().getShape(0)
        if isinstance(shape, BulletBoxShape):
            half_extents = shape.getHalfExtentsWithMargin()
            width = half_extents.x * unit.bodyNP.getScale().x
            height = half_extents.y * unit.bodyNP.getScale().y

        angleToRotate = angleDefender - angleAttacker
        angleToRotate = (angleToRotate) % 360
        unitloc = unit.bodyNP.getPos(defenderNP)
        hitloc = unitloc

        angle_between = math.acos(Vec3(0, 1, 0).dot(hitloc.normalized())) * (180.0 / math.pi)
        frontArcAngle = 90 - math.atan2(height, width) * (180.0 / math.pi)

        if angle_between > frontArcAngle + 90:
            flank = "rear"
            if angleToRotate > 90:
                angleToRotate = (360 - angleToRotate) * -1

        elif angle_between > frontArcAngle and hitloc.x < 0:
            flank = "flank"
            if angleToRotate > 90:
                angleToRotate -= 90
            else:
                angleToRotate = 90 - angleToRotate
                angleToRotate *= -1

        elif angle_between < frontArcAngle:
            flank = "front"
            if angleToRotate > 90:
                angleToRotate -= 180

        elif angle_between > frontArcAngle and hitloc.x > 0:
            flank = "flank"
            if angleToRotate < 0:
                angleToRotate += 90
            if angleToRotate > 90:
                angleToRotate = 360 - 90 - angleToRotate
                angleToRotate *= -1

        return flank, angleToRotate

    # ─── Battle Output ────────────────────────────────────────────────────

    def printBattleResults(self, attackerUnit, defenderUnit, attacks, total_hits,
                           suffered_wounds, saves_made, total_wounds):
        from battleFunctions import take_last_combat_report, format_combat_report
        for line in format_combat_report(take_last_combat_report()):
            print(line)
        weapon = attackerUnit.unit.model.equipedWeapon or {}
        verb = 'shots' if weapon.get('tag') == 'ranged' else 'attacks'
        print(f"   {attacks} {verb} -> {total_hits} hit -> {suffered_wounds} wound "
              f"-> {saves_made} saved -> {total_wounds} slain "
              f"({defenderUnit.unit.name})")

    @staticmethod
    def printCombatResult(rows, totals, unit_strengths):
        """Itemise the combat result so a draw or a rout can be read back.

        *rows* is label -> (player 1 points, player 2 points). A line the
        rulebook never awarded is still worth showing as a zero: the question
        after a lost combat is usually which bonus the other side had.
        """
        p1_total, p2_total = totals
        p1_us, p2_us = unit_strengths
        width = max(len(label) for label in rows)
        print(f"\n   {'Combat result':<{width}}  Player 1  Player 2")
        print(f"   {'-' * (width + 20)}")
        for label, (p1, p2) in rows.items():
            print(f"   {label:<{width}}  {p1:>8}  {p2:>8}")
        print(f"   {'-' * (width + 20)}")
        print(f"   {'TOTAL':<{width}}  {p1_total:>8}  {p2_total:>8}")
        print(f"   {'(Unit Strength)':<{width}}  {p1_us:>8}  {p2_us:>8}")
        if p1_total == p2_total:
            print("   -> drawn combat")
        else:
            winner = 1 if p1_total > p2_total else 2
            print(f"   -> Player {winner} wins by {abs(p1_total - p2_total)}")

    # ─── Impact Hits ──────────────────────────────────────────────────────

    def impactHits(self, modRemoveSequence):
        """Resolve Impact Hits before any blows are struck (Rulebook p. 172).

        Returns each player's combat result contribution.
        """
        player1_score = player2_score = 0
        resolved = set()
        for striker, target in zip(self.game.attackers, self.game.defenders):
            if id(striker) in resolved or striker.hasAttackedThisTurn:
                continue
            resolved.add(id(striker))
            if not getattr(striker, 'chargedThisTurn', False):
                continue
            if getattr(striker, 'chargeDistance', 0.0) < MIN_IMPACT_HIT_CHARGE:
                continue
            hits, wounds, saves, unsaved = resolve_impact_hits(striker.unit,
                                                               target.unit)
            if not hits:
                continue
            for line in impact_hit_report(striker.unit, target.unit):
                print(line)
            print(f"   {hits} impact hits -> {wounds} wound -> {saves} saved "
                  f"-> {unsaved} unsaved ({target.unit.name})")
            if not unsaved:
                continue
            # Thin the target now so it strikes back with its losses, the way
            # the attack loop does; applyWounds later confirms the count.
            W = max(1, _stat_int(target.unit.model.characteristics, 'W', 1))
            slain = (getattr(target, 'woundsOnModel', 0) + unsaved) // W
            target.unit.nmodels = max(0, target.unit.nmodels - slain)
            if striker in self.game.player1Units:
                player1_score += unsaved
            else:
                player2_score += unsaved
            modRemoveSequence.append(Func(self.game.applyWounds, target, unsaved))
        return player1_score, player2_score

    # ─── Battle Start & Resolution ────────────────────────────────────────

    async def verySimpleBattleStart(self, task):
        self.game.resolvingCombat = True
        weps = self.game.unitToMove.unit.model.weapons

        wepchoice = await taskMgr.add(self.game.makeChoiceNew(weps, Vec3(0, 0, 10)))

        self.game.unitToMove.unit.model.equip_weapon(wepchoice)

        await taskMgr.add(self.verySimpleBattle, "verySimpleBattleTask")
        return task.done

    async def verySimpleBattle(self, task):
        # Hold nearby-friend Panic tests until the whole combat (incl. flee /
        # pursuit) is resolved, so their moves/reforms don't clash with the
        # charge-reaction / pursuit choices.
        psy = getattr(self.game, 'psychology', None)
        if psy:
            psy.hold_panic()
        try:
            await self._verySimpleBattleInner(task)
        except Exception as e:
            print(f"ERROR in verySimpleBattle: {e}")
            import traceback
            traceback.print_exc()
            self.game.resolvingCombat = False
            messenger.send('unit-move-complete')
        finally:
            if psy:
                psy.release_panic()
        return task.done

    async def _verySimpleBattleInner(self, task):
        attacker = self.game.unitToMove.bodyNP
        defender = self.game.unitToMove.isInCombatWith[0].bodyNP
        flank = self.game.unitToMove.isInCombatFlank[0]
        engagedWith = [x.unitName for x in self.game.unitToMove.isInCombatWith]

        selected_choice = await taskMgr.add(self.game.makeChoiceNew(engagedWith, Vec3(0, 0, 10)))

        for unit in self.game.unitToMove.isInCombatWith:
            if unit.unitName == selected_choice:
                defender = unit.bodyNP
                break
        attackerUnit = self.game.getSelectedUnit(attacker.node())
        defenderUnit = self.game.getSelectedUnit(defender.node())
        defender_nmodels = defenderUnit.unit.nmodels
        print(f"{attackerUnit.unit.name} attacks {defenderUnit.unit.name} in {flank}!")
        # Fight with each model's best melee weapon (not a bare hand weapon) so
        # its stats and hooks come from the actually-equipped weapon. The
        # attacker keeps a melee weapon it was given; only swap off a ranged one.
        if attackerUnit.unit.model.equipedWeapon.get('tag') == 'ranged':
            attackerUnit.unit.model.equip_best_melee()
        defenderUnit.unit.model.equip_best_melee()

        self.game.attackSequence = Sequence()
        self.game.attackers = []
        self.game.attackers.append(attackerUnit)
        self.game.defenders = []
        self.game.defenders.append(defenderUnit)
        for unit in self.game.unitToMove.isInCombatWith:
            self.game.attackers.append(self.game.getSelectedUnit(unit.bodyNP.node()))
            self.game.defenders.append(self.game.unitToMove)
        for unit in defenderUnit.isInCombatWith:
            self.game.attackers.append(self.game.getSelectedUnit(unit.bodyNP.node()))
            self.game.defenders.append(defenderUnit)
        # Snapshot each unit's model count at the start of combat so that
        # casualties inflicted earlier this round (e.g. by a charger striking
        # first) thin the fighting ranks of a unit that strikes back.
        self._combatStartModels = {id(g.unit): g.unit.nmodels
                                   for g in set(self.game.attackers) | set(self.game.defenders)}
        player1_score = 0
        player1_flank_bonus = 0
        player1_rank_bonus = 0
        player2_score = 0
        player2_flank_bonus = 0
        player2_rank_bonus = 0
        modRemoveSequence = Sequence()
        impact1, impact2 = self.impactHits(modRemoveSequence)
        player1_score += impact1
        player2_score += impact2
        for i in range(len(self.game.attackers)):
            unit = self.game.attackers[i]
            if unit.hasAttackedThisTurn:
                print(f"Unit {unit.unit.name} has already attacked this turn, skipping.")
                continue
            attackerUnit = self.game.defenders[i]
            attacker = attackerUnit.bodyNP
            defender = unit.bodyNP
            defenderUnit = self.game.getSelectedUnit(defender.node())
            defenderUnit.hasAttackedThisTurn = True
            defenderUnit.updateTextNode()
            if defenderUnit.unit.model.equipedWeapon.get('tag') == 'ranged':
                defenderUnit.unit.model.equip_weapon('hand weapon')
            if attackerUnit.unit.model.equipedWeapon.get('tag') == 'ranged':
                attackerUnit.unit.model.equip_weapon('hand weapon')
            apos = defender.getPos()
            back_int = LerpPosHprInterval(
                defender,
                duration=0.5,
                pos=defender.getPos() - (attacker.getPos() - defender.getPos()).normalized() * 2,
                hpr=defender.getHpr(),
                blendType='easeInOut'
            )
            forward_int = LerpPosHprInterval(
                defender,
                duration=0.5,
                pos=apos,
                hpr=defender.getHpr(),
                blendType='easeInOut'
            )
            self.game.attackSequence.append(back_int)
            self.game.attackSequence.append(forward_int)

            # A joined character occupies one front-rank slot: the unit fights
            # with one fewer model of its own profile, the character adds its own
            # attacks below.
            joinedRule = next((r for r in defenderUnit.unit.model.special_rules
                               if isinstance(r, dict) and r.get('tag') == JOIN_TAG), None)
            origFiles = defenderUnit.unit.files
            if joinedRule and origFiles > 1:
                defenderUnit.unit.files -= 1
            # The charging unit fights with its charge bonus (and front rank
            # only); everyone else fights as normal (front + supporting rank).
            # Casualties suffered this round (charger struck first) come off
            # the fighting rank of a unit that strikes back: the slain and the
            # models that stepped into their place cannot attack.
            casualties = max(0, self._combatStartModels.get(
                id(defenderUnit.unit), defenderUnit.unit.nmodels) - defenderUnit.unit.nmodels)
            attacks, total_hits, suffered_wounds, saves_made, total_wounds = simulate_battle(
                defenderUnit.unit, attackerUnit.unit,
                charge=getattr(defenderUnit, 'chargedThisTurn', False),
                casualties=casualties)
            defenderUnit.unit.files = origFiles
            self.printBattleResults(defenderUnit, attackerUnit, attacks, total_hits,
                                    suffered_wounds, saves_made, total_wounds)
            attackerUnit.unit.nmodels -= total_wounds

            if defenderUnit in self.game.player1Units:
                player1_score += total_wounds
                for faceing in defenderUnit.isInCombatFlank:
                    if faceing == 'flank':
                        player2_flank_bonus += 1
                    elif faceing == 'rear':
                        player2_flank_bonus += 2
                    else:
                        player2_flank_bonus += 0
                player1_rank_bonus = min(player1_rank_bonus + rank_bonus(
                    defenderUnit.unit, getattr(defenderUnit, 'isDisrupted', False)),
                    MAX_RANK_BONUS)
            else:
                player2_score += total_wounds
                for faceing in defenderUnit.isInCombatFlank:
                    if faceing == 'flank':
                        player1_flank_bonus += 1
                    elif faceing == 'rear':
                        player1_flank_bonus += 2
                    else:
                        player1_flank_bonus += 0
                player2_rank_bonus = min(player2_rank_bonus + rank_bonus(
                    defenderUnit.unit, getattr(defenderUnit, 'isDisrupted', False)),
                    MAX_RANK_BONUS)

            combWounds = 0
            combWounds += total_wounds
            for rule in defenderUnit.unit.model.special_rules:
                if rule.get('mountUnit'):
                    attacks, total_hits, suffered_wounds, saves_made, total_wounds = simulate_battle(
                        rule['mountUnit'], attackerUnit.unit,
                        charge=getattr(defenderUnit, 'chargedThisTurn', False))
                    self.printBattleResults(defenderUnit, attackerUnit, attacks, total_hits,
                                            suffered_wounds, saves_made, total_wounds)
                    attackerUnit.unit.nmodels -= total_wounds

                    if defenderUnit in self.game.player1Units:
                        player1_score += total_wounds
                    else:
                        player2_score += total_wounds
                    combWounds += total_wounds
            for partUnit in self.chariotParts(defenderUnit):
                attacks, total_hits, suffered_wounds, saves_made, total_wounds = simulate_battle(
                    partUnit, attackerUnit.unit,
                    charge=getattr(defenderUnit, 'chargedThisTurn', False))
                self.printBattleResults(defenderUnit, attackerUnit, attacks, total_hits,
                                        suffered_wounds, saves_made, total_wounds)
                attackerUnit.unit.nmodels -= total_wounds
                if defenderUnit in self.game.player1Units:
                    player1_score += total_wounds
                else:
                    player2_score += total_wounds
                combWounds += total_wounds
            # A joined character fights with its own profile (single model).
            if joinedRule:
                charUnit = joinedRule['characterUnit']
                cw = charUnit.model.equipedWeapon
                if cw is None or cw.get('tag') == 'ranged':
                    charUnit.model.equip_weapon('hand weapon')
                attacks, total_hits, suffered_wounds, saves_made, total_wounds = simulate_battle(
                    charUnit, attackerUnit.unit,
                    charge=getattr(defenderUnit, 'chargedThisTurn', False))
                self.printBattleResults(defenderUnit, attackerUnit, attacks, total_hits,
                                        suffered_wounds, saves_made, total_wounds)
                attackerUnit.unit.nmodels -= total_wounds
                if defenderUnit in self.game.player1Units:
                    player1_score += total_wounds
                else:
                    player2_score += total_wounds
                combWounds += total_wounds
            modRemoveSequence.append(
                Func(self.game.applyWounds, attackerUnit, combWounds))

        engaged = set(self.game.attackers) | set(self.game.defenders)
        p1_units = [u for u in engaged if u in self.game.player1Units]
        p2_units = [u for u in engaged if u in self.game.player2Units]
        # Wounds are everything banked so far bar the Impact Hits.
        p1_wounds = player1_score - impact1
        p2_wounds = player2_score - impact2
        player1_score += player1_flank_bonus + player1_rank_bonus
        player2_score += player2_flank_bonus + player2_rank_bonus
        player1_standard = battle_standard_bonus(p1_units)
        player2_standard = battle_standard_bonus(p2_units)
        player1_score += player1_standard
        player2_score += player2_standard
        p1_us = side_unit_strength(p1_units)
        p2_us = side_unit_strength(p2_units)
        player1_massed = massed_infantry_bonus(p1_units, p1_us, p2_us)
        player2_massed = massed_infantry_bonus(p2_units, p2_us, p1_us)
        player1_score += player1_massed
        player2_score += player2_massed
        for who, bonus, us, other in (('Player 1', player1_massed, p1_us, p2_us),
                                      ('Player 2', player2_massed, p2_us, p1_us)):
            if bonus:
                rule_log('Massed Infantry', who,
                         f"Unit Strength {us} against {other} -> +1 combat result")
        self.printCombatResult(
            {'Wounds caused': (p1_wounds, p2_wounds),
             'Impact Hits': (impact1, impact2),
             'Flank / rear': (player1_flank_bonus, player2_flank_bonus),
             'Rank Bonus': (player1_rank_bonus, player2_rank_bonus),
             'Battle Standard': (player1_standard, player2_standard),
             'Massed Infantry': (player1_massed, player2_massed)},
            (player1_score, player2_score), (p1_us, p2_us))
        await self.game.attackSequence
        await modRemoveSequence

        self.game.attackSequence2 = Sequence()
        loserUnits = []
        if player2_score == player1_score:
            print("Combat is a draw, no units flee.")
            self.game.resolvingCombat = False
            messenger.send('unit-move-complete')
            return
        elif player2_score < player1_score:
            for atu in self.game.attackers:
                if atu in self.game.player2Units and atu.bodyNP.isEmpty() == False and atu not in loserUnits:
                    loserUnits.append(atu)
            diff = player1_score - player2_score
        else:
            for atu in self.game.attackers:
                if atu in self.game.player1Units and atu.bodyNP.isEmpty() == False and atu not in loserUnits:
                    loserUnits.append(atu)
            diff = player2_score - player1_score

        # Follow Up & Pursuit (p. 156) is four passes over the whole combat, not
        # one loser at a time: every Break test is made, then every winner
        # declares and names its quarry, then the losers move, and only then do
        # the pursuits run. Interleaving them gave the wrong answer as soon as a
        # combat had two losing units.
        outcomes = await self.breakTestPass(loserUnits, diff)
        responses = await self.declarePass(outcomes)
        await self.loserMovePass(outcomes, responses)
        await self.pursuitPass(outcomes, responses)

        self.game.resolvingCombat = False
        messenger.send('unit-move-complete')
        return task.done

    # ─── Post-Combat: pass 1, the Break tests ─────────────────────────────

    async def breakTestPass(self, loserUnits, diff):
        """Every losing unit takes its Break test (p. 154).

        Returns ``[(unit, outcome)]`` and moves nothing: the winners have to
        declare before any of these units budge.
        """
        outcomes = []
        for loserUnit in loserUnits:
            if loserUnit.bodyNP.isEmpty():
                continue

            if any(rule.get('Unbreakable', False) for rule in loserUnit.unit.model.special_rules):
                print(f"{loserUnit.unit.name} is Unbreakable and does not flee!, only gives ground.")
                outcomes.append((loserUnit, 'give_ground'))
                continue

            ld = _stat_int(loserUnit.unit.model.characteristics, 'Ld', 7)
            psy = getattr(self.game, 'psychology', None)
            if psy is not None:
                ld, general = psy.leadership_of(loserUnit)
                if general is not None:
                    print(f"{loserUnit.unit.name} takes its Break test on the "
                          f"General's Leadership ({general.unit.name}, Ld {ld}) "
                          f"— Inspiring Presence.")
            overwhelm = self.isOverwhelmed(loserUnit, loserUnits)

            if stubborn_available(loserUnit):
                if self.game.roundCounter.current_player in [1, 2] and self.game.AIplayer2.active:
                    useStubborn = should_use_stubborn(ld, diff, overwhelm)
                else:
                    stubbornChoice = [loserUnit.unitName + '\nStand Firm',
                                      loserUnit.unitName + '\nBreak test']
                    selected = await taskMgr.add(
                        self.game.makeChoiceNew(stubbornChoice, Vec3(0, 0, 10)))
                    useStubborn = selected == stubbornChoice[0]
                if useStubborn:
                    loserUnit.usedStubborn = True
                    print(f"{loserUnit.unit.name} is Stubborn and refuses its Break "
                          f"test — Falls Back in Good Order.")
                    self.notifyFleesCombat(loserUnit)
                    outcomes.append((loserUnit, 'fall_back'))
                    continue

            ldDice = await self.rollBreakDice()
            print("Leadership dice results for fleeing unit:", ldDice,
                  "sum:", sum(ldDice), "Ld:", ld, "combat result diff:", diff,
                  "overwhelmed:", overwhelm)
            outcome = break_test_outcome(ldDice, ld, diff, overwhelm)

            bsb = psy.battle_standard_of(loserUnit) if psy is not None else None
            if bsb is not None:
                if self.game.roundCounter.current_player in [1, 2] and self.game.AIplayer2.active:
                    reroll = should_reroll_break(outcome, ld, diff, overwhelm)
                else:
                    rerollChoice = [loserUnit.unitName + f'\nRe-roll ({outcome})',
                                    loserUnit.unitName + '\nKeep']
                    selected = await taskMgr.add(
                        self.game.makeChoiceNew(rerollChoice, Vec3(0, 0, 10)))
                    reroll = selected == rerollChoice[0]
                if reroll:
                    ldDice = await self.rollBreakDice()
                    # The second roll stands, even if it is worse than the first.
                    outcome = break_test_outcome(ldDice, ld, diff, overwhelm)
                    print(f"{loserUnit.unit.name} re-rolls its Break test "
                          f"(Hold Your Ground: {bsb.unit.name}): {ldDice} "
                          f"sum {sum(ldDice)} -> {outcome}")

            if outcome in ('break', 'fall_back'):
                print("losing unit flees from combat!" if outcome == 'break'
                      else "losing unit FBIG!")
                self.notifyFleesCombat(loserUnit)
            else:
                print("losing unit gives ground!")
            outcomes.append((loserUnit, outcome))
        return outcomes

    # ─── Post-Combat: pass 2, the winners declare ─────────────────────────

    def stillEngaged(self, unit, exclude=None):
        """Still Engaged (p. 156): base contact with any enemy other than the
        one it is going after stops a follow up or a pursuit."""
        for other in list(unit.isInCombatWith):
            if other is exclude or other.bodyNP.isEmpty():
                continue
            if self.game.world.contactTestPair(
                    unit.bodyNP.node(), other.bodyNP.node()).getNumContacts():
                return True
        return False

    async def declarePass(self, outcomes):
        """Each winner declares restrain, follow up or pursue — and which unit
        it is after — before any loser moves or rolls (p. 156).

        Returns ``[{'winner', 'target', 'action'}]``.
        """
        responses = []
        declared = []
        for loserUnit, outcome in outcomes:
            for winner in list(loserUnit.isInCombatWith):
                if winner.bodyNP.isEmpty() or any(winner is w for w in declared):
                    continue
                declared.append(winner)

                targets = [(u, o) for u, o in outcomes
                           if any(winner is w for w in u.isInCombatWith)]
                if not targets:
                    continue
                if len(targets) > 1:
                    names = [u.unitName + '\nChase' for u, _ in targets]
                    picked = await taskMgr.add(
                        self.game.makeChoiceNew(names, Vec3(0, 0, 10)))
                    target, targetOutcome = next(
                        (u, o) for (u, o), n in zip(targets, names) if n == picked)
                else:
                    target, targetOutcome = targets[0]

                move = winner_response(targetOutcome)
                action = await self.restrainChoice(winner, target, move)
                responses.append({'winner': winner, 'target': target,
                                  'action': action})
        return responses

    async def restrainChoice(self, winner, target, move):
        """Restrain & Reform (p. 156) is a Leadership test, not a free choice:
        a unit that elects to hold back and fails must go anyway."""
        verb = 'Follow up' if move == 'follow_up' else 'Pursue'
        if self.game.roundCounter.current_player in [1, 2] and self.game.AIplayer2.active:
            chosen = move
        else:
            options = [winner.unitName + '\n' + verb,
                       winner.unitName + '\nRestrain']
            selected = await taskMgr.add(
                self.game.makeChoiceNew(options, Vec3(0, 0, 10)))
            chosen = move if selected == options[0] else 'restrain'

        if chosen != 'restrain':
            print(f"{winner.unit.name} chooses to {verb.lower()} "
                  f"{target.unit.name}!")
            return move

        ld = _stat_int(winner.unit.model.characteristics, 'Ld', 7)
        psy = getattr(self.game, 'psychology', None)
        if psy is not None:
            ld, _ = psy.leadership_of(winner)
        dice = await self.rollBreakDice()
        if restraint_test(ld, dice):
            rule_log('Restrain & Reform', winner,
                     f"Restraint test {dice} = {sum(dice)} vs Ld {ld} -> holds "
                     f"its ground")
            winner.request("Idle")
            return 'restrain'
        rule_log('Restrain & Reform', winner,
                 f"Restraint test {dice} = {sum(dice)} vs Ld {ld} -> fails, and "
                 f"must {'follow up' if move == 'follow_up' else 'pursue'} "
                 f"{target.unit.name}")
        return move

    # ─── Post-Combat: pass 3, the losers move ─────────────────────────────

    async def loserMovePass(self, outcomes, responses):
        """Give Ground, Fall Back in Good Order and Flee, in that order.

        A Give Ground and the Follow Up answering it are the same 2" in the same
        direction, so they move together: keeping them in one interval means the
        two stay in base contact by construction, with no separation test to
        push the follower back out again.
        """
        for loserUnit, outcome in outcomes:
            if loserUnit.bodyNP.isEmpty():
                continue
            if outcome == 'give_ground':
                followers = [r['winner'] for r in responses
                             if r['target'] is loserUnit and r['action'] == 'follow_up']
                await self.giveGroundMove(loserUnit, followers)
            else:
                await self.fleeMove(loserUnit, outcome)

    async def giveGroundMove(self, loserUnit, followers):
        """The loser backs off 2" and anyone following up comes with it."""
        winners = [u for u in loserUnit.isInCombatWith if not u.bodyNP.isEmpty()]
        direction = self.giveGroundDirection(loserUnit, winners)
        moving = [loserUnit] + [f for f in followers if not f.bodyNP.isEmpty()]

        crashFraction = 1.0
        for unit in moving:
            crashFraction = min(crashFraction,
                                self.game.sweepTest(unit, direction, GIVE_GROUND) * .95)
        step = direction * (GIVE_GROUND * crashFraction)
        if step.length() < 0.05:
            # Surrounded (p. 155): a loser that cannot break contact stays put
            # and the combat is fought again as though it had been a draw.
            rule_log('Surrounded', loserUnit,
                     "cannot break contact, so it stays locked in place and "
                     "fights again next turn")
            return

        self.game.attackSequence2 = Parallel()
        for unit in moving:
            print(f"{unit.unit.name} moves {step.length():.1f}\" "
                  f"({'gives ground' if unit is loserUnit else 'follows up'})")
            self.game.attackSequence2.append(
                LerpPosInterval(unit.bodyNP, duration=1.0,
                                pos=unit.bodyNP.getPos() + step,
                                blendType='easeInOut'))
        if not self.game.attackSequence.isPlaying():
            await self.game.attackSequence2

    async def fleeMove(self, loserUnit, outcome):
        """Break or Fall Back: away from the single strongest winner, at a
        distance the outcome decides."""
        winner = self.fleesFrom(loserUnit)
        if winner is None:
            return
        pos = loserUnit.bodyNP.getPos()
        wpos = winner.bodyNP.getPos()
        dx, dy = flee_direction((pos.x, pos.y), (wpos.x, wpos.y))
        direction = Vec3(dx, dy, 0)

        kind = 'flee' if outcome == 'break' else 'fall back'
        bonus = await self.swiftstrideChoice(
            loserUnit, kind, distance_to_edge=board_edge_distance(pos.x, pos.y))
        dice = await self.rollMoveDice(loserUnit, 3 if bonus else 2, bonus)
        distance = flee_roll(dice) if outcome == 'break' else fall_back_roll(dice)
        print(f"{loserUnit.unit.name} {kind}s {distance}\" away from "
              f"{winner.unit.name} (highest Unit Strength): {dice}")

        await self.game.fallBack2(loserUnit.bodyNP, direction, length=distance * 1.0,
                                  rally=(outcome == 'fall_back'),
                                  flee=(outcome == 'break'))
        # The state is set after the move, as it was before the four passes:
        # a unit that Falls Back rallies at the end of it and is not fleeing.
        loserUnit.request("IsFleeing" if outcome == 'break' else "Moved")

    # ─── Post-Combat: pass 4, the pursuits ────────────────────────────────

    async def pursuitPass(self, outcomes, responses):
        """Pursuit moves are made one at a time, once every loser has moved
        (p. 156)."""
        outcome_of = {id(u): o for u, o in outcomes}
        for r in responses:
            if r['action'] != 'pursue':
                continue
            winner, target = r['winner'], r['target']
            if winner.bodyNP.isEmpty():
                continue
            if self.stillEngaged(winner, exclude=target):
                rule_skipped('Pursuit', winner,
                             "still in base contact with another enemy")
                continue
            if target.bodyNP.isEmpty():
                continue
            await self.pursuitMove(winner, target, outcome_of.get(id(target)))

    async def pursuitMove(self, winner, target, outcome):
        """Pivot to face the quarry and run the pursuit through the charge
        machinery, which rolls the 2D6, sums it, and handles the wheel, the
        align and the contact — the same things a charge needs."""
        targetPos = target.bodyNP.getPos()
        rFrom = winner.bodyNP.getHpr()
        winner.bodyNP.lookAt(targetPos)
        rTo = winner.bodyNP.getHpr()
        winner.bodyNP.setHpr(rFrom)
        await LerpPosHprInterval(winner.bodyNP, duration=0.5,
                                 pos=winner.bodyNP.getPos(), hpr=rTo,
                                 blendType='easeInOut')

        winner.request("IsPursuing")
        winner.hasMovedThisTurn = False
        self.game.autoCharge = True
        self.game.autoHold = True
        self.game.pathTowardsMouse(winner, targetPos.x, targetPos.y)
        self.game.moveUnit(winner)
        await Wait(5.0)

    # ─── Post-Combat: Give Ground ─────────────────────────────────────────

    async def rollBreakDice(self):
        """Roll the physical 2D6 of a Break test and return their values."""
        terningerLd = []
        for i in range(2):
            terning = Dice(self.game.world, position=Vec3(20 + i * 2, 0, 10),
                           size=1.0, color=(1, 0, 0, 1))
            terningerLd.append(terning)
        for terning in terningerLd:
            terning.roll()
        await taskMgr.add(checkDice, "checkDiceTaskFlee",
                          extraArgs=[terningerLd], appendTask=True)
        values = [terning.currentValue for terning in terningerLd]
        for terning in terningerLd:
            terning.remove(self.game.world)
        return values

    async def rollMoveDice(self, unit, count, swiftstride=False):
        """Roll the physical dice of a Flee, Fall Back or Pursuit roll.

        The Swiftstride die is thrown last and in its own colour, because it is
        added rather than being one of the two the roll may discard between.
        """
        dice = []
        for i in range(count):
            swift = swiftstride and i == count - 1
            dice.append(Dice(self.game.world,
                             position=unit.bodyNP.getPos() + Vec3(-20 + i * 4, 0, 10),
                             size=1.0,
                             body_color=SWIFTSTRIDE_DIE_COLOR if swift else None))
        for terning in dice:
            terning.roll()
        await taskMgr.add(checkDice, "checkDiceTaskPersuit" + str(unit.unitName),
                          extraArgs=[dice], appendTask=True)
        values = [terning.currentValue for terning in dice]
        for terning in dice:
            terning.remove(self.game.world)
        return values

    def fleesFrom(self, loserUnit):
        """The winner a Break or Fall Back runs directly away from: the highest
        Unit Strength, settled at random between equals (p. 133)."""
        winners = [(u, unit_strength_total(u))
                   for u in loserUnit.isInCombatWith if not u.bodyNP.isEmpty()]
        return flees_from(winners)

    def giveGroundDirection(self, loserUnit, winners):
        """Give Ground backs away from every unit engaging it at once, so this
        is not the same direction a flee would take (p. 155)."""
        pos = loserUnit.bodyNP.getPos()
        others = [(w.bodyNP.getPos().x, w.bodyNP.getPos().y) for w in winners]
        dx, dy = give_ground_direction((pos.x, pos.y), others)
        return Vec3(dx, dy, 0)

    def chariotParts(self, hostUnit):
        """A chariot's crew and beasts as fighting units of their own.

        Each uses its own Weapon Skill, Strength and Attacks (Rulebook p. 194);
        the chariot itself has no Attacks. The catalogue says how many of each
        a chariot carries -- a War Wagon has 6 crew and 2 horses -- and they all
        fight, so the part's frontage is its own count times the number of
        chariots in the unit.
        """
        model = hostUnit.unit.model
        parts = []
        for tag, part in (('crew', model.get_crew()), ('beasts', model.get_beasts())):
            if part is None:
                continue
            count = model.part_count(tag) * hostUnit.unit.nmodels
            parts.append(SimpleNamespace(
                name=part.name, model=part,
                nmodels=count, files=count, ranks=1))
        return parts

    def notifyFleesCombat(self, loserUnit):
        """A US>=5 unit breaking or falling back panics nearby friends. The Unit
        Strength is the one it had at the start of this combat, and friends are
        measured before it moves."""
        if not getattr(self.game, 'psychology', None):
            return
        start_models = self._combatStartModels.get(
            id(loserUnit.unit), loserUnit.unit.nmodels)
        us0 = loserUnit.unit.model.unit_strength() * start_models
        self.game.psychology.on_unit_flees_combat(loserUnit, unit_strength=us0)

    def isOverwhelmed(self, loserUnit, loserUnits):
        """True if the winning side's total Unit Strength is more than twice the
        losing side's, which turns a Fall Back in Good Order into a Break.
        Unit Strength is totalled per side at the end of the Combat phase."""
        winners = [u for u in loserUnit.isInCombatWith if not u.bodyNP.isEmpty()]
        losers = [loserUnit] + [u for u in loserUnits
                                if u is not loserUnit and not u.bodyNP.isEmpty()
                                and any(w in u.isInCombatWith for w in winners)]
        return overwhelmed(sum(unit_strength_total(u) for u in winners),
                           sum(unit_strength_total(u) for u in losers))
