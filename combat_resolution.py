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
from direct.interval.LerpInterval import LerpPosHprInterval
from direct.interval.IntervalGlobal import Sequence, Parallel, Wait
from direct.interval.FunctionInterval import Func
from direct.task.Task import Task

from dice import Dice, checkDice
from battleFunctions import simulate_battle
from characters import JOIN_TAG
from psychology import (battle_standard_bonus, break_test_outcome, overwhelmed,
                       should_reroll_break, should_use_stubborn,
                       stubborn_available, unit_strength_total)


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
        self.terningerCharge = []
        for i in range(2):
            terning = Dice(self.game.world, position=Vec3(-20 + i * 2, 0, 10), size=1.0)
            self.terningerCharge.append(terning)
        for terning in self.terningerCharge:
            terning.roll()
        chtask = taskMgr.add(checkDice, "checkDiceTaskCharge",
                             extraArgs=[self.terningerCharge], appendTask=True)

        self.terningerFlee = []
        for i in range(2):
            terning = Dice(self.game.world, position=Vec3(20 + i * 2, 0, 10), size=1.0, color=(1, 0, 0, 1))
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

        chdist = _stat_int(unit.unit.model.characteristics, 'M') + max(chdice)
        for rule in unit.unit.model.special_rules:
            if rule.get('mountUnit'):
                chdist = _stat_int(rule['mountUnit'].model.characteristics, 'M') + max(chdice)
        fldist = sum(fldice)
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

    async def rullTerninger(self, antall):
        terninger = []
        for i in range(2):
            terning = Dice(self.game.world, position=Vec3(0 + i * 4, 0, 10), size=1.0)
            terninger.append(terning)
        for terning in terninger:
            terning.roll()
        await taskMgr.add(checkDice, "checkDiceTask", extraArgs=[terninger], appendTask=True)

        chdice = []
        for terning in terninger:
            chdice.append(terning.currentValue)
        return terninger, chdice

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
        self.game.diceInfoText.setText(f"Roll needed: {(math.ceil(self.game.moveArceDistance) - int(maxmove)):.0f}")
        if not self.game.autoRoll:
            self.game.diceInfoText.setText(
                f"Roll needed: {(math.ceil(self.game.moveArceDistance) - int(maxmove)):.0f}")

            terninger, chdice = await self.rullTerninger(2)

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

        chdist = _stat_int(unit.unit.model.characteristics, 'M') + max(chdice)
        for rule in unit.unit.model.special_rules:
            if rule.get('mountUnit'):
                chdist = _stat_int(rule['mountUnit'].model.characteristics, 'M') + max(chdice)
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
            print("Charge fell short \u2014 unit did not reach the enemy.")
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

        self.game.diceInfoText.setText(
            f"Roll needed: {(math.ceil(self.game.moveArceDistance) - int(maxmove)):.0f}")
        if not self.game.autoRoll:
            terninger, chdice = await self.rullTerninger(2)
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

        chdist = maxmove + max(chdice)
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
            # Casualties suffered this round (charger struck first) thin the
            # supporting rank of a unit that strikes back.
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
                # Skirmishers claim no Rank Bonus when engaged.
                if not defenderUnit.unit.model.is_skirmisher():
                    player1_rank_bonus += defenderUnit.unit.ranks - 1
                    if defenderUnit.unit.nmodels % defenderUnit.unit.files > 0 and \
                       defenderUnit.unit.nmodels % defenderUnit.unit.files < 4:
                        player1_rank_bonus -= 1
                    player1_rank_bonus = max(player1_rank_bonus, 0)
                    player1_rank_bonus = min(player1_rank_bonus, 2)
            else:
                player2_score += total_wounds
                for faceing in defenderUnit.isInCombatFlank:
                    if faceing == 'flank':
                        player1_flank_bonus += 1
                    elif faceing == 'rear':
                        player1_flank_bonus += 2
                    else:
                        player1_flank_bonus += 0
                # Skirmishers claim no Rank Bonus when engaged.
                if not defenderUnit.unit.model.is_skirmisher():
                    player2_rank_bonus += defenderUnit.unit.ranks - 1
                    if defenderUnit.unit.nmodels % defenderUnit.unit.files > 0 and \
                       defenderUnit.unit.nmodels % defenderUnit.unit.files < 4:
                        player2_rank_bonus -= 1
                    player2_rank_bonus = max(player2_rank_bonus, 0)
                    player2_rank_bonus = min(player2_rank_bonus, 2)

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
                Func(self.game.removeModelsFromUnit, attackerUnit, combWounds))

        player1_score += player1_flank_bonus + player1_rank_bonus
        player2_score += player2_flank_bonus + player2_rank_bonus
        engaged = set(self.game.attackers) | set(self.game.defenders)
        player1_standard = battle_standard_bonus(
            [u for u in engaged if u in self.game.player1Units])
        player2_standard = battle_standard_bonus(
            [u for u in engaged if u in self.game.player2Units])
        player1_score += player1_standard
        player2_score += player2_standard
        if player1_standard or player2_standard:
            print(f"Battle Standard combat result bonus: "
                  f"P1 +{player1_standard}, P2 +{player2_standard}")
        print(f"Player 2 score: {player2_score}, Player 1 score: {player1_score}")
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

        for loserUnit in loserUnits:
            if loserUnit.bodyNP.isEmpty():
                continue

            if any(rule.get('Unbreakable', False) for rule in loserUnit.unit.model.special_rules):
                print(f"{loserUnit.unit.name} is Unbreakable and does not flee!, only gives ground.")
                await taskMgr.add(self.GiveGroundFromCombat, "fleeFromCombatTask",
                                   extraArgs=[loserUnit], appendTask=False)
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
                    await taskMgr.add(self.FBIGFromCombat, "fleeFromCombatTask",
                                       extraArgs=[loserUnit], appendTask=False)
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

            if outcome == 'break':
                print("losing unit flees from combat!")
                self.notifyFleesCombat(loserUnit)
                await taskMgr.add(self.fleeFromCombat, "fleeFromCombatTask",
                                   extraArgs=[loserUnit], appendTask=False)
            elif outcome == 'fall_back':
                print("losing unit FBIG!")
                self.notifyFleesCombat(loserUnit)
                await taskMgr.add(self.FBIGFromCombat, "fleeFromCombatTask",
                                   extraArgs=[loserUnit], appendTask=False)
            else:
                print("losing unit gives ground!")
                await taskMgr.add(self.GiveGroundFromCombat, "fleeFromCombatTask",
                                   extraArgs=[loserUnit], appendTask=False)

        for loserUnit in loserUnits:
            if loserUnit.bodyNP.isEmpty():
                continue
            loserUnit.madePursuitChoice = False
            for unit in loserUnit.isInCombatWith:
                unit.madePursuitChoice = False

        self.game.resolvingCombat = False
        messenger.send('unit-move-complete')
        return task.done

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

    async def GiveGroundFromCombat(self, loserUnit):
        direction = self.game.fleeDirectionMultUnits(
            loserUnit,
            [self.game.getSelectedUnit(u.bodyNP.node()) for u in loserUnit.isInCombatWith])

        persuingUnit = []
        persuingUnit.append(loserUnit)
        self.game.attackSequence2 = Sequence()
        for i, unit in enumerate(loserUnit.isInCombatWith):
            if unit.madePursuitChoice:
                loserUnit.isInCombatWith.remove(unit)
                loserUnit.isInCombatFlank.remove(loserUnit.isInCombatFlank[i])
                loserUnit.request("Idle")
                continue
            unit.madePursuitChoice = True
            persuitOrNot = [unit.unitName + '\nPersuit', unit.unitName + '\nRestrain']
            selected_choice = await taskMgr.add(
                self.game.makeChoiceNew(persuitOrNot, Vec3(0, 0, 10)))
            print(f"Selected choice: {selected_choice}")
            if selected_choice == persuitOrNot[0]:
                print(f"{unit.unit.name} chooses to pursue!")
                persuingUnit.append(unit)

        crashFractionMin = 1.0
        for i, unit in enumerate(persuingUnit):
            persuit_results = 2
            persuit_score = persuit_results
            print(f"Persuit dice results for {unit.unit.name}: {persuit_results}, total: {persuit_score}")
            crashFraction = self.game.sweepTest(unit, direction, persuit_score) * .95
            crashFractionMin = min(crashFraction, crashFractionMin)

            print(f"{unit.unit.name} successfully pursues the fleeing unit!")
            self.game.attackSequence2.append(
                Func(self.game.fallBack, unit.bodyNP, direction,
                     length=persuit_score * crashFractionMin, GG=True))
            self.game.attackSequence2.append(Wait(0.25))

        if not self.game.attackSequence.isPlaying():
            await self.game.attackSequence2

    # ─── Post-Combat: Fall Back In Good Order ─────────────────────────────

    async def FBIGFromCombat(self, loserUnit):
        direction = self.game.fleeDirectionMultUnits(
            loserUnit,
            [self.game.getSelectedUnit(u.bodyNP.node()) for u in loserUnit.isInCombatWith])
        persuitDiceTasks = []
        persuitDiceDices = []
        persuingUnit = []

        self.game.attackSequence2 = Sequence()
        for unit in loserUnit.isInCombatWith:
            persuitOrNot = [unit.unitName + '\nPersuit', unit.unitName + '\nRestrain']
            selected_choice = await taskMgr.add(
                self.game.makeChoiceNew(persuitOrNot, Vec3(0, 0, 10)))
            print(f"Selected choice: {selected_choice}")
            if selected_choice == persuitOrNot[0]:
                print(f"{unit.unit.name} chooses to pursue!")
                unit.request("IsPursuing")
                persuingUnit.append(unit)
            else:
                print(f"{unit.unit.name} chooses to restrain.")
                unit.request("Idle")

        if len(persuingUnit) == 0:
            print("No units chose to pursue, ending FBIG.")
            loserUnit.request("Idle")

        persuingUnit.append(loserUnit)

        for i, unit in enumerate(persuingUnit):
            if unit != loserUnit:
                continue
            terningerPersuit = []
            for j in range(2):
                terning = Dice(self.game.world,
                               position=unit.bodyNP.getPos() + Vec3(-20 + j * 4, 0, 10), size=1.0)
                terningerPersuit.append(terning)
            for terning in terningerPersuit:
                terning.roll()

            persuitDiceTasks.append(
                taskMgr.add(checkDice,
                            "checkDiceTaskPersuit" + str(loserUnit.unitName),
                            extraArgs=[terningerPersuit], appendTask=True))
            persuitDiceDices.append(terningerPersuit)

        for task in persuitDiceTasks:
            await task

        maxmove = max([terning.currentValue for terning in persuitDiceDices[-1]])
        for i in range(len(persuingUnit) - 1, -1, -1):
            unit = persuingUnit[i]

            if unit == loserUnit:
                persuitDices = persuitDiceDices[0]
                persuit_results = [terning.currentValue for terning in persuitDices]
                persuit_score = max(persuit_results)
            else:
                pass
            print(f"Persuit dice results for {unit.unit.name}: {persuit_results}, total: {persuit_score}")

            print(f"{unit.unit.name} successfully pursues the fleeing unit!")
            if unit != loserUnit:
                pass
            else:
                self.game.attackSequence2.append(
                    Func(self.game.fallBack, unit.bodyNP, direction,
                         length=persuit_score * 1.0, rally=True))
            self.game.attackSequence2.append(Wait(0.7))
        for dices in persuitDiceDices:
            for terning in dices:
                terning.remove(self.game.world)

        self.game.attackSequence2.append(Wait(2 * (len(persuingUnit) - 1)))
        await self.game.attackSequence2
        
        for i in range(0, len(persuingUnit) - 1):
            unit = persuingUnit[i]
            rFrom = unit.bodyNP.getHpr()
            unit.bodyNP.lookAt(loserUnit.bodyNP)
            rTo = unit.bodyNP.getHpr()
            unit.bodyNP.setHpr(rFrom)
            rotation_interval = LerpPosHprInterval(
                unit.bodyNP,
                duration=0.5,
                pos=unit.bodyNP.getPos(),
                hpr=rTo,
                blendType='easeInOut'
            )
            await rotation_interval
            unit.request("IsPursuing")
            unit.hasMovedThisTurn = False
            opos = unit.bodyNP.getPos() - direction
            orot = unit.bodyNP.getHpr()
            self.game.autoCharge = True
            self.game.autoHold = True
            self.game.pathTowardsMouse(unit, loserUnit.bodyNP.getPos().x, loserUnit.bodyNP.getPos().y)
            self.game.moveUnit(unit)
            await Wait(5.0)
        loserUnit.request("Moved")

    # ─── Post-Combat: Flee ────────────────────────────────────────────────

    async def fleeFromCombat(self, loserUnit):
        direction = self.game.fleeDirectionMultUnits(
            loserUnit,
            [self.game.getSelectedUnit(u.bodyNP.node()) for u in loserUnit.isInCombatWith])
        persuitDiceTasks = []
        persuitDiceDices = []
        persuingUnit = []

        self.game.attackSequence2 = Sequence()
        for unit in loserUnit.isInCombatWith:
            if unit.madePursuitChoice:
                continue
            unit.madePursuitChoice = True
            persuitOrNot = [unit.unitName + '\nPersuit', unit.unitName + '\nRestrain']
            selected_choice = await taskMgr.add(
                self.game.makeChoiceNew(persuitOrNot, Vec3(0, 0, 10)))
            print(f"Selected choice: {selected_choice}")
            unit.request("Idle")
            if selected_choice == persuitOrNot[0]:
                print(f"{unit.unit.name} chooses to pursue!")
                unit.request("IsPursuing")
                persuingUnit.append(unit)

        persuingUnit.append(loserUnit)

        for i, unit in enumerate(persuingUnit):
            if unit != loserUnit:
                continue
            terningerPersuit = []
            for j in range(2):
                terning = Dice(self.game.world,
                               position=unit.bodyNP.getPos() + Vec3(-20 + j * 4, 0, 10), size=1.0)
                terningerPersuit.append(terning)
            for terning in terningerPersuit:
                terning.roll()

            persuitDiceTasks.append(
                taskMgr.add(checkDice,
                            "checkDiceTaskPersuit" + str(loserUnit.unitName),
                            extraArgs=[terningerPersuit], appendTask=True))
            persuitDiceDices.append(terningerPersuit)

        for task in persuitDiceTasks:
            await task
        for i in range(len(persuingUnit) - 1, -1, -1):
            unit = persuingUnit[i]

            if unit == loserUnit:
                persuitDices = persuitDiceDices[0]
                persuit_results = [terning.currentValue for terning in persuitDices]
                persuit_score = sum(persuit_results)
                print(f"Persuit dice results for {unit.unit.name}: {persuit_results}, total: {persuit_score}")

                print(f"{unit.unit.name} successfully pursues the fleeing unit!")
                await self.game.fallBack2(unit.bodyNP, direction, length=persuit_score * 1.0, flee=True)
            else:
                pass

        for dices in persuitDiceDices:
            for terning in dices:
                terning.remove(self.game.world)

        loserUnit.request("IsFleeing")

        for n, persuing in enumerate(persuingUnit):
            if persuing == loserUnit:
                continue
            taskMgr.doMethodLater(
                1.7 * (len(persuingUnit) - 1), self.game.checkFleeCaught,
                "checkFleeCaughtTask" + str(n),
                extraArgs=[loserUnit, persuing], appendTask=True)

        if not self.game.attackSequence.isPlaying():
            await self.game.attackSequence2

        for n, persuing in enumerate(persuingUnit):
            if taskMgr.hasTaskNamed("checkFleeCaughtTask" + str(n)):
                taskMgr.remove("checkFleeCaughtTask" + str(n))

        loserPos = loserUnit.bodyNP.getPos()
        for i in range(0, len(persuingUnit) - 1):
            unit = persuingUnit[i]
            rFrom = unit.bodyNP.getHpr()
            unit.bodyNP.lookAt(loserPos)
            rTo = unit.bodyNP.getHpr()
            unit.bodyNP.setHpr(rFrom)
            rotation_interval = LerpPosHprInterval(
                unit.bodyNP,
                duration=0.5,
                pos=unit.bodyNP.getPos(),
                hpr=rTo,
                blendType='easeInOut'
            )
            await rotation_interval
            unit.request("IsPursuing")
            unit.hasMovedThisTurn = False
            opos = unit.bodyNP.getPos() - direction
            orot = unit.bodyNP.getHpr()
            self.game.autoCharge = True
            self.game.autoHold = True
            self.game.pathTowardsMouse(unit, loserPos.x, loserPos.y)
            self.game.moveUnit(unit)
            await Wait(5.0)
