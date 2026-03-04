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
from panda3d.bullet import BulletBoxShape
from panda3d.core import LRotationf
from direct.interval.LerpInterval import LerpPosHprInterval
from direct.interval.IntervalGlobal import Sequence, Parallel, Wait
from direct.interval.FunctionInterval import Func
from direct.task.Task import Task

from dice import Dice, checkDice
from battleFunctions import simulate_battle


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
        if self.game.autoCharge:
            cynchoice = "Yes"
        else:
            cynchoice = await taskMgr.add(self.game.makeChoiceNew(chargeYesNo, Vec3(-20, 0, 10)))

        if cynchoice == "Yes":
            print("Charging into combat...")

            chargeReaction = ["hold", "flee"]
            if self.game.autoHold:
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
            print(f"Defender unit width: {width}")
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
        print("rotate from to", newnode.getHpr(), contactRot)
        newnode_hpr = newnode.getHpr()
        positive_h = newnode_hpr.x % 360
        positive_p = newnode_hpr.y % 360
        positive_r = newnode_hpr.z % 360

        if positive_h > 180:
            positive_h -= 360

        if positive_h - contactRot.x > 180:
            contactRot = Vec3(contactRot.x + 360, contactRot.y, contactRot.z)
        newnode.setHpr(positive_h, positive_p, positive_r)

        print("rotate from to", newnode.getHpr(), contactRot)

        wheel1Angle = contactRot.x - orotUnit.x
        print("wheel1Angle:", wheel1Angle)
        newnode.setHpr(contactRot)
        wheel1Pos = newnode.getPos(render)

        if 1:
            direction = self.game.playerNP.getPos() - wheel1Pos
            wdistance = abs(math.radians(wheel1Angle) * width * 2)
            cdistance = self.game.moveArceDistance - wdistance
            print("Calculated distance to move forward:", cdistance, wdistance, width * 2)

        chdist = int(unit.unit.model.characteristics['M']) + max(chdice)
        for rule in unit.unit.model.special_rules:
            if rule.get('mountUnit'):
                chdist = int(rule['mountUnit'].model.characteristics['M']) + max(chdice)
        fldist = sum(fldice)
        print("Charge distance:", chdist)
        print("Flee distance:", fldist)
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
        print((contactPos - newnode.getPos()).normalized() * cdistance)
        print(cdistance, "aplpied to vector:", Vec3(vector.x, vector.y, 0), wdistance, chdist)
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
        maxmove = int(unit.unit.model.characteristics['M'])
        durIntConst = 1.0
        for rule in unit.unit.model.special_rules:
            if rule.get('mountUnit'):
                maxmove = int(rule['mountUnit'].model.characteristics['M'])
        if unit.state == "IsPursuing":
            maxmove = 0
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
            print(f"Defender unit width: {width}")
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
        print("rotate from to", newnode.getHpr(), contactRot)
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
        print("rotate from to", newnode.getHpr(), contactRot)

        wheel1Angle = contactRot.x - orotUnit.x
        if wheel1Angle > 180:
            wheel1Angle -= 360
        print("wheel1Angle:", wheel1Angle)
        newnode.setHpr(contactRot)
        wheel1Pos = newnode.getPos(render)

        if 1:
            direction = self.game.playerNP.getPos() - wheel1Pos
            wdistance = abs(math.radians(wheel1Angle) * width * 2)
            cdistance = self.game.moveArceDistance - wdistance
            print("Calculated distance to move forward:", cdistance, wdistance, width * 2)

        chdist = int(unit.unit.model.characteristics['M']) + max(chdice)
        for rule in unit.unit.model.special_rules:
            if rule.get('mountUnit'):
                chdist = int(rule['mountUnit'].model.characteristics['M']) + max(chdice)
        if unit.state == "IsPursuing":
            chdist = sum(chdice)
        print("Charge distance:", chdist)
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
            print("Charge distance less than wheel distance, returning.")
            unit.request("Moved")
            return
        ocdistance = cdistance
        if chdist < wdistance + cdistance:
            cdistance = chdist - wdistance

        angle = contactRot.x
        vector = Vec2(-math.sin(math.radians(angle)), math.cos(math.radians(angle)))
        print((contactPos - newnode.getPos()).normalized() * cdistance)
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
            print("Charge distance less than total distance, returning.",
                  chdist, wdistance, ocdistance, self.game.moveArceDistance)
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
        print("Final HPR:", finalHpr)
        print("Angle to rotate:", angleToRotate)
        print("Current HPR before final rotation:", newnode.getHpr())
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

    # ─── Flank Detection ──────────────────────────────────────────────────

    def getFlankFromContact(self, unit, contact):
        flank = "front"
        print("Unit collision detected!")
        angleAttacker = unit.bodyNP.getH()
        defenderNP = render.find(f"**/{contact.getNode1().getName()}")
        angleDefender = defenderNP.getH()
        print(f"contact position in defender coordsystem: {self.game.playerNP.getPos(defenderNP)}")
        hitloc = self.game.playerNP.getPos(defenderNP)

        shape = contact.getNode1().getShape(0)
        if isinstance(shape, BulletBoxShape):
            half_extents = shape.getHalfExtentsWithMargin()
            width = half_extents.x * unit.bodyNP.getScale().x
            height = half_extents.y * unit.bodyNP.getScale().y
            print(f"Defender unit width: {width}, height: {height}")

        angleToRotate = angleDefender - angleAttacker
        print(f"Attacker angle: {angleAttacker}, Defender angle: {angleDefender}")
        print(f"Rotating attacker by {angleToRotate} degrees to face defender.")
        angleToRotate = (angleToRotate) % 360
        print(f"normalized {angleToRotate} degrees to face defender.")
        print(f"Hit location in defender coords: {hitloc}")
        unitloc = unit.bodyNP.getPos(defenderNP)
        print(f"Attacker unit center location in defender coords: {unitloc}")
        hitloc = unitloc

        angle_between = math.acos(Vec3(0, 1, 0).dot(hitloc.normalized())) * (180.0 / math.pi)
        print("Angle between forward and hit location vector:", angle_between)
        frontArcAngle = 90 - math.atan2(height, width) * (180.0 / math.pi)
        print("Front arc angle:", frontArcAngle)

        if angle_between > frontArcAngle + 90:
            print("Hit rear side of defender")
            flank = "rear"
            print(f"Initial angle to rotate: {angleToRotate}")
            if angleToRotate > 90:
                angleToRotate = (360 - angleToRotate) * -1

        elif angle_between > frontArcAngle and hitloc.x < 0:
            print("Hit on left side of defender")
            flank = "flank"
            print(f"Initial angle to rotate: {angleToRotate}")
            if angleToRotate > 90:
                angleToRotate -= 90
            else:
                angleToRotate = 90 - angleToRotate
                angleToRotate *= -1
            print(f"Adjusted angle to rotate: {angleToRotate}")

        elif angle_between < frontArcAngle:
            print("Hit front side of defender")
            flank = "front"
            print(f"Initial angle to rotate: {angleToRotate}")
            if angleToRotate > 90:
                angleToRotate -= 180
            print(f"Adjusted angle to rotate: {angleToRotate}")

        elif angle_between > frontArcAngle and hitloc.x > 0:
            print("Hit on right side of defender")
            flank = "flank"
            print(f"Initial angle to rotate: {angleToRotate}")
            if angleToRotate < 0:
                angleToRotate += 90
            if angleToRotate > 90:
                angleToRotate = 360 - 90 - angleToRotate
                angleToRotate *= -1
            print(f"Adjusted angle to rotate: {angleToRotate}")

        else:
            print("Hit i dont know where")
        return flank, angleToRotate

    # ─── Battle Output ────────────────────────────────────────────────────

    def printBattleResults(self, attackerUnit, defenderUnit, attacks, total_hits,
                           suffered_wounds, saves_made, total_wounds):
        print(f"Battle results for {attackerUnit.unit.name} attacking with weapon "
              f"{attackerUnit.unit.model.equipedWeapon.get('name')}:")
        print(f"Total hits by {attackerUnit.unit.name} on {defenderUnit.unit.name}: {total_hits}")
        print(f"suffered wounds by {attackerUnit.unit.name} on {defenderUnit.unit.name}: {suffered_wounds}")
        print(f"Saves made by {defenderUnit.unit.name}: {saves_made}")
        print(f"Total wounds by {attackerUnit.unit.name} on {defenderUnit.unit.name}: {total_wounds}")

    # ─── Battle Start & Resolution ────────────────────────────────────────

    async def verySimpleBattleStart(self, task):
        weps = self.game.unitToMove.unit.model.weapons

        wepchoice = await taskMgr.add(self.game.makeChoiceNew(weps, Vec3(0, 0, 10)))

        self.game.unitToMove.unit.model.equip_weapon(wepchoice)
        print('Event delivered with args:', wepchoice)

        taskMgr.add(self.verySimpleBattle, "verySimpleBattleTask")
        return task.done

    async def verySimpleBattle(self, task):
        print("Starting very simple battle...")
        attacker = self.game.unitToMove.bodyNP
        defender = self.game.unitToMove.isInCombatWith[0].bodyNP
        flank = self.game.unitToMove.isInCombatFlank[0]
        engagedWith = [x.unitName for x in self.game.unitToMove.isInCombatWith]
        print("Attacker:", attacker.node().getName())
        print("engaged in battle with:", engagedWith)
        print("on flanks:", self.game.unitToMove.isInCombatFlank)

        selected_choice = await taskMgr.add(self.game.makeChoiceNew(engagedWith, Vec3(0, 0, 10)))

        print(f"Selected choice: {selected_choice}")
        for unit in self.game.unitToMove.isInCombatWith:
            if unit.unitName == selected_choice:
                defender = unit.bodyNP
                break
        print(f"{attacker.node().getName()} attacks {defender.node().getName()} in {flank}!")
        attackerUnit = self.game.getSelectedUnit(attacker.node())
        defenderUnit = self.game.getSelectedUnit(defender.node())
        defender_nmodels = defenderUnit.unit.nmodels
        print(f"{attackerUnit.unit.name} attacks {defenderUnit.unit.name} in {flank}!")
        if attackerUnit.unit.model.equipedWeapon.get('tag') == 'ranged':
            print("attacker unit has ranged weapon equiped, switch to melee weapon for combat.")
            attackerUnit.unit.model.equip_weapon('hand weapon')
        if defenderUnit.unit.model.equipedWeapon.get('tag') == 'ranged':
            print("defender unit has ranged weapon equiped, switch to melee weapon for combat.")
            defenderUnit.unit.model.equip_weapon('hand weapon')

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
        player1_score = 0
        player1_flank_bonus = 0
        player1_rank_bonus = 0
        player2_score = 0
        player2_flank_bonus = 0
        player2_rank_bonus = 0
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
                print("defender unit has ranged weapon equiped, switch to melee weapon for combat.")
                defenderUnit.unit.model.equip_weapon('hand weapon')
            if attackerUnit.unit.model.equipedWeapon.get('tag') == 'ranged':
                print("attacker unit has ranged weapon equiped, switch to melee weapon for combat.")
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

            attacks, total_hits, suffered_wounds, saves_made, total_wounds = simulate_battle(
                defenderUnit.unit, attackerUnit.unit, charge=False)
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
                        rule['mountUnit'], attackerUnit.unit, charge=False)
                    self.printBattleResults(defenderUnit, attackerUnit, attacks, total_hits,
                                            suffered_wounds, saves_made, total_wounds)
                    attackerUnit.unit.nmodels -= total_wounds

                    if defenderUnit in self.game.player1Units:
                        player1_score += total_wounds
                    else:
                        player2_score += total_wounds
                    combWounds += total_wounds
            self.game.attackSequence.append(
                Func(self.game.removeModelsFromUnit, attackerUnit, combWounds))

        player1_score += player1_flank_bonus + player1_rank_bonus
        player2_score += player2_flank_bonus + player2_rank_bonus
        print(f"Player 2 score: {player2_score}, Player 1 score: {player1_score}")
        print(f"Player 2 flank bonus: {player2_flank_bonus}, Player 1 flank bonus: {player1_flank_bonus}")
        print(f"Player 2 rank bonus: {player2_rank_bonus}, Player 1 rank bonus: {player1_rank_bonus}")
        await self.game.attackSequence
        self.game.attackSequence2 = Sequence()
        loserUnits = []
        if player2_score == player1_score:
            print("Combat is a draw, no units flee.")
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
                return

            if any(rule.get('Unbreakable', False) for rule in loserUnit.unit.model.special_rules):
                print(f"{loserUnit.unit.name} is Unbreakable and does not flee!, only gives ground.")
                await taskMgr.add(self.GiveGroundFromCombat, "fleeFromCombatTask",
                                   extraArgs=[loserUnit], appendTask=False)
                continue

            print("losing unit original LD:", loserUnit.unit.model.characteristics['Ld'],
                  "modified by combat diff:", diff,
                  "final LD to beat:", int(loserUnit.unit.model.characteristics['Ld']) - diff)
            terningerLd = []
            for i in range(2):
                terning = Dice(self.game.world, position=Vec3(20 + i * 2, 0, 10),
                               size=1.0, color=(1, 0, 0, 1))
                terningerLd.append(terning)
            for terning in terningerLd:
                terning.roll()
            await taskMgr.add(checkDice, "checkDiceTaskFlee",
                              extraArgs=[terningerLd], appendTask=True)
            ldDice = []
            for terning in terningerLd:
                ldDice.append(terning.currentValue)
            leadership_score = sum(ldDice)
            for terning in terningerLd:
                terning.remove(self.game.world)
            print("Leadership dice results for fleeing unit:", ldDice, "sum:", leadership_score)

            if leadership_score > int(loserUnit.unit.model.characteristics['Ld']):
                print("losing unit flees from combat!")
                await taskMgr.add(self.fleeFromCombat, "fleeFromCombatTask",
                                   extraArgs=[loserUnit], appendTask=False)
            elif leadership_score > int(loserUnit.unit.model.characteristics['Ld']) - diff:
                print("losing unit FBIG!")
                await taskMgr.add(self.FBIGFromCombat, "fleeFromCombatTask",
                                   extraArgs=[loserUnit], appendTask=False)
            else:
                print("losing unit gives ground!")
                await taskMgr.add(self.GiveGroundFromCombat, "fleeFromCombatTask",
                                   extraArgs=[loserUnit], appendTask=False)

        for loserUnit in loserUnits:
            loserUnit.madePursuitChoice = False
            for unit in loserUnit.isInCombatWith:
                unit.madePursuitChoice = False

        messenger.send('unit-move-complete')
        return task.done

    # ─── Post-Combat: Give Ground ─────────────────────────────────────────

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
            print("sweep test for fallback")
            crashFraction = self.game.sweepTest(unit, direction, persuit_score) * .95
            crashFractionMin = min(crashFraction, crashFractionMin)

            print(f"{unit.unit.name} successfully pursues the fleeing unit!")
            self.game.attackSequence2.append(
                Func(self.game.fallBack, unit.bodyNP, direction,
                     length=persuit_score * crashFractionMin, GG=True))
            self.game.attackSequence2.append(Wait(0.25))

        if not self.game.attackSequence.isPlaying():
            self.game.attackSequence2.start()

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
        loserUnit.request("Moved")
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
                print("removing task", "checkFleeCaughtTask" + str(n))
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
