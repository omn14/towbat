"""
Spell system for Warhammer: The Old World tabletop battle game.

Defines the base Spell class and concrete spell implementations used
during the Strategy / Spell phases.
"""

import math
from panda3d.core import Point3, Vec3, LRotationf
from panda3d.bullet import BulletBoxShape
from dice import Dice, checkDice
from rulesFunctions import plusSTAT


class Spell:
    """Base class for all spells."""

    def __init__(self, name, casting_value, duration_list=None):
        self.name = name
        self.casting_value = casting_value
        self.duration_list = duration_list

    async def cast(self, target_unit):
        """Override in subclasses to implement spell effects."""
        raise NotImplementedError

    def endSpell(self):
        """Called at end of turn to revert temporary spell effects."""
        pass

    # ─── Shared dice-rolling helper ─────────────────────────────────

    @staticmethod
    async def _roll_casting_dice(num_dice=2, position_base=Vec3(20, 0, 10)):
        """Roll dice and return the total score."""
        dice = []
        for i in range(num_dice):
            die = Dice(
                base.world,
                position=position_base + Vec3(i * 2, 0, 0),
                size=1.0,
                color=(1, 0, 0, 1),
            )
            dice.append(die)

        for die in dice:
            die.roll()

        await taskMgr.add(
            checkDice, "checkDiceTaskSpell", extraArgs=[dice], appendTask=True
        )

        values = [die.currentValue for die in dice]
        total = sum(values)

        for die in dice:
            die.remove(base.world)

        return total, values


class DevilsVisitSpell(Spell):
    """Increases an ally unit's Movement characteristic for one turn."""

    def __init__(self, name, casting_value, duration_list):
        super().__init__(name, casting_value, duration_list)
        self.affected_unit = None

    async def spellFunction(self, unit):
        total, values = await self._roll_casting_dice()

        if total < self.casting_value:
            print(
                f"Devil's Visit failed for unit: {unit.unit.name} "
                f"with score: {total}"
            )
            return

        print(
            f"Devil's Visit succeeded for unit: {unit.unit.name} "
            f"with score: {total}"
        )
        self.affected_unit = unit
        self.duration_list.append(self)
        plusSTAT(unit.unit.model, 'M', 11, -99)

    def endSpell(self):
        if self.affected_unit:
            plusSTAT(self.affected_unit.unit.model, 'M', -11, -99)


class RaiseDeadSpell(Spell):
    """Raises fallen models back into an allied Undead unit."""

    def __init__(self, name, casting_value, duration_list):
        super().__init__(name, casting_value, duration_list)

    def endSpell(self):
        pass

    async def spellFunction(self, unit):
        taskMgr.remove("taskShootingTrajectoryDrawLine")

        # Roll to cast
        cast_total, _ = await self._roll_casting_dice()
        if cast_total > 7:
            print(
                f"Raise Dead failed for unit: {unit.unit.name} "
                f"with LD score: {cast_total}"
            )
            return

        print(
            f"Raise Dead succeeded for unit: {unit.unit.name} "
            f"with LD score: {cast_total}"
        )
        old_ranks = (unit.unit.nmodels - 1) // unit.unit.files

        # Roll D3 for number of models raised
        d3_total, _ = await self._roll_casting_dice(num_dice=1)
        d3_score = d3_total / 2

        print(f"Dead models to raise for unit: {unit.unit.name} is: {d3_score}")
        unit.unit.nmodels += int(math.ceil(d3_score)) + 2

        children = unit.model.getChildren()
        files = unit.unit.files
        new_ranks = (unit.unit.nmodels - 1) // files
        unit.unit.ranks = new_ranks
        rank_diff = new_ranks - old_ranks
        print(
            f"Raising dead for unit: {unit.unit.name}, "
            f"Old ranks: {old_ranks}, New ranks: {new_ranks}, "
            f"Rank difference: {rank_diff}"
        )

        # Clone models if needed
        if unit.unit.nmodels != len(children):
            diff = unit.unit.nmodels - len(children)
            for _ in range(diff):
                clone = children[0].copyTo(unit.model)
                children.append(clone)

        # Remove excess models
        while len(children) > unit.unit.nmodels:
            children[-1].removeNode()
            children = unit.model.getChildren()

        # Reposition all models in formation
        for i, child in enumerate(children):
            row = i // files
            col = i % files
            p = Point3(col * unit.modelWidth, -row * unit.modelHeight, 0)
            child.setPos(p)

        # Rebuild collision shape
        base.world.removeRigidBody(unit.bodyNP.node())
        for shape in unit.bodyNP.node().shapes:
            unit.bodyNP.node().removeShape(shape)

        bounds = unit.model.getTightBounds()
        box_size = bounds[1] - bounds[0]
        shape = BulletBoxShape(box_size * 0.5)
        unit.bodyNP.node().addShape(shape)
        unit.bodyNP.node().setMass(0)
        base.world.attachRigidBody(unit.bodyNP.node())

        unit.model.setPos(0, 0, 0)
        unit.model.setPos(
            -box_size.x / 2 + unit.modelWidth / 2,
            box_size.y / 2 - unit.modelHeight / 2,
            0,
        )

        rot = LRotationf()
        rot.setHpr(unit.bodyNP.getHpr())
        fwd = rot.getForward()
        unit.bodyNP.setPos(
            unit.bodyNP.getPos() - fwd * unit.modelHeight / 2 * rank_diff
        )
