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


def casting_result(dice_total: int, wizard_level: int) -> int:
    """A Casting roll of 2D6 plus half the caster's Level of Wizardry, rounding
    fractions up (Rulebook p. 108). The spell is cast if this equals or exceeds
    the spell's casting value."""
    return dice_total + math.ceil(max(0, wizard_level) / 2)


# ── Casting outcomes (Rulebook p. 108-109) ────────────────────────────────

CAST_FAILED = 'failed'
CAST_SUCCESS = 'cast'
CAST_MISCAST = 'miscast'
CAST_PERFECT = 'perfect invocation'


def casting_outcome(dice, wizard_level: int, casting_value: int):
    """(outcome, casting result) for a Casting roll.

    A natural double 6 is a perfect invocation and cannot be dispelled; a
    natural double 1 is a miscast whatever the result would have been.
    """
    result = casting_result(sum(dice), wizard_level)
    pair = list(dice[:2])
    if pair == [6, 6]:
        return CAST_PERFECT, result
    if pair == [1, 1]:
        return CAST_MISCAST, result
    if result >= casting_value:
        return CAST_SUCCESS, result
    return CAST_FAILED, result


# 2D6, Rulebook p. 109. 'cast' says whether the spell still goes off; a Wizard
# that survives the last two rows may attempt no more spells this turn.
MISCAST_TABLE = (
    (2, 4, 'Dimensional Cascade',
     'Centre a 5" blast template over the Wizard. Every model beneath it risks '
     'a single Strength 10 hit with an AP of -4.',
     {'cast': False, 'blast': 5, 'strength': 10, 'ap': 4}),
    (5, 6, 'Calamitous Detonation',
     'Centre a 3" blast template over the Wizard. Every model beneath it risks '
     'a single Strength 6 hit with an AP of -2.',
     {'cast': False, 'blast': 3, 'strength': 6, 'ap': 2}),
    (7, 7, 'Careless Conjuration',
     'The Wizard suffers a single Strength 4 hit with an AP of -1.',
     {'cast': False, 'strength': 4, 'ap': 1}),
    (8, 9, 'Barely Controlled Power',
     'The spell is cast at its casting value. The Wizard cannot attempt to '
     'cast any more spells this turn.',
     {'cast': True, 'at_casting_value': True, 'no_more_spells': True}),
    (10, 12, 'Power Drain',
     'The spell is cast with a perfect invocation. You cannot attempt to cast '
     'any more spells this turn.',
     {'cast': True, 'perfect': True, 'no_more_spells': True}),
)


def miscast_result(roll: int) -> dict:
    """The Miscast table entry for a 2D6 roll."""
    for low, high, name, effect, flags in MISCAST_TABLE:
        if low <= roll <= high:
            entry = {'roll': roll, 'name': name, 'effect': effect,
                     'cast': False, 'perfect': False, 'no_more_spells': False,
                     'at_casting_value': False, 'blast': 0, 'strength': 0,
                     'ap': 0}
            entry.update(flags)
            return entry
    raise ValueError(f"a 2D6 Miscast roll cannot be {roll}")


# ── Dispelling (Rulebook p. 110) ──────────────────────────────────────────

def dispel_result(dice, wizard_level: int = 0, wizardly: bool = False) -> int:
    """A Dispel roll of 2D6. A Wizardly dispel adds half the dispelling
    Wizard's Level, rounding up; a Fated dispel adds nothing."""
    total = sum(dice)
    return casting_result(total, wizard_level) if wizardly else total


def is_dispelled(dispel: int, casting: int) -> bool:
    """The dispel result must *exceed* the casting result; a tie fails."""
    return dispel > casting


# ── How many spells a turn (Rulebook p. 108) ──────────────────────────────

def may_attempt(spells_cast, spell_name: str, wizard_level: int,
                blocked: bool = False) -> bool:
    """A Wizard may attempt as many spells a turn as its Level of Wizardry, and
    may only attempt each spell once."""
    if blocked or spell_name in (spells_cast or ()):
        return False
    return len(spells_cast or ()) < max(1, wizard_level)


class Spell:
    """Base class for all spells."""

    def __init__(self, name, casting_value, duration_list=None, wizard_level=1,
                 effect=''):
        self.name = name
        self.casting_value = casting_value
        self.duration_list = duration_list
        self.wizard_level = wizard_level
        self.effect = effect
        self.casting = 0            # result a dispel attempt must beat
        self.perfect = False        # a perfect invocation cannot be dispelled
        self.no_more_spells = False # the caster is spent for this turn

    async def cast(self, target_unit):
        """Override in subclasses to implement spell effects."""
        raise NotImplementedError

    def endSpell(self):
        """Called at end of turn to revert temporary spell effects."""
        pass

    async def _attempt(self, unit):
        """Roll to cast and report. Returns True if the spell goes off.

        Sets `self.casting` to the casting result a dispel attempt must beat,
        and `self.perfect` when the spell cannot be dispelled at all.
        """
        total, values = await self._roll_casting_dice()
        outcome, result = casting_outcome(values, self.wizard_level,
                                          self.casting_value)
        self.casting = result
        self.perfect = outcome == CAST_PERFECT
        self.no_more_spells = False
        print(f"{self.name}: casting roll {values} = {total} "
              f"+ {result - total} (Level {self.wizard_level}) = {result} "
              f"vs {self.casting_value}+ -> {outcome}")

        if outcome != CAST_MISCAST:
            return outcome in (CAST_SUCCESS, CAST_PERFECT)

        miscast_roll, _ = await self._roll_casting_dice()
        entry = miscast_result(miscast_roll)
        print(f"   Miscast! {miscast_roll}: {entry['name']} — {entry['effect']}")
        self.perfect = entry['perfect']
        self.no_more_spells = entry['no_more_spells']
        if entry['at_casting_value']:
            self.casting = self.casting_value
        return entry['cast']

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

    def __init__(self, name, casting_value, duration_list, **kw):
        super().__init__(name, casting_value, duration_list, **kw)
        self.affected_unit = None

    async def spellFunction(self, unit):
        if not await self._attempt(unit):
            return
        self.affected_unit = unit
        self.duration_list.append(self)
        plusSTAT(unit.unit.model, 'M', 11, -99)

    def endSpell(self):
        if self.affected_unit:
            plusSTAT(self.affected_unit.unit.model, 'M', -11, -99)


class CatalogueSpell(Spell):
    """A spell loaded from the catalogue that has no coded effect yet.

    It rolls to cast properly and prints the rulebook's own wording, so the
    effect can be applied by hand until it is implemented.
    """

    async def spellFunction(self, unit):
        if not await self._attempt(unit):
            return
        target = getattr(getattr(unit, 'unit', None), 'name', unit)
        print(f"   target: {target}")
        print(f"   {self.effect or 'No effect text in the catalogue.'}")
        print("   (not yet applied automatically)")


class RaiseDeadSpell(Spell):
    """Raises fallen models back into an allied Undead unit."""

    def __init__(self, name, casting_value, duration_list, **kw):
        super().__init__(name, casting_value, duration_list, **kw)

    def endSpell(self):
        pass

    async def spellFunction(self, unit):
        taskMgr.remove("taskShootingTrajectoryDrawLine")

        if not await self._attempt(unit):
            return
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
