"""
Spell system for Warhammer: The Old World tabletop battle game.

Defines the base Spell class and concrete spell implementations used
during the Strategy / Spell phases.
"""

import math
import random
import textwrap
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

    # True for a spell aimed at a point on the board rather than at a unit.
    targets_ground = False

    def __init__(self, name, casting_value, duration_list=None, wizard_level=1,
                 effect='', game=None, caster=None):
        self.name = name
        self.casting_value = casting_value
        self.duration_list = duration_list
        self.wizard_level = wizard_level
        self.effect = effect
        self.game = game
        self.caster = caster
        self.casting = 0            # result a dispel attempt must beat
        self.perfect = False        # a perfect invocation cannot be dispelled
        self.no_more_spells = False # the caster is spent for this turn
        # End-of-turn ticks this spell survives before endSpell() is called.
        self.ticks_remaining = 1

    async def spellFunction(self, target):
        """Cast at *target*: roll, offer the Dispel, then apply the effect.

        The Dispel belongs between the roll and the effect. A dispelled spell
        never happens at all, so nothing may have been worked out yet — the
        engine used to resolve the effect first and undo it afterwards, which
        showed the player damage that was then taken back.
        """
        if not self.canTarget(target):
            return
        if not await self._attempt(target):
            return
        if await self._dispelled():
            return
        await self.apply(target)

    def canTarget(self, target) -> bool:
        """Whether *target* is legal. Overrides print why when they refuse."""
        return True

    async def apply(self, target):
        """The spell's effect, once it is cast and has survived the Dispel."""
        raise NotImplementedError

    async def _dispelled(self) -> bool:
        """Offer the other side its single Dispel attempt."""
        if self.perfect or not self.casting:
            return False
        if self.game is None or self.caster is None:
            return False
        return await self.game.dispelAttempt(self, self.caster)

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

    # ─── Shared effect helpers ──────────────────────────────────────

    def _game_of(self, unit):
        return self.game or getattr(unit, 'game', None)

    def _magic_hits(self, unit, dice, strength, ap, flaming=False):
        """Roll *dice* automatic hits of the given Strength and AP onto *unit*.

        A spell has no attacking model, so nothing rolls To Hit; the hits go
        straight to the To Wound roll and the target's usual saves.
        """
        from battleFunctions import resolve_magic_hits
        from models import roll_dice_expr
        hits = roll_dice_expr(dice)
        wounds, saves, unsaved = resolve_magic_hits(unit.unit, hits, strength, ap)
        ap_str = f"AP-{ap}" if ap else "AP0"
        flame = ", Flaming Attacks" if flaming else ""
        print(f"   {unit.unit.name}: {hits} S{strength} {ap_str} hit(s){flame} "
              f"-> {wounds} wound(s), {saves} saved, {unsaved} unsaved")
        game = self._game_of(unit)
        if unsaved and game is not None:
            game.movement.applyWounds(unit, unsaved)
            game.psychology.check_heavy_casualties(unit, 'shooting',
                                                   attacker=self.caster)
        return unsaved


class DevilsVisitSpell(Spell):
    """Increases an ally unit's Movement characteristic for one turn."""

    def __init__(self, name, casting_value, duration_list, **kw):
        super().__init__(name, casting_value, duration_list, **kw)
        self.affected_unit = None

    async def apply(self, unit):
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

    async def apply(self, unit):
        target = getattr(getattr(unit, 'unit', None), 'name', unit)
        print(f"   target: {target}")
        print(f"   {self.effect or 'No effect text in the catalogue.'}")
        print("   (not yet applied automatically)")


# ── Battle Magic (Rulebook p. 320) ────────────────────────────────────────

# "Until your next Start of Turn sub-phase" spans a whole round: the spell has
# to outlive the end of the caster's turn and the end of the opponent's.
UNTIL_NEXT_START_OF_TURN = 2

# A small (3") blast template, in world units.
BLAST_TEMPLATE_SMALL = 3.0


class FireballSpell(Spell):
    """1. Fireball — 2D6 Strength 4 hits with no AP, and Flaming Attacks."""

    async def apply(self, unit):
        self._magic_hits(unit, '2D6', strength=4, ap=0, flaming=True)


class CurseOfArrowAttractionSpell(Spell):
    """2. Curse of Arrow Attraction — shots at the target re-roll natural 1s
    To Hit, until the caster's next Start of Turn."""

    affected_unit = None

    async def apply(self, unit):
        self.attach(unit, UNTIL_NEXT_START_OF_TURN)
        print(f"   {unit.unit.name} is cursed: shooting at it re-rolls "
              f"natural 1s To Hit.")

    def attach(self, unit, ticks):
        """Put the curse on *unit*; also how a save restores it."""
        self.affected_unit = unit
        unit.unit.model.arrow_attraction = True
        self.ticks_remaining = ticks
        self.duration_list.append(self)

    def endSpell(self):
        affected = getattr(self, 'affected_unit', None)
        if affected is not None:
            affected.unit.model.arrow_attraction = False


def distance_to_segment(px, py, ax, ay, bx, by) -> float:
    """Shortest distance from a point to the segment a-b, on the table plane."""
    dx, dy = bx - ax, by - ay
    span = dx * dx + dy * dy
    t = 0.0 if span < 1e-9 else ((px - ax) * dx + (py - ay) * dy) / span
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (ax + dx * t), py - (ay + dy * t))


# Templates are placed *not touching* a base, so clearance is a shade over the
# radii that just meet.
_TOUCHING = 0.01


def nudge_clear(centre, radius, obstacles, directions: int = 72):
    """The smallest shift putting a template of *radius* centred on *centre*
    clear of every obstacle, each given as (x, y, base radius).

    A Magical Vortex is never placed touching a model's base, so one that ends
    a move over a unit is moved by the smallest amount possible, in any
    direction (Rulebook p. 107). Returns (dx, dy), zero when it already stands
    clear. "Any direction" is sampled rather than solved, which costs at most
    half a degree of slack on a shift of a few inches.
    """
    cx, cy = centre
    blocking = [(x, y, radius + r + _TOUCHING) for x, y, r in obstacles
                if math.hypot(cx - x, cy - y) < radius + r + _TOUCHING]
    if not blocking:
        return 0.0, 0.0
    best = None
    for i in range(directions):
        a = 2.0 * math.pi * i / directions
        dx, dy = math.cos(a), math.sin(a)
        need = 0.0
        for x, y, clear in blocking:
            vx, vy = cx - x, cy - y
            proj = vx * dx + vy * dy
            disc = max(0.0, proj * proj - (vx * vx + vy * vy) + clear * clear)
            need = max(need, math.sqrt(disc) - proj)
        if best is None or need < best[0]:
            best = (need, dx, dy)
    need, dx, dy = best
    return dx * need, dy * need


class PillarOfFireSpell(Spell):
    """3. Pillar of Fire — a Magical Vortex that Remains in Play.

    A 3" template of difficult terrain, placed with its central hole within
    12" of the caster and never touching a base. It scatters D6" every Start
    of Turn and burns any enemy unit that walks through it, or that it drifts
    over, for D3+3 Strength 3 hits at AP -2.
    """

    targets_ground = True

    piece = None

    def canTarget(self, point):
        caster = self.caster
        if caster is None:
            return True
        reach = Vec3(Point3(point) - caster.bodyNP.getPos())
        reach.z = 0
        if reach.length() > self.RANGE:
            print(f"   that point is {reach.length():.0f}\" away; the template "
                  f"must be placed within {self.RANGE:.0f}\".")
            return False
        return True

    RANGE = 12.0

    async def apply(self, point):
        game = self.game
        if game is None:
            return
        # Placing it hurts nobody: the template goes down clear of every base,
        # so there is no unit under it to burn.
        self.place(game, Point3(point))

    def place(self, game, point):
        """Put the template on the board; also how a save restores it."""
        self.game = game
        self.piece = game.terrain_manager.add_terrain(
            'pillar_of_fire', Point3(point.x, point.y, 0.1),
            BLAST_TEMPLATE_SMALL, BLAST_TEMPLATE_SMALL)
        # Remains in Play: it lives on the board, not on the turn timer.
        game.remainsInPlay.append(self)
        self.settle(game)

    def scatter(self, game):
        """Drift D6" in a random direction, burning whatever it crosses.

        Called once per Start of Turn sub-phase while the spell is in play.
        """
        if self.piece is None:
            return
        angle = random.uniform(0.0, 2.0 * math.pi)
        distance = random.randint(1, 6)
        start = Point3(self.piece.center)
        self.shift(Vec3(math.cos(angle) * distance,
                        math.sin(angle) * distance, 0))
        print(f"[Magic] {self.name} scatters {distance}\" to "
              f"({self.piece.center.x:.0f}, {self.piece.center.y:.0f}).")
        # "any enemy unit that the template moves over": the ground it swept
        # counts, not only where it came to rest.
        self.burn_units_between(game, start, self.piece.center)
        self.settle(game)

    def shift(self, move):
        """Move the template and everything drawn for it."""
        self.piece.center = Point3(self.piece.center + move)
        for np in (self.piece.visual, self.piece.ghost_np, self.piece.outline):
            if np is not None and not np.isEmpty():
                np.setPos(np.getPos() + move)

    def settle(self, game):
        """Shift the template off any base it came to rest on.

        A Magical Vortex is never placed touching a model's base, so one whose
        scatter ends over a unit is moved the least it can be, in any
        direction, until it stands clear (Rulebook p. 107).
        """
        obstacles = []
        for unit in self.all_units(game):
            reach = self.base_radius(unit)
            obstacles += [(x, y, reach) for x, y in self.model_positions(unit)]
        centre = self.piece.center
        dx, dy = nudge_clear((centre.x, centre.y), self.piece.width / 2.0,
                             obstacles)
        if dx or dy:
            self.shift(Vec3(dx, dy, 0))
            print(f"[Magic] {self.name} may not rest on a base; nudged "
                  f"{math.hypot(dx, dy):.1f}\" to "
                  f"({self.piece.center.x:.0f}, {self.piece.center.y:.0f}).")

    def burn_units_between(self, game, start, end):
        """Hit every enemy unit the template covered on its way from *start*
        to *end*."""
        from characters import friendly_units
        for other in self.enemies(game):
            if self.caught(other, start, end):
                self.burn(other)
        # Only enemy units burn, which reads as a bug when the template drifts
        # back over the caster's own line and nothing happens.
        for own in friendly_units(game, self.caster):
            if not own.bodyNP.isEmpty() and self.caught(own, start, end):
                print(f"   {own.unit.name} is passed over but is friendly to "
                      f"the caster — {self.name} burns enemy units only.")

    def caught(self, unit, start, end) -> bool:
        """Whether any of *unit*'s models lie under the template's path.

        A model counts when its *base* meets the template, which is the same
        reach `settle` uses to decide the template is touching one. Measuring
        centres alone let a vortex sweep between two ranks, close enough to be
        nudged off the unit afterwards, and burn nobody.
        """
        reach = self.piece.width / 2.0 + self.base_radius(unit)
        return any(distance_to_segment(x, y, start.x, start.y,
                                       end.x, end.y) <= reach
                   for x, y in self.model_positions(unit))

    @staticmethod
    def model_positions(unit):
        """Where each of the unit's models stands, in world space."""
        if unit.model.isEmpty():
            p = unit.bodyNP.getPos()
            return [(p.x, p.y)]
        out = []
        for child in unit.model.getChildren():
            p = child.getPos(render)
            out.append((p.x, p.y))
        return out

    @staticmethod
    def base_radius(unit):
        """Half a model base's diagonal — the round template only needs to
        know how far a base reaches, not which way it faces."""
        return math.hypot(getattr(unit, 'modelWidth', 1.0),
                          getattr(unit, 'modelHeight', 1.0)) / 2.0

    def burn(self, unit):
        self._magic_hits(unit, 'D3+3', strength=3, ap=2, flaming=True)

    @staticmethod
    def all_units(game):
        """Every live unit. The template may not touch *any* base, friend or
        foe; only the damage cares whose side a unit is on."""
        units = getattr(game, 'units', None)
        if units is None:
            units = list(game.player1Units) + list(game.player2Units)
        return [u for u in units if not u.bodyNP.isEmpty()]

    def enemies(self, game):
        from characters import enemy_units
        return [u for u in enemy_units(game, self.caster)
                if not u.bodyNP.isEmpty()]

    def endSpell(self):
        if self.piece is None:
            return
        game = self.game
        if game is not None:
            game.terrain_manager.remove_terrain(self.piece)
            if self in game.remainsInPlay:
                game.remainsInPlay.remove(self)
        self.piece = None


class ArcaneUrgencySpell(Spell):
    """4. Arcane Urgency — a friendly unit that has already moved this phase
    may immediately move again."""

    def canTarget(self, unit):
        if unit.state == 'IsFleeing':
            print(f"   {unit.unit.name} is fleeing and cannot be hurried.")
            return False
        if not getattr(unit, 'hasMovedThisTurn', False):
            print(f"   {unit.unit.name} has not moved yet this phase.")
            return False
        return True

    async def apply(self, unit):
        unit.hasMovedThisTurn = False
        unit.updateTextNode()
        print(f"   {unit.unit.name} may move again.")


class OakenShieldSpell(Spell):
    """5. Oaken Shield — the caster, and any unit it has joined, gain a 5+ Ward
    save until the caster's next Start of Turn."""

    WARDING_VALUE = 5

    affected_unit = None
    rule = None

    async def apply(self, unit):
        # Range 'Self': the target is the caster whatever was clicked.
        target = self.caster or unit
        self.attach(target, UNTIL_NEXT_START_OF_TURN)
        print(f"   {target.unit.name} gains a {self.WARDING_VALUE}+ Ward save.")

    def attach(self, unit, ticks):
        """Put the Ward on *unit*; also how a save restores it."""
        self.affected_unit = unit
        self.rule = {'name': self.name, 'ward': self.WARDING_VALUE}
        unit.unit.model.special_rules.append(self.rule)
        self.ticks_remaining = ticks
        self.duration_list.append(self)

    def endSpell(self):
        affected = getattr(self, 'affected_unit', None)
        rule = getattr(self, 'rule', None)
        if affected is None or rule is None:
            return
        rules = affected.unit.model.special_rules
        if rule in rules:
            rules.remove(rule)


class CurseOfCowardlyFlightSpell(Spell):
    """6. Curse of Cowardly Flight — an immediate Panic test the target cannot
    duck, even if it normally passes them automatically."""

    async def apply(self, unit):
        game = self._game_of(unit)
        if game is not None:
            game.psychology.panic_test(unit, flee_from=self.caster,
                                       cause=self.name, compulsory=True)


class HammerhandSpell(Spell):
    """Signature — an enemy unit the caster is engaged with suffers 2D3
    Strength 4 hits at AP -2."""

    def canTarget(self, unit):
        caster = self.caster
        if caster is not None and unit not in getattr(caster, 'isInCombatWith', []):
            print(f"   {caster.unit.name} is not engaged in combat with "
                  f"{unit.unit.name}.")
            return False
        return True

    async def apply(self, unit):
        self._magic_hits(unit, '2D3', strength=4, ap=2)


# The catalogue gives a spell's name, casting value, range and wording; what it
# cannot give is the effect, so each one is matched to its class by name.
BATTLE_MAGIC = {
    'Fireball': FireballSpell,
    'Curse of Arrow Attraction': CurseOfArrowAttractionSpell,
    'Pillar of Fire': PillarOfFireSpell,
    'Arcane Urgency': ArcaneUrgencySpell,
    'Oaken Shield': OakenShieldSpell,
    'Curse of Cowardly Flight': CurseOfCowardlyFlightSpell,
    'Hammerhand': HammerhandSpell,
}


def spell_class(name: str):
    """The coded class for a spell, or None if only its wording is known."""
    return BATTLE_MAGIC.get((name or '').strip())


def spell_readout(name: str, spell: dict, width: int = 46) -> str:
    """The card for one spell: type, casting value, range and wording.

    Wrapped here rather than by the GUI because both the hover panel and the
    status line are plain text nodes.
    """
    spell = spell or {}
    reach = spell.get('range')
    reach = f'{reach}"' if isinstance(reach, (int, float)) else (reach or '-')
    head = (f"{name}  ({spell.get('type') or 'Spell'})\n"
            f"Casting {spell.get('casting_value') or '?'}+   Range {reach}")
    body = ' '.join((spell.get('effect') or '').split())
    return f"{head}\n{textwrap.fill(body, width)}" if body else head


# ── Saving and restoring spells in play ───────────────────────────────────
#
# A hex, a ward or a vortex outlives the turn it was cast in, so a quicksave
# taken while one is up has to carry it or the spell silently ends on load.

def save_spells(game) -> list:
    """A JSON-safe record of every spell still in play."""
    out = []
    for spell in list(game.fsm.endOfTurnSpells) + list(game.remainsInPlay):
        target = getattr(spell, 'affected_unit', None)
        piece = getattr(spell, 'piece', None)
        out.append({
            'name': spell.name,
            'casting_value': spell.casting_value,
            'wizard_level': spell.wizard_level,
            'effect': spell.effect,
            'ticks': spell.ticks_remaining,
            'caster': spell.caster.unitName if spell.caster is not None else None,
            'target': target.unitName if target is not None else None,
            'center': ([piece.center.x, piece.center.y]
                       if piece is not None else None),
        })
    return out


def load_spells(game, records, unit_map):
    """Put saved spells back in play. Nothing is re-rolled: the casting is
    history, only its lingering effect is restored."""
    for data in records or ():
        cls = spell_class(data.get('name'))
        if cls is None:
            continue
        caster = unit_map.get(data.get('caster'))
        spell = cls(data['name'], data.get('casting_value') or 12,
                    game.fsm.endOfTurnSpells,
                    wizard_level=data.get('wizard_level') or 1,
                    effect=data.get('effect', ''), game=game, caster=caster)
        center = data.get('center')
        if center is not None and hasattr(spell, 'place'):
            spell.place(game, Point3(center[0], center[1], 0.1))
            continue
        target = unit_map.get(data.get('target'))
        if target is not None and hasattr(spell, 'attach'):
            spell.attach(target, data.get('ticks') or 1)


class RaiseDeadSpell(Spell):
    """Raises fallen models back into an allied Undead unit."""

    def __init__(self, name, casting_value, duration_list, **kw):
        super().__init__(name, casting_value, duration_list, **kw)

    def endSpell(self):
        pass

    async def apply(self, unit):
        taskMgr.remove("taskShootingTrajectoryDrawLine")

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
