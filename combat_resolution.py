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
import random
from types import SimpleNamespace

from panda3d.core import Vec2, Vec3, Point3, NodePath, TransformState

from collision_masks import CollisionMask as CM


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
from direct.interval.LerpInterval import LerpFunc
from direct.task.Task import Task

from dice import Dice, checkDice
from battleFunctions import (MIN_IMPACT_HIT_CHARGE, base_initiative,
                             charge_initiative_bonus, impact_hit_report,
                             resolve_impact_hits, simulate_battle,
                             strike_initiative, take_last_slaying_blows)
from characters import JOIN_TAG, slay_character
from challenges import (Challenge, MAX_OVERKILL, add_challenge, can_accept,
                        duellist, end_challenge, find_challenge,
                        overkill_bonus, refusal_barred, wounds_remaining)
from battlescribe import has_quick_shot
from special_rules import (board_edge_distance, can_stand_and_shoot, charge_roll,
                           max_charge_range, max_pursuit_range,
                           should_use_swiftstride, unit_has_swiftstride)
from psychology import (MAX_RANK_BONUS, battle_standard_bonus, break_test_outcome,
                       massed_infantry_bonus, obb_distance, overwhelmed,
                       rank_bonus, should_reroll_break, should_use_stubborn,
                       side_unit_strength, stubborn_available,
                       unit_strength_total)
from psychology import shieldwall_unavailable_reason, reroll_leadership, veteran_reroll_allowed
from post_combat import (GIVE_GROUND, detour_angles, facing_vector,
                         fall_back_roll, fire_and_flee_roll, flee_direction,
                         flee_roll, flees_from,
                         give_ground_direction, nearest_corner, peril_wounds,
                         pursuit_roll, restraint_test, segment_crosses_box,
                         turn_direction, winner_response)
from rules_log import rule_log, rule_skipped, battle_log
from scouts import scout_charge_blocked
from vanguard import in_vanguard, vanguard_charge_blocked

# The Swiftstride die is thrown in its own colour so it is never mistaken for
# one of the dice a Charge or Fall Back roll discards between.
SWIFTSTRIDE_DIE_COLOR = (0.85, 0.05, 0.05, 1)

# A unit that Broke or Fell Back must end up this far from the enemy (p. 154).
ONE_INCH_APART = 1.0

# Bases this close are still touching, for deciding whether a Give Ground
# actually broke contact (p. 155).
CONTACT_GAP = 0.05

# A move halted by a sweep stops this fraction short of what it struck, so it
# does not come to rest touching it and re-contact on the next test.
CRASH_MARGIN = 0.05


class CombatResolver:
    """Encapsulates all combat resolution logic for the game."""

    def __init__(self, game):
        self.game = game
        # Dice state used during charge/flee intervals
        self.terningerCharge = []
        self.terningerFlee = []

    # ─── Contact Detection ────────────────────────────────────────────────

    def checkUnitContactSmall(self, unit):
        # Deployment and move commits teleport the body before asking about
        # contact. Bullet's world query still uses the last physics frame's
        # broadphase bounds; pair tests use the new transforms immediately.
        for other in self.game.units:
            if (other is unit or other.bodyNP.isEmpty()
                    or getattr(other, 'hostUnit', None) is not None):
                continue
            contacts = self.game.world.contactTestPair(unit.bodyNP.node(), other.bodyNP.node())
            for contact in contacts.getContacts():
                return contact
        return None

    def removeUnitFromPlay(self, unit):
        """Take a run-down unit off the board, node and bookkeeping alike."""
        from spell_effects import caster_removed
        caster_removed(self.game, unit)
        self.game.world.removeRigidBody(unit.bodyNP.node())
        unit.model.removeNode()
        unit.bodyNP.removeNode()
        if unit in self.game.units:
            self.game.units.remove(unit)
        if unit in self.game.player1Units:
            self.game.player1Units.remove(unit)
        if unit in self.game.player2Units:
            self.game.player2Units.remove(unit)

    # ─── Charge & Charge Reaction ─────────────────────────────────────────

    def counterChargeOption(self, defender, charger, fromPos, fromHpr):
        """Measure the declared charge, not its tentative contact (p. 167)."""
        from counter_charge import has_counter_charge, unavailable_reason
        from magic_items import current_turn
        from skirmish_charge import formed_contact
        if not has_counter_charge(defender):
            return None
        target = self.game.psychology._unit_box(defender)
        source = self.game.psychology._unit_box(charger)
        source = (fromPos.x, fromPos.y, source[2], source[3], fromHpr.x)
        flank = formed_contact([source], [target], fromPos)[-1]
        distance = obb_distance(source, target)
        profile = charger.unit.model
        movement = profile.get_fly_movement(4) if profile.is_flying() else profile.get_movement(4)
        reason = unavailable_reason(
            defender, charger, distance=distance, movement=movement, flank=flank,
            turn=current_turn(self.game),
            declared=not self.game.autoHold and charger.state != 'IsPursuing')
        if reason is None and (getattr(defender, 'isSkirmisher', False)
                               or getattr(charger, 'isSkirmisher', False)):
            reason = 'loose-formation Counter Charge geometry is not implemented (LEFTOVER)'
        if reason:
            rule_skipped('Counter Charge', defender, reason)
            return None
        return SimpleNamespace(distance=distance, movement=movement)

    async def counterChargeInterval(self, charger, defender, origin, facing, *, defer_charge=False):
        """Resolve a formed single-charge reaction (p. 167; D3+1 is not a Charge roll)."""
        from counter_charge import counter_charge_distance
        from first_charge import begin_charge_attempt, finish_charge_attempt
        from formed_skirmish_charge import route_to_model
        from magic_items import current_turn
        from psychology import _box_corners
        from scouts import BOARD_HALF_DEPTH, BOARD_HALF_WIDTH

        defender.counterChargeTurn = current_turn(self.game)
        begin_charge_attempt(defender)
        begin_charge_attempt(charger)
        defender.isChargingMove = charger.isChargingMove = True
        charger.bodyNP.setPos(origin)
        charger.bodyNP.setHpr(facing)
        charger.marchedThisTurn = charger.wouldMarch = False
        dice_models = []
        try:
            from drilled import before_move
            await before_move(self.game, defender, 'Counter Charge')
            dice_models, rolls = await self.rullTerninger(1)
            distance = counter_charge_distance(rolls[0])
            for die in dice_models:
                die.remove(self.game.world)
            dice_models = []
            start = Vec3(defender.bodyNP.getPos())
            direction = origin - start
            direction.z = 0
            direction.normalize()
            heading = math.degrees(math.atan2(-direction.x, direction.y))
            previous_heading = defender.bodyNP.getH()
            turn = (heading - previous_heading + 180) % 360 - 180
            target_box = self.game.psychology._unit_box(defender)
            pivoted_box = (*target_box[:4], heading)
            others = [member for member in self.game.units
                      if member is not defender and not member.bodyNP.isEmpty()
                      and getattr(member, 'hostUnit', None) is None]
            blocked = any(obb_distance(pivoted_box, self.game.psychology._unit_box(member)) <= 0
                          for member in others)
            blocked = blocked or any(abs(corner[0]) > BOARD_HALF_WIDTH or abs(corner[1]) > BOARD_HALF_DEPTH
                                     for corner in _box_corners(*pivoted_box))
            fraction, _, _ = self.game.movement.sweepTestRot(
                defender, Vec2(start.x, start.y), turn, mask=CM.TERRAIN_IMPASSABLE, pass_over=False)
            moved = 0.0
            if not blocked and fraction >= 1:
                await LerpPosHprInterval(defender.bodyNP, duration=.3, pos=start,
                                         hpr=Vec3(previous_heading + turn, 0, 0))
                transform = TransformState.makePosHpr(start, defender.bodyNP.getHpr())
                fraction, _ = self.game.movement.sweepTestDir(
                    defender, transform, direction, distance, pass_over=False)
                moved = max(0, distance * fraction - (CRASH_MARGIN if fraction < 1 else 0))
                end = start + direction * moved
                await LerpPosInterval(defender.bodyNP, duration=.4, pos=end)
                self.game.movement.dangerousTerrainTests(defender, start, end)
            else:
                rule_skipped('Counter Charge', defender,
                             'centre pivot is blocked; reaction spent with no advance')
            defender.chargedThisTurn = True
            defender.chargeDistance = moved
            rule_log('Counter Charge', defender,
                     f'D6 {rolls[0]} -> D3+1 = {distance}"; advanced {moved:.2f}" '
                     f'towards {charger.unit.name}; both count as charging, no Swiftstride bonus (p. 167)')
            if defender.bodyNP.isEmpty() or defender.unit.nmodels <= 0:
                charger.request('Moved')
                return
            if defer_charge:
                return

            source = self.game.psychology._unit_box(charger)
            target = self.game.psychology._unit_box(defender)
            obstacles = [self.game.psychology._unit_box(member) for member in self.game.units
                         if member not in (charger, defender) and not member.bodyNP.isEmpty()
                         and getattr(member, 'hostUnit', None) is None]
            route = route_to_model([source], [target], 0, origin, obstacles)
            if route is None:
                rule_skipped('Counter Charge', charger,
                             'no supported charge route after the defender moved; charge fails (LEFTOVER)')
                charger.request('Moved')
                return
            self.game.moveArceDistance = route.distance
            self.game.playerNP.setPos(*route.destination)
            bonus = await self.swiftstrideChargeChoice(charger)
            dice_models, rolls = await self.rullTerninger(3 if bonus else 2, bonus)
            allowance = self.chargeDistance(charger, origin, rolls)
            travel = min(route.distance, allowance)
            if allowance < route.distance:
                travel = min(route.distance, charge_roll(rolls, self.chargeThroughDifficult(charger, origin)))
            reached_distance = 0.0
            for stop in (route.lead, route.lead + route.wheel_distance, route.distance):
                stop = min(stop, travel)
                if stop <= reached_distance:
                    continue
                position, end_heading = route.pose(stop)
                start_position = Vec3(charger.bodyNP.getPos())
                angle = end_heading - charger.bodyNP.getH()
                if abs(angle) > 1e-6:
                    fraction, _, _ = self.game.movement.sweepTestRot(
                        charger, Vec2(*route.pivot), angle, pass_over=False)
                else:
                    advance = Vec3(*position) - start_position
                    length = advance.length()
                    advance.normalize()
                    fraction, _ = self.game.movement.sweepTestDir(
                        charger, TransformState.makePosHpr(start_position, charger.bodyNP.getHpr()),
                        advance, length, pass_over=False)
                endpoint = reached_distance + (stop - reached_distance) * fraction

                def place(distance):
                    pose, rotation = route.pose(distance)
                    charger.bodyNP.setPos(*pose)
                    charger.bodyNP.setH(rotation)

                await Parallel(LerpFunc(place, fromData=reached_distance, toData=endpoint, duration=.4))
                reached_distance = endpoint
                if fraction < 1:
                    break
            self.game.movement.dangerousTerrainTests(charger, origin, charger.bodyNP.getPos())
            if charger.bodyNP.isEmpty() or charger.unit.nmodels <= 0:
                return
            if reached_distance + CONTACT_GAP < route.distance:
                rule_log('Failed Charge', charger,
                         f'after Counter Charge, required {route.distance:.2f}", '
                         f'rolled range {allowance:g}"; moved {reached_distance:.2f}" without contact')
                charger.request('Moved')
                return
            angle = (defender.bodyNP.getH() + 180 - charger.bodyNP.getH() + 180) % 360 - 180
            await self.alignToEnemy(charger, angle, pivot=self.contactPointOn(charger, defender.bodyNP))
            for member, enemy in ((charger, defender), (defender, charger)):
                member.request('InCombat')
                member.isInCombat = True
                member.chargedThisTurn = member.wasChargedThisTurn = True
                member.isInCombatWith = [enemy]
                member.isInCombatFlank = ['front']
                member.isChargingMove = False
                finish_charge_attempt(member, enemy)
                member.updateTextNode()
            charger.chargeDistance = reached_distance
            rule_log('Counter Charge', defender,
                     f'contact with {charger.unit.name}: charge distances '
                     f'{moved:.2f}" / {reached_distance:.2f}"; both receive charging benefits')
        finally:
            for die in dice_models:
                die.remove(self.game.world)
            if not defer_charge:
                finish_charge_attempt(charger)
                finish_charge_attempt(defender)
            defender.isChargingMove = charger.isChargingMove = False
            self.game.autoCharge = self.game.autoHold = False

    def standAndShootOption(self, defender, charger, fromPos=None, fromHpr=None):
        """Whether *defender* may Stand & Shoot at *charger* (p. 120).

        Measured from where the charge was *declared*, which `fromPos` and
        `fromHpr` carry: by the time this runs the charger has already been
        swept into contact, so its own transform reads 0" away and would refuse
        every reaction in the game. Line of sight is taken from there too —
        "when the charge reaction is declared" (Official FAQ).

        Returns the missile weapon it would fire, or None with the reason
        logged. Every refusal looks the same on the board, so each one says
        which condition it failed.
        """
        if defender is None or charger is None:
            return None
        weapon = defender.unit.model.missile_weapon()
        if not weapon:
            return None   # not armed with missile weapons: not a refusal
        if defender.state == "IsFleeing":
            rule_skipped('Stand & Shoot', defender,
                         "fleeing units cannot Stand & Shoot")
            return None
        if defender.state == "InCombat":
            rule_skipped('Stand & Shoot', defender,
                         "already engaged in combat when charged")
            return None
        chargerPos = charger.bodyNP.getPos() if fromPos is None else fromPos
        from shooting_geometry import shooting_solution, uses_individual_shooting
        target_boxes = None
        if uses_individual_shooting(self.game, defender, charger):
            from formed_skirmish_charge import starting_boxes
            target_boxes = starting_boxes(charger, fromPos, fromHpr)
            shooting = shooting_solution(self.game, defender, charger, weapon=weapon,
                                         stand_and_shoot=True, target_boxes=target_boxes)
            if not shooting.eligible:
                rule_skipped('Stand & Shoot', defender, f'{shooting.detail()} at declaration (pp. 120, 137)')
                return None
        else:
            blocked = self.game.terrain_manager.los_block_point(
                defender.bodyNP.getPos(), chargerPos)
            if blocked is not None:
                rule_skipped('Stand & Shoot', defender,
                             f"no line of sight to {charger.unit.name}")
                return None
        psy = self.game.psychology
        chargerBox = psy._unit_box(charger)
        if fromPos is not None:
            heading = chargerBox[4] if fromHpr is None else fromHpr.x
            chargerBox = (chargerPos.x, chargerPos.y,
                          chargerBox[2], chargerBox[3], heading)
        distance = obb_distance(psy._unit_box(defender), chargerBox)
        if target_boxes is not None:
            from scouts import model_base_boxes
            distance = min(obb_distance(source, target) for source in model_base_boxes(defender)
                           for target in target_boxes)
        movement = charger.unit.model.get_movement(4)
        quick = bool(weapon.get('quick_shot')
                     or has_quick_shot(weapon.get('special_rules')))
        if not can_stand_and_shoot(distance, movement, quick):
            rule_skipped('Stand & Shoot', defender,
                         f"{charger.unit.name} is {distance:.1f}\" away, inside "
                         f"its own Movement of {movement}\" -> no time to raise "
                         f"weapons")
            return None
        if quick and distance < movement:
            rule_log('Quick Shot', defender,
                     f"{weapon.get('name', 'weapon')}: {charger.unit.name} is "
                     f"{distance:.1f}\" away, inside its Movement of {movement}\", "
                     f"and may still be Stood & Shot at (p. 175)")
        return SimpleNamespace(weapon=weapon, distance=distance,
                               movement=movement, quick=quick, target_boxes=target_boxes)

    def fireAndFleeOption(self, defender, charger, opt):
        """Whether *defender* may Fire & Flee (p. 169), given its shooting option.

        The rule restates the distance gate, but the reaction *is* a Stand &
        Shoot followed by a flee, so Quick Shot's exemption from that distance
        carries: whatever may be Stood & Shot at may be Fired & Fled from. The
        gate is therefore only ever applied in one place.
        """
        return bool(opt) and defender.unit.model.has_fire_and_flee()

    async def standAndShoot(self, defender, charger, weapon, distance=None, *, target_boxes=None):
        """Fire the charged unit's missile weapon at the charging unit (p. 120).

        The weapon is equipped for the shot: a unit expecting a fight may have
        drawn its sword, and `shootAt` fires whatever is in hand.
        """
        previous = (defender.unit.model.equipedWeapon or {}).get('name')
        slot = defender.unit.model.weapon_slot(weapon.get('name', ''))
        if slot is not None:
            defender.unit.model.equip_weapon(slot)
        rule_log('Stand & Shoot', defender,
                 f"reacts to {charger.unit.name}'s charge with "
                 f"{weapon.get('name', 'its missile weapon')} at -1 To Hit, "
                 f"and no long range modifier (p. 139)")
        await self.game.shootAt(defender, charger, stand_and_shoot=True,
                                distance=distance, **({'target_boxes': target_boxes} if target_boxes is not None else {}))
        # equip_weapon also rewrites special_rules, so put the melee weapon
        # back through it rather than by assignment.
        if slot is not None and previous:
            defender.unit.model.equip_weapon(previous)

    async def resolveDeclaredCharge(self, declaration):
        """Resolve one reserved charge without asking for a second declaration (p. 121)."""
        from first_charge import finish_charge_attempt
        from formed_skirmish_charge import route_to_model, starting_boxes
        from skirmish_charge import formed_contact

        unit, defender = declaration.charger, declaration.defender
        if (unit.bodyNP.isEmpty() or defender.bodyNP.isEmpty()
                or unit.unit.nmodels <= 0 or defender.unit.nmodels <= 0):
            finish_charge_attempt(unit)
            return
        origin, facing = Vec3(*declaration.origin), Vec3(*declaration.facing)
        self.game.unitToMove = unit
        unit.bodyNP.setPos(origin)
        unit.bodyNP.setHpr(facing)
        unit.isChargingMove = True
        from drilled import before_move, has_drilled, marching_column
        if has_drilled(unit) or marching_column(unit):
            self.game.playerNP.setPos(*declaration.destination)
            self.game.moveArceDistance = declaration.distance
            if self.game.autoRoll:
                declaration.charge_dice = [6, 6]
            else:
                bonus = await self.swiftstrideChargeChoice(unit)
                dice_models, declaration.charge_dice = await self.rullTerninger(3 if bonus else 2, bonus)
                for die in dice_models:
                    die.remove(self.game.world)
            self.chargeDistance(unit, origin, declaration.charge_dice)
            await before_move(self.game, unit, 'charge move', compulsory=declaration.compulsory)
            origin = Vec3(unit.bodyNP.getPos())
            if marching_column(unit):
                rule_log('Marching Column', unit,
                         f'{unit.unit.files} files / {unit.unit.ranks} ranks: cannot make a charge move; '
                         'charge fails without moving (p. 101; FAQ v1.5.3)')
                finish_charge_attempt(unit)
                unit.isChargingMove = False
                unit.request('Moved')
                return
        if not getattr(unit, 'isSkirmisher', False) and not getattr(defender, 'isSkirmisher', False):
            source = self.game.psychology._unit_box(unit)
            target = self.game.psychology._unit_box(defender)
            obstacles = [self.game.psychology._unit_box(member) for member in self.game.units
                         if member not in (unit, defender) and not member.bodyNP.isEmpty()
                         and getattr(member, 'hostUnit', None) is None]
            route = route_to_model([source], [target], 0, origin, obstacles)
            if route is None:
                rule_skipped('Charge Move', unit, 'declared target has no supported clear route; charge spent (LEFTOVER)')
                finish_charge_attempt(unit)
                unit.isChargingMove = False
                unit.request('Moved')
                return
            unit.bodyNP.setPos(*route.destination)
            unit.bodyNP.setH(route.heading + route.wheel)
            self.game.playerNP.setPos(*route.destination)
            self.game.moveArceDistance = route.distance
            contact = formed_contact([source], [target], origin)
            direction = contact[4]
            heading = math.degrees(math.atan2(-direction[0], direction[1]))
            angle = (heading - unit.bodyNP.getH() + 180) % 360 - 180
            declaration.flank_angle = (contact[-1], angle)
            declaration.route = route
        elif declaration.target_index is not None:
            from scouts import model_base_boxes
            obstacles = [box for member in self.game.units if member not in (unit, defender)
                         and not member.bodyNP.isEmpty() and member.isDeployed
                         and getattr(member, 'hostUnit', None) is None
                         for box in model_base_boxes(member)]
            obstacles.extend((piece.center.x, piece.center.y, piece.width / 2, piece.height / 2, 0)
                             for piece in self.game.terrain_manager.terrain_pieces if piece.is_impassable)
            targets = model_base_boxes(defender)
            route = None
            if (not defender.skirmishCombat and defender.state != 'IsFleeing'
                    and declaration.target_index < len(targets)):
                route = route_to_model(starting_boxes(unit, origin, facing), targets,
                                       declaration.target_index, origin, obstacles)
            if route is None:
                rule_skipped('Skirmishers', unit,
                             'reserved loose target has no supported route after reactions or form-up; charge spent (LEFTOVER)')
                finish_charge_attempt(unit)
                unit.isChargingMove = False
                unit.request('Moved')
                return
            declaration.preview = SimpleNamespace(target=defender, route=route, error=None)
            self.game.playerNP.setPos(*route.destination)
            self.game.moveArceDistance = route.distance
        else:
            unit.bodyNP.setPos(*declaration.contact_position)
            unit.bodyNP.setHpr(*declaration.contact_facing)
            self.game.playerNP.setPos(*declaration.destination)
            self.game.moveArceDistance = declaration.distance
            declaration.flank_angle = ('front', 0)
        try:
            unit.declaredCharge = declaration
            await self.chargeAndChargeReaction(unit, None, origin, facing,
                                              SimpleNamespace(done=None), defender=defender,
                                              declaration=declaration)
        finally:
            finish_charge_attempt(unit)
            unit.isChargingMove = False
            unit.declaredCharge = unit.formedSkirmishCharge = None

    async def chargeAndChargeReaction(self, unit, c, oposUnit, orotUnit, task, defender=None,
                                     declaration=None):
        pregame_rule = ('Scouts' if scout_charge_blocked(self.game, unit) else
                        'Vanguard' if in_vanguard(self.game) or vanguard_charge_blocked(self.game, unit) else None)
        if unit.state != 'IsPursuing' and pregame_rule:
            rule_log(pregame_rule, unit, 'charge declaration refused before reactions or dice: pre-game move or first own turn')
            unit.bodyNP.setPos(oposUnit)
            unit.bodyNP.setHpr(orotUnit)
            unit.bodyNP.node().setTransformDirty()
            unit.isChargingMove = False
            self.game.autoCharge = False
            self.game.autoHold = False
            self.game.startTaskFunction(self.game.taskLoopPathTowardsMouse, 'taskLoopPathTowardsMouse')
            return task.done
        chargeYesNo = ["Yes", "No"]
        # Declaring the charge is the charger's call; reacting to it is the
        # defender's, and they need not belong to the same player.
        if defender is None:
            defenderNP = render.find(f"**/{c.getNode1().getName()}")
            defender = self.game.getSelectedUnit(defenderNP.node())
        else:
            defenderNP = defender.bodyNP
        if (declaration is None and unit.state != 'IsPursuing'
                and self.game.fsm.state == 'MovementPhase'
                and getattr(self.game, 'chargeStage', None) in ('resolving', 'remaining', 'blocked')):
            rule_skipped('Charge Declaration', unit, 'declarations are closed; no new charge is allowed (p. 119)')
            unit.bodyNP.setPos(oposUnit)
            unit.bodyNP.setHpr(orotUnit)
            unit.bodyNP.node().setTransformDirty()
            unit.isChargingMove = unit.wouldMarch = False
            self.game.autoCharge = self.game.autoHold = False
            return task.done
        from formed_skirmish_charge import preview_charge
        from skirmish_charge import supported_skirmish_defender
        formed_preview = declaration.preview if declaration is not None else None
        unit.formedSkirmishCharge = formed_preview
        if declaration is None and (supported_skirmish_defender(unit, defender) or c is None):
            formed_preview = preview_charge(self.game, unit, defender, oposUnit, orotUnit)
            if formed_preview.error:
                rule_skipped('Skirmishers', unit,
                             f'charge refused before reactions or dice: {formed_preview.error}; movement retained (p. 186)')
                unit.bodyNP.setPos(oposUnit)
                unit.bodyNP.setHpr(orotUnit)
                unit.bodyNP.node().setTransformDirty()
                unit.isChargingMove = unit.wouldMarch = False
                self.game.autoCharge = self.game.autoHold = False
                self.game.startTaskFunction(self.game.taskLoopPathTowardsMouse, 'taskLoopPathTowardsMouse')
                return task.done
            self.game.moveArceDistance = formed_preview.route.distance
            self.game.playerNP.setPos(*formed_preview.route.destination)
            unit.formedSkirmishCharge = formed_preview
        visibility = None
        if (getattr(unit, 'isSkirmisher', False) and not getattr(unit, 'skirmishCombat', False)
            and unit.state != 'IsPursuing' and declaration is None):
            from skirmish_visibility import charge_visibility
            visibility = charge_visibility(self.game, unit, defender, oposUnit, orotUnit)
            if not visibility.allowed:
                rule_skipped('Skirmishers', unit,
                             f'charge refused before reactions or dice: {visibility.detail(defender.unitName)} (p. 186)')
                unit.bodyNP.setPos(oposUnit)
                unit.bodyNP.setHpr(orotUnit)
                unit.bodyNP.node().setTransformDirty()
                unit.isChargingMove = False
                unit.wouldMarch = False
                self.game.autoCharge = False
                self.game.autoHold = False
                self.game.startTaskFunction(self.game.taskLoopPathTowardsMouse, 'taskLoopPathTowardsMouse')
                return task.done
        if declaration is not None or self.game.autoCharge or self.game.aiControls(unit):
            cynchoice = "Yes"
        else:
            cynchoice = await taskMgr.add(self.game.makeChoiceNew(
                chargeYesNo, Vec3(-20, 0, 10), owner=unit,
                prompt=f"{unit.unit.name}: charge {defender.unit.name}?",
                detail=self.chargeRangeText(unit, unit.unit.model.get_movement(4))
                       + (f'\n{visibility.detail(defender.unitName)}' if visibility is not None else '')))

        if cynchoice == "Yes":
            if unit.state != 'IsPursuing':
                from first_charge import begin_charge_attempt
                begin_charge_attempt(unit)
            if formed_preview is not None:
                route = formed_preview.route
                rule_log('Skirmishers', unit,
                         f'closest visible model {route.target_index + 1} of {defender.unitName}: '
                         f'wheel {abs(route.wheel):.1f} degrees costs {route.wheel_distance:.2f}", '
                         f'approach {route.lead + route.advance:.2f}" -> {route.distance:.2f}" charge declared (p. 186)')
            if visibility is not None:
                rule_log('Skirmishers', unit,
                         f'{visibility.detail(defender.unitName)} -> charge declared (p. 186)')
            from charge_declarations import collecting, queue_charge
            if declaration is None and unit.state != 'IsPursuing' and collecting(self.game):
                queue_charge(self.game, unit, defender, oposUnit, orotUnit)
                rule_log('Charge Declaration', unit,
                         f'targets {defender.unit.name}; {len(self.game.chargeDeclarations)} charge(s) queued, no charge moves yet (p. 119)')
                unit.updateTextNode()
                messenger.send('unit-move-complete')
                return task.done
            print("Charging into combat...")

            chargeReaction = ["hold", "flee"]
            counterOption = self.counterChargeOption(defender, unit, oposUnit, orotUnit) if declaration is None else None
            if counterOption:
                chargeReaction.insert(0, 'counter charge')
            shootOption = self.standAndShootOption(defender, unit, oposUnit, orotUnit) if declaration is None else None
            fireFlee = self.fireAndFleeOption(defender, unit, shootOption)
            if fireFlee:
                chargeReaction.insert(0, "fire & flee")
            if shootOption:
                chargeReaction.insert(0, "stand & shoot")
            fireAndFlee = False
            from magic_items import current_turn
            counterTurn = getattr(defender, 'counterChargeTurn', None)
            counterSpent = counterTurn is not None and counterTurn == current_turn(self.game)
            if declaration is not None:
                crchoice = declaration.reaction
            elif self.game.autoHold or counterSpent:
                # A pursuit was never declared as a charge, so the unit it
                # reaches gets no reaction to it (p. 157).
                crchoice = "hold"
                if shootOption and self.game.autoHold:
                    rule_skipped('Stand & Shoot', defender,
                                 f"reached by {unit.unit.name}'s pursuit rather "
                                 f"than a declared charge — no reaction")
            elif self.game.aiControls(defender):
                # Shooting costs the unit nothing a hold would have kept: it
                # holds afterwards either way, keeps its Shooting phase, and
                # the charger tests for Panic in neither case. Fire & Flee is
                # a real trade — the volley for the fight — and the AI has no
                # policy to weigh it, so it shoots and stands.
                if counterOption:
                    crchoice = 'counter charge'
                else:
                    crchoice = "stand & shoot" if shootOption else "hold"
                if fireFlee:
                    rule_skipped('Fire & Flee', defender,
                                 "the AI stands its ground; it has no policy "
                                 "for trading the combat away")
            else:
                crchoice = await taskMgr.add(self.game.makeChoiceNew(
                    chargeReaction, Vec3(20, 0, 10), owner=defender,
                    prompt=f"{defender.unit.name}: charged by {unit.unit.name} "
                           f"— how does it react?",
                    detail=(f'Counter Charge: {counterOption.distance:.2f}" away, '
                            f'charger M{counterOption.movement:g}; pivot and move D3+1".'
                            if counterOption else '')))
            if crchoice == 'counter charge' and counterOption:
                rule_log('Counter Charge', defender,
                         f'accepts frontal charge from {unit.unit.name} at '
                         f'{counterOption.distance:.2f}" (charger M{counterOption.movement:g})')
                unit.hasMovedThisTurn = True
                unit.marchedThisTurn = unit.wouldMarch = False
                await self.counterChargeInterval(unit, defender, oposUnit, orotUnit)
                return task.done
            if counterOption:
                rule_skipped('Counter Charge', defender, f'chose {crchoice}; reaction remains unused')
            if crchoice == "fire & flee":
                await self.standAndShoot(defender, unit, shootOption.weapon,
                                         shootOption.distance, target_boxes=getattr(shootOption, 'target_boxes', None))
                rule_log('Fire & Flee', defender,
                         f"volley fired at {unit.unit.name}, now turning tail: "
                         f"the Flee roll discards its lowest die rather than "
                         f"summing both (p. 169)")
                crchoice = "flee"
                fireAndFlee = True
            elif crchoice == "stand & shoot":
                await self.standAndShoot(defender, unit, shootOption.weapon,
                                         shootOption.distance, target_boxes=getattr(shootOption, 'target_boxes', None))
                # "Once this shooting has been resolved, the charged unit will
                # Hold and await the charging unit" (p. 120).
                crchoice = "hold"
            if formed_preview is not None and (unit.unit.nmodels <= 0 or unit.bodyNP.isEmpty()):
                from first_charge import finish_charge_attempt
                finish_charge_attempt(unit)
                unit.formedSkirmishCharge = unit.formedSkirmishPreview = None
                unit.isChargingMove = False
                rule_skipped('Skirmishers', unit, 'charger destroyed by reaction; no charge roll or form-up (p. 120)')
                return task.done
            if crchoice == "hold":
                print("Defender holds position.")

                flank, angleToRotate = (declaration.flank_angle if declaration is not None and declaration.flank_angle is not None
                                       else ('front', 0) if formed_preview is not None else self.getFlankFromContact(unit, c))

                unit.hasMovedThisTurn = True
                unit.updateTextNode()
                if declaration is not None or unit.state == 'IsPursuing':
                    await self.chargeInterval(unit, defenderNP, angleToRotate, oposUnit, orotUnit, flank,
                                              chdice=declaration.charge_dice if declaration is not None else None)
                else:
                    taskMgr.add(self.chargeInterval, "chargeIntervalTask",
                                extraArgs=[unit, defenderNP, angleToRotate, oposUnit, orotUnit, flank],
                                appendTask=False)

            elif crchoice == "flee":
                flank, angleToRotate = (declaration.flank_angle if declaration is not None and declaration.flank_angle is not None
                                       else ('front', 0) if formed_preview is not None else self.getFlankFromContact(unit, c))
                if formed_preview is not None:
                    rule_skipped('Skirmishers', unit,
                                 'defender flees: legacy chase/redirect movement replaces the planned contact route; '
                                 'no defender form-up (LEFTOVER, p. 186)')
                    unit.bodyNP.setPos(*formed_preview.route.destination)
                    unit.bodyNP.setH(formed_preview.route.heading + formed_preview.route.wheel)
                    unit.formedSkirmishCharge = None
                print("Defender flees!")
                loserUnit = self.game.getSelectedUnit(defenderNP)
                loserUnit.request("IsFleeing")
                if declaration is not None:
                    await self.fleeInterval(unit, defenderNP, angleToRotate, oposUnit, orotUnit, fireAndFlee)
                    return task.done
                taskMgr.add(self.fleeInterval, "fleeIntervalTask",
                            extraArgs=[unit, defenderNP, angleToRotate, oposUnit, orotUnit, fireAndFlee],
                            appendTask=False)
                fleeDirection = defenderNP.getPos() - unit.bodyNP.getPos()
                storeRotation = defenderNP.getHpr()
                defenderNP.lookAt(defenderNP.getPos() + fleeDirection)
                fleeRotation = defenderNP.getHpr()
                defenderNP.setHpr(storeRotation)

        else:
            print("Charge cancelled.")
            unit.formedSkirmishCharge = None
            unit.bodyNP.setPos(oposUnit)
            unit.bodyNP.setHpr(orotUnit)
            unit.bodyNP.node().setTransformDirty()
            unit.isChargingMove = False
            unit.wouldMarch = False
            self.game.startTaskFunction(self.game.taskLoopPathTowardsMouse, "taskLoopPathTowardsMouse")
            self.game.autoCharge = False
            self.game.autoHold = False

        return task.done

    # ─── Flee Interval ────────────────────────────────────────────────────

    async def fleeInterval(self, unit, defenderNP, angleToRotate, oposUnit, orotUnit,
                           fireAndFlee=False):
        """A fleeing target still consumes the declared charge attempt (p. 169)."""
        from first_charge import begin_charge_attempt, finish_charge_attempt
        begin_charge_attempt(unit)
        try:
            await self._resolveFleeInterval(unit, defenderNP, angleToRotate, oposUnit, orotUnit, fireAndFlee)
        finally:
            finish_charge_attempt(unit)

    async def _resolveFleeInterval(self, unit, defenderNP, angleToRotate, oposUnit, orotUnit,
                                  fireAndFlee=False):
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

        chdist = self.chargeDistance(unit, oposUnit, chdice)
        # The bonus die is already among *fldice*; adding `fleeBonus` as well
        # added a literal 1" for taking it. Both flee rolls live in one place
        # now, which is also where The Limits of Endurance is applied.
        spent = getattr(fleeingUnit, 'fledThisPhase', False)
        fldist = (fire_and_flee_roll(fldice, spent) if fireAndFlee
                  else flee_roll(fldice, spent))
        fleeingUnit.fledThisPhase = True
        pair, bonus = fldice[:2], sum(fldice[2:])
        swift = f" +{bonus} Swiftstride" if bonus else ""
        if spent:
            rule_log('The Limits of Endurance', fleeingUnit,
                     "already fled this phase -> this flee covers 0\" (p. 133)")
        elif fireAndFlee:
            rule_log('Fire & Flee', fleeingUnit,
                     f"flee roll {pair} keeps the {max(pair)} and discards the "
                     f"{min(pair)}{swift} -> flees {fldist}\", where an ordinary "
                     f"Flee would have summed them for {sum(fldice)}\" (p. 169)")
        else:
            battle_log(f"{fleeingUnit.unit.name} flees {fldist}\" "
                       f"(2D6 {pair} summed{swift})")
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
        if self.game.aiControls(unit):
            use = should_use_swiftstride(kind, distance_to_edge)
        else:
            choice = ['Swiftstride\n+D6', 'No bonus']
            selected = await taskMgr.add(
                self.game.makeChoiceNew(
                    choice, Vec3(0, 0, 10), owner=unit,
                    prompt=f"{unit.unit.name}: take the Swiftstride bonus "
                           f"to its {kind}?"))
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
        needed = math.ceil(self.game.moveArceDistance - 0.001)
        if unit.state == "IsPursuing":
            return (f"Roll needed: {needed:.0f}"
                    f"   (max pursuit {max_pursuit_range(swiftstride)}\")")
        needed = max(0, math.ceil(self.game.moveArceDistance - maxmove - 0.001))
        return (f"Roll needed: {needed:.0f}"
            f"   (max charge {max_charge_range(maxmove, swiftstride):g}\")")

    def chargeThroughDifficult(self, unit, from_pos) -> bool:
        """True if the charge path meets terrain that hinders movement, which
        makes the Charge roll discard the highest die instead of the lowest
        (Rulebook p. 269)."""
        tm = getattr(self.game, 'terrain_manager', None)
        if tm is None:
            return False
        profiles = [participant.unit.model
                    for participant in self.game.movement.movementParticipants(unit)]
        from special_rules import is_ethereal
        if all(profile.is_flying() for profile in profiles) or all(is_ethereal(profile) for profile in profiles):
            return False
        planned = getattr(unit, 'formedSkirmishCharge', None)
        from tempest import tempest_features
        if planned is not None:
            from formed_skirmish_charge import route_features
            features = tempest_features(self.game, unit, planned.route.origin, planned.route.destination,
                         route_features(self.game, planned.route))
        else:
            endpoint = self.game.playerNP.getPos()
            features = tempest_features(self.game, unit, from_pos, endpoint,
                         tm.get_terrain_between(from_pos, endpoint))
        return (any(piece.movement_modifier < 0 for piece in features)
            and not all(profile.is_move_through_cover() or is_ethereal(profile) for profile in profiles))

    def chargeDistance(self, unit, from_pos, dice):
        """Terrain-adjusted M plus Charge roll; Move Through Cover keeps the high
        die (Rulebook pp. 174, 269; Official FAQ v1.5.3). Pursuit is not a charge.
        """
        if unit.state == 'IsPursuing':
            return sum(dice)
        planned = getattr(unit, 'formedSkirmishCharge', None)
        if planned is not None:
            from formed_skirmish_charge import route_allowance, route_features
            movement = route_allowance(self.game, unit, planned.route)
            difficult = any(piece.movement_modifier < 0 for piece in route_features(self.game, planned.route))
        else:
            movement = self.game.movement.movementAllowance(
                unit, from_pos, self.game.playerNP.getPos(), log=True)
            tm = getattr(self.game, 'terrain_manager', None)
            difficult = tm is not None and tm.crosses_difficult(from_pos, self.game.playerNP.getPos())
        rough = self.chargeThroughDifficult(unit, from_pos)
        result = charge_roll(dice, rough)
        if planned is not None:
            rule_log('Charge Move', unit,
                     f'route M{movement:g}, dice {dice} keep {"lowest" if rough else "highest"} '
                     f'-> {movement + result:g}" range (pp. 121, 269)')
        profiles = [participant.unit.model
                    for participant in self.game.movement.movementParticipants(unit)]
        from special_rules import is_ethereal
        if difficult and all(is_ethereal(profile) for profile in profiles):
            rule_log('Ethereal', unit, f'charge treats terrain as open ground: dice {dice} '
                     f'keep highest -> {result}; M{movement:g} -> {movement + result:g}" (p. 167)')
        if (difficult
            and any(profile.is_move_through_cover() for profile in profiles)
            and not all(profile.is_flying() for profile in profiles)):
            if rough:
                rule_skipped('Move Through Cover', unit,
                             f'not every charging model has the rule; dice {dice} '
                             f'discard the highest -> {result}; M{movement:g} -> {movement + result:g}"')
            else:
                rule_log('Move Through Cover', unit,
                         f'charge through terrain: dice {dice} keep the highest '
                         f'of the first two -> {result}; M{movement:g} -> {movement + result:g}"')
        return movement + result

    # ─── Charge Interval ──────────────────────────────────────────────────

    async def chargeInterval(self, unit, defenderNP, angleToRotate, oposUnit, orotUnit, flank, chdice=None):
        """Track one attempt across all charge geometries (First Charge, p. 169)."""
        from first_charge import begin_charge_attempt, finish_charge_attempt
        if unit.state != 'IsPursuing':
            begin_charge_attempt(unit)
        try:
            await self._resolveChargeInterval(unit, defenderNP, angleToRotate, oposUnit, orotUnit, flank, chdice)
        finally:
            finish_charge_attempt(unit)

    async def _resolveChargeInterval(self, unit, defenderNP, angleToRotate, oposUnit, orotUnit, flank, chdice=None):
        declaration = getattr(unit, 'declaredCharge', None)
        if declaration is not None and declaration.route is not None:
            planned = SimpleNamespace(target=declaration.defender, route=declaration.route,
                                      formed_target=True, flank=flank, angle=angleToRotate)
            unit.formedSkirmishCharge = planned
            await self._formedSkirmishChargeInterval(unit, planned, oposUnit, orotUnit, chdice)
            return
        planned = getattr(unit, 'formedSkirmishCharge', None)
        if planned is not None and planned.target.bodyNP == defenderNP:
            await self._formedSkirmishChargeInterval(unit, planned, oposUnit, orotUnit, chdice)
            return
        # Skirmishers charge straight in — no wheel, no flank-align pivot — but
        # the charge roll is still made and must reach the target to connect.
        if getattr(unit, 'isSkirmisher', False):
            await self._skirmishChargeInterval(unit, defenderNP, oposUnit, orotUnit, flank, chdice)
            return
        maxmove = self.game.movement.movementAllowance(unit, oposUnit, self.game.playerNP.getPos())
        durIntConst = 1.0
        wasPursuing = unit.state == "IsPursuing"
        if wasPursuing:
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
        chdist = self.chargeDistance(unit, oposUnit, chdice)
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
                if wasPursuing:
                    rule_log('Pursuit into an Obstacle', unit,
                             f"halted against {defenderUnit.unit.name}, a "
                             f"friendly unit")
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
                if wasPursuing:
                    rule_log('Pursuit into an Obstacle', unit,
                             f"halted against {defenderUnit.unit.name}, a "
                             f"friendly unit")
                direction = unit.bodyNP.getPos() - defenderNP.getPos()
                direction.normalize()
                self.game.fallBackContactTest(unit.bodyNP, direction * .3)
                for terning in terninger:
                    terning.remove(self.game.world)
                del terninger
                unit.request("Moved")
                return

        if await self._formChargedSkirmishers(unit, defenderUnit):
            flank = 'front'
        else:
            await self.alignToEnemy(unit, angleToRotate, 0.5 * durIntConst)

        # A pursuer often reaches something other than the unit it set off
        # after, and the two cases read identically on the board.
        quarry = getattr(unit, 'pursuitQuarry', None)
        strayed = wasPursuing and quarry is not None and defenderUnit is not quarry
        escaped = (f", and {quarry.unit.name} is not caught" if strayed else "")

        if defenderUnit.state == "IsFleeing":
            print("Contact detected between fleeing unit and pursuer!")
            rule_log('Pursuit into a Fleeing Enemy' if strayed
                     else 'Catching the Curs!', unit,
                     f"caught the fleeing {defenderUnit.unit.name} and hacked "
                     f"it to pieces{escaped}")
            from first_charge import finish_charge_attempt
            finish_charge_attempt(unit, defenderUnit)
            from command_groups import capture_standard
            capture_standard(self.game, defenderUnit, unit)
            self.removeUnitFromPlay(defenderUnit)
            unit.request("Moved")
            for terning in terninger:
                terning.remove(self.game.world)
            await self.freeReform(unit)
            return

        unit.request("InCombat")
        unit.isInCombat = True
        unit.chargedThisTurn = True
        # Impact Hits need to know the charge covered 3" or more (p. 172).
        defenderUnit.wasChargedThisTurn = True
        unit.chargeDistance = float(self.game.moveArceDistance)
        from first_charge import finish_charge_attempt
        finish_charge_attempt(unit, defenderUnit)
        from magic_items import current_turn
        counter_turn = getattr(defenderUnit, 'counterChargeTurn', None)
        if counter_turn is not None and counter_turn == current_turn(self.game):
            unit.wasChargedThisTurn = True
            finish_charge_attempt(defenderUnit, unit)
            rule_log('Counter Charge', defenderUnit,
                     f'contact with {unit.unit.name}: charge distances '
                     f'{defenderUnit.chargeDistance:.2f}" / {unit.chargeDistance:.2f}"; both receive charging benefits')
        if wasPursuing:
            joins, whyNot = (self.joinsCombatThisPhase(defenderUnit) if strayed
                             else (False, ""))
            if joins:
                # Pursuit into a New Combat (p. 157): the enemy has not fought
                # yet, so the pursuer joins that combat and fights again in it
                # — chargedThisTurn already stands — but may not pursue out of
                # it, restraining and reforming for free instead.
                # Its attack for this phase is spent, and the attack loop skips
                # anyone carrying that, so the allowance has to be given back.
                unit.hasAttackedThisTurn = False
                unit.cannotPursueThisTurn = True
                rule_log('Pursuit into a New Combat', unit,
                         f"joined {defenderUnit.unit.name}'s unfought combat by "
                         f"its {flank}: fights again this phase counting as "
                         f"charged, and may not pursue again{escaped}")
            elif strayed:
                # Catching the Curs! and Pursuit into a New Combat (p. 157)
                # agree: locked together and fought *next* turn, which is when
                # the pursuer counts as having charged. chargedThisTurn is
                # cleared at the end of this phase, so the claim has to be
                # carried over separately.
                unit.countsAsChargedNextTurn = True
                defenderUnit.countsAsChargeTargetNextTurn = True
                rule_log('Pursuit into a Fresh Enemy', unit,
                         f"ran into {defenderUnit.unit.name}'s {flank} instead: "
                         f"locked together and fought next turn, counting as "
                         f"charged, because {whyNot}{escaped}")
            else:
                unit.countsAsChargedNextTurn = True
                defenderUnit.countsAsChargeTargetNextTurn = True
                rule_log('Catching the Curs!', unit,
                         f"caught {defenderUnit.unit.name}, which fell back: "
                         f"locked together, and counts as charging next turn")
            from first_charge import count_as_charge
            count_as_charge(unit, defenderUnit, next_turn=not joins)
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

    async def _formedSkirmishChargeInterval(self, unit, preview, origin, facing, chdice=None):
        """Follow a planned charge route; align only after contact (pp. 121, 126, 186)."""
        from direct.interval.IntervalGlobal import Parallel
        from direct.interval.LerpInterval import LerpFunc
        from formed_skirmish_charge import route_to_model, starting_boxes, route_allowance, route_features
        from scouts import model_base_boxes
        defender = preview.target
        route = preview.route
        self.game.diceInfoText.setText(self.chargeRangeText(
            unit, route_allowance(self.game, unit, route)))
        if chdice is not None:
            dice_models = []
        elif not self.game.autoRoll:
            bonus = await self.swiftstrideChargeChoice(unit)
            dice_models, chdice = await self.rullTerninger(3 if bonus else 2, bonus)
        else:
            dice_models = []
            chdice = [6, 6] if chdice is None else chdice
        self.game.autoCharge = self.game.autoHold = False
        formed_target = getattr(preview, 'formed_target', False)
        original = starting_boxes(unit, origin, facing)
        if not formed_target and original != route.original_boxes:
            obstacles = [box for other in self.game.units if other not in (unit, defender)
                         and other.isDeployed and getattr(other, 'hostUnit', None) is None
                         for box in model_base_boxes(other)]
            obstacles.extend((piece.center.x, piece.center.y, piece.width / 2, piece.height / 2, 0)
                             for piece in self.game.terrain_manager.terrain_pieces if piece.is_impassable)
            route = route_to_model(original, model_base_boxes(defender), route.target_index, origin, obstacles)
        if route is not None:
            preview.route = route
            self.game.moveArceDistance = route.distance
            self.game.playerNP.setPos(*route.destination)
        distance = self.chargeDistance(unit, origin, chdice)
        reached = route is not None and distance + 1e-5 >= route.distance
        if route is None:
            route = preview.route
        travel = route.distance if reached else min(route.distance, charge_roll(chdice, self.chargeThroughDifficult(unit, origin)))
        unit.bodyNP.setPos(origin)
        unit.bodyNP.setHpr(facing)

        def position_at(amount):
            position, heading = route.pose(amount)
            unit.bodyNP.setPos(*position)
            unit.bodyNP.setHpr(heading, facing.y, facing.z)
            unit.bodyNP.node().setTransformDirty()

        await Parallel(LerpFunc(position_at, fromData=0, toData=travel, duration=0.7, blendType='easeInOut'))
        for die in dice_models:
            die.remove(self.game.world)
        unit.formedSkirmishCharge = unit.formedSkirmishPreview = None
        self.game.diceInfoText.setText('')
        self.game.debugTextInfo.setText('')
        self.game.movement.dangerousTerrainTests(unit, origin, unit.bodyNP.getPos(),
                               features=route_features(self.game, route, travel))
        unit.isChargingMove = False
        if unit.unit.nmodels <= 0 or unit.bodyNP.isEmpty():
            return
        if not reached:
            rule_log('Failed Charge', unit,
                     f'route {route.distance:.2f}", rolled range {distance:g}"; dice {chdice} '
                     f'-> moves {travel:.2f}" without adding M (p. 121)')
            unit.request('Moved')
            return
        if formed_target:
            await self.alignToEnemy(unit, preview.angle, pivot=self.contactPointOn(unit, defender.bodyNP))
            if defender.state == 'IsFleeing':
                from first_charge import finish_charge_attempt
                from command_groups import capture_standard
                rule_log('Catching the Curs!', unit, f'caught the fleeing {defender.unit.name} (p. 121)')
                finish_charge_attempt(unit, defender)
                capture_standard(self.game, defender, unit)
                self.removeUnitFromPlay(defender)
                unit.request('Moved')
                await self.freeReform(unit)
                return
        elif not await self._formChargedSkirmishers(unit, defender):
            rule_skipped('Skirmishers', unit, 'contact reached but defender cannot form; charge not engaged (p. 186)')
            unit.request('Moved')
            return
        for participant, opponent in ((unit, defender), (defender, unit)):
            if participant.state != 'InCombat':
                participant.request('InCombat')
            participant.isInCombat = True
            participant.isInCombatWith.append(opponent)
            participant.isInCombatFlank.append(preview.flank if formed_target and participant is defender else 'front')
            participant.updateTextNode()
        unit.chargedThisTurn = True
        unit.chargeDistance = route.distance
        defender.wasChargedThisTurn = True
        from first_charge import finish_charge_attempt
        finish_charge_attempt(unit, defender)
        from magic_items import current_turn
        counter_turn = getattr(defender, 'counterChargeTurn', None)
        if counter_turn is not None and counter_turn == current_turn(self.game):
            unit.wasChargedThisTurn = True
            finish_charge_attempt(defender, unit)
            rule_log('Counter Charge', defender,
                     f'contact with {unit.unit.name}: charge distances '
                     f'{defender.chargeDistance:.2f}" / {unit.chargeDistance:.2f}"; both receive charging benefits')

    async def _formChargedSkirmishers(self, attacker, defender):
        """Loose defenders align to the formed charger, not vice versa (p. 186)."""
        from direct.interval.IntervalGlobal import Parallel
        from scouts import model_base_boxes
        from skirmish_charge import apply_fighting_rank, plan_skirmish_defence, supported_skirmish_defender
        if not supported_skirmish_defender(attacker, defender):
            if getattr(defender, 'isSkirmisher', False) and not defender.skirmishCombat:
                rule_skipped('Skirmishers', defender,
                             f'defender-only form-up unsupported: charger state {attacker.state}, '
                             f'defender state {defender.state}, joined models '
                             f'{int(getattr(attacker, "joinedCharacter", None) is not None) + int(getattr(defender, "joinedCharacter", None) is not None)}; '
                             'using legacy alignment (p. 186 LEFTOVER)')
            return False
        movement = self.game.movement.movementAllowance(defender)
        rank = plan_skirmish_defence(model_base_boxes(attacker), model_base_boxes(defender), movement)
        if rank is None:
            rule_skipped('Skirmishers', defender,
                         f'cannot form a supported rank within M{movement:g} at actual front-base '
                         'contact; using legacy alignment (p. 186 LEFTOVER)')
            return False
        children = list(defender.model.getChildren())
        moves = []
        for index, position in zip(rank.order, rank.positions):
            child = children[index]
            world = Point3(*position, child.getZ(render))
            local = child.getParent().getRelativePoint(render, world)
            moves.append(LerpPosHprInterval(
                child, duration=0.3, pos=local,
                hpr=(rank.heading - child.getParent().getH(render), 0, 0), blendType='easeInOut'))
        await Parallel(*moves)
        apply_fighting_rank(self.game, defender, rank)
        rule_log('Skirmishers', defender,
                 f'{rank.files} models form the fighting rank within M{movement:g}; '
                 f'{len(rank.positions) - rank.files} form behind; '
                 f'{attacker.unit.name} stays fixed without an alignment wheel (p. 186)')
        return True

    async def _skirmishChargeInterval(self, unit, defenderNP, oposUnit, orotUnit, flank, chdice=None):
        """Resolve the roll and anchor Skirmisher ranks to contact (pp. 186-187)."""
        maxmove = self.game.movement.movementAllowance(unit, oposUnit, self.game.playerNP.getPos())

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

        chdist = self.chargeDistance(unit, oposUnit, chdice)

        defenderUnit = self.game.getSelectedUnit(defenderNP.node())
        from skirmish_charge import supported_pair, supported_formed_target
        formation = None
        formed_target = supported_formed_target(unit, defenderUnit)
        planned_pair = supported_pair(unit, defenderUnit) or formed_target
        if not planned_pair:
            rule_skipped('Skirmishers', unit,
                         f'individual form-up against {defenderUnit.unit.name} is not supported '
                         'for this pairing; using legacy alignment (p. 187 LEFTOVER)')
        if planned_pair:
            from scouts import model_base_boxes
            from skirmish_charge import plan_skirmish_charge, plan_formed_charge
            unit.bodyNP.setPos(oposUnit)
            unit.bodyNP.setHpr(orotUnit)
            if formed_target:
                formation = plan_formed_charge(
                    model_base_boxes(unit), model_base_boxes(defenderUnit), chdist, oposUnit)
            else:
                formation = plan_skirmish_charge(
                    model_base_boxes(unit), model_base_boxes(defenderUnit), chdist,
                    self.game.movement.movementAllowance(defenderUnit))
            if formation is not None:
                await self._formSkirmishCharge(unit, defenderUnit, formation)

        target = self.game.playerNP.getPos()
        d = target - oposUnit
        dist = d.length()
        dirn = d / dist if dist > 1e-6 else Vec3(0, 1, 0)
        reached = chdist + 0.001 >= self.game.moveArceDistance
        if planned_pair:
            reached = formation is not None
        failed_distance = (chdist if unit.state == 'IsPursuing' else
                           charge_roll(chdice, self.chargeThroughDifficult(unit, oposUnit)))
        travel = dist if reached else min(failed_distance, dist)
        endp = oposUnit + dirn * travel

        if formation is None:
            unit.bodyNP.setPos(oposUnit)
            unit.bodyNP.setHpr(orotUnit)
            await LerpPosHprInterval(unit.bodyNP, duration=0.5,
                                     pos=endp, hpr=orotUnit, blendType='easeInOut')

        for terning in terninger:
            terning.remove(self.game.world)

        if not reached:
            print("Charge fell short \u2014 skirmishers did not reach the enemy.")
            rule_log('Failed Charge', unit,
                     f'range {chdist:g}" cannot reach; dice {chdice} -> '
                     f'advances {travel:.2f}" without adding M (p. 121)')
            self.game.movement.dangerousTerrainTests(unit, oposUnit, endp)
            unit.isChargingMove = False
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
            from first_charge import finish_charge_attempt
            finish_charge_attempt(unit, defenderUnit)
            self.removeUnitFromPlay(defenderUnit)
            unit.request("Moved")
            return

        was_pursuing = unit.state == 'IsPursuing'
        joins, why_not = (self.joinsCombatThisPhase(defenderUnit) if was_pursuing
                          else (False, ''))
        unit.request("InCombat")
        unit.isInCombat = True
        unit.chargedThisTurn = True
        unit.chargeDistance = float(formation.distance if formation is not None else travel)
        defenderUnit.wasChargedThisTurn = True
        from first_charge import count_as_charge, finish_charge_attempt
        if was_pursuing:
            if joins:
                unit.hasAttackedThisTurn = False
                unit.cannotPursueThisTurn = True
                rule_log('Pursuit into a New Combat', unit,
                         f'contacted {defenderUnit.unit.name}: fights this phase '
                         'counting as charged; cannot pursue again (p. 157)')
            else:
                unit.countsAsChargedNextTurn = True
                defenderUnit.countsAsChargeTargetNextTurn = True
                rule_log('Catching the Curs!', unit,
                         f'contacted {defenderUnit.unit.name}: counts as charged '
                         f'next turn, because {why_not} (p. 157)')
            count_as_charge(unit, defenderUnit, next_turn=not joins)
        else:
            finish_charge_attempt(unit, defenderUnit)
        self.game.movement.dangerousTerrainTests(unit, oposUnit, endp)
        if defenderUnit.state != "InCombat":
            defenderUnit.request("InCombat")
        unit.isInCombatWith.append(defenderUnit)
        unit.isInCombatFlank.append("front")
        defenderUnit.isInCombatWith.append(unit)
        defenderUnit.isInCombat = True
        defenderUnit.isInCombatFlank.append(formation.flank if formation is not None else flank)
        unit.updateTextNode()
        defenderUnit.updateTextNode()

    async def _formSkirmishCharge(self, unit, defender, formation):
        """Chargers first; only loose defenders subsequently form up (pp. 186-187)."""
        from direct.interval.IntervalGlobal import Parallel
        from skirmish_charge import apply_fighting_rank

        def intervals(member, rank, indices):
            children = list(member.model.getChildren())
            moves = []
            for model_index in indices:
                slot = rank.order.index(model_index)
                child = children[model_index]
                world = Point3(*rank.positions[slot], child.getZ(render))
                local = child.getParent().getRelativePoint(render, world)
                moves.append(LerpPosHprInterval(
                    child, duration=0.3, pos=local,
                    hpr=(rank.heading - child.getParent().getH(render), 0, 0),
                    blendType='easeInOut'))
            return moves

        await Parallel(*intervals(unit, formation.attacker, [formation.first_attacker]))
        remaining = [index for index in formation.attacker.order if index != formation.first_attacker]
        if remaining:
            await Parallel(*intervals(unit, formation.attacker, remaining))
        apply_fighting_rank(self.game, unit, formation.attacker)
        if formation.defender is not None:
            await Parallel(*intervals(defender, formation.defender, formation.defender.order))
            apply_fighting_rank(self.game, defender, formation.defender)
            result = (f'{formation.attacker.files} chargers face {formation.defender.files} defenders '
                      'in touching fighting ranks (p. 187)')
        else:
            result = (f'{formation.attacker.files} chargers touch {defender.unit.name}\'s '
                      f'{formation.flank}; formed defender stays fixed (p. 186)')
        rule_log('Skirmishers', unit,
                 f'first model {formation.first_attacker + 1} moves {formation.distance:.2f}"; {result}')

    # ─── Flank Detection ──────────────────────────────────────────────────

    async def alignToEnemy(self, unit, angleToRotate, duration=0.5, *, pivot=None):
        """Pivot to align, about the point of contact (p. 157).

        A charge leaves that point on `playerNP`; an overrun has to supply the
        corner that struck, never having gone through the move code that sets
        it. Nothing else moves: pivoting about the point the two bases share
        leaves them sharing it.

        `pivot` is keyword-only so it cannot swallow a positional duration.
        """
        parent = unit.bodyNP.getParent()
        newnode = render.attachNewNode(f"Temp-{unit.unitName}")
        newnode.setPos(self.game.playerNP.getPos() if pivot is None else pivot)
        newnode.setHpr(unit.bodyNP.getHpr())
        unit.bodyNP.wrtReparentTo(newnode)
        await LerpPosHprInterval(
            newnode,
            duration=duration,
            pos=newnode.getPos(),
            hpr=(newnode.getH() + angleToRotate, newnode.getP(), newnode.getR()),
            blendType='easeInOut'
        )
        unit.bodyNP.wrtReparentTo(parent)
        newnode.removeNode()

    def contactPointOn(self, unit, otherNP):
        """The corner of *unit*'s base that struck *otherNP*.

        Bullet's manifold carries a contact point of its own, but for a contact
        test on a body that was just placed there -- rather than a collision the
        solver worked out -- it is not reliably on the struck face.
        """
        mine = unit.bodyNP.node().getShape(0).getHalfExtentsWithMargin()
        theirs = otherNP.node().getShape(0).getHalfExtentsWithMargin()
        local = [Point3(sx * mine.x, sy * mine.y, 0)
                 for sx in (-1, 1) for sy in (-1, 1)]
        seen = [otherNP.getRelativePoint(unit.bodyNP, p) for p in local]
        i = nearest_corner([(p.x, p.y) for p in seen], theirs.x, theirs.y)
        return render.getRelativePoint(unit.bodyNP, local[i])

    def joinsCombatThisPhase(self, enemy):
        """Pursuit into a New Combat (p. 157): the pursuer fights again only if
        the enemy was already engaged when the phase began and that combat has
        not been fought yet. `hasAttackedThisTurn` is set on every unit in a
        combat as it is resolved, so it doubles as "this combat has been
        fought".

        Returns ``(joins, why_not)`` — three conditions decline this rule and
        they are indistinguishable on the board, so the caller has to be able
        to say which one did.
        """
        if not getattr(enemy, 'startOfPhaseEngaged', False):
            return False, "it was not in a combat when this phase began"
        if enemy.hasAttackedThisTurn:
            return False, "its combat has already been fought this phase"
        if not any(not u.bodyNP.isEmpty() for u in enemy.isInCombatWith):
            return False, "the combat it began the phase in is already over"
        return True, ""

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
        battle_log(
            f"{attackerUnit.unit.name} v {defenderUnit.unit.name}: "
            f"{attacks} {verb}, {total_hits} hit, {total_wounds} slain",
            'good' if total_wounds else 'combat')

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
            unsaved, _ = self.commandWoundLimit(target, unsaved)
            self.previewCombatWounds(target, unsaved)
            if target.unit.nmodels == 0:
                from command_groups import capture_standard
                capture_standard(self.game, target, striker)
            if striker in self.game.player1Units:
                player1_score += unsaved
            else:
                player2_score += unsaved
            modRemoveSequence.append(Func(self.applyCombatWounds, target, unsaved))
        return player1_score, player2_score

    # ─── Battle Start & Resolution ────────────────────────────────────────

    def previewCombatWounds(self, target, wounds, slaying=0):
        """Bank multi-Wound losses until the casualty animation runs (p. 150)."""
        if not hasattr(self, '_pendingWounds'):
            self._pendingWounds = {}
        remaining = self._pendingWounds.get(id(target), getattr(target, 'woundsOnModel', 0))
        if slaying:
            remaining = 0
        total = remaining + max(0, wounds)
        per_model = max(1, _stat_int(target.unit.model.characteristics, 'W', 1))
        casualties, remainder = divmod(total, per_model)
        target.unit.nmodels = max(0, target.unit.nmodels - casualties - slaying)
        self._pendingWounds[id(target)] = remainder

    def commandWoundLimit(self, target, wounds, slaying=0, challenge=None):
        """Undirected attacks cannot spill onto a champion (Rulebook p. 199)."""
        from command_groups import champions
        protected = champions(target, include_retired=True)
        if not protected:
            return wounds, slaying
        duelling = challenge is not None and any(challenge.involves(champion) for champion in protected)
        reserve = 1 if target.unit.nmodels > 1 or duelling else 0
        available = max(0, target.unit.nmodels - reserve)
        per_model = max(1, _stat_int(target.unit.model.characteristics, 'W', 1))
        killed = min(slaying, available)
        partial = 0 if killed else getattr(self, '_pendingWounds', {}).get(id(target), getattr(target, 'woundsOnModel', 0))
        admitted = min(wounds, max(0, (available - killed) * per_model - partial))
        if admitted != wounds or killed != slaying:
            rule_log('Champion', target,
                     f'{wounds} wounds and {slaying} slaying blows against ordinary models -> '
                     f'{admitted} wounds and {killed} slaying blows; champion protected (p. 199)')
        return admitted, killed

    def applyCombatWounds(self, target, wounds, slaying=0):
        """Replay the banked losses against rendered bodies, once per exchange."""
        if target not in self.game.units:
            return
        target.unit.nmodels = len(target.model.getChildren())
        self.game.applyWounds(target, wounds, slaying)

    def resolveMeleeProfiles(self, challenge, removals):
        from assailment import drain_steps
        return drain_steps(self._meleeProfileSteps(challenge, removals))

    async def resolveMeleeWithSpells(self, challenge, removals):
        from assailment import cast_at_initiative
        steps = self._meleeProfileSteps(challenge, removals)
        while True:
            try:
                caster, targets, damage, miscast_damage = next(steps)
            except StopIteration as finished:
                return finished.value
            await cast_at_initiative(self.game, caster, targets, damage, challenge=challenge,
                                     miscast_damage=miscast_damage)

    def _meleeProfileSteps(self, challenge, removals):
        """Resolve every rider, mount and crew at its own Initiative (pp. 146, 192-194)."""
        from combat_profiles import profile_strike_order
        order = profile_strike_order(self.game.attackers, self.game.defenders,
                                     self._engagedFacing, challenge)
        engaged = {id(member): member for member in self.game.attackers + self.game.defenders}
        scores = [0, 0]
        initiative_step = None
        snapshots = {}
        animated = set()
        cast = set()
        self._pendingWounds = getattr(self, '_pendingWounds', {})
        for initiative, part in order:
            host, target = part.host, part.target
            if host.bodyNP.isEmpty() or target.bodyNP.isEmpty():
                rule_skipped('Combat', host, f'I{initiative}: combatant removed during the challenge; no attacks')
                continue
            if initiative != initiative_step:
                initiative_step = initiative
                snapshots = {identity: member.unit.nmodels for identity, member in engaged.items()}
                attacks_at_step = {id(candidate): candidate.attacks(
                    snapshots[id(candidate.host)], self._combatStartModels.get(
                        id(candidate.host.unit), snapshots[id(candidate.host)]), challenge)
                    for step, candidate in order if step == initiative}
            if snapshots[id(target)] <= 0:
                rule_skipped('Combat', host, f'I{initiative}: opponent already slain before this Initiative; no attacks')
                continue
            models = snapshots[id(host)]
            caster = (part.fighter if part.role == 'character' and part.fighter in self.game.units
                      else host if part.role == 'main' else None)
            if caster is not None and id(caster) not in cast and attacks_at_step[id(part)] > 0:
                cast.add(id(caster))

                def spell_damage(victim, wounds, regenerated=0):
                    specific = getattr(victim, 'command_host', None) or getattr(victim, 'hostUnit', None)
                    if specific is not None:
                        left = max(0, wounds_remaining(victim)) if victim.unit.nmodels else 0
                        admitted = min(left, wounds)
                        scores[0 if host in self.game.player1Units else 1] += admitted + regenerated
                        if admitted:
                            previous = victim.woundsOnModel
                            victim.woundsOnModel += admitted
                            if admitted == left:
                                victim.unit.nmodels = 0
                                if getattr(victim, 'command_host', None) is not None:
                                    victim.command_entry['active'] = False
                                    specific.unit.nmodels = max(0, specific.unit.nmodels - 1)
                            removals.append(Func(self.applyAssailmentModelWounds, victim, previous, admitted))
                        return
                    admitted, _ = self.commandWoundLimit(victim, wounds, challenge=challenge)
                    per_model = max(1, _stat_int(victim.unit.model.characteristics, 'W', 1))
                    left = max(0, victim.unit.nmodels * per_model - self._pendingWounds.get(
                        id(victim), getattr(victim, 'woundsOnModel', 0)))
                    scores[0 if host in self.game.player1Units else 1] += min(left, admitted) + regenerated
                    self.previewCombatWounds(victim, admitted)
                    if victim.unit.nmodels == 0:
                        from command_groups import capture_standard
                        capture_standard(self.game, victim, host)
                    removals.append(Func(self.applyCombatWounds, victim, admitted))

                targets = [enemy for enemy in getattr(host, 'isInCombatWith', []) if enemy.unit.nmodels > 0
                           and not (challenge and challenge.involves(enemy))]
                def miscast_damage(victim, wounds, regenerated=0):
                    victim_host = getattr(victim, 'command_host', None) or getattr(victim, 'hostUnit', None) or victim
                    credit = min(wounds + regenerated, self.miscastWoundsRemaining(victim))
                    admitted = self.previewMiscastWounds(victim, wounds, removals)
                    if id(victim_host) in engaged:
                        scores[1 if victim_host in self.game.player1Units else 0] += credit
                        rule_log('Miscast', victim, f'{admitted} Wounds lost in this combat -> '
                                 f'+{credit} opposing combat result including {regenerated} regenerated, '
                                 'no overkill (pp. 151, 176)')

                yield caster, targets, spell_damage, miscast_damage
            if id(host) not in animated:
                animated.add(id(host))
                host.hasAttackedThisTurn = True
                host.updateTextNode()
                origin = host.bodyNP.getPos()
                direction = target.bodyNP.getPos() - origin
                if direction.lengthSquared():
                    self.game.attackSequence.append(LerpPosHprInterval(
                        host.bodyNP, duration=0.5, pos=origin - direction.normalized() * 2,
                        hpr=host.bodyNP.getHpr(), blendType='easeInOut'))
                    self.game.attackSequence.append(LerpPosHprInterval(
                        host.bodyNP, duration=0.5, pos=origin,
                        hpr=host.bodyNP.getHpr(), blendType='easeInOut'))
            if models <= 0 and part.role != 'character':
                rule_skipped('Split Profile' if part.role != 'main' else 'Combat', host,
                             f'{part.profile.name} at I{initiative}: no eligible attacks; '
                             f'{models} models remain at this Initiative step')
                continue
            if target.unit.model.equipedWeapon.get('tag') == 'ranged':
                target.unit.model.equip_best_melee()
            def attack_count():
                return attacks_at_step[id(part)]

            result = simulate_battle(part.unit(attack_count), target.unit,
                                     charge=getattr(host, 'chargedThisTurn', False),
                                     first_round=getattr(host, 'roundsFought', 0) == 1)
            slaying = take_last_slaying_blows()
            if not result[0]:
                rule_skipped('Champion' if part.role == 'champion' else 'Split Profile', host,
                             f'{part.profile.name} at I{initiative}: no eligible attacks after modifiers')
                continue
            wounds, slaying = self.commandWoundLimit(target, result[-1] - slaying, slaying, challenge)
            total_wounds = wounds + slaying
            self.printBattleResults(host, target, *result)
            rule_log('Champion' if part.role == 'champion' else 'Split Profile', host,
                     f"{part.profile.name} ({part.role}) at I{initiative}, "
                     f"{(part.profile.equipedWeapon or {}).get('name', 'unarmed')}: "
                     f'{result[0]} attacks -> {total_wounds} unsaved wounds')
            self.previewCombatWounds(target, wounds, slaying)
            if target.unit.nmodels == 0:
                from command_groups import capture_standard
                capture_standard(self.game, target, host)
            scores[0 if host in self.game.player1Units else 1] += total_wounds
            removals.append(Func(self.applyCombatWounds, target, wounds, slaying))
        return scores

    def applyAssailmentModelWounds(self, victim, previous, wounds):
        """Replay a selected model's damage after simultaneous attacks (pp. 146, 199)."""
        victim.unit.nmodels = 1
        victim.woundsOnModel = previous
        command_host = getattr(victim, 'command_host', None)
        if command_host is not None:
            command_host.unit.nmodels = len(command_host.model.getChildren())
            victim.command_entry['active'] = True
        self.woundDuellist(victim, wounds)

    def previewMiscastWounds(self, victim, wounds, removals):
        """Bank incidental damage alongside pending combat losses (pp. 109, 146, 150-151)."""
        from spell_effects import caster_removed
        specific = getattr(victim, 'command_host', None) or getattr(victim, 'hostUnit', None)
        if specific is not None:
            left = max(0, wounds_remaining(victim)) if victim.unit.nmodels else 0
            admitted = min(left, wounds)
            if admitted:
                previous = victim.woundsOnModel
                victim.woundsOnModel += admitted
                if admitted == left:
                    victim.unit.nmodels = 0
                    if getattr(victim, 'command_host', None) is not None:
                        victim.command_entry['active'] = False
                        specific.unit.nmodels = max(0, specific.unit.nmodels - 1)
                removals.append(Func(self.applyAssailmentModelWounds, victim, previous, admitted))
        else:
            per_model = max(1, _stat_int(victim.unit.model.characteristics, 'W', 1))
            partial = getattr(self, '_pendingWounds', {}).get(id(victim), getattr(victim, 'woundsOnModel', 0))
            admitted = min(wounds, max(0, victim.unit.nmodels * per_model - partial))
            self.previewCombatWounds(victim, admitted)
            removals.append(Func(self.applyCombatWounds, victim, admitted))
        if victim.unit.nmodels == 0:
            caster_removed(self.game, victim)
        return admitted

    def miscastWoundsRemaining(self, victim):
        if getattr(victim, 'command_host', None) or getattr(victim, 'hostUnit', None):
            return max(0, wounds_remaining(victim)) if victim.unit.nmodels else 0
        per_model = max(1, _stat_int(victim.unit.model.characteristics, 'W', 1))
        partial = getattr(self, '_pendingWounds', {}).get(id(victim), getattr(victim, 'woundsOnModel', 0))
        return max(0, victim.unit.nmodels * per_model - partial)

    async def verySimpleBattleStart(self, task):
        self.game.resolvingCombat = True
        weps = self.game.unitToMove.unit.model.weapons

        wepchoice = await taskMgr.add(self.game.makeChoiceNew(
            weps, Vec3(0, 0, 10), owner=self.game.unitToMove,
            prompt=f"{self.game.unitToMove.unit.name}: fight with which weapon?"))

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

    @staticmethod
    def _engagedFacing(target, striker):
        """Which of *target*'s arcs *striker* is engaged in.

        isInCombatWith and isInCombatFlank are appended in step, so the arc a
        striker charged into is the entry at its own index in its victim's list.
        """
        try:
            return target.isInCombatFlank[target.isInCombatWith.index(striker)]
        except (ValueError, IndexError):
            return 'front'

    def strikeOrder(self):
        """(Initiative, index) for every striker, highest Initiative first.

        Who Strikes First (p. 146): work down the Initiative values, models
        attacking as their value is reached.
        """
        order = []
        # A unit reachable from both sides of the engagement is listed twice,
        # and it strikes once.
        seen = set()
        for i, striker in enumerate(self.game.attackers):
            if striker.hasAttackedThisTurn:
                print(f"Unit {striker.unit.name} has already attacked this turn, skipping.")
                continue
            if id(striker) in seen:
                continue
            seen.add(id(striker))
            target = self.game.defenders[i]
            facing = self._engagedFacing(target, striker)
            charged = bool(getattr(striker, 'chargedThisTurn', False))
            inches = float(getattr(striker, 'chargeDistance', 0.0) or 0.0)
            model = striker.unit.model
            flanking = facing in ('flank', 'rear')
            profile = _stat_int(model.characteristics, 'I', 1)
            base = base_initiative(model)
            initiative = strike_initiative(model, charged=charged, inches=inches,
                                           flank_or_rear=flanking,
                                           first_round=getattr(striker, 'roundsFought', 0) == 1, log=True)
            if base != profile:
                name = 'Strike First' if base > profile else 'Strike Last'
                weapon = (model.active_melee_weapon() or {}).get('name')
                source = f" from its {weapon}" if weapon and weapon != 'Hand Weapon' else ''
                rule_log(name, striker,
                         f"strikes at I{base} instead of its profile I{profile}"
                         f"{source}, before any other modifier "
                         f"(p. {177 if base > profile else 178})")
            elif model.has_strike_first() and model.has_strike_last():
                rule_skipped('Strike First', striker,
                             f"cancelled out by Strike Last, so it strikes at its "
                             f"profile I{profile} (p. 178)")
            if charged:
                before_charge = strike_initiative(model, first_round=getattr(striker, 'roundsFought', 0) == 1)
                bonus = charge_initiative_bonus(inches, flanking)
                if not bonus:
                    rule_skipped('Charging Units', striker,
                                 f"charged only {inches:.1f}\", not a full inch, so no "
                                 f"Initiative bonus (stays I{initiative})")
                elif initiative > before_charge:
                    rule_log('Charging Units', striker,
                             f"charged {inches:.1f}\" into {target.unit.name}'s {facing} "
                             f"-> +{initiative - before_charge} Initiative (I{before_charge} -> I{initiative}, "
                             f"max +{4 if flanking else 3}) (p. 146)")
                else:
                    rule_skipped('Charging Units', striker,
                                 f"charged {inches:.1f}\" for +{bonus} Initiative, but "
                                 f"I{base} is already the maximum of 10 (p. 146)")
            order.append((initiative, i))
        order.sort(key=lambda e: -e[0])
        if len(order) > 1:
            names = ", ".join(f"{self.game.attackers[i].unit.name} I{v}" for v, i in order)
            battle_log(f"Strike order: {names}")
            for (v1, i1), (v2, i2) in zip(order, order[1:]):
                if v1 == v2:
                    rule_log('Simultaneous Combat', self.game.attackers[i2],
                             f"strikes at I{v2} alongside "
                             f"{self.game.attackers[i1].unit.name}: the active player "
                             f"resolves first, but neither side's casualties reduce the "
                             f"other's attacks (p. 146)")
        return order

    # ─── Challenges (Rulebook p. 210-211) ─────────────────────────────────

    @staticmethod
    def _duelName(model):
        return model.unit.name if model is not None else '-'

    async def challengeExchange(self, attackerUnit, defenderUnit):
        """Issue, accept or refuse, at Step 1.1 (p. 210).

        The active player is offered first; only if they decline may the
        inactive player issue. One challenge per combat, and none at all while
        an earlier one is still running (To The Death!, p. 211).
        """
        live = find_challenge(self.game, attackerUnit, defenderUnit)
        if live is not None:
            live.rounds += 1
            rule_log('To The Death!', live.challenger,
                     f"the challenge against {self._duelName(live.accepter)} carries "
                     f"into round {live.rounds + 1}; no other may be issued in this "
                     f"combat until it resolves (p. 211)")
            await self.armDuellists(live)
            return live

        for issuer, target in ((attackerUnit, defenderUnit),
                               (defenderUnit, attackerUnit)):
            challenger = duellist(issuer)
            if challenger is None:
                continue
            # The AI never issues, so it is only ever asked to accept.
            if self.game.aiControls(issuer):
                rule_skipped('Challenges', issuer,
                             "the AI does not issue challenges")
                continue
            answer = await taskMgr.add(self.game.makeChoiceNew(
                ["Issue a challenge", "No challenge"],
                Vec3(0, 0, 12), owner=issuer,
                prompt=f"{challenger.unit.name} may challenge "
                       f"{target.unit.name}"))
            if answer != "Issue a challenge":
                rule_skipped('Challenges', challenger,
                             "its player declined to issue a challenge")
                continue
            challenge = Challenge(challenger, issuer)
            rule_log('Challenges', challenger,
                     f"issues a challenge to {target.unit.name} (p. 210)")
            await self.answerChallenge(challenge, target)
            add_challenge(self.game, challenge)
            await self.armDuellists(challenge)
            return challenge
        return None

    async def armDuellists(self, challenge):
        """Each duellist picks its own weapon — the duel is its own fight.

        The weapon chosen at the start of the combat was the *unit's*; a
        character keeps its own profile and weapons, so nothing had asked it.
        """
        for model, host in ((challenge.challenger, challenge.host),
                            (challenge.accepter, challenge.accepter_host)):
            if model is None:
                continue
            weapons = [name for name, w in model.unit.model.weapons.items()
                       if (w or {}).get('tag') != 'ranged']
            if len(weapons) < 2 or self.game.aiControls(host or model):
                model.unit.model.equip_best_melee()
                continue
            choice = await taskMgr.add(self.game.makeChoiceNew(
                weapons, Vec3(0, 0, 12), owner=host or model,
                prompt=f"{model.unit.name}: which weapon for the duel?"))
            if choice:
                model.unit.model.equip_weapon(choice)
            rule_log('Fighting a Challenge', model,
                     f"duels with its {model.unit.model.equipedWeapon['name']}")

    async def answerChallenge(self, challenge, target):
        """Accept or refuse, and retire a coward (p. 210-211)."""
        if not can_accept(target):
            rule_log('Challenges', challenge.challenger,
                     f"{target.unit.name} has no character to answer, so the "
                     f"challenge goes unanswered (p. 210)")
            return
        accepter = duellist(target)
        barred = refusal_barred(accepter, target if accepter is not target else None)
        # The AI always accepts.
        if self.game.aiControls(target) or barred is not None:
            answer = "Accept"
            if barred is not None:
                rule_log('Nowhere to Run', accepter,
                         f"cannot refuse — {barred} — and must meet the challenge "
                         f"(p. 211)")
        else:
            answer = await taskMgr.add(self.game.makeChoiceNew(
                ["Accept", "Refuse"], Vec3(0, 0, 12), owner=target,
                prompt=f"{self._duelName(challenge.challenger)} challenges "
                       f"{accepter.unit.name}"))
        if answer != "Refuse":
            challenge.accepter = accepter
            challenge.accepter_host = target
            rule_log('Challenges', accepter,
                     f"accepts the challenge from "
                     f"{self._duelName(challenge.challenger)} (p. 210)")
            return
        challenge.refused = True
        self.retireFromCombat(accepter, target)

    def retireFromCombat(self, model, host):
        """A model that refused a challenge hides in the rear ranks (p. 210)."""
        model.retiredFromCombat = True
        from spell_effects import refresh_self_spells
        refresh_self_spells(model)
        if host is not None and host is not model:
            host.placeCharacter()
            host.layOutRanks()
        rule_log('Refusing a Challenge', model,
                 "retires from combat: makes no attacks, has none directed at it, "
                 "and confers no Leadership or special rules on its unit while its "
                 "unit stays engaged (p. 210)")

    def duelCombatants(self, model, host):
        """A duellist and whatever fights alongside it, each with its own I.

        A mount, or a chariot's crew, must direct its attacks at the other
        participant (p. 211).
        """
        out = [(model.unit, '')]
        for rule in model.unit.model.special_rules:
            if isinstance(rule, dict) and rule.get('mountUnit'):
                mount = getattr(rule['mountUnit'], 'model', rule['mountUnit'])
                out.append((SimpleNamespace(name=mount.name, model=mount, nmodels=1, files=1, ranks=1), ' (mount)'))
        for part in self.chariotParts(model):
            out.append((part, ' (crew)'))
        return out

    def woundDuellist(self, model, wounds):
        """Put unsaved wounds on one duellist. Returns True if it falls."""
        if wounds <= 0:
            return False
        left = wounds_remaining(model)
        model.woundsOnModel = getattr(model, 'woundsOnModel', 0) + wounds
        if wounds < left:
            return False
        if getattr(model, 'command_host', None) is not None:
            model.command_entry['active'] = False
            self.game.movement.removeModelsFromUnit(model.command_host, 1)
        elif getattr(model, 'hostUnit', None) is not None:
            slay_character(self.game, model)
        else:
            self.game.movement.removeModelsFromUnit(model, 1)
        return True

    def resolveChallenge(self, challenge):
        from assailment import drain_steps
        return drain_steps(self._challengeSteps(challenge))

    async def resolveChallengeWithSpells(self, challenge, removals=None):
        from assailment import cast_at_initiative
        steps = self._challengeSteps(challenge, removals)
        while True:
            try:
                caster, targets, damage, miscast_damage = next(steps)
            except StopIteration as finished:
                return finished.value
            await cast_at_initiative(self.game, caster, targets, damage, challenge=challenge,
                                     miscast_damage=miscast_damage)

    def _challengeSteps(self, challenge, removals=None):
        """Fight the duel, in Initiative order (p. 211).

        Returns (player 1 wounds, player 2 wounds, player 1 overkill,
        player 2 overkill). The duellists attack only each other and nothing
        else may attack them, so this is sealed off from the combat around it.
        """
        if not challenge.answered:
            return 0, 0, 0, 0
        player_one = challenge.host in self.game.player1Units
        order = []
        for model, host in ((challenge.challenger, challenge.host),
                            (challenge.accepter, challenge.accepter_host)):
            charged = bool(getattr(host or model, 'chargedThisTurn', False))
            inches = float(getattr(host or model, 'chargeDistance', 0.0) or 0.0)
            first = getattr(host or model, 'roundsFought', 0) == 1
            for unit, label in self.duelCombatants(model, host):
                order.append((strike_initiative(unit.model, charged=charged,
                                                inches=inches, first_round=first, log=True),
                              model, unit, label, charged, first))
        order.sort(key=lambda e: -e[0])
        battle_log("Challenge: " + " vs ".join(
            self._duelName(m) for m in challenge.participants()))
        scores = {id(challenge.challenger): 0, id(challenge.accepter): 0}
        overkill = {id(challenge.challenger): 0, id(challenge.accepter): 0}
        incidental = [0, 0]
        affected_hosts = {id(member) for member in (getattr(self.game, 'attackers', [])
                  + getattr(self.game, 'defenders', []) + [challenge.host, challenge.accepter_host])}
        fallen = set()
        cast = set()
        from itertools import groupby
        for initiative, step in groupby(order, key=lambda entry: entry[0]):
            inflicted = {id(participant): 0 for participant in challenge.participants()}
            hazard_removals = Sequence()
            hazard_fallen = set()
            for _, model, unit, label, charged, first in step:
                rival = challenge.opponent_of(model)
                if id(model) in fallen or id(rival) in fallen:
                    rule_skipped('Challenges & Mounts', model,
                                 f'I{initiative}{label}: a participant was slain at a higher Initiative (p. 211)')
                    continue
                if unit.model is model.unit.model and id(model) not in cast:
                    cast.add(id(model))

                    def spell_damage(target, wounds, regenerated=0):
                        inflicted[id(model)] += wounds
                        scores[id(model)] += min(regenerated, wounds_remaining(target))

                    def miscast_damage(victim, wounds, regenerated=0):
                        victim_host = getattr(victim, 'command_host', None) or getattr(victim, 'hostUnit', None) or victim
                        credit = min(wounds + regenerated, self.miscastWoundsRemaining(victim))
                        queue = hazard_removals if removals is None else removals
                        admitted = self.previewMiscastWounds(victim, wounds, queue)
                        if victim.unit.nmodels == 0:
                            hazard_fallen.add(id(victim))
                        if id(victim_host) in affected_hosts:
                            incidental[1 if victim_host in self.game.player1Units else 0] += credit
                            rule_log('Miscast', victim, f'{admitted} Wounds lost -> +{credit} opposing '
                                     f'combat result including {regenerated} regenerated, '
                                     'no challenge overkill (pp. 151, 176)')

                    yield model, [rival], spell_damage, miscast_damage
                weapon = unit.model.equipedWeapon
                if weapon is None or weapon.get('tag') == 'ranged':
                    unit.model.equip_best_melee()
                attacks, hits, suffered, saved, wounds = simulate_battle(
                    unit, rival.unit, charge=charged, first_round=first)
                slaying = take_last_slaying_blows()
                if slaying:
                    wounds = max(wounds, wounds_remaining(rival))
                rule_log('Fighting a Challenge', model,
                         f'strikes{label} at I{initiative}: {attacks} attack(s) -> '
                         f'{hits} hit -> {wounds} unsaved wound(s) on {self._duelName(rival)} (p. 211)')
                inflicted[id(model)] += wounds
            hazard_removals.finish()
            fallen.update(hazard_fallen)
            for model in challenge.participants():
                rival = challenge.opponent_of(model)
                wounds = inflicted[id(model)]
                if not wounds or id(rival) in fallen:
                    continue
                left = self.miscastWoundsRemaining(rival) if removals is not None else wounds_remaining(rival)
                scores[id(model)] += min(wounds, left)
                if removals is not None:
                    self.previewMiscastWounds(rival, min(wounds, left), removals)
                    slain = wounds >= left
                else:
                    slain = self.woundDuellist(rival, wounds)
                if slain:
                    fallen.add(id(rival))
                    bonus = overkill_bonus(wounds, left)
                    overkill[id(model)] += bonus
                    rule_log('Challenges', model,
                             f'slays {self._duelName(rival)} in the challenge')
                    if bonus:
                        rule_log('Overkill', model,
                                 f'{wounds} unsaved wounds against {left} Wounds remaining -> '
                                 f'{left} wounds +{bonus} overkill (max {MAX_OVERKILL}) (p. 211)')
        if fallen:
            end_challenge(self.game, challenge)
        first, second = challenge.challenger, challenge.accepter
        if not player_one:
            first, second = second, first
        return (scores[id(first)] + incidental[0], scores[id(second)] + incidental[1],
                overkill[id(first)], overkill[id(second)])

    @staticmethod
    def takeAssailmentWounds(units):
        """Consume pre-fight Assailment wounds once for this combat (p. 151)."""
        total = 0
        for unit in units:
            banked = getattr(unit, 'assailmentWounds', 0)
            if banked:
                rule_log('Assailment', unit, f'{banked} wounds add +{banked} combat result (p. 151)')
                total += banked
                unit.assailmentWounds = 0
        return total

    async def _verySimpleBattleInner(self, task):
        attacker = self.game.unitToMove.bodyNP
        defender = self.game.unitToMove.isInCombatWith[0].bodyNP
        engagedWith = [x.unitName for x in self.game.unitToMove.isInCombatWith]

        selected_choice = await taskMgr.add(self.game.makeChoiceNew(
            engagedWith, Vec3(0, 0, 10), owner=self.game.unitToMove,
            prompt=f"{self.game.unitToMove.unit.name}: which enemy will it fight?"))

        for unit in self.game.unitToMove.isInCombatWith:
            if unit.unitName == selected_choice:
                defender = unit.bodyNP
                break
        attackerUnit = self.game.getSelectedUnit(attacker.node())
        defenderUnit = self.game.getSelectedUnit(defender.node())
        defender_nmodels = defenderUnit.unit.nmodels
        # A unit's own isInCombatFlank entry is the facing *it* is engaged on,
        # so the facing being struck has to be read from the defender's side,
        # indexed by this attacker rather than by whichever engagement is first.
        try:
            flank = defenderUnit.isInCombatFlank[
                defenderUnit.isInCombatWith.index(attackerUnit)]
        except (ValueError, IndexError):
            flank = 'front'
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
        for unit in dict.fromkeys(self.game.attackers + self.game.defenders):
            await self.shieldwallWeaponChoice(unit)
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
        self._pendingWounds = {}
        engaged = set(self.game.attackers) | set(self.game.defenders)
        foesBefore = {id(unit): list(unit.isInCombatWith) for unit in engaged}
        p1_units = [unit for unit in engaged if unit in self.game.player1Units]
        p2_units = [unit for unit in engaged if unit in self.game.player2Units]
        impact1, impact2 = self.impactHits(modRemoveSequence)
        player1_score += impact1
        player2_score += impact2
        # Challenges are issued when the combat is chosen, at Step 1.1 (p. 210).
        challenge = await self.challengeExchange(attackerUnit, defenderUnit)
        duel1, duel2, overkill1, overkill2 = (
            await self.resolveChallengeWithSpells(challenge, modRemoveSequence) if challenge else (0, 0, 0, 0))
        player1_score += duel1 + overkill1
        player2_score += duel2 + overkill2
        melee1, melee2 = await self.resolveMeleeWithSpells(challenge, modRemoveSequence)
        player1_score += melee1
        player2_score += melee2

        player1_score += self.takeAssailmentWounds(p1_units)
        player2_score += self.takeAssailmentWounds(p2_units)
        # Wounds are everything banked so far bar the Impact Hits. The
        # challenge's own wounds count as wounds like any other, but Overkill
        # is a separate bonus and has its own row.
        p1_wounds = player1_score - impact1 - overkill1
        p2_wounds = player2_score - impact2 - overkill2
        from psychology import combat_flank_bonus, combat_rank_bonus
        player1_flank_bonus = sum(combat_flank_bonus(unit, log=True) for unit in p2_units if unit.unit.nmodels > 0)
        player2_flank_bonus = sum(combat_flank_bonus(unit, log=True) for unit in p1_units if unit.unit.nmodels > 0)
        player1_rank_bonus = min(MAX_RANK_BONUS, sum(combat_rank_bonus(unit, log=True)
                               for unit in p1_units if unit.unit.nmodels > 0))
        player2_rank_bonus = min(MAX_RANK_BONUS, sum(combat_rank_bonus(unit, log=True)
                               for unit in p2_units if unit.unit.nmodels > 0))
        player1_score += player1_flank_bonus + player1_rank_bonus
        player2_score += player2_flank_bonus + player2_rank_bonus
        player1_standard = battle_standard_bonus(p1_units)
        player2_standard = battle_standard_bonus(p2_units)
        player1_score += player1_standard
        player2_score += player2_standard
        from command_groups import standard_bonus, musician_bonus
        ordinary1 = standard_bonus(p1_units, log=True)
        ordinary2 = standard_bonus(p2_units, log=True)
        player1_score += ordinary1
        player2_score += ordinary2
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
        # Wounds from a Stand & Shoot count for the combat that follows it
        # (p. 151), so they are banked in the Movement phase and spent here.
        p1_shot = sum(getattr(u, 'standAndShootWounds', 0) for u in p1_units)
        p2_shot = sum(getattr(u, 'standAndShootWounds', 0) for u in p2_units)
        player1_score += p1_shot
        player2_score += p2_shot
        for units in (p1_units, p2_units):
            for u in units:
                banked = getattr(u, 'standAndShootWounds', 0)
                if banked:
                    rule_log('Stand & Shoot', u,
                             f"{banked} unsaved wound(s) from the charge "
                             f"reaction count towards this combat (p. 151)")
        music1, music2 = musician_bonus(p1_units, p2_units, player1_score, player2_score, log=True)
        player1_score += music1
        player2_score += music2
        self.printCombatResult(
            {'Wounds caused': (p1_wounds, p2_wounds),
             'Impact Hits': (impact1, impact2),
             'Stand & Shoot': (p1_shot, p2_shot),
             'Overkill': (overkill1, overkill2),
             'Flank / rear': (player1_flank_bonus, player2_flank_bonus),
             'Rank Bonus': (player1_rank_bonus, player2_rank_bonus),
             'Battle Standard': (player1_standard, player2_standard),
             'Standard Bearer': (ordinary1, ordinary2),
             'Musician': (music1, music2),
             'Massed Infantry': (player1_massed, player2_massed)},
            (player1_score, player2_score), (p1_us, p2_us))
        await self.game.attackSequence
        await modRemoveSequence

        # A unit that wiped out its enemy overruns instead of taking any part
        # in the Break test sub-phase, which is why this comes first (p. 156).
        await self.overrunPass(engaged, foesBefore)

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
                if self.game.aiControls(loserUnit):
                    useStubborn = should_use_stubborn(ld, diff, overwhelm)
                else:
                    stubbornChoice = ['Stand Firm', 'Break test']
                    selected = await taskMgr.add(
                        self.game.makeChoiceNew(
                            stubbornChoice, Vec3(0, 0, 10), owner=loserUnit,
                            prompt=f"{loserUnit.unit.name} is Stubborn: "
                                   f"stand firm?"))
                    useStubborn = selected == stubbornChoice[0]
                if useStubborn:
                    loserUnit.usedStubborn = True
                    print(f"{loserUnit.unit.name} is Stubborn and refuses its Break "
                          f"test — Falls Back in Good Order.")
                    outcome = await self.shieldwallOutcome(loserUnit, 'fall_back')
                    if outcome == 'fall_back':
                        self.notifyFleesCombat(loserUnit)
                    outcomes.append((loserUnit, outcome))
                    continue

            ldDice = await self.rollBreakDice()
            print("Leadership dice results for fleeing unit:", ldDice,
                  "sum:", sum(ldDice), "Ld:", ld, "combat result diff:", diff,
                  "overwhelmed:", overwhelm)
            outcome = break_test_outcome(ldDice, ld, diff, overwhelm)
            veteran_reroll_allowed(loserUnit, 'Break', sum(ldDice), ld)

            bsb = psy.battle_standard_of(loserUnit) if psy is not None else None
            from magic_items import reroll_break_test
            ldDice = await reroll_break_test(self.game, loserUnit, ldDice, ld, diff,
                                             overwhelm, self.rollBreakDice, bsb)
            outcome = break_test_outcome(ldDice, ld, diff, overwhelm)

            outcome = await self.shieldwallOutcome(loserUnit, outcome)
            if outcome in ('break', 'fall_back'):
                print("losing unit flees from combat!" if outcome == 'break'
                      else "losing unit FBIG!")
                self.notifyFleesCombat(loserUnit)
            else:
                print("losing unit gives ground!")
            outcomes.append((loserUnit, outcome))
        return outcomes

    # ─── Post-Combat: pass 2, the winners declare ─────────────────────────

    async def shieldwallWeaponChoice(self, unit):
        """Choose shields before fighting, never after seeing the result (p. 177)."""
        model = unit.unit.model
        if shieldwall_unavailable_reason(unit, check_weapon=False) is not None:
            return
        if not model.melee_weapon_requires_two_hands():
            return
        weapon = model.equipedWeapon['name']
        shields = 'Hand weapon & shield'
        choice = shields if self.game.aiControls(unit) else await taskMgr.add(
            self.game.makeChoiceNew(
                [shields, weapon], Vec3(0, 0, 10), owner=unit,
                prompt=f'{unit.unit.name}: shields or {weapon}?'))
        if choice == shields:
            model.equip_weapon('hand weapon')
            rule_log('Shieldwall', unit,
                     f'chooses hand weapon and shield instead of {weapon} before fighting; '
                     'once-per-game use remains available')
        else:
            rule_skipped('Shieldwall', unit,
                         f'chooses {weapon} instead of shields for this combat')

    async def shieldwallOutcome(self, unit, outcome):
        """Optional Shieldwall after Stubborn or the final Break roll (p. 177).

        Resolve before flee notifications: Giving Ground does not cause Panic.
        A Break, including one caused by overwhelming Unit Strength, is not a
        Fall Back in Good Order result and cannot be replaced.
        """
        if not getattr(unit.unit.model, 'is_shieldwall', lambda: False)():
            return outcome
        if outcome != 'fall_back':
            rule_skipped('Shieldwall', unit,
                         f'result is {outcome.replace("_", " ")}, not Fall Back in Good Order')
            return outcome
        reason = shieldwall_unavailable_reason(unit)
        if reason is not None:
            rule_skipped('Shieldwall', unit, f'{reason}; Falls Back in Good Order')
            return outcome
        use_shieldwall = self.game.aiControls(unit)
        if not use_shieldwall:
            choices = ['Shieldwall', 'Fall Back in Good Order']
            selected = await taskMgr.add(self.game.makeChoiceNew(
                choices, Vec3(0, 0, 10), owner=unit,
                prompt=f'{unit.unit.name}: spend Shieldwall to Give Ground?'))
            use_shieldwall = selected == choices[0]
        if not use_shieldwall:
            rule_skipped('Shieldwall', unit,
                         'player keeps its once-per-game use; Falls Back in Good Order')
            return outcome
        unit.usedShieldwall = True
        rule_log('Shieldwall', unit,
                 'charged this turn, in Close Order and using shields; '
                 'spends its once-per-game use: Fall Back in Good Order -> Give Ground 2"')
        return 'give_ground'

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
                        self.game.makeChoiceNew(
                            names, Vec3(0, 0, 10), owner=winner,
                            prompt=f"{winner.unit.name}: which enemy will it "
                                   f"chase?"))
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
        verb = {'follow_up': 'Follow up', 'overrun': 'Overrun'}.get(move, 'Pursue')
        quarry = f" {target.unit.name}" if target is not None else ""
        if move != 'follow_up' and getattr(winner, 'cannotPursueThisTurn', False):
            # Pursuit into a New Combat (p. 157): a unit that joined a combat
            # mid-phase restrains and reforms with no Restraint test.
            rule_log('Pursuit into a New Combat', winner,
                     f"already joined a new combat this turn, so it may not "
                     f"{verb.lower()}{quarry}: restrains and reforms with no "
                     f"Restraint test")
            winner.request("Idle")
            return 'restrain'
        if self.game.aiControls(winner):
            chosen = move
        else:
            options = [verb, 'Restrain']
            selected = await taskMgr.add(
                self.game.makeChoiceNew(
                    options, Vec3(0, 0, 10), owner=winner,
                    prompt=f"{winner.unit.name}: {verb.lower()}{quarry}?"))
            chosen = move if selected == options[0] else 'restrain'

        if chosen != 'restrain':
            print(f"{winner.unit.name} chooses to {verb.lower()}{quarry}!")
            return move

        ld = _stat_int(winner.unit.model.characteristics, 'Ld', 7)
        psy = getattr(self.game, 'psychology', None)
        if psy is not None:
            ld, _ = psy.leadership_of(winner)
        dice = await self.rollBreakDice()
        dice = await reroll_leadership(self.game, winner, 'Restraint', dice, ld,
                          self.rollBreakDice)
        if restraint_test(ld, dice):
            rule_log('Restrain & Reform', winner,
                     f"Restraint test {dice} = {sum(dice)} vs Ld {ld} -> holds "
                     f"its ground, and reforms once the enemy has drawn off")
            winner.request("Idle")
            return 'restrain'
        rule_log('Restrain & Reform', winner,
                 f"Restraint test {dice} = {sum(dice)} vs Ld {ld} -> fails, and "
                 f"must {verb.lower()}{quarry}")
        return move

    # ─── Post-Combat: Overrun ─────────────────────────────────────────────

    async def overrunPass(self, engaged, foesBefore):
        """Overrun (p. 156) is offered before the Break test sub-phase, to any
        unit that has just wiped out everything it was fighting.

        *foesBefore* is the engagement map from before the casualties were
        taken off, because a destroyed unit clears its foes' combat lists.
        """
        for winner in list(engaged):
            if winner.bodyNP.isEmpty():
                continue
            foes = foesBefore.get(id(winner)) or []
            if not foes or any(not u.bodyNP.isEmpty() for u in foes):
                continue            # it was fighting nothing, or something survived
            action = await self.restrainChoice(winner, None, 'overrun')
            if action == 'restrain':
                winner.request("Idle")
                await self.freeReform(winner)
            else:
                await self.overrunMove(winner)

    async def overrunMove(self, winner):
        """A normal pursuit move, but directly forwards and without pivoting
        (p. 156)."""
        pos = winner.bodyNP.getPos()
        bonus = await self.swiftstrideChoice(
            winner, 'pursuit', distance_to_edge=board_edge_distance(pos.x, pos.y))
        dice = await self.rollMoveDice(winner, 3 if bonus else 2, bonus)
        rolled = pursuit_roll(dice)

        heading = facing_vector(winner.bodyNP.getH())
        forward = Vec3(heading[0], heading[1], 0)
        # Whatever is in the way stops the move here; which of the p. 157 cases
        # it turns into depends on what was hit, and that is read off the
        # contact once the unit has arrived.
        clear = self.game.sweepTest(winner, forward, rolled)
        moved = rolled * clear

        rule_log('Overrun', winner,
                 f"destroyed everything it was fighting: {dice} = {rolled}\" "
                 f"straight ahead, no pivot"
                 + ("" if clear >= 1.0 else
                    f", halted after {moved:.1f}\" by what is in front of it"))
        winner.request("Moved")          # leaving InCombat clears the stale foe list
        if moved > 0.05:
            await LerpPosInterval(winner.bodyNP, duration=1.0,
                                  pos=pos + forward * moved,
                                  blendType='easeInOut')
        if clear < 1.0:
            await self.overrunContact(winner, moved)

    async def overrunContact(self, winner, moved):
        """Resolve whatever an overrun ran into (p. 157).

        An overrun is a normal pursuit move, so the pursuit cases apply to it:
        a friend or impassable terrain simply stops it, a fleeing enemy is run
        down, and a fresh enemy is charged.
        """
        c = self.game.checkUnitContactSmall(winner)
        if c is None:
            rule_log('Pursuit into an Obstacle', winner,
                     "halted against impassable terrain")
            return
        blockerNP = render.find(f"**/{c.getNode1().getName()}")
        blocker = self.game.getSelectedUnit(blockerNP.node())
        if blocker is None:
            return

        if (winner in self.game.player1Units) == (blocker in self.game.player1Units):
            rule_log('Pursuit into an Obstacle', winner,
                     f"halted against {blocker.unit.name}, a friendly unit")
            return

        if blocker.state == "IsFleeing":
            rule_log('Pursuit into a Fleeing Enemy', winner,
                     f"overran into the fleeing {blocker.unit.name} and ran it "
                     f"down")
            self.removeUnitFromPlay(blocker)
            await self.freeReform(winner)
            return

        flank, angleToRotate = self.getFlankFromContact(winner, c)
        # Read the flank from the contact as it was made, then wheel about the
        # point on the overrunning unit that struck.
        await self.alignToEnemy(
            winner, angleToRotate,
            pivot=self.contactPointOn(winner, blockerNP))

        # "That combat has not yet been fought" decides whether this becomes a
        # fight this phase or a lock until the next turn.
        joins, whyNot = self.joinsCombatThisPhase(blocker)

        winner.request("InCombat")
        winner.isInCombat = True
        if blocker.state != "InCombat":
            blocker.request("InCombat")
        blocker.isInCombat = True
        blocker.wasChargedThisTurn = True
        winner.isInCombatWith.append(blocker)
        winner.isInCombatFlank.append("front")
        blocker.isInCombatWith.append(winner)
        blocker.isInCombatFlank.append(flank)
        winner.updateTextNode()
        blocker.updateTextNode()

        if joins:
            # Pursuit into a New Combat (p. 157).
            winner.chargedThisTurn = True
            winner.chargeDistance = moved
            # Its attack for this phase is spent, and the attack loop skips
            # anyone carrying that, so the allowance has to be given back.
            winner.hasAttackedThisTurn = False
            winner.cannotPursueThisTurn = True
            # Both fight this phase, after enterCombatPhase has already counted
            # the round, so anyone not already fighting is on their first.
            for u in (winner, blocker):
                if not getattr(u, 'roundsFought', 0):
                    u.roundsFought = 1
            rule_log('Pursuit into a New Combat', winner,
                     f"overran into {blocker.unit.name}'s unfought combat by "
                     f"its {flank}: fights again this phase counting as "
                     f"charged, and may not pursue again")
        else:
            winner.countsAsChargedNextTurn = True
            blocker.countsAsChargeTargetNextTurn = True
            rule_log('Pursuit into a Fresh Enemy', winner,
                     f"overran into {blocker.unit.name}'s {flank}, wheeling to "
                     f"align: locked together and fought next turn, counting "
                     f"as charged, because {whyNot}")
        from first_charge import count_as_charge
        count_as_charge(winner, blocker, next_turn=not joins)

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
            loserUnit.surroundedThisPhase = False
            if outcome == 'give_ground':
                followers = [r['winner'] for r in responses
                             if r['target'] is loserUnit and r['action'] == 'follow_up']
                await self.giveGroundMove(loserUnit, followers)
            else:
                await self.fleeMove(loserUnit, outcome)

    async def giveGroundMove(self, loserUnit, followers):
        """The loser backs off 2" and anyone following up comes with it."""
        from drilled import before_move
        await before_move(self.game, loserUnit, 'Giving Ground')
        winners = [u for u in loserUnit.isInCombatWith if not u.bodyNP.isEmpty()]
        direction = self.giveGroundDirection(loserUnit, winners)
        moving = [loserUnit] + [f for f in followers if not f.bodyNP.isEmpty()]

        crashFraction = 1.0
        for unit in moving:
            hit = self.game.sweepTest(unit, direction, GIVE_GROUND)
            # Stop short only of something actually struck, so that a clear
            # path gives the full 2" the rule asks for.
            crashFraction = min(crashFraction,
                                hit if hit >= 1.0 else hit * (1.0 - CRASH_MARGIN))
        step = direction * (GIVE_GROUND * crashFraction)
        if self.surrounded(loserUnit, winners, step):
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

    def surrounded(self, loserUnit, winners, step):
        """Surrounded (p. 155): the Give Ground cannot break contact, so nobody
        moves and the combat is fought again next turn as if it had been drawn.

        Measured against where the winners stand now, not where a Follow Up
        will put them: a follower closing the gap again is a later choice, not
        a failure to break away. The unit that cannot move at all is only the
        obvious case -- one shoved half an inch by terrain and still touching
        is surrounded just the same.
        """
        psy = getattr(self.game, 'psychology', None)
        if psy is None or not winners:
            return False
        cx, cy, hx, hy, heading = psy._unit_box(loserUnit)
        after = (cx + step.x, cy + step.y, hx, hy, heading)
        stuck = [w for w in winners
                 if obb_distance(after, psy._unit_box(w)) <= CONTACT_GAP]
        if not stuck:
            return False
        names = ', '.join(w.unit.name for w in stuck)
        rule_log('Surrounded', loserUnit,
                 f"can give only {step.length():.1f}\" of its 2\" and would "
                 f"still be in base contact with {names}, so nobody moves: "
                 f"locked in place, fighting again next turn as if the combat "
                 f"had been a draw")
        loserUnit.surroundedThisPhase = True
        return True

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
        spent = getattr(loserUnit, 'fledThisPhase', False)
        bonus = False if spent else await self.swiftstrideChoice(
            loserUnit, kind, distance_to_edge=board_edge_distance(pos.x, pos.y))
        dice = await self.rollMoveDice(loserUnit, 3 if bonus else 2, bonus)
        distance = (flee_roll(dice, spent) if outcome == 'break'
                    else fall_back_roll(dice, spent))
        if spent:
            rule_log('The Limits of Endurance', loserUnit,
                     f"has already fled this phase, so it {kind}s 0\" instead "
                     f"of {sum(dice)}\"")
        else:
            print(f"{loserUnit.unit.name} {kind}s {distance}\" away from "
                  f"{winner.unit.name} (highest Unit Strength): {dice}")
        loserUnit.fledThisPhase = True

        direction = self.fleeAroundImpassable(loserUnit, direction, distance)
        before = Vec3(loserUnit.bodyNP.getPos())
        await self.game.fallBack2(loserUnit.bodyNP, direction, length=distance * 1.0,
                                  rally=(outcome == 'fall_back'),
                                  flee=(outcome == 'break'))
        self.nudgeOneInchApart(loserUnit, direction)
        after = Vec3(loserUnit.bodyNP.getPos())
        # A unit can be finished off by the ground it ran over, so the position
        # is taken before the tests that might remove it.
        self.fleeTerrainTests(loserUnit, before, after)
        self.perilTests(loserUnit, after - before)
        # The state is set after the move, as it was before the four passes:
        # a unit that Falls Back rallies at the end of it and is not fleeing.
        if not loserUnit.bodyNP.isEmpty():
            loserUnit.request("IsFleeing" if outcome == 'break' else "Moved")

    def enemiesOf(self, unit):
        """Every enemy unit still on the board."""
        ours = unit in self.game.player1Units
        return [u for u in self.game.units
                if u is not unit and not u.bodyNP.isEmpty()
                and (u in self.game.player1Units) != ours]

    def impassableAhead(self, unit, direction, distance):
        """Whether the unit's own base, turned to face *direction*, meets
        impassable terrain within *distance*.

        A swept box and not a centre line: a unit is wider than the line
        through the middle of it, and a house its centre misses is a house its
        flank still walks into. Only terrain is in the mask -- a fleeing unit
        goes *through* units, which is what the Peril tests are for.
        """
        body = unit.bodyNP
        pos, hpr = body.getPos(), body.getHpr()
        body.lookAt(pos + direction)
        fleeHpr = body.getHpr()
        body.setHpr(hpr)
        fraction, _ = self.game.movement.sweepTestDir(
            unit, TransformState.makePosHpr(pos, fleeHpr), direction, distance,
            mask=CM.TERRAIN_IMPASSABLE, pass_over=False)
        return fraction < 1.0

    def fleeAroundImpassable(self, unit, direction, distance):
        """Fleeing Through Terrain (p. 133): impassable terrain is gone around,
        "by the shortest possible route", not run through.
        """
        if unit.bodyNP.isEmpty() or distance < 0.05:
            return direction
        if unit.unit.model.is_flying():
            return direction
        if not self.impassableAhead(unit, direction, distance):
            return direction
        for turn in detour_angles():
            if turn == 0.0:
                continue
            x, y = turn_direction((direction.x, direction.y), turn)
            candidate = Vec3(x, y, 0)
            if self.impassableAhead(unit, candidate, distance):
                continue
            rule_log('Fleeing Through Terrain', unit,
                     f"impassable terrain blocks its {distance}\" flee, so it "
                     f"pivots {abs(turn):.0f} deg "
                     f"{'left' if turn > 0 else 'right'} to go around it")
            return candidate
        rule_log('Fleeing Through Terrain', unit,
                 f"impassable terrain blocks every route within 90 deg either "
                 f"way, so its {distance}\" flee runs straight into it")
        return direction

    def fleeTerrainTests(self, unit, from_pos, to_pos):
        """A flee takes no Movement penalty from difficult or dangerous terrain
        but still makes the Dangerous Terrain tests (p. 133)."""
        if unit.bodyNP.isEmpty():
            return
        self.game.movement.dangerousTerrainTests(unit, from_pos, to_pos)

    def perilTests(self, unit, displacement):
        """Peril tests (p. 133): a D6 for each model that fled *through* an
        enemy unit, losing a Wound on a 1-3.

        The models all shift by the same vector, so winding each one back by it
        gives the path it swept. There is no limit to how many tests a single
        move can call for.
        """
        psy = getattr(self.game, 'psychology', None)
        if psy is None or unit.bodyNP.isEmpty() or unit.model.isEmpty():
            return
        if displacement.length() < 0.05:
            return
        boxes = [(psy._unit_box(f), f) for f in self.enemiesOf(unit)]
        if not boxes:
            return

        through = {}
        for child in unit.model.getChildren():
            end = child.getPos(render)
            start = end - displacement
            for box, foe in boxes:
                if segment_crosses_box((start.x, start.y), (end.x, end.y), box):
                    through[id(foe)] = through.get(id(foe), 0) + 1
                    break
        if not through:
            # Crossing nobody is not the rule declining, it is the rule never
            # being reached, so there is nothing to report.
            return

        tests = sum(through.values())
        dice = [random.randint(1, 6) for _ in range(tests)]
        wounds = peril_wounds(dice)
        names = ', '.join(f.unit.name for _, f in boxes if id(f) in through)
        rule_log('Peril tests', unit,
                 f"{tests} model(s) fled through {names}: {dice} -> "
                 f"{wounds} wound(s) on a 1-3")
        if wounds:
            self.game.movement.applyWounds(unit, wounds)

    def nudgeOneInchApart(self, loserUnit, direction):
        """1" Apart (p. 154): a unit that Broke or Fell Back and is still in
        base contact is nudged apart by the smallest amount that restores the
        1" rule.

        The existing fall-back contact test only resolves overlap, which leaves
        the two touching rather than an inch clear. In practice this fires for a
        unit that fled 0" under The Limits of Endurance, one whose move was
        blocked, or one still close to an enemy other than the one it ran from.

        Every enemy counts, not only the one it was fighting: a flee move that
        ends within 1" of *any* enemy carries on until it is clear (p. 133).
        """
        psy = getattr(self.game, 'psychology', None)
        if psy is None or loserUnit.bodyNP.isEmpty():
            return
        enemies = self.enemiesOf(loserUnit)
        if not enemies:
            return
        step = Vec3(direction)
        step.z = 0
        if step.length() < 1e-6:
            return
        step.normalize()

        def gap():
            return min(obb_distance(psy._unit_box(loserUnit), psy._unit_box(e))
                       for e in enemies)

        moved = 0.0
        for _ in range(3):
            short = ONE_INCH_APART - gap()
            if short <= 0:
                break
            # Never shove the unit into terrain or a friend to make the room:
            # the sweep ignores the enemies it is backing away from.
            clear = self.game.sweepTest(loserUnit, step, short)
            room = short * clear
            if room < 0.05:
                break
            loserUnit.bodyNP.setPos(loserUnit.bodyNP.getPos() + step * room)
            moved += room

        left = gap()
        if moved:
            rule_log('1" Apart', loserUnit,
                     f"still too close after falling back: nudged {moved:.1f}\" "
                     f"further, leaving {left:.1f}\"")
        elif left < ONE_INCH_APART:
            rule_skipped('1" Apart', loserUnit,
                         f"only {left:.1f}\" clear, and there is nowhere to "
                         f"give — terrain or a friend is directly behind it")

    # ─── Post-Combat: pass 4, the pursuits ────────────────────────────────

    async def pursuitPass(self, outcomes, responses):
        """Pursuit moves are made one at a time, once every loser has moved
        (p. 156).

        A unit that restrained reforms here rather than at the moment it
        declared: the reform is a move, and until the loser has drawn off the
        two are still nose to nose with no room to turn in.
        """
        outcome_of = {id(u): o for u, o in outcomes}
        destinations = {id(response['target']): Vec3(response['target'].bodyNP.getPos())
                for response in responses if not response['target'].bodyNP.isEmpty()}
        for r in responses:
            winner, target = r['winner'], r['target']
            if winner.bodyNP.isEmpty():
                continue
            if r['action'] == 'restrain':
                if getattr(target, 'surroundedThisPhase', False):
                    # A drawn combat grants no reform, and the two are still
                    # nose to nose with no room to turn in anyway (p. 155).
                    rule_skipped('Restrain & Reform', winner,
                                 f"{target.unit.name} is Surrounded and never "
                                 f"drew off, so the combat continues as a draw")
                    continue
                await self.freeReform(winner)
                continue
            if r['action'] != 'pursue':
                continue
            if self.stillEngaged(winner, exclude=target):
                rule_skipped('Pursuit', winner,
                             "still in base contact with another enemy")
                continue
            destination = destinations.get(id(target))
            if destination is None:
                continue
            if target.bodyNP.isEmpty():
                rule_log('Pursuit', winner,
                         f'{target.unit.name} already caught; completes the declared pursuit '
                         f'towards ({destination.x:.2f}, {destination.y:.2f}) using current obstacles (p. 156)')
            await self.pursuitMove(winner, target, outcome_of.get(id(target)), destination=destination)

    async def pursuitMove(self, winner, target, outcome, *, destination=None):
        """Pivot to face the quarry and run the pursuit through the charge
        machinery, which rolls the 2D6, sums it, and handles the wheel, the
        align and the contact — the same things a charge needs."""
        targetPos = Vec3(destination) if destination is not None else target.bodyNP.getPos()
        rFrom = winner.bodyNP.getHpr()
        winner.bodyNP.lookAt(targetPos)
        rTo = winner.bodyNP.getHpr()
        winner.bodyNP.setHpr(rFrom)
        await LerpPosHprInterval(winner.bodyNP, duration=0.5,
                                 pos=winner.bodyNP.getPos(), hpr=rTo,
                                 blendType='easeInOut')

        winner.request("IsPursuing")
        winner.hasMovedThisTurn = False
        # The charge machinery resolves whatever the pursuer actually reaches,
        # and needs this to tell the quarry from a fresh enemy (p. 157).
        winner.pursuitQuarry = target
        self.game.autoCharge = True
        self.game.autoHold = True
        try:
            self.game.pathTowardsMouse(winner, targetPos.x, targetPos.y)
            movement_task = self.game.moveUnit(winner, wait_for_completion=True)
            if movement_task is not None and movement_task is not False:
                await movement_task
        finally:
            winner.pursuitQuarry = None
            self.game.autoCharge = self.game.autoHold = False

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

    async def freeReform(self, unit):
        """A free reform, awaited so the caller does not run on while the
        player is still placing the unit (p. 156, p. 157).

        The AI has no way to answer the interactive prompt, so it declines.
        Ordering is not decided here: several reforms fall due at once and the
        Panic pass raises its own, so they queue in ``startFreeReform``, which
        is the one point both paths pass through.
        """
        if self.game.aiControls(unit):
            return
        rule_log('Free Reform', unit, "may reform after the pursuit")
        finished = []
        self.game.startFreeReform(unit, on_done=lambda: finished.append(True))
        while not finished:
            await Task.pause(0.1)

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
