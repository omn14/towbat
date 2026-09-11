"""Shared loose-formation movement preview and commit (pp. 118, 123, 184-185)."""

from dataclasses import dataclass
import math

from panda3d.core import LineSegs, Point3, Vec2

from characters import side_of, slay_character
from psychology import _box_corners, obb_distance
from rules_log import rule_log, rule_skipped
from scouts import BOARD_HALF_DEPTH, BOARD_HALF_WIDTH, model_base_boxes
from skirmish import EPSILON, coherency_error, swept_base_overlaps
from skirmish_visibility import ChargeVisibility, charge_visibility
from terrain_system import dangerous_terrain_wounds
from toHitAndToWound import stat_value


@dataclass
class MovePreview:
    positions: list
    destination: tuple
    boxes: list
    distances: list
    allowance: float
    marched: bool
    terrain: list
    error: str | None = None
    charge_target: object | None = None
    charge_maximum: float | None = None
    charge_distance: float | None = None
    visibility: ChargeVisibility | None = None

    @property
    def distance(self):
        if self.charge_distance is not None:
            return self.charge_distance
        return max(self.distances, default=0.0)


def plotted_destination(game):
    position = game.arcPoint * 2 - Vec2(1, 1)
    return (position.x * 50, position.y * 50, 0)


def contact_target_at(game, unit, destination):
    """Probe Bullet contact without moving the live unit or spending its move."""
    origin = unit.bodyNP.getPos()
    try:
        unit.bodyNP.setPos(*destination)
        unit.bodyNP.node().setTransformDirty()
        contact = game.checkUnitContactSmall(unit)
        return game.getSelectedUnit(contact.getNode1()) if contact else None
    finally:
        unit.bodyNP.setPos(origin)
        unit.bodyNP.node().setTransformDirty()


def preview_action(game, unit):
    """Predict declaration, limited to M + 6 (+3 Swiftstride), p. 121."""
    from scouts import scout_charge_blocked
    from special_rules import max_charge_range, unit_has_swiftstride
    from vanguard import vanguard_charge_blocked
    preview = preview_move(game, unit, destination=plotted_destination(game))
    preview.charge_maximum = max_charge_range(preview.allowance, unit_has_swiftstride(unit))
    target = contact_target_at(game, unit, preview.destination)
    if target is not None and side_of(game, target) != side_of(game, unit):
        from skirmish_charge import first_contact, formed_contact, supported_pair, supported_formed_target
        preview.charge_target = target
        preview.marched = False
        preview.error = unavailable_reason(game, unit)
        preview.visibility = charge_visibility(game, unit, target)
        if not preview.visibility.allowed:
            preview.error = preview.error or preview.visibility.detail(target.unitName)
        if supported_pair(unit, target):
            preview.charge_distance = first_contact(model_base_boxes(unit), model_base_boxes(target))[2]
        elif supported_formed_target(unit, target):
            preview.charge_distance = formed_contact(
                model_base_boxes(unit), model_base_boxes(target), unit.bodyNP.getPos(game.render))[2]
        if preview.distance > preview.charge_maximum + 0.001:
            preview.error = f'Charge {preview.distance:.2f}" exceeds maximum {preview.charge_maximum:g}"'
        if getattr(unit, 'cannotChargeThisTurn', False):
            preview.error = 'Cannot charge this turn'
        if scout_charge_blocked(game, unit) or vanguard_charge_blocked(game, unit):
            preview.error = 'Cannot charge on the first own turn after Scouts or Vanguard'
    else:
        preview.error = unavailable_reason(game, unit) or preview.error
    return preview


def current_positions(unit):
    positions = [(record['x'], record['y']) for record in unit.skirmishLayout]
    character = getattr(unit, 'joinedCharacter', None)
    if character is not None:
        position = character.bodyNP.getPos(unit.bodyNP)
        positions.append((position.x, position.y))
    return positions


def preview_move(game, unit, positions=None, destination=None):
    """Read-only base paths and costs; flyers suffer landing terrain only (p. 170)."""
    positions = list(current_positions(unit) if positions is None else positions)
    root = unit.bodyNP.getTop()
    origin = unit.bodyNP.getPos(root)
    destination = tuple(origin if destination is None else destination)
    original = model_base_boxes(unit)
    if len(positions) != len(original):
        raise ValueError('A move must retain every living model')
    offset = Point3(*destination) - origin
    matrix = unit.bodyNP.getMat(root)
    boxes = []
    for position, old_box in zip(positions, original):
        world = matrix.xformPoint(Point3(position[0], position[1], 0)) + offset
        boxes.append((world.x, world.y, *old_box[2:]))
    distances = [math.hypot(after[0] - before[0], after[1] - before[1])
                 for before, after in zip(original, boxes)]
    error = coherency_error(boxes)
    participants = game.movement.movementParticipants(unit)
    flying = all(member.unit.model.is_flying() for member in participants)
    from special_rules import is_ethereal
    ethereal = all(is_ethereal(member.unit.model) for member in participants)
    pieces = getattr(getattr(game, 'terrain_manager', None), 'terrain_pieces', [])
    terrain = [[piece for piece in pieces if swept_base_overlaps(
        after if flying else before, after,
        (piece.center.x, piece.center.y, piece.width / 2, piece.height / 2, 0))]
        for before, after in zip(original, boxes)]
    modifier = min([0] + [piece.movement_modifier for features in terrain for piece in features])
    allowances = []
    for participant in participants:
        profile = participant.unit.model
        movement = profile.get_fly_movement(0) if flying else profile.get_movement(0)
        if not flying and not profile.is_move_through_cover() and not is_ethereal(profile) and modifier:
            movement = max(1, movement + modifier)
        allowances.append(movement)
    allowance = min(allowances)
    remaining = max(0.0, 2 * allowance - unit.moveSpentThisTurn)
    longest = max(distances, default=0)
    if longest > remaining + 1e-4:
        error = error or f'Longest model move {longest:.2f}" exceeds remaining {remaining:.2f}"'
    for index, (before, after, features) in enumerate(zip(original, boxes, terrain)):
        if any(abs(corner[0]) > BOARD_HALF_WIDTH + EPSILON or
               abs(corner[1]) > BOARD_HALF_DEPTH + EPSILON for corner in _box_corners(*after)):
            error = error or f'Model {index + 1} would leave the battlefield'
        for piece in features:
            if piece.is_impassable and (not (flying or ethereal) or obb_distance(
                    after, (piece.center.x, piece.center.y, piece.width / 2, piece.height / 2, 0)) <= EPSILON):
                error = error or f'Model {index + 1} is blocked by impassable terrain'
        for other in game.units:
            if (other is unit or getattr(other, 'hostUnit', None) is not None
                    or not other.isDeployed or other.unit.nmodels <= 0):
                continue
            centre = other.bodyNP.getPos(root)
            obstacle = (centre.x, centre.y, other.unitWidth / 2, other.unitHeight / 2,
                        other.bodyNP.getH(root))
            if ((not flying and swept_base_overlaps(before, after, obstacle))
                    or (flying and obb_distance(after, obstacle) <= EPSILON)):
                error = error or f'Model {index + 1} is blocked by {other.unit.name}'
            if side_of(game, other) != side_of(game, unit):
                if any(obb_distance(after, enemy) < 1 - EPSILON for enemy in model_base_boxes(other)):
                    error = error or f'Model {index + 1} must finish at least 1" from {other.unit.name}'
    marched = longest > max(0, allowance - unit.moveSpentThisTurn) + EPSILON
    return MovePreview(positions, destination, boxes, distances, allowance, marched, terrain, error)


def limited_move_preview(game, unit, destination):
    """Clamp a straight destination to the shared legal-move check (pp. 123, 185)."""
    origin = unit.bodyNP.getPos(unit.bodyNP.getTop())
    offset = Point3(*destination) - origin
    preview = preview_move(game, unit, destination=destination)
    if preview.error is None:
        return preview
    length = offset.length()
    if length <= EPSILON:
        return preview
    maximum = min(length, max(0.0, 2 * preview.allowance - unit.moveSpentThisTurn))
    direction = offset / length
    candidate = preview_move(game, unit, destination=origin + direction * maximum)
    if candidate.error is None:
        return candidate
    nearest = preview_move(game, unit, destination=origin)
    if nearest.error:
        return preview
    lower, upper = 0.0, maximum
    for _ in range(14):
        distance = (lower + upper) / 2
        candidate = preview_move(game, unit, destination=origin + direction * distance)
        if candidate.error:
            upper = distance
        else:
            lower, nearest = distance, candidate
    return nearest


def unavailable_reason(game, unit):
    if unit is None or not getattr(unit, 'isSkirmisher', False):
        return 'Select a unit in Skirmish formation'
    if game.fsm.state != 'MovementPhase':
        return 'Formation adjustment is available during Movement'
    if side_of(game, unit) != game.roundCounter.current_player:
        return 'Not the active player\'s unit'
    if not unit.isDeployed or getattr(unit, 'hostUnit', None) is not None:
        return 'Select a deployed unit, not an attached character'
    if unit.state != 'Idle' or unit.hasMovedThisTurn or unit.isInCombat or unit.skirmishCombat:
        return 'This unit cannot make an ordinary move now'
    if getattr(game, 'awaitingChoice', False):
        return 'Finish the current choice first'
    return None


def commit_move(game, unit, positions=None, destination=None):
    """Validate again, then spend one move; ghost manipulation never rolls dice."""
    from charge_declarations import ordinary_move_allowed
    if not ordinary_move_allowed(game):
        return False
    reason = unavailable_reason(game, unit)
    if reason:
        rule_skipped('Skirmishers', unit, reason)
        return False
    preview = preview_move(game, unit, positions, destination)
    if preview.error or preview.distance <= EPSILON:
        rule_skipped('Skirmishers', unit, preview.error or 'no model changed position; movement retained')
        return False
    if preview.marched:
        from marching import request_march
        if not request_march(game, unit, lambda: commit_move(game, unit, preview.positions, preview.destination)):
            return False
    origin = unit.bodyNP.getPos()
    unit.bodyNP.setPos(*preview.destination)
    for record, position in zip(unit.skirmishLayout, preview.positions):
        record['x'], record['y'] = position
    character = getattr(unit, 'joinedCharacter', None)
    if character is not None:
        unit.skirmishCharacterPosition = list(preview.positions[-1])
        unit.placeCharacter()
    unit.rebuildFootprint()
    unit.bodyNP.node().setTransformDirty()
    unit.moveSpentThisTurn += preview.distance
    unit.marchedThisTurn = unit.marchedThisTurn or preview.marched
    unit.hasMovedThisTurn = True
    game.movement.alignModelsToHillNormal(unit)
    rule_log('Skirmishers', unit,
             f'{len(preview.boxes)} models remain coherent; longest move {preview.distance:.2f}" '
             f'of M{preview.allowance:g}{" (march)" if preview.marched else ""} (pp. 123, 184-185)')
    game.movement.movementAllowance(
        unit, features=[piece for crossed in preview.terrain for piece in crossed], log=True)
    if preview.marched:
        rule_log('Marching', unit, 'per-model movement exceeded M; shooting restrictions apply (p. 123)')
    for participant, features in ((unit, preview.terrain[:unit.unit.nmodels]),
                                  (character, preview.terrain[unit.unit.nmodels:])):
        if participant is None or participant not in game.units:
            continue
        tests = sum(sum(piece.is_dangerous for piece in crossed) for crossed in features)
        from special_rules import is_ethereal
        if tests and is_ethereal(participant.unit.model):
            rule_log('Ethereal', participant, f'open ground: skips {tests} dangerous-terrain tests (p. 167)')
            continue
        wounds = dangerous_terrain_wounds(tests, 1,
                                         reroll_sources=participant.unit.model.dangerous_terrain_reroll_sources(),
                                         subject=participant)
        if tests:
            rule_log('Dangerous Terrain', participant,
                     f'{tests} model/feature tests -> {wounds} wounds (p. 269)')
            if participant is character:
                participant.woundsOnModel += wounds
                if participant.woundsOnModel >= max(1, stat_value(
                        participant.unit.model.characteristics.get('W'), 1)):
                    slay_character(game, participant)
            else:
                game.movement.applyWounds(participant, wounds)
    if unit in game.units:
        game.movement.magicalVortexTests(unit, origin, unit.bodyNP.getPos())
    if unit in game.units:
        game.movement.updateDisrupted(unit)
        unit.request('Moved')
        game.refreshSelectedUnit()
    return True


def preview_colour(preview):
    if preview.error:
        return (0.95, 0.25, 0.2, 1)
    if preview.charge_target is not None:
        return (0.15, 0.85, 1, 1)
    return (1, 0.72, 0.25, 1) if preview.marched else (0.3, 1, 0.65, 1)


def draw_preview(parent, preview):
    """Draw legal-base outlines unshaded, after the board and miniatures."""
    lines = LineSegs('skirmish-preview')
    lines.setThickness(3)
    lines.setColor(*preview_colour(preview))
    for box in preview.boxes:
        corners = _box_corners(*box)
        lines.moveTo(*corners[0], 0.3)
        for corner in corners[1:] + corners[:1]:
            lines.drawTo(*corner, 0.3)
    node = parent.attachNewNode(lines.create())
    node.setLightOff()
    node.setShaderOff()
    node.setBin('fixed', 40)
    node.setDepthTest(False)
    node.setDepthWrite(False)
    return node