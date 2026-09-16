"""Remaining Moves character attachment, departure and base previews (Rulebook p. 207)."""

import math

from panda3d.core import Point3
from direct.showbase.DirectObject import DirectObject
from shapely import shortest_line
from shapely.geometry import Polygon

from battlefield import battlefield_for
from characters import (get_joined_characters, join_reason, join_unit, leave_reason,
                        rank_placements, release_character, side_of)
from psychology import _box_corners, obb_distance
from rules_log import rule_log, rule_skipped
from scouts import model_base_boxes
from skirmish import EPSILON, coherency_error, swept_base_overlaps
from skirmish_movement import MovePreview
from special_rules import is_ethereal
from terrain_system import terrain_obstacle, terrain_path_contact


def polygon(box):
    return Polygon(_box_corners(*box))


def joined_boxes(host, character, contact):
    """Predict the final rank geometry without parenting or moving live nodes (p. 207)."""
    if host.isSkirmisher and not host.skirmishCombat:
        return [*model_base_boxes(host), contact]
    from command_groups import command_positions, living_command
    members = [*get_joined_characters(host), character]
    commands = command_positions(host)
    slots, placements = rank_placements(host.unit.nmodels, host.unit.files, host.modelWidth, host.modelHeight,
        [(member.unitName, member.modelWidth, member.modelHeight,
          getattr(member, 'retiredFromCombat', False)) for member in members],
        [commands[id(entry)] for entry in living_command(host)])
    matrix = host.model.getMat(host.bodyNP.getTop())
    heading = host.bodyNP.getH(host.bodyNP.getTop())
    boxes = []
    for slot in slots:
        position = matrix.xformPoint(Point3(slot % host.unit.files * host.modelWidth,
                                           -(slot // host.unit.files) * host.modelHeight, 0))
        boxes.append((position.x, position.y, host.modelWidth / 2, host.modelHeight / 2, heading))
    for member in members:
        record = placements[member.unitName]
        position = matrix.xformPoint(Point3(record['x'], record['y'], 0))
        boxes.append((position.x, position.y, member.modelWidth / 2, member.modelHeight / 2, heading))
    return boxes


def preview(game, character, *, host=None, destination=None):
    """A contact move joins; a departure is measured from the character's own base (pp. 123, 207)."""
    joining = host is not None
    error = join_reason(game, character, host, movement=True) if joining else leave_reason(game, character)
    original_host = getattr(character, 'hostUnit', None)
    root = character.bodyNP.getTop()
    origin = character.bodyNP.getPos(root)
    if error:
        return MovePreview([], tuple(origin), [], [], 0, False, [], error)
    source = model_base_boxes(character)[0]
    if joining:
        segment = min((shortest_line(polygon(source), polygon(box)) for box in model_base_boxes(host)),
                      key=lambda line: line.length)
        coordinates = list(segment.coords)
        start, end = coordinates if coordinates else ((0, 0), (0, 0))
        destination = (origin.x + end[0] - start[0], origin.y + end[1] - start[1], origin.z)
    destination = tuple(origin if destination is None else destination)
    offset = Point3(*destination) - origin
    contact = (source[0] + offset.x, source[1] + offset.y, *source[2:])
    distance = math.hypot(offset.x, offset.y)
    profile = character.unit.model
    flying = profile.is_flying()
    from copy import copy
    independent = copy(profile)
    independent._joined_skirmish = None
    restricted_translation = False
    if not independent.is_skirmisher():
        forward = character.bodyNP.getQuat(root).getForward()
        right = character.bodyNP.getQuat(root).getRight()
        sideways, longitudinal = abs(offset.dot(right)), offset.dot(forward)
        if sideways > EPSILON and abs(longitudinal) > EPSILON:
            error = 'formed characters must wheel before moving diagonally (p. 123)'
        restricted_translation = sideways > EPSILON or longitudinal < -EPSILON
    pieces = getattr(getattr(game, 'terrain_manager', None), 'terrain_pieces', [])
    features = [piece for piece in pieces if terrain_path_contact(piece, contact if flying else source, contact)]
    from tempest import tempest_features
    features = tempest_features(game, character, source, contact, features, base_paths=[(source, contact)])
    allowance = profile.get_fly_movement(0) if flying else profile.get_movement(0)
    penalty = min([0, *[piece.movement_modifier for piece in features]])
    if penalty and not (flying or is_ethereal(profile) or profile.is_move_through_cover()):
        allowance = max(1, allowance + penalty)
    spent = getattr(character, 'moveSpentThisTurn', 0)
    maximum = (allowance / 2 if restricted_translation else 2 * allowance) - spent
    if getattr(character, 'marchTestResult', None) == 'failed':
        maximum = min(maximum, allowance - spent)
    if distance > maximum + EPSILON:
        error = error or f'{distance:.2f}" exceeds remaining movement {max(0, maximum):.2f}"'
    if not joining and distance <= EPSILON:
        error = error or 'character must move out of the unit'
    for other in game.units:
        if (other is character or not other.isDeployed or other.unit.nmodels <= 0
                or getattr(other, 'hostUnit', None) is not None):
            continue
        for box in model_base_boxes(other):
            if other is original_host:
                continue
            if other is host:
                if polygon(contact).intersection(polygon(box)).area > EPSILON:
                    error = error or 'contact move crosses the host'
                continue
            blocked = polygon(contact).intersection(polygon(box)).area > EPSILON
            if not flying:
                blocked = blocked or swept_base_overlaps(source, contact, box)
            if blocked:
                error = error or f'path blocked by {other.unit.name}'
    for piece in features:
        from chariot_terrain import linear_impassable
        if (piece.is_impassable or linear_impassable(piece, character)) and not (flying or is_ethereal(profile)):
            error = error or f'path blocked by {piece.terrain_type}'
    boxes = joined_boxes(host, character, contact) if joining else [contact]
    if joining and host.isSkirmisher and not host.skirmishCombat:
        error = error or coherency_error(boxes)
    for box in boxes:
        if not battlefield_for(game).contains_box(box, EPSILON):
            error = error or 'final formation would leave the battlefield'
        for other in game.units:
            if (other is character or other is host or not other.isDeployed or other.unit.nmodels <= 0
                    or getattr(other, 'hostUnit', None) is not None):
                continue
            obstacles = model_base_boxes(other)
            if other is original_host:
                obstacles = obstacles[:other.unit.nmodels] + [base for member in get_joined_characters(other)
                                                            if member is not character for base in model_base_boxes(member)]
            for obstacle in obstacles:
                if polygon(box).intersection(polygon(obstacle)).area > EPSILON:
                    error = error or f'final bases overlap {other.unit.name}'
                if side_of(game, other) != side_of(game, character) and obb_distance(box, obstacle) < 1 - EPSILON:
                    error = error or f'must finish at least 1" from {other.unit.name}'
        for piece in pieces:
            from chariot_terrain import linear_impassable
            if (piece.is_impassable or linear_impassable(piece, character)) and obb_distance(box, terrain_obstacle(piece)) <= EPSILON:
                error = error or f'final bases overlap impassable {piece.terrain_type}'
    return MovePreview([], destination, boxes, [distance], allowance, distance > allowance - spent + EPSILON,
                       [features], error)


def commit(game, character, *, host=None, destination=None):
    """Commit only after a fresh preview; cancellation and invalid previews spend nothing (p. 207)."""
    result = preview(game, character, host=host, destination=destination)
    action = 'Joining a Unit' if host is not None else 'Leaving a Unit'
    if result.error:
        rule_skipped(action, character, result.error)
        return False
    if result.marched:
        from marching import request_march
        if not request_march(game, character, lambda: commit(game, character, host=host, destination=destination)):
            return False
    old_host = getattr(character, 'hostUnit', None)
    origin = character.bodyNP.getPos(character.bodyNP.getTop())
    if old_host is not None:
        anchor = old_host.model.getPos()
        release_character(game, character)
        old_host.layOutRanks()
        old_host.rebuildFootprint()
        if not old_host.isSkirmisher or old_host.skirmishCombat:
            old_host.bodyNP.setPos(old_host.bodyNP, anchor - old_host.model.getPos())
        old_host.placeCharacter()
    character.bodyNP.setPos(character.bodyNP.getTop(), Point3(*result.destination))
    character.bodyNP.node().setTransformDirty()
    character.moveSpentThisTurn += result.distance
    character.hasMovedThisTurn = True
    character.marchedThisTurn = character.marchedThisTurn or result.marched
    game.movement.dangerousTerrainTests(character, origin, Point3(*result.destination), features=result.terrain[0])
    if character not in game.units or character.unit.nmodels <= 0:
        rule_log(action, character, 'character was lost on the movement path')
        return True
    if host is not None:
        contact = character.bodyNP.getPos(character.bodyNP.getTop())
        if not join_unit(game, character, host):
            return False
        if host.isSkirmisher and not host.skirmishCombat:
            local = host.bodyNP.getRelativePoint(host.bodyNP.getTop(), contact)
            character.joinedPosition = [local.x, local.y]
            if get_joined_characters(host)[0] is character:
                host.skirmishCharacterPosition = character.joinedPosition
            host.placeCharacter()
            host.rebuildFootprint()
        host.joinedMovementLocked = True
    else:
        character.request('Moved')
    game.movement.alignModelsToHillNormal(character)
    for member in (old_host, host):
        if member is not None:
            game.movement.alignModelsToHillNormal(member)
    game.roundCounter.apply_selection_masks()
    rule_log(action, character, f'{result.distance:.2f}" of M{result.allowance:g}'
             + (' (march)' if result.marched else '')
             + (f'; joins {host.unit.name}, host cannot move again this phase, '
                f'host moved={host.hasMovedThisTurn} (p. 207)' if host is not None
                else f'; leaves {old_host.unit.name} before the host moves (p. 207)'))
    game.refreshSelectedUnit()
    return True


def refresh_controls(game, unit):
    from direct.gui.DirectGui import DirectButton, DGG
    from characters import is_character, remaining_move_reason
    import gui_theme as theme
    button = getattr(game, 'characterMoveButton', None)
    if button is None:
        button = DirectButton(parent=game.a2dTopLeft, text='', text_font=theme.get_font(),
            text_fg=theme.BTN_TEXT, text_scale=.030, text_pos=(.21, -.024),
            frameColor=theme.BTN_RED, frameSize=(0, .42, -.055, .012),
            relief=DGG.FLAT, pos=(.04, 0, -.15), command=lambda: open_editor(game))
        game.characterMoveButton = button
    attached = get_joined_characters(unit) if unit is not None else []
    eligible = unit is not None and (attached or is_character(unit))
    blocked = not eligible or remaining_move_reason(game, unit)
    if blocked or getattr(game, 'characterMoveEditor', None) or getattr(game, 'skirmishEditor', None):
        button.hide()
    else:
        button['text'] = 'Leave unit' if attached else 'Join unit'
        button.show()


def try_ai_join(game, character):
    """Use the same legal contact move for nearby compatible escorts (p. 207)."""
    from characters import is_character, remaining_move_reason
    if not is_character(character) or remaining_move_reason(game, character):
        return False
    choices = []
    for host in game.units:
        if join_reason(game, character, host, movement=True) is not None:
            continue
        result = preview(game, character, host=host)
        if not result.error and not result.marched:
            choices.append((result.distance, host))
    if not choices:
        return False
    return commit(game, character, host=min(choices, key=lambda choice: choice[0])[1])


def open_editor(game):
    unit = getattr(game, 'unitToMove', None)
    from characters import is_character, remaining_move_reason
    if (unit is None or remaining_move_reason(game, unit) or getattr(game, 'characterMoveEditor', None)
            or getattr(game, 'skirmishEditor', None)):
        return None
    attached = get_joined_characters(unit)
    if not attached and not is_character(unit):
        return None
    return CharacterMoveEditor(game, unit, attached)


class CharacterMoveEditor(DirectObject):
    def __init__(self, game, unit, attached):
        from direct.gui.DirectGui import DirectButton, DirectFrame, DirectOptionMenu, DGG
        from panda3d.core import TextNode
        from skirmish_ui import clear_plot_preview
        import gui_theme as theme
        super().__init__()
        self.game, self.unit = game, unit
        self.joining = not attached
        self.candidates = attached or [host for host in game.units if host.isDeployed
                                      and join_reason(game, unit, host) is None]
        self.character = unit if self.joining else attached[0]
        self.host = self.candidates[0] if self.joining and self.candidates else None
        self.destination = tuple(self.character.bodyNP.getPos(game.render))
        self.ghost = None
        self.ready = False
        game.characterMoveEditor = self
        game.taskMgr.remove('taskLoopPathTowardsMouse')
        clear_plot_preview(game)
        for name in ('skirmishAdjustButton', 'characterMoveButton'):
            if getattr(game, name, None) is not None:
                getattr(game, name).hide()
        self.panel = DirectFrame(parent=game.a2dTopLeft, frameTexture=theme.TEX_PARCHMENT,
            frameColor=(1, 1, 1, 1), frameSize=(0, .72, -.46, 0), relief=DGG.FLAT, pos=(.04, 0, -.15))
        self.title = theme.styled_text(text='Join unit' if self.joining else 'Leave unit', parent=self.panel,
            pos=(.03, -.05), scale=.034, fg=theme.INK, align=TextNode.ALeft)
        labels = [f'{member.unit.name} [{index + 1}]' for index, member in enumerate(self.candidates)]
        self.menu = DirectOptionMenu(parent=self.panel, items=labels or ['No eligible units'],
            scale=.030, pos=(.03, 0, -.11), text_font=theme.get_font(), text_fg=theme.INK,
            frameColor=theme.PARCHMENT_DARK, popupMarker_borderWidth=(.02, .02),
            popupMarker_scale=.3, command=self.choose)
        self.menu.setScale(min(.030, .65 / max(1, self.menu.getWidth())))
        self.status = theme.styled_text(text='', parent=self.panel, pos=(.03, -.20), scale=.030,
                                       fg=theme.INK, align=TextNode.ALeft, wordwrap=21)
        self.confirm_button = DirectButton(parent=self.panel, text='Confirm', text_font=theme.get_font(),
            text_fg=theme.BTN_TEXT, text_scale=.030, text_pos=(0, -.01), frameColor=theme.BTN_RED,
            frameSize=(-.14, .14, -.03, .03), relief=DGG.FLAT, pos=(.19, 0, -.40), command=self.confirm)
        self.cancel_button = DirectButton(parent=self.panel, text='Cancel', text_font=theme.get_font(),
            text_fg=theme.BTN_TEXT, text_scale=.030, text_pos=(0, -.01), frameColor=theme.BTN_RED,
            frameSize=(-.14, .14, -.03, .03), relief=DGG.FLAT, pos=(.53, 0, -.40), command=self.cancel)
        self.accept('mouse1', self.place)
        self.accept('escape', self.cancel)
        self.ready = True
        self.redraw()

    def choose(self, _label):
        if not self.ready or not self.candidates:
            return
        chosen = self.candidates[self.menu.selectedIndex]
        if self.joining:
            self.host = chosen
        else:
            self.character = chosen
            self.destination = tuple(chosen.bodyNP.getPos(self.game.render))
        self.redraw()

    def place(self):
        if self.joining:
            return
        watcher = self.game.mouseWatcherNode
        if not watcher.hasMouse():
            return
        mouse = watcher.getMouse()
        horizontal, vertical = (mouse.x + 1) * self.game.getAspectRatio(), 1 - mouse.y
        if .04 <= horizontal <= .76 and .15 <= vertical <= .15 - self.panel['frameSize'][2]:
            return
        near, far = Point3(), Point3()
        self.game.camLens.extrude(mouse, near, far)
        near = self.game.render.getRelativePoint(self.game.cam, near)
        far = self.game.render.getRelativePoint(self.game.cam, far)
        direction = far - near
        if abs(direction.z) < 1e-6:
            return
        self.destination = tuple(near - direction * (near.z / direction.z))
        self.redraw()

    def redraw(self):
        from direct.gui.DirectGui import DGG
        from skirmish_movement import draw_preview
        if self.ghost is not None:
            self.ghost.removeNode()
            self.ghost = None
        self.result = preview(self.game, self.character, host=self.host, destination=self.destination)
        if not self.candidates:
            self.result.error = 'No eligible units'
        self.status.setText(self.result.error or f'{self.character.unit.name}\n'
                            f'{self.result.distance:.2f}" / M{self.result.allowance:g}'
                            + (' (march)' if self.result.marched else ''))
        bounds = self.status.getTightBounds(self.panel)
        button_height = min(-.40, bounds[0].z - .08) if bounds else -.40
        self.confirm_button.setZ(button_height)
        self.cancel_button.setZ(button_height)
        self.panel['frameSize'] = (0, .72, button_height - .06, 0)
        self.confirm_button['state'] = DGG.DISABLED if self.result.error else DGG.NORMAL
        self.ghost = draw_preview(self.game.render, self.result)

    def confirm(self):
        if self.result.error:
            return False
        accepted = commit(self.game, self.character, host=self.host, destination=self.destination)
        if accepted or getattr(self.character, 'marchTestResult', None) == 'pending':
            self.close()
        else:
            self.redraw()
        return accepted

    def cancel(self):
        self.close()

    def close(self):
        self.ignoreAll()
        self.panel.destroy()
        if self.ghost is not None:
            self.ghost.removeNode()
        self.game.characterMoveEditor = None
        self.game.refreshSelectedUnit()