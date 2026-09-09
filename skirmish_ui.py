"""Optional ghost-only formation adjustment using the shared movement validator."""

import math

from direct.gui.DirectGui import DGG, DirectButton, DirectFrame, DirectLabel, DirectRadioButton, DirectSlider
from direct.showbase.DirectObject import DirectObject
from panda3d.core import Point3, TextNode

import gui_theme as theme
from psychology import obb_distance
from skirmish import EPSILON, layout_positions
from skirmish_movement import (commit_move, current_positions, draw_preview,
                               preview_action, preview_move, unavailable_reason)


def clear_plot_preview(game):
    range_input = game.ground.getShaderInput('skirmishRangeActive')
    if range_input.getVector().x:
        game.setGroundOverlay(False)
    if getattr(game, 'skirmishMoveStatus', None) is not None:
        game.skirmishMoveStatus.hide()
    if getattr(game, 'skirmMoveGhost', None) is not None:
        game.skirmMoveGhost.removeNode()
        game.skirmMoveGhost = None


def show_plot_status(game, preview):
    status = getattr(game, 'skirmishMoveStatus', None)
    if status is None:
        status = DirectLabel(
            parent=game.a2dTopLeft, text='', text_font=theme.get_font(),
            text_fg=theme.INK, text_scale=0.035, text_align=TextNode.ALeft,
            text_wordwrap=28, frameTexture=theme.TEX_PARCHMENT,
            frameColor=(1, 1, 1, 1), pad=(0.025, 0.02), relief=DGG.FLAT,
            pos=(0.065, 0, -0.20))
        game.skirmishMoveStatus = status
    if preview.error:
        text = f'BLOCKED\n{preview.error}'
    elif preview.charge_target is not None:
        text = (f'CHARGE: {preview.charge_target.unitName}\n'
            f'{preview.distance:.2f}" to contact / max {preview.charge_maximum:g}"')
        if preview.visibility is not None:
            text += f'\n{preview.visibility.visible}/{preview.visibility.total} models can see'
    else:
        action = 'MARCH' if preview.marched else 'MOVE'
        text = f'{action} (not a charge)\n{preview.distance:.2f}" / M{preview.allowance:g}'
    status['text'] = text
    status.resetFrameSize()
    status.show()


def confirm_plotted_move(game, unit):
    """A human click stages an ordinary move; only Confirm spends it."""
    if game.arcPoint is None:
        return
    preview = preview_action(game, unit)
    show_plot_status(game, preview)
    if preview.error:
        if preview.visibility is not None and not preview.visibility.allowed:
            from rules_log import rule_skipped
            rule_skipped('Skirmishers', unit,
                         f'charge refused: {preview.error}; movement retained (p. 186)')
        return
    if preview.charge_target is not None:
        game.moveUnit(unit)
    else:
        SkirmishEditor(game, unit, destination=preview.destination)


def refresh_adjust_button(game, unit):
    clear_plot_preview(game)
    button = getattr(game, 'skirmishAdjustButton', None)
    if button is None:
        button = DirectButton(
            parent=game.a2dTopLeft, text='Adjust formation',
            text_font=theme.get_font(), text_fg=theme.BTN_TEXT,
            text_scale=0.032, text_pos=(0.31, -0.027),
            frameColor=theme.BTN_RED, frameSize=(0, 0.62, -0.055, 0.015),
            relief=DGG.FLAT, pos=(0.04, 0, -0.07),
            command=lambda: open_editor(game))
        game.skirmishAdjustButton = button
    if unavailable_reason(game, unit) or getattr(game, 'skirmishEditor', None):
        button.hide()
    else:
        button.show()


def open_editor(game):
    unit = getattr(game, 'unitToMove', None)
    if unavailable_reason(game, unit) or getattr(game, 'skirmishEditor', None):
        return None
    return SkirmishEditor(game, unit)


class SkirmishEditor(DirectObject):
    def __init__(self, game, unit, *, destination=None):
        DirectObject.__init__(self)
        self.game, self.unit = game, unit
        self.positions = current_positions(unit)
        self.destination = tuple(unit.bodyNP.getPos(game.render) if destination is None else destination)
        self.dragging = None
        self.ghost = None
        self.mode = ['group']
        self.ready = False
        game.skirmishEditor = self
        game.taskMgr.remove('taskLoopPathTowardsMouse')
        game.setGroundOverlay(False)
        clear_plot_preview(game)
        if getattr(game, 'skirmishAdjustButton', None):
            game.skirmishAdjustButton.hide()
        self.panel = DirectFrame(
            parent=game.a2dTopLeft, frameTexture=theme.TEX_PARCHMENT,
            frameColor=(1, 1, 1, 1), frameSize=(0, 0.70, -0.77, 0),
            pos=(0.04, 0, -0.07), relief=DGG.FLAT)
        self.label(unit.unit.name, 0.03, -0.048, scale=0.032)
        group = DirectRadioButton(
            parent=self.panel, text='Move group', text_font=theme.get_font(),
            text_fg=theme.INK, variable=self.mode, value=['group'],
            text_align=TextNode.ALeft, text_pos=(0.6, 0),
            indicator_pos=(0, 0, 0.25), relief=None, frameColor=(0, 0, 0, 0),
            scale=0.032, pos=(0.06, 0, -0.11))
        models = DirectRadioButton(
            parent=self.panel, text='Place models', text_font=theme.get_font(),
            text_fg=theme.INK, variable=self.mode, value=['models'],
            text_align=TextNode.ALeft, text_pos=(0.6, 0),
            indicator_pos=(0, 0, 0.25), relief=None, frameColor=(0, 0, 0, 0),
            scale=0.032, pos=(0.39, 0, -0.11))
        self.mode_buttons = [group, models]
        group.setOthers([models])
        models.setOthers([group])
        self.labels = {}
        self.sliders = {}
        specifications = [('width', 'Models across', (1, max(2, unit.unit.nmodels)),
                           math.ceil(math.sqrt(unit.unit.nmodels))),
                          ('gap', 'Gap', (0.1, 1.0), 0.6),
                          ('angle', 'Angle', (-180, 180), 0)]
        for index, (key, title, limits, value) in enumerate(specifications):
            vertical = -0.20 - index * 0.095
            self.labels[key] = self.label(title, 0.03, vertical, scale=0.026)
            self.sliders[key] = DirectSlider(
                parent=self.panel, range=limits, value=value,
                pageSize=1 if key != 'gap' else 0.1,
                scale=0.23, pos=(0.43, 0, vertical + 0.008),
                frameSize=(-1, 1, -0.028, 0.028),
                thumb_frameSize=(-0.05, 0.05, -0.065, 0.065),
                command=self.reshape)
        self.status = self.label('', 0.03, -0.50, scale=0.029)
        self.confirm_button = self.button('Confirm move', 0.19, -0.70, self.confirm)
        self.button('Cancel', 0.53, -0.70, self.cancel)
        self.accept('mouse1', self.press)
        self.accept('mouse1-up', self.release)
        self.ready = True
        self.redraw()
        game.taskMgr.add(self.update, 'skirmishEditor')

    def label(self, text, horizontal, vertical, *, scale):
        return theme.styled_text(text=text, parent=self.panel,
                                 pos=(horizontal, vertical), scale=scale,
                                 fg=theme.INK, align=TextNode.ALeft,
                                 wordwrap=0.64 / scale)

    def button(self, text, horizontal, vertical, command):
        return DirectButton(
            parent=self.panel, text=text, text_font=theme.get_font(),
            text_fg=theme.BTN_TEXT, text_scale=0.028, text_pos=(0, -0.010),
            frameColor=theme.BTN_RED, frameSize=(-0.145, 0.145, -0.028, 0.028),
            relief=DGG.FLAT, pos=(horizontal, 0, vertical), command=command)

    def reshape(self):
        if not self.ready:
            return
        width = min(self.unit.unit.nmodels, max(1, round(self.sliders['width']['value'])))
        gap = self.sliders['gap']['value']
        angle = math.radians(self.sliders['angle']['value'])
        positions = layout_positions(self.unit.unit.nmodels, self.unit.modelWidth,
                                     self.unit.modelHeight, width, gap)
        self.positions[:self.unit.unit.nmodels] = [
            (point[0] * math.cos(angle) - point[1] * math.sin(angle),
             point[0] * math.sin(angle) + point[1] * math.cos(angle)) for point in positions]
        self.labels['width'].setText(f'Across: {width}')
        self.labels['gap'].setText(f'Gap: {gap:.2f}"')
        self.labels['angle'].setText(f'Angle: {math.degrees(angle):.0f}')
        self.redraw()

    def pointer(self):
        watcher = self.game.mouseWatcherNode
        if not watcher.hasMouse():
            return None
        mouse = watcher.getMouse()
        horizontal = (mouse.x + 1) * self.game.getAspectRatio()
        vertical = 1 - mouse.y
        if 0.04 <= horizontal <= 0.74 and 0.07 <= vertical <= 0.84:
            return None
        near, far = Point3(), Point3()
        self.game.camLens.extrude(mouse, near, far)
        near = self.game.render.getRelativePoint(self.game.cam, near)
        far = self.game.render.getRelativePoint(self.game.cam, far)
        direction = far - near
        if abs(direction.z) < 1e-6:
            return None
        return near - direction * (near.z / direction.z)

    def press(self):
        point = self.pointer()
        if point is None:
            return
        if self.mode[0] == 'group':
            self.dragging = 'group'
            self.drag_offset = Point3(*self.destination) - point
        else:
            cursor = (point.x, point.y, 0.001, 0.001, 0)
            closest = min(range(len(self.preview.boxes)),
                          key=lambda index: obb_distance(cursor, self.preview.boxes[index]))
            if obb_distance(cursor, self.preview.boxes[closest]) <= 0.35:
                self.dragging = closest

    def release(self):
        self.dragging = None

    def update(self, task):
        if self.unit not in self.game.units or self.game.fsm.state != 'MovementPhase':
            self.close(resume=False)
            return task.done
        point = self.pointer() if self.dragging is not None else None
        if point is not None:
            if self.dragging == 'group':
                self.destination = tuple(point + self.drag_offset)
            else:
                origin = self.unit.bodyNP.getPos(self.game.render)
                offset = Point3(*self.destination) - origin
                local = self.unit.bodyNP.getRelativePoint(self.game.render, point - offset)
                self.positions[self.dragging] = (local.x, local.y)
            self.redraw()
        return task.cont

    def redraw(self):
        self.preview = preview_move(self.game, self.unit, self.positions, self.destination)
        if self.ghost is not None:
            self.ghost.removeNode()
        self.ghost = draw_preview(self.game.render, self.preview)
        if self.preview.error:
            self.status.setText(self.preview.error)
        else:
            kind = 'March' if self.preview.marched else 'Move'
            self.status.setText(f'{kind.upper()} (not a charge)\n'
                                f'Coherent: {len(self.positions)} models\n'
                                f'{kind}: {self.preview.distance:.2f}" / M{self.preview.allowance:g}')
        self.confirm_button['state'] = (DGG.DISABLED if self.preview.error or
                                        self.preview.distance <= EPSILON else DGG.NORMAL)

    def confirm(self):
        if commit_move(self.game, self.unit, self.positions, self.destination):
            self.close(resume=False)
            return True
        self.redraw()
        return False

    def cancel(self):
        self.close(resume=True)

    def close(self, *, resume=False):
        self.ignoreAll()
        self.game.taskMgr.remove('skirmishEditor')
        self.panel.destroy()
        if self.ghost is not None:
            self.ghost.removeNode()
        self.game.skirmishEditor = None
        if resume and unavailable_reason(self.game, self.unit) is None:
            self.game.startTaskFunction(self.game.taskLoopPathTowardsMouse, 'taskLoopPathTowardsMouse')
        refresh_adjust_button(self.game, getattr(self.game, 'unitToMove', None))