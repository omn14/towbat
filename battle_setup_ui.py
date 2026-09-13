"""Interactive terrain placement for Battle March (Companion p. 24; Rulebook p. 268)."""

from panda3d.core import AsyncFuture, Point3
from direct.gui.DirectGui import DirectEntry
from shapely.affinity import translate

import gui_theme as theme
from battle_config import ConfigError, _number
from battle_terrain import footprint, placement_report
from rules_log import rule_log, rule_skipped
from terrain_system import TerrainPiece


class TerrainPlacement:
    def __init__(self, game, specification, placed, player):
        self.game = game
        self.specification = dict(specification)
        self.placed = placed
        self.player = player
        self.future = AsyncFuture()
        self.panel = None
        self.preview = None
        self.position = Point3(0, 0, 0)
        self.report = None
        self.resize(specification['width'], specification['height'])

    def resize(self, width, height):
        _number(width, 'terrain.width', .1, self.game.battlefield.width)
        _number(height, 'terrain.height', .1, self.game.battlefield.depth)
        specification = {**self.specification, 'width': width, 'height': height}
        previous = self.preview
        self.preview = TerrainPiece(specification['type'], Point3(0, 0, 0),
                                    width, height, self.game,
                                    specification.get('going'))
        if self.preview.ghost_np is not None:
            self.game.world.removeRigidBody(self.preview.ghost_np.node())
            self.preview.ghost_np.removeNode()
            self.preview.ghost_np = None
        self.geometry = footprint(self.preview)
        self.specification = specification
        if previous is not None:
            previous.destroy()
        self.set_position(self.position)

    def apply_size(self):
        try:
            self.resize(float(self.width_entry.get()), float(self.height_entry.get()))
        except (ValueError, ConfigError) as error:
            self.status.setText(str(error))

    def cancel(self):
        if not self.future.done():
            self.future.set_result(None)

    def set_position(self, position):
        self.position = Point3(position)
        self.position.z = 0
        shape = translate(self.geometry, self.position.x, self.position.y)
        self.report = placement_report(self.game.battlefield, self.game.battle_config, shape,
                                       self.placed, self.player,
                                       special=self.specification.get('special', False))
        self.preview.visual.setPos(self.position)
        if self.preview.outline is not None:
            self.preview.outline.setPos(self.position)
        if self.preview.trees_np is not None:
            self.preview.trees_np.setPos(self.position)
        self.preview.visual.setColorScale(*((1, .35, .3, .8) if self.report['errors'] else (.45, 1, .65, .8)))
        if self.panel is not None:
            self.status.setText('\n'.join(self.report['errors'] or self.report['warnings'] or ['Legal placement']))
            self.status.setScale(min(.032, .15 / max(.01, self.status.textNode.getHeight())))
            self.panel.setX(-self.game.getAspectRatio() + .49)
        return self.report

    def move(self, task):
        if self.game.mouseWatcherNode is not None and self.game.mouseWatcherNode.hasMouse():
            mouse = self.game.mouseWatcherNode.getMouse()
            pointer = self.game.aspect2d.getRelativePoint(self.game.render2d, Point3(mouse.x, 0, mouse.y))
            if (self.panel is not None and -.46 <= pointer.x - self.panel.getX() <= .46
                    and -.40 <= pointer.z - self.panel.getZ() <= .14):
                return task.cont
            near, far = Point3(), Point3()
            self.game.cam.node().getLens().extrude(mouse, near, far)
            near = self.game.render.getRelativePoint(self.game.cam, near)
            far = self.game.render.getRelativePoint(self.game.cam, far)
            if abs(far.z - near.z) > 1e-8:
                fraction = -near.z / (far.z - near.z)
                if fraction >= 0:
                    self.set_position(near + (far - near) * fraction)
        return task.cont

    def commit(self):
        if self.future.done():
            return
        self.set_position(self.position)
        if self.report['errors']:
            rule_skipped('Terrain placement', f'Player {self.player}', '; '.join(self.report['errors']))
            return
        record = {**self.specification, 'center': [self.position.x, self.position.y, 0],
                  'player': self.player}
        rule_log('Terrain placement', f'Player {self.player}',
                 f'{record["type"]} {record["width"]:g} x {record["height"]:g} at '
                 f'({self.position.x:.2f}, {self.position.y:.2f}); '
                 + ('; '.join(self.report['warnings']) if self.report['warnings'] else 'all placement clearances satisfied'))
        self.future.set_result(record)

    def ai_position(self):
        field = self.game.battlefield
        candidates = [(horizontal, vertical)
                      for horizontal in range(-int(field.width / 2), int(field.width / 2) + 1)
                      for vertical in range(-int(field.depth / 2), int(field.depth / 2) + 1)]
        candidates.sort(key=lambda point: (abs(point[0]) + abs(point[1]), point))
        for horizontal, vertical in candidates:
            if not self.set_position(Point3(horizontal, vertical, 0))['errors']:
                return self.position
        raise ConfigError('No legal sampled placement for this feature; choose another size or revise earlier terrain')

    async def choose(self, owner):
        try:
            if self.game.aiControls(owner):
                self.ai_position()
                self.commit()
            else:
                self.panel = theme.styled_panel((-.46, .46, -.40, .14),
                                                pos=(-self.game.getAspectRatio() + .49, 0, .70), parent=self.game.aspect2d)
                theme.styled_text(f'Player {self.player}: {self.specification["type"]}',
                                  parent=self.panel, pos=(-.41, .065), scale=.039)
                self.status = theme.styled_text('', parent=self.panel, pos=(-.41, -.01),
                                                scale=.032, wordwrap=24)
                theme.styled_text('Width', parent=self.panel, pos=(-.41, -.18), scale=.028)
                theme.styled_text('Depth', parent=self.panel, pos=(-.05, -.18), scale=.028)
                self.width_entry = DirectEntry(parent=self.panel, initialText=str(self.specification['width']),
                                               pos=(-.41, 0, -.24), scale=.036, width=5, numLines=1)
                self.height_entry = DirectEntry(parent=self.panel, initialText=str(self.specification['height']),
                                                pos=(-.05, 0, -.24), scale=.036, width=5, numLines=1)
                theme.tex_button('Size', (.31, 0, -.23), self.apply_size, parent=self.panel, scale=.035)
                theme.tex_button('Cancel', (-.22, 0, -.35), self.cancel, parent=self.panel, scale=.038)
                theme.tex_button('Place', (.22, 0, -.35), self.commit, parent=self.panel, scale=.038)
                self.game.accept('mouse1', self.commit)
                self.game.taskMgr.add(self.move, 'battleMarchTerrainPreview')
                self.set_position(self.position)
            return await self.future
        finally:
            self.game.ignore('mouse1')
            self.game.taskMgr.remove('battleMarchTerrainPreview')
            if self.panel is not None:
                self.panel.destroy()
            self.preview.destroy()