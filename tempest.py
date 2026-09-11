"""Stationary Tempest and source-owned terrain escalation (Rulebook pp. 107, 329)."""

from dataclasses import dataclass

from panda3d.core import Point3

from characters import side_of
from rules_log import rule_log, rule_skipped
from spell_effects import register
from spell_system import PillarOfFireSpell
from spell_templates import circle_distance, swept_circle_distance


class TempestSpell(PillarOfFireSpell):
    spell_type = 'Magical Vortex'

    def canTarget(self, point):
        from scouts import BOARD_HALF_DEPTH, BOARD_HALF_WIDTH, model_base_boxes
        if not super().canTarget(point):
            return False
        if abs(point.x) + 1.5 > BOARD_HALF_WIDTH or abs(point.y) + 1.5 > BOARD_HALF_DEPTH:
            rule_skipped(self.name, self.caster, '3-inch template must fit on the battlefield')
            return False
        if any(circle_distance(point, box) <= 1.5 + 1e-6
               for member in self.game.units for box in model_base_boxes(member)):
            rule_skipped(self.name, self.caster, 'vortex cannot be placed touching a model base (p. 107)')
            return False
        return True

    def place(self, game, point):
        self.game = game
        self.piece = game.terrain_manager.add_terrain('pillar_of_fire', Point3(point.x, point.y, .1),
                                                     3, 3, going='dangerous')
        self.piece.visual.setColorScale(.35, .65, 1, 1)
        register(self, duration='remains')
        rule_log(self.name, self.caster, f'stationary 3-inch dangerous template at '
                 f'({point.x:.2f}, {point.y:.2f}); enemy terrain escalates within 6 inches of its edge')

    def scatter(self, game):
        pass

    def enemies(self, game):
        return []


@dataclass(eq=False)
class TempestTerrain:
    source: object
    terrain_type: str
    center: object
    width: float
    height: float
    is_dangerous: bool = False
    movement_modifier: int = -1
    is_impassable: bool = False
    disrupts: bool = True
    tempest_aura: bool = True


def tempest_features(game, unit, start, end, features, *, log=False, base_paths=None):
    """Temporary circular template/aura views, without changing native terrain (p. 329)."""
    spells = [spell for spell in getattr(game, 'remainsInPlay', [])
              if isinstance(spell, TempestSpell) and spell.piece is not None]
    if not spells or start is None or end is None:
        return features
    if base_paths is None:
        from scouts import model_base_boxes
        current = unit.bodyNP.getPos(unit.bodyNP.getTop())
        base_paths = [((box[0] + start[0] - current.x, box[1] + start[1] - current.y, *box[2:]),
                       (box[0] + end[0] - current.x, box[1] + end[1] - current.y, *box[2:]))
                      for box in model_base_boxes(unit)]
    sources = {spell.piece for spell in spells}
    originals = []
    for feature in features:
        piece = feature.source if isinstance(feature, TempestTerrain) else feature
        if piece not in sources and piece not in originals:
            originals.append(piece)
    active = [spell for spell in spells if side_of(game, spell.caster, None) != side_of(game, unit, None)
              and any(swept_circle_distance(spell.piece.center, before, after) <= 7.5 + 1e-8
                      for before, after in base_paths)]
    result = []
    for piece in originals:
        bounds = (piece.center.x, piece.center.y, piece.width / 2, piece.height / 2, 0)
        dangerous = getattr(piece, 'going', None) == 'difficult' and any(
            swept_circle_distance(spell.piece.center, before, after, bounds) <= 7.5 + 1e-8
            for spell in active for before, after in base_paths)
        result.append(TempestTerrain(piece, 'Tempest: difficult becomes dangerous', piece.center,
                                     piece.width, piece.height, True) if dangerous else piece)
    for spell in spells:
        if any(swept_circle_distance(spell.piece.center, before, after) <= 1.5 + 1e-8
               for before, after in base_paths):
            result.append(TempestTerrain(spell.piece, 'Tempest template', spell.piece.center,
                                         3, 3, True, tempest_aura=False))
    if active:
        result.append(TempestTerrain(active[0].piece, 'Tempest: open becomes difficult',
                                     active[0].piece.center, 15, 15))
    if log and (active or any(isinstance(piece, TempestTerrain) for piece in result)):
        rule_log('Tempest', unit, f'{len(active)} enemy aura(s) crossed; -1 Movement, non-cumulative; '
                 f'{sum(piece.is_dangerous for piece in result)} dangerous features; protections still apply')
    return result