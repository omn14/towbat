"""Individual missile eligibility (Rulebook pp. 103, 137, 139, 184-185)."""

from dataclasses import dataclass

from panda3d.core import Point3

from battleFunctions import firing_rank_count
from psychology import is_skirmish_unit, obb_distance
from rules_log import rule_log, rule_skipped
from scouts import model_base_boxes
from skirmish import EPSILON
from skirmish_visibility import model_can_see
from terrain_system import sees_over


@dataclass(frozen=True)
class Shooter:
    unit: object
    index: int
    distance: float | None
    long_range: bool
    reason: str | None = None


@dataclass(frozen=True)
class ShootingSolution:
    models: tuple[Shooter, ...]

    @property
    def eligible(self):
        return tuple(model for model in self.models if model.reason is None)

    def detail(self):
        eligible = self.eligible
        long = sum(model.long_range for model in eligible)
        blocked = sum(model.reason == 'no line of sight' for model in self.models)
        distant = sum(model.reason == 'out of range' for model in self.models)
        return (f'{len(eligible)}/{len(self.models)} can shoot: {len(eligible) - long} short, {long} long; '
                f'{blocked} blocked, {distant} out of range')


def model_shot(observer, targets, blockers, reach, *, facing=None, sight_origin=None,
               stand_and_shoot=False, terrain=()):
    """A visible target within each model's own range; reactions waive range (p. 137)."""
    source = observer if sight_origin is None else sight_origin
    seen = []
    for target in targets:
        terrain_boxes = [(piece.center.x, piece.center.y, piece.width / 2, piece.height / 2, 0)
                         for piece in terrain if piece.blocks_line_of_sight
                         and not piece.contains(Point3(*source[:2], 0))
                         and not piece.contains(Point3(*target[:2], 0))]
        if model_can_see(source, [target], [*blockers, *terrain_boxes], facing=facing):
            seen.append(obb_distance(observer, target))
    if not seen:
        return None, False, 'no line of sight'
    distance = min(seen)
    if not stand_and_shoot and distance > reach + EPSILON:
        return distance, True, 'out of range'
    return distance, not stand_and_shoot and distance > reach / 2 + EPSILON, None


def enemy_fire_modifier(target, *, log=False):
    """The -1 applies only when every live model is US1 (p. 185), including a joined character."""
    if not is_skirmish_unit(target):
        return False
    members = [target]
    joined = getattr(target, 'joinedCharacter', None)
    if joined is not None:
        members.append(joined)
    live = [member for member in members if member.unit.nmodels > 0]
    total = sum(member.unit.nmodels for member in live)
    strength_one = sum(member.unit.nmodels for member in live if member.unit.model.unit_strength() == 1)
    applies = total > 0 and strength_one == total
    if log:
        report = rule_log if applies else rule_skipped
        report('Skirmishers', target,
               f'{strength_one}/{total} live models are US1, including joined characters '
               f'-> {"enemy -1 To Hit" if applies else "no enemy-fire modifier"} (p. 185)')
    return applies


def uses_individual_shooting(game, unit, target):
    if is_skirmish_unit(unit) or is_skirmish_unit(target):
        return True
    return any(is_skirmish_unit(member) and member.isDeployed and member.unit.nmodels > 0
               and getattr(member, 'hostUnit', None) is None
               for member in getattr(game, 'units', []))


def shooting_solution(game, unit, target, *, weapon=None, stand_and_shoot=False,
                      target_boxes=None):
    """Read-only shared preview/volley query; rear ranks inherit file-front sight (p. 137)."""
    boxes = model_base_boxes(unit)
    targets = model_base_boxes(target) if target_boxes is None else target_boxes
    pieces = getattr(getattr(game, 'terrain_manager', None), 'terrain_pieces', [])
    hill = game.movement.hillUnderUnit(unit)
    target_hill = game.movement.hillUnderUnit(target)
    others = []
    for other in game.units:
        if (other in (unit, target) or getattr(other, 'hostUnit', None) is not None
                or not other.isDeployed or other.unit.nmodels <= 0):
            continue
        other_hill = game.movement.hillUnderUnit(other)
        if (hill is not None or target_hill is not None) and other_hill is None:
            continue
        if hill is not None and other_hill is hill and sees_over(
                unit.bodyNP.getPos(), other.bodyNP.getPos(), hill.center):
            continue
        others.extend(model_base_boxes(other))
    loose = is_skirmish_unit(unit) and not getattr(unit, 'skirmishCombat', False)
    extra = int(game.movement.entirelyOnHill(unit)) if not loose else 0
    count = min(unit.unit.nmodels, len(boxes))
    files = max(1, unit.unit.files)
    volley = any(rule.get('volley_fire') for rule in unit.unit.model.special_rules)
    joined = getattr(unit, 'joinedCharacter', None)
    has_joined = joined is not None and joined.unit.nmodels > 0 and len(boxes) > count
    reserved = getattr(unit, 'characterSlot', None) if has_joined and not loose else None
    slots = {index + int(reserved is not None and index >= reserved): (unit, index)
             for index in range(count)}
    if has_joined:
        slots[reserved if reserved is not None else count] = (joined, count)
    firing = len(slots) if loose else firing_rank_count(files, len(slots), extra, volley)
    participants = [(slot, member, index) for slot, (member, index) in sorted(slots.items())
                    if slot < firing and not (member is joined and getattr(joined, 'retiredFromCombat', False))]
    result = []
    for slot, member, index in participants:
        profile = member.unit.model
        missile = weapon if member is unit and weapon is not None else profile.equipedWeapon or {}
        if missile.get('tag') != 'ranged':
            continue
        source_index = index if loose else slots[slot % files][1]
        source = boxes[source_index]
        own = boxes[:source_index] + boxes[source_index + 1:]
        facing = None if loose or profile.has_all_round_vision() else source[4]
        distance, long_range, reason = model_shot(
            boxes[index], targets, [*own, *others], missile.get('ranged_range', 0),
            facing=facing, sight_origin=source, stand_and_shoot=stand_and_shoot, terrain=pieces)
        result.append(Shooter(member, index, distance, long_range, reason))
    return ShootingSolution(tuple(result))