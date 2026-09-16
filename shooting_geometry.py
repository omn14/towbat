"""Individual missile eligibility (Rulebook pp. 103, 137, 139, 184-185)."""

from dataclasses import dataclass

from panda3d.core import Point3

from battleFunctions import firing_rank_count
from psychology import is_skirmish_unit, obb_distance
from rules_log import rule_log, rule_skipped
from scouts import model_base_boxes
from skirmish import EPSILON
from skirmish_visibility import model_can_see, model_visible_fraction
from terrain_system import sees_over


@dataclass(frozen=True)
class Shooter:
    unit: object
    index: int
    distance: float | None
    long_range: bool
    reason: str | None = None
    wood_sight: bool = False
    cover: int = 0
    cover_detail: str = ''

    @property
    def cover_modifiers(self):
        return dict(partial_cover=self.cover == 1, full_cover=self.cover == 2)


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
        protected = sum(model.reason == 'protected by Shadowed Mantle' for model in self.models)
        partial = sum(model.cover == 1 for model in eligible)
        full = sum(model.cover == 2 for model in eligible)
        return (f'{len(eligible)}/{len(self.models)} can shoot: {len(eligible) - long} short, {long} long; '
            f'{blocked} blocked, {distant} out of range'
            + (f'; {protected} denied by Shadowed Mantle' if protected else '')
            + (f'; cover: {partial} partial, {full} full' if partial or full else ''))


def target_cover(observer, targets, blockers=(), *, terrain=()):
    """Cover from existing XY silhouettes; exactly half is partial (p. 139).

    This shares sight's see-onto terrain convention, not sculpt heights or
    foliage density. Target-unit bases do not screen one another.
    """
    fractions = [model_visible_fraction(observer, target, blockers, terrain=terrain) for target in targets]
    if not fractions:
        return 0, 'no target bases'
    if len(fractions) == 1:
        obscured = max(0.0, 1 - fractions[0])
        detail = f'{obscured:.1%} of lone-target silhouette obscured'
        return (2 if obscured > .5 + EPSILON else 1 if obscured > EPSILON else 0), detail
    obscured = sum(fraction < 1 - EPSILON for fraction in fractions)
    detail = f'{obscured}/{len(fractions)} target models obscured'
    return (2 if obscured * 2 > len(fractions) else 1 if obscured else 0), detail


def model_shot(observer, targets, blockers, reach, *, facing=None, sight_origin=None,
               stand_and_shoot=False, terrain=()):
    """A visible target within each model's own range; reactions waive range (p. 137)."""
    source = observer if sight_origin is None else sight_origin
    seen = []
    for target in targets:
        if model_can_see(source, [target], blockers, facing=facing, terrain=terrain):
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
    from characters import get_joined_characters
    members = [target, *get_joined_characters(target)]
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
    from characters import get_joined_characters
    if get_joined_characters(unit) or get_joined_characters(target):
        return True
    from magic_items import EffectKind, effects_for
    if any(effects_for(member, kind) for member in getattr(game, 'units', [])
           for kind in (EffectKind.TARGET_PROTECTION, EffectKind.SHOOTING_SIGHT)):
        return True
    pieces = getattr(getattr(game, 'terrain_manager', None), 'terrain_pieces', [])
    if any(getattr(piece, 'blocks_line_of_sight', False)
           or getattr(piece, 'terrain_type', None) == 'landmark' for piece in pieces):
        return True
    if is_skirmish_unit(unit) or is_skirmish_unit(target):
        return True
    return any(member not in (unit, target) and member.isDeployed and member.unit.nmodels > 0
               and getattr(member, 'hostUnit', None) is None
               for member in getattr(game, 'units', []))


def shooting_solution(game, unit, target, *, weapon=None, stand_and_shoot=False,
                      target_boxes=None):
    """Read-only shared preview/volley query; rear ranks inherit file-front sight (p. 137)."""
    boxes = model_base_boxes(unit)
    from magic_items import item_target_protected
    if not stand_and_shoot and item_target_protected(game, unit, target):
        return ShootingSolution(tuple(Shooter(unit, index, None, False, 'protected by Shadowed Mantle')
                                      for index in range(min(unit.unit.nmodels, len(boxes)))))
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
    from characters import get_joined_characters
    joined = get_joined_characters(unit)
    children = list(unit.model.getChildren())[:count]
    slots = {index: (round(-child.getY() / unit.modelHeight) * files
                    + round(child.getX() / unit.modelWidth)) for index, child in enumerate(children)}
    records = [(slots[index], unit, index) for index in range(count)]
    records += [(getattr(member, 'formationSlot', getattr(unit, 'characterSlot', 0)) or 0, member, count + offset)
                for offset, member in enumerate(joined) if member.unit.nmodels > 0]
    firing = firing_rank_count(files, max([slot for slot, _, _ in records], default=0) + 1, extra, volley)
    participants = [(slot, member, index) for slot, member, index in records
                    if (loose or slot < firing) and not getattr(member, 'retiredFromCombat', False)]
    front_sources = {slot: index for slot, member, index in records if member is unit and slot < files}
    for offset, member in enumerate(joined):
        cells = getattr(unit, 'characterPlacements', {}).get(member.unitName, {}).get('cells', [])
        front_sources.update({cell: count + offset for cell in cells if cell < files})
    result = []
    cover_by_source = {}
    for slot, member, index in participants:
        profile = member.unit.model
        missile = weapon if member is unit and weapon is not None else profile.equipedWeapon or {}
        if missile.get('tag') != 'ranged':
            continue
        moved = any(getattr(part, flag, False) for part in (unit, member)
                    for flag in ('hasMovedThisTurn', 'manoeuvreThisTurn', 'moveSpentThisTurn', 'attemptedRallyThisTurn'))
        marched = any(getattr(part, 'marchedThisTurn', False) for part in (unit, member))
        blocked = ('marched' if marched and not profile.fires_after_marching() else
                   'Move or Shoot' if moved and not stand_and_shoot and profile.cannot_shoot_after_moving(missile) else None)
        if blocked:
            result.append(Shooter(member, index, None, False, blocked))
            continue
        source_index = index if loose or member is not unit else front_sources.get(slot % files, index)
        source = boxes[source_index]
        own = boxes[:source_index] + boxes[source_index + 1:]
        facing = None if loose or profile.has_all_round_vision() else source[4]
        from magic_items import EffectKind, profile_effects
        ignored = {entry.effect.value for entry in profile_effects(profile, EffectKind.SHOOTING_SIGHT)}
        sight_pieces = [piece for piece in pieces if getattr(piece, 'terrain_type', None) not in ignored]
        distance, long_range, reason = model_shot(
            boxes[index], targets, [*own, *others], missile.get('ranged_range', 0),
            facing=facing, sight_origin=source, stand_and_shoot=stand_and_shoot, terrain=sight_pieces)
        wood_sight = bool(reason is None and len(sight_pieces) != len(pieces) and model_shot(
            boxes[index], targets, [*own, *others], missile.get('ranged_range', 0),
            facing=facing, sight_origin=source, stand_and_shoot=stand_and_shoot, terrain=pieces)[2] == 'no line of sight')
        cover, cover_detail = 0, ''
        if reason is None:
            if source_index not in cover_by_source:
                cover_by_source[source_index] = target_cover(source, targets, [*own, *others], terrain=pieces)
            cover, cover_detail = cover_by_source[source_index]
        result.append(Shooter(member, index, distance, long_range, reason, wood_sight, cover, cover_detail))
    return ShootingSolution(tuple(result))


def report_item_sight(solution):
    """Only a resolved volley reports item sight, not previews (Companion p. 48)."""
    from magic_items import EffectKind, profile_effects
    seen = set()
    for shooter in solution.models:
        member = shooter.unit
        if id(member) in seen:
            continue
        seen.add(id(member))
        own = [record for record in solution.models if record.unit is member]
        recovered = sum(record.wood_sight for record in own)
        for entry in profile_effects(member.unit.model, EffectKind.SHOOTING_SIGHT):
            report = rule_log if recovered else rule_skipped
            report(entry.item.name, member, f'{recovered}/{len(own)} firing models gain line of sight by ignoring woods; '
                   'other terrain and models still block shooting')