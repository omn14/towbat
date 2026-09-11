"""Circular spell coverage and scatter (Rulebook pp. 95, 169, 329)."""

import math
import random

from panda3d.core import Point3

from psychology import _box_corners


def circle_distance(center, box):
    angle = math.radians(box[4])
    horizontal, vertical = center[0] - box[0], center[1] - box[1]
    local_x = horizontal * math.cos(angle) + vertical * math.sin(angle)
    local_y = -horizontal * math.sin(angle) + vertical * math.cos(angle)
    return math.hypot(max(0, abs(local_x) - box[2]), max(0, abs(local_y) - box[3]))


def template_coverage(center, radius, box):
    """Return automatic, partial or miss, including a base under the hole (p. 95)."""
    if circle_distance(center, box) <= 1e-8:
        return 'automatic'
    if all(math.hypot(corner[0] - center[0], corner[1] - center[1]) <= radius + 1e-8
           for corner in _box_corners(*box)):
        return 'automatic'
    return 'partial' if circle_distance(center, box) <= radius + 1e-8 else 'miss'


def scatter_template(center, distance):
    """Two Hit faces and four arrows; arrow direction is uniform (p. 95)."""
    if random.randint(1, 6) <= 2:
        return Point3(center), 'Hit; no scatter'
    angle = random.uniform(0, math.tau)
    destination = Point3(center.x + math.cos(angle) * distance,
                         center.y + math.sin(angle) * distance, 0)
    return destination, f'arrow {math.degrees(angle):.1f} degrees, {distance:g} inches'


def swept_circle_distance(center, start, end, clip=None):
    """Exact distance to a translating base's swept polygon, optionally terrain-clipped."""
    from spell_system import distance_to_segment
    def cross(origin, first, second):
        return ((first[0] - origin[0]) * (second[1] - origin[1])
                - (first[1] - origin[1]) * (second[0] - origin[0]))
    points = sorted(set(_box_corners(*start) + _box_corners(*end)))
    lower, upper = [], []
    for sequence, chain in ((points, lower), (reversed(points), upper)):
        for point in sequence:
            while len(chain) >= 2 and cross(chain[-2], chain[-1], point) <= 0:
                chain.pop()
            chain.append(point)
    polygon = lower[:-1] + upper[:-1]
    if clip is not None:
        for axis, boundary, direction in ((0, clip[0] - clip[2], 1), (0, clip[0] + clip[2], -1),
                                           (1, clip[1] - clip[3], 1), (1, clip[1] + clip[3], -1)):
            result = []
            for index, first in enumerate(polygon):
                second = polygon[(index + 1) % len(polygon)]
                inside_first = direction * (first[axis] - boundary) >= 0
                inside_second = direction * (second[axis] - boundary) >= 0
                if inside_first:
                    result.append(first)
                if inside_first != inside_second:
                    fraction = (boundary - first[axis]) / (second[axis] - first[axis])
                    result.append(tuple(first[coordinate] + fraction * (second[coordinate] - first[coordinate])
                                        for coordinate in (0, 1)))
            polygon = result
    if not polygon:
        return float('inf')
    edges = list(zip(polygon, polygon[1:] + polygon[:1]))
    if len(polygon) >= 3 and all(cross(first, second, center) >= -1e-8 for first, second in edges):
        return 0.0
    return min(distance_to_segment(center[0], center[1], *first, *second) for first, second in edges)


async def fiery_template(spell, center):
    """Snapshot template hits, including command/character protection (pp. 199, 209, 329)."""
    from battleFunctions import resolve_magic_hits
    from characters import side_of
    from command_groups import champions, living_command
    from panda3d.core import Vec3
    from rules_log import rule_log, rule_skipped
    from scouts import model_base_boxes
    from troop_types import is_cavalry, is_infantry
    game = spell.game
    hits_by_model = {}
    for member in list(game.units):
        if member.bodyNP.isEmpty() or not member.isDeployed or member.unit.nmodels <= 0:
            continue
        boxes = model_base_boxes(member)[:member.unit.nmodels]
        command = living_command(member)
        promoted = champions(member, include_retired=True)
        for index, box in enumerate(boxes):
            coverage = template_coverage(center, 2.5, box)
            if coverage == 'miss':
                continue
            if side_of(game, member, None) == side_of(game, spell.caster, None):
                rule_skipped(spell.name, member, 'friendly base under template: no hit')
                continue
            roll = random.randint(1, 6) if coverage == 'partial' else None
            if roll is not None:
                report = rule_log if roll >= 4 else rule_skipped
                report(spell.name, member, f'partially covered base {index + 1}: {roll} vs 4+ to hit')
                if roll < 4:
                    continue
            recipient = next((champion for champion in promoted if index < len(command)
                              and champion.command_entry is command[index]), member)
            host = getattr(recipient, 'command_host', None) or getattr(recipient, 'hostUnit', None)
            if host is not None:
                ordinary = host.unit.nmodels - len(champions(host, include_retired=True))
                same_type = bool(getattr(recipient, 'command_host', None)) or any(
                    category(host.unit.model.troop_type()) and category(recipient.unit.model.troop_type())
                    for category in (is_infantry, is_cavalry))
                if ordinary >= 5 and same_type:
                    choice = 'Look Out, Sir!'
                    if not game.aiControls(host):
                        choice = await game.makeChoiceNew(['Look Out, Sir!', 'Take hit'], Vec3(0, 0, 10),
                                                         owner=host, prompt=f'{recipient.unit.name}: template hit')
                    if choice == 'Look Out, Sir!':
                        protection = random.randint(1, 6)
                        rule_log('Look Out, Sir!', recipient, f'{ordinary} rank-and-file models; roll {protection} '
                                 f'vs 2+ -> {"unit takes hit" if protection >= 2 else "model takes hit"} (pp. 199, 209)')
                        if protection >= 2:
                            recipient = host
                    else:
                        rule_skipped('Look Out, Sir!', recipient, 'owner elects to take the template hit')
                else:
                    rule_skipped('Look Out, Sir!', recipient, f'{ordinary} rank-and-file models; '
                                 f'same troop type: {same_type}; needs at least 5 (pp. 199, 209)')
            entry = hits_by_model.setdefault(id(recipient), [recipient, 0])
            entry[1] += 1
    results = []
    for member, hits in hits_by_model.values():
        flammable = any(rule.get('flammable') or rule.get('name', '').lower() == 'flammable'
                        for rule in member.unit.model.special_rules)
        wounds, saves, unsaved = resolve_magic_hits(member.unit, hits, 4, 2, allow_regeneration=not flammable)
        rule_log(spell.name, member, f'{hits} Flaming S4 AP-2 template hits -> {wounds} wounds, '
                 f'{saves} saved, {unsaved} unsaved (p. 329)')
        if flammable:
            rule_log('Flammable', member, f'{wounds} Flaming wounds: Regeneration prohibited (p. 169)')
        results.append((member, unsaved))
    affected = []
    ordered = sorted(results, key=lambda result: not (getattr(result[0], 'command_host', None)
                     or getattr(result[0], 'hostUnit', None)))
    for member, wounds in ordered:
        host = getattr(member, 'command_host', None) or getattr(member, 'hostUnit', None) or member
        if wounds:
            if member is not host:
                game.combat.woundDuellist(member, wounds)
            elif member in game.units:
                game.movement.applyWounds(member, wounds)
            if host not in affected:
                affected.append(host)
    for host in affected:
        if host in game.units and not host.bodyNP.isEmpty():
            game.psychology.check_heavy_casualties(host, 'shooting', attacker=spell.caster)