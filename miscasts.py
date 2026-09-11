"""Miscast and Outclassed casualties (Rulebook pp. 95, 109-110)."""

import random

from battleFunctions import resolve_magic_hits
from characters import side_of
from command_groups import champions, living_command
from rules_log import rule_log, rule_skipped
from scouts import model_base_boxes
from spell_templates import template_coverage


def miscast_targets(game, wizard, entry, context):
    """Snapshot every covered base, with no shooting-only Look Out, Sir! (pp. 109, 209)."""
    if not entry['blast']:
        return [(wizard, 1)]
    boxes = model_base_boxes(wizard)
    if not boxes:
        return []
    center = boxes[0][:2]
    radius = entry['blast'] / 2
    rule_log(context, wizard, f'{entry["name"]}: {entry["blast"]}" template centred '
             f'on the Wizard at ({center[0]:.2f}, {center[1]:.2f}); friend and foe (p. 109)')
    groups = {}
    for member in list(game.units):
        if (member.bodyNP.isEmpty() or member.unit.nmodels <= 0
                or not getattr(member, 'isDeployed', True)):
            continue
        command = living_command(member)
        promoted = champions(member, include_retired=True)
        for index, box in enumerate(model_base_boxes(member)[:member.unit.nmodels]):
            coverage = template_coverage(center, radius, box)
            if coverage == 'miss':
                continue
            recipient = next((champion for champion in promoted if index < len(command)
                              and champion.command_entry is command[index]), member)
            packet = groups.setdefault(id(recipient), [recipient, 0, []])
            if coverage == 'automatic':
                packet[1] += 1
            else:
                packet[2].append(random.randint(1, 6))
    result = []
    for member, automatic, partial in groups.values():
        hits = automatic + sum(roll >= 4 for roll in partial)
        report = rule_log if hits else rule_skipped
        report(context, member, f'{automatic} automatic hits + partial rolls {partial} '
               f'vs 4+ -> {hits} hits; no Look Out, Sir! for a Miscast (pp. 95, 109, 209)')
        if hits:
            result.append((member, hits))
    return result


def resolve_miscast_damage(game, wizard, entry, *, context='Miscast'):
    """Resolve the table's automatic magical hits, not the attempted spell (pp. 109-110)."""
    if not entry['strength']:
        return []
    if (game is None or wizard is None or not hasattr(wizard, 'bodyNP') or wizard.bodyNP.isEmpty()
            or wizard.unit.nmodels <= 0):
        rule_skipped(context, wizard, f'{entry["name"]}: no live Wizard on the battlefield to damage')
        return []
    results = []
    for member, hits in miscast_targets(game, wizard, entry, context):
        regenerated = []
        wounds, saves, unsaved = resolve_magic_hits(member.unit, hits, entry['strength'], entry['ap'],
                                                   regenerated=regenerated)
        rule_log(context, member, f'{entry["name"]}: {hits} S{entry["strength"]} AP-{entry["ap"]} '
                 f'magical hits -> {wounds} wounds, {saves} saved, {unsaved} unsaved; '
                 'normal armour, Ward and Regeneration permitted (pp. 109-110)')
        results.append((member, unsaved, len(regenerated)))
    ordered = sorted(results, key=lambda result: not (getattr(result[0], 'command_host', None)
                     or getattr(result[0], 'hostUnit', None)))
    psychology = getattr(game, 'psychology', None)
    affected = {}
    owner = side_of(game, wizard, None)
    window = getattr(game, 'assailmentWindow', None) or {}
    deferred = window.get('miscast_damage')
    if psychology:
        psychology.hold_panic()
    try:
        for member, wounds, regenerated in ordered:
            if not wounds and not regenerated:
                continue
            host = getattr(member, 'command_host', None) or getattr(member, 'hostUnit', None) or member
            if wounds:
                affected[id(host)] = host
            if deferred is not None:
                deferred(member, wounds, regenerated)
            elif wounds and member is not host:
                game.combat.woundDuellist(member, wounds)
            elif wounds and member in game.units:
                game.movement.applyWounds(member, wounds)
        phase = game.castingPhase() if hasattr(game, 'castingPhase') else 'strategy'
        if psychology and phase != 'combat':
            for host in affected.values():
                if host in game.units and not host.bodyNP.isEmpty():
                    attacker = wizard if (not wizard.bodyNP.isEmpty()
                                          and side_of(game, host, None) != owner) else None
                    psychology.check_heavy_casualties(host, phase, attacker=attacker)
    finally:
        if psychology:
            psychology.release_panic()
    return [(member, wounds) for member, wounds, regenerated in results]