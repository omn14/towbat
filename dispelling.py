"""Dispel choices and Remains in Play attempts (Rulebook pp. 110-111)."""

from panda3d.core import Vec3

from characters import side_of
from magic_items import current_turn
from psychology import PsychologySystem, obb_distance
from rules_log import rule_log, rule_skipped


def side_members(game, side):
    members = list(game.player1Units if side == 1 else game.player2Units)
    for host in list(members):
        joined = getattr(host, 'joinedCharacter', None)
        if joined is not None and joined not in members:
            members.append(joined)
    return [member for member in members if not member.bodyNP.isEmpty()]


def wizard_reason(game, wizard, spell, *, remains=False):
    """Range is base-to-base; a vortex also supplies its circular template (p. 111)."""
    from chaos_gifts import succumbed
    if succumbed(wizard):
        return 'Wizard or host succumbed to Stupidity (p. 178)'
    host = getattr(wizard, 'hostUnit', None) or wizard
    if getattr(host, 'state', None) == 'IsFleeing':
        return 'Wizard or host is fleeing'
    if not getattr(wizard, 'isDeployed', True):
        return 'Wizard is not on the battlefield'
    if getattr(wizard, 'dispelBlockedTurn', None) == current_turn(game):
        return 'Wizard cannot dispel again this turn after being Outclassed'
    target = getattr(spell, 'affected_unit', None) if remains else getattr(spell, 'target', None)
    target_host = getattr(target, 'command_host', None) or getattr(target, 'hostUnit', None) or target
    if getattr(host, 'isInCombat', False) and target_host is not host:
        return 'engaged Wizard can only dispel a spell targeting their unit'
    level = wizard.unit.model.wizard_level(0)
    radius = 24 if level >= 3 else 18
    box = PsychologySystem._unit_box(wizard)
    distances = []
    if spell.caster is not None and not spell.caster.bodyNP.isEmpty():
        distances.append(obb_distance(box, PsychologySystem._unit_box(spell.caster)))
    piece = getattr(spell, 'piece', None) if remains else None
    if piece is not None:
        point_box = (piece.center.x, piece.center.y, 0, 0, 0)
        distances.append(max(0, obb_distance(box, point_box) - piece.width / 2))
    distance = min(distances, default=float('inf'))
    if distance > radius:
        return f'{distance:.2f}" away, outside Level {level} Dispel range {radius}"'
    return None


async def attempt(game, spell, caster, *, remains=False):
    """One defending-player choice; Fated use is shared across both timing windows."""
    from spell_system import Spell, dispel_result, is_dispelled, miscast_result
    side = 3 - side_of(game, caster)
    token = current_turn(game)
    if spell.perfect and not remains:
        rule_skipped('Dispel', caster, f'{spell.name}: perfect invocation cannot be dispelled immediately')
        return False
    blocked = getattr(game, 'dispelBlockedTurns', {})
    if blocked.get(str(side)) == token:
        rule_skipped('Dispel', caster, f'{spell.name}: player {side} cannot dispel again this turn')
        return False
    members = side_members(game, side)
    if not members:
        return False
    owner = members[0]
    options = {}
    for wizard in members:
        if not wizard.unit.model.is_wizard():
            continue
        reason = wizard_reason(game, wizard, spell, remains=remains)
        if reason:
            rule_skipped('Wizardly Dispel', wizard, f'{spell.name}: {reason}')
        else:
            options[f'Wizardly: {wizard.unitName}'] = wizard
    fated = getattr(game, 'fatedDispelTurns', {})
    if fated.get(str(side)) != token:
        options['Fated dispel'] = None
    else:
        rule_skipped('Fated Dispel', owner, f'{spell.name}: player {side} already used its one attempt this turn')
    if not options:
        return False
    threshold = spell.casting_value if remains else spell.casting
    if game.aiControls(owner):
        choice = max(options, key=lambda label: options[label].unit.model.wizard_level(0)
                     if options[label] is not None else -1)
    else:
        choice = await game.makeChoiceNew(
            [*options, 'Pass'], Vec3(0, 0, 10), owner=owner,
            prompt=f'{spell.name}: dispel?',
            detail=f'Beat {threshold}; ' + ('minimum casting value (Remains in Play)' if remains
                                            else 'casting result'))
    if choice not in options:
        rule_skipped('Dispel', owner, f'{spell.name}: player {side} declines; Fated use retained')
        return False
    wizard = options[choice]
    if wizard is None:
        game.fatedDispelTurns = {**fated, str(side): token}
    if remains:
        spell.dispel_attempt_turn = token
    total, dice = await Spell._roll_casting_dice(position_base=Vec3(-20, 0, 10))
    level = wizard.unit.model.wizard_level(0) if wizard is not None else 0
    result = dispel_result(dice, level, wizardly=wizard is not None)
    stopped = dice == [6, 6] or is_dispelled(result, threshold)
    if dice == [1, 1] and wizard is not None:
        roll, table_dice = await Spell._roll_casting_dice()
        entry = miscast_result(roll)
        stopped = entry['cast']
        if entry['no_more_spells']:
            wizard.dispelBlockedTurn = token
            if entry['perfect']:
                game.dispelBlockedTurns = {**blocked, str(side): token}
        rule_log('Outclassed in the Art', wizard,
                 f'{spell.name}: double 1; table {table_dice} = {roll}, {entry["name"]}; '
                 f'{"dispelled" if stopped else "not dispelled"}; '
                 f'{"further dispels blocked this turn" if entry["no_more_spells"] else "no dispel lockout"} (pp. 109-110)')
        if entry['strength']:
            from miscasts import resolve_miscast_damage
            resolve_miscast_damage(game, wizard, entry, context='Outclassed in the Art')
    rule_log('Wizardly Dispel' if wizard is not None else 'Fated Dispel', wizard or owner,
             f'{spell.name}: {dice} + {result - total} = {result} vs {threshold}; '
             f'{"Unbinding; " if dice == [6, 6] else ""}'
             f'{"dispelled" if stopped else "holds (must exceed)"} (p. 110)')
    return stopped


def conjuration_spells(game):
    token = current_turn(game)
    return [spell for spell in getattr(game, 'remainsInPlay', [])
            if side_of(game, spell.caster) != game.roundCounter.current_player
            and (not spell.lifecycle or spell.lifecycle['cast_turn'] != token)
            and getattr(spell, 'dispel_attempt_turn', None) != token]


async def conjuration(game):
    """Resolve remaining RIP choices before leaving Conjuration (pp. 111, 117)."""
    from spell_effects import end_effect
    game.magicBusy = True
    try:
        for spell in conjuration_spells(game):
            if await attempt(game, spell, spell.caster, remains=True):
                end_effect(spell, 'dispelled during Conjuration against minimum casting value')
        game.conjurationDoneTurn = current_turn(game)
    finally:
        game.magicBusy = False