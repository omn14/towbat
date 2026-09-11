"""Source-owned spell lifetimes (Rulebook p. 111; Magic FAQ v1.5.3)."""

from characters import side_of
from magic_items import current_turn
from rules_log import rule_log, rule_skipped


def active_spells(game):
    if game is None:
        return []
    return [*getattr(getattr(game, 'fsm', None), 'endOfTurnSpells', []),
            *getattr(game, 'remainsInPlay', [])]


def register(spell, target=None, *, duration='next_start', phase=None):
    """Record ownership and an explicit boundary, without touching base stats (p. 111)."""
    spell.affected_unit = target
    spell.ended = False
    game = spell.game
    if game is not None and spell.caster is not None and current_turn(game) is not None:
        spell.lifecycle = dict(duration=duration, cast_turn=current_turn(game),
                               owner=side_of(game, spell.caster), phase=phase)
    collection = (game.remainsInPlay if duration == 'remains' and game is not None
                  else spell.duration_list)
    if collection is not None and spell not in collection:
        collection.append(spell)
    if game is not None:
        rule_log(spell.name, target or spell.caster,
                 f'effect active; duration {duration}; source '
                 f'{getattr(getattr(spell.caster, "unit", None), "name", "unknown caster")} (p. 111)')


def grant_rule(spell, target, rule):
    """Identical spells share a grant, but retain independent lifetimes (FAQ v1.5.3)."""
    existing = next((other for other in active_spells(spell.game)
                     if other is not spell and other.name == spell.name
                     and getattr(other, 'affected_unit', None) is target
                     and getattr(other, 'rule', None) is not None), None)
    spell.rule = existing.rule if existing is not None else dict(rule)
    if not any(entry is spell.rule for entry in target.unit.model.special_rules):
        target.unit.model.special_rules.append(spell.rule)
    elif existing is not None:
        rule_skipped(spell.name, target, 'same spell already grants this effect; not cumulative (FAQ v1.5.3)')


def revoke_rule(spell):
    target, rule = getattr(spell, 'affected_unit', None), getattr(spell, 'rule', None)
    if target is None or rule is None:
        return
    if any(other is not spell and not getattr(other, 'ended', False)
           and getattr(other, 'rule', None) is rule for other in active_spells(spell.game)):
        return
    target.unit.model.special_rules[:] = [entry for entry in target.unit.model.special_rules if entry is not rule]


def end_effect(spell, reason):
    """Remove exactly this source, once, before its cleanup queries other sources."""
    if getattr(spell, 'ended', False):
        return
    spell.ended = True
    collections = [spell.duration_list]
    if spell.game is not None:
        collections.extend([spell.game.fsm.endOfTurnSpells, spell.game.remainsInPlay])
    for collection in collections:
        if collection is not None and spell in collection:
            collection.remove(spell)
    cleanup = getattr(spell, 'remove_effect', spell.endSpell)
    cleanup()
    rule_log(spell.name, getattr(spell, 'affected_unit', None) or spell.caster,
             f'effect ends: {reason} (p. 111)')


def start_turn(game):
    token = current_turn(game)
    for spell in active_spells(game):
        state = getattr(spell, 'lifecycle', None)
        if (state and token and state['duration'] == 'next_start'
                and state['owner'] == token[0] and state['cast_turn'] != token):
            end_effect(spell, f'player {token[0]} next Start of Turn')


def end_turn(game):
    for spell in list(game.fsm.endOfTurnSpells):
        state = getattr(spell, 'lifecycle', None)
        if state:
            if state['duration'] == 'end_turn':
                end_effect(spell, 'end of turn')
        else:
            spell.ticks_remaining -= 1
            if spell.ticks_remaining <= 0:
                end_effect(spell, 'legacy duration elapsed')


def end_phase(game, phase):
    for spell in active_spells(game):
        state = getattr(spell, 'lifecycle', None)
        if state and state['duration'] == 'end_phase' and state.get('phase') == phase:
            end_effect(spell, f'end of {phase}')


def recasting(game, caster, name):
    """A Remains in Play spell ends when its caster attempts it again (FAQ v1.5.3)."""
    for spell in list(getattr(game, 'remainsInPlay', [])):
        if spell.caster is caster and spell.name == name:
            end_effect(spell, 'caster attempts to cast the same spell again')


def caster_removed(game, caster):
    joined = getattr(caster, 'joinedCharacter', None)
    for spell in active_spells(game):
        if getattr(spell, 'self_scope', False) and (spell.caster is caster or spell.caster is joined):
            end_effect(spell, 'Self spell caster slain or leaves the battlefield')
    for spell in list(getattr(game, 'remainsInPlay', [])):
        if spell.caster is caster or (joined is not None and spell.caster is joined):
            end_effect(spell, 'caster slain or leaves the battlefield')


def refresh_self_spells(caster):
    """Self-spell host benefits follow the caster's presence (Magic FAQ v1.5.3)."""
    for spell in list(getattr(caster, '_self_spells', [])):
        if not spell.ended:
            spell.refresh()


async def choose_ending(game):
    """Voluntary RIP ending at exposed phase boundaries (Rulebook p. 111)."""
    from panda3d.core import Vec3
    for spell in list(getattr(game, 'remainsInPlay', [])):
        if spell.caster is None:
            continue
        if game.aiControls(spell.caster):
            choice = 'Keep'
        else:
            choice = await game.makeChoiceNew(
                ['Keep', 'End spell'], Vec3(0, 0, 10), owner=spell.caster,
                prompt=f'{spell.name}: Remains in Play',
                detail=f'Caster: {spell.caster.unit.name}')
        if choice == 'End spell':
            end_effect(spell, 'caster voluntarily ends it at the phase boundary')
        else:
            rule_skipped(spell.name, spell.caster, 'caster keeps Remains in Play effect at phase boundary')


async def advance_with_choices(fsm):
    game = fsm.game
    game.magicBusy = True
    try:
        await choose_ending(game)
    finally:
        game.magicBusy = False
    fsm._magic_boundary_ready = True
    try:
        fsm.nextPhase()
    finally:
        fsm._magic_boundary_ready = False