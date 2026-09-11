"""Gaze of the Gods (RH pp. 81, 116) and amended Stupidity (Rulebook p. 178)."""

import random

from characters import side_of
from magic_items import current_turn
from panda3d.core import Vec3
from psychology import _stat_int, leadership_passed, reroll_leadership
from rules_log import rule_log, rule_skipped


GIFTS = {
    2: ('Unnatural Quickness', ('I',), True),
    3: ('Iron Skin', ('T',), True),
    4: ('Murderous Mutation', ('WS',), False),
    5: ('Dark Fury', ('A',), False),
    6: ('Apotheosis', ('S', 'Ld'), False),
}


def has_rule(member, name):
    return any(rule.get('name', '').casefold() == name.casefold()
               for rule in getattr(member.unit.model, 'special_rules', []))


def has_stupidity(member):
    if has_rule(member, 'Stupidity') or getattr(member, 'gazeState', {}).get('stupidity'):
        return True
    mount = getattr(member.unit.model, 'get_mount', lambda: None)()
    return bool(mount and any(rule.get('name', '').casefold() == 'stupidity'
                             for rule in mount.special_rules))


def subject_to_stupidity(member):
    host = getattr(member, 'hostUnit', None) or member
    joined = getattr(host, 'joinedCharacter', None)
    return has_stupidity(host) or bool(joined and has_stupidity(joined))


def succumbed(member):
    host = getattr(member, 'hostUnit', None) or member
    return bool(getattr(host, 'stupidityFailed', False)) and subject_to_stupidity(host)


def apply_gift(member, roll):
    """Only the character changes; temporary deltas expire next own Start of Turn (RH p. 116)."""
    if roll not in range(1, 7):
        raise ValueError('Gaze of the Gods requires a D6 result')
    state = dict(getattr(member, 'gazeState', {}))
    changes = []
    if roll == 1 and not has_stupidity(member):
        state['stupidity'] = True
        name = 'Damned by Chaos'
        changes.append('gains Stupidity for the remainder of the game')
    else:
        name, stats, temporary = ('Damned by Chaos', ('Ld',), False) if roll == 1 else GIFTS[roll]
        for stat in stats:
            before = _stat_int(member.unit.model.characteristics, stat, 0)
            after = max(2, before - 1) if roll == 1 else min(10, before + 1)
            member.unit.model.characteristics[stat] = after
            baseline = getattr(member.unit.model, '_base_characteristics', None)
            if baseline is not None:
                baseline[stat] = _stat_int(baseline, stat, 0) + after - before
            if temporary:
                pending = dict(state.get('temporary', {}))
                pending[stat] = pending.get(stat, 0) + after - before
                state['temporary'] = pending
            changes.append(f'{stat} {before} -> {after}' + (' until next Start of Turn' if temporary else ' permanently'))
    member.gazeState = state
    rule_log('Gaze of the Gods', member, f'D6={roll}, {name}: {"; ".join(changes)}; character only (RH p. 116)')


def expire_gifts(member):
    state = dict(getattr(member, 'gazeState', {}))
    for stat, delta in state.pop('temporary', {}).items():
        before = _stat_int(member.unit.model.characteristics, stat, 0)
        member.unit.model.characteristics[stat] = before - delta
        baseline = getattr(member.unit.model, '_base_characteristics', None)
        if baseline is not None:
            baseline[stat] = _stat_int(baseline, stat, 0) - delta
        rule_log('Gaze of the Gods', member, f'Start of Turn: temporary {stat} {before} -> {before - delta}')
    member.gazeState = state


async def start_and_command(game):
    """Resolve mandatory tests before offering the character's optional Command roll (pp. 117, 178)."""
    token = current_turn(game)
    if getattr(game, 'chaosCommandTurn', None) == token:
        return
    owned = [member for member in game.units if side_of(game, member, None) == game.roundCounter.current_player]
    for member in owned:
        expire_gifts(member)
        member.stupidityFailed = False
    async def dice():
        return [random.randint(1, 6), random.randint(1, 6)]
    for member in owned:
        if getattr(member, 'hostUnit', None) is not None:
            continue
        if member.unit.nmodels <= 0 or member.bodyNP.isEmpty() or not getattr(member, 'isDeployed', True):
            continue
        if not subject_to_stupidity(member):
            continue
        if member.state == 'IsFleeing' or getattr(member, 'isInCombat', False):
            rule_skipped('Stupidity', member, 'fleeing or engaged: no Start of Turn test (p. 178)')
            continue
        from warband import leadership_for_test
        leadership = leadership_for_test(game.psychology, member, 'Stupidity')[0]
        rolled = await reroll_leadership(game, member, 'Stupidity', await dice(), leadership, dice)
        member.stupidityFailed = not leadership_passed(sum(rolled), leadership)
        joined = getattr(member, 'joinedCharacter', None)
        if joined is not None:
            joined.stupidityFailed = member.stupidityFailed
        rule_log('Stupidity', member, f'2D6={rolled} vs Ld {leadership}: '
                 + ('failed; no movement/shooting/casting/dispelling, Hold only until next Start of Turn'
                    if member.stupidityFailed else 'passed; acts normally'))
    for member in owned:
        if not has_rule(member, 'Gaze of the Gods'):
            continue
        if member.bodyNP.isEmpty() or member.unit.nmodels <= 0 or not getattr(member, 'isDeployed', True):
            continue
        choice = 'Roll' if game.aiControls(member) else await game.makeChoiceNew(
            ['Roll', 'Decline'], Vec3(0, 0, 10), owner=member, prompt=f'{member.unit.name}: Gaze of the Gods')
        if choice == 'Roll':
            apply_gift(member, random.randint(1, 6))
        else:
            rule_skipped('Gaze of the Gods', member, 'owner declines the optional Command roll')
    game.chaosCommandTurn = token


def begin_turn(game):
    """Hold Strategy input until required Start of Turn tests and Gaze choices finish (p. 117)."""
    if getattr(game, 'restoringBattle', False):
        return
    owned = [member for member in game.units
             if side_of(game, member, None) == game.roundCounter.current_player]
    if not any(has_rule(member, 'Gaze of the Gods') or subject_to_stupidity(member)
               or getattr(member, 'gazeState', {}).get('temporary') for member in owned):
        return
    if getattr(game, 'chaosCommandTurn', None) == current_turn(game):
        return
    if getattr(game, 'chaosCommandBusy', False):
        return
    game.chaosCommandBusy = True
    previous = getattr(game, 'magicBusy', False)
    game.magicBusy = True

    async def resolve():
        try:
            await start_and_command(game)
        finally:
            game.chaosCommandBusy = False
            game.magicBusy = previous
    game.taskMgr.add(resolve(), 'chaosCommandTask')