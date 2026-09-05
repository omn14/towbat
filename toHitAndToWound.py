
import random

from battlescribe import has_ponderous, has_quick_shot

# A needed roll of 10 or more cannot be made even by the 7+ chain (p. 139).
TO_HIT_IMPOSSIBLE = 10


def stat_value(value, default=0):
    """A characteristic as a number. Profiles write an absent characteristic as
    '-', which the rules treat as 0 (Rulebook p. 97)."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def to_hit(model1,model2):
    # A chariot attacks with its crew's Weapon Skill and is hit on it too, since
    # the chariot's own profile has none (Rulebook p. 194).
    ws1 = (model1.defending_ws() if hasattr(model1, 'defending_ws')
           else stat_value(model1.characteristics.get('WS')))
    ws2 = (model2.defending_ws() if hasattr(model2, 'defending_ws')
           else stat_value(model2.characteristics.get('WS')))

    # A model with WS 0 cannot defend itself: its own attacks all miss, and
    # blows struck against it hit automatically (Rulebook p. 158).
    if ws1 == 0:
        return 7
    if ws2 == 0:
        return 1

    if ws1 > 2*ws2:
        return 2
    if ws1 > ws2:
        return 3
    if 2*ws1 < ws2:
        return 5
    return 4

def to_wound(model1,model2,strength=None):
    str1 = stat_value(model1.characteristics.get('S')) if strength is None else strength

    # Mounted defenders always use the rider's Toughness.
    if hasattr(model2, 'get_toughness'):
        toughness2 = model2.get_toughness()
    else:
        toughness2 = stat_value(model2.characteristics.get('T'), 4)

    if str1 <= 0 or toughness2 <= 0:
        return 7   # no Strength to wound with, or nothing left to wound

    if str1 == toughness2:
        return 4
    if str1 - toughness2 == 1:
        return 3
    if str1 - toughness2 >= 2:
        return 2
    if str1 - toughness2 == -1:
        return 5
    return 6

def ranged_hit_requirement(model1, moved=False, long_range=False,
                           stand_and_shoot=False, partial_cover=False,
                           full_cover=False, multiple_shots=False,
                           target_skirmisher=False):
    """The D6 numbers a shot needs: (first, re-roll), or None if it cannot shoot.

    Modifiers move the target number rather than Ballistic Skill. For BS1-5 the
    two are the same thing, but not for BS6+, where the target is already 2+ and
    a reduced BS would be a different row of the table rather than a harder roll.
    """
    # A chariot shoots with its crew's Ballistic Skill; its own profile has none.
    bs1 = (model1.firing_bs() if hasattr(model1, 'firing_bs')
           else stat_value(model1.characteristics.get('BS')))
    if bs1 <= 0:
        return None   # BS 0: no ranged ability at all

    # Some weapons (e.g. Blunderbuss) ignore certain To Hit penalties.
    _weapon = getattr(model1, 'equipedWeapon', None) or {}
    ignore = set(_weapon.get('ignore_to_hit_penalties', None) or [])
    # Read from the rule names too, since a save predating the flags keeps them.
    _rules = _weapon.get('special_rules')
    _quick = bool(_weapon.get('quick_shot') or has_quick_shot(_rules))
    _ponderous = bool(_weapon.get('ponderous') or has_ponderous(_rules))
    penalty = 0
    # Moving and Shooting is -1, or -2 for a Ponderous weapon (p. 175), and
    # Quick Shot waives it. A weapon with both rules takes the plain -1: the
    # FAQ has them "effectively cancel one another out".
    if moved and 'moved' not in ignore:
        if _ponderous:
            penalty += 1 if _quick else 2
        elif not _quick:
            penalty += 1
    # A Stand & Shoot suffers no additional modifier for long range (p. 139).
    if long_range and not stand_and_shoot and 'long_range' not in ignore:
        penalty += 1
    if stand_and_shoot and 'stand_and_shoot' not in ignore:
        penalty += 1
    if partial_cover:
        penalty += 1
    if full_cover:
        penalty += 2
    if multiple_shots and 'multiple_shots' not in ignore:
        penalty += 1
    # Enemy fire at a unit of US1 Skirmishers suffers -1 To Hit (not ignorable).
    if target_skirmisher:
        penalty += 1

    if bs1 >= 6:
        # BS6+ hits on 2+ and re-rolls a failure, the re-roll growing easier
        # with Ballistic Skill (p. 138): BS6 2+/6+ through BS10+ 2+/2+.
        return 2 + penalty, max(2, 12 - bs1) + penalty
    return (7 - bs1) + penalty, None


def _target_chance(target: int) -> float:
    """Probability one D6 attempt meets *target*, following the 7+ chain."""
    if target <= 1:
        return 1.0
    if target <= 6:
        return (7 - target) / 6
    if target < TO_HIT_IMPOSSIBLE:
        return (1 / 6) * (7 - (target - 3)) / 6
    return 0.0


def _attempt(target: int, die: int) -> bool:
    """Resolve one attempt whose first D6 has already been rolled."""
    if target <= 6:
        return die >= target
    if target >= TO_HIT_IMPOSSIBLE:
        return False
    # 7+ To Hit: a natural 6 earns a second roll (p. 139) — 7 needs 4+, 8 a 5+,
    # 9 a 6, and 10 or more cannot be rolled at all.
    return die == 6 and random.randint(1, 6) >= target - 3


def to_hit_ranged(model1,moved=False,long_range=False,stand_and_shoot=False,partial_cover=False,full_cover=False,multiple_shots=False,target_skirmisher=False):
    req = ranged_hit_requirement(
        model1, moved=moved, long_range=long_range,
        stand_and_shoot=stand_and_shoot, partial_cover=partial_cover,
        full_cover=full_cover, multiple_shots=multiple_shots,
        target_skirmisher=target_skirmisher)
    if req is None:
        return False
    first, reroll = req
    if _attempt(first, model1.attack_roll):
        return True
    return reroll is not None and _attempt(reroll, random.randint(1, 6))


def ranged_hit_chance(model1, **mods) -> float:
    """Probability that one shot from *model1* hits, given the same keyword
    modifiers to_hit_ranged() takes.

    Read from the same requirement the roll is resolved against, so a caller
    weighing a choice cannot disagree with the dice it is predicting.
    """
    req = ranged_hit_requirement(model1, **mods)
    if req is None:
        return 0.0
    first, reroll = req
    chance = _target_chance(first)
    if reroll is not None:
        chance += (1 - chance) * _target_chance(reroll)
    return chance