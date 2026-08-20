

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

def to_wound(model1,model2):
    str1 = stat_value(model1.characteristics.get('S'))

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

def to_hit_ranged(model1,moved=False,long_range=False,stand_and_shoot=False,partial_cover=False,full_cover=False,multiple_shots=False,target_skirmisher=False):
    bs1 = stat_value(model1.characteristics.get('BS'))
    if bs1 <= 0:
        return False   # BS 0: no ranged ability at all

    #hit_roll = random.randint(1, 6)
    hit_roll = model1.attack_roll
    #print(f"Ranged attack roll: {hit_roll} against BS {bs1}")
    # Some weapons (e.g. Blunderbuss) ignore certain To Hit penalties.
    ignore = set((getattr(model1, 'equipedWeapon', None) or {}).get('ignore_to_hit_penalties', []))
    if moved:
        bs1 -= 1
    if long_range and 'long_range' not in ignore:
        bs1 -= 1
    if stand_and_shoot and 'stand_and_shoot' not in ignore:
        bs1 -= 1
    if partial_cover:
        bs1 -= 1
    if full_cover:
        bs1 -= 2
    if multiple_shots and 'multiple_shots' not in ignore:
        bs1 -= 1
    # Enemy fire at a unit of US1 Skirmishers suffers -1 To Hit (not ignorable).
    if target_skirmisher:
        bs1 -= 1
    
    if bs1 == 1 and hit_roll >= 6:
        return True
    elif bs1 == 2 and hit_roll >= 5:
        return True
    elif bs1 == 3 and hit_roll >= 4:
        return True
    elif bs1 == 4 and hit_roll >= 3:
        return True
    elif bs1 == 5 and hit_roll >= 2:
        return True
    else:
        return False