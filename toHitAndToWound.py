

def to_hit(model1,model2):
    ws1 = model1.characteristics.get('WS')
    ws2 = model2.characteristics.get('WS')

    if ws1 is None or ws2 is None:
        return f"One of the models does not have the characteristic 'WS'."

    try:
        ws1 = int(ws1)
        ws2 = int(ws2)
    except ValueError:
        return f"Characteristic 'WS' is not a numeric value."

    
    if ws1 > 2*ws2:
        return 2
    if ws1 > ws2:
        return 3
    if 2*ws1 < ws2:
        return 5
    return 4

def to_wound(model1,model2):
    str1 = model1.characteristics.get('S')
    toughness2 = model2.characteristics.get('T')

    if str1 is None or toughness2 is None:
        return f"One of the models does not have the characteristic 'S' or 'T'."

    try:
        str1 = int(str1)
        toughness2 = int(toughness2)
    except ValueError:
        return f"Characteristic 'S' or 'T' is not a numeric value."

    if str1 == toughness2:
        return 4
    if str1 - toughness2 == 1:
        return 3
    if str1 - toughness2 >= 2:
        return 2
    if str1 - toughness2 == -1:
        return 5
    if str1 - toughness2 <= -2:
        return 6
    
    return f"could not calculate to wound"

def to_hit_ranged(model1,moved=False,long_range=False,stand_and_shoot=False,partial_cover=False,full_cover=False):
    bs1 = model1.characteristics.get('BS')

    if bs1 is None:
        return f"The model does not have the characteristic 'BS'."

    try:
        bs1 = int(bs1)
    except ValueError:
        return f"Characteristic 'BS' is not a numeric value."

    #hit_roll = random.randint(1, 6)
    hit_roll = model1.attack_roll
    print(f"Ranged attack roll: {hit_roll} against BS {bs1}")
    if moved:
        bs1 -= 1
    if long_range:
        bs1 -= 1
    if stand_and_shoot:
        bs1 -= 1
    if partial_cover:
        bs1 -= 1
    if full_cover:
        bs1 -= 2
    
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