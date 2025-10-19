import random
from toHitAndToWound import *

def simulate_attack(model1,model2):
    model1.attack_roll = random.randint(1, 6)
    model1.wound_roll = random.randint(1, 6)
    for rule in model1.special_rules:
        if rule.get('to_wound'):
            model1.wound_roll = rule['to_wound'](model1.wound_roll,model1)
    to_hit_roll = to_hit(model1,model2)
    to_wound_roll = to_wound(model1,model2)

    if model1.attack_roll >= to_hit_roll:
        hit = True
    else:
        hit = False
    if hit and model1.wound_roll >= to_wound_roll:
        wound = True
    else:
        wound = False
    return hit, wound


    

def simulate_attack_ranged(model1,model2):
    model1.attack_roll = random.randint(1, 6)
    model1.wound_roll = random.randint(1, 6)
    for rule in model1.special_rules:
        if rule.get('to_wound'):
            model1.wound_roll = rule['to_wound'](model1.wound_roll,model1)
        if rule.get('to_hit'):
            model1.attack_roll = rule['to_hit'](model1.attack_roll,model1)
    
    hit = to_hit_ranged(model1,long_range=False)
    model1.characteristics['S'] = model1.weapons['short bow']['ranged_strength']
    print(model1.characteristics['S'])
    to_wound_roll = to_wound(model1,model2)
    print(f"Ranged wound roll: {model1.wound_roll} against to wound {to_wound_roll}")
    if hit and model1.wound_roll >= to_wound_roll:
        wound = True
    else:
        wound = False
    return hit, wound



def check_armor_save(model, armor_save_value, AP):
    armor_save_roll = random.randint(1, 6)
    for rule in model.special_rules:
        if rule.get('to_save'):
            armor_save_roll = rule['to_save'](armor_save_roll)
    if armor_save_roll - AP >= armor_save_value:
        return True
    return False

def simulate_battle(unit1, unit2,charge: bool):

    # how many attacks
    if charge:
        unit1.model.charging = True
        for rule in unit1.model.special_rules:
            if rule.get('charge'):
                #attacks = (int(unit1.model.characteristics.get('A', 0)) + 1) * unit1.files #front rank attacks
                #print(unit1.model.special_rules[1]['charge'](unit1.model))
                rule['charge'](unit1.model)
            else:
                attacks = int(unit1.model.characteristics.get('A', 0)) * unit1.files #front rank attacks
        attacks = int(unit1.model.characteristics.get('A', 0)) * unit1.files
    else: #defends
        attacks = int(unit1.model.characteristics.get('A', 0)) * unit1.files #front rank attacks
        if attacks >= int(unit1.model.characteristics.get('A', 0)) *unit1.nmodels: 
            attacks = int(unit1.model.characteristics.get('A', 0)) *unit1.nmodels # cannot attack more than you have models in front rank
        elif unit1.nmodels % unit1.files > 0: # uncomplete second rank
            attacks +=   (unit1.nmodels % unit1.files) # only one attack if not in base contact
        else:
            attacks += unit1.files # full second rank
    attacks1 = attacks 
    print(f"Total attacks by {unit1.name} on {unit2.name}: {attacks1}")
    total_hits = 0
    total_wounds = 0
    suffered_wounds = 0
    saves_made = 0
    for i in range(attacks1):
        hit,wound = simulate_attack(unit1.model, unit2.model)
        if hit:
            total_hits += 1
        if wound:
            total_wounds += 1
            suffered_wounds += 1
        if wound:
            print(unit1.model.AP)
            if check_armor_save(unit2.model,unit2.model.armor_save, unit1.model.AP):
                saves_made += 1
                total_wounds -= 1
    
    unit1.model.reset_characteristics()
    unit2.model.reset_characteristics()
    

    return attacks,total_hits, suffered_wounds,  saves_made, total_wounds

def simulate_battle_ranged(unit1, unit2, charge: bool):

    # how many attacks
    if charge:
        unit1.model.charging = True
        for rule in unit1.model.special_rules:
            if rule.get('charge'):
                #attacks = (int(unit1.model.characteristics.get('A', 0)) + 1) * unit1.files #front rank attacks
                #print(unit1.model.special_rules[1]['charge'](unit1.model))
                rule['charge'](unit1.model)
            else:
                attacks = int(unit1.model.characteristics.get('A', 0)) * unit1.files #front rank attacks
        attacks = int(unit1.model.characteristics.get('A', 0)) * unit1.files
    else: #defends
        attacks = int(unit1.model.characteristics.get('A', 0)) * unit1.files #front rank attacks
        if attacks >= int(unit1.model.characteristics.get('A', 0)) *unit1.nmodels: 
            attacks = int(unit1.model.characteristics.get('A', 0)) *unit1.nmodels # cannot attack more than you have models in front rank
        elif unit1.nmodels % unit1.files > 0: # uncomplete second rank
            attacks +=   (unit1.nmodels % unit1.files) # only one attack if not in base contact
        else:
            attacks += unit1.files # full second rank
    attacks1 = attacks 
    print(f"Total attacks by {unit1.name} on {unit2.name}: {attacks1}")
    total_hits = 0
    total_wounds = 0
    suffered_wounds = 0
    saves_made = 0
    for i in range(attacks1):
        hit,wound = simulate_attack_ranged(unit1.model, unit2.model)
        if hit:
            total_hits += 1
        if wound:
            total_wounds += 1
            suffered_wounds += 1
        if wound:
            print(unit1.model.AP)
            if check_armor_save(unit2.model,unit2.model.armor_save, unit1.model.AP):
                saves_made += 1
                total_wounds -= 1
    
    unit1.model.reset_characteristics()
    unit2.model.reset_characteristics()
    

    return attacks,total_hits, suffered_wounds,  saves_made, total_wounds