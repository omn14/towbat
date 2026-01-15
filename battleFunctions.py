import random
from toHitAndToWound import *

def simulate_attack(model1,model2):
    model1.attack_roll = random.randint(1, 6)
    model1.wound_roll = random.randint(1, 6)
    for rule in model1.special_rules:
        if rule.get('to_wound'):
            model1.wound_roll = rule['to_wound'](model1.wound_roll,model1)
        if rule.get('to_hit'):
            model1.attack_roll = rule['to_hit'](model1.attack_roll,model1)
    
    weaponIsRanged = False

    if model1.equipedWeapon.get('tag') == 'ranged':
        weaponIsRanged = True
    #print(f"Weapon is ranged: {weaponIsRanged}")
    if not weaponIsRanged: # for to hit modifications
        to_hit_target_roll = to_hit(model1,model2)
        if model1.attack_roll >= to_hit_target_roll:
            hit = True
        else:
            hit = False
    else:
        hit = to_hit_ranged(model1,long_range=False)
        model1.characteristics['S'] = model1.equipedWeapon.get('ranged_strength')
        #print(model1.characteristics['S'])


    to_wound_roll = to_wound(model1,model2)

    
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
    print(unit1.name, "has", unit1.nmodels, "models", unit1.files, "files,", unit1.ranks, "ranks")
    unit1.nmodels = max(0, unit1.nmodels)  # Ensure at least one model
    if charge:
        unit1.model.charging = True
        for rule in unit1.model.special_rules:
            if rule.get('charge'):
                rule['charge'](unit1.model)
            else:
                attacks = int(unit1.model.characteristics.get('A', 0)) * unit1.files #front rank attacks
        attacks = int(unit1.model.characteristics.get('A', 0)) * unit1.files
        if attacks >= int(unit1.model.characteristics.get('A', 0)) *unit1.nmodels: 
            attacks = int(unit1.model.characteristics.get('A', 0)) *unit1.nmodels # cannot attack more than you have models in front rank
    else: #defends
        attacks = int(unit1.model.characteristics.get('A', 0)) * unit1.files #front rank attacks
        if attacks >= int(unit1.model.characteristics.get('A', 0)) *unit1.nmodels: 
            attacks = int(unit1.model.characteristics.get('A', 0)) *unit1.nmodels # cannot attack more than you have models in front rank
        elif unit1.nmodels % unit1.files > 0: # uncomplete second rank
            attacks +=   (unit1.nmodels % unit1.files) # only one attack if not in base contact
        else:
            attacks += unit1.files # full second rank
    
    if unit1.model.equipedWeapon.get('tag') == 'ranged':
        attacks = unit1.model.equipedWeapon.get('ranged_shots') * unit1.files #front rank attacks
        if attacks >= unit1.model.equipedWeapon.get('ranged_shots') * unit1.nmodels: 
            attacks = unit1.model.equipedWeapon.get('ranged_shots') * unit1.nmodels # cannot attack more than you have models in front rank

        for rule in unit1.model.special_rules:
            if rule.get('volley_fire'):
                
                if unit1.nmodels % unit1.files > 0: # incomplete second rank
                    attacks += int(((unit1.nmodels % unit1.files) + 1) / 2)  # only one attack if not in base contact
                else:
                    attacks += int((unit1.files + 1) / 2)  # half attacks from second rank
                    print("Volly fire rule applied",unit1.files,attacks)

    for rule in unit1.model.special_rules:
        if rule.get('to_modify_stat'):
            rule['to_modify_stat'](unit1.model)

    for rule in unit2.model.special_rules:
        if rule.get('to_modify_stat'):
            rule['to_modify_stat'](unit2.model)


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
            else:
                for rule in unit2.model.special_rules:
                    if rule.get('regen'):
                        if check_armor_save(unit2.model,rule['regen'], 0):
                            saves_made += 1
                            total_wounds -= 1
            #if total_wounds >= unit2.nmodels:
            #    total_wounds = unit2.nmodels
            #    break # cannot wound more models than you have
    
    unit1.model.reset_characteristics()
    unit2.model.reset_characteristics()
    

    return attacks1,total_hits, suffered_wounds,  saves_made, total_wounds


