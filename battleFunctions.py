import random
from toHitAndToWound import *

def simulate_attack(model1,model2):
    model1.attack_roll = random.randint(1, 6)
    model1.wound_roll = random.randint(1, 6)
    natural_wound = model1.wound_roll
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
        model1.AP = model1.melee_ap()
        if model1.attack_roll >= to_hit_target_roll:
            hit = True
        else:
            hit = False
    else:
        hit = to_hit_ranged(model1,long_range=getattr(model1,'at_long_range',False),multiple_shots=getattr(model1,'firing_multiple',False))
        model1.characteristics['S'] = model1.equipedWeapon.get('ranged_strength')
        model1.AP = model1.equipedWeapon.get('ranged_AP', model1.AP)
        #print(model1.characteristics['S'])


    to_wound_roll = to_wound(model1,model2)

    
    if hit and model1.wound_roll >= to_wound_roll:
        wound = True
    else:
        wound = False
    # Armour Bane (X): a natural 6 to wound improves this attack's AP by X.
    bane = model1.armour_bane_for_attack() if (wound and natural_wound == 6) else 0
    model1.attack_AP = model1.AP + bane
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
    unit1.nmodels = max(0, unit1.nmodels)  # Ensure at least one model
    # Apply the active melee weapon's always-on/charge Strength bonus once.
    unit1.model.charging = bool(charge)
    unit1.model.apply_melee_strength()
    if charge:
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
        w = unit1.model.equipedWeapon
        # Multiple Shots: fire multiple by default (-1 To Hit); dice count is
        # rolled separately for each firing model.
        unit1.model.firing_multiple = bool(w.get('ranged_shots_dice')) or (w.get('ranged_shots') or 1) > 1
        firing_models = min(unit1.files, unit1.nmodels)  # front rank
        if any(r.get('volley_fire') for r in unit1.model.special_rules):
            second_rank = min(unit1.files, max(0, unit1.nmodels - firing_models))
            firing_models += (second_rank + 1) // 2  # half of the second rank
        attacks = sum(unit1.model.roll_ranged_shots() for _ in range(firing_models))

    for rule in unit1.model.special_rules:
        if rule.get('to_modify_stat'):
            rule['to_modify_stat'](unit1.model)

    for rule in unit2.model.special_rules:
        if rule.get('to_modify_stat'):
            rule['to_modify_stat'](unit2.model)


    attacks1 = attacks
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
            if check_armor_save(unit2.model,unit2.model.armor_save, getattr(unit1.model, 'attack_AP', unit1.model.AP)):
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


