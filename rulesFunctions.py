import random

def plus1attacks(model_instance):
            """
            Adds +1 to a model's Attacks characteristic when charging.
            
            Args:
                model_instance: The model to modify
                
            Returns:
                int: The modified number of attacks
            """
            base_attacks = int(model_instance.characteristics.get('A', 0))
            model_instance.characteristics['A'] = base_attacks + 1
            return

def reroll1d6(roll,value_to_reroll,doCheckOrNot=False):
    if doCheckOrNot and roll in value_to_reroll:
        return random.randint(1, 6)
    return roll

def plusAP(model_instance, AP_increase, roll):
    base_AP = model_instance.AP
    model_instance.AP = base_AP + AP_increase
    return roll
    
def plusSTAT(model_instance, STAT, STAT_increase, roll):
    base_STAT = int(model_instance.characteristics.get(STAT, 0))
    model_instance.characteristics[STAT] = base_STAT + STAT_increase
    return roll
