import requests
from bs4 import BeautifulSoup
import random
import json
import matplotlib.pyplot as plt
import numpy as np
import os

class model:
    def __init__(self, name: str, url: str):
        self.name = name
        self.url = url
        self.characteristics = {}
        self.model_data = self.fetch_model_data(url)
        self.json_file_path = self.name.replace(" ", "_").lower() + '_characteristics.json'
        if os.path.isfile(self.json_file_path):
            self.characteristics = load_dict_from_file(self.json_file_path)
        else:
            self.characteristics = self.get_characteristics_from_html(self.model_data)
            store_dict_to_file(self.characteristics, self.json_file_path)
        self.armor_save = 7
        self.AP = 0  # Armor Penetration

    def reset_characteristics(self):
        if os.path.isfile(self.json_file_path):
            self.characteristics = load_dict_from_file(self.json_file_path)
        
        self.AP = 0  # Armor Penetration


    def fetch_model_data(self,url: str) -> dict:
        """
        Fetch data from the model wiki page.

        Returns:
            dict: Parsed JSON response from the wiki page.
        """
        #url = "https://www.newrecruit.eu/wiki/tow/warhammer-the-old-world/orc-and-goblin-tribes/907e-90b-b5a5-a8a3/black-orc"
        response = requests.get(url)
        response.raise_for_status()
        return response

    def get_characteristics_from_html(self,html_content: str) -> dict:
        """
        Extract characteristics from HTML content.

        Args:
            html_content (str): HTML content as a string.

        Returns:
            dict: Dictionary of characteristics and their values.
        """
        soup = BeautifulSoup(html_content.text, 'html.parser')
        table = soup.find('table')
        if table:
            print("Table found in the HTML response.")
        else:
            print("No table found in the HTML response.")
            return {}

        pairs = []
        characteristics = []
        values = []
        if table:
            rows = table.find_all('tr')
            #print(rows)
            for row in rows:
                #print(str(row)+"\n")
                cols = row.find_all(['td', 'th'])
                #print(cols)
                if len(cols) >= 2:
                    for c in cols:
                        #print(c.get_text(strip=True))
                        characteristics.append(c.get_text(strip=True))
            
            # Reshape the characteristics list into two rows
            if len(characteristics) % 2 == 0:
                characteristics = [characteristics[:len(characteristics)//2], characteristics[len(characteristics)//2:]]
            else:
                characteristics = [characteristics[:len(characteristics)//2], characteristics[len(characteristics)//2:]]
            #print(characteristics)
            pairs = dict(zip(characteristics[0], characteristics[1]))
            #print(list(pairs.keys()))
            #print(pairs.get('Ld'))
        return pairs

class BlackOrc(model):
    def __init__(self, name: str, url: str):
        super().__init__(name, url)
        # Additional Black Orc specific attributes can be added here
        self.special_rules = []
        self.special_rules.append({'name': 'Always strikes first', 
                                   'description': 'This model always strikes first in combat.',
                                   'tag': 'combat'})
        self.special_rules.append({'name': 'Furious charge',
                                   'description': 'This model adds +1 to its Attacks characteristic when it charges.',
                                   'tag': 'combat',
                                   'charge': plus1attacks})
        self.special_rules.append({'name': 'extra AP on charge',
                                   'description': 'This model adds +1 to its Armor Penetration (AP) when it charges.',
                                   'tag': 'combat',
                                   'charge': lambda model_instance: setattr(model_instance, 'AP', (model_instance.AP + 1)*1)})
        self.AP = 1  # Example Armor Penetration value for Black Orcs

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

def plus1AP(model_instance):
    base_AP = model_instance.AP
    model_instance.AP = base_AP + 1
    return

class DetailedModel(model):
    def __init__(self, name: str, url: str, model_type: str = "infantry", points_cost: int = 0):
        super().__init__(name, url)
        self.model_type = model_type  # infantry, cavalry, monster, etc.
        self.points_cost = points_cost
        self.weapons = []
        self.special_rules = []
        
    def add_weapon(self, weapon_name: str, strength_modifier: int = 0, attacks_modifier: int = 0):
        """Add a weapon to the model with optional stat modifiers."""
        self.weapons.append({
            'name': weapon_name,
            'strength_modifier': strength_modifier,
            'attacks_modifier': attacks_modifier
        })
        
    def add_special_rule(self, rule_name: str, rule_description: str):
        """Add a special rule or ability to the model."""
        self.special_rules.append({
            'name': rule_name,
            'description': rule_description
        })
        
    def get_effective_strength(self):
        """Calculate effective strength including weapon modifiers."""
        base_strength = int(self.characteristics.get('S', 0))
        max_modifier = max([w['strength_modifier'] for w in self.weapons], default=0)
        return base_strength + max_modifier
        
    def get_effective_attacks(self):
        """Calculate effective attacks including weapon modifiers."""
        base_attacks = int(self.characteristics.get('A', 0))
        total_modifier = sum(w['attacks_modifier'] for w in self.weapons)
        return base_attacks + total_modifier

class unit:
    def __init__(self, name: str, model: model, nmodels: int, files: int, ranks: int):
        self.name = name
        self.nmodels = nmodels
        self.model = model
        self.files = files
        self.ranks = ranks
        

def compare_model_stats(model1: model, model2: model,stat: str):
    """
    Compare a specific stat between two models.

    Args:
        model1 (model): First model instance.
        model2 (model): Second model instance.
        stat (str): The characteristic to compare.

    Returns:
        str: Comparison result.
    """
    val1 = model1.characteristics.get(stat)
    val2 = model2.characteristics.get(stat)

    if val1 is None or val2 is None:
        return f"One of the models does not have the characteristic '{stat}'."

    try:
        val1 = int(val1)
        val2 = int(val2)
    except ValueError:
        return f"Characteristic '{stat}' is not a numeric value."

    if val1 > val2:
        return f"{model1.name} has a higher {stat} ({val1}) than {model2.name} ({val2})."
    elif val1 < val2:
        return f"{model2.name} has a higher {stat} ({val2}) than {model1.name} ({val1})."
    else:
        return f"Both models have the same {stat} ({val1})."
    
def store_dict_to_file(data: dict, filename: str):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def load_dict_from_file(filename: str) -> dict:
    with open(filename, 'r') as f:
        return json.load(f)
    
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
    
    return f"could not calculate to hit"

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

def simulate_attack(model1,model2):
    to_hit_roll = to_hit(model1,model2)
    to_wound_roll = to_wound(model1,model2)
    attack_roll = random.randint(1, 6)
    wound_roll = random.randint(1, 6)
    if attack_roll >= to_hit_roll:
        hit = True
    else:
        hit = False
    if hit and wound_roll >= to_wound_roll:
        wound = True
    else:
        wound = False
    return hit, wound

def check_armor_save(armor_save_value, AP):
    armor_save_roll = random.randint(1, 6)
    if armor_save_roll - AP >= armor_save_value:
        return True
    return False

def simulate_battle(unit1, unit2,charge: bool):

    # how many attacks
    if charge:
        
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
        if attacks > int(unit1.model.characteristics.get('A', 0)) *unit1.nmodels: 
            attacks = int(unit1.model.characteristics.get('A', 0)) *unit1.nmodels # cannot attack more than you have models in front rank
        elif unit1.nmodels % unit1.files > 0: # uncomplete second rank
            attacks +=   (unit1.nmodels % unit1.files)
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
            if check_armor_save(unit2.model.armor_save, unit1.model.AP):
                saves_made += 1
                total_wounds -= 1
    
    unit1.model.reset_characteristics()
    unit2.model.reset_characteristics()
    

    return attacks,total_hits, suffered_wounds,  saves_made, total_wounds

def battle_graph(total_attacks, hits,suffered_wounds , saves, total_wounds):
    plt.figure()
    values = [total_attacks, hits,suffered_wounds , saves, total_wounds]
    labels = ['Total Attacks', 'Hits', 'Wounds', 'Saves', 'Total Wounds']

    plt.bar(labels, values, color=['blue', 'green', 'red', 'orange', 'purple'])
    plt.ylabel('Count')
    plt.title('Battle Simulation Results')
    plt.ylim(0, 15)
    # Annotate each bar with its value
    for i, v in enumerate(values):
        plt.text(i, v + 0.1, f"{v:.2f}", ha='center', va='bottom')
    

url_black_orc = "https://www.newrecruit.eu/wiki/tow/warhammer-the-old-world/orc-and-goblin-tribes/907e-90b-b5a5-a8a3/black-orc"
url_man_at_arm = "https://www.newrecruit.eu/wiki/tow/warhammer-the-old-world/kingdom-of-bretonnia/3ddf-271a-aaec-73eb/man-at-arms"
url_saurus_warrior = "https://www.newrecruit.eu/wiki/tow/warhammer-the-old-world/lizardmen/65aee1f-83430cad/saurus-warrior"

black_orc = BlackOrc("Black Orc", url_black_orc)
black_orc.armor_save = 3
man_at_arm = model("Man_at_Arm", url_man_at_arm)
man_at_arm.armor_save = 7
saurus_warrior = model("Saurus Warrior", url_saurus_warrior)
saurus_warrior.armor_save = 3

black_orc_unit = unit("Black Orc Unit", black_orc, 10,5,2)
man_at_arm_unit = unit("Man_at_Arm Unit", man_at_arm, 10,5,2)
saurus_warrior_unit = unit("Saurus Warrior Unit", saurus_warrior, 10,5,2)

print(black_orc.characteristics)
print(man_at_arm.characteristics)
print(saurus_warrior.characteristics)

#store_dict_to_file(black_orc.characteristics, 'black_orc_characteristics.json')
#store_dict_to_file(man_at_arm.characteristics, 'man_at_arm_characteristics.json')

comparison_result = compare_model_stats(black_orc,man_at_arm,'WS')
print(comparison_result)

to_hit_result = to_hit(black_orc, man_at_arm)
print(to_hit_result)
to_wound_result = to_wound(black_orc, man_at_arm)
print(to_wound_result)

results_attacker = []
results_defender = []
attacker = black_orc_unit
defender = saurus_warrior_unit
for i in range(1000):
    defender.nmodels=10
    attacks, total_hits, suffered_wounds,  saves_made, total_wounds = simulate_battle(attacker, defender,True)
    result = [attacks, total_hits, suffered_wounds,  saves_made, total_wounds]
    results_attacker.append(result)
    print(f"Total hits by {attacker.name} on {defender.name}: {total_hits}")
    print(f"suffered wounds by {attacker.name} on {defender.name}: {suffered_wounds}")
    print(f"Saves made by {defender.name}: {saves_made}")
    print(f"Total wounds by {attacker.name} on {defender.name}: {total_wounds}")
    #battle_graph(attacks, total_hits, suffered_wounds, saves_made, total_wounds)
    defender.nmodels-=total_wounds
    attacks, total_hits, suffered_wounds,  saves_made, total_wounds = simulate_battle(defender, attacker,False)
    result = [attacks, total_hits, suffered_wounds,  saves_made, total_wounds]
    results_defender.append(result)
    print(f"Total hits by {defender.name} on {attacker.name}: {total_hits}")
    print(f"suffered wounds by {defender.name} on {attacker.name}: {suffered_wounds}")
    print(f"Saves made by {attacker.name}: {saves_made}")
    print(f"Total wounds by {defender.name} on {attacker.name}: {total_wounds}\n")
    #battle_graph(attacks, total_hits, suffered_wounds, saves_made, total_wounds)

results_attacker = np.array(results_attacker)
attacks=results_attacker[:,0].mean()
hits=results_attacker[:,1].mean()
suffered_wounds=results_attacker[:,2].mean()
saves=results_attacker[:,3].mean()
total_wounds=results_attacker[:,4].mean()
battle_graph(attacks, hits, suffered_wounds, saves, total_wounds)
results_defender = np.array(results_defender)
attacks=results_defender[:,0].mean()
hits=results_defender[:,1].mean()
suffered_wounds=results_defender[:,2].mean()
saves=results_defender[:,3].mean()
total_wounds=results_defender[:,4].mean()
battle_graph(attacks, hits, suffered_wounds, saves, total_wounds)

attacker_wins = results_attacker[:,4]-results_defender[:,4]
plt.figure()
plt.plot(attacker_wins)
plt.title('Attacker Wins (Positive means attacker wins)')
plt.figure()
#plt.hist(attacker_wins, bins=10, edgecolor='black', align='mid')
plt.title('Histogram of Attacker Wins')

# Annotate each bar with its value
# Center bins on integer numbers
min_win = int(np.floor(attacker_wins.min()))
max_win = int(np.ceil(attacker_wins.max()))
bins = np.arange(min_win - 0.5, max_win + 1.5, 1)
counts, bins, patches = plt.hist(attacker_wins, bins=bins, edgecolor='black', align='mid')
for count, bin_left, patch in zip(counts, bins[:-1], patches):
    plt.text(bin_left + (bins[1] - bins[0]) / 2, count, f"{int(count)}", ha='center', va='bottom')
#print(results_attacker)
#print(results_attacker[:,0])
#print(results_defender)
plt.show()