from rulesFunctions import *
from utilityFunctions import *
from bs4 import BeautifulSoup
import requests

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
        self.charging = False
        self.special_rules = []
        self.weapons = {}
        self.weapons.update({'hand weapon': {'name': 'hand weapon',
                                             'description': 'basic melee weapon',
                                             'tag': 'combat'}})
        self.equipedWeapon = None
        self.equip_weapon('hand weapon')
        self.attack_roll = 0
        self.wound_roll = 0

    def reset_characteristics(self):
        if os.path.isfile(self.json_file_path):
            self.characteristics = load_dict_from_file(self.json_file_path)
        
        self.AP = 0  # Armor Penetration
        self.charging = False


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
    def equip_weapon(self, weapon_name: str):
        try:
            self.equipedWeapon = self.weapons.get(weapon_name)
            self.special_rules.append(self.equipedWeapon)
        except Exception as e:
            print(f"Error equipping weapon '{weapon_name}' for {self.name}: {e}")

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
        self.special_rules.append({'name': 'Reroll 1s to wound when charging',
                                      'description': 'This model can reroll wound rolls of 1 when charging.',
                                      'tag': 'combat',
                                      'charge': lambda model_instance: print("Reroll 1s to wound when charging"),
                                      'to_wound': lambda roll,model_instance: reroll1d6(roll,[1],model_instance.charging)})
        self.AP = 0  # Example Armor Penetration value for Black Orcs

class OrcBoyz(model):
    def __init__(self, name: str, url: str):
        super().__init__(name, url)
        # Additional Orc Boyz specific attributes can be added here
        self.special_rules.append({'name': 'Orc Boyz',
                                   'description': 'This model has special rules for Orc Boyz.',
                                   'tag': 'special'})
        self.AP = 0  # Example Armor Penetration value for Orc Boyz

class SaurusWarrior(model):
    def __init__(self, name: str, url: str):
        super().__init__(name, url)
        # Additional Saurus Warrior specific attributes can be added here
        
        self.special_rules.append({'name': 'Stubborn',
                                   'description': 'This model is stubborn and has a higher Leadership.',
                                   'tag': 'psychology'})
        self.special_rules.append({'name': 'Morks curse',
                                   'description': 'This model must reroll saves of 6',
                                   'tag': 'saving throw',
                                   'to_save': lambda roll: reroll1d6(roll,[6],False)})
        self.AP = 0  # Example Armor Penetration value for Saurus Warriors

        self.weapons.update({
            'spear': {'name': 'spear'},
            'halberd': {'name': 'halberd',
                        'description': 'This model adds +1 to its Armor Penetration (AP) when it charges.',
                        'tag': 'combat',
                        'charge': lambda model_instance: setattr(model_instance, 'AP', (model_instance.AP + 1)*1)}
        })

class NightGoblin(model):
    def __init__(self, name: str, url: str):
        super().__init__(name, url)
        # Additional Night Goblin specific attributes can be added here
        self.special_rules.append({'name': 'Night Goblin',
                                   'description': 'This model has special rules for Night Goblins.',
                                   'tag': 'special'})
        self.AP = 0  # Example Armor Penetration value for Night Goblins

        """ 
        self.special_rules.append({'name': 'AP +2 when wounding roll of 6',
                                   'description': 'This model adds +2 to its Armor Penetration (AP) when it rolls a 6 to wound.',
                                   'tag': 'combat',
                                   'to_wound': lambda roll, model_instance: plusAP(model_instance, 2,roll) if roll == 6 else roll})
         """
        """ 
        self.special_rules.append({'name': 'poison',
                                   'description': 'adds +2 to roll to wound if hit roll is 6.',
                                   'tag': 'combat',
                                   'to_wound': lambda roll, model_instance: roll+2 if model_instance.attack_roll == 6 else roll})
         """
        """ 
        self.special_rules.append({'name': 'Reroll 1s to hit',
                                      'description': 'This model can reroll hit rolls of 1.',
                                      'tag': 'combat',
                                      'to_hit': lambda roll,model_instance: reroll1d6(roll,[1],True)})
        """

        self.weapons.update({
            'short bow': {'name': 'short bow',
                          'description': 'weaker ranged weapon',
                          'tag': 'ranged',
                          'ranged_range': 12,
                          'ranged_shots': 1,
                          'ranged_strength': 3,
                          'ranged_AP': 0,
                          'volley_fire': True}
        })

class MountedKnightOfTheRealm(model):
    def __init__(self, name: str, url: str, mountUnit: model = None):
        super().__init__(name, url)
        # Additional Mounted Knight of the Realm specific attributes can be added here
        self.special_rules.append({'name': 'Mounted Knight of the Realm',
                                   'description': 'This model has special rules for Mounted Knights of the Realm.',
                                   'tag': 'special'})
        self.special_rules.append({'name': 'Mounted',
                                   'description': 'This model has a mount, which grants it additional movement and combat abilities.',
                                   'tag': 'mount',
                                   'mountUnit': mountUnit})


        self.AP = 0  # Example Armor Penetration value for Mounted Knights of the Realm

        self.weapons.update({
            'lance': {'name': 'lance',
                      'description': 'This model adds +2 to its Armor Penetration (AP) when it charges.',
                      'tag': 'combat',
                      'charge': lambda model_instance: setattr(model_instance, 'AP', (model_instance.AP + 2)*1),
                      'charge': lambda model_instance: plusSTAT(model_instance, 'S', 2, -99) },
            'sword': {'name': 'sword'}
        })

class BretonnianWarhorse(model):
    def __init__(self, name: str, url: str):
        super().__init__(name, url)
        # Additional Bretonnian Warhorse specific attributes can be added here
        
        self.special_rules.append({'name': 'Bretonnian Warhorse',
                                   'description': 'This model has special rules for Bretonnian Warhorses.',
                                   'tag': 'special'})
        self.AP = 0  # Example Armor Penetration value for Bretonnian Warhorses

class BardedPegasus(model):
    def __init__(self, name: str, url: str):
        super().__init__(name, url)
        # Additional Barded Pegasus specific attributes can be added here
        self.special_rules.append({'name': 'Barded Pegasus',
                                   'description': 'This model has special rules for Barded Pegasi.',
                                   'tag': 'special'})
        self.AP = 0  # Example Armor Penetration value for Barded Pegasi

class PegasusKnight(model):
    def __init__(self, name: str, url: str, mountUnit: model = None):
        super().__init__(name, url)
        # Additional Pegasus Knight specific attributes can be added here
        self.special_rules.append({'name': 'Pegasus Knight',
                                   'description': 'This model has special rules for Pegasus Knights.',
                                   'tag': 'special'})
        self.special_rules.append({'name': 'Flying Mount',
                                   'description': 'This model has a flying mount, granting it enhanced mobility and combat advantages.',
                                   'tag': 'mount',
                                   'mountUnit': mountUnit})

        self.AP = 0  # Example Armor Penetration value for Pegasus Knights