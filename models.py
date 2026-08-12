from rulesFunctions import *
from utilityFunctions import *
from bs4 import BeautifulSoup
import requests

import copy
import os
import random
import re

from battlescribe import get_catalogue, NAME_ALIASES as _NAME_ALIASES

# Base directory for unit characteristic JSON files, organised by faction
ARMY_UNITS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'army_units')

# Table scale: 1 game unit == 1 inch == 25.4 mm (a 6'x4' table is 72x48 units).
MM_PER_UNIT = 25.4


def _find_json_file(filename: str) -> str:
    """Search army_units/ subfolders for a characteristics JSON file.
    Returns the full path if found, otherwise returns the filename as-is
    (for legacy/fallback behaviour)."""
    if os.path.isfile(filename):
        return filename
    for root, _dirs, files in os.walk(ARMY_UNITS_DIR):
        if filename in files:
            return os.path.join(root, filename)
    return filename  # fallback: will trigger fetch from URL


def stat_int(characteristics: dict, key: str, default: int = 0) -> int:
    """Read a numeric stat, returning *default* for missing/non-numeric (e.g. '-')."""
    try:
        return int(characteristics.get(key))
    except (KeyError, TypeError, ValueError):
        return default


def armour_bane_x(rules) -> int:
    """Return the X from an 'Armour Bane (X)' entry in a rules list, else 0."""
    for r in rules or []:
        m = re.search(r"armou?r bane\s*\(?\s*(\d+)", str(r), re.I)
        if m:
            return int(m.group(1))
    return 0


def roll_dice_expr(expr) -> int:
    """Roll a shots/dice expression: '2', 'D3', 'D6', '2D6', 'D3+1'. Returns int >= 0."""
    s = str(expr).upper().replace(" ", "")
    if s.isdigit():
        return int(s)
    m = re.match(r"(\d*)D(\d+)([+-]\d+)?$", s)
    if not m:
        return 1
    count = int(m.group(1)) if m.group(1) else 1
    sides = int(m.group(2))
    mod = int(m.group(3)) if m.group(3) else 0
    return max(0, sum(random.randint(1, sides) for _ in range(count)) + mod)


class model:
    def __init__(self, name: str, url: str):
        self.name = name
        self.url = url
        self.characteristics = {}
        self.json_file_path = None
        # Source of truth for reset_characteristics(): a pristine copy of the stats.
        self._base_characteristics = None

        # 1. Prefer the BattleScribe catalogue (all factions, official data).
        catalogue_chars = get_catalogue().characteristics(self.name)
        if catalogue_chars:
            self.characteristics = catalogue_chars
            self._base_characteristics = copy.deepcopy(catalogue_chars)

        json_filename = self.name.replace(" ", "_").replace("-", "_").lower() + '_characteristics.json'
        # Check for a name alias (e.g. "Orc Boyz" -> "orc_boy")
        stem = json_filename.replace('_characteristics.json', '')
        if stem in _NAME_ALIASES:
            json_filename = _NAME_ALIASES[stem] + '_characteristics.json'
        if self.characteristics:
            pass  # already sourced from the catalogue
        elif os.path.isfile(_find_json_file(json_filename)):
            self.json_file_path = _find_json_file(json_filename)
            self.characteristics = load_dict_from_file(self.json_file_path)
        elif url:
            self.model_data = self.fetch_model_data(url)
            self.characteristics = self.get_characteristics_from_html(self.model_data)
            # Save into army_units/ root if no faction folder matched
            os.makedirs(ARMY_UNITS_DIR, exist_ok=True)
            self.json_file_path = os.path.join(ARMY_UNITS_DIR, json_filename)
            store_dict_to_file(self.characteristics, self.json_file_path)
        else:
            print(f"Warning: No characteristics file found for '{self.name}' and no URL provided.")
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
        if self._base_characteristics is not None:
            self.characteristics = copy.deepcopy(self._base_characteristics)
        elif self.json_file_path and os.path.isfile(self.json_file_path):
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
            self.special_rules = [rule for rule in self.special_rules if rule != self.equipedWeapon]
            self.equipedWeapon = self.weapons.get(weapon_name)
            # Remove existing weapon rule if it has the same name
            #self.special_rules = [rule for rule in self.special_rules if rule.get('name') != weapon_name]
            self.special_rules = [rule for rule in self.special_rules if rule != self.equipedWeapon]
            self.special_rules.append(self.equipedWeapon)
        except Exception as e:
            print(f"Error equipping weapon '{weapon_name}' for {self.name}: {e}")

    def get_mount(self):
        """Return the mount's model if this unit is mounted, else None."""
        for rule in self.special_rules:
            if isinstance(rule, dict) and rule.get('tag') == 'mount' and rule.get('mountUnit'):
                mount = rule['mountUnit']
                return getattr(mount, 'model', mount)
        return None

    def attach_mount(self, mount_unit):
        """Attach (or replace) a mount so this model counts as mounted."""
        self.special_rules = [r for r in self.special_rules
                              if not (isinstance(r, dict) and r.get('tag') == 'mount')]
        self.special_rules.append({'name': 'Mounted',
                                   'description': 'This model is mounted.',
                                   'tag': 'mount',
                                   'mountUnit': mount_unit})

    def is_mounted(self) -> bool:
        return self.get_mount() is not None

    def get_movement(self, default: int = 0) -> int:
        """Movement value; mounted units always use their mount's Movement."""
        mount = self.get_mount()
        if mount is not None:
            return stat_int(mount.characteristics, 'M', default)
        return stat_int(self.characteristics, 'M', default)

    def get_toughness(self, default: int = 4) -> int:
        """Toughness value; mounted units always use the rider's own Toughness."""
        return stat_int(self.characteristics, 'T', default)

    def get_base_size(self):
        """Base size (width_mm, depth_mm) from the catalogue; mounted models use
        their mount's base. Returns None when the database has no base data."""
        mount = self.get_mount()
        name = mount.name if mount is not None else self.name
        size = get_catalogue().base_size(name)
        if size:
            return size
        # Fall back to any base fields on the resolved model's characteristics.
        chars = getattr(mount, 'characteristics', None) or self.characteristics
        w, d = chars.get('base_width_mm'), chars.get('base_depth_mm')
        if w and d:
            return (float(w), float(d))
        return None

    def give_weapon(self, name: str):
        """Add a weapon by name, resolving its stats from the catalogue."""
        w = get_catalogue().weapon(name)
        if w:
            if w.get('tag') == 'ranged' and not w.get('ranged_strength'):
                w['ranged_strength'] = stat_int(self.characteristics, 'S', 3)
            self.weapons[w['name']] = w
        return w

    def roll_ranged_shots(self) -> int:
        """Shots for one firing model from the equipped ranged weapon.
        Rolls the dice (e.g. Multiple Shots (D3)) when the count is random."""
        w = self.equipedWeapon or {}
        dice = w.get('ranged_shots_dice')
        return roll_dice_expr(dice) if dice else (w.get('ranged_shots') or 1)

    def melee_strength_bonus(self) -> int:
        """Strength bonus of the active melee weapon. Charge-only weapons (Lance)
        count only while charging; others (Halberd, Great Weapon) are always on."""
        best = 0
        for w in self.weapons.values():
            if w.get('tag') == 'ranged':
                continue
            if w.get('charge_only') and not self.charging:
                continue
            best = max(best, w.get('strength_bonus', 0))
        return best

    def apply_melee_strength(self):
        """Add the active melee weapon's Strength bonus once per combat.
        Reset by reset_characteristics() afterwards."""
        bonus = self.melee_strength_bonus()
        if bonus:
            self.characteristics['S'] = str(stat_int(self.characteristics, 'S', 3) + bonus)
        return bonus

    def melee_ap(self) -> int:
        """AP penetration of the active melee weapon; charge value while charging."""
        best = 0
        for w in self.weapons.values():
            if w.get('tag') == 'ranged':
                continue
            if w.get('charge_only') and not self.charging:
                continue
            ap = w.get('ap_penetration', 0)
            if self.charging and w.get('ap_penetration_charge') is not None:
                ap = w['ap_penetration_charge']
            best = max(best, ap)
        return best

    def armour_bane_for_attack(self) -> int:
        """Armour Bane (X) of the weapon used for the current attack.
        Ranged uses the equipped weapon; melee uses the best applicable weapon
        (charge-only weapons count only while charging)."""
        if self.equipedWeapon and self.equipedWeapon.get('tag') == 'ranged':
            return armour_bane_x(self.equipedWeapon.get('special_rules', []))
        best = 0
        for w in self.weapons.values():
            if w.get('tag') == 'ranged':
                continue
            if w.get('charge_only') and not self.charging:
                continue
            best = max(best, armour_bane_x(w.get('special_rules', [])))
        return best

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
        """ 
        self.special_rules.append({'name': 'Morks curse',
                                   'description': 'This model must reroll saves of 6',
                                   'tag': 'saving throw',
                                   'to_save': lambda roll: reroll1d6(roll,[6],False)})
         """
        self.AP = 0  # Example Armor Penetration value for Saurus Warriors

        self.weapons.update({
            'spear': {'name': 'spear'},
            'halberd': {'name': 'halberd',
                        'description': '+1 Strength in combat.',
                        'tag': 'combat',
                        'to_modify_stat': lambda model_instance: setattr(model_instance, 'AP', model_instance.AP + 1)}
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
        self.armor_save = 3  # Example improved armor save for Pegasus Knights

class GiantWolf(model):
    def __init__(self, name: str, url: str):
        super().__init__(name, url)
        # Additional Giant Wolf specific attributes can be added here
        self.special_rules.append({'name': 'Giant Wolf',
                                   'description': 'This model has special rules for Giant Wolves.',
                                   'tag': 'special'})
        self.AP = 0  # Example Armor Penetration value for Giant Wolves

class GoblinWolfRider(model):
    def __init__(self, name: str, url: str, mountUnit: model = None):
        super().__init__(name, url)
        # Additional Goblin Wolf Rider specific attributes can be added here
        self.special_rules.append({'name': 'Goblin Wolf Rider',
                                   'description': 'This model has special rules for Goblin Wolf Riders.',
                                   'tag': 'special'})
        self.special_rules.append({'name': 'Mounted on Giant Wolf',
                                   'description': 'This model is mounted on a Giant Wolf, granting it additional movement and combat abilities.',
                                   'tag': 'mount',
                                   'mountUnit': mountUnit})

        self.AP = 0  # Example Armor Penetration value for Goblin Wolf Riders
        self.armor_save = 6  # Example improved armor save for Goblin Wolf Riders

        self.weapons.update({
            'cavalry spear': {'name': 'cavalry spear',
                      'description': 'This model adds +1 to its Armor Penetration (AP) and Strength (S) when it charges.',
                      'tag': 'combat',
                      'charge': lambda model_instance: setattr(model_instance, 'AP', (model_instance.AP + 1)*1),
                      'charge': lambda model_instance: plusSTAT(model_instance, 'S', 1, -99) },
            'sword': {'name': 'sword'}
        })

class SkeletalSteed(model):
    def __init__(self, name: str, url: str):
        super().__init__(name, url)
        # Additional Skeletal Steed specific attributes can be added here
        
        self.special_rules.append({'name': 'Skeletal Steed',
                                   'description': 'This model has special rules for Skeletal Steeds.',
                                   'tag': 'special',
                                   'move': 0.5
                                   })
        self.AP = 0  # Example Armor Penetration value for Skeletal Steeds

class BlackKnight(model):
    def __init__(self, name: str, url: str, mountUnit: model = None):
        super().__init__(name, url)
        # Additional Black Knight specific attributes can be added here
        self.special_rules.append({'name': 'Black Knight',
                                   'description': 'This model has special rules for Black Knights.',
                                   'tag': 'special'})
        self.special_rules.append({'name': 'Mounted on Skeletal Steed',
                                   'description': 'This model is mounted on a Skeletal Steed, granting it additional movement and combat abilities.',
                                   'tag': 'mount',
                                   'mountUnit': mountUnit})
        self.special_rules.append({'name': 'regeneration',
                                   'description': 'This model has the regeneration special rule, allowing it to recover wounds.',
                                   'tag': 'special',
                                   'regen': 6,
                                   'move': 0.5
                                   })
        self.special_rules.append({'name': 'Fearless',
                                   'description': 'This model has the Fearless special rule.',
                                   'tag': 'special',
                                   'Unbreakable': True
                                   })

        self.AP = 0  # Example Armor Penetration value for Black Knights
        self.armor_save = 4  # Example improved armor save for Black Knights

        self.weapons.update({
            'lance': {'name': 'lance',
                      'description': 'This model adds +2 to its Armor Penetration (AP) when it charges.',
                      'tag': 'combat',
                      'charge': lambda model_instance: setattr(model_instance, 'AP', (model_instance.AP + 2)*1),
                      'charge': lambda model_instance: plusSTAT(model_instance, 'S', 2, -99) },
            'sword': {'name': 'sword'}
        })

class Zombie(model):
    def __init__(self, name: str, url: str):
        super().__init__(name, url)
        # Additional Zombie specific attributes can be added here
        
        self.special_rules.append({'name': 'Zombie',
                                   'description': 'This model has special rules for Zombies.',
                                   'tag': 'special',
                                   'move': 0.5
                                   })
        self.special_rules.append({'name': 'Fearless',
                                   'description': 'This model has the Fearless special rule.',
                                   'tag': 'special',
                                   'Unbreakable': True
                                   })
        self.AP = 0  # Example Armor Penetration value for Zombies



class DireWolf(model):
    def __init__(self, name: str, url: str):
        super().__init__(name, url)
        # Additional Dire Wolf specific attributes can be added here
        
        self.special_rules.append({'name': 'Dire Wolf',
                                   'description': 'This model has special rules for Dire Wolves.',
                                   'tag': 'special',
                                   #'move': lambda model_instance: plusSTAT(model_instance, 'M', int(int(model_instance.characteristics['M'])/-2), .5)
                                   'move': 0.5
                                   })
        self.special_rules.append({'name': 'Fearless',
                                   'description': 'This model has the Fearless special rule.',
                                   'tag': 'special',
                                   'Unbreakable': True
                                   })
        self.AP = 0  # Example Armor Penetration value for Dire Wolves


class JadeWarrior(model):
    def __init__(self, name: str, url: str):
        super().__init__(name, url)
        # Additional Jade Warrior specific attributes can be added here
        
        self.special_rules.append({'name': 'Jade Warrior',
                                   'description': 'This model has special rules for Jade Warriors.',
                                   'tag': 'special'})
        self.AP = 0  # Example Armor Penetration value for Jade Warriors
        self.armor_save = 3  # Example improved armor save for Jade Warriors

        self.weapons.update({
            'halberd': {'name': 'halberd',
                        'description': 'This model adds +1 to its Armor Penetration (AP) when it charges.',
                        'tag': 'combat',
                        'charge': lambda model_instance: setattr(model_instance, 'AP', (model_instance.AP + 1)*1),
                        'to_modify_stat': lambda model_instance: plusSTAT(model_instance, 'S', 1, -99),
                        'to_modify_stat': lambda model_instance: setattr(model_instance, 'AP', (model_instance.AP + 1)*1) }
        })

class CathayanWarhorse(model):
    def __init__(self, name: str, url: str):
        super().__init__(name, url)
        # Additional Cathayan Warhorse specific attributes can be added here
        
        self.special_rules.append({'name': 'Cathayan Warhorse',
                                   'description': 'This model has special rules for Cathayan Warhorses.',
                                   'tag': 'special'})
        self.AP = 0  # Example Armor Penetration value for Cathayan Warhorses

class JadeLancer(model):
    def __init__(self, name: str, url: str, mountUnit: model = None):
        super().__init__(name, url)
        # Additional Jade Lancer specific attributes can be added here
        self.special_rules.append({'name': 'Jade Lancer',
                                   'description': 'This model has special rules for Jade Lancers.',
                                   'tag': 'special'})
        self.special_rules.append({'name': 'Mounted on Cathayan Warhorse',
                                   'description': 'This model is mounted on a Cathayan Warhorse, granting it additional movement and combat abilities.',
                                   'tag': 'mount',
                                   'mountUnit': mountUnit})

        self.AP = 0  # Example Armor Penetration value for Jade Lancers
        self.armor_save = 3  # Example improved armor save for Jade Lancers

        self.weapons.update({
            'lance': {'name': 'cathayan lance',
                      'description': 'This model adds +2 to its Armor Penetration (AP) when it charges.',
                      'tag': 'combat',
                      'charge': lambda model_instance: setattr(model_instance, 'AP', (model_instance.AP + 1)*1),
                      'charge': lambda model_instance: plusSTAT(model_instance, 'S', 1, -99) },
            'sword': {'name': 'sword'}
        })

class Necromancer(model):
    def __init__(self, name: str, url: str, spells: dict = None):
        super().__init__(name, url)
        # Additional Necromancer specific attributes can be added here
        
        self.special_rules.append({'name': 'Necromancer',
                                   'description': 'This model has special rules for Necromancers.',
                                   'tag': 'wizard',
                                   'wizard_level': 2,
                                   'wizard': True
                                   })
        
        self.spells =spells

        self.AP = 0  # Example Armor Penetration value for Necromancers


# ═══════════════════════════════════════════════════════════════════════
#  Orc & Goblin Tribes — additional units
# ═══════════════════════════════════════════════════════════════════════

class WarBoar(model):
    def __init__(self, name: str, url: str):
        super().__init__(name, url)
        self.special_rules.append({'name': 'War Boar',
                                   'description': 'Tusked charge.',
                                   'tag': 'special'})
        self.AP = 0

class OrcBoarBoy(model):
    """Hammer (Fast) — heavy cavalry of the Waaagh."""
    def __init__(self, name: str, url: str, mountUnit: model = None):
        super().__init__(name, url)
        self.special_rules.append({'name': 'Orc Boar Boy',
                                   'description': 'Orc mounted on a war boar.',
                                   'tag': 'special'})
        self.special_rules.append({'name': 'Mounted on War Boar',
                                   'description': 'Grants additional movement and charge impact.',
                                   'tag': 'mount',
                                   'mountUnit': mountUnit})
        self.special_rules.append({'name': 'Furious charge',
                                   'description': '+1 Strength on the charge.',
                                   'tag': 'combat',
                                   'charge': plus1attacks})
        self.AP = 0
        self.armor_save = 4

        self.weapons.update({
            'cavalry spear': {'name': 'cavalry spear',
                              'description': '+1 S when charging.',
                              'tag': 'combat',
                              'charge': lambda mi: plusSTAT(mi, 'S', 1, -99)},
        })

class Troll(model):
    """Anvil — regeneration and 3 wounds make them nearly impossible to shift."""
    def __init__(self, name: str, url: str):
        super().__init__(name, url)
        self.special_rules.append({'name': 'Troll',
                                   'description': 'Large, regenerating creature.',
                                   'tag': 'special'})
        self.special_rules.append({'name': 'regeneration',
                                   'description': 'Regeneration (4+).',
                                   'tag': 'special',
                                   'regen': 4})
        self.special_rules.append({'name': 'Stupidity',
                                   'description': 'Must test for Stupidity each turn.',
                                   'tag': 'psychology'})
        self.AP = 0


# ═══════════════════════════════════════════════════════════════════════
#  Vampire Counts — additional units
# ═══════════════════════════════════════════════════════════════════════

class SkeletonWarrior(model):
    """Cannon Fodder — weak stats but Unbreakable keeps them on the field."""
    def __init__(self, name: str, url: str):
        super().__init__(name, url)
        self.special_rules.append({'name': 'Fearless',
                                   'description': 'Unbreakable undead.',
                                   'tag': 'special',
                                   'Unbreakable': True})
        self.AP = 0
        self.armor_save = 6

class CryptGhoul(model):
    """Basic — decent stats but no armour, fights on par."""
    def __init__(self, name: str, url: str):
        super().__init__(name, url)
        self.special_rules.append({'name': 'Fearless',
                                   'description': 'Unbreakable undead.',
                                   'tag': 'special',
                                   'Unbreakable': True})
        self.special_rules.append({'name': 'Poison',
                                   'description': 'Attacks are Poisoned (to-hit roll of 6 auto-wounds).',
                                   'tag': 'combat'})
        self.AP = 0

class GraveGuard(model):
    """Superior — elite undead infantry with heavy armour and Killing Blow."""
    def __init__(self, name: str, url: str):
        super().__init__(name, url)
        self.special_rules.append({'name': 'Fearless',
                                   'description': 'Unbreakable undead.',
                                   'tag': 'special',
                                   'Unbreakable': True})
        self.special_rules.append({'name': 'Killing Blow',
                                   'description': 'To-wound roll of 6 causes instant death.',
                                   'tag': 'combat'})
        self.AP = 0
        self.armor_save = 4

        self.weapons.update({
            'great weapon': {'name': 'great weapon',
                             'description': '+2 Strength, strikes last.',
                             'tag': 'combat',
                             'to_modify_stat': lambda mi: plusSTAT(mi, 'S', 2, -99)},
        })


# ═══════════════════════════════════════════════════════════════════════
#  Bretonnia — additional units
# ═══════════════════════════════════════════════════════════════════════

class PeasantBowman(model):
    """Cannon Fodder (Shooting) — cheap ranged peasants."""
    def __init__(self, name: str, url: str):
        super().__init__(name, url)
        self.special_rules.append({'name': 'Peasant Bowman',
                                   'description': 'Poorly trained peasant levy.',
                                   'tag': 'special'})
        self.AP = 0

        self.weapons.update({
            'longbow': {'name': 'longbow',
                        'description': 'Standard ranged weapon.',
                        'tag': 'ranged',
                        'ranged_range': 24,
                        'ranged_shots': 1,
                        'ranged_strength': 3,
                        'ranged_AP': 0,
                        'volley_fire': True},
        })

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

class GrailKnight(model):
    """Hammer (Fast) — the elite of Bretonnia, blessed by the Lady."""
    def __init__(self, name: str, url: str, mountUnit: model = None):
        super().__init__(name, url)
        self.special_rules.append({'name': 'Grail Knight',
                                   'description': 'Blessed warriors of the Grail.',
                                   'tag': 'special'})
        self.special_rules.append({'name': 'Mounted on Bretonnian Warhorse',
                                   'description': 'Mounted cavalry.',
                                   'tag': 'mount',
                                   'mountUnit': mountUnit})
        self.special_rules.append({'name': 'Blessing of the Lady',
                                   'description': 'Ward save (5+).',
                                   'tag': 'special'})
        self.AP = 0
        self.armor_save = 2

        self.weapons.update({
            'lance': {'name': 'lance',
                      'description': '+2 S when charging.',
                      'tag': 'combat',
                      'charge': lambda mi: plusSTAT(mi, 'S', 2, -99)},
            'sword': {'name': 'sword'},
        })

class BattlePilgrim(model):
    """Anvil — fanatical peasants with Stubborn and decent Leadership."""
    def __init__(self, name: str, url: str):
        super().__init__(name, url)
        self.special_rules.append({'name': 'Stubborn',
                                   'description': 'Always uses unmodified Leadership for Break tests.',
                                   'tag': 'psychology'})
        self.special_rules.append({'name': 'Grail Reliquae',
                                   'description': 'Inspired by holy relics — Stubborn and immune to Fear.',
                                   'tag': 'special'})
        self.AP = 0


# ═══════════════════════════════════════════════════════════════════════
#  Grand Cathay — additional units
# ═══════════════════════════════════════════════════════════════════════

class PeasantSpearman(model):
    """Cannon Fodder — cheap Cathayan levy with spears."""
    def __init__(self, name: str, url: str):
        super().__init__(name, url)
        self.special_rules.append({'name': 'Peasant Spearman',
                                   'description': 'Low-quality infantry levy.',
                                   'tag': 'special'})
        self.AP = 0

        self.weapons.update({
            'spear': {'name': 'spear',
                      'description': 'Fight in extra rank.',
                      'tag': 'combat'},
        })

class IronHailGunner(model):
    """Basic (Shooting) — handgun-armed Cathayan infantry."""
    def __init__(self, name: str, url: str):
        super().__init__(name, url)
        self.special_rules.append({'name': 'Iron Hail Gunner',
                                   'description': 'Cathayan firearms regiment.',
                                   'tag': 'special'})
        self.AP = 0
        self.armor_save = 5

        self.weapons.update({
            'handgun': {'name': 'iron hail handgun',
                        'description': 'Armour-piercing firearm.',
                        'tag': 'ranged',
                        'ranged_range': 24,
                        'ranged_shots': 1,
                        'ranged_strength': 4,
                        'ranged_AP': 1,
                        'volley_fire': False},
        })


# ═══════════════════════════════════════════════════════════════════════
#  Lizardmen — additional units
# ═══════════════════════════════════════════════════════════════════════

class Skink(model):
    """Cannon Fodder (Shooting) — nimble skirmishers with blowpipes."""
    def __init__(self, name: str, url: str):
        super().__init__(name, url)
        self.special_rules.append({'name': 'Skink',
                                   'description': 'Fast, expendable skirmisher.',
                                   'tag': 'special'})
        self.AP = 0

        self.weapons.update({
            'blowpipe': {'name': 'blowpipe',
                         'description': 'Short-ranged poisoned missile weapon.',
                         'tag': 'ranged',
                         'ranged_range': 12,
                         'ranged_shots': 2,
                         'ranged_strength': 3,
                         'ranged_AP': 0,
                         'volley_fire': False},
        })

class TempleGuard(model):
    """Anvil — Stubborn elite Saurus guarding the temple-cities."""
    def __init__(self, name: str, url: str):
        super().__init__(name, url)
        self.special_rules.append({'name': 'Stubborn',
                                   'description': 'Always uses unmodified Leadership for Break tests.',
                                   'tag': 'psychology'})
        self.special_rules.append({'name': 'Temple Guard',
                                   'description': 'Elite guardians of the Slann.',
                                   'tag': 'special'})
        self.AP = 0
        self.armor_save = 4

        self.weapons.update({
            'halberd': {'name': 'halberd',
                        'description': '+1 Strength in combat.',
                        'tag': 'combat',
                        'to_modify_stat': lambda mi: plusSTAT(mi, 'S', 1, -99)},
        })

class ColdOne(model):
    def __init__(self, name: str, url: str):
        super().__init__(name, url)
        self.special_rules.append({'name': 'Cold One',
                                   'description': 'Cold-blooded mount.',
                                   'tag': 'special'})
        self.AP = 0

class ColdOneRider(model):
    """Hammer (Fast) — heavy Saurus cavalry on savage Cold One mounts."""
    def __init__(self, name: str, url: str, mountUnit: model = None):
        super().__init__(name, url)
        self.special_rules.append({'name': 'Cold One Rider',
                                   'description': 'Saurus mounted on a Cold One.',
                                   'tag': 'special'})
        self.special_rules.append({'name': 'Mounted on Cold One',
                                   'description': 'Grants additional movement and attacks.',
                                   'tag': 'mount',
                                   'mountUnit': mountUnit})
        self.AP = 0
        self.armor_save = 3

        self.weapons.update({
            'cavalry spear': {'name': 'cavalry spear',
                              'description': '+1 S when charging.',
                              'tag': 'combat',
                              'charge': lambda mi: plusSTAT(mi, 'S', 1, -99)},
        })