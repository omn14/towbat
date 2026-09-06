from rulesFunctions import *
from utilityFunctions import *
from bs4 import BeautifulSoup
import requests

import copy
import os
import random
import re

from battlescribe import (get_catalogue, NAME_ALIASES as _NAME_ALIASES,
                          has_move_and_shoot,
                          has_move_or_shoot as _has_move_or_shoot,
                          has_strike_first as _has_strike_first,
                          has_strike_last as _has_strike_last)
from special_rules import build_special_rules
import troop_types

# Base directory for unit characteristic JSON files, organised by faction
ARMY_UNITS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'army_units')

# Table scale: 1 game unit == 1 inch == 25.4 mm (a 6'x4' table is 72x48 units).
MM_PER_UNIT = 25.4

# The playing surface, in those units. The out-of-bounds walls, the painted
# board edge and the grass all measure from here, so they cannot drift apart.
BOARD_WIDTH = 72
BOARD_DEPTH = 48

# Parry improves an armour value by 1, but never past this (Rulebook p. 190).
PARRY_BEST_SAVE = 3


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


# Armour value each worn piece grants on its own (target number; lower is
# better; 7 means no save).  Names match the catalogue 'Armour' profiles.
ARMOUR_VALUES = {
    'light armour': 6,
    'heavy armour': 5,
    'full plate armour': 4,
    'plate armour': 4,
    'gromril armour': 5,
}
# Pieces that improve (lower) the armour value by 1 rather than setting it.
ARMOUR_MODIFIERS = {'shield', 'barding'}


def armour_save_from_equipment(items, ignore_shield: bool = False) -> int:
    """Armour-save target (2-7; 7 = no save) for a list of armour item names.

    The best worn body armour sets the base value; shields and barding each
    improve it by 1.  E.g. ['Heavy Armour', 'Shield'] -> 4+, ['Shield'] -> 6+.
    With *ignore_shield* the shield grants no bonus (a two-handed weapon in
    melee cannot also use a shield).
    """
    best = 7
    improve = 0
    for it in items or []:
        key = str(it).strip().lower()
        if key in ARMOUR_VALUES:
            best = min(best, ARMOUR_VALUES[key])
        elif key in ARMOUR_MODIFIERS:
            if key == 'shield' and ignore_shield:
                continue
            improve += 1
        else:
            # Some pieces (e.g. chariot hulls) state the value directly, as an
            # 'Armour Value N+' name/description.
            av = re.search(r"armou?r value\s*:?\s*(\d+)", key)
            if av:
                best = min(best, int(av.group(1)))
    if best == 7 and improve == 0:
        return 7
    return max(2, min(7, best - improve))



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


def dice_expr_mean(expr) -> float:
    """Average of the same expressions roll_dice_expr() rolls: 'D3' -> 2.0.

    Multiple Shots has to be chosen before the dice are thrown, so the choice
    is weighed on the average instead. The clamp at zero in roll_dice_expr has
    no counterpart here because every catalogue expression adds a positive
    modifier, if any.
    """
    s = str(expr).upper().replace(" ", "")
    if s.isdigit():
        return float(s)
    m = re.match(r"(\d*)D(\d+)([+-]\d+)?$", s)
    if not m:
        return 1.0
    count = int(m.group(1)) if m.group(1) else 1
    sides = int(m.group(2))
    mod = int(m.group(3)) if m.group(3) else 0
    return count * (sides + 1) / 2 + mod


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
        self.armour = []  # equipped armour item names (from the roster)
        self.special_rules = []
        self.special_rules = build_special_rules(self)
        self.weapons = {}
        # Every model carries a hand weapon unless told otherwise (Rulebook
        # p. 190). Taken from the catalogue so the roster's own copy merges
        # into it later instead of arriving as a second, differently-cased
        # weapon. The stub is only for running without the catalogue.
        if not self.give_weapon('Hand Weapon'):
            self.weapons['hand weapon'] = {'name': 'hand weapon',
                                           'description': 'basic melee weapon',
                                           'tag': 'combat'}
        self.equipedWeapon = None
        self.equip_weapon('hand weapon')
        # War machines carry their piece as a ranged weapon (e.g. Great Cannon),
        # named the same as the model in the catalogue.
        if str(self.characteristics.get('Troop Type', '')).lower() == 'war machine':
            self.give_weapon(self.name)
        # A chariot is one model made of several profiles; pull in the parts the
        # catalogue lists so combat can use the right one (Rulebook p. 194).
        for part in (self.characteristics.get('Crew') or []):
            self.attach_crew(model(part['name'], ""), part.get('count', 1))
        for part in (self.characteristics.get('Beasts') or []):
            self.attach_beasts(model(part['name'], ""), part.get('count', 1))
        self.attack_roll = 0
        self.wound_roll = 0
        self.spells = {}       # name -> spell dict; only Wizards have any

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
    def weapon_slot(self, name):
        """The key in ``weapons`` matching *name*, ignoring case, or None.

        The catalogue names weapons in title case ("Hand Weapon") while the
        engine asks for them in lower case, and a dict keyed on the raw string
        kept both as separate weapons.
        """
        target = str(name).strip().lower()
        for key in self.weapons:
            if str(key).strip().lower() == target:
                return key
        return None

    def equip_weapon(self, weapon_name: str):
        try:
            slot = self.weapon_slot(weapon_name)
            self.special_rules = [rule for rule in self.special_rules if rule != self.equipedWeapon]
            self.equipedWeapon = self.weapons.get(slot) if slot else None
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

    def is_chariot(self) -> bool:
        """True if this model is a chariot, which has a split profile."""
        return 'chariot' in str(self.characteristics.get('Troop Type', '')).lower()

    def troop_type(self) -> str:
        return str(self.characteristics.get('Troop Type', '') or '')

    def troop_type_rule(self, rule_name: str) -> bool:
        """True if this model's troop type grants *rule_name*. These rules are
        implied by the troop type and never appear in the catalogue data."""
        return troop_types.has_rule(self.troop_type(), rule_name)

    def max_rank_bonus(self, default: int) -> int:
        return troop_types.max_rank_bonus(self.troop_type(), default)

    def models_per_rank(self, default: int) -> int:
        return troop_types.models_per_rank(self.troop_type(), default)

    def starting_wounds(self, default: int = 1) -> int:
        """Wounds as printed, before anything on the table wounded it."""
        base = self._base_characteristics or self.characteristics
        return stat_int(base, 'W', default)

    def has_all_round_vision(self) -> bool:
        """360° vision arc for shooting and casting: Firing Platform (p. 194)
        for a chariot, and Skirmishers, who have no formed facing."""
        return self.is_skirmisher() or self.troop_type_rule('Firing Platform')

    def impact_hit_ap(self) -> int:
        """Armour Piercing of this model's Impact Hits: Scythed Wheels (p. 195)
        gives a heavy chariot -2."""
        return 2 if self.troop_type_rule('Scythed Wheels') else 0

    def _tagged_part(self, tag):
        for rule in self.special_rules:
            if isinstance(rule, dict) and rule.get('tag') == tag and rule.get('partUnit'):
                part = rule['partUnit']
                return getattr(part, 'model', part)
        return None

    def part_count(self, tag: str) -> int:
        """How many models of a chariot part there are (6 crew, 2 horses...)."""
        for rule in self.special_rules:
            if isinstance(rule, dict) and rule.get('tag') == tag and rule.get('partUnit'):
                return rule.get('count', 1)
        return 0

    def _attach_part(self, tag, name, description, part, count=1):
        self.special_rules = [r for r in self.special_rules
                              if not (isinstance(r, dict) and r.get('tag') == tag)]
        self.special_rules.append({'name': name, 'description': description,
                                   'tag': tag, 'partUnit': part, 'count': count})

    def get_crew(self):
        """The chariot's crew model, or None."""
        return self._tagged_part('crew')

    def attach_crew(self, crew_model, count=1):
        self._attach_part('crew', 'Chariot Crew',
                          "Enemy rolls To Hit are made against the crew's Weapon Skill.",
                          crew_model, count)

    def get_beasts(self):
        """The beasts drawing the chariot, or None."""
        return self._tagged_part('beasts')

    def attach_beasts(self, beast_model, count=1):
        self._attach_part('beasts', 'Chariot Beasts',
                          'The chariot moves at the speed of the beasts drawing it.',
                          beast_model, count)

    def defending_ws(self, default: int = 0) -> int:
        """The Weapon Skill enemies roll against: a chariot is hit on its crew's
        (Rulebook p. 194)."""
        crew = self.get_crew()
        source = crew if crew is not None else self
        return stat_int(source.characteristics, 'WS', default)

    def firing_bs(self, default: int = 0) -> int:
        """The Ballistic Skill this model shoots with. A chariot's own profile
        has none; its crew shoot with theirs (Rulebook p. 194)."""
        crew = self.get_crew()
        source = crew if crew is not None else self
        return stat_int(source.characteristics, 'BS', default)

    def shooting_strength(self, default: int = 3) -> int:
        """Strength for a weapon that has none of its own, e.g. a bow. The crew
        shoot with their own Strength, not the chariot's."""
        crew = self.get_crew()
        source = crew if crew is not None else self
        return stat_int(source.characteristics, 'S', default)

    def is_wizard(self) -> bool:
        return any(isinstance(r, dict) and r.get('wizard')
                   for r in self.special_rules)

    def wizard_level(self, default: int = 0) -> int:
        """Level of Wizardry, which sets how many spells may be cast a turn and
        adds half itself to a Casting roll (Rulebook p. 106-108)."""
        for rule in self.special_rules:
            if isinstance(rule, dict) and rule.get('wizard'):
                return int(rule.get('wizard_level') or 1)
        return default

    def is_skirmisher(self) -> bool:
        """True if the model has the Skirmishers special rule."""
        return any(isinstance(r, dict) and r.get('skirmish')
                   for r in self.special_rules)

    def has_fire_and_flee(self) -> bool:
        """True if the model has the Fire & Flee special rule (p. 169)."""
        return any(isinstance(r, dict) and r.get('fire_and_flee')
                   for r in self.special_rules)

    def _strike_rule(self, key, from_weapon) -> bool:
        """Whether a Strike First / Strike Last rule is in force.

        Strike Last is almost always the weapon's doing rather than the
        model's — a great weapon carries it — so the melee weapon actually in
        hand counts as well as the profile.
        """
        if any(isinstance(r, dict) and r.get(key) for r in self.special_rules):
            return True
        w = self.active_melee_weapon()
        return bool(w.get(key) or from_weapon(w.get('special_rules')))

    def has_strike_first(self) -> bool:
        """True if the model strikes at Initiative 10 (p. 177)."""
        return self._strike_rule('strike_first', _has_strike_first)

    def has_strike_last(self) -> bool:
        """True if the model strikes at Initiative 1 (p. 178)."""
        return self._strike_rule('strike_last', _has_strike_last)

    def is_venerable(self) -> bool:
        """True if the model has the Venerable special rule."""
        return any(isinstance(r, dict) and r.get('venerable')
                   for r in self.special_rules)

    def is_stubborn(self) -> bool:
        """True if the model has the Stubborn special rule."""
        return any(isinstance(r, dict) and r.get('stubborn')
                   for r in self.special_rules)

    def is_general(self) -> bool:
        """True if the army list nominates this model as the General."""
        return any(isinstance(r, dict) and r.get('general')
                   for r in self.special_rules)

    def is_battle_standard(self) -> bool:
        """True if this model carries the army's Battle Standard."""
        return any(isinstance(r, dict) and r.get('battle_standard')
                   for r in self.special_rules)

    def is_flying(self) -> bool:
        """True if the model has the Fly special rule."""
        return any(isinstance(r, dict) and r.get('fly')
                   for r in self.special_rules)

    def is_swiftstride(self) -> bool:
        """True if this model has Swiftstride, or rides a mount that does."""
        if any(isinstance(r, dict) and r.get('swiftstride')
               for r in self.special_rules):
            return True
        mount = self.get_mount()
        return bool(mount is not None and mount.is_swiftstride())

    def get_fly_movement(self, default: int = 0) -> int:
        """Fly Movement characteristic (the X in 'Fly (X)'), else *default*."""
        for r in self.special_rules:
            if isinstance(r, dict) and r.get('fly') and r.get('fly_movement'):
                return int(r['fly_movement'])
        return default

    def set_armour(self, items):
        """Record equipped armour item names and recompute the armour save."""
        self.armour = list(items or [])
        self.armor_save = armour_save_from_equipment(self.armour)
        return self.armor_save

    def melee_weapon_requires_two_hands(self) -> bool:
        """True if the model's active melee weapon has 'Requires Two Hands'.
        Combat equips the best melee weapon first, so the equipped weapon is it."""
        w = self.equipedWeapon
        if not w or w.get('tag') == 'ranged':
            return False
        return any('requires two hands' in str(r).lower()
                   for r in (w.get('special_rules') or []))

    def has_shield(self) -> bool:
        return any(str(a).strip().lower() == 'shield' for a in (self.armour or []))

    def uses_hand_weapon(self) -> bool:
        """True if the active melee weapon is a plain hand weapon."""
        w = self.equipedWeapon
        return bool(w) and str(w.get('name', '')).strip().lower() == 'hand weapon'

    def parry_applies(self) -> bool:
        """Parry (Rulebook p. 190): infantry fighting with a hand weapon and a
        shield deflect blows with it."""
        return (self.troop_type_rule('Parry') and self.has_shield()
                and self.uses_hand_weapon())

    def fights_in_extra_rank(self) -> bool:
        """True if the equipped melee weapon allows a supporting attack.

        The rule lives on the weapon as 'Fight in Extra Rank' (Rulebook p. 169)
        — a spear or polearm has it, bare hands do not.

        This reads equipedWeapon rather than active_melee_weapon(): a cavalry
        spear is charge-only for Strength and AP, but its extra rank works the
        other way round, being denied on the turn the wielder charged.
        """
        w = self.equipedWeapon or {}
        if w.get('tag') == 'ranged':
            return False
        return any('fight in extra rank' in str(r).lower()
                   for r in (w.get('special_rules') or []))

    def melee_armour_save(self) -> int:
        """Armour save used in melee; a two-handed weapon disables the shield,
        and Parry improves the value for a hand weapon and shield.
        Based on the stored save so hard-coded units keep their value."""
        save = self.armor_save
        if self.has_shield() and self.melee_weapon_requires_two_hands():
            return min(7, save + 1)  # lose the shield's improvement
        if self.parry_applies():
            # Improves by 1 but no further than 3+, and a better save stands.
            return min(save, max(PARRY_BEST_SAVE, save - 1))
        return save

    def unit_strength(self) -> int:
        """Unit Strength per model. The troop type decides it where the rulebook
        value is known (a heavy chariot is US5, a monster is worth its starting
        Wounds); otherwise mounted models count as US2 and everything else US1."""
        return troop_types.unit_strength(self.troop_type(),
                                         2 if self.is_mounted() else 1,
                                         self.starting_wounds())

    def get_movement(self, default: int = 0) -> int:
        """Movement value; mounted units always use their mount's Movement."""
        mount = self.get_mount()
        if mount is not None:
            return stat_int(mount.characteristics, 'M', default)
        # A chariot moves at the speed of the beasts that draw it, if any.
        beasts = self.get_beasts()
        if beasts is not None:
            return stat_int(beasts.characteristics, 'M', default)
        # A split profile is two rows with gaps in each (p. 97). A war machine
        # has no Movement of its own: it is shifted by the crew that work it.
        if stat_int(self.characteristics, 'M', 0) <= 0:
            crew = self.get_crew()
            if crew is not None:
                return stat_int(crew.characteristics, 'M', default)
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
                w['ranged_strength'] = self.shooting_strength()
            # Replace any entry of the same name rather than sitting beside it.
            slot = self.weapon_slot(w['name'])
            if slot is not None and slot != w['name']:
                del self.weapons[slot]
            self.weapons[w['name']] = w
        return w

    def roll_ranged_shots(self, multiple: bool = True) -> int:
        """Shots for one firing model from the equipped ranged weapon.

        Multiple Shots is a choice, not a property of the weapon (Rulebook
        p. 174): *multiple* False fires the single accurate shot instead. A
        random count is rolled here, so a caller rolling once per model gets a
        separate roll for each, as the rule requires.
        """
        w = self.equipedWeapon or {}
        if not multiple:
            return 1
        dice = w.get('ranged_shots_dice')
        return roll_dice_expr(dice) if dice else (w.get('ranged_shots') or 1)

    def has_multiple_shots(self) -> bool:
        """True if the equipped ranged weapon offers the choice at all."""
        w = self.equipedWeapon or {}
        return bool(w.get('ranged_shots_dice')) or (w.get('ranged_shots') or 1) > 1

    def expected_ranged_shots(self, multiple: bool = True) -> float:
        """Average shots one firing model gets, for weighing the choice."""
        w = self.equipedWeapon or {}
        if not multiple:
            return 1.0
        dice = w.get('ranged_shots_dice')
        return dice_expr_mean(dice) if dice else float(w.get('ranged_shots') or 1)

    def fires_after_marching(self) -> bool:
        """Move & Shoot: the equipped weapon may fire even after a march (p. 174)."""
        w = self.equipedWeapon or {}
        if 'move_and_shoot' in w:
            return bool(w['move_and_shoot'])
        # A save written before the flag existed still carries the rule names.
        return has_move_and_shoot(w.get('special_rules'))

    def cannot_shoot_after_moving(self, weapon=None) -> bool:
        """Move or Shoot: artillery is impossible to fire on the move (p. 174).

        The weapon is passed in by the war-machine paths, which pick their own
        rather than firing whatever happens to be equipped.
        """
        w = self.equipedWeapon if weapon is None else weapon
        w = w or {}
        if 'move_or_shoot' in w:
            return bool(w['move_or_shoot'])
        # A save written before the flag existed still carries the rule names.
        return _has_move_or_shoot(w.get('special_rules'))

    def active_melee_weapon(self) -> dict:
        """The melee weapon this model is actually fighting with.

        A charge-only weapon (Lance) is not in use outside a charge, and a
        ranged weapon is never used in melee; both fall back to bare hands.
        """
        w = self.equipedWeapon or {}
        if w.get('tag') == 'ranged':
            return {}
        if w.get('charge_only') and not self.charging:
            return {}
        return w

    def missile_weapon(self) -> dict:
        """A missile weapon the model carries, equipped or not.

        Stand & Shoot asks whether the unit *is armed* with one (p. 120), which
        a unit that has equipped its sword for the coming fight still is.
        """
        w = self.equipedWeapon or {}
        if w.get('tag') == 'ranged':
            return w
        return next((x for x in self.weapons.values()
                     if x.get('tag') == 'ranged'), {})

    def melee_strength_bonus(self) -> int:
        """Strength bonus of the equipped melee weapon. Charge-only weapons
        (Lance) count only while charging; others (Halberd, Great Weapon) are
        always on."""
        return self.active_melee_weapon().get('strength_bonus', 0)

    def apply_melee_strength(self):
        """Add the active melee weapon's Strength bonus once per combat.
        Reset by reset_characteristics() afterwards."""
        bonus = self.melee_strength_bonus()
        if bonus:
            self.characteristics['S'] = str(stat_int(self.characteristics, 'S', 3) + bonus)
        return bonus

    def melee_ap(self) -> int:
        """AP penetration of the equipped melee weapon; charge value while charging."""
        w = self.active_melee_weapon()
        if self.charging and w.get('ap_penetration_charge') is not None:
            return w['ap_penetration_charge']
        return w.get('ap_penetration', 0)

    def armour_bane_for_attack(self) -> int:
        """Armour Bane (X) of the weapon used for the current attack."""
        if self.equipedWeapon and self.equipedWeapon.get('tag') == 'ranged':
            return armour_bane_x(self.equipedWeapon.get('special_rules', []))
        return armour_bane_x(self.active_melee_weapon().get('special_rules', []))
        return best

    def equip_best_melee(self) -> str:
        """Equip the strongest applicable melee weapon so the equipped weapon
        matches what's actually used in combat (and its hooks fire).  Charge-only
        weapons (e.g. Lance) count only while charging."""
        best_name, best_score = 'hand weapon', -1
        for name, w in self.weapons.items():
            if w.get('tag') == 'ranged':
                continue
            if w.get('charge_only') and not self.charging:
                continue
            ap = w.get('ap_penetration', 0)
            if self.charging and w.get('ap_penetration_charge') is not None:
                ap = w['ap_penetration_charge']
            score = (w.get('strength_bonus', 0) + ap
                     + armour_bane_x(w.get('special_rules', [])))
            if score > best_score:
                best_name, best_score = name, score
        self.equip_weapon(best_name)
        return best_name