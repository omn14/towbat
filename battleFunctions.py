import random
from toHitAndToWound import *


# ── Combat report (debug printout) ─────────────────────────────────────────
# The most recent battle's descriptive detail, captured inside simulate_battle
# just before characteristics are reset.  Consumed once by printBattleResults;
# because each simulate_battle call is immediately followed by its own print,
# a single module-level slot correctly pairs every (sub-)attack with its report.
LAST_COMBAT_REPORT = None

# Special-rule hook -> human label.  Any rule carrying one of these hooks is
# reported as "in effect", so new rules that reuse a hook appear automatically.
_RULE_HOOK_LABELS = {
    'to_hit': 'To Hit mod/reroll',
    'to_wound': 'To Wound mod/reroll',
    'to_save': 'enemy Save mod/reroll',
    'to_modify_stat': 'stat modifier',
    'regen': 'Regeneration',
    'skirmish': 'Skirmishers',
    'fly': 'Fly',
    'Unbreakable': 'Unbreakable',
}


def _si(chars, key, default=0):
    try:
        return int(chars.get(key))
    except (TypeError, ValueError):
        return default


def _active_rule_effects(model, charge):
    """List rules currently in effect on *model* (name + which hooks fire).
    Charge-only bonuses are included only while charging."""
    out = []
    for r in getattr(model, 'special_rules', []) or []:
        if not isinstance(r, dict):
            continue
        name = r.get('name') or 'rule'
        labels = [lbl for hook, lbl in _RULE_HOOK_LABELS.items() if r.get(hook)]
        if r.get('charge') and charge:
            labels.append('charge bonus')
        if labels:
            out.append(f"{name} [{', '.join(labels)}]")
    return out


def _weapon_effects(weapon):
    """Weapon special-rule names (e.g. Armour Bane, Requires Two Hands)."""
    rules = (weapon or {}).get('special_rules') or []
    if isinstance(rules, str):
        rules = [rules]
    return [str(r) for r in rules]


def _ranged_tohit_report(model):
    """Effective BS, To Hit target and the modifiers in effect for a shot."""
    bs = (model.firing_bs(3) if hasattr(model, 'firing_bs')
          else _si(model.characteristics, 'BS', 3))
    ignore = set((getattr(model, 'equipedWeapon', None) or {})
                 .get('ignore_to_hit_penalties', []))
    mods = []
    if getattr(model, 'at_long_range', False) and 'long_range' not in ignore:
        bs -= 1
        mods.append('long range -1')
    if getattr(model, 'firing_multiple', False) and 'multiple_shots' not in ignore:
        bs -= 1
        mods.append('multiple shots -1')
    if getattr(model, 'target_skirmisher', False):
        bs -= 1
        mods.append('skirmisher target -1')
    target = {1: 6, 2: 5, 3: 4, 4: 3, 5: 2}.get(bs, 2 if bs >= 6 else 7)
    return target, mods, bs


def build_combat_report(unit1, unit2, charge, attacks):
    """Capture the descriptive state of a resolved battle for the debug print.
    Reads the models *after* stat bonuses were applied, so values are the ones
    actually in effect.  Never raises fatally (wrapped by the caller)."""
    m1, m2 = unit1.model, unit2.model
    w = m1.equipedWeapon or {}
    ranged = w.get('tag') == 'ranged'

    if ranged:
        strength = w.get('ranged_strength') or (
            m1.shooting_strength() if hasattr(m1, 'shooting_strength')
            else _si(m1.characteristics, 'S', 3))
        ap = w.get('ranged_AP', 0)
        to_hit_target, hit_mods, _bs = _ranged_tohit_report(m1)
    else:
        strength = _si(m1.characteristics, 'S', 3)
        ap = m1.melee_ap() if hasattr(m1, 'melee_ap') else m1.AP
        to_hit_target, hit_mods = to_hit(m1, m2), []

    # To Wound with the strength actually used for this attack.
    saved_s = m1.characteristics.get('S')
    m1.characteristics['S'] = str(strength)
    to_wound_target = to_wound(m1, m2)
    m1.characteristics['S'] = saved_s

    toughness = (m2.get_toughness() if hasattr(m2, 'get_toughness')
                 else _si(m2.characteristics, 'T', 4))
    save = m2.armor_save
    if not ranged and hasattr(m2, 'melee_armour_save'):
        save = m2.melee_armour_save()
    armour = list(getattr(m2, 'armour', []) or [])
    regen = next((r.get('regen') for r in getattr(m2, 'special_rules', [])
                  if isinstance(r, dict) and r.get('regen')), None)

    mods = list(hit_mods)
    if charge:
        mods.append('charging')
    if (not ranged and hasattr(m2, 'melee_weapon_requires_two_hands')
            and m2.melee_weapon_requires_two_hands()
            and any(str(a).strip().lower() == 'shield' for a in armour)):
        mods.append("defender's shield disabled (two-handed)")

    return {
        'attacker': getattr(m1, 'name', '?'),
        'defender': getattr(m2, 'name', '?'),
        'weapon': w.get('name', 'hand weapon'),
        'mode': 'ranged' if ranged else 'melee',
        'strength': strength, 'ap': ap,
        'to_hit': to_hit_target, 'to_wound': to_wound_target,
        'toughness': toughness, 'save': save, 'armour': armour, 'regen': regen,
        'modifiers': mods,
        'attacker_effects': _active_rule_effects(m1, charge) + _weapon_effects(w),
        'defender_effects': _active_rule_effects(m2, charge),
    }


def format_combat_report(r):
    """Render a report dict into printable debug lines."""
    if not r:
        return []
    ap = r['ap']
    ap_str = f"AP-{ap}" if ap else "AP0"
    hit = r['to_hit']
    tw = r['to_wound']
    hit_str = f"{hit}+" if isinstance(hit, int) else str(hit)
    tw_str = f"{tw}+" if isinstance(tw, int) else str(tw)
    mod_str = f"  ({', '.join(r['modifiers'])})" if r['modifiers'] else ""
    save = r['save']
    save_str = f"{save}+" if isinstance(save, int) and save <= 6 else "none"
    armour = ", ".join(r['armour']) if r['armour'] else "-"
    regen = f"  regen {r['regen']}+" if r['regen'] else ""
    lines = [
        f"   Weapon : {r['weapon']} ({r['mode']})  S{r['strength']} {ap_str}  "
        f"[hit {hit_str}, wound {tw_str}]{mod_str}",
        f"   Target : {r['defender']}  T{r['toughness']}  save {save_str}  "
        f"armour [{armour}]{regen}",
    ]
    if r['attacker_effects']:
        lines.append(f"   Attacker rules : {', '.join(r['attacker_effects'])}")
    if r['defender_effects']:
        lines.append(f"   Defender rules : {', '.join(r['defender_effects'])}")
    return lines


def take_last_combat_report():
    """Return and clear the most recent combat report."""
    global LAST_COMBAT_REPORT
    r = LAST_COMBAT_REPORT
    LAST_COMBAT_REPORT = None
    return r


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
        hit = to_hit_ranged(model1,long_range=getattr(model1,'at_long_range',False),multiple_shots=getattr(model1,'firing_multiple',False),target_skirmisher=getattr(model1,'target_skirmisher',False))
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


# ── Impact Hits (Rulebook p. 172) ──────────────────────────────────────────

MIN_IMPACT_HIT_CHARGE = 3.0


def _impact_hit_rule(m):
    for rule in getattr(m, 'special_rules', []) or []:
        if isinstance(rule, dict) and rule.get('impact_hits'):
            return rule['impact_hits']
    return None


def impact_hit_profile(unit):
    """(model, dice expression) for the model in *unit* causing Impact Hits.

    A mount's Impact Hits are made with the mount's Strength, and a chariot's
    with the chariot's own rather than its crew's, so the model that carries
    the rule is also the one that resolves it.
    """
    m = getattr(unit, 'model', None)
    if m is None:
        return None
    expr = _impact_hit_rule(m)
    if expr:
        return m, expr
    mount = m.get_mount() if hasattr(m, 'get_mount') else None
    expr = _impact_hit_rule(mount) if mount is not None else None
    return (mount, expr) if expr else None


def unmodified_strength(m):
    """The model's Strength as printed, ignoring weapon and charge bonuses."""
    base = getattr(m, '_base_characteristics', None) or m.characteristics
    return stat_value(base.get('S'))


def resolve_impact_hits(unit1, unit2):
    """Impact Hits from charging *unit1* against *unit2*.

    Returns (hits, wounds, saves, unsaved). The hits are automatic, so no To
    Hit roll is made; every model in base contact causes them, which is the
    charging unit's front rank.
    """
    found = impact_hit_profile(unit1)
    if not found:
        return 0, 0, 0, 0
    m, expr = found
    from models import roll_dice_expr
    contacting = max(0, min(unit1.files, unit1.nmodels))
    hits = sum(roll_dice_expr(expr) for _ in range(contacting))

    target = to_wound(m, unit2.model, strength=unmodified_strength(m))
    wounds = sum(1 for _ in range(hits) if random.randint(1, 6) >= target)

    ap = m.impact_hit_ap() if hasattr(m, 'impact_hit_ap') else 0
    saves = 0
    for _ in range(wounds):
        if check_armor_save(unit2.model, unit2.model.melee_armour_save(), ap):
            saves += 1
            continue
        for rule in unit2.model.special_rules:
            if rule.get('regen') and check_armor_save(unit2.model, rule['regen'], 0):
                saves += 1
                break
    return hits, wounds, saves, wounds - saves


def impact_hit_report(unit1, unit2):
    """Printable lines describing an Impact Hits attack, or []."""
    found = impact_hit_profile(unit1)
    if not found:
        return []
    m, expr = found
    strength = unmodified_strength(m)
    target = to_wound(m, unit2.model, strength=strength)
    ap = m.impact_hit_ap() if hasattr(m, 'impact_hit_ap') else 0
    save = unit2.model.melee_armour_save()
    save_str = f"{save}+" if isinstance(save, int) and save <= 6 else "none"
    return [f"   Impact Hits ({expr}) : {m.name}  S{strength} "
            f"{f'AP-{ap}' if ap else 'AP0'}  [wound {target}+]",
            f"   Target : {unit2.model.name}  "
            f"T{stat_value(unit2.model.characteristics.get('T'), 4)}  "
            f"save {save_str}"]


def firing_rank_count(files: int, nmodels: int, extra_ranks: int = 0,
                      volley_fire: bool = False) -> int:
    """How many models of a unit may shoot.

    Only the front rank shoots normally; rules that let the rear ranks fire
    stack (Rulebook p. 137). Vantage Point adds a whole rank for a unit on a
    hill, and Volley Fire then adds half of the rank below that.
    """
    if files <= 0 or nmodels <= 0:
        return 0
    firing = min(files * (1 + max(0, extra_ranks)), nmodels)
    if volley_fire:
        next_rank = min(files, nmodels - firing)
        firing += (next_rank + 1) // 2
    return firing


def simulate_battle(unit1, unit2,charge: bool, casualties: int = 0,
                    extra_ranks: int = 0):

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
                attacks = stat_value(unit1.model.characteristics.get('A')) * unit1.files #front rank attacks
        attacks = stat_value(unit1.model.characteristics.get('A')) * unit1.files
        if attacks >= stat_value(unit1.model.characteristics.get('A')) *unit1.nmodels: 
            attacks = stat_value(unit1.model.characteristics.get('A')) *unit1.nmodels # cannot attack more than you have models in front rank
    else: #defends
        A = stat_value(unit1.model.characteristics.get('A'))
        attacks = A * unit1.files  # front rank fights with full Attacks
        if attacks >= A * unit1.nmodels:
            attacks = A * unit1.nmodels  # fewer models than a full front rank
        else:
            # The second rank adds one supporting attack per model (a full rank
            # if the unit is deep enough).  Casualties are taken from the back
            # and thin the supporting rank first, so each one costs an attack.
            second_rank = min(unit1.files, unit1.nmodels - unit1.files)
            second_rank = max(0, second_rank - max(0, casualties))
            attacks += second_rank
    
    if unit1.model.equipedWeapon.get('tag') == 'ranged':
        w = unit1.model.equipedWeapon
        # Multiple Shots: fire multiple by default (-1 To Hit); dice count is
        # rolled separately for each firing model.
        unit1.model.firing_multiple = bool(w.get('ranged_shots_dice')) or (w.get('ranged_shots') or 1) > 1
        firing_models = firing_rank_count(
            unit1.files, unit1.nmodels, extra_ranks,
            any(r.get('volley_fire') for r in unit1.model.special_rules))
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
            if check_armor_save(unit2.model,unit2.model.melee_armour_save(), getattr(unit1.model, 'attack_AP', unit1.model.AP)):
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
    
    # Capture the descriptive report before stats are reset (best-effort).
    global LAST_COMBAT_REPORT
    try:
        LAST_COMBAT_REPORT = build_combat_report(unit1, unit2, charge, attacks1)
    except Exception as e:
        LAST_COMBAT_REPORT = None
        print(f"[combat-report] skipped: {e}")

    unit1.model.reset_characteristics()
    unit2.model.reset_characteristics()
    

    return attacks1,total_hits, suffered_wounds,  saves_made, total_wounds


