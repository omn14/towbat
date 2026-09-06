import random
from toHitAndToWound import *
from rules_log import rule_log, rule_skipped


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
    """The To Hit arithmetic for a shot: base, each modifier, final target.

    Read from `ranged_hit_requirement`, not a second copy of the ladder, so
    the report cannot print a number the dice will not use. The size of each
    modifier is measured the same way rather than assumed, which is what makes
    a Ponderous -2 read as -2.
    """
    applied = dict(
        moved=getattr(model, 'moved_this_turn', False),
        long_range=getattr(model, 'at_long_range', False),
        multiple_shots=getattr(model, 'firing_multiple', False),
        target_skirmisher=getattr(model, 'target_skirmisher', False),
    )
    req = ranged_hit_requirement(model, **applied)
    if req is None:
        return {'target': 7, 'base': 7, 'mods': [], 'reroll': None, 'bs': 0}
    base = ranged_hit_requirement(model)
    w = getattr(model, 'equipedWeapon', None) or {}
    labels = {'moved': 'moved', 'long_range': 'long range',
              'multiple_shots': 'multiple shots',
              'target_skirmisher': 'skirmisher target'}
    mods = []
    for key, on in applied.items():
        if not on:
            continue
        alone = ranged_hit_requirement(model, **{key: True})
        delta = alone[0] - base[0]
        why = ''
        if key == 'moved':
            if w.get('ponderous') and w.get('quick_shot'):
                why = ' (Ponderous and Quick Shot cancel out)'
            elif w.get('ponderous'):
                why = ' (Ponderous)'
            elif not delta:
                why = ' (Quick Shot)'
        mods.append(f"{labels[key]} -{delta}{why}" if delta
                    else f"{labels[key]} waived{why}")
    bs = (model.firing_bs(3) if hasattr(model, 'firing_bs')
          else _si(model.characteristics, 'BS', 3))
    return {'target': req[0], 'base': base[0], 'mods': mods,
            'reroll': req[1], 'bs': bs}


def _tohit_summary(rep):
    """One line showing how the To Hit target was arrived at."""
    parts = f"BS{rep['bs']} {rep['base']}+"
    if rep['mods']:
        parts += "  " + "  ".join(rep['mods'])
    line = f"{parts}  =  {rep['target']}+"
    if rep['target'] > 6:
        line += (f" (natural 6, then {rep['target'] - 3}+)"
                 if rep['target'] < TO_HIT_IMPOSSIBLE else " (cannot hit)")
    elif rep['reroll'] is not None:
        line += f", re-rolling a failure at {rep['reroll']}+"
    return line


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
        _rep = _ranged_tohit_report(m1)
        to_hit_target, hit_mods = _rep['target'], []
        tohit_summary = _tohit_summary(_rep)
    else:
        strength = _si(m1.characteristics, 'S', 3)
        ap = m1.melee_ap() if hasattr(m1, 'melee_ap') else m1.AP
        to_hit_target, hit_mods = to_hit(m1, m2), []
        tohit_summary = None

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
        'to_hit_summary': tohit_summary,
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
    ]
    if r.get('to_hit_summary'):
        lines.append(f"   To Hit : {r['to_hit_summary']}")
    lines.append(
        f"   Target : {r['defender']}  T{r['toughness']}  save {save_str}  "
        f"armour [{armour}]{regen}")
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
        def shoot():
            return to_hit_ranged(model1,long_range=getattr(model1,'at_long_range',False),multiple_shots=getattr(model1,'firing_multiple',False),target_skirmisher=getattr(model1,'target_skirmisher',False),moved=getattr(model1,'moved_this_turn',False))
        hit = shoot()
        # Curse of Arrow Attraction: a natural 1 To Hit may be re-rolled.
        if not hit and model1.attack_roll == 1 and getattr(model2, 'arrow_attraction', False):
            model1.attack_roll = random.randint(1, 6)
            hit = shoot()
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


def ward_save_value(model) -> int:
    """The model's Warding value, or 0 for none.

    Only one Ward save may ever be attempted and two never combine, so a model
    carrying more than one simply uses the best (Rulebook p. 141).
    """
    best = 0
    for rule in getattr(model, 'special_rules', []) or []:
        if isinstance(rule, dict) and rule.get('ward'):
            best = rule['ward'] if not best else min(best, rule['ward'])
    return best


def check_saves(model, armor_save_value, AP):
    """The whole save sequence against one wound: Armour, then Ward, then
    Regeneration (Rulebook p. 141, p. 176). True if the wound is saved.

    Rules that modify armour values leave Warding and Regeneration values
    alone, so neither of those is touched by the attack's AP.
    """
    if check_armor_save(model, armor_save_value, AP):
        return True
    ward = ward_save_value(model)
    if ward and random.randint(1, 6) >= ward:
        return True
    for rule in getattr(model, 'special_rules', []) or []:
        if rule.get('regen') and check_armor_save(model, rule['regen'], 0):
            return True
    return False


def resolve_magic_hits(unit, hits: int, strength: int, ap: int):
    """*hits* automatic hits of the given Strength and AP against *unit*.

    Returns (wounds, saves, unsaved). A spell has no attacking model, so there
    is no To Hit roll and the hits wound on the spell's own Strength.
    """
    if hits <= 0:
        return 0, 0, 0
    m = unit.model
    # to_wound reads its first model only for a Strength, which is given here.
    target = to_wound(m, m, strength=strength)
    wounds = sum(1 for _ in range(hits) if random.randint(1, 6) >= target)
    saves = sum(1 for _ in range(wounds)
                if check_saves(m, m.melee_armour_save(), ap))
    return wounds, saves, wounds - saves


# ── Who Strikes First (Rulebook p. 146) ────────────────────────────────────

MAX_INITIATIVE = 10
MIN_INITIATIVE = 1
MAX_CHARGE_INITIATIVE_FRONT = 3
MAX_CHARGE_INITIATIVE_FLANK = 4


def charge_initiative_bonus(inches: float, flank_or_rear: bool = False) -> int:
    """+1 Initiative per *full* inch a charge moved before contact (p. 146)."""
    cap = (MAX_CHARGE_INITIATIVE_FLANK if flank_or_rear
           else MAX_CHARGE_INITIATIVE_FRONT)
    return max(0, min(int(inches), cap))


def base_initiative(model) -> int:
    """A model's Initiative before any modifier.

    Strike First sets it to 10 and Strike Last to 1 (p. 177 and p. 178), both
    "before any other modifiers are applied", and a model with both is left on
    its own characteristic because the two cancel out.
    """
    profile = _si(model.characteristics, 'I', 1)
    first = model.has_strike_first() if hasattr(model, 'has_strike_first') else False
    last = model.has_strike_last() if hasattr(model, 'has_strike_last') else False
    if first and not last:
        return MAX_INITIATIVE
    if last and not first:
        return MIN_INITIATIVE
    return profile


def strike_initiative(model, charged: bool = False, inches: float = 0.0,
                      flank_or_rear: bool = False) -> int:
    """The Initiative a model strikes at this round.

    The charge bonus is capped at +3 into a front arc and +4 into a flank or
    rear, and the total may not exceed 10 (p. 146, as amended by the errata).
    """
    base = base_initiative(model)
    if not charged:
        return base
    return min(MAX_INITIATIVE,
               base + charge_initiative_bonus(inches, flank_or_rear))


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
    saves = sum(1 for _ in range(wounds)
                if check_saves(unit2.model, unit2.model.melee_armour_save(), ap))
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


def melee_attacks(unit, charge: bool, casualties: int = 0) -> int:
    """Attacks a unit makes in one round of combat.

    A model in base contact attacks with its full Attacks characteristic; a
    model that can fight but is not in base contact attacks once, whatever its
    profile says (Rulebook p. 146).

    Which ranks may fight is two separate rules that stack rather than overlap.
    Press of Battle makes the fighting rank *two* ranks deep (p. 190), and
    Fight in Extra Rank lets the rank directly behind the fighting rank make
    supporting attacks (p. 169) — which a model already in a fighting rank may
    not do (p. 145). Infantry with thrusting spears therefore fight three ranks
    deep: the spear rank is pushed back to the third, not absorbed into the
    second. Both are denied on the turn the model charged.

    `casualties` are the models lost earlier in this phase, already gone from
    `unit.nmodels`. They cost the unit a *second* attacker only where a model
    behind the fighting rank stepped into the gap, since neither the slain nor
    the model that replaced it may attack (Stepping Forward, p. 102 and p. 150).
    A unit no deeper than its own fighting rank has nobody to step forward, so
    its fighting rank simply narrows.
    """
    m = unit.model
    A = stat_value(m.characteristics.get('A'))
    files = max(0, unit.files)
    spare = max(0, unit.nmodels)
    fallen = max(0, casualties)
    press = m.troop_type_rule('Press of Battle') and not charge

    # Only models that were behind the fighting rank can have stepped into it.
    behind = max(0, spare + fallen - files * (2 if press else 1))
    stepped = min(fallen, behind)

    def survivors(rank: int) -> int:
        """Models of a rank still able to attack once those that stepped
        forward into it are taken off."""
        nonlocal stepped
        able = max(0, rank - stepped)
        stepped -= rank - able
        return able

    front = min(files, spare)
    spare -= front
    fighting = survivors(front)
    attacks = A * fighting
    if fighting < front:
        rule_log('Stepping Forward', unit,
                 f"{front - fighting} of {front} models in the fighting rank "
                 f"stepped up over the {fallen} fallen and cannot attack: "
                 f"{fighting} attack instead of {front}")

    if charge:
        if m.troop_type_rule('Press of Battle'):
            rule_skipped('Press of Battle', unit, "it charged this turn")
        return attacks

    weapon = (m.equipedWeapon or {}).get('name', 'bare hands')

    if press:
        rank = min(spare, files)
        spare -= rank
        able = survivors(rank)
        attacks += able
        if able:
            rule_log('Press of Battle', unit,
                     f"fighting rank is two deep: {able} model(s) in the second "
                     f"rank attack once each")

    if m.fights_in_extra_rank():
        rank = min(spare, files)
        spare -= rank
        able = survivors(rank)
        attacks += able
        rule_log('Fight in Extra Rank', unit,
                 f"{weapon}: rank {2 + press} supports the fighting rank with "
                 f"{able} attack(s)")
    elif spare:
        why = f"{weapon} grants no supporting attack"
        if not press:
            why += " and the troop type has no Press of Battle"
        rule_skipped('Fight in Extra Rank', unit,
                     f"{spare} model(s) behind the fighting rank idle: {why}")
    return attacks


def simulate_battle(unit1, unit2,charge: bool, casualties: int = 0,
                    extra_ranks: int = 0, multiple_shots: bool = True):

    # how many attacks
    unit1.nmodels = max(0, unit1.nmodels)  # Ensure at least one model
    # Apply the active melee weapon's always-on/charge Strength bonus once.
    unit1.model.charging = bool(charge)
    unit1.model.apply_melee_strength()
    if charge:
        for rule in unit1.model.special_rules:
            if rule.get('charge'):
                rule['charge'](unit1.model)
    attacks = melee_attacks(unit1, charge, casualties)
    
    if unit1.model.equipedWeapon.get('tag') == 'ranged':
        w = unit1.model.equipedWeapon
        # Multiple Shots is the firer's choice (p. 174), taken for the whole
        # unit and passed in; volume costs -1 To Hit on every shot.
        fire_multiple = bool(multiple_shots) and unit1.model.has_multiple_shots()
        unit1.model.firing_multiple = fire_multiple
        firing_models = firing_rank_count(
            unit1.files, unit1.nmodels, extra_ranks,
            any(r.get('volley_fire') for r in unit1.model.special_rules))
        # Rolled per model, not once for the unit: "where the number of
        # Multiple Shots is generated by a dice roll, roll separately for
        # each model".
        attacks = sum(unit1.model.roll_ranged_shots(fire_multiple)
                      for _ in range(firing_models))

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
    # Reported once for the exchange, not once per save roll.
    if unit2.model.parry_applies():
        rule_log('Parry', unit2, f"hand weapon and shield: armour "
                                 f"{unit2.model.armor_save}+ -> "
                                 f"{unit2.model.melee_armour_save()}+ in melee")
    elif unit2.model.troop_type_rule('Parry') and unit2.model.has_shield():
        rule_skipped('Parry', unit2,
                     f"fighting with "
                     f"{(unit2.model.equipedWeapon or {}).get('name', 'nothing')}, "
                     f"not a hand weapon")
    for i in range(attacks1):
        hit,wound = simulate_attack(unit1.model, unit2.model)
        if hit:
            total_hits += 1
        if wound:
            total_wounds += 1
            suffered_wounds += 1
        if wound:
            if check_saves(unit2.model, unit2.model.melee_armour_save(),
                           getattr(unit1.model, 'attack_AP', unit1.model.AP)):
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


