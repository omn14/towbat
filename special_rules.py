"""Data-driven special rules.

The BattleScribe cat/gst only carry rule *keywords* on each unit (e.g. a unit
lists "Regeneration", never "Regeneration (4+)"). Core rulebook effects and their
numeric values are NOT in the data, so this module is the code-side rulebook: it
maps a catalogue keyword to the game's special-rule dict (the same shape the
engine already consumes via `model.special_rules`).

`build_special_rules(model)` reads `model.characteristics['Special Rules']` and
returns the list of rule dicts. Army-specific abilities with no coded mechanic
become plain flag dicts (name + catalogue description) so they still display.

This module is intentionally standalone (no import of models.py) to avoid an
import cycle; the base model imports `build_special_rules` from here.
"""

from __future__ import annotations

import re

from battlescribe import get_catalogue
from rulesFunctions import plus1attacks


def parse_special_rule(text: str):
    """Split a catalogue rule string into (display_name, param).

    'Regeneration'      -> ('Regeneration', None)
    'Armour Bane (1)'   -> ('Armour Bane', '1')
    'Multiple Shots (D3)' -> ('Multiple Shots', 'D3')
    """
    s = str(text).strip()
    m = re.match(r"^(?P<name>.*?)\s*(?:\((?P<param>[^)]*)\))?\s*$", s)
    if not m:
        return s, None
    name = (m.group("name") or "").strip()
    param = m.group("param")
    return (name or s), (param.strip() if param else None)


def _param_save(param, default=None):
    """Parse a save value like '4+' or '4' from a rule param, else *default*."""
    if param:
        m = re.search(r"(\d+)", str(param))
        if m:
            return int(m.group(1))
    return default


def _param_dice(param, default=None):
    """Leading dice expression of a rule param: '2', 'D6', 'D6+1', '2D6+2'.

    A param often carries prose the engine cannot model, e.g. Impact Hits
    '(D6+1, War Wagon only)'; only the expression is kept.
    """
    m = re.match(r"\s*(\d*[Dd]\d+(?:\s*[+-]\s*\d+)?|\d+)", str(param or ""))
    return m.group(1).replace(" ", "").upper() if m else default


def _flag(name: str, desc: str | None, tag: str = "special") -> dict:
    """A display-only special rule with no coded mechanic (yet)."""
    return {"name": name, "description": desc or "", "tag": tag}


# ── Builders for keywords that have a coded engine effect ──────────────────
# Each builder takes (model, param, description) and returns a special_rules
# dict, or None to skip. Keys are matched case-insensitively on the rule name.

def _furious_charge(model, param, desc):
    return {"name": "Furious Charge",
            "description": desc or "+1 Attack on the charge.",
            "tag": "combat",
            "charge": plus1attacks}


def _regeneration(model, param, desc):
    # The save value is taken from the catalogue name modifier, e.g.
    # 'Regeneration (5+)'. With no value in the data there is no regen save.
    entry = {"name": "Regeneration",
             "description": desc or "Regeneration save.",
             "tag": "special"}
    save = _param_save(param)
    if save is not None:
        entry["regen"] = save
    return entry


def _unbreakable(model, param, desc):
    return {"name": "Unbreakable",
            "description": desc or "Never flees; only gives ground when it loses combat.",
            "tag": "psychology",
            "Unbreakable": True}


def _skirmishers(model, param, desc):
    return {"name": "Skirmishers",
            "description": desc or "May adopt a Skirmish formation.",
            "tag": "formation",
            "skirmish": True}


def _fire_and_flee(model, param, desc):
    return {"name": "Fire & Flee",
            "description": desc or ("May Stand & Shoot and then flee as a "
                                    "charge reaction."),
            "tag": "reaction",
            "fire_and_flee": True}


def _strike_first(model, param, desc):
    return {"name": "Strike First",
            "description": desc or ("Improves its Initiative to 10 before any "
                                    "other modifiers are applied."),
            "tag": "combat",
            "strike_first": True}


def _strike_last(model, param, desc):
    return {"name": "Strike Last",
            "description": desc or ("Reduces its Initiative to 1 before any "
                                    "other modifiers are applied."),
            "tag": "combat",
            "strike_last": True}


def _killing_blow(model, param, desc):
    return {"name": "Killing Blow",
            "description": desc or ("A natural 6 To Wound in combat allows no "
                                    "armour or Regeneration save, and an "
                                    "infantry or cavalry model that suffers an "
                                    "unsaved wound from it loses all of its "
                                    "remaining Wounds."),
            "tag": "combat",
            "killing_blow": True}


def _fly(model, param, desc):
    # 'Fly (9)' -> flies with Movement 9, passing freely over models/terrain.
    entry = {"name": "Fly",
             "description": desc or "May move by flying over models and terrain.",
             "tag": "movement",
             "fly": True}
    m = re.search(r"(\d+)", str(param or ""))
    if m:
        entry["fly_movement"] = int(m.group(1))
    return entry


def _venerable(model, param, desc):
    return {"name": "Venerable",
            "description": desc or ("Friendly units within 6\" of this model may "
                                    "re-roll failed Panic tests."),
            "tag": "psychology",
            "venerable": True}


def _stubborn(model, param, desc):
    return {"name": "Stubborn",
            "description": desc or ("The first Break test this unit is required to "
                                    "make may be refused, Falling Back in Good "
                                    "Order instead."),
            "tag": "psychology",
            "stubborn": True}


def _general(model, param, desc):
    # Marks the army commander; Inspiring Presence is applied from the unit.
    return {"name": "General",
            "description": desc or ("Friendly units within this model's Command "
                                    "range may use its Leadership."),
            "tag": "psychology",
            "general": True}


def _battle_standard(model, param, desc):
    return {"name": "Battle Standard Bearer",
            "description": desc or ("Friendly units within this model's Command "
                                    "range may re-roll Panic, Rally and Break "
                                    "tests; +1 combat result."),
            "tag": "psychology",
            "battle_standard": True}


def _swiftstride(model, param, desc):
    return {"name": "Swiftstride",
            "description": desc or ("+3\" maximum charge range; may add a D6 to "
                                    "Charge, Flee and Pursuit rolls."),
            "tag": "movement",
            "swiftstride": True}


def _impact_hits(model, param, desc):
    return {"name": "Impact Hits",
            "description": desc or ("A charging model that moved 3\" or more "
                                    "causes automatic hits at its unmodified "
                                    "Strength before any attacks are made."),
            "tag": "combat",
            "impact_hits": _param_dice(param, "1")}


# Normalised (lowercase) keyword -> builder.
SPECIAL_RULE_BUILDERS = {
    "furious charge": _furious_charge,
    "regeneration": _regeneration,
    "unbreakable": _unbreakable,
    "skirmishers": _skirmishers,
    "fire & flee": _fire_and_flee,
    "fire and flee": _fire_and_flee,
    "strike first": _strike_first,
    "strikes first": _strike_first,
    "strike last": _strike_last,
    "strikes last": _strike_last,
    "killing blow": _killing_blow,
    "fly": _fly,
    "venerable": _venerable,
    "stubborn": _stubborn,
    "general": _general,
    "battle standard bearer": _battle_standard,
    "swiftstride": _swiftstride,
    "impact hits": _impact_hits,
}


# ── Swiftstride (Rulebook p. 178) ──────────────────────────────────────────

# Swiftstride adds this much to the maximum possible charge range. Note it is
# NOT the 6 the bonus die could roll: the rulebook fixes the declaration range
# at +3" whatever the die later does.
SWIFTSTRIDE_CHARGE_BONUS = 3
# Flee further than this and the board edge stops being a worry.
SAFE_FLEE_MARGIN = 12.0


def unit_has_swiftstride(unit) -> bool:
    """True if *unit* consists entirely of Swiftstride models.

    The engine keeps one profile per unit, so the rank and file are covered by
    the unit's own model (or its mount). A joined character with different
    profile is the one way to get a mixed unit, and it costs the unit the rule.
    """
    model = getattr(getattr(unit, 'unit', None), 'model', None)
    check = getattr(model, 'is_swiftstride', None)
    if not callable(check) or not check():
        return False
    joined = getattr(unit, 'joinedCharacter', None)
    if joined is None:
        return True
    return unit_has_swiftstride(joined)


def max_charge_range(movement: int, swiftstride: bool = False) -> int:
    """Maximum possible charge range: Movement plus the best Charge roll (6),
    plus 3\" for Swiftstride (Rulebook p. 121)."""
    return movement + 6 + (SWIFTSTRIDE_CHARGE_BONUS if swiftstride else 0)


def charge_roll(dice, difficult: bool = False) -> int:
    """Result of a Charge roll: 2D6 discarding the lowest.

    Charging through difficult terrain discards the highest instead, so the
    lowest of the two is the result (Rulebook p. 269). Any further dice are
    Swiftstride's bonus and are *added* -- the bonus die is never one of the
    two the roll discards between.
    """
    if not dice:
        return 0
    pair = min(dice[:2]) if difficult else max(dice[:2])
    return pair + sum(dice[2:])


def max_pursuit_range(swiftstride: bool = False) -> int:
    """Furthest a Pursuit roll can carry a unit: 2D6 summed, with no Movement
    added, plus Swiftstride's bonus die. The +3\" is a charge-declaration rule
    and does not apply here.
    """
    return 12 + (6 if swiftstride else 0)


def should_use_swiftstride(kind: str, distance_to_edge=None) -> bool:
    """AI policy for Swiftstride's optional bonus die.

    The choice is made before the roll, so it is judged on consequences, not
    results. Charging further is free -- the unit stops at its target -- but a
    fleeing unit must move the full distance even off the battlefield, where it
    is destroyed, so decline when the board edge is close. A pursuit is treated
    the same way, cautiously: a long one can carry the unit off the table.
    """
    if kind in ('flee', 'fall back', 'pursuit') and distance_to_edge is not None:
        return distance_to_edge >= SAFE_FLEE_MARGIN
    return True


def should_fire_multiple(single_chance: float, multi_chance: float,
                         expected_shots: float) -> bool:
    """AI policy for the Multiple Shots choice (Rulebook p. 174).

    Volume against accuracy, weighed as expected hits per firing model: one
    shot at the unmodified To Hit, or *expected_shots* of them at -1. Nothing
    else about the shot changes, so the comparison is the whole decision.

    A tie goes to the single shot. The extra dice stop paying once the -1 has
    pushed the roll out of reach, and an unmodified roll keeps its value under
    any further penalty the target's cover or range might add.
    """
    return expected_shots * multi_chance > single_chance


def can_stand_and_shoot(distance: float, charger_movement: int,
                        quick_shot: bool = False) -> bool:
    """Whether the charged unit has time to raise its weapons (p. 120).

    A charger closing from inside its own Movement characteristic is on the
    unit before it can shoot; Quick Shot ignores the distance entirely
    (p. 175). Exactly the Movement is far enough — the rule bars a distance
    *less than* it.
    """
    if quick_shot:
        return True
    return distance >= charger_movement


def board_edge_distance(x: float, y: float, half_x: float = 36.0,
                        half_y: float = 24.0) -> float:
    """Distance from (x, y) to the nearest table edge, in inches.

    Feeds the flee decision above; the board is 72x48 centred on the origin.
    """
    return min(half_x - abs(x), half_y - abs(y))


def build_special_rules(model) -> list:
    """Return the special-rule dicts for a model from its catalogue keywords.

    Rules with a coded builder get their engine hooks; everything else becomes a
    display-only flag carrying the catalogue description when one exists.
    """
    cat = get_catalogue()
    rules: list = []
    for raw in model.characteristics.get("Special Rules", []) or []:
        display, param = parse_special_rule(raw)
        desc = cat.rule_description(display) or cat.rule_description(raw)
        builder = SPECIAL_RULE_BUILDERS.get(display.lower())
        if builder is None and "(" in display:
            # A rule that replaces another keeps both values, e.g. 'Impact Hits
            # (2) (D3+1)'; the trailing one is the replacement that applies.
            display = display.split("(")[0].strip()
            builder = SPECIAL_RULE_BUILDERS.get(display.lower())
        entry = builder(model, param, desc) if builder else _flag(display, desc)
        if entry:
            rules.append(entry)
    return rules


def _built_names(model, names) -> set:
    """The rule names *names* would build for this model."""
    keep = model.characteristics.get("Special Rules")
    model.characteristics["Special Rules"] = list(names)
    try:
        return {e.get("name") for e in build_special_rules(model)
                if isinstance(e, dict)}
    finally:
        model.characteristics["Special Rules"] = keep


def apply_rule_keywords(model, names, replace=False) -> None:
    """Give *model* the rule keywords an army list or a save names.

    With `replace`, *names* is the whole list and anything it no longer
    mentions is taken away again. A save carries the roster's complete list,
    and merging instead lets a rule outlive the save that granted it — Strike
    First stayed on a unit after loading a save that never had it.
    """
    current = model.characteristics.get("Special Rules")
    current = list(current) if isinstance(current, list) else []
    if replace:
        wanted = [n for n in (names or []) if n]
        stale = _built_names(model, current) - _built_names(model, wanted)
        model.characteristics["Special Rules"] = wanted
        if stale:
            model.special_rules = [
                r for r in model.special_rules
                if not (isinstance(r, dict) and r.get("name") in stale)]
    else:
        if not names:
            return
        for name in names:
            if name and name not in current:
                current.append(name)
        model.characteristics["Special Rules"] = current
    have = {r.get("name") for r in model.special_rules if isinstance(r, dict)}
    for entry in build_special_rules(model):
        if isinstance(entry, dict) and entry.get("name") not in have:
            model.special_rules.append(entry)
            have.add(entry.get("name"))
