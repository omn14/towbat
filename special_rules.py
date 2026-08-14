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


# Normalised (lowercase) keyword -> builder.
SPECIAL_RULE_BUILDERS = {
    "furious charge": _furious_charge,
    "regeneration": _regeneration,
    "unbreakable": _unbreakable,
}


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
        entry = builder(model, param, desc) if builder else _flag(display, desc)
        if entry:
            rules.append(entry)
    return rules
