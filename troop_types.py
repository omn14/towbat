"""Troop type properties (Rulebook, Troop Types in Detail).

Most rules reach a model as a keyword in its `Special Rules` list, which
`special_rules.py` turns into an engine hook. A troop type's rules are not
written down anywhere in the catalogue, though: nothing in the .cat/.gst
mentions Scythed Wheels, Lumbering, Iron Shod Wheels or Firing Platform, and a
model is expected to have them purely because its Troop Type says
"Heavy Chariot". This module is that missing table.

Only the entries whose values have been checked against the rulebook are
listed; every other troop type falls back to the engine's existing behaviour.
"""

from __future__ import annotations

# Rulebook p. 194-195. `models_per_rank` and `max_rank_bonus` of None mean the
# type cannot form ranks at all, printed as '-' in the rulebook's tables.
TROOP_TYPES = {
    "heavy chariot": {
        "models_per_rank": None,
        "max_rank_bonus": None,
        "unit_strength": 5,
        "rules": ("Split Profile (Chariots)", "Scythed Wheels", "Lumbering",
                  "Iron Shod Wheels", "Firing Platform"),
    },
    "light chariot": {
        "models_per_rank": 3,
        "max_rank_bonus": 1,
        "unit_strength": 3,
        "rules": ("Split Profile (Chariots)", "Iron Shod Wheels",
                  "Churning Wheels", "Firing Platform"),
    },
}


def normalise(troop_type) -> str:
    """The bare troop type, e.g. 'Heavy chariot (named character)' -> 'heavy chariot'.

    The data is inconsistently cased, tags characters in parentheses, and gives
    a two-profile unit both types separated by a comma.
    """
    name = str(troop_type or "").split("(")[0].split(",")[0]
    return " ".join(name.lower().split())


def properties(troop_type) -> dict | None:
    return TROOP_TYPES.get(normalise(troop_type))


def unit_strength(troop_type, default: int) -> int:
    props = properties(troop_type)
    return props["unit_strength"] if props else default


def max_rank_bonus(troop_type, default: int) -> int:
    """The most Rank Bonus this troop type can claim; 0 if it cannot form ranks."""
    props = properties(troop_type)
    if props is None:
        return default
    return props["max_rank_bonus"] or 0


def has_rule(troop_type, rule_name: str) -> bool:
    props = properties(troop_type)
    if props is None:
        return False
    rule_name = rule_name.lower()
    return any(rule_name == r.lower().split("(")[0].strip()
               or rule_name == r.lower() for r in props["rules"])
