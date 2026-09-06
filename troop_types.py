"""Troop type properties (Rulebook, Troop Types in Detail).

Most rules reach a model as a keyword in its `Special Rules` list, which
`special_rules.py` turns into an engine hook. A troop type's rules are not
written down anywhere in the catalogue, though: nothing in the .cat/.gst
mentions Parry, Scythed Wheels or Lumbering, and a model is expected to have
them purely because its Troop Type says "Regular Infantry" or "Heavy Chariot".
This module is that missing table.

The `rules` entries are the rulebook's names. Only some of them are coded yet;
`has_rule` answers what a troop type *grants*, not what the engine does with it.
"""

from __future__ import annotations

# Monsters and war machines have a Unit Strength equal to their starting
# Wounds, which is a per-model lookup rather than a number in the table.
AS_STARTING_WOUNDS = 'as starting wounds'

# Rulebook p. 190-197. `models_per_rank` and `max_rank_bonus` of None mean the
# type cannot form ranks at all, printed as '-' in the rulebook's tables.
TROOP_TYPES = {
    # ── Infantry (p. 190-191) ──────────────────────────────────────────
    "regular infantry": {
        "models_per_rank": 5,
        "max_rank_bonus": 2,
        "unit_strength": 1,
        "rules": ("Press of Battle", "Massed Infantry", "Parry"),
    },
    "heavy infantry": {
        "models_per_rank": 4,
        "max_rank_bonus": 2,
        "unit_strength": 1,
        "rules": ("Steady in the Ranks", "Press of Battle", "Massed Infantry",
                  "Parry"),
    },
    "monstrous infantry": {
        "models_per_rank": 3,
        "max_rank_bonus": 2,
        "unit_strength": 3,
        "rules": ("Clumsy",),
    },
    "swarms": {
        "models_per_rank": None,
        "max_rank_bonus": None,
        "unit_strength": 3,
        "rules": ("Insignificant", "No One Cares", "Undisciplined"),
    },
    # ── Cavalry (p. 192-193) ───────────────────────────────────────────
    "light cavalry": {
        "models_per_rank": 5,
        "max_rank_bonus": 1,
        "unit_strength": 2,
        "rules": ("Split Profile (Cavalry)", "Cavalry Support"),
    },
    "heavy cavalry": {
        "models_per_rank": 4,
        "max_rank_bonus": 1,
        "unit_strength": 2,
        "rules": ("Split Profile (Cavalry)", "Cavalry Support"),
    },
    "monstrous cavalry": {
        "models_per_rank": 3,
        "max_rank_bonus": 1,
        "unit_strength": 3,
        "rules": ("Split Profile (Cavalry)", "Clumsy"),
    },
    "war beasts": {
        "models_per_rank": 5,
        "max_rank_bonus": 1,
        "unit_strength": 1,
        "rules": ("Undisciplined",),
    },
    # ── Chariots (p. 194-195) ──────────────────────────────────────────
    "light chariot": {
        "models_per_rank": 3,
        "max_rank_bonus": 1,
        "unit_strength": 3,
        "rules": ("Split Profile (Chariots)", "Iron Shod Wheels",
                  "Churning Wheels", "Firing Platform"),
    },
    "heavy chariot": {
        "models_per_rank": None,
        "max_rank_bonus": None,
        "unit_strength": 5,
        "rules": ("Split Profile (Chariots)", "Scythed Wheels", "Lumbering",
                  "Iron Shod Wheels", "Firing Platform"),
    },
    # ── Monsters (p. 196) ──────────────────────────────────────────────
    "monstrous creature": {
        "models_per_rank": None,
        "max_rank_bonus": None,
        "unit_strength": AS_STARTING_WOUNDS,
        "rules": ("Ridden Monster", "Lumbering"),
    },
    "behemoth": {
        "models_per_rank": None,
        "max_rank_bonus": None,
        "unit_strength": AS_STARTING_WOUNDS,
        "rules": ("Ridden Monster", "Lumbering", "Thunderstomp"),
    },
    # ── War machines (p. 197) ──────────────────────────────────────────
    "war machine": {
        "models_per_rank": None,
        "max_rank_bonus": None,
        "unit_strength": AS_STARTING_WOUNDS,
        "rules": ("Split Profile (War Machine)", "We're Not Paid to Fight",
                  "Weapon of War"),
    },
}

# The data writes several of these both ways round and in the singular or
# plural; the rulebook's own tables are no more consistent.
_ALIASES = {
    "swarm": "swarms",
    "monstrous creatures": "monstrous creature",
    "behemoths": "behemoth",
    "war beast": "war beasts",
    "war machines": "war machine",
    "light chariots": "light chariot",
    "heavy chariots": "heavy chariot",
}


def normalise(troop_type) -> str:
    """The bare troop type, e.g. 'Heavy chariot (named character)' -> 'heavy chariot'.

    The data is inconsistently cased, tags characters in parentheses, and gives
    a two-profile unit both types separated by a comma.
    """
    name = str(troop_type or "").split("(")[0].split(",")[0]
    name = " ".join(name.lower().split())
    return _ALIASES.get(name, name)


def properties(troop_type) -> dict | None:
    return TROOP_TYPES.get(normalise(troop_type))


def unit_strength(troop_type, default: int, wounds: int = 0) -> int:
    """Unit Strength per model. Monsters and war machines are worth their
    starting Wounds, so those need the model's profile passed in."""
    props = properties(troop_type)
    if props is None:
        return default
    value = props["unit_strength"]
    if value == AS_STARTING_WOUNDS:
        return max(1, wounds) if wounds else default
    return value


def models_per_rank(troop_type, default: int) -> int:
    """Models a rank needs to count towards Rank Bonus; 0 if the type cannot
    form ranks at all."""
    props = properties(troop_type)
    if props is None:
        return default
    return props["models_per_rank"] or 0


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


# A sub-category follows its parent unless it says otherwise, so a rule that
# names 'infantry' means monstrous infantry and swarms as well (p. 188).
INFANTRY = ("regular infantry", "heavy infantry", "monstrous infantry", "swarms")
CAVALRY = ("light cavalry", "heavy cavalry", "monstrous cavalry")


def is_infantry(troop_type) -> bool:
    return normalise(troop_type) in INFANTRY


def is_cavalry(troop_type) -> bool:
    return normalise(troop_type) in CAVALRY
