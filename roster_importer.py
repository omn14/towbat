"""
Import a NewRecruit / BattleScribe roster export (.json) into the game's flat
army-list format used by strategy_armies/ and the list builder.

A roster nests unit -> model/crew -> upgrade/mount. This flattens each top-level
unit into {name, faction, nmodels, files, ranks, points_cost, category, mounted}.
Formation (files/ranks) is not stored in a roster, so sensible defaults are used.

Usage:
    python roster_importer.py strategy_armies/Bm.json
    python roster_importer.py strategy_armies/Bm.json -o strategy_armies/empire.json
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
import re

from battlescribe import get_catalogue, slugify, spell_from_profile, spell_key, weapon_from_profile

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FILES = 5  # default frontage when the roster has no formation info
MAGIC_ITEM_TYPES = {
    "Magic Weapons", "Magic Armour", "Talismans", "Enchanted Items",
    "Arcane Items", "Magic Standards",
}


def _selected_copy(selection: dict) -> dict:
    """Retain selected data, excluding available choices and zero-count subtrees."""
    result = {key: deepcopy(value) for key, value in selection.items()
              if key not in {"selections", "selectionEntries", "selectionEntryGroups"}}
    result["selections"] = [_selected_copy(child)
                            for child in selection.get("selections", [])
                            if child.get("number", 1) > 0]
    return result


def _selection_metadata(unit: dict, prefix: str) -> dict:
    """Keep source data and structural owners; no item or command effects are applied."""
    selections = []

    def walk(selection, parent_ref, owner_ref, index):
        identity = selection.get("id") or f"selection-{index}"
        reference = f"{parent_ref}/{identity}"
        profiles = selection.get("profiles", [])
        if (selection.get("type") in {"unit", "model", "crew", "mount"}
                or any(profile.get("typeName") in {"Model", "Command"}
                       for profile in profiles)):
            owner_ref = reference
        record = {key: deepcopy(value) for key, value in selection.items()
                  if key != "selections"}
        record.update(ref=reference, parent_ref=parent_ref, owner_ref=owner_ref)
        selections.append(record)
        for child_index, child in enumerate(selection.get("selections", [])):
            walk(child, reference, owner_ref, child_index)

    walk(unit, prefix, prefix, 0)
    references = {selection["id"]: selection["ref"]
                  for selection in selections if selection.get("id")}
    items = []
    command = []
    equipment = []
    spell_sources = []
    item_owners = {}
    for selection in selections:
        profiles = selection.get("profiles", [])
        common = {
            "selection_ref": selection["ref"],
            "owner_ref": selection["owner_ref"],
            "name": selection.get("name", "Unknown"),
            "number": selection.get("number", 1),
            "points_cost": _pts(selection),
        }
        item_profiles = [profile for profile in profiles
                         if profile.get("typeName") in MAGIC_ITEM_TYPES]
        item_ref = selection["ref"] if item_profiles else item_owners.get(selection["parent_ref"])
        item_owners[selection["ref"]] = item_ref
        if item_profiles:
            primary = item_profiles[0]
            items.append(dict(common, name=primary.get("name", common["name"]),
                              definition_id=primary.get("id") or selection.get("entryId"),
                              category=primary["typeName"], profiles=deepcopy(item_profiles),
                              effect_status="unsupported"))
        for profile in profiles:
            profile_type = profile.get("typeName")
            if profile_type == "Command":
                targets = [references[association["to"]]
                           for association in selection.get("associations", [])
                           if association.get("type") == "outgoing"
                           and association.get("to") in references]
                command.append(dict(common, role=slugify(profile.get("name", "")),
                                    model_refs=targets, profiles=deepcopy(profiles)))
            elif profile_type in {"Weapon", "Armour"}:
                equipment.append(dict(common, name=profile.get("name", common["name"]),
                                      category=profile_type, profile=deepcopy(profile)))
            elif profile_type == "Spell":
                explicit = (selection.get("type") == "upgrade" and profile.get("name")
                            and slugify(common["name"]) == slugify(profile["name"]))
                spell_sources.append(dict(common, name=profile.get("name"),
                                          kind="item" if item_ref else "selected" if explicit else "pool",
                                          item_ref=item_ref,
                                          profile=deepcopy(profile)))
    return {"roster_selections": selections, "magic_items": items,
            "command": command, "equipment": equipment, "spell_sources": spell_sources}


def _pts(selection: dict) -> int:
    return sum(c.get("value", 0) for c in selection.get("costs", []) if c.get("name") == "pts")


def _sum_pts(selection: dict) -> int:
    """Total points of a selection including all nested upgrades/mounts."""
    total = _pts(selection)
    for sub in selection.get("selections", []):
        total += _sum_pts(sub)
    return total


def _count_models(unit: dict) -> int:
    """Number of physical models. Command (crew) are promotions of existing
    rank-and-file models, so they are not counted as extra bodies."""
    models = sum(sub.get("number", 1) for sub in unit.get("selections", [])
                 if sub.get("type") == "model")
    if models:
        return models
    # Fallback for units represented only by crew (e.g. war-machine crews).
    return sum(sub.get("number", 1) for sub in unit.get("selections", [])
               if sub.get("type") == "crew")


def _primary_model_name(unit: dict) -> str:
    """Name of the unit's fighting profile (first model, else crew, else unit)."""
    for kind in ("model", "crew"):
        for sub in unit.get("selections", []):
            if sub.get("type") == kind:
                return sub.get("name", unit.get("name", "Unknown"))
    return unit.get("name", "Unknown")


def _is_mounted(selection: dict) -> bool:
    for sub in selection.get("selections", []):
        if sub.get("type") == "mount" or _is_mounted(sub):
            return True
    return False


def _mount_name(selection: dict):
    """Name of the chosen mount (first nested 'mount' selection), or None."""
    for sub in selection.get("selections", []):
        if sub.get("type") == "mount":
            return sub.get("name")
        found = _mount_name(sub)
        if found:
            return found
    return None


def _collect_weapons(selection: dict, out: list) -> None:
    """Gather weapon upgrades from a unit, skipping mount subtrees."""
    for sub in selection.get("selections", []):
        if sub.get("type") == "mount":
            continue
        for p in sub.get("profiles", []):
            if p.get("typeName") == "Weapon":
                chars = {c["name"]: c.get("$text", "") for c in p.get("characteristics", [])}
                out.append(weapon_from_profile(sub.get("name", "Weapon"), chars))
                break
        _collect_weapons(sub, out)


def _collect_armour(selection: dict, out: list) -> None:
    """Gather worn armour names (Armour profiles) from a unit and its upgrades,
    including barding on a mount (it improves the rider's save)."""
    for p in selection.get("profiles", []):
        if p.get("typeName") == "Armour" and p.get("name"):
            out.append(p["name"])
    for sub in selection.get("selections", []):
        _collect_armour(sub, out)


def _collect_special_rules(selection: dict, out: list) -> None:
    """Gather special-rule names from a unit and its (non-mount) upgrades.

    Rules can sit on the unit itself or on any nested upgrade selection (e.g.
    the Skirmishers formation upgrade), as a 'Special Rule' profile or a rule.

    Mount subtrees are left to ``_collect_mount_rules``: they belong to the
    mount's own profile, and the engine looks through to it.
    """
    for p in selection.get("profiles", []):
        if p.get("typeName") == "Special Rule" and p.get("name"):
            out.append(p["name"])
    for r in selection.get("rules", []):
        if r.get("name"):
            out.append(r["name"])
    for sub in selection.get("selections", []):
        if sub.get("type") == "mount":
            continue
        _collect_special_rules(sub, out)


def _collect_mount_rules(selection: dict, out: list) -> None:
    """Special rules carried by the mount, wherever it sits in the unit.

    A Captain on a Demigryph takes Swiftstride from the beast, not from
    himself, and it decides whether the unit he joins may add Swiftstride's
    die to its charge.
    """
    for sub in selection.get("selections", []):
        if sub.get("type") == "mount":
            _collect_special_rules(sub, out)
        else:
            _collect_mount_rules(sub, out)


def _collect_spells(selection: dict, out: list) -> None:
    """Gather exported spell definitions, including pools and selected bound spells."""
    # Only a selected item grants its spell; available upgrades grant nothing (p. 342).
    if (selection.get('name') == 'Ruby Ring of Ruin'
            or any(p.get('name') == 'Ruby Ring of Ruin'
                   for p in selection.get('profiles', []))):
        spell = get_catalogue().spell('Fireball')
        if spell:
            out.append(dict(spell, bound=True, power_level=1, source='Ruby Ring of Ruin'))
        if selection.get('name') == 'Ruby Ring of Ruin':
            return
    if any(profile.get("typeName") in MAGIC_ITEM_TYPES
           for profile in selection.get("profiles", [])):
        return
    for p in selection.get("profiles", []):
        if p.get("typeName") == "Spell" and p.get("name"):
            chars = {c["name"]: c.get("$text", "") for c in p.get("characteristics", [])}
            out.append(spell_from_profile(p["name"], chars))
    for sub in selection.get("selections", []):
        _collect_spells(sub, out)


def _selected_spell_names(selection: dict) -> set:
    """Only a matching spell upgrade explicitly selects an ordinary spell."""
    names = set()
    if any(profile.get("typeName") in MAGIC_ITEM_TYPES
           for profile in selection.get("profiles", [])):
        return names
    if selection.get("type") == "upgrade":
        for profile in selection.get("profiles", []):
            if (profile.get("typeName") == "Spell" and profile.get("name")
                    and slugify(selection.get("name", "")) == slugify(profile["name"])):
                names.add(profile["name"])
    for child in selection.get("selections", []):
        names.update(_selected_spell_names(child))
    return names


def _wizard_level(selection: dict):
    """A Wizard's Level of Wizardry, taken from its 'Wizard Level N' upgrade.

    Levels 1 and 2 are often the profile's own, with only an upgrade to 3 or 4
    listed, so a wizard with spells but no upgrade counts as Level 1.
    """
    best = None
    def walk(sel):
        nonlocal best
        m = re.match(r"wizard level\s*(\d+)", str(sel.get("name", "")).strip(), re.I)
        if m:
            best = max(best or 0, int(m.group(1)))
        for sub in sel.get("selections", []):
            walk(sub)
    walk(selection)
    return best


def _primary_category(unit: dict):
    for cat in unit.get("categories", []):
        if cat.get("primary"):
            return cat.get("name")
    return None


def import_roster(path: str) -> dict:
    """Convert a roster JSON file into the game's army-list dict."""
    with open(path, "r", encoding="utf-8") as f:
        roster = json.load(f)["roster"]

    forces = roster.get("forces", [])
    faction_name = forces[0].get("catalogueName", "Unknown") if forces else "Unknown"
    faction_slug = slugify(faction_name)

    limit = next((c["value"] for c in roster.get("costLimits", []) if c["name"] == "pts"), None)
    total = next((c["value"] for c in roster.get("costs", []) if c["name"] == "pts"), 0)

    units = []
    for force_index, force in enumerate(forces):
        for unit_index, raw_unit in enumerate(force.get("selections", [])):
            if raw_unit.get("type") != "unit" or raw_unit.get("number", 1) <= 0:
                continue
            unit = _selected_copy(raw_unit)
            force_ref = force.get("id") or f"force-{force_index}"
            unit_ref = unit.get("id") or f"unit-{unit_index}"
            prefix = f"{roster.get('id', 'roster')}/{force_ref}/{unit_ref}"
            metadata = _selection_metadata(unit, prefix)
            nmodels = max(1, _count_models(unit))
            files = min(DEFAULT_FILES, nmodels)
            ranks = math.ceil(nmodels / files)
            mount = _mount_name(unit)
            weapons: list = []
            _collect_weapons(unit, weapons)
            # De-duplicate by weapon name (keep first occurrence).
            seen: set = set()
            weapons = [w for w in weapons if not (w["name"] in seen or seen.add(w["name"]))]
            special_rules: list = []
            _collect_special_rules(unit, special_rules)
            # De-duplicate, preserving order.
            special_rules = list(dict.fromkeys(special_rules))
            mount_rules: list = []
            _collect_mount_rules(unit, mount_rules)
            mount_rules = list(dict.fromkeys(mount_rules))
            armour: list = []
            _collect_armour(unit, armour)
            armour = list(dict.fromkeys(armour))
            spells: list = []
            _collect_spells(unit, spells)
            seen = set()
            spells = [s for s in spells
                      if not (spell_key(s) in seen or seen.add(spell_key(s)))]
            selected_spells = _selected_spell_names(unit)
            spell_pool = [spell for spell in spells
                          if not spell.get("bound") and spell["name"] not in selected_spells]
            spells = [spell for spell in spells
                      if spell.get("bound") or spell["name"] in selected_spells]
            level = _wizard_level(unit)
            if (spell_pool or any(not s.get('bound') for s in spells)) and not level:
                level = 1
            units.append({
                **metadata,
                "roster_source": {key: deepcopy(value) for key, value in force.items()
                                  if key not in {"selections", "forces"}},
                "name": _primary_model_name(unit),
                "faction": slugify(force.get("catalogueName", faction_name)),
                "nmodels": nmodels,
                "files": files,
                "ranks": ranks,
                "points_cost": _sum_pts(unit),
                "category": _primary_category(unit),
                "mounted": bool(mount),
                "mount": mount,
                "mount_special_rules": mount_rules,
                "weapons": weapons,
                "special_rules": special_rules,
                "armour": armour,
                "spells": spells,
                "spell_pool": spell_pool,
                "spell_generation_pending": bool(spell_pool),
                "wizard_level": level,
            })

    return {"budget": limit or total, "faction": faction_slug, "units": units,
            "roster_source": {key: deepcopy(value) for key, value in roster.items()
                              if key != "forces"}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roster", help="Path to a NewRecruit/BattleScribe roster .json")
    parser.add_argument("-o", "--out", help="Output army-list path (default: alongside input).")
    args = parser.parse_args()

    army = import_roster(args.roster)

    out_path = args.out
    if not out_path:
        stem = os.path.splitext(os.path.basename(args.roster))[0]
        out_path = os.path.join(REPO_DIR, "strategy_armies", f"{stem}_army.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(army, f, indent=4, ensure_ascii=False)

    print(f"Imported {len(army['units'])} units ({army['budget']} pts, {army['faction']}) -> {out_path}")
    for u in army["units"]:
        mount = f" on {u['mount']}" if u.get("mount") else ""
        print(f"  {u['name']:26} {u['nmodels']:>3} models  {u['points_cost']:>4} pts  {u['category']}{mount}")


if __name__ == "__main__":
    main()
