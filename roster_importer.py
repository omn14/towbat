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
import json
import math
import os

from battlescribe import slugify, weapon_from_profile

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FILES = 5  # default frontage when the roster has no formation info


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


def _collect_special_rules(selection: dict, out: list) -> None:
    """Gather special-rule names from a unit and its (non-mount) upgrades.

    Rules can sit on the unit itself or on any nested upgrade selection (e.g.
    the Skirmishers formation upgrade), as a 'Special Rule' profile or a rule.
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
    for force in forces:
        for unit in force.get("selections", []):
            if unit.get("type") != "unit":
                continue
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
            units.append({
                "name": _primary_model_name(unit),
                "faction": faction_slug,
                "nmodels": nmodels,
                "files": files,
                "ranks": ranks,
                "points_cost": _sum_pts(unit),
                "category": _primary_category(unit),
                "mounted": bool(mount),
                "mount": mount,
                "weapons": weapons,
                "special_rules": special_rules,
            })

    return {"budget": limit or total, "faction": faction_slug, "units": units}


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
