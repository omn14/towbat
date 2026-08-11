"""
Runtime loader for BattleScribe catalogues (.cat).

Parses the catalogues in `Warhammer-The-Old-World/` into an in-memory index so the
game and list builder can source unit characteristics directly from the official
data instead of the per-unit JSON files. The characteristics dict returned matches
the shape the rest of the codebase already expects (Model, M..Ld, Points), with a
few harmless extra keys (Unit, Troop Type, Unit Size, Special Rules).

This module has no dependency on the rest of the project (stdlib only), so it is
safe to import from models.py without creating an import cycle.
"""

from __future__ import annotations

import copy
import os
import re
import xml.etree.ElementTree as ET

# Stat keys in the canonical order used throughout the codebase.
STAT_KEYS = ["M", "WS", "BS", "S", "T", "W", "I", "A", "Ld"]

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CAT_DIR = os.path.join(REPO_DIR, "Warhammer-The-Old-World")

# The BattleScribe schema namespace; ElementTree prefixes every tag with it.
NS = "{http://www.battlescribe.net/schema/catalogueSchema}"

# Map display-name slugs to the canonical catalogue model slug when they differ.
NAME_ALIASES = {
    "orc_boyz": "orc_boy",
    "goblin_wolf_rider": "wolf_rider",
    "orc_boar_boy": "boar_boy",
}

# Army-organisation category ids, defined once in the .gst and shared by all
# factions. Units are assigned one via a primary categoryLink.
ORG_CATEGORY_BY_ID = {
    "3ba8-a41e-b6ae-d4ba": "Characters",   # Named Characters
    "a4cc-15c9-cfae-1b3b": "Characters",
    "f0e3-2e32-8866-ea32": "Core",
    "633f-f67a-1b6a-d203": "Special",
    "2bfe-5863-46fe-d284": "Rare",
    "5b84-2c3c-869d-3522": "Mercenaries",
}
ORG_CATEGORY_ORDER = ["Characters", "Core", "Special", "Rare", "Mercenaries", "Other"]


def _tag(elem: ET.Element) -> str:
    """Return an element's local tag name without the namespace prefix."""
    return elem.tag.split("}", 1)[-1]


def slugify(name: str) -> str:
    """Turn a display name into the filename-stem convention used by the game."""
    name = name.replace("'", "").replace("\u2019", "")
    name = name.replace("-", "_").replace(" ", "_").lower()
    name = re.sub(r"[^a-z0-9_]", "", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def _index_model_profiles(root: ET.Element) -> dict:
    """Map profile id -> {name, stats} for every Model-type stat profile."""
    profiles: dict = {}
    for profile in root.iter(f"{NS}profile"):
        if profile.get("typeName") != "Model":
            continue
        stats = {}
        for char in profile.iter(f"{NS}characteristic"):
            key = char.get("name")
            if key in STAT_KEYS:
                stats[key] = (char.text or "").strip()
        profiles[profile.get("id")] = {
            "name": profile.get("name", "Unknown"),
            "stats": stats,
        }
    return profiles


def _direct_child(parent: ET.Element, local_name: str):
    for child in parent:
        if _tag(child) == local_name:
            return child
    return None


def _model_points(model_entry: ET.Element):
    """Read the direct per-model points cost from a model selectionEntry."""
    costs = _direct_child(model_entry, "costs")
    if costs is None:
        return None
    for cost in costs:
        if cost.get("typeId") == "points":
            try:
                return int(float(cost.get("value", "0")))
            except ValueError:
                return None
    return None


def _first_model_profile_id(model_entry: ET.Element, profiles: dict):
    """Find the stat-profile id this model entry links to via its infoLinks."""
    info_links = _direct_child(model_entry, "infoLinks")
    if info_links is None:
        return None
    for link in info_links:
        if link.get("type") == "profile" and link.get("targetId") in profiles:
            return link.get("targetId")
    return None


def _index_org_categories(root: ET.Element) -> dict:
    """Map unit id -> org category via the root entryLinks that assign one."""
    mapping: dict = {}
    for link in root.iter(f"{NS}entryLink"):
        target = link.get("targetId")
        if not target:
            continue
        cats = _direct_child(link, "categoryLinks")
        if cats is None:
            continue
        for cat in cats:
            cid = cat.get("targetId")
            if cat.get("primary") == "true" and cid in ORG_CATEGORY_BY_ID:
                mapping[target] = ORG_CATEGORY_BY_ID[cid]
                break
    return mapping


def _unit_org_category(unit: ET.Element, org_map: dict) -> str:
    """Determine a unit's org category from the entryLink map or its own links."""
    cat = org_map.get(unit.get("id"))
    if cat:
        return cat
    for cat_link in unit.iter(f"{NS}categoryLink"):
        cid = cat_link.get("targetId")
        if cat_link.get("primary") == "true" and cid in ORG_CATEGORY_BY_ID:
            return ORG_CATEGORY_BY_ID[cid]
    return "Other"


def _unit_context(unit: ET.Element) -> dict:
    """Pull troop type, unit size and special-rule names from a unit entry."""
    troop_type = unit_size = None
    profiles = _direct_child(unit, "profiles")
    if profiles is not None:
        for profile in profiles:
            if profile.get("typeName") != "Unit":
                continue
            for char in profile.iter(f"{NS}characteristic"):
                if char.get("name") == "Troop Type":
                    troop_type = (char.text or "").strip()
                elif char.get("name") == "Unit Size":
                    unit_size = (char.text or "").strip()

    special_rules: list = []
    info_groups = _direct_child(unit, "infoGroups")
    if info_groups is not None:
        for group in info_groups:
            if group.get("name") != "Special Rules":
                continue
            links = _direct_child(group, "infoLinks")
            if links is None:
                continue
            for link in links:
                rule = link.get("name")
                if rule and rule not in special_rules:
                    special_rules.append(rule)

    return {
        "Troop Type": troop_type,
        "Unit Size": unit_size,
        "Special Rules": special_rules,
    }


def weapon_from_profile(name: str, chars: dict) -> dict:
    """Convert a BattleScribe Weapon profile (R/S/AP/Special Rules) into the
    game's weapon dict. Ranged weapons get range/strength/AP/shots/volley."""
    rng = (chars.get("R") or "").strip()
    strength = (chars.get("S") or "").strip()
    ap = (chars.get("AP") or "").strip()
    rules_txt = (chars.get("Special Rules") or "").strip()
    rules = [r.strip() for r in rules_txt.split(",") if r.strip() and r.strip() != "-"]

    is_ranged = rng.lower() not in ("combat", "", "-")
    weapon = {"name": name, "tag": "ranged" if is_ranged else "combat", "special_rules": rules}
    # Raw profile values for display (S like 'S+2', AP like '-2').
    if strength and strength != "S":
        weapon["strength"] = strength
    if ap and ap != "-":
        weapon["ap"] = ap
    if is_ranged:
        m = re.search(r"\d+", rng)
        weapon["ranged_range"] = int(m.group()) if m else 0
        # Numeric Strength, else None meaning 'use the wielder's Strength'.
        weapon["ranged_strength"] = int(strength) if strength.isdigit() else None
        # Catalogue AP is a save modifier ('-1'); game penetration is its negation.
        ap_m = re.search(r"-?\d+", ap)
        weapon["ranged_AP"] = -int(ap_m.group()) if ap_m else 0
        shots = 1
        for r in rules:
            sm = re.search(r"multiple shots.*?(\d+)", r, re.I)
            if sm:
                shots = int(sm.group(1))
        weapon["ranged_shots"] = shots
        weapon["volley_fire"] = any("volley" in r.lower() for r in rules)
    return weapon


def _weapon_profiles(root: ET.Element) -> list:
    """Return game weapon dicts for every Weapon profile (namespace-agnostic,
    so it works for both .cat catalogues and the .gst game system)."""
    out: list = []
    for profile in root.iter():
        if _tag(profile) != "profile" or profile.get("typeName") != "Weapon":
            continue
        chars = {}
        for c in profile.iter():
            if _tag(c) == "characteristic":
                chars[c.get("name")] = (c.text or "").strip()
        out.append(weapon_from_profile(profile.get("name", "Weapon"), chars))
    return out


def parse_catalogue_full(cat_path: str):
    """Parse one catalogue into (faction, unit_records, profile_records, weapons)."""
    tree = ET.parse(cat_path)
    root = tree.getroot()

    faction_name = root.get("name", os.path.splitext(os.path.basename(cat_path))[0])
    profiles = _index_model_profiles(root)
    org_map = _index_org_categories(root)
    records: list = []

    for unit in root.iter(f"{NS}selectionEntry"):
        if unit.get("type") != "unit":
            continue
        unit_name = unit.get("name", "Unknown")
        context = _unit_context(unit)
        category = _unit_org_category(unit, org_map)

        for model_entry in unit.iter(f"{NS}selectionEntry"):
            if model_entry.get("type") != "model":
                continue
            profile_id = _first_model_profile_id(model_entry, profiles)
            if profile_id is None:
                continue
            prof = profiles[profile_id]
            if not prof["stats"]:
                continue

            points = _model_points(model_entry)
            record = {"Model": prof["name"]}
            for key in STAT_KEYS:
                record[key] = prof["stats"].get(key, "")
            record["Points"] = points if points is not None else 0
            record["Unit"] = unit_name
            record["Troop Type"] = context["Troop Type"]
            record["Unit Size"] = context["Unit Size"]
            record["Special Rules"] = list(context["Special Rules"])
            record["Category"] = category
            record["Faction"] = faction_name
            records.append(record)

    # Standalone model profiles (mounts, champion variants) that are not tied to
    # their own 'model' selection entry, used only as a lookup fallback.
    profile_records: list = []
    for prof in profiles.values():
        if not prof["stats"]:
            continue
        rec = {"Model": prof["name"]}
        for key in STAT_KEYS:
            rec[key] = prof["stats"].get(key, "")
        rec["Points"] = 0
        rec["Unit"] = None
        rec["Troop Type"] = None
        rec["Unit Size"] = None
        rec["Special Rules"] = []
        rec["Category"] = "Other"
        rec["Faction"] = faction_name
        profile_records.append(rec)

    weapons = _weapon_profiles(root)
    return faction_name, records, profile_records, weapons


def parse_weapons(cat_path: str) -> list:
    """Parse a .cat or .gst file and return its game weapon dicts."""
    root = ET.parse(cat_path).getroot()
    return _weapon_profiles(root)


def parse_catalogue(cat_path: str):
    """Parse one catalogue into (faction_name, list_of_unit_model_records)."""
    faction_name, records, _profiles, _weapons = parse_catalogue_full(cat_path)
    return faction_name, records


class Catalogue:
    """In-memory index of every model in every .cat under a directory."""

    def __init__(self, cat_dir: str = DEFAULT_CAT_DIR):
        self.cat_dir = cat_dir
        self.by_slug: dict = {}
        # Fallback index of every model profile (includes mounts) by slug.
        self.all_by_slug: dict = {}
        # Weapon profiles (from every .cat and the .gst) by slug.
        self.weapons_by_slug: dict = {}
        self.factions: dict = {}
        self._load()

    def _load(self):
        if not os.path.isdir(self.cat_dir):
            print(f"[battlescribe] catalogue directory not found: {self.cat_dir}")
            return
        for filename in sorted(os.listdir(self.cat_dir)):
            path = os.path.join(self.cat_dir, filename)
            if filename.endswith(".gst"):
                # Game system holds the common (shared) weapons.
                try:
                    for w in parse_weapons(path):
                        self.weapons_by_slug.setdefault(slugify(w["name"]), w)
                except ET.ParseError as exc:
                    print(f"[battlescribe] failed to parse {filename}: {exc}")
                continue
            if not filename.endswith(".cat"):
                continue
            try:
                faction_name, records, profile_records, weapons = parse_catalogue_full(path)
            except ET.ParseError as exc:
                print(f"[battlescribe] failed to parse {filename}: {exc}")
                continue
            for record in records:
                slug = slugify(record["Model"])
                # First occurrence wins; prefer an entry that has a points cost.
                existing = self.by_slug.get(slug)
                if existing is None or (not existing.get("Points") and record.get("Points")):
                    self.by_slug[slug] = record
                self.factions.setdefault(faction_name, set()).add(record["Model"])
            for record in profile_records:
                self.all_by_slug.setdefault(slugify(record["Model"]), record)
            for w in weapons:
                self.weapons_by_slug.setdefault(slugify(w["name"]), w)
        print(f"[battlescribe] loaded {len(self.by_slug)} models, "
              f"{len(self.weapons_by_slug)} weapons across {len(self.factions)} factions")

    def characteristics(self, name: str):
        """Return a fresh characteristics dict for a display name, or None."""
        slug = slugify(name)
        slug = NAME_ALIASES.get(slug, slug)
        record = self.by_slug.get(slug) or self.all_by_slug.get(slug)
        return copy.deepcopy(record) if record else None

    def weapon(self, name: str):
        """Return a fresh game weapon dict for a weapon name, or None."""
        record = self.weapons_by_slug.get(slugify(name))
        return copy.deepcopy(record) if record else None

    def iter_models(self):
        """Yield (model_name, faction, characteristics) for every model."""
        for record in self.by_slug.values():
            yield record["Model"], record.get("Faction", "Unknown"), copy.deepcopy(record)


_catalogue: Catalogue | None = None


def get_catalogue(cat_dir: str = DEFAULT_CAT_DIR) -> Catalogue:
    """Return the shared Catalogue singleton, loading it on first use."""
    global _catalogue
    if _catalogue is None:
        _catalogue = Catalogue(cat_dir)
    return _catalogue
