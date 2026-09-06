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


def has_move_and_shoot(rules) -> bool:
    """True if *rules* names Move & Shoot (p. 174).

    The catalogue also carries `Move or Shoot`, which is the opposite rule and
    differs by one word, so the joiner is what has to match.
    """
    return any(re.search(r"move\s*(?:&|and)\s*shoot", str(r), re.I)
               for r in (rules or []))


def has_move_or_shoot(rules) -> bool:
    """True if *rules* names Move or Shoot (p. 174).

    The opposite of the rule above, and one word apart from it, so the joiner
    is again what has to match. 25 weapons carry it, all of them artillery.
    """
    return any(re.search(r"move\s*or\s*shoot", str(r), re.I)
               for r in (rules or []))


def has_quick_shot(rules) -> bool:
    """True if *rules* names Quick Shot (p. 175).

    The catalogue spells it both `Quick Shot` (32 weapons) and `Quick Shoot`
    (1), so the second 'o' is optional.
    """
    return any(re.search(r"quick\s*shoo?t", str(r), re.I)
               for r in (rules or []))


def has_ponderous(rules) -> bool:
    """True if *rules* names Ponderous (p. 175)."""
    return any(re.search(r"ponderous", str(r), re.I)
               for r in (rules or []))


# Anchored, because the catalogue also has a Lightning Strike that is neither.
_STRIKE_FIRST = re.compile(r"\s*strikes?\s*first\s*$", re.I)
_STRIKE_LAST = re.compile(r"\s*strikes?\s*last\s*$", re.I)


def has_strike_first(rules) -> bool:
    """True if *rules* names Strike First (p. 177).

    The catalogue spells it both `Strike First` and `Strikes First`.
    """
    return any(_STRIKE_FIRST.match(str(r)) for r in (rules or []))


def has_strike_last(rules) -> bool:
    """True if *rules* names Strike Last (p. 178)."""
    return any(_STRIKE_LAST.match(str(r)) for r in (rules or []))


def has_killing_blow(rules) -> bool:
    """True if *rules* names Killing Blow (p. 172).

    One spelling across the catalogues, on 20 weapons and 4 models.
    """
    return any(re.search(r"killing\s*blow", str(r), re.I)
               for r in (rules or []))


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


def _parse_base_size(text: str):
    """Parse a base-size string ('25x50', '50', '50x50 min...') into (width_mm, depth_mm)."""
    if not text:
        return None
    m = re.search(r"(\d+)\s*[xX]\s*(\d+)", text)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    m = re.search(r"(\d+)", text)
    if m:
        d = int(m.group(1))  # round base: same value both dimensions
        return (d, d)
    return None


def _model_base_size(model_entry: ET.Element):
    """Base size (width_mm, depth_mm) from a model entry's own 'Base (WxD)' infoLink."""
    info_links = _direct_child(model_entry, "infoLinks")
    if info_links is None:
        return None
    for link in info_links:
        name = link.get("name") or ""
        if name.startswith("Base"):
            size = _parse_base_size(name)
            if size:
                return size
    return None


def _index_base_by_profile(root: ET.Element, profiles: dict) -> dict:
    """Map model-profile id -> base size, so standalone profiles (mounts) get a base.
    Mount-subtype entries win, so a mount's own base is preferred over, e.g., a
    chariot that also references the same creature profile."""
    mapping: dict = {}
    for want_mount in (True, False):
        for entry in root.iter(f"{NS}selectionEntry"):
            if entry.get("type") != "model":
                continue
            if (entry.get("subType") == "mount") != want_mount:
                continue
            pid = _first_model_profile_id(entry, profiles)
            size = _model_base_size(entry)
            if pid and size:
                mapping.setdefault(pid, size)
    return mapping


def _apply_base_size(record: dict, size) -> None:
    """Store base dimensions on a record; leaves them unset when unknown."""
    if not size:
        return
    record["Base Size"] = f"{size[0]}x{size[1]}"
    record["base_width_mm"] = size[0]
    record["base_depth_mm"] = size[1]


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


def _effective_link_name(link: ET.Element) -> str:
    """Return an infoLink's display name after applying its <modifier> children.
    Units carry a rule's parameter this way, e.g. name 'Regeneration' + an
    append modifier '(5+)' -> 'Regeneration (5+)'."""
    name = link.get("name") or ""
    mods = _direct_child(link, "modifiers")
    if mods is None:
        return name
    for mod in mods:
        if _tag(mod) != "modifier" or mod.get("field") != "name":
            continue
        value = mod.get("value") or ""
        mtype = mod.get("type")
        if mtype == "append":
            sep = " " if value.startswith("(") and not name.endswith(" ") else ""
            name = f"{name}{sep}{value}"
        elif mtype == "prepend":
            name = f"{value}{name}"
        elif mtype == "set":
            name = value
    return name


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
                rule = _effective_link_name(link)
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
    notes = (chars.get("Notes") or "").strip()
    rules = [r.strip() for r in rules_txt.split(",") if r.strip() and r.strip() != "-"]

    is_ranged = rng.lower() not in ("combat", "", "-")
    weapon = {"name": name, "tag": "ranged" if is_ranged else "combat", "special_rules": rules}
    # Raw profile values for display (S like 'S+2', AP like '-2').
    if strength and strength != "S":
        weapon["strength"] = strength
    if ap and ap != "-":
        weapon["ap"] = ap
    # Notes often carry the actual rule (e.g. charge-only modifiers) — keep them.
    if notes:
        weapon["notes"] = notes
        # Some weapons (e.g. Blunderbuss) ignore certain To Hit penalties.
        low = notes.lower()
        if "no negative modifier" in low or "no penalt" in low:
            ignore = []
            if "long range" in low:
                ignore.append("long_range")
            if "multiple shots" in low:
                ignore.append("multiple_shots")
            if "stand & shoot" in low or "stand and shoot" in low:
                ignore.append("stand_and_shoot")
            if ignore:
                weapon["ignore_to_hit_penalties"] = ignore
    # Melee rules, so set for every weapon rather than inside the ranged branch
    # below — where they sat at first, which left them off exactly the weapons
    # that carry them.
    weapon["strike_first"] = has_strike_first(rules)
    weapon["strike_last"] = has_strike_last(rules)
    weapon["killing_blow"] = has_killing_blow(rules)
    if not is_ranged:
        # Combat modifiers: 'S+2' -> +2 Strength; '-2' -> AP 2 penetration.
        sb = re.search(r"S\s*\+\s*(\d+)", strength)
        if sb:
            weapon["strength_bonus"] = int(sb.group(1))
        apb = re.search(r"-\s*(\d+)", ap)
        if apb:
            weapon["ap_penetration"] = int(apb.group(1))
        # A parenthetical AP (e.g. '-1 (-2)') is the value on the charge.
        apc = re.search(r"\(\s*-\s*(\d+)\s*\)", ap)
        if apc:
            weapon["ap_penetration_charge"] = int(apc.group(1))
        # Lance-style weapons apply their modifiers only on the charge.
        low = notes.lower()
        if "charged" in low and "only" in low:
            weapon["charge_only"] = True
    if is_ranged:
        # Range may be a single value or a 'min-max' band (e.g. Mortar '12-48').
        nums = re.findall(r"\d+", rng)
        if len(nums) >= 2:
            weapon["ranged_range_min"] = int(nums[0])
            weapon["ranged_range"] = int(nums[1])
        elif nums:
            weapon["ranged_range"] = int(nums[0])
        else:
            weapon["ranged_range"] = 0
        # Strength: leading number (handles '2 (6)'); a bracket gives the
        # central-hole Strength used by Bombardment weapons.
        sm = re.match(r"\s*(\d+)", strength)
        weapon["ranged_strength"] = int(sm.group(1)) if sm else None
        sc = re.search(r"\(\s*(\d+)\s*\)", strength)
        if sc:
            weapon["ranged_strength_central"] = int(sc.group(1))
        # Catalogue AP is a save modifier ('-1'); game penetration is its negation.
        ap_m = re.search(r"-?\d+", ap)
        weapon["ranged_AP"] = -int(ap_m.group()) if ap_m else 0
        apc = re.search(r"\(\s*(-?\d+)\s*\)", ap)
        if apc:
            weapon["ranged_AP_central"] = -int(apc.group(1))
        weapon["ranged_shots"] = 1
        for r in rules:
            ms = re.search(r"multiple shots\s*\(([^)]+)\)", r, re.I)
            if ms:
                expr = ms.group(1).strip().upper().replace(" ", "")
                if expr.isdigit():
                    weapon["ranged_shots"] = int(expr)
                else:
                    # Random count (e.g. 'D3', 'D6', 'D3+1') rolled when firing.
                    weapon["ranged_shots_dice"] = expr
            mw = re.search(r"multiple wounds\s*\(([^)]+)\)", r, re.I)
            if mw:
                weapon["multiple_wounds"] = mw.group(1).strip().upper().replace(" ", "")
        weapon["volley_fire"] = any("volley" in r.lower() for r in rules)
        weapon["move_and_shoot"] = has_move_and_shoot(rules)
        weapon["move_or_shoot"] = has_move_or_shoot(rules)
        weapon["quick_shot"] = has_quick_shot(rules)
        weapon["ponderous"] = has_ponderous(rules)
        # Blast template diameter (e.g. '5" blast template') from the Notes.
        bm = re.search(r"(\d+)\s*[\"\u201d]?\s*blast", notes, re.I)
        if bm:
            weapon["blast_diameter"] = int(bm.group(1))
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


# ── Spells (Rulebook p. 106) ──────────────────────────────────────────────

# A spell's Type decides the phase it may be cast in (Rulebook p. 108).
SPELL_PHASES = {
    'enchantment': 'strategy',
    'hex': 'strategy',
    'conveyance': 'movement',
    'magic missile': 'shooting',
    'magical vortex': 'shooting',
    'assailment': 'combat',
    'prayer': 'strategy',
}


def _casting_value(text):
    """Leading casting value of a spell: '8+' -> 8. A boosted spell writes both
    versions as '8+/11+'; the basic one is what the engine casts."""
    m = re.search(r"(\d+)", str(text or ""))
    return int(m.group(1)) if m else None


def _spell_range(text):
    """A spell's range in inches, or the string 'Self'/'Combat' when it has no
    measured range. Returns None when the data gives none."""
    s = str(text or "").strip()
    low = s.lower()
    if low.startswith("self"):
        return "Self"
    if low.startswith("combat"):
        return "Combat"
    m = re.search(r"(\d+)", s.replace("D6", "").replace("d6", ""))
    return int(m.group(1)) if m else None


def spell_from_profile(name: str, chars: dict) -> dict:
    """Convert a BattleScribe Spell profile into the game's spell dict."""
    kind = (chars.get("Type") or "").strip()
    number = chars.get("Number") or ""
    return {
        "name": name,
        "type": kind,
        "phase": SPELL_PHASES.get(kind.lower(), "strategy"),
        "casting_value": _casting_value(chars.get("Casting Value")),
        "range": _spell_range(chars.get("Range")),
        "effect": (chars.get("Effect") or "").strip(),
        # The seventh spell of a lore is its signature and carries no number.
        "number": int(number) if str(number).strip().isdigit() else None,
    }


def _spell_lores(cat_path: str) -> dict:
    """Map lore name -> list of game spell dicts for one catalogue.

    Each Lore of Magic is a shared infoGroup holding its spells; the eight full
    lores hold seven, six numbered and a signature.
    """
    try:
        root = ET.parse(cat_path).getroot()
    except ET.ParseError:
        return {}
    out: dict = {}
    for group in root.iter():
        if _tag(group) != "infoGroup":
            continue
        spells = []
        for profile in group.iter():
            if _tag(profile) != "profile" or profile.get("typeName") != "Spell":
                continue
            chars = {c.get("name"): (c.text or "").strip()
                     for c in profile.iter() if _tag(c) == "characteristic"}
            spells.append(spell_from_profile(profile.get("name", "Spell"), chars))
        if spells:
            out.setdefault(group.get("name") or "Unknown Lore", spells)
    return out


def _index_entries_by_name(root: ET.Element) -> dict:
    """Map selectionEntry name -> element, for following entryLinks."""
    out: dict = {}
    for entry in root.iter(f"{NS}selectionEntry"):
        out.setdefault(entry.get("name"), entry)
    return out


def _linked_entries(entry: ET.Element, by_name: dict) -> list:
    """Model entries a unit reaches through <entryLinks> rather than nesting.

    Chariots are built this way: 'Empire War Wagons' carries the Troop Type and
    links out to a sibling 'War Wagon' model entry.
    """
    links = _direct_child(entry, "entryLinks")
    if links is None:
        return []
    out = []
    for link in links:
        if link.get("type") != "selectionEntry":
            continue
        target = by_name.get(link.get("name"))
        if target is not None and target is not entry:
            out.append(target)
    return out


def _entry_count(entry: ET.Element, default: int = 1) -> int:
    """How many of an entry a unit takes, from its selection constraints.
    A War Wagon's crew is one entry constrained to exactly 6, its horses to 2."""
    constraints = _direct_child(entry, "constraints")
    if constraints is None:
        return default
    counts = {}
    for c in constraints:
        if c.get("field") != "selections":
            continue
        try:
            counts[c.get("type")] = int(float(c.get("value", "0")))
        except (TypeError, ValueError):
            continue
    for key in ("max", "min"):
        if counts.get(key, 0) > 0:
            return counts[key]
    return default


def _model_parts(model_entry: ET.Element, by_name: dict) -> dict:
    """Crew and beast models belonging to a split-profile model, with how many
    of each the entry takes.

    A chariot's crew is a nested entry marked subType='crew'; the beasts that
    draw it are linked out and tagged with a CHARIOT CREW category. A war
    machine's crew is neither — it is a plain entry link, named for what the
    crew are ('Gun Crew', 'Dwarf Crew'), which is why war machines came back
    with no crew at all and so no Movement of their own.
    """
    crew, beasts = [], []
    entries = _direct_child(model_entry, "selectionEntries")
    if entries is not None:
        for sub in entries:
            if sub.get("type") == "model" and sub.get("subType") == "crew":
                crew.append({"name": sub.get("name"), "count": _entry_count(sub)})
    links = _direct_child(model_entry, "entryLinks")
    if links is not None:
        for link in links:
            name = (link.get("name") or "").strip()
            cats = _direct_child(link, "categoryLinks")
            drawn = False
            for cat in (cats if cats is not None else []):
                if (cat.get("name") or "").strip().upper() == "CHARIOT CREW":
                    beasts.append({"name": name, "count": _entry_count(link)})
                    drawn = True
                    break
            if not drawn and name.lower().endswith("crew"):
                crew.append({"name": name, "count": _entry_count(link)})
    return {"Crew": crew, "Beasts": beasts}


def parse_catalogue_full(cat_path: str):
    """Parse one catalogue into (faction, unit_records, profile_records, weapons)."""
    tree = ET.parse(cat_path)
    root = tree.getroot()

    faction_name = root.get("name", os.path.splitext(os.path.basename(cat_path))[0])
    profiles = _index_model_profiles(root)
    org_map = _index_org_categories(root)
    base_by_profile = _index_base_by_profile(root, profiles)
    entries_by_name = _index_entries_by_name(root)
    records: list = []

    for unit in root.iter(f"{NS}selectionEntry"):
        if unit.get("type") != "unit":
            continue
        unit_name = unit.get("name", "Unknown")
        context = _unit_context(unit)
        category = _unit_org_category(unit, org_map)

        model_entries = list(unit.iter(f"{NS}selectionEntry"))
        for linked in _linked_entries(unit, entries_by_name):
            model_entries.extend(linked.iter(f"{NS}selectionEntry"))
        for model_entry in model_entries:
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
            base = _model_base_size(model_entry) or base_by_profile.get(profile_id)
            _apply_base_size(record, base)
            record["Unit"] = unit_name
            record["Troop Type"] = context["Troop Type"]
            record["Unit Size"] = context["Unit Size"]
            record["Special Rules"] = list(context["Special Rules"])
            record["Category"] = category
            record["Faction"] = faction_name
            record.update(_model_parts(model_entry, entries_by_name))
            records.append(record)

    # Standalone model profiles (mounts, champion variants) that are not tied to
    # their own 'model' selection entry, used only as a lookup fallback.
    profile_records: list = []
    for pid, prof in profiles.items():
        if not prof["stats"]:
            continue
        rec = {"Model": prof["name"]}
        for key in STAT_KEYS:
            rec[key] = prof["stats"].get(key, "")
        rec["Points"] = 0
        _apply_base_size(rec, base_by_profile.get(pid))
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


def _rule_descriptions(cat_path: str) -> dict:
    """Map Special Rule name -> description text for a catalogue or game system.

    Army-specific abilities are defined in the .cat files; the core rulebook
    keywords (Impact Hits, Fear, the chariot rules...) are defined only in the
    .gst, which uses a different XML namespace, so match on the local tag.
    """
    try:
        root = ET.parse(cat_path).getroot()
    except ET.ParseError:
        return {}
    out: dict = {}
    for prof in root.iter():
        if _tag(prof) != "profile" or prof.get("typeName") != "Special Rule":
            continue
        name = (prof.get("name") or "").strip()
        if not name:
            continue
        desc = " ".join(
            (c.text or "").strip()
            for c in prof.iter()
            if _tag(c) == "characteristic" and c.text
        ).strip()
        if desc:
            out.setdefault(name, desc)
    return out


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
        # Per-faction unit records (faction -> slug -> record); avoids dropping
        # same-named units that exist in more than one faction (e.g. Mortar).
        self.by_faction: dict = {}
        # Weapon profiles (from every .cat and the .gst) by slug.
        self.weapons_by_slug: dict = {}
        # Special Rule profile descriptions by slug (army-specific abilities).
        self.rule_desc_by_slug: dict = {}
        # Lore of Magic name -> its spells, and every spell by slug.
        self.lores: dict = {}
        self.spells_by_slug: dict = {}
        self.factions: dict = {}
        self._load()

    def _load(self):
        if not os.path.isdir(self.cat_dir):
            print(f"[battlescribe] catalogue directory not found: {self.cat_dir}")
            return
        for filename in sorted(os.listdir(self.cat_dir)):
            path = os.path.join(self.cat_dir, filename)
            if filename.endswith(".gst"):
                # Game system holds the common (shared) weapons and the core
                # rulebook's special-rule descriptions.
                try:
                    for w in parse_weapons(path):
                        self.weapons_by_slug.setdefault(slugify(w["name"]), w)
                    for name, desc in _rule_descriptions(path).items():
                        self.rule_desc_by_slug.setdefault(slugify(name), desc)
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
                self.by_faction.setdefault(faction_name, {}).setdefault(slug, record)
            for record in profile_records:
                self.all_by_slug.setdefault(slugify(record["Model"]), record)
            for w in weapons:
                self.weapons_by_slug.setdefault(slugify(w["name"]), w)
            for name, desc in _rule_descriptions(path).items():
                self.rule_desc_by_slug.setdefault(slugify(name), desc)
            for lore, spells in _spell_lores(path).items():
                self.lores.setdefault(lore, spells)
                for s in spells:
                    self.spells_by_slug.setdefault(slugify(s["name"]), s)
        print(f"[battlescribe] loaded {len(self.by_slug)} models, "
              f"{len(self.weapons_by_slug)} weapons, "
              f"{len(self.spells_by_slug)} spells in {len(self.lores)} lores "
              f"across {len(self.factions)} factions")

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

    def spell(self, name: str):
        """Return a fresh game spell dict for a spell name, or None."""
        record = self.spells_by_slug.get(slugify(name))
        return copy.deepcopy(record) if record else None

    def lore(self, name: str):
        """Return a Lore of Magic's spells by lore name, or None."""
        for lore, spells in self.lores.items():
            if slugify(lore) == slugify(name):
                return copy.deepcopy(spells)
        return None

    def rule_description(self, name: str):
        """Return the catalogue description for a Special Rule name, or None.
        Only army-specific abilities are defined in the data; core rulebook
        keywords (Regeneration, Fear, ...) are not and return None."""
        return self.rule_desc_by_slug.get(slugify(name))

    def base_size(self, name: str):
        """Base size (width_mm, depth_mm) for a model, or None. Prefers the
        standalone/mount profile so a mount's true base isn't shadowed by a
        chariot component that reuses the same creature name."""
        slug = slugify(name)
        slug = NAME_ALIASES.get(slug, slug)
        for record in (self.all_by_slug.get(slug), self.by_slug.get(slug)):
            if record and record.get("base_width_mm") and record.get("base_depth_mm"):
                return (float(record["base_width_mm"]), float(record["base_depth_mm"]))
        return None

    def iter_models(self):
        """Yield (model_name, faction, characteristics) for every faction's models,
        so units that share a name across factions are all emitted."""
        for faction, records in self.by_faction.items():
            for record in records.values():
                yield record["Model"], faction, copy.deepcopy(record)


_catalogue: Catalogue | None = None


def get_catalogue(cat_dir: str = DEFAULT_CAT_DIR) -> Catalogue:
    """Return the shared Catalogue singleton, loading it on first use."""
    global _catalogue
    if _catalogue is None:
        _catalogue = Catalogue(cat_dir)
    return _catalogue
