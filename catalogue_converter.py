"""
Stage 1 tool: convert BattleScribe catalogues (.cat) into the flat per-unit
characteristics JSON format used by army_units/<faction>/. It reuses the shared
parser in battlescribe.py so the offline export and the runtime loader stay in sync.

Usage:
    python catalogue_converter.py                      # convert the Orc & Goblin file
    python catalogue_converter.py path/to/File.cat ...
    python catalogue_converter.py --out-dir army_units_cat *.cat
"""

from __future__ import annotations

import argparse
import json
import os

from battlescribe import STAT_KEYS, parse_catalogue, slugify

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CAT = os.path.join(
    REPO_DIR, "Warhammer-The-Old-World", "Orc and Goblin Tribes.cat"
)
DEFAULT_OUT_DIR = os.path.join(REPO_DIR, "army_units_cat")


def convert_catalogue(cat_path: str, out_dir: str) -> None:
    """Parse one catalogue and write per-model characteristics JSON files."""
    faction_name, records = parse_catalogue(cat_path)
    faction_dir = os.path.join(out_dir, slugify(faction_name))
    os.makedirs(faction_dir, exist_ok=True)

    misses = [f"{r['Unit']} / {r['Model']}" for r in records if not r.get("Points")]

    # The same stat line can appear under several units; keep the first per file.
    written: set = set()
    for record in records:
        stem = slugify(record["Model"])
        filename = f"{stem}_characteristics.json"
        if filename in written:
            continue
        written.add(filename)
        out = {"Model": record["Model"]}
        for key in STAT_KEYS:
            out[key] = record.get(key, "")
        out["Points"] = record.get("Points", 0)
        for extra in ("Unit", "Troop Type", "Unit Size", "Special Rules", "Category"):
            out[extra] = record.get(extra)
        with open(os.path.join(faction_dir, filename), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=4, ensure_ascii=False)

    print(f"[{faction_name}] {len(written)} models -> {faction_dir}")
    if misses:
        print(f"  ({len(misses)} model(s) with no points cost found)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalogues", nargs="*", default=[DEFAULT_CAT],
                        help="Paths to .cat files (default: Orc and Goblin Tribes).")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                        help="Output directory (default: army_units_cat/).")
    args = parser.parse_args()

    for cat_path in (args.catalogues or [DEFAULT_CAT]):
        convert_catalogue(cat_path, args.out_dir)


if __name__ == "__main__":
    main()
