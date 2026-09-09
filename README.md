# towbat

## Roster Imports

[roster_importer.py](roster_importer.py) converts selected NewRecruit/BattleScribe
JSON data into an army list. Existing flat combat fields remain available;
additional fields preserve the information needed by future item/command support:

| Field | Meaning |
| --- | --- |
| `roster_source` | Original roster metadata at army level; force/catalogue metadata on each unit |
| `roster_selections` | Selected nodes, with original IDs, profiles, costs, categories and associations; `ref`, `parent_ref` and `owner_ref` preserve their hierarchy |
| `magic_items` | Selected item records, category, definition ID, quantity, points and owner reference; not applied effects |
| `command` | Command roles, promoted-model references and original profiles, including champion characteristics |
| `equipment` | Weapon/ordinary-armour profiles with structural owner references, including mounts and model-profile upgrades |
| `spell_sources` | Original spell profiles and their selected source, classified as `pool`, `selected` or `item` |
| `spell_pool` | Exported ordinary spell options that were not explicitly selected as known spells |
| `spells` | Explicitly selected ordinary spell upgrades plus supported bound spells |
| `spell_generation_pending` | Unselected ordinary spell options remain; generation/substitution is not performed by import |

References resolve against the unit's `roster_selections`. They use scoped
selection IDs, with positional fallbacks where IDs are absent. Reimporting an
unchanged export is deterministic. Available `selectionEntries` and
`selectionEntryGroups`, and selections with zero quantity, grant nothing and
are excluded from the selected-node records. Command upgrades do not add bodies.

Ownership describes what the export actually specifies, not guessed combat
allocation. In particular, Skycutter bows/spears exported on the containing
model remain there; Wicked Claws retain their Roc owner. Assigning shared
chariot equipment to individual crew is still runtime work.

An exported lore is no longer a ready-to-cast spellbook. The High Elf roster's
Level 2 Mage retains ten options but no generated known spells. Silvery Wand
is preserved as item data; its extra spell and generation/substitution UI are
not implemented here. A selected upgrade whose name matches a Spell profile
is treated as an explicit spell choice, but import does not certify that the
number or combination of choices is legal. Unknown item spells remain metadata,
not Wizard levels or castable spells. The existing Ruby Ring bound-Fireball
handler remains supported separately from the planned general item system.

These records survive army-list JSON serialization. This is **not** battle-save
inventory support: item activation, command effects, runtime owners and their
persistence remain separate checklist tasks. Import does not modify its source
file or apply prose-described magic-item effects.

Focused import checks use constructed fixtures, not the local army exports:

```bash
source .venv/bin/activate && python -m pytest tests/test_roster_importer.py tests/test_spells.py tests/test_bound_spells.py -q
```

## Save Profiles

Saves keep the live `characteristics` separately from `base_characteristics`,
the roster-adjusted profile used by combat resets. A temporary stat bonus must
not become permanent just because the game was saved while it was active.
Mounts, crew and beasts carry the same separate snapshots in `profile_parts`.
Permanent roster rule selections update the baseline without copying temporary
numeric changes into it.

Supported ongoing spells are stored in `spells_in_play` with their target and
remaining duration. Their runtime effects are restored once, separately from
the profile snapshots; resetting combat stats does not expire those spells.
This does not add persistence support for non-catalogue demonstration spells
such as Devil's Visit or introduce a general modifier engine.

Older saves without a baseline retain their saved profile as the baseline.
The loader does not guess which old values were temporary or replace valid
custom stats with current catalogue values. Entirely statless records, such as
the formerly unresolved Lothern Skycutter, are repaired from the catalogue while
preserving saved rules and battle state. Loading does not rewrite the save file.

## Testing

Activate the project environment for every command:

```bash
source .venv/bin/activate && python -m pytest tests/test_persistence.py tests/test_special_rules.py -q
```

Use calculation and serialization tests while editing. Add the relevant real-game
scene tests when changing live behavior or save/load integration:

```bash
source .venv/bin/activate && python -m pytest tests/test_persistence.py tests/test_skycutter_scene.py -q --durations=8
```

Scene tests start Panda3D offscreen and should reuse a module-scoped scene with
a saved baseline restored between cases. In a measured persistence run, scene
setup took about 8 seconds after catalogue initialization, while the individual
new save/load cases took about 0.5-0.8 seconds. Do not pay for a new game instance
in every test. Use `--durations` to measure before changing fixture isolation.

Run the complete suite using separate, memory-bounded processes per test module:

```bash
source .venv/bin/activate && python run_tests_isolated.py
```

On this 8 GB machine, a monolithic scene-test run caused `systemd-oomd` to kill
the entire VS Code process group. The isolated runner uses Linux systemd user
services outside that group, sequentially, with a 1536 MiB RAM limit, no swap,
and a 180-second timeout per module. Each pytest process exits before the next
starts, releasing its native scene resources. A module that exceeds its limit
fails instead of allowing unrestricted memory growth. Isolation cannot protect
against unrelated system-wide pressure; the runner also checks available memory
before each module and refuses to start without 512 MiB extra headroom.
The Skirmisher scene module measured 1155 MiB peak RSS in isolation; a 1024 MiB
cap stopped it near the end. The failed bounded run killed only its test service,
and the same 82 tests passed under the 1536 MiB cap.

Use the same runner for individual scene modules or a subset:

```bash
source .venv/bin/activate && python run_tests_isolated.py tests/test_skirmish_scene.py tests/test_skycutter_scene.py
```

Reports are written under `.pytest_cache/isolated/<timestamp>-<pid>/` by default:
an incremental `summary.json` plus each module's `pytest.log`, `junit.xml`, and
GNU time `memory.json`. Peak RSS is reported per pytest process; it is not a
measurement of total GPU memory or the entire desktop. Incomplete/resource-killed
runs remain explicitly incomplete. `--output` takes a new results directory;
`--memory-mb` and `--timeout` override the limits. Do not raise the memory cap
without first checking headroom. There is no unbounded fallback if user services
are unavailable. Reuse completed results across interruptions rather than
restarting the full suite.

Report existing failures separately; do not repeatedly run the full suite to
reconfirm them. Automated visual checks must render offscreen rather than
starting the interactive game window.
