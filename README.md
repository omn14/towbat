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
chariot equipment to individual crew happens in the runtime ownership layer.

An exported lore is no longer a ready-to-cast spellbook. The High Elf roster's
Level 2 Mage retains ten options but no generated known spells during import.
At runtime, Silvery Wand adds one known spell and the pre-deployment generator
offers the normal signature substitution. A selected upgrade whose name matches a Spell profile
is treated as an explicit spell choice, but import does not certify that the
number or combination of choices is legal. Unknown item spells remain metadata,
not Wizard levels or castable spells. The existing Ruby Ring bound-Fireball
handler remains supported separately from the general item registry.

These records survive army-list JSON serialization. The live loader applies
command/profile ownership and installs purchased item instances separately.
Import does not modify its source file or apply prose-described magic-item effects.

### Magic Item Inventory

[magic_items.py](magic_items.py) separates immutable definitions, purchased
instances and source-tagged effect contributions. Instances keep stable unit,
selection and copy identities, structural owners, per-ability use counters and
whole-item suppression. Repeated installation preserves use state. Spending an
ability does not disable unrelated passive effects.

`active_effects` is a read-only query boundary for explicit coded definitions.
It follows living bearers, command losses and character joins/retirement without
baking item grants into native rules or baseline profiles. `activate_ability`
is the shared eligibility/commit gate for player and AI choices;
`disable_item` suppresses registry contributions for the rest of the battle.
Consumers apply and log supported outcomes at their owning rule implementations.
This is not a second combat or reroll engine.

The three selected items now have coded effects:

- Silvery Wand adds one known spell without changing Wizard level or cast limit.
- Helm of Courage improves its bearer's armour by one and offers one Break
	reroll per battle, also usable by a joined unit. Spending it leaves the armour
	bonus intact; an available BSB reroll takes priority and preserves the item.
- The Banner of the Bold grants Veteran to its unit and joined characters.
	Losing or disabling the standard removes only that source, not native Veteran.

Vaul's Unmaking targeting, other item effects, general equipment legality and
adaptation of the existing Ruby Ring handler remain pending. The selected
High Magic pool now has coded Shield of Saphery, Fury of Khaine, Walk Between
Worlds and Corporeal Unmaking effects. Corporeal Unmaking's combat timing and
challenge handling remain partial. The other six High Magic/Saphery effects
remain unimplemented; generating those spells does not make their wording executable.

Battle saves contain explicit `magic_item_inventory` records. Recreating a
bearer restores spent/disabled state; old saves without this field load an empty
inventory even if raw metadata mentions items. The unit card's detail arrows
expose item names, bearers, support status and ability-use state, including items
carried by a joined character.

### Spell Generation

[spell_generation.py](spell_generation.py) runs before deployment (Rulebook
pp. 106, 319). Each player can order their Wizards; dice generate distinct
numbered spells, rerolling duplicates. One optional substitution allows the
normal signature or one Lore of Saphery alternative (Forces of Fantasy p. 186).
The AI keeps its generated spells. Incomplete or ambiguous numbered pools remain
pending with a diagnostic instead of silently generating an invalid spellbook.

The Level 2 Mage with Silvery Wand knows three spells, not all ten options and
not a third casting slot. Explicit known spells and Bound spells are preserved.
Intermediate rolls and a pending substitution live in saved roster metadata;
the final spellbook uses the existing save path. Reloading does not roll again.
Finish an open generation choice before loading another battle; saving during
the choice is supported. Imported replacement armies generate on their first
deployment action. Long choice labels fit inside the existing fixed controls.

Both generation choices include a spell reference. Select a spell in its list
to read the imported type, casting value, range, phase and full effect text
without choosing or replacing it. The list distinguishes generated spells,
numbered spells not generated, already-known spells and signature alternatives.
Hovering a signature or replacement answer also previews its profile; only
clicking an answer commits the choice. Long effects scroll within the reader,
and the selected signature remains visible when resuming a saved replacement.
Missing roster text and unimplemented engine effects are labelled explicitly.

Startup uses the converted lists in
[strategy_armies/my_army_he.json](strategy_armies/my_army_he.json) and
[strategy_armies/my_arm_chaosy.json](strategy_armies/my_arm_chaosy.json), not the
original roster exports. These lists have been refreshed with the current
importer. Older conversions can incorrectly contain ten known spells and no
item records, so they need reimporting; updating engine code alone cannot recover
the missing purchases. Old battle saves likewise retain their saved state.
Select a unit and use the small up/down arrows beside the HUD's two detail lines
to see item names, bearers and use status. A joined character's inventory also
appears on its host unit's card; the Banner is on the Chaos Warriors' card.

### Casting With Joined Wizards

Select the host unit and press `C` in the spell's applicable phase. A sole
eligible joined Wizard opens its spell menu directly; if both the host and its
character can cast (including Bound spells), choose the caster first. The
Strategy-phase selection path also finds joined casters after the Command window.
Spells and attempts remain on their original bearer. Casting or cancelling
returns selection to the host without detaching the character. Host fleeing,
marching and combat restrictions still apply, and ranged aiming uses the
joined model's world position and facing.

### Magic Lifetimes and Dispelling

The High Elf Mage can now use Lileath's Blessing to reroll one failed Casting
roll per turn. Declining retains the use; a natural Miscast cannot be rerolled,
and a replacement roll can Miscast. The reroll spends no extra casting attempt.
Its usage survives reload and remains on the Mage when joined to a unit.

Successful spells offer the defending player legal Wizardly choices, a Fated
dispel, or Pass. Chaos can attempt one Fated dispel per turn despite having no
Wizard. Joined Wizards, Dispel range and engaged/fleeing restrictions are checked.
Double 6 unbinds, ties fail, and immediate perfect invocations bypass dispelling.
When advancing out of Strategy, surviving enemy Remains in Play spells offer
Conjuration dispels against their minimum casting value. Fated use is shared
with immediate dispels. Normal main-phase advance also offers each caster the
choice to keep or end their own Remains in Play spells.

Existing Battle Magic wards/hexes now have explicit expiry and independent
source ownership. Recasting a RIP spell ends its old effect before the roll;
caster removal ends RIP spells. Reload restores effects and dispel usage without
duplicate grants. Saving/loading and phase advance wait for in-flight magic.

This is a lifecycle foundation, not a complete magic engine. Shield's Enchantment
replacement and the two High Magic unit grants below are implemented; dynamic
Self/area auras beyond Walk Between Worlds and every internal subphase's voluntary-ending window remain
unfinished. Miscast/Outclassed damage still requires manual resolution and is
logged as such. See the exact scope in
[SPECIAL_RULES_CHECKLIST.md](SPECIAL_RULES_CHECKLIST.md).

```bash
source .venv/bin/activate && python run_tests_isolated.py --memory-mb 768 tests/test_lileaths_blessing.py tests/test_spell_effects.py tests/test_dispelling.py tests/test_magic_lifecycle_scene.py
```

### Shield of Saphery and Fury of Khaine

Both use the existing casting menu when known, and expire at the **end of the
current turn**, not the caster's next Start of Turn (Rulebook p. 329).

- Shield of Saphery: 8+, range 18", friendly unengaged unit. Grants a 5+ Ward
	against wounds; a better existing Ward still wins. A successful, undispelled
	cast removes earlier Enchantments on the target, not Hexes or native rules.
- Fury of Khaine: 9+, range 12", friendly unit, including one engaged in combat.
	Grants Extra Attacks (+1) to the applicable main, champion, mount, crew and
	joined-character profiles. Identical grants do not stack. Supporting attacks
	remain limited; shooting volume is unchanged.

Targets need the caster's vision arc and base-to-base range, but not line of
sight. The caster's own unit is selectable. Joined Wizards use world position
and facing. Character joins spread either grant without renewing its duration;
received grants remain until expiry after separation. Shield removes them only
from its current target's members, preserving separate recipients.

Temporary attacks do not rewrite profile characteristics, so combat resets do
not erase Fury. Saves restore recipients and expiry without duplicate bonuses,
including surviving recipients after the original target/caster dies. Numeric
grant and combat-save outcomes appear in the rule log. Target geometry uses
unit footprints; dispersed formations and challenge-specific targeting still
need broader verification. Six other lore effects remain unimplemented.

```bash
source .venv/bin/activate && python run_tests_isolated.py --memory-mb 768 tests/test_high_magic.py tests/test_high_magic_scene.py
```

### Walk Between Worlds

The Mage can cast this **10+ Self-range Conveyance** through the normal Movement
phase spell menu. It grants the caster and current joined unit **Ethereal and
Reserve Move until the caster's next Start of Turn** (Rulebook p. 329). It is not
an Enchantment, so Shield of Saphery does not remove it.

Ethereal prevents non-magical attack wounds, including Killing Blow and mundane
Impact Hits or artillery. Magical attacks and spells still wound normally.
Ensorcelled hand weapons qualify; mundane Chaos lances do not. Movement ignores
terrain penalties and dangerous-terrain tests. Impassable terrain may be crossed,
but not occupied at the end of a move; other units still block movement.
Magical vortex damage remains magical. Ethereal and non-Ethereal characters and
units cannot join each other.

After shooting resolves, End Phase opens **Reserve Move** when eligible units
exist. Select a unit and use the existing movement controls; End Phase declines
unused moves and proceeds to Combat. A majority must have the rule. Units that
charged, attempted a charge, marched or fled during Movement cannot use it;
engaged, fleeing and newly rallied units are excluded. This is a separate Basic
Movement allowance, not a march or charge, and cannot reopen shooting or casting.
Normal redress and the Dragon Princes' free Drilled redress are supported. The
AI currently declines optional Reserve Moves; after reloading directly into its
Reserve window, End Phase may be needed to continue.

Reload preserves the extra movement budget and completed moves. Self grants
follow the Mage's host: leaving or retiring removes the host's benefit, returning
to the fighting rank restores it while the spell lasts, and caster removal ends
this Self-dependent effect. Native rules remain intact.

The tested core covers the actual lone and joined Mage, exact-M Reserve Moves
through a house, illegal endpoints, combat damage, expiry and repeated reload.
Complex precomputed charge routes may still conservatively block Ethereal
terrain passage. Impassable endpoints in other charge/reaction/pursuit paths,
detailed swept wheel/formation geometry and broader Self/aura spells remain
unfinished. See the checklist for those limits.

```bash
source .venv/bin/activate && python run_tests_isolated.py --memory-mb 768 tests/test_walk_between_worlds.py tests/test_walk_between_worlds_scene.py
```

### Corporeal Unmaking

The Mage's **8+ Combat-range Assailment** now inflicts **D3 automatic magical S5
hits**, with **no armour or Regeneration saves**; Ward saves work normally
(Rulebook p. 329). It uses the spell's Strength, not the caster's weapon or
special attacks. Ethereal does not prevent these magical wounds. Multi-wound
targets lose wounds rather than whole models, and rule logs report the hits,
wounds and Ward saves.

When known, select the Mage or its host during Combat and use the existing spell
menu before starting the fight. Targets are the unit's engaged enemies, including
for a joined Mage; shooting line of sight is not required. Retired Wizards,
already-fought units and casts during an active combat resolution are rejected.
Cancelled, failed and dispelled casts cause no damage. Successful wounds count
toward the caster's unit's combat result once, capped at the target's remaining
Wounds. Save/reload preserves casualties, casting usage and unspent combat credit.
No shooting heavy-casualty Panic test is requested by this Assailment.

**Partial combat integration:** the existing spell menu resolves before the
fight, not at the Wizard's Initiative as p. 108 requires. Automatic casting
opportunities for both sides, isolated challenge allocation and spell-only
wipeout/post-combat routing remain unfinished. Casts by a duelling Wizard or
against a unit in an active challenge are refused with a diagnostic, not allowed
to spill wounds into protected models. This does not change Hammerhand's existing
behavior or complete the shared Assailment engine.

```bash
source .venv/bin/activate && python run_tests_isolated.py --memory-mb 768 tests/test_corporeal_unmaking.py tests/test_corporeal_unmaking_scene.py
```

### High Elf and Chaos Faction Effects

The loaded rosters now use Dragon/Chaos Armour Ward saves, first-round Elven
Reflexes on the correct rider or crew profile, Ithilmar hand-weapon rerolls,
Ensorcelled hand-weapon AP, contextual Valour/Mark Panic rerolls, and Ithilmar
Barding terrain rerolls. Ward sources use the best value, separate from armour.
The Skycutter's crew gains Reflexes, not its Roc; joined characters do not borrow
barding. Rule logs include deciding rolls and relevant reasons for not applying.
Native rules survive save/reload independently of magic-item suppression.

Mark's live Fear/Terror tests, magical-damage defences needed by Ensorcelled
Weapons, and Wizard armour exceptions remain unfinished. This does not add the
missing High Magic spell effects or complete the matchup. Per-rule status and
verification caveats are in [SPECIAL_RULES_CHECKLIST.md](SPECIAL_RULES_CHECKLIST.md).

```bash
source .venv/bin/activate && python run_tests_isolated.py --memory-mb 768 tests/test_faction_rules.py tests/test_faction_rules_scene.py
```

### First Charge

Silver Helms, Dragon Princes and Chaos Knights now track their first charge
attempt. A failed first attempt spends the benefit; successful contact disrupts
the target's rank bonus until that turn's Combat phase ends. Disruption is
separate from terrain and flank effects. Pursuit/overrun contacts that count as
charging apply it in the turn when that combat is fought, including next-turn
combat. Rule logs report application, nonapplication and expiry.

New saves retain attempts and active/deferred disruption. Old saves without
charge history conservatively treat First Charge as spent: start a new battle
to use it with accurate history. Pending state is saved, but suspended charge
animations are not resumed by this feature.

This is the First Charge core of the movement work, not completion of the
matchup. The selected cavalry now also have Impetuous and Drilled support;
detailed formation boundaries are in
[SPECIAL_RULES_CHECKLIST.md](SPECIAL_RULES_CHECKLIST.md).

```bash
source .venv/bin/activate && python run_tests_isolated.py --memory-mb 768 tests/test_first_charge.py tests/test_first_charge_scene.py tests/test_shieldwall_scene.py
```

### Counter Charge

Chaos Knights and Dragon Princes can now Counter Charge eligible frontal charges
in the formed-unit flow. The defender's reaction menu includes the
option; AI defenders select it automatically when eligible. Distance is measured
from the charger's original position, not its tentative contact position.
Too-close, wrong-arc/type, fleeing, engaged and already-used cases are logged.

The defender pivots and advances D3+1", with no Swiftstride bonus to that roll.
The charger then rolls and moves against the defender's new position. Contact
gives both units charging benefits, including eligible First Charge effects.
Once-per-turn use survives save/reload. Charging into an enemy no longer produces
the erroneous marching log or sets the marching flag.

Drilled defenders can freely redress before their Counter Charge. This remains
partial: wider Marching Column/reaction and loose-formation interactions need
further work; see the checklist's explicit
limitations. Loading a save does not resume an in-flight reaction animation.

```bash
source .venv/bin/activate && python run_tests_isolated.py --memory-mb 768 tests/test_counter_charge.py tests/test_counter_charge_scene.py
```

### Charge Declarations

Movement now starts with charge declarations. Confirm each charge using the
existing movement preview; the charger stays at its original position until
resolution. Click **Resolve Charges** when declarations are complete, including
when there are no charges. Ordinary movement is unavailable until then.

Defenders choose their reactions after all declarations. Counter Charge and
Stand & Shoot select one incoming charger; Flee runs once away from the strongest
charger, with ties chosen randomly. Reactions finish before the active player
chooses which charge to move next. Formed routes and contact arcs are recalculated
against the moved defender, and existing combat links survive later charges.
Failed planned charges move their Charge roll without adding Movement.

Both AI implementations close declarations before their remaining moves. EnhancedAI
currently attempts declarations against each eligible unit's nearest enemy;
this is not a multi-charge tactical search.

Declaration-stage saves retain queued units, original poses and selected loose
target indices without repeating First Charge attempts. Saves during active or
interrupted resolution are refused. Reload a declaration-stage save after an
interruption; snapshots do not resume animation tasks. Legacy Movement saves
without a recorded stage resume in Remaining Moves to avoid inventing declarations.

Remaining limits include redirection, simultaneous frontage maximisation,
complex flying/obstructed routes, and multiple form-ups or fleeing loose targets.
Unsupported reserved loose-target routes are logged and spent rather than silently
retargeted.

```bash
source .venv/bin/activate && python run_tests_isolated.py --memory-mb 768 tests/test_charge_declarations.py tests/test_charge_declarations_scene.py
```

### Drilled and Impetuous

Dragon Princes use the amended Impetuous Leadership test, not the old 4+ roll
(Rulebook p. 172). Resolve Charges tests eligible units before reactions; failure
adds a compulsory charge, with a target choice when several legal targets exist.
An already-declared unit still tests, because failure can require Drilled redress.
Shared Leadership and Veteran rerolls apply. Both human and AI paths use this
resolver, and declaration-stage reloads do not duplicate charge attempts.

Drilled offers one free redress of up to five front-rank models before a committed
Remaining Move, Counter Charge, Giving Ground, or queued charge move (pp. 125,
167; FAQ v1.5.3). The front rank remains anchored; normal movement and manoeuvre
allowances are preserved. Charge dice precede the redress, then the route is
rebuilt. Candidate formations must fit other units, impassable terrain and the
board, and cannot move a model more than 2M during the manoeuvre.

A Marching Column has no rank bonus and marches at 3M (p. 101). It can declare a
charge but cannot make the charge move until it leaves column. A compulsory
Drilled charger must adopt a fitting Combat Order; if it cannot, the charge fails
without movement. Voluntary chargers may decline redress. Drilled's existing
Enemy Sighted exemption still applies.

Pending Drilled movement blocks other movement input, phase advance and save/load.
Ordinary redress cannot be used during declarations to evade Impetuous.
The 25 focused offscreen scenarios cover the actual three Dragon Princes against
Chaos Knights, including Counter Charge, First Charge, human/AI choices and reload.
This is not a complete-matchup verification. Loose Impetuous chargers, advanced
blocked/flight routes, and free-redress hooks for other movement sources remain
explicit checklist limitations.

```bash
source .venv/bin/activate && python run_tests_isolated.py --memory-mb 768 tests/test_drilled_impetuous_scene.py
```

Focused inventory checks use the memory-bounded runner:

```bash
source .venv/bin/activate && python run_tests_isolated.py tests/test_selected_items.py tests/test_spell_generation.py tests/test_magic_items.py tests/test_magic_item_scene.py tests/test_persistence.py tests/test_choice_layout.py
```

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
