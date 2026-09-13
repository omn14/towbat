# Battle March Implementation

Source: [General's Companion section](https://tow.whfb.app/battle-march),
reviewed 2026-09-13. This is not the older Settra's Fury ruleset.

## Current Scope: Revised 2026-09-13

This revision supersedes the original requirement to implement all seven stages
in full. Continue stages 1-4, the active portions of stage 5, and stage 7.

- Stop further magic-item implementation for now. Completing all 20 items is
  not required. Preserve existing implementations; unfinished item work,
  including the in-progress scroll changes, is paused rather than completed
  or reverted. Uncommitted work is not a verified milestone.
- Tournament features are deferred, including event rotation, two-list
  selection, event-wide secret-objective pools and tournament-only settings.
- Stage 6 narrative presets and their unique mechanics are deferred in full.
- Standalone secret objectives and all three random-happening tables remain
  in scope, as do Raid & Burn and baggage carts.
- Stage 7 still requires verification and milestone commits for active work.
  Deferred features are not completion gates and must not be marked implemented.

## Approved Stages

1. Preset and validation: versioned configuration, unchanged standard defaults,
   army violations reported without rewriting rosters.
2. Shared battlefield geometry: six maps and mirrors, full-base validation,
   deployment, Scouts, AI, movement, fleeing, pursuit, overlays and physical edges.
3. Complete standard setup: terrain policies, objectives, zone choice,
   alternating deployment, separate first-turn roll-off and choice.
4. Playable objectives and scoring: control, temporary landmark effects,
   per-player-turn awards, HUD/AI, idempotent saves, five complete rounds.
5. Active optional modules: Raid & Burn, baggage carts, ten private secret
  objectives for standalone games, and all three random-happening tables.
  Further magic-item implementation and tournament options are deferred.
6. Deferred narrative presets: Weisnicht Bridge, Bel-Cedas, Drazgog,
  Karak Ziflin, Sturdham and Gemelwald, including their unique mechanics.
7. Verification and milestones: both sizes, all maps/mirrors, rotated/dispersed
   bases, thresholds, terrain clearance, control ties, expiry, both turn orders,
  save/reload, offscreen visuals and isolated bounded tests; milestone commits
  for the revised active scope, without requiring deferred features.

## Implemented Foundation

- `config/battle_march.json` is the editable 500-point, 44 x 30, five-round preset.
- `battle_config.load_config` rejects malformed, unknown and duplicated fields.
- `battle_setup.resolve_setup` records map/objective/property dice using a seed.
- Saved configuration and resolved choices are independent copies. Restoring
  checks the recorded dice without RNG calls. Older saves clear custom geometry.
- `battlefield` owns playable edges and all six map shapes; the visual table
  remains 72 x 48. Units and terrain are not scaled.
- Runtime placement and boundary consumers use per-game geometry; standard
  games retain legacy defaults. Overlay lines and AI candidates use those zones.
- Mustering reports check available paid-point/category/starting-strength data;
  absent evaluated faction composition and restricted-option metadata is
  explicitly unverified. Imported roster category is preserved.
- Objective records use the verified diagrams: two at (0, +/-7.5), three at
  (-11, 0), (0, 0), (11, 0), or a central landmark. The earlier planning
  summary's horizontal two-trove description was incorrect.
- Shapely-backed terrain calculations cover rendered natural-feature rims,
  centre/opponent spacing, recommendation warnings, first-contact scatter,
  and minimum terrain translation clearing fixed objectives. Ordinary terrain
  selection and mouse-driven dimension/position previews are connected; river
  footprints and coordinated multi-feature relocation remain pending. Per-feature
  clearance now searches around fixed neighbours instead of rejecting an obstructed
  shortest move. Concave footprints, holes and board edges constrain the search;
  accepted positions are committed only after all features can be cleared.
- Offscreen 44 x 30 and 48 x 36 captures verify colored overlays and removal
  of the legacy rectangle. Real-scene saves restore geometry without RNG.
- Objective control now uses actual bases, joined Unit Strength, distance/US
  tie-breaks and explicit choices only for multiple controlled objectives.
  Both players score at each player-turn end; the saved ledger prevents repeats.
- Landmark MR(-2), Frenzy and Stubborn grants expire at the next player-turn
  end, preserve permanent sources and survive reload, including lost Frenzy.
- Frenzy now has majority psychology/no-Flee/no-Restraint checks, loss on a
  defeated combat round, conditional attacks with split-profile exclusions,
  follow-up turn tracking and compulsory declarations without Leadership.
  Supported loose form-ups use the same planner as actual charge moves.
- Battle March uses 50/25/25 General/standard/BSB bonuses and most VP wins.
  Standard-game scoring is unchanged.
- Numbered trove tokens and circular landmark terrain now render and restore.
  Shared movement and sight checks retain the landmark's round footprint and
  all-LOS prohibition. Both HUD orientations show control/property/objective VP,
  and battle results include objective award categories. Offscreen visual
  checks cover both objective layouts and both HUD orientations.
- A separate post-deployment roll-off and winner's first/second choice now
  determine the starting player. Pending choices can be saved and resumed
  without dice. Both starting orders have been tested through all ten queued
  player-turn scoring boundaries, including intermediate and final reloads.

## Starting The Current Implementation

Run `python game.py --battle-config` to open the native Panda3D configuration
screen with the default preset, or pass an existing JSON path after
`--battle-config`. Battle, Terrain, Objectives, Scoring, Muster and Modules tabs
edit the preset. The file field selects the load/save path; Load reads that file,
Save Config writes it without starting, and Start Game validates and saves it
before initializing the battle in the same window. Exit does not save edits.

An optional `--battle-seed 19` prepopulates the editable setup seed. An empty seed
uses a new random value when starting; this launch-only value is not written into
the versioned config schema. No armies, battle terrain or deployment tasks are
created until Start Game. Invalid entries and write failures leave the editor
open. Unsupported modules can be inspected and disabled in loaded presets but
cannot start a battle. Normal startup without `--battle-config` retains the
existing armies and terrain and bypasses this screen.

Setup reports army violations and unverified composition without rewriting units.
Players acknowledge their reports, select a shared terrain pool, roll off and
alternate terrain placement, then choose zones and independently roll first drop.
Accepted footprints, ownership, pool choices and dice survive saves. Pending zone
choices resume after reload without repeating terrain dice. Preview-only edits are
not committed to saves. Deployment and spell generation wait for setup completion.
Impossible objective-clearance layouts offer a terrain revision or setup pause.

This remains an incremental implementation, not completion of the approved plan.
Scattered placement is also connected: the winner places all features and the
loser chooses D3 to scatter; Hit/arrow dice and first-contact movement are saved.
Raid & Burn can be enabled in `optional_rules.secondary_objectives`: actual contact
during Remaining Moves begins a saved destruction attempt, restricts shooting and
casting, and awards 30 VP after the next own Start of Turn checks succeed.
Baggage Carts is also available as `baggage_carts` in that list. Setup chooses
one defender's cart or Guard Duty for both players. Each cart deploys as a single
60 x 100mm Heavy Chariot with its published driver/draft profiles and a distinct
miniature. Non-Combatant restrictions, enemy-zone edge escape, literal final
bonuses, saved ownership and once-only setup are connected. Escaped carts are
not destroyed and never earn a separate Dead or Fled award.
The committed magic-item milestone covers ten Battle March items: the existing Helm
of Courage and Banner of the Bold, plus Ranger's Glass, Warding Talisman, Padded
Hauberk, Wyrdstone Shard, Banner of Renown, Diestro's Blade, Skirmisher's Blade and
Thornspitter Stave. Inventory ownership, suppression and single-use state remain
the source of those effects. Additional uncommitted item work is paused. The
current item-module activation gate remains closed; this scope revision does not
enable unsupported items. Completing the remaining items or enabling the full
item module is no longer required for the active plan.
Runtime activation explicitly rejects the agreed time limit and other enabled
optional modules until their handlers are complete. Standard ordinary
terrain currently offers hills, woods and impassable buildings. Special features,
remaining terrain categories, objective-aware strategic AI, private objectives,
and random happenings remain active work in stages 1 and 3-5. Tournament features,
further magic-item work and stage 6 narrative scenarios are deferred.

## Current Terrain-Clearance Increment: 2026-09-13

General's Companion pp. 24-25: objective placement now finds the shortest legal
translation for each feature with its neighbours held fixed. Existing valid
clearance stays unchanged, including exact boundary distances. Circular clearance
regions use a conservative polygon approximation; the search does not substitute
convex hulls for concave terrain. Movement, no-move and impossible-placement logs
report the deciding distances or reason. Failed searches leave accepted setup
records and rendered terrain unchanged.

Verification: 120 configuration/geometry/controller tests pass in a 512 MiB
isolated service. Four added real-scene cases pass for buildings/hills on both
board sizes: legal placement, six-inch scatter, obstructed relocation, completed
setup and two reloads without further movement or dice. Offscreen screenshots
and terrain projection/pixel checks pass. The scene module's latest run has
57 passes and one failure in the previously recorded HUD layout NaN assertion;
that paused investigation remains untouched. This terrain-clearance milestone
does not complete stage 7; no full-suite pass is claimed.

LEFTOVER: jointly relocating several features to obtain a globally minimal layout
is not solved; the sequential fixed-neighbour search can still require revision.
River footprints, remaining terrain categories, special features, objective-aware
AI, standalone private objectives and random happenings remain active work.
The deferred item, event and narrative scope is unchanged.

## Native Configuration Screen: 2026-09-13

The launcher uses DirectGUI and the existing game theme inside the game's single
ShowBase instance, not Qt or a separate process. Typed numeric fields, enum menus,
checkboxes and scrolling tabs cover configurable preset values while preserving
source metadata and fixed schema fields. Saves use validated atomic replacement;
opening, resizing and cancelling do not rewrite files. Ordinary startup and
direct programmatic battle construction retain their existing behavior.

Verification: 15 native editor tests, one real startup handoff and 120 existing
configuration tests pass in isolated bounded services. The startup test edits
board dimensions, rounds and seed through the GUI, dispatches the Start Game
button event, verifies the saved JSON, reuses the same window and completes
pre-deployment setup. Desktop 1280 x 720, 800 x 600 and portrait 720 x 960 captures,
full-screen coverage, control/label bounds, popup sizing and event handling pass.
Invalid edits, unsupported modules, failed atomic writes, reload and duplicate
Start events are covered. The actual preset and user armies were not modified.

LEFTOVER: this is a configuration editor, not an implementation of the remaining
terrain, AI, private-objective or random-happening rules. The seed remains a
launch setting; saved battles retain their resolved setup as before. No roster
selection UI is added and no full-suite or paused HUD fix is claimed.

## Source Decisions

- Follow Battle March's explicit most-VP wording, not the core 100-VP margin.
- User confirmed literal cart scoring: an escaped cart earns its owner the
  25 VP escape bonus and its opponent the 25 VP "otherwise" award. Escape
  remains distinct from destruction/Dead or Fled.
- User confirmed written narrative rules over conflicting illustrations or
  cross-references: Gemelwald village goes in the defender half; the Weisnicht
  Bridge duel determines first turn. Retained for when stage 6 resumes.
- Narrative building-size feet/inches typos still need the applicable source
  verified if those deferred scenarios resume; do not stretch printed distances.

## Verification Constraints

Every terminal command activates `.venv`. Use `run_tests_isolated.py`; never
run several Panda3D test modules in one pytest process. The full-suite guard
stays at 1536 MiB plus 256 MiB available headroom. Render visual checks offscreen.
Never overwrite or stage user saves, local armies, startup army edits or the
paused HUD layout probe. No changes to Devil's Visit.