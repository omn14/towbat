# Skirmishers — Implementation Plan

> Status: **in progress**. The loose-movement foundation and optional formation
> editor are implemented and tested. This is not complete Skirmish Formation:
> two loose units now form fighting ranks in sequence, and loose chargers form
> against an unengaged formed target. Compact fleeing units wait for successful
> Rally before separating (requested house rule). Per-model sight, other charge
> pairings, combat scoring and constrained separation remain outstanding.
> The historical phases below describe the earlier baseline.
> Note: a unit's skirmisher status comes from the army list's `special_rules`
> (unit-level rule), not the base catalogue model profile (e.g. Cathay
> "Peasant Soldier" has no Skirmishers on its model profile).

Rule (catalogue): *"A unit consisting of models with this special rule may adopt
a Skirmish formation."* Source rules: <https://tow.whfb.app/unusual-formations/skirmish-formation>
(Rulebook pp. 184–187; Official FAQ & Errata v1.5.3 takes precedence).

## Current milestone: positions, casualties and movement

- Ordinary models have saved IDs and local base positions. A deterministic,
  non-touching layout replaces jitter; edge-to-edge adjacency must form one
  connected group within 1", not several independently connected pairs (p. 184).
- Ordinary casualties preserve coherency and surviving positions. Removal uses
  all rendered bases even when a combat caller has already reduced the logical
  count. A slain bridge character can be replaced by a legally removable ordinary
  model (FAQ, Unusual Formations); the last surviving character remains on-table.
- Save/load restores individual positions and joined-character placement. Older
  saves regenerate a coherent layout. Generic rank/footprint rebuilding no longer
  silently collapses a loose formation or shifts its surviving bases.
- Normal group movement and optional adjustment share per-base validation and
  commit. Maximum individual travel spends the movement allowance, including
  reshaping, with a float32 tolerance at the march boundary (pp. 123, 185).
- The contextual **Adjust formation** control offers group/model dragging,
  models-across, gap and angle, actual-base ghosts, refusal reasons, movement cost,
  Confirm and Cancel. Preview never mutates live positions or rolls dice. Cancel
  retains movement; confirm spends it once; phase advance is blocked while editing.
- The normal cursor now labels **CHARGE: target** with cyan squares, **MOVE**
  with green squares, **MARCH** with amber squares, or **BLOCKED** with a reason.
  It probes the same contact and first-turn restrictions as the click action;
  CHARGE means a declaration prompt, not a guaranteed successful charge roll.
  Ordinary human right-clicks stage the destination in Confirm/Cancel without
  spending movement. Cancel (or another right-click) restores the cursor preview.
  Charge clicks retain Yes/No. AI/direct movement calls do not need UI confirmation.
  Verified contact gets a 0.0001" sweep-boundary tolerance to avoid a rounding gap;
  declining a charge clears the charging flag as well as restoring position.
  Indicators clear on selection/phase changes, editor opening and a missed ray.
- Crossed difficult terrain affects movement. Dangerous tests count individual
  bases per feature; joined-character wounds use character removal. Flying models
  suffer landing terrain, not features overflown (pp. 170, 269). Other units and
  terrain currently use conservative rectangular obstacles.
- Corrections during validation: ordinary deaths previously ranked survivors;
  saved layouts were missing; joined bodies could be reattached to physics on
  load; a pre-reduced count hid casualty candidates; lethal character terrain
  damage used host removal; floating-point error could turn exactly M into a
  march; mode labels initially extended outside the panel.

Verification: 349 tests plus 75 geometry subtests passed across Skirmishers,
characters, persistence, terrain, Fly, psychology, Scouts and Vanguard. An existing
Vanguard screenshot test needed Windows filename normalization in the test runner;
its source was not changed. The new scene tests use normalized paths directly.
Charge-indicator follow-up: 94 tests passed across the Skirmishers, Scouts,
Vanguard, Quick Shot and choice-layout modules, including 32 Skirmishers scene
tests. Pixel checks verify all cyan charge-square edges; screenshots were inspected
at 1280x720 and 800x600.

## Ground-range and legal-destination follow-up

- Shared ground/terrain shader bands show remaining normal movement in green,
  remaining march distance in amber, and maximum charge reach with a dashed cyan
  boundary. Cyan hatching marks reach beyond the march allowance. Charge reach
  remains independent when 2M exceeds M + 6, and includes Swiftstride's existing
  +3 maximum (pp. 121, 123, 185).
- Range inputs use the preview's movement allowance and spent movement. The cyan
  boundary disappears for charge restrictions, including Scouts/Vanguard and
  cannotChargeThisTurn. Selection/phase changes, missing ground hits, the editor
  and unavailable unit states clear the overlay and ordinary destination squares.
- Empty-ground destinations clamp to a legal straight-line endpoint through the
  shared per-model move validator: remaining allowance, battlefield edges,
  impassable terrain, units and enemy clearance. Clamping is read-only and never
  spends movement or rolls terrain dice; confirmation still validates again.
  Refused charge destinations have no squares and retain their refusal message.
- Corrected during validation: extra charge reach could previously display an
  impossible ordinary move; clamping must not turn an aimed out-of-range enemy
  into a march. Shader defaults must live on copied/baked surfaces, not only the
  scene root. Offscreen render tests use dedicated buffers at each resolution.

Verification: 761 tests and 82 subtests passed. Pixel comparisons at 1280x720 and
800x600 verify green, amber and cyan bands and unchanged pixels beyond maximum
reach; both screenshots were inspected. Tests also cover terrain shader inputs
and clearing, remaining movement, fast units, Swiftstride, charge restrictions,
board/impassable/clearance clamping and unchanged live movement state.

LEFTOVER: circular shading is a range guide using the current preview allowance,
not an obstacle-aware legal-destination map or a promise of charge success. Exact
ordinary destination squares use the existing conservative straight-path validator;
detours, irregular obstacles and the charge-planning limitations below remain.
Vanguard and pursuit retain their existing phase-specific indicators.

## Charge-range and alignment follow-up

- Charge plotting and direct commits now enforce current M + 6, with the existing
  Swiftstride +3 maximum (p. 121). M3 Dwarfs can declare up to 9", not just their
  6" march distance. Extra charge reach cannot be spent on ordinary movement.
  The cyan status names the target and shows distance/maximum; the roll-needed
  readout tolerates the contact epsilon instead of asking for an impossible 7.
- Two loose Skirmisher units without joined characters use a contact-anchored
  planner (p. 187). The closest model moves first, the other chargers form around
  it, then the defenders form an opposing rank within M. Frontage is determined
  by reachable slots rather than the saved files setting. Rear ranks retain
  reachable models; unassigned models are coherency casualties (FAQ v1.5.3).
  Planned model identities, facing and positions survive combat entry/save/load.
- The first model's planned distance drives the two-Skirmisher range check and
  success depends on the actual roll. A failed Skirmisher charge advances only
  the Charge-roll result, not M plus that roll (p. 121). Unsupported pairings
  log that legacy alignment is being used; previews remain silent.
- Corrections during validation: an exact-range contact needed a smaller sweep
  epsilon; a failed individual plan must not fall back to successful rank snapping;
  direct calls previously bypassed the range gate; float32 save comparisons need
  tolerance. Tests now refresh Bullet after teleports, and Scout eligibility
  fixtures use legal charge distances instead of 31.85" direct jumps.

Verification: 612 tests and 75 geometry subtests passed across the affected
movement, combat-rule, character, persistence and scene modules, including 192
uneven-frontage combinations, actual charge success/failure and form-up casualties.
Real intervals were stepped offscreen; chargers moved before defenders. Alignment
renders were checked at 1280x720 and 800x600, with model pixels verified at 800x600.
The existing Vanguard screenshot test still uses runner-only Windows path
normalization. No new diagnostics in the charge implementation or its tests.

### Formed-target rear-charge correction

Source: <https://tow.whfb.app/unusual-formations/skirmishers-and-charging>
(Rulebook p. 186). The reported column-to-rear charge still used whole-body
translation followed by compact rank snapping, leaving the new fighting rank
short of the target even though both units entered combat.

- Loose Skirmishers without joined characters now form against an unengaged
  formed target's actual base edge. The closest charger moves first, reachable
  models form a touching rank and the remainder use rear slots within their roll.
  The formed defender never translates, rotates or reforms during this sequence.
- Preview and resolution share the first-contact distance. The charged face is
  selected from the starting position, and rear/flank combat bookkeeping is
  preserved instead of relabelling every planned charge as front contact.
- Frontage is limited to actual enemy-base contact. The first charger need not
  be centred against a formed target; candidate slots are compared for surviving
  models and travel. Equal closest-model distances use a float tolerance so
  rotating the same fixture does not change its first charger.

Verification: 627 tests plus 75 geometry subtests passed. The original gap was
reproduced before fixing it. Scene regressions cover rear charges at 0, 37 and
180 degrees, unchanged defenders, model identity and each charger's travel.
Twelve pure fixtures cover every face, frontage limits, rear ranks and insufficient
reach. Real charge intervals completed offscreen at 1280x720 and 800x600; both
renders were inspected and model pixels/contact geometry checked.

LEFTOVER: the new planner assumes uniform bases within each unit, direct centre
translations to a target edge, and no joined characters. It does not search all
legal contact orientations, alternative rank arrangements or individual detours;
forced losses are based on its chosen slots, not an exhaustive formation search.
Path crossing/obstacles and per-model charge terrain tests still need integration.
Formed-target planning uses a single outer face; stepped/incomplete rear ranks and
arc-straddling declarations need further adjudication. Formed units charging loose
defenders, already-engaged targets, fleeing/pursuing units, joined/command models
and multiple charges retain the older alignment path. These are not complete
implementations of pp. 186-187. Existing saved misaligned combats are not migrated.

### Compact fleeing units and Rally

Source: <https://tow.whfb.app/unusual-formations/skirmishers-in-combat>
(Rulebook p. 185; no overriding exception found in Official FAQ v1.5.3).
The printed rule keeps Skirmishers compact until unengaged at the end of a Combat
phase, then separates models by the smallest amount possible.

- **Requested house rule:** compact fleeing units stay compact across phase
  changes and failed Rally tests, separating only after successful Rally. This
  deliberately differs from the printed end-of-Combat-phase timing for fleeing
  units. Normal Rally and Rallying Cry share the success hook, after any human
  Rally reform; AI uses the same separation operation.
- Non-fleeing, unengaged compact units separate at Combat-phase end. Engaged
  units remain compact, including units caught again before phase end. Temporary
  transitions to spell resolution do not trigger separation.
- Corrected: leaving the InCombat FSM state previously spread immediately and
  regenerated a blob. Separation now uses current base positions, preserves IDs,
  rank order and world facing, and includes an attached character. A tiny uniform
  expansion gives touching bases a numerical gap of at least 0.0001", rather than
  granting a free reform. Already separated bases do not move.
- Compact fleeing state survives save/load and subsequent casualties. Logs explain
  separation and maximum displacement, or why an engaged/fleeing unit stays compact.
  Invalid overlapping layouts are refused and logged rather than silently rebuilt.

Verification: 747 tests and 82 subtests passed across the Skirmishers, movement,
combat, character, persistence, psychology, Rally and scene suites. New regressions
cover successful/failed normal Rally and Rallying Cry, human/AI branches, save/load,
phase-end and spell-transition timing, re-engagement, attached bases, actual flee
animation and compact casualties. Human input is stubbed in these tests.

LEFTOVER: separation uses uniform expansion of local axis-aligned bases, not a
general minimum-displacement search constrained by terrain, board edges or other
units. Invalid compact layouts remain compact with a refusal log. Previously
saved units that were already spread while fleeing cannot recover their old ranks.
Combat scoring/disruption and the other charge-pairing limitations remain below.

The reproducible offscreen scenario is `python -m tests.test_skirmish_scene`.
It writes `saves/skirmishers.json` and `screenshots/skirmishers.png`, without
overwriting the player's quicksave. Load that save, select **Normal Rangers** in
Movement, then use **Adjust formation**. Saving does not persist an unconfirmed ghost.

## Remaining implementation order

1. **Authoritative formation state:** separate permission to adopt Skirmish from
   the active formation; implement legal reform switching. Harden malformed-save
  handling.
2. **Individual sight and shooting:** real base blockers and visible gaps,
   per-shooter range/LoS, joined-model Unit Strength for the all-US1 shooting
   penalty. Current centre rays/circular blockers are not per-model LoS.
3. **Charge declaration:** strict >50% visibility at declaration, with a visible
   count and deterministic failure reasons; preserve declaration-time reactions.
4. **Charge movement/form-up:** extend the loose/formed-target planners to joined
  and mixed bases, exact contact/path selection and alternative rear arrangements.
  Implement formed chargers contacting loose defenders without a second alignment
  wheel. Multiple chargers, command placement and per-model terrain still need
  integration and fixtures.
5. **Compact combat and constrained separation:** account for terrain, board
  edges and other units when finding the smallest legal separation; timing and
  the fleeing Rally-only house rule are implemented above. No rank bonus from
  starting Skirmish, no flank/rear combat-result bonuses against compact
  Skirmishers, and no disruption by Skirmishers; do not generalize that exemption
  to all geometry.
6. **Shared model mechanics:** exact troop-subcategory joining, command/champion
   representation and fighting-rank placement, directed casualties, multi-combat
   attack allocation and coherent resurrection/reinforcements.
7. **Remaining interactions:** terrain outlines/low obstacles, centre-based
   templates including joined bases, fleeing and forced incoherency resolution,
   normal march Leadership tests near enemies, and movement-path selection when
   a legal route requires detouring rather than the current direct translation.
8. **QoL and integration:** shared charge/sight previews, highlight specific
   invalid models, presets that retain joined-model coherence, human choice when
   several legal casualty groups exist, broader AI/state integration and boundary
   scenarios. Keep the unit-level workflow as the default.

LEFTOVER: every numbered item above. The current normal cursor preview still uses
the existing whole-footprint sweep to choose a destination before displaying the
per-base validator result; it may conservatively stop short. The editor permits
only straight start-to-end base paths and does not resolve crossing own-model
paths. Formation rotation changes proposed centres, not individual base headings.
Combat entry preserves a planned fighting rank; other pairings still use the
generic snapping hook. Separation retains current positions with a tiny expansion,
but does not yet solve surrounding-obstacle or boundary constraints.

## How the rule works
- **Loose formation** — models are ~1" apart in a contiguous blob, *not* in
  ranks/files. Each model moves individually in any direction, no wheeling, and
  must keep coherency (within 1" of another model in the unit).
- **360° facing / LoS** — no flank or rear arcs; may shoot and charge in any
  direction. Individual models block LoS as normal, but enemies can see
  *through the gaps* between models.
- **Enemy fire −1** — a unit shooting at Skirmishers (all models Unit Strength 1)
  suffers **−1 To Hit**.
- **No rank bonus** — a unit that is in Skirmish formation when it becomes
  engaged cannot claim a Rank Bonus.
- **In combat** — Skirmishers "form up" into base contact (a fighting rank) when
  they charge or are charged, then spread back out once combat ends. May charge
  a target visible to **more than 50%** of its models.
- **Panic** — fleeing Skirmishers do not cause panic in *formed* friendly units
  they flee through (they still panic other Skirmishers / cause normal panic when
  annihilated or broken).

## Where this lands in the code
- Units are a single rigid `bodyNP` with `unit.ranks/files/nmodels`; models are
  arranged in a grid in `units.py`. Movement is whole-unit (wheel/rotate) in
  `movement_system.py`.
- Rank/flank bonus is computed in `combat_resolution.py` (`_verySimpleBattleInner`,
  ~L687–706) from `unit.ranks`.
- Shooting arc is a ~90° front arc: `shootingArc(..., rotationangle=getH()+45)`
  (`movement_system.py` ~L208, driven from `taskShootingArcUpdate` in `game.py`).
  LoS currently uses centre rays/circular blockers; `losBlockUnit` is not a
  complete per-model implementation.
- Ranged To-Hit modifiers already flow through `to_hit_ranged` (it has
  `long_range`) in `toHitAndToWound.py` / `battleFunctions.py`.

## Historical baseline (not full-rule completion)

### Phase 0 — flag & state (small) — DONE
- Added a `_skirmishers` builder to `special_rules.py` (`tag:'formation', skirmish:True`).
- Added `model.is_skirmisher()` and `model.unit_strength()` helpers.

### Phase 1 — combat/shooting effects (cheap, high value, testable) — DONE
- **No rank bonus**: `combat_resolution.py` skips the rank-bonus increment for a
  skirmisher unit in both player branches.
- **Enemy fire -1**: `game.shootAt` sets `model.target_skirmisher` (US1 skirmisher
  target); `to_hit_ranged` applies a non-ignorable -1; `battleFunctions` threads
  the flag through.
- Tests: `tests/test_skirmishers.py` (flag helpers + -1 To Hit).

### Phase 2 — 360° arc (medium) — DONE
- `shootingArc` takes a `full_circle` flag; skirmishers get a 2π circle instead
  of the 90° front cone. Wired into the shooting and magic arc updates in
  `game.py` (and the point count still lands at the shader's 83).
- Existing LoS clipping (forest/units) still applies to the full circle. True
  per-model sight and seeing through gaps remain deferred.
- Note: charge direction isn't front-arc-gated in this engine (charging uses the
  movement swing, not an arc), so 360° charging already works; the exact
  ">50% of models must see the target" gate is deferred (needs a per-model LoS
  count).

### Phase 3 — loose formation + per-model movement (hard, architectural)
Minimal loose layout — DONE:
- `units.py` originally used a jittered loose blob. The current milestone
  replaces it with saved coherent positions and keeps one `bodyNP`.
Free 360° movement — DONE:
- `movement_system._skirmishMovePreview`: straight-line translation up to the
  move allowance in any direction with a circular range indicator + a ghost
  footprint at the destination (no wheel arc); `pathTowardsMouse` routes
  skirmishers to it.
- `moveUnit` skips the wheel rotation and rear-pivot for skirmishers (they
  translate freely and keep facing).
Visual form-up in combat — baseline only:
- `units.formUpForCombat()` snaps a skirmisher's models into a tight fighting
  rank on `enterInCombat`; `spreadToSkirmish()` returns them to the loose blob on
  `exitInCombat` (deterministic blob so it reproduces the same layout).
Current movement/coherency work is described above. A single `bodyNP` does not
prevent per-model base validation; it remains a conservative physics proxy.

### Phase 4 — panic & terrain nuance (small, later)
- Skirmishers fleeing don't panic formed friendlies — DONE. The Fled-Through
  panic cause in `psychology.py` (`_after_unit_done`) is guarded by the pure
  predicate `fled_through_panics(fleer_skirmish, target_skirmish)`: a fleeing /
  falling-back Skirmisher unit queues no Panic test for *formed* friendlies it
  passes through, but still panics friendly Skirmishers. Skirmishers still cause
  Panic as normal when annihilated or when they Break and flee (those go through
  `on_unit_destroyed` / `on_unit_flees_combat`, which are untouched).
  Rulebook p. 185 — <https://tow.whfb.app/unusual-formations/skirmishers-and-panic>.
  Same guard also restricts Fled-Through tests to *friendly* units (an enemy
  unit fled through no longer takes a Panic test). Tests in
  `tests/test_psychology.py::SkirmisherPanicTests`.
- Terrain nuance (shelter / cover from terrain) is still open.

Follow the remaining implementation order above; the historical DONE labels
cover only their listed baseline behaviour, not complete rules compliance.
