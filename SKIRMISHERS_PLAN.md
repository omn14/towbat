# Skirmishers — Implementation Plan

> Status: **complete for now** (Phases 0–4 panic guard done). Leftovers: true
> per-model coherency, "see through gaps" LoS, >50%-visible charge gate, terrain
> nuance.
> Note: a unit's skirmisher status comes from the army list's `special_rules`
> (unit-level rule), not the base catalogue model profile (e.g. Cathay
> "Peasant Soldier" has no Skirmishers on its model profile).

Rule (catalogue): *"A unit consisting of models with this special rule may adopt
a Skirmish formation."* Source rules: <https://tow.whfb.app/unusual-formations/skirmish-formation>
(rulebook p.184–186).

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
  LoS is already **per-model** via `losBlockUnit` / `los_block_point`.
- Ranged To-Hit modifiers already flow through `to_hit_ranged` (it has
  `long_range`) in `toHitAndToWound.py` / `battleFunctions.py`.

## Plan (staged by value vs. risk)

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
- Per-model LoS clipping (forest/units) still applies to the full circle. The
  "see through gaps" nuance is deferred.
- Note: charge direction isn't front-arc-gated in this engine (charging uses the
  movement swing, not an arc), so 360° charging already works; the exact
  ">50% of models must see the target" gate is deferred (needs a per-model LoS
  count).

### Phase 3 — loose formation + per-model movement (hard, architectural)
Minimal loose layout — DONE:
- `units.py` lays skirmisher models out as a loose blob (roughly square, ~1"
  gaps, deterministic jitter) instead of a rigid grid, and sizes the collision
  footprint to cover the blob. Single `bodyNP` kept.
Free 360° movement — DONE:
- `movement_system._skirmishMovePreview`: straight-line translation up to the
  move allowance in any direction with a circular range indicator + a ghost
  footprint at the destination (no wheel arc); `pathTowardsMouse` routes
  skirmishers to it.
- `moveUnit` skips the wheel rotation and rear-pivot for skirmishers (they
  translate freely and keep facing).
Form-up in combat — DONE:
- `units.formUpForCombat()` snaps a skirmisher's models into a tight fighting
  rank on `enterInCombat`; `spreadToSkirmish()` returns them to the loose blob on
  `exitInCombat` (deterministic blob so it reproduces the same layout).
Still TODO:
- Full per-model movement with per-model coherency (currently one `bodyNP`).

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

## Recommendation
Do **Phase 0 + Phase 1** first (localized edits + tests; immediate combat/shooting
behaviour) → then **Phase 2** (360° arc) → treat **Phase 3** (true loose
movement / form-up) as a dedicated follow-up since it's the only architecturally
heavy part.
