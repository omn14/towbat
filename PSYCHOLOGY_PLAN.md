# Psychology System — Implementation Plan

> Status: **planning only — no code yet.** Start with the **Panic test** (Phase 0).
> Source: https://tow.whfb.app/the-psychology-of-war (Rulebook p. 160–161).
> This document is a task list; each phase can be built and committed on its own.

The Old World "psychology" umbrella covers several tests and states. This plan
scopes the **whole** system for context but **begins with Panic**, which is the
foundation the rest build on (they all end in a Leadership test → flee/fall back).

---

## 0. Where this plugs into the existing code

Reuse what already exists rather than reinventing:

- **Leadership test** — 2D6 vs `Ld`. Already done ad-hoc in
  `combat_resolution.py` (Break test around L900+, `rullTerninger`) and
  `game.py:rallyUnit`. Extract a single reusable `leadership_test(unit, modifier=0)`.
- **Fleeing** — `unit.request("IsFleeing")`, `units.py:enterIsFleeing`,
  `movement_system.py:fallBack` / `fleeDirectionMultUnits`, and
  `combat_resolution.py:GiveGroundFromCombat` / `fleeInterval`.
- **Flee direction** — `movement_system.py:fleeDirectionMultUnits` already flees
  away from enemies; panic needs "flee directly away from the nearest enemy unit
  that is not itself fleeing".
- **Unit Strength** — `models.py:unit_strength()` (needs a unit-level sum:
  US × models; currently per-model only — see Data model below).
- **Per-turn/phase flags** — mirror the `chargedThisTurn` pattern (set on the
  unit, cleared by the FSM; saved in `persistence.py`).
- **Phase transitions** — `game_fsm.py` (Strategy/Movement/Shooting/Combat
  enter/exit hooks) is where "start of phase" snapshots and "one test per phase"
  resets live.
- **Catalogue rules already present**: `Ignore Panic`, `Ignore Goblin Panic`,
  and "automatically passes any Panic tests" (BSB-style) appear in army data —
  wire these as exemptions.

---

## Phase 0 — Panic test core  ← START HERE

Goal: a single, reusable Panic test that any cause can call, with the correct
pass/fail outcome. No causes wired yet (trigger it from a debug key first).

> Status: **DONE.** Implemented in `psychology.py` (`PsychologySystem`,
> `leadership_test`, `panic_fail_outcome`, `unit_strength_total`). Unit state
> `startOfBattleModels` / `panicTestedThisPhase` on the unit (persisted); the
> FSM clears `panicTestedThisPhase` on each phase enter. Debug key **Shift+P**
> forces a Panic test on the selected unit. Tests in `tests/test_psychology.py`.

Rules (panic-tests page):
- A Panic test = test against the unit's `Ld` (2D6 ≤ Ld → pass).
- **Pass** → unit holds, nothing happens.
- **Fail** → outcome depends on casualties vs the unit's *start-of-battle* size:
  - **> 50%** of start-of-battle models remain → **Fall Back in Good Order**.
  - **≤ 50%** remain → **Flee** immediately.

Tasks:
- [ ] Record each unit's **start-of-battle model count** once (e.g.
  `unit.startOfBattleModels` set at deploy/first load). Persist it.
- [ ] Add `leadership_test(unit, modifier=0) -> bool` (shared helper; roll 2D6,
  compare to `Ld`; return pass/fail). Refactor Break/rally to use it later.
- [ ] Add `panic_test(unit, flee_from=None, cause="") -> None` that:
  - returns immediately if the unit is exempt (see Phase 2 exemptions);
  - rolls `leadership_test`;
  - on fail, picks **Fall Back in Good Order** vs **Flee** using the ≥/≤50%
    start-of-battle rule;
  - triggers the existing flee / fall-back movement, fleeing away from
    `flee_from` (or nearest non-fleeing enemy if `None`).
- [ ] "Flee directly away from the nearest enemy unit that is not itself
  fleeing" — small helper `nearest_non_fleeing_enemy(unit)` feeding the flee
  direction (reuse `fleeDirectionMultUnits`).
- [ ] Debug hook: a key (e.g. Shift+P) that forces the selected unit to take a
  Panic test, so the outcome branches can be verified before any cause exists.
- [ ] Console log line consistent with the combat debug printout (unit, Ld,
  2D6 roll, pass/fail, outcome, % remaining).

Acceptance: pressing the debug key on a fresh unit either holds or falls back;
on a unit reduced below 50% it flees; direction is away from the nearest live
enemy.

---

## Phase 1 — Common causes of Panic

Each cause calls `panic_test`. Respect "one test per phase" (Phase 2) and
"leave the triggering unit in place until all tests are made" (measure point).

> Status: **DONE.** Cause hooks wired via `PsychologySystem`
> (`check_heavy_casualties`, `on_unit_destroyed`, `on_unit_flees_combat`, and
> Fled-Through in the flee move). Start-of-phase model counts snapshot in the
> FSM (non-combat phases) and persist. Pure helpers `heavy_casualties`,
> `unit_strength_total`, constants `PANIC_US_THRESHOLD`/`PANIC_RADIUS`; tests in
> `tests/test_psychology.py`.

- [x] **Heavy Casualties** (heavy-casualties): in **any phase except Combat**,
  if a unit loses **> 25%** of the models it had **at the start of that phase**,
  it tests. Flee from the enemy unit that caused the casualties (or nearest
  non-fleeing enemy if none). → snapshot each unit's model count at the start of
  each non-combat phase (`game_fsm` enter hooks); check after shooting/magic
  resolves. Note: heavy casualties in the **Combat phase do NOT** cause panic.
- [x] **Nearby Friend Destroyed** (nearby-friend-destroyed): when a unit of
  **US ≥ 5** is destroyed, all friendlies within **6"** test. Leave the
  destroyed unit in place as the measure point; failed units flee from nearest
  non-fleeing enemy.
- [x] **Nearby Friend Flees Combat** (nearby-friend-flees-combat): when a
  friendly **US ≥ 5** unit loses combat and either Breaks/flees **or** Falls
  Back in Good Order, friendlies within **6"** test (measure before it moves).
  Hook into the existing Break-test outcome in `combat_resolution.py`.
- [x] **Fled Through** (fled-through): when a fleeing / falling-back friendly
  unit moves **through** another unit, that unit tests (resolve the movement
  first). Can cascade — a panicked unit fleeing through another triggers a
  further test. Hook into the flee/fall-back movement path.
- [x] **Shooting casualties panic** — already referenced by the Shooting rules;
  folded into Heavy Casualties (shooting is a non-combat phase; cannon +
  bombardment wired too).

Acceptance: each cause reliably produces exactly one test at the right time,
measured from the correct point, fleeing in the correct direction.

---

## Phase 2 — "No Need for Hysterics" & bookkeeping

Rules (no-need-for-hysterics):
- A unit makes **only one Panic test per phase**, even with multiple causes.
- A unit is **not required** to test if it is:
  - making a **Charge** move,
  - **engaged in combat**,
  - **already fleeing** (and has yet to rally).

> Status: **DONE.** Full exemption list confirmed against the rulebook (charge
> move / engaged / already fleeing) plus rule immunities (Ignore Panic, Immune
> to Psychology, Unbreakable) and one-test-per-phase. `panic_exempt_reason`
> reports the specific reason. `isChargingMove` flag set when a charge is
> declared and cleared at end of the Movement phase. Simultaneous tests resolve
> via the sequential panic queue in the owner's unit-list order. Exemption tests
> in `tests/test_psychology.py`.

Tasks:
- [x] `unit.panicTestedThisPhase` flag; set on any panic test, cleared by the
      FSM on each phase enter. Persist it.
- [x] Exemption check used by `panic_test`: charging, in combat, already
      fleeing, and rule-based (`Ignore Panic`, Unbreakable, Immune to
      Psychology -> auto-pass/skip).
- [x] Simultaneous-test ordering: resolved by the sequential panic queue in the
      owner's unit-list order (documented simplification).
---

## Phase 3+ — Rest of the psychology system (later)

These build on the same Leadership-test + flee/hold plumbing. Listed for scope;
detailed tasks to be written when Panic is done.

- [ ] **Fear** — units in combat with a Feared enemy test; effects on charging /
  being charged and combat.
- [ ] **Terror** — causes a Panic-like test when charged by / charging a Terror
  causer; Terror causers are immune to Fear.
- [ ] **Immune to Psychology** — ignores Fear/Terror/Panic-from-psychology.
- [ ] **Stubborn** — Break tests (and some panic) on **unmodified** Ld. Partial
  data already read by the AI classifier; not yet applied.
- [ ] **Unbreakable** — never flees; already a flag in `special_rules.py`, wire
  the exemption.
- [ ] **Hatred** — reroll misses first round (combat, not strictly psychology).
- [ ] **Frenzy** — must charge, extra attacks, immune to psychology while
  frenzied, frenzy can be broken.
- [ ] **Cold-Blooded / army-specific** Ld modifiers (e.g. re-roll, best-of-3).

---

## Data model / state additions (summary)

Added to the unit (persist in `persistence.py`, save/restore like
`chargedThisTurn`):
- `startOfBattleModels` — set once; drives the 50% flee/fall-back split.
- `startOfPhaseModels` — snapshot on each non-combat phase enter; drives the
  25% Heavy-Casualties check.
- `panicTestedThisPhase` — reset by the FSM per phase.

Helpers to add:
- `models.py` / unit: `unit_strength_total(unit)` = per-model US × current
  models (for the US ≥ 5 thresholds).
- `combat_resolution.py` (or a new `psychology.py`): `leadership_test`,
  `panic_test`, `nearest_non_fleeing_enemy`.

Consider a dedicated `psychology.py` module so the causes, tests, and exemptions
live in one place (mirrors `special_rules.py` / `terrain_system.py`).

---

## Open questions / to verify against the rulebook

- Exact remaining bullets of "No Need for Hysterics" (page text was truncated).
- Whether Fall-Back-in-Good-Order vs Flee uses **start-of-battle** count in all
  panic cases (confirmed on the panic-tests page) — reuse `startOfBattleModels`.
- Interaction of Heavy-Casualties direction ("flee from the causing enemy") when
  casualties came from a spell/template with no single source unit.
- How our simultaneous multi-unit tests should order for the human player.

## Testing plan

- Unit tests (no Panda needed) for the pure logic: `leadership_test`
  distribution edge cases; 50% split (Fall Back vs Flee); 25% heavy-casualty
  threshold; US ≥ 5 total; exemption predicate; one-test-per-phase gating.
- Manual: debug-key panic; kill a US ≥ 5 unit near friends; rout a US ≥ 5 unit
  in combat near friends; flee a unit through a friendly line (cascade).
