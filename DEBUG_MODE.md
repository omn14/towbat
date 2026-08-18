# Debug Mode

Developer tools for testing special-rule implementations without playing a full
game: free unit movement, forced combat setup, rule granting, loaded dice and
state snapshots.

Implemented in `debug_tools.py`. Constructed in `game.py` only when debug mode is
requested at launch — in a normal game nothing is imported into the input map,
no tasks run and no overlay is drawn.

---

## Activation

Debug mode is gated twice: once at launch, once at runtime.

**1. Launch with the flag or env var**

```powershell
# PowerShell
$env:WH_DEBUG = "1"; python game.py

# or, per-run
python game.py --debug
```

Without one of these, `DebugTools` is never constructed and `game.debug_tools`
is `None`.

**2. Press `F12` in game**

`F12` binds every other debug key and shows the HUD overlay. `F12` again unbinds
them, restores normal dice and hides the overlay. A key reference is printed to
the console on activation, and `h` reprints it.

> The project runs on the poetry environment
> `wh-rFkACKCq-py3.10`. The default `python` on PATH does not have Panda3D.

---

## Key reference

### Movement

| Key | Action |
| --- | --- |
| `g` | Grab / drop the unit under the cursor (falls back to the selected unit) |
| arrow keys | Nudge 0.5" along the world X/Y axes |
| `shift` + arrows | Nudge 0.1" — for exact base-to-base contact |
| `z` / `x` | Rotate ±15° |
| `shift-z` / `shift-x` | Rotate ±1° |
| `b` / `n` | Select the previous / next unit |

While a unit is grabbed it follows the mouse every frame. Movement is committed
straight to the unit's body node, so **move allowance, terrain penalties, wheel
pivots, sweep tests, table boundaries and unit collision are all bypassed** — the
normal `moveUnit()` path is never entered. Models are re-seated on the terrain
surface after every grab, nudge and rotation.

### Turn and phase

| Key | Action |
| --- | --- |
| `r` | Reset every turn flag on every unit and return them to `Idle` |
| `1` `2` `3` `4` | Jump to Strategy / Movement / Shooting / Combat phase |

`r` clears `hasMovedThisTurn`, `hasAttackedThisTurn`, `attemptedRallyThisTurn`,
`chargedThisTurn`, `cannotChargeThisTurn`, `isChargingMove`,
`panicTestedThisPhase`, `usedStubborn` and `madePursuitChoice`, and resets
`startOfPhaseModels`. Units already `InCombat` or `IsFleeing` keep their state.

### Combat setup

| Key | Action |
| --- | --- |
| `e` | Engage the selected unit with the nearest enemy |
| `shift-e` | Disengage the selected unit from everything it is fighting |
| `k` | Remove 1 model from the selected unit |
| `shift-k` | Remove 5 models |

`e` fills in both sides of the engagement (`isInCombat`, `isInCombatWith`,
`isInCombatFlank`) and derives the facing — front, flank or rear — from where the
attacker actually sits relative to the defender's footprint. Position the unit
with `g` and the arrow keys first, then press `e` to lock in that geometry.

`k` routes through the real casualty-removal path, so unit destruction, Panic on
nearby friends and collision-shape rebuilding all behave normally.

### Special rules

| Key | Action |
| --- | --- |
| `shift-b` / `shift-n` | Cycle the rule under test |
| `shift-g` | Grant the current rule to the selected unit |
| `shift-d` | Revoke it from the selected unit |
| `i` | Dump the selected unit to the console |
| `shift-i` | Dump every unit |

The cycle covers every keyword with a coded builder in `special_rules.py`:

```
fly · furious charge · regeneration · skirmishers · stubborn · unbreakable · venerable
```

Granting supplies a working parameter where the catalogue normally omits one
(Regeneration → `5+`, Fly → `9`), so the rule produces a real engine hook rather
than an inert flag. Granting replaces any existing rule of the same name.

The dump marks each rule `[CODED]` or `[flag ]`:

```
── Black Orc Mob (player 2) ─────────────────────
  state    : Idle   pos (12.40, -3.10, 0.00)  H 180.0
  models   : 20 (battle start 20, phase start 20)
  flags    : moved=False attacked=False charged=False noCharge=False charging=False
           : panicked=False stubbornUsed=False rallied=False skirmisher=False
  [CODED] Furious Charge -> charge
  [CODED] Regeneration -> regen
  [flag ] Choppas
```

`[CODED]` means the rule dict carries an engine hook key; `[flag ]` means it is
display-only and has no implementation yet. This is the quickest way to answer
"is this rule actually wired up?".

### Dice and state

| Key | Action |
| --- | --- |
| `y` | Cycle loaded d6: off → all 1s → all 6s → off |
| `F8` | Write a snapshot to `debug_snapshot.json` |
| `shift-F8` | Restore that snapshot |
| `h` | Reprint the key reference |
| `F12` | Turn debug mode off |

Loaded dice force every logic roll to the minimum or maximum, which makes
save/wound/Leadership outcomes deterministic — use all 1s to guarantee failures
and all 6s to guarantee passes.

---

## HUD overlay

While debug mode is on, the top-right overlay shows live state:

```
DEBUG  phase MovementPhase  round 3 P1
Black Orc Mob  [Idle]  GRABBED
  ( 12.40, -3.10)  H  180.0  x20
  moved 0  atk 0  chg 0  combat 0
  coded: Furious Charge, Regeneration
  flags: Choppas
rule under test: regeneration   dice: random
```

---

## Typical workflows

**Test a rule that only triggers on a flank charge**

1. `F12`, then `n` until the attacking unit is selected.
2. `g`, move it beside the target's flank, `g` again to drop.
3. `shift` + arrows and `shift-z` / `shift-x` to line up base contact exactly.
4. `e` — the console confirms the facing that was registered.
5. `4` to jump to the Combat phase and resolve.

**Check whether a rule is implemented at all**

1. `F12`, select the unit with `b` / `n`.
2. `i` — anything listed as `[flag ]` has no coded effect.

**Isolate one rule's effect**

1. `i` to record the baseline.
2. `shift-b` / `shift-n` to pick the rule, `shift-g` to grant it.
3. `F8` to snapshot, fight the combat, `shift-F8` to rewind.
4. `shift-d` to revoke, then fight the same combat again and compare.

**Make an outcome reproducible**

1. `y` once for all 1s (everything fails) or twice for all 6s (everything passes).
2. Run the combat.
3. `y` again to return to random rolls.

**Re-run the same move repeatedly**

`r` after each attempt clears the "already moved" flags, so a unit can be moved
as many times as you like within one phase.

---

## Calling the tools from code

Every binding is a plain method on `game.debug_tools`, so the same operations can
be scripted or driven from a breakpoint:

```python
dbg = base.debug_tools

dbg.select(base.player1Units[0])
dbg.nudge(2.0, 0.0)
dbg.rotate(90)
dbg.reset_flags()
dbg.goto_phase("CombatPhase")

dbg.grant_rule("regeneration", param="4+")   # override the default test value
dbg.revoke_rule("stubborn")
dbg.dump()

dbg.set_loaded_dice(1)      # every d6 rolls 1
dbg.set_loaded_dice(None)   # back to random
dbg.engage_nearest()
dbg.snapshot(); dbg.restore()
```

---

## Limitations

- **Physics dice are unaffected by `y`.** Loaded dice work by intercepting
  `random.randint(1, 6)`, which covers the logic rolls in `battleFunctions.py`,
  `psychology.py`, `rulesFunctions.py`, `cannon_fire.py` and `bombardment.py`.
  The rolling `Dice` objects in `dice.py` read their result from the physics
  simulation and have no shared chokepoint, so they still roll freely. Making
  those deterministic requires routing all rolls through a single function first.
- **Free movement does not create combats.** Teleporting a unit into contact
  will not trigger a charge or engagement, because the collision and charge
  reaction logic lives in `moveUnit()`. Use `e` afterwards.
- **Granted rules are not persisted.** They live on the model instance, so a
  snapshot restore or a reload drops them. Re-grant after restoring.
- **Restoring a snapshot can rebuild units**, which invalidates the grab. The
  grab is released automatically before both snapshot and restore.
- Debug key bindings are only registered while debug mode is on, so they cannot
  interfere with the list builder, the tutorial or the campaign map when it is
  off. They can still overlap with those screens while it is on — turn debug
  mode off before using them.

### Keyboard layout

Every binding is a letter, a digit, an arrow or a function key — no punctuation.
Panda3D names punctuation keys after the character a US layout produces, so on a
Norwegian keyboard `[` and `]` (AltGr) and `,` / `.` never emit the events the
bindings would listen for, and the keys would silently do nothing. Rotation uses
`z` / `x` and cycling uses `b` / `n` for that reason.

---

# How it is built

Notes for anyone extending the debug system.

## Files

| File | Role |
| --- | --- |
| `debug_tools.py` | The whole system: `debug_enabled()`, `DebugTools`, key map, help text |
| `game.py` | Two lines only — the import, and construction inside `MyApp.__init__` |

Nothing else in the codebase knows the debug system exists. There are no
`if debug:` branches in the engine, and no debug state is written to saves. If
`debug_tools.py` were deleted, only those two lines in `game.py` would break.

## Wiring into the game

```python
# game.py, top
from debug_tools import DebugTools, debug_enabled

# game.py, MyApp.__init__, after the other subsystems are constructed
self.debug_tools = DebugTools(self) if debug_enabled() else None
```

Construction order matters: `DebugTools` is created after `self.movement`,
`self.fsm` and the unit lists exist, because `_bind()` reads `game.fsm.PHASES`
to build the phase-jump bindings. It is placed just after `Bombardment`.

`debug_enabled()` checks `WH_DEBUG` in the environment and `--debug` in
`sys.argv`. When it is false, `DebugTools` is never instantiated, so no events
are accepted, no task runs and no GUI node is created.

## Class shape

`DebugTools` subclasses `DirectObject`, which gives it its own event and task
scope. This matters: Panda3D's `accept()` replaces a handler *per DirectObject*,
so binding `mouse1` here would add a third handler alongside the ones on `MyApp`
and `GamePhaseFSM` rather than replacing them. The system therefore avoids mouse
buttons entirely and only uses keys.

State lives in a handful of attributes set in `__init__`:

| Attribute | Meaning |
| --- | --- |
| `enabled` | Debug mode on/off; `F12` flips it |
| `grabbed` | The unit currently following the cursor, or `None` |
| `rule_index` | Index into `TESTABLE_RULES` for the rule under test |
| `loaded_dice` | `None`, `1` or `6` |
| `_orig_randint` | The real `random.randint`, stashed while dice are loaded |
| `overlay` | The HUD `OnscreenText`, created lazily on first enable |

## Two-stage activation

`__init__` binds exactly one event, `f12`, and prints a one-line notice.
`enable()` calls `_bind()`, creates or shows the overlay and starts the
`debugOverlay` task. `disable()` releases the grab, restores the dice, calls
`_unbind()`, stops the task and hides the overlay.

`_bind()` and `_unbind()` must stay in sync — every key accepted in one has to
be ignored in the other, or bindings leak after toggling off. The phase digits
are bound in a loop over `game.fsm.PHASES` and unbound in a matching loop.

## How free movement works

There is no bypass flag anywhere in the movement system. Free movement simply
never calls `MovementSystem.moveUnit()`; it writes to the node directly:

```python
unit.bodyNP.setPos(x, y, 0)
unit.bodyNP.setH(heading)
```

`bodyNP` is the `BulletRigidBodyNode` root and the model is its child, so moving
it moves everything. The box shape is centred on the body origin, which is why
`setPos` places the unit's centre under the cursor and `setH` rotates about that
centre rather than the rear pivot the real wheel uses. Because nothing calls the
sweep tests, the move allowance, terrain multipliers, boundary ghosts and unit
collision are all skipped as a side effect, not as an explicit exception.

After every placement `_settle()` calls
`MovementSystem.alignModelsToHillNormal(unit)` so individual models re-seat on
the terrain surface. That call is idempotent, so repeating it is harmless.

The cursor position comes from `_ground_point()`, which extrudes the mouse ray
through the lens and solves analytically for where it crosses `z = 0`. This is
deliberately *not* a `rayTestClosest` against the board: a physics ray would hit
the grabbed unit's own collision box, and would need mask juggling to avoid it.

Unit picking under the cursor (`unit_under_cursor()`) does use a physics ray,
with `CM.HOVER_PICK`, and maps the hit node's `UnitCollision-<name>` name back to
an entry in `game.units` — the same convention `MyApp.mouseHoverUnit` uses.

## How rule granting works

`TESTABLE_RULES` is derived at import time from `SPECIAL_RULE_BUILDERS` in
`special_rules.py`, so any keyword that gains a builder automatically appears in
the cycle. Nothing needs updating here when a rule is implemented.

`grant_rule()` calls the builder directly and appends the resulting dict to
`model.special_rules`, replacing any entry with the same name. `RULE_TEST_PARAMS`
supplies the param the catalogue normally omits, so `regeneration` grants an
actual `regen: 5` hook instead of an inert flag. Pass `param=` to override.

`_rule_hooks()` decides whether a rule is implemented by listing the dict keys
that are not `name`, `description` or `tag` — anything left is an engine hook.
This is what drives the `[CODED]` / `[flag ]` split in the dump and the
`coded:` / `flags:` lines in the overlay. **If a new rule dict ever adds a
non-hook metadata key, add it to that exclusion set** or it will be misreported
as implemented.

## How loaded dice work

`set_loaded_dice()` swaps `random.randint` for a wrapper that returns the fixed
value when called as `randint(1, 6)` and defers to the original otherwise. This
works because every module in the project calls `random.randint(...)` through
the module attribute; none of them do `from random import randint`, which would
capture the original function and escape the patch.

It is a monkeypatch rather than a proper seam because there is no single roll
function to hook. The physics `Dice` objects in `dice.py` read their value from
the simulation and are unaffected. Introducing a `roll_d6()` chokepoint and
routing the scattered `random.randint(1, 6)` calls through it would let this be
replaced with a real forced-roll queue.

The original function is always restored on `disable()`, so toggling debug mode
off can never leave the game with loaded dice.

## Combat forcing

`engage_nearest()` reproduces by hand what `CombatResolver` does when a charge
connects: set `isInCombat` on both units, append each to the other's
`isInCombatWith`, append the facing to `isInCombatFlank`, request the `InCombat`
state and refresh the text nodes. The facing comes from `_flank_of()`, which
compares the attacker's position in the defender's local space against the
defender's footprint aspect ratio, giving front, flank or rear.

`kill_models()` deliberately routes through
`MovementSystem.removeModelsFromUnit()` rather than removing nodes itself, so
unit destruction, the nearby-friend Panic test and collision-shape rebuilding
all behave exactly as they do in a real game.

## Adding a new tool

1. Write the method on `DebugTools`. Read the unit through the `selected`
   property, which falls back safely when the current unit has been destroyed.
2. Add the `accept()` in `_bind()` **and** the matching name in `_unbind()`.
3. Use a letter, digit, arrow or function key — see the keyboard-layout note
   above — and check it against the keys already taken in `game.py`
   (`q w a l t c m f5`–`f10 shift-p`) and those already used here.
4. Add a line to `HELP_TEXT` and a row to the key tables in this file.

Keep new tools inside `debug_tools.py`. The value of this system is that it is
entirely removable; adding debug branches to engine code would lose that.

