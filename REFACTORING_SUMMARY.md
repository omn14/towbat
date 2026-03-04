# Refactoring Summary — `game.py`

## Overview

The original `game.py` was a **4,899-line monolith** containing the entire game application: FSM phase management, spell logic, save/load persistence, physics setup, unit movement, combat resolution, campaign map, rendering, and UI — all in a single file with scattered imports, dead code blocks, and commented-out experiments.

The refactored `game.py` is **2,753 lines** (~2,150 lines removed), with four cohesive subsystems extracted into their own modules and the remaining code organized with section headers and cleaned imports.

---

## Changes Made

### 1. Extracted `game_fsm.py` (365 lines)

**What:** The inline `gameFSM(FSM)` class (~330 lines) that managed all game phase transitions (Deploy, Strategy, Movement, Shooting, Combat, Spell, Campaign, MakeChoice) was extracted to its own module as `GamePhaseFSM`.

**Why:** The FSM is a self-contained state machine with its own `enter*`/`exit*` handlers. Keeping it inline obscured the game's phase flow and made it hard to modify phase logic without wading through thousands of unrelated lines. The extracted version is a clean, readable map of all game states.

**Key decision:** Added backward-compatible properties (`currentPhaseIndex`, `endOfTurnSpells`, `phases`) so existing code continues to reference `self.fsm.currentPhaseIndex` etc. without changes.

### 2. Extracted `spell_system.py` (180 lines)

**What:** Three inline spell classes (`spell`, `spellDevilsVisit`, `spellRaiseDead`) were extracted and renamed to `Spell`, `DevilsVisitSpell`, and `RaiseDeadSpell`.

**Why:** The spell classes had their own casting logic, dice rolling, and lifecycle management — entirely independent of the game loop. Extracting them makes it straightforward to add new spells without touching the main game file.

**Key decision:** Used PEP 8 naming (CamelCase class names) instead of the original lowercase names. Updated the spell dictionary in `__init__` to reference the new class names.

### 3. Extracted `persistence.py` (163 lines)

**What:** The `save_game_state` and `load_game_state` methods (~180 lines of JSON serialization/deserialization) were extracted as standalone functions that accept the game instance as a parameter.

**Why:** Save/load is a cross-cutting concern that serializes many game objects. Having it inline made the main class harder to navigate and coupled persistence logic with game logic.

**Key decision:** The methods in `MyApp` now delegate to the extracted functions with one-line wrappers, preserving the `self.save_game_state()` / `self.load_game_state()` call interface.

### 4. Removed Dead Code (~250+ lines)

| Removed | Lines | Reason |
|---------|-------|--------|
| Commented-out unit creation blocks | ~140 | Old hardcoded unit setups replaced by `load_army_from_json()` |
| `if 0:` test blocks | ~100 | Debug/test scenarios wrapped in `if 0:` (never executed) |
| `upAndDown()` method | ~60 | Debug method referencing `self.goblins` (no longer exists) |
| `waitForChoice()` method | ~3 | Empty `pass` method, never called |

### 5. Cleaned Imports (68 → 57 lines, zero duplicates)

**Before:** 68 lines of scattered imports with duplicates (`Vec3` imported 3×, `BulletRigidBodyNode` 3×, `CardMaker` 2×, etc.), unused imports (`FSM`, `datetime`, `NurbsCurve`, `Mopath`), and no organization.

**After:** Grouped into logical sections:
- Standard Library
- Panda3D Core (single consolidated import)
- Panda3D Bullet Physics (single consolidated import)
- Panda3D Direct (ShowBase, intervals, GUI)
- Shaders
- Project Modules
- Extracted Subsystems

Removed: `FSM` (now in game_fsm.py), `datetime` (now in persistence.py), `NurbsCurve` / `Mopath` / `MopathInterval` (unused).

### 6. Added Section Headers

Added 18 section headers (`# ─── Section Name ───`) to organize `MyApp` methods into logical groups:

- Initialization
- Army Loading
- Texture Baking
- Projectiles & Visual Effects
- Task Management & Phase Loops
- Phase Task Loops
- Camera & UI
- Unit Selection & Interaction
- Campaign Map
- Shader & Physics Setup
- Drawing Helpers
- Movement & Pathfinding
- Unit Movement Execution
- Combat Resolution
- Flee, Pursuit & Rally
- Persistence
- Camera Zoom & Controls
- List Builder & Army Management UI

### 7. Extracted `combat_resolution.py` (1,071 lines)

**What:** Twelve combat-related methods were extracted into a `CombatResolver` class: `checkUnitContactSmall`, `chargeAndChargeReaction`, `fleeInterval`, `rullTerninger`, `chargeInterval`, `getFlankFromContact`, `printBattleResults`, `verySimpleBattleStart`, `verySimpleBattle`, `GiveGroundFromCombat`, `FBIGFromCombat`, and `fleeFromCombat`.

**Why:** These methods form a tightly coupled combat subsystem covering charge resolution, dice rolling, melee battle simulation, leadership tests, pursuit/flee/give-ground outcomes, and FBIG (Fall Back in Good Order). They had no business living in the main game class alongside camera controls and UI code.

**Key decision:** `CombatResolver` receives the game instance (`self.game`) and accesses all game state through it. Game attributes like `world`, `units`, `player1Units`, `player2Units`, `playerNP`, `unitToMove`, `attackSequence`, `autoRoll`, `autoCharge`, `autoHold`, and `diceInfoText` are accessed via `self.game.*`. Panda3D globals (`render`, `taskMgr`, `messenger`, `loader`, `base`) remain as globals. Thin delegate methods on `MyApp` preserve the `self.methodName()` call interface for all internal callers and external callers (e.g., `deployPhase.py` calling `game.checkUnitContactSmall()`).

### 8. Naming Improvements

- `gameFSM` → `GamePhaseFSM` (PEP 8 class naming)
- `spell` / `spellDevilsVisit` / `spellRaiseDead` → `Spell` / `DevilsVisitSpell` / `RaiseDeadSpell`
- `persuitMove` → `pursuitMove` (typo fix)
- `"event recieced"` → `"event received"` (typo fix)
- `"Ingrease"` → `"Increase"` (typo fix in spell description)

**Not renamed (risk vs. reward):** Norwegian variable names (`terninger` = dice, `terning` = die) appear 50+ times across combat methods. A bulk rename risks breaking functional code without test coverage, so these were left in place with the recommendation to rename them in a future pass with tests.

---

## File Structure After Refactoring

```
game.py              (2,753 lines)  — Main game application
combat_resolution.py (1,071 lines)  — Combat resolution subsystem
game_fsm.py          (365 lines)    — Game phase state machine
spell_system.py      (180 lines)    — Spell base class + implementations
persistence.py       (163 lines)    — Save/load game state
game_original.py     (4,899 lines)  — Backup of original (can be deleted)
```

## What's Left (Future Work)

1. **Rename Norwegian variables** — `terninger`→`dice_set`, `terning`→`die` across combat methods (50+ occurrences, needs test coverage first)
2. ~~**Extract combat resolution**~~ — Done. 12 methods (1,071 lines) extracted to `combat_resolution.py`
3. **Extract movement system** — `pathTowardsMouse`, `moveUnit`, sweep tests (~400 lines) are another candidate
4. **Extract drawing helpers** — Circle/arc/rectangle drawing methods (~200 lines) are self-contained
5. **Remove remaining commented-out blocks** — A few `""" ... """` blocks remain inside active methods (e.g., inside `mouseHoverUnit`)
