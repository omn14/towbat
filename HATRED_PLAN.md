# Hatred (X) — implementation plan

Rulebook p. 171. Last checked against <https://tow.whfb.app/special-rules/hatred>
on 2026-09-06; the page carries **no FAQ or errata cross-references**, so the
printed wording is the whole of it:

> A model with this special rule may re-roll any failed rolls To Hit made
> against a hated enemy during the first round of combat.
>
> Which enemies are hated varies from model to model and will be shown in
> brackets after the name of this special rule (shown here as 'X'). Some models
> hate 'all enemies', meaning they hate all enemy models equally.

Three separate problems, only one of which is the rule itself:

1. the engine cannot tell which round of a combat it is in;
2. `X` is prose that has to be resolved to something the engine can compare;
3. the re-roll.

---

## 1. What the catalogue actually contains

Surveyed with `has_*`-style matching over every model and weapon record:
**111 carriers, 8 distinct spellings.**

| X | carriers | resolves to |
| --- | ---: | --- |
| `Orcs & Goblins` | 38 | faction `Orc and Goblin Tribes` |
| `Beastman Brayherds` | 30 | faction `Beastmen Brayherds` |
| `High Elves` | 24 | faction `High Elf Realms` |
| `Dwarfs` | 8 | factions `Dwarfen Mountain Holds` **and** `Chaos Dwarfs` |
| `All Enemies` | 4 | everything |
| `all enemies` | 3 | everything (a second casing of the same thing) |
| `Warriors of Chaos & Daemonic models` | 3 | faction + keyword |
| `Warriors of Chaos, Beastmen Breyherds & Daemonic models` | 1 | two factions + keyword |

Not one of the eight matches a faction name exactly, so a bare string compare
against `Faction` finds nothing at all.

Two things that decide the design:

- **`X` cannot be split on `&`.** `Orcs & Goblins` is a single faction;
  `Warriors of Chaos & Daemonic models` is two different tests. The whole
  string is matched against an alias table first, and only what fails to match
  whole is split on `,` and `&` and matched piece by piece.
- **`Daemonic` is a keyword, not a faction.** It is a real special rule on 31
  models across Daemons of Chaos and elsewhere, so "Daemonic models" is a test
  against the target's rules list rather than its army.

`Breyherds` in the last row is a typo in the source data — the same army is
spelled `Brayherds` everywhere else, including in the Bretonnian Grand Master's
roster rule that grants this. Aliased rather than corrected: the catalogues are
a vendored dependency and get re-cloned.

### Identifying the enemy needs no new plumbing

Verified rather than assumed: `characteristics['Faction']` is present on a live
`model`, survives `reset_characteristics()`, and survives a save reload (the
rebase fix that went in with Monster Slayer). A model whose name is not in the
catalogue has no `Faction` at all and can never be hated — which is a
`rule_skipped` line, not a crash.

---

## 2. The first round of combat

**The engine has no per-combat round counter.** `chargedThisTurn` is cleared at
the end of every turn, so it cannot see round one of a combat that began on the
previous turn, and `startOfPhaseEngaged` answers a different question (was this
unit already fighting when the phase began, for pursuits).

The lifecycle to hang a counter on already exists:

- `units.py` — `enterInCombat` / `exitInCombat`, which already clear
  `isInCombatWith` and drop challenges;
- `game_fsm.py` — `enterCombatPhase`, which already walks every unit once per
  Combat phase to set `startOfPhaseEngaged`.

So: `roundsFought`, an int on the unit.

- incremented for every engaged unit in `enterCombatPhase`;
- reset to 0 in `exitInCombat`;
- saved and loaded, or a reload mid-combat quietly re-grants Hatred;
- Hatred applies while `roundsFought == 1`.

**Scope decision — the counter is per unit, not per pairing.** When a fresh
enemy charges a fight already in progress, the charger has just entered combat
and gets its Hatred; the unit already fighting does not. A per-pairing counter
would be a dict on every unit and in every save, and the rule's own wording
("the first round of combat") does not clearly ask for it. Recorded as a
LEFTOVER.

**This counter is worth more than Hatred.** Frenzy, Impetuous, and the several
rules phrased "in the first round of combat" all need exactly this, and it is
the reason to build it properly rather than inferring round one from the charge
flag.

---

## 3. The re-roll

`simulate_attack` in `battleFunctions.py`, the melee branch. Melee only:
shooting does not happen in a round of combat.

Two subtleties that are easy to get wrong:

- `rule['to_hit']` modifiers are applied to `attack_roll` *before* it is
  compared to the target, so a re-rolled die has to go back through the same
  modifiers rather than being compared raw.
- A re-roll is never itself re-rolled. Nothing else re-rolls a melee To Hit
  today, so the guard has to be deliberate rather than accidental.

Logged once per exchange with the numbers that decided it, never inside the
per-attack loop:

```
[Rule] Hatred — Dwarf Warriors: 4 failed hits re-rolled against Orc and Goblin
       Tribes in the first round of combat, 2 of them hit (p. 171)
```

and the negative case, which matters as much:

```
[Rule] Hatred — Dwarf Warriors: not claimed (round 2 of this combat)
[Rule] Hatred — Dwarf Warriors: not claimed (Skeleton Warriors are Vampire
       Counts, not a hated enemy)
```

---

## 4. Dead data to remove

Seven files under `army_units_cat/orc_and_goblin_tribes/` carry a bare
`"Hatred"` with no bracketed enemy: `night_goblin`, `night_goblin_bigboss`,
`night_goblin_warboss`, `night_goblin_oddgit`, `night_goblin_oddnob`,
`squig_hopper`, `mangler_squigs`. Every one of those models is in the
catalogue, which supplies `Hatred (Dwarfs)` and wins, so the entries are
unreachable. They are deleted rather than guessed at — a bare `Hatred` would
otherwise have to mean either "all enemies" or nothing, and both are wrong.

---

## 5. Tests

The one that earns its place, in the manner of the "every Killing Blow weapon
is flagged" test that turned up the melee-flag bug: **every distinct `Hatred`
spelling in the catalogue must resolve to a non-empty set of factions or
keywords.** A new army book adding `Hatred (Skaven)` then fails loudly instead
of silently hating nobody.

Beyond that:

- `Orcs & Goblins` is one faction, not two — the `&` trap;
- the compound targets, including the three-part one;
- both casings of "all enemies";
- the `Breyherds` typo;
- a target with no `Faction` at all;
- Chaos Dwarfs are hated by `Hatred (Dwarfs)`;
- the counter: 1 on the phase after engaging, 2 the phase after, 0 again once
  the unit leaves combat, and that it survives a save;
- the re-roll: only failures are re-rolled, only once, only in melee, only in
  round one.

## 6. Test save

`saves/hatred.json` — Dwarfs against Orcs & Goblins would be ideal but the
board is Empire against Undead, so the matchup is built from what is on it and
the log lines are what to read.

---

## Order of work

1. `HATRED_PLAN.md` (this file)
2. delete the dead bare-`Hatred` entries
3. `roundsFought` — units, phase, persistence
4. resolve `X` — alias table and matcher
5. the re-roll and its logging
6. tests
7. `SPECIAL_RULES_CHECKLIST.md`
8. `saves/hatred.json`

## Related rules this unblocks or touches

- `Ancestral Grudge` — `Hatred (enemy characters)`, so the matcher must accept
  categories as well as factions and keywords from the start.
- The Bretonnian Grand Master grants
  `Hatred (Warriors of Chaos, Beastmen Brayherds & Daemonic models)` by roster
  rule, spelled correctly there, which is a second route into the same matcher.
- Frenzy and Impetuous need `roundsFought`.
