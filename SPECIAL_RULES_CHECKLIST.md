# Special Rules — Implementation Checklist

Tracks weapon and unit special rules from the BattleScribe catalogues. Rules
carry only *keywords* in the data; the coded effects live in `special_rules.py`,
`battleFunctions.py`, `combat_resolution.py`, and `models.py`.

## Done
- [x] Armour Bane (X) — natural 6 to wound improves that attack's AP by X
- [x] Lance / charge-only melee (strength & AP bonus only while charging)
- [x] Charge-conditional AP (e.g. Halberd `-1 (-2)`)
- [x] Melee strength bonus (e.g. Halberd S+1 always, Lance S+2 on charge)
- [x] Furious Charge (registry builder)
- [x] Regeneration (registry builder; save value only if in data)
- [x] Unbreakable (registry builder)
- [x] Multiple Shots (N) — parsed (D3 approximated as 3)
- [x] Volley Fire — parsed from weapon rules
- [x] Equip best melee weapon in combat — models fight with their strongest
      applicable melee weapon (Lance only while charging) so its stats/hooks
      come from the equipped weapon, not a bare hand weapon
- [x] Armour save from equipment — the roster importer collects worn armour
      (Armour profiles: Light/Heavy/Full Plate, Shield, Barding) and the save
      is derived at unit creation (best body armour, -1 per shield/barding,
      capped at 2+). Shown on the hover panel; persisted in saves.
- [x] Break test, all three outcomes — natural roll > Ld Breaks, natural <= Ld
      but modified > Ld Falls Back in Good Order, otherwise (or on a natural
      double 1) Gives Ground. A losing side whose enemies total more than twice
      its Unit Strength is overwhelmed: a Fall Back result becomes a Break.
      `break_test_outcome` / `overwhelmed` in `psychology.py`.

## Weapon rules — TODO (high frequency)
- [ ] Multiple Shots (D3) — roll D3/D6+N instead of the hardcoded 3
- [ ] Move or Shoot — cannot shoot after moving
- [ ] Ponderous — move-or-shoot / initiative penalty
- [ ] Killing Blow — natural 6 to wound = no armour save (auto-kill)
- [ ] Heroic Killing Blow
- [ ] Strike First
- [ ] Strike Last
- [x] Requires Two Hands — disables the shield's +1 in melee (a two-handed
      weapon cannot also use a shield); melee_armour_save() drops the shield
      bonus when the active melee weapon has this rule. Shooting save keeps it.

## Unit rules — TODO (displayed, not yet applied)
- [ ] Impetuous
- [ ] Frenzy
- [ ] Immune to Psychology
- [ ] Fear
- [ ] Terror
- [x] Stubborn — see General rules below

## General (core `.gst`) rules — TODO
From scanning `strategy_armies/nr/dv.json` (Dwarf army) against the engine:
of 35 distinct rules only **Armour Bane** is coded. These core rules are
army-agnostic and would benefit every faction.

- [x] Stubborn — DONE (2024 wording, Rulebook p. 178): the first Break test a
      Stubborn unit is required to make may be refused, Falling Back in Good
      Order instead, even when the winning side's Unit Strength is more than
      twice its own. Once per battle (`usedStubborn`, persisted); the player is
      prompted, the AI decides via `should_use_stubborn`. Only the unit's own
      profile counts, so a joined Stubborn character neither confers nor uses
      the rule. NOTE: this is *not* the old "unmodified Ld" mechanic.
- [x] Fly (X) — DONE: flyers use their Fly Movement characteristic and pass
      freely over terrain (no difficult-terrain penalty, no forest-edge block)
      and over other units (unit-sweep skipped for flyers). Leftover: the
      end-of-move "not on top of a unit / within 1in of an enemy" restriction
      is not enforced.
- [x] General — Inspiring Presence (Ld bubble): the General is the character
      with the highest Leadership (an explicit `General` rule in the army list
      wins), nominated once per army load and never replaced when slain. Unless
      it is fleeing, friendly units within its Command range — a flat 12",
      18" with Large Target — test on its Leadership instead of their own
      (Break, Panic and Rally). Measured edge to edge from the General's own
      base, including when it has joined a unit. See
      `PsychologySystem.general_of` / `leadership_of` in `psychology.py`.
- [x] Battle Standard Bearer — DONE: nominated from the army list's `Battle
      Standard Bearer` keyword and never the General. Unless it is fleeing,
      friendly units in its Command range re-roll failed Panic and Rally tests
      and may re-roll a Break test's 2D6 (the second roll stands, even if
      worse), and its side gets +1 combat result — once, even with two
      bearers. `PsychologySystem.battle_standard_of` /
      `battle_standard_bonus` / `should_reroll_break` in `psychology.py`.
      NOTE: ordinary standard bearers are not modelled at all, so the Battle
      Standard is currently the only source of a combat result standard bonus.
- [ ] Hatred (X) — re-roll misses to hit in the first combat round
- [ ] Magic Resistance (-1/-2) — to-cast / ward penalty vs magic
- [x] Impact Hits (X) — DONE (Rulebook p. 172): the `(X)` is parsed off the rule
      name (`_param_dice` copes with prose such as `(D6+1, War Wagon only)`), and
      `impactHits` resolves them for every charging unit before any blows are
      struck, once per combat. Each model in base contact — the front rank —
      causes `X` automatic hits, so no To Hit roll is made; they wound with the
      unmodified Strength of the model that owns the rule, which is the mount
      for a rider and the chariot itself for a chariot. Armour and Regeneration
      saves apply as normal. The 3" condition needed the charge to record how
      far it actually moved (`chargeDistance`).
      LEFTOVER: no Armour Piercing (Crushing Weight gives AP-1/-2/-3),
      no Rank Bonus improvement, and no weapon-profile variants (Grinding
      Attacks, whirling blades).
- [x] Skirmishers — DONE (Phases 0–3 + the Phase 4 panic guard): rule flag, no
      rank bonus, enemy-fire -1, 360° arc, loose-blob layout, free 360° move with
      destination ghost, form-up/spread in combat, and fleeing Skirmishers no
      longer panic formed friendlies they flee through. Leftovers: true per-model
      coherency (one bodyNP), "see through gaps" LoS, >50%-visible charge gate,
      terrain nuance — see SKIRMISHERS_PLAN.md. Note: skirmisher status comes
      from the army list's special_rules (unit-level rule), not the base
      catalogue model profile.
- [ ] Scouts / Vanguard — pre-game deployment / free move
- [x] Swiftstride — DONE: a unit made entirely of Swiftstride models (the rule
      may come from the mount; a joined character without it breaks the unit's
      claim) adds 3" to its maximum possible charge range and may add a D6 to
      its Charge, Flee, Fall Back and Pursuit rolls. The bonus die is rolled in
      its own colour and is *added*, never one of the two a Charge or Fall Back
      roll discards between. `unit_has_swiftstride` / `max_charge_range` /
      `max_pursuit_range` / `charge_roll` / `should_use_swiftstride` in
      `special_rules.py`. Pursuit moves run through the charge machinery
      (`IsPursuing` -> `maxmove = 0`, `chdist = sum(chdice)`), so the bonus die
      is summed there rather than discarded.
- [ ] Move Through Cover — no difficult-terrain movement penalty
- [ ] Shieldwall — defensive bonus vs charges
- [ ] Resolute — strikes with full ranks when charged
- [ ] Veteran — Ld / re-roll bonus
- [ ] Rallying Cry — bonus to rally tests
- [ ] Close Order / Open Order / Dispersed Formation — formation modes
- [ ] Gromril Weapons — hand weapon with AP -1 in melee
- [ ] Detachment — list-building support (may not need a runtime effect)

## Dwarf-specific (`Dwarfen Mountain Holds.cat`) rules — TODO
- [ ] Ancestral Grudge — Hatred (enemy characters)
- [ ] Gromril Armour — re-roll natural 1s on armour saves
- [ ] Stoic Defenders — +1 Initiative & Attacks when charged
- [x] Venerable — DONE: friendly units within 6" of a Venerable unit (edge to
      edge, the same bubble as nearby-friend Panic) re-roll failed Panic tests.
      The Venerable unit benefits itself; a fleeing one inspires nobody. See
      `PsychologySystem.venerable_source` / `leadership_test_with_reroll` in
      `psychology.py`.
- [ ] Runes of Warding — 5+ ward vs Flaming Attacks
- [ ] Rune Lore — may attempt a Wizardly Dispel
- [ ] Forgefire — joined unit gains Armour Bane (2) + Flaming Attacks
- [ ] Dwarf Crafted — no -1 To Hit on a Stand & Shoot reaction
- [ ] Fire & Flee — shooting-unit flee reaction
- [ ] Dive Bomb — once-per-game flyer attack
- [ ] Borne Aloft — Shieldbearers (4 models on one base)
- [ ] Royal Guard — army-list allowance (list-building, not runtime)

## Loose ends
- [ ] Test/CI hardening; broaden `tests/` to a couple of full factions
- [ ] Empire units render with the generic model (no `.bam`) — add mappings
- [x] Non-numeric stats (e.g. Giant A="*", a chariot's WS="-") no longer break
      the combat maths: `stat_value()` in `toHitAndToWound.py` reads them as 0,
      which is what the rules mean by them (Rulebook p. 97), and WS 0 follows
      p. 158 — its attacks all miss, attacks against it hit automatically.
      To Hit/To Wound used to return an error *string*, which then blew up on
      the `>=` comparison.
- [x] Chariot split profiles (Rulebook p. 194) — the catalogue marks the crew
      `subType="crew"` and the beasts with a CHARIOT CREW category link, and
      the unit profile gives the troop type. `battlescribe.py` now also follows
      a unit's `entryLinks` to sibling model entries (which is how chariots are
      declared, and why they had no troop type before), and records `Crew` /
      `Beasts`, each with the count its selection constraints fix (a War Wagon
      takes exactly 6 crew and 2 horses). `models.py` attaches them: enemies
      roll To Hit against the crew's WS (`defending_ws`), the chariot moves at
      its beasts' Movement, Toughness/Wounds stay on the chariot, and the crew
      and beasts each fight with their own WS/S/A at full count while the
      chariot itself has no Attacks (`CombatResolver.chariotParts`).
      LEFTOVER: Impact Hits use the chariot's Strength but have no AP.
- [x] Multi-wound models \u2014 `MovementSystem.applyWounds` converts unsaved wounds
      into slain models using the profile's Wounds, keeping the remainder on the
      wounded model (`woundsOnModel`, persisted). Combat and shooting passed
      their wound totals straight to `removeModelsFromUnit`, which counts
      *models*, so a single wound destroyed a 6-Wound War Wagon.

## Deferred war-machine items
- [ ] Multiple Wounds (D3+1) generic rule
- [ ] Black Powder / Misfire tables
- [ ] True line-of-sight and rank/file hit caps for cannon/bombardment
