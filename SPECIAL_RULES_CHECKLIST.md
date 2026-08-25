# Special Rules — Implementation Checklist

Tracks weapon and unit special rules from the BattleScribe catalogues. Rules
carry only *keywords* in the data; the coded effects live in `special_rules.py`,
`battleFunctions.py`, `combat_resolution.py`, and `models.py`.

**Every rule that fires has to say so.** Use `rules_log.rule_log` when a rule
changes an outcome and `rules_log.rule_skipped` when one could have applied but
did not, carrying the numbers that decided it. Nothing in this engine is
visible on screen, so a rule that works and a rule that was never coded look
identical without the log. See `.github/copilot-instructions.md`.

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
- [ ] **Necromantic Undead / Nehekharan Undead** — every undead unit carries one
      of these and neither does anything. It matters now: deleting the
      hand-written model subclasses took away the invented `Fearless`
      (`Unbreakable: True`) they had been leaning on, so Zombies, Skeletons,
      Dire Wolves, Black Knights, Crypt Ghouls and Grave Guard take Break tests
      and flee like anything else. Losing Unbreakable is correct — undead do
      not have it in this edition — but the rule that replaces it, crumbling,
      is not coded, so at the moment they have nothing at all in its place.
      First thing to fix before playing an undead army.
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
      LEFTOVER: no Rank Bonus improvement, and no weapon-profile variants
      (Grinding Attacks, whirling blades). Armour Piercing is done for a heavy
      chariot's Scythed Wheels; Crushing Weight still has no effect.
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

## Troop Types in Detail (Rulebook p. 188-197)

A troop type's rules are the one thing the catalogue never states: nothing in
the `.cat`/`.gst` mentions Parry or Lumbering, and a model is meant to have
them purely because its Troop Type reads "Regular Infantry" or "Heavy Chariot".
`troop_types.py` is that missing table, and all thirteen rows are now in it —
532 of the 533 model entries in the catalogues resolve (the one that does not
is a `Special Feature`, which is terrain).

Five main categories, each with sub-categories. A sub-category follows its
parent's rules unless it states otherwise, so "infantry" in a rule means
monstrous infantry and swarms too.

| Troop Type | Models/Rank | Max Rank | US | Special Rules |
| --- | --- | --- | --- | --- |
| Regular Infantry | 5 | +2 | 1 | Press of Battle, Massed Infantry, Parry |
| Heavy Infantry | 4 | +2 | 1 | Steady in the Ranks, Press of Battle, Massed Infantry, Parry |
| Monstrous Infantry | 3 | +2 | 3 | Clumsy |
| Swarms | - | - | 3 | Insignificant, No One Cares, Undisciplined |
| Light Cavalry | 5 | +1 | 2 | Split Profile (Cavalry), Cavalry Support |
| Heavy Cavalry | 4 | +1 | 2 | Split Profile (Cavalry), Cavalry Support |
| Monstrous Cavalry | 3 | +1 | 3 | Split Profile (Cavalry), Clumsy |
| War Beasts | 5 | +1 | 1 | Undisciplined |
| Light Chariots | 3 | +1 | 3 | Split Profile (Chariots), Iron Shod Wheels, Churning Wheels, Firing Platform |
| Heavy Chariots | - | - | 5 | Split Profile (Chariots), Scythed Wheels, Lumbering, Iron Shod Wheels, Firing Platform |
| Monstrous Creatures | - | - | =W | Ridden Monster, Lumbering |
| Behemoths | - | - | =W | Ridden Monster, Lumbering, Thunderstomp |
| War Machines | - | - | =W | Split Profile (War Machine), "We're Not Paid to Fight", Weapon of War |

### Phase 1 — fill in the table — DONE
- [x] All thirteen rows in `TROOP_TYPES`. Before this only the two chariot
      rows existed, so **every other unit in the game had the wrong Unit
      Strength and no rank cap**: monstrous infantry counted as US1 instead of
      US3, cavalry claimed +2 Rank Bonus instead of +1, a Giant was US1.
      That feeds Overwhelmed, nearby-friend Panic and pursuit routs.
- [x] Unit Strength "As Starting Wounds" for monsters and war machines. The
      table cannot hold a number for these, so `AS_STARTING_WOUNDS` is a
      sentinel and `unit_strength(troop_type, default, wounds)` resolves it
      against `model.starting_wounds()` — read from the pristine profile, so
      a wounded Giant is still US6.
- [x] Models per rank now means something. `psychology.rank_bonus` hard-coded
      "a trailing rank of fewer than 4 does not count", which was a stand-in
      for this very column. It reads `models_per_rank` per troop type, and a
      unit narrower than its own models per rank claims no Rank Bonus at all,
      which is what the rulebook's table means by that number.
- [x] Singular/plural aliases — the catalogue writes 'War beast' and 'War
      beasts', 'Monstrous creature' and 'Monstrous Creatures'.

### Phase 2 — rules with a hook already in place
- [x] Parry — DONE (p. 190): a model with the rule fighting with a hand weapon
      *and* a shield improves its armour value by 1, to a maximum of 3+.
      `model.parry_applies()` gates on the troop type, a shield in the armour
      list and the *equipped* melee weapon being a plain hand weapon;
      `melee_armour_save()` applies `min(save, max(3, save - 1))`, so a 2+ is
      never made worse and a 3+ gains nothing. Shooting is untouched — it uses
      the stored `armor_save`. A two-handed weapon already dropped the shield
      entirely, so it cannot parry either.
      Regular and heavy infantry have it, which is most of the game: goblins in
      light armour and shield went from a 5+ to a 4+ in melee, and two existing
      tests had to be re-baselined because of it.
      Logged once per exchange rather than once per save roll, and the negative
      case is logged too — a unit that could parry but swung a great weapon
      instead says so, because that is the question a player actually asks.
      The weapon choice now decides the stats: see "the equipped weapon is the
      one you fight with" under Loose ends.
- [x] Massed Infantry — DONE (p. 190): a side with the *higher* total Unit
      Strength that includes at least one unit with the rule claims +1 combat
      result. Both halves are needed — numbers without the infantry, or the
      infantry without the numbers, claim nothing, and an equal Unit Strength
      is not "higher". Worth one point however many such units are present.
      `psychology.massed_infantry_bonus` / `side_unit_strength`; a destroyed
      unit contributes nothing to the total.
      The combat result is now printed as a table instead of
      `Player 2 score: 3, Player 1 score: 2`, itemising wounds, Impact Hits,
      flank/rear, Rank Bonus, Battle Standard and Massed Infantry for both
      sides, with each side's Unit Strength beneath the total — because that is
      what this rule turns on. Zeroes are shown too: after a lost combat the
      question is usually which bonus the *other* side had.
- [ ] Undisciplined — cannot use the General's Inspiring Presence nor the
      Battle Standard's "Hold Your Ground". A gate in
      `PsychologySystem.leadership_of` / `battle_standard_of`. Swarms and war
      beasts.
- [ ] No One Cares — swarms never cause Panic in friendly units, whatever
      happens to them. A gate in the nearby-friend-destroyed and
      flees-combat panic causes.
- [ ] Clumsy / Churning Wheels — a unit with the rule may only be joined by a
      character that also has it. Both are one check in `characters.join_unit`,
      which already refuses some joins.
- [x] Press of Battle — DONE (p. 190). `simulate_battle` was already giving the
      rank behind the front rank one attack per model, but it gave it to
      *everyone*, which is Press of Battle applied universally. The maths moved
      into `battleFunctions.extra_rank_attacks`, which grants it only to a
      troop type that has the rule (regular and heavy infantry). The "except on
      the turn it charged" exception needed no code: the extra ranks were only
      ever counted in the non-charging branch.
      Gating on the troop type alone would have silently taken the extra rank
      away from spearmen, so the weapon half went in with it: the .gst gives a
      thrusting spear and a cavalry spear the `Fight in Extra Rank` rule ("a
      model with this special rule may make a supporting attack", p. 169), read
      by `model.fights_in_extra_rank()`. That deliberately reads `equipedWeapon`
      and not `active_melee_weapon()` — a cavalry spear is charge-only for
      Strength and AP, but its extra rank works the other way round, being
      denied on the turn the wielder charged, so the usual charge-only fallback
      to bare hands would have been backwards here.
      **The two rules stack rather than overlap**, which is the whole point of
      keeping the fighting rank and supporting attacks apart. Press of Battle
      deepens the *fighting rank* to two ranks; Fight in Extra Rank lets the
      rank directly behind the fighting rank support it, and p. 145 bars a
      model that is itself in a fighting rank from making a supporting attack.
      So the spear rank is pushed back to the third rank rather than absorbed
      into the second, and infantry with thrusting spears fight three ranks
      deep: 20 models five wide make 5 full-Attack + 5 + 5. Cavalry with a
      cavalry spear have no Press of Battle, so their spears support from the
      second rank and they fight two deep. A first pass collapsed both rules
      into a single second rank and undercounted spear-armed infantry by a
      whole rank.
      None of these models are in base contact, so they attack once each
      whatever their Attacks characteristic says (p. 146) — which is what the
      old code did, so the numbers only changed for units whose rank claim was
      wrong.
      LEFTOVER: a polearm also denies a supporting attack on the charge turn,
      but the .gst carries no `Fight in Extra Rank` on it and its own note is
      prose only, so a polearm currently grants no extra rank at all.
      LEFTOVER: the "within a number of inches equal to its Movement of the
      enemy unit" clause is unmodelled; the extra ranks are always assumed
      close enough. Base-to-base geometry is not tracked at this level.
      LEFTOVER: supporting attacks cannot be made against a flank or rear, and
      Press of Battle needs the unit to be in Combat Order. Neither is checked
      — combat facing does not reach `simulate_battle`.
- [x] Stepping Forward — DONE (p. 102 and p. 150). Fell out of Press of Battle
      and was **the opposite of what the engine did**. Casualties are removed
      from the back of a unit, and the code took that literally: the front rank
      always swung at full strength and the casualty count was subtracted from
      the *supporting* rank. Removing from the back is only bookkeeping —
      "models removed as casualties before having a chance to attack, and
      models that stepped forward during the current phase, cannot attack"
      (Set Casualties Aside, p. 150).
      The clause that decides the arithmetic is the combat-phase one: a model
      cannot attack in a phase in which it *stepped forward into the fighting
      rank*. The slain model is already gone from `nmodels`, so a casualty
      costs a **second** attacker only where a model behind the fighting rank
      stepped into the gap. A unit no deeper than its own fighting rank has
      nobody to step forward and its fighting rank simply narrows: 10 Jade
      Warriors five wide are two ranks, both of them fighting rank under Press
      of Battle, so losing one leaves nine models and nine attacks. A first
      pass deducted the casualty from the rank as well as from `nmodels` and
      reported eight — caught from a game log.
      A deep unit pays twice over, which is the point of the rule: 20 State
      Troopers five wide that lose two answer with 8 attacks, not 10, because
      two models from the rear ranks are clambering over the fallen.
      `battleFunctions.melee_attacks` is now the single place attack counts are
      worked out, for chargers and defenders alike, and it takes the steppers
      off the front rank, then the Press of Battle rank, then supporting
      attacks. Losses come off a whole model at a time, so a two-Attack model
      that steps forward costs the unit both of its attacks.
      NOTE: which part of a two-deep fighting rank the steppers land in is not
      spelled out. Filling the front rank first is the reading used here,
      because that is where models are in base contact and where the rulebook
      says the casualties fall; it is the harsher of the two readings.
      LEFTOVER: the cascade is deliberately not modelled — a model shuffling
      from the fourth rank into the third has also stepped forward by the
      wording on p. 102, but the combat rule speaks only of stepping into the
      fighting rank, so the supporting rank is left alone.
      LEFTOVER: Simultaneous Combat (p. 146) says casualties do *not* reduce
      the attacks of enemy models with the same Initiative value. The engine
      has no Initiative ordering at all — the charger always strikes first and
      always thins the defender. That is usually right by accident, since a
      charge grants +1 Initiative per inch moved to a maximum of +3 (p. 146),
      but a defender with equal or higher Initiative should be striking back at
      full strength and currently does not.
- [ ] Cavalry Support — when a cavalry model makes a supporting attack, only
      the rider attacks, not the mount. Needs the supporting-attack maths in
      `simulate_battle` to know rider from mount, which `get_mount()` gives it.
      `extra_rank_attacks` is now the one place this has to happen.
- [ ] Dividing Attacks / Fighting on Multiple Fronts (p. 147) — a unit engaged
      in more than one of its arcs has **more than one fighting rank**: its
      front rank and the file engaged in the flank both fight, and each model
      attacks an enemy it is in base contact with.
      The engine resolves combat strictly pairwise — `attackers[i]` strikes
      `defenders[i]` — and a unit appears in that list once, so a unit fought
      on two fronts strikes one enemy at full strength and the other not at
      all. Seen in play once Pursuit into a New Combat started putting a second
      unit onto an already-engaged enemy: Jade Warriors engaged to the front by
      Longbeards and in the flank by Hammerers made all 10 attacks against the
      Hammerers and none against the Longbeards.
      Predates this work — any multiple-unit combat has it — but it was hard to
      notice before, because the second front usually arrived by charge in the
      Movement phase and both fights were fought separately.
      `battleFunctions.melee_attacks` takes one unit and one charge flag, so it
      would need to know the facings engaged; the combat facing does not reach
      it today, which is the same blocker the Press of Battle leftovers name.

### Phase 3 — rules that need something built first
- [ ] Steady in the Ranks — heavy infantry in Close or Open Order is not
      Disrupted by a flank or rear engagement unless the enemy has US 10+.
      BLOCKED: the engine's `isDisrupted` only ever means "a quarter of the
      models are in difficult terrain"; **flank/rear Disruption does not exist
      at all**, which is a missing core rule in its own right.
- [ ] Thunderstomp — a behemoth's Stomp Attacks have AP -2, and cannot be used
      against another monster. BLOCKED: Stomp Attacks are not implemented.
- [ ] Iron Shod Wheels — treats difficult terrain as dangerous, treats linear
      obstacles as impassable, and loses D3 Wounds on a failed Dangerous
      Terrain test. The D3 half is ready (`dangerousTerrainTests(damage='D3')`);
      the difficult-as-dangerous half is a small change to `crosses_difficult`
      / `dangerous_between`; the obstacle half is BLOCKED on linear obstacles.
- [ ] Insignificant — line of sight is drawn across a swarm as if it were not
      there, and swarms are ignored when targeting enemy characters. Needs the
      unit-blocking half of line of sight, which `markHillTargets` and the arc
      clipping only half model.
- [ ] Split Profile (Cavalry) — rider and mount each use their own WS/BS/S/I/A
      and weapons; enemies roll To Hit against the *rider's* WS; Impact Hits
      and Stomp use the mount's Strength; the armour save uses the rider's
      value; the model dies when the rider does. The chariot split profile
      already does the equivalent (`defending_ws`, `CombatResolver.chariotParts`),
      so this is largely reuse — but it changes every mounted unit in the game,
      so it wants its own pass.

### Phase 4 — war machines
Their three rules are a self-contained block and would suit being done with the
existing `cannon_fire.py` / `bombardment.py` work.
- [ ] Split Profile (War Machine) — crew's Toughness and Wounds in combat, the
      machine's when not; -1 Attack per Wound the crew has lost; armour save
      from the crew; either element at zero Wounds removes the model.
- [ ] "We're Not Paid to Fight" — a war machine that Breaks and flees from
      combat is destroyed outright. Fall Back and Give Ground are normal.
- [ ] Weapon of War — cannot march, declare a charge or pursue; -1 to any Flee
      roll (minimum 1); may pivot freely about its centre immediately before
      shooting without that counting as moving; may follow up as normal.

### Not part of this section
Ridden Monster is listed for monstrous creatures and behemoths but is defined
under Characters, so it belongs with that work rather than here.

## Post-Combat: Break Test, Follow Up & Pursuit (Rulebook p. 144-157)

The fourth sub-phase of a combat, and the one the engine had least of. The
rulebook resolves it in four passes over the *whole* combat (p. 156): every
Break test is made, then every winning unit declares what it will do and which
losing unit it is going after, then the losers move, and only then are the
pursuit moves made, one at a time.

Much of the pursuit *move* was already right, in a place that is easy to miss:
a pursuit is resolved through the charge machinery (`chargeInterval`, with
`maxmove = 0` and `chdist = sum(chdice)`), which is where its own 2D6 is rolled,
where Swiftstride's die is added rather than discarded, and where the wheel, the
align and the contact test happen. That is also where a fleeing unit that is
caught is run down. Twice during this work a rule was called missing when the
charge path was quietly handling it — check there before believing an absence.

### Done
- [x] The four passes — `breakTestPass`, `declarePass`, `loserMovePass`,
      `pursuitPass` in `combat_resolution.py`. The engine used to do all four
      per losing unit inside one loop, so in a combat with two losing units the
      first fled and was pursued before the second had taken its Break test.
      A side effect worth naming: nearby-friend Panic is measured "before it
      moves" (p. 161), which is only true now that no loser moves until every
      Break test is made.
- [x] `post_combat.py` — the arithmetic, with no Panda3D in it, so it tests
      without a window: `flee_roll` / `fall_back_roll` / `pursuit_roll`,
      `flees_from`, `flee_direction`, `give_ground_direction`, `restraint_test`,
      `winner_response`, `catch_outcome`, `may_pursue`. `fall_back_roll`
      delegates to `special_rules.charge_roll` rather than keeping a second copy
      of "2D6 discard the lowest", which is the same arithmetic.
- [x] Flee direction — a Break or a Fall Back runs directly away from the
      winning unit with the **highest Unit Strength**, chosen at random between
      equals (The Greater the Danger, p. 133; p. 154). Every outcome used to use
      `fleeDirectionMultUnits`, which averages the direction away from all the
      winners. That function is still right for Give Ground, which really does
      move as directly as possible away from *all* of them (p. 155), so the two
      are now separate functions instead of one shared by accident.
- [x] Restrain & Reform is a **Leadership test** (p. 156), not the free choice
      the menu offered: electing to hold back can fail and force the move.
- [x] Still Engaged (p. 156) — a winner still in base contact with another enemy
      does not follow up or pursue.
- [x] Surrounded (p. 155) — a Give Ground with nowhere to go reports itself and
      stays locked rather than silently moving 0".
- [x] Declaring a target — with more than one losing unit the winner is asked
      which it is chasing, before any Flee roll is made (p. 156).
- [x] Three ~90-line copy-pasted coroutines replaced by `giveGroundMove`,
      `fleeMove` and `pursuitMove`, plus `rollMoveDice` for the dice loop that
      existed in triplicate. Removed with them: dead `if x: pass` branches, and
      a variable `FBIGFromCombat` printed for every pursuer while only ever
      assigning it for the loser — which is why two units used to report the
      same dice, one of them the unit that was running away.
- [x] Wording: the menu said "Persuit" whether the answer was a Follow Up or a
      Pursuit, and a pursuit that did not reach reported "Charge fell short".
      A pursuit that does not reach is not a failure; the unit still moves its
      full roll and halts.

### Phase 3 — catching — DONE
- [x] The free reform a pursuer may attempt after running an enemy down
      (p. 157). `freeReform()` wraps the existing interactive reform in
      something awaitable so the sequence does not run on while the player is
      still placing the unit; the AI declines, having no way to answer.
      Reforms queue on `reformInProgress`. Two can fall due at once and did in
      play: a pursuer that catches its quarry reforms from the charge task,
      which `pursuitPass` does not await, so the pass had already moved on and
      started the next unit's reform. Both prompts were live together and the
      player could only place one. Awaiting one caller is not enough when the
      other reform is reached down a task the caller never waits for.
- [x] "During the next turn, the pursuing unit counts as having charged"
      (p. 157). Catching a unit that Fell Back set `chargedThisTurn`, but
      `exitCombatPhase` clears that at the end of the very phase it was set, so
      the claim was gone before the combat it applies to was fought.
      `countsAsChargedNextTurn` carries it over and is promoted at the end of
      the phase, taking `chargeDistance` with it so Impact Hits keep their 3"
      test. Persisted, and `test_persistence` now derives its flag list from
      `units.py` so the next one cannot be forgotten.

### Phase 4 — the rest, cheapest first
- [x] The free reform on a *passed* Restraint test (p. 156) — the unit held its
      ground but was not offered the reform the rule grants it. The reform is
      taken in pass 4 with the other post-combat moves, not at the moment the
      unit declares in pass 2: a declaration is not a move, and until the loser
      has drawn off the two are still nose to nose with no room to turn in.
- [x] The Limits of Endurance (p. 133) — one flee move per phase; a second is 0"
      and does not pivot. A Fall Back "moves exactly like a fleeing unit"
      (p. 134), so it spends the allowance too. Found in play: a unit that lost
      a combat, fell back 6", then failed a Panic test from a nearby friend
      fled a second time on a fresh 2D6. `fledThisPhase` is reset at every phase
      entry beside the Panic allowance, persisted, and folded into
      `flee_roll` / `fall_back_roll` so the distance is decided in one place.
      Swiftstride's die is not offered for a move that cannot go anywhere.
- [x] 1" Apart (p. 154) — `nudgeOneInchApart` pushes a unit that Broke or Fell
      Back the smallest distance that leaves it an inch clear. The existing
      `fallBackContactTest` only resolves overlap, which leaves the two
      touching; this measures edge to edge with `obb_distance`, the same
      oriented-box maths the Leadership bubbles use.
- [x] Overrun (p. 156) — a unit that destroys its enemy outright, before the
      Break test sub-phase, may attempt to restrain and reform or overrun: a
      normal pursuit move (2D6 summed, Swiftstride's die added) but directly
      forwards and without pivoting. `overrunPass` runs before the Break tests,
      because a unit whose enemy is gone takes no part in them; it fires for a
      winner whose `isInCombatWith` still lists units but all of them are
      destroyed. Restraining is the same Leadership test as anywhere else, so
      electing to hold back can fail and force the overrun.
      The move sweeps first, which covers Pursuit into an Obstacle (p. 157) for
      this case: it stops at a friendly unit or impassable terrain rather than
      running through. Leaving `InCombat` clears the stale foe list through the
      FSM's own `exitInCombat`, so nothing has to unpick it by hand.
      The heading maths moved to `post_combat.facing_vector` to be tested: a
      flipped sign would send the unit backwards through the enemy it had just
      destroyed, which nothing but a game would have caught.
      An overrun *is* a normal pursuit move, so `overrunContact` resolves what
      it ran into by the pursuit cases on p. 157 rather than just halting. The
      sweep only reports how far it got, so what was hit is read afterwards
      from the contact: a friend or terrain stops it, a fleeing enemy is run
      down and a free reform offered, and a fresh enemy is engaged with
      `countsAsChargedNextTurn`.
      LEFTOVER: no wheel to maximise contact before the align.
- [x] Pursuit into a Fresh Enemy (p. 157) — a pursuer that reaches an enemy
      other than its quarry counts as charging *that* unit, and the quarry is
      not caught. The move itself was already right and had been since the
      charge machinery was reused for pursuits: `moveUnit` hands a pursuer's
      contact to `chargeAndChargeReaction` whatever it hit, so `chargeInterval`
      engages whoever is actually there and never touches the quarry. What was
      wrong was the report — it announced `Catching the Curs!` and claimed to
      have caught a unit "which fell back", naming the wrong unit and the wrong
      rule. `chargeInterval` cannot see past the contact it was handed, so
      `pursuitMove` leaves the quarry on `unit.pursuitQuarry` for it to compare
      against, and the two cases now log apart. This is the third time a rule
      turned out to be handled by the charge path already and only the logging
      was missing.
      Done for an overrun too, in `overrunContact`.
      Both paths pivot to align about the point of contact — for an overrun,
      the **corner that struck**, which `contactPointOn` finds as the corner of
      its own base nearest the enemy's. Pivoting about the point the two bases
      share leaves them sharing it: measured across three angles the gap after
      the pivot equals the gap before it, to three decimal places, and the
      pivot itself drifts 0.000.
      Four wrong turns before that, all of them mine and all from reasoning
      about geometry in prose instead of measuring it: a stale `playerNP`; then
      "back off, turn, drive straight in", which squares the unit up but slides
      it somewhere it never touched; then a pivot on the *middle* of the struck
      edge, which swings the touching corner away and opens a gap; then a
      closing slide to take up that gap, which was patching a symptom of the
      pivot being wrong. `getHalfExtentsWithoutMargin` was wrong too — Bullet
      keeps the box shrunk by the margin, so `WithMargin` is the true size.
      `tests/harness_align.py` is what settled it: two real Bullet unit boxes
      offscreen, the real `contactPointOn`, and printed numbers for pivot
      position, pivot drift and the gap either side of the turn. It found the
      margin inversion on its first run. Reach for it before arguing about
      what the board looks like.
      LEFTOVER: no pivot to *maximise contact* before aligning, on either
      path — the unit engages wherever it happened to touch.
- [x] Pursuit into a Fleeing Enemy (p. 157) — run down exactly as if caught by
      a charging unit, then the pursuer may reform. `chargeInterval`'s Catching
      the Curs! branch already did this for any fleeing unit it touched; it now
      says which rule ran it down and notes when the quarry got away. The
      overrun path shares the removal through `removeUnitFromPlay`.
- [x] Pursuit into an obstacle (p. 157) — stop on contact with a friendly unit
      or impassable terrain. The friendly half was happening silently in
      `chargeInterval`'s "both units belong to Player N" guard, which reads as
      a rejected charge rather than a rule; it logs as one for a pursuer now.
      LEFTOVER: impassable terrain does not stop a pursuit — the sweep the
      overrun uses is not on this path, which moves through
      `pathTowardsMouse`/`moveUnit`.
- [x] Pursuit into a New Combat (p. 157) — both branches. If the enemy reached
      was already engaged when the phase began *and* that combat has not been
      fought yet, the pursuer joins it, fights again in it counting as charged,
      and may not pursue out of it — restraining and reforming with no
      Restraint test. Otherwise the two are locked until the next turn, carried
      by `countsAsChargedNextTurn`.
      The snapshot it needs turned out to be half-built already:
      `hasAttackedThisTurn` is set on every unit in a combat as that combat is
      resolved, so it doubles as "this combat has been fought". Only the other
      half was missing — whether the enemy was engaged *before* the phase
      began, which pursuits themselves change — so `startOfPhaseEngaged` is
      taken in `enterCombatPhase` beside `startOfPhaseModels`. Both halves are
      in `joinsCombatThisPhase`.
      "Cannot pursue again this turn" is `cannotPursueThisTurn`, cleared in
      `exitCombatPhase` with the other per-turn charge flags and persisted.
      The gate sits in `restrainChoice`, which both the pursuit and the overrun
      declaration go through, so neither can slip past it; a Follow Up is not a
      pursuit and is left alone.
      Corrected on the way: the first pass logged this rule as *skipped* every
      time a pursuer reached a fresh enemy, which contradicted the line above
      it — that line announces the locked-together outcome, and that outcome
      *is* this rule's "Otherwise" branch. A rule reported as not claimed when
      it had just fired is worse than no line at all.
      Corrected once the branch actually fired in a game: the pursuer joined
      the combat and was then skipped when it was fought, because the attack
      loop passes over anyone with `hasAttackedThisTurn` and it had spent that
      winning its own combat. Joining gives the allowance back, which is what
      "will fight again when that combat is fought" asks for. The join happens
      after the current combat's attack loop has run, so returning it cannot
      make the unit strike twice in the fight it has just won.
      SUSPECT: `hasAttackedThisTurn` is a rough proxy for "this combat has been
      fought". `self.game.attackers` is built from the chosen defender's whole
      engagement list, so resolving one combat marks every unit connected to it,
      including units whose own fight has not been chosen yet. It errs towards
      the locked-until-next-turn branch, which is the safe way to be wrong, but
      a proper "combats fought this phase" record would be better.
      All three declining conditions log which one it was, because they look
      identical on the board and the first game after this went in could not be
      read to find out. That paid for itself immediately: the rule was
      declining because the engagement had been made with the debug `e` key,
      which set `isInCombat` and both combat lists but not
      `startOfPhaseEngaged`, so a hand-made combat read as brand new. The debug
      engage/disengage keys maintain it now — a tool that fabricates state has
      to fabricate all of it, or it tests something other than the game.
- [x] Peril tests (p. 133) — a D6 for each model that flees *through* an enemy
      unit, losing a Wound on a 1-3, with no limit on how many a single move
      calls for. `perilTests` winds each model node back by the move's
      displacement to get the path it swept, and
      `post_combat.segment_crosses_box` says whether that path went through an
      enemy's footprint — the enemy's own facing turns the box, so a unit
      presenting its flank is a different obstacle from one facing the runner.
      A model that ends up inside counts as having gone through.
      The same page moves the 1" rule off the unit it was fighting and onto
      *any* enemy: a flee move that ends within 1" of one carries on until it is
      clear. `nudgeOneInchApart` measured against `isInCombatWith` alone, which
      is both too narrow and empty by the time a unit has fled, so it reads
      every enemy on the board now (`enemiesOf`).
      The negative case is deliberately *not* logged. Crossing nobody is not
      the rule declining, it is the rule never being reached, and a line on
      every flee in the game would bury the ones that matter.
      Both flee paths run them: the post-combat `fleeMove`, and the Panic and
      charge-reaction flees in `psychology._start_flee_move`, which calls
      through `game.combat` rather than growing a second copy. A Give Ground is
      excluded — it is 2" backwards and runs through nobody. A Fall Back is
      not, because it "moves exactly like a fleeing unit" (p. 134).
      A unit can be wiped out by its own Peril tests, so the flee sequence
      checks for that before rallying it or offering it a reform.
- [ ] Pursuit off the Battlefield (p. 157) — removed but *not* destroyed,
      returning in the next Compulsory Moves sub-phase as reinforcements.
      BLOCKED: no reinforcement mechanism exists.
- [ ] Surrounded, the rest of it (p. 155) — the units stay locked, but "fight
      another round exactly as if the combat had been a draw" is not modelled.
- [ ] Give Ground moves 1.9", not 2" — `crashFraction` multiplies by 0.95 even
      on a clear path, as a margin against immediately re-contacting. Predates
      this work.

### Not in this section
Whether a unit that Falls Back in Good Order panics its friends was checked and
is correct as coded: the rulebook is explicit that it does, because "amidst the
clamour of battle, friendly units are seldom able to tell the difference"
(p. 161).

## Loose ends
- [ ] Test/CI hardening; broaden `tests/` to a couple of full factions
- [ ] Empire units render with the generic model (no `.bam`) — add mappings
- [x] The equipped weapon is the one you fight with — `melee_strength_bonus`,
      `melee_ap` and `armour_bane_for_attack` all said "the active melee
      weapon" in their docstrings and then looped over *every* melee weapon the
      model owned, taking the best. So a State Trooper carrying a halberd swung
      at the halberd's S+1 and AP-2 even when the player picked the hand weapon
      from the combat menu, and the report printed the equipped weapon's name
      beside the best weapon's numbers — which is how it was spotted.
      All three now read `active_melee_weapon()`, which is the equipped weapon,
      or bare hands if that is ranged or a charge-only weapon out of a charge.
      Seven tests had encoded the old behaviour by giving a model a weapon and
      never equipping it; they equip now, as combat does.
      This also settles Parry's "the player chooses" clause: choosing the hand
      weapon actually means fighting with it.
- [x] Casting allowance survives a save — `spellsCastThisTurn` and
      `cannotCastThisTurn` were the only per-unit turn flags `persistence.py`
      never wrote, so loading left whatever the running session had: a spell
      attempted *after* the save was still marked as attempted, and a Wizard
      that had failed to cast could not try again. `tests/test_persistence.py`
      now asserts every turn flag is written.
- [x] Whose side is a joined character on — `join_unit` takes a character out
      of both player lists (it keeps `_player` for the save), so any code that
      answered that question by list membership got it wrong for exactly the
      models that cast most of the spells. A joined Wizard was treated as being
      on player 2 whatever its actual side, which made its Pillar of Fire sweep
      over the real enemy and burn nobody, and had the Dispel offered to its own
      side. `characters.side_of` / `friendly_units` / `enemy_units` resolve it
      through `_player`, then through the host unit, and the spell and the
      Dispel both go through them now.
      LEFTOVER: about forty other sites still ask by membership
      (`combat_resolution`, `movement_system`, `psychology`); they are all
      reached with host units rather than joined characters, so none is known
      to be wrong, but they would be better off using the helper.
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
      The crew also shoot with their own Ballistic Skill and Strength
      (`firing_bs` / `shooting_strength`) — a chariot's own BS is '-', which
      reads as 0, and `to_hit_ranged` rejects BS 0 outright, so a War Wagon
      fired its blunderbuss every turn and could never hit with it.
      LEFTOVER: one weapon per unit, so the crew cannot each carry their own —
      a War Wagon's 6 crew take exactly 6 different weapon upgrades in the
      catalogue, 3 of them missile weapons; and "special rules that apply to
      one element apply to the others".
- [x] Troop types (Rulebook p. 194-195) — `troop_types.py`. A troop type's rules
      are the one thing the catalogue never states: nothing in the .cat/.gst
      mentions Scythed Wheels, Lumbering, Iron Shod Wheels or Firing Platform,
      and a model is meant to have them purely because its Troop Type reads
      "Heavy Chariot". The table supplies models per rank, maximum rank bonus,
      Unit Strength and the implied rule list, normalising the data's
      inconsistent casing, `(named character)` suffixes and comma-separated
      compound types. Only the chariot rows are filled in; every other type
      keeps the engine's previous behaviour rather than guessing at values.
      Done from it: Unit Strength (a heavy chariot is US5, not US1, which
      changes Overwhelmed, panic and pursuit routs), no Rank Bonus for a heavy
      chariot and at most +1 for a light one (`psychology.rank_bonus`, which
      also de-duplicates the two copies of that sum in `combat_resolution`),
      Scythed Wheels (`model.impact_hit_ap()` — Impact Hits at AP-2) and
      Firing Platform (`model.has_all_round_vision()` — the 360° arc
      skirmishers already had, for shooting and casting).
      LEFTOVER: Lumbering (pivot 90° about centre, cannot join or be joined),
      and Iron Shod Wheels, which needs linear obstacles and "difficult counts
      as dangerous for me"; the D3 damage hook is already in place
      (`dangerousTerrainTests(..., damage='D3')`). The table now covers all
      thirteen troop types — see "Troop Types in Detail" above for the rules
      each one grants and the plan for coding them.
- [x] Categories of terrain (Rulebook p. 269-270) — what a piece *is* is now
      separate from what it *looks like*, because the rulebook is explicit that
      a wood "might be classed as difficult, dangerous or even impassable
      terrain, based upon its size and density". `TERRAIN_CATEGORIES` holds the
      rules (open / difficult / dangerous / impassable); `TERRAIN_RULES` holds
      the look plus the category that type presents by default, and a map may
      override it per piece with a `going` tag. Old maps without the tag load
      unchanged.
      Done: -1 Movement (already there), the Dangerous Terrain test (every
      model tests once per feature it meets, losing a Wound on a 1 —
      `dangerous_terrain_wounds`, applied on normal moves and on charges,
      including charges that fall short), the difficult-terrain Charge roll
      (`charge_roll(dice, difficult=True)` discards the *highest* die, with
      Swiftstride's bonus die still added), and Disrupted (a quarter or more of
      a unit's models in difficult terrain costs it its Rank Bonus — counted
      from the unit's own model nodes, recomputed after a move and at the start
      of the Combat phase, and persisted).
      Corrected on the way: marsh was -2 Movement (no such modifier exists) and
      is now dangerous; river invented a no-charges rule and is now dangerous;
      hills gave +1 combat result and are now open ground as the rulebook says,
      keeping their line-of-sight behaviour. `combat_modifier`,
      `charge_allowed` and `formation_break` were dead keys, contradicted by
      the code that ignored them, and are gone; `blocks_line_of_sight` was
      likewise ignored (hills were flagged False yet blocked) and now drives
      `los_block_point`.
      LEFTOVER: linear obstacles are not represented at all.
- [x] Hills (Rulebook p. 271) — Beyond the Crest was already right: a hill
      blocks sight only when neither model is upon it, which is what
      `los_block_point` does (the same rule as a wood's Arboreal Gloom, hence
      one function for both).
      Vantage Point: a unit fires with one additional rank
      (`firing_rank_count`, `simulate_battle(..., extra_ranks=1)`), and the
      rules that let rear ranks fire stack, so on a hill a Volley Fire unit
      shoots with its front rank, the whole second and half of the third.
      The unit must be *entirely* on the hill to claim any of it (Official FAQ
      1.5.3), which bites because hills are organic shapes — a unit on the rim
      usually has a model hanging off. `MovementSystem.entirelyOnHill` counts
      the unit's own model nodes, sharing `modelsInTerrain` with the Disrupted
      check.
      Line of sight, all three cases: a unit on a hill sees over units on lower
      ground; on the *same* hill it sees over only those closer to the bottom,
      the top being the hill's centre (`terrain_system.sees_over`, Official FAQ
      1.5.3); and a unit entirely on a hill can be seen across or through
      intervening units by anyone. That last one does not fit the shooting arc,
      which is clipped target-agnostically, so `markHillTargets` re-marks such
      units as targetable afterwards. Only units are seen over — a wood or
      another hill in the way still blocks.
      LEFTOVER: the arc overlay is still drawn clipped short of a hill-standing
      target even though it can be shot; the magenta target highlight is the
      only cue. The FAQ also calls the battlefield edge a hill's top, for hills
      that run off the table; only the centre is modelled.
- [x] Impassable terrain (Rulebook p. 270) — a `house` terrain type, built as a
      procedural medieval timber-framed building (plaster walls, corner posts
      and a mid rail, gabled roof with overhanging eaves, stone chimney, door
      and shuttered windows) sized to its footprint and turned to run its ridge
      along the longer side. It blocks line of sight like a wood, and it stops
      movement: terrain bodies already carried collision bits 20-24, but the
      movement sweeps tested bit 9 (units) alone, so no terrain had ever
      physically blocked anything. `CollisionMask.MOVE_BLOCKERS` is now the
      sweep default, and only the impassable bit is in it, so woods and hills
      still let units walk in. Flyers pass over as before.
      LEFTOVER: a charge that cannot align because of impassable terrain should
      become a disordered charge, which is not modelled.
- [x] Multi-wound models \u2014 `MovementSystem.applyWounds` converts unsaved wounds
      into slain models using the profile's Wounds, keeping the remainder on the
      wounded model (`woundsOnModel`, persisted). Combat and shooting passed
      their wound totals straight to `removeModelsFromUnit`, which counts
      *models*, so a single wound destroyed a 6-Wound War Wagon.
- [x] Battle Magic (Rulebook p. 320) — all seven spells do something now. The
      catalogue gives a spell's name, casting value, range, type and wording but
      never its effect, so `BATTLE_MAGIC` in `spell_system.py` matches each one
      to a class by name and `game.py` prefers that over `CatalogueSpell`.
      Fireball: 2D6 S4 AP0 automatic hits. Hammerhand: 2D3 S4 AP-2 on an enemy
      the caster is engaged with. Both go through `resolve_magic_hits`, which
      skips To Hit entirely (a spell has no attacking model) and rolls To Wound
      on the spell's own Strength.
      Curse of Arrow Attraction: sets `arrow_attraction` on the target's model;
      `simulate_attack` re-rolls a natural 1 To Hit when shooting at it. Oaken
      Shield: appends a `{'ward': 5}` rule to the caster's model. Both last
      "until your next Start of Turn", which is a whole round — `Spell` now
      carries `ticks_remaining` and the FSM only calls `endSpell()` when it
      reaches zero, so one turn (the old blanket behaviour) is still the
      default and two spans the opponent's turn.
      Arcane Urgency: clears `hasMovedThisTurn` so a unit that has already moved
      may move again; refuses a fleeing unit and one that has not moved.
      Curse of Cowardly Flight: `panic_test(..., compulsory=True)` — the test is
      taken even by a unit that would pass automatically, and *that* unit Gives
      Ground (2") on a failure instead of fleeing.
      Pillar of Fire: a Magical Vortex (Rulebook p. 107). The template is
      placed by clicking a point on the board — a range ring shows the 12" its
      central hole must fall within — and becomes a 3" `pillar_of_fire` terrain
      piece (difficult going), drawn as the ring it is on the tabletop and with
      a round footprint to match, rather than a scaled terrain mesh in a square
      box. A Vortex is never placed touching a base, so `nudge_clear` finds the
      smallest shift, in any direction, that puts it clear of every model,
      friend or foe; placing it therefore hurts nobody, and one whose scatter
      ends over a unit steps off again.
      It burns for D3+3 S3 AP-2 anything that walks through it
      (`MovementSystem.magicalVortexTests`, which rides the same move hook as
      the Dangerous Terrain test) and anything its D6" Start of Turn scatter
      sweeps over — the whole swept path counts, not just where it comes to
      rest, which is what "that the template moves over" means. A model is
      under the template when its *base* meets it (`caught` /
      `distance_to_segment`), the same reach `settle` uses to decide the
      template is touching one; measuring model centres alone let a pillar
      sweep down the gap between two ranks, close enough to be nudged off the
      unit afterwards, and burn nobody.
      `TerrainManager.remove_terrain` takes it away when it ends.
      Ward saves did not exist at all and were needed for Oaken Shield:
      `ward_save_value` picks the best of several (they never combine) and
      `check_saves` runs the whole sequence — Armour, then Ward, then
      Regeneration (p. 141, p. 176) — with AP applying only to the armour value.
      `simulate_battle` and `resolve_impact_hits` both went through it, so every
      Ward save in the game now works, not just this spell's.
      The Dispel now happens between the Casting roll and the effect, which is
      where the rulebook puts it (p. 110). It used to resolve the effect first
      and undo it afterwards, so the player was shown damage that was then
      taken back. `Spell.spellFunction` is a fixed sequence — `canTarget`,
      `_attempt`, `_dispelled`, `apply` — and every spell supplies the pieces
      rather than the whole thing.
      A spell in play survives a quicksave: `save_spells` / `load_spells`
      record the caster, the target and either the remaining duration or the
      template's centre, and put the effect back without re-rolling anything.
      Choosing a spell shows its card — type, casting value, range and the
      catalogue's own wording — under the cursor and on the status line
      (`spell_readout`, `Choice(descriptions=...)`).
      LEFTOVER: Magic Resistance, Unbinding and Outclassed in the Art are still
      unimplemented, so nothing reduces a spell's chance beyond the single
      Dispel attempt, and a Remains in Play spell cannot be dispelled after the
      turn it was cast. The vortex is not removed when it drifts off the table
      edge, and it is nudged clear of bases but not of impassable terrain.
- [ ] Too Tough to Wound — `to_wound` returns 6+ for any Strength shortfall,
      but a difference of -3 or worse cannot wound at all (Rulebook p. 143).
      Found while testing Battle Magic; affects all combat, not just spells.

## Deferred war-machine items
- [ ] Multiple Wounds (D3+1) generic rule
- [ ] Black Powder / Misfire tables
- [ ] True line-of-sight and rank/file hit caps for cannon/bombardment
