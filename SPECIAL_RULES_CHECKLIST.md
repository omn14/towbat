# Special Rules — Implementation Checklist

Tracks weapon and unit special rules from the BattleScribe catalogues. Rules
carry only *keywords* in the data; the coded effects live in `special_rules.py`,
`battleFunctions.py`, `combat_resolution.py`, and `models.py`.

**Every rule that fires has to say so.** Use `rules_log.rule_log` when a rule
changes an outcome and `rules_log.rule_skipped` when one could have applied but
did not, carrying the numbers that decided it. Nothing in this engine is
visible on screen, so a rule that works and a rule that was never coded look
identical without the log. See `.github/copilot-instructions.md`.

## Army Readiness: High Elves vs Warriors of Chaos

Roster-specific audit, 2026-09-09. Sources:
[High Elves: Grow league 500](strategy_armies/nr/he500.json) and
[Warriors of Chaos: Bm500](strategy_armies/nr/chaos500.json).
Both exports total 500 points. These are the actual selected armies, not a
request to implement every option in either faction's catalogue. The roster
files are local inputs and must not be staged with this checklist.

Here, `[x]` means an existing implementation can be reused, not that the whole
army has passed an end-to-end readiness test. `[ ]` includes missing effects,
lost import data, partial support and outstanding roster-specific verification.
This section tracks implementation work; completed import metadata does not
itself add gameplay effects.

### Selected Units

| Army | Unit | Models | Points | Selected equipment or upgrades |
| --- | --- | ---: | ---: | --- |
| High Elves | Mage, General | 1 | 125 | Level 2, High Magic, Silvery Wand, hand weapon |
| High Elves | Elven Archers | 6 | 54 | Hand weapons, longbows; no command group selected |
| High Elves | Silver Helms | 5 | 120 | Barded Elven Steeds, heavy armour, shields, lances; no command group |
| High Elves | Dragon Princes | 3 | 111 | Barded Elven Steeds, full plate, shields, lances; no command group |
| High Elves | Lothern Skycutter | 1 | 90 | Sea Guard crew, Swiftfeather Roc, cavalry spears, shortbows, Wicked Claws, armour 4+ |
| Chaos | Aspiring Champion, General | 1 | 99 | Great weapon, heavy armour, Helm Of Courage, Mark of Chaos Undivided |
| Chaos | Chaos Knights | 4 | 134 | Chaos Steeds, heavy armour, shields, barding, lances; champion, standard, musician |
| Chaos | Chaos Warhounds | 5 | 30 | Hand weapons; no optional upgrades selected |
| Chaos | Chaos Warriors | 10 | 172 | Halberds, heavy armour, shields; standard, musician, The Banner Of The Bold; no champion |
| Chaos | Marauder Horsemen | 5 | 65 | Warhorses, light armour, shields, throwing spears, musician, Skirmishers |

### Import Blockers

- [x] Preserve the three selected magic items and their owners: Silvery Wand
      (Mage), Helm Of Courage (Aspiring Champion), The Banner Of The Bold
      (Chaos Warriors' standard bearer). `magic_items` now retains definition
      and selection identity, category, quantity, points and structural owner.
      Magic Armour remains distinct from ordinary Armour. Unknown items are
      retained with unsupported status. Runtime roster metadata now survives
      battle saves. Runtime inventory and these three selected effects are now
      implemented in points 3 and 4 below; other items remain unsupported.
- [x] Preserve command selections and their model associations. Four Chaos
      Knights remain four bodies; `command` retains champion/standard/musician
      roles, model references and the champion's A2 profile. Original incoming
      and outgoing associations are retained in `roster_selections`.
      Runtime promotions, combat bonuses, casualties and challenges are now
      connected; see Step 2 below for verification and remaining limits.
- [x] Separate the Mage's available spell pool from known spells. The ten
      High Magic/Lore of Saphery options now occupy `spell_pool`; `spells` is
      empty, Level 2 is retained and `spell_generation_pending` is true.
      Explicit spell upgrades and the existing Ruby Ring bound spell remain
      distinguishable. Unsupported item spells do not create Wizards.
      Runtime generation, the single signature substitution and the Wand's
      extra known spell are now implemented in point 4 below.
- [x] Retain selected source records, equipment owners and spell provenance.
      Exclude unselected/zero-count subtrees; retain nested mount and Roc
      weapons without deduplicating different owners. Constructed fixtures
      cover both armies' counts/totals, all three items, command promotions,
      source reconstruction, repeat imports and JSON serialization. Actual
      local exports also verified at 500 points each without modifying them.
      Runtime owners now select separate weapons for rider, mount, crew and
      beasts. Chariot hull-level personal weapons are explicitly crew-operated
      under p. 194. LEFTOVER: ambiguous owners outside these supported profiles.

### Step 2: Command and Profile Gameplay

- [x] Command promotions occupy existing bodies (pp. 198-201): four Chaos
      Knights attack as three ordinary riders plus one A2 champion, with four
      steeds when all four bases can fight. Champion characteristics and
      personal wounds/retirement are separate from the ordinary profile.
- [x] Ordinary standards add one combat-result point per side, independently
      of the BSB (pp. 153, 200). Musicians break only the final tie, after all
      other contributions including Stand & Shoot; opposing front-rank
      musicians cancel (p. 201). Both have visible combat-result rows and logs.
- [x] Musicians add +1 Leadership, capped at 10, to Rally and Enemy Sighted
      tests only. Committed formed and Skirmisher marches within 8" of a
      non-fleeing enemy now test before moving (pp. 123, 201). A failed test
      leaves normal movement available but still counts as marching; it
      cannot be retried that turn. Drilled and flying movers are exempt
      (pp. 167, 170). Pending tests block phase advance and cannot change the
      declared destination. Results survive reload and reset next turn.
- [x] Command occupy front-rank slots with the standard nearest the centre;
      characters cannot displace them. Ordinary casualties remove rank-and-file
      first, then musician, standard and champion. Undirected melee/Impact Hit
      overflow cannot remove a champion while ordinary models remain (p. 199).
      Champions issue, accept and refuse challenges; their own mounts join the
      duel without bringing the regiment's other mounts (pp. 199, 210-211).
- [x] Standards captured on combat destruction or a fleeing catch are recorded
      once, permanently, with their 50 VP value and winning player; trophies
      survive the captured unit disappearing and save/load (pp. 200, 286).
- [x] Rider, champion, mount, crew, beast and joined-character profiles use
      their own weapons and Initiative. Each Initiative step shares a casualty
      snapshot; later mounts/crew lose attacks with their base. Cavalry mounts
      cannot support (pp. 146, 192, 194). The Skycutter uses three Sea Guard
      with spears/bows and one Roc with Wicked Claws, including three BS4 bow
      shots rather than one hull shot. Each part's weapons and armour persist.
- [x] CORRECTED on the way: multi-Wound casualties now accumulate across
      profile steps rather than subtracting one chariot per unsaved wound;
      delayed casualty animation uses rendered counts. Reloading a regiment
      restores rendered bodies as well as logical counts, rebuilding its
      footprint only when the formation changes so existing contacts survive.
      Per-profile attack counts run after charge hooks, preserving existing
      Furious Charge bonuses without duplicate attack-count logs. Duel overkill no
      longer counts excess wounds twice, and equal-Initiative duellists may
      slay each other before either side's casualties are removed.
- [x] Verified the actual two local rosters' runtime owners and counts, pure
      command/profile tests, and an offscreen command scene covering physical
      slots, casualties/reload, later mount attacks, champion challenge
      persistence, Rally, three-crew shooting, final-score tie-breaking and
      failed-march persistence. Local roster files are not test dependencies.
      Full-suite verification uses `run_tests_isolated.py`, never a monolithic
      pytest process inside Code. Final verification: 55 modules, 1,916 JUnit
      cases including subtests, zero failures/errors after focused rechecks.
      Full-run peak RSS 1,207.7 MiB under the 1,536 MiB service cap. The first
      full report retains two reload-contact failures; the corrected
      Shieldwall, command-scene and persistence rechecks supersede those results.
- LEFTOVER: arbitrary attacks directed at a champion outside a challenge,
  champion-specific template/Look Out Sir allocation, and champion-specific
  missile attacks still need per-model attack allocation. Exact contact-based
  full-versus-single attack limits remain the combat engine's existing
  formation approximation. No bespoke command miniature assets are added.
- LEFTOVER: when a unit has both a joined character and champion, the current
  challenge picker prefers the character; explicit participant/refusal
  nomination is not yet offered. AI still never issues and always accepts.
  The duel remains a separate pass from the surrounding combat.
- LEFTOVER: captured-standard VP are persisted, but the tactical assessment
  is not an end-of-battle Victory Points adjudicator. Warband's march modifier,
      remaining movement extensions and magic-item/faction effects belong
  to later steps. This does not mark either army fully rules-complete.

### Existing Support to Reuse

- [x] All ten selected primary model names resolve through the catalogue.
      The Skycutter correction supplies S5/T4/W4, Heavy Chariot, three Sea Guard
      crew and one Roc; Impact Hits (D3+1) use S5 AP-2 from Scythed Wheels.
- [x] Ordinary equipment infrastructure: armour/shield/barding, lances,
      halberds, great weapons, Armour Bane, Requires Two Hands, Strike Last,
      ranged weapon profiles, Volley Fire, Quick Shot and supporting attacks.
      These are reusable primitives, not proof of correct rider/crew ownership.
- [x] General/Inspiring Presence for both Generals; Rallying Cry for the
      Aspiring Champion; Veteran resolution for a future Banner of the Bold
      grant. Veteran exists, but this roster's banner currently grants nothing.
- [x] Swiftstride; Warhounds' Move Through Cover; Fly and Firing Platform
      infrastructure for the Skycutter; core Skirmishers for Marauder Horsemen.
      Retain the partial-implementation limits in the detailed sections below.
- [x] Generic Ward saves, Panic/Break tests, reroll limits, Furious Charge,
      Press of Battle, Massed Infantry and Parry have existing resolution paths.
      A generic Ward save does not implement Dragon Armour or Chaos Armour.
- [x] Separate live/base profiles and split-part snapshots in persistence;
      spell-in-play records and a Ruby Ring of Ruin bound-spell import precedent.
      The Ruby Ring is NOT selected here, and is not a general item system.

### Shared Battle Dependencies

- [x] **First Charge: core** - Silver Helms, Dragon Princes, Chaos Knights
      (pp. 101, 169; pursuit p. 157 and Official FAQ v1.5.3). An accepted first
      charge attempt spends eligibility even if it fails; actual contact grants
      Disruption through that turn's Combat phase. Cancelled/refused declarations
      do not spend it. Pending attempt tracking is idempotent across repeated
      entry, and handles charges against fleeing targets and supported contact
      geometries. Rank scoring consumes a separate source, so terrain refreshes
      cannot erase it and expiry cannot clear terrain/flank disruption.
      **Corrected:** pursuit/overrun contact that counts as charging also counts
      towards First Charge. Unfought-combat contact applies now; deferred combat
      queues the source for next turn instead of expiring it in the current one.
      Combat spell detours do not expire sources. Attempts, pending state, active
      and deferred sources survive save/reload. Outcome logs explain application,
      failure/spent eligibility, expiry, and rank points already suppressed by
      another source. Quiet queries do not log.
      **Legacy saves:** absent charge history is treated as spent, with a log;
      start a new battle for accurate first-attempt eligibility.
      Counter Charge's supported formed-unit reaction below also resolves
      each participant's First Charge attempt on actual contact.
      **LEFTOVER:** Phalanx immunity/formation is not implemented (neither roster uses it).
      Independent joined-character charge history, other disrupted-state
      consumers outside combat rank scoring, and a complete redirected-charge
      UI remain unaudited. Saving pending state does not restore suspended
      charge/reaction animation tasks. This is not a general effect scheduler.
- [ ] **Counter Charge: PARTIAL, formed-unit declaration flow** - Dragon Princes
      and Chaos Knights (p. 167; Official FAQ v1.5.3). Eligible defenders now
      get a Counter Charge choice; AI defenders take it when available, falling
      back to Stand & Shoot/Hold otherwise. Checks use declaration-time arc and
      base distance, charging troop type, active ground/Fly Movement, own rider
      rule and eligible joined character, fleeing/engaged state, and saved
      once-per-game-turn usage. Declining spends nothing; later charges against
      a unit that used it force Hold. Refusals include deciding conditions.
      The reaction pivots about the centre and advances D3+1" with collision
      checks. The D6-to-D3 conversion has no Swiftstride choice or charge reroll.
      The incoming charge is rerouted to the moved defender using existing
      route geometry, with its own charge roll and Swiftstride choice. Both
      units receive charge flags/distances and First Charge on contact; failed
      contact spends pending first attempts. Charge-move flags cover both
      participants during casualty checks and clear afterwards.
      **Corrected:** the enemy-contact movement preview no longer marks/logs
      the charger as marching before its charge declaration. Callback route
      animation uses the existing awaitable Parallel interval pattern.
      **Declaration sequencing (pp. 119-121):** confirmed chargers now remain at
      their original poses until Resolve Charges closes the declaration stage.
      Counter Charge/Stand & Shoot choose one incoming charger after all
      declarations; the other chargers face Hold. Flee runs once from the highest
      Unit Strength charger, with random ties. Reactions finish before the active
      player chooses the charge-move order. Routes/arcs are rebuilt after defensive
      movement; a front and flank charge can engage the same moved defender.
      Ordinary moves and late charges are gated by the stage. Basic and Enhanced
      AI each make a declaration pass before Remaining Moves; EnhancedAI currently
      uses nearest-enemy attempts, not a multi-charge tactical search.
      Declaration-stage saves retain the queue, original poses, loose target
      index and pending First Charge attempts. Active/interrupted resolution
      refuses saves; legacy Movement snapshots resume in Remaining Moves.
      **Corrected during sequencing:** planned formed routes now use the exact
      route animator and contact-point alignment, not the legacy reconstructed
      wheel which left a 0.15-inch gap. Re-entering InCombat cleared earlier combat
      links; already-engaged defenders now keep them. Failed planned charges move
      only the Charge roll. Flee's board-distance helper imports from special_rules.
      A queued loose target is not reselected by sight after reactions or reload.
      Drilled now offers free redress before the formed Counter Charge advance.
      **LEFTOVER:** wider Marching Column/Counter Charge restrictions and loose-formation
      Counter Charge remain unfinished (loose pairs are refused with a log). Complex
      obstructed/flying routes, pivot/terrain edge cases and joined-model geometry
      need broader verification, as do simultaneous frontage maximisation and
      redirected charges. Multiple loose-defender form-ups and fleeing loose
      targets are not supported by the queued per-model route; these fail with
      explicit spent-charge logs. Suspended reaction/animation tasks are not
      resumed from saves: reload a declaration-stage snapshot after interruption.
      Do not mark the full rule or formation dependencies complete from these tests.
- [x] **Cavalry split profiles and Cavalry Support** - separate rider/mount
      WS/S/I/A, weapons, Initiative snapshots and casualty accounting; no
      supporting mounts. Remaining contact-allocation limits are noted above.
      **Corrected (movement preview):** selecting Silver Helms crashed because
      the mount-modifier loop assumed `mountUnit.model`, but imported mounts
      are already model profiles. It now uses the existing `get_mount()`
      accessor for direct and legacy wrapped mounts. Seven offscreen preview
      checks cover Silver Helms, Dragon Princes and Chaos Knights in both
      representations, plus an unmounted Mage; range, finite preview points,
      unchanged position/facing and mount identity are verified. No movement
      rule effects or formation support were added by this crash fix.
- [x] **Command groups** - champion A2 on one of the four Chaos Knights,
      standard bearers' combat-result bonus on Knights and Warriors, and
      musicians on Knights, Warriors and Horsemen. Tie-breaking, Rally/march
      modifiers, front-rank placement, casualties, champion challenges,
      captured standards and persistence are wired; see Step 2 limits above.
      A normal standard is
      not a Battle Standard Bearer and does not grant Hold Your Ground.
- [ ] **Formation-specific effects** - verify Close Order combat-result
      eligibility at current Unit Strength, Open Order Quick Turn for
      Warhounds, and Skirmish formation for Horsemen. Do not infer an active
      formation solely from a list of available formation keywords.
- [ ] **Enemy Sighted / march tests**, and the Drilled/Fly exemptions; connect
      musician and Warband modifiers to the actual Leadership-test context.
      Core test, musician and exemptions are done. LEFTOVER: Warband modifier.
- [ ] **Heavy-infantry Steady in the Ranks** - Chaos Warriors and the Aspiring
      Champion's troop type; apply its disruption protection where relevant.
- [ ] **Weapon choices and charge conditions** - preserve hand-weapon choices
      when Ithilmar/Ensorcelled Weapons or Parry can beat the automatic choice;
      no shield in melee with a great weapon/halberd. Verify the 3" threshold
      for Furious Charge/Impact Hits separately from lance/spear conditions.
- [ ] **Flaming and Magical Attacks dependencies** - distinguish weapon,
      spell and model sources; High Magic and Ensorcelled Weapons need these
      against Ethereal and any relevant saves. Warhounds' Fear interaction
      with Flaming Attacks must follow the troop-type rule and FAQ.

### High Elf Rules

- [x] **Elven Reflexes** - all five High Elf units, but only Sea Guard crew
      on the Skycutter. Apply first-combat-round +1 Initiative, capped at 10,
      to the correct profile; not to steeds, Roc or hull. Coordinate with
      separate split-profile attack timing, not a whole-unit Initiative bump.
      Live scheduler and challenges use the combat-round state; no profile
      stat mutation. Corrected the catalogue/roster name collision that dropped
      the crew-only keyword. Derived crew grants withdraw and restore with the
      source, including save/reload. Charge logs separate the two bonuses.
- [x] **Valour of Ages** - all five units. Re-roll failed Panic tests only
      for the specified heavy-casualty/friendly-flee-through causes, not every
      Leadership or Panic test. Preserve cause information and the one-reroll cap.
      The live Panic queue retains the cause and waits for an optional human
      choice; AI re-rolls failures. Positive, ineligible and declined cases log.
- [x] **Ithilmar Weapons** - Mage and Dragon Princes. Natural-1 melee To Hit
      re-rolls with a single non-magical hand weapon only; not lances, mounts,
      other weapons or spells. Do not grant this to Silver Helms by faction.
      Shares one replacement die with Hatred; logs aggregate attempts/results,
      not every attack. LEFTOVER: tactical weapon selection remains above.
- [x] **Ithilmar Barding** - Silver Helms and Dragon Princes. Dangerous Terrain
      natural-1 re-rolls, without also granting Move Through Cover's Movement
      immunity. Reuse the terrain reroll machinery with a distinct source.
      Formed and Skirmish terrain callers share the same one-reroll cap. The
      actual joined Mage tests separately and cannot borrow the host's barding.
      LEFTOVER: the combined Ithilmar Armour/Barding rule's Wizard armour
      permission is not implemented; these selected riders are not Wizards.
- [x] **Dragon Armour** - Dragon Princes' 6+ Ward save, independent of their
      full plate/shield/barding armour save. Combine Ward sources by taking
      the best, not adding them. Shared saves cover melee, shooting, magic,
      Impact Hits, cannon and bombardment; Slaying Blows still permit the Ward.
      AP never modifies the Ward. Report aggregate rolls and superseded sources.
      LEFTOVER: armour-wearing Wizard permission is not implemented or needed
      by these three non-Wizard models.
- [x] **Impetuous (selected formed cavalry)** - Dragon Princes (p. 172,
      amended wording; FAQ v1.5.3). Resolve Charges tests once before reactions,
      using Leadership/Inspiring Presence, active joined-character Leadership and
      shared Veteran rerolls. Failure adds a compulsory declaration with owner
      target selection; AI selects the nearest supported legal route. A prior
      voluntary declaration still tests, so failure cannot evade compulsory
      Drilled. Ordinary redress is blocked during declarations. Rule presence
      includes joined and split-profile models; the old 4+ mechanism is not used.
      **LEFTOVER:** loose Impetuous chargers are explicitly logged as unsupported.
      Routing still needs broader flight/large-target/height and obstructed-charge
      support, including the FAQ's friendly Skirmisher screen that may move away.
      Other compulsory-charge sources (Frenzy, spell restrictions) are not added
      by this implementation. Legacy saves without a declaration stage still
      resume in Remaining Moves rather than inventing retroactive tests.
- [x] **Drilled (selected movement paths)** - Dragon Princes (pp. 125, 167;
      FAQ v1.5.3). Free redress immediately before committed Remaining Moves,
      queued charge moves, Counter Charge and Giving Ground. The front is anchored;
      up to five files change without spending normal movement/manoeuvre allowance.
      Charge dice precede redress; routes are rebuilt afterwards. Joined characters
      accompany the host. Candidate shapes roll back cleanly, retain character
      slots and reject unit/impassable/board overlap or model movement beyond 2M.
      Human choices and AI keep/required-redress policy use the same helper.
      Drilled retains its Enemy Sighted exemption. Pending Remaining Move choices
      block extra movement, phase advance and save/load.
      **Marching Column (p. 101):** no rank bonus, 3M march, declaration allowed
      but queued charge fails without movement if still in column. A compulsory
      Drilled charger must redress into Combat Order when a legal shape fits;
      a voluntary charger may keep its column. No-room and declined cases log.
      **Corrected during implementation:** replot from the actual clicked target,
      not the stale collision marker; complete empty declaration queues; use the
      active-character Leadership helper. Existing declaration/rally tests now
      supply Impetuous dice or close declarations before ordinary redress.
      **LEFTOVER:** free-redress hooks for Vanguard, pursuit/overrun, Follow Up,
      reserve and spell-granted movement; broader column/Counter Charge restrictions;
      swept manoeuvre obstruction, 1-inch separation and complex joined footprints.
      Endpoint fit is not a full model-by-model legal-path proof. AI does not
      optimize optional frontage. Suspended movement tasks cannot be saved.
- [ ] **Sons of Caledor** - restrict who may join Dragon Princes. The Mage
      is this army's General, so that exception must allow them to join;
      being a Wizard alone is neither permission nor a prohibition.
- [ ] **Lileath's Blessing** - Mage. Optional failed Casting-roll re-roll once
      per turn, with human/AI choice and a saved usage flag. FAQ v1.5.3 excludes
      natural double-1 Miscasts from failed-Casting-roll re-rolls.
- [ ] **Lore of Saphery** - Mage. Implement the permitted generated-spell
      substitution into the lore signature or one of the three faction spells;
      do not grant all alternatives automatically. See spell generation below.
- [ ] **Fear** - Skycutter. Charge/combat tests, relative Unit Strength,
      immunity and once-per-turn state; show why equal/higher-strength enemies
      do not test. Extend Mark of Chaos Undivided to this test once implemented.
- [ ] **Skycutter split equipment and remaining chariot rules** - keep crew
      cavalry spears/shortbows separate from the Roc's Wicked Claws: now
      implemented, with separate Initiative and three-crew shooting. The
      export's hull-level personal weapons are assigned to crew under p. 194.
      LEFTOVER: audit Lumbering,
      Iron Shod Wheels, flying versus ground movement, landing terrain and
      ground-only follow-up/pursuit. Do not duplicate the Roc as an extra mount.

**Conditional, not extra purchases:** Archers have **Detachment**, but this
export does not select a parent regimental unit or detachment relationship.
Do not invent supporting-fire/charge benefits. Detachment runtime support is
only a blocker if that relationship is actually chosen later.

### Chaos Rules

- [x] **Chaos Armour (5+) / (6+)** - 5+ Ward on the Aspiring Champion; 6+ on
      Knights and Warriors. These are Ward values, not body-armour values.
      Reuse Ward resolution and retain each source through save/load and
      item suppression: disabling the Helm removes its armour bonus, not the
      native Chaos Ward. Missing/X values never invent a save.
      LEFTOVER: Wizard armour permission is not implemented; no Chaos Wizard
      is selected. Future spell Ward grants must use the existing best-save rule.
- [ ] **Ensorcelled Weapons** - Aspiring Champion, Knights and Warriors.
      A single non-magical hand weapon gains AP-1 and Magical Attacks; great
      weapons, halberds, lances and mounts do not receive those benefits.
      AP-1 is live and attacks carry a Magical classification without mutating
      their stored weapon. Eligible/ineligible weapon batches log once.
      LEFTOVER: magical-versus-mundane defence consumers (including Ethereal)
      are not implemented; classification alone is not complete rule support.
- [ ] **Mark of Chaos Undivided** - Aspiring Champion, Knights, Warriors and
      Horsemen. Failed Fear/Panic/Terror re-rolls, not Break or ordinary Rally
      re-rolls. Coordinate with Veteran and never re-roll a re-roll.
      Live Panic and the shared optional choice are implemented and tested with
      the Warriors' actual Banner. Fear/Terror eligibility is tested in the
      shared helper. LEFTOVER: no live Fear/Terror test handlers yet, so those
      benefits cannot fire in battle. Do not mark the full rule complete.
- [ ] **Gaze of the Gods** - Aspiring Champion. Optional Command-phase table,
      roll/result log, persistent changes versus effects expiring at the next
      Start of Turn, characteristic caps and save/load. Include **Stupidity**
      from the adverse result and its joined-unit effects; a table that applies
      only the beneficial outcomes is not implemented.
- [ ] **Warband** - Horsemen. Charge-roll re-roll and context-limited Rank
      Bonus to Leadership. Their selected Skirmish formation gives no Rank
      Bonus, but that does not remove the charge-reroll benefit. Test fleeing,
      Restraint and non-Warband character Leadership explicitly.
- [ ] **Loner and Undisciplined** - Warhounds. Enforce character-joining and
      General restrictions, and do not lend them the General's Inspiring
      Presence through the generic Leadership helper. Undisciplined comes
      from the War Beasts troop type even though absent from the roster keywords.
- [x] **The Banner Of The Bold and Helm Of Courage** - owned inventory,
      sourced Veteran, passive armour and optional once-per-game Break reroll
      are implemented below. Chaos Armour's separate Ward is now also live.

**Present but inactive in this loadout:** **Fast Cavalry** needs Open Order;
these Horsemen selected Skirmishers instead. **Fire & Flee** has an existing
engine hook, but these Horsemen have no missile weapon: their **Throwing Spear**
has range `Combat`, Fight in Extra Rank, and charge-only use. It is not a thrown
missile. **Chaotic Cults** is a listed option, but no specific cult is selected;
do not grant a patron benefit or display a usable blessing without one.

### Selected Magic Items

All three selected items now have coded effects through the roster path.
Names below preserve the exports' spelling; match canonical IDs/normalised
names rather than relying on capitalisation.
Point 3 now retains each purchase as a runtime instance, resolves all three
bearers in the actual armies, and persists state. Point 4 adds the consumers,
outcome logs and generation choices. Unknown items still show **unsupported**.

- [x] **Silvery Wand**, Mage, 15 points, `Arcane Items`.
      Generate **three known spells for this Level 2 Mage**, with the normal
      generation/substitution rules. The bearer remains Level 2 for casting
      and dispelling, and does not gain a third spell attempt per turn.
      Source: [Silvery Wand](https://tow.whfb.app/magic-item/silvery-wand),
      Forces of Fantasy p. 183; casting limit clarified by FAQ v1.5.3.
- [x] **Helm Of Courage**, Aspiring Champion, 25 points, `Magic Armour`.
      Improve the wearer's armour value by 1 while retaining heavy armour:
      this loadout should have armour 4+ before AP, plus its separate Chaos
      Armour 5+ Ward. Offer the wearer/joined unit a
      once-per-game 2D6 Break-test re-roll, with one shared item-use record.
      Spending the re-roll must not switch off the passive armour bonus.
      Source: [Helm of Courage](https://tow.whfb.app/magic-item/helm-of-courage),
      Battle March: General's Companion p. 47.
- [x] **The Banner Of The Bold**, Chaos Warriors' standard, 10 points,
      `Magic Standards`. Grant Veteran through the existing rule machinery;
      do not implement a second reroll algorithm or extend Veteran to Break.
      Handle the unit and joined-character scope required by magic-standard
      rules, and revoke only this item's grant if the banner ceases to apply.
      Source: [The Banner of the Bold](https://tow.whfb.app/magic-item/the-banner-of-the-bold),
      Battle March: General's Companion p. 47; FAQ Characters on standards.

### Magic Item System Plan

- [x] **Import identity and structural ownership** in `roster_importer.py`:
      selected item/definition IDs, names, categories, available source metadata,
      quantities, points and owners now survive import and army-list JSON.
      Unknown items remain unsupported metadata, without inferred prose effects.
      Runtime instances now resolve parent, attachment and command owners;
      ambiguous or missing owners remain inactive with a reason. The three
      selected definitions have explicit names/aliases and book references.
      LEFTOVER: register catalogue-ID aliases and definitions for more items.
- [x] **Typed registry foundation** in `magic_items.py`: immutable definitions,
      explicit category-checked ID/name matching, typed armour modifiers,
      rule grants, extra known spells and optional rerolls. Categories and
      description text never generate effects. The three selected definitions
      now drive armour saves, existing Veteran/Break resolution and spell
      generation. LEFTOVER: other item definitions and effect types.
- [x] **Separate definitions, instances and active contributions**: purchased
      IDs include the runtime unit, selection reference and copy ordinal.
      Each copy owns its retained metadata, per-ability usage and whole-item
      disabled/destroyed state. Spending an ability does not disable passive
      effects. Source IDs derive from the purchased instance and effect key.
- [x] **Source-aware query boundary**: `active_effects` derives live recipient
      contributions without editing native rules or `_base_characteristics`.
      Repeated queries/loads/joins do not multiply a source; suppression
      withdraws only that source. Synthetic tests preserve another grant and
      native Veteran, and retain passive effects after ability exhaustion.
      Point 4 queries these contributions in combat, armour, psychology and
      spell generation, logging deciding numbers and relevant nonapplication.
      LEFTOVER: wider stacking policies for future item effects.
- [x] **Core lifecycle and activation gate**: live queries follow character
      joins/leaves/removal, command casualties and retirement. A retired
      bearer retains personal passive protection but grants no host benefit
      (p. 210). `activate_ability` checks the owner, recipient, context and use
      limit; declined/invalid choices spend nothing and log the reason.
      Per-turn state uses the saved player/round identity; per-game state stays
      on the owning item through host changes. A new army starts fresh; a
      battle reload restores state.
      Helm now offers a human choice and uses the existing AI Break policy;
      BSB priority prevents spending both sources on the same roll.
      LEFTOVER: other item-specific timing and transient effect durations
      beyond per-turn uses/rest-of-battle suppression.
- [ ] **Item suppression for Vaul's Unmaking**: select a carried item on a
      legal enemy character, mark it unusable for the rest of the battle and
      remove its active effects. This directly matters for the Chaos General's
      Helm. Do not allow targeting the Warriors' non-character standard bearer
      with a spell restricted to enemy characters.
      Core `disable_item` now records the reason/destroyed state independently
      of spent abilities and removes all registry contributions.
      LEFTOVER: the actual spell, character-only target selection and adapter
      for effects implemented outside this registry (including Ruby Ring).
- [x] **Inventory persistence** in `persistence.py`: explicit
      `magic_item_inventory` records save stable IDs/owners, raw source data,
      per-ability use counters and whole-item disabled/destroyed state.
      Contributions are derived, not serialized as profile mutations. Existing
      and recreated bearers reload without duplication or refunded use. Legacy
      saves without the field have an empty inventory, even if raw roster
      metadata names purchased items.
      Point 4 also persists intermediate generation rolls/substitution stage
      in roster metadata and final known choices through the spellbook save
      path; completed spellbooks are never generated again on battle load.
- [ ] **Equipment and validation boundaries**: distinguish additive helmets
      from replacement body armour; use best Ward rather than adding saves;
      enforce rider-only/weapon-only scope. Preserve item categories and
      allowances for future list validation. Agree the league/Battle March
      rules pack before claiming roster legality; this audit is not a legality check.
      Bearer-only registry scope already separates rider/champion/attachment
      profiles. The additive Helm now improves heavy armour 5+ to 4+ without
      modifying baseline stats or protecting its host unit. LEFTOVER: broader
      equipment/Ward/weapon-only stacking, monster/chariot armour-choice
      restrictions and roster legality checks belong to later handlers.
- [x] **Inventory visibility and state diagnostics**: the unit card includes
      owned items and joined-character inventories, bearer names, support and
      use/disabled status. Its existing two-line detail area now has bounded
      scroll arrows so inventory entries are not silently truncated. Selection
      changes reset scrolling; refreshes preserve it. Load reports unsupported
      purchases; activation/suppression log at state changes, never in queries.
      Point 4 logs actual reroll results, declined/spent/lost/disabled sources
      and armour changes once per attack batch, not per attack or UI query.

Point 3 verification: 13 synthetic registry/lifecycle tests and eight offscreen
scene cases cover all three selected owners, independent copies, use-state JSON
roundtrips, no legacy inventory inference, existing/recreated bearers, joins,
standard loss, native-source preservation and both HUD layouts. Corrected during
implementation: hidden detail lines, retired personal-vs-host scope, serialized
turn-token equality and hidden-card controls reappearing on resize. Point 3 was
committed as `9b90bce`; selected effects were deliberately left for point 4.

### Point 4: Selected Effects and Generation

- [x] Banner Veteran uses the existing strict-majority Leadership reroll,
      including personal tests for joined characters under the magic-standard
      FAQ. Break tests remain excluded. Lost/disabled sources withdraw cleanly
      without removing native Veteran; human decline and actual results log.
- [x] Helm armour is derived for melee, missile, magic, impact, cannon and
      bombardment save paths. Once-per-game Break state belongs to the item,
      survives reload and host changes, and does not disable passive armour.
      BSB and Helm never reroll the same dice twice; Shieldwall follows the
      final result through the existing combat resolver.
- [x] Before deployment, D6 results select unique numbered spells, rerolling
      duplicates (Rulebook pp. 106, 319). Saphery's numbered export profiles
      are excluded from the D6 table and offered only as signature alternatives
      (Forces of Fantasy p. 186). Exactly one generated spell may be replaced.
      [Generation](https://tow.whfb.app/the-lores-of-magic/spells-and-spell-generation);
      [Saphery](https://tow.whfb.app/the-lores-of-magic/lore-of-saphery).
- [x] Silvery Wand adds one known spell without increasing Wizard level,
      casting allowance or dispelling. Explicit known/Bound spells remain.
      Saved intermediate choices resume without rerolls, including a pending
      replacement. Player-selected Wizard order is supported; AI keeps rolls.
- [x] Pending generation blocks placement and phase advance. Fresh imports
      start on their first deployment action; saved deployment games schedule
      only after unit restoration. Loading another battle while a generation
      choice is open is rejected with a message; saving the choice is allowed.
- [x] Corrected during implementation: empty known-spell lists had skipped
      selected Wizard level restoration; Saphery 1-3 collided with High Magic's
      table; runtime spell classes needed excluding from saved generation data;
      long choice labels overflowed fixed buttons. Dialog geometry and rendered
      selection are covered by the existing offscreen choice-layout tests.
- LEFTOVER: all ten High Magic/Saphery spell effects, Chaos Armour's Ward,
      Mark of Chaos Undivided and other faction effects remain later points.
      Unsupported generated spells are explicitly logged as known but inert.
- LEFTOVER: only a complete, unambiguous six-spell lore plus one normal
      signature is accepted. Missing/ambiguous export pools stay pending with
      a diagnostic; other special generation rules and a full legality engine
      are not implemented. Mid-battle loss of a Wand-granted spell needs the
      eventual Vaul/item-suppression spell handler's explicit resolution.

Point 4 verification covers pure selected-item/generation rules, actual offscreen
Break resolution and deployment gating, saved spellbooks, and the real exports'
six numbered High Magic spells/four signature choices. Final isolated runs pass
all selected-item, generation, inventory, persistence, armour, weapon, magic,
choice-layout, command, Veteran and Shieldwall slices, including all 11 item
scene checks. Earlier runs encountered an intermittent Panda3D `!mat.is_nan()`
assertion in the existing horizontal HUD text check; it did not recur on the
final run, but no fix for that intermittent assertion is claimed. All 17 changed
Python files pass syntax validation. No full-suite, full-matchup or complete-
lore-effect claim.

Startup follow-up: the game's two checked-in converted army lists still predated
the lossless importer, despite the original exports passing the runtime audit.
The old Mage listed all ten spells as known and neither list retained magic
items, so no generation prompt or inventory entries could appear. Regenerated
both startup lists from the local exports without changing those exports or
battle saves; both remain five units and 500 points. Added an offscreen regression
using the actual default startup files: one pending Level 2 Mage, generation
prompt, three final known spells, all three resolved items visible through HUD
detail scrolling, and Champion armour 5+ to 4+. Old battle saves are not migrated.

### High Magic and Lore of Saphery

- [x] Joined-Wizard casting selection: select the host and use the cast command;
      Strategy-phase clicks also discover the joined caster after Command.
      If both host and character can cast, choose the bearer before its spell.
      Keep spellbooks, ordinary/Bound allowances and spent flags on that bearer;
      restore host selection on success, failure or cancel without detaching.
      Respect host fleeing/marching/combat and retired Assailment restrictions
      (pp. 108, 123, 207, 210). Corrected joined-world facing and vortex distance
      (p. 208), and Hammerhand's host-opponent check. Pure and offscreen tests
      cover actual selection, targeting, successful/failed/cancelled casting,
      per-bearer accounting and save/reload.
      LEFTOVER: broader joined-unit spell-effect propagation and challenge-specific
      spell targeting are separate from caster selection; this adds no missing
      High Magic effects and does not expand the one-joined-character model.

Runtime audit: `spell_class(name)` returns `None` for **all ten** exported
spells below. `CatalogueSpell` rolls to cast and prints wording, but does not
apply the effect. They form the Mage's available pool, not ten known spells.
Implement the pool if arbitrary legal spell generation is to work; a smaller
first playable milestone must explicitly restrict the selected known spells.

- [x] **Spell generation and Lore of Saphery substitution**: Level 2 plus
      Silvery Wand gives three known spells, with normal duplicate handling
      and permitted signature replacement. Persist the final choices; casting
      allowance remains Level 2. Do not use `spells` for both pool and choices.
- [x] **Generation spell reference**: both signature and replacement prompts
      show imported type, casting value, range, phase and full effect wording.
      Read-only browsing separates generated, not-generated, already-known and
      signature spells; a pending signature is labelled on save/resume. Hover
      previews do not spend a choice. Long text scrolls in a fixed reader, with
      wide and narrow offscreen coverage and a real startup-dialog regression.
      Corrected horizontal overflow from Panda scroll-frame border space.
      LEFTOVER: this displays roster wording, not an errata-synchronised rules
      reference; missing text and uncoded effects are labelled, not invented.
- [ ] **Drain Magic** - Remains in Play casting-value aura. This Chaos list
      contains no Wizard or Bound item, so it normally has no enemy caster to
      affect here; retain correct eligibility rather than inventing a target.
- [ ] **Walk Between Worlds** - caster/host Ethereal and Reserve Move until
      the specified next Start of Turn. Requires those rules, magical versus
      mundane damage, movement timing and removal without permanent stat drift.
- [ ] **Fiery Convocation** - scattered large template, per-base hits,
      Flaming Attacks and correct S/AP, including Skirmisher targets.
- [ ] **Tempest** - stationary Remains in Play vortex and terrain-changing
      aura; interact with Fly, Move Through Cover and Ithilmar Barding.
- [ ] **Corporeal Unmaking** - Assailment hits with armour and Regeneration
      prohibited, but Ward saves allowed; respect combat/challenge targeting.
- [ ] **Fury of Khaine** - timed Extra Attacks (+1), including correct model
      scope and expiry without losing the buff at an unrelated combat reset.
- [ ] **Shield of Saphery** - timed 5+ Ward and replacement of existing
      Enchantments as specified, not an extra cumulative Ward save.
- [ ] **Hand of Khaine** - single-model Assailment hit; no armour save,
      but Ward and Regeneration remain available; validate model targeting.
- [ ] **Courage of Aenarion** - Remains in Play Unbreakable grant, legal
      engaged target, specified Enchantment replacement and reversible removal.
- [ ] **Vaul's Unmaking** - character-targeted item disabling; requires the
      item system above, including targeting a joined or engaged character.
- [ ] **Shared magic lifecycle** - casting/dispelling and Lileath choices,
      Remains in Play dispels/recasts, caster loss, conflicting Enchantments,
      aura refresh, correct duration boundaries and save/load. Verify the
      Chaos player's dispel options despite having no Wizard. Current saved
      live/base snapshots are not a general temporary-effect engine.

### Implementation Order and Acceptance

1. Lossless roster import: completed.
2. Command groups and split-profile ownership: completed core, leftovers above.
3. Minimum magic-item system: completed and committed as `9b90bce`.
4. Selected Banner/Helm/Wand effects and spell generation: completed as above,
      with unsupported spell effects and verification caveats explicitly retained.
5. Frequent faction effects: completed core Ward grants, profile-local Elven
      Reflexes, hand-weapon effects and contextual rerolls. Remaining Fear/Terror,
      magical-defence and Wizard-armour dependencies are explicit above.
6. Charges/movement: in progress. First Charge core, counted-pursuit timing and
      save/load are implemented, as are the declaration queue, post-declaration
      reaction selection and formed Counter Charge routing. Selected-cavalry
      Impetuous, Drilled and core Marching Column dependencies are now implemented
      and tested together. The explicit movement/geometry leftovers remain open.
7. Magic and remaining dependencies: selected lore effects, Lileath choices,
      effect lifetimes and Gaze of the Gods including Stupidity.
8. Complete matchup verification using both exact roster imports offscreen.
      Each completed entry needs a rule citation, positive/negative tests, useful
      logs and an explicit `LEFTOVER:` if partial.

- [ ] Import/reload keeps both totals at 500, all ten units, exact model counts,
      command roles, three item identities and correct crew/mount equipment.
- [x] Mage remains Level 2 with three generated known spells; no duplicate
      grants, extra casting slot or new random generation after loading.
- [x] General's Helm changes armour 5+ to 4+, grants one optional Break
      re-roll, stays spent across host changes/reloads, and loses both benefits
      when made unusable; spending only the re-roll preserves its armour effect.
- [x] Warriors gain Veteran from their banner and use only one allowed
      reroll alongside Mark of Chaos Undivided; failed Break tests do not get
      Veteran. Verified through the live Panic choice; Mark's Fear/Terror callers
      remain pending. Banner loss and joined-character scope are tested separately.
- [ ] Silver Helms/Dragon Princes versus Chaos Knights test first charge,
      counter charge, riders versus mounts, armour/Ward, and first-round timing.
- [x] Combined movement slice: failed Impetuous, required Drilled from column,
      Counter Charge and First Charge against Chaos Knights, human/AI choices,
      live EnhancedAI resolver entry, voluntary declarations and reload. This
      does not complete the broader combat acceptance item above.
- [ ] Skycutter versus Chaos tests S5 AP-2 Impact Hits, Fear-strength boundaries,
      crew shooting/melee, Roc attacks, terrain and flight; Horsemen cannot
      shoot or Fire & Flee with their melee-only Throwing Spears.
- [ ] Test expiry/disable/reload at least once for every timed or limited-use
      source, plus a rendered playable two-army save with no silent unsupported
      effects. Use focused checks per feature and one full suite at completion.

**Point 5 verification:** 41 focused faction checks and four actual-roster
offscreen checks pass. The latter load all ten models from the tracked startup
lists and exercise crew/rider/mount Initiative, Ward values, joined terrain
ownership, the Banner/Mark choice, item suppression and repeated save/reload.
Additional Initiative, terrain, Move Through Cover (including scene), command,
challenge and persistence checks pass. Cannon/template save functions and one
aggregate template Ward log are covered. All runs use sequential memory-bounded
services; no monolithic or full-matchup run. The adjacent item scene passed
10/11 checks but reproduced the previously recorded horizontal HUD
`!mat.is_nan()` assertion; this unrelated intermittent failure remains unfixed.

**Drilled/Impetuous verification (2026-09-11):** 25 actual-roster offscreen checks
pass, including free/declined/blocked redress, anchored frontage, engaged Giving
Ground, Counter Charge timing, actual 3M marching, column rank denial, pass/fail
and ineligible Impetuous, joined-model scope/Veteran, pending-choice guards and
combined human/AI/reload charges. All 297 distinct focused/adjacent checks in this
work pass through sequential bounded services (768/1024 MiB; peak 850.3 MiB).
Adjacent coverage includes declarations, First Charge, Counter Charge, marching,
mounted preview, persistence, pursuit ordering, rally movement, Vanguard,
formed/Skirmisher routes, Shieldwall and post-combat results. No monolithic suite
or full-matchup acceptance run. Sources: rule pages pp. 101, 125, 167, 172 and
the online Movement/Universal Special Rules FAQ v1.5.3.

**Earlier point 6 verification:** 14 focused First Charge checks and five
actual-roster offscreen state checks pass, covering failures, repeated entry,
rank disruption, spent/pending/active/deferred save state, legacy saves and
Combat-phase expiry. The 48 formed-versus-Skirmisher route checks include
First Charge success/failure assertions; the real-task scheduler regression
still passes. Thirteen Shieldwall scene checks cover live formed/Skirmisher
pursuit and immediate/deferred overrun contacts. Psychology, post-combat,
bound-spell and persistence regressions pass under sequential memory caps.
Counter Charge adds 12 rule checks and 11 actual-roster scene checks: human/AI
reaction selection, retained shooting fallback, declaration-distance refusal,
Fly distance boundary, D3+1 and charger-only Swiftstride, straight/angled contact,
failed contact, mutual First Charge, saved turn usage/next-turn availability,
charge-move state and the false marching log. Both angled scenes render offscreen
and were inspected. Stand & Shoot, marching, formed-to-Skirmisher, First Charge,
persistence, mount-preview and task-scheduling regressions pass after this change.
These checks do not complete point 6 or the full two-army acceptance gate.

**Declaration sequencing verification (2026-09-11):** two queue checks and 15
actual-roster scene checks cover delayed reactions, independent reaction-target
and charge-move choices, real front/flank contacts, roll-only failed movement,
strongest-source Flee and its live animation, both AI entry points, save/reload,
blocked snapshots, input guards and the Resolve Charges phase button. Its
offscreen HUD screenshot was inspected. The formed-to-Skirmisher suite now has
49 checks, including queued loose-target save/reload without sight reselection;
82 Skirmisher scene checks pass with ordinary moves after declaration closure.
Counter Charge, Stand & Shoot, marching, persistence, First Charge, Shieldwall,
mounted-preview and named-task regressions also pass under sequential memory
caps. No full-suite or complete two-army acceptance run was performed.

Rule wording verified for point 5:
[Dragon Armour](https://tow.whfb.app/special-rules/dragon-armour), FoF p. 184;
[Chaos Armour](https://tow.whfb.app/special-rules/chaos-armour-warriors-of-chaos), RH p. 81;
[Elven Reflexes](https://tow.whfb.app/special-rules/elven-reflexes), FoF pp. 145, 185;
[Valour of Ages](https://tow.whfb.app/special-rules/valour-of-ages), FoF p. 185;
[Ithilmar Weapons](https://tow.whfb.app/special-rules/ithilmar-weapons), FoF p. 185;
[Ithilmar Barding](https://tow.whfb.app/special-rules/ithilmar-armour-ithilmar-barding), FoF p. 185;
[Ensorcelled Weapons](https://tow.whfb.app/special-rules/ensorcelled-weapons), RH pp. 81, 115;
[Mark of Chaos Undivided](https://tow.whfb.app/special-rules/mark-of-chaos-undivided), RH pp. 82, 116.

**Sources and boundaries:** faction-rule and spell summaries above are based
on the selected exports, not inferred from names. The three linked item pages,
[First Charge](https://tow.whfb.app/special-rules/first-charge),
[Counter Charge](https://tow.whfb.app/special-rules/counter-charge),
[Impetuous](https://tow.whfb.app/special-rules/impetuous) and
[Official FAQ index v1.5.3](https://tow.whfb.app/faq) were checked for this audit.
Before implementing each entry, verify its full current wording, book page
and applicable errata. Neither roster selects a Terror-causing model, a Chaos
Wizard, a Bound item, or a specific Chaotic Cult. Do not add those as unrelated
army-readiness prerequisites. Non-catalogue demonstration spells remain excluded.
**LEFTOVER:** import, command/profile ownership, selected items, spell generation
and the point-5 faction core are implemented as marked above. Unchecked movement,
psychology, spell effects and listed partial-rule dependencies remain. This does
not certify list legality or claim the armies passed all acceptance scenarios.

## Done
- [x] Armour Bane (X) — natural 6 to wound improves that attack's AP by X
- [x] Lance / charge-only melee (strength & AP bonus only while charging)
- [x] Charge-conditional AP (e.g. Halberd `-1 (-2)`)
- [x] Melee strength bonus (e.g. Halberd S+1 always, Lance S+2 on charge)
- [x] Furious Charge (registry builder)
- [x] Regeneration (registry builder; save value only if in data)
- [x] Unbreakable (registry builder)
- [x] Multiple Shots (X) — the firer chooses, per unit, before rolling (p. 174).
      `has_multiple_shots()` / `roll_ranged_shots(multiple)` /
      `expected_ranged_shots()` in `models.py`, `multiple_shots=` through
      `simulate_battle`, prompt in `MyApp.shootAt` (now async), AI policy in
      `special_rules.should_fire_multiple`. The -1 To Hit per shot and the
      per-model dice roll were already correct; the *choice* was what was
      missing — the old code always fired multiple. `ranged_hit_chance` in
      `toHitAndToWound.py` derives P(hit) by walking all six faces through
      `to_hit_ranged`, so the decision cannot drift from the dice.
      Corrected on the way: the previous line here claimed "D3 approximated
      as 3", which had not been true since the dice parsing landed.
      LEFTOVER: a joined character follows the unit's call rather than being
      asked separately. The rule binds the *unit*, and a character is not one
      of its models, so it arguably gets its own choice; one prompt per volley
      was preferred over two.
      Found from this and fixed with it: `to_hit_ranged` returned a flat miss
      for any effective BS above 5 or below 1 — see the two To Hit entries
      below, which Multiple Shots' -1 is what made reachable.
- [x] BS of 6 or Higher (p. 138) and 7+ To Hit (p. 139) — the ranged To Hit
      ladder stopped at BS5 and sent everything else to `return False`, so all
      20 BS6/BS7 models in the catalogue (Keeper of Secrets, Lord of Change,
      Dark Elf Dreadlord, Glade Captain...) **missed with every shot they ever
      took**, and a shot pushed past a 6 by modifiers silently missed instead
      of taking its natural-6 second roll.
      BS6+ now hits on 2+ and re-rolls a failure, the re-roll growing easier
      with Ballistic Skill (BS6 2+/6+ through BS10+ 2+/2+). A needed roll of
      7-9 hits on a natural 6 followed by a 4+, 5+ or 6; 10 or more is
      genuinely impossible.
      Modifiers now move the **target number** rather than Ballistic Skill.
      For BS1-5 those are the same thing, since the target is 7 - BS, which is
      why the old code got away with it; for BS6+ they are not, because a
      reduced BS is a different row of the table rather than a harder roll.
      `ranged_hit_requirement` in `toHitAndToWound.py` is the single table,
      read both by `to_hit_ranged`, which rolls the dice, and by
      `ranged_hit_chance`, which works the probability out exactly for the
      Multiple Shots decision. Keeping one source of truth is the point: the
      first draft of `ranged_hit_chance` sampled `to_hit_ranged` over all six
      faces to avoid a second copy of the ladder, which stopped being exact
      the moment a rule needed a second die.
      Checked against 200k rolls per case over 54 BS/modifier combinations:
      predicted and rolled agree to within 0.0022, which is the sampling
      noise of that many trials.
      NOTE: Curse of Arrow Attraction re-rolls a natural 1, and a BS6+ model
      that rolls a 1 has already had the p. 138 re-roll; the rulebook does not
      say which wins, and the two are left to stack.
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
- [x] Move & Shoot (p. 174) — a weapon with this rule fires in the Shooting
      phase even if the model marched, so it is the exception to Marching.
      Parsed onto the weapon as `move_and_shoot` in `battlescribe.py`, read by
      `model.fires_after_marching()`, and `MyApp.shootAt` lets the volley
      through and says which weapon earned it.
      The catalogue carries **both** this and its opposite, `Move or Shoot`,
      and the two names differ by a single word — 16 weapons have `Move &
      Shoot`, 25 have `Move or Shoot` (one of them spelled `Move Or Shoot`).
      Matching on "move ... shoot" would therefore have let 25 weapons fire
      that must not, so the regex matches the joiner itself and two tests hold
      that line from both directions: nothing with `or` is flagged, nothing
      with `&` is missed.
      LEFTOVER: read from the *unit's* equipped weapon, so a joined character
      firing its own weapon is judged by the unit's. Magic Missiles stay
      barred either way — this is a weapon rule, not a licence to cast.
      Corrected from a game log: a Gyrocopter loaded from a quicksave was
      refused its shot despite carrying a Clattergun. `persistence.py` stores
      the weapon *dicts*, so a save written before `move_and_shoot` was parsed
      has no such key, and reading it returned False — the exemption failed
      silently on exactly the saves most likely to exist.
      Fixed at the root rather than per-flag: `_create_unit` already treated a
      catalogue weapon as the source of truth and let saved data fill only the
      fields it lacked, but *only* for weapons already on the model's own
      profile. A Clattergun is not part of the Gyrocopter's profile, so it fell
      to the branch that took the stored dict wholesale. It now calls
      `give_weapon()` first for anything the catalogue knows, so every weapon
      is re-derived on load and a save can no longer pin a weapon to the flags
      that existed when it was written. Only weapons the catalogue has never
      heard of are taken from the save as-is.
      Verified by loading the failing quicksave offscreen: all 13 ranged
      weapons come back with the flag, both Clatterguns True, and the Warbow
      and Grand cannon — which carry `Move or Shoot` — correctly False.
      `fires_after_marching()` also falls back to the rule *names* the saved
      dict keeps, which covers a weapon the catalogue no longer lists; both it
      and the parser go through one `has_move_and_shoot()`, so the `&`-not-`or`
      distinction is defined once.
- [x] Moving and Shooting (p. 139) — "models that have moved for any reason
      during this turn (including rallying and reforming) have less time to aim
      and suffer a -1 To Hit modifier".
      Found missing while implementing Quick Shot, which exists only to negate
      it: `to_hit_ranged` had taken a `moved=` argument since it was written,
      but nothing in the game ever passed it True, so the penalty had never
      once been applied. Quick Shot would have been a rule that switched off
      nothing.
      "For any reason" is wider than the engine's `hasMovedThisTurn`, which is
      set by an ordinary move, a Panic flee and a flee reaction, but
      deliberately *not* by a manoeuvre on the spot. `MyApp.shootAt` therefore
      asks four flags — `hasMovedThisTurn`, `manoeuvreThisTurn`,
      `moveSpentThisTurn` and `attemptedRallyThisTurn` — and the log says which
      one fired, so a -1 can always be traced to the thing that caused it.
      This closes the LEFTOVER left by Redress the Ranks: redressing counts as
      moving for shooting, and there is now a penalty for it to hook to.
      Corrected on the way: `battleFunctions._ranged_tohit_report` kept a
      *second* copy of the To Hit ladder and re-derived the modifiers itself,
      so the log's stated target was computed separately from the dice that
      rolled against it. It now asks `ranged_hit_requirement`, and works out
      which modifiers a weapon waives by asking the same function rather than
      restating the list — a test holds the two together across BS3/BS6 and
      both weapon kinds, because a report that drifts from the dice is worse
      than no report.
      LEFTOVER: a rally is counted by `attemptedRallyThisTurn`, so a *failed*
      rally also takes the -1. It is still fleeing and cannot shoot at all, so
      nothing observable rides on it today.
- [x] Quick Shot (p. 175) — first half done, second half blocked.
      "Does not suffer the usual -1 To Hit modifier for Moving and Shooting":
      parsed onto the weapon as `quick_shot` and folded into the existing
      `ignore_to_hit_penalties` set that Blunderbusses already used, so the
      waiver lives in the one function that owns the To Hit table rather than
      in a new branch beside it.
      The catalogue spells it two ways — 32 weapons `Quick Shot` and one
      `Quick Shoot` — the same trap Move & Shoot set, so `has_quick_shot()`
      makes the second 'o' optional and a test checks all 33 are flagged.
      As with Move & Shoot the check falls back to the rule *names*, so a
      quicksave written before the flag existed still gets the waiver.
      Logged either way: `Quick Shot` when a moved unit still hits on its
      unmodified number, `Moving and Shooting` when it does not, and
      `rule_skipped('Quick Shot')` when the unit stood still and there was no
      penalty to ignore — the three cases are indistinguishable on screen.
      BLOCKED, then DONE: "can use them to make a Stand & Shoot charge
      reaction regardless of how close the charging unit is" had nothing to
      attach to, because the reaction did not exist. It does now — see Stand &
      Shoot below — and `standAndShootOption` passes the weapon's Quick Shot
      through to `can_stand_and_shoot`, which waives the distance test and
      says so. Both halves of the rule are therefore coded.
- [x] Stand & Shoot (p. 120) and Standing and Shooting (p. 139) — the third
      charge reaction. `combat_resolution.py` offered only `["hold", "flee"]`,
      so a unit of handgunners watched a charge come in with its weapons
      loaded; it is now offered first when the unit is entitled to it.
      The entitlement is five conditions and each refusal logs which one it
      failed, because on the board they are indistinguishable: armed with a
      missile weapon, line of sight to the charger, not fleeing, not already
      engaged, and the charger no closer than its own Movement characteristic.
      That last one is `special_rules.can_stand_and_shoot`, kept pure and
      tested — the rule bars a distance *less than* the Movement, so exactly
      the Movement may still shoot, and Quick Shot skips the test entirely.
      Distance is measured edge to edge with `obb_distance`, the same oriented
      box maths the 1" rule and the Leadership bubbles use, and the charger's
      Movement comes from `get_movement()`, which looks through to a mount or
      a chariot's beasts.
      "Armed with missile weapons" is not the same question as "has one in
      hand": a unit expecting a fight may have equipped its melee weapon, so
      `model.missile_weapon()` finds the bow whether or not it is equipped,
      and the reaction equips it for the shot and puts the melee weapon back
      afterwards — through `equip_weapon`, which also rewrites `special_rules`,
      rather than by assigning `equipedWeapon` and leaving the rules behind.
      The shot itself reuses `MyApp.shootAt`, which already knows about ranks,
      hills, Multiple Shots and joined characters. Three things differ and all
      three are the rule rather than convenience: -1 To Hit, **no** long range
      modifier however far away the charger is (p. 139 — verified: a Handgun
      needing 4+ needs 5+ either way, where an ordinary shot at long range
      would need 6+), and no Panic test for the charging unit (p. 120).
      `hasAttackedThisTurn` is deliberately NOT set. It is the unit's Shooting
      phase allowance, which a reaction does not spend — and it doubles as
      "this combat has been fought", so setting it would have barred the unit
      from the fight it had just shot at. That trap is recorded under Pursuit
      into a New Combat and would have been silent here.
      Verified offscreen with the real armies: State Missile Troopers are
      offered their Handgun against a Captain 20" off, refused at 0" (inside
      his Movement of 7"), and refused while fleeing.
      Corrected from a game log, and it had disabled the whole reaction: every
      charge reported `Grave Guard Unit is 0.0" away, inside its own Movement
      of 4"` and refused. `chargeAndChargeReaction` runs *after* the charger
      has been swept into contact — that is why it is handed `oposUnit`, to put
      the unit back if the charge is cancelled — so the charger's own transform
      always reads zero and no unit could ever have Stood & Shot.
      The distance and the line of sight are both taken from the declaration
      position now, which is also what the FAQ asks for: "when are line of
      sight and cover determined for a unit that declares a Stand & Shoot
      charge reaction? When the charge reaction is declared."
      The offscreen check above passed because it called `standAndShootOption`
      directly with the units where they stood; only a real charge moved the
      charger first. Two tests now drive the function with the charger nose to
      nose and its declaration position given separately, which is the shape
      the bug had.
      **Its casualties count towards the combat that follows** (p. 151): "each
      side's basic combat result is equal to the number of unsaved wounds it
      caused during this Combat phase, plus any unsaved wounds a unit caused by
      shooting if it chose to Stand & Shoot as a charge reaction during this
      turn". Missing from the first pass, which resolved the shot and then
      threw the number away, so the reaction could kill five models and count
      for nothing in the fight it had just softened.
      The tally is banked on the shooter as `standAndShootWounds` in the
      Movement phase and spent in `_verySimpleBattleInner`, because the two
      halves happen in different phases and nothing else carries state between
      them. It is reset at the Start of Turn with the other per-turn flags,
      which is exactly the "during this turn" the rule asks for, and it is
      persisted — the guard test derived from the FSM's reset block caught it
      unsaved on the first run, which is the second time that test has paid
      for itself.
      It gets its own row in the combat result table rather than being folded
      into `Wounds caused`, because a combat can be won on it alone: 2 wounds
      against 3 with equal ranks is a loss by 1, and the same combat with 3
      banked from the reaction is a win by 2.
      The AI takes the reaction whenever it is entitled to it. There is no
      trade-off to weigh: the unit holds afterwards either way, the reaction
      does not spend its Shooting phase, and the charger tests for Panic in
      neither case — so holding instead is simply worse. It was written to
      hold at first, which quietly wasted every volley the AI was owed.
      `autoHold` is *not* the AI and still holds: it is set for a pursuit, and
      a unit reached by one was never declared a charge against, so it gets no
      reaction at all (p. 157). That is logged rather than silent, since it
      looks identical to the rule failing.
      LEFTOVER: line of sight is `los_block_point`, which is terrain only. The
      unit-blocking half is only half modelled anywhere in the engine, so a
      unit can Stand & Shoot through a friend.
      LEFTOVER: the FAQ case of being charged by two units where only one is
      too close is not modelled — reactions are resolved per contact, so each
      charge asks its own question and the two never meet.
      LEFTOVER: the FAQ's ruling that a unit which Stands & Shoots (or Fires &
      Flees) may not then cast a Magic Missile or Magical Vortex is not
      enforced.
      Fire & Flee is built on this reaction and is done — see the Dwarf rules
      below, where the catalogue keeps it.
      Now unblocked by this: `Dwarf Crafted` (no -1 To Hit on a Stand & Shoot)
      finally has a modifier to cancel.
- [x] Move or Shoot (p. 174) — DONE. "Cannot be used in the Shooting phase if
      the model equipped with it moved for any reason during this turn
      (including rallying and reforming)." All 25 weapons that carry it are
      artillery, which is what the flavour text describes.
      The hard part was already built. "Moved for any reason" is wider than
      `hasMovedThisTurn`, which a manoeuvre deliberately does not set, and
      Moving and Shooting had already had to answer that question — so the
      same four flags decide both, computed once at the top of `shootAt` and
      read by each. The rule names rallying and reforming explicitly, and both
      are among them.
      Parsed as `move_or_shoot` in `battlescribe.py` and read by
      `model.cannot_shoot_after_moving()`, with the same fallback to the rule
      *names* that Move & Shoot uses, so a quicksave written before the flag
      existed is still barred.
      The `&`-not-`or` trap is now guarded from both directions: matching
      "move ... shoot" here would silence the 16 weapons written to fire on the
      move, exactly as the reverse would have let 25 artillery pieces fire
      after moving. Each rule has a test asserting the other's weapons are not
      caught.
      A Stand & Shoot is exempt. The rule bars the *Shooting phase*, and a
      charge reaction is not it — nor has the defender moved, since the flags
      are cleared at the start of every turn.
      Logged both ways, because a cannon that does not fire looks identical to
      a cannon the player forgot to shoot with: the bar names what the unit did
      (`marched`, `rallied`, the manoeuvre, or `moved`), and a war machine that
      held still says so as it fires.
- [x] Ponderous (p. 175) — "a weapon with this special rule suffers a To Hit
      modifier of -2 for Moving and Shooting, rather than the usual -1". That
      is the whole rule. The line here previously described it as
      "move-or-shoot / initiative penalty", which is two rules it does not
      have — a guess from the name that would have barred these weapons from
      firing after moving at all.
      Parsed as `ponderous` in `battlescribe.py` (9 weapons, one spelling) and
      applied in `ranged_hit_requirement`, which now decides the Moving and
      Shooting penalty in one place for all three cases.
      A weapon with **both** Ponderous and Quick Shot takes the plain -1: the
      FAQ has them "effectively cancel one another out". Not hypothetical —
      `Naptha bombs` carries both, so the branch is reachable from the
      catalogue and is tested against that weapon rather than a fixture.
      This is not a fringe rule: **the Handgun and the Crossbow are both
      Ponderous**, which is most of the missile fire in an Empire or Dwarf
      army. Moving with either now costs 4+ -> 6+.
      Corrected on the way, and the tests caught it: two Moving and Shooting
      tests asserted a plain -1 *using a Handgun*, so they were describing the
      general rule with a weapon that does not follow it. They are re-baselined
      on an Asrai Longbow, which carries neither rule, and the Handgun now has
      its own -2 tests.
      Corrected with it: `_ranged_tohit_report` printed `-1` for any modifier
      that changed the target at all, so a Ponderous weapon would have reported
      a number the dice did not use. It reads the size of the change now, which
      is why it says `moved -2`.
      Corrected from a game log: the rule line still read `Moving and Shooting
      — moved this turn -> -1 To Hit (5+ -> 7+)`, which is a two-point change
      described as one point, and never named Ponderous as the cause. The rule
      that fired now says its own name and carries the real delta, and the
      shot readout gained a `To Hit` line showing the whole sum:
      `BS3 4+  moved -2 (Ponderous)  long range -1  =  7+ (natural 6, then 4+)`.
      Each modifier names the rule that sized it, so `moved waived (Quick
      Shot)` and `moved -1 (Ponderous and Quick Shot cancel out)` are both
      readable without knowing the weapon's rule list.
- [x] Killing Blow (p. 172) — DONE. A natural 6 To Wound for an attack made
      **in combat** allows an infantry or cavalry model no armour or
      Regeneration save (a Ward save is attempted as normal), and an unsaved
      wound from one costs the victim **all of its remaining Wounds**.
      `slaying_blow_struck` in `battleFunctions.py` is the whole condition
      (named for the pair once Monster Slayer joined it), and
      each clause is one the rule or its FAQ states outright: not a missile
      attack, a natural 6, the attack actually wounded, the model has the rule,
      and the target is infantry or cavalry.
      "Infantry" and "cavalry" are the *parent* categories, and a sub-category
      follows its parent unless it says otherwise (p. 188), so
      `troop_types.is_infantry` covers monstrous infantry and swarms, and
      `is_cavalry` covers monstrous cavalry. A behemoth, monstrous creature,
      chariot, war beast or war machine cannot be felled by it.
      FAQ: "If a model cannot wound an enemy, it cannot kill it" — so a target
      the attacker could not have wounded at all is safe, which
      `slaying_blow_struck` reads from the To Wound target being above 6.
      Too Tough to Wound now supplies that impossible target when Strength is
      six or more below Toughness (p. 140). It is also why an attack that wounds
      *automatically* cannot use the rule: there
      is no roll to be a natural 6.
      `check_saves` grew a `slaying_blow` flag rather than a second copy of the
      save sequence, so the Ward save keeps its place in the middle of the
      order.
      The count is carried out of band, as `take_last_slaying_blows()` beside
      the existing `take_last_combat_report()`. `simulate_battle`'s five-value
      return is read from a dozen call sites, and a Killing Blow cannot simply
      be one more wound in the pool: `applyWounds` divides the pool by the
      profile's Wounds, so a Killing Blow against a W3 model would have spilled
      its excess onto the next model. The FAQ is explicit that it does not —
      "the excess wounds are lost" — so `applyWounds` takes killing blows off
      first, a whole model at a time, and pools only the ordinary wounds.
      The same FAQ gives the order to apply them in: Killing Blows first, then
      Multiple Wounds, then single wounds. Only the first half of that ordering
      is modelled, since Multiple Wounds (X) is still a deferred item.
      CORRECTED on the way, and it predates this rule: the whole weapon-flag
      block in `weapon_from_profile` sits **inside the `if is_ranged:` branch**,
      so `strike_first` and `strike_last` were only ever set on ranged weapons
      — which is to say never on the melee weapons that are the only ones
      carrying them. Strike Last worked at all only through its fallback to the
      rule *names*. The three melee flags are set for every weapon now, and the
      test that every Killing Blow weapon is flagged is what found it.
      LEFTOVER: not applied to Impact Hits. Those are hits rather than attacks,
      and the rule says "an attack made in combat" — but the FAQ does let a
      model's *own* Armour Bane reach Impact Hits, so the same might be argued
      here.
      LEFTOVER: spells never get it, which the FAQ is blunt about
      ("Absolutely not") and which holds for free, since `resolve_magic_hits`
      has no attacking model to carry the rule.
- [x] Monster Slayer (p. 173) — DONE. Word for word Killing Blow with one
      word changed: a natural 6 To Wound in combat, no armour or Regeneration
      save, Ward as normal, and the victim loses every Wound it has left — but
      only where the target's troop type is **monster**.
      Implemented by generalising Killing Blow rather than copying it, because
      the two are one mechanic and share every part of the machinery: the
      counter (`take_last_slaying_blows`), the `slaying_blow` flag on
      `check_saves`, and the whole-model removal in `applyWounds`. What differs
      is only which target each can fell, so `slaying_rule_for` answers that in
      one place and `slaying_blow_struck` returns the *name* of the rule that
      struck instead of a bool.
      That generalisation is not tidiness for its own sake — it is the only way
      to get **the weapons that carry both** right. Clearver-Limbs,
      Decapitating Claws, Decapitating Strike and Great Blade (Deadly Strike)
      have Killing Blow *and* Monster Slayer, so the weapon cannot decide which
      applies; the target does. Since infantry-or-cavalry and monster are
      disjoint, at most one can ever be in play, which is also why `applyWounds`
      can name the rule from the victim's troop type without being told.
      TRAP, and it has its own test: **"monstrous" is not "monster"**.
      Monstrous infantry and monstrous cavalry are sub-categories of infantry
      and cavalry, and are felled by Killing Blow, not by this. Only monstrous
      creatures and behemoths are monsters (p. 196), which is what
      `troop_types.MONSTERS` holds. Any substring match on "monst" would have
      got this exactly backwards for two whole troop types.
      Where it lands is on 7 weapons and, unlike Killing Blow, **no model
      profile at all** — so it only ever reaches a model through the weapon in
      hand, which is what `_strike_rule` already handled.
      The same FAQ answers apply, being asked of both rules together: an enemy
      too tough to be wounded cannot be killed, and an attack that wounds
      automatically cannot use it.
      LEFTOVER: as with Killing Blow, not applied to Impact Hits or Stomp
      Attacks, and never to spells.
      CORRECTED on the way, and nothing to do with the rule: building the test
      save exposed that **a reloaded profile reverts to the catalogue after the
      first combat**. `reset_characteristics()` restores
      `_base_characteristics`, which a fresh model fills from the catalogue,
      and the load set only `characteristics` — so a save's stats held until
      the first exchange ended and were then silently thrown away, taking any
      roster change with them. The monster on the test board turned back into
      a Dire Wolf between rounds, which is how it was noticed. The load rebases
      the profile now.
- [x] Strike First — DONE (p. 177). Initiative becomes 10 before any other
      modifier; `battleFunctions.base_initiative` does the substitution and
      `strike_initiative` applies the charge bonus after it, so a charge cannot
      push it past the cap of 10.
- [x] Strike Last — DONE (p. 178). Initiative becomes 1 the same way. A model
      with both is left on its own characteristic, because the rules say they
      cancel one another out.
      Both come from the weapon far more often than the profile: 12 weapons
      carry Strike Last against no models at all, and the plain **great weapon**
      is one of them, so this fires constantly. `model.has_strike_first` and
      `has_strike_last` therefore read the melee weapon actually in hand
      (`active_melee_weapon`, so a sheathed great weapon or a Lance outside a
      charge does not count) as well as the model's own rules.
      The catalogue spells them `Strike First`/`Strikes First` and
      `Strike Last`/`Strikes Last`, so `battlescribe.has_strike_first` and
      `has_strike_last` take the optional 's'. The match is anchored: the Warp
      Lightning Cannon has **Lightning Strike**, which a substring test reads as
      a Strike rule. Weapons parsed by `weapon_from_profile` also get
      `strike_first`/`strike_last` flags, and the string test is kept as a
      fallback for weapons a save wrote before the flags existed.
      LEFTOVER: "before any other modifiers are applied" is read literally, so
      a Strike Last model that charges 5" strikes at I4, not I1 — the charge
      bonus is a modifier and the rule only replaces the characteristic. That
      is what the wording says, but it does mean a great-weapon unit that
      charges is not actually striking last.
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
- [x] Hatred (X) (p. 171) — DONE. A failed To Hit may be re-rolled against a
      hated enemy, in the first round of a combat only, in melee only. The
      rules page carries no FAQ or errata, so the printed wording is all of it.
      Three separate problems, only one of which was the rule; see
      `HATRED_PLAN.md`.
      **The engine could not tell which round of a combat it was in.**
      `chargedThisTurn` is cleared every turn, so it cannot see round one of a
      combat that began last turn, and `startOfPhaseEngaged` answers a
      different question (for pursuits). So `roundsFought` on the unit, counted
      up in `enterCombatPhase` beside `startOfPhaseEngaged`, reset in
      `exitInCombat` beside `isInCombatWith`, and saved — without that last
      part a reload mid-combat hands Hatred back. Pursuit into a New Combat is
      the awkward case: those units are dragged in *after* the phase has
      counted, so the two of them are set to their first round there.
      This counter is worth more than Hatred: Frenzy and Impetuous both want
      it, and it is the honest version of what "first round of combat" means.
      DECIDED: the counter is **per unit, not per pairing**. When a fresh enemy
      charges a fight already in progress, the charger gets its Hatred and the
      unit already fighting does not. LEFTOVER: strictly the rule is per
      combat, which would cost a dict on every unit and in every save.
      **X is prose, and not one of the eight spellings in the data is a faction
      name.** `Orcs & Goblins` is the army called `Orc and Goblin Tribes`;
      `Dwarfs` is `Dwarfen Mountain Holds`; `High Elves` is `High Elf Realms`.
      `HATRED_TARGETS` in `special_rules.py` maps the prose to factions and to
      keywords, and `resolve_hatred_target` looks up the **whole string first**
      and only then splits — because `Orcs & Goblins` is one faction while
      `Warriors of Chaos & Daemonic models` is two different tests, and
      splitting on `&` first invents an army called Orcs.
      `Daemonic` is a keyword, not an army: it is a real rule on 31 models, so
      that half of the test reads the target's own rules list.
      DECIDED: `Hatred (Dwarfs)` covers Chaos Dwarfs as well as the mountain
      holds.
      One entry in the data reads `Beastmen Breyherds`, a typo for
      `Brayherds`. Aliased rather than corrected: the catalogues are vendored
      and get re-cloned. The Bretonnian Grand Master's roster rule grants the
      same Hatred spelled correctly, so both spellings reach the matcher.
      Identifying the enemy needed no new plumbing: `characteristics['Faction']`
      is on a live model, survives `reset_characteristics()`, and survives a
      reload since the rebase fix that went in with Monster Slayer. A model the
      catalogue has never heard of has no faction and can only be hated by
      'all enemies' — which is a logged line, not a crash.
      The re-roll is in the melee branch of `simulate_attack`. Two things that
      were easy to get wrong and are done deliberately: the re-rolled die goes
      back through the same `to_hit` rule modifiers the first one did rather
      than being compared raw, and a re-roll is never itself re-rolled.
      The guard-rail test is that **every Hatred spelling in the catalogue
      resolves to a non-empty set of factions or keywords**, so an army book
      adding `Hatred (Skaven)` fails loudly instead of silently hating nobody.
      REMOVED on the way: seven profiles under
      `army_units_cat/orc_and_goblin_tribes/` carried a bare `"Hatred"` with no
      enemy named. Every one of those models is in the catalogue, which
      supplies `Hatred (Dwarfs)` and wins, so the entries were unreachable.
      Deleted rather than guessed at — a bare `Hatred` could only have meant
      'all enemies' or nothing, and both are wrong. A test keeps them gone.
      LEFTOVER: not applied to Impact Hits or Stomp Attacks, which are hits
      rather than To Hit rolls.
- [x] Magic Resistance (-X) — DONE (Rulebook pp. 108, 173): an enemy spell
      targeting the unit takes the strongest negative casting modifier among
      its living members, including a joined character and mount/crew/beast
      profiles. Values never add together. This is **not** a Ward save, damage
      reduction, or a bonus to a dispel roll.
      Static signed values are parsed strictly. The two Ushabti profiles that
      spell it `Magic Resistance (2)` mean -2; an unknown parameter, bare rule,
      or `-D3` is logged as unresolved, never guessed or rolled per incoming
      spell. Parameter-aware names allow replacement, downgrade and removal
      without keeping a stale stronger value.
      `Spell._attempt` checks the actual target and confirmed ownership before
      rolling: friendly and Self spells take no penalty; ground-targeted
      vortices and their subsequent damage never acquire one. The casting
      value is unchanged, and dispelling must beat the reduced casting result.
      Natural doubles still decide ordinary miscasts/perfect invocations;
      Miscast-table overrides retain their stated casting result (p. 109).
      Logs identify the strongest source, dice, separate casting bonus,
      resistance, final result and threshold, or the reason it was skipped.
      **Bound spells (p. 109):** add their full Power Level, not Wizard Level,
      and never miscast or invoke perfectly. A selected **Ruby Ring of Ruin**
      imports its real Fireball grant at Power Level 1 (p. 342), is selectable
      by a non-Wizard, and does not turn the bearer into a Wizard. Ordinary
      and Bound copies coexist. One Bound attempt per phase is tracked
      separately from ordinary spell slots, survives reload, and still obeys
      marching, fleeing, combat and explicit no-more-spells restrictions.
      CORRECTED on the way: casting detours no longer reset phase state,
      scatter vortices again, increment combat rounds, or end a player's turn.
      Self spells resolve on the caster without an arbitrary target click;
      legal Assailments are no longer rejected by the shooting-only UI gate.
      Loading now replaces spell metadata while retaining coded classes and
      rebuilds saved character/mount links: a character or mount acquired
      after the save cannot leave its resistance on the restored unit.
      Tests cover real casting/dispel order, logs, parser edge cases, live
      sources, roster/UI Bound selection, phase detours and offscreen reloads.
      `saves/magicresistance.json` starts in Shooting: a Level 2 Master Mage,
      a non-Wizard Ruby Ring bearer, -1 Dwarfs, a -2 Runesmith joined to -1
      Dwarfs, a friendly resistant unit and an unresistant control. The scene
      generator is `tests/test_magic_resistance_scene.py`; targets have clear
      Fireball lanes within 24". The save was loaded and rendered offscreen.
      LEFTOVER: indirect army/item grants (Mark of Tzeentch, A Tingle in the
      Air, Witch Hunter choices, etc.) still need their own rules. Sorcerous
      Void's pre-deployment -D3 must be resolved and saved by that grant.
      LEFTOVER: other Bound item/special-rule grants, single-use item limits
      and uncoded spell effects are not implemented by the Ruby Ring path.
      Multi-model units still share one caster record rather than independent
      per-model Bound allowances. Existing wider magic limitations, including
      army-wide Power Drain and Miscast damage, are unchanged.
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
- [ ] Skirmishers — PARTIAL (Rulebook pp. 184–187; FAQ v1.5.3): baseline rule
      flag, no rank bonus, enemy-fire -1, 360° arc and fled-through Panic guard.
      Added saved per-model IDs/positions, connected edge-to-edge coherency,
      coherency-safe ordinary casualties, bridge-character replacement, legacy
      save migration and a ghost-only formation editor. Group/model movement
      shares validation and spends the longest individual travel once. Actual
      base ghosts show validity; Confirm/Cancel, width/gap/angle and phase locking
      make adjustment optional. Terrain tests count crossed models/features;
      flyers suffer landing terrain only. A lone surviving character stays alive.
      Corrected: jitter gaps, rank cleanup moving loose survivors, missing saved
      layouts, pre-reduced casualty counts, joined-character terrain death,
      exactly-M float precision and clipped editor labels. Movement squares now
      render unshaded after the 3D scene, so terrain/models cannot hide their edges;
      pixel checks cover every edge in ordinary movement, charging and the editor.
      The cursor now names CHARGE targets in cyan, MOVE in green, MARCH in amber,
      and BLOCKED in red with its reason. Human non-charge clicks open Confirm/Cancel
      without spending movement; charges retain Yes/No. Both preview and click use
      the same destination/contact checks. Corrected the sweep/contact rounding
      gap and charging flags after declining; stale indicators are cleared.
      Follow-up: 94 tests passed, including 32 Skirmishers scene tests; inspected
      1280x720 and 800x600 renders. The original milestone passed 349 neighbouring
      tests plus 75 geometry subtests; offscreen demo in saves/skirmishers.json.
      Charge follow-up (pp. 121, 187): cursor/direct commits now use M + 6
      (+3 Swiftstride maximum), not 2M; ordinary moves cannot spend extra charge
      reach. Status shows distance/maximum and the roll-needed readout handles
      contact epsilon. Two loose units without joined characters now move the
      closest charger first, form its reachable rank, then align defenders within
      M. Facing, frontage and model identities survive combat entry/save/load.
      Chosen-slot coherency casualties are removed and logged (FAQ v1.5.3).
      Corrected failure fallback falsely reaching contact, failed charges adding
      M, direct-call range bypass and exact-range precision. Unsupported pairings
      explicitly log legacy alignment. Validation: 612 tests and 75 geometry
      subtests, including 192 uneven-frontage combinations; real animations and
      1280x720/800x600 alignment renders checked.
      Formed-target correction (p. 186): a column charging a formed rear no longer
      snaps into a rank short of the enemy. Closest-model-first planning anchors
      every fighting base to the charged edge, keeps the formed defender fixed,
      preserves rear/flank bookkeeping and uses rear slots for other reachable
      models. Preview/resolution share the distance; tied closest models remain
      deterministic under rotation. No joined characters or already-engaged target
      in this path. Verification: 627 tests and 75 subtests, including three real
      rear-charge rotations and twelve face/frontage fixtures. Real animations
      and contact renders checked at 1280x720 and 800x600.
      Compact lifecycle correction (p. 185): leaving combat no longer immediately
      rebuilds a loose blob. Non-fleeing, unengaged units separate at Combat-phase
      end; engaged units and temporary spell transitions retain compact ranks.
      HOUSE RULE requested by the user: fleeing compact units wait for successful
      Rally (including Rallying Cry), not the printed end-phase timing. Failed
      Rally, phase changes, save/load and casualties preserve compact survivors.
      Separation preserves current positions, IDs, facing and attached characters
      through tiny uniform expansion; applied/refused changes are logged.
      Verification: 747 tests and 82 subtests passed, including human/AI Rally
      branches, actual flee animation, compact casualties and phase/save boundaries.
      Ground indicator follow-up: green normal-move and amber march bands show
      remaining movement; dashed cyan charge reach is independent and respects
      charge restrictions. Shared shader code wraps terrain and clears with the
      preview. Ordinary destination squares clamp through the existing per-model
      legality check; refused charges hide their squares without becoming marches.
      Corrected extra-charge-reach ordinary previews and copied-surface shader
      defaults. Verification: 761 tests and 82 subtests; inspected 1280x720 and
      800x600 screenshots with band-color and outside-range pixel assertions.
      Charge sight (pp. 103, 184, 186): strict >50% visibility now gates previews,
      human clicks, direct AI moves and declarations before reactions/dice.
      Actual rotated model bases block sight, including friendly/own models;
      angular intervals detect visible target edges and narrow gaps. Attached
      characters count once. UI and outcome logs carry visible/total and required
      counts. Corrected contact-position sight by querying original transforms
      without mutation; Stand & Shoot casualties do not revoke valid declarations,
      and pursuit is exempt. Verification: 804 tests and 82 subtests, including
      43 visibility tests and inspected allowed/blocked panels at both resolutions.
      Defender frontage correction (pp. 145, 187): replaced the charging-width
      cap with actual front-base contact, including corners. One charging file
      can now face three equal-base defending files when each model can reach
      within M; models without enemy contact still form behind. The existing
      log reports both frontages. Verification: 407 tests and 82 subtests;
      ten new rotated, Movement-limit and real charge/render regressions, with
      inspected contact and model pixels at 1280x720 and 800x600. Already-saved
      compact combats are not automatically re-formed.
      Formed-versus-loose contact stage (p. 186): once a supported formed charger
      reaches an actual front-base contact, only the loose defenders form up
      within M; the charger stays fixed without a second alignment wheel. Front
      slots include corner contact, with remaining models behind. Applied logs
      give rank/rear counts and M; unsupported loose pairings and missing real
      contact explicitly log legacy fallback. Front-edge checks reject side/rear
      contact with a front-row model. Verification: 844 tests and
      82 subtests, including 30 new geometry, pairing and actual charge cases;
      inspected 1280x720/800x600 renders with all-model pixel/framing checks.
      Formed ground-charge approach (pp. 103, 124, 126, 186): cursor, declaration
      and resolution now share the closest visible model, actual-base contact,
      one paid approach wheel and optional straight lead. Declaration revalidates
      before reactions/dice; blocked routes do not choose a farther model instead.
      Swept formation terrain drives M, charge rolls and hazards; Move Through
      Cover outcome logs remain shared. Failed charges leave defenders loose;
      Stand & Shoot survivors retain the target and use revised route terrain.
      Preview/cache cleanup covers missed rays, leaving the window and save loading.
      Corrected SAT corner precision, awaitable animation, stale distance labels
      and removed-target checks. Verification: 48 new route cases, 84 tests in
      the final focused gate, plus inspected live preview/contact renders at both
      resolutions. Earlier full run: 1599 passes, 165 subtests, eight failures,
      all reproduced on isolated committed 7414299 with matching catalogue data
      (bound-spell phase, older movement logging, Shieldwall and Veteran fixtures).
      Charge-click crash follow-up: a Skycutter charging loose Marauder Horsemen
      reached an unnamed `functools.partial` task, triggering Panda3D's
      `task.hasName()` assertion before declaration. Added an explicit task
      name without changing charge rules or arguments. A real isolated Panda
      task-manager regression reproduces the old failure and now passes,
      including deferred target/original-transform forwarding and completion;
      all 48 existing offscreen formed-charge tests also pass.
      Combat scoring (pp. 101, 152, 185; Unusual Formations FAQ v1.5.3): compact
      Skirmishers grant no flank/rear points and receive no rank bonus, without
      changing physical arcs. Skirmisher attackers can score against formed arcs
      but cannot disrupt ranks. Surviving formed US5+ flank/rear enemies do disrupt,
      including joined-character strength; terrain disruption remains independent.
      Corrected stale ranks and slain flankers contributing after casualties.
      Scores and numerical outcome logs are now resolved once after attacks.
      Individual shooting (pp. 137, 139, 184-185): shared per-model range/sight
      drives target highlighting, aiming counts, direct volleys and declaration-time
      Stand & Shoot. Own/friendly/enemy bases block sight, gaps are transparent,
      and eligible models fire in profile/range-band groups without editing files.
      Multiple Shots has one unit choice with mixed-range advice. Formed firing
      ranks inherit front-file sight; joined characters occupy their actual slot
      and shoot with their own weapon/range. The all-US1 enemy-fire check includes
      every live joined model and logs qualifying/total counts. Existing hill
      visibility and extra firing rank are preserved. Corrected reaction modifiers
      missing from dice/reports, centre-range summaries and unarmed character slots.
      Refused volleys retain shooting and roll no dice. Verification: 42 scoring
      and 34 shooting regressions, 164 scoring/psychology and 72 shooting/reaction
      passes; final charge/visibility/scoring/shooting gate 167 passes. Both aiming
      resolutions rendered and inspected. Full run: 1678 passes, 165 subtests,
      eight known baseline failures and one optional-keyword mock failure, since
      corrected and covered by that passing focused gate.
      LEFTOVER: active formation switching; shooting height/cover refinements;
      spell sight and non-Skirmisher-only battles retain legacy sight; charge-sight
      height/Large Target exceptions and detailed terrain (currently base-centre
      XY sight and conservative rectangles); arc-straddling declarations;
      continuous formed-route search and global contact maximization (currently
      bounded angles through +/-90 degrees and 0.5-inch delayed-wheel samples,
      not rule limits); formed-route wheel distance still uses arc length instead
      of the Movement FAQ's outside-front-corner straight-line displacement, and
      conservative sweeps still reject some permitted rear-corner crossings;
      flying obstacle/landing paths and fleeing-target redirects
      (explicitly logged legacy chase); mixed/joined/multi-charge form-up and stepped
      rear faces;
      exact contact orientation, alternative ranks before forced losses and
      individual charge paths/terrain; command/champion
      and exact troop-subcategory joining; minimum legal separation constrained
      by terrain, other units and board edges (invalid compact layouts are refused);
      joined-model rank contributions; forced incoherency choices; reinforcements,
      templates/low obstacles, individual detours, march Leadership and broader
      AI integration. Normal cursor destination sweeps remain conservative, and
      circular shader shading shows range, not obstacle-aware legal routes;
      unsupported pairings retain visual approximations. See SKIRMISHERS_PLAN.md for the
      staged remainder and precise limitations. Skirmishers permission can come
      from army-list special_rules, not only the catalogue model profile.
- [x] Scouts — DONE for deployment and first-own-turn charges (Rulebook p. 177;
      Official FAQ & Errata v1.5.3). Selecting an eligible unit offers ordinary
      deployment or setting it aside. Reserving Scouts is not a deployment drop:
      both armies finish ordinary deployment before any reserved unit is placed.
      When both sides scout, a D6 roll-off (ties re-rolled) chooses the first
      player, then drops alternate, skipping an exhausted side. Scouts count
      towards `firstFinishedDeploying`, not just ordinary deployment completion.
      Split-profile grants are recognised from rider/mount and live crew/beasts
      (pp. 192, 194, 204); a joined character must qualify separately.
      Placement is strictly MORE THAN 12" from each deployed enemy model's base,
      measured between rotated base rectangles, including joined characters and
      real skirmisher positions rather than empty corners of a regiment box.
      FAQ correction: there is NO own-deployment-zone exception. All bases must
      be on the table; overlap and impassable terrain are refused. Ordinary
      deployment also requires every base to fit inside its own deployment zone.
      Joining a character validates the final host footprint and restores ranks,
      ownership, physics and position if the join would make placement illegal.
      `deployedAsScouts` is deployment history, not the current keyword. Both
      human/AI movement commits and direct charge declarations refuse Scouts
      during their owner's first turn, before reactions or dice. The other
      player's completed turn does not remove the ban. Ordinary-deployed Scouts
      are unrestricted; normal movement and pursuit are not charge declarations.
      Refusal restores position/facing without spending movement or announcing
      a fictitious march. A joined late Scout also prevents its host charging.
      Corrected on the way: deployment could be skipped with End Phase, and the
      old wall-contact test accepted bases entirely beyond the walls. Bullet's
      broadphase contact query also missed units moved since the last physics
      frame; synchronous pair queries now find actual charge/join contact.
      Deployment stage, roll-off winner, first-finished player, per-unit choices
      and history are saved/restored. Loading does not re-roll, retains the active
      deployment player, clears stale placement tasks and resumes AI deployment.
      Old saves clear later Scout state. Both AI implementations select only
      current-stage candidates; a 200-attempt placement limit pauses the AI for
      manual placement instead of spinning forever. Rule decisions/refused drops
      log reasons and the decisive clearance/rolls; frame-by-frame previews stay
      quiet. `tests/test_scouts.py` and `tests/test_scouts_scene.py` cover 47 cases,
      including real offscreen application save/load, async AI task chaining,
      character joins, first-turn expiry, allowed pursuits and move rollback.
      Running the scene test as a script creates `saves/scouts.json`, ready for
      Player 1's first late Scout drop, and `screenshots/scouts.png`; the generated
      save was loaded and rendered offscreen.
      Vanguard now consumes `scouts_block_vanguard` and saved deployment history:
      an actual late Scout deployment excludes the unit from the Vanguard step.
      LEFTOVER: the engine still starts battle with Player 1; `firstFinishedDeploying`
      is recorded correctly but no scenario first-turn roll-off/bonus consumes it.
      LEFTOVER: irregular impassable terrain uses conservative bounding rectangles;
      custom scenario deployment zones and army-specific Scouts variants are not
      covered. AI charge planning is unchanged: illegal first-turn proposals are
      rejected by the shared execution gate rather than tactically replanned.
- [x] Vanguard — implemented with the movement-control limitations below
      (Rulebook p. 180, updated 25 June 2025; Official FAQ v1.5.3). Separate final
      substage of Deploy Phase, after every ordinary and Scout drop. Optional
      Move/Skip choice; both eligible armies roll off, re-roll ties and alternate
      units, continuing with the remaining side when the other has finished.
      End Phase declines the current player's remaining moves or completes its
      active manoeuvre; it cannot bypass an opponent or an open choice.
      `vanguard.py` owns eligibility, sequencing, movement commits and history.
      Basic Movement uses the slowest participating model (including mounts and
      joined Vanguard characters), with flight only when all participants can
      fly. No march multiplier or charge-range bonus. Table bounds, model-base
      overlaps, terrain and the normal 1" enemy end-position clearance apply
      (p. 118); there is no fixed 12" move or Vanguard-specific 12" exclusion.
      Existing wheel/advance, backwards/sideways at half M, Skirmisher movement
      and half-M redress controls work in Vanguard. Commit-time validation
      rejects excess distance and combining redress with another manoeuvre.
      Pre-game allowances are cleared on completion without spending the first
      battle turn's movement, marching, shooting or charge flags.
      FAQ: actual late Scouts cannot Vanguard, but normally deployed Scouts can.
      A non-Vanguard character prevents its formed host moving; Skirmishers may
      leave it at its original world position, restoring independent selection,
      physics and ownership. Invalid/zero-distance moves do not detach it. Moving
      Vanguard characters share their host's history. Only actual movement or
      redress records `madeVanguardMove`: both move-contact and direct charge
      paths refuse a first-own-turn charge before reactions/dice, using the
      owner's completed-turn counter. Declining keeps ordinary charge rights;
      pursuit and later-turn charges remain available.
      Save/load retains stage, roll-off winner, active unit, completion/history
      and spent manoeuvre allowance. No re-roll, movement refund or duplicate
      deployment resume on loading; older saves clear stale Vanguard fields.
      Both AI selectors use a bounded forward-move policy and alternate through
      real Panda tasks. Eligibility failures, declines, moves and refused charge
      declarations log the deciding values/reasons; previews remain quiet.
      Corrected on the way: explicit AI coordinates no longer require a mouse;
      Vanguard uses the actual front edge, not the inset movement marker; wheel
      sampling is capped at the allowance, backwards uses half M, and loading an
      active move restores its right-click binding. Selection cannot switch
      units halfway through Vanguard.
      Tests: 40 cases across `tests/test_vanguard.py` and
      `tests/test_vanguard_scene.py`, including actual offscreen movement/render,
      both AI task chains, wheel/redress, own-turn charge enforcement, character
      separation/rollback, enemy clearance, saved active manoeuvres and old saves.
      `python -m tests.test_vanguard_scene` creates `saves/vanguard.json` and
      `screenshots/vanguard.png`, then verifies the saved state by loading it.
      Starts at Player 1's Vanguard choice with AI off: Normal Rangers may leave
      their non-Vanguard character behind, P1 Ranked Vanguard can manoeuvre,
      P1 Blocked Vanguard cannot move with its character, Warriors provide the
      opposing Vanguard move, and the late Scouts remain ineligible. These are
      test-only Vanguard grants, not a legal army list.
      LEFTOVER: inherited movement UI offers one wheel followed by advance, not
      repeated wheel/advance segments, paid turns or full reforms (pp. 124-125).
      These missing ordinary manoeuvres are not supplied by Vanguard itself.
      LEFTOVER: path collision sweeps use the regiment footprint (conservative
      for Skirmishers), and irregular impassable terrain uses bounding rectangles.
      The AI advances or holds rather than choosing tactical pre-game manoeuvres.
      Scenario-specific deployment/first-turn rolls are unchanged: battle still
      starts with Player 1, independently of the Vanguard roll-off.
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
      Corrected since: a mounted character was losing the rule. The roster
      importer skipped mount subtrees when gathering special rules, so a
      Captain of the Empire on a Demigryph arrived with the beast's name and
      none of its rules; `model('Demigryph', '')` then found nothing, because
      the catalogue has no standalone Demigryph profile, only Demigryph
      Knight. `is_swiftstride` looks through to the mount, found an empty
      model, and said no — and because a joined character without the rule
      breaks the unit's claim, the Demigryph Knights it joined lost it too.
      The data was never wrong: the roster carries Swiftstride directly under
      the Captain's mount selection. `_collect_mount_rules` now exports it as
      `mount_special_rules` and `MyApp.applyDataRules` puts it on the mount
      model, where the look-through expects it. Fear, Counter Charge and First
      Charge were being dropped the same way and come back with it.
      LEFTOVER: converted army files under `strategy_armies/` predate this and
      carry no `mount_special_rules`; `Bm_army.json` was regenerated, the rest
      need re-importing before their mounts' rules reach the table.
- [x] Move Through Cover — DONE (Rulebook p. 174; difficult/dangerous terrain
      p. 269; Official FAQ & Errata v1.5.3). Checked the
      [rule](https://tow.whfb.app/special-rules/move-through-cover),
      [difficult terrain](https://tow.whfb.app/battlefield-terrain/difficult-terrain),
      [dangerous terrain](https://tow.whfb.app/battlefield-terrain/dangerous-terrain),
      [joined-character FAQ](https://tow.whfb.app/faq#what-happens-if-a-unit-with-the-move-through-cover-special-rule-is-joined-by)
      and [charge FAQ](https://tow.whfb.app/faq#does-a-unit-with-move-through-cover-discard-the-highest-dice-when-making-a-charg).
      Registered the keyword; the model ignores terrain's -1 Movement and
      re-rolls initial Dangerous Terrain 1s once, not repeatedly until safe.
      A second 1 still loses a Wound. Tests remain once per model per feature;
      the existing variable-damage hook is retained. Shared mount/live crew/
      draught-beast profiles qualify, but a joined character must have its
      own protection and suffers its own wounds, not the regiment's.
      Movement now uses the slowest participant AFTER each model's own
      penalty: M4 troops with an unprotected M4 character move 3", while M3
      troops with an unprotected M5 character still move 3". This applies to
      ordinary ranked/Skirmisher previews, march allowances, Vanguard and
      charge resolution. A Vanguard character left behind is not counted.
      Protected charges keep the highest of the first two dice, with any
      Swiftstride bonus die added separately. Mixed units containing an
      unprotected model still discard the highest; pursuit keeps its normal
      summed dice. Move Through Cover does NOT remove Disruption, grant
      passage through impassable terrain, or cancel non-terrain modifiers.
      Logs at move commitment and charge resolution include adjusted model
      Movements and the resulting allowance; Dangerous Terrain logs aggregate
      rerolled dice, avoided mishaps and remaining wounds. No-1s and mixed
      charge refusals explain why the benefit did not apply, without logging
      per-preview or per-test-roll spam.
      Follow-up: the shared Skirmisher move commit had lost this Movement log.
      It now passes the actual swept-base terrain features to the allowance
      logger once per committed move; previews remain silent. Explicit empty
      features do not fall back to centre sampling, and flying reports why
      Move Through Cover does not apply. Movement and march limits are unchanged.
      Corrected on the way: ordinary Skirmishers skipped terrain penalties,
      charges used raw M instead of the adjusted allowance, joined characters
      were omitted from Dangerous Terrain tests, and saves omitted terrain.
      Saves now store static terrain using the map record format; old saves
      without terrain retain the current battlefield. Spell-created terrain
      is restored through spells, not duplicated in the static terrain list.
      Initialized the terrain shader's missing movement-overlay colour so
      newly loaded terrain renders before the first movement preview.
      Tests: `tests/test_move_through_cover.py`,
      `tests/test_move_through_cover_scene.py`, plus reroll regressions in
      `tests/test_terrain.py`. Run `python -m tests.test_move_through_cover_scene`
      to create `saves/move_through_cover.json` and its offscreen screenshot.
      The save starts Player 1's Movement phase, AI off: Cover Woods (M4),
      Ordinary Woods (M4 -> 3), Cover Slow Escort (M4 with unprotected M4
      captain -> 3), Ordinary Marsh (M4 -> 3), and Cover Fast Escort (M3 with
      unprotected M5 captain -> 3). Two enemy units face the woods units for
      charge comparisons. Marsh is dangerous, woods difficult. These are
      test-only rule/stat grants, not a legal army list.
      LEFTOVER: ranked-unit terrain detection samples the regiment's straight
      centre-to-centre path, not every base's swept area during a wheel;
      Dangerous Terrain tests consequently use that shared feature list for
      rank-and-file and joined characters. Narrow features and edge-only
      contact can be missed. Optional beneficial re-rolls are automatically
      taken, consistent with the existing automatic Dangerous Terrain roller.
- [x] Shieldwall (Rulebook p. 177) — once per game, a unit charged this turn,
      in Close Order and using shields, may Give Ground instead of Falling
      Back in Good Order. It is not an armour or combat-result bonus.
      Sources checked: https://tow.whfb.app/special-rules/shieldwall,
      https://tow.whfb.app/forming-units/close-order-formation (p. 100),
      https://tow.whfb.app/movement-in-detail/give-ground and
      https://tow.whfb.app/movement-in-detail/fall-back-in-good-order (p. 134),
      plus Official FAQ v1.5.3 at https://tow.whfb.app/faq. A single survivor
      retains Close Order; no minimum rank bonus, frontage or Unit Strength
      is required. Disruption does not prevent Shieldwall.
      Registry keyword and model predicate feed `shieldwall_unavailable_reason`
      and `CombatResolver.shieldwallOutcome`. The defending player may spend
      or retain the rule after the final Break roll, including a BSB re-roll,
      or after choosing Stubborn's automatic Fall Back. A Break (including
      an overwhelmed result) cannot be converted. Stubborn followed by
      Shieldwall spends both rules. Giving Ground does not trigger flee Panic,
      roll flee dice or mark the unit as having fled; the existing 2-inch
      Give Ground / Follow Up movement, collision and Surrounded rules apply.
      Shields must be equipped and usable with the chosen melee weapon.
      An eligible unit auto-equipped with a two-handed weapon gets a choice
      of keeping it or using hand weapon and shield BEFORE attacks. Choosing
      shields does not itself spend Shieldwall. AI chooses shields and spends
      the rule on an eligible Fall Back result. Unit-level checks do not grant
      an ordinary regiment Shieldwall merely because a character has it.
      `wasChargedThisTurn` records successful incoming charges independently
      of the unit's own charge bonuses; failed charges do not set it. Pursuit
      and overrun contacts carry `countsAsChargeTargetNextTurn` when their
      combat is deferred, matching the charger's existing next-turn flag.
      Both flags and `usedShieldwall` survive save/load. Turn end clears or
      promotes charge history but never refunds Shieldwall. Old saves default
      to unused Shieldwall and no recorded incoming charge.
      Effect and skipped logs explain the outcome, spent use, missing charge,
      formation, missing shields, two-handed weapon or player decision.
      Corrections made while implementing: Stubborn's early return bypassed
      ordinary Break-result processing; the defender was always auto-equipped
      with its strongest melee weapon; and the catalogue's Additional Hand
      Weapon uses "Require Two Hands" rather than "Requires Two Hands".
      The shield-use predicate now accepts both spellings (also fixing its
      shield armour-save check).
      Tests: `tests/test_shieldwall.py`, `tests/test_shieldwall_scene.py`, and
      persistence flag coverage. Real offscreen combats verify shield choice,
      Stubborn + Shieldwall, no flee Panic and paired 2-inch movement. Charge,
      pursuit and overrun handlers, legacy saves and spent-state reloads are
      exercised. Run `python -m tests.test_shieldwall_scene` to regenerate
      `saves/shieldwall.json` and `screenshots/shieldwall.png`.
      Save: Player 2 Combat phase, AI off. Three pairs left to right: ready
      Shieldwall, already-spent Shieldwall, and a great-weapon/shield choice.
      Select a blue Charger, resolve combat, then choose Stand Firm for the
      red defender; choose Shieldwall when offered. Choose Follow up to watch
      both sides move 2 inches. All profiles have test-only A0: red defenders
      have one rank of five (no second-rank Press of Battle attacks), blue
      units two ranks of five, ensuring the defenders lose without random
      casualties. Stubborn is also granted for repeatable results. Reload to
      compare keeping Shieldwall or keeping the great weapon. Not a legal list.
      LEFTOVER: formation switching is still unimplemented: Shieldwall uses
      the Close Order keyword and excludes Skirmishers (even when formed up
      for combat); Open Order-only units cannot use it. It does not introduce
      Close/Open Order transitions. Old saves cannot recover incoming-charge
      history they never recorded. AI always uses an eligible Shieldwall,
      rather than evaluating whether retreating further would be preferable.
- [x] Veteran (Rulebook p. 180, Official FAQ v1.5.3) — a strict majority of
      models with Veteran permits one re-roll of a failed Leadership test.
      It grants no Leadership bonus and never re-rolls a Break test.
      Sources checked: https://tow.whfb.app/special-rules/veteran,
      https://tow.whfb.app/model-profiles/leadership-tests (p. 97),
      https://tow.whfb.app/faq/universal-special-rules, and Split Profile
      (Cavalry/Chariots), pp. 192/194. The keyword is active in the registry;
      `model.is_veteran` includes mount and surviving crew/beast profiles.
      A split-profile model counts once, not once per component or wound.
      `veteran_counts` counts the regiment's surviving models plus its joined
      character, not Unit Strength. Ties fail. Casualties change eligibility
      immediately. A retired character counts as a model but contributes no
      rule benefit (p. 210). There is no once-per-game limit or spent flag.
      Shared `reroll_leadership` is wired into Rally, Panic and Restraint.
      A human may Re-roll or Keep; AI re-rolls automatically. Existing
      Venerable/BSB re-rolls remain automatic and cannot grant a third roll.
      The replacement stands even if it fails (p. 93). Panic's queue remains
      paused until the owner's choice and subsequent movement finish.
      Actual Break resolution explicitly rejects Veteran, independently of
      BSB Break re-rolls. Restraint is still eligible even though it uses the
      physical dice method named `rollBreakDice`.
      FAQ: a personal character test must pass the character itself, its own
      Leadership and `personal=True`; an ordinary character cannot borrow
      its host's Veteran. See Rallying Cry below for that rule's target test.
      Correction made along the way: Leadership tests now automatically pass
      on natural double 1 and fail on natural double 6 (p. 97), including
      Rally and Restraint. Older tests assuming Ld 1/12 always fail/pass were
      corrected. Rule logs carry model counts, initial/final rolls and Ld,
      or explain the failed majority, passed test, Break exclusion, personal
      test restriction or player's decision to keep the failure.
      Tests: `tests/test_veteran.py` and `tests/test_veteran_scene.py` cover
      majority, split profiles, FAQ, single re-roll, live Rally/Restraint/Break
      methods, queued Panic, actual casualties and real save/load/render.
      Run `python -m tests.test_veteran_scene` to regenerate
      `saves/veteran.json` and `screenshots/veteran.png`.
      Save: Player 1 Strategy phase, AI off. Five fleeing cases left to right:
      five Veterans; five ordinary models; five Veterans with an ordinary
      captain (5/6); one Veteran with an ordinary captain (1/2, no majority);
      and a Veteran captain alone. Select a red unit to attempt Rally.
      All have test-only Ld 6, with no General/BSB support. A failed test
      offers Veteran only in cases 1, 3 and 5. Reload to repeat or compare
      keeping the failure. Dice are not forced; an initial pass needs no
      re-roll. Complete the existing free reform after a successful Rally.
      These are test profiles, not a legal army list.
      LEFTOVER: Fear, Terror and other not-yet-implemented
      Leadership-test causes must call the shared re-roll flow when added.
      No new test causes or general Leadership-source rules are implemented
      here. Majority counts use the existing homogeneous regiment profile
      plus one joined character; arbitrary mixed-model regiments are not
      represented. AI does not evaluate reasons to keep a failed test.
- [x] Rallying Cry (Rulebook pp. 117, 175, 202-203) — once per Command
      sub-phase, a non-fleeing, unengaged character may nominate one fleeing
      friendly unit within Command range for an immediate Rally test.
      Sources checked: https://tow.whfb.app/special-rules/rallying-cry,
      https://tow.whfb.app/the-strategy-phase/command,
      https://tow.whfb.app/the-strategy-phase/rally-fleeing-units,
      https://tow.whfb.app/the-strategy-phase/rallied-units,
      https://tow.whfb.app/the-strategy-phase/insurmountable-losses,
      https://tow.whfb.app/characters/command-range, and Official FAQ v1.5.3
      at https://tow.whfb.app/faq/universal-special-rules.
      `rallying_cry.py` owns eligibility, target nomination and Command flow.
      Ordinary characters use their own Ld in inches; General/BSB use 12",
      or 18" with Large Target. Range is inclusive, base-edge to base-edge,
      from the character's own world position even when joined to a regiment.
      A fleeing or engaged host, retired character, wrong turn, enemy target,
      non-fleeing target or spent use is rejected with a reason. The keyword
      supports shared mount/live crew/beast profiles without extra uses.
      Strategy opens a Command window when the active side has a character
      with Rallying Cry. Selecting that character, or its host regiment,
      opens the eligible-target menu. Cancelling retains the use; confirming
      a valid nomination spends it before the target rolls. End Phase closes
      Command before the existing Conjuration/Rally actions become available.
      End Phase cannot interrupt nomination, dice resolution or free reform.
      AI chooses the largest eligible unit and declines the optional reform.
      The target uses the shared Rally path, including its own Veteran and
      BSB eligibility, with at most one re-roll of each test. Failed Command
      Rally does not consume normal Rally; success permits a free reform,
      bars charges and counts as moved for shooting, but does not consume
      the ordinary move. A later normal Rally is a fresh test, not a third
      roll of the earlier one. Different callers may each nominate a unit
      that is still fleeing; each caller is limited to one use.
      `usedRallyingCry` and `strategyCommandDone` survive save/load. A fresh
      Strategy entry resets uses exactly once; returning from a spell does
      not. Legacy saves default to Command already complete, avoiding an
      extra opportunity in a turn whose Command history was never saved.
      FAQ interpretation: the Veteran answer mentions Rallying Cry as an
      example of a character's personal test, but p. 175 explicitly gives
      the nominated UNIT a Rally test and specifies no character test.
      This implementation follows that operative text, adds no extra test,
      and does not lend the caller's or its host's Veteran to the target.
      Existing personal-character test restrictions remain unchanged.
      Corrections made while implementing: normal failed Rally attempts now
      reset even for units still fleeing on the next Strategy entry; shared
      Rally applies Insurmountable Losses (-1 Ld below half starting models,
      only natural double 1 below a quarter); Large Target command detection
      recognises the catalogue keyword and mounts; joined AI characters use
      their actual side. Command initialization was moved outside the unit
      reset loop. Setting `hasMovedThisTurn` on Rally incorrectly barred
      normal movement; the existing `attemptedRallyThisTurn` shooting query
      supplies the required penalty without spending movement instead.
      Logs report caller, target, measured range, spent use, dice/Ld, outcome,
      loss penalties and why actions were unavailable or declined.
      Tests: `tests/test_rallying_cry.py` and `tests/test_rallying_cry_scene.py`.
      Run `python -m tests.test_rallying_cry_scene` to regenerate
      `saves/rallying_cry.json`, `screenshots/rallying_cry.png` and the target
      menu screenshot `screenshots/rallying_cry_choice.png`.
      Save: Player 1 Strategy/Command, AI off. Select Ready Caller on the
      left, then Left Veterans or Left Ordinary. The centre Veteran Guard
      contains an ordinary Joined Caller who can nominate Centre Runners.
      Spent Caller on the right cannot act again; enemy and out-of-range
      runners are excluded. Callers have test-only Ld 8; runners Ld 6, with
      no General/BSB support. Dice are not forced. After an early failure,
      press End Phase once and select that fleeing unit for normal Rally.
      Complete the existing free reform on success. Reload to repeat.
      These are test profiles, not a legal army list.
      LEFTOVER: this adds a Command window to Strategy, not a complete
      four-sub-phase interface. Conjuration and normal Rally still share
      the existing selection flow, and End Phase does not force every normal
      Rally attempt. Other Command abilities are not added. AI chooses by
      model count, not tactical value, and does not optimise the free reform.
      The FAQ's example remains inconsistent with the published operative
      Rallying Cry text; no unsupported character Leadership test is added.
- [ ] Close Order / Open Order / Dispersed Formation — formation modes
- [ ] Detachment — list-building support (may not need a runtime effect)

## Dwarf-specific (`Dwarfen Mountain Holds.cat`) rules — TODO
- [ ] Ancestral Grudge — Hatred (enemy characters)
- [ ] Gromril Armour — re-roll natural 1s on armour saves
- [ ] Gromril Weapons — hand weapon with AP -1 in melee
- [ ] Resolute — 
- [ ] Stoic Defenders — +1 Initiative & Attacks when charged
- [x] Venerable — DONE: friendly units within 6" of a Venerable unit (edge to
      edge, the same bubble as nearby-friend Panic) re-roll failed Panic tests.
      The Venerable unit benefits itself; a fleeing one inspires nobody. See
      `PsychologySystem.venerable_source` / `_resolve_panic` in
      `psychology.py`.
- [ ] Runes of Warding — 5+ ward vs Flaming Attacks
- [ ] Rune Lore — may attempt a Wizardly Dispel
- [ ] Forgefire — joined unit gains Armour Bane (2) + Flaming Attacks
- [ ] Dwarf Crafted — no -1 To Hit on a Stand & Shoot reaction. Unblocked now
      that the reaction and its -1 exist; `ignore_to_hit_penalties` already
      carries a `stand_and_shoot` key for exactly this.
- [x] Fire & Flee (p. 169) — the fourth charge reaction: the unit Stands &
      Shoots and *then* flees, "however, due to the time spent shooting at the
      charging foe, when making its Flee roll the unit rolls two D6 and
      discards the lowest result".
      That is a penalty, not a bonus, and it is easy to read the other way
      round: an ordinary Flee roll *sums* 2D6, so keeping one die is the
      shorter run. `post_combat.fire_and_flee_roll` is the same arithmetic as
      a Fall Back, so it delegates to `charge_roll` exactly as `fall_back_roll`
      does rather than being a third copy of "2D6 discard the lowest".
      Measured: [2,5] flees 7" normally and 5" here, [6,6] flees 12" and 6".
      The rule is unit-level and comes from the army list, like Skirmishers, so
      it goes through a `SPECIAL_RULE_BUILDERS` entry to `fire_and_flee` on the
      model and is read by `model.has_fire_and_flee()`. Both spellings of the
      joiner are registered. It is not in the catalogue's model profiles at
      all — the Gyrocopter carries it in `player1_army.json`, which is the only
      place it appears.
      NOTE: the rule restates the distance gate in its own words — "if the
      distance between this unit and the charging unit is less than the
      Movement characteristic of the charging unit, this unit must either Hold
      or Flee" — without repeating Quick Shot's exemption from it. Taken as
      the same gate rather than a second one: the reaction *is* a Stand &
      Shoot followed by a flee, so whatever may be Stood & Shot at may be
      Fired & Fled from, and a Quick Shot weapon may do both from inside the
      charger's Movement. It is therefore tested in one place —
      `can_stand_and_shoot` — and `fireAndFleeOption` only asks whether the
      unit has the rule.
      Two bugs fell out of the flee path while wiring this up, both predating
      it and both in `fleeInterval`, which is the charge-reaction flee:
      `fldist = sum(fldice) + fleeBonus` added `fleeBonus` — a **bool** from
      `swiftstrideChoice` — to the distance, so any unit that took its
      Swiftstride die fled a literal 1" too far, on top of the bonus die
      already among `fldice`.
      And that line was a second copy of the flee arithmetic, so this path
      never consulted `fledThisPhase`: The Limits of Endurance (p. 133) was
      applied to every flee in the game *except* the one a charge causes, and
      a unit that had already fled could flee again on a fresh 2D6. Both flee
      rolls now go through `flee_roll` / `fire_and_flee_roll`, which is where
      that clause lives, and the reaction spends the allowance.
      LEFTOVER: the AI never takes it. Unlike Stand & Shoot this is a genuine
      trade — the volley in exchange for giving up the combat and taking a
      Panic-adjacent flee move — and there is no policy to weigh it, so it
      shoots and stands, and logs that it declined.
      LEFTOVER: "if the *majority* of the models in a unit" is read as the
      unit's shared model having the rule, which is what the engine's one
      model per unit can express.
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
      the attacks of enemy models with the same Initiative value. CLOSED — see
      Who Strikes First below.
- [x] Who Strikes First / Charging Units / Simultaneous Combat — DONE (p. 146,
      Charging Units as amended by the errata). `battleFunctions`
      `charge_initiative_bonus` and `strike_initiative` do the arithmetic:
      +1 Initiative per *full* inch moved before contact, capped at +3 into a
      front arc and +4 into a flank or rear, with the total capped at 10 by the
      errata. `CombatResolver.strikeOrder` sorts the engagement list by that
      value, highest first, and `_engagedFacing` reads which arc a striker
      charged into — `isInCombatWith` and `isInCombatFlank` are appended in
      step, so the arc is the entry at the striker's own index in its victim's
      list.
      Before this the loop simply walked `self.game.attackers` in list order:
      the charger struck first because it happened to be built first, and a
      defender with equal or higher Initiative struck back thinned. Now
      `_verySimpleBattleInner` re-snapshots every unit's model count each time
      the Initiative value changes, and a striker fights with the models it had
      when its step began — so a blow landed alongside its own cannot cost it
      attacks, while one landed at a higher Initiative still does.
      CORRECTED on the way: `strike_initiative` defaults a missing `I` to 1
      rather than 0, so an unparsed profile still strikes rather than being
      sorted below everything and then treated as Initiative 0.
      CORRECTED on the way: `self.game.attackers` lists a unit reachable from
      both sides of the engagement twice, and the old loop weeded the copy out
      with `hasAttackedThisTurn` as each striker came up. Drawing the order up
      in advance reads that flag before anyone has fought, so both copies got
      through and the unit attacked twice — seen in play, Skeleton Warriors
      making two sets of 5 attacks either side of the reply. `strikeOrder` now
      keeps the first occurrence of each striker.
      LEFTOVER: a unit strikes as one at its own model's Initiative. A joined
      character, a mount and a chariot's parts are all resolved inside that
      unit's step, though each has its own Initiative and should be placed in
      the order separately — a I6 hero in a I3 regiment ought to strike before
      the rank and file, not with them.
      LEFTOVER: Strike First and Strike Last (above) are now DONE — they turned
      out to be a substitution of the characteristic before modifiers rather
      than a sort outside the sequence, so they needed no ordering hook at all.
      LEFTOVER: Stomp Attacks are absent entirely, and the errata to p. 177 has
      them made last of all, after attacks at Initiative 1.
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
- [x] Split Profile (Cavalry) — rider and mount each use their own WS/BS/S/I/A
      and weapons; enemies roll To Hit against the *rider's* WS; Impact Hits
      and Stomp use the mount's Strength; the armour save uses the rider's
      value; the model dies when the rider does. `combat_profiles.py` now
      schedules parts separately and restricts mounts to the fighting rank.
      LEFTOVER: exact individual-base contact allocation and uncoded Stomp.

### Phase 4 — war machines
Their three rules are a self-contained block and would suit being done with the
existing `cannon_fire.py` / `bombardment.py` work.
- [ ] Split Profile (War Machine) — crew's Toughness and Wounds in combat, the
      machine's when not; -1 Attack per Wound the crew has lost; armour save
      from the crew; either element at zero Wounds removes the model.
      PART DONE — **Movement**. A split profile is two rows with gaps in each
      (p. 97), and a war machine's own row has no Movement at all: it is shifted
      by the crew that work it. `get_movement` fell through to whatever default
      the caller happened to pass, so an artillery piece moved 4", 0" or
      anything else depending on which code asked.
      The crew were not being read either. A chariot declares its crew as a
      nested entry marked `subType='crew'`, which is what `_model_parts` looked
      for; a war machine's crew is a plain entry link named for what the crew
      are — `Gun Crew`, `Dwarf Crew`, `Peasant Crew`. All 26 of them end in
      "Crew", so that is what the parser matches, and 33 models now carry crew
      where 7 did before — every one of them a war machine or a chariot.
      `get_movement` only falls through to the crew when the model's own
      Movement is missing, so a Stegadon (M6, with Skink Crew) is untouched and
      a chariot still moves at the speed of the beasts that draw it.
      Verified: Great Cannon, Helblaster and Helstorm all resolve to M4 from
      their Gun Crew; the War Wagon stays at 7 from its horses.
      NOTE: `model('Mortar')` resolves to the Renegade Crowns profile, which
      genuinely has no crew in the data, rather than the Empire's. Bare model
      names collide across factions — that predates this and is unrelated to
      the split profile.
- [ ] "We're Not Paid to Fight" — a war machine that Breaks and flees from
      combat is destroyed outright. Fall Back and Give Ground are normal.
- [ ] Weapon of War — cannot march, declare a charge or pursue; -1 to any Flee
      roll (minimum 1); may pivot freely about its centre immediately before
      shooting without that counting as moving; may follow up as normal.

### Not part of this section
Ridden Monster is listed for monstrous creatures and behemoths but is defined
under Characters, so it belongs with that work rather than here.

## Challenges (Rulebook p. 210-211)

Planned in `CHALLENGE_PLAN.md`. The rules with no Panda3D in them live in
`challenges.py`; `combat_resolution` runs the exchange and the duel.

- [x] Challenges / Issuing / Accepting — DONE (p. 210). `challengeExchange`
      runs at Step 1.1, right after the combat is chosen and before Impact
      Hits. The active player is offered first and only then the inactive one,
      one challenge per combat. `duellist(unit)` finds the model that may fight.
- [x] Refusing a Challenge — DONE (p. 210). The refusing model is retired:
      `retiredFromCombat` takes it out of its unit's attacks, and
      `units.placeCharacter` moves it behind the last rank so the retreat is
      visible. `psychology.active_character` makes it confer nothing — a
      retired General or Battle Standard stops counting, which is the clause
      that actually bites. `units.exitInCombat` lets it return once the unit is
      no longer engaged.
- [x] Nowhere to Run — DONE (p. 211). `refusal_barred` reports *why* a model
      cannot refuse so the log can say it: not part of a unit, the last model,
      or surrounded.
- [x] Fighting a Challenge — DONE (p. 211). `resolveChallenge` fights the duel
      in Initiative order using the same `strike_initiative` as everything else.
      A duellist adds nothing to its unit's fight, which is why the `joinedRule`
      block is skipped for it.
- [x] Overkill — DONE (p. 211). `overkill_bonus` is the excess unsaved wounds
      over the loser's remaining Wounds, capped at +5, and only when the rival
      actually falls. It has its own row in the combat result, and is subtracted
      out of 'Wounds caused' so it is not counted twice.
- [x] To The Death! — DONE (p. 211). Challenges live on `game.challenges` and
      are saved, so one carries across turns; `challengeExchange` finds a live
      one and refuses to start another in that combat.
- [x] Challenges & Mounts — DONE (p. 211). `duelCombatants` adds a mount and a
      chariot's crew to the duel with their own Initiative, and a participant
      slain before its attacks are made loses them.
      CORRECTED on the way: a joined character could not be wounded at all.
      `applyWounds` is only ever called on the host unitGraphics, so a character
      dealt attacks but nothing could direct wounds at it — it died only when
      its host was destroyed. `woundDuellist` and `characters.slay_character`
      close that. Slaying a joined character is deliberately not
      `removeModelsFromUnit`: joining takes the character out of the physics
      world and parents its nodes under the host, so the ordinary path would
      remove a rigid body twice and leave the host pointing at a dead model.
      Champions are now promoted profiles with personal wounds, refusal state,
      mount-only duel attacks and stable identities across save/load. Overkill
      counts actual wounds plus excess once; equal-Initiative duels share a step.
      LEFTOVER: explicit choice between a joined character and champion, and
      the issuing player's choice of which eligible model must retire.
      LEFTOVER: "within, or adjacent to, the fighting rank" is never tested. A
      joined character always stands in the front rank at `host.characterSlot`,
      so it is always eligible.
      LEFTOVER: "engaged in all four arcs" is approximated. The engine records
      only `front`, `flank` and `rear` per engagement and cannot tell a left
      flank from a right one, so `surrounded` asks for a front, a rear and two
      flanks.
      LEFTOVER: the AI never issues and always accepts. A real heuristic would
      weigh the two profiles against what refusing would cost.
      LEFTOVER: the duel is resolved as its own pass rather than woven into the
      unit-level strike order. p. 211 seals a challenge off — the duellists
      attack only each other and nothing else may attack them — so the ordering
      only has to be internally consistent, but a rule that cares about the
      order of the whole combat would not see it.
      LEFTOVER: the pair are not moved into base contact. p. 211 says that is
      optional ("perfectly acceptable to leave them in place").

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
      Corrected since: that queue was the combat resolver's own, and the Panic
      pass never touched it — `psychology.py` calls `game.startFreeReform`
      directly. One break can raise four reforms at once, the pursuer's plus a
      Fall Back in Good Order for every friend that then failed its Panic test,
      and each new one force-released the previous waiter and started on top.
      The queue now lives in `MyApp.startFreeReform`, the single point both
      paths reach, as a FIFO with the callbacks fired in turn. A unit run down
      while it waited is skipped but its waiter is still released, and a reform
      raised from another's callback queues behind rather than on top, because
      the active flag stays set across the callback.
      **Corrected (2026-09-11):** the queue alone did not serialize pursuits:
      `pursuitMove` still waited a fixed five seconds, while the charge/reform
      task continued independently. Dragon Princes could be reforming when
      Silver Helms opened Swiftstride and planned against their unfinished pose.
      The game/movement wrappers now expose the actual charge task through an
      opt-in completion return; pursuit awaits the charge animation and reform
      callback before starting the next unit. Normal scheduled AI callbacks retain
      their None return. Quarry state clears in finally, after completion.
      The pursuit pass retains the quarry's last position so its removal by the
      first pursuer does not silently skip later declared pursuits. Later paths
      are computed only after earlier reforms finish, against current obstacles.
      Four offscreen regression checks include 12-second dice/reform delays, a
      removed quarry, both real cavalry movement tasks and the live reform queue,
      plus the scheduled-callback return contract. Another 144 adjacent checks
      pass (post-combat, Shieldwall, Counter Charge, declarations, formed-to-loose
      routes and task scheduling), in bounded services; no full-suite run.
      Wording rechecked at [Pursuit](https://tow.whfb.app/the-combat-phase/pursuit)
      (p. 156) and [Catching the Curs!](https://tow.whfb.app/the-combat-phase/catching-the-curs)
      (p. 157).
      **LEFTOVER:** this repairs task sequencing, not the complete legacy pursuit
      range/route or multi-unit geometry rules. Suspended dice/reform tasks are
      not resumed from saves.
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
      A fifth wrong turn, and not a geometric one: `pivot` went in as the
      *third* positional parameter of `alignToEnemy`, ahead of `duration`. The
      new overrun call passed it by keyword and was fine, but the charge call
      has always passed its duration positionally, so every charge align since
      has handed `0.5` to `setPos` and raised. Adding a parameter in the middle
      of a signature is silent in Python — no error at the call site, nothing
      from pyflakes — and no test touches `alignToEnemy`, which is async and
      needs `render`. `pivot` is keyword-only now so a positional argument
      cannot land on it again.
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
- [x] Fleeing Through Terrain (p. 133) — three clauses, and only one of them
      needed much.
      "Without suffering any negative modifiers to its Movement characteristic"
      was already true, by construction rather than by intent: a Flee roll is
      2D6 summed and a Fall Back is 2D6 discarding the lowest, neither adds
      Movement, and `fallBack2` moves the distance it is handed without
      consulting the terrain. Checked rather than assumed — `charge_roll` does
      take a `difficult` flag that discards the *highest* die, and
      `fall_back_roll` delegates to it, so the wrong default there would have
      quietly shortened every Fall Back over rough ground.
      "It must make any Dangerous Terrain tests required" was missing on both
      flee paths, though the machinery was already there and already used by
      normal moves and charges. `fleeTerrainTests` is a thin wrapper on
      `dangerousTerrainTests` that exists so the two flee paths call one thing.
      "Should a fleeing unit come into contact with impassable terrain, it must
      pivot around its centre in order to move around it by the shortest
      possible route" — `fleeAroundImpassable` reads "shortest" as the smallest
      pivot that gets past, offering each turn to both sides before trying a
      larger one, which is `post_combat.detour_angles`. The angle sequence and
      the rotation are pure and tested; only the "is this way blocked?" question
      needs the board.
      Corrected from a game log: the first pass asked `get_terrain_between`,
      which samples the **centre line** at 1" intervals. A 10 degree pivot over
      a 3" flee moves that line about half an inch, enough to step it off the
      corner of a house while the unit's flank walks straight into the wall —
      and the log cheerfully announced it had gone around. A unit is wider than
      the line through the middle of it. `impassableAhead` now sweeps the
      unit's own box, turned to the heading it will flee on, against
      `TERRAIN_IMPASSABLE` alone; units are left out of the mask deliberately,
      because a fleeing unit goes *through* those and that is what the Peril
      tests are for.
      `tests/harness_flee_terrain.py` is what settled it — a real unit box and
      a real impassable body offscreen, printing blocked/clear for every
      candidate turn and re-testing the direction finally chosen. It shows the
      shape of the answer as well as its correctness: a 5" unit with a 6" house
      two inches off its nose has no small way past, because pivoting about its
      centre sweeps its own corners into the wall, so it turns the full 90
      degrees and runs along the frontage. Drastic, but it is what "pivot about
      its centre by the shortest possible route" means for a wide unit.
      A Give Ground is excluded: it is not a flee, and stops at terrain rather
      than going round it (p. 155).
      LEFTOVER: nothing stops a fleeing unit at impassable terrain, so when
      every route within 90 degrees either way is blocked it says so and then
      runs straight through the building. Going round is now right; being
      stopped by it was never modelled and still is not.
      LEFTOVER: the detour is measured over the flee distance, but
      `psychology._flee_until_clear` may carry the unit further than that
      looking for a spot clear of other units, and the extra ground is not
      re-tested.
- [ ] Pursuit off the Battlefield (p. 157) — removed but *not* destroyed,
      returning in the next Compulsory Moves sub-phase as reinforcements.
      BLOCKED: no reinforcement mechanism exists.
- [x] Surrounded, the rest of it (p. 155) — "the unit's movement stops
      immediately and the units instead remain locked in place until the next
      player's turn when they will fight another round of combat, exactly as if
      the combat had been a draw."
      Two halves were missing, and the interesting one was the *detection*. The
      rule fires when a loser "is unable to break contact with one or more of
      the enemy units engaging it"; the code asked whether the Give Ground
      moved it at all. For two units nose to nose those are the same question —
      any move over a hair breaks contact — so the gap only shows with three.
      A unit engaged in front and both flanks has its two flank directions
      cancel, so it gives ground straight backwards, breaks cleanly from the
      unit in front and *scrapes along the faces of the two beside it*, still
      touching both. It had been moving off as though it had got away.
      `surrounded()` projects the loser's box by the step it is about to take
      and measures it against every winner with `obb_distance`, the same
      oriented-box maths the 1" rule and the Leadership bubbles use. Measured
      against where the winners stand *now*, deliberately: a Follow Up closing
      the gap again is a later choice by the winner, not a failure by the loser
      to break away, and testing against the followers' destinations would have
      called every followed-up Give Ground in the game Surrounded.
      The second half is the reform. A winner that restrained was still being
      offered its free reform in pass 4, but the loser never drew off, so the
      two are nose to nose with no room to turn in — and a drawn combat grants
      no reform at all. It is skipped, and says which unit's Surrounded result
      took it away.
      Nothing else was needed for "as if the combat had been a draw": neither
      side moves, so both stay engaged and fight again next turn, which is what
      the draw branch does by returning early.
      The log carries how far it could actually have gone and who it is still
      stuck against, because "locked in place" and "the rule never fired" look
      identical on the board.
- [x] Give Ground moves 2", not 1.9" — `crashFraction` multiplied the sweep's
      hit fraction by 0.95 whether or not the sweep hit anything, so every
      Give Ground in the game was short. The 5% is a margin against coming to
      rest touching what was struck and re-contacting on the next test, which
      is only meaningful when something *was* struck: `sweepTest` returns 1.0
      for a clear path, and that is now taken in full. Predates this work.
      It also removes a coupling this section had just introduced: `surrounded`
      measures the step *after* the margin, so a Give Ground that should barely
      break contact could have been judged Surrounded on the strength of a
      fudge factor. With a clear 2" it cannot.
      LEFTOVER: untested. The margin needs a real Bullet sweep to reach, so it
      is two lines of arithmetic behind the same async/`render` boundary that
      `alignToEnemy` sits behind.

### Not in this section
Whether a unit that Falls Back in Good Order panics its friends was checked and
is correct as coded: the rulebook is explicit that it does, because "amidst the
clamour of battle, friendly units are seldom able to tell the difference"
(p. 161).

## Loose ends
- [x] Marching (p. 123) — a unit may double its Movement to march, but a unit
      that marched cannot shoot that turn, nor cast a Magic Missile or a
      Magical Vortex.
      Marching was not so much missing as **always on and free**: the movement
      arc offered `M * 2` unconditionally, so every ordinary move in the game
      was already a march-length move with nothing charged for it. This change
      mostly takes freedom away — the same reach is now split into an ordinary
      band (M) and a march band (up to 2M), and crossing into the second costs
      the unit its Shooting phase.
      `is_march(distance, movement, spent)` in `movement_system.py` is the
      whole decision, kept pure so it can be tested: the first M is free, and
      exactly M is not a march. It reads the arc distance the cursor has
      already reached — wheel included, and recomputed every frame — so the
      answer exists before the click rather than after it.
      The overlay tints warm (`OVERLAY_MARCH`) the moment the cursor crosses
      into the march band, which is what tells the player what the click will
      cost. Both `c1.frag` and `terrain.frag` hard-coded the pale blue, so both
      take a `overlayColor` / `moveColor` uniform now; the terrain one matters
      because the indicator wraps over hills. Verified by rendering offscreen
      under each tint and measuring the frame: red +0.086, blue -0.160, so the
      uniform is genuinely driving the colour rather than being ignored.
      A pursuit is exempt on both paths — it is a compulsory post-combat move
      of up to 21", not a march, and would otherwise trip the band every time.
      The barred spell categories come from the catalogue's own `type` field
      ('Magic Missile', 28 spells; 'Magical Vortex', 18), not from a list
      invented here, and a test asserts those strings still match the data.
      Move & Shoot (p. 174) is the exception and is done — see the weapon
      rules above.
      LEFTOVER: Enemy Sighted (p. 123) is not implemented — beginning a march
      within 8" of a non-fleeing enemy needs a Leadership test, and a *failed*
      test still counts as having marched even if the unit then does not move
      at all. Marching is currently unrestricted near the enemy. Deliberately
      deferred to its own commit.
      LEFTOVER: "whilst marching a unit can wheel ... but cannot perform any
      other manoeuvres" is only half enforced. Move Sideways is withdrawn from
      a unit that has already marched, but the sideways band is part of the
      same arc widget and is sized before the cursor's band is known, so a
      unit can still sidestep and then march in the same move.
      LEFTOVER: the march threshold is only visible once crossed. Drawing both
      bands at once would show where the line falls beforehand, but it needs a
      second polygon array in the shader.
      LEFTOVER: the spell gate reads the flag on `unitToMove`, which is the
      host unit. A joined Wizard is covered by its host, but the marching flag
      of a lone character Wizard is its own and has not been exercised.
      LEFTOVER: only `is_march` and the spell categories are tested. Reading
      the band from the cursor, tinting the overlay and setting the flag on
      the click all sit inside Panda3D-bound methods.
- [x] Redress the Ranks (p. 125) — a unit spends half its Movement to move up
      to five models to or from its front rank; the remaining ranks rearrange,
      every rank full but the rear. `redress_formation()` in
      `movement_system.py` is the pure arithmetic (`ranks = ceil(n/files)`
      gives the "only the rear rank may have fewer" clause for nothing);
      `MovementSystem.redressRanks()` performs it. `v` widens the front rank,
      `shift-v` narrows it, up to five models in total for one cost.
      The front rank holds its ground: the new frontage is built about the
      front edge's centre, not the unit's centre, or widening would drag the
      unit backwards and narrowing would walk it forwards. That also satisfies
      the FAQ's "as equally as possible on either side" without doing anything
      further, since the models are interchangeable.
      Manoeuvres are now metered. A unit may perform only ONE per move
      (p. 124), held in `unit.manoeuvreThisTurn`, and the half-Movement cost
      goes into `unit.moveSpentThisTurn`, which both movement arcs subtract.
      This also fixed a rule the arc broke silently: it offered a wheel and a
      Move Sideways in the same move, which is two manoeuvres. `sidemove` is
      now zero once a manoeuvre has been performed.
      A redress that would not fit is reverted — the reshape happens, the
      contact test runs, and the old formation and position go back if a
      widened frontage fouls a neighbour.
      LEFTOVER: redressing is meant to count as moving for shooting penalties,
      but nothing applies a to-hit penalty for having moved, so there is
      nothing to hook it to. `hasMovedThisTurn` is deliberately NOT set —
      `moveUnit()` refuses to move a unit carrying it, which would forbid the
      rest of the move the rule explicitly allows.
      CLOSED by Moving and Shooting (p. 139): the -1 now exists, and `shootAt`
      reads `manoeuvreThisTurn` alongside `hasMovedThisTurn` precisely so a
      redress is counted as moving without setting the flag that would bar the
      rest of the move.
      LEFTOVER: the other four manoeuvres (wheel, turn, move backwards, move
      sideways, reform) do not set `manoeuvreThisTurn`, so only a redress
      currently blocks a second manoeuvre.
      Corrected since: none of the three fields the metering runs on —
      `moveSpentThisTurn`, `manoeuvreThisTurn`, `redressDelta` — were written
      to a save, so a quicksave taken mid-move refunded the half Movement a
      manoeuvre had cost, lifted the one-per-move limit and handed back the
      five-model redress allowance. All three are saved and loaded now.
      The guard test that should have caught this matched flag *names* ending
      in ThisTurn/ThisPhase/NextTurn, which found the first two and missed
      `redressDelta` entirely. It now derives the list from the start-of-turn
      reset in `game_fsm.enterStrategyPhase`, which is what actually defines
      per-turn state: anything cleared there must appear in the save. That
      reads 10 fields today, `redressDelta` among them.
- [x] A save carries the army list's special rules — `persistence.py` stored
      only what the catalogue knows, so the *roster's* rules were lost for any
      unit a load had to rebuild: Skirmishers, Swiftstride, Fire & Flee, Fear
      and the rest all arrive from the army list through `applyDataRules`, not
      from the model profile.
      It hid because a load normally restores state onto units that already
      exist, and those keep the rules they were given at army load. Only a unit
      missing from the scene goes through `_create_unit`, which has nothing but
      the catalogue profile to build from — and that is exactly the path a save
      from a different army takes.
      Found while building a quicksave to test Fire & Flee: the rule vanished
      on load. The names are saved per unit and re-applied to every unit on
      load, which is idempotent, so a save also repairs a unit that predates a
      rule the roster grants.
      LEFTOVER: `mount_special_rules` are not saved with them, so a mount's
      rules still depend on the army file being the one that was played.
- [ ] Test/CI hardening; broaden `tests/` to a couple of full factions
      Fixed eleven regressions after Skirmisher integration: bound-spell test
      games now explicitly have no formation editor, and lifecycle test doubles
      support spreading after combat or a successful rally. Detours still skip
      turn-end cleanup; failed rallies do not spread. Panda3D ShaderInput vectors
      borrow their owner's storage: overlay assertions now copy with the owner
      alive, and live preview cleanup retains the input while testing its flag.
      Added active/inactive cleanup coverage without scene startup. Restored
      the missing committed Skirmisher Move Through Cover log as detailed above.
      LEFTOVER: broader full-faction coverage and CI setup remain outstanding.
- [ ] Empire units render with the generic model (no `.bam`) — add mappings
- [x] One hand weapon per model, and it is the catalogue's — every model was
      given an invented `'hand weapon'` carrying a made-up description, while
      the roster supplied the real `'Hand Weapon'`. `weapons` is keyed on the
      raw name, so the two capitalisations were two separate weapons and
      models carried both.
      The invented one is gone: models take the catalogue entry, which brings
      its actual rules and the rulebook note with it. A stub remains only for
      running with no catalogue at all. `model.weapon_slot` matches names
      case-insensitively, so `equip_weapon('hand weapon')` still resolves, a
      roster's copy merges into the existing entry instead of sitting beside
      it, and saves written before the fix collapse to one entry on load.
      Two tests asserted the invented lower-case name; the fallback is the
      point, not its capitalisation, so they assert `uses_hand_weapon()`.
- [x] A cocked die no longer reports the previous roll — `checkDice` tested
      each of the six body axes against a fixed `dot > 0.9` and set
      `currentValue` when one cleared. A die that landed cocked, resting on
      another die or on the rim, cleared none of them, and `currentValue` kept
      whatever the *last* roll had put there. Across a sweep of 197k
      orientations the old read produced no answer for 56% of them.
      It reads the axis pointing most nearly at the sky instead, so there is
      always a face, and prints a line when the die is under 0.75 square-on —
      the one result worth distrusting, and previously invisible. Verified
      against the old logic over the same sweep: wherever the old code read a
      face the new one returns the identical face, 0 mismatches, so the
      numbering is unchanged.
      This reached further than the dice strip that exposed it. `game.py`
      Leadership tests, charge and flee distances in `combat_resolution.py`
      and the casting roll in `spell_system.py` all sum `currentValue`
      straight, so each would have silently reused a stale number.
      LEFTOVER: nothing re-rolls or nudges a cocked die, which is what a
      player would do; it reads the nearest face and says so.
- [x] One world unit is one inch, everywhere — `game.WORLD_UNITS_PER_INCH` was
      3.0, so weapon ranges alone were tripled. Nothing else on the table
      agreed with it: `models.MM_PER_UNIT` sizes bases at `base_mm / 25.4`, the
      board rectangle is 72x48 (a 6'x4' table) and the deployment zones are 12
      deep, all of which are plain inches, and Cannon Fire, Bombardment, spell
      ranges and Command range all compare a range in inches straight against a
      world-space `length()`.
      The effect was that a 24" bow drew a 72-unit arc — the full width of the
      table — and Long Range (-1 To Hit) began at 36" instead of 12", so almost
      every shot counted as short range. Cannons, measuring honestly, were
      out-ranged by bows.
      Scale is now 1.0 and the three baked-in `* 1.5` half-range factors and
      the `* 3 / 100` arc radius go through the constant. The `/ 100` in the
      arc radius stays: it is the world-to-normalised-space conversion for
      `movement_system.shootingArc`, not part of the table scale.
      LEFTOVER: only ranged weapons were ever scaled, so nothing else needed
      correcting — but the scale is still a bare module constant in `game.py`
      rather than living beside `MM_PER_UNIT` in `models.py`, which is where a
      reader would look for it.
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
      Step 2 now uses separate part Initiative, owned weapons, casualties per
      base and full eligible crew counts for crew-operated missile weapons;
      the hull is no longer counted as a single bow shooter.
      Skycutter correction (Forces of Fantasy p. 172; Rulebook pp. 194-195):
      the roster calls it "Lothern Skycutter", while the Model profile is
      "Skycutter". An alias now resolves both names. Its outer unit entry has
      no Unit profile; the linked model carries Heavy Chariot, size and rules.
      When the outer troop type is absent, parsing falls back to that model
      context. Existing outer profiles remain authoritative; this is not a
      blanket merge of every model-level rule list into its parent unit.
      Verified S5/T4/W4, three Sea Guard crew, one Swiftfeather Roc, Fly (10),
      60x100mm base and Impact Hits (D3+1) at S5 AP-2 (Scythed Wheels), wounding
      a T4 Chaos Knight on 3+. Sources checked:
      https://tow.whfb.app/unit/lothern-skycutter and
      https://tow.whfb.app/troop-types-in-detail/scythed-wheels.
      Old failed lookups saved only rule keywords and no stat keys. Loading
      such a record now fills its profile from the catalogue before unit
      recreation/restoration, preserving saved rules, wounds and battle state.
      A console/HUD notice names the repair. The restored base profile survives
      combat resets and re-saving; the original save is not rewritten by Load.
      Tests in `tests/test_chariots.py`, `tests/test_persistence.py` and
      `tests/test_skycutter_scene.py` cover fresh creation, existing/missing
      units on load, actual S5/AP-2 wounds, split parts, footprint, reset and
      round-trip stability, plus preserving explicitly saved custom stats.
      Follow-up save-state correction: `characteristics` is now the live
      snapshot and `base_characteristics` separately records the roster
      baseline. A live S9 over a custom S7 baseline reloads as S9, then resets
      to S7, never to the catalogue S5 or a permanently boosted S9. Mounts,
      crew and beasts use the same split in `profile_parts`, preserving
      component identity, counts, custom stats and rule selections. Permanent
      roster keyword updates also update the baseline without promoting live
      numeric bonuses. Supported ongoing spell effects remain separate in
      `spells_in_play`; repeated loads restore one Oaken Shield with its saved
      duration, combat reset preserves it, and expiry removes it. Tests share
      the existing offscreen scene instead of creating an app for each case.
      LEFTOVER: old saves without a baseline keep the saved profile as their
      baseline; temporary bonuses embedded in those values cannot be reliably
      distinguished from custom roster stats. Live numeric snapshots are not
      a general modifier engine. Non-catalogue demonstration spells, including
      Devil's Visit, are intentionally outside this work. No new spell effects
      or mid-combat coroutine resumption are implemented.
      LEFTOVER: recovery deliberately skips profiles containing any stat key,
      including explicit zero or '-' values; partial or unknown profiles are
      not guessed. This does not change chariot crew weapon allocation or
      introduce new aliases for other mismatched roster/model names.
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
- [x] Too Tough to Wound (Rulebook p. 140; combat chart p. 149) - Strength
      six or more points below the target's Toughness cannot wound. Checked
      https://tow.whfb.app/the-shooting-phase/too-tough-to-wound and the combat
      To Wound chart. CORRECTED: this entry previously claimed a three-point
      gap and p. 143; gaps of two through five still wound on 6+.
      `to_wound` returns the existing impossible target of 7. Melee, shooting,
      spell hits and Impact Hits share it, using the attack's effective
      Strength (unmodified Strength for Impact Hits) and target's Toughness.
      A modified wound die cannot bypass an impossible target. Natural sixes
      cannot trigger Killing Blow or Monster Slayer against it either.
      Batch logs show hits, Strength, Toughness, gap and zero wounds, or why
      the rule was skipped; there is no per-die logging. Combat readouts say
      "impossible", and slaying-blow refusals name the unwoundable target.
      Tests: `tests/test_too_tough_to_wound.py` covers boundaries, attack paths,
      modifiers, saves, both slaying rules and logs.
      LEFTOVER: no additional rule work; automatic-wound exceptions, when
      implemented by their own rules, must explicitly define their interaction.

## Deferred war-machine items
- [ ] Multiple Wounds (D3+1) generic rule
- [ ] Black Powder / Misfire tables
- [ ] True line-of-sight and rank/file hit caps for cannon/bombardment
