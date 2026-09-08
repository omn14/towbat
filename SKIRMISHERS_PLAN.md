# Skirmishers — Implementation Plan

> Status: **in progress**. The loose-movement foundation and optional formation
> editor are implemented and tested. This is not complete Skirmish Formation:
> two loose units now form fighting ranks in sequence, and loose chargers form
> against an unengaged formed target. Formed ground chargers now share an actual-base,
> closest-visible-model approach across cursor, declaration and resolution against
> supported loose defenders. Compact fleeing units wait for successful
> Rally before separating (requested house rule). Combat scoring and individual
> shooting are implemented below; other charge pairings, full height/cover rules
> and constrained separation remain outstanding.
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

### Defending fighting-rank corner-contact correction

Sources: [Skirmishers Charging Skirmishers](https://tow.whfb.app/unusual-formations/skirmishers-charging-skirmishers)
(Rulebook p. 187) and [Base Contact](https://tow.whfb.app/the-combat-phase/base-contact)
(p. 145). Defenders form after the chargers, within each defending model's
Movement characteristic. Corner-to-corner contact counts as enemy base contact.

- Corrected the defender frontage cap, which copied the charging rank's width
  and could force both units into one-file columns. The existing contact-aware
  rank builder now checks each front slot against the actual charging front bases,
  including their corners, and considers which slot holds the contacted defender.
- With equal bases, a single charging file can face three defending files if
  those models can reach. Further defenders still form behind: spare Movement
  does not permit a front slot without enemy contact. Per-model Movement limits,
  attacker-first formation and the existing frontage-count log remain in force.

Verification: reproduced one defending file where three could contact with only
0.36 inches of travel by the outer two models. The correction passed 407 tests
and 82 subtests across Skirmishers, characters and persistence. Ten new regressions
cover five rotations, insufficient/asymmetric Movement and actual charge resolution
at 1280x720 and 800x600. Both renders were inspected; checks cover front-base
contact, surviving model identities, travel limits, logs and model pixels.

LEFTOVER: existing compact combats in saved games are not re-formed. Joined or
mixed bases, unsupported charge pairings, exact contact/path selection and
obstacle-aware formation remain subject to the limitations below.

### Formed chargers: contact-stage increment

Source: [Formed Units Charging Skirmishers](https://tow.whfb.app/unusual-formations/formed-units-charging-skirmishers)
(Rulebook p. 186), with corner contact from p. 145.

- Implemented the post-contact stage for formed chargers against unengaged loose
  Skirmishers with uniform defending bases and no joined characters. At actual
  front-base contact, the charger retains its position/facing and does not take
  the normal alignment wheel. Only the defenders animate into fighting ranks.
  Corrected contact eligibility to test front-edge segments, not the side/rear
  of a model merely because it occupies the front row.
- Defenders face the charging front, with front slots constrained to enemy base
  contact and every model limited to its Movement allowance. Rear slots and
  coherency losses use the existing rank builder. Combat records the charged
  Skirmishers as facing the charger, rather than assigning them a loose flank.
- Applied logs report the fighting-rank count, rear count, M and the stationary
  charger. Missing real contact, unsupported rank geometry and unsupported loose
  pairings retain legacy alignment with an explicit LEFTOVER log. Failed charges
  never trigger defender form-up.

Verification: reproduced the unwanted alignment call before the fix; 844 tests
and 82 subtests passed after it. Thirty new cases cover rotated fronts,
corner contact, individual Movement, missing front contact, mixed/unsupported
pairings and real charge success/failure at 0, 37 and 180 degrees. Offscreen
1280x720 and 800x600 contact renders were inspected; model-hidden comparisons
verified model pixels and all ranks were checked inside the viewport.

The initial contact-stage increment is now connected to the shared approach below.
Unsupported pairings still retain the explicitly logged legacy fallback.

### Formed chargers: shared approach

Sources: [Formed Units Charging Skirmishers](https://tow.whfb.app/unusual-formations/formed-units-charging-skirmishers)
(p. 186), [Wheel](https://tow.whfb.app/movement-in-detail/wheel) (p. 124),
[Manoeuvring During A Charge](https://tow.whfb.app/movement-in-detail/manoeuvring-during-a-charge)
(p. 126), line of sight (p. 103) and charge/terrain rolls (pp. 121, 269).

- `formed_skirmish_charge.py` selects the closest visible defending model using
  actual base distances and the formed charger's front arc. Other models and
  sight-blocking terrain screen individual targets; a blocked route does not
  silently substitute a farther visible model.
- The same route supplies the cursor's cyan base ghosts, target index, wheel
  cost, distance/maximum, declaration validation and live animation. The loose
  footprint is only a mouse hit area: contact is with an actual model base.
  Declaration revalidates from the original transform before reactions or dice,
  without trusting the cursor cache or moving live models. Human Yes/No and
  direct/AI calls share the gate.
- Routes contain an optional straight lead, one paid front-corner wheel and a
  straight approach. Wheel cost is frontage times angle in radians (p. 124).
  Board edges, other models and impassable rectangles constrain ground paths.
  After contact, only defenders form within M; no charger alignment wheel is added.
- Crossed terrain is measured along the swept formation, not merely its centre
  line. Route M and difficult-terrain charge dice share the existing Move Through
  Cover protection/logs. Hazard tests use the features crossed by actual travel.
  Failed charges advance only the charge roll along the route and leave defenders
  loose. Stand & Shoot survivors retain the declared target model; a rebuilt route
  supplies the revised terrain cost. A destroyed charger never rolls or forms up.
- Preview caching invalidates on geometry, terrain and charge-permission changes.
  Leaving the target/board/window clears the preview; save loading clears transient
  route/cache state. Logs name target, paid wheel, approach, roll and final frontage;
  cursor queries remain silent. Flee reactions explicitly log the legacy handoff.
- Corrections during validation: tightened SAT precision to avoid a corner-contact
  rounding miss; wrapped `LerpFunc` in an awaitable `Parallel`; removed stale legacy
  distance labels; rejected removed targets before reading their bases; recalculated
  terrain cost after reaction casualties and retained Move Through Cover outcome logs.

Verification: 48 new route tests cover rotated/offset approaches, closest-visible
selection, blockers and a delayed wheel, range refusal, human/AI confirmation,
success/failure, terrain, cache invalidation, save cleanup and Stand & Shoot.
The final focused run passed 84 tests including Move Through Cover and Quick Shot.
Live offscreen animation and preview/contact renders at 1280x720 and 800x600 were
checked for model pixels, framing, labels and defender-only movement after contact.
The full-suite run before the final four regression cases reported 1599 passes,
165 subtests and eight failures. All eight were independently reproduced on an
isolated `7414299` export with the same catalogue data: four bound-spell phase
cases, one older Move Through Cover movement-log case, two Shieldwall fixtures
and one Veteran Rally fixture. They were not changed by this work.

LEFTOVER: route search is bounded, not an exhaustive continuous solver: it samples
wheel angles through +/-90 degrees (an implementation limit, not a claimed rule
limit), refines the first contact boundary and tries delayed wheels in 0.5-inch
steps. It prefers shortest clear routes rather than globally maximizing eventual
contact across all possible routes. Conservative rotational/rectangular sweeps may
refuse tight legal paths. Flying obstacle overflight/landing, fleeing-target chase
and redirects still need route integration; flee currently uses legacy movement.
Joined/mixed defending bases, multiple chargers, already-engaged defenders and
pursuit remain unsupported by this planner. Sight is planar; defender form-up
does not yet resolve individual terrain/obstacle detours or alternate assignments
before coherency losses. These limits must not be read as completed rule coverage.

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
arc-straddling declarations need further adjudication. The formed ground-charge
approach above supersedes the earlier loose-defender limitation. Already-engaged targets, fleeing/pursuing units,
joined/command models and multiple charges retain the older alignment path. These are not complete
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
Combat scoring/disruption is implemented below; other charge-pairing limitations remain.

The reproducible offscreen scenario is `python -m tests.test_skirmish_scene`.
It writes `saves/skirmishers.json` and `screenshots/skirmishers.png`, without
overwriting the player's quicksave. Load that save, select **Normal Rangers** in
Movement, then use **Adjust formation**. Saving does not persist an unconfirmed ghost.

## Charge-visibility follow-up

Sources: [Facing & Line of Sight](https://tow.whfb.app/unusual-formations/facing-and-line-of-sight-skirmishers)
(p. 184), [Skirmishers & Charging](https://tow.whfb.app/unusual-formations/skirmishers-and-charging)
(p. 186), and [Line of Sight](https://tow.whfb.app/model-and-unit-facing/line-of-sight)
(p. 103). Official FAQ v1.5.3 does not replace the strict-majority requirement.

- Charge previews and declarations now require strictly more than half of the
  charging models to see at least one target model. Exactly half is refused.
  Attached characters count once, with their own base; undeployed units do not
  create blockers. Friendly, enemy and other models in the charging unit block
  sight individually. Loose formations do not create an opaque bounding rectangle.
- The read-only planar sight helper uses actual rotated bases and tests the open
  angular intervals between their corners. A visible part of a target is enough;
  its centre need not be visible. Narrow gaps work without fixed-angle sampling,
  while seams between touching bases do not grant sight.
- The preview and confirmation show visible/total counts. A refused click or
  direct movement call logs the count and required majority without spending
  movement. Successful declarations log once; previews and cancelled choices do
  not log that a charge was declared. Existing AI movement uses the same gate.
- Corrected declaration timing: the engine temporarily moves chargers into
  contact before offering reactions. Sight is calculated from the original
  position/facing without moving live bodies, and an invalid declaration restores
  that transform before choices or dice. Stand & Shoot casualties do not revoke
  a valid declaration. Pursuit contact is not checked as a fresh declaration.

Verification: 804 tests and 82 subtests passed, including 43 new visibility tests.
The original exactly-half preview failure was reproduced before implementation.
Coverage includes rotated narrow gaps, complete screens on either side, target
edges, attached models, terrain, human/AI refusal, cancellation, pursuit and
Stand & Shoot timing. Allowed/blocked panels were rendered and inspected at
1280x720 and 800x600, with bounds and nonblank-image assertions.

LEFTOVER: sight is from each base centre in XY, not from arbitrary points on a
sculpt or a complete 3D height model. Large Target and hill-height exceptions,
irregular terrain silhouettes and detailed woodland visibility need integration.
Terrain uses conservative rectangles and the existing see-onto/not-through
convention. The angular interval method assumes non-overlapping model bases.
This charge milestone did not change shooting; the individual-shooting follow-up
below now supplies per-shooter range, sight and the all-US1 modifier. Spell sight
and cover retain their older paths. AI target
selection is still its existing policy, although it cannot commit an invalid
Skirmisher declaration. Arc-straddling charges remain a separate task.

## Combat scoring and individual shooting

Sources: Rulebook pp. 101, 137, 139, 152, 184-185 and Official FAQ v1.5.3:
[Skirmishers & Rank Bonus](https://tow.whfb.app/unusual-formations/skirmishers-and-rank-bonus),
[Skirmishers & Disruption](https://tow.whfb.app/unusual-formations/skirmishers-and-disruption),
[Unusual Formations FAQ](https://tow.whfb.app/faq/unusual-formations),
[Check Line of Sight](https://tow.whfb.app/the-shooting-phase/check-line-of-sight),
[Check Range](https://tow.whfb.app/the-shooting-phase/check-range),
[Skirmishers & Shooting](https://tow.whfb.app/unusual-formations/skirmishers-and-shooting),
[Enemy Fire](https://tow.whfb.app/unusual-formations/enemy-fire-skirmishers),
and [Shooting FAQ](https://tow.whfb.app/faq/shooting).

- Compact Skirmishers retain their rank-bonus exemption and grant no enemy
  flank/rear combat-result points. Their physical contact arcs are unchanged.
  Skirmishers attacking formed flanks/rears can still score those points, but do
  not disrupt ranks. Formed enemies with surviving US5+ do disrupt; joined
  character strength and independent terrain disruption are included.
- Score these effects once after attack casualties, with outcome logs. Corrected
  stale stored ranks and slain flank opponents contributing points. Explicit
  active formation overrides the model keyword; retained compact combat state
  preserves Skirmisher exemptions. Legal formation switching is still pending.
- `shooting_geometry.py` supplies the shared target-highlighting, aiming,
  direct-volley and reaction query whenever a shooter, target or live deployed
  intervening unit is a Skirmisher. Actual rotated bases block sight, including
  the shooter's own models; gaps remain transparent. Loose shooters have 360-degree
  sight. Each eligible model measures its own base-to-base range and range band.
- Volleys group eligible models by profile and short/long range without changing
  live files/ranks. Multiple Shots is chosen once, with weighted mixed-range
  advice. Formed firing/Volley Fire ranks inherit front-file sight while using
  their own ranges; the joined character's reserved slot is respected, including
  an unarmed character. Joined shooters use their own weapon and range.
- Existing whole-unit hill visibility exceptions and the extra firing rank are
  retained. Stand & Shoot uses declaration-time target bases and sight, waives
  maximum/long range, and now sends its reaction modifier to both dice and shot
  reports. Ordinary volleys reset that modifier. Completely screened volleys and
  invalid direct targets are refused before choices/dice without spending shooting.
- The enemy-fire penalty requires every live model, including a joined character,
  to be US1. Applied/skipped logs give the qualifying/total count. Eligibility
  summaries report short, long, blocked and out-of-range counts; group logs report
  shots, hits and wounds. Per-model queries and dice loops remain silent.

Verification: 42 scoring and 34 shooting regressions; 164 scoring/psychology tests,
72 shooting/reaction tests and a final 167-test charge/visibility/scoring/shooting
gate passed. Inspected aiming renders at 1280x720 and 800x600, with text-bound checks.
The full run had 1678 passes and 165 passing subtests, with the eight previously
reproduced baseline failures plus one reaction mock missing the new optional
keyword; that mock was corrected and the affected 167-test gate passed.

LEFTOVER: this is planar base-centre sight, not complete 3D sculpt/height or Large
Target visibility. Hill exceptions use the existing whole-unit approximation;
irregular terrain, detailed woodland visibility and partial/full-cover modifiers
are not newly implemented. Spell targeting and battles without active Skirmishers
retain their older sight paths. The circular range overlay remains a unit-centre
guide, not an exact map of individual legal shots; counts and target highlighting
are authoritative. Mixed-size formed character placement, general command models,
joined-model rank contributions and multi-combat allocation remain separate work.
The AI shares execution legality but has no new per-model tactical planner.

Separate charge LEFTOVER discovered while checking FAQ: the existing formed-route
wheel cost uses frontage times angle (arc length), whereas the Movement FAQ v1.5.3
measures the outside front corner's straight-line displacement. Its conservative
sweep also does not allow every FAQ-permitted rear-corner crossing. This was not
changed as part of combat scoring/shooting and needs a focused charge correction.

## Remaining implementation order

1. **Authoritative formation state:** separate permission to adopt Skirmish from
   the active formation; implement legal reform switching. Harden malformed-save
  handling.
2. **Sight and shooting refinements:** full height/Large Target, detailed terrain
  and cover; exact per-model overlays and broader tactical AI. Individual range,
  planar sight and joined-model Unit Strength checks are implemented above.
3. **Charge sight and arcs:** integrate height/Large Target exceptions and
  detailed terrain visibility; adjudicate arc-straddling declarations. The
  strict-majority gate, visible count and declaration timing are implemented above.
4. **Charge movement/form-up:** extend the loose/formed-target planners to joined
  and mixed bases, exact contact/path selection and alternative rear arrangements.
  Closest-visible-model targeting, approach and defender-only form-up for supported
  formed ground chargers are implemented above; correct the FAQ wheel measurement
  and improve the bounded route search. Multiple chargers, command placement and per-model
  terrain still need integration and fixtures.
5. **Compact combat and constrained separation:** account for terrain, board
  edges and other units when finding the smallest legal separation; timing and
  the fleeing Rally-only house rule are implemented above. Rank, flank/rear
  scoring and Skirmisher disruption exemptions are now implemented without
  changing physical contact arcs.
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
- Rank/flank bonus is computed after casualties in `combat_resolution.py`, using
  `psychology.combat_rank_bonus` and `combat_flank_bonus` with live model counts.
- Shooting arc is a ~90° front arc: `shootingArc(..., rotationangle=getH()+45)`
  (`movement_system.py` ~L208, driven from `taskShootingArcUpdate` in `game.py`).
  Individual eligibility comes from `shooting_geometry.py` when Skirmishers are
  present; the old range overlay and non-Skirmisher-only sight path remain guides.
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
