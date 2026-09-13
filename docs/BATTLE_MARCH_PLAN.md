# Battle March Implementation

Source: [General's Companion section](https://tow.whfb.app/battle-march),
reviewed 2026-09-13. This is not the older Settra's Fury ruleset.

## Approved Stages

1. Preset and validation: versioned configuration, unchanged standard defaults,
   army violations reported without rewriting rosters.
2. Shared battlefield geometry: six maps and mirrors, full-base validation,
   deployment, Scouts, AI, movement, fleeing, pursuit, overlays and physical edges.
3. Complete standard setup: terrain policies, objectives, zone choice,
   alternating deployment, separate first-turn roll-off and choice.
4. Playable objectives and scoring: control, temporary landmark effects,
   per-player-turn awards, HUD/AI, idempotent saves, five complete rounds.
5. Optional modules: Raid & Burn, baggage carts, ten private secret objectives,
   all three random-happening tables, 20 magic items, tournament options.
6. Narrative presets: Weisnicht Bridge, Bel-Cedas, Drazgog, Karak Ziflin,
   Sturdham and Gemelwald, including their unique mechanics.
7. Verification and milestones: both sizes, all maps/mirrors, rotated/dispersed
   bases, thresholds, terrain clearance, control ties, expiry, both turn orders,
   save/reload, offscreen visuals and isolated bounded tests; milestone commits.

## Implemented Foundation

- `config/battle_march.json` is the editable 500-point, 44 x 30, five-round preset.
- `battle_config.load_config` rejects malformed, unknown and duplicated fields.
- `battle_setup.resolve_setup` records map/objective/property dice using a seed.
- Saved configuration and resolved choices are independent copies. Restoring
  checks the recorded dice without RNG calls. Older saves clear custom geometry.
- `battlefield` owns playable edges and all six map shapes; the visual table
  remains 72 x 48. Units and terrain are not scaled.
- Runtime placement and boundary consumers use per-game geometry; standard
  games retain legacy defaults. Overlay lines and AI candidates use those zones.
- Mustering reports check available paid-point/category/starting-strength data;
  absent evaluated faction composition and restricted-option metadata is
  explicitly unverified. Imported roster category is preserved.
- Objective records use the verified diagrams: two at (0, +/-7.5), three at
  (-11, 0), (0, 0), (11, 0), or a central landmark. The earlier planning
  summary's horizontal two-trove description was incorrect.
- Shapely-backed terrain calculations cover rendered natural-feature rims,
  centre/opponent spacing, recommendation warnings, first-contact scatter,
  and minimum terrain translation clearing fixed objectives. UI integration,
  river footprints and other-feature relocation conflicts are pending.
- Offscreen 44 x 30 and 48 x 36 captures verify colored overlays and removal
  of the legacy rectangle. Real-scene saves restore geometry without RNG.
- Objective control now uses actual bases, joined Unit Strength, distance/US
  tie-breaks and explicit choices only for multiple controlled objectives.
  Both players score at each player-turn end; the saved ledger prevents repeats.
- Landmark MR(-2), Frenzy and Stubborn grants expire at the next player-turn
  end, preserve permanent sources and survive reload, including lost Frenzy.
- Frenzy now has majority psychology/no-Flee/no-Restraint checks, loss on a
  defeated combat round, conditional attacks with split-profile exclusions,
  follow-up turn tracking and compulsory declarations without Leadership.
  Supported loose form-ups use the same planner as actual charge moves.
- Battle March uses 50/25/25 General/standard/BSB bonuses and most VP wins.
  Standard-game scoring is unchanged.
- Numbered trove tokens and circular landmark terrain now render and restore.
  Shared movement and sight checks retain the landmark's round footprint and
  all-LOS prohibition. Both HUD orientations show control/property/objective VP,
  and battle results include objective award categories. Offscreen visual
  checks cover both objective layouts and both HUD orientations.

This is not yet a playable Battle March mode. No startup option is exposed.
The presence of objective, terrain or optional settings does not mean their
effects are implemented. These are remaining work in stages 1 and 3-6.

## Source Decisions

- Follow Battle March's explicit most-VP wording, not the core 100-VP margin.
- User confirmed literal cart scoring: an escaped cart earns its owner the
  25 VP escape bonus and its opponent the 25 VP "otherwise" award. Escape
  remains distinct from destruction/Dead or Fled.
- User confirmed written narrative rules over conflicting illustrations or
  cross-references: Gemelwald village goes in the defender half; the Weisnicht
  Bridge duel determines first turn.
- Narrative building-size feet/inches typos still need the applicable source
  verified when implementing those scenarios; do not stretch printed distances.

## Verification Constraints

Every terminal command activates `.venv`. Use `run_tests_isolated.py`; never
run several Panda3D test modules in one pytest process. The full-suite guard
stays at 1536 MiB plus 256 MiB available headroom. Render visual checks offscreen.
Never overwrite or stage user saves, local armies, startup army edits or the
paused HUD layout probe. No changes to Devil's Visit.