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

This is not yet a playable Battle March mode. No startup option is exposed.
The presence of objective, terrain or optional settings does not mean their
effects are implemented. These are remaining work in stages 1 and 3-6.

## Source Questions

- Battle March's most-VP wording versus the linked core 100-VP victory margin.
- Whether escaped carts also award the opponent the "otherwise" 25 VP.
- Feet/inches typos on narrative feature sizes.
- Weisnicht Bridge's incorrect first-turn cross-reference.
- Gemelwald village text/diagram disagreement about the designated half.

Resolve these from the supplement or errata before implementing the affected
mechanics. Do not silently invent scenario rules or stretch printed distances.

## Verification Constraints

Every terminal command activates `.venv`. Use `run_tests_isolated.py`; never
run several Panda3D test modules in one pytest process. The full-suite guard
stays at 1536 MiB plus 256 MiB available headroom. Render visual checks offscreen.
Never overwrite or stage user saves, local armies, startup army edits or the
paused HUD layout probe. No changes to Devil's Visit.