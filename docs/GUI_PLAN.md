# GUI plan

The simulation is well ahead of its interface. This is the plan for closing
that gap, what has been built so far, and what deliberately has not.

Design sketch: [hud_mockup.svg](hud_mockup.svg) — open the VS Code preview.
Current state rendered offscreen: [hud_render.png](hud_render.png).

## The problem this solves

The engine's most valuable output is invisible. `rules_log.rule_log` and
`rule_skipped` are the only evidence that a rule fired or declined — a Ward
save that works and a Ward save that was never coded look identical on screen —
and until now that trace went to the console alone. Combat results, morale
tests and modifier stacks were in the same position.

Everything else followed from four text nodes at hardcoded aspect2d
coordinates, each owning whichever corner it was written for, with no layout
system to stop them colliding.

## Screen zones

One owner per zone. Nothing may draw outside the zone it was given, and
nothing covers the centre of the board.

| Zone | Anchor | Owner |
|---|---|---|
| Top left | `a2dTopLeft` | Turn banner — player, round |
| Top centre | `a2dTopCenter` | Phase track |
| Top right | `a2dTopRight` | *free* — army strength |
| Bottom left | `a2dBottomLeft` | Selected unit readout |
| Bottom centre | `a2dBottomCenter` | Transient readouts: dice, targeting, status |
| Bottom right | `a2dBottomRight` | Battle log |

Anchor to the corner nodes, never to raw aspect2d coordinates. `(-1.3, 0.9)`
is 16:9 only; on any other window shape it drifts off the edge or into the
middle of the table.

## Architecture rules

- **Build once, `setText` after.** No destroy-and-recreate on every update.
- **Publish, don't reach in.** Game systems post `messenger.send('hud-log',
  [text, category])`; they never import or touch a widget. The HUD is a
  subscriber. Events: `hud-turn`, `hud-phase`, `hud-log`.
- **One source of truth per fact.** When the phase track went in, the nine
  `debugText.setText(f"Current phase: ...")` echoes came out. A duplicated
  readout is a readout that will eventually disagree with itself.
- **A display that throws must not stop a rule resolving.** `rules_log`
  catches and reports listener failures rather than letting them propagate.

## Built

- `hud.py` — `HUD` class: turn banner, phase track, battle log, `F3` to hide
  all chrome for screenshots.
- Phase track lit from `GamePhaseFSM.request`, which is overridden to
  broadcast. Phases are requested directly from about six places as well as
  through `nextPhase()`, so hooking the transition is the only way the track
  sees all of them. Off-track states (`SpellPhase`, `MakeChoice`,
  `CampaignPhase`) show as a suffix chip rather than blanking the track.
- Battle log: 9 entries, colour-coded by category — rules fired gold, rules
  *declined* dim grey, dice pale blue, morale red, kills green. Bottom-anchored
  so the newest line stays put however much earlier lines wrap. Categories are
  inline `TextProperties` runs in a single text node, not one widget per line.
- `rules_log.add_listener` / `remove_listener`. Printing is unchanged; the
  trace is now also emitted as `(kind, rule, subject, detail)`.
- `RoundCounter.update_round_display` publishes `hud-turn` instead of
  destroying and recreating an `OnscreenText`.
- `CombatResolver.printBattleResults` also posts a one-line summary.
- The four loose text nodes anchored to corners; the selected-unit readout no
  longer dumps the raw characteristics dict.

## Next, in order

| Item | Effort | Payoff |
|---|---|---|
| Second font — sans for numbers and body | Low | High |
| Unit card: proper panel, models bar with the 50%/25% thresholds, state chips | Low–Med | High |
| In-world: selection decal, facing chevron, status badges, strength bar | Low–Med | High |
| `legal_actions(unit)` + contextual action bar with disabled reasons | Med | High |
| End Phase button + army strength readout | Low | High |
| Target preview: to-hit, to-wound, save, modifier stack, expected casualties | Med | Very high |
| Combat result card | Low | Medium |
| Log scrollback and category filtering | Low | Medium |

The two genuinely hard ones:

- **Target preview** needs an expected-value path that provably cannot drift
  from the dice path. Compute it from the same `toHitAndToWound` helpers;
  never re-derive the numbers in the UI.
- **Action bar** needs legality extracted out of the task loops.
  `taskLoopPathTowardsMouse` currently prints "Unit is not idle" and returns —
  the reason a unit cannot act is computed but not queryable. Extract
  `legal_actions(unit)` and have both the task loops and the bar call it.

## Look and feel

- **Two fonts.** MedievalSharp for headings and unit names only; a clean
  humanist sans for stat lines, numbers and the log. Display faces at 0.03
  scale are illegible, and a stat block is tabular data.
- **A type scale, not arbitrary values.** 0.075 title, 0.05 heading, 0.038
  body, 0.03 caption.
- **Semantic colour.** Gold means important, red means bad for you, green means
  good for you, and nothing else may use them. Parchment and ink for
  everything neutral.
- **Panels float, the board does not.** Translucent, edge-anchored, never over
  the centre of the table.
- **Keep the physical dice.** Tumbling d6s are the most characterful thing in
  the build. Add a 2D strip beside them showing values against the target
  number so the *result* reads while the *rolling* stays tactile.

## LEFTOVER

- No second font yet — MedievalSharp is still setting the stat lines, and it
  is the weakest part of the current screen.
- Battle log has no scrollback; it is a fixed 9-entry window with no filter.
- Nothing posts to the log from `psychology.py`, `spell_system.py`,
  `cannon_fire.py` or `bombardment.py` yet. They only need a
  `messenger.send('hud-log', ...)` at the point of resolution.
- Dice results still go to `diceInfoText` as plain text, not the roll strip.
- `debugText`, `debugTextInfo`, `debugTextUnit` keep their debug-era names
  despite carrying real gameplay information.
- The `endPhase` collision cube is still the only way to advance a phase, and
  it competes with unit picking for the same mouse ray.
