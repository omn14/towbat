# GUI plan

The simulation is well ahead of its interface. This is the plan for closing
that gap, what has been built so far, and what deliberately has not.

Design sketch: [hud_mockup.svg](hud_mockup.svg) — open the VS Code preview.
It predates the bottom command bar and still shows the four-corner layout.
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

Everything the player reads lives in one command bar along the bottom of the
screen. One owner per section. Nothing may draw outside the section it was
given, and nothing covers the centre of the board.

| Section | Span | Owner |
|---|---|---|
| Rail | 0.000–0.050 | Army list and spellbook slots |
| Regiment | 0.050–0.315 | Selected unit: portrait, name, stats, models bar |
| Log | 0.315–0.560 | Battle log |
| Centre | 0.560–0.790 | Recent dice, phase track, player and round |
| Rules | 0.790–0.838 | Game rules slot |
| Objectives | 0.838–0.930 | Objectives |
| End | 0.930–1.000 | End Phase button |
| Follows the cursor | `aspect2d` | Hover tooltip (drawn above everything) |

Spans are fractions of the bar's width, re-flowed on `window-event`. Never
write a raw aspect2d x coordinate: `(-1.3, 0.9)` is 16:9 only, and on any
other window shape it drifts off the edge or into the middle of the table.
The bar itself is anchored to `a2dBottomCenter` and sized from
`base.getAspectRatio()`.

### The vertical ledger

`F2` swaps the bottom bar for a right-hand ledger, and the choice is written
to `settings.json` so it survives a restart. The board then keeps nearly the
full height instead of the full width, which suits a tall window where the
bar eats a quarter of the play area.

| Section | Span | Owner |
|---|---|---|
| Turn | 0.000–0.058 | Player and round |
| Regiment | 0.058–0.348 | Portrait, name, models bar, stats, chips |
| Phase | 0.348–0.508 | The turn sequence as a lit list |
| Dice | 0.508–0.622 | Recent dice and their total |
| Tabs | 0.622–0.838 | Battle log, game rules, objectives |
| Controls | 0.838–0.905 | Army and spellbook |
| End | 0.905–1.000 | End Phase button |

Spans run top to bottom and are fractions of aspect2d's fixed two-unit
height, so unlike the bar the ledger does not move with the aspect ratio.
`_layout` places section anchors and `_section_width` reports the space
across the short axis — the section's own slice when sections tile left to
right, the whole ledger width when they stack. Everything else is shared:
`_place`, `_span`, `_slot`, `_label`, the tooltip, and every setter.

Only the panels whose arrangement genuinely differs are built twice. The
regiment stats go two columns by four rather than nine across, and the phase
track becomes five lit rows rather than one text node, because neither fits
the other shape. `set_phase` and `set_dice` drive whichever exists.

**Switching rebuilds rather than re-flows.** The two layouts arrange their
panels differently enough that there is nothing to re-flow, so the HUD is
destroyed and remade. `snapshot`/`restore` carry what the player would
otherwise lose — the log, the phase, the turn, the dice — and `game.py`
re-publishes the selected unit.

`HUD.view_shift` reports which way the board has to move for the layout in
force, so the camera correction is the same code for both: the bar displaces
it upwards, the ledger leftwards, and collapsing the ledger gives the screen
back.

**The bar is drawn over the board, not carved out of it.** Shrinking the
camera's display region would be the tidier answer, but every pick site
extrudes raw mouse coordinates through `base.camLens`, which assumes the
region covers the whole window; shrinking it silently offsets every pick.
Nothing is lost by overlaying: DirectGui regions suppress mouse-button
events, so a click on the bar cannot also reach a unit behind it.

Overlaying does move the apparent centre, though: with the bottom `BAR_H`
covered, the middle of what you can see sits `BAR_H / 2` above the middle of
the window, and the board looked low. `MyApp.centreViewAboveHud` corrects it
with a lens film offset rather than by moving the camera — the offset lives
in the lens and `camLens.extrude` reads it back, so every pick site stays
aligned with what is on screen for free. Moving the camera would have needed
the same correction applied by hand at each of them.

**Textures must be opaque where they cover the board.** `ImageDraw` writes
RGBA verbatim instead of blending, so a translucent fill punches a hole
rather than tinting. Both the bar's wood grain and the round button's dome
hit this; they composite through `alpha_composite`, and `command_bar.png`
and `slot.png` finish with `putalpha(255)` so the board cannot show through.

**Keep what you act on in one place.** Bycer's criticism of Planetary
Annihilation is exactly this: build commands at the bottom, resources at the
top and unit orders on the right forces the player to split their attention.
The bar is the answer to it — the selected unit, what is happening to it, and
the button that ends the phase are all within one glance.

## Architecture rules

- **Build once, `setText` after.** No destroy-and-recreate on every update.
- **Publish, don't reach in.** Game systems post `messenger.send('hud-log',
  [text, category])`; they never import or touch a widget. The HUD is a
  subscriber. Events: `hud-turn`, `hud-phase`, `hud-log`, `hud-unit`.

  Rules modules post through `rules_log.battle_log` instead. They are imported
  by the tests without a ShowBase, so the Panda3D `messenger` builtin may not
  exist; `battle_log` is the one place that knows about it, and it no-ops when
  there is no game running.
- **Nothing decisive resolves off screen.** If an event changes the battle and
  the player would not otherwise see it, it goes in the log. This is the whole
  argument for the log, and the reason morale, casting, cannon and bombardment
  were wired into it rather than left printing to a terminal.
- **One source of truth per fact.** When the phase track went in, the nine
  `debugText.setText(f"Current phase: ...")` echoes came out. A duplicated
  readout is a readout that will eventually disagree with itself.
- **A display that throws must not stop a rule resolving.** `rules_log`
  catches and reports listener failures rather than letting them propagate.

## Built

- `hud.py` — `HUD` class: one bottom command bar holding every readout, `F3`
  to hide all chrome for screenshots. The four corner panels it replaced were
  each anchored to whichever corner they were written for, and the bar is the
  single owner of the bottom of the screen instead.

  Sections are declared once in `HUD.SECTIONS` as fractional spans and placed
  by `_layout()`. Children register through `_place` (a fraction across their
  section) or `_span` (a frame stretching between two fractions); `_layout`
  re-applies both on `window-event`, so the bar re-flows rather than being
  drawn for 16:9 and clipped everywhere else. `OnscreenText` overrides
  `setPos` with a flat `(x, z)` signature, which is why `_layout` branches on
  the node type.

  Textures are procedural, generated by `generate_tutorial_textures.py`:
  `command_bar.png` (oak backing), `slot.png` (an empty reserved frame) and
  `button_round.png` / `_hover`. Note that `ImageDraw` writes RGBA verbatim
  instead of blending, so a translucent fill punches a hole in the image
  rather than lightening it — the round button's dome is composited through
  `alpha_composite` for that reason.

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
- Morale, casting, cannon and bombardment post their outcomes too: the Panic
  test with its roll against Leadership, the casting roll against the casting
  value, Miscasts, and the war-machine damage summaries. Between those and the
  rule trace, the things that decide a battle now happen on screen.
- The four loose text nodes anchored to corners; the selected-unit readout no
  longer dumps the raw characteristics dict.
- Regiment page (bar, left): name, troop type, Unit Strength, formation, save,
  Ward, rank bonus; a stat row with above-average values green and
  below-average red; two detail lines — effective Leadership and where it
  comes from, and the equipped weapon; a models bar marked at 50%
  (flee vs fall back) and 25% (heavy-casualties Panic), whose fill turns amber
  then red as it crosses them; and state chips — fleeing, engaged, disrupted,
  charged, general, battle standard, moved, attacked.

  Stat columns are placed as fractions across the section and centred, which
  keeps the table aligned without needing a fixed-width face. With nothing
  selected the readouts are hidden but the parchment and heading stay, so the
  bar does not develop a hole in the middle of it.
  `game.showSelectedUnit` gathers the facts and publishes `hud-unit`; the page
  only lays them out.

  Colours on the pages are ink tints, not the bright on-black set the corner
  panels used — same hues, darkened until they read on parchment.

  **The page is not the hover tooltip.** The tooltip in `units.py` is the full
  dossier — every weapon, every special rule, both stat blocks. The page
  answers a narrower question: what can this unit do right now, and what is
  about to happen to it. Facts earn a place on it by deciding something this
  phase. Duplicating the dossier would cost the page the thing that makes it
  readable at a glance.

- Recent dice. `dice.checkDice` is the one point where a roll's faces become
  known whoever threw it, so it posts them through `rules_log.dice_roll`,
  which carries the same no-ShowBase guard as `battle_log`. Unused slots stay
  on screen dimmed, so the strip does not change width between a 2D6 and a
  5D6 roll.

- End Phase button. It posts `hud-end-phase` rather than holding a reference
  to the FSM; `game.py` binds that to `fsm.nextPhase`. It is labelled for what
  it does — the engine cycles phases and has no separate end-of-turn step.

  It replaced the floating `endPhase` collision cube: a camera-parented Bullet
  body with a `models/box` on it, which advanced the phase when a mouse ray
  hit it. That cost a per-frame task to flag its transform dirty, because
  Bullet does not re-read a body when an ancestor moves, and it answered the
  same mouse ray that picks units.


- Hover tooltip moved from world space to screen space. It was a billboarded
  `TextNode` parented to the unit, so its position was whatever the camera
  projection gave you — there was nothing to clamp against, and a unit low on
  the board put its dossier off the bottom of the screen.

  Now one shared `aspect2d` widget with a panel behind it. Its top-left corner
  is placed at the cursor, then shifted by the *least* amount that brings it
  back on screen, so the anchor stays as near the pointer as it can. Placed on
  hover-enter, not followed every frame: a tooltip that chases the cursor
  jitters, and re-running the fit each frame makes it jump as it crosses an
  edge. `units.py` keeps the `TextNode` purely as the string builder.

  Free consequences: no scaling with camera zoom, no depth-sort fighting, a
  unit's own models can no longer occlude its label, and the text is set in
  MedievalSharp rather than `cmtt12`, so `4+` renders as `4+` instead of `4−`.

## Next, in order

| Item | Effort | Payoff |
|---|---|---|
| In-world: selection decal, facing chevron, status badges, strength bar | Low–Med | High |
| `legal_actions(unit)` + action bar **in the regiment section** | Med | High |
| Every action shows its key, plus a key-list overlay | Low | High |
| Army strength readout (the macro glance) — the End Phase button is in | Low | High |
| Fill the empty art slots: portrait, army list, spellbook, rules, objectives | Med | Medium |
| Target preview: to-hit, to-wound, save, modifier stack, expected casualties | Med | Very high |
| Combat result card | Low | Medium |
| Log scrollback and category filtering | Low | Medium |
| Second font — **blocked**, see below | Low | High |

The two genuinely hard ones:

- **Target preview** needs an expected-value path that provably cannot drift
  from the dice path. Compute it from the same `toHitAndToWound` helpers;
  never re-derive the numbers in the UI.
- **Action bar** needs legality extracted out of the task loops.
  `taskLoopPathTowardsMouse` currently prints "Unit is not idle" and returns —
  the reason a unit cannot act is computed but not queryable. Extract
  `legal_actions(unit)` and have both the task loops and the bar call it.

## Look and feel

- **Every action shows its key, and every key has a visible action.** Hotkeys
  are for speed; the visible control is how the player finds out the hotkey
  exists. Today `t`, `c`, `l`, `a`, `F3`–`F10`, right-click-to-commit and the
  End Phase button are the entire command set, and only the last of them is
  discoverable. Put the key on the button (`SHOOT (S)`) and add an
  always-available key list — `debug_tools` has one on `h`, but only in debug
  mode.

  Bycer's hotkey list — control groups, attack-move, select-all-of-type, idle
  workers — does not apply. That is base-management RTS under APM pressure;
  this is turn-based with six units a side, so discoverability matters far
  more than speed.

- **Two fonts.** MedievalSharp for headings and unit names only; a clean
  humanist sans for stat lines, numbers and the log. Display faces at 0.03
  scale are illegible, and a stat block is tabular data.

  **This is blocked on an asset, not on effort.** `fonts/` holds only
  MedievalSharp, and Panda3D's bundled `cmss12` / `cmtt12` are not a
  substitute: they are OT1-encoded Computer Modern, so `+` renders as a minus
  sign and `>` as an inverted question mark. In a wargame UI where `4+` is the
  most common string on screen, that is disqualifying. They are also
  StaticTextFont, pre-rendered and unfilterable, so they blur when scaled.

  To unblock: drop an OFL sans TTF (Inter, Lato, Open Sans) into `fonts/`, add
  `BODY_FONT_PATH` beside `FONT_PATH` in `gui_theme`, and a `get_body_font()`
  next to `get_font()`. Everything else is already parameterised —
  `styled_text` and `setup_text_node` both take a `font`.

- **Mind the glyph set.** MedievalSharp has no `→` (U+2192); an arrow in a log
  line renders as nothing at all, silently. Stick to `->`, `v` and `-`.
- **A type scale, not arbitrary values.** 0.075 title, 0.05 heading, 0.038
  body, 0.03 caption.
- **Semantic colour.** Gold means important, red means bad for you, green means
  good for you, and nothing else may use them. Parchment and ink for
  everything neutral.
- **Panels float, the board does not.** Translucent, edge-anchored, never over
  the centre of the table.
- **The board is a board.** It is finite, and its limit is where the rules stop
  applying, so it has a drawn wooden edge and sits on a dark tabletop rather
  than fading into open ground. Extending the grass outwards would read as a
  continuous world and leave the out-of-bounds line looking arbitrary.

  The edge is restrained — 2.4 units, 3.3% of the board width — with a brass
  fillet and corner studs to tie it to the book-themed bar. The table is
  darker and flatter than either the board or the HUD, and the dressing at its
  extreme edges is sited to be cropped by the frame. Reading order is
  miniatures, board, HUD, table dressing, and nothing in the last two may
  compete with the first.

  `MyApp.setup_table` builds it. The grass is clipped to the board in
  `c1.frag` rather than by resizing the ground card: the card is also the
  coordinate space the movement and shooting overlay polygons live in, where
  world ±50 maps to 0..1, so resizing it would silently move every arc.
- **Keep the physical dice.** Tumbling d6s are the most characterful thing in
  the build. Add a 2D strip beside them showing values against the target
  number so the *result* reads while the *rolling* stays tactile.
- **Graduated detail, not everything at once.** Bycer contrasts Rise of Nations
  (show it all) with Age of Empires 3 (degrees of detail, to avoid overload).
  That is the argument for the two-stage tooltip: a short card on hover, the
  full dossier on demand.

Source for several of the above: Josh Bycer, *UI Strategy Game Design Dos and
Don'ts*, Game Developer, 2015.

## LEFTOVER

- No second font — attempted with Panda3D's bundled faces and reverted; see
  the reason under Look and feel. Needs a sans TTF added to `fonts/`.
  MedievalSharp is still setting the stat lines, and it remains the weakest
  part of the screen.
- Battle log has no scrollback; it is a fixed 6-entry window with no filter.
  The bar is slimmer than the old corner panel, so it lost two lines.
- **The art slots are empty.** The portrait, the two rail slots (army list,
  spellbook), the game-rules slot and the objectives slot are framed
  placeholders with corner ticks, not features. Nothing is behind them: the
  rail and rules slots have no click handler, and there is no objectives
  system in the engine for that panel to read.
- Detail lines dropped from five to two to fit the bar. The mount profile,
  casting attempts left, and who a unit is fighting and from which arc are no
  longer on the page — they are still in the hover dossier.
- The bar overlays the board rather than shrinking the display region, so the
  board still draws behind it. Carving out the region needs every pick site
  to clip mouse coordinates to it first; see Screen zones.
- Only the last roll shows in the dice strip, and it is capped at five dice.
  A 10D6 shooting volley shows its last five faces with no indication that
  more were thrown.
- The unit card refreshes on selection, on `applyWounds` and on every phase
  change, which covers casualties and flag resets. It does *not* refresh when
  a unit is disrupted by terrain, joined by a character, or has its Ward
  granted mid-turn by a spell — those change the card without touching any of
  the three triggers.
- Detail lines are truncated at 54 characters rather than wrapped, because the
  card lays them out on a fixed step. A unit with a long name fighting two
  others loses the tail of the "Fighting" line.
- Not on the card, and deliberately: points value, the full special-rules list,
  the full weapon list, and armour pieces. They belong to the hover dossier.
- The tooltip's stat table now aligns on real tab stops (`TextNode.setTabWidth`),
  set by `HUD.TIP_TAB_WIDTH` and emitted as `\t` by `units._stat_table`. Space
  padding could never work: the display face is proportional, so a space is
  narrower than a digit. The tab stop is sized for the widest cell a stat can
  produce, `10+`; a stat wider than that would push its column to the next stop.
- The tooltip can cover the unit card when hovering a unit low on the left.
  Acceptable while it is transient and drawn on top, but it argues for the
  two-stage split: a short card on hover, the dossier in the free top-right
  zone. That split is still not done — hover shows the whole dossier, so on a
  tall unit entry the panel is still most of the screen height.
- Nothing posts to the log from the pursuit, break-test or Fear/Terror paths in
  `psychology.py` — only the Panic test does. A broken unit routing off the
  board is still console-only.
- Dice results still go to `diceInfoText` as plain text, not the roll strip.
- `debugText`, `debugTextInfo`, `debugTextUnit` keep their debug-era names
  despite carrying real gameplay information.
