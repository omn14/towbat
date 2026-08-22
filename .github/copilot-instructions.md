# Copilot instructions — Warhammer: The Old World battle engine

## Environment

Every terminal command starts with `source .venv/bin/activate`. The default
`python` on PATH has no Panda3D.

- Tests: `python -m pytest tests -q`
- `python game.py` blocks on window creation. To check anything visual, render
  offscreen instead: `loadPrcFileData("", "window-type offscreen ...")`, build
  the scene, `base.graphicsEngine.renderFrame()`, `base.screenshot(...)`.

Never stage `quicksave.json`, `quicksave.json.bak`, `my_army.json` or
`strategy_armies/nr/`. They change every time the game is played.

## Rules work

Check the wording at <https://tow.whfb.app/> before coding a rule, and cite the
rulebook page in the code comment or docstring. The Official FAQ & Errata
overrides the printed book.

The BattleScribe catalogues carry rule *keywords* only, never their effects.
Coded effects live in `special_rules.py`, `battleFunctions.py`,
`combat_resolution.py`, `psychology.py`, `troop_types.py` and `models.py`.

Track every rule in `SPECIAL_RULES_CHECKLIST.md`: what was done, what was
corrected on the way, and an explicit `LEFTOVER:` for what was not. A rule that
is half-implemented and not written down is worse than one that is missing.

## Every rule that fires must say so

**When a special rule changes an outcome, log it.** Use `rules_log.rule_log`,
and `rules_log.rule_skipped` when a rule could have applied but did not:

```python
from rules_log import rule_log, rule_skipped

rule_log('Swiftstride', unit, "adds +3\" to the maximum charge range (12\" -> 15\")")
rule_skipped('Vantage Point', unit, "only 4/10 models are on the hill")
```

A rule in this engine has no visible effect on screen — a Ward save that works
and a Ward save that was never coded look identical — so the console is the
only way to tell them apart, and it is what a bug report gets written from.

The line has to earn its place. Carry the numbers that decided it and what they
changed, so it answers "why did that happen?" without a re-run:

- Good: `Impact Hits (D6+1) — War Wagon: charged 7", 3 models in contact -> 11 hits at S5 AP-2`
- Useless: `Impact Hits triggered`

Log where a rule *changes something*, not where it is queried. `unit_strength()`
and `max_rank_bonus()` are read constantly; logging those floods the console and
buries the lines that matter. Nothing inside a per-attack dice loop should log
either — report the total once the rolls are resolved.

The negative case matters as much as the positive one. A rule that quietly
declines looks exactly like a rule that is broken; two separate debugging
sessions on this project went that way.

## Style

Comments say what the code cannot: why a rule works the way it does, what a
magic number came from, what a workaround is for. Not what the next line does.
