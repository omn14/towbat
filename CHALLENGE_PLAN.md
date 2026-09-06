# Challenges — implementation plan

Rulebook p. 210–211, the Characters chapter. Eight pages, all read at
<https://tow.whfb.app/characters/challenges> and its children.

## The rules, condensed

| Page | Rule |
| --- | --- |
| p. 210 | **Issuing.** At Step 1.1 when a combat is chosen. **One challenge per combat.** The active player is offered first; if they decline, the inactive player may issue. The challenger must be a character or champion **within, or adjacent to, the fighting rank**. No eligible model, no challenge. |
| p. 210 | **Accepting.** The opposing player nominates an eligible character or champion. If the enemy unit holds none, the challenge **goes unanswered**. |
| p. 210 | **Refusing.** The issuer nominates one model that *could* have accepted. It **retires**: moves out of the fighting rank, makes no attacks and has none directed at it, and **confers no benefits — Leadership, special rules, anything else**. It cannot return while its unit is still engaged. |
| p. 211 | **Nowhere to Run.** Cannot refuse if the model is not part of a unit, is the last model in its unit, or its unit is engaged **in all four arcs**. |
| p. 211 | **Fighting.** Both direct **all** their attacks at one another, **in Initiative order**. No other model in the combat may direct attacks at either duellist. Moving the pair into base contact is optional. |
| p. 211 | **Overkill.** If the winner causes more unsaved wounds than the loser had Wounds remaining, each excess is **+1 combat result, to a maximum of +5**. Explicitly an exception: normally only Wounds *lost* count. |
| p. 211 | **To The Death!** If both survive and the combat continues into the next player's turn, so does the challenge. **No further challenge in that combat until it resolves.** |
| p. 211 | **Challenges & Mounts.** A mount — including a chariot's crew — must direct its attacks at the other participant. **If a participant is slain before their rival or a mount has attacked, those attacks are lost.** |

One wording gap: the Refusing page truncates on fetch at "…cannot direct
attacks against the model that issued the challenge – they are far too occupied
with their cowardice!". Confirm the full paragraph against the printed book
before relying on that clause.

## What the engine had to gain first

**A joined character could not be wounded.** `applyWounds` was only ever called
on the host unitGraphics, so a character dealt attacks but nothing could ever
direct wounds at it; it died only when its host was destroyed. A duel in which
neither duellist can fall is not a duel, so model-level wound allocation is a
prerequisite rather than part of the challenge itself.

**Champions do not exist.** Searching the engine and the catalogue loader turns
up only the Battle Standard Bearer, and that is a keyword on a unit, not a
model. So "character or champion" is read here as "joined character" alone.
This is the largest deliberate gap — unit champions are the commoner case at
the table.

## Why a challenge can be resolved on its own

p. 211 says the duellists direct **all** attacks at each other and that no other
model may attack either of them. A challenge is therefore *sealed off* from the
combat around it: nothing it does can change another model's attacks, and
nothing another model does can change its own. Its Initiative ordering only has
to be internally consistent, so it is resolved as a self-contained ordered pass
rather than being woven into the unit-level strike order. That is what keeps
this from requiring a rewrite of `_verySimpleBattleInner`.

## Shape

- `challenges.py` — the rules with no Panda3D in them: eligibility, Nowhere to
  Run, Overkill arithmetic, and the `Challenge` record. Testable directly.
- `combat_resolution.py` — orchestration: the issue/accept/refuse exchange, the
  duel, and the Overkill row in the combat result.
- `units.py` — placing a retired model behind the ranks.
- `psychology.py` — a retired model confers no Leadership.
- `persistence.py` — the challenge survives a save, for To The Death.

## Order of work

1. Wound a joined character. Standalone and useful on its own.
2. `Challenge` state, eligibility and Nowhere to Run.
3. The issue / accept / refuse exchange at Step 1.1.
4. Retiring, visually and in its effects.
5. The duel, in Initiative order, mounts included, attacks lost on death.
6. Overkill as a combat result row.
7. To The Death across turns, and persistence.
8. Tests and the checklist.

## Known simplifications

- **All four arcs.** The engine records only `front`, `flank` and `rear` per
  engagement — it cannot tell a left flank from a right one. "Surrounded" is
  approximated as engaged to the front, the rear and in two flanks.
- **Within or adjacent to the fighting rank.** A joined character always stands
  in the front rank at `host.characterSlot`, so it is always eligible; the
  proximity test is never actually made.
- **AI.** Never issues, always accepts. A real heuristic would weigh the
  profiles and what a refusal would cost.
