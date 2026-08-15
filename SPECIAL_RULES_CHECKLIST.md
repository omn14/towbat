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

## Weapon rules — TODO (high frequency)
- [ ] Multiple Shots (D3) — roll D3/D6+N instead of the hardcoded 3
- [ ] Move or Shoot — cannot shoot after moving
- [ ] Ponderous — move-or-shoot / initiative penalty
- [ ] Killing Blow — natural 6 to wound = no armour save (auto-kill)
- [ ] Heroic Killing Blow
- [ ] Strike First
- [ ] Strike Last
- [ ] Requires Two Hands — disables shield/parry bonus

## Unit rules — TODO (displayed, not yet applied)
- [ ] Impetuous
- [ ] Frenzy
- [ ] Immune to Psychology
- [ ] Fear
- [ ] Terror
- [ ] Stubborn — see General rules below

## General (core `.gst`) rules — TODO
From scanning `strategy_armies/nr/dv.json` (Dwarf army) against the engine:
of 35 distinct rules only **Armour Bane** is coded. These core rules are
army-agnostic and would benefit every faction.

- [ ] Stubborn — break tests on unmodified Ld (AI classifier reads it, but the
      mechanic isn't applied)
- [ ] Fly (X) — flying movement (AI reads `is_flying` only)
- [ ] General — Inspiring Presence (Ld bubble)
- [ ] Battle Standard Bearer — re-roll failed break tests near the BSB
- [ ] Hatred (X) — re-roll misses to hit in the first combat round
- [ ] Magic Resistance (-1/-2) — to-cast / ward penalty vs magic
- [ ] Impact Hits (D3) — auto-hits on the charge
- [ ] Skirmishers — loose formation, 360° LoS, no rank bonus
- [ ] Scouts / Vanguard — pre-game deployment / free move
- [ ] Swiftstride — better charge / flee / pursuit distance
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
- [ ] Venerable — friendly units within 6" re-roll failed Panic tests
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
- [ ] Non-numeric stats (e.g. Giant A="*") still break `int()` in some call sites

## Deferred war-machine items
- [ ] Multiple Wounds (D3+1) generic rule
- [ ] Black Powder / Misfire tables
- [ ] True line-of-sight and rank/file hit caps for cannon/bombardment
