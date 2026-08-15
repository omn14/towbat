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
- [ ] Stubborn
- [ ] Immune to Psychology
- [ ] Fear
- [ ] Terror

## Loose ends
- [ ] Test/CI hardening; broaden `tests/` to a couple of full factions
- [ ] Empire units render with the generic model (no `.bam`) — add mappings
- [ ] Non-numeric stats (e.g. Giant A="*") still break `int()` in some call sites

## Deferred war-machine items
- [ ] Multiple Wounds (D3+1) generic rule
- [ ] Black Powder / Misfire tables
- [ ] True line-of-sight and rank/file hit caps for cannon/bombardment
