"""Choose weapons before attacks and challenges (amended Rulebook pp. 213-215)."""

from panda3d.core import Vec3

from command_groups import champions
from rules_log import rule_log, rule_skipped


def available_weapons(profile, charged):
    weapons = {name: weapon for name, weapon in profile.weapons.items()
               if weapon and weapon.get('tag') != 'ranged'
               and (charged or not weapon.get('charge_only'))}
    magical = {name: weapon for name, weapon in weapons.items()
               if weapon.get('magic_item') or weapon.get('magical')}
    return magical or weapons


async def choose_profile(game, owner, profile, charged):
    profile.charging = charged
    weapons = available_weapons(profile, charged)
    if not weapons:
        return None
    current = profile.weapon_slot((profile.equipedWeapon or {}).get('name', ''))
    if current not in weapons:
        current = profile.equip_best_melee()
    if current not in weapons:
        current = next(iter(weapons))
    labels = {('Hand weapon & shield' if profile.is_shieldwall() and profile.has_shield()
               and name.casefold() == 'hand weapon' else name): name for name in weapons}
    if len(weapons) > 1:
        if game.aiControls(owner):
            preferred = profile.equip_best_melee()
            current = preferred if preferred in weapons else current
            if profile.is_shieldwall() and profile.has_shield() and getattr(owner, 'wasChargedThisTurn', False):
                current = labels.get('Hand weapon & shield', current)
        else:
            selected = await game.makeChoiceNew(list(labels), Vec3(0, 0, 10), owner=owner,
                                               prompt=f'{profile.name}: combat weapon')
            current = labels.get(selected, current)
    profile.equip_weapon(current)
    rule_log('Combat Weapons', owner, f'{profile.name} chooses {current} for this combat (p. 213)')
    if profile.has_shield() and profile.melee_weapon_requires_two_hands():
        rule_skipped('Shield', owner, f'{profile.name} uses {current}, which Requires Two Hands; '
                     f'melee armour {profile.effective_armour_save()}+ -> {profile.melee_armour_save()}+')
    return current


async def choose_unit_weapons(game, host, chosen, *, charged=None):
    charged = bool(getattr(host, 'chargedThisTurn', False)) if charged is None else charged
    profile = host.unit.model
    selected = await choose_profile(game, host, profile, charged)
    chosen.add(id(profile))
    for champion in champions(host):
        promoted = champion.unit.model
        if set(available_weapons(promoted, charged)) == set(available_weapons(profile, charged)):
            promoted.charging = charged
            promoted.equip_weapon(selected)
        else:
            await choose_profile(game, host, promoted, charged)
        chosen.add(id(promoted))
    for tag in ('mount', 'crew', 'beasts'):
        part = getattr(profile, f'get_{tag}')()
        if part is not None:
            await choose_profile(game, host, part, charged)
            chosen.add(id(part))
    joined = getattr(host, 'joinedCharacter', None)
    if joined is not None and not getattr(joined, 'retiredFromCombat', False):
        await choose_unit_weapons(game, joined, chosen, charged=charged)