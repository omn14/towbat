"""Per-profile attacks on a shared base (Rulebook pp. 146, 192, 194, 199)."""

from copy import copy
from dataclasses import dataclass
from types import SimpleNamespace

from battleFunctions import attack_characteristic, melee_attacks, strike_initiative
from command_groups import living_command, command_positions
from characters import get_joined_character


@dataclass
class CombatProfile:
    host: object
    target: object
    profile: object
    role: str
    count: int = 1
    entry: object = None
    fighter: object = None

    def attacks(self, models, initial, challenge=None):
        """Use the start of this Initiative step, not the start of combat (p. 146)."""
        group = copy(self.host.unit)
        group.nmodels = max(0, models)
        charged = bool(getattr(self.host, 'chargedThisTurn', False))
        fallen = max(0, initial - models)
        positions = command_positions(group)
        commands = [entry for entry in living_command(group)
                if positions[id(entry)] < group.files]
        champions = [entry for entry in commands if entry.get('role') == 'champion'
                 and not entry.get('retired', False)]
        joined = get_joined_character(self.host)
        if self.role == 'character':
            return melee_attacks(self.fighter.unit, charged) if self.fighter.unit.nmodels > 0 else 0
        if self.role == 'champion':
            return attack_characteristic(self.profile) if self.entry in champions else 0
        blocked = len(champions)
        if joined is not None and not getattr(joined, 'retiredFromCombat', False) and (
            getattr(self.host, 'characterSlot', 0) or 0) < group.files:
            group.files = max(0, group.files - 1)
        if self.role == 'main':
            ordinary = melee_attacks(group, charged, fallen)
            return max(0, ordinary - blocked * attack_characteristic(group.model))
        behind = max(0, models + fallen - group.files)
        fighting = max(0, min(group.files, models) - min(fallen, behind))
        if challenge is not None and self.host in challenge.hosts():
            for participant in challenge.participants():
                if getattr(participant, 'command_host', None) is self.host:
                    fighting = max(0, fighting - 1)
        return fighting * self.count * attack_characteristic(self.profile)

    def unit(self, attacks):
        return SimpleNamespace(name=self.profile.name, model=self.profile,
                               nmodels=1, files=1, ranks=1, _attack_count=attacks)


def combat_profiles(host, target, challenge=None):
    profile = host.unit.model
    if challenge is not None and challenge.involves(host):
        return []
    parts = []
    if not profile.is_chariot():
        parts.append(CombatProfile(host, target, profile, 'main'))
    mount = profile.get_mount()
    if mount is not None:
        parts.append(CombatProfile(host, target, mount, 'mount'))
    for tag in ('crew', 'beasts'):
        part = getattr(profile, f'get_{tag}')()
        if part is not None:
            parts.append(CombatProfile(host, target, part, tag, profile.part_count(tag)))
    for index, entry in enumerate(getattr(host.unit, 'command', [])):
        key = entry.get('selection_ref', str(index))
        champion = getattr(host.unit, 'command_models', {}).get(key)
        duelling = challenge is not None and any(
            getattr(participant, 'command_entry', None) is entry for participant in challenge.participants())
        if champion is not None and entry.get('active', True) and not entry.get('retired', False) and not duelling:
            parts.append(CombatProfile(host, target, champion, 'champion', entry=entry))
    joined = get_joined_character(host)
    if joined is not None and not getattr(joined, 'retiredFromCombat', False) and not (
            challenge is not None and challenge.involves(joined)) and (
            getattr(host, 'characterSlot', 0) or 0) < host.unit.files:
        parts.append(CombatProfile(host, target, joined.unit.model, 'character', fighter=joined))
        for tag in ('mount', 'crew', 'beasts'):
            part = getattr(joined.unit.model, f'get_{tag}')()
            if part is not None:
                count = 1 if tag == 'mount' else joined.unit.model.part_count(tag)
                fighting = SimpleNamespace(unit=SimpleNamespace(name=part.name, model=part,
                                           nmodels=count, files=count, ranks=1))
                parts.append(CombatProfile(host, target, part, 'character', fighter=fighting))
    return parts


def profile_strike_order(attackers, defenders, facing, challenge=None):
    """Each weapon's Initiative modifiers belong to its wielder (pp. 192-194)."""
    seen = set()
    order = []
    for host, target in zip(attackers, defenders):
        if id(host) in seen or host.hasAttackedThisTurn:
            continue
        seen.add(id(host))
        for part in combat_profiles(host, target, challenge):
            if not part.profile.equipedWeapon or part.profile.equipedWeapon.get('tag') == 'ranged':
                part.profile.equip_best_melee()
            initiative = strike_initiative(
                part.profile, charged=bool(getattr(host, 'chargedThisTurn', False)),
                inches=float(getattr(host, 'chargeDistance', 0) or 0),
                flank_or_rear=facing(target, host) in ('flank', 'rear'),
                first_round=getattr(host, 'roundsFought', 0) == 1, log=True)
            order.append((initiative, part))
    return sorted(order, key=lambda entry: -entry[0])


def crew_shooting_unit(group, firing_models=None):
    """Each armed crew member shoots with its own profile (Rulebook p. 194)."""
    crew = group.model.get_crew() if group.model.is_chariot() else None
    weapon = group.model.equipedWeapon or {}
    if crew is None or crew.weapon_slot(weapon.get('name', '')) is None:
        return group, firing_models
    crew.equip_weapon(weapon['name'])
    for flag in ('at_long_range', 'target_skirmisher', 'moved_this_turn'):
        setattr(crew, flag, getattr(group.model, flag, False))
    count = group.model.part_count('crew')
    shooters = SimpleNamespace(name=crew.name, model=crew, nmodels=group.nmodels * count,
                               files=group.files * count, ranks=group.ranks)
    return shooters, None if firing_models is None else firing_models * count