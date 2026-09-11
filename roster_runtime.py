"""Apply selected equipment to its profile, not every part of a base (pp. 192-195)."""

from copy import copy, deepcopy

from battlescribe import slugify, weapon_from_profile
from command_groups import install_command


def champion_profile(group, entry):
    """A champion replaces one ordinary profile (Rulebook p. 199)."""
    profile = copy(group.model)
    profile.characteristics = deepcopy(group.model.characteristics)
    profile._base_characteristics = deepcopy(group.model._base_characteristics)
    profile.weapons = deepcopy(group.model.weapons)
    profile.special_rules = [rule for rule in group.model.special_rules
                             if rule is not group.model.equipedWeapon]
    profile.equipedWeapon = None
    for record in entry.get('profiles', []):
        if record.get('typeName') == 'Model':
            values = {value['name']: value.get('$text', '') for value in record.get('characteristics', [])}
            profile.characteristics.update(values)
            profile._base_characteristics.update(values)
            profile.name = record.get('name', entry.get('name', 'Champion'))
    profile.equip_best_melee()
    return profile


def apply_roster_ownership(group, data):
    """Use selected structural owners; chariot weapons are crew-operated (p. 194)."""
    install_command(group, data.get('command', []))
    group.roster_metadata = {key: deepcopy(data[key]) for key in
                             ('points_cost', 'roster_selections', 'equipment', 'magic_items', 'spell_sources',
                              'spell_pool', 'spell_generation_pending', 'roster_source') if key in data}
    group.command_models = {entry.get('selection_ref', str(index)): champion_profile(group, entry)
                            for index, entry in enumerate(group.command) if entry.get('role') == 'champion'}
    profiles = [group.model] + [part for tag in ('mount', 'crew', 'beasts')
                                if (part := getattr(group.model, f'get_{tag}')()) is not None]
    by_name = {slugify(profile.name): profile for profile in profiles}
    owners = {record['ref']: record for record in data.get('roster_selections', [])}
    assigned = {}
    for equipment in data.get('equipment', []):
        if equipment.get('category') != 'Weapon':
            continue
        owner = owners.get(equipment.get('owner_ref'), {})
        profile = by_name.get(slugify(owner.get('name', '')))
        if profile is None:
            profile = group.command_models.get(equipment.get('owner_ref'))
        if profile is None:
            continue
        if profile is group.model and profile.is_chariot() and profile.get_crew() is not None:
            profile = profile.get_crew()
        record = equipment['profile']
        values = {value['name']: value.get('$text', '') for value in record.get('characteristics', [])}
        weapon = weapon_from_profile(record['name'], values)
        assigned.setdefault(id(profile), (profile, []))[1].append(weapon)
    for profile, weapons in assigned.values():
        profile.special_rules = [rule for rule in profile.special_rules if rule is not profile.equipedWeapon]
        profile.equipedWeapon = None
        profile.weapons = {}
        for weapon in weapons:
            name = weapon['name']
            if profile.give_weapon(name):
                continue
            profile.weapons[name] = weapon
        profile.equip_best_melee()
    for index, entry in enumerate(group.command):
        if entry.get('role') == 'champion' and entry.get('selection_ref') not in {
                equipment.get('owner_ref') for equipment in data.get('equipment', [])}:
            group.command_models[entry.get('selection_ref', str(index))] = champion_profile(group, entry)
    crew = group.model.get_crew()
    if group.model.is_chariot() and crew is not None and data.get('equipment'):
        group.model.special_rules = [rule for rule in group.model.special_rules
                                     if rule is not group.model.equipedWeapon]
        group.model.weapons = deepcopy(crew.weapons)
        group.model.equipedWeapon = None
        group.model.equip_best_melee()