"""Explicit item definitions and purchased instances; never execute roster prose.

Magic item categories and single-use rules: Rulebook pp. 337-340. Individual
definitions must cite their own rule source before any effects are enabled.
"""

from copy import deepcopy
from dataclasses import dataclass, field
from enum import StrEnum
import json
import textwrap
from types import SimpleNamespace
import weakref

from battlescribe import slugify
from rules_log import rule_log, rule_skipped


class EffectKind(StrEnum):
    ARMOUR = 'armour_modifier'
    RULE = 'rule_grant'
    SPELLS = 'extra_known_spells'
    REROLL = 'reroll'
    FIRST_TURN = 'first_turn_modifier'
    WARD = 'ward_save'
    BODY_ARMOUR = 'body_armour'
    AP_ZERO_ARMOUR = 'ap_zero_armour'
    MAGIC_ROLL = 'magic_roll_modifier'
    COMBAT_RESULT = 'combat_result'
    WEAPON = 'weapon_profile'
    INITIATIVE = 'initiative_modifier'
    TARGET_PROTECTION = 'target_protection'
    SHOOTING_SIGHT = 'shooting_sight'
    IGNORE_COVER = 'ignore_cover'
    START_RULE = 'start_turn_rule'


class Scope(StrEnum):
    BEARER = 'bearer'
    UNIT = 'unit'
    UNIT_AND_JOINED = 'unit_and_joined'
    BEARER_AND_UNIT = 'bearer_and_unit'


@dataclass(frozen=True)
class ItemEffect:
    key: str
    kind: EffectKind
    value: int | str
    scope: Scope = Scope.BEARER
    ability: str | None = None


@dataclass(frozen=True)
class ItemAbility:
    key: str
    context: str
    limit: int = 1
    per_turn: bool = False


@dataclass(frozen=True)
class ItemDefinition:
    key: str
    name: str
    category: str
    reference: str
    effects: tuple[ItemEffect, ...] = ()
    abilities: tuple[ItemAbility, ...] = ()
    aliases: tuple[str, ...] = ()
    catalogue_ids: tuple[str, ...] = ()

    @property
    def supported(self):
        return bool(self.effects)


class ItemRegistry:
    def __init__(self, definitions=()):
        self.definitions = {}
        self.names = {}
        self.catalogue_ids = {}
        for definition in definitions:
            self.register(definition)

    def register(self, definition):
        if not definition.reference:
            raise ValueError('An item definition needs a rules reference')
        keys = [effect.key for effect in definition.effects]
        abilities = [ability.key for ability in definition.abilities]
        if len(set(keys)) != len(keys) or len(set(abilities)) != len(abilities):
            raise ValueError('Duplicate effect or ability key')
        if any(effect.ability and effect.ability not in abilities for effect in definition.effects):
            raise ValueError('Effect refers to an undefined ability')
        if any(ability.limit < 1 for ability in definition.abilities):
            raise ValueError('Ability use limit must be positive')
        names = {slugify(name) for name in (definition.name, *definition.aliases)}
        if (definition.key in self.definitions or names.intersection(self.names)
                or set(definition.catalogue_ids).intersection(self.catalogue_ids)):
            raise ValueError('Ambiguous item definition, name or catalogue ID')
        self.definitions[definition.key] = definition
        self.names.update({name: definition for name in names})
        self.catalogue_ids.update({identity: definition for identity in definition.catalogue_ids})

    def resolve(self, source):
        definition = (self.catalogue_ids.get(source.get('definition_id'))
                      or self.names.get(slugify(source.get('name', ''))))
        if definition is None or slugify(source.get('category', '')) != slugify(definition.category):
            return None
        return definition


REGISTRY = ItemRegistry((
    ItemDefinition('silvery_wand', 'Silvery Wand', 'Arcane Items', 'Forces of Fantasy p. 183',
                   effects=(ItemEffect('extra_spell', EffectKind.SPELLS, 1),)),
    ItemDefinition('helm_of_courage', 'Helm of Courage', 'Magic Armour',
                   "Battle March: General's Companion p. 47",
                   effects=(ItemEffect('armour', EffectKind.ARMOUR, 1),
                            ItemEffect('break_reroll', EffectKind.REROLL, 'Break',
                                       Scope.BEARER_AND_UNIT, 'courage')),
                   abilities=(ItemAbility('courage', 'Break'),)),
    ItemDefinition('banner_of_the_bold', 'The Banner of the Bold', 'Magic Standards',
                   "Battle March: General's Companion p. 47; FAQ v1.5.3 Characters",
                   effects=(ItemEffect('veteran', EffectKind.RULE, 'Veteran', Scope.UNIT_AND_JOINED),),
                   aliases=('Banner of the Bold',)),
    ItemDefinition('rangers_glass', "The Ranger's Glass", 'Enchanted Items',
                   "Battle March: General's Companion p. 48",
                   effects=(ItemEffect('first_turn', EffectKind.FIRST_TURN, 1),),
                   aliases=("Ranger's Glass",)),
    ItemDefinition('warding_talisman', 'The Warding Talisman', 'Talismans',
                   "Battle March: General's Companion p. 47",
                   effects=(ItemEffect('ward', EffectKind.WARD, 6),),
                   aliases=('Warding Talisman',)),
    ItemDefinition('padded_hauberk', 'Padded Hauberk', 'Magic Armour',
                   "Battle March: General's Companion p. 47",
                   effects=(ItemEffect('heavy_armour', EffectKind.BODY_ARMOUR, 5),
                            ItemEffect('padding', EffectKind.AP_ZERO_ARMOUR, 1))),
    ItemDefinition('wyrdstone_shard', 'Wyrdstone Shard', 'Arcane Items',
                   "Battle March: General's Companion p. 48",
                   effects=(ItemEffect('magic_roll', EffectKind.MAGIC_ROLL, 1, ability='shard'),),
                   abilities=(ItemAbility('shard', 'Magic roll'),)),
    ItemDefinition('banner_of_renown', 'Banner of Renown', 'Magic Standards',
                   "Battle March: General's Companion p. 47",
                   effects=(ItemEffect('combat_result', EffectKind.COMBAT_RESULT, 1, Scope.UNIT, 'renown'),),
                   abilities=(ItemAbility('renown', 'Combat result'),)),
    ItemDefinition('diestros_blade', "Diestro's Blade", 'Magic Weapons',
                   "Battle March: General's Companion p. 46",
                   effects=(ItemEffect('blade', EffectKind.WEAPON, 'diestros_blade'),
                            ItemEffect('quickness', EffectKind.INITIATIVE, 1))),
    ItemDefinition('skirmishers_blade', "Skirmisher's Blade", 'Magic Weapons',
                   "Battle March: General's Companion p. 46",
                   effects=(ItemEffect('blade', EffectKind.WEAPON, 'skirmishers_blade'),)),
    ItemDefinition('thornspitter_stave', 'Thornspitter Stave', 'Magic Weapons',
                   "Battle March: General's Companion p. 46",
                   effects=(ItemEffect('combat', EffectKind.WEAPON, 'thornspitter_combat'),
                            ItemEffect('ranged', EffectKind.WEAPON, 'thornspitter_ranged'))),
    ItemDefinition('shadowed_mantle', 'Shadowed Mantle', 'Magic Armour',
                   "Battle March: General's Companion p. 47; Rulebook p. 206",
                   effects=(ItemEffect('light_armour', EffectKind.BODY_ARMOUR, 6),
                            ItemEffect('concealment', EffectKind.TARGET_PROTECTION, 6))),
    ItemDefinition('all_seeing_eye_of_numas', 'The All-Seeing Eye of Numas', 'Enchanted Items',
                   "Battle March: General's Companion p. 48",
                   effects=(ItemEffect('woods', EffectKind.SHOOTING_SIGHT, 'forest'),
                            ItemEffect('cover', EffectKind.IGNORE_COVER, True)),
                   aliases=('All-Seeing Eye of Numas',)),
    ItemDefinition('trailblazers_hatchet', "Trailblazer's Hatchet", 'Magic Weapons',
                   "Battle March: General's Companion p. 46",
                   effects=(ItemEffect('weapon', EffectKind.WEAPON, 'trailblazers_hatchet'),
                            ItemEffect('cover', EffectKind.RULE, 'Move Through Cover'),
                            ItemEffect('path', EffectKind.START_RULE, 'Move Through Cover', ability='path')),
                   abilities=(ItemAbility('path', 'Start of Turn'),)),
))


ITEM_WEAPONS = {
    'diestros_blade': {'name': "Diestro's Blade", 'tag': 'combat', 'strength_bonus': 0,
                      'ap_penetration': 1, 'magical': True, 'magic_item': True,
                      'special_rules': ['Magical Attacks']},
    'skirmishers_blade': {'name': "Skirmisher's Blade", 'tag': 'combat', 'strength_bonus': 0,
                         'ap_penetration': 1, 'magical': True, 'magic_item': True, 'extra_attacks': 1,
                         'wound_reroll_flank': True, 'special_rules': ['Magical Attacks', 'Extra Attacks (+1)']},
    'thornspitter_combat': {'name': 'Thornspitter Stave', 'tag': 'combat', 'strength_bonus': 1,
                           'ap_penetration': 0, 'magical': True, 'magic_item': True,
                           'special_rules': ['Armour Bane (1)', 'Magical Attacks']},
    'thornspitter_ranged': {'name': 'Thornspitter Stave (ranged)', 'tag': 'ranged', 'ranged_range': 24,
                          'ranged_strength': 4, 'ranged_AP': 0, 'ranged_shots': 1, 'ponderous': True,
                          'magical': True, 'magic_item': True,
                          'special_rules': ['Armour Bane (1)', 'Magical Attacks', 'Ponderous']},
        'trailblazers_hatchet': {'name': "Trailblazer's Hatchet", 'tag': 'combat', 'strength_bonus': 1,
                                                        'ap_penetration': 1, 'magical': True, 'magic_item': True, 'flaming_attacks': True,
                                                        'special_rules': ['Flaming Attacks', 'Magical Attacks', 'Move Through Cover']},
}


@dataclass
class ItemInstance:
    instance_id: str
    source: dict
    disabled_reason: str | None = None
    destroyed: bool = False
    uses: dict = field(default_factory=dict)

    @property
    def owner_ref(self):
        return self.source.get('owner_ref')

    @property
    def name(self):
        return self.source.get('name', 'Unknown item')

    def to_record(self):
        return {'instance_id': self.instance_id, 'source': deepcopy(self.source),
                'disabled_reason': self.disabled_reason, 'destroyed': self.destroyed,
                'uses': deepcopy(self.uses)}

    @classmethod
    def from_record(cls, record):
        return cls(record['instance_id'], deepcopy(record['source']),
                   record.get('disabled_reason'), bool(record.get('destroyed', False)),
                   deepcopy(record.get('uses', {})))


def inventory(member):
    return getattr(getattr(member, 'unit', member), 'magic_item_inventory', [])


def install_inventory(member, sources):
    """Reinstalling the same purchase keeps its state, but never aliases copies."""
    group = getattr(member, 'unit', member)
    namespace = getattr(member, 'unitName', group.name)
    existing = {item.instance_id: item for item in inventory(member)}
    instances = []
    seen = set()
    for index, source in enumerate(sources):
        for ordinal in range(max(0, int(source.get('number', 1)))):
            identity = json.dumps([namespace, source.get('selection_ref', f'item-{index}'), ordinal],
                                  separators=(',', ':'))
            if identity in seen:
                raise ValueError(f'Duplicate purchased item: {identity}')
            seen.add(identity)
            item = existing.get(identity)
            if item is None:
                item = ItemInstance(identity, deepcopy(source))
            instances.append(item)
    group.magic_item_inventory = instances
    bind_inventory(member)
    return instances


def save_inventory(member):
    return [item.to_record() for item in inventory(member)]


def restore_inventory(member, records):
    instances = [ItemInstance.from_record(record) for record in records]
    if len({item.instance_id for item in instances}) != len(instances):
        raise ValueError('Duplicate saved magic-item instance')
    getattr(member, 'unit', member).magic_item_inventory = instances
    bind_inventory(member)


def item_profiles(member):
    group = getattr(member, 'unit', member)
    profiles = [group.model] + list(getattr(group, 'command_models', {}).values())
    for profile in list(profiles):
        for tag in ('mount', 'crew', 'beasts'):
            part = getattr(profile, f'get_{tag}')()
            if part is not None and not any(existing is part for existing in profiles):
                profiles.append(part)
    return profiles


def bind_inventory(member):
    """Profiles keep a non-owning link; item bonuses never enter saved base stats."""
    group = getattr(member, 'unit', member)
    model = getattr(group, 'model', None)
    if model is None or not hasattr(member, 'unit'):
        return
    try:
        reference = weakref.ref(member)
    except TypeError:
        reference = lambda: member
    profiles = item_profiles(member)
    for profile in profiles:
        profile._magic_item_member = reference
    refresh_item_weapons(member, profiles)


def refresh_item_weapons(member, profiles):
    """Owned weapon profiles are derived from live inventory, not roster prose (p. 46)."""
    for profile in profiles:
        equipped = profile.equipedWeapon
        owned_equipped = bool(equipped and equipped.get('item_source'))
        profile.weapons = {name: weapon for name, weapon in profile.weapons.items() if not weapon.get('item_source')}
        for item in inventory(member):
            definition = REGISTRY.resolve(item.source)
            bearer = resolve_bearer(member, item)
            if definition is None or bearer is None or bearer.profile is not profile:
                continue
            for effect in definition.effects:
                if effect.kind != EffectKind.WEAPON:
                    continue
                weapon = deepcopy(ITEM_WEAPONS[effect.value])
                slot = profile.weapon_slot(weapon['name'])
                if slot is not None:
                    if profile.weapons[slot] is equipped:
                        owned_equipped = True
                    del profile.weapons[slot]
                if bearer_unavailable(member, item, allow_retired=True):
                    continue
                weapon['item_source'] = item.instance_id
                profile.weapons[weapon['name']] = weapon
        if owned_equipped:
            slot = profile.weapon_slot(equipped['name']) or profile.weapon_slot('Hand Weapon')
            profile.special_rules = [rule for rule in profile.special_rules if rule is not equipped]
            profile.equipedWeapon = profile.weapons.get(slot)
            if profile.equipedWeapon is not None:
                profile.special_rules.append(profile.equipedWeapon)


async def activate_start_items(game):
    """Trailblazer's once-only unit grant lasts this player's turn (Companion p. 46)."""
    from characters import side_of
    from panda3d.core import Vec3
    from troop_types import is_infantry
    turn = current_turn(game)
    if turn is None:
        return
    for member in list(game.units):
        if side_of(game, member, None) != turn[0]:
            continue
        for entry in effects_for(member, EffectKind.START_RULE, context='Start of Turn'):
            item = entry.item
            if item.uses.get('start_offer', {}).get('turn') == turn:
                continue
            if not is_infantry(entry.bearer.profile.characteristics.get('Troop Type')):
                rule_skipped(item.name, member, 'requires an infantry bearer; no unit grant')
                continue
            use = game.aiControls(member)
            if not use:
                answer = await game.makeChoiceNew(['Reveal path', 'Keep use'], Vec3(0, 0, 10), owner=member,
                                                  prompt=f'{member.unit.name}: Trailblazer\'s Hatchet?',
                                                  detail='Move Through Cover for the unit until this turn ends; once per game')
                use = answer == 'Reveal path'
            item.uses['start_offer'] = {'turn': list(turn)}
            if not activate_ability(game, member, item, 'path', 'Start of Turn', confirmed=use):
                continue
            host = getattr(member, 'hostUnit', None) or member
            recipients = [host]
            joined = getattr(host, 'joinedCharacter', None)
            if joined is not None:
                recipients.append(joined)
            for recipient in recipients:
                for profile in item_profiles(recipient):
                    profile.special_rules.append({'name': item.name, 'move_through_cover': True,
                                                  'item_rule_source': item.instance_id, 'item_rule_turn': list(turn)})
            count = sum(recipient.unit.nmodels for recipient in recipients)
            rule_log(item.name, host, f'{count} models gain Move Through Cover until the end of player '
                     f'{turn[0]} turn {turn[1]}; single use spent')


def end_item_turn(game):
    """Remove activated item grants without removing native rules (Companion p. 46)."""
    for member in game.units:
        removed = {}
        for profile in item_profiles(member):
            for rule in profile.special_rules:
                if isinstance(rule, dict) and rule.get('item_rule_source'):
                    removed[rule['item_rule_source']] = rule['name']
            profile.special_rules[:] = [rule for rule in profile.special_rules
                                        if not (isinstance(rule, dict) and rule.get('item_rule_source'))]
        for name in removed.values():
            rule_log(name, member, 'end of turn: temporary Move Through Cover removed; native rules retained')


def report_item_cover(profile, shots):
    """Report once per volley, never inside the hit-dice loop (Companion p. 48)."""
    penalty = 2 if getattr(profile, 'full_cover', False) else int(bool(getattr(profile, 'partial_cover', False)))
    for entry in profile_effects(profile, EffectKind.IGNORE_COVER):
        if penalty:
            rule_log(entry.item.name, entry.bearer.carrier, f'{shots} shots: cover modifier -{penalty} -> 0 To Hit')
        else:
            rule_skipped(entry.item.name, entry.bearer.carrier, f'{shots} shots: no full or partial cover modifier to ignore')


def item_target_protected(game, attacker, target, *, log=False):
    """Shadowed Mantle extends lone-character screening to 6 inches (pp. 47, 206)."""
    if game is None or attacker is None or not hasattr(target, 'unit'):
        return False
    entries = effects_for(target, EffectKind.TARGET_PROTECTION)
    if not entries:
        return False
    from characters import side_of
    from psychology import obb_distance
    from scouts import model_base_boxes
    from troop_types import is_infantry
    side = side_of(game, target, None)
    if side is None or side_of(game, attacker, None) in (None, side):
        return False
    if (target.unit.nmodels != 1 or getattr(target, 'hostUnit', None) is not None
            or not is_infantry(target.unit.model.characteristics.get('Troop Type'))):
        return False
    boxes = model_base_boxes(target)
    for friend in game.units:
        if (friend is target or side_of(game, friend, None) != side or friend.unit.nmodels < 5
                or getattr(friend, 'hostUnit', None) is not None
                or not getattr(friend, 'isDeployed', False) or getattr(friend, 'state', None) == 'IsFleeing'
                or not is_infantry(friend.unit.model.characteristics.get('Troop Type'))):
            continue
        distance = min((obb_distance(own, other) for own in boxes for other in model_base_boxes(friend)),
                       default=float('inf'))
        for entry in entries:
            if distance <= float(entry.effect.value) + 1e-6:
                if log:
                    rule_log(entry.item.name, target, f'{friend.unit.nmodels} friendly infantry in {friend.unit.name} '
                             f'at {distance:.2f}" <= {entry.effect.value}": enemy targeting prohibited even when closest')
                return True
    if log:
        for entry in entries:
            rule_skipped(entry.item.name, target, f'no non-fleeing friendly unit of 5+ infantry within '
                         f'{entry.effect.value}"; no targeting protection')
    return False


def item_armour_save(profile, base_save, *, log=False):
    """Additive helmets, Battle March p. 47; minimum armour value 2+ (p. 141)."""
    reference = getattr(profile, '_magic_item_member', None)
    member = reference() if callable(reference) else None
    if member is None:
        return base_save
    body_armour = effects_for(member, EffectKind.BODY_ARMOUR, profile=profile)
    original = base_save
    if body_armour:
        from models import ARMOUR_MODIFIERS
        modifiers = sum(str(piece).strip().casefold() in ARMOUR_MODIFIERS for piece in profile.armour)
        base_save = max(2, min(int(entry.effect.value) for entry in body_armour) - modifiers)
    effects = effects_for(member, EffectKind.ARMOUR, profile=profile)
    result = max(2, base_save - sum(int(entry.effect.value) for entry in effects)) if effects else base_save
    if log:
        for entry in body_armour:
            rule_log(entry.item.name, member, f'{profile.name}: body armour {entry.effect.value}+, '
                     f'with equipment modifiers {original}+ -> {base_save}+ before AP')
        report_inactive_effects(member, EffectKind.ARMOUR, f'armour remains {base_save}+ before AP', profile=profile)
        for entry in effects:
            if result != base_save:
                rule_log(entry.item.name, member,
                         f'{profile.name}: armour {base_save}+ -> {result}+ before AP; '
                         'passive protection independent of the Break re-roll use')
            else:
                rule_skipped(entry.item.name, member, f'{profile.name}: armour already at the 2+ limit before AP')
    return result


def item_ap_armour_save(profile, save, penetration, permitted, history=None):
    """Padded Hauberk improves only AP '-' saves, capped at 2+ (Companion p. 47)."""
    entries = profile_effects(profile, EffectKind.AP_ZERO_ARMOUR)
    result = max(2, save - max((int(entry.effect.value) for entry in entries), default=0)) if entries and permitted and penetration == 0 else save
    if history is not None:
        history.extend((entry, save, result, penetration, permitted) for entry in entries)
    return result


def report_ap_armour(profile, history):
    from collections import Counter
    groups = Counter((entry.item.instance_id, before, after, penetration, permitted)
                     for entry, before, after, penetration, permitted in history)
    entries = {entry.item.instance_id: entry for entry, *_ in history}
    for (identity, before, after, penetration, permitted), count in groups.items():
        entry = entries[identity]
        if before != after:
            rule_log(entry.item.name, entry.bearer.carrier,
                     f'{count} wound(s) at AP 0: armour {before}+ -> {after}+ (maximum 2+)')
        else:
            reason = 'armour saves prohibited' if not permitted else f'AP -{penetration}' if penetration else 'already at 2+'
            rule_skipped(entry.item.name, entry.bearer.carrier, f'{count} wound(s): no padding improvement; {reason}')


async def reroll_break_test(game, unit, dice, ld, diff, overwhelm, roll_dice, bsb=None):
    """One shared optional re-roll: Battle March p. 47; Rulebook pp. 93, 154, 203."""
    from panda3d.core import Vec3
    from psychology import break_test_outcome, should_reroll_break

    outcome = break_test_outcome(dice, ld, diff, overwhelm)
    candidates = effects_for(unit, EffectKind.REROLL, value='Break', context='Break')
    report_inactive_effects(unit, EffectKind.REROLL,
                           f'Break: 2D6={sum(dice)}, Ld {ld}, difference {diff}: {outcome}',
                           value='Break', context='Break')
    chosen = candidates[0] if candidates and bsb is None else None
    if bsb is None and chosen is None:
        return dice
    name = chosen.item.name if chosen else 'Hold Your Ground'
    if bsb is not None:
        for entry in candidates:
            rule_skipped(entry.item.name, unit, 'Hold Your Ground is available; preserve the once-per-game use')
    if game.aiControls(unit):
        use = should_reroll_break(outcome, ld, diff, overwhelm)
    else:
        options = [f'Re-roll\n({outcome})', 'Keep']
        selected = await game.makeChoiceNew(
            options, Vec3(0, 0, 10), owner=unit,
            prompt=f'{unit.unit.name}: {name} Break test re-roll?',
            detail=f'2D6={sum(dice)}, Ld {ld}, combat difference {diff}; {outcome}')
        use = selected == options[0]
    if chosen is not None:
        if not use:
            rule_skipped(name, unit, f'keeps 2D6={sum(dice)}, Ld {ld}, difference {diff}: {outcome}; no use spent')
            return dice
        if not activate_ability(game, chosen.bearer.carrier, chosen.item, 'courage', 'Break',
                                confirmed=use, recipient=unit):
            return dice
    elif not use:
        rule_skipped(name, unit, f'keeps 2D6={sum(dice)}, Ld {ld}, difference {diff}: {outcome}')
        return dice
    result = await roll_dice()
    replacement = break_test_outcome(result, ld, diff, overwhelm)
    rule_log(name, unit, f'Break: 2D6={sum(dice)}, Ld {ld}, difference {diff}, '
             f'overwhelmed={overwhelm}: {outcome} -> {sum(result)} ({replacement}); no further re-roll')
    return result


def use_count(item, ability, turn=None):
    record = item.uses.get(ability.key, {})
    token = list(turn) if isinstance(turn, (list, tuple)) else turn
    if ability.per_turn and record.get('turn') != token:
        return 0
    return record.get('count', 0)


def current_turn(game):
    counter = getattr(game, 'roundCounter', None)
    if counter is None:
        return None
    player = counter.current_player
    return [player, counter.currentRoundPlayer[player - 1]]


def ability_unavailable(item, ability_key, context, *, turn=None, registry=REGISTRY):
    definition = registry.resolve(item.source)
    if definition is None or not definition.supported:
        return 'item effects are not implemented'
    if item.destroyed or item.disabled_reason is not None:
        return item.disabled_reason or 'item destroyed'
    ability = next((ability for ability in definition.abilities if ability.key == ability_key), None)
    if ability is None:
        return 'unknown ability'
    if context != ability.context:
        return f'ability requires {ability.context}, not {context}'
    if ability.per_turn and turn is None:
        return 'turn identity is required'
    if use_count(item, ability, turn) >= ability.limit:
        return 'ability spent'
    return None


def spend_ability(item, ability_key, context, *, confirmed, turn=None, registry=REGISTRY):
    """Commit only after the caller's choice and outcome checks have succeeded."""
    if not confirmed or ability_unavailable(item, ability_key, context, turn=turn, registry=registry):
        return False
    definition = registry.resolve(item.source)
    ability = next(ability for ability in definition.abilities if ability.key == ability_key)
    item.uses[ability.key] = {'count': use_count(item, ability, turn) + 1,
                            'turn': (list(turn) if isinstance(turn, (list, tuple)) else deepcopy(turn))
                            if ability.per_turn else None}
    return True


@dataclass(frozen=True)
class ItemBearer:
    carrier: object
    profile: object
    command: dict | None = None


def resolve_bearer(member, item):
    """Resolve selected owners only, never deduce a bearer from an item name."""
    group = member.unit
    if not item.owner_ref:
        return None
    for index, entry in enumerate(getattr(group, 'command', [])):
        if entry.get('selection_ref') == item.owner_ref:
            profile = getattr(group, 'command_models', {}).get(entry.get('selection_ref', str(index)), group.model)
            return ItemBearer(member, profile, entry)
    records = getattr(group, 'roster_metadata', {}).get('roster_selections', [])
    owner = next((record for record in records if record.get('ref') == item.owner_ref), None)
    if owner is None:
        return None
    if owner.get('type') == 'unit':
        return ItemBearer(member, group.model)
    names = {slugify(owner.get('name', ''))} | {
        slugify(profile.get('name', '')) for profile in owner.get('profiles', [])
        if profile.get('typeName') == 'Model'}
    candidates = []
    for profile in [group.model] + [part for tag in ('mount', 'crew', 'beasts')
                                   if (part := getattr(group.model, f'get_{tag}')()) is not None]:
        if slugify(profile.name) in names:
            candidates.append(profile)
    return ItemBearer(member, candidates[0]) if len(candidates) == 1 else None


def bearer_unavailable(member, item, *, game=None, allow_retired=False):
    """Command losses and retired characters cannot confer benefits (pp. 199-210)."""
    if not any(owned is item for owned in inventory(member)):
        return 'item is not in this bearer inventory'
    if game is not None and not any(unit is member for unit in game.units):
        return 'bearer is no longer in play'
    if member.unit.nmodels <= 0:
        return 'bearer has no surviving models'
    if item.destroyed or item.disabled_reason is not None:
        return item.disabled_reason or 'item destroyed'
    bearer = resolve_bearer(member, item)
    if bearer is None:
        return 'roster owner cannot be resolved'
    definition = REGISTRY.resolve(item.source)
    if definition is not None and definition.key in ('trailblazers_hatchet', 'shadowed_mantle'):
        from troop_types import is_infantry
        if not is_infantry(bearer.profile.characteristics.get('Troop Type')):
            return 'item requires an infantry bearer (Companion pp. 46-47)'
    if not allow_retired and getattr(member, 'retiredFromCombat', False):
        return 'bearer retired from combat'
    if bearer.command is not None:
        from command_groups import living_command
        if not any(entry is bearer.command for entry in living_command(member)):
            return 'command bearer was lost'
        if not allow_retired and bearer.command.get('retired', False):
            return 'command bearer retired from combat'
    if slugify(item.source.get('category', '')) == 'magic_standards':
        if not (bearer.command is not None and bearer.command.get('role') == 'standard_bearer') and not getattr(member, 'isBSB', False):
            return 'magic standard has no standard bearer (p. 341)'
    return None


@dataclass(frozen=True)
class EffectContribution:
    source_id: str
    item: ItemInstance
    effect: ItemEffect
    bearer: ItemBearer


def _affects(bearer, effect, recipient, profile):
    carrier = bearer.carrier
    host = getattr(carrier, 'hostUnit', None) or carrier
    personal = (recipient is carrier and profile is bearer.profile and bearer.command is None) or (
        bearer.command is not None and getattr(recipient, 'command_entry', None) is bearer.command)
    if bearer.command is not None and bearer.command.get('role') == 'champion':
        personal = personal or (recipient is carrier and profile is bearer.profile)
    retired = getattr(carrier, 'retiredFromCombat', False) or (
        bearer.command is not None and bearer.command.get('retired', False))
    if retired and not personal:
        return False
    if effect.scope == Scope.BEARER:
        return personal
    if effect.scope == Scope.UNIT:
        return recipient is host
    if effect.scope == Scope.UNIT_AND_JOINED:
        return recipient is host or getattr(recipient, 'hostUnit', None) is host
    if effect.scope == Scope.BEARER_AND_UNIT:
        return personal or (recipient is host and host is not carrier)
    return False


def active_effects(game, recipient, *, profile=None, context=None, turn=None, registry=REGISTRY):
    """Derived source-tagged contributions; no mutation of native or base stats.

    Callers apply the typed effect at its outcome and log the deciding numbers.
    Activated effects are candidates only; querying never consumes an ability.
    """
    turn = current_turn(game) if turn is None else turn
    profile = recipient.unit.model if profile is None else profile
    contributions = {}
    for member in game.units:
        for item in inventory(member):
            definition = registry.resolve(item.source)
            if definition is None or not definition.supported or bearer_unavailable(
                    member, item, game=game, allow_retired=True):
                continue
            bearer = resolve_bearer(member, item)
            for effect in definition.effects:
                if effect.ability and ability_unavailable(item, effect.ability, context, turn=turn, registry=registry):
                    continue
                if _affects(bearer, effect, recipient, profile):
                    identity = json.dumps([item.instance_id, effect.key], separators=(',', ':'))
                    contributions[identity] = EffectContribution(identity, item, effect, bearer)
    return list(contributions.values())


def effects_for(member, kind, *, value=None, profile=None, context=None):
    """Read item contributions through the recipient's live battle ownership."""
    game = getattr(member, 'game', None)
    if not isinstance(getattr(game, 'units', None), (list, tuple)):
        host = getattr(member, 'hostUnit', None)
        candidates = [member, host, getattr(member, 'joinedCharacter', None)]
        game = SimpleNamespace(units=[candidate for candidate in candidates if candidate is not None])
    return [entry for entry in active_effects(game, member, profile=profile, context=context)
            if entry.effect.kind == kind and (value is None or entry.effect.value == value)]


def profile_effects(profile, kind):
    reference = getattr(profile, '_magic_item_member', None)
    member = reference() if callable(reference) else None
    return effects_for(member, kind, profile=profile) if member is not None else []


def report_inactive_effects(member, kind, detail, *, value=None, profile=None, context=None):
    """Explain inactive nearby purchases only at an outcome, never from a query."""
    active = {entry.item.instance_id for entry in effects_for(member, kind, value=value,
                                                             profile=profile, context=context)}
    profile = member.unit.model if profile is None else profile
    carriers = [member, getattr(member, 'hostUnit', None), getattr(member, 'joinedCharacter', None)]
    seen = set()
    for carrier in carriers:
        if carrier is None:
            continue
        for item in inventory(carrier):
            if item.instance_id in active or item.instance_id in seen:
                continue
            seen.add(item.instance_id)
            definition = REGISTRY.resolve(item.source)
            effects = [effect for effect in definition.effects
                       if effect.kind == kind and (value is None or effect.value == value)] if definition else []
            bearer = resolve_bearer(carrier, item)
            if not effects or (all(effect.scope == Scope.BEARER for effect in effects)
                               and bearer is not None and bearer.profile is not profile):
                continue
            reason = bearer_unavailable(carrier, item, allow_retired=True)
            if reason is None:
                reason = next((reason for effect in effects if effect.ability
                               if (reason := ability_unavailable(item, effect.ability, context))), None)
            rule_skipped(item.name, member, f'{detail}; {reason or "bearer cannot confer this benefit on the recipient"}')


def known_spell_count(member, *, log=False):
    """Silvery Wand adds a known spell, not a Wizard level (Forces of Fantasy p. 183)."""
    level = member.unit.model.wizard_level()
    effects = effects_for(member, EffectKind.SPELLS) if level > 0 else []
    count = level + sum(int(entry.effect.value) for entry in effects)
    if log:
        report_inactive_effects(member, EffectKind.SPELLS, f'Level {level}: {count} known spells')
        for entry in effects:
            rule_log(entry.item.name, member,
                     f'Level {level}: {level} -> {count} known spells; '
                     f'Wizard level and per-turn casting allowance remain {level}')
    return count


def activate_ability(game, member, item, ability_key, context, *, confirmed, recipient=None,
                     profile=None, turn=None, registry=REGISTRY):
    """Shared human/AI commit gate; failed choices never spend an item ability."""
    turn = current_turn(game) if turn is None else turn
    recipient = member if recipient is None else recipient
    reason = (bearer_unavailable(member, item, game=game, allow_retired=True)
              or ability_unavailable(item, ability_key, context, turn=turn, registry=registry))
    if reason is None and not any(
            contribution.item is item and contribution.effect.ability == ability_key
            for contribution in active_effects(game, recipient, profile=profile, context=context,
                                                turn=turn, registry=registry)):
        reason = 'ability has no eligible effect for this recipient'
    if reason or not confirmed:
        rule_skipped(item.name, member, f'{ability_key}: {reason or "activation declined"}; no use spent')
        return False
    spent = spend_ability(item, ability_key, context, confirmed=True, turn=turn, registry=registry)
    ability = next(ability for ability in registry.resolve(item.source).abilities if ability.key == ability_key)
    rule_log(item.name, member,
             f'{ability_key} in {context}: used {use_count(item, ability, turn)}/{ability.limit} '
             f'{"this turn" if ability.per_turn else "this battle"}; other abilities and passive effects unchanged')
    return spent


async def combat_result_bonus(game, units, own_score, enemy_score):
    """Optional Banner of Renown before the musician tie-break (Companion p. 47)."""
    from panda3d.core import Vec3
    bonus = 0
    seen = set()
    for unit in units:
        entries = effects_for(unit, EffectKind.COMBAT_RESULT, context='Combat result')
        report_inactive_effects(unit, EffectKind.COMBAT_RESULT, 'no combat-result bonus', context='Combat result')
        for entry in entries:
            if entry.item.instance_id in seen:
                continue
            seen.add(entry.item.instance_id)
            use = own_score + bonus <= enemy_score
            if not game.aiControls(unit):
                answer = await game.makeChoiceNew(['Raise banner', 'Keep banner'], Vec3(0, 0, 10), owner=unit,
                                                  prompt=f'{unit.unit.name}: Banner of Renown?',
                                                  detail=f'Combat result {own_score + bonus} vs {enemy_score}; single-use +1')
                use = answer == 'Raise banner'
            if activate_ability(game, entry.bearer.carrier, entry.item, 'renown', 'Combat result',
                                confirmed=use, recipient=unit):
                before = own_score + bonus
                bonus += int(entry.effect.value)
                rule_log(entry.item.name, unit, f'combat result {before} -> {own_score + bonus} vs {enemy_score}; '
                         'single-use +1, before musician tie-break')
    return bonus


async def magic_roll_bonus(game, member, kind):
    """Declare the Shard before casting/dispelling dice, once total (Companion p. 48)."""
    if game is None or member is None:
        return 0
    from panda3d.core import Vec3
    entries = effects_for(member, EffectKind.MAGIC_ROLL, context='Magic roll')
    report_inactive_effects(member, EffectKind.MAGIC_ROLL, f'no {kind} modifier', context='Magic roll')
    bonus = 0
    for entry in entries:
        use = game.aiControls(member)
        if not use:
            answer = await game.makeChoiceNew(['Use shard', 'Keep shard'], Vec3(0, 0, 10), owner=member,
                                              prompt=f'{member.unit.name}: Wyrdstone Shard?',
                                              detail=f'+1 to this {kind} roll; single use, declared before rolling')
            use = answer == 'Use shard'
        if activate_ability(game, member, entry.item, 'shard', 'Magic roll', confirmed=use):
            bonus += int(entry.effect.value)
            rule_log(entry.item.name, member, f'declared before {kind} dice; +1 to this roll, shared casting/dispel use spent')
    return bonus


def item_spell_available(member, spell):
    """An unmade item cannot supply a Bound spell (Forces of Fantasy p. 186)."""
    if not spell.get('bound'):
        return True
    owners = [item for item in inventory(member) if item.name == spell.get('source')]
    return not owners or any(not item.destroyed and item.disabled_reason is None for item in owners)


def bind_generated_spells(member, known):
    """Retain ownership of extra generated slots, not extra Wizard levels (FoF p. 183)."""
    extra = iter(known[member.unit.model.wizard_level():])
    for contribution in effects_for(member, EffectKind.SPELLS):
        for _ in range(int(contribution.effect.value)):
            record = next(extra, None)
            if record is not None:
                record['granted_by_item'] = contribution.item.instance_id


def disable_item(member, item, reason, *, destroyed=False):
    """Persist whole-item suppression separately from ability exhaustion."""
    if not reason:
        raise ValueError('Item suppression requires a reason')
    if not any(owned is item for owned in inventory(member)):
        raise ValueError('Item does not belong to this inventory')
    if item.destroyed or (item.disabled_reason is not None and not destroyed):
        rule_skipped(item.name, member, f'already disabled: {item.disabled_reason or "destroyed"}')
        return False
    profile = member.unit.model
    if item.name == 'Silvery Wand':
        ordinary = [record for record in profile.spells.values() if not record.get('bound')]
        if not any(record.get('granted_by_item') for record in ordinary):
            bind_generated_spells(member, ordinary)
    item.disabled_reason = reason
    item.destroyed = destroyed
    bind_inventory(member)
    for key, record in list(profile.spells.items()):
        if record.get('granted_by_item') == item.instance_id:
            del profile.spells[key]
            rule_log(item.name, member, f'{key}: item-granted spell removed; Wizard level unchanged (FoF pp. 183, 186)')
    rule_log(item.name, member, f'{reason}; all item effects disabled for the rest of the battle')
    return True


def inventory_lines(member, *, width=54, turn=None, registry=REGISTRY):
    """Read-only inventory facts for the existing unit details panel."""
    lines = []
    for item in inventory(member):
        definition = registry.resolve(item.source)
        bearer = resolve_bearer(member, item)
        owner = (bearer.command.get('name', bearer.command['role']) if bearer.command is not None
                 else bearer.profile.name) if bearer else 'unresolved owner'
        status = 'supported' if definition is not None and definition.supported else 'unsupported'
        reason = bearer_unavailable(member, item, allow_retired=True)
        if reason:
            status += f'; {reason}'
        if bearer and (getattr(member, 'retiredFromCombat', False)
                       or (bearer.command is not None and bearer.command.get('retired', False))):
            status += '; retired, no unit benefits'
        facts = [f'Item: {item.name}', f'Bearer: {owner}', status]
        if definition is not None and definition.supported:
            for ability in definition.abilities:
                count = use_count(item, ability, turn)
                state = 'unavailable' if reason or (ability.per_turn and turn is None) else (
                    'spent' if count >= ability.limit else 'ready')
                facts.append(f'{ability.key}: {state}, {count}/{ability.limit} used')
        for fact in facts:
            lines.extend(textwrap.wrap(fact, width=width))
    return lines


def report_inventory(member, *, registry=REGISTRY):
    """Report unsupported purchases once at roster load, never in query loops."""
    for item in inventory(member):
        definition = registry.resolve(item.source)
        reason = bearer_unavailable(member, item)
        if definition is None or not definition.supported:
            rule_skipped(item.name, member,
                         f'purchased item retained; effects not implemented'
                         + (f'; {reason}' if reason else ''))
        elif reason:
            rule_skipped(item.name, member, reason)