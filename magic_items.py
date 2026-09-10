"""Explicit item definitions and purchased instances; never execute roster prose.

Magic item categories and single-use rules: Rulebook pp. 337-340. Individual
definitions must cite their own rule source before any effects are enabled.
"""

from copy import deepcopy
from dataclasses import dataclass, field
from enum import StrEnum
import json
import textwrap

from battlescribe import slugify
from rules_log import rule_log, rule_skipped


class EffectKind(StrEnum):
    ARMOUR = 'armour_modifier'
    RULE = 'rule_grant'
    SPELLS = 'extra_known_spells'
    REROLL = 'reroll'


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
    ItemDefinition('silvery_wand', 'Silvery Wand', 'Arcane Items', 'Forces of Fantasy p. 183'),
    ItemDefinition('helm_of_courage', 'Helm of Courage', 'Magic Armour',
                   "Battle March: General's Companion p. 47"),
    ItemDefinition('banner_of_the_bold', 'The Banner of the Bold', 'Magic Standards',
                   "Battle March: General's Companion p. 47", aliases=('Banner of the Bold',)),
))


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
    return instances


def save_inventory(member):
    return [item.to_record() for item in inventory(member)]


def restore_inventory(member, records):
    instances = [ItemInstance.from_record(record) for record in records]
    if len({item.instance_id for item in instances}) != len(instances):
        raise ValueError('Duplicate saved magic-item instance')
    getattr(member, 'unit', member).magic_item_inventory = instances


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


def disable_item(member, item, reason, *, destroyed=False):
    """Persist whole-item suppression separately from ability exhaustion."""
    if not reason:
        raise ValueError('Item suppression requires a reason')
    if not any(owned is item for owned in inventory(member)):
        raise ValueError('Item does not belong to this inventory')
    if item.destroyed or (item.disabled_reason is not None and not destroyed):
        rule_skipped(item.name, member, f'already disabled: {item.disabled_reason or "destroyed"}')
        return False
    item.disabled_reason = reason
    item.destroyed = destroyed
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