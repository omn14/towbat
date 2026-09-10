"""Inventory infrastructure tests use synthetic effects, not inferred item rules."""

import json
from types import SimpleNamespace

import pytest

from magic_items import (EffectKind, ItemAbility, ItemDefinition, ItemEffect, ItemRegistry,
                         REGISTRY, ability_unavailable, install_inventory, inventory,
                         restore_inventory, save_inventory, spend_ability)


def member(name='Bearer'):
    return SimpleNamespace(unitName=name, unit=SimpleNamespace(name=name, nmodels=1))


def source(identity='purchase', number=1):
    return {'selection_ref': identity, 'definition_id': 'test-definition', 'owner_ref': 'bearer',
            'name': 'Test Relic', 'category': 'Talismans', 'number': number, 'points_cost': 10,
            'profiles': [{'characteristics': [{'name': 'Description', '$text': 'Not executable'}]}]}


@pytest.fixture
def registry():
    return ItemRegistry((ItemDefinition(
        'test_relic', 'Test Relic', 'Talismans', 'Synthetic test definition',
        effects=(ItemEffect('armour', EffectKind.ARMOUR, 1),
                 ItemEffect('reroll', EffectKind.REROLL, 'Break', ability='courage')),
        abilities=(ItemAbility('courage', 'Break'), ItemAbility('turn_power', 'Casting', per_turn=True)),
        catalogue_ids=('test-definition',)),))


def test_registry_ids_aliases_and_categories_are_explicit(registry):
    assert registry.resolve(source()).key == 'test_relic'
    renamed = dict(source(), name='RENAMED IN EXPORT')
    assert registry.resolve(renamed).key == 'test_relic'
    assert registry.resolve(dict(source(), definition_id='unknown', name='TEST RELIC')).key == 'test_relic'
    assert registry.resolve(dict(source(), category='Magic Weapons')) is None
    with pytest.raises(ValueError, match='Ambiguous'):
        registry.register(registry.definitions['test_relic'])


def test_real_items_are_recognized_but_effects_remain_point_four():
    for name, category in [('Silvery Wand', 'Arcane Items'), ('Helm Of Courage', 'Magic Armour'),
                           ('The Banner Of The Bold', 'Magic Standards')]:
        definition = REGISTRY.resolve({'name': name, 'category': category})
        assert definition is not None and definition.reference
        assert not definition.supported and not definition.effects


def test_copies_and_bearers_have_distinct_stable_instances(registry):
    bearer = member()
    first, second = install_inventory(bearer, [source(number=2)])
    assert first.instance_id != second.instance_id
    assert spend_ability(first, 'courage', 'Break', confirmed=True, registry=registry)
    assert second.uses == {}
    reinstalled = install_inventory(bearer, [source(number=2)])
    assert reinstalled[0] is first and reinstalled[0].uses
    other = install_inventory(member('Other'), [source(number=2)])
    assert not {item.instance_id for item in other}.intersection(item.instance_id for item in reinstalled)


def test_activation_spends_only_confirmed_eligible_ability(registry):
    item = install_inventory(member(), [source()])[0]
    assert not spend_ability(item, 'courage', 'Break', confirmed=False, registry=registry)
    assert not spend_ability(item, 'courage', 'Rally', confirmed=True, registry=registry)
    assert item.uses == {}
    assert spend_ability(item, 'courage', 'Break', confirmed=True, registry=registry)
    assert not spend_ability(item, 'courage', 'Break', confirmed=True, registry=registry)
    assert ability_unavailable(item, 'courage', 'Break', registry=registry) == 'ability spent'
    assert not item.destroyed and item.disabled_reason is None


def test_state_roundtrips_without_refunding_or_aliasing(registry):
    bearer = member()
    item = install_inventory(bearer, [source()])[0]
    spend_ability(item, 'courage', 'Break', confirmed=True, registry=registry)
    item.disabled_reason = 'Test suppression'
    records = json.loads(json.dumps(save_inventory(bearer)))
    other = member()
    restore_inventory(other, records)
    restored = inventory(other)[0]
    assert restored.to_record() == item.to_record()
    restored.uses.clear()
    assert item.uses and records[0]['uses']


def test_turn_uses_require_identity_and_reset_only_for_new_turn(registry):
    item = install_inventory(member(), [source()])[0]
    assert not spend_ability(item, 'turn_power', 'Casting', confirmed=True, registry=registry)
    assert spend_ability(item, 'turn_power', 'Casting', confirmed=True, turn=[1, 1], registry=registry)
    assert not spend_ability(item, 'turn_power', 'Casting', confirmed=True, turn=[1, 1], registry=registry)
    assert spend_ability(item, 'turn_power', 'Casting', confirmed=True, turn=[1, 2], registry=registry)


def test_unknown_item_data_remains_inert_and_lossless():
    raw = dict(source(), name='Unknown Relic')
    item = install_inventory(member(), [raw])[0]
    assert item.source == raw
    assert not spend_ability(item, 'courage', 'Break', confirmed=True)
    assert item.uses == {}
    with pytest.raises(ValueError, match='Duplicate'):
        install_inventory(member(), [raw, raw])


def live_member(name='Bearer', count=1):
    from models import model

    bearer = member(name)
    bearer.unit.nmodels = count
    bearer.unit.files = count
    bearer.unit.model = model('Chaos Warrior', '')
    bearer.unit.roster_metadata = {'roster_selections': [
        {'ref': 'bearer', 'name': 'Chaos Warrior', 'type': 'model'}]}
    return bearer


def test_spent_ability_keeps_passive_effect_without_mutating_baseline(registry):
    from copy import deepcopy
    from magic_items import activate_ability, active_effects, disable_item

    bearer = live_member()
    item = install_inventory(bearer, [source()])[0]
    game = SimpleNamespace(units=[bearer])
    baseline = deepcopy(bearer.unit.model._base_characteristics)
    effects = active_effects(game, bearer, context='Break', registry=registry)
    assert len(effects) == 2
    assert len({effect.source_id for effect in effects}) == 2
    assert activate_ability(game, bearer, item, 'courage', 'Break', confirmed=True, registry=registry)
    assert [entry.effect.key for entry in active_effects(game, bearer, context='Break', registry=registry)] == ['armour']
    assert disable_item(bearer, item, 'Test suppression')
    assert not active_effects(game, bearer, registry=registry)
    bearer.unit.model.reset_characteristics()
    assert bearer.unit.model._base_characteristics == baseline


def test_owner_resolution_is_conservative_and_rider_effects_do_not_reach_mount(registry):
    from magic_items import active_effects, bearer_unavailable
    from models import model

    bearer = live_member()
    bearer.unit.model.attach_mount(model('Chaos Steed', ''))
    item = install_inventory(bearer, [source()])[0]
    game = SimpleNamespace(units=[bearer])
    assert active_effects(game, bearer, registry=registry)
    assert not active_effects(game, bearer, profile=bearer.unit.model.get_mount(), registry=registry)
    item.source['owner_ref'] = 'missing'
    assert bearer_unavailable(bearer, item) == 'roster owner cannot be resolved'
    assert not active_effects(game, bearer, registry=registry)


def test_sourced_banner_grant_tracks_command_loss_and_join_leave():
    from command_groups import install_command, remove_command_casualties
    from magic_items import Scope, active_effects, disable_item

    definition = ItemDefinition('test_banner', 'Test Banner', 'Magic Standards', 'Synthetic test',
                                effects=(ItemEffect('veteran', EffectKind.RULE, 'Veteran', Scope.UNIT_AND_JOINED),))
    registry = ItemRegistry((definition,))
    host = live_member('Regiment', 4)
    character = live_member('Character')
    character.hostUnit = host
    host.joinedCharacter = character
    install_command(host, [{'role': 'champion'}, {'role': 'standard_bearer', 'selection_ref': 'standard'}])
    items = install_inventory(host, [dict(source(), name='Test Banner', category='Magic Standards',
                                         owner_ref='standard', number=2)])
    game = SimpleNamespace(units=[host, character, host])
    assert len(active_effects(game, host, registry=registry)) == 2
    assert len(active_effects(game, character, registry=registry)) == 2
    disable_item(host, items[0], 'One source removed')
    assert len(active_effects(game, character, registry=registry)) == 1
    character.hostUnit = None
    assert not active_effects(game, character, registry=registry)
    assert len(active_effects(game, host, registry=registry)) == 1
    host.unit.nmodels = 1
    remove_command_casualties(host)
    assert not active_effects(game, host, registry=registry)


def test_retired_bearer_keeps_personal_protection_but_cannot_benefit_host():
    from magic_items import Scope, activate_ability, active_effects

    bearer = live_member()
    host = live_member('Host', 4)
    bearer.hostUnit = host
    registry = ItemRegistry((ItemDefinition(
        'test_relic', 'Test Relic', 'Talismans', 'Synthetic test definition',
        effects=(ItemEffect('armour', EffectKind.ARMOUR, 1),
                 ItemEffect('reroll', EffectKind.REROLL, 'Break', Scope.BEARER_AND_UNIT, 'courage')),
        abilities=(ItemAbility('courage', 'Break'),)),))
    item = install_inventory(bearer, [source()])[0]
    game = SimpleNamespace(units=[bearer, host])
    assert active_effects(game, host, context='Break', registry=registry)
    bearer.retiredFromCombat = True
    assert [entry.effect.key for entry in active_effects(game, bearer, registry=registry)] == ['armour']
    assert not active_effects(game, host, context='Break', registry=registry)
    assert not activate_ability(game, bearer, item, 'courage', 'Break', confirmed=True,
                                recipient=host, registry=registry)
    assert item.uses == {}
    bearer.retiredFromCombat = False
    game.units.clear()
    assert not activate_ability(game, bearer, item, 'courage', 'Break', confirmed=True, registry=registry)


def test_turn_token_survives_json_roundtrip_and_live_counter(registry):
    from magic_items import current_turn, inventory_lines

    bearer = live_member()
    game = SimpleNamespace(roundCounter=SimpleNamespace(current_player=2, currentRoundPlayer=[3, 2]))
    assert current_turn(game) == [2, 2]
    item = install_inventory(bearer, [source()])[0]
    assert spend_ability(item, 'turn_power', 'Casting', confirmed=True, turn=(2, 2), registry=registry)
    restore_inventory(bearer, json.loads(json.dumps(save_inventory(bearer))))
    restored = inventory(bearer)[0]
    assert not spend_ability(restored, 'turn_power', 'Casting', confirmed=True, turn=(2, 2), registry=registry)
    assert 'turn_power: spent, 1/1 used' in inventory_lines(bearer, turn=current_turn(game), registry=registry)
    game.roundCounter.currentRoundPlayer[1] += 1
    assert spend_ability(restored, 'turn_power', 'Casting', confirmed=True, turn=current_turn(game), registry=registry)


def test_item_queries_do_not_remove_native_rules_or_other_sources():
    from magic_items import Scope, active_effects, disable_item

    bearer = live_member()
    bearer.unit.model.special_rules.append({'name': 'Veteran', 'veteran': True})
    registry = ItemRegistry((ItemDefinition(
        'test_relic', 'Test Relic', 'Talismans', 'Synthetic test',
        effects=(ItemEffect('veteran', EffectKind.RULE, 'Veteran', Scope.BEARER),)),))
    items = install_inventory(bearer, [source(number=2)])
    game = SimpleNamespace(units=[bearer])
    assert bearer.unit.model.is_veteran()
    initial = active_effects(game, bearer, registry=registry)
    assert len(initial) == 2
    disable_item(bearer, items[0], 'Test source loss')
    remaining = active_effects(game, bearer, registry=registry)
    assert len(remaining) == 1 and remaining[0].source_id == initial[1].source_id
    disable_item(bearer, items[1], 'Other source lost')
    assert not active_effects(game, bearer, registry=registry)
    assert bearer.unit.model.is_veteran()