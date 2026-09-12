"""Actual model-base contact and ground reach drive live combat snapshots (pp. 145-146)."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from direct.interval.IntervalGlobal import Sequence
from panda3d.core import Vec3

from combat_contacts import CombatContactSnapshot
from combat_profiles import combat_profiles
from challenges import duellists
from command_groups import champions
from characters import join_unit
from persistence import load_game_state
from scouts import model_base_boxes
from tests.test_faction_rules_scene import members, scene as scene
from tests.test_shieldwall_scene import combat_tasks


def edge_contact(host, enemy):
    host.bodyNP.setPos(0, 0, 0)
    host.bodyNP.setH(0)
    enemy.bodyNP.setPos(0, 0, 0)
    enemy.bodyNP.setH(180)
    own, other = model_base_boxes(host)[0], model_base_boxes(enemy)[0]
    enemy.bodyNP.setPos(Vec3(own[0] - other[0], own[1] + own[3] + other[3] - other[1], 0))
    for member, opponent in ((host, enemy), (enemy, host)):
        member.isInCombatWith = [opponent]
        member.isInCombatFlank = ['front']
        member.isInCombat = True
        member.hasAttackedThisTurn = False


@pytest.mark.parametrize('target_name', ['Chaos Knight', 'Mage'])
def test_dragon_princes_fury_reaches_live_attack_counts(scene, target_name, capsys):
    from high_magic import FuryOfKhaineSpell
    from spell_effects import end_turn
    app, baseline = scene
    load_game_state(app, baseline)
    roster = members(app)
    host, enemy = roster['Dragon Prince'], roster[target_name]
    edge_contact(host, enemy)
    parts = combat_profiles(host, enemy)
    snapshot = CombatContactSnapshot([host, enemy])
    before = {part.role: snapshot.attacks(part, host.unit.nmodels) for part in parts}
    spell = FuryOfKhaineSpell('Fury of Khaine', 9, game=app, caster=roster['Mage'])
    asyncio.run(spell.apply(host))
    contact_count = sum(position.contact for position in snapshot.positions(host, enemy)[1])
    expected = {role: attacks + contact_count for role, attacks in before.items()}
    assert {part.role: snapshot.attacks(part, host.unit.nmodels) for part in parts} == expected
    assert expected == ({'main': 9, 'mount': 6} if target_name == 'Chaos Knight'
                        else {'main': 5, 'mount': 4})
    app.attackers, app.defenders = [host], [enemy]
    app.attackSequence = Sequence()
    app.combat._pendingWounds = {}
    app.combat._combatStartModels = {id(member.unit): member.unit.nmodels for member in (host, enemy)}
    observed = {}

    def fight(group, target, **kwargs):
        count = group._attack_count() if callable(group._attack_count) else group._attack_count
        if group.model in (host.unit.model, host.unit.model.get_mount()):
            name = 'main' if group.model is host.unit.model else 'mount'
            observed[name] = observed.get(name, 0) + count
        return count, 0, 0, 0, 0

    with patch('combat_resolution.simulate_battle', side_effect=fight), \
            patch.object(app, 'aiControls', return_value=True), patch('assailment.cast_at_initiative'):
        asyncio.run(app.combat.resolveCombatWithSpells(None, Sequence()))
    assert observed == expected
    output = capsys.readouterr().out
    assert 'A2 -> A3 (Fury of Khaine +1)' in output
    assert 'A1 -> A2 (Fury of Khaine +1)' in output
    assert ('0' if target_name == 'Chaos Knight' else '2') + ' noncontact bases limited' in output
    end_turn(app)
    assert {part.role: snapshot.attacks(part, host.unit.nmodels) for part in parts} == before


def test_live_fighting_rank_uses_ground_m_and_noncontact_one_attack(scene, capsys):
    app, baseline = scene
    load_game_state(app, baseline)
    host, enemy = members(app)['Chaos Warrior'], members(app)['Mage']
    host.unit.files = host.unit.nmodels
    host.layOutRanks()
    edge_contact(host, enemy)
    profile = host.unit.model
    with patch.object(profile, 'characteristics', {**profile.characteristics, 'A': '3', 'M': '1'}), \
            patch.object(profile, 'special_rules', [*profile.special_rules,
                         {'name': 'Fly (20)', 'fly': True, 'fly_movement': 20}]):
        assert profile.get_fly_movement() > profile.get_movement()
        part, = combat_profiles(host, enemy)
        assert CombatContactSnapshot([host, enemy]).attacks(part, host.unit.nmodels) == 5
        app.attackers, app.defenders = [host, enemy], [enemy, host]
        app.attackSequence = Sequence()
        app.combat._pendingWounds = {}
        app.combat._combatStartModels = {id(member.unit): member.unit.nmodels for member in (host, enemy)}
        counts = []
        def fight(group, target, **kwargs):
            count = group._attack_count() if callable(group._attack_count) else group._attack_count
            if group.model is profile:
                counts.append(count)
            return count, 0, 0, 0, 0
        with patch('combat_resolution.simulate_battle', side_effect=fight), \
                patch('assailment.cast_at_initiative'):
            assert asyncio.run(app.combat.resolveCombatWithSpells(None, Sequence())) == (0, 0, 0, 0)
        assert counts == [5]
    assert 'ground M1' in capsys.readouterr().out


def test_champion_is_not_duplicated_and_later_mounts_lose_fallen_bases(scene):
    app, baseline = scene
    load_game_state(app, baseline)
    host, enemy = members(app)['Chaos Knight'], members(app)['Mage']
    edge_contact(host, enemy)
    parts = {part.role: part for part in combat_profiles(host, enemy)}
    snapshot = CombatContactSnapshot([host, enemy])
    before = {role: snapshot.attacks(part, 4) for role, part in parts.items()}
    assert before == {'main': 3, 'mount': 4, 'champion': 2}
    assert snapshot.attacks(parts['mount'], 3) == 3
    assert snapshot.attacks(parts['champion'], 3) == 2


def test_target_geometry_refresh_waits_for_next_initiative(scene):
    app, baseline = scene
    load_game_state(app, baseline)
    host, enemy = members(app)['Mage'], members(app)['Chaos Knight']
    enemy.unit.files = 1
    enemy.layOutRanks()
    edge_contact(host, enemy)
    snapshot = CombatContactSnapshot([host, enemy])
    original = snapshot.target_boxes(enemy)
    app.combat.previewCombatWounds(enemy, 2)
    assert snapshot.target_boxes(enemy) == original
    snapshot.refresh()
    assert len(snapshot.target_boxes(enemy)) == 2
    command_indices = snapshot.formations[id(enemy)][2]
    assert list(dict(snapshot.targets[id(enemy)])) == list(command_indices)[:2]
    assert len(enemy.model.getChildren()) == 4


@pytest.mark.parametrize('casualty_initiative,expected', [(6, 2), (5, 4), (4, 4)])
def test_live_initiative_uses_surviving_enemy_footprint(scene, casualty_initiative, expected):
    app, baseline = scene
    load_game_state(app, baseline)
    host, enemy, caster = (members(app)[name] for name in ('Silver Helm', 'Chaos Knight', 'Mage'))
    assert join_unit(app, caster, host)
    host.layOutRanks()
    host.placeCharacter()
    edge_contact(host, enemy)
    snapshot = CombatContactSnapshot([host, enemy])
    app.attackers, app.defenders = [host, enemy], [enemy, host]
    app.attackSequence = Sequence()
    app.combat._pendingWounds = {}
    app.combat._combatStartModels = {id(member.unit): member.unit.nmodels for member in (host, enemy)}
    observed = []

    def initiative(profile, **kwargs):
        if profile is caster.unit.model:
            return casualty_initiative
        return 5 if profile is host.unit.model else 1

    async def cast(game, member, targets, damage, **kwargs):
        if member is caster:
            damage(enemy, 2)

    def fight(group, target, **kwargs):
        if group.model is host.unit.model:
            observed.append(len(snapshot.target_boxes(enemy)))
        return group._attack_count, 0, 0, 0, 0

    with patch('combat_profiles.strike_initiative', side_effect=initiative), \
            patch('assailment.cast_at_initiative', side_effect=cast), \
            patch('combat_resolution.simulate_battle', side_effect=fight), \
            patch.object(app, 'aiControls', return_value=True):
        asyncio.run(app.combat.resolveCombatWithSpells(None, Sequence(), contacts=snapshot))
    assert observed and set(observed) == {expected}
    assert enemy.unit.nmodels == 2 and len(enemy.model.getChildren()) == 4


def test_dead_joined_character_base_leaves_next_target_snapshot(scene):
    app, baseline = scene
    load_game_state(app, baseline)
    host, character, enemy = (members(app)[name] for name in ('Chaos Knight', 'Aspiring Champion', 'Mage'))
    assert join_unit(app, character, host)
    host.layOutRanks()
    host.placeCharacter()
    edge_contact(enemy, host)
    snapshot = CombatContactSnapshot([host, enemy])
    initial = len(snapshot.target_boxes(host))
    app.combat.previewMiscastWounds(character, app.combat.miscastWoundsRemaining(character), Sequence())
    assert len(snapshot.target_boxes(host)) == initial
    snapshot.refresh()
    assert len(snapshot.target_boxes(host)) == initial - 1
    assert all(index < host.unit.nmodels for index, _ in snapshot.targets[id(host)])


def test_horsemen_support_reaches_only_next_rank_and_not_mounts(scene):
    app, baseline = scene
    load_game_state(app, baseline)
    host, enemy = members(app)['Marauder Horsemen'], members(app)['Mage']
    host.skirmishCombat = True
    host.unit.files = 3
    host.unit.ranks = 2
    host.layOutRanks()
    host.chargedThisTurn = True
    host.unit.model.equip_weapon('Throwing Spear')
    edge_contact(host, enemy)
    parts = {part.role: part for part in combat_profiles(host, enemy)}
    snapshot = CombatContactSnapshot([host, enemy])
    assert snapshot.attacks(parts['main'], 5) == 5
    assert snapshot.attacks(parts['mount'], 5) == 3
    host.isInCombatFlank = ['rear']
    assert snapshot.attacks(parts['main'], 5) == 3


def test_challenge_candidate_must_be_within_or_adjacent_to_fighting_rank(scene):
    app, baseline = scene
    load_game_state(app, baseline)
    host, character, enemy = members(app)['Chaos Knight'], members(app)['Aspiring Champion'], members(app)['Mage']
    assert join_unit(app, character, host)
    host.unit.files = 1
    host.layOutRanks()
    host.placeCharacter()
    edge_contact(host, enemy)
    assert host.characterSlot >= 3
    assert character not in duellists(host)
    assert len(duellists(host)) == 1
    host.unit.files = 4
    host.layOutRanks()
    host.placeCharacter()
    edge_contact(host, enemy)
    assert character in duellists(host)


def test_two_enemy_units_share_each_models_attack_budget(scene):
    app, baseline = scene
    load_game_state(app, baseline)
    host, first, second = (members(app)[name] for name in ('Chaos Warrior', 'Mage', 'Silver Helm'))
    app.movement.removeModelsFromUnit(second, second.unit.nmodels - 1)
    host.unit.files = host.unit.nmodels
    host.layOutRanks()
    edge_contact(host, first)
    second.bodyNP.setPos(first.bodyNP.getPos())
    second.bodyNP.setH(180)
    own = sorted(model_base_boxes(host), key=lambda box: box[0])[-2]
    other = model_base_boxes(second)[0]
    second.bodyNP.setPos(second.bodyNP.getPos() + Vec3(
        own[0] - other[0], own[1] + own[3] + other[3] - other[1], 0))
    host.isInCombatWith = [first, second]
    host.isInCombatFlank = ['front', 'front']
    second.isInCombatWith, second.isInCombatFlank = [host], ['front']
    second.isInCombat, second.hasAttackedThisTurn = True, False
    part, = combat_profiles(host, first)
    with patch.object(part.profile, 'characteristics', {**part.profile.characteristics, 'A': '1', 'M': '10'}):
        snapshot = CombatContactSnapshot([host, first, second])
        allocation = snapshot.allocation(part, host.unit.nmodels)
        assert sum(count for _, count, _ in allocation.batches) == host.unit.nmodels
        assert any(targets == [first] for _, _, targets in allocation.batches)
        assert any(targets == [second] for _, _, targets in allocation.batches)
        with patch.object(app, 'makeChoiceNew', AsyncMock(side_effect=lambda choices, *args, **kwargs: choices[0])):
            asyncio.run(allocation.resolve(app))
        assert {id(target) for target, _ in allocation.attacks} == {id(first), id(second)}
        assert sum(count for _, count in allocation.attacks) == host.unit.nmodels
        app.attackers, app.defenders = [host, first, second], [first, host, host]
        app.attackSequence = Sequence()
        app.combat._pendingWounds = {}
        app.combat._combatStartModels = {id(member.unit): member.unit.nmodels for member in (host, first, second)}
        directed = []
        def fight(group, target, **kwargs):
            if group.model is part.profile:
                directed.append((target, group._attack_count))
            return group._attack_count, 0, 0, 0, 0
        with patch('combat_resolution.simulate_battle', side_effect=fight), \
                patch('assailment.cast_at_initiative'), \
                patch.object(app, 'makeChoiceNew', AsyncMock(side_effect=lambda choices, *args, **kwargs: choices[0])):
            assert asyncio.run(app.combat.resolveCombatWithSpells(None, Sequence())) == (0, 0, 0, 0)
        assert {id(target) for target, _ in directed} == {id(first.unit), id(second.unit)}
        assert sum(count for _, count in directed) == host.unit.nmodels


def test_joined_character_in_rear_fighting_rank_gets_live_attacks(scene):
    app, baseline = scene
    load_game_state(app, baseline)
    host, character, enemy = (members(app)[name] for name in ('Chaos Knight', 'Aspiring Champion', 'Mage'))
    assert join_unit(app, character, host)
    host.unit.files = 1
    host.layOutRanks()
    host.placeCharacter()
    edge_contact(host, enemy)
    assert host.characterSlot >= 3
    parts = [part for part in combat_profiles(host, enemy) if part.role == 'character']
    assert len(parts) == 1
    part = parts[0]
    assert CombatContactSnapshot([host, enemy]).attacks(part, host.unit.nmodels) == 0
    own, other = model_base_boxes(host)[-1], model_base_boxes(enemy)[0]
    enemy.bodyNP.setPos(enemy.bodyNP.getPos() + Vec3(
        own[0] - other[0], own[1] - own[3] - other[3] - other[1], 0))
    host.isInCombatFlank = ['rear']
    snapshot = CombatContactSnapshot([host, enemy])
    attacks = int(character.unit.model.characteristics['A'])
    assert snapshot.attacks(part, host.unit.nmodels) == attacks
    app.attackers, app.defenders = [host, enemy], [enemy, host]
    app.attackSequence = Sequence()
    app.combat._pendingWounds = {}
    app.combat._combatStartModels = {id(member.unit): member.unit.nmodels for member in (host, enemy)}
    counts = []
    def fight(group, target, **kwargs):
        if group.model is character.unit.model:
            counts.append(group._attack_count)
        return group._attack_count, 0, 0, 0, 0
    with patch('combat_resolution.simulate_battle', side_effect=fight), \
            patch('assailment.cast_at_initiative'), \
            patch.object(app, 'makeChoiceNew', AsyncMock(side_effect=lambda choices, *args, **kwargs: choices[0])):
        assert asyncio.run(app.combat.resolveCombatWithSpells(None, Sequence())) == (0, 0, 0, 0)
    assert counts == [attacks]


@pytest.mark.parametrize('kind', ['champion', 'character'])
def test_specific_targets_require_contact_and_directed_damage_cannot_spill(scene, kind):
    app, baseline = scene
    load_game_state(app, baseline)
    host, enemy, character = (members(app)[name] for name in ('Mage', 'Chaos Knight', 'Aspiring Champion'))
    if kind == 'character':
        assert join_unit(app, character, enemy)
        enemy.layOutRanks()
        enemy.placeCharacter()
        victim = character
        victim_index = enemy.unit.nmodels
    else:
        victim, = champions(enemy)
        victim_index = next(index for index, entry in CombatContactSnapshot([enemy]).formations[id(enemy)][2].items()
                            if entry is victim.command_entry)
    edge_contact(host, enemy)
    own, other = model_base_boxes(host)[0], model_base_boxes(enemy)[victim_index]
    host.bodyNP.setPos(host.bodyNP.getPos() + Vec3(other[0] - own[0], other[1] - other[3] - own[3] - own[1], 0))
    part, = combat_profiles(host, enemy)
    allocation = CombatContactSnapshot([host, enemy]).allocation(part, 1)
    assert any(victim in targets for _, _, targets in allocation.batches)
    host.bodyNP.setX(host.bodyNP.getX() + 20)
    distant = CombatContactSnapshot([host, enemy]).allocation(part, 1)
    assert all(victim not in targets for _, _, targets in distant.batches)
    before = enemy.unit.nmodels
    remaining = int(victim.unit.model.characteristics['W']) - victim.woundsOnModel
    removals = Sequence()
    app.combat._pendingWounds = {}
    with combat_tasks(app) as run, \
            patch('combat_resolution.simulate_battle', return_value=(10, 10, 10, 0, 10)), \
            patch('combat_resolution.take_last_slaying_blows', return_value=0):
        assert app.combat.resolveProfileAttacks(part, victim, 10, 5, None, removals) == remaining
        assert victim.unit.nmodels == 0
        assert enemy.unit.nmodels == before - (kind == 'champion')
        async def apply_removals():
            await removals
        run(apply_removals())
    assert enemy.unit.nmodels == before - (kind == 'champion')
    assert len(enemy.model.getChildren()) == enemy.unit.nmodels


def test_move_through_rear_rank_preserves_command_and_returns_after_combat(scene, tmp_path, capsys):
    from characters import move_through_ranks
    from command_groups import command_positions
    from persistence import save_game_state
    app, baseline = scene
    load_game_state(app, baseline)
    host, character, enemy = (members(app)[name] for name in ('Chaos Warrior', 'Aspiring Champion', 'Mage'))
    assert join_unit(app, character, host)
    host.request('InCombat')
    edge_contact(host, enemy)
    previous = host.characterSlot
    own = min(model_base_boxes(host), key=lambda box: box[1])
    other = model_base_boxes(enemy)[0]
    enemy.bodyNP.setPos(enemy.bodyNP.getPos() + Vec3(
        own[0] - other[0], own[1] - own[3] - other[3] - other[1], 0))
    host.isInCombatFlank = ['rear']
    command = command_positions(host)
    with patch.object(app, 'makeChoiceNew', AsyncMock(side_effect=lambda options, *args, **kwargs: options[1])) as choice:
        asyncio.run(move_through_ranks(app, [host, enemy]))
    choice.assert_awaited_once()
    assert host.characterSlot >= host.unit.files
    assert host.characterSlot not in command.values()
    assert host.characterCombatReturnSlot == previous
    destination = host.characterSlot
    host.layOutRanks()
    host.placeCharacter()
    assert host.characterSlot == destination
    assert command_positions(host) == command
    part = next(part for part in combat_profiles(host, enemy) if part.role == 'character')
    assert CombatContactSnapshot([host, enemy]).attacks(part, host.unit.nmodels) == 1
    saved = save_game_state(app, str(tmp_path / 'rank-move.json'))
    assert saved is not None
    for _ in range(2):
        load_game_state(app, saved)
        host = members(app)['Chaos Warrior']
        assert host.characterSlot == destination
        assert host.characterCombatReturnSlot == previous
    host.request('Idle')
    assert host.characterSlot == previous
    assert host.characterCombatReturnSlot is None
    assert 'Moving Through the Ranks' in capsys.readouterr().out


@pytest.mark.parametrize('active_player', [1, 2])
def test_rank_move_choices_start_with_inactive_player_and_may_be_declined(scene, active_player, capsys):
    from characters import move_through_ranks, side_of
    app, baseline = scene
    load_game_state(app, baseline)
    armies = members(app)
    first, second = armies['Elven Archer'], armies['Chaos Warrior']
    assert join_unit(app, armies['Mage'], first)
    assert join_unit(app, armies['Aspiring Champion'], second)
    for host in (first, second):
        host.unit.files = 3
        host.layOutRanks()
        host.placeCharacter()
    edge_contact(first, second)
    own = min(model_base_boxes(first), key=lambda box: box[1])
    other = max(model_base_boxes(second), key=lambda box: box[1])
    second.bodyNP.setPos(second.bodyNP.getPos() + Vec3(
        own[0] - other[0], own[1] - own[3] - other[3] - other[1], 0))
    first.isInCombatFlank = second.isInCombatFlank = ['rear']
    before = [model_base_boxes(host) for host in (first, second)]
    with patch.object(app.roundCounter, 'current_player', active_player), \
            patch.object(app, 'aiControls', return_value=False), \
            patch.object(app, 'makeChoiceNew', AsyncMock(return_value='Stay in place')) as choice:
        asyncio.run(move_through_ranks(app, [first, second]))
    assert [side_of(app, call.kwargs['owner']) for call in choice.await_args_list] == [3 - active_player, active_player]
    assert [model_base_boxes(host) for host in (first, second)] == before
    assert 'declines' in capsys.readouterr().out