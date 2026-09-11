"""Actual model-base contact and ground reach drive live combat snapshots (pp. 145-146)."""

import asyncio
from unittest.mock import patch

from direct.interval.IntervalGlobal import Sequence
from panda3d.core import Vec3

from combat_contacts import CombatContactSnapshot
from combat_profiles import combat_profiles
from challenges import duellists
from characters import join_unit
from persistence import load_game_state
from scouts import model_base_boxes
from tests.test_faction_rules_scene import members, scene as scene


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