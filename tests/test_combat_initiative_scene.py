"""Cross-group casualties use one Initiative clock (Rulebook pp. 146, 211)."""

import asyncio
from unittest.mock import patch

import pytest
from direct.interval.IntervalGlobal import Sequence

from challenges import Challenge
from characters import join_unit
from command_groups import champions, install_command
from models import model
from tests.test_corporeal_unmaking_scene import prepare_combat
from tests.test_faction_rules_scene import members, scene as scene


@pytest.mark.parametrize('victim_initiative', [6, 5, 4])
@pytest.mark.parametrize('caster_in_duel', [True, False])
def test_miscast_across_groups_preserves_equal_but_not_lower_attacks(scene, victim_initiative, caster_in_duel):
    app, baseline = scene
    mage, host, enemy = prepare_combat(app, baseline)
    opposing_character = members(app)['Aspiring Champion']
    assert join_unit(app, opposing_character, enemy)
    if caster_in_duel:
        duellist = mage
        victim = host
    else:
        install_command(host, [{'role': 'champion', 'name': 'Test Champion', 'selection_ref': 'test/helms'}])
        host.unit.command_models = {'test/helms': model('Silver Helm', '')}
        duellist = champions(host)[0]
        victim = duellist
    challenge = Challenge(duellist, host, opposing_character, enemy)
    app.attackers, app.defenders = [host, enemy], [enemy, host]
    app.attackSequence = Sequence()
    app.combat._pendingWounds = {}
    app.combat._combatStartModels = {id(member.unit): member.unit.nmodels for member in (host, enemy)}
    removals = Sequence()
    attacks = []
    initial = host.unit.nmodels

    def initiative(profile, **kwargs):
        if profile is mage.unit.model:
            return 5
        if profile is victim.unit.model:
            return victim_initiative
        return 1

    async def cast(game, caster, targets, damage, *, challenge, miscast_damage):
        if caster is mage:
            miscast_damage(victim, app.combat.miscastWoundsRemaining(victim))

    def fight(group, target, **kwargs):
        count = group._attack_count() if callable(group._attack_count) else group._attack_count
        if count:
            attacks.append(group.model)
        return count, 0, 0, 0, 0

    with patch('combat_resolution.strike_initiative', side_effect=initiative), \
            patch('combat_profiles.strike_initiative', side_effect=initiative), \
            patch('assailment.cast_at_initiative', side_effect=cast), \
            patch('combat_resolution.simulate_battle', side_effect=fight):
        scores = asyncio.run(app.combat.resolveCombatWithSpells(challenge, removals))
    assert (victim.unit.model in attacks) is (victim_initiative >= 5)
    assert scores == (0, initial if caster_in_duel else 1, 0, 0)
    assert len(host.model.getChildren()) == initial
    assert victim.unit.nmodels == 0
    removals.finish()
    if caster_in_duel:
        assert host not in app.units
    else:
        assert host.unit.nmodels == len(host.model.getChildren()) == initial - 1
        assert not champions(host)