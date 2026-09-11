"""Walk Between Worlds dependencies (Rulebook pp. 167, 172, 177, 329)."""

from types import SimpleNamespace
from unittest.mock import patch
import asyncio

import pytest

from battleFunctions import resolve_magic_hits, simulate_attack, simulate_battle
from tests.test_faction_rules import profile
from tests.test_move_through_cover import troops as troops


@pytest.mark.parametrize('source', ['mundane', 'keyword', 'weapon', 'ensorcelled', 'lance'])
def test_ethereal_only_allows_magical_attack_wounds(source):
    attacker, defender = profile('Killing Blow'), profile('Ethereal')
    if source == 'keyword':
        attacker = profile('Killing Blow', 'Magical Attacks')
    elif source == 'weapon':
        attacker.equipedWeapon['magical'] = True
    elif source in ('ensorcelled', 'lance'):
        attacker = profile('Killing Blow', 'Ensorcelled Weapons')
        if source == 'lance':
            attacker.equipedWeapon = {'name': 'Lance', 'tag': 'combat'}
    magical = source in ('keyword', 'weapon', 'ensorcelled')
    with patch('battleFunctions.random.randint', return_value=6):
        assert simulate_attack(attacker, defender) == (True, magical)
    assert bool(attacker.slaying_blow) == magical


def test_ethereal_batch_reports_prevented_wounds_once(capsys):
    attacker = SimpleNamespace(name='Warriors', model=profile(), nmodels=5, files=5)
    defender = SimpleNamespace(name='Spirits', model=profile('Ethereal'), nmodels=5, files=5)
    with patch('battleFunctions.random.randint', return_value=6):
        assert simulate_battle(attacker, defender, charge=True) == (5, 5, 0, 0, 0)
    output = capsys.readouterr().out
    assert output.count('Ethereal') == 1 and 'prevents 5 wound(s)' in output


def test_spells_can_wound_ethereal_models():
    defender = SimpleNamespace(name='Spirits', model=profile('Ethereal'))
    with patch('battleFunctions.random.randint', return_value=6):
        assert resolve_magic_hits(defender, 2, 5, 0) == (2, 0, 2)


def test_impact_hits_do_not_borrow_magical_weapon():
    from battleFunctions import resolve_impact_hits
    attacker = SimpleNamespace(name='Chariot', model=profile('Impact Hits (2)'), nmodels=1, files=1)
    defender = SimpleNamespace(name='Spirits', model=profile('Ethereal'), nmodels=5, files=5)
    attacker.model.equipedWeapon = dict(attacker.model.equipedWeapon, magical=True)
    with patch('battleFunctions.random.randint', return_value=6):
        assert resolve_impact_hits(attacker, defender) == (2, 0, 0, 0)
        attacker.model.special_rules.append({'name': 'Magical Attacks'})
        assert resolve_impact_hits(attacker, defender) == (2, 2, 0, 2)


@pytest.mark.parametrize('magical', [False, True])
def test_artillery_damage_uses_explicit_magical_source(magical):
    from cannon_fire import CannonFire
    from bombardment import Bombardment
    defender = SimpleNamespace(unit=SimpleNamespace(name='Spirits', model=profile('Ethereal')),
                               model=SimpleNamespace(getChildren=lambda: [object()]))
    cannon = CannonFire.__new__(CannonFire)
    bombardment = Bombardment.__new__(Bombardment)
    with patch('battleFunctions.random.randint', return_value=6):
        assert cannon._apply_wounds(defender, 1, 10, 3, magical=magical) == (
            (1, 1, 0) if magical else (0, 0, 0))
        assert bombardment._wound_unsaved(defender.unit.model, 10, 3, magical=magical) == magical


def test_ethereal_ignores_terrain_penalties_and_danger_not_other_members(troops, capsys):
    from panda3d.core import Point3
    from special_rules import apply_rule_keywords
    movement, host, character = troops
    apply_rule_keywords(host.unit.model, ['Ethereal'], replace=True)
    host.unit.model.characteristics['M'] = character.unit.model.characteristics['M'] = '4'
    assert movement.movementAllowance(host, Point3(0), Point3(1)) == 3
    with patch('terrain_system.random.randint', return_value=1) as dice:
        assert movement.dangerousTerrainTests(host, Point3(0), Point3(1)) == 1
    assert dice.call_count == 1
    apply_rule_keywords(character.unit.model, ['Ethereal'], replace=True)
    assert movement.movementAllowance(host, Point3(0), Point3(1), log=True) == 4
    with patch('terrain_system.random.randint') as dice:
        assert movement.dangerousTerrainTests(host, Point3(0), Point3(1)) == 0
    dice.assert_not_called()
    assert 'Ethereal' in capsys.readouterr().out


def test_reserve_move_majority_and_saved_movement_restrictions(troops):
    from reserve_move import has_majority, record_movement, unavailable
    movement, host, character = troops
    game = movement.game
    game.units = [host, character]
    game.chargeDeclarations = []
    game.roundCounter = SimpleNamespace(current_player=1, currentRoundPlayer=[0, 0])
    host.state = character.state = 'Moved'
    character.hostUnit = host
    host.unit.model.special_rules.append({'name': 'Walk Between Worlds', 'reserve_move': True})
    assert has_majority(host)
    assert unavailable(game, host) is None
    host.unit.nmodels = 1
    assert not has_majority(host)
    character.unit.model.special_rules.append({'name': 'Reserve Move'})
    assert has_majority(host)
    host.fledThisPhase = True
    record_movement(game)
    host.fledThisPhase = False
    assert unavailable(game, host) == 'fled during Movement'


@pytest.mark.parametrize('field', ['marchedThisTurn', 'chargedThisTurn', 'attemptedRallyThisTurn'])
def test_reserve_move_rejects_forbidden_prior_movement(troops, field):
    from reserve_move import unavailable
    movement, host, character = troops
    game = movement.game
    game.roundCounter = SimpleNamespace(current_player=1, currentRoundPlayer=[0, 0])
    host.state = 'Moved'
    host.unit.model.special_rules.append({'name': 'Reserve Move'})
    setattr(host, field, True)
    assert unavailable(game, host) is not None


def test_walk_is_self_conveyance_survives_turn_end_and_shield_until_owner_start():
    from high_magic import ShieldOfSapherySpell, WalkBetweenWorldsSpell
    from special_rules import is_ethereal
    from spell_effects import end_turn, start_turn
    from spell_system import spell_class
    from tests.test_high_magic import spell_case
    game, caster, host, enemy, unused = spell_case()
    walk = WalkBetweenWorldsSpell('Walk Between Worlds', 10, game=game, caster=caster)
    assert spell_class(walk.name) is WalkBetweenWorldsSpell
    assert walk.canTarget(caster) and not walk.canTarget(host)
    caster.hostUnit = host
    host.joinedCharacter = caster
    asyncio.run(walk.apply(caster))
    assert is_ethereal(caster.unit.model) and is_ethereal(host.unit.model)
    shield = ShieldOfSapherySpell('Shield of Saphery', 8, game=game, caster=caster)
    asyncio.run(shield.apply(host))
    assert not walk.ended
    end_turn(game)
    assert is_ethereal(host.unit.model)
    game.roundCounter.current_player = 2
    start_turn(game)
    assert is_ethereal(host.unit.model)
    game.roundCounter.current_player = 1
    game.roundCounter.currentRoundPlayer = [1, 1]
    start_turn(game)
    assert not is_ethereal(host.unit.model) and not is_ethereal(caster.unit.model)


def test_self_host_grants_follow_detachment_and_retirement():
    from high_magic import WalkBetweenWorldsSpell
    from special_rules import is_ethereal
    from spell_effects import refresh_self_spells
    from tests.test_high_magic import spell_case
    game, caster, host, enemy, unused = spell_case()
    caster.hostUnit = host
    host.joinedCharacter = caster
    walk = WalkBetweenWorldsSpell('Walk Between Worlds', 10, game=game, caster=caster)
    asyncio.run(walk.apply(caster))
    caster.retiredFromCombat = True
    refresh_self_spells(caster)
    assert not is_ethereal(host.unit.model) and is_ethereal(caster.unit.model)
    caster.retiredFromCombat = False
    refresh_self_spells(caster)
    assert is_ethereal(host.unit.model)
    caster.hostUnit = None
    refresh_self_spells(caster)
    assert not is_ethereal(host.unit.model) and is_ethereal(caster.unit.model)