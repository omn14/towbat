"""High Magic temporary grants and combat consumers (Rulebook pp. 168, 329)."""

from copy import deepcopy
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from panda3d.core import NodePath

from battleFunctions import attack_characteristic, melee_attacks, ward_save_value
from combat_profiles import CombatProfile
from high_magic import FuryOfKhaineSpell, ShieldOfSapherySpell
from spell_effects import end_turn
from spell_system import CurseOfArrowAttractionSpell, OakenShieldSpell, spell_class
from tests.test_magic_resistance import _UnitGraphics


def test_temporary_attack_grants_do_not_rewrite_or_stack_same_spell():
    target = _UnitGraphics('Target', nmodels=5)
    profile = target.unit.model
    before = deepcopy(profile.characteristics)
    baseline = attack_characteristic(profile)
    grant = {'name': 'Fury of Khaine', 'extra_attacks': 1}
    profile.special_rules.extend([grant, dict(grant)])
    assert attack_characteristic(profile) == baseline + 1
    assert melee_attacks(target.unit, charge=True) == (baseline + 1) * 5
    profile.characteristics = deepcopy(before)
    assert attack_characteristic(profile) == baseline + 1
    assert profile.characteristics == before
    profile.special_rules.remove(grant)
    profile.special_rules.remove(grant)
    assert attack_characteristic(profile) == baseline


def test_split_profile_attack_consumer_uses_temporary_grant():
    target = _UnitGraphics('Target', nmodels=3)
    profile = target.unit.model
    baseline = attack_characteristic(profile)
    profile.special_rules.append({'name': 'Fury of Khaine', 'extra_attacks': 1})
    part = CombatProfile(target, None, profile, 'crew', count=2)
    assert part.attacks(3, 3) == 6 * (baseline + 1)


def spell_case(cls=FuryOfKhaineSpell):
    caster, target, enemy = [_UnitGraphics(name, nmodels=5) for name in ('Mage', 'Target', 'Enemy')]
    root = NodePath('world')
    for unit in (caster, target, enemy):
        unit.bodyNP = root.attachNewNode(unit.unitName)
        unit.unitWidth = unit.unitHeight = 2
        unit.isInCombat = False
        unit.isDeployed = True
    target.bodyNP.setY(8)
    enemy.bodyNP.setY(9)
    game = SimpleNamespace(player1Units=[caster, target], player2Units=[enemy], remainsInPlay=[],
                           fsm=SimpleNamespace(endOfTurnSpells=[]),
                           roundCounter=SimpleNamespace(current_player=1, currentRoundPlayer=[0, 0]))
    name = 'Fury of Khaine' if cls is FuryOfKhaineSpell else 'Shield of Saphery'
    spell = cls(name, 9 if cls is FuryOfKhaineSpell else 8, game=game, caster=caster)
    return game, caster, target, enemy, spell


def test_lookup_grants_expire_at_end_of_current_turn_and_do_not_stack():
    game, caster, target, enemy, spell = spell_case()
    assert spell_class(spell.name) is FuryOfKhaineSpell
    before = attack_characteristic(target.unit.model)
    asyncio.run(spell.apply(target))
    other = FuryOfKhaineSpell(spell.name, 9, game=game, caster=caster)
    asyncio.run(other.apply(target))
    assert attack_characteristic(target.unit.model) == before + 1
    spell.endSpell()
    assert attack_characteristic(target.unit.model) == before + 1
    end_turn(game)
    assert attack_characteristic(target.unit.model) == before


def test_shield_replaces_earlier_enchantments_but_not_hex_or_native_ward():
    game, caster, target, enemy, shield = spell_case(ShieldOfSapherySpell)
    assert spell_class(shield.name) is ShieldOfSapherySpell
    native = {'name': 'Native Ward', 'ward': 4}
    target.unit.model.special_rules.append(native)
    fury = FuryOfKhaineSpell('Fury of Khaine', 9, game=game, caster=caster)
    asyncio.run(fury.apply(target))
    oaken = OakenShieldSpell('Oaken Shield', 7, game.fsm.endOfTurnSpells, game=game, caster=caster)
    oaken.attach(target, 2)
    curse = CurseOfArrowAttractionSpell('Curse of Arrow Attraction', 7, game.fsm.endOfTurnSpells,
                                       game=game, caster=enemy)
    curse.attach(target, 2)
    asyncio.run(shield.apply(target))
    assert fury.ended and oaken.ended and not curse.ended
    assert ward_save_value(target.unit.model) == 4
    shield.endSpell()
    assert any(rule is native for rule in target.unit.model.special_rules)
    assert target.unit.model.arrow_attraction


def test_failed_or_dispelled_shield_does_not_remove_fury():
    game, caster, target, enemy, shield = spell_case(ShieldOfSapherySpell)
    fury = FuryOfKhaineSpell('Fury of Khaine', 9, game=game, caster=caster)
    asyncio.run(fury.apply(target))
    shield._attempt = AsyncMock(return_value=False)
    shield._dispelled = AsyncMock(return_value=True)
    asyncio.run(shield.spellFunction(target))
    assert not fury.ended
    shield._attempt.return_value = True
    asyncio.run(shield.spellFunction(target))
    assert not fury.ended


def test_target_rules_ownership_arc_range_and_engaged_exception():
    game, caster, target, enemy, fury = spell_case()
    shield = ShieldOfSapherySpell('Shield of Saphery', 8, game=game, caster=caster)
    assert fury.canTarget(target) and shield.canTarget(target)
    assert not fury.canTarget(enemy)
    target.bodyNP.setY(-8)
    assert not fury.canTarget(target)
    target.bodyNP.setY(15)
    assert not fury.canTarget(target) and shield.canTarget(target)
    target.bodyNP.setY(8)
    target.isInCombat = True
    assert fury.canTarget(target) and not shield.canTarget(target)


def test_extra_attacks_respects_characteristic_cap():
    game, caster, target, enemy, fury = spell_case()
    target.unit.model.characteristics['A'] = 10
    asyncio.run(fury.apply(target))
    assert attack_characteristic(target.unit.model) == 10


def test_supporting_attacks_remain_one_each_with_fury():
    from models import model
    game, caster, target, enemy, fury = spell_case()
    target.unit.model = model('High Elf Spearman', '')
    target.unit.nmodels = 15
    target.unit.files = 5
    before = melee_attacks(target.unit, charge=False)
    asyncio.run(fury.apply(target))
    assert melee_attacks(target.unit, charge=False) == before + 5


def test_target_boundary_world_space_and_all_round_vision():
    game, caster, target, enemy, fury = spell_case()
    target.bodyNP.setY(14)
    assert fury.canTarget(target)
    target.bodyNP.setY(14.01)
    assert not fury.canTarget(target)
    caster.bodyNP.setPos(20, 10, 0)
    caster.bodyNP.setH(90)
    target.bodyNP.setPos(12, 10, 0)
    assert fury.canTarget(target)
    target.bodyNP.setPos(28, 10, 0)
    assert not fury.canTarget(target)
    caster.unit.model.has_all_round_vision = lambda: True
    assert fury.canTarget(target)