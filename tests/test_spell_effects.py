"""Source ownership, expiry, and recasting (Rulebook p. 111; FAQ v1.5.3)."""

from types import SimpleNamespace

from spell_effects import caster_removed, end_effect, end_phase, end_turn, recasting, register, start_turn
from spell_system import CurseOfArrowAttractionSpell, OakenShieldSpell, Spell
from tests.test_magic_resistance import _UnitGraphics


def effect_case(cls=OakenShieldSpell, name='Oaken Shield'):
    caster = _UnitGraphics('Mage', nmodels=1)
    target = _UnitGraphics('Target')
    game = SimpleNamespace(player1Units=[caster, target], player2Units=[],
                           roundCounter=SimpleNamespace(current_player=1, currentRoundPlayer=[2, 2]),
                           fsm=SimpleNamespace(endOfTurnSpells=[]), remainsInPlay=[])
    spell = cls(name, 8, game.fsm.endOfTurnSpells, game=game, caster=caster)
    return game, caster, target, spell


def test_expiry_is_owner_start_not_unrelated_end_or_same_turn_reentry():
    game, caster, target, spell = effect_case()
    spell.attach(target, 2)
    start_turn(game)
    end_turn(game)
    game.roundCounter.current_player = 2
    start_turn(game)
    end_turn(game)
    assert spell.rule in target.unit.model.special_rules
    game.roundCounter.current_player = 1
    game.roundCounter.currentRoundPlayer[0] = 3
    start_turn(game)
    assert spell.rule not in target.unit.model.special_rules
    assert not game.fsm.endOfTurnSpells
    end_effect(spell, 'duplicate callback')


def test_identical_ward_sources_share_grant_until_last_source_ends():
    game, caster, target, spell = effect_case()
    spell.attach(target, 2)
    other = OakenShieldSpell(spell.name, 8, game.fsm.endOfTurnSpells, game=game, caster=caster)
    other.attach(target, 2)
    assert other.rule is spell.rule
    assert sum(entry is spell.rule for entry in target.unit.model.special_rules) == 1
    spell.endSpell()
    spell.endSpell()
    assert other.rule in target.unit.model.special_rules
    end_effect(other, 'expired')
    assert other.rule not in target.unit.model.special_rules


def test_identical_hex_sources_do_not_clear_each_other():
    game, caster, target, spell = effect_case(CurseOfArrowAttractionSpell, 'Curse of Arrow Attraction')
    spell.attach(target, 2)
    other = CurseOfArrowAttractionSpell(spell.name, 8, game.fsm.endOfTurnSpells, game=game, caster=caster)
    other.attach(target, 2)
    end_effect(spell, 'expired')
    assert target.unit.model.arrow_attraction
    end_effect(other, 'expired')
    assert not target.unit.model.arrow_attraction


def test_remains_recast_and_caster_loss_do_not_remove_other_casters_or_timed_effects():
    game, caster, target, spell = effect_case()
    spell.attach(target, 2)
    first = Spell('Vortex', 9, game=game, caster=caster)
    second = Spell('Vortex', 9, game=game, caster=target)
    register(first, duration='remains')
    register(second, duration='remains')
    recasting(game, caster, 'Vortex')
    assert game.remainsInPlay == [second]
    caster_removed(game, target)
    assert not game.remainsInPlay
    assert game.fsm.endOfTurnSpells == [spell]


def test_phase_expiry_does_not_expire_on_an_unrelated_phase():
    game, caster, target, spell = effect_case()
    register(spell, target, duration='end_phase', phase='ShootingPhase')
    end_phase(game, 'MovementPhase')
    assert spell in game.fsm.endOfTurnSpells
    end_phase(game, 'ShootingPhase')
    assert not game.fsm.endOfTurnSpells


def test_native_equal_ward_is_never_revoked():
    game, caster, target, spell = effect_case()
    native = {'name': 'Oaken Shield', 'ward': 5}
    target.unit.model.special_rules.append(native)
    spell.attach(target, 2)
    spell.endSpell()
    assert any(rule is native for rule in target.unit.model.special_rules)


def test_host_loss_also_ends_joined_caster_vortex():
    game, caster, target, spell = effect_case()
    target.joinedCharacter = caster
    register(spell, duration='remains')
    caster_removed(game, target)
    assert not game.remainsInPlay


def test_saved_legacy_ticks_are_not_reinterpreted_as_new_start_boundary():
    from spell_system import load_spells, save_spells
    game, caster, target, spell = effect_case()
    spell.attach(target, 1)
    records = save_spells(game)
    records[0].pop('lifecycle')
    spell.endSpell()
    load_spells(game, records, {caster.unitName: caster, target.unitName: target})
    end_turn(game)
    assert not game.fsm.endOfTurnSpells