"""Battle Magic — Rulebook p. 320.

The catalogue supplies each spell's name, casting value, range and wording; it
says nothing about what the spell actually does, so the effects are coded here
and matched back to the catalogue by name.
"""

import asyncio
import json
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from battleFunctions import (check_saves, resolve_magic_hits,  # noqa: E402
                             ward_save_value)
from battlescribe import get_catalogue  # noqa: E402
from models import model  # noqa: E402
from panda3d.core import Point3  # noqa: E402
from psychology import GIVE_GROUND  # noqa: E402
from spell_system import (BATTLE_MAGIC, UNTIL_NEXT_START_OF_TURN,  # noqa: E402
                          ArcaneUrgencySpell, CurseOfArrowAttractionSpell,
                          CurseOfCowardlyFlightSpell, FireballSpell,
                          HammerhandSpell, OakenShieldSpell, PillarOfFireSpell,
                          Spell, load_spells, save_spells, spell_class,
                          spell_readout)


def _model(name="Goblin", **rules):
    """A catalogue model with extra special rules bolted on."""
    m = model(name, "")
    for key, value in rules.items():
        m.special_rules.append({'name': key, key: value})
    return m


def _unit(m, nmodels=10, files=5):
    return SimpleNamespace(model=m, nmodels=nmodels, files=files,
                           name=f"{m.name} Unit")


class TestTheLoreIsWiredUp(unittest.TestCase):
    """Every spell the catalogue lists for Battle Magic has a coded effect."""

    def test_every_catalogue_spell_has_a_class(self):
        spells = get_catalogue().lore("Battle Magic")
        self.assertEqual(len(spells), 7)
        for spell in spells:
            self.assertIsNotNone(spell_class(spell['name']),
                                 f"{spell['name']} has no coded effect")

    def test_the_classes_match_the_spell_types(self):
        self.assertIs(spell_class('Fireball'), FireballSpell)
        self.assertIs(spell_class('Pillar of Fire'), PillarOfFireSpell)
        self.assertIs(spell_class('Hammerhand'), HammerhandSpell)

    def test_an_unknown_spell_has_none(self):
        self.assertIsNone(spell_class('Curse of the Second Breakfast'))
        self.assertIsNone(spell_class(None))

    def test_names_are_matched_after_stripping(self):
        self.assertIs(spell_class('  Oaken Shield '), OakenShieldSpell)

    def test_casting_values_come_from_the_catalogue(self):
        values = {s['name']: s['casting_value']
                  for s in get_catalogue().lore("Battle Magic")}
        self.assertEqual(values['Fireball'], 8)
        self.assertEqual(values['Hammerhand'], 7)
        self.assertEqual(values['Pillar of Fire'], 9)

    def test_the_lore_is_seven_spells_of_the_right_names(self):
        names = {s['name'] for s in get_catalogue().lore("Battle Magic")}
        self.assertEqual(names, set(BATTLE_MAGIC))


class TestWardSaves(unittest.TestCase):
    """Rulebook p. 141."""

    def test_no_ward_by_default(self):
        self.assertEqual(ward_save_value(_model()), 0)

    def test_the_ward_is_read_off_the_rule(self):
        self.assertEqual(ward_save_value(_model(ward=5)), 5)

    def test_two_wards_do_not_combine_the_best_is_used(self):
        m = _model(ward=5)
        m.special_rules.append({'name': 'Talisman', 'ward': 4})
        self.assertEqual(ward_save_value(m), 4)


class TestTheSaveSequence(unittest.TestCase):
    """Armour first, then Ward, then Regeneration (p. 141, p. 176)."""

    def test_a_ward_catches_what_the_armour_missed(self):
        m = _model(ward=5)
        # 1 fails the (absent) armour save, 6 passes the 5+ Ward.
        with mock.patch('battleFunctions.random.randint', side_effect=[1, 6]):
            self.assertTrue(check_saves(m, 7, 0))

    def test_a_failed_ward_is_a_wound(self):
        m = _model(ward=5)
        with mock.patch('battleFunctions.random.randint', side_effect=[1, 2]):
            self.assertFalse(check_saves(m, 7, 0))

    def test_armour_piercing_does_not_touch_the_warding_value(self):
        # A 5+ Ward stays a 5+ however much AP the attack carries.
        m = _model(ward=5)
        with mock.patch('battleFunctions.random.randint', side_effect=[1, 5]):
            self.assertTrue(check_saves(m, 7, 6))

    def test_regeneration_is_the_last_chance(self):
        m = _model(ward=5, regen=4)
        # armour fails, ward fails, Regeneration saves.
        with mock.patch('battleFunctions.random.randint', side_effect=[1, 1, 5]):
            self.assertTrue(check_saves(m, 7, 0))

    def test_a_model_with_nothing_saves_nothing(self):
        with mock.patch('battleFunctions.random.randint', return_value=6):
            self.assertFalse(check_saves(_model(), 7, 0))


class TestMagicHits(unittest.TestCase):
    """A spell's hits are automatic: no To Hit roll, straight To Wound."""

    def test_no_hits_no_wounds(self):
        self.assertEqual(resolve_magic_hits(_unit(_model()), 0, 4, 0), (0, 0, 0))

    def test_every_wound_is_either_saved_or_suffered(self):
        unit = _unit(_model())
        wounds, saves, unsaved = resolve_magic_hits(unit, 20, 4, 0)
        self.assertEqual(wounds, saves + unsaved)
        self.assertLessEqual(wounds, 20)

    def test_strength_four_wounds_toughness_four_on_a_four(self):
        unit = _unit(_model())
        unit.model.characteristics['T'] = 4
        with mock.patch('battleFunctions.random.randint', return_value=4):
            wounds, _, _ = resolve_magic_hits(unit, 6, 4, 0)
        self.assertEqual(wounds, 6)

    def test_strength_three_needs_a_six_against_toughness_five(self):
        unit = _unit(_model())
        unit.model.characteristics['T'] = 5
        with mock.patch('battleFunctions.random.randint', return_value=5):
            wounds, _, _ = resolve_magic_hits(unit, 6, 3, 0)
        self.assertEqual(wounds, 0)


class TestCurseOfArrowAttraction(unittest.TestCase):
    """A Hex lasting until the caster's next Start of Turn."""

    def test_it_flags_and_unflags_the_target(self):
        unit = SimpleNamespace(unit=_unit(_model()))
        spell = CurseOfArrowAttractionSpell('Curse of Arrow Attraction', 7, [])
        spell.affected_unit = unit
        unit.unit.model.arrow_attraction = True
        spell.endSpell()
        self.assertFalse(unit.unit.model.arrow_attraction)

    def test_a_hex_outlives_the_turn_it_was_cast_in(self):
        # One tick would expire it at the end of the caster's own turn.
        self.assertEqual(UNTIL_NEXT_START_OF_TURN, 2)


class TestOakenShield(unittest.TestCase):
    """A 5+ Ward save on the caster and the unit it has joined."""

    def _cast(self):
        unit = SimpleNamespace(unit=_unit(_model()))
        spell = OakenShieldSpell('Oaken Shield', 7, [])
        spell.affected_unit = unit
        spell.rule = {'name': 'Oaken Shield', 'ward': OakenShieldSpell.WARDING_VALUE}
        unit.unit.model.special_rules.append(spell.rule)
        return unit, spell

    def test_the_ward_is_five_up(self):
        unit, _ = self._cast()
        self.assertEqual(ward_save_value(unit.unit.model), 5)

    def test_the_ward_goes_away_again(self):
        unit, spell = self._cast()
        spell.endSpell()
        self.assertEqual(ward_save_value(unit.unit.model), 0)

    def test_ending_it_twice_is_harmless(self):
        unit, spell = self._cast()
        spell.endSpell()
        spell.endSpell()
        self.assertEqual(ward_save_value(unit.unit.model), 0)


class TestSpellDuration(unittest.TestCase):

    def test_a_spell_lasts_one_turn_by_default(self):
        self.assertEqual(Spell('Devil\'s Visit', 7).ticks_remaining, 1)


class TestGiveGround(unittest.TestCase):
    """Curse of Cowardly Flight makes an otherwise-immune unit give ground."""

    def test_giving_ground_is_two_inches(self):
        self.assertEqual(GIVE_GROUND, 2.0)


class TestSpellsSurviveASave(unittest.TestCase):
    """A ward or hex outlives the turn it was cast in, so a quicksave taken
    while one is up has to carry it."""

    def setUp(self):
        self.game = SimpleNamespace(
            fsm=SimpleNamespace(endOfTurnSpells=[]), remainsInPlay=[])
        self.wizard = SimpleNamespace(unitName='Battle Wizard',
                                      unit=_unit(_model('Wizard')))
        self.foe = SimpleNamespace(unitName='Goblin Mob',
                                   unit=_unit(_model()))
        self.unit_map = {u.unitName: u for u in (self.wizard, self.foe)}

    def _round_trip(self, spell):
        records = save_spells(self.game)
        for live in list(self.game.fsm.endOfTurnSpells):
            live.endSpell()
        self.game.fsm.endOfTurnSpells = []
        load_spells(self.game, records, self.unit_map)
        return records

    def test_a_ward_comes_back(self):
        spell = OakenShieldSpell('Oaken Shield', 7, self.game.fsm.endOfTurnSpells,
                                 game=self.game, caster=self.wizard)
        spell.attach(self.wizard, UNTIL_NEXT_START_OF_TURN)
        self._round_trip(spell)
        self.assertEqual(ward_save_value(self.wizard.unit.model), 5)
        self.assertEqual(len(self.game.fsm.endOfTurnSpells), 1)

    def test_a_hex_comes_back(self):
        spell = CurseOfArrowAttractionSpell(
            'Curse of Arrow Attraction', 7, self.game.fsm.endOfTurnSpells,
            game=self.game, caster=self.wizard)
        spell.attach(self.foe, UNTIL_NEXT_START_OF_TURN)
        self._round_trip(spell)
        self.assertTrue(self.foe.unit.model.arrow_attraction)

    def test_the_remaining_duration_is_kept(self):
        spell = OakenShieldSpell('Oaken Shield', 7, self.game.fsm.endOfTurnSpells,
                                 game=self.game, caster=self.wizard)
        spell.attach(self.wizard, 1)
        self._round_trip(spell)
        self.assertEqual(self.game.fsm.endOfTurnSpells[0].ticks_remaining, 1)

    def test_the_record_is_json_safe(self):
        spell = OakenShieldSpell('Oaken Shield', 7, self.game.fsm.endOfTurnSpells,
                                 game=self.game, caster=self.wizard)
        spell.attach(self.wizard, 2)
        json.dumps(save_spells(self.game))

    def test_nothing_in_play_saves_nothing(self):
        self.assertEqual(save_spells(self.game), [])
        load_spells(self.game, None, self.unit_map)
        self.assertEqual(self.game.fsm.endOfTurnSpells, [])


class _SpyGame:
    """Just enough game for a spell to run against."""

    def __init__(self, dispels=False):
        self.dispels = dispels
        self.dispel_calls = 0

    async def dispelAttempt(self, spell, caster):
        self.dispel_calls += 1
        return self.dispels


class _TestSpell(Spell):
    """A spell whose casting always succeeds, recording whether it applied."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.applied = False
        self.order = []

    async def _attempt(self, target):
        self.order.append('cast')
        self.casting = 9
        return True

    async def apply(self, target):
        self.order.append('apply')
        self.applied = True


class TestTheCastingSequence(unittest.TestCase):
    """Cast, then Dispel, then the effect (Rulebook p. 108-110)."""

    def _cast(self, dispels, **kw):
        game = _SpyGame(dispels)
        spell = _TestSpell('Test', 7, [], game=game, caster=object(), **kw)
        spell.game.dispelAttempt = game.dispelAttempt
        asyncio.run(spell.spellFunction(None))
        return game, spell

    def test_a_spell_that_holds_is_applied(self):
        game, spell = self._cast(dispels=False)
        self.assertTrue(spell.applied)
        self.assertEqual(spell.order, ['cast', 'apply'])
        self.assertEqual(game.dispel_calls, 1)

    def test_a_dispelled_spell_never_happens(self):
        # It used to be resolved and then undone, which showed the player
        # damage that was taken back again.
        game, spell = self._cast(dispels=True)
        self.assertFalse(spell.applied)
        self.assertEqual(spell.order, ['cast'])

    def test_a_perfect_invocation_is_not_offered_for_dispel(self):
        game = _SpyGame(dispels=True)
        spell = _TestSpell('Test', 7, [], game=game, caster=object())
        spell.perfect = True

        async def attempt(target):
            spell.casting = 12
            spell.perfect = True
            return True
        spell._attempt = attempt
        asyncio.run(spell.spellFunction(None))
        self.assertEqual(game.dispel_calls, 0)
        self.assertTrue(spell.applied)

    def test_an_illegal_target_is_never_rolled_for(self):
        game = _SpyGame()
        spell = _TestSpell('Test', 7, [], game=game, caster=object())
        spell.canTarget = lambda target: False
        asyncio.run(spell.spellFunction(None))
        self.assertEqual(spell.order, [])


class TestTargetChecks(unittest.TestCase):

    def test_arcane_urgency_needs_a_unit_that_has_moved(self):
        spell = ArcaneUrgencySpell('Arcane Urgency', 9, [])
        unit = SimpleNamespace(unit=_unit(_model()), state='Idle',
                               hasMovedThisTurn=False)
        self.assertFalse(spell.canTarget(unit))
        unit.hasMovedThisTurn = True
        self.assertTrue(spell.canTarget(unit))

    def test_arcane_urgency_will_not_hurry_a_fleeing_unit(self):
        spell = ArcaneUrgencySpell('Arcane Urgency', 9, [])
        unit = SimpleNamespace(unit=_unit(_model()), state='IsFleeing',
                               hasMovedThisTurn=True)
        self.assertFalse(spell.canTarget(unit))

    def test_hammerhand_needs_an_engaged_enemy(self):
        foe = SimpleNamespace(unit=_unit(_model()))
        caster = SimpleNamespace(unit=_unit(_model('Wizard')),
                                 isInCombatWith=[])
        spell = HammerhandSpell('Hammerhand', 7, [], caster=caster)
        self.assertFalse(spell.canTarget(foe))
        caster.isInCombatWith.append(foe)
        self.assertTrue(spell.canTarget(foe))


class TestTheVortexIsPlacedOnTheBoard(unittest.TestCase):
    """A Magical Vortex is aimed at a point, not at a unit."""

    def _caster_at(self, x, y):
        return SimpleNamespace(unit=_unit(_model('Wizard')),
                               bodyNP=SimpleNamespace(
                                   getPos=lambda: Point3(x, y, 0)))

    def test_it_targets_the_ground(self):
        self.assertTrue(PillarOfFireSpell.targets_ground)
        self.assertFalse(FireballSpell.targets_ground)

    def test_a_point_within_range_is_legal(self):
        spell = PillarOfFireSpell('Pillar of Fire', 9, [],
                                  caster=self._caster_at(0, 0))
        self.assertTrue(spell.canTarget(Point3(8, 0, 0)))

    def test_a_point_beyond_twelve_inches_is_refused(self):
        spell = PillarOfFireSpell('Pillar of Fire', 9, [],
                                  caster=self._caster_at(0, 0))
        self.assertFalse(spell.canTarget(Point3(0, 13, 0)))

    def test_the_range_matches_the_catalogue(self):
        reach = next(s['range'] for s in get_catalogue().lore("Battle Magic")
                     if s['name'] == 'Pillar of Fire')
        self.assertEqual(PillarOfFireSpell.RANGE, reach)


class TestTheSpellReadout(unittest.TestCase):
    """The player has to be able to read a spell before casting it."""

    def setUp(self):
        self.spell = next(s for s in get_catalogue().lore("Battle Magic")
                          if s['name'] == 'Fireball')
        self.text = spell_readout('Fireball', self.spell)

    def test_it_names_the_spell_and_its_type(self):
        self.assertIn('Fireball', self.text)
        self.assertIn('Magic Missile', self.text)

    def test_it_gives_the_casting_value_and_range(self):
        self.assertIn('Casting 8+', self.text)
        self.assertIn('24"', self.text)

    def test_it_carries_the_rulebook_wording(self):
        wide = spell_readout('Fireball', self.spell, width=200)
        self.assertIn('2D6 Strength 4 hits', wide)
        self.assertIn('Flaming Attacks', wide)

    def test_it_wraps_to_the_given_width(self):
        for line in spell_readout('Fireball', self.spell, width=30).splitlines():
            self.assertLessEqual(len(line), 40)

    def test_a_self_range_is_printed_as_written(self):
        self.assertIn('Range Self', spell_readout('Oaken Shield',
                                                  {'range': 'Self'}))

    def test_a_spell_with_no_wording_still_reads(self):
        self.assertIn('Mystery', spell_readout('Mystery', {}))


if __name__ == "__main__":
    unittest.main()
