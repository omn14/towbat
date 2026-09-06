"""Challenges — Rulebook p. 210-211."""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import combat_resolution  # noqa: E402
from challenges import (Challenge, MAX_OVERKILL, add_challenge,  # noqa: E402
                        can_accept, can_issue, duellist, end_challenge,
                        find_challenge, may_refuse, overkill_bonus,
                        refusal_barred, surrounded, wounds_remaining)
from combat_resolution import CombatResolver  # noqa: E402


def _graphics(name, category=None, wounds=1, taken=0, nmodels=1):
    chars = {'Model': name, 'W': str(wounds)}
    if category:
        chars['Category'] = category
    return SimpleNamespace(
        unitName=f"{name} Unit",
        unit=SimpleNamespace(name=f"{name} Unit", nmodels=nmodels, files=5,
                             model=SimpleNamespace(characteristics=chars,
                                                   special_rules=[])),
        woundsOnModel=taken, joinedCharacter=None, retiredFromCombat=False,
        isInCombatFlank=[])


def _character(name="Captain", wounds=2, taken=0):
    return _graphics(name, category='Characters', wounds=wounds, taken=taken)


def _regiment(name="Spearmen", nmodels=20, character=None):
    unit = _graphics(name, nmodels=nmodels)
    if character is not None:
        unit.joinedCharacter = character
        character.hostUnit = unit
    return unit


class FindingADuellistTests(unittest.TestCase):
    """Only characters — the engine has no champions."""

    def test_a_units_joined_character_is_its_duellist(self):
        captain = _character()
        unit = _regiment(character=captain)
        self.assertIs(duellist(unit), captain)
        self.assertTrue(can_issue(unit))
        self.assertTrue(can_accept(unit))

    def test_a_unit_with_no_character_has_none(self):
        unit = _regiment()
        self.assertIsNone(duellist(unit))
        self.assertFalse(can_issue(unit))
        # p. 210: any challenge issued goes unanswered.
        self.assertFalse(can_accept(unit))

    def test_a_lone_character_is_its_own_duellist(self):
        captain = _character()
        self.assertIs(duellist(captain), captain)

    def test_a_retired_character_cannot_be_nominated_again(self):
        captain = _character()
        unit = _regiment(character=captain)
        captain.retiredFromCombat = True
        self.assertIsNone(duellist(unit))

    def test_nothing_has_no_duellist(self):
        self.assertIsNone(duellist(None))


class SurroundedTests(unittest.TestCase):
    """Engaged in all four arcs (p. 211)."""

    def test_a_front_and_rear_and_two_flanks_counts(self):
        unit = _regiment()
        unit.isInCombatFlank = ['front', 'flank', 'flank', 'rear']
        self.assertTrue(surrounded(unit))

    def test_three_arcs_is_not_enough(self):
        unit = _regiment()
        unit.isInCombatFlank = ['front', 'flank', 'rear']
        self.assertFalse(surrounded(unit))

    def test_an_unengaged_unit_is_not_surrounded(self):
        self.assertFalse(surrounded(_regiment()))


class NowhereToRunTests(unittest.TestCase):
    """When a challenge cannot be refused (p. 211)."""

    def test_an_ordinary_character_in_a_unit_may_refuse(self):
        captain = _character()
        unit = _regiment(character=captain)
        self.assertIsNone(refusal_barred(captain, unit))
        self.assertTrue(may_refuse(captain, unit))

    def test_a_model_not_in_a_unit_cannot_refuse(self):
        captain = _character()
        self.assertEqual(refusal_barred(captain, None), "it is not part of a unit")
        self.assertFalse(may_refuse(captain, None))

    def test_the_last_model_in_a_unit_cannot_refuse(self):
        captain = _character()
        unit = _regiment(nmodels=1, character=captain)
        self.assertEqual(refusal_barred(captain, unit),
                         "it is the last model in its unit")

    def test_a_surrounded_unit_cannot_refuse(self):
        captain = _character()
        unit = _regiment(character=captain)
        unit.isInCombatFlank = ['front', 'flank', 'flank', 'rear']
        self.assertEqual(refusal_barred(captain, unit),
                         "its unit is engaged in all four arcs")


class WoundsRemainingTests(unittest.TestCase):

    def test_an_unhurt_model_has_its_whole_profile(self):
        self.assertEqual(wounds_remaining(_character(wounds=3)), 3)

    def test_wounds_already_taken_come_off(self):
        self.assertEqual(wounds_remaining(_character(wounds=3, taken=2)), 1)

    def test_it_never_goes_below_zero(self):
        self.assertEqual(wounds_remaining(_character(wounds=1, taken=4)), 0)


class OverkillTests(unittest.TestCase):
    """Excess unsaved wounds are combat result, to a maximum of +5 (p. 211)."""

    def test_no_bonus_if_the_rival_survives(self):
        self.assertEqual(overkill_bonus(1, 2), 0)

    def test_no_bonus_for_a_clean_kill(self):
        self.assertEqual(overkill_bonus(2, 2), 0)

    def test_each_excess_wound_is_a_point(self):
        self.assertEqual(overkill_bonus(5, 2), 3)

    def test_the_bonus_is_capped(self):
        self.assertEqual(overkill_bonus(20, 1), MAX_OVERKILL)

    def test_a_model_with_nothing_left_still_scores(self):
        self.assertEqual(overkill_bonus(3, 0), 3)


class ChallengeRecordTests(unittest.TestCase):

    def setUp(self):
        self.a = _character("Captain")
        self.b = _character("Champion")
        self.host_a = _regiment("Spearmen", character=self.a)
        self.host_b = _regiment("Orcs", character=self.b)
        self.challenge = Challenge(self.a, self.host_a, self.b, self.host_b)

    def test_an_unanswered_challenge_has_no_accepter(self):
        lonely = Challenge(self.a, self.host_a)
        self.assertFalse(lonely.answered)
        self.assertEqual(lonely.participants(), [self.a])

    def test_the_participants_are_the_two_duellists(self):
        self.assertTrue(self.challenge.answered)
        self.assertEqual(self.challenge.participants(), [self.a, self.b])
        self.assertTrue(self.challenge.involves(self.a))
        self.assertTrue(self.challenge.involves(self.b))

    def test_a_bystander_is_not_involved(self):
        self.assertFalse(self.challenge.involves(_character("Someone")))
        self.assertFalse(self.challenge.involves(None))

    def test_each_duellist_faces_the_other(self):
        self.assertIs(self.challenge.opponent_of(self.a), self.b)
        self.assertIs(self.challenge.opponent_of(self.b), self.a)
        self.assertIsNone(self.challenge.opponent_of(_character("Someone")))

    def test_the_hosts_are_both_units(self):
        self.assertEqual(self.challenge.hosts(), [self.host_a, self.host_b])


class ToTheDeathTests(unittest.TestCase):
    """A challenge carries into the next turn (p. 211)."""

    def setUp(self):
        self.game = SimpleNamespace(challenges=[])
        self.a = _character("Captain")
        self.b = _character("Champion")
        self.host_a = _regiment("Spearmen", character=self.a)
        self.host_b = _regiment("Orcs", character=self.b)
        self.challenge = Challenge(self.a, self.host_a, self.b, self.host_b)

    def test_a_live_challenge_is_found_from_either_unit(self):
        add_challenge(self.game, self.challenge)
        self.assertIs(find_challenge(self.game, self.host_a), self.challenge)
        self.assertIs(find_challenge(self.game, self.host_b), self.challenge)

    def test_an_uninvolved_unit_finds_nothing(self):
        add_challenge(self.game, self.challenge)
        self.assertIsNone(find_challenge(self.game, _regiment("Bystanders")))

    def test_a_resolved_challenge_is_gone(self):
        add_challenge(self.game, self.challenge)
        end_challenge(self.game, self.challenge)
        self.assertIsNone(find_challenge(self.game, self.host_a))

    def test_a_game_with_no_challenges_yet_is_fine(self):
        self.assertIsNone(find_challenge(SimpleNamespace(), self.host_a))


class DuelResolutionTests(unittest.TestCase):
    """Fighting a challenge, and what it scores (p. 211)."""

    def setUp(self):
        self.a = _character("Captain", wounds=2)
        self.b = _character("Champion", wounds=2)
        self.host_a = _regiment("Spearmen", character=self.a)
        self.host_b = _regiment("Orcs", character=self.b)
        for model, initiative in ((self.a, 5), (self.b, 3)):
            model.unit.model.characteristics['I'] = str(initiative)
            model.unit.model.equipedWeapon = {'name': 'Hand Weapon', 'tag': 'combat'}
            model.hostUnit = None      # slain by removeModelsFromUnit, not detached
        self.challenge = Challenge(self.a, self.host_a, self.b, self.host_b)
        self.slain = []
        self.resolver = object.__new__(CombatResolver)
        self.resolver.game = SimpleNamespace(
            player1Units=[self.host_a], player2Units=[self.host_b],
            challenges=[self.challenge],
            movement=SimpleNamespace(
                removeModelsFromUnit=lambda u, n: self.slain.append(u)))
        self.resolver.chariotParts = lambda model: []

    def _fight(self, wounds_by_name):
        """Resolve the duel with each model's wounds fixed rather than rolled."""
        def fake(unit, target, charge=False, **kw):
            n = wounds_by_name.get(unit.name, 0)
            return 1, 1, n, 0, n
        with mock.patch.object(combat_resolution, 'simulate_battle', fake):
            return self.resolver.resolveChallenge(self.challenge)

    def test_an_unanswered_challenge_scores_nothing(self):
        self.assertEqual(
            self.resolver.resolveChallenge(Challenge(self.a, self.host_a)),
            (0, 0, 0, 0))

    def test_both_duellists_wound_each_other(self):
        p1, p2, ok1, ok2 = self._fight({'Captain Unit': 1, 'Champion Unit': 1})
        self.assertEqual((p1, p2), (1, 1))
        self.assertEqual((ok1, ok2), (0, 0))

    def test_the_higher_initiative_strikes_first_and_can_end_it(self):
        # The Captain is I5 against I3 and kills outright, so the Champion's
        # own attacks are never made.
        p1, p2, _, _ = self._fight({'Captain Unit': 2, 'Champion Unit': 5})
        self.assertEqual(p1, 2)
        self.assertEqual(p2, 0, "a slain duellist should not strike back")

    def test_overkill_is_scored_for_the_excess(self):
        p1, p2, ok1, ok2 = self._fight({'Captain Unit': 5, 'Champion Unit': 0})
        self.assertEqual(p1, 5)
        self.assertEqual(ok1, 3, "5 wounds against 2 remaining is +3")
        self.assertEqual(ok2, 0)

    def test_a_resolved_challenge_leaves_play(self):
        self._fight({'Captain Unit': 2, 'Champion Unit': 0})
        self.assertEqual(self.resolver.game.challenges, [])

    def test_a_challenge_both_survive_stays_in_play(self):
        self._fight({'Captain Unit': 1, 'Champion Unit': 1})
        self.assertEqual(self.resolver.game.challenges, [self.challenge])

    def test_the_scores_are_reported_per_player(self):
        # The challenger's host is player 2's, so its wounds must land second.
        self.resolver.game.player1Units = [self.host_b]
        self.resolver.game.player2Units = [self.host_a]
        p1, p2, _, _ = self._fight({'Captain Unit': 1, 'Champion Unit': 0})
        self.assertEqual((p1, p2), (0, 1))


class WoundingADuellistTests(unittest.TestCase):
    """A joined character could not be wounded at all before challenges."""

    def setUp(self):
        self.resolver = object.__new__(CombatResolver)
        self.removed = []
        self.resolver.game = SimpleNamespace(
            movement=SimpleNamespace(
                removeModelsFromUnit=lambda u, n: self.removed.append((u, n))))

    def test_a_wound_short_of_the_profile_only_marks_the_model(self):
        captain = _character(wounds=3)
        captain.hostUnit = None
        self.assertFalse(self.resolver.woundDuellist(captain, 2))
        self.assertEqual(captain.woundsOnModel, 2)
        self.assertEqual(self.removed, [])

    def test_enough_wounds_slay_it(self):
        captain = _character(wounds=2)
        captain.hostUnit = None
        self.assertTrue(self.resolver.woundDuellist(captain, 2))
        self.assertEqual(self.removed, [(captain, 1)])

    def test_wounds_already_taken_count_towards_the_kill(self):
        captain = _character(wounds=3, taken=2)
        captain.hostUnit = None
        self.assertTrue(self.resolver.woundDuellist(captain, 1))

    def test_no_wounds_changes_nothing(self):
        captain = _character(wounds=2)
        self.assertFalse(self.resolver.woundDuellist(captain, 0))
        self.assertEqual(captain.woundsOnModel, 0)

    def test_a_joined_character_is_taken_out_of_its_host(self):
        captain = _character(wounds=1)
        host = _regiment(character=captain)
        captain.hostUnit = host
        with mock.patch.object(combat_resolution, 'slay_character') as slain:
            self.assertTrue(self.resolver.woundDuellist(captain, 1))
        slain.assert_called_once()
        self.assertEqual(self.removed, [], "not an ordinary casualty")


if __name__ == '__main__':
    unittest.main()
