"""Saving and loading a battle.

A quicksave has to carry everything the rules track per turn, or reloading
quietly hands the player a different game — a Wizard that has used up its
casting allowance, a unit that has already charged, and so on.
"""

import json
import os
import re
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import persistence  # noqa: E402
from models import model  # noqa: E402
from persistence import save_game_state  # noqa: E402
from special_rules import apply_rule_keywords  # noqa: E402

# Flags the rules reset each turn or each battle. Every one has to survive a
# save; anything missing keeps whatever the running session happened to have.
TURN_FLAGS = (
    'hasMovedThisTurn', 'hasAttackedThisTurn', 'attemptedRallyThisTurn',
    'chargedThisTurn', 'countsAsChargedNextTurn', 'chargeDistance',
    'cannotChargeThisTurn', 'panicTestedThisPhase', 'fledThisPhase',
    'usedStubborn', 'isDisrupted', 'woundsOnModel', 'spellsCastThisTurn',
    'cannotCastThisTurn', 'cannotPursueThisTurn', 'moveSpentThisTurn',
    'manoeuvreThisTurn', 'redressDelta', 'marchedThisTurn',
)


def _model():
    return SimpleNamespace(
        name='Battle Wizard',
        characteristics={'M': '4', 'W': '1', 'Points': 90},
        armor_save=7, armour=[], charging=False, weapons={}, spells={},
        equipedWeapon=None,
        wizard_level=lambda default=0: 2,
        is_mounted=lambda: False,
        get_mount=lambda: None,
    )


def _unit(name='Battle Wizard'):
    body = SimpleNamespace(getPos=lambda: (1.0, 2.0, 0.0), getH=lambda: 90.0,
                           getP=lambda: 0.0, getR=lambda: 0.0)
    unit = SimpleNamespace(
        unitName=name, bodyNP=body, state='Idle', color=(1, 0, 0, 1),
        isInCombat=False, isDeployed=True, isInCombatWith=[],
        isInCombatFlank=[], joinedCharacter=None, isGeneral=False, isBSB=False,
        startOfBattleModels=1, startOfPhaseModels=1, startOfPhaseEngaged=False,
        unit=SimpleNamespace(nmodels=1, files=1, ranks=1, model=_model()),
    )
    for flag in TURN_FLAGS:
        setattr(unit, flag, False)
    unit.chargeDistance = 0.0
    unit.woundsOnModel = 0
    unit.spellsCastThisTurn = []
    unit.moveSpentThisTurn = 0.0
    unit.manoeuvreThisTurn = None
    unit.redressDelta = 0
    return unit


def _game(units):
    return SimpleNamespace(
        fsm=SimpleNamespace(getCurrentOrNextState=lambda: 'StrategyPhase',
                            phases=['StrategyPhase'], currentPhaseIndex=0,
                            endOfTurnSpells=[]),
        roundCounter=SimpleNamespace(currentRoundPlayer=1, current_player=1,
                                     max_rounds=6),
        AIplayer2=SimpleNamespace(active=False),
        remainsInPlay=[], units=units, player1Units=units, player2Units=[],
    )


class TestSavingTurnState(unittest.TestCase):

    def setUp(self):
        self.unit = _unit()
        self.game = _game([self.unit])
        self.tmp = tempfile.mkdtemp()

    def _save(self):
        path = os.path.join(self.tmp, 'save.json')
        save_game_state(self.game, path)
        with open(path) as f:
            return json.load(f)['units'][0]

    def test_every_turn_flag_is_written(self):
        saved = self._save()
        for flag in TURN_FLAGS:
            self.assertIn(flag, saved, f"{flag} is not saved")

    def test_the_flag_list_has_not_fallen_behind_the_unit(self):
        """TURN_FLAGS is hand-written, so a flag added to `units.py` slips past
        the test above unnoticed. Anything named like a per-turn flag on a real
        unit has to be listed here, and so saved."""
        import units
        source = open(units.__file__, encoding='utf-8').read()
        named = set(re.findall(
            r'self\.(\w*(?:ThisTurn|ThisPhase|NextTurn))\s*=', source))
        missing = named - set(TURN_FLAGS)
        self.assertFalse(
            missing,
            f"per-turn flags on the unit that are neither listed nor saved: "
            f"{sorted(missing)}")

    def test_a_wizards_spent_spells_are_recorded(self):
        # Reloading used to leave the running session's list in place, so a
        # spell attempted after the save could not be attempted again.
        self.unit.spellsCastThisTurn = ['Fireball', 'Oaken Shield']
        self.assertEqual(self._save()['spellsCastThisTurn'],
                         ['Fireball', 'Oaken Shield'])

    def test_everything_the_turn_resets_is_saved(self):
        """The start-of-turn reset in `game_fsm.py` is what *defines* per-turn
        state, so anything cleared there has to be saved. A name pattern is not
        enough: `redressDelta` is reset with the rest and reads like a counter,
        and went unsaved for exactly that reason."""
        path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'game_fsm.py')
        src = open(path, encoding='utf-8').read()
        block = re.search(r'def enterStrategyPhase\(self\):(.*?)\n    def ',
                          src, re.S)
        self.assertIsNotNone(block, "could not find the start-of-turn reset")
        reset = set(re.findall(r'unit\.(\w+)\s*=(?!=)', block.group(1)))
        saved = set(self._save())
        self.assertFalse(
            reset - saved,
            f"cleared at the start of every turn but not saved: "
            f"{sorted(reset - saved)}")

    def test_the_manoeuvre_allowance_survives(self):
        # Without these a reload handed back the half Movement a redress had
        # spent and let the unit manoeuvre a second time in the same move.
        self.unit.moveSpentThisTurn = 2.0
        self.unit.manoeuvreThisTurn = 'Redress the Ranks'
        self.unit.redressDelta = 3
        saved = self._save()
        self.assertEqual(saved['moveSpentThisTurn'], 2.0)
        self.assertEqual(saved['manoeuvreThisTurn'], 'Redress the Ranks')
        self.assertEqual(saved['redressDelta'], 3)

    def test_a_spent_wizard_is_recorded(self):
        self.unit.cannotCastThisTurn = True
        self.assertTrue(self._save()['cannotCastThisTurn'])

    def test_the_spell_list_is_a_copy_not_a_reference(self):
        self.unit.spellsCastThisTurn = ['Fireball']
        saved = self._save()
        self.unit.spellsCastThisTurn.append('Hammerhand')
        self.assertEqual(saved['spellsCastThisTurn'], ['Fireball'])

    def test_the_whole_save_is_json(self):
        self.unit.spellsCastThisTurn = ['Fireball']
        json.dumps(self._save())


class TheReloadedProfileTests(unittest.TestCase):
    """A save is the source of truth for the stats it stores.

    `reset_characteristics()` runs at the end of every exchange, and it restores
    `_base_characteristics` — which a fresh model fills from the catalogue. A
    load that set only `characteristics` therefore held its saved profile until
    the first combat and then silently reverted to the bare catalogue entry,
    taking any roster change with it.
    """

    def test_a_restored_profile_survives_a_reset(self):
        m = model('Grave Guard', '')
        m.characteristics['A'] = '4'
        m._base_characteristics = dict(m.characteristics)
        m.characteristics['A'] = '9'    # a temporary combat modifier
        m.reset_characteristics()
        self.assertEqual(m.characteristics['A'], '4')

    def test_the_catalogue_profile_is_what_it_reverts_to_otherwise(self):
        m = model('Grave Guard', '')
        catalogue_attacks = m.characteristics['A']
        m.characteristics['A'] = '4'
        m.reset_characteristics()
        self.assertEqual(m.characteristics['A'], catalogue_attacks)

    def test_the_load_rebases_the_profile_it_restores(self):
        """Source-level, because the assignment sits inside `load_game_state`,
        which needs a running app. Reverting the one line would otherwise pass
        every test in this file."""
        src = open(persistence.__file__, encoding='utf-8').read()
        block = re.search(
            r"model\.characteristics = unit_data\['characteristics'\](.{0,400})",
            src, re.S)
        self.assertIsNotNone(block, "the load no longer restores characteristics")
        self.assertIn('_base_characteristics', block.group(1),
                      "a loaded profile is not rebased, so it reverts to the "
                      "catalogue at the end of the first combat")


class TheSavesFolderTests(unittest.TestCase):
    """Saves live in saves/, so the working tree stays clear of them."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        patcher = mock.patch.object(persistence, 'SAVE_DIR', self.tmp)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _write(self, name, when=None):
        path = os.path.join(self.tmp, name)
        with open(path, 'w') as f:
            f.write('{}')
        if when is not None:
            os.utime(path, (when, when))
        return path

    def test_a_bare_name_goes_in_the_saves_folder(self):
        self.assertEqual(persistence.save_path('quicksave.json'),
                         os.path.join(self.tmp, 'quicksave.json'))

    def test_a_name_with_a_directory_is_left_alone(self):
        self.assertEqual(persistence.save_path('elsewhere/quicksave.json'),
                         'elsewhere/quicksave.json')

    def test_an_absolute_path_is_left_alone(self):
        self.assertEqual(persistence.save_path('/tmp/quicksave.json'),
                         '/tmp/quicksave.json')

    def test_saving_writes_into_the_folder(self):
        unit = _unit()
        persistence.save_game_state(_game([unit]), 'quicksave.json')
        self.assertTrue(os.path.exists(os.path.join(self.tmp, 'quicksave.json')))

    def test_the_folder_is_created_if_it_is_missing(self):
        nested = os.path.join(self.tmp, 'fresh')
        with mock.patch.object(persistence, 'SAVE_DIR', nested):
            persistence.save_game_state(_game([_unit()]), 'quicksave.json')
            self.assertTrue(os.path.exists(os.path.join(nested, 'quicksave.json')))

    def test_the_newest_save_is_listed_first(self):
        self._write('old.json', when=1000)
        self._write('new.json', when=2000)
        self.assertEqual(persistence.list_saves(), ['new.json', 'old.json'])

    def test_backups_and_temp_files_are_not_offered(self):
        self._write('quicksave.json')
        self._write('quicksave.json.bak')
        self._write('quicksave.json.tmp')
        self.assertEqual(persistence.list_saves(), ['quicksave.json'])

    def test_a_missing_folder_lists_nothing(self):
        with mock.patch.object(persistence, 'SAVE_DIR',
                               os.path.join(self.tmp, 'nope')):
            self.assertEqual(persistence.list_saves(), [])

    def test_the_label_drops_the_extension_and_dates_the_save(self):
        self._write('quicksave.json', when=0)
        label = persistence.save_label('quicksave.json')
        self.assertTrue(label.startswith('quicksave'))
        self.assertNotIn('.json', label)


class RoleKeywordsSurviveALoadTests(unittest.TestCase):
    """A save carries the roster's whole list, so loading one replaces it."""

    def _model(self, *keywords):
        m = model("State Missile Trooper", "")
        apply_rule_keywords(m, list(keywords))
        return m

    def _names(self, m):
        return {r.get('name') for r in m.special_rules if isinstance(r, dict)}

    def test_a_granted_rule_is_taken_away_by_the_next_save(self):
        # Loading a save with Strike First and then one without used to leave
        # the unit striking first for the rest of the session.
        m = self._model("Strike First")
        self.assertTrue(m.has_strike_first())
        apply_rule_keywords(m, [], replace=True)
        self.assertFalse(m.has_strike_first())
        self.assertNotIn('Strike First', self._names(m))

    def test_the_rules_the_save_does_name_are_kept(self):
        m = self._model("Strike First", "Fire & Flee")
        apply_rule_keywords(m, ["Fire & Flee"], replace=True)
        self.assertFalse(m.has_strike_first())
        self.assertTrue(m.has_fire_and_flee())

    def test_the_keyword_list_matches_the_save(self):
        m = self._model("Strike First")
        apply_rule_keywords(m, ["Skirmishers"], replace=True)
        self.assertEqual(m.characteristics['Special Rules'], ["Skirmishers"])

    def test_a_save_can_still_grant_a_rule(self):
        m = self._model()
        apply_rule_keywords(m, ["Strike Last"], replace=True)
        self.assertTrue(m.has_strike_last())

    def test_merging_is_still_available_for_an_army_list(self):
        m = self._model("Skirmishers")
        apply_rule_keywords(m, ["Strike First"])
        self.assertIn('Skirmishers', m.characteristics['Special Rules'])
        self.assertTrue(m.has_strike_first())


if __name__ == "__main__":
    unittest.main()
