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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from persistence import save_game_state  # noqa: E402

# Flags the rules reset each turn or each battle. Every one has to survive a
# save; anything missing keeps whatever the running session happened to have.
TURN_FLAGS = (
    'hasMovedThisTurn', 'hasAttackedThisTurn', 'attemptedRallyThisTurn',
    'chargedThisTurn', 'countsAsChargedNextTurn', 'chargeDistance',
    'cannotChargeThisTurn', 'panicTestedThisPhase', 'fledThisPhase',
    'usedStubborn', 'isDisrupted', 'woundsOnModel', 'spellsCastThisTurn',
    'cannotCastThisTurn', 'cannotPursueThisTurn', 'moveSpentThisTurn',
    'manoeuvreThisTurn', 'redressDelta',
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


if __name__ == "__main__":
    unittest.main()
