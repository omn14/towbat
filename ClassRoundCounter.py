from direct.fsm.FSM import FSM
from panda3d.core import BitMask32


class RoundCounter(FSM):
    def __init__(self, Game, max_rounds):
        FSM.__init__(self, 'counterFSM')
        self.game=Game
        self.max_rounds = max_rounds
        self.nPlayers = 2
        self.currentRoundPlayer = [0] * self.nPlayers
        self.current_player = 1  # 1 for Player One, 2 for Player Two
        self.request('PlayerOne')  # Start with Player One
        self.update_round_display()


    def enterPlayerOne(self):
        print(f"Entering Player One's turn in Round {self.currentRoundPlayer[0] + 1}")
        self.current_player = 1
        self.apply_selection_masks()

    def enterPlayerTwo(self):
        print(f"Entering Player Two's turn in Round {self.currentRoundPlayer[1] + 1}")
        self.current_player = 2
        self.apply_selection_masks()

    def apply_selection_masks(self):
        """Put every unit back on the mask a selection click looks for.

        Shooting and magic move their targets onto targeting masks, and a shot
        moves both parties onto the melee one. Those have to be undone when a
        phase ends as well as when the turn does, or the click that should
        select a unit in the next phase finds a target instead.
        """
        for units, player in ((self.game.player1Units, 1),
                              (self.game.player2Units, 2)):
            mask = BitMask32.bit(1) if player == self.current_player \
                else BitMask32.bit(7)
            for unit in list(units):
                if unit.bodyNP.isEmpty():
                    units.remove(unit)
                else:
                    unit.bodyNP.setCollideMask(mask)

    @property
    def finished(self):
        return all(rounds >= self.max_rounds for rounds in self.currentRoundPlayer)

    @property
    def final_turn(self):
        """A round includes both players' turns (Rulebook pp. 114, 286)."""
        return not self.finished and all(rounds + (index == self.current_player - 1) >= self.max_rounds
                                         for index, rounds in enumerate(self.currentRoundPlayer))

    def next_turn(self):
        if self.finished:
            return False
        self.currentRoundPlayer[self.current_player - 1] += 1
        if self.finished:
            print("Game Over! Both players have completed all rounds.")
            return False
        self.request('PlayerTwo' if self.current_player == 1 else 'PlayerOne')
        return True

    def update_round_display(self):
        """Publish the turn state; the HUD owns the widget that shows it."""
        messenger.send('hud-turn', [
            self.current_player,
            min(self.max_rounds, self.currentRoundPlayer[self.current_player - 1] + 1),
            self.max_rounds,
        ])