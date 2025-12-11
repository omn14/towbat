from direct.fsm.FSM import FSM
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import TextNode, BitMask32


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
        # Additional logic for Player One's turn can be added here
        for unit in self.game.player1Units:
            unit.bodyNP.setCollideMask(BitMask32.bit(1))
        for unit in self.game.player2Units:
            unit.bodyNP.setCollideMask(BitMask32.bit(7))

    def enterPlayerTwo(self):
        print(f"Entering Player Two's turn in Round {self.currentRoundPlayer[1] + 1}")
        self.current_player = 2
        # Additional logic for Player Two's turn can be added here
        for unit in self.game.player1Units:
            unit.bodyNP.setCollideMask(BitMask32.bit(7))
        for unit in self.game.player2Units:
            unit.bodyNP.setCollideMask(BitMask32.bit(1))

    def next_turn(self):
        if self.current_player == 1:
            self.currentRoundPlayer[0] += 1
            if self.currentRoundPlayer[0] < self.max_rounds:
                self.request('PlayerTwo')
            else:
                print("Game Over! Player One has completed all rounds.")
        elif self.current_player == 2:
            self.currentRoundPlayer[1] += 1
            if self.currentRoundPlayer[1] < self.max_rounds:
                self.request('PlayerOne')
            else:
                print("Game Over! Player Two has completed all rounds.")

    def update_round_display(self):
        
        # Remove existing text if it exists
        if hasattr(self, 'round_text'):
            self.round_text.destroy()
        
        # Create round info text
        round_info = f"Player {self.current_player} | Round {self.currentRoundPlayer[self.current_player-1] + 1}/{self.max_rounds}"
        
        self.round_text = OnscreenText(
            text=round_info,
            pos=(1.3, 0.9),  # Top right corner
            scale=0.07,
            fg=(1, 1, 1, 1),  # White text
            align=TextNode.ARight,
            mayChange=True
        )