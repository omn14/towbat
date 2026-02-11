"""
Integration example: Using MinimaxTree with the existing AI system
This shows how to enhance ClassAI with minimax decision-making
"""

from gameStateTree import GameState, GameAction
from minimaxOptimizations import OptimizedMinimaxTree
from gameStateAnalyzer import GameStateAnalyzer
from treeVisualization import DecisionExplainer, TreeVisualizer


class EnhancedAI:
    """
    Enhanced AI that uses minimax with alpha-beta pruning for decision making.
    Can replace or supplement the existing ClassAI.
    """
    
    def __init__(self, game, player_units, enemy_units, player_num=2, 
                 use_minimax=True, minimax_depth=3):
        self.game = game
        self.player_units = player_units
        self.enemy_units = enemy_units
        self.player_num = player_num
        self.active = False
        
        # Initialize analyzer and tree
        self.analyzer = GameStateAnalyzer(game)
        self.use_minimax = use_minimax
        self.minimax_depth = minimax_depth
        
        if use_minimax:
            # Use optimized tree with transposition table, move ordering, and iterative deepening
            self.tree = OptimizedMinimaxTree(
                self.analyzer, 
                max_depth=minimax_depth,
                use_transposition_table=True,
                use_move_ordering=True,
                use_iterative_deepening=True
            )
        
        # Statistics
        self.decisions_made = 0
        self.minimax_decisions = 0
        self.heuristic_decisions = 0
    
    def make_decision(self):
        """
        Main decision-making function.
        Decides between minimax (slow but optimal) or heuristics (fast but suboptimal).
        """
        # Capture current game state
        current_state = GameState.from_game(self.game)
        
        # Use minimax for critical decisions, heuristics for simple ones
        if self.use_minimax and self._should_use_minimax(current_state):
            return self._minimax_decision(current_state)
        else:
            return self._heuristic_decision(current_state)
    
    def _should_use_minimax(self, state: GameState) -> bool:
        """
        Determine if minimax should be used for this decision.
        Use minimax for important decisions, heuristics for trivial ones.
        """
        return True  # For simplicity, always use minimax
        # Always use minimax in combat phase
        if state.current_phase == 'CombatPhase':
            return True
        
        # Use minimax when army strength is close
        player_units = state.get_player_units(self.player_num)
        enemy_units = state.get_player_units(3 - self.player_num)
        
        p_strength = sum(u['nmodels'] for u in player_units)
        e_strength = sum(u['nmodels'] for u in enemy_units)
        
        strength_ratio = p_strength / max(1, e_strength)
        
        # Use minimax when game is close (ratio between 0.7 and 1.4)
        if 0.7 <= strength_ratio <= 1.4:
            return True
        
        # Use heuristics when we're clearly winning or losing
        return False
    
    def _minimax_decision(self, state: GameState) -> GameAction:
        """Make decision using minimax algorithm"""
        self.minimax_decisions += 1
        self.decisions_made += 1
        
        # Find best move with time limit (iterative deepening)
        # Set time_limit based on game urgency (in seconds)
        time_limit = 150.0  # Adjust as needed
        best_action, expected_value = self.tree.find_best_move_timed(state, time_limit)
        
        #self.tree.print_tree(self.tree.root.best_child)  # Optional: Print the tree for debugging
        # Get statistics
        stats = self.tree.get_statistics()
        print(f"\n[AI Minimax] Player {self.player_num}")
        print(f"  Best action: {best_action}")
        print(f"  Expected value: {expected_value:.2f}")
        print(f"  Nodes evaluated: {stats['nodes_evaluated']}")
        print(f"  Nodes pruned: {stats['nodes_pruned']}")
        print(f"  Pruning efficiency: {stats['pruning_efficiency']:.1f}%")
        
        # Show transposition table stats if available
        if 'tt_hit_rate' in stats:
            print(f"  TT size: {stats['tt_size']} | Hit rate: {stats['tt_hit_rate']:.1f}%")
        
        return best_action
    
    def _heuristic_decision(self, state: GameState) -> GameAction:
        """Make decision using fast heuristics (existing AI logic)"""
        self.heuristic_decisions += 1
        self.decisions_made += 1
        
        # Use existing evaluation functions for quick decisions
        evaluation = self.analyzer.evaluate_overall_state(self.player_num)
        strategy = self.analyzer.suggest_strategy(self.player_num)
        
        print(f"\n[AI Heuristic] Player {self.player_num}")
        print(f"  Assessment: {evaluation['assessment']}")
        print(f"  Strategy: {strategy}")
        
        # Generate a simple action based on strategy
        if "AGGRESSIVE" in strategy:
            return self._aggressive_action(state)
        elif "DEFENSIVE" in strategy:
            return self._defensive_action(state)
        else:
            return self._balanced_action(state)
    
    def _aggressive_action(self, state: GameState) -> GameAction:
        """Generate aggressive action - prioritize attacks and charges"""
        player_units = state.get_player_units(self.player_num)
        enemy_units = state.get_player_units(3 - self.player_num)
        
        if state.current_phase == 'MovementPhase':
            # Move towards nearest enemy
            for unit in player_units:
                if not unit['hasMovedThisTurn']:
                    # Find nearest enemy
                    nearest_enemy = min(enemy_units, 
                        key=lambda e: self._distance(unit['position'], e['position']))
                    
                    # Move towards enemy
                    dx = nearest_enemy['position'][0] - unit['position'][0]
                    dy = nearest_enemy['position'][1] - unit['position'][1]
                    
                    # Normalize and scale by movement
                    dist = (dx**2 + dy**2)**0.5
                    if dist > 0:
                        move_dist = 8  # Simplified movement
                        dx = (dx / dist) * move_dist
                        dy = (dy / dist) * move_dist
                    
                    return GameAction('move', unit['name'], {
                        'target_x': unit['position'][0] + dx,
                        'target_y': unit['position'][1] + dy
                    })
        
        elif state.current_phase == 'ShootingPhase':
            # Shoot at weakest enemy
            for unit in player_units:
                
                if not unit['hasAttackedThisTurn']:
                    weakest = min(enemy_units, key=lambda e: e['nmodels'])
                    return GameAction('shoot', unit['name'], {'target': weakest['name']})
        
        elif state.current_phase == 'CombatPhase':
            # Attack in combat
            for unit in player_units:
                if unit['isInCombat'] and not unit['hasAttackedThisTurn']:
                    if unit['isInCombatWith']:
                        return GameAction('attack', unit['name'], 
                                        {'target': unit['isInCombatWith'][0]})
        
        return GameAction('end_phase', 'system', {})
    
    def _defensive_action(self, state: GameState) -> GameAction:
        """Generate defensive action - consolidate and protect"""
        player_units = state.get_player_units(self.player_num)
        
        if state.current_phase == 'MovementPhase':
            # Move towards center/friendly units
            for unit in player_units:
                if not unit['hasMovedThisTurn']:
                    # Move to battlefield center
                    dx = -unit['position'][0] * 0.3  # Gentle move towards center
                    dy = -unit['position'][1] * 0.3
                    
                    return GameAction('move', unit['name'], {
                        'target_x': unit['position'][0] + dx,
                        'target_y': unit['position'][1] + dy
                    })
        
        elif state.current_phase == 'ShootingPhase':
            # Shoot at nearest threat
            enemy_units = state.get_player_units(3 - self.player_num)
            for unit in player_units:
                if not unit['hasAttackedThisTurn']:
                    nearest = min(enemy_units, 
                        key=lambda e: self._distance(unit['position'], e['position']))
                    return GameAction('shoot', unit['name'], {'target': nearest['name']})
        
        return GameAction('end_phase', 'system', {})
    
    def _balanced_action(self, state: GameState) -> GameAction:
        """Generate balanced action - opportunistic"""
        # Mix of aggressive and defensive
        player_units = state.get_player_units(self.player_num)
        enemy_units = state.get_player_units(3 - self.player_num)
        
        if state.current_phase == 'ShootingPhase':
            for unit in player_units:
                if not unit['hasAttackedThisTurn']:
                    # Target most valuable enemy
                    best_target = max(enemy_units, 
                        key=lambda e: e['nmodels'] * e['A'])
                    return GameAction('shoot', unit['name'], {'target': best_target['name']})
        
        return GameAction('end_phase', 'system', {})
    
    def _distance(self, pos1, pos2):
        """Calculate distance between two positions"""
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        return (dx**2 + dy**2)**0.5
    
    def execute_action(self, action: GameAction):
        """
        Execute an action in the actual game.
        Translates GameAction to actual game commands.
        """
        if action is None:
            print("[AI] No action to execute")
            return
        
        print(f"[AI] Executing: {action}")
        
        if action.action_type == 'move':
            # Find the actual unit object
            unit = self._get_unit_by_name(action.unit_name)
            if unit:
                target_pos = (action.parameters['target_x'], 
                            action.parameters['target_y'], 0)
                # Use existing game movement system
                # This depends on your actual game implementation
                # Example: self.game.moveUnit(unit, target_pos)
                pass
        
        elif action.action_type == 'shoot':
            unit = self._get_unit_by_name(action.unit_name)
            target = self._get_unit_by_name(action.parameters['target'])
            if unit and target:
                # Execute shooting
                # Example: self.game.shootAt(unit, target)
                pass
        
        elif action.action_type == 'attack':
            unit = self._get_unit_by_name(action.unit_name)
            target = self._get_unit_by_name(action.parameters['target'])
            if unit and target:
                # Execute melee attack
                # Example: self.game.meleeAttack(unit, target)
                pass
        
        elif action.action_type == 'end_phase':
            # End current phase
            # Example: self.game.fsm.nextPhase()
            pass
    
    def _get_unit_by_name(self, name: str):
        """Get actual unit object by name"""
        for unit in self.game.units:
            if unit.unitName == name:
                return unit
        return None
    
    def take_turn(self):
        """
        Main entry point for AI turn.
        Makes decision and executes it.
        """
        if not self.active:
            return
        
        # Make decision
        action = self.make_decision()
        
        # Execute action
        self.execute_action(action)

        visualizer = TreeVisualizer(self.tree)
        visualizer.print_best_path()

        """ explainer = DecisionExplainer(self.tree)
        explainer.explain_decision()

        visualizer.print_statistics_detailed() """
        
        return action
    
    def print_statistics(self):
        """Print AI performance statistics"""
        print(f"\n=== AI Statistics (Player {self.player_num}) ===")
        print(f"Total decisions: {self.decisions_made}")
        print(f"Minimax decisions: {self.minimax_decisions} "
              f"({self.minimax_decisions/max(1, self.decisions_made)*100:.1f}%)")
        print(f"Heuristic decisions: {self.heuristic_decisions} "
              f"({self.heuristic_decisions/max(1, self.decisions_made)*100:.1f}%)")


# Example: Replacing existing AI in game.py
"""
In game.py, replace:
    self.AIplayer2 = ClassAI(self, self.player2Units, self.player1Units)

With:
    from aiMinimaxIntegration import EnhancedAI
    self.AIplayer2 = EnhancedAI(
        self, 
        self.player2Units, 
        self.player1Units,
        player_num=2,
        use_minimax=True,
        minimax_depth=3  # Adjust based on performance
    )

The EnhancedAI has the same interface as ClassAI but uses minimax when appropriate.

You can also use both AIs for comparison:
    self.AIplayer2_simple = ClassAI(self, self.player2Units, self.player1Units)
    self.AIplayer2_minimax = EnhancedAI(self, self.player2Units, self.player1Units)
    
    # Switch between them:
    self.AIplayer2 = self.AIplayer2_minimax  # Use the smart AI

Performance tuning:
- minimax_depth=2: Fast, decent decisions (~100-500 nodes)
- minimax_depth=3: Medium speed, good decisions (~500-5000 nodes)
- minimax_depth=4: Slow, excellent decisions (~5000-50000 nodes)
- minimax_depth=5+: Very slow, near-optimal (50000+ nodes)

Alpha-beta pruning typically reduces nodes by 50-90% compared to pure minimax.
"""
