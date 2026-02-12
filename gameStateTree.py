"""
Game State Tree Structure for Minimax Algorithm with Alpha-Beta Pruning
For Warhammer-style turn-based strategy game
"""

import copy
import math
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
import json


@dataclass
class GameState:
    """
    Lightweight representation of game state that can be efficiently copied.
    This is what gets stored in each tree node.
    """
    # FSM state
    current_phase: str
    current_phase_index: int
    
    # Round info
    current_round: int
    current_player: int
    max_rounds: int
    
    # Unit states (simplified for speed)
    units: List[Dict[str, Any]] = field(default_factory=list)
    
    # Evaluation score (cached)
    score: Optional[float] = None
    
    @classmethod
    def from_game(cls, game):
        """Create a GameState from the actual game object"""
        units = []
        
        for unit in game.units:
            unit_data = {
                'name': unit.unitName,
                'position': tuple(unit.bodyNP.getPos()),
                'heading': unit.bodyNP.getH(),
                'state': unit.state,
                'player': 1 if unit in game.player1Units else 2,
                
                # Combat/turn state
                'isInCombat': unit.isInCombat,
                'hasMovedThisTurn': unit.hasMovedThisTurn,
                'hasAttackedThisTurn': unit.hasAttackedThisTurn,
                'isDeployed': unit.isDeployed,
                
                # Unit composition
                'nmodels': unit.unit.nmodels,
                'files': unit.unit.files,
                'ranks': unit.unit.ranks,
                
                # Key characteristics (for evaluation)
                'WS': int(unit.unit.model.characteristics.get('WS', 3)),
                'S': int(unit.unit.model.characteristics.get('S', 3)),
                'T': int(unit.unit.model.characteristics.get('T', 3)),
                'A': int(unit.unit.model.characteristics.get('A', 1)),
                'Ld': int(unit.unit.model.characteristics.get('Ld', 7)),
                'armor_save': unit.unit.model.armor_save,
                'charging': unit.unit.model.charging,
                'ranged' : any(unit.unit.model.weapons.get(weapon).get('tag') == 'ranged' for weapon in unit.unit.model.weapons),
                    
                
                # Combat relationships (store indices instead of references)
                'isInCombatWith': [u.unitName for u in unit.isInCombatWith],
                'isInCombatFlank': unit.isInCombatFlank.copy() if unit.isInCombatFlank else []
            }
            units.append(unit_data)
        
        return cls(
            current_phase=game.fsm.phases[game.fsm.currentPhaseIndex],
            current_phase_index=game.fsm.currentPhaseIndex,
            current_round=game.roundCounter.currentRoundPlayer[game.roundCounter.current_player - 1],
            current_player=game.roundCounter.current_player,
            max_rounds=game.roundCounter.max_rounds,
            units=units
        )
    
    def clone(self):
        """Create a deep copy of this state"""
        return GameState(
            current_phase=self.current_phase,
            current_phase_index=self.current_phase_index,
            current_round=self.current_round,
            current_player=self.current_player,
            max_rounds=self.max_rounds,
            units=copy.deepcopy(self.units),
            score=self.score
        )
    
    def get_unit_by_name(self, name: str) -> Optional[Dict]:
        """Get unit data by name"""
        for unit in self.units:
            if unit['name'] == name:
                return unit
        return None
    
    def get_player_units(self, player: int) -> List[Dict]:
        """Get all units belonging to a player"""
        return [u for u in self.units if u['player'] == player]


@dataclass
class GameAction:
    """
    Represents a possible action/move in the game.
    Used to transition between states.
    """
    action_type: str  # 'move', 'charge', 'shoot', 'cast_spell', 'end_phase', etc.
    unit_name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self):
        return f"{self.action_type}({self.unit_name}, {self.parameters})"


class GameStateNode:
    """
    A node in the game tree. Contains a state and can have child nodes.
    """
    def __init__(self, state: GameState, parent: Optional['GameStateNode'] = None, 
                 action: Optional[GameAction] = None, depth: int = 0):
        self.state = state
        self.parent = parent
        self.action = action  # Action that led to this state
        self.children: List['GameStateNode'] = []
        self.depth = depth
        
        # Minimax values
        self.value: Optional[float] = None
        self.alpha: float = float('-inf')
        self.beta: float = float('inf')
        
        # Flags
        self.is_terminal = False
        self.is_fully_expanded = False
        self.best_child: Optional['GameStateNode'] = None
    
    def add_child(self, child_state: GameState, action: GameAction) -> 'GameStateNode':
        """Add a child node with the given state and action"""
        child = GameStateNode(child_state, parent=self, action=action, depth=self.depth + 1)
        self.children.append(child)
        return child
    
    def is_max_node(self) -> bool:
        """True if this is a maximizing node (player turn)"""
        # Assuming player 1 is maximizing, player 2 is minimizing
        return self.state.current_player == 1
    
    def is_min_node(self) -> bool:
        """True if this is a minimizing node (opponent turn)"""
        return not self.is_max_node()
    
    def get_path_from_root(self) -> List['GameStateNode']:
        """Get the path from root to this node"""
        path = []
        node = self
        while node is not None:
            path.append(node)
            node = node.parent
        return list(reversed(path))
    
    def get_action_sequence(self) -> List[GameAction]:
        """Get the sequence of actions from root to this node"""
        path = self.get_path_from_root()
        return [node.action for node in path[1:] if node.action is not None]
    
    def __repr__(self):
        action_str = f" via {self.action}" if self.action else ""
        value_str = f" value={self.value:.2f}" if self.value is not None else ""
        return f"Node(depth={self.depth}, player={self.state.current_player}{action_str}{value_str})"


class MinimaxTree:
    """
    Game tree with minimax algorithm and alpha-beta pruning.
    """
    def __init__(self, game_state_analyzer, max_depth: int = 3):
        self.analyzer = game_state_analyzer
        self.max_depth = max_depth
        self.nodes_evaluated = 0
        self.nodes_pruned = 0
        self.root: Optional[GameStateNode] = None
    
    def build_tree(self, initial_state: GameState, depth: int = None) -> GameStateNode:
        """Build game tree from initial state"""
        if depth is None:
            depth = self.max_depth
        
        self.root = GameStateNode(initial_state)
        self.nodes_evaluated = 0
        self.nodes_pruned = 0
        
        self._expand_node(self.root, depth)
        return self.root
    
    def minimax(self, node: GameStateNode, depth: int, alpha: float, beta: float, 
                maximizing: bool) -> float:
        """
        Minimax algorithm with alpha-beta pruning.
        
        Args:
            node: Current node
            depth: Remaining depth to search
            alpha: Best value for maximizer
            beta: Best value for minimizer
            maximizing: True if maximizing player's turn
        
        Returns:
            Best evaluation score
        """
        self.nodes_evaluated += 1
        
        # Terminal node or max depth reached
        if depth == 0 or self._is_terminal_state(node.state):
            node.value = self._evaluate_state(node.state)
            node.is_terminal = True
            return node.value
        
        if maximizing:
            max_eval = float('-inf')
            best_child = None
            
            # Generate or use existing children
            if not node.children:
                self._expand_node(node, 1)
            
            for child in node.children:
                # Determine if child is maximizing based on which player moves next
                child_maximizing = (child.state.current_player == 1)
                eval_score = self.minimax(child, depth - 1, alpha, beta, child_maximizing)
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_child = child
                
                alpha = max(alpha, eval_score)
                
                # Alpha-beta pruning
                if beta <= alpha:
                    self.nodes_pruned += len(node.children) - node.children.index(child) - 1
                    break
            
            node.value = max_eval
            node.best_child = best_child
            node.alpha = alpha
            return max_eval
        
        else:  # Minimizing
            min_eval = float('inf')
            best_child = None
            
            if not node.children:
                self._expand_node(node, 1)
            
            for child in node.children:
                # Determine if child is maximizing based on which player moves next
                child_maximizing = (child.state.current_player == 1)
                eval_score = self.minimax(child, depth - 1, alpha, beta, child_maximizing)
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_child = child
                
                beta = min(beta, eval_score)
                
                # Alpha-beta pruning
                if beta <= alpha:
                    self.nodes_pruned += len(node.children) - node.children.index(child) - 1
                    break
            
            node.value = min_eval
            node.best_child = best_child
            node.beta = beta
            return min_eval
    
    def find_best_move(self, initial_state: GameState) -> Tuple[Optional[GameAction], float]:
        """
        Find the best move from the current state.
        
        Returns:
            Tuple of (best_action, expected_value)
        """
        self.root = GameStateNode(initial_state)
        
        is_maximizing = initial_state.current_player == 1
        best_value = self.minimax(self.root, self.max_depth, float('-inf'), float('inf'), is_maximizing)
        
        if self.root.best_child:
            return (self.root.best_child.action, best_value)
        
        return (None, best_value)
    
    def _expand_node(self, node: GameStateNode, depth: int):
        """Generate child nodes for all possible actions"""
        if depth <= 0 or self._is_terminal_state(node.state):
            return
        
        possible_actions = self._generate_possible_actions(node.state)
        
        for action in possible_actions:
            # Create new state by applying action
            new_state = self._apply_action(node.state, action)
            node.add_child(new_state, action)
    
    def _generate_possible_actions(self, state: GameState) -> List[GameAction]:
        """
        Generate all possible legal actions from current state.
        This is game-specific and needs to be customized.
        """
        actions = []
        current_player = state.current_player
        player_units = state.get_player_units(current_player)
        
        phase = state.current_phase
        
        if phase == 'MovementPhase':
            # Generate movement actions for each unit
            for unit in player_units:
                if not unit['hasMovedThisTurn'] and unit['state'] != 'InCombat':
                    # Sample movement positions (in real implementation, use pathfinding)
                    for dx in [-5, 0, 5]:
                        for dy in [-10, -5, 0, 5, 10]:
                            if dx == 0 and dy == 0:
                                continue
                            actions.append(GameAction(
                                'move',
                                unit['name'],
                                {'target_x': unit['position'][0] + dx, 
                                 'target_y': unit['position'][1] + dy}
                            ))
            
            # Always can end phase
            actions.append(GameAction('end_phase', 'system', {}))
        
        elif phase == 'ShootingPhase':
            # Generate shooting actions
            enemy_units = state.get_player_units(3 - current_player)
            for unit in player_units:
                if not unit['hasAttackedThisTurn'] and unit['ranged']:
                    for enemy in enemy_units:
                        actions.append(GameAction(
                            'shoot',
                            unit['name'],
                            {'target': enemy['name']}
                        ))
            
            actions.append(GameAction('end_phase', 'system', {}))
        
        elif phase == 'CombatPhase':
            # Generate combat actions
            for unit in player_units:
                if unit['isInCombat'] and not unit['hasAttackedThisTurn']:
                    for enemy_name in unit['isInCombatWith']:
                        actions.append(GameAction(
                            'attack',
                            unit['name'],
                            {'target': enemy_name}
                        ))
            
            actions.append(GameAction('end_phase', 'system', {}))
        
        elif phase == 'StrategyPhase':
            # Generate strategy actions (rallying, spell casting, etc.)
            actions.append(GameAction('end_phase', 'system', {}))
        
        # IMPORTANT: Always put end_phase at the end of the list
        # This ensures tactical actions are evaluated first in minimax
        end_phase_actions = [a for a in actions if a.action_type == 'end_phase']
        other_actions = [a for a in actions if a.action_type != 'end_phase']
        actions = other_actions + end_phase_actions
        
        # Limit action space for performance
        #print(len(actions), "possible actions generated")
        if len(actions) > 20:
            # Prune to most promising actions using heuristics
            actions = self._prune_actions(state, actions)[:20]
        
        return actions
    
    def _apply_action(self, state: GameState, action: GameAction) -> GameState:
        """
        Apply an action to a state and return the resulting new state.
        This creates a simulation of the action without affecting the real game.
        """
        new_state = state.clone()
        
        if action.action_type == 'move':
            unit = new_state.get_unit_by_name(action.unit_name)
            
            if unit and not unit['isInCombat'] and not unit['hasMovedThisTurn']:
                dx = action.parameters['target_x'] - unit['position'][0]
                dy = action.parameters['target_y'] - unit['position'][1]
                unit['position'] = (
                    action.parameters['target_x'],
                    action.parameters['target_y'],
                    unit['position'][2]
                )
                # Calculate new heading based on movement direction
                
                new_heading = math.degrees(math.atan2(dy, dx))
                unit['heading'] = new_heading
                unit['hasMovedThisTurn'] = True

                # Check if unit is within 5 units of an enemy unit
                enemy_units = new_state.get_player_units(3 - unit['player'])
                for enemy in enemy_units:
                    dx = enemy['position'][0] - unit['position'][0]
                    dy = enemy['position'][1] - unit['position'][1]
                    distance = math.sqrt(dx*dx + dy*dy)
                    if distance < 5:
                        unit['isInCombat'] = True
                        unit['state'] = 'InCombat'
                        unit['isInCombatWith'].append(enemy['name'])
                        enemy['isInCombat'] = True
                        enemy['state'] = 'InCombat'
                        enemy['isInCombatWith'].append(unit['name'])
                        break
        
        elif action.action_type == 'shoot':
            unit = new_state.get_unit_by_name(action.unit_name)
            target = new_state.get_unit_by_name(action.parameters['target'])
            if unit and target and not unit['isInCombat'] and not unit['hasAttackedThisTurn']:
                # Simulate combat (simplified)
                unit['hasAttackedThisTurn'] = True
                # Rough damage calculation
                damage = max(0, int(unit['nmodels'] * 0.1))
                target['nmodels'] = max(0, target['nmodels'] - damage)
        
        elif action.action_type == 'attack':
            unit = new_state.get_unit_by_name(action.unit_name)
            target = new_state.get_unit_by_name(action.parameters['target'])
            if unit and target:
                unit['hasAttackedThisTurn'] = True
                # Simulate melee combat
                damage = max(0, int(unit['nmodels'] * unit['A'] * 0.15))
                target['nmodels'] = max(0, target['nmodels'] - damage)
                if target['nmodels'] == 0:
                    print(f"{target['name']} has been destroyed in combat!")
        
        elif action.action_type == 'end_phase':
            # Advance to next phase
            new_state.current_phase_index = (new_state.current_phase_index + 1) % 4
            phases = ['StrategyPhase', 'MovementPhase', 'ShootingPhase', 'CombatPhase']
            new_state.current_phase = phases[new_state.current_phase_index]
            
            # Reset turn flags if new turn (going back to StrategyPhase)
            if new_state.current_phase_index == 0:
                new_state.current_player = 3 - new_state.current_player
                # Reset ALL units' flags at start of new round
                for unit in new_state.units:
                    unit['hasMovedThisTurn'] = False
                    unit['hasAttackedThisTurn'] = False
        
        return new_state
    
    def _evaluate_state(self, state: GameState) -> float:
        """
        Evaluate a game state and return a score.
        Positive values favor player 1, negative favor player 2.
        """
        if state.score is not None:
            return state.score
        
        player1_units = state.get_player_units(1)
        player2_units = state.get_player_units(2)
        
        # Quick evaluation based on army strength
        p1_strength = sum(u['nmodels'] * u['A'] * u['S'] * (7 - u['armor_save'])
                         for u in player1_units)
        p2_strength = sum(u['nmodels'] * u['A'] * u['S'] * (7 - u['armor_save']) 
                         for u in player2_units)
        
        score = p1_strength - p2_strength
        
        # Positional bonuses - encourage both players to move towards center/enemy
        # Player 1 starts at negative Y, should move towards positive Y (towards enemy)
        # Player 2 starts at positive Y, should move towards negative Y (towards enemy)
        p1_pos_bonus = 0
        p2_pos_bonus = 0
        for unit in player1_units:
            p1_pos_bonus += unit['position'][1] * 5  # Higher Y = better for P1 (maximizing)
            
        
        for unit in player2_units:
            p2_pos_bonus += unit['position'][1] * 5  # Track P2's Y position
        
        # P2 minimizes score, so higher Y values should increase score (making it less desirable for P2)
        # Lower Y values decrease score (making it more desirable for P2)
        score += p1_pos_bonus + p2_pos_bonus

        # Flanking bonus - reward units positioned to attack enemy flanks/rear
        import math
        p1_flank_bonus = 0
        p2_flank_bonus = 0
        
        for unit in player1_units:
            if unit['nmodels'] == 0:
                continue
            unit_pos = unit['position']
            
            for enemy in player2_units:
                if enemy['nmodels'] == 0:
                    continue
                enemy_pos = enemy['position']
                
                # Calculate distance
                dx = enemy_pos[0] - unit_pos[0]
                dy = enemy_pos[1] - unit_pos[1]
                distance = math.sqrt(dx*dx + dy*dy)
                
                if distance < 20:  # Within threatening range (charge distance)
                    # Calculate angle from enemy facing to our unit
                    direction_x = unit_pos[0] - enemy_pos[0]
                    direction_y = unit_pos[1] - enemy_pos[1]
                    angle_to_unit = math.degrees(math.atan2(direction_y, direction_x))
                    relative_angle = angle_to_unit - enemy['heading']
                    
                    # Normalize to -180 to 180
                    while relative_angle > 180:
                        relative_angle -= 360
                    while relative_angle < -180:
                        relative_angle += 360
                    
                    if 45 < abs(relative_angle) < 135:  # Flank position
                        p1_flank_bonus += 15
                    elif abs(relative_angle) > 135:  # Rear position
                        p1_flank_bonus += 30
        
        for unit in player2_units:
            if unit['nmodels'] == 0:
                continue
            unit_pos = unit['position']
            
            for enemy in player1_units:
                if enemy['nmodels'] == 0:
                    continue
                enemy_pos = enemy['position']
                
                # Calculate distance
                dx = enemy_pos[0] - unit_pos[0]
                dy = enemy_pos[1] - unit_pos[1]
                distance = math.sqrt(dx*dx + dy*dy)
                
                if distance < 20:  # Within threatening range
                    # Calculate angle from enemy facing to our unit
                    direction_x = unit_pos[0] - enemy_pos[0]
                    direction_y = unit_pos[1] - enemy_pos[1]
                    angle_to_unit = math.degrees(math.atan2(direction_y, direction_x))
                    relative_angle = angle_to_unit - enemy['heading']
                    
                    # Normalize to -180 to 180
                    while relative_angle > 180:
                        relative_angle -= 360
                    while relative_angle < -180:
                        relative_angle += 360
                    
                    if 45 < abs(relative_angle) < 135:  # Flank position
                        p2_flank_bonus += 15
                    elif abs(relative_angle) > 135:  # Rear position
                        p2_flank_bonus += 30
        
        score += p1_flank_bonus - p2_flank_bonus

        # CRITICAL: Penalize unused action potential
        # If units haven't moved/attacked yet, that's wasted opportunity
        unused_p1_actions = sum(
            1 for u in player1_units 
            if not u['hasMovedThisTurn'] and u['state'] not in ['InCombat', 'IsFleeing']
        )
        unused_p2_actions = sum(
            1 for u in player2_units 
            if not u['hasMovedThisTurn'] and u['state'] not in ['InCombat', 'IsFleeing']
        )
        
        # Penalize the current player for having unused actions
        # This prevents ending phase early
        if state.current_player == 1:
            score -= unused_p1_actions * 20  # P1 loses points for not using their units
        else:
            score += unused_p2_actions * 20  # P2 loses points (adds to score because P2 wants lower scores)
        
        # Bonus for units in combat (taking action)
        p1_engaged = sum(10 for u in player1_units if u['isInCombat'])
        p2_engaged = sum(10 for u in player2_units if u['isInCombat'])
        score += p1_engaged - p2_engaged
        
        state.score = score
        
        # Debug output (can be commented out for performance)
        if False:  # Set to True to enable debug
            print(f"[Eval] P{state.current_player} | Score: {score:.1f} | "
                  f"Strength: {p1_strength-p2_strength:.0f} | "
                  f"Flank: P1={p1_flank_bonus} P2={p2_flank_bonus} | "
                  f"Unused P1/P2: {unused_p1_actions}/{unused_p2_actions} | "
                  f"PosBonus: {p1_pos_bonus:.0f}+{p2_pos_bonus:.0f}")
        
        return score
    
    def _is_terminal_state(self, state: GameState) -> bool:
        """Check if state is terminal (game over)"""
        player1_units = state.get_player_units(1)
        player2_units = state.get_player_units(2)
        
        # Check if either player has no units
        p1_alive = sum(1 for u in player1_units if u['nmodels'] > 0)
        p2_alive = sum(1 for u in player2_units if u['nmodels'] > 0)
        
        if p1_alive == 0 or p2_alive == 0:
            return True
        
        # Check if max rounds reached
        if state.current_round >= state.max_rounds:
            return True
        
        return False
    
    def _prune_actions(self, state: GameState, actions: List[GameAction]) -> List[GameAction]:
        """Prune less promising actions using heuristics"""
        # Sort actions by estimated value
        scored_actions = []
        for action in actions:
            # Quick heuristic scoring
            score = 0
            if action.action_type == 'attack':
                score = 100  # Prioritize attacks
            elif action.action_type == 'shoot':
                score = 80  # Prioritize shooting
            elif action.action_type == 'move':
                # Prioritize moving towards enemy
                target_x = action.parameters.get('target_x', 0)
                target_y = action.parameters.get('target_y', 0)
                # Score based on forward movement
                score = 50 #+ target_y #abs(target_y)
            elif action.action_type == 'end_phase':
                # ALWAYS evaluate end_phase LAST
                # This ensures tactical moves are tried first
                score = -1000
            scored_actions.append((score, action))
        
        scored_actions.sort(reverse=True, key=lambda x: x[0])
        return [action for _, action in scored_actions]
    
    def print_tree(self, node: Optional[GameStateNode] = None, indent: int = 0):
        """Print the tree structure for debugging"""
        if node is None:
            node = self.root
        
        if node is None:
            print("Empty tree")
            return
        
        prefix = "  " * indent
        print(f"{prefix}{node}")
        
        if node.best_child and node.best_child in node.children:
            best_idx = node.children.index(node.best_child)
            for i, child in enumerate(node.children):
                marker = " [BEST]" if i == best_idx else ""
                print(f"{prefix}  └─{marker}")
                self.print_tree(child, indent + 2)
        else:
            for child in node.children[:5]:  # Limit output
                self.print_tree(child, indent + 1)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get tree statistics"""
        return {
            'nodes_evaluated': self.nodes_evaluated,
            'nodes_pruned': self.nodes_pruned,
            'max_depth': self.max_depth,
            'pruning_efficiency': self.nodes_pruned / max(1, self.nodes_evaluated) * 100
        }


# Example usage:
"""
from gameStateTree import GameState, MinimaxTree
from gameStateAnalyzer import GameStateAnalyzer

# In your game class or AI:
analyzer = GameStateAnalyzer(game)
tree = MinimaxTree(analyzer, max_depth=4)

# Capture current state
current_state = GameState.from_game(game)

# Find best move
best_action, expected_value = tree.find_best_move(current_state)
print(f"Best action: {best_action}")
print(f"Expected value: {expected_value}")

# Get statistics
stats = tree.get_statistics()
print(f"Evaluated {stats['nodes_evaluated']} nodes")
print(f"Pruned {stats['nodes_pruned']} nodes ({stats['pruning_efficiency']:.1f}% efficiency)")

# Execute best action in real game
if best_action:
    # Apply the action to the real game state
    pass
"""
