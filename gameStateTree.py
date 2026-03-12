"""
Game State Tree Structure for Minimax Algorithm with Alpha-Beta Pruning
For Warhammer-style turn-based strategy game
"""

import math
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
from unitTypeClassifier import UnitTypeClassifier, UnitType, SupportRole, MATCHUP_TABLE, FLANK_BONUS, REAR_BONUS

# Module-level classifier singleton (caches classifications across calls)
_classifier = UnitTypeClassifier()


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
                'Points': int(unit.unit.model.characteristics.get('Points', 0)),
                'armor_save': unit.unit.model.armor_save,
                'charging': unit.unit.model.charging,
                'ranged' : any(unit.unit.model.weapons.get(weapon).get('tag') == 'ranged' for weapon in unit.unit.model.weapons),
                
                # Special rule flags for classifier
                'is_unbreakable': any(r.get('Unbreakable', False) for r in unit.unit.model.special_rules if isinstance(r, dict)),
                'is_stubborn': any('stubborn' in r.get('name', '').lower() for r in unit.unit.model.special_rules if isinstance(r, dict)),
                'has_regen': any(r.get('regen') for r in unit.unit.model.special_rules if isinstance(r, dict)),
                'has_mount': any(r.get('tag') == 'mount' for r in unit.unit.model.special_rules if isinstance(r, dict)),
                'is_flying': any('fly' in r.get('name', '').lower() for r in unit.unit.model.special_rules if isinstance(r, dict)),
                'has_charge_bonus': any(r.get('charge') for r in unit.unit.model.special_rules if isinstance(r, dict)),
                'M': int(unit.unit.model.characteristics.get('M', 4)) if unit.unit.model.characteristics.get('M', '4') not in ('-', '0', '') else 4,
                'W': int(unit.unit.model.characteristics.get('W', 1)),
                    
                # Combat relationships (store indices instead of references)
                'isInCombatWith': [u.unitName for u in unit.isInCombatWith],
                'isInCombatFlank': unit.isInCombatFlank.copy() if unit.isInCombatFlank else []
            }
            
            # Classify unit type and store on the dict
            main_type, support_role = _classifier.classify_from_dict(unit_data)
            unit_data['unit_type'] = main_type.value
            unit_data['support_role'] = support_role.value
            
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
        """Create a fast copy of this state (avoids slow copy.deepcopy)"""
        new_units = []
        for u in self.units:
            new_u = u.copy()  # shallow dict copy
            new_u['isInCombatWith'] = u['isInCombatWith'][:]
            new_u['isInCombatFlank'] = u['isInCombatFlank'][:] if u['isInCombatFlank'] else []
            new_units.append(new_u)
        return GameState(
            current_phase=self.current_phase,
            current_phase_index=self.current_phase_index,
            current_round=self.current_round,
            current_player=self.current_player,
            max_rounds=self.max_rounds,
            units=new_units,
            score=None,  # Don't carry cached score from parent
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
            
            for i, child in enumerate(node.children):
                # Determine if child is maximizing based on which player moves next
                child_maximizing = (child.state.current_player == 1)
                eval_score = self.minimax(child, depth - 1, alpha, beta, child_maximizing)
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_child = child
                
                alpha = max(alpha, eval_score)
                
                # Alpha-beta pruning
                if beta <= alpha:
                    self.nodes_pruned += len(node.children) - i - 1
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
            
            for i, child in enumerate(node.children):
                # Determine if child is maximizing based on which player moves next
                child_maximizing = (child.state.current_player == 1)
                eval_score = self.minimax(child, depth - 1, alpha, beta, child_maximizing)
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_child = child
                
                beta = min(beta, eval_score)
                
                # Alpha-beta pruning
                if beta <= alpha:
                    self.nodes_pruned += len(node.children) - i - 1
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
        Generate possible actions from current state using SINGLE-UNIT SELECTION.
        
        Instead of generating actions for ALL units (which explodes combinatorially),
        we pick the single highest-priority unit that hasn't acted yet, and only
        generate its actions. After it acts, the next call picks the next unit.
        
        This changes branching from O(all_units × moves_per_unit) to O(moves_per_unit).
        With 5 units × 4 moves each: old = 20 branches, new = 4 branches per node.
        """
        actions = []
        current_player = state.current_player
        player_units = state.get_player_units(current_player)
        enemy_units = state.get_player_units(3 - current_player)
        
        phase = state.current_phase
        
        # Max units the AI will consider moving (limits branching, not actual gameplay)
        max_units_to_consider = 4
        
        if phase == 'MovementPhase':
            # Filter to only the top N most relevant unmoved units
            unmoved = [u for u in player_units if not u['hasMovedThisTurn'] and u['state'] != 'InCombat']
            candidates = self._pick_top_n_units(unmoved, enemy_units, max_units_to_consider)
            
            best_unit = self._pick_most_relevant_unit(candidates, enemy_units)
            
            if best_unit:
                move_dist = 8  # Standard movement distance
                
                # Only consider the 3 closest enemies (limits branching)
                living_enemies = [e for e in enemy_units if e['nmodels'] > 0]
                living_enemies.sort(key=lambda e: (
                    (e['position'][0] - best_unit['position'][0])**2 +
                    (e['position'][1] - best_unit['position'][1])**2
                ))
                closest_enemies = living_enemies[:3]
                
                for enemy in closest_enemies:
                    dx = enemy['position'][0] - best_unit['position'][0]
                    dy = enemy['position'][1] - best_unit['position'][1]
                    dist = max(1.0, math.sqrt(dx*dx + dy*dy))
                    
                    actions.append(GameAction(
                        'move',
                        best_unit['name'],
                        {'target_x': best_unit['position'][0] + dx/dist * move_dist,
                         'target_y': best_unit['position'][1] + dy/dist * move_dist}
                    ))
                
                # Generic forward advance (toward enemy deployment zone)
                forward_dir = 1 if current_player == 1 else -1
                actions.append(GameAction(
                    'move',
                    best_unit['name'],
                    {'target_x': best_unit['position'][0],
                     'target_y': best_unit['position'][1] + forward_dir * move_dist}
                ))
            
            # Always can end phase
            actions.append(GameAction('end_phase', 'system', {}))
        
        elif phase == 'ShootingPhase':
            # Find the single best ranged unit that hasn't shot yet
            ranged_units = [u for u in player_units if not u['hasAttackedThisTurn'] and u['ranged']]
            best_unit = self._pick_most_relevant_unit(ranged_units, enemy_units)
            
            if best_unit:
                for enemy in enemy_units:
                    if enemy['nmodels'] > 0:
                        actions.append(GameAction(
                            'shoot',
                            best_unit['name'],
                            {'target': enemy['name']}
                        ))
            
            actions.append(GameAction('end_phase', 'system', {}))
        
        elif phase == 'CombatPhase':
            # Find the single best combat unit that hasn't attacked yet
            combat_units = [u for u in player_units if u['isInCombat'] and not u['hasAttackedThisTurn']]
            best_unit = self._pick_most_relevant_unit(combat_units, enemy_units)
            
            if best_unit:
                for enemy_name in best_unit['isInCombatWith']:
                    actions.append(GameAction(
                        'attack',
                        best_unit['name'],
                        {'target': enemy_name}
                    ))
            
            actions.append(GameAction('end_phase', 'system', {}))
        
        elif phase == 'StrategyPhase':
            actions.append(GameAction('end_phase', 'system', {}))
        
        # Put end_phase last so tactical actions are evaluated first
        end_phase_actions = [a for a in actions if a.action_type == 'end_phase']
        other_actions = [a for a in actions if a.action_type != 'end_phase']
        actions = other_actions + end_phase_actions
        
        # Limit action space for performance
        if len(actions) > 12:
            actions = self._prune_actions(state, actions)[:12]
        
        return actions
    
    def _pick_most_relevant_unit(self, candidates: List[Dict], enemy_units: List[Dict]) -> Optional[Dict]:
        """
        Pick the single most relevant/urgent unit to act next.
        
        Priority order:
        1. Units already in combat (must fight)
        2. Units closest to an enemy (most impactful moves)
        3. Strongest units (biggest impact)
        
        Units far from all enemies are deprioritized — their moves matter less.
        """
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        
        living_enemies = [e for e in enemy_units if e['nmodels'] > 0]
        if not living_enemies:
            return candidates[0]
        
        scored = []
        for unit in candidates:
            priority = 0
            
            # Highest priority: already in combat
            if unit['isInCombat']:
                priority += 1000
            
            # Distance to nearest enemy (closer = higher priority)
            min_dist = float('inf')
            for enemy in living_enemies:
                dx = enemy['position'][0] - unit['position'][0]
                dy = enemy['position'][1] - unit['position'][1]
                dist = math.sqrt(dx*dx + dy*dy)
                min_dist = min(min_dist, dist)
            
            # Invert distance: close units get high score, far units low
            # Units within charge range (~10) get big bonus
            if min_dist < 10:
                priority += 500
            elif min_dist < 20:
                priority += 200
            elif min_dist < 40:
                priority += 50
            # Units > 40 away get minimal priority (their exact move matters less)
            
            priority += 100.0 / max(1.0, min_dist)
            
            # Tiebreak: stronger units first (more impactful)
            priority += unit['nmodels'] * unit['A'] * unit['S'] * 0.1
            
            # Unit type priority bonuses
            u_type = unit.get('unit_type', 'basic')
            u_role = unit.get('support_role', 'none')
            if u_type == 'hammer' and min_dist < 15:
                priority += 300  # Hammer units near enemies are highest priority (charge!)
            elif u_type == 'anvil':
                priority += 50   # Anvils are less urgent to move (they hold anyway)
            elif u_type == 'cannon_fodder':
                priority += 30   # Fodder is low priority for movement
            if u_role == 'fast':
                priority += 150  # Fast units should maneuver for flanks
            
            scored.append((priority, unit))
        
        scored.sort(reverse=True, key=lambda x: x[0])
        return scored[0][1]
    
    def _pick_top_n_units(self, candidates: List[Dict], enemy_units: List[Dict], n: int) -> List[Dict]:
        """
        Return the top N most relevant units from candidates.
        Uses the same priority logic as _pick_most_relevant_unit but returns
        multiple units. Units outside top N are ignored by the AI search tree,
        reducing branching without limiting actual gameplay.
        """
        if len(candidates) <= n:
            return candidates
        
        living_enemies = [e for e in enemy_units if e['nmodels'] > 0]
        if not living_enemies:
            return candidates[:n]
        
        scored = []
        for unit in candidates:
            priority = 0
            if unit['isInCombat']:
                priority += 1000
            
            min_dist = float('inf')
            for enemy in living_enemies:
                dx = enemy['position'][0] - unit['position'][0]
                dy = enemy['position'][1] - unit['position'][1]
                dist = math.sqrt(dx*dx + dy*dy)
                min_dist = min(min_dist, dist)
            
            if min_dist < 10:
                priority += 500
            elif min_dist < 20:
                priority += 200
            elif min_dist < 40:
                priority += 50
            
            priority += 100.0 / max(1.0, min_dist)
            priority += unit['nmodels'] * unit['A'] * unit['S'] * 0.1
            
            # Unit type bonuses for top-N selection
            u_type = unit.get('unit_type', 'basic')
            u_role = unit.get('support_role', 'none')
            if u_type == 'hammer':
                priority += 200
            if u_role == 'fast':
                priority += 100
            
            scored.append((priority, unit))
        
        scored.sort(reverse=True, key=lambda x: x[0])
        return [unit for _, unit in scored[:n]]
    
    def _apply_action(self, state: GameState, action: GameAction) -> GameState:
        """
        Apply an action to a state and return the resulting new state.
        This creates a simulation of the action without affecting the real game.
        """
        new_state = state.clone()
        
        if action.action_type == 'move':
            unit = new_state.get_unit_by_name(action.unit_name)
            maxMovesAllowed = 2
            if unit and not unit['isInCombat'] and not unit['hasMovedThisTurn']:
                
                dx = action.parameters['target_x'] - unit['position'][0]
                dy = action.parameters['target_y'] - unit['position'][1]
                unit['position'] = (
                    action.parameters['target_x'],
                    action.parameters['target_y'],
                    unit['position'][2]
                )
                # Use explicit heading when provided (e.g. screening),
                # otherwise derive heading from movement direction.
                if 'target_heading' in action.parameters:
                    unit['heading'] = action.parameters['target_heading']
                else:
                    new_heading = math.degrees(math.atan2(dy, dx))
                    unit['heading'] = new_heading
                unit['hasMovedThisTurn'] = True
                already_moved = sum(1 for u in new_state.units if u['hasMovedThisTurn'])
                if already_moved >= maxMovesAllowed:
                    for u in new_state.units:
                        u['hasMovedThisTurn'] = True  # Force all units to have moved to encourage end phase
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
                damage = max(0, int(unit['nmodels'] * 0.1))+1
                target['nmodels'] = max(0, target['nmodels'] - damage)
        
        elif action.action_type == 'attack':
            unit = new_state.get_unit_by_name(action.unit_name)
            target = new_state.get_unit_by_name(action.parameters['target'])
            if unit and target and unit['isInCombat'] and not unit['hasAttackedThisTurn']:
                unit['hasAttackedThisTurn'] = True
                # Simulate melee combat
                damage = max(0, int(unit['nmodels'] * unit['A'] * 0.15))+1
                target['nmodels'] = max(0, target['nmodels'] - damage)
                """ if target['nmodels'] == 0:
                    print(f"{target['name']} has been destroyed in combat!") """
        
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
        Lightweight version - avoids O(n^2) flanking calculations for speed.
        """
        if state.score is not None:
            return state.score
        
        player1_units = state.get_player_units(1)
        player2_units = state.get_player_units(2)
        
        # ── Type-weighted army strength ──
        # Each unit type has a base value multiplier reflecting its strategic worth
        TYPE_VALUE = {
            'hammer': 2.5,
            'anvil': 1.6,
            'superior': 1.3,
            'basic': 1.0,
            'cannon_fodder': 0.5,
        }

        def _unit_value(u):
            """Value a single unit incorporating its type and point cost."""
            pts = u.get('Points', 0)
            # Use Points per model as the base value; fall back to A*S if unset
            base = u['nmodels'] * (pts if pts > 0 else u['A'] * u['S'])
            type_mult = TYPE_VALUE.get(u.get('unit_type', 'basic'), 1.0)
            return base * type_mult

        p1_strength = sum(_unit_value(u) for u in player1_units)
        p2_strength = sum(_unit_value(u) for u in player2_units)
        
        score = p1_strength - p2_strength
        
        # ── Matchup bonuses: reward good combat pairings ──
        for u in player1_units:
            if u['isInCombat'] and u['nmodels'] > 0:
                u_type = UnitType(u.get('unit_type', 'basic'))
                for enemy_name in u['isInCombatWith']:
                    enemy = state.get_unit_by_name(enemy_name)
                    if enemy and enemy['nmodels'] > 0:
                        e_type = UnitType(enemy.get('unit_type', 'basic'))
                        matchup = MATCHUP_TABLE[u_type][e_type]
                        # Check flanking
                        is_flanking = 'flank' in (u.get('isInCombatFlank') or [])
                        is_rear = 'rear' in (u.get('isInCombatFlank') or [])
                        if is_rear:
                            matchup += REAR_BONUS
                        elif is_flanking:
                            matchup += FLANK_BONUS
                        score += (matchup - 1.0) * 15  # reward favorable matchups

        for u in player2_units:
            if u['isInCombat'] and u['nmodels'] > 0:
                u_type = UnitType(u.get('unit_type', 'basic'))
                for enemy_name in u['isInCombatWith']:
                    enemy = state.get_unit_by_name(enemy_name)
                    if enemy and enemy['nmodels'] > 0:
                        e_type = UnitType(enemy.get('unit_type', 'basic'))
                        matchup = MATCHUP_TABLE[u_type][e_type]
                        is_flanking = 'flank' in (u.get('isInCombatFlank') or [])
                        is_rear = 'rear' in (u.get('isInCombatFlank') or [])
                        if is_rear:
                            matchup += REAR_BONUS
                        elif is_flanking:
                            matchup += FLANK_BONUS
                        score -= (matchup - 1.0) * 15  # P2 good matchup hurts P1

        # ── Positional bonuses ──
        score += sum(u['position'][1] * 5 for u in player1_units)
        score += sum(u['position'][1] * 5 for u in player2_units)

        # ── Fast unit flanking threat bonus ──
        for u in player1_units:
            if u.get('support_role') == 'fast' and not u['isInCombat'] and u['nmodels'] > 0:
                score += 8  # latent flanking threat
        for u in player2_units:
            if u.get('support_role') == 'fast' and not u['isInCombat'] and u['nmodels'] > 0:
                score -= 8

        # ── Shooting unit value: ranged units that haven't fired ──
        for u in player1_units:
            if u.get('support_role') == 'shooting' and not u['hasAttackedThisTurn'] and u['nmodels'] > 0:
                score += 5
        for u in player2_units:
            if u.get('support_role') == 'shooting' and not u['hasAttackedThisTurn'] and u['nmodels'] > 0:
                score -= 5

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
                  f"Unused P1/P2: {unused_p1_actions}/{unused_p2_actions}")
        
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
        enemy_units = state.get_player_units(3 - state.current_player)
        living_enemies = [e for e in enemy_units if e['nmodels'] > 0]
        
        scored_actions = []
        for action in actions:
            score = 0
            if action.action_type == 'attack':
                score = 100  # Prioritize attacks
            elif action.action_type == 'shoot':
                score = 80  # Prioritize shooting
            elif action.action_type == 'move':
                # Score by how close the move gets to nearest enemy
                tx = action.parameters.get('target_x', 0)
                ty = action.parameters.get('target_y', 0)
                if living_enemies:
                    min_dist = min(
                        math.sqrt((e['position'][0]-tx)**2 + (e['position'][1]-ty)**2)
                        for e in living_enemies
                    )
                    score = 50 + max(0, 30 - min_dist)  # Closer = higher
                else:
                    score = 50
            elif action.action_type == 'end_phase':
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
