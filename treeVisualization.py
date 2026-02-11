"""
Visualization and debugging utilities for the Minimax game tree.
Helps understand and debug AI decision-making.
"""

from gameStateTree import GameState, GameStateNode, MinimaxTree, GameAction
from typing import Optional, List, TextIO
import json
import sys


class TreeVisualizer:
    """
    Visualize the minimax tree in various formats.
    """
    
    def __init__(self, tree: MinimaxTree):
        self.tree = tree
    
    def print_tree_ascii(self, node: Optional[GameStateNode] = None, 
                        prefix: str = "", is_last: bool = True, 
                        max_depth: int = 4, file: TextIO = sys.stdout):
        """
        Print tree in ASCII art format.
        
        Example output:
        └─ Root [P1] value=10.5
           ├─ Move(Unit1, forward) [P2] value=8.2 [BEST]
           │  ├─ Attack(Unit2, Enemy1) [P1] value=8.2
           │  └─ EndPhase() [P1] value=5.1
           └─ Move(Unit1, left) [P2] value=5.0
        """
        if node is None:
            node = self.tree.root
        
        if node is None or node.depth > max_depth:
            return
        
        # Prepare node label
        action_str = f"{node.action}" if node.action else "Root"
        player_str = f"P{node.state.current_player}"
        value_str = f"value={node.value:.2f}" if node.value is not None else "value=?"
        phase_str = node.state.current_phase[:3]
        
        # Mark best path
        is_best = (node.parent and node.parent.best_child == node)
        best_marker = " [★ BEST]" if is_best else ""
        
        # Build full label
        label = f"{action_str} [{player_str}|{phase_str}] {value_str}{best_marker}"
        
        # Print with tree structure
        connector = "└─ " if is_last else "├─ "
        file.write(prefix + connector + label + "\n")
        
        # Prepare prefix for children
        extension = "   " if is_last else "│  "
        new_prefix = prefix + extension
        
        # Sort children to show best first
        children = sorted(node.children, 
                         key=lambda c: (c != node.best_child, -(c.value or 0)))
        
        # Print children
        for i, child in enumerate(children[:10]):  # Limit to 10 children
            is_last_child = (i == len(children) - 1)
            self.print_tree_ascii(child, new_prefix, is_last_child, max_depth, file)
        
        if len(node.children) > 10:
            file.write(new_prefix + "... (" + str(len(node.children) - 10) + 
                      " more children)\n")
    
    def print_best_path(self, file: TextIO = sys.stdout):
        """Print the best path found by minimax"""
        if not self.tree.root:
            file.write("No tree to display\n")
            return
        
        file.write("=== Best Path Found by Minimax ===\n\n")
        
        node = self.tree.root
        depth = 0
        
        while node:
            # Node info
            player = node.state.current_player
            phase = node.state.current_phase
            value = node.value if node.value is not None else "?"
            
            file.write(f"Step {depth}: ")
            
            if node.action:
                file.write(f"{node.action}\n")
            else:
                file.write("Initial State\n")
            
            file.write(f"  Player: {player}, Phase: {phase}, Value: {value}\n")
            
            # Show some state info
            p1_units = node.state.get_player_units(1)
            p2_units = node.state.get_player_units(2)
            p1_models = sum(u['nmodels'] for u in p1_units)
            p2_models = sum(u['nmodels'] for u in p2_units)
            
            file.write(f"  Army Sizes: P1={p1_models} models, P2={p2_models} models\n")
            file.write("\n")
            
            # Move to best child
            if node.best_child:
                node = node.best_child
                depth += 1
            else:
                break
        
        file.write(f"Total depth: {depth}\n")
    
    def export_to_json(self, filename: str, max_depth: int = 3):
        """Export tree to JSON format for external visualization"""
        if not self.tree.root:
            return
        
        tree_dict = self._node_to_dict(self.tree.root, max_depth)
        
        with open(filename, 'w') as f:
            json.dump(tree_dict, f, indent=2)
        
        print(f"Tree exported to {filename}")
    
    def _node_to_dict(self, node: GameStateNode, max_depth: int) -> dict:
        """Convert node and children to dictionary"""
        if node.depth >= max_depth:
            return None
        
        node_dict = {
            'action': str(node.action) if node.action else 'Root',
            'player': node.state.current_player,
            'phase': node.state.current_phase,
            'value': node.value,
            'alpha': node.alpha,
            'beta': node.beta,
            'depth': node.depth,
            'is_best': node.parent and node.parent.best_child == node,
            'children': []
        }
        
        for child in node.children:
            child_dict = self._node_to_dict(child, max_depth)
            if child_dict:
                node_dict['children'].append(child_dict)
        
        return node_dict
    
    def print_statistics_detailed(self, file: TextIO = sys.stdout):
        """Print detailed statistics about the tree"""
        stats = self.tree.get_statistics()
        
        file.write("=== Minimax Tree Statistics ===\n\n")
        file.write(f"Nodes evaluated: {stats['nodes_evaluated']}\n")
        file.write(f"Nodes pruned: {stats['nodes_pruned']}\n")
        file.write(f"Pruning efficiency: {stats['pruning_efficiency']:.1f}%\n")
        file.write(f"Max depth: {stats['max_depth']}\n")
        
        if 'tt_hit_rate' in stats:
            file.write(f"\nTransposition Table:\n")
            file.write(f"  Size: {stats['tt_size']} entries\n")
            file.write(f"  Hits: {stats['tt_hits']}\n")
            file.write(f"  Misses: {stats['tt_misses']}\n")
            file.write(f"  Hit rate: {stats['tt_hit_rate']:.1f}%\n")
        
        # Calculate tree metrics
        if self.tree.root:
            total_nodes = self._count_nodes(self.tree.root)
            max_depth_actual = self._max_depth(self.tree.root)
            avg_branching = (stats['nodes_evaluated'] / max(1, max_depth_actual))
            
            file.write(f"\nTree Structure:\n")
            file.write(f"  Total nodes in memory: {total_nodes}\n")
            file.write(f"  Actual max depth reached: {max_depth_actual}\n")
            file.write(f"  Average branching factor: {avg_branching:.1f}\n")
    
    def _count_nodes(self, node: GameStateNode) -> int:
        """Count total nodes in tree"""
        count = 1
        for child in node.children:
            count += self._count_nodes(child)
        return count
    
    def _max_depth(self, node: GameStateNode) -> int:
        """Find maximum depth in tree"""
        if not node.children:
            return node.depth
        return max(self._max_depth(child) for child in node.children)


class GameStateDebugger:
    """
    Debug utilities for inspecting game states.
    """
    
    @staticmethod
    def print_state(state: GameState, file: TextIO = sys.stdout):
        """Print detailed game state information"""
        file.write("=== Game State ===\n\n")
        file.write(f"Phase: {state.current_phase}\n")
        file.write(f"Round: {state.current_round}/{state.max_rounds}\n")
        file.write(f"Current Player: {state.current_player}\n\n")
        
        file.write("Units:\n")
        for unit in state.units:
            player_marker = f"[P{unit['player']}]"
            file.write(f"  {player_marker} {unit['name']:<20} ")
            file.write(f"Models: {unit['nmodels']:>3} ")
            file.write(f"Pos: ({unit['position'][0]:>6.1f}, {unit['position'][1]:>6.1f}) ")
            file.write(f"State: {unit['state']:<12} ")
            
            if unit['isInCombat']:
                file.write(f"[IN COMBAT with {', '.join(unit['isInCombatWith'])}]")
            
            file.write("\n")
        
        file.write("\n")
    
    @staticmethod
    def compare_states(state1: GameState, state2: GameState, file: TextIO = sys.stdout):
        """Compare two game states and show differences"""
        file.write("=== State Comparison ===\n\n")
        
        # Phase/round changes
        if state1.current_phase != state2.current_phase:
            file.write(f"Phase: {state1.current_phase} -> {state2.current_phase}\n")
        
        if state1.current_player != state2.current_player:
            file.write(f"Player: {state1.current_player} -> {state2.current_player}\n")
        
        # Unit changes
        file.write("\nUnit Changes:\n")
        for unit1 in state1.units:
            unit2 = state2.get_unit_by_name(unit1['name'])
            if not unit2:
                file.write(f"  {unit1['name']}: DESTROYED\n")
                continue
            
            changes = []
            
            if unit1['nmodels'] != unit2['nmodels']:
                changes.append(f"models {unit1['nmodels']} -> {unit2['nmodels']}")
            
            if unit1['position'] != unit2['position']:
                changes.append(f"moved from {unit1['position'][:2]} to {unit2['position'][:2]}")
            
            if unit1['state'] != unit2['state']:
                changes.append(f"state {unit1['state']} -> {unit2['state']}")
            
            if unit1['hasMovedThisTurn'] != unit2['hasMovedThisTurn']:
                changes.append(f"moved: {unit2['hasMovedThisTurn']}")
            
            if unit1['hasAttackedThisTurn'] != unit2['hasAttackedThisTurn']:
                changes.append(f"attacked: {unit2['hasAttackedThisTurn']}")
            
            if changes:
                file.write(f"  {unit1['name']}: {', '.join(changes)}\n")
    
    @staticmethod
    def validate_state(state: GameState) -> List[str]:
        """Validate state for inconsistencies"""
        errors = []
        
        # Check for units with negative models
        for unit in state.units:
            if unit['nmodels'] < 0:
                errors.append(f"{unit['name']} has negative models: {unit['nmodels']}")
        
        # Check for units out of bounds
        for unit in state.units:
            x, y, _ = unit['position']
            if abs(x) > 100 or abs(y) > 100:
                errors.append(f"{unit['name']} is out of bounds: {unit['position']}")
        
        # Check combat consistency
        for unit in state.units:
            if unit['isInCombat'] and not unit['isInCombatWith']:
                errors.append(f"{unit['name']} marked in combat but no opponents")
        
        return errors


class DecisionExplainer:
    """
    Explain why AI made a particular decision.
    """
    
    def __init__(self, tree: MinimaxTree):
        self.tree = tree
    
    def explain_decision(self, file: TextIO = sys.stdout):
        """Explain the AI's decision process"""
        if not self.tree.root or not self.tree.root.best_child:
            file.write("No decision to explain\n")
            return
        
        file.write("=== AI Decision Explanation ===\n\n")
        
        root = self.tree.root
        best = root.best_child
        
        file.write(f"Chosen Action: {best.action}\n")
        file.write(f"Expected Value: {best.value:.2f}\n\n")
        
        file.write("Reasoning:\n")
        
        # Compare with alternative actions
        alternatives = [c for c in root.children if c != best]
        alternatives.sort(key=lambda c: c.value or float('-inf'), reverse=True)
        
        file.write(f"\nTop alternatives considered:\n")
        for i, alt in enumerate(alternatives[:5], 1):
            value_diff = (best.value or 0) - (alt.value or 0)
            file.write(f"  {i}. {alt.action} (value={alt.value:.2f}, ")
            file.write(f"diff={value_diff:+.2f})\n")
        
        # Explain what happens if we take best action
        file.write(f"\nIf we take this action:\n")
        self._explain_consequences(best, file)
    
    def _explain_consequences(self, node: GameStateNode, file: TextIO):
        """Explain consequences of an action"""
        # Compare before/after states
        if node.parent:
            before = node.parent.state
            after = node.state
            
            # Model count changes
            for unit in after.units:
                before_unit = before.get_unit_by_name(unit['name'])
                if before_unit and before_unit['nmodels'] != unit['nmodels']:
                    diff = unit['nmodels'] - before_unit['nmodels']
                    file.write(f"  - {unit['name']} models: {before_unit['nmodels']} -> "
                             f"{unit['nmodels']} ({diff:+d})\n")


# Example usage in main application:
"""
from treeVisualization import TreeVisualizer, GameStateDebugger, DecisionExplainer

# After AI makes a decision:
visualizer = TreeVisualizer(ai.tree)

# Print ASCII tree
visualizer.print_tree_ascii(max_depth=3)

# Print best path
visualizer.print_best_path()

# Export to JSON for web visualization
visualizer.export_to_json('tree_visualization.json')

# Print detailed statistics
visualizer.print_statistics_detailed()

# Explain decision
explainer = DecisionExplainer(ai.tree)
explainer.explain_decision()

# Debug current state
GameStateDebugger.print_state(current_state)

# Validate state
errors = GameStateDebugger.validate_state(current_state)
if errors:
    print("State validation errors:")
    for error in errors:
        print(f"  - {error}")

# Save detailed analysis to file
with open('ai_decision_log.txt', 'w') as f:
    visualizer.print_best_path(file=f)
    visualizer.print_statistics_detailed(file=f)
    explainer.explain_decision(file=f)
"""
