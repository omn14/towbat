"""
Complete example demonstrating the minimax game tree system.
Run this to see the AI in action with full debugging output.
"""

from gameStateTree import GameState, MinimaxTree, GameAction
from minimaxOptimizations import OptimizedMinimaxTree, TranspositionTable
from treeVisualization import TreeVisualizer, GameStateDebugger, DecisionExplainer
from gameStateAnalyzer import GameStateAnalyzer


class MockGame:
    """
    Mock game object for testing without full game engine.
    Replace with your actual game object in production.
    """
    def __init__(self):
        # Mock FSM
        self.fsm = type('FSM', (), {
            'phases': ['StrategyPhase', 'MovementPhase', 'ShootingPhase', 'CombatPhase'],
            'currentPhaseIndex': 1
        })()
        
        # Mock round counter
        self.roundCounter = type('RoundCounter', (), {
            'current_round': 1,
            'current_player': 1,
            'max_rounds': 6
        })()
        
        # Mock units
        self.units = []
        self.player1Units = []
        self.player2Units = []
        
        # Create some test units
        self._create_test_units()
    
    def _create_test_units(self):
        """Create mock units for testing"""
        # Player 1 units
        for i in range(2):
            unit = self._create_mock_unit(
                name=f"P1_Unit_{i}",
                player=1,
                pos=(i * 10 - 5, -10, 0),
                nmodels=10
            )
            self.units.append(unit)
            self.player1Units.append(unit)
        
        # Player 2 units
        for i in range(2):
            unit = self._create_mock_unit(
                name=f"P2_Unit_{i}",
                player=2,
                pos=(i * 10 - 5, 10, 0),
                nmodels=10
            )
            self.units.append(unit)
            self.player2Units.append(unit)
    
    def _create_mock_unit(self, name, player, pos, nmodels):
        """Create a mock unit object"""
        return type('Unit', (), {
            'unitName': name,
            'bodyNP': type('NodePath', (), {
                'getPos': lambda: type('Vec3', (), {'__iter__': lambda s: iter(pos)})(),
                'getH': lambda: 0 if player == 1 else 180,
                'isEmpty': lambda: False
            })(),
            'unit': type('UnitData', (), {
                'nmodels': nmodels,
                'files': 5,
                'ranks': 2,
                'model': type('Model', (), {
                    'characteristics': {
                        'WS': 3, 'S': 3, 'T': 3, 'A': 1, 'Ld': 7
                    },
                    'armor_save': 6,
                    'charging': False,
                    'special_rules': []
                })()
            })(),
            'state': 'Idle',
            'isInCombat': False,
            'hasMovedThisTurn': False,
            'hasAttackedThisTurn': False,
            'attemptedRallyThisTurn': False,
            'isDeployed': True,
            'isInCombatWith': [],
            'isInCombatFlank': []
        })()


def example_basic_minimax():
    """Basic minimax example"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Minimax (Depth 2)")
    print("="*70)
    
    # Create mock game
    game = MockGame()
    
    # Create analyzer and tree
    analyzer = GameStateAnalyzer(game)
    tree = MinimaxTree(analyzer, max_depth=2)
    
    # Capture current state
    current_state = GameState.from_game(game)
    
    print("\n--- Initial State ---")
    GameStateDebugger.print_state(current_state)
    
    # Find best move
    print("\n--- Searching for best move ---")
    best_action, expected_value = tree.find_best_move(current_state)
    
    print(f"\n✓ Best action: {best_action}")
    print(f"✓ Expected value: {expected_value:.2f}")
    
    # Show statistics
    stats = tree.get_statistics()
    print(f"\nStatistics:")
    print(f"  Nodes evaluated: {stats['nodes_evaluated']}")
    print(f"  Nodes pruned: {stats['nodes_pruned']}")
    print(f"  Pruning efficiency: {stats['pruning_efficiency']:.1f}%")


def example_optimized_minimax():
    """Optimized minimax with all features"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Optimized Minimax (Depth 4)")
    print("="*70)
    
    game = MockGame()
    analyzer = GameStateAnalyzer(game)
    
    # Create optimized tree
    tree = OptimizedMinimaxTree(
        analyzer,
        max_depth=4,
        use_transposition_table=True,
        use_move_ordering=True,
        use_iterative_deepening=True
    )
    
    current_state = GameState.from_game(game)
    
    print("\n--- Searching with time limit (3 seconds) ---")
    best_action, expected_value = tree.find_best_move_timed(
        current_state,
        time_limit=3.0
    )
    
    print(f"\n✓ Best action: {best_action}")
    print(f"✓ Expected value: {expected_value:.2f}")
    
    # Extended statistics
    stats = tree.get_statistics()
    print(f"\nDetailed Statistics:")
    print(f"  Nodes evaluated: {stats['nodes_evaluated']}")
    print(f"  Nodes pruned: {stats['nodes_pruned']}")
    print(f"  Pruning efficiency: {stats['pruning_efficiency']:.1f}%")
    print(f"  Transposition table size: {stats['tt_size']} entries")
    print(f"  Cache hit rate: {stats['tt_hit_rate']:.1f}%")


def example_tree_visualization():
    """Visualize the decision tree"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Tree Visualization")
    print("="*70)
    
    game = MockGame()
    analyzer = GameStateAnalyzer(game)
    tree = MinimaxTree(analyzer, max_depth=2)
    
    current_state = GameState.from_game(game)
    best_action, expected_value = tree.find_best_move(current_state)
    
    print("\n--- ASCII Tree Structure ---")
    visualizer = TreeVisualizer(tree)
    visualizer.print_tree_ascii(max_depth=2)
    
    print("\n--- Best Path ---")
    visualizer.print_best_path()
    
    print("\n--- Decision Explanation ---")
    explainer = DecisionExplainer(tree)
    explainer.explain_decision()


def example_state_comparison():
    """Compare states before and after action"""
    print("\n" + "="*70)
    print("EXAMPLE 4: State Comparison")
    print("="*70)
    
    game = MockGame()
    analyzer = GameStateAnalyzer(game)
    tree = MinimaxTree(analyzer, max_depth=2)
    
    # Get initial state
    initial_state = GameState.from_game(game)
    
    # Find and simulate best move
    best_action, _ = tree.find_best_move(initial_state)
    
    # Apply action to get new state
    new_state = tree._apply_action(initial_state, best_action)
    
    print("\n--- State Comparison ---")
    GameStateDebugger.compare_states(initial_state, new_state)


def example_iterative_deepening():
    """Show iterative deepening in action"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Iterative Deepening")
    print("="*70)
    
    game = MockGame()
    analyzer = GameStateAnalyzer(game)
    
    tree = OptimizedMinimaxTree(
        analyzer,
        max_depth=6,  # Maximum depth
        use_transposition_table=True,
        use_move_ordering=True,
        use_iterative_deepening=True
    )
    
    current_state = GameState.from_game(game)
    
    print("\n--- Iterative Deepening Search (5 second limit) ---")
    print("Will search depths 1, 2, 3... until time expires\n")
    
    best_action, value = tree.find_best_move_timed(
        current_state,
        time_limit=5.0
    )
    
    print(f"\n✓ Best action found: {best_action}")
    print(f"✓ Value: {value:.2f}")


def example_performance_comparison():
    """Compare basic vs optimized performance"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Performance Comparison")
    print("="*70)
    
    game = MockGame()
    analyzer = GameStateAnalyzer(game)
    current_state = GameState.from_game(game)
    
    import time
    
    # Test basic tree
    print("\n--- Basic Minimax (depth 3) ---")
    basic_tree = MinimaxTree(analyzer, max_depth=3)
    start = time.time()
    basic_action, basic_value = basic_tree.find_best_move(current_state)
    basic_time = time.time() - start
    basic_stats = basic_tree.get_statistics()
    
    print(f"Time: {basic_time:.3f}s")
    print(f"Nodes: {basic_stats['nodes_evaluated']}")
    print(f"Best action: {basic_action}")
    
    # Test optimized tree
    print("\n--- Optimized Minimax (depth 3) ---")
    opt_tree = OptimizedMinimaxTree(
        analyzer,
        max_depth=3,
        use_transposition_table=True,
        use_move_ordering=True
    )
    start = time.time()
    opt_action, opt_value = opt_tree.find_best_move(current_state)
    opt_time = time.time() - start
    opt_stats = opt_tree.get_statistics()
    
    print(f"Time: {opt_time:.3f}s")
    print(f"Nodes: {opt_stats['nodes_evaluated']}")
    print(f"Best action: {opt_action}")
    
    # Compare
    print("\n--- Comparison ---")
    speedup = basic_time / max(opt_time, 0.001)
    node_reduction = (1 - opt_stats['nodes_evaluated'] / basic_stats['nodes_evaluated']) * 100
    
    print(f"Speedup: {speedup:.2f}x faster")
    print(f"Node reduction: {node_reduction:.1f}% fewer nodes")
    print(f"Same answer? {basic_action == opt_action}")


def example_validation():
    """Validate game states"""
    print("\n" + "="*70)
    print("EXAMPLE 7: State Validation")
    print("="*70)
    
    game = MockGame()
    current_state = GameState.from_game(game)
    
    print("\n--- Validating state ---")
    errors = GameStateDebugger.validate_state(current_state)
    
    if errors:
        print("❌ Validation errors found:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("✓ State is valid!")
    
    # Create invalid state
    print("\n--- Creating invalid state ---")
    invalid_state = current_state.clone()
    invalid_state.units[0]['nmodels'] = -5  # Invalid!
    invalid_state.units[1]['position'] = (200, 200, 0)  # Out of bounds!
    
    errors = GameStateDebugger.validate_state(invalid_state)
    print(f"\n❌ Found {len(errors)} errors:")
    for error in errors:
        print(f"  - {error}")


def run_all_examples():
    """Run all examples"""
    print("\n" + "="*70)
    print("MINIMAX GAME TREE SYSTEM - COMPLETE EXAMPLES")
    print("="*70)
    
    examples = [
        ("Basic Minimax", example_basic_minimax),
        ("Optimized Minimax", example_optimized_minimax),
        ("Tree Visualization", example_tree_visualization),
        ("State Comparison", example_state_comparison),
        ("Iterative Deepening", example_iterative_deepening),
        ("Performance Comparison", example_performance_comparison),
        ("State Validation", example_validation),
    ]
    
    for name, func in examples:
        try:
            func()
        except Exception as e:
            print(f"\n❌ Error in {name}: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "-"*70)
        input("Press Enter to continue to next example...")
    
    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETED")
    print("="*70)
    print("\nKey Takeaways:")
    print("1. Basic minimax works but optimized version is much faster")
    print("2. Alpha-beta pruning typically cuts 50-90% of nodes")
    print("3. Transposition table provides additional 30-60% speedup")
    print("4. Iterative deepening allows time-bounded search")
    print("5. Tree visualization helps understand AI decisions")
    print("6. State validation catches simulation errors")
    print("\nReady to integrate with your game!")


if __name__ == "__main__":
    # Run specific example or all
    import sys
    
    if len(sys.argv) > 1:
        example_num = int(sys.argv[1])
        examples = [
            example_basic_minimax,
            example_optimized_minimax,
            example_tree_visualization,
            example_state_comparison,
            example_iterative_deepening,
            example_performance_comparison,
            example_validation,
        ]
        
        if 1 <= example_num <= len(examples):
            examples[example_num - 1]()
        else:
            print(f"Example {example_num} not found. Use 1-{len(examples)}")
    else:
        # Run all examples interactively
        run_all_examples()
