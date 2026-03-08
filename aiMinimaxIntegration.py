"""
Integration example: Using MinimaxTree with the existing AI system
This shows how to enhance ClassAI with minimax decision-making
"""

from concurrent.futures import ThreadPoolExecutor
import math
from panda3d.core import ClockObject

globalClock = ClockObject.getGlobalClock()

from gameStateTree import GameState, GameAction
from minimaxOptimizations import OptimizedMinimaxTree
from gameStateAnalyzer import GameStateAnalyzer
from strategyAdvisor import StrategyAdvisor
from unitTypeClassifier import UnitTypeClassifier, UnitType, SupportRole, MATCHUP_TABLE
from treeVisualization import DecisionExplainer, TreeVisualizer
from direct.showbase.DirectObject import DirectObject
from direct.task import Task

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
        self.active = True
        
        # Initialize analyzer, classifier and strategy advisor
        self.analyzer = GameStateAnalyzer(game)
        self.classifier = UnitTypeClassifier()
        self.advisor = StrategyAdvisor(self.classifier)
        self.use_minimax = use_minimax
        self.minimax_depth = minimax_depth
        
        # Cached strategy info (refreshed each decision)
        self._current_strategy = None
        self._tactical_roles = {}
        
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

        self._move_complete = False
        self.helper1 = DirectObject()
        self.helper1.accept('unit-move-complete', self.endLoopWaitForMoveComplete)
    
    async def make_decision(self):
        """
        Main decision-making function.
        Decides between minimax (slow but optimal) or heuristics (fast but suboptimal).
        """
        # Capture current game state
        current_state = GameState.from_game(self.game)
        
        # Use minimax for critical decisions, heuristics for simple ones
        if self.use_minimax and self._should_use_minimax(current_state):
            return await self._minimax_decision(current_state)
        else:
            return self._heuristic_decision(current_state)
    
    def _should_use_minimax(self, state: GameState) -> bool:
        """
        Determine if minimax should be used for this decision.
        Use minimax for important decisions, heuristics for trivial ones.
        """
        #return True  # For simplicity, always use minimax
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
    
    _executor = ThreadPoolExecutor(max_workers=1)

    async def _minimax_decision(self, state: GameState) -> GameAction:
        """Make decision using minimax algorithm (runs search in background thread)"""
        self.minimax_decisions += 1
        self.decisions_made += 1
        
        # Find best move with time limit (iterative deepening)
        # Set time_limit based on game urgency (in seconds)
        time_limit = 3.0
        # Submit to a background thread so the main loop stays responsive
        future = self._executor.submit(
            self.tree.find_best_move_timed, state, time_limit
        )
        # Poll the future, yielding back to Panda3D each frame
        while not future.done():
            await Task.pause(0)  # yield one frame to the task manager
        best_action, expected_value = future.result()
        
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
        """Make decision using strategy-advisor-driven heuristics.
        
        Each unit receives a tactical role from the StrategyAdvisor based on
        its unit-type classification and the army-level strategy.  The role
        determines *how* the unit moves, *what* it targets, and *why*.
        """
        self.heuristic_decisions += 1
        self.decisions_made += 1

        current = state.current_player
        player_units = state.get_player_units(current)
        enemy_units  = state.get_player_units(3 - current)

        # ── 1. Pick army-level strategy ──────────────────────────────
        top_strats = self.advisor.recommend_strategies(
            player_units, from_dict=True, top_n=1)
        if top_strats:
            self._current_strategy, fit = top_strats[0]
        else:
            self._current_strategy = None
            fit = 0.0

        # ── 2. Assign per-unit tactical roles ────────────────────────
        self._tactical_roles = self.advisor.assign_tactical_roles(
            player_units, enemy_units)

        strat_name = self._current_strategy.name if self._current_strategy else 'None'
        print(f"\n[AI Heuristic] Player {current}")
        print(f"  Strategy: {strat_name} ({fit:.0%} fit)")
        for uname, info in self._tactical_roles.items():
            print(f"    {uname:20s} -> {info['role']:10s} "
                  f"target={info['target'] or '-':20s} ({info['reason']})")

        # ── Push roles onto the actual unit objects for display ───────
        for uname, info in self._tactical_roles.items():
            unit_obj = self._get_unit_by_name(uname)
            if unit_obj:
                unit_obj.tacticalRole = info
                unit_obj.updateTextNode()

        # ── 3. Generate the next concrete action ─────────────────────
        return self._strategy_aware_action(state, player_units, enemy_units)

    # ------------------------------------------------------------------
    # Strategy-aware action generation (replaces aggressive/defensive/balanced)
    # ------------------------------------------------------------------

    def _strategy_aware_action(self, state: GameState,
                               player_units, enemy_units) -> GameAction:
        """Pick the next action by iterating units in priority order and
        translating their tactical role into a concrete GameAction.

        The army-level strategy modifies which units act first and how
        each role translates into movement."""

        phase = state.current_phase
        strat_name = self._current_strategy.name if self._current_strategy else None

        # ── Role priority varies by strategy ──────────────────────────
        ROLE_PRIORITY = self._get_role_priority(strat_name)

        def _unit_priority(u):
            role_info = self._tactical_roles.get(u['name'])
            if not role_info:
                return 99
            return ROLE_PRIORITY.get(role_info['role'], 50)

        ordered = sorted(player_units, key=_unit_priority)

        # ── MOVEMENT PHASE ───────────────────────────────────────────
        if phase == 'MovementPhase':
            for unit in ordered:
                if unit['hasMovedThisTurn']:
                    continue
                role_info = self._tactical_roles.get(unit['name'])
                if not role_info:
                    continue

                role   = role_info['role']
                target_name = role_info['target']
                target = self._find_unit_dict(target_name, enemy_units) if target_name else None

                move = self._movement_for_role(
                    unit, role, target, player_units, enemy_units,
                    strat_name)
                if move:
                    return move

        # ── SHOOTING PHASE ───────────────────────────────────────────
        elif phase == 'ShootingPhase':
            for unit in ordered:
                if unit['hasAttackedThisTurn'] or not unit.get('ranged'):
                    continue
                role_info = self._tactical_roles.get(unit['name'])
                if not role_info:
                    continue

                # Shooting units use their advisor-assigned target;
                # non-shooting units with ranged weapons fire opportunistically
                target_name = role_info['target']
                if role_info['role'] == 'SHOOT' and target_name:
                    return GameAction('shoot', unit['name'],
                                     {'target': target_name})
                elif target_name:
                    # Non-shooting roles still fire if they have a ranged weapon.
                    # Pick the advisor target if reachable, else nearest enemy.
                    return GameAction('shoot', unit['name'],
                                     {'target': target_name})
                elif enemy_units:
                    nearest = min(enemy_units,
                                  key=lambda e: self._distance(
                                      unit['position'], e['position']))
                    return GameAction('shoot', unit['name'],
                                     {'target': nearest['name']})

        # ── COMBAT PHASE ─────────────────────────────────────────────
        elif phase == 'CombatPhase':
            for unit in ordered:
                if not unit['isInCombat'] or unit['hasAttackedThisTurn']:
                    continue
                if unit['isInCombatWith']:
                    return GameAction('attack', unit['name'],
                                     {'target': unit['isInCombatWith'][0]})

        return GameAction('end_phase', 'system', {})

    # ------------------------------------------------------------------
    # Strategy-dependent role priorities
    # ------------------------------------------------------------------

    @staticmethod
    def _get_role_priority(strat_name):
        """Return role-priority dict tuned for the active strategy.

        Lower number = acts first.  The strategy shifts which units
        the AI processes first, so the most doctrinally-important
        units always get their move before the rest.
        """
        BASE = {
            'RALLY': 0, 'SHOOT': 1, 'REDIRECT': 2, 'SCREEN': 3,
            'BLOCK': 4, 'HOLD': 5, 'FLANK': 6, 'CHARGE': 7,
            'ENGAGE': 8, 'ADVANCE': 9, 'FIGHT': 10,
        }
        if strat_name == 'Hammer and Anvil':
            # Anvils pin first, then hammers strike the engaged foe
            return {**BASE, 'BLOCK': 1, 'HOLD': 2, 'CHARGE': 6, 'FLANK': 7}
        if strat_name == 'Gunline':
            # Shooting is king; screens protect; melee stays back
            return {**BASE, 'SHOOT': 0, 'SCREEN': 1, 'HOLD': 2,
                    'ADVANCE': 10, 'CHARGE': 10, 'ENGAGE': 10}
        if strat_name == 'Refused Flank':
            # Hammers & flankers first to create the concentration
            return {**BASE, 'CHARGE': 1, 'FLANK': 2, 'ENGAGE': 3}
        if strat_name == 'Horde Rush':
            # Everyone charges — advance roles first
            return {**BASE, 'ADVANCE': 1, 'CHARGE': 2, 'ENGAGE': 3,
                    'SCREEN': 8, 'HOLD': 9}
        if strat_name == 'Fast Strike':
            # Flanking units are the core; nobody goes head-on
            return {**BASE, 'FLANK': 1, 'CHARGE': 5, 'ADVANCE': 8}
        if strat_name == 'Attrition Grind':
            # Blockers and holders first; measured advance
            return {**BASE, 'BLOCK': 1, 'HOLD': 2, 'ENGAGE': 3,
                    'ADVANCE': 4, 'CHARGE': 7}
        return BASE

    # ------------------------------------------------------------------
    # Per-role movement vectors (strategy-modified)
    # ------------------------------------------------------------------

    def _movement_for_role(self, unit, role, target, friendlies, enemies,
                           strat_name=None) -> GameAction | None:
        """Translate a tactical role into a movement GameAction.

        The active strategy name modifies how each role behaves:
          - Hammer & Anvil : hammers wait for an anvil to engage first
          - Gunline        : shooters never kite; melee units screen
          - Refused Flank  : all movement biased toward one battlefield flank
          - Fast Strike    : everything tries to flank; no direct advances
          - Horde Rush     : everyone rushes forward, no screening
          - Attrition Grind: slower, tighter advance

        Returns None if this unit shouldn't move.
        """
        ux, uy = unit['position'][0], unit['position'][1]
        move_speed = unit.get('M', 4) * 2  # base movement allowance

        # ── RALLY ─────────────────────────────────────────────────────
        if role == 'RALLY':
            return None

        # ── SHOOT ─────────────────────────────────────────────────────
        if role == 'SHOOT':
            if strat_name == 'Gunline':
                # Gunline: shooters NEVER move — hold the firing line
                return None
            nearest_enemy_dist = self._nearest_enemy_distance(unit, enemies)
            if nearest_enemy_dist < 12:
                ne = min(enemies,
                         key=lambda e: self._distance(unit['position'], e['position']))
                dx = ux - ne['position'][0]
                dy = uy - ne['position'][1]
                dx, dy = self._normalize(dx, dy, move_speed * 0.5)
                return self._move_action(unit, ux + dx, uy + dy)
            return None  # stay and shoot

        # ── HOLD ──────────────────────────────────────────────────────
        if role == 'HOLD':
            if strat_name == 'Hammer and Anvil':
                # Anvil with no target still advances slowly to create
                # contact for the hammer to exploit
                return self._move_toward_target(
                    unit, enemies, move_speed * 0.5)
            if strat_name == 'Attrition Grind':
                # Slowly close the gap in a measured advance
                return self._move_toward_target(
                    unit, enemies, move_speed * 0.4)
            return None  # default: hold position

        # ── SCREEN ────────────────────────────────────────────────────
        if role == 'SCREEN':
            if strat_name == 'Horde Rush':
                # Horde Rush: no screening — rush forward instead
                return self._move_toward_target(unit, enemies, move_speed)
            act = self._screen_friendlies(unit, friendlies, enemies,
                                          move_speed)
            if act:
                return act
            return None

        # ── REDIRECT / BLOCK ──────────────────────────────────────────
        if role in ('REDIRECT', 'BLOCK'):
            if target:
                dx = target['position'][0] - ux
                dy = target['position'][1] - uy
                dx, dy = self._clamp_movement(dx, dy, move_speed)
                return self._move_action(unit, ux + dx, uy + dy)
            return None

        # ── FLANK ─────────────────────────────────────────────────────
        if role == 'FLANK':
            if target:
                tx, ty = target['position'][0], target['position'][1]
                dx, dy = tx - ux, ty - uy
                dist = (dx*dx + dy*dy) ** 0.5
                if dist > 0:
                    perp_x, perp_y = -dy / dist, dx / dist
                    approach_x, approach_y = dx / dist, dy / dist
                    blend = min(1.0, dist / 30.0)
                    # Fast Strike: wider flanking arc (60% perp)
                    swing = 0.6 if strat_name == 'Fast Strike' else 0.4
                    fx = approach_x * (1 - swing * blend) + perp_x * swing * blend
                    fy = approach_y * (1 - swing * blend) + perp_y * swing * blend
                    fx, fy = self._normalize(fx, fy, move_speed)
                    # Refused Flank: bias toward chosen flank
                    if strat_name == 'Refused Flank':
                        fx, fy = self._bias_to_flank(fx, fy, friendlies)
                    return self._move_action(unit, ux + fx, uy + fy)
            return self._move_toward_target(unit, enemies, move_speed)

        # ── CHARGE ────────────────────────────────────────────────────
        if role == 'CHARGE':
            if strat_name == 'Hammer and Anvil':
                # Only charge if an anvil is already in combat (pinning)
                anvil_engaged = any(
                    f.get('isInCombat') and
                    self._tactical_roles.get(f['name'], {}).get('role') in ('HOLD', 'BLOCK', 'FIGHT')
                    for f in friendlies if f['name'] != unit['name']
                )
                if not anvil_engaged:
                    # Anvil hasn't pinned yet — advance cautiously
                    return self._move_toward_target(
                        unit, enemies, move_speed * 0.4)
            if strat_name == 'Fast Strike':
                # Fast Strike: hammers also flank rather than charge head-on
                if target:
                    return self._flank_toward(unit, target, move_speed, swing=0.5)
            if strat_name == 'Refused Flank' and target:
                dx = target['position'][0] - ux
                dy = target['position'][1] - uy
                dx, dy = self._clamp_movement(dx, dy, move_speed)
                dx, dy = self._bias_to_flank_vec(dx, dy, friendlies)
                return self._move_action(unit, ux + dx, uy + dy)
            if target:
                dx = target['position'][0] - ux
                dy = target['position'][1] - uy
                dx, dy = self._clamp_movement(dx, dy, move_speed)
                return self._move_action(unit, ux + dx, uy + dy)
            return self._move_toward_target(unit, enemies, move_speed)

        # ── ENGAGE ────────────────────────────────────────────────────
        if role == 'ENGAGE':
            speed_mult = 0.6 if strat_name == 'Attrition Grind' else 1.0
            if strat_name == 'Gunline':
                # Gunline: melee units screen instead of engaging
                return self._screen_friendlies(unit, friendlies, enemies, move_speed)
            if strat_name == 'Fast Strike':
                # Swing to flank instead of direct engagement
                if target:
                    return self._flank_toward(unit, target, move_speed, swing=0.35)
            if target:
                dx = target['position'][0] - ux
                dy = target['position'][1] - uy
                dx, dy = self._clamp_movement(dx, dy, move_speed * speed_mult)
                return self._move_action(unit, ux + dx, uy + dy)
            return self._move_toward_target(unit, enemies, move_speed * speed_mult)

        # ── ADVANCE ───────────────────────────────────────────────────
        if role == 'ADVANCE':
            if strat_name == 'Gunline':
                # Gunline: basic units screen rather than advance
                return self._screen_friendlies(unit, friendlies, enemies, move_speed)
            if strat_name == 'Fast Strike':
                # Even basic units try a slight flank
                if enemies:
                    nearest = min(enemies,
                                  key=lambda e: self._distance(
                                      unit['position'], e['position']))
                    return self._flank_toward(unit, nearest, move_speed, swing=0.25)
            speed_mult = 0.6 if strat_name == 'Attrition Grind' else 1.0
            return self._move_toward_target(unit, enemies, move_speed * speed_mult)

        # ── FIGHT ─────────────────────────────────────────────────────
        if role == 'FIGHT':
            return None

        # Fallback: advance
        return self._move_toward_target(unit, enemies, move_speed)

    # ------------------------------------------------------------------
    # Movement helpers
    # ------------------------------------------------------------------

    def _move_action(self, unit, tx, ty, target_heading=None) -> GameAction:
        params = {'target_x': tx, 'target_y': ty}
        if target_heading is not None:
            params['target_heading'] = target_heading
        return GameAction('move', unit['name'], params)

    def _move_toward_target(self, unit, enemies, move_speed) -> GameAction | None:
        if not enemies:
            return None
        nearest = min(enemies,
                      key=lambda e: self._distance(
                          unit['position'], e['position']))
        ux, uy = unit['position'][0], unit['position'][1]
        dx = nearest['position'][0] - ux
        dy = nearest['position'][1] - uy
        dx, dy = self._clamp_movement(dx, dy, move_speed)
        return self._move_action(unit, ux + dx, uy + dy)

    def _find_unit_dict(self, name, unit_list):
        """Find a unit dict by name in a list."""
        if not name:
            return None
        for u in unit_list:
            if u.get('name') == name:
                return u
        return None

    def _nearest_enemy_distance(self, unit, enemies):
        if not enemies:
            return float('inf')
        return min(self._distance(unit['position'], e['position'])
                   for e in enemies)

    def _most_valuable_friendly(self, unit, friendlies):
        """Return the most valuable friendly unit (highest combat power) that
        isn't this unit itself."""
        best = None
        best_val = -1
        for f in friendlies:
            if f['name'] == unit['name']:
                continue
            val = f.get('nmodels', 1) * f.get('A', 1) * f.get('S', 3)
            if val > best_val:
                best_val = val
                best = f
        return best

    @staticmethod
    def _normalize(dx, dy, length):
        dist = (dx*dx + dy*dy) ** 0.5
        if dist == 0:
            return 0, 0
        return dx / dist * length, dy / dist * length

    @staticmethod
    def _clamp_movement(dx, dy, max_dist):
        dist = (dx*dx + dy*dy) ** 0.5
        if dist <= max_dist or dist == 0:
            return dx, dy
        return dx / dist * max_dist, dy / dist * max_dist

    def _distance(self, pos1, pos2):
        """Calculate distance between two positions"""
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        return (dx**2 + dy**2)**0.5

    # ------------------------------------------------------------------
    # Strategy-dependent movement helpers
    # ------------------------------------------------------------------

    def _flank_toward(self, unit, target, move_speed, swing=0.4):
        """Move toward *target* with a perpendicular swing component.
        *swing* controls how wide the arc is (0 = straight, 1 = pure perp).
        """
        ux, uy = unit['position'][0], unit['position'][1]
        tx, ty = target['position'][0], target['position'][1]
        dx, dy = tx - ux, ty - uy
        dist = (dx*dx + dy*dy) ** 0.5
        if dist == 0:
            return None
        perp_x, perp_y = -dy / dist, dx / dist
        approach_x, approach_y = dx / dist, dy / dist
        blend = min(1.0, dist / 30.0)
        fx = approach_x * (1 - swing * blend) + perp_x * swing * blend
        fy = approach_y * (1 - swing * blend) + perp_y * swing * blend
        fx, fy = self._normalize(fx, fy, move_speed)
        return self._move_action(unit, ux + fx, uy + fy)

    def _screen_friendlies(self, unit, friendlies, enemies, move_speed):
        """Position between the most valuable friendly and the nearest enemy.

        The unit advances *toward* the nearest enemy so that its heading
        naturally faces the threat (the movement arc determines heading).
        A small lateral component drifts the unit toward the ideal screen
        position (the midpoint on the valuable↔enemy axis) without turning
        so far that the flank is exposed.
        """
        valuable = self._most_valuable_friendly(unit, friendlies)
        if not valuable or not enemies:
            return None

        ne = min(enemies,
                 key=lambda e: self._distance(
                     valuable['position'], e['position']))
        ux, uy = unit['position'][0], unit['position'][1]
        ex, ey = ne['position'][0], ne['position'][1]
        vx, vy = valuable['position'][0], valuable['position'][1]

        # --- direction toward the enemy (unit will face this way) ---
        enemy_dx, enemy_dy = ex - ux, ey - uy
        dist_to_enemy = math.sqrt(enemy_dx ** 2 + enemy_dy ** 2)
        if dist_to_enemy < 0.5:
            return None
        enx, eny = enemy_dx / dist_to_enemy, enemy_dy / dist_to_enemy

        # --- ideal screen position: midpoint of valuable ↔ enemy ---
        mid_x = (vx + ex) / 2
        mid_y = (vy + ey) / 2
        screen_dx, screen_dy = mid_x - ux, mid_y - uy

        # Decompose the screen vector into advance (along enemy dir)
        # and lateral (perpendicular) components.
        advance = screen_dx * enx + screen_dy * eny
        lat_dx = screen_dx - advance * enx
        lat_dy = screen_dy - advance * eny
        lat_mag = math.sqrt(lat_dx ** 2 + lat_dy ** 2)

        # Cap lateral drift so the arc turn stays small (~17 °).
        max_lateral = max(abs(advance) * 0.3, 0.5)
        if lat_mag > max_lateral:
            scale = max_lateral / lat_mag
            lat_dx *= scale
            lat_dy *= scale

        dx = advance * enx + lat_dx
        dy = advance * eny + lat_dy
        dx, dy = self._clamp_movement(dx, dy, move_speed)

        # target_heading is used only by the minimax evaluator;
        # the real game derives heading from the movement arc.
        dest_x, dest_y = ux + dx, uy + dy
        face_dx, face_dy = ex - dest_x, ey - dest_y
        heading = math.degrees(math.atan2(face_dy, face_dx))
        return self._move_action(unit, dest_x, dest_y,
                                 target_heading=heading)


    def _bias_to_flank(self, fx, fy, friendlies):
        """Bias a movement vector toward the army's weighted flank.
        Returns modified (fx, fy)."""
        return self._bias_to_flank_vec(fx, fy, friendlies)

    def _bias_to_flank_vec(self, dx, dy, friendlies):
        """Shift a vector toward the side of the battlefield where friendly
        units are already concentrated (the 'refused flank' side)."""
        if not friendlies:
            return dx, dy
        avg_x = sum(f['position'][0] for f in friendlies) / len(friendlies)
        # Nudge 30 % toward the army centroid's X side
        bias = 0.3
        if avg_x > 0:
            dx += abs(dx) * bias
        else:
            dx -= abs(dx) * bias
        return dx, dy
    
    async def execute_action(self, action: GameAction):
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
                self.game.ball.setPos(target_pos)  # Placeholder for movement command
                self.game.pathTowardsMouse(unit,action.parameters['target_x'],
                                           action.parameters['target_y'])
                #self.game.moveUnit(unit)
                taskMgr.doMethodLater(0.1, self.game.moveUnit, "moveTask", extraArgs=[unit], appendTask=False)
                self._move_complete = False
                await taskMgr.add(self.loopWaitForMoveComplete, "waitTask", extraArgs=[unit], appendTask=True)
                pass
        
        elif action.action_type == 'shoot':
            unit = self._get_unit_by_name(action.unit_name)
            target = self._get_unit_by_name(action.parameters['target'])
            if unit and target:
                self.game.unitToMove=unit
                #target_pos = target['position']
                # Execute shooting
                # Example: self.game.shootAt(unit, target)
                #self.game.ball.setPos(target_pos)  # Placeholder for movement command
                self.game.shootAt(unit,target)
                self._move_complete = False
                await taskMgr.add(self.loopWaitForMoveComplete, "waitTask", extraArgs=[unit], appendTask=True)
                # Mark the unit as having attacked so it can't shoot again this turn
                unit.hasAttackedThisTurn = True
                pass
        
        elif action.action_type == 'attack':
            unit = self._get_unit_by_name(action.unit_name)
            target = self._get_unit_by_name(action.parameters['target'])
            if unit and target:
                # Mark the unit as having attacked so it can't attack again this turn
                #unit.hasAttackedThisTurn = True
                # Execute melee attack
                # Example: self.game.meleeAttack(unit, target)
                self.game.unitToMove=unit
                self._move_complete = False
                taskMgr.add(self.game.combat.verySimpleBattleStart, "meleeTask", appendTask=True)
                await taskMgr.add(self.loopWaitForMoveComplete, "waitTask", extraArgs=[unit], appendTask=True)
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
    
    async def take_turn(self):
        """
        Main entry point for AI turn.
        Loops, making and executing decisions, until an end_phase action is produced.
        """
        if not self.active or getattr(self, '_turn_running', False):
            return
        self.game.save_game_state('previous_phase.json')
        self._turn_running = True

        while True:
            self._move_complete = False  # Reset move completion flag at start of each decision loop
            # Make decision (await directly so the return value is preserved)
            action = await self.make_decision()

            visualizer = TreeVisualizer(self.tree)
            visualizer.print_best_path()

            print(self.game.analyzer.get_strategy_report(player_num=self.game.roundCounter.current_player))

            # Execute action
            await self.execute_action(action)

            #visualizer.print_tree_ascii(max_depth=19)

            """ explainer = DecisionExplainer(self.tree)
            explainer.explain_decision()

            visualizer.print_statistics_detailed() """

            if action.action_type == 'end_phase':
                self.game.fsm.nextPhase()
                #break

        self._turn_running = False
        return action
    
    def print_statistics(self):
        """Print AI performance statistics"""
        print(f"\n=== AI Statistics (Player {self.player_num}) ===")
        print(f"Total decisions: {self.decisions_made}")
        print(f"Minimax decisions: {self.minimax_decisions} "
              f"({self.minimax_decisions/max(1, self.decisions_made)*100:.1f}%)")
        print(f"Heuristic decisions: {self.heuristic_decisions} "
              f"({self.heuristic_decisions/max(1, self.decisions_made)*100:.1f}%)")

    def deployUnits(self):
        for unit in self.player_units:
            if not unit.isDeployed:
                self.game.unitToMove=unit
                taskMgr.add(self.game.taskLoopDeploy, "taskLoopDeploy", extraArgs=[], appendTask=True)
                break
    
    def loopWaitForMoveComplete(self,unit,task):
        if not hasattr(task, '_wait_elapsed'):
            task._wait_elapsed = 0.0
        task._wait_elapsed += globalClock.getDt()
        if task._wait_elapsed % 2.0 < globalClock.getDt():
            print(f"Waiting for move complete for unit: {unit.unit.name} ({task._wait_elapsed:.1f}s)")
        if self._move_complete:
            self._move_complete = False
            print(f"signal received for unit: {unit.unit.name}")
            return task.done
        # Safety timeout: if we've been waiting too long, force-advance
        if task._wait_elapsed > 30.0:
            print(f"TIMEOUT: unit-move-complete never received for {unit.unit.name} after {task._wait_elapsed:.1f}s, forcing advance")
            self._move_complete = False
            return task.done
        return task.cont
    
    def endLoopWaitForMoveComplete(self):
        self._move_complete = True
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
