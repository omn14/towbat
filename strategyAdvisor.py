"""
Strategy Advisor Module

Analyses army composition by unit type and suggests high-level strategies,
counter-strategies, and per-unit tactical roles based on the classification
system from unitTypeClassifier.py.

Designed to plug into both the pre-game list-building phase and the in-game
AI evaluation.  The advisor works on three levels:

  1. Army-level strategy  — overall doctrine given your type mix
  2. Matchup advice       — how your composition fares against the opponent
  3. Per-unit tactical role — what each unit should be doing this turn
"""

from typing import Dict, List, Tuple, Any, Optional
from unitTypeClassifier import (
    UnitTypeClassifier, UnitType, SupportRole,
    MATCHUP_TABLE, FLANK_BONUS, REAR_BONUS,
)


# ── Pre-canned strategy doctrines ────────────────────────────────────

class Strategy:
    """Named strategy with description, prerequisites, and counter."""
    def __init__(self, name: str, description: str,
                 requires: Dict[str, int],
                 strengths: str, weaknesses: str):
        self.name = name
        self.description = description
        self.requires = requires       # minimum unit-type counts
        self.strengths = strengths
        self.weaknesses = weaknesses

    def __repr__(self):
        return f"Strategy({self.name})"


# Canonical strategies (expandable)
STRATEGIES = [
    Strategy(
        name="Hammer and Anvil",
        description=(
            "Pin the enemy with your anvil units while hammer units strike "
            "the flanks or rear. The anvil holds indefinitely; the hammer "
            "breaks anything it touches. Classic combined-arms."
        ),
        requires={'anvil': 1, 'hammer': 1},
        strengths="Dominates when anvil contacts first and hammer gets rear/flank.",
        weaknesses="Vulnerable if hammer is shot off before contact, or if fast enemy "
                   "units redirect the hammer.",
    ),
    Strategy(
        name="Refused Flank",
        description=(
            "Concentrate your striking power on one side of the battlefield. "
            "Use cannon fodder or shooting to delay the enemy on the other "
            "side. Overwhelm locally before the rest of their army arrives."
        ),
        requires={'hammer': 1},
        strengths="Achieves local superiority. Fast units excel at exploiting the gap.",
        weaknesses="If the delay screen folds too quickly, you get surrounded.",
    ),
    Strategy(
        name="Gunline",
        description=(
            "Deploy shooting units in a defensive formation. Anvil or basic "
            "units screen the front. Whittle the enemy down before contact. "
            "Engage only a weakened foe."
        ),
        requires={'shooting': 2},
        strengths="Forces the enemy to come to you under fire. Great vs slow hammer armies.",
        weaknesses="Fast units and flying units bypass the screen. Magic-heavy armies "
                   "can outshoot you.",
    ),
    Strategy(
        name="Horde Rush",
        description=(
            "Overwhelm with numbers. Flood the field with basic and cannon "
            "fodder units. Accept losses but ensure you can always bring "
            "more bodies to every fight and win through combat resolution."
        ),
        requires={'basic': 3},
        strengths="Hard to stop with few hammer units. Flanks are always available.",
        weaknesses="Shooting and magic tear through hordes. One hammer unit on the flank "
                   "can cascade panic.",
    ),
    Strategy(
        name="Fast Strike",
        description=(
            "Use fast and flying units to pick engagements. Avoid unfavorable "
            "fights. Flank and rear-charge everything. The opponent cannot "
            "protect every angle."
        ),
        requires={'fast': 2},
        strengths="Incredible mobility allows favorable matchups every turn.",
        weaknesses="Low model count means casualties hurt. Shooting phases are dangerous.",
    ),
    Strategy(
        name="Attrition Grind",
        description=(
            "Engage with superior and anvil units across the line. You don't "
            "break enemy units quickly, but you don't break either. Slowly "
            "grind them down through combat resolution and static combat res."
        ),
        requires={'anvil': 1, 'superior': 1},
        strengths="Very resilient. Hard to outfight on even terms.",
        weaknesses="Slow. Hammer units or flank charges can break the grind.",
    ),
]


class StrategyAdvisor:
    """
    Analyses army lists and game states to recommend strategies and
    per-unit tactical roles.
    """

    def __init__(self, classifier: Optional[UnitTypeClassifier] = None):
        self.classifier = classifier or UnitTypeClassifier()

    # ──────────────────────────────────────────────────────────────────
    # 1. Army composition analysis
    # ──────────────────────────────────────────────────────────────────

    def analyse_composition(self, units: list, from_dict: bool = False
                            ) -> Dict[str, Any]:
        """
        Break down an army into type counts and percentages.
        
        Args:
            units: list of model objects or unit dicts
            from_dict: True if units are dicts (GameState.units format)
        
        Returns:
            Dict with 'types', 'roles', 'classifications', 'summary'
        """
        classifications = self.classifier.classify_army(units, from_dict=from_dict)

        type_counts: Dict[str, int] = {}
        role_counts: Dict[str, int] = {}
        for name, (main_t, support_r) in classifications.items():
            t = main_t.value
            r = support_r.value
            type_counts[t] = type_counts.get(t, 0) + 1
            role_counts[r] = role_counts.get(r, 0) + 1

        total = max(1, len(classifications))
        type_pct = {k: v / total * 100 for k, v in type_counts.items()}

        summary_lines = [f"Army Composition ({total} units):"]
        for t in UnitType:
            count = type_counts.get(t.value, 0)
            pct = type_pct.get(t.value, 0)
            if count > 0:
                summary_lines.append(f"  {t.value.replace('_',' ').title():15s}: {count} ({pct:.0f}%)")
        for r in SupportRole:
            if r == SupportRole.NONE:
                continue
            count = role_counts.get(r.value, 0)
            if count > 0:
                summary_lines.append(f"  + {r.value.title():13s}: {count}")

        return {
            'types': type_counts,
            'roles': role_counts,
            'classifications': classifications,
            'type_pct': type_pct,
            'summary': '\n'.join(summary_lines),
        }

    # ──────────────────────────────────────────────────────────────────
    # 2. Strategy recommendation
    # ──────────────────────────────────────────────────────────────────

    def recommend_strategies(self, units: list, from_dict: bool = False,
                             top_n: int = 3) -> List[Tuple[Strategy, float]]:
        """
        Score every strategy against the army composition and return
        the best matches.
        
        Returns:
            List of (Strategy, fit_score) sorted best-first.
            fit_score 1.0 = perfect fit, 0.0 = not viable.
        """
        comp = self.analyse_composition(units, from_dict=from_dict)
        type_counts = comp['types']
        role_counts = comp['roles']

        # Merge support roles into type counts for matching
        merged = dict(type_counts)
        for role_name, count in role_counts.items():
            if role_name != 'none':
                merged[role_name] = merged.get(role_name, 0) + count

        scored: List[Tuple[Strategy, float]] = []
        for strat in STRATEGIES:
            fit = self._score_strategy_fit(strat, merged)
            scored.append((strat, fit))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_n]

    def _score_strategy_fit(self, strategy: Strategy,
                            merged_counts: Dict[str, int]) -> float:
        """
        How well does the army fit this strategy?
        Returns 0.0 – 1.0.
        """
        if not strategy.requires:
            return 0.5  # generic

        met = 0
        total_req = len(strategy.requires)
        bonus = 0.0

        for req_type, req_count in strategy.requires.items():
            have = merged_counts.get(req_type, 0)
            if have >= req_count:
                met += 1
                # Extra units of the required type are a bonus
                bonus += min(0.15, (have - req_count) * 0.05)
            elif have > 0:
                # Partial credit
                met += have / req_count * 0.5

        base = met / total_req
        return min(1.0, base + bonus)

    # ──────────────────────────────────────────────────────────────────
    # 3. Matchup analysis (your army vs opponent)
    # ──────────────────────────────────────────────────────────────────

    def analyse_matchup(self, my_units: list, enemy_units: list,
                        from_dict: bool = False) -> Dict[str, Any]:
        """
        Compare two armies and assess overall advantage plus key matchups.
        """
        my_comp = self.analyse_composition(my_units, from_dict=from_dict)
        enemy_comp = self.analyse_composition(enemy_units, from_dict=from_dict)

        my_types = my_comp['types']
        enemy_types = enemy_comp['types']

        # Aggregate expected matchup score across all pairings
        total_score = 0.0
        pairings = 0
        for my_type_str, my_count in my_types.items():
            my_t = UnitType(my_type_str)
            for e_type_str, e_count in enemy_types.items():
                e_t = UnitType(e_type_str)
                interaction_count = my_count * e_count
                matchup = MATCHUP_TABLE[my_t][e_t]
                total_score += (matchup - 1.0) * interaction_count
                pairings += interaction_count

        avg_matchup = total_score / max(1, pairings)

        # Identify biggest threats and best targets
        threats = []
        targets = []
        for e_name, (e_type, e_role) in enemy_comp['classifications'].items():
            # A threat is something we don't matchup well against
            worst_matchup = 3.0
            for my_name, (my_type, my_role) in my_comp['classifications'].items():
                m = MATCHUP_TABLE[my_type][e_type]
                worst_matchup = min(worst_matchup, m)
            if worst_matchup < 0.8:
                threats.append((e_name, e_type, e_role, worst_matchup))

            # A target is something we can easily crush
            best_matchup = 0.0
            for my_name, (my_type, my_role) in my_comp['classifications'].items():
                m = MATCHUP_TABLE[my_type][e_type]
                best_matchup = max(best_matchup, m)
            if best_matchup > 1.5:
                targets.append((e_name, e_type, e_role, best_matchup))

        threats.sort(key=lambda x: x[3])
        targets.sort(key=lambda x: x[3], reverse=True)

        if avg_matchup > 0.15:
            verdict = "FAVORABLE — your army composition has the edge"
        elif avg_matchup > -0.15:
            verdict = "EVEN — tactical execution will decide the battle"
        else:
            verdict = "UNFAVORABLE — opponent has compositional advantage"

        return {
            'verdict': verdict,
            'avg_matchup': avg_matchup,
            'threats': threats,
            'targets': targets,
            'my_composition': my_comp,
            'enemy_composition': enemy_comp,
        }

    # ──────────────────────────────────────────────────────────────────
    # 4. Per-unit tactical role assignment (in-game)
    # ──────────────────────────────────────────────────────────────────

    def assign_tactical_roles(self, my_units: List[Dict], enemy_units: List[Dict]
                              ) -> Dict[str, Dict[str, Any]]:
        """
        Given the current game state (units as dicts), assign each friendly
        unit a tactical role and a priority target.
        
        Returns:
            {unit_name: {'role': str, 'target': str|None, 'reason': str}}
        """
        my_comp = self.classifier.classify_army(my_units, from_dict=True)
        enemy_comp = self.classifier.classify_army(enemy_units, from_dict=True)
        living_enemies = [e for e in enemy_units if e.get('nmodels', 0) > 0]

        assignments: Dict[str, Dict[str, Any]] = {}

        for unit in my_units:
            if unit.get('nmodels', 0) <= 0:
                continue

            name = unit.get('name', 'unknown')
            main_type, support_role = my_comp.get(name, (UnitType.BASIC, SupportRole.NONE))

            role, target, reason = self._decide_role(
                unit, main_type, support_role, living_enemies, enemy_comp
            )

            assignments[name] = {
                'role': role,
                'target': target,
                'reason': reason,
                'unit_type': main_type.value,
                'support_role': support_role.value,
            }

        return assignments

    def _decide_role(self, unit: Dict, main_type: UnitType, support_role: SupportRole,
                     living_enemies: List[Dict],
                     enemy_comp: Dict[str, Tuple[UnitType, SupportRole]]
                     ) -> Tuple[str, Optional[str], str]:
        """Return (role_label, target_name_or_None, reason_string)."""

        if unit.get('state') == 'IsFleeing':
            return ('RALLY', None, 'Unit is fleeing — attempt to rally')

        if unit.get('isInCombat'):
            return ('FIGHT', unit.get('isInCombatWith', [None])[0] if unit.get('isInCombatWith') else None,
                    'Already engaged — fight the current opponent')

        if not living_enemies:
            return ('ADVANCE', None, 'No enemies remaining — advance')

        # ── SHOOTING units: stay back, pick the juiciest target ──
        if support_role == SupportRole.SHOOTING:
            best_target = self._best_shooting_target(unit, living_enemies, enemy_comp)
            return ('SHOOT', best_target, 'Ranged unit — whittle down threats from range')

        # ── FAST units: hunt flanks ──
        if support_role == SupportRole.FAST:
            best_target = self._best_flank_target(unit, living_enemies, enemy_comp)
            return ('FLANK', best_target, 'Fast unit — maneuver for a flank or rear charge')

        # ── HAMMER units: charge the most impactful target ──
        if main_type == UnitType.HAMMER:
            best_target = self._best_charge_target(unit, living_enemies, enemy_comp, main_type)
            return ('CHARGE', best_target, 'Hammer unit — seek the decisive charge')

        # ── ANVIL units: move to block enemy hammers or hold center ──
        if main_type == UnitType.ANVIL:
            enemy_hammer = self._find_enemy_type(living_enemies, enemy_comp, UnitType.HAMMER)
            if enemy_hammer:
                return ('BLOCK', enemy_hammer, 'Anvil unit — intercept enemy hammer')
            return ('HOLD', None, 'Anvil unit — hold the center line')

        # ── SUPERIOR units: pick favorable fights ──
        if main_type == UnitType.SUPERIOR:
            best_target = self._best_charge_target(unit, living_enemies, enemy_comp, main_type)
            return ('ENGAGE', best_target, 'Superior unit — engage targets you outclass')

        # ── CANNON FODDER: redirect or screen ──
        if main_type == UnitType.CANNON_FODDER:
            enemy_hammer = self._find_enemy_type(living_enemies, enemy_comp, UnitType.HAMMER)
            if enemy_hammer:
                return ('REDIRECT', enemy_hammer,
                        'Cannon fodder — redirect enemy hammer into a bad position')
            return ('SCREEN', None, 'Cannon fodder — screen more valuable units')

        # ── BASIC: advance toward the nearest enemy ──
        nearest = self._nearest_enemy(unit, living_enemies)
        return ('ADVANCE', nearest, 'Basic unit — advance to combat')

    # ── Target-picking helpers ─────────────────────────────────────────

    def _best_charge_target(self, unit: Dict, enemies: List[Dict],
                            enemy_comp: Dict, my_type: UnitType) -> Optional[str]:
        """Pick the enemy the hammer/superior unit will get the best matchup against."""
        best_score = -999
        best_name = None
        for e in enemies:
            e_name = e.get('name', '')
            e_type, _ = enemy_comp.get(e_name, (UnitType.BASIC, SupportRole.NONE))
            matchup = MATCHUP_TABLE[my_type][e_type]
            # Prefer closer targets
            dist = self._dist(unit, e)
            closeness_bonus = max(0, 40 - dist) * 0.02
            score = matchup + closeness_bonus
            if score > best_score:
                best_score = score
                best_name = e_name
        return best_name

    def _best_flank_target(self, unit: Dict, enemies: List[Dict],
                           enemy_comp: Dict) -> Optional[str]:
        """Fast units want to hit flanks of engaged enemy units (they're pinned)."""
        # Prefer enemies already in combat (pinned — can't turn)
        best_score = -999
        best_name = None
        for e in enemies:
            e_name = e.get('name', '')
            e_type, _ = enemy_comp.get(e_name, (UnitType.BASIC, SupportRole.NONE))
            score = 0.0
            if e.get('isInCombat'):
                score += 2.0  # pinned targets are ideal flank targets
            # Non-hammer units are easier to break on a flank
            if e_type != UnitType.HAMMER:
                score += 0.5
            dist = self._dist(unit, e)
            score -= dist * 0.02
            if score > best_score:
                best_score = score
                best_name = e_name
        return best_name

    def _best_shooting_target(self, unit: Dict, enemies: List[Dict],
                              enemy_comp: Dict) -> Optional[str]:
        """Shooting units want to hit expensive/small units — maximum impact per model removed."""
        best_score = -999
        best_name = None
        for e in enemies:
            e_name = e.get('name', '')
            e_type, _ = enemy_comp.get(e_name, (UnitType.BASIC, SupportRole.NONE))
            # Hammer units are usually small and expensive — best shooting targets
            score = 0.0
            if e_type == UnitType.HAMMER:
                score += 3.0
            elif e_type == UnitType.SUPERIOR:
                score += 1.5
            elif e_type == UnitType.ANVIL:
                score -= 1.0  # hard to remove, low priority
            # Fewer models = each removal hurts more
            nmodels = e.get('nmodels', 10)
            score += max(0, 20 - nmodels) * 0.1
            dist = self._dist(unit, e)
            score -= dist * 0.01
            if score > best_score:
                best_score = score
                best_name = e_name
        return best_name

    def _find_enemy_type(self, enemies: List[Dict], enemy_comp: Dict,
                         target_type: UnitType) -> Optional[str]:
        """Find the nearest enemy matching a given type."""
        matches = []
        for e in enemies:
            e_name = e.get('name', '')
            e_type, _ = enemy_comp.get(e_name, (UnitType.BASIC, SupportRole.NONE))
            if e_type == target_type:
                matches.append(e_name)
        return matches[0] if matches else None

    def _nearest_enemy(self, unit: Dict, enemies: List[Dict]) -> Optional[str]:
        """Return name of nearest enemy."""
        nearest_name = None
        nearest_dist = float('inf')
        for e in enemies:
            d = self._dist(unit, e)
            if d < nearest_dist:
                nearest_dist = d
                nearest_name = e.get('name', '')
        return nearest_name

    @staticmethod
    def _dist(a: Dict, b: Dict) -> float:
        """Euclidean distance between two units (using position tuples)."""
        pa = a.get('position', (0, 0, 0))
        pb = b.get('position', (0, 0, 0))
        dx = pa[0] - pb[0]
        dy = pa[1] - pb[1]
        return (dx*dx + dy*dy) ** 0.5

    # ──────────────────────────────────────────────────────────────────
    # 5. Human-readable report
    # ──────────────────────────────────────────────────────────────────

    def full_report(self, my_units: list, enemy_units: list = None,
                    from_dict: bool = False) -> str:
        """
        Generate a complete strategy report.
        
        Args:
            my_units: your army (models or dicts)
            enemy_units: opponent's army (optional)
            from_dict: True if units are dicts
        
        Returns:
            Multi-line human-readable strategy report
        """
        lines = []
        lines.append("=" * 70)
        lines.append("  STRATEGY ADVISOR REPORT")
        lines.append("=" * 70)

        # Composition
        comp = self.analyse_composition(my_units, from_dict=from_dict)
        lines.append("\n" + comp['summary'])

        # Unit classifications
        lines.append("\n  Unit Classifications:")
        for name, (main_t, support_r) in comp['classifications'].items():
            label = self.classifier.get_type_label(main_t, support_r)
            lines.append(f"    {name:30s} -> {label}")

        # Strategy recommendations
        lines.append("\n  Recommended Strategies:")
        strats = self.recommend_strategies(my_units, from_dict=from_dict)
        for i, (strat, fit) in enumerate(strats, 1):
            lines.append(f"\n  {i}. {strat.name} (fit: {fit:.0%})")
            lines.append(f"     {strat.description}")
            lines.append(f"     Strengths:  {strat.strengths}")
            lines.append(f"     Weaknesses: {strat.weaknesses}")

        # Matchup analysis
        if enemy_units:
            lines.append("\n" + "-" * 70)
            lines.append("  MATCHUP ANALYSIS")
            lines.append("-" * 70)

            matchup = self.analyse_matchup(my_units, enemy_units, from_dict=from_dict)
            lines.append(f"\n  Verdict: {matchup['verdict']}")
            lines.append(f"  Average matchup score: {matchup['avg_matchup']:+.2f}")

            if matchup['threats']:
                lines.append("\n  Key Threats:")
                for name, e_type, e_role, score in matchup['threats'][:5]:
                    label = self.classifier.get_type_label(e_type, e_role)
                    lines.append(f"    ! {name:30s} [{label}] (matchup: {score:.1f})")

            if matchup['targets']:
                lines.append("\n  Priority Targets:")
                for name, e_type, e_role, score in matchup['targets'][:5]:
                    label = self.classifier.get_type_label(e_type, e_role)
                    lines.append(f"    > {name:30s} [{label}] (matchup: {score:.1f})")

            # Tactical roles
            if from_dict:
                lines.append("\n  Tactical Role Assignments:")
                roles = self.assign_tactical_roles(my_units, enemy_units)
                for name, info in roles.items():
                    lines.append(
                        f"    {name:30s} -> {info['role']:10s} "
                        f"{'target: ' + info['target'] if info['target'] else ''}"
                    )
                    lines.append(f"      {info['reason']}")

        lines.append("\n" + "=" * 70)
        return '\n'.join(lines)
