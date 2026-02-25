"""
Unit Type Classification System

Classifies units into strategic archetypes based on their characteristics
and special rules, following Warhammer strategy doctrine:

Main Unit Types:
  - BASIC:        The reference unit. Matches its caliber but nothing more.
  - CANNON_FODDER: Weaker than basic. Never wins combat head-on, but can break
                   flanked targets. Assumed to lose to everything it doesn't flank.
  - SUPERIOR:     Slightly above basic (extra WS, better save, etc.).
                   Beats basic in ~2 turns, but not quickly.
  - ANVIL:        Holds against almost anything (stubborn, unbreakable, unkillable).
                   On par with basic offensively, but will hold indefinitely.
                   Only hammer units break anvils.
  - HAMMER:       Crushes anything it fights. Breaks everything on contact;
                   anvils take 2 turns. Hammer vs hammer: advantage wins.

Support Unit Types:
  - FAST:         Fast cavalry, flyers, eagles. Ignores obstacles, flanks easily.
                   Not strong head-on but devastating on flanks.
  - SHOOTING:     Ranged purpose only. Pushovers in combat but whittle
                   enemies down so they lose fights they should have won.

A unit can have one main type and optionally one support type
(e.g. a fast hammer, or a shooting cannon-fodder).
"""

from enum import Enum
from typing import Dict, Any, Optional, Tuple, List


class UnitType(Enum):
    """Main strategic unit archetypes"""
    BASIC = "basic"
    CANNON_FODDER = "cannon_fodder"
    SUPERIOR = "superior"
    ANVIL = "anvil"
    HAMMER = "hammer"


class SupportRole(Enum):
    """Support roles that overlay the main type"""
    NONE = "none"
    FAST = "fast"
    SHOOTING = "shooting"


# ── Combat outcome assumptions (from the strategy guide) ──────────────
# These encode the guide's stated assumptions as multipliers.
# A matchup value > 1.0 means the row unit is expected to beat the column unit.
# The value represents how many "combat rounds" the column survives.

MATCHUP_TABLE = {
    # Attacker ->           CANNON_FODDER  BASIC  SUPERIOR  ANVIL  HAMMER
    UnitType.CANNON_FODDER: {UnitType.CANNON_FODDER: 1.0, UnitType.BASIC: 0.3,
                             UnitType.SUPERIOR: 0.2, UnitType.ANVIL: 0.2,
                             UnitType.HAMMER: 0.1},
    UnitType.BASIC:         {UnitType.CANNON_FODDER: 1.5, UnitType.BASIC: 1.0,
                             UnitType.SUPERIOR: 0.6, UnitType.ANVIL: 0.8,
                             UnitType.HAMMER: 0.3},
    UnitType.SUPERIOR:      {UnitType.CANNON_FODDER: 2.0, UnitType.BASIC: 1.4,
                             UnitType.SUPERIOR: 1.0, UnitType.ANVIL: 1.0,
                             UnitType.HAMMER: 0.4},
    UnitType.ANVIL:         {UnitType.CANNON_FODDER: 1.2, UnitType.BASIC: 1.0,
                             UnitType.SUPERIOR: 0.9, UnitType.ANVIL: 1.0,
                             UnitType.HAMMER: 0.5},
    UnitType.HAMMER:        {UnitType.CANNON_FODDER: 3.0, UnitType.BASIC: 2.5,
                             UnitType.SUPERIOR: 2.0, UnitType.ANVIL: 1.5,
                             UnitType.HAMMER: 1.0},
}

# Flank bonus: cannon fodder can break non-hammer units on a flank
FLANK_BONUS = 0.6  # added to matchup when flanking
REAR_BONUS = 1.0   # added to matchup when in rear


class UnitTypeClassifier:
    """
    Analyses a unit's characteristics and special rules to assign
    a main UnitType and an optional SupportRole.
    
    Classification is heuristic-based using stat thresholds calibrated
    to the existing roster (Orc Boys ≈ BASIC reference line).
    """

    # ── Reference line: Orc Boyz stats ────────────────────────────────
    REF_WS = 3
    REF_S = 3
    REF_T = 4
    REF_A = 1
    REF_LD = 6
    REF_SAVE = 7   # no armour
    REF_M = 4

    def __init__(self):
        # Cache classified results so we don't recalculate
        self._cache: Dict[str, Tuple[UnitType, SupportRole]] = {}

    def classify_from_model(self, model_obj) -> Tuple[UnitType, SupportRole]:
        """
        Classify a model class instance (from models.py).
        Returns (UnitType, SupportRole).
        """
        name = model_obj.name
        if name in self._cache:
            return self._cache[name]

        chars = model_obj.characteristics
        stats = {
            'WS': int(chars.get('WS', 3)),
            'BS': int(chars.get('BS', 0)) if chars.get('BS', '0') not in ('-', '') else 0,
            'S': int(chars.get('S', 3)),
            'T': int(chars.get('T', 3)),
            'W': int(chars.get('W', 1)),
            'I': int(chars.get('I', 3)),
            'A': int(chars.get('A', 1)),
            'Ld': int(chars.get('Ld', 7)),
            'M': int(chars.get('M', 4)) if chars.get('M', '4') not in ('-', '0') else 4,
            'armor_save': getattr(model_obj, 'armor_save', 7),
        }

        # Gather special rule flags
        rules = getattr(model_obj, 'special_rules', [])
        has_unbreakable = any(r.get('Unbreakable', False) for r in rules if isinstance(r, dict))
        has_stubborn = any('stubborn' in r.get('name', '').lower() for r in rules if isinstance(r, dict))
        has_regen = any(r.get('regen') for r in rules if isinstance(r, dict))
        has_mount = any(r.get('tag') == 'mount' for r in rules if isinstance(r, dict))
        has_flying = any('fly' in r.get('name', '').lower() for r in rules if isinstance(r, dict))
        has_ranged = any(
            w.get('tag') == 'ranged'
            for w in (model_obj.weapons.values() if hasattr(model_obj, 'weapons') else [])
        )
        has_charge_bonus = any(
            r.get('charge') for r in rules if isinstance(r, dict)
        )
        # Also check weapons for charge bonuses (e.g. lances)
        if not has_charge_bonus and hasattr(model_obj, 'weapons'):
            for w in model_obj.weapons.values():
                if isinstance(w, dict) and w.get('charge'):
                    has_charge_bonus = True
                    break

        main_type = self._classify_main_type(stats, has_unbreakable, has_stubborn,
                                              has_regen, has_charge_bonus, has_mount)
        support_role = self._classify_support_role(stats, has_mount, has_flying, has_ranged)

        self._cache[name] = (main_type, support_role)
        return main_type, support_role

    def classify_from_dict(self, unit_data: Dict[str, Any]) -> Tuple[UnitType, SupportRole]:
        """
        Classify from a lightweight dict (as stored in GameState.units).
        Used during AI tree evaluation so we don't need the full model object.
        """
        name = unit_data.get('name', '')
        if name in self._cache:
            return self._cache[name]

        stats = {
            'WS': unit_data.get('WS', 3),
            'S': unit_data.get('S', 3),
            'T': unit_data.get('T', 3),
            'A': unit_data.get('A', 1),
            'Ld': unit_data.get('Ld', 7),
            'armor_save': unit_data.get('armor_save', 7),
            'M': unit_data.get('M', 4),
            'W': unit_data.get('W', 1),
        }

        # Read flags stored on the unit dict (set during GameState.from_game)
        has_unbreakable = unit_data.get('is_unbreakable', False)
        has_stubborn = unit_data.get('is_stubborn', False)
        has_regen = unit_data.get('has_regen', False)
        has_mount = unit_data.get('has_mount', False)
        has_flying = unit_data.get('is_flying', False)
        has_ranged = unit_data.get('ranged', False)
        has_charge_bonus = unit_data.get('has_charge_bonus', False)

        main_type = self._classify_main_type(stats, has_unbreakable, has_stubborn,
                                              has_regen, has_charge_bonus, has_mount)
        support_role = self._classify_support_role(stats, has_mount, has_flying, has_ranged)

        self._cache[name] = (main_type, support_role)
        return main_type, support_role

    # ── Internal classification heuristics ─────────────────────────────

    def _combat_power_score(self, stats: dict) -> float:
        """
        Single number representing offensive + defensive combat power,
        normalised so that the BASIC reference (Orc Boy) scores ~1.0.
        """
        ws_ratio = stats['WS'] / self.REF_WS
        s_ratio = stats['S'] / self.REF_S
        t_ratio = stats['T'] / self.REF_T
        a_ratio = stats['A'] / self.REF_A
        save_bonus = max(0, (self.REF_SAVE - stats['armor_save'])) * 0.15
        w_bonus = (stats.get('W', 1) - 1) * 0.3  # multi-wound models

        offensive = ws_ratio * s_ratio * a_ratio
        defensive = t_ratio * (1.0 + save_bonus + w_bonus)
        return (offensive + defensive) / 2.0

    def _staying_power_score(self, stats: dict, has_unbreakable: bool,
                             has_stubborn: bool, has_regen: bool) -> float:
        """How long the unit can hold in combat regardless of killing power."""
        score = 0.0
        ld = stats['Ld']

        if has_unbreakable:
            score += 3.0   # infinite staying power
        if has_stubborn:
            score += 1.5 + ld * 0.1
        if has_regen:
            score += 0.8

        # High toughness / good save = hard to shift
        t_bonus = max(0, stats['T'] - self.REF_T) * 0.4
        save_bonus = max(0, self.REF_SAVE - stats['armor_save']) * 0.3
        score += t_bonus + save_bonus

        # Leadership contribution
        score += max(0, ld - self.REF_LD) * 0.2

        return score

    def _classify_main_type(self, stats: dict, has_unbreakable: bool,
                            has_stubborn: bool, has_regen: bool,
                            has_charge_bonus: bool, has_mount: bool) -> UnitType:
        """Determine the main unit archetype."""
        combat = self._combat_power_score(stats)
        staying = self._staying_power_score(stats, has_unbreakable, has_stubborn, has_regen)

        # ── HAMMER: devastating offensive output ──
        # High combat power, especially with charge bonuses or mounted
        hammer_score = combat
        if has_charge_bonus:
            hammer_score += 0.5
        if has_mount:
            hammer_score += 0.4
        if stats.get('W', 1) >= 2:
            hammer_score += 0.3

        if hammer_score >= 2.0:
            return UnitType.HAMMER

        # ── ANVIL: exceptional staying power ──
        if staying >= 2.5 or (has_unbreakable and combat < 1.5):
            return UnitType.ANVIL
        if has_stubborn and stats['Ld'] >= 7:
            return UnitType.ANVIL

        # ── CANNON FODDER: clearly below basic ──
        # Low combat output AND poor leadership = unreliable
        ld = stats.get('Ld', 7)
        if combat < 0.8 and ld <= 5:
            return UnitType.CANNON_FODDER
        if combat < 0.65:
            return UnitType.CANNON_FODDER

        # ── SUPERIOR: noticeably above basic ──
        if combat >= 1.2 or (combat >= 1.0 and staying >= 1.5):
            return UnitType.SUPERIOR

        # ── BASIC: everything else ──
        return UnitType.BASIC

    def _classify_support_role(self, stats: dict, has_mount: bool,
                               has_flying: bool, has_ranged: bool) -> SupportRole:
        """Determine if the unit fills a support role."""
        m = stats.get('M', 4)

        # FAST: high movement, mounted, or flying
        if has_flying:
            return SupportRole.FAST
        if has_mount:
            # All cavalry/mounted units count as fast —
            # the rider's M stat in the JSON is irrelevant when mounted
            return SupportRole.FAST
        if m >= 7:  # significantly faster than standard infantry
            return SupportRole.FAST

        # SHOOTING: primarily a ranged unit
        if has_ranged:
            return SupportRole.SHOOTING

        return SupportRole.NONE

    # ── Public helpers ─────────────────────────────────────────────────

    def get_matchup_score(self, attacker_type: UnitType, defender_type: UnitType,
                          is_flanking: bool = False, is_rear: bool = False) -> float:
        """
        Return expected combat performance of attacker vs defender.
        >1.0 means attacker is expected to win.
        """
        base = MATCHUP_TABLE[attacker_type][defender_type]
        if is_rear:
            base += REAR_BONUS
        elif is_flanking:
            base += FLANK_BONUS
        return base

    def classify_army(self, units: list, from_dict: bool = False) -> Dict[str, Tuple[UnitType, SupportRole]]:
        """Classify every unit in a list. Returns {name: (UnitType, SupportRole)}."""
        result = {}
        for u in units:
            if from_dict:
                name = u.get('name', 'unknown')
                main_t, support_r = self.classify_from_dict(u)
            else:
                name = u.name if hasattr(u, 'name') else str(u)
                main_t, support_r = self.classify_from_model(u)
            result[name] = (main_t, support_r)
        return result

    def get_type_label(self, main_type: UnitType, support_role: SupportRole) -> str:
        """Human-readable label like 'Hammer (Fast)' or 'Basic'."""
        label = main_type.value.replace('_', ' ').title()
        if support_role != SupportRole.NONE:
            label += f" ({support_role.value.title()})"
        return label

    def clear_cache(self):
        """Clear the classification cache (call if units change mid-game)."""
        self._cache.clear()
