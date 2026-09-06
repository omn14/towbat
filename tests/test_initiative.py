"""Who Strikes First, and the Initiative a charge buys — Rulebook p. 146."""

import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from battleFunctions import (base_initiative,  # noqa: E402
                             charge_initiative_bonus, melee_attacks,
                             strike_initiative)
from battlescribe import (get_catalogue, has_strike_first,  # noqa: E402
                          has_strike_last)
from combat_resolution import CombatResolver  # noqa: E402
from models import model  # noqa: E402
from special_rules import build_special_rules  # noqa: E402


def _model(name, initiative):
    m = model(name, "")
    m.characteristics['I'] = initiative
    m.characteristics['A'] = 1
    return m


def _unit(name, initiative, nmodels=1, files=1):
    return SimpleNamespace(model=_model(name, initiative), nmodels=nmodels,
                           files=files, ranks=1, name=f"{name} Unit")


def _fighter(name, initiative, charged=False, distance=0.0):
    """A game unit as strikeOrder sees it."""
    unit = _unit(name, initiative)
    return SimpleNamespace(unit=unit, unitName=unit.name,
                           hasAttackedThisTurn=False,
                           chargedThisTurn=charged, chargeDistance=distance,
                           isInCombatWith=[], isInCombatFlank=[])


def _engage(charger, target, flank='front'):
    charger.isInCombatWith.append(target)
    charger.isInCombatFlank.append('front')
    target.isInCombatWith.append(charger)
    target.isInCombatFlank.append(flank)


def _order(*pairs):
    """strikeOrder over a list of (striker, target) pairs."""
    resolver = object.__new__(CombatResolver)
    resolver.game = SimpleNamespace(attackers=[p[0] for p in pairs],
                                    defenders=[p[1] for p in pairs])
    return resolver.strikeOrder()


class ChargeBonusTests(unittest.TestCase):
    """+1 per full inch moved, to a maximum of +3 or +4 (p. 146)."""

    def test_one_point_per_full_inch(self):
        for inches, bonus in ((0.0, 0), (0.9, 0), (1.0, 1), (1.9, 1), (2.5, 2)):
            self.assertEqual(charge_initiative_bonus(inches), bonus, inches)

    def test_front_arc_caps_at_three(self):
        self.assertEqual(charge_initiative_bonus(3.0), 3)
        self.assertEqual(charge_initiative_bonus(12.0), 3)

    def test_flank_or_rear_caps_at_four(self):
        self.assertEqual(charge_initiative_bonus(4.0, flank_or_rear=True), 4)
        self.assertEqual(charge_initiative_bonus(12.0, flank_or_rear=True), 4)

    def test_a_partial_inch_is_never_negative(self):
        self.assertEqual(charge_initiative_bonus(-2.0), 0)


class StrikeInitiativeTests(unittest.TestCase):

    def test_a_unit_that_did_not_charge_uses_its_profile(self):
        self.assertEqual(strike_initiative(_model("Grave Guard", 2)), 2)
        self.assertEqual(
            strike_initiative(_model("Grave Guard", 2), charged=False, inches=8.0), 2)

    def test_the_charge_bonus_is_added(self):
        self.assertEqual(
            strike_initiative(_model("Knight", 3), charged=True, inches=6.0), 6)

    def test_the_total_is_capped_at_ten(self):
        # Errata p. 146: "to a maximum of 10".
        self.assertEqual(
            strike_initiative(_model("Elf", 8), charged=True, inches=9.0,
                              flank_or_rear=True), 10)

    def test_a_missing_characteristic_does_not_crash(self):
        m = model("Nameless", "")
        m.characteristics.pop('I', None)
        self.assertEqual(strike_initiative(m), 1)


class StrikeOrderTests(unittest.TestCase):
    """Work down from the highest Initiative (p. 146)."""

    def test_highest_initiative_strikes_first(self):
        fast = _fighter("Elf", 6)
        slow = _fighter("Orc", 2)
        _engage(fast, slow)
        order = _order((slow, fast), (fast, slow))
        self.assertEqual([v for v, _ in order], [6, 2])
        self.assertEqual(order[0][1], 1)

    def test_a_charge_can_overtake_a_faster_defender(self):
        charger = _fighter("Knight", 3, charged=True, distance=7.0)
        holder = _fighter("Skink", 5)
        _engage(charger, holder)
        order = _order((charger, holder), (holder, charger))
        self.assertEqual([v for v, _ in order], [6, 5])

    def test_a_defender_can_still_strike_first(self):
        charger = _fighter("Zombie", 1, charged=True, distance=2.0)
        holder = _fighter("Elf", 6)
        _engage(charger, holder)
        order = _order((charger, holder), (holder, charger))
        self.assertEqual([v for v, _ in order], [6, 3])

    def test_a_flank_charge_may_take_the_larger_bonus(self):
        charger = _fighter("Knight", 3, charged=True, distance=9.0)
        holder = _fighter("Skink", 5)
        _engage(charger, holder, flank='rear')
        order = _order((charger, holder), (holder, charger))
        self.assertEqual(order[0], (7, 0))

    def test_a_unit_that_has_already_fought_is_left_out(self):
        fast = _fighter("Elf", 6)
        slow = _fighter("Orc", 2)
        _engage(fast, slow)
        fast.hasAttackedThisTurn = True
        self.assertEqual(_order((slow, fast), (fast, slow)), [(2, 0)])

    def test_a_unit_listed_twice_strikes_once(self):
        # attackers is built from both sides of the engagement, so a unit
        # reachable from either turns up twice; nothing has fought yet when the
        # order is drawn up, so hasAttackedThisTurn cannot weed the copy out.
        fast = _fighter("Elf", 6)
        slow = _fighter("Orc", 2)
        _engage(fast, slow)
        order = _order((slow, fast), (fast, slow), (fast, slow))
        self.assertEqual(order, [(6, 1), (2, 0)])


class EngagedFacingTests(unittest.TestCase):

    def test_the_arc_is_read_at_the_strikers_own_index(self):
        first = _fighter("Wolves", 3)
        second = _fighter("Knights", 3)
        holder = _fighter("Spearmen", 3)
        _engage(first, holder, flank='front')
        _engage(second, holder, flank='flank')
        self.assertEqual(CombatResolver._engagedFacing(holder, first), 'front')
        self.assertEqual(CombatResolver._engagedFacing(holder, second), 'flank')

    def test_an_unrecorded_engagement_counts_as_the_front(self):
        striker = _fighter("Wolves", 3)
        holder = _fighter("Spearmen", 3)
        self.assertEqual(CombatResolver._engagedFacing(holder, striker), 'front')


class SimultaneousCombatTests(unittest.TestCase):
    """Casualties do not reduce the attacks of models at the same Initiative."""

    def test_equal_initiative_shares_one_step(self):
        a = _fighter("Spearmen", 3)
        b = _fighter("Swordsmen", 3)
        _engage(a, b)
        self.assertEqual([v for v, _ in _order((a, b), (b, a))], [3, 3])

    def test_the_models_lost_in_the_same_step_still_attack(self):
        # The engine feeds melee_attacks the count as it stood when the step
        # began, so a blow struck alongside cannot thin the fighting rank.
        unit = _unit("Spearmen", 3, nmodels=20, files=5)
        before = melee_attacks(unit, charge=False, casualties=0)
        unit.nmodels = 17
        after = melee_attacks(unit, charge=False, casualties=3)
        self.assertLess(after, before)


def _rule_model(initiative, *keywords):
    """A model carrying *keywords* the way an imported roster supplies them."""
    m = _model("Warrior", initiative)
    m.characteristics['Special Rules'] = list(keywords)
    have = {r.get('name') for r in m.special_rules if isinstance(r, dict)}
    for entry in build_special_rules(m):
        if entry.get('name') not in have:
            m.special_rules.append(entry)
    return m


def _armed(initiative, slug):
    """A model wielding the catalogue weapon *slug*."""
    m = _model("Warrior", initiative)
    m.weapons['Test Weapon'] = dict(get_catalogue().weapons_by_slug[slug])
    m.equip_weapon('Test Weapon')
    return m


class RuleSpellingTests(unittest.TestCase):

    def test_both_spellings_are_recognised(self):
        for raw in ("Strike First", "Strikes First"):
            self.assertTrue(has_strike_first([raw]), raw)
        for raw in ("Strike Last", "Strikes Last"):
            self.assertTrue(has_strike_last([raw]), raw)

    def test_lightning_strike_is_neither(self):
        # The catalogue's Warp Lightning Cannon has Lightning Strike, which a
        # loose substring match would read as Strike First.
        self.assertFalse(has_strike_first(["Lightning Strike"]))
        self.assertFalse(has_strike_last(["Lightning Strike"]))


class StrikeFirstTests(unittest.TestCase):
    """Initiative becomes 10 before any other modifier (p. 177)."""

    def test_the_profile_is_replaced(self):
        self.assertEqual(base_initiative(_rule_model(3, "Strike First")), 10)

    def test_a_weapon_can_grant_it(self):
        self.assertEqual(base_initiative(_armed(3, 'celestial_blade')), 10)

    def test_a_charge_cannot_push_it_past_ten(self):
        m = _rule_model(3, "Strike First")
        self.assertEqual(strike_initiative(m, charged=True, inches=8.0), 10)


class StrikeLastTests(unittest.TestCase):
    """Initiative becomes 1 before any other modifier (p. 178)."""

    def test_the_profile_is_replaced(self):
        self.assertEqual(base_initiative(_rule_model(6, "Strike Last")), 1)

    def test_a_great_weapon_grants_it(self):
        # The plain great weapon carries Strike Last, so this is the common case.
        self.assertEqual(base_initiative(_armed(3, 'great_weapon')), 1)

    def test_the_charge_bonus_is_added_afterwards(self):
        # "before any other modifiers are applied" — the charge still counts.
        m = _armed(3, 'great_weapon')
        self.assertEqual(strike_initiative(m, charged=True, inches=5.0), 4)

    def test_a_weapon_not_in_hand_does_not_count(self):
        m = _armed(4, 'great_weapon')
        m.equip_weapon('Hand Weapon')
        self.assertFalse(m.has_strike_last())
        self.assertEqual(base_initiative(m), 4)


class CancellingOutTests(unittest.TestCase):

    def test_a_model_with_both_uses_its_profile(self):
        m = _rule_model(4, "Strike First", "Strike Last")
        self.assertTrue(m.has_strike_first())
        self.assertTrue(m.has_strike_last())
        self.assertEqual(base_initiative(m), 4)

    def test_a_strike_first_model_wielding_a_great_weapon_cancels(self):
        m = _rule_model(5, "Strike First")
        m.weapons['Great Weapon'] = dict(
            get_catalogue().weapons_by_slug['great_weapon'])
        m.equip_weapon('Great Weapon')
        self.assertEqual(base_initiative(m), 5)


class StrikeOrderWithTheStrikeRulesTests(unittest.TestCase):

    def test_strike_first_overtakes_a_charge(self):
        charger = _fighter("Knight", 3, charged=True, distance=7.0)
        holder = _fighter("Assassin", 5)
        holder.unit.model.characteristics['Special Rules'] = ["Strike First"]
        for entry in build_special_rules(holder.unit.model):
            holder.unit.model.special_rules.append(entry)
        _engage(charger, holder)
        order = _order((charger, holder), (holder, charger))
        self.assertEqual([v for v, _ in order], [10, 6])
        self.assertEqual(order[0][1], 1)


if __name__ == '__main__':
    unittest.main()
