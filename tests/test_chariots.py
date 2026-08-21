"""
Tests for chariot split profiles (Rulebook p. 194).

A chariot is one model built from several profiles: the chariot itself, its crew
and the beasts that draw it. The catalogue marks the crew with subType='crew'
and the beasts with a CHARIOT CREW category link, and the unit profile carries
the 'Heavy chariot' / 'Light chariot' troop type.

Run:  python -m unittest discover -s tests
"""

import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from battlescribe import get_catalogue  # noqa: E402
from battleFunctions import simulate_battle  # noqa: E402
from combat_resolution import CombatResolver  # noqa: E402
from models import model  # noqa: E402
from movement_system import MovementSystem  # noqa: E402
from toHitAndToWound import to_hit, to_hit_ranged, to_wound, stat_value  # noqa: E402


class TestChariotCatalogue(unittest.TestCase):
    """The parts have to come out of the catalogue before anything can use them."""

    def test_unit_context_reaches_a_linked_model(self):
        # 'Empire War Wagons' holds the troop type and links out to a sibling
        # 'War Wagon' model entry, rather than nesting it.
        ch = get_catalogue().characteristics("War Wagon")
        self.assertEqual(ch["Unit"], "Empire War Wagons")
        self.assertEqual(ch["Troop Type"], "Heavy chariot")

    def test_crew_and_beasts_are_listed(self):
        ch = get_catalogue().characteristics("War Wagon")
        self.assertEqual(ch["Crew"], [{"name": "War Wagon Crew", "count": 6}])
        self.assertEqual(ch["Beasts"], [{"name": "Barded Warhorse", "count": 2}])

    def test_chariot_without_beasts(self):
        # The Black Orc Chariot's crew is nested, but it links no beasts.
        ch = get_catalogue().characteristics("Chariot")
        self.assertEqual([c["name"] for c in ch["Crew"]], ["Black Orc Crew"])
        self.assertEqual(ch["Beasts"], [])

    def test_ordinary_model_has_no_parts(self):
        ch = get_catalogue().characteristics("Crypt Ghoul")
        self.assertEqual(ch["Crew"], [])
        self.assertEqual(ch["Beasts"], [])


class TestChariotModel(unittest.TestCase):
    def setUp(self):
        self.wagon = model("War Wagon", "")
        self.ghoul = model("Crypt Ghoul", "")

    def test_detected_as_a_chariot(self):
        self.assertTrue(self.wagon.is_chariot())
        self.assertFalse(self.ghoul.is_chariot())

    def test_parts_are_attached(self):
        self.assertEqual(self.wagon.get_crew().name, "War Wagon Crew")
        self.assertEqual(self.wagon.get_beasts().name, "Barded Warhorse")

    def test_part_counts_come_from_the_catalogue(self):
        self.assertEqual(self.wagon.part_count('crew'), 6)
        self.assertEqual(self.wagon.part_count('beasts'), 2)
        self.assertEqual(self.ghoul.part_count('crew'), 0)

    def test_ordinary_model_has_no_parts(self):
        self.assertIsNone(self.ghoul.get_crew())
        self.assertIsNone(self.ghoul.get_beasts())

    def test_moves_at_the_speed_of_its_beasts(self):
        # The chariot's own Movement is '-'.
        self.assertEqual(self.wagon.characteristics["M"], "-")
        self.assertEqual(self.wagon.get_movement(),
                         stat_value(self.wagon.get_beasts().characteristics["M"]))

    def test_wounds_and_toughness_stay_on_the_chariot(self):
        self.assertEqual(self.wagon.characteristics["T"], "5")
        self.assertEqual(self.wagon.characteristics["W"], "6")


class TestChariotToHit(unittest.TestCase):
    """In combat, all enemy rolls To Hit are made against the crew's WS."""

    def setUp(self):
        self.wagon = model("War Wagon", "")
        self.ghoul = model("Crypt Ghoul", "")

    def test_enemies_hit_against_the_crew(self):
        # Ghoul WS 3 vs crew WS 3 -> 4+, not the automatic hit a WS of '-'
        # would otherwise give.
        self.assertEqual(self.wagon.defending_ws(), 3)
        self.assertEqual(to_hit(self.ghoul, self.wagon), 4)

    def test_chariot_attacks_with_its_crew(self):
        self.assertEqual(to_hit(self.wagon, self.ghoul), 4)

    def test_wounds_are_rolled_against_the_chariot(self):
        # S3 against the chariot's T5 is a 6, not against the crew's T of '-'.
        self.assertEqual(to_wound(self.ghoul, self.wagon), 6)


class TestChariotShooting(unittest.TestCase):
    """The crew shoot with their own Ballistic Skill and Strength (p. 194)."""

    BLUNDERBUSS = {'name': 'Blunderbuss', 'tag': 'ranged', 'ranged_range': 12,
                   'ranged_strength': 3, 'ranged_AP': 1, 'ranged_shots': 1,
                   'ranged_shots_dice': 'D3'}

    def setUp(self):
        self.wagon = model("War Wagon", "")
        self.wagon.weapons['Blunderbuss'] = dict(self.BLUNDERBUSS)

    def test_the_chariot_has_no_ballistic_skill_of_its_own(self):
        self.assertEqual(self.wagon.characteristics['BS'], '-')

    def test_it_shoots_with_the_crews(self):
        self.assertEqual(self.wagon.firing_bs(), 3)
        self.assertEqual(self.wagon.get_crew().characteristics['BS'], '3')

    def test_a_shot_can_actually_hit(self):
        # A BS of '-' reads as 0, which to_hit_ranged rejects outright, so every
        # shot missed however good the roll.
        self.wagon.equip_weapon('Blunderbuss')
        self.wagon.attack_roll = 4          # crew BS 3 hits on a 4+
        self.assertTrue(to_hit_ranged(self.wagon))
        self.wagon.attack_roll = 3
        self.assertFalse(to_hit_ranged(self.wagon))

    def test_a_weapon_without_a_strength_uses_the_crews(self):
        # The chariot is S5, its crew S3; a bow has no Strength of its own.
        self.assertEqual(self.wagon.shooting_strength(), 3)
        self.assertEqual(stat_value(self.wagon.characteristics['S']), 5)

    def test_an_ordinary_model_is_unaffected(self):
        ghoul = model("Crypt Ghoul", "")
        self.assertEqual(ghoul.firing_bs(),
                         stat_value(ghoul.characteristics.get('BS')))
        self.assertEqual(ghoul.shooting_strength(),
                         stat_value(ghoul.characteristics.get('S')))


class TestChariotAttacks(unittest.TestCase):
    """The crew and beasts each fight with their own profile; the chariot has
    no Attacks of its own."""

    def setUp(self):
        self.wagon = model("War Wagon", "")
        self.target = SimpleNamespace(
            model=model("Crypt Ghoul", ""), nmodels=10, files=5, ranks=2,
            name="Crypt Ghouls")

    @staticmethod
    def _as_unit(m):
        return SimpleNamespace(model=m, nmodels=1, files=1, ranks=1, name=m.name)

    def test_the_chariot_itself_makes_no_attacks(self):
        attacks = simulate_battle(self._as_unit(self.wagon), self.target,
                                  charge=False)[0]
        self.assertEqual(attacks, 0)

    def test_the_crew_attacks(self):
        crew = self.wagon.get_crew()
        self.assertEqual(stat_value(crew.characteristics["A"]), 1)
        attacks = simulate_battle(self._as_unit(crew), self.target,
                                  charge=False)[0]
        self.assertEqual(attacks, 1)

    def test_all_six_crew_fight(self):
        host = SimpleNamespace(unit=SimpleNamespace(model=self.wagon, nmodels=1,
                                                    files=1, ranks=1))
        crew = next(p for p in CombatResolver.chariotParts(None, host)
                    if p.name == "War Wagon Crew")
        self.assertEqual(crew.nmodels, 6)
        self.assertEqual(simulate_battle(crew, self.target, charge=False)[0], 6)

    def test_both_horses_fight(self):
        host = SimpleNamespace(unit=SimpleNamespace(model=self.wagon, nmodels=1,
                                                    files=1, ranks=1))
        beasts = next(p for p in CombatResolver.chariotParts(None, host)
                      if p.name == "Barded Warhorse")
        self.assertEqual(beasts.nmodels, 2)
        self.assertEqual(simulate_battle(beasts, self.target, charge=False)[0], 2)

    def test_the_beasts_attack(self):
        beasts = self.wagon.get_beasts()
        attacks = simulate_battle(self._as_unit(beasts), self.target,
                                  charge=False)[0]
        self.assertEqual(attacks, stat_value(beasts.characteristics["A"]))

    def test_parts_scale_with_the_number_of_chariots(self):
        host = SimpleNamespace(unit=SimpleNamespace(model=self.wagon, nmodels=3,
                                                    files=3, ranks=1))
        parts = CombatResolver.chariotParts(None, host)
        self.assertEqual([p.name for p in parts],
                         ["War Wagon Crew", "Barded Warhorse"])
        self.assertEqual([p.nmodels for p in parts], [18, 6])   # 3 chariots

    def test_a_model_without_parts_contributes_none(self):
        host = SimpleNamespace(unit=SimpleNamespace(model=model("Crypt Ghoul", ""),
                                                    nmodels=10, files=5, ranks=2))
        self.assertEqual(CombatResolver.chariotParts(None, host), [])

    def test_the_beasts_are_not_also_a_mount(self):
        # The roster lists the draught beasts as a 'mount' selection; attaching
        # them as one too would give the chariot a second set of horse attacks.
        self.assertIsNone(self.wagon.get_mount())
        self.assertFalse(self.wagon.is_mounted())


class TestMultiWoundCasualties(unittest.TestCase):
    """A chariot has 6 Wounds, so one unsaved wound must not remove it."""

    class _Movement:
        """MovementSystem.applyWounds with the node removal stubbed out."""
        def __init__(self):
            self.removed = []

        removeModelsFromUnit = lambda self, unit, n: self.removed.append(n)
        applyWounds = MovementSystem.applyWounds

    @staticmethod
    def _unit(wounds_characteristic):
        return SimpleNamespace(
            woundsOnModel=0,
            unit=SimpleNamespace(name="War Wagons",
                                 model=SimpleNamespace(
                                     characteristics={'W': wounds_characteristic})))

    def test_single_wound_models_are_removed_one_for_one(self):
        mv, unit = self._Movement(), self._unit('1')
        mv.applyWounds(unit, 3)
        self.assertEqual(mv.removed, [3])

    def test_a_wound_short_of_the_profile_removes_nothing(self):
        mv, unit = self._Movement(), self._unit('6')
        mv.applyWounds(unit, 5)
        self.assertEqual(mv.removed, [])
        self.assertEqual(unit.woundsOnModel, 5)

    def test_wounds_accumulate_across_rounds(self):
        mv, unit = self._Movement(), self._unit('6')
        mv.applyWounds(unit, 4)
        mv.applyWounds(unit, 2)
        self.assertEqual(mv.removed, [1])
        self.assertEqual(unit.woundsOnModel, 0)

    def test_leftovers_stay_on_the_next_model(self):
        mv, unit = self._Movement(), self._unit('6')
        mv.applyWounds(unit, 8)
        self.assertEqual(mv.removed, [1])
        self.assertEqual(unit.woundsOnModel, 2)

    def test_a_dash_reads_as_one_wound(self):
        mv, unit = self._Movement(), self._unit('-')
        mv.applyWounds(unit, 2)
        self.assertEqual(mv.removed, [2])

    def test_nothing_happens_without_wounds(self):
        mv, unit = self._Movement(), self._unit('6')
        mv.applyWounds(unit, 0)
        self.assertEqual(mv.removed, [])


class TestCharacteristicsOfZero(unittest.TestCase):
    """A '-' characteristic is 0, and WS 0 has its own combat rule (p. 158)."""

    @staticmethod
    def _model(**chars):
        return SimpleNamespace(characteristics=chars)

    def test_dash_reads_as_zero(self):
        self.assertEqual(stat_value("-"), 0)
        self.assertEqual(stat_value(None), 0)
        self.assertEqual(stat_value("3"), 3)

    def test_default_is_used_for_a_missing_value(self):
        self.assertEqual(stat_value("-", 4), 4)

    def test_attacks_against_ws_zero_hit_automatically(self):
        target = self._model(WS="-")
        self.assertEqual(to_hit(self._model(WS="4"), target), 1)

    def test_a_ws_zero_model_always_misses(self):
        self.assertEqual(to_hit(self._model(WS="-"), self._model(WS="4")), 7)

    def test_to_hit_never_returns_a_string(self):
        # It used to return an error message, which then blew up on '>='.
        self.assertIsInstance(to_hit(self._model(WS="-"), self._model()), int)


if __name__ == "__main__":
    unittest.main()
