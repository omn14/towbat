"""Impact Hits (X) — Rulebook p. 172."""

import os
import random
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from battleFunctions import (MIN_IMPACT_HIT_CHARGE, impact_hit_profile,  # noqa: E402
                             resolve_impact_hits, unmodified_strength)
from combat_resolution import CombatResolver  # noqa: E402
from models import model  # noqa: E402
from special_rules import (SPECIAL_RULE_BUILDERS, _param_dice,  # noqa: E402
                           build_special_rules)


def _with_rules(name, *raw_rules):
    """A model carrying *raw_rules*, the way an imported roster supplies them."""
    m = model(name, "")
    m.characteristics['Special Rules'] = list(raw_rules)
    have = {r.get('name') for r in m.special_rules if isinstance(r, dict)}
    for entry in build_special_rules(m):
        if entry.get('name') not in have:
            m.special_rules.append(entry)
    return m


def _unit(m, nmodels=1, files=1, ranks=1, name=None):
    return SimpleNamespace(model=m, nmodels=nmodels, files=files, ranks=ranks,
                           name=name or f"{m.name} Unit")


class TestParsingTheParameter(unittest.TestCase):

    def test_plain_expressions(self):
        for raw, expected in (("2", "2"), ("D6", "D6"), ("D3+1", "D3+1"),
                              ("2D6+2", "2D6+2")):
            self.assertEqual(_param_dice(raw), expected)

    def test_trailing_prose_is_dropped(self):
        # The War Wagon's rule reads 'Impact Hits (D6+1, War Wagon only)'.
        self.assertEqual(_param_dice("D6+1, War Wagon only"), "D6+1")

    def test_a_missing_parameter_falls_back(self):
        self.assertEqual(_param_dice(None, "1"), "1")

    def test_the_builder_is_registered(self):
        entry = SPECIAL_RULE_BUILDERS["impact hits"](None, "D6", None)
        self.assertEqual(entry["impact_hits"], "D6")
        self.assertEqual(entry["tag"], "combat")

    def test_a_bare_rule_causes_one_hit(self):
        self.assertEqual(SPECIAL_RULE_BUILDERS["impact hits"](None, None, None)
                         ["impact_hits"], "1")


class TestCatalogueModels(unittest.TestCase):
    """The parameter reaches the model from the catalogue."""

    @staticmethod
    def _expr(unit_name):
        return next((r["impact_hits"] for r in model(unit_name, "").special_rules
                     if r.get("impact_hits")), None)

    def test_razorgor(self):
        self.assertEqual(self._expr("Razorgor"), "D3")

    def test_chariot(self):
        self.assertEqual(self._expr("Chariot"), "D6+1")

    def test_a_replacement_rule_uses_the_later_value(self):
        # 'Impact Hits (2) (D3+1)': the second value replaces the first.
        self.assertEqual(self._expr("Slaughtermaster"), "D3+1")

    def test_a_model_without_the_rule_has_none(self):
        self.assertIsNone(self._expr("Zombie"))

    def test_the_war_wagon_gets_it_from_the_roster(self):
        # Its rules live on the linked model entry, which the catalogue parser
        # does not read; the imported roster supplies them instead.
        wagon = _with_rules("War Wagon", "Impact Hits (D6+1, War Wagon only)")
        self.assertEqual(impact_hit_profile(_unit(wagon))[1], "D6+1")


class TestWhichProfileCausesThem(unittest.TestCase):

    def test_a_chariot_uses_its_own_strength_not_its_crew(self):
        wagon = _with_rules("War Wagon", "Impact Hits (D6+1, War Wagon only)")
        m, expr = impact_hit_profile(_unit(wagon))
        self.assertIs(m, wagon)
        self.assertEqual(unmodified_strength(m), 5)
        self.assertEqual(unmodified_strength(wagon.get_crew()), 3)

    def test_a_mounts_impact_hits_use_the_mounts_strength(self):
        rider = model("Orc Boar Boy", "")
        boar = model("War Boar", "")          # Impact Hits (D6+1) in the catalogue
        rider.attach_mount(SimpleNamespace(model=boar))
        m, expr = impact_hit_profile(_unit(rider))
        self.assertIs(m, boar)
        self.assertEqual(expr, "D6+1")
        self.assertEqual(unmodified_strength(m), unmodified_strength(boar))

    def test_the_models_own_rule_wins_over_its_mount(self):
        rider = _with_rules("Orc Boar Boy", "Impact Hits (D6)")
        rider.attach_mount(SimpleNamespace(model=model("War Boar", "")))
        self.assertEqual(impact_hit_profile(_unit(rider))[1], "D6")

    def test_no_rule_no_profile(self):
        self.assertIsNone(impact_hit_profile(_unit(model("Zombie", ""))))

    def test_the_strength_ignores_a_weapon_bonus(self):
        wagon = _with_rules("War Wagon", "Impact Hits (D6+1, War Wagon only)")
        wagon.characteristics['S'] = '9'   # as a lance bonus would leave it
        self.assertEqual(unmodified_strength(wagon), 5)


class TestResolution(unittest.TestCase):

    def setUp(self):
        self.wagon = _with_rules("War Wagon", "Impact Hits (D6+1, War Wagon only)")
        self.charger = _unit(self.wagon)
        # A Goblin has neither armour nor Regeneration, so unsaved == wounds.
        self.target = _unit(model("Goblin", ""), nmodels=20, files=5, ranks=4)

    def test_a_model_without_the_rule_causes_nothing(self):
        plain = _unit(model("Zombie", ""), nmodels=10, files=5, ranks=2)
        self.assertEqual(resolve_impact_hits(plain, self.target), (0, 0, 0, 0))

    def test_the_number_of_hits_follows_the_expression(self):
        for _ in range(30):
            hits = resolve_impact_hits(self.charger, self.target)[0]
            self.assertIn(hits, range(2, 8))   # D6+1

    def test_hits_are_automatic(self):
        # Every roll a 1: To Hit would fail, but Impact Hits skip it entirely.
        with mock.patch('battleFunctions.random.randint', return_value=1):
            hits, wounds, saves, unsaved = resolve_impact_hits(self.charger,
                                                               self.target)
        self.assertEqual(hits, 2)          # D6+1 with every die a 1
        self.assertEqual(wounds, 0)        # a 1 still fails To Wound
        with mock.patch('battleFunctions.random.randint', return_value=6):
            hits, wounds, saves, unsaved = resolve_impact_hits(self.charger,
                                                               self.target)
        self.assertEqual(hits, 7)
        self.assertEqual(wounds, 7)        # every hit wounds, none rolled To Hit
        self.assertEqual(unsaved, 7)

    def test_armour_saves_still_apply(self):
        armoured = model("Goblin", "")
        armoured.set_armour(["full plate armour", "shield"])   # 2+, AP-2 leaves 4+
        with mock.patch('battleFunctions.random.randint', return_value=6):
            hits, wounds, saves, unsaved = resolve_impact_hits(
                self.charger, _unit(armoured, nmodels=20, files=5, ranks=4))
        self.assertEqual(saves, wounds)    # a 6 always saves
        self.assertEqual(unsaved, 0)

    def test_scythed_wheels_cut_through_light_armour(self):
        # Light armour + shield is a 5+, improved to 4+ by Parry; a heavy
        # chariot's AP-2 takes that back to 6+.
        armoured = model("Goblin", "")
        armoured.set_armour(["light armour", "shield"])
        self.assertEqual(armoured.melee_armour_save(), 4)
        self.assertEqual(self.wagon.impact_hit_ap(), 2)
        with mock.patch('battleFunctions.random.randint', return_value=5):
            hits, wounds, saves, unsaved = resolve_impact_hits(
                self.charger, _unit(armoured, nmodels=20, files=5, ranks=4))
        self.assertEqual(saves, 0)
        self.assertEqual(unsaved, wounds)

    def test_regeneration_applies_after_the_armour_save(self):
        # A Zombie has no armour but does have Regeneration.
        zombies = _unit(model("Zombie", ""), nmodels=20, files=5, ranks=4)
        with mock.patch('battleFunctions.random.randint', return_value=6):
            hits, wounds, saves, unsaved = resolve_impact_hits(self.charger,
                                                               zombies)
        self.assertEqual(saves, wounds)
        self.assertEqual(unsaved, 0)

    def test_every_model_in_contact_causes_them(self):
        rider = model("Orc Boar Boy", "")
        rider.attach_mount(SimpleNamespace(model=model("War Boar", "")))
        # Five models wide, two ranks deep: only the front rank is in contact.
        unit = _unit(rider, nmodels=10, files=5, ranks=2)
        with mock.patch('battleFunctions.random.randint', return_value=1):
            self.assertEqual(resolve_impact_hits(unit, self.target)[0], 10)  # 5 x D6+1

    def test_a_unit_smaller_than_its_frontage(self):
        rider = model("Orc Boar Boy", "")
        rider.attach_mount(SimpleNamespace(model=model("War Boar", "")))
        unit = _unit(rider, nmodels=2, files=5, ranks=1)
        with mock.patch('battleFunctions.random.randint', return_value=1):
            self.assertEqual(resolve_impact_hits(unit, self.target)[0], 4)   # 2 x D6+1


class TestChargeConditions(unittest.TestCase):
    """Only a charging model that moved 3\" or more causes Impact Hits."""

    class _Game:
        player1Units = []
        player2Units = []

        def applyWounds(self, unit, wounds):
            pass

    def setUp(self):
        self.resolver = CombatResolver.__new__(CombatResolver)
        self.resolver.game = self._Game()
        wagon = _with_rules("War Wagon", "Impact Hits (D6+1, War Wagon only)")
        self.striker = SimpleNamespace(unit=_unit(wagon), hasAttackedThisTurn=False,
                                       chargedThisTurn=True, chargeDistance=8.0)
        self.target = SimpleNamespace(
            unit=_unit(model("Goblin", ""), nmodels=20, files=5, ranks=4),
            hasAttackedThisTurn=False, woundsOnModel=0)
        self.resolver.game.player1Units = [self.striker]
        self.resolver.game.player2Units = [self.target]
        self.resolver.game.attackers = [self.striker]
        self.resolver.game.defenders = [self.target]

    def _run(self):
        from direct.interval.IntervalGlobal import Sequence
        return self.resolver.impactHits(Sequence())

    def test_a_charge_of_three_inches_or_more_causes_them(self):
        p1, p2 = self._run()
        self.assertGreater(p1, 0)
        self.assertLess(self.target.unit.nmodels, 20)

    def test_a_short_charge_causes_none(self):
        self.striker.chargeDistance = MIN_IMPACT_HIT_CHARGE - 0.5
        self.assertEqual(self._run(), (0, 0))
        self.assertEqual(self.target.unit.nmodels, 20)

    def test_a_unit_that_did_not_charge_causes_none(self):
        self.striker.chargedThisTurn = False
        self.assertEqual(self._run(), (0, 0))

    def test_a_unit_that_already_fought_causes_none(self):
        self.striker.hasAttackedThisTurn = True
        self.assertEqual(self._run(), (0, 0))

    def test_they_are_caused_once_per_combat(self):
        # The same striker appears twice when engaged with two enemies.
        self.resolver.game.attackers = [self.striker, self.striker]
        self.resolver.game.defenders = [self.target, self.target]
        with mock.patch('battleFunctions.random.randint', return_value=6):
            p1, _ = self._run()
        self.assertEqual(p1, 7)   # D6+1 = 7, not 14

    def test_multi_wound_targets_lose_whole_models_only(self):
        wagon = _with_rules("War Wagon", "Impact Hits (D6+1, War Wagon only)")
        self.target.unit = _unit(wagon, nmodels=3, files=3)
        self.target.woundsOnModel = 0
        with mock.patch('battleFunctions.random.randint', return_value=6):
            self._run()
        # 7 unsaved wounds on W6 chariots: one dies, the rest carry over.
        self.assertEqual(self.target.unit.nmodels, 2)


if __name__ == "__main__":
    random.seed(0)
    unittest.main()
