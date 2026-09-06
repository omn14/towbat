"""Hatred (X) — Rulebook p. 171."""

import itertools
import os
import random
import re
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import persistence  # noqa: E402
import units as units_module  # noqa: E402
from battleFunctions import simulate_battle  # noqa: E402
from battlescribe import get_catalogue  # noqa: E402
from models import model  # noqa: E402
from special_rules import (parse_special_rule,  # noqa: E402
                           resolve_hatred_target)

HATRED = re.compile(r"^hatred\b", re.I)


def catalogue_hatred_spellings():
    """Every distinct 'Hatred (X)' string in the catalogue."""
    cat = get_catalogue()
    found = set()
    for record in (list(cat.by_slug.values()) + list(cat.all_by_slug.values())
                   + list(cat.weapons_by_slug.values())):
        rules = record.get('Special Rules') or record.get('special_rules') or []
        if not isinstance(rules, list):
            rules = [rules]
        for rule in rules:
            if HATRED.match(str(rule).strip()):
                found.add(str(rule).strip())
    return found


def _model(name="Warrior", faction=None, hatred=None, rules=(), ws=4, a=1):
    m = model(name, "")
    m.characteristics.update({'WS': str(ws), 'A': str(a), 'S': '4', 'T': '4',
                              'W': '1'})
    if faction is not None:
        m.characteristics['Faction'] = faction
    if rules:
        m.special_rules.extend({'name': r} for r in rules)
    if hatred:
        from special_rules import _hatred
        m.special_rules.append(_hatred(m, hatred, None))
    return m


def _unit(m, nmodels=1, name="Unit"):
    return SimpleNamespace(model=m, nmodels=nmodels, files=1, ranks=1, name=name)


class ResolvingTheHatedEnemyTests(unittest.TestCase):
    """X is prose, and none of it matches a faction name exactly."""

    def test_every_spelling_in_the_catalogue_resolves(self):
        """The guard-rail: a new army book adding 'Hatred (Skaven)' must fail
        here rather than silently hate nobody."""
        unresolved = []
        for spelling in sorted(catalogue_hatred_spellings()):
            _, param = parse_special_rule(spelling)
            factions, keywords, all_enemies = resolve_hatred_target(param)
            if not (factions or keywords or all_enemies):
                unresolved.append(spelling)
        self.assertEqual(unresolved, [],
                         "Hatred targets the engine cannot name an enemy for")

    def test_the_catalogue_still_has_the_eight_it_had(self):
        # If this drops, the survey above is testing less than it looks.
        self.assertGreaterEqual(len(catalogue_hatred_spellings()), 8)

    def test_orcs_and_goblins_is_one_faction_not_two(self):
        # The '&' trap: splitting first would invent an army called 'Orcs'.
        factions, keywords, all_enemies = resolve_hatred_target('Orcs & Goblins')
        self.assertEqual(factions, ('Orc and Goblin Tribes',))
        self.assertEqual(keywords, ())
        self.assertFalse(all_enemies)

    def test_a_compound_target_splits(self):
        factions, keywords, _ = resolve_hatred_target(
            'Warriors of Chaos & Daemonic models')
        self.assertEqual(factions, ('Warriors of Chaos',))
        self.assertEqual(keywords, ('Daemonic',))

    def test_the_three_part_target_splits(self):
        factions, keywords, _ = resolve_hatred_target(
            'Warriors of Chaos, Beastmen Breyherds & Daemonic models')
        self.assertEqual(factions,
                         ('Warriors of Chaos', 'Beastmen Brayherds'))
        self.assertEqual(keywords, ('Daemonic',))

    def test_the_misspelling_in_the_data_is_aliased(self):
        # 'Breyherds' for 'Brayherds', in one Cathayan entry.
        factions, _, _ = resolve_hatred_target('Beastmen Breyherds')
        self.assertEqual(factions, ('Beastmen Brayherds',))

    def test_both_casings_of_all_enemies(self):
        for text in ('All Enemies', 'all enemies'):
            self.assertEqual(resolve_hatred_target(text), ((), (), True), text)

    def test_dwarfs_covers_both_dwarf_armies(self):
        factions, _, _ = resolve_hatred_target('Dwarfs')
        self.assertIn('Dwarfen Mountain Holds', factions)
        self.assertIn('Chaos Dwarfs', factions)

    def test_an_empty_target_names_nobody(self):
        self.assertEqual(resolve_hatred_target(None), ((), (), False))


class WhoIsHatedTests(unittest.TestCase):

    def test_a_hated_faction_is_hated(self):
        dwarf = _model(hatred='Orcs & Goblins')
        self.assertTrue(dwarf.hates(_model(faction='Orc and Goblin Tribes')))

    def test_another_army_is_not(self):
        dwarf = _model(hatred='Orcs & Goblins')
        self.assertFalse(dwarf.hates(_model(faction='Vampire Counts')))

    def test_all_enemies_hates_anyone(self):
        zealot = _model(hatred='All Enemies')
        self.assertTrue(zealot.hates(_model(faction='Vampire Counts')))
        self.assertTrue(zealot.hates(_model(faction=None)))

    def test_a_keyword_is_matched_against_the_targets_rules(self):
        # 'Daemonic models' is a keyword, not an army.
        cathayan = _model(hatred='Warriors of Chaos & Daemonic models')
        daemon = _model(faction='Daemons of Chaos', rules=['Daemonic'])
        self.assertTrue(cathayan.hates(daemon))

    def test_that_keyword_does_not_catch_a_similar_name(self):
        cathayan = _model(hatred='Warriors of Chaos & Daemonic models')
        mortal = _model(faction='Vampire Counts', rules=['Daemonslayer'])
        self.assertFalse(cathayan.hates(mortal))

    def test_a_model_the_catalogue_never_heard_of_has_no_army_to_hate(self):
        dwarf = _model(hatred='Orcs & Goblins')
        self.assertFalse(dwarf.hates(_model(name='Nobody', faction=None)))

    def test_a_model_without_the_rule_hates_nothing(self):
        self.assertFalse(_model().hates(_model(faction='Orc and Goblin Tribes')))

    def test_a_real_dwarf_hates_a_real_orc(self):
        # Straight from the catalogue, no fixtures.
        self.assertTrue(model('Dwarf Warrior', '').hates(model('Orc Boy', '')))
        self.assertFalse(model('Dwarf Warrior', '').hates(model('Zombie', '')))


class TheRerollTests(unittest.TestCase):
    """Failed To Hit rolls only, once each, melee only, first round only."""

    def _fight(self, rolls, hatred='Orcs & Goblins', first_round=True,
               attacks=1):
        attacker = _model(hatred=hatred, a=attacks)
        victim = _model(faction='Orc and Goblin Tribes', ws=4)
        victim.armor_save = 7
        with mock.patch.object(random, 'randint',
                               side_effect=itertools.chain(rolls,
                                                           itertools.repeat(1))):
            return simulate_battle(_unit(attacker), _unit(victim, name='Orcs'),
                                   charge=False, first_round=first_round)

    def test_a_missed_hit_is_rerolled_and_can_land(self):
        # WS4 vs WS4 needs 4+: 1 misses, the re-rolled 6 hits.
        _, hits, *_ = self._fight([1, 1, 6])
        self.assertEqual(hits, 1)

    def test_the_reroll_is_not_itself_rerolled(self):
        _, hits, *_ = self._fight([1, 1, 1])
        self.assertEqual(hits, 0)

    def test_a_hit_is_never_rerolled(self):
        # A 5 already hits; if it were re-rolled the 1 behind it would miss.
        _, hits, *_ = self._fight([5, 1, 1])
        self.assertEqual(hits, 1)

    def test_nothing_is_rerolled_after_the_first_round(self):
        _, hits, *_ = self._fight([1, 1, 6], first_round=False)
        self.assertEqual(hits, 0)

    def test_nothing_is_rerolled_against_an_unhated_enemy(self):
        attacker = _model(hatred='High Elves')
        victim = _model(faction='Orc and Goblin Tribes')
        victim.armor_save = 7
        with mock.patch.object(random, 'randint',
                               side_effect=itertools.chain([1, 1, 6],
                                                           itertools.repeat(1))):
            _, hits, *_ = simulate_battle(_unit(attacker), _unit(victim),
                                          charge=False, first_round=True)
        self.assertEqual(hits, 0)

    def test_shooting_gets_no_reroll_from_it(self):
        # "during the first round of combat" — a missile attack is not.
        attacker = _model(hatred='All Enemies')
        attacker.characteristics['BS'] = '4'
        bow = dict(get_catalogue().weapon('Asrai Longbow') or
                   {'name': 'Bow', 'tag': 'ranged'})
        attacker.weapons[bow['name']] = bow
        attacker.equip_weapon(bow['name'])
        self.assertEqual(attacker.equipedWeapon.get('tag'), 'ranged')
        victim = _model(faction='Orc and Goblin Tribes')
        victim.armor_save = 7
        with mock.patch.object(random, 'randint',
                               side_effect=itertools.chain([1, 1, 6],
                                                           itertools.repeat(1))):
            _, hits, *_ = simulate_battle(_unit(attacker), _unit(victim),
                                          charge=False, first_round=True)
        self.assertEqual(hits, 0)


class TheCombatRoundCounterTests(unittest.TestCase):
    """'The first round of combat' had no representation in the engine."""

    def test_the_unit_starts_a_battle_having_fought_no_rounds(self):
        src = open(units_module.__file__, encoding='utf-8').read()
        self.assertIn('self.roundsFought=0', src)

    def test_leaving_combat_forgets_the_count(self):
        src = open(units_module.__file__, encoding='utf-8').read()
        block = re.search(r'def exitInCombat\(self\):(.*?)\n    def ', src, re.S)
        self.assertIsNotNone(block)
        self.assertIn('roundsFought', block.group(1),
                      "a unit that leaves combat keeps its round count, so its "
                      "next fight never counts as the first")

    def test_the_combat_phase_counts_the_round(self):
        path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'game_fsm.py')
        block = re.search(r'def enterCombatPhase\(self\):(.*?)\n    def ',
                          open(path, encoding='utf-8').read(), re.S)
        self.assertIsNotNone(block)
        self.assertIn('roundsFought', block.group(1))

    def test_the_count_survives_a_save(self):
        src = open(persistence.__file__, encoding='utf-8').read()
        self.assertIn("'roundsFought'", src,
                      "reloading mid-combat would hand Hatred back")


class DeadDataTests(unittest.TestCase):

    def test_no_profile_carries_a_bare_hatred(self):
        """A bare 'Hatred' names no enemy, so it can only be guessed at. The
        seven Night Goblin profiles that carried one were unreachable anyway:
        the catalogue supplies Hatred (Dwarfs) and wins."""
        import glob
        import json
        bare = []
        for path in glob.glob(os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'army_units_cat', '**', '*.json'), recursive=True):
            with open(path, encoding='utf-8') as f:
                rules = json.load(f).get('Special Rules') or []
            if isinstance(rules, list) and 'Hatred' in rules:
                bare.append(os.path.basename(path))
        self.assertEqual(bare, [])


if __name__ == '__main__':
    unittest.main()
