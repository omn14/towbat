"""Magic Resistance and casting order — Rulebook pp. 108–110, 173."""

import asyncio
import contextlib
import io
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import model  # noqa: E402
from panda3d.core import Point3  # noqa: E402
from rules_log import add_listener, remove_listener  # noqa: E402
from special_rules import apply_rule_keywords, unit_magic_resistance  # noqa: E402
from spell_system import Spell, is_dispelled  # noqa: E402


def _model(name='Bearer', *keywords):
    m = model('Zombie', '')
    m.name = name
    apply_rule_keywords(m, list(keywords), replace=True)
    return m


def _resistance_rules(m):
    return [r for r in m.special_rules
            if isinstance(r, dict) and 'magic_resistance' in r]


class _UnitGraphics:
    """Identity-based ownership and joined links, without Panda nodes."""

    def __init__(self, name, *keywords, nmodels=10):
        self.unitName = name
        self.unit = SimpleNamespace(name=name, model=_model(name, *keywords),
                                    nmodels=nmodels, files=5)
        self.hostUnit = None
        self.joinedCharacter = None


def _join(host, character):
    host.joinedCharacter = character
    character.hostUnit = host


class _LoggedTestCase(unittest.TestCase):

    def setUp(self):
        self.logs = []

        def listener(kind, rule, subject, detail):
            self.logs.append((kind, rule, subject, detail))

        add_listener(listener)
        self.addCleanup(remove_listener, listener)

    def resistance_logs(self, kind):
        return [entry for entry in self.logs
                if entry[:2] == (kind, 'Magic Resistance')]


class TestMagicResistanceKeywords(unittest.TestCase):

    def test_signed_unsigned_and_unicode_values_are_penalties(self):
        for keyword, expected in (
                ('Magic Resistance (-1)', -1),
                ('Magic Resistance (-12)', -12),
                ('Magic Resistance (2)', -2),
                ('Magic Resistance (−2)', -2),
                ('  mAgIc ReSiStAnCe ( -3 )  ', -3),
                ('Magic Resistance (0)', 0)):
            with self.subTest(keyword=keyword):
                m = _model('Bearer', keyword)
                rules = _resistance_rules(m)
                self.assertEqual(len(rules), 1)
                self.assertEqual(rules[0]['magic_resistance'], expected)
                self.assertEqual(rules[0]['name'],
                                 f'Magic Resistance ({expected})')
                self.assertEqual(m.magic_resistance(), expected)
                self.assertNotIn('ward', rules[0])
                self.assertNotIn('regen', rules[0])

    def test_unsupported_parameters_never_guess_a_number(self):
        for keyword in (
                'Magic Resistance', 'Magic Resistance ()',
                'Magic Resistance (unknown)', 'Magic Resistance (-X)',
                'Magic Resistance (-D3)', 'Magic Resistance (D3)',
                'Magic Resistance (+2)', 'Magic Resistance (-2.5)',
                'Magic Resistance (-2, bearer only)',
                'Magic Resistance (-2 or -3)', 'Magic Resistance (--2)',
                'Magic Resistance (- 2)'):
            with self.subTest(keyword=keyword):
                m = _model('Bearer', keyword)
                rules = _resistance_rules(m)
                self.assertEqual(len(rules), 1)
                self.assertIsNone(rules[0]['magic_resistance'])
                self.assertEqual(m.magic_resistance(), 0)
                value, source, unresolved = unit_magic_resistance(m)
                self.assertEqual((value, source), (0, ''))
                self.assertEqual(unresolved, [f"Bearer: {rules[0]['name']}"])

    def test_no_rule_means_no_resistance(self):
        m = _model()
        self.assertEqual(m.magic_resistance(), 0)
        self.assertEqual(unit_magic_resistance(m), (0, '', []))

    def test_multiple_grants_use_the_strongest_not_the_sum(self):
        m = _model('Bearer', 'Magic Resistance (-1)',
                   'Magic Resistance (-3)', 'Magic Resistance (-2)',
                   'Magic Resistance (-D3)')
        m.special_rules.extend([None, 'display-only'])
        self.assertEqual(m.magic_resistance(), -3)
        self.assertEqual(unit_magic_resistance(m),
                         (-3, 'Bearer', ['Bearer: Magic Resistance (-D3)']))

    def test_replacement_upgrades_downgrades_and_removes_the_parameter(self):
        m = _model('Bearer', 'Magic Resistance (-1)')
        ward = {'name': 'Oaken Shield', 'ward': 5}
        m.special_rules.append(ward)
        for value in (-3, -1, None):
            with self.subTest(value=value):
                wanted = [] if value is None else [f'Magic Resistance ({value})']
                apply_rule_keywords(m, wanted, replace=True)
                self.assertEqual(m.characteristics['Special Rules'], wanted)
                self.assertEqual([r['name'] for r in _resistance_rules(m)], wanted)
                self.assertEqual(m.magic_resistance(), value or 0)
                self.assertIn(ward, m.special_rules)

    def test_replacement_with_unresolved_rule_removes_old_numeric_grant(self):
        for keyword in ('Magic Resistance', 'Magic Resistance (-D3)'):
            with self.subTest(keyword=keyword):
                m = _model('Bearer', 'Magic Resistance (-3)')
                apply_rule_keywords(m, [keyword], replace=True)
                self.assertEqual(m.magic_resistance(), 0)
                self.assertEqual(len(_resistance_rules(m)), 1)
                self.assertEqual(unit_magic_resistance(m),
                                 (0, '', [f'Bearer: {keyword}']))

    def test_resolving_an_unknown_rule_removes_the_unresolved_grant(self):
        m = _model('Bearer', 'Magic Resistance (-D3)')
        apply_rule_keywords(m, ['Magic Resistance (-2)'], replace=True)
        self.assertEqual(len(_resistance_rules(m)), 1)
        self.assertEqual(unit_magic_resistance(m), (-2, 'Bearer', []))

    def test_canonical_replacement_and_repeated_load_do_not_duplicate(self):
        m = _model('Bearer', 'Magic Resistance (2)')
        for keyword in ('Magic Resistance (−2)', 'Magic Resistance (-2)',
                        'Magic Resistance (-2)'):
            apply_rule_keywords(m, [keyword], replace=True)
            self.assertEqual(len(_resistance_rules(m)), 1)
            self.assertEqual(m.magic_resistance(), -2)
        apply_rule_keywords(m, [], replace=True)
        self.assertEqual(_resistance_rules(m), [])


class TestLiveMagicResistance(_LoggedTestCase):

    def test_graphics_unit_and_model_wrappers_agree(self):
        host = _UnitGraphics('Host', 'Magic Resistance (-2)')
        for wrapped in (host, host.unit, host.unit.model):
            with self.subTest(wrapper=type(wrapped).__name__):
                self.assertEqual(unit_magic_resistance(wrapped), (-2, 'Host', []))

    def test_joined_character_protects_the_host_and_itself(self):
        host = _UnitGraphics('Host', 'Magic Resistance (-1)')
        character = _UnitGraphics('Character', 'Magic Resistance (-3)', nmodels=1)
        _join(host, character)
        for target in (host, character):
            self.assertEqual(unit_magic_resistance(target), (-3, 'Character', []))

    def test_host_protects_a_weaker_joined_character(self):
        host = _UnitGraphics('Host', 'Magic Resistance (-3)')
        character = _UnitGraphics('Character', 'Magic Resistance (-1)', nmodels=1)
        _join(host, character)
        self.assertEqual(unit_magic_resistance(character), (-3, 'Host', []))

    def test_detaching_a_character_immediately_removes_its_grant(self):
        host = _UnitGraphics('Host', 'Magic Resistance (-1)')
        character = _UnitGraphics('Character', 'Magic Resistance (-3)', nmodels=1)
        _join(host, character)
        self.assertEqual(unit_magic_resistance(host)[0], -3)
        host.joinedCharacter = character.hostUnit = None
        self.assertEqual(unit_magic_resistance(host), (-1, 'Host', []))
        self.assertEqual(unit_magic_resistance(character), (-3, 'Character', []))

    def test_dead_joined_character_contributes_neither_value_nor_unknown(self):
        host = _UnitGraphics('Host', 'Magic Resistance (-1)')
        character = _UnitGraphics('Character', 'Magic Resistance (-3)',
                                  'Magic Resistance (-D3)', nmodels=1)
        _join(host, character)
        self.assertEqual(unit_magic_resistance(host)[0], -3)
        character.unit.nmodels = 0
        self.assertEqual(unit_magic_resistance(host), (-1, 'Host', []))

    def test_dead_host_does_not_supply_resistance(self):
        host = _UnitGraphics('Host', 'Magic Resistance (-3)',
                             'Magic Resistance (-D3)', nmodels=0)
        self.assertEqual(unit_magic_resistance(host), (0, '', []))

    def test_mount_crew_and_beasts_each_supply_live_resistance(self):
        for tag in ('mount', 'crew', 'beasts'):
            with self.subTest(part=tag):
                host = _UnitGraphics('Host', 'Magic Resistance (-1)')
                part = _model(tag, 'Magic Resistance (-3)')
                getattr(host.unit.model, f'attach_{tag}')(part)
                self.assertEqual(unit_magic_resistance(host), (-3, tag, []))
                apply_rule_keywords(part, ['Magic Resistance (-2)'], replace=True)
                self.assertEqual(unit_magic_resistance(host), (-2, tag, []))
                apply_rule_keywords(part, [], replace=True)
                self.assertEqual(unit_magic_resistance(host), (-1, 'Host', []))

    def test_wrapped_dead_parts_are_ignored(self):
        for tag in ('mount', 'crew', 'beasts'):
            with self.subTest(part=tag):
                host = _UnitGraphics('Host', 'Magic Resistance (-1)')
                part = _UnitGraphics(tag, 'Magic Resistance (-3)',
                                     'Magic Resistance (-D3)', nmodels=1)
                getattr(host.unit.model, f'attach_{tag}')(part.unit)
                self.assertEqual(unit_magic_resistance(host)[0], -3)
                part.unit.nmodels = 0
                self.assertEqual(unit_magic_resistance(host), (-1, 'Host', []))

    def test_joined_mount_and_other_parts_do_not_stack(self):
        host = _UnitGraphics('Host', 'Magic Resistance (-1)')
        character = _UnitGraphics('Character', 'Magic Resistance (-2)', nmodels=1)
        mount = _model('Mount', 'Magic Resistance (-4)')
        character.unit.model.attach_mount(mount)
        host.unit.model.attach_crew(_model('Crew', 'Magic Resistance (-3)'), 6)
        host.unit.model.attach_beasts(_model('Beasts', 'Magic Resistance (-2)'), 2)
        _join(host, character)
        self.assertEqual(unit_magic_resistance(host), (-4, 'Mount', []))
        apply_rule_keywords(mount, [], replace=True)
        self.assertEqual(unit_magic_resistance(host), (-3, 'Crew', []))

    def test_repeated_profile_links_are_visited_once_without_query_logs(self):
        host = _UnitGraphics('Host', 'Magic Resistance (-1)')
        part = _model('Shared part', 'Magic Resistance (-D3)')
        host.unit.model.attach_mount(part)
        host.unit.model.attach_crew(part)
        part.attach_mount(host.unit)
        for _ in range(3):
            self.assertEqual(unit_magic_resistance(host),
                             (-1, 'Host', ['Shared part: Magic Resistance (-D3)']))
        self.assertEqual(self.logs, [])


class TestMagicResistanceCasting(_LoggedTestCase):
    """Use Spell._attempt itself; only dice, dispel choice and effects are fakes."""

    def setUp(self):
        super().setUp()
        self.caster = _UnitGraphics('Caster')
        self.foe = _UnitGraphics('Enemy', 'Magic Resistance (-2)')
        self.order = []
        self.dispel_roll = 0

        async def dispel(spell, caster):
            self.order.append(('dispel', spell.casting))
            self.assertIs(caster, spell.caster)
            spell.apply.assert_not_awaited()
            return is_dispelled(self.dispel_roll, spell.casting)

        self.game = SimpleNamespace(player1Units=[self.caster],
                                    player2Units=[self.foe],
                                    dispelAttempt=mock.AsyncMock(side_effect=dispel))

    def _spell(self, **kwargs):
        options = dict(wizard_level=3, casting_value=7,
                       game=self.game, caster=self.caster)
        options.update(kwargs)
        spell = Spell('Test Hex', **options)

        async def apply(target):
            self.order.append(('apply', target))

        spell.apply = mock.AsyncMock(side_effect=apply)
        return spell

    def _cast(self, spell, target=None, rolls=((3, 4),)):
        self.order.clear()
        self.logs.clear()
        self.game.dispelAttempt.reset_mock()
        spell.apply.reset_mock()
        values = iter(rolls)

        async def roll():
            pair = next(values)
            self.order.append(('roll', pair))
            return sum(pair), list(pair)

        with contextlib.redirect_stdout(io.StringIO()) as out:
            with mock.patch.object(spell, '_roll_casting_dice',
                                   new_callable=mock.AsyncMock,
                                   side_effect=roll) as dice:
                asyncio.run(spell.spellFunction(self.foe if target is None else target))
        return dice, out.getvalue()

    def test_resistance_turns_a_success_into_failure_before_dispel(self):
        spell = self._spell(casting_value=8)
        dice, _ = self._cast(spell)
        self.assertEqual(spell.casting, 7)
        dice.assert_awaited_once_with()
        self.game.dispelAttempt.assert_not_awaited()
        spell.apply.assert_not_awaited()
        self.assertEqual(self.order, [('roll', (3, 4))])

    def test_live_downgrade_and_removal_cross_the_casting_threshold(self):
        spell = self._spell(casting_value=8)
        for keyword, result, applied in (
                ('Magic Resistance (-2)', 7, False),
                ('Magic Resistance (-1)', 8, True),
                ('Magic Resistance (-3)', 6, False),
                (None, 9, True)):
            with self.subTest(keyword=keyword):
                apply_rule_keywords(self.foe.unit.model,
                                    [keyword] if keyword else [], replace=True)
                self._cast(spell)
                self.assertEqual(spell.casting, result)
                self.assertEqual(spell.apply.await_count, int(applied))
                self.assertEqual(self.game.dispelAttempt.await_count, int(applied))

    def test_dispel_tie_uses_final_reduced_result_then_applies(self):
        spell = self._spell()
        self.dispel_roll = 7
        self._cast(spell)
        self.assertEqual(spell.casting, 7)
        self.assertEqual(self.order, [('roll', (3, 4)), ('dispel', 7),
                                     ('apply', self.foe)])
        self.game.dispelAttempt.assert_awaited_once_with(spell, self.caster)
        spell.apply.assert_awaited_once_with(self.foe)

    def test_dispel_beating_reduced_result_prevents_any_effect(self):
        spell = self._spell()
        self.dispel_roll = 8
        self._cast(spell)
        self.assertEqual(self.order, [('roll', (3, 4)), ('dispel', 7)])
        spell.apply.assert_not_awaited()

    def test_fired_log_reports_source_arithmetic_threshold_and_outcome(self):
        spell = self._spell(casting_value=8)
        _, output = self._cast(spell)
        entries = self.resistance_logs('fired')
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0][2], 'Enemy')
        for text in ('-2 from Enemy', 'strongest, not cumulative', 'Test Hex',
                     'dice 7 + bonus 2 -2 = 7', 'vs 8+ -> failed'):
            self.assertIn(text, entries[0][3])
        self.assertIn('[Rule] Magic Resistance', output)
        self.assertEqual(self.resistance_logs('skipped'), [])

    def test_unresolved_grants_log_a_skip_without_guessing_or_rerolling(self):
        for keyword in ('Magic Resistance', 'Magic Resistance (unknown)',
                        'Magic Resistance (-D3)'):
            with self.subTest(keyword=keyword):
                apply_rule_keywords(self.foe.unit.model, [keyword], replace=True)
                spell = self._spell(casting_value=9)
                dice, output = self._cast(spell)
                dice.assert_awaited_once_with()
                self.assertEqual(spell.casting, 9)
                spell.apply.assert_awaited_once_with(self.foe)
                entries = self.resistance_logs('skipped')
                self.assertEqual(len(entries), 1)
                self.assertEqual(entries[0][2], 'Enemy')
                self.assertIn(f'Enemy: {keyword}', entries[0][3])
                self.assertIn('no resolved numeric modifier', entries[0][3])
                self.assertIn('not guessed or re-rolled', entries[0][3])
                self.assertIn('not claimed', output)
                self.assertEqual(self.resistance_logs('fired'), [])

    def test_known_and_unknown_grants_log_their_independent_decisions(self):
        apply_rule_keywords(self.foe.unit.model,
                            ['Magic Resistance (-2)', 'Magic Resistance (-D3)'],
                            replace=True)
        dice, _ = self._cast(self._spell())
        dice.assert_awaited_once_with()
        self.assertEqual(len(self.resistance_logs('fired')), 1)
        self.assertEqual(len(self.resistance_logs('skipped')), 1)

    def test_friendly_spells_ignore_resistance_on_either_players_side(self):
        for side in (1, 2):
            for bound in (False, True):
                with self.subTest(side=side, bound=bound):
                    self.game.player1Units = [self.caster, self.foe] if side == 1 else []
                    self.game.player2Units = [self.caster, self.foe] if side == 2 else []
                    spell = self._spell(bound=bound, power_level=2, casting_value=9)
                    self._cast(spell)
                    self.assertEqual(spell.casting, 9)
                    spell.apply.assert_awaited_once_with(self.foe)
                    skipped = self.resistance_logs('skipped')
                    self.assertEqual(len(skipped), 1)
                    self.assertIn('friendly unit', skipped[0][3])
                    self.assertEqual(self.resistance_logs('fired'), [])

    def test_self_flag_and_range_redirect_to_caster_without_resistance(self):
        apply_rule_keywords(self.caster.unit.model,
                            ['Magic Resistance (-4)'], replace=True)
        for flag, spell_range in ((True, None), (False, 'Self'), (False, 'sElF')):
            with self.subTest(flag=flag, spell_range=spell_range):
                spell = self._spell(spell_range=spell_range, casting_value=9)
                spell.targets_self = flag
                self._cast(spell)
                self.assertEqual(spell.casting, 9)
                spell.apply.assert_awaited_once_with(self.caster)
                skipped = self.resistance_logs('skipped')
                self.assertEqual(len(skipped), 1)
                self.assertEqual(skipped[0][2], 'Caster')
                self.assertIn('Self spell has no enemy target', skipped[0][3])
                self.assertEqual(self.resistance_logs('fired'), [])

    def test_ground_target_ignores_resistant_units_under_the_effect(self):
        spell = self._spell(casting_value=9)
        spell.targets_ground = True
        point = Point3(2, 3, 0)
        with mock.patch('spell_system.unit_magic_resistance',
                        wraps=unit_magic_resistance) as resistance:
            self._cast(spell, target=point)
        resistance.assert_not_called()
        self.assertEqual(spell.casting, 9)
        spell.apply.assert_awaited_once_with(point)
        self.assertEqual(self.resistance_logs('fired'), [])

    def test_enemy_ownership_is_not_assumed_to_mean_player_two(self):
        self.game.player1Units, self.game.player2Units = [self.foe], [self.caster]
        spell = self._spell(casting_value=8)
        self._cast(spell)
        self.assertEqual(spell.casting, 7)
        spell.apply.assert_not_awaited()
        self.assertEqual(len(self.resistance_logs('fired')), 1)

    def test_joined_caster_uses_host_ownership_outside_player_lists(self):
        host = _UnitGraphics('Caster host')
        _join(host, self.caster)
        self.game.player1Units, self.game.player2Units = [self.foe], [host]
        spell = self._spell(casting_value=8)
        self._cast(spell)
        self.assertEqual(spell.casting, 7)
        spell.apply.assert_not_awaited()
        self.assertEqual(len(self.resistance_logs('fired')), 1)

    def test_detached_caster_can_use_recorded_ownership(self):
        self.caster._player = 2
        self.game.player1Units, self.game.player2Units = [self.foe], []
        spell = self._spell(casting_value=8)
        self._cast(spell)
        self.assertEqual(spell.casting, 7)
        spell.apply.assert_not_awaited()

    def test_joined_target_uses_host_resistance_and_ownership(self):
        character = _UnitGraphics('Enemy character', nmodels=1)
        _join(self.foe, character)
        spell = self._spell(casting_value=8)
        self._cast(spell, target=character)
        self.assertEqual(spell.casting, 7)
        spell.apply.assert_not_awaited()
        entries = self.resistance_logs('fired')
        self.assertEqual(len(entries), 1)
        self.assertIn('-2 from Enemy', entries[0][3])

    def test_missing_game_or_caster_skips_unknown_ownership(self):
        for missing in ('game', 'caster'):
            with self.subTest(missing=missing):
                spell = self._spell(**{missing: None}, casting_value=9)
                self._cast(spell)
                self.assertEqual(spell.casting, 9)
                spell.apply.assert_awaited_once_with(self.foe)
                self.game.dispelAttempt.assert_not_awaited()
                entries = self.resistance_logs('skipped')
                self.assertEqual(len(entries), 1)
                self.assertIn('ownership is unavailable', entries[0][3])

    def _assert_unknown_ownership_is_skipped(self):
        spell = self._spell(casting_value=9)
        self._cast(spell)
        self.assertEqual(spell.casting, 9,
                         'Unknown ownership must not invent an enemy casting penalty')
        spell.apply.assert_awaited_once_with(self.foe)
        self.assertEqual(self.resistance_logs('fired'), [])
        entries = self.resistance_logs('skipped')
        self.assertEqual(len(entries), 1)
        self.assertIn('ownership is unavailable', entries[0][3])

    def test_unlisted_caster_without_side_or_host_does_not_guess_ownership(self):
        self.game.player1Units, self.game.player2Units = [], [self.foe]
        self._assert_unknown_ownership_is_skipped()

    def test_unlisted_target_without_side_or_host_does_not_guess_ownership(self):
        self.game.player1Units, self.game.player2Units = [], [self.caster]
        self._assert_unknown_ownership_is_skipped()

    def test_natural_double_six_overrides_a_penalty_that_would_fail(self):
        apply_rule_keywords(self.foe.unit.model, ['Magic Resistance (-6)'], replace=True)
        spell = self._spell(casting_value=13)
        dice, _ = self._cast(spell, rolls=((6, 6),))
        dice.assert_awaited_once_with()
        self.assertEqual(spell.casting, 8)
        self.assertTrue(spell.perfect)
        self.assertFalse(spell.no_more_spells)
        self.game.dispelAttempt.assert_not_awaited()
        spell.apply.assert_awaited_once_with(self.foe)
        entries = self.resistance_logs('skipped')
        self.assertEqual(len(entries), 1)
        for text in ('-6 from Enemy', 'natural [6, 6]', 'perfect invocation'):
            self.assertIn(text, entries[0][3])
        self.assertEqual(self.resistance_logs('fired'), [])

    def test_natural_double_one_miscasts_even_if_modified_result_would_pass(self):
        spell = self._spell(casting_value=2)
        dice, output = self._cast(spell, rolls=((1, 1), (3, 4)))
        self.assertEqual(dice.await_count, 2)
        self.assertEqual(spell.casting, 2)
        self.assertFalse(spell.perfect)
        self.assertFalse(spell.no_more_spells)
        self.game.dispelAttempt.assert_not_awaited()
        spell.apply.assert_not_awaited()
        self.assertIn('Careless Conjuration', output)
        self.assertIn('natural [1, 1]: miscast', self.resistance_logs('skipped')[0][3])
        self.assertEqual(self.resistance_logs('fired'), [])

    def test_lower_miscast_rows_never_offer_dispel_or_apply(self):
        for table_dice in ((1, 1), (2, 2), (2, 3), (3, 3), (3, 4)):
            with self.subTest(table_dice=table_dice):
                spell = self._spell()
                dice, _ = self._cast(spell, rolls=((1, 1), table_dice))
                self.assertEqual(dice.await_count, 2)
                self.assertFalse(spell.perfect)
                self.assertFalse(spell.no_more_spells)
                self.game.dispelAttempt.assert_not_awaited()
                spell.apply.assert_not_awaited()

    def test_barely_controlled_power_overrides_resistance_for_dispel_threshold(self):
        for table_dice in ((4, 4), (4, 5)):
            for dispel in (8, 9):
                with self.subTest(table_dice=table_dice, dispel=dispel):
                    spell = self._spell(casting_value=8)
                    self.dispel_roll = dispel
                    dice, _ = self._cast(spell, rolls=((1, 1), table_dice))
                    self.assertEqual(dice.await_count, 2)
                    self.assertEqual(spell.casting, 8)
                    self.assertTrue(spell.no_more_spells)
                    self.assertFalse(spell.perfect)
                    expected = [('roll', (1, 1)), ('roll', table_dice), ('dispel', 8)]
                    if dispel == 8:
                        expected.append(('apply', self.foe))
                    self.assertEqual(self.order, expected)
                    self.assertEqual(spell.apply.await_count, int(dispel == 8))

    def test_power_drain_overrides_failure_and_bypasses_dispel(self):
        for table_dice in ((4, 6), (5, 6), (6, 6)):
            with self.subTest(table_dice=table_dice):
                spell = self._spell(casting_value=9)
                dice, output = self._cast(spell, rolls=((1, 1), table_dice))
                self.assertEqual(dice.await_count, 2)
                self.assertTrue(spell.perfect)
                self.assertTrue(spell.no_more_spells)
                self.game.dispelAttempt.assert_not_awaited()
                spell.apply.assert_awaited_once_with(self.foe)
                self.assertIn('Power Drain', output)

    def test_numeric_two_is_not_a_miscast_without_raw_double_one(self):
        spell = self._spell(wizard_level=2, casting_value=2)
        dice, _ = self._cast(spell, rolls=((1, 2),))
        dice.assert_awaited_once_with()
        self.assertEqual(spell.casting, 2)
        self.assertFalse(spell.perfect)
        self.assertFalse(spell.no_more_spells)
        spell.apply.assert_awaited_once_with(self.foe)

    def test_numeric_twelve_is_not_perfect_without_raw_double_six(self):
        apply_rule_keywords(self.foe.unit.model, [], replace=True)
        spell = self._spell(casting_value=12)
        self.dispel_roll = 13
        dice, _ = self._cast(spell, rolls=((5, 5),))
        dice.assert_awaited_once_with()
        self.assertEqual(spell.casting, 12)
        self.assertFalse(spell.perfect)
        self.game.dispelAttempt.assert_awaited_once_with(spell, self.caster)
        spell.apply.assert_not_awaited()

    def test_bound_adds_full_power_not_half_power_or_wizard_bonus(self):
        for level in (0, 1, 4):
            with self.subTest(wizard_level=level):
                spell = self._spell(bound=True, power_level=3,
                                    wizard_level=level, casting_value=8)
                self.dispel_roll = 8
                dice, _ = self._cast(spell)
                dice.assert_awaited_once_with()
                self.assertEqual(spell.casting, 8)
                self.assertEqual(self.order, [('roll', (3, 4)), ('dispel', 8),
                                             ('apply', self.foe)])
                self.assertFalse(spell.perfect)
                self.assertFalse(spell.no_more_spells)
                entries = [entry for entry in self.logs
                           if entry[:2] == ('fired', 'Bound Spells')]
                self.assertEqual(len(entries), 1)
                for text in ('2D6 7 + Power Level 3', '-2 (Magic Resistance)',
                             '= 8', 'no Wizard bonus, miscast or perfect invocation'):
                    self.assertIn(text, entries[0][3])

    def test_bound_with_no_power_does_not_fall_back_to_wizard_bonus(self):
        spell = self._spell(bound=True, wizard_level=4, casting_value=6)
        dice, _ = self._cast(spell)
        dice.assert_awaited_once_with()
        self.assertEqual(spell.casting, 5)
        self.game.dispelAttempt.assert_not_awaited()
        spell.apply.assert_not_awaited()

    def test_bound_double_one_is_ordinary_success_or_failure_never_miscast(self):
        for casting_value in (5, 6):
            with self.subTest(casting_value=casting_value):
                spell = self._spell(bound=True, power_level=5,
                                    casting_value=casting_value)
                self.dispel_roll = 6
                dice, output = self._cast(spell, rolls=((1, 1),))
                dice.assert_awaited_once_with()
                self.assertEqual(spell.casting, 5)
                self.assertFalse(spell.perfect)
                self.assertFalse(spell.no_more_spells)
                self.assertNotIn('Miscast!', output)
                self.assertEqual(self.game.dispelAttempt.await_count,
                                 int(casting_value == 5))
                spell.apply.assert_not_awaited()
                self.assertEqual(len(self.resistance_logs('fired')), 1)
                self.assertEqual(self.resistance_logs('skipped'), [])

    def test_bound_double_six_can_fail_or_be_dispelled_never_perfect(self):
        for casting_value in (13, 14):
            with self.subTest(casting_value=casting_value):
                spell = self._spell(bound=True, power_level=3,
                                    casting_value=casting_value)
                self.dispel_roll = 14
                dice, _ = self._cast(spell, rolls=((6, 6),))
                dice.assert_awaited_once_with()
                self.assertEqual(spell.casting, 13)
                self.assertFalse(spell.perfect)
                self.assertFalse(spell.no_more_spells)
                self.assertEqual(self.game.dispelAttempt.await_count,
                                 int(casting_value == 13))
                spell.apply.assert_not_awaited()
                self.assertEqual(len(self.resistance_logs('fired')), 1)
                self.assertEqual(self.resistance_logs('skipped'), [])


if __name__ == '__main__':
    unittest.main()