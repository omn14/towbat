"""Skycutter roster/save recovery (Forces of Fantasy p. 172, Rulebook p. 195)."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from panda3d.core import getModelPath, loadPrcFileData

from battleFunctions import impact_hit_report, resolve_impact_hits, unmodified_strength
from game import MyApp
from models import model
from persistence import load_game_state, save_game_state
from special_rules import apply_rule_keywords
from spell_system import OakenShieldSpell

ROOT = Path(__file__).resolve().parents[1]


def build_scene():
    loadPrcFileData('', 'window-type offscreen\nwin-size 1280 720\naudio-library-name null')
    getModelPath().appendDirectory(str(ROOT))

    def player_one(app, _):
        app._create_unit({'name': 'Lothern Skycutter', 'nmodels': 1, 'files': 1,
                          'ranks': 1, 'armour': ['Armour Value : 4+'],
                          'special_rules': ['Impact Hits (D3+1)', 'Fly (10)']},
                         1, 'Skycutter')

    def player_two(app, _):
        app._create_unit({'name': 'Chaos Knight', 'nmodels': 5, 'files': 5, 'ranks': 1},
                         2, 'Knights')

    with patch.object(MyApp, 'load_player1_army', player_one), \
            patch.object(MyApp, 'load_player2_army', player_two):
        app = MyApp()
    app.AIplayer2.active = False
    app.fsm.request('MovementPhase')
    for index, unit in enumerate(app.units):
        unit.bodyNP.setPos(0, -5 + index * 10, 0)
        unit.bodyNP.setH(180 if index else 0)
        unit.isDeployed = True
    app.unitToMove = app.player1Units[0]
    return app


def assert_skycutter(unit):
    profile = unit.unit.model
    assert unmodified_strength(profile) == 5
    assert profile.characteristics['T'] == '4'
    assert profile.characteristics['W'] == '4'
    assert profile.is_chariot() and profile.impact_hit_ap() == 2
    assert profile.armor_save == 4
    assert profile.get_fly_movement() == 10
    assert profile.get_crew().name == 'Sea Guard Crew'
    assert profile.part_count('crew') == 3
    assert profile.firing_bs() == 4 and profile.defending_ws() == 4
    assert profile.get_beasts().name == 'Swiftfeather Roc'
    assert profile.part_count('beasts') == 1
    assert unit.modelWidth == pytest.approx(60 / 25.4)
    assert unit.modelHeight == pytest.approx(100 / 25.4)


@pytest.fixture(scope='module')
def scene(tmp_path_factory):
    app = build_scene()
    path = save_game_state(app, str(tmp_path_factory.mktemp('skycutter') / 'baseline.json'))
    yield app, path
    app.destroy()


@pytest.mark.parametrize('recreate', [False, True])
def test_statless_saved_skycutter_recovers_live_profile(scene, tmp_path, recreate, capsys):
    app, baseline = scene
    data = json.loads(Path(baseline).read_text())
    record = next(record for record in data['units'] if record['name'] == 'Skycutter')
    record['characteristics'] = {'Special Rules': record['special_rules']}
    record.pop('base_characteristics', None)
    record['woundsOnModel'] = 2
    record['chargedThisTurn'] = True
    path = tmp_path / 'broken.json'
    path.write_text(json.dumps(data))
    if recreate:
        absent = dict(data, units=[record for record in data['units'] if record['name'] != 'Skycutter'])
        absent_path = tmp_path / 'absent.json'
        absent_path.write_text(json.dumps(absent))
        load_game_state(app, str(absent_path))
    load_game_state(app, str(path))
    skycutter = app.player1Units[0]
    assert_skycutter(skycutter)
    assert skycutter.woundsOnModel == 2 and skycutter.chargedThisTurn
    assert 'Restored missing profile for Skycutter' in capsys.readouterr().out
    assert json.loads(path.read_text()) == data
    profile = skycutter.unit.model
    profile.characteristics['S'] = '9'
    profile.reset_characteristics()
    assert_skycutter(skycutter)
    report = impact_hit_report(skycutter.unit, app.player2Units[0].unit)
    assert 'S5 AP-2  [wound 3+]' in report[0]
    with patch('battleFunctions.random.randint', return_value=3), \
            patch('battleFunctions.check_saves', return_value=False) as saves:
        assert resolve_impact_hits(skycutter.unit, app.player2Units[0].unit) == (4, 4, 0, 4)
    assert saves.call_count == 4
    assert all(call.args[2] == 2 for call in saves.call_args_list)
    saved = save_game_state(app, str(tmp_path / 'recovered.json'))
    load_game_state(app, saved)
    assert_skycutter(app.player1Units[0])
    app.graphicsEngine.renderFrame()
    app.graphicsEngine.renderFrame()
    assert app.screenshot(str(tmp_path / 'skycutter.png'), defaultFilename=False)


def test_valid_custom_saved_stats_are_not_replaced(scene, tmp_path):
    app, baseline = scene
    data = json.loads(Path(baseline).read_text())
    record = next(record for record in data['units'] if record['name'] == 'Skycutter')
    record.pop('base_characteristics', None)
    record['characteristics']['S'] = '7'
    record['characteristics']['T'] = '6'
    path = tmp_path / 'custom.json'
    path.write_text(json.dumps(data))
    load_game_state(app, str(path))
    profile = app.player1Units[0].unit.model
    profile.reset_characteristics()
    assert unmodified_strength(profile) == 7
    assert profile.characteristics['T'] == '6'


def test_temporary_strength_does_not_become_saved_baseline(scene, tmp_path):
    app, baseline = scene
    load_game_state(app, baseline)
    profile = app.player1Units[0].unit.model
    profile._base_characteristics['S'] = '7'
    profile.characteristics['S'] = '9'
    path = save_game_state(app, str(tmp_path / 'temporary-strength.json'))
    load_game_state(app, path)
    profile = app.player1Units[0].unit.model
    assert profile.characteristics['S'] == '9'
    assert unmodified_strength(profile) == 7
    profile.reset_characteristics()
    assert profile.characteristics['S'] == '7'


@pytest.mark.parametrize('tag', ['mount', 'crew', 'beasts'])
def test_split_profile_baseline_and_bonus_survive_reload(scene, tmp_path, tag):
    app, baseline = scene
    load_game_state(app, baseline)
    profile = app.player1Units[0].unit.model
    if tag == 'mount':
        profile = app.player2Units[0].unit.model
        profile.attach_mount(model('Barded Warhorse', ''))
    part = getattr(profile, f'get_{tag}')()
    part._base_characteristics['S'] = '6'
    part.characteristics['S'] = '8'
    apply_rule_keywords(part, ['Veteran'], replace=True)
    path = save_game_state(app, str(tmp_path / f'{tag}.json'))
    load_game_state(app, path)
    restored = getattr(profile, f'get_{tag}')()
    assert restored.characteristics['S'] == '8'
    assert unmodified_strength(restored) == 6
    restored.reset_characteristics()
    assert restored.characteristics['S'] == '6'
    assert restored.characteristics['Special Rules'] == ['Veteran']
    assert restored.is_veteran()
    if tag != 'mount':
        assert profile.part_count(tag) == (3 if tag == 'crew' else 1)


def test_ongoing_spell_restores_once_and_expires_without_rebasing(scene, tmp_path):
    app, baseline = scene
    load_game_state(app, baseline)
    target = app.player1Units[0]
    profile = target.unit.model
    profile._base_characteristics['S'] = '7'
    profile.characteristics['S'] = '9'
    spell = OakenShieldSpell('Oaken Shield', 7, app.fsm.endOfTurnSpells,
                            game=app, caster=target)
    spell.attach(target, 2)
    path = save_game_state(app, str(tmp_path / 'ongoing.json'))
    for _ in range(2):
        load_game_state(app, path)
        assert len(app.fsm.endOfTurnSpells) == 1
        restored = app.fsm.endOfTurnSpells[0]
        assert restored.ticks_remaining == 2
        assert sum(rule.get('name') == 'Oaken Shield' for rule in profile.special_rules) == 1
        assert profile.characteristics['S'] == '9'
        profile.reset_characteristics()
        assert profile.characteristics['S'] == '7'
        assert restored.rule in profile.special_rules
    restored.endSpell()
    assert not any(rule.get('name') == 'Oaken Shield' for rule in profile.special_rules)
    assert profile.characteristics['S'] == '7'