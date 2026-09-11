"""Selected army joining and command restrictions (pp. 172, 190, 193; FoF p. 170)."""

from unittest.mock import patch

from characters import join_unit
from persistence import load_game_state
from psychology import close_order_bonus, combat_rank_bonus, select_general
from tests.test_faction_rules_scene import members, scene as scene


def test_warhounds_cannot_be_joined_or_use_general_or_battle_standard(scene):
    app, baseline = scene
    load_game_state(app, baseline)
    units = members(app)
    champion, hounds = units['Aspiring Champion'], units['Chaos Warhound']
    assert not join_unit(app, champion, hounds)
    with patch.object(app.psychology, '_command_source', return_value=champion) as source:
        assert app.psychology.general_of(hounds) is None
        assert app.psychology.battle_standard_of(hounds) is None
    source.assert_not_called()


def test_mage_general_can_join_dragon_princes_but_ordinary_mage_cannot(scene):
    app, baseline = scene
    load_game_state(app, baseline)
    units = members(app)
    mage, princes = units['Mage'], units['Dragon Prince']
    with patch.object(mage, 'isGeneral', False):
        assert not join_unit(app, mage, princes)
    assert mage.isGeneral
    assert join_unit(app, mage, princes)


def test_loner_character_cannot_be_general_even_if_roster_nominated(scene):
    app, baseline = scene
    load_game_state(app, baseline)
    mage = members(app)['Mage']
    with patch.object(mage.unit.model, 'special_rules', [*mage.unit.model.special_rules, {'name': 'Loner'}]):
        assert select_general([mage]) is None
    assert not mage.isGeneral


def test_heavy_infantry_resists_us_below_ten_but_not_first_charge(scene):
    app, baseline = scene
    load_game_state(app, baseline)
    units = members(app)
    warriors, enemy = units['Chaos Warrior'], units['Dragon Prince']
    warriors.isInCombatWith = [enemy]
    warriors.isInCombatFlank = ['flank']
    warriors.isDisrupted = False
    assert combat_rank_bonus(warriors) == 1
    warriors.firstChargeDisruptedBy = [enemy.unit.name]
    assert combat_rank_bonus(warriors) == 0
    warriors.firstChargeDisruptedBy = []
    with patch.object(enemy.unit, 'nmodels', 5):
        assert combat_rank_bonus(warriors) == 0


def test_lumbering_skycutter_cannot_be_joined(scene):
    app, baseline = scene
    load_game_state(app, baseline)
    units = members(app)
    assert not join_unit(app, units['Mage'], units['Lothern Skycutter'])


def test_close_order_uses_current_strength_and_active_formation(scene, capsys):
    app, baseline = scene
    load_game_state(app, baseline)
    units = members(app)
    warriors = units['Chaos Warrior']
    with patch.object(warriors.unit, 'nmodels', 10), patch.object(warriors.unit, 'files', 5):
        assert close_order_bonus(warriors, log=True) == 1
        with patch.object(warriors.unit, 'nmodels', 9):
            assert close_order_bonus(warriors, log=True) == 0
        with patch.object(warriors.unit, 'files', 2):
            assert close_order_bonus(warriors, log=True) == 0
        with patch.object(warriors, 'isDisrupted', True):
            assert close_order_bonus(warriors) == 1
    for name in ('Dragon Prince', 'Lothern Skycutter', 'Chaos Warhound', 'Marauder Horsemen'):
        assert close_order_bonus(units[name], log=True) == 0
    horsemen = units['Marauder Horsemen']
    assert horsemen.isSkirmisher
    with patch.object(horsemen, 'isSkirmisher', False), patch.object(horsemen, 'skirmishCombat', True):
        assert close_order_bonus(horsemen) == 0
    output = capsys.readouterr().out
    assert 'Unit Strength 9 is below 10' in output
    assert 'Marching Column' in output


def test_joined_character_strength_counts_but_not_as_another_formation(scene):
    app, baseline = scene
    load_game_state(app, baseline)
    units = members(app)
    champion, warriors = units['Aspiring Champion'], units['Chaos Warrior']
    assert join_unit(app, champion, warriors)
    with patch.object(warriors.unit, 'nmodels', 9), patch.object(warriors.unit, 'files', 5):
        assert close_order_bonus(warriors) == 1
        assert close_order_bonus(champion) == 0