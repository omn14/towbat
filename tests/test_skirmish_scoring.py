"""Skirmisher scoring, Rulebook p. 185 and Unusual Formations FAQ v1.5.3."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from psychology import combat_flank_bonus, combat_rank_bonus
from tests.test_shieldwall_scene import combat_tasks, scene as scene


@pytest.mark.parametrize('skirmisher', [False, True])
@pytest.mark.parametrize('compact', [False, True])
@pytest.mark.parametrize('face,points', [('front', 0), ('flank', 1), ('rear', 2)])
def test_scoring_exemption_does_not_change_contact_geometry(skirmisher, compact, face, points, capsys):
    unit = SimpleNamespace(unitName='Target', isSkirmisher=skirmisher, skirmishCombat=compact,
                           isInCombatWith=[SimpleNamespace(unit=SimpleNamespace(nmodels=1))],
                           isInCombatFlank=[face], unit=SimpleNamespace(
                               model=SimpleNamespace(is_skirmisher=lambda: skirmisher, special_rules=[])))
    assert combat_flank_bonus(unit, log=True) == (0 if skirmisher or compact else points)
    assert unit.isInCombatFlank == [face]
    output = capsys.readouterr().out
    assert ('Skirmishers' in output) is ((skirmisher or compact) and points > 0)


def test_scoring_queries_are_silent(capsys):
    unit = SimpleNamespace(isSkirmisher=True, isInCombatFlank=['flank', 'rear'])
    assert combat_flank_bonus(unit) == 0
    assert capsys.readouterr().out == ''


def regiment(name, count=15, skirmisher=False):
    profile = SimpleNamespace(is_skirmisher=lambda: skirmisher, unit_strength=lambda: 1,
                              models_per_rank=lambda default: 5, max_rank_bonus=lambda default: 2)
    return SimpleNamespace(unitName=name, isSkirmisher=skirmisher, skirmishCombat=False,
                           isDisrupted=False, isInCombatWith=[], isInCombatFlank=[],
                           unit=SimpleNamespace(name=name, model=profile, nmodels=count, files=5,
                                                ranks=(count + 4) // 5))


@pytest.mark.parametrize('skirmisher', [False, True])
@pytest.mark.parametrize('strength', [4, 5, 10])
@pytest.mark.parametrize('face', ['front', 'flank', 'rear'])
def test_only_formed_enemy_of_sufficient_strength_disrupts(skirmisher, strength, face):
    target = regiment('Spears')
    enemy = regiment('Enemy', strength, skirmisher)
    target.isInCombatWith, target.isInCombatFlank = [enemy], [face]
    expected = 0 if not skirmisher and strength >= 5 and face != 'front' else 2
    assert combat_rank_bonus(target, log=True) == expected
    assert not target.isDisrupted
    assert combat_flank_bonus(target) == {'front': 0, 'flank': 1, 'rear': 2}[face]


def test_terrain_and_other_formed_enemy_still_disrupt():
    target = regiment('Spears')
    target.isInCombatWith = [regiment('Scouts', skirmisher=True), regiment('Knights')]
    target.isInCombatFlank = ['rear', 'flank']
    assert combat_rank_bonus(target) == 0
    target.isInCombatWith.pop()
    target.isInCombatFlank.pop()
    assert combat_rank_bonus(target) == 2
    target.isDisrupted = True
    assert combat_rank_bonus(target) == 0


@pytest.mark.parametrize('survivors,points', [(15, 2), (10, 1), (9, 0), (4, 0), (0, 0)])
def test_rank_points_use_survivors_before_visual_ranks_refresh(survivors, points):
    target = regiment('Spears')
    target.unit.nmodels = survivors
    assert target.unit.ranks == 3
    assert combat_rank_bonus(target) == points


def test_slain_enemy_neither_scores_flank_points_nor_disrupts():
    target, enemy = regiment('Spears'), regiment('Enemy')
    target.isInCombatWith, target.isInCombatFlank = [enemy], ['rear']
    enemy.unit.nmodels = 0
    assert combat_flank_bonus(target) == 0
    assert combat_rank_bonus(target) == 2
    assert target.isInCombatFlank == ['rear']


def test_joined_character_counts_towards_disrupting_strength():
    target, enemy = regiment('Spears'), regiment('Enemy', 4)
    target.isInCombatWith, target.isInCombatFlank = [enemy], ['rear']
    assert combat_rank_bonus(target) == 2
    enemy.joinedCharacter = regiment('Hero', 1)
    assert combat_rank_bonus(target) == 0


def test_active_formation_and_compact_combat_override_keyword():
    target = regiment('Scouts', skirmisher=True)
    target.isSkirmisher = False
    assert combat_rank_bonus(target) == 2
    target.skirmishCombat = True
    assert combat_rank_bonus(target) == 0
    target.isInCombatFlank = ['rear']
    assert combat_flank_bonus(target) == 0


@pytest.mark.parametrize('skirmisher_side', [1, 2])
def test_real_combat_table_preserves_geometry_and_skirmisher_exemptions(scene, skirmisher_side):
    from persistence import load_game_state
    app, baseline = scene
    load_game_state(app, baseline)
    defender, charger = app.player1Units[0], app.player2Units[0]
    skirmisher = defender if skirmisher_side == 1 else charger
    defender.isInCombatFlank = ['rear']
    charger.isInCombatFlank = ['front']
    app.unitToMove = charger

    async def choose(options, *args, **kwargs):
        return next((choice for choice in ('Hand weapon & shield', 'Stand Firm', 'Shieldwall', 'Follow up')
                     if choice in options), next(iter(options)))

    with combat_tasks(app) as run, \
            patch.object(skirmisher, 'isSkirmisher', True), \
            patch.object(skirmisher, 'skirmishCombat', True), \
            patch.object(app, 'makeChoiceNew', AsyncMock(side_effect=choose)), \
            patch.object(app.combat, 'breakTestPass', AsyncMock(return_value=[])), \
            patch.object(app.combat, 'printCombatResult') as table:
        run(app.combat._verySimpleBattleInner(SimpleNamespace(done='done')))
    rows = table.call_args.args[0]
    assert rows['Flank / rear'] == (0, 0 if skirmisher_side == 1 else 2)
    assert rows['Rank Bonus'] == (0, 1 if skirmisher_side == 1 else 0)
    assert defender.isInCombatFlank == ['rear']