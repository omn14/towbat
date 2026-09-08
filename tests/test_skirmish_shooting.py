"""Individual Skirmisher shooting (Rulebook pp. 137, 139, 184-185)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
import asyncio

import pytest

from shooting_geometry import enemy_fire_modifier, model_shot, shooting_solution
from tests.test_skirmish_scene import restore, scene as scene


@pytest.mark.parametrize('distance,expected', [(5, (False, None)), (10, (True, None)),
                                               (11, (True, 'out of range'))])
def test_range_is_base_to_base_per_model(distance, expected):
    measured, long_range, reason = model_shot((0, 0, .5, .5, 0),
                                             [(0, distance + 1, .5, .5, 0)], [], 10)
    assert measured == pytest.approx(distance)
    assert (long_range, reason) == expected


def test_own_models_block_but_gaps_allow_sight():
    observer, target = (0, -4, .5, .5, 0), (0, 4, .5, .5, 0)
    assert model_shot(observer, [target], [(0, 0, 2, .5, 0)], 20)[2] == 'no line of sight'
    assert model_shot(observer, [target], [(-1, 0, .5, .5, 0), (1, 0, .5, .5, 0)], 20)[2] is None


def test_skirmisher_sight_is_360_but_formed_shooters_use_front_arc():
    observer, target = (0, 0, .5, .5, 0), (0, -4, .5, .5, 0)
    assert model_shot(observer, [target], [], 20)[2] is None
    assert model_shot(observer, [target], [], 20, facing=0)[2] == 'no line of sight'


def test_stand_and_shoot_waives_range_not_sight():
    observer, target = (0, 0, .5, .5, 0), (0, 40, .5, .5, 0)
    assert model_shot(observer, [target], [], 12, stand_and_shoot=True)[1:] == (False, None)
    assert model_shot(observer, [target], [(0, 20, 3, 1, 0)], 12,
                      stand_and_shoot=True)[2] == 'no line of sight'


@pytest.mark.parametrize('strength,alive,expected', [(1, True, True), (2, True, False), (2, False, True)])
def test_enemy_fire_checks_joined_model_strength(strength, alive, expected):
    target = SimpleNamespace(isSkirmisher=True, unit=SimpleNamespace(
        nmodels=5, model=SimpleNamespace(unit_strength=lambda: 1)))
    target.joinedCharacter = SimpleNamespace(unit=SimpleNamespace(
        nmodels=int(alive), model=SimpleNamespace(unit_strength=lambda: strength)))
    assert enemy_fire_modifier(target) is expected


@pytest.mark.parametrize('eligible', [0, 1, 4])
def test_explicit_firing_models_does_not_change_formation(eligible):
    from battleFunctions import simulate_battle
    from models import model
    from units import unit
    shooter = unit('Archers', model('State Trooper', ''), 10, 5, 2)
    target = unit('Target', model('State Trooper', ''), 10, 5, 2)
    shooter.model.give_weapon('Asrai Longbow')
    shooter.model.equip_weapon('Asrai Longbow')
    target.model.equip_best_melee()
    before = shooter.nmodels, shooter.files, shooter.ranks
    with patch('battleFunctions.simulate_attack', return_value=(False, False)):
        result = simulate_battle(shooter, target, charge=False, multiple_shots=False, firing_models=eligible)
    assert result[0] == eligible
    assert (shooter.nmodels, shooter.files, shooter.ranks) == before


def test_normal_volley_clears_previous_reaction_modifier():
    from battleFunctions import _ranged_tohit_report, simulate_battle
    from models import model
    from units import unit
    shooter = unit('Archers', model('State Trooper', ''), 5, 5, 1)
    target = unit('Target', model('State Trooper', ''), 5, 5, 1)
    shooter.model.give_weapon('Asrai Longbow')
    shooter.model.equip_weapon('Asrai Longbow')
    shooter.model.characteristics['BS'] = '3'
    target.model.equip_best_melee()
    with patch('battleFunctions.simulate_attack', return_value=(False, False)):
        simulate_battle(shooter, target, charge=False, firing_models=1, stand_and_shoot=True)
        assert _ranged_tohit_report(shooter.model)['target'] == 5
        simulate_battle(shooter, target, charge=False, firing_models=1)
        assert _ranged_tohit_report(shooter.model)['target'] == 4


def shooting_scene(scene):
    app, shooter = restore(scene)
    app.fsm.request('ShootingPhase')
    app.terrain_manager.clear()
    target = next(member for member in app.units if member.unitName == 'Warriors')
    for member in app.units:
        member.isDeployed = member in (shooter, target)
    shooter.bodyNP.setPos(0, 0, 0)
    shooter.bodyNP.setH(0)
    target.bodyNP.setPos(0, 8, 0)
    target.bodyNP.setH(180)
    shooter.unit.model.give_weapon('Asrai Longbow')
    shooter.unit.model.equip_weapon('Asrai Longbow')
    shooter.unit.model.equipedWeapon['ranged_range'] = 12
    for index, child in enumerate(shooter.model.getChildren()):
        child.setPos(app.render, (index - 2) * 1.2, 2 if index < 2 else 0, 0)
    shooter.rebuildFootprint()
    app.unitToMove = shooter
    app.roundCounter.current_player = 1
    app.world.doPhysics(1 / 60)
    return app, shooter, target


def test_real_volley_uses_preview_counts_and_individual_ranges(scene, capsys):
    app, shooter, target = shooting_scene(scene)
    geometry = shooting_solution(app, shooter, target)
    assert {model.long_range for model in geometry.eligible} == {False, True}
    original = shooter.unit.files, shooter.unit.ranks, shooter.bodyNP.getTransform()
    groups = []

    def volley(unit, defender, **kwargs):
        groups.append((unit, kwargs['firing_models'], unit.model.at_long_range))
        return kwargs['firing_models'], 0, 0, 0, 0

    with patch('game.simulate_battle', side_effect=volley), \
            patch.object(app.taskMgr, 'add', side_effect=lambda coroutine: coroutine.close()), \
            patch.object(app, 'shootingAnimation', AsyncMock()), \
            patch.object(app, 'printBattleResults'):
        asyncio.run(app.shootAt(shooter, target))
    assert sum(count for unit, count, long_range in groups) == len(geometry.eligible)
    for unit, count, long_range in groups:
        assert count == sum(model.long_range == long_range for model in geometry.eligible)
    assert shooter.hasAttackedThisTurn
    assert (shooter.unit.files, shooter.unit.ranks, shooter.bodyNP.getTransform()) == original
    assert geometry.detail() in capsys.readouterr().out


def test_screened_volley_is_refused_without_spending_shooting(scene):
    app, shooter, target = shooting_scene(scene)
    from panda3d.core import Point3
    wall = SimpleNamespace(center=Point3(0, 4, 0), width=20, height=.5, blocks_line_of_sight=True,
                           contains=lambda point: abs(point.x) <= 10 and abs(point.y - 4) <= .25)
    with patch.object(app.terrain_manager, 'terrain_pieces', [wall]), \
            patch('game.simulate_battle') as volley, patch.object(app, 'makeChoiceNew') as choice:
        asyncio.run(app.shootAt(shooter, target))
    assert not shooter.hasAttackedThisTurn
    volley.assert_not_called()
    choice.assert_not_called()


def test_target_highlighting_matches_individual_query(scene):
    from panda3d.core import BitMask32
    app, shooter, target = shooting_scene(scene)
    app.shootingArcPoints = []
    assert shooting_solution(app, shooter, target).eligible
    assert app.checkArrows()
    assert target.bodyNP.getCollideMask() == BitMask32.bit(3)


@pytest.mark.parametrize('invalid', ['friendly', 'undeployed', 'engaged', 'spent'])
def test_direct_volley_rejects_invalid_targets_without_dice(scene, invalid):
    app, shooter, target = shooting_scene(scene)
    if invalid == 'friendly':
        target = shooter
    elif invalid == 'undeployed':
        target.isDeployed = False
    elif invalid == 'engaged':
        target.isInCombat = True
    else:
        shooter.hasAttackedThisTurn = True
    with patch('game.simulate_battle') as volley, patch.object(app, 'makeChoiceNew') as choice:
        asyncio.run(app.shootAt(shooter, target))
    volley.assert_not_called()
    choice.assert_not_called()
    assert shooter.hasAttackedThisTurn is (invalid == 'spent')


@pytest.mark.parametrize('hill_position,visible', [('none', False), ('shooter', True),
                                                 ('target', True), ('higher_blocker', False)])
def test_hill_visibility_preserves_unit_screening_exceptions(scene, hill_position, visible):
    from panda3d.core import Point3
    from scouts import model_base_boxes
    app, shooter, target = shooting_scene(scene)
    blocker = next(member for member in app.units if member not in (shooter, target))
    blocker.isDeployed = True
    blocker.bodyNP.setPos(0, 4, 0)
    hill = SimpleNamespace(center=Point3(0, 4, 0))

    def hill_under(member):
        if (hill_position == 'shooter' and member is shooter
                or hill_position == 'target' and member is target
                or hill_position == 'higher_blocker' and member in (shooter, blocker)):
            return hill
        return None

    with patch.object(app.movement, 'hillUnderUnit', side_effect=hill_under), \
            patch('shooting_geometry.model_base_boxes', side_effect=lambda member:
                  [(0, 4, 10, .5, 0)] if member is blocker else model_base_boxes(member)):
        assert bool(shooting_solution(app, shooter, target).eligible) is visible


def test_mixed_ranges_share_one_multiple_shots_choice(scene):
    from tests.test_shieldwall_scene import combat_tasks
    app, shooter, target = shooting_scene(scene)
    shooter.unit.model.equipedWeapon.update(ranged_shots=2, multiple_shots=True)
    geometry = shooting_solution(app, shooter, target)
    assert {model.long_range for model in geometry.eligible} == {False, True}
    calls = []

    def volley(unit, defender, **kwargs):
        calls.append(kwargs)
        return kwargs['firing_models'] * 2, 0, 0, 0, 0

    async def choose(options, *args, **kwargs):
        return next(option for option in options if option.startswith('Multiple Shots'))

    with combat_tasks(app) as run, patch('game.simulate_battle', side_effect=volley), \
            patch.object(app, 'makeChoiceNew', AsyncMock(side_effect=choose)) as choice, \
            patch.object(app, 'shootingAnimation', AsyncMock()), patch.object(app, 'printBattleResults'):
        run(app.shootAt(shooter, target))
    choice.assert_awaited_once()
    assert len(calls) == 2 and all(call['multiple_shots'] for call in calls)
    assert sum(call['firing_models'] for call in calls) == len(geometry.eligible)


def test_reaction_uses_original_target_bases_without_moving_live_models(scene):
    from scouts import model_base_boxes
    app, shooter, target = shooting_scene(scene)
    declared, facing = target.bodyNP.getPos(), target.bodyNP.getHpr()
    original_boxes = model_base_boxes(target)
    target.bodyNP.setPos(0, 2, 0)
    target.bodyNP.setH(23)
    current = target.bodyNP.getTransform()
    option = app.combat.standAndShootOption(shooter, target, declared, facing)
    assert option is not None
    for actual, expected in zip(option.target_boxes, original_boxes):
        assert actual == pytest.approx(expected, abs=1e-5)
    assert target.bodyNP.getTransform() == current
    assert option.distance > 3


def test_reaction_modifier_reaches_real_hit_roll():
    from battleFunctions import _ranged_tohit_report, simulate_attack
    from models import model
    shooter, target = model('State Trooper', ''), model('State Trooper', '')
    shooter.give_weapon('Asrai Longbow')
    shooter.equip_weapon('Asrai Longbow')
    shooter.characteristics['BS'] = '3'
    shooter.stand_and_shoot = True
    assert _ranged_tohit_report(shooter)['target'] == 5
    with patch('battleFunctions.random.randint', side_effect=[4, 6]):
        assert simulate_attack(shooter, target)[0] is False
    shooter.stand_and_shoot = False
    with patch('battleFunctions.random.randint', side_effect=[4, 6]):
        assert simulate_attack(shooter, target)[0] is True


@pytest.mark.parametrize('hill', [False, True])
def test_formed_firing_ranks_keep_their_limits_and_inherit_front_sight(scene, hill):
    app, target, shooter = shooting_scene(scene)
    shooter.unit.model.give_weapon('Asrai Longbow')
    shooter.unit.model.equip_weapon('Asrai Longbow')
    with patch.object(app.movement, 'entirelyOnHill', side_effect=lambda member: hill and member is shooter):
        geometry = shooting_solution(app, shooter, target)
    assert len(geometry.models) == (10 if hill else 8)
    assert len(geometry.eligible) == len(geometry.models)


def test_joined_shooter_has_own_weapon_range_and_does_not_replace_a_loose_model(scene):
    from characters import join_unit
    app, shooter, target = shooting_scene(scene)
    character = app._create_unit(dict(name='Captain of the Empire', nmodels=1, files=1, ranks=1),
                                 1, 'Shooting Captain')
    assert join_unit(app, character, shooter)
    character.unit.model.give_weapon('Asrai Longbow')
    character.unit.model.equip_weapon('Asrai Longbow')
    character.unit.model.equipedWeapon['ranged_range'] = 1
    character.model.getChild(0).setPos(app.render, 4, 1, 0)
    geometry = shooting_solution(app, shooter, target)
    assert len(geometry.models) == shooter.unit.nmodels + 1
    own = next(model for model in geometry.models if model.unit is character)
    assert own.reason == 'out of range'
    character.unit.model.equipedWeapon['ranged_range'] = 30
    geometry = shooting_solution(app, shooter, target)
    assert any(model.unit is character for model in geometry.eligible)
    fired = []

    def volley(unit, defender, **kwargs):
        fired.append((unit, kwargs['firing_models']))
        return kwargs['firing_models'], 0, 0, 0, 0

    with patch('game.simulate_battle', side_effect=volley), \
            patch.object(app.taskMgr, 'add', side_effect=lambda coroutine: coroutine.close()), \
            patch.object(app, 'shootingAnimation', AsyncMock()), patch.object(app, 'printBattleResults'):
        asyncio.run(app.shootAt(shooter, target))
    assert sum(count for unit, count in fired) == len(geometry.eligible)
    assert (character.unit, 1) in fired


@pytest.mark.parametrize('size', [(1280, 720), (800, 600)])
def test_individual_aiming_readout_renders(scene, tmp_path, size):
    from panda3d.core import Filename, PNMImage
    app, shooter, target = shooting_scene(scene)
    old_size = app.win.getXSize(), app.win.getYSize()
    camera_transform = app.camera.getTransform()
    window = app.openWindow(type='offscreen', size=size, makeCamera=False)
    for order, camera in enumerate((app.cam, app.cam2d, app.cam2dp)):
        region = window.makeDisplayRegion()
        region.setCamera(camera)
        region.setSort(order * 10)
    try:
        app.adjustWindowAspectRatio(size[0] / size[1])
        app.camera.setPos(0, -24, 34)
        app.camera.lookAt(0, 4, 0)
        for other in app.units:
            if other not in (shooter, target):
                other.model.hide()
        app.taskShootingArcUpdate(SimpleNamespace(done='done'))
        with patch.object(app, 'targetUnderMouse', return_value=target):
            app.taskShootingTrajectoryDrawLine(SimpleNamespace(cont='cont'))
        expected = shooting_solution(app, shooter, target).detail()
        assert expected.replace('; ', '\n') in app.debugTextInfo.getText()
        lower, upper = app.debugTextInfo.getTightBounds(app.aspect2d)
        assert -size[0] / size[1] <= lower.x < upper.x <= size[0] / size[1]
        assert -1 <= lower.z < upper.z <= 1
        app.graphicsEngine.renderFrame()
        app.graphicsEngine.renderFrame()
        image = PNMImage()
        assert window.getScreenshot(image)
        assert image.write(Filename.fromOsSpecific(str(tmp_path / f'shooting-{size[0]}x{size[1]}.png')))
        assert image.getAverageXel().length() > .1
    finally:
        for other in app.units:
            other.model.show()
        app.camera.setTransform(camera_transform)
        app.closeWindow(window, keepCamera=True)
        app.adjustWindowAspectRatio(old_size[0] / old_size[1])


@pytest.mark.parametrize('armed', [False, True])
def test_formed_joined_character_occupies_one_front_rank_slot(scene, armed):
    from characters import join_unit
    app, target, shooter = shooting_scene(scene)
    shooter.unit.model.give_weapon('Asrai Longbow')
    shooter.unit.model.equip_weapon('Asrai Longbow')
    shooter.unit.model.special_rules = [rule for rule in shooter.unit.model.special_rules
                                      if not rule.get('volley_fire')]
    character = app._create_unit(dict(name='Captain of the Empire', nmodels=1, files=1, ranks=1),
                                 2, 'Formed Shooting Captain')
    assert join_unit(app, character, shooter)
    if armed:
        character.unit.model.give_weapon('Asrai Longbow')
        character.unit.model.equip_weapon('Asrai Longbow')
    geometry = shooting_solution(app, shooter, target)
    assert len(geometry.eligible) == shooter.unit.files - 1 + int(armed)
    assert sum(model.unit is character for model in geometry.eligible) == int(armed)
    with patch.object(app.movement, 'entirelyOnHill', side_effect=lambda member: member is shooter):
        geometry = shooting_solution(app, shooter, target)
    assert len(geometry.eligible) == shooter.unit.files * 2 - 1 + int(armed)