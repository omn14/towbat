import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from panda3d.core import NodePath

from aiMinimaxIntegration import ActionOutcome, EnhancedAI
from gameStateTree import GameAction, GameState, MinimaxTree
from minimaxOptimizations import OptimizedMinimaxTree
from models import model


@pytest.fixture
def snapshot_game():
    root = NodePath('ai-test-world')
    game = SimpleNamespace(
        render=root, units=[], player1Units=[], player2Units=[],
        chargeStage='declarations',
        fsm=SimpleNamespace(
            state='MovementPhase', currentPhaseIndex=1,
            phases=['StrategyPhase', 'MovementPhase', 'ShootingPhase', 'CombatPhase']),
        roundCounter=SimpleNamespace(
            currentRoundPlayer=[1, 0], current_player=2, max_rounds=5))

    def add_unit(name, player, profile=None):
        profile = profile if profile is not None else model('Silver Helm', '')
        member = SimpleNamespace(
            unitName=name, bodyNP=root.attachNewNode(name),
            unit=SimpleNamespace(name=name, model=profile, nmodels=5, files=5, ranks=1),
            state='Idle', isInCombat=False, hasMovedThisTurn=False,
            hasAttackedThisTurn=False, isDeployed=True,
            isInCombatWith=[], isInCombatFlank=[], hostUnit=None,
            updateTextNode=Mock())
        game.units.append(member)
        (game.player1Units if player == 1 else game.player2Units).append(member)
        return member

    yield game, add_unit
    root.removeNode()


@pytest.mark.parametrize('kind', ['cannon', 'bombardment'])
def test_artillery_misfire_spends_exactly_one_shot(snapshot_game, monkeypatch, kind):
    from cannon_fire import CannonFire
    from bombardment import Bombardment
    game, add_unit = snapshot_game
    shooter = add_unit('artillery', 1)
    target = add_unit('target', 2)
    target.bodyNP.setY(12)
    game.debugTextInfo = game.diceInfoText = Mock()
    monkeypatch.setattr('battle_secondary.shooting_blocked', lambda *args: False)
    monkeypatch.setattr('magic_items.item_target_protected', lambda *args, **kwargs: False)
    weapon = {'ranged_range': 48, 'ranged_range_min': 6}
    if kind == 'cannon':
        resolver = CannonFire(game)
        resolver.cannon_weapon = lambda unit: weapon
        resolver._place_marker = resolver._draw_path = Mock()
        roll = resolver.roll_artillery = AsyncMock(return_value='Misfire')
        destination = target.bodyNP.getPos()
    else:
        resolver = Bombardment(game)
        resolver.bombardment_weapon = lambda unit: weapon
        resolver._place_template = Mock()
        roll = resolver.roll_scatter_dice = AsyncMock(return_value=('Misfire', 'Hit!'))
        destination = target
    asyncio.run(resolver.fire(shooter, destination))
    assert shooter.hasAttackedThisTurn
    asyncio.run(resolver.fire(shooter, destination))
    assert roll.await_count == 1


def test_charge_policy_requires_support_and_keeps_committed_support(snapshot_game, monkeypatch):
    from ai_policy import charge_candidates
    game, add_unit = snapshot_game
    striker = add_unit('striker', 1)
    support = add_unit('support', 1)
    enemy = add_unit('enemy', 2)
    striker.unit.nmodels = striker.unit.files = 2
    support.unit.nmodels = support.unit.files = 4
    route = SimpleNamespace(distance=2)
    monkeypatch.setattr('impetuous.legal_targets', lambda game, unit: [(enemy, route, 0)])
    assert charge_candidates(game, [striker]) == []
    assert any(candidate.action.unit_name == 'striker' for candidate in charge_candidates(game, [striker, support]))
    game.chargeDeclarations = [SimpleNamespace(charger=support, defender=enemy)]
    assert charge_candidates(game, [striker])


@pytest.mark.parametrize('distance,threshold,expected', [(40, 8, False), (12, 3, False),
                                                       (12, 8, True), (12, 20, False)])
def test_scarce_magic_requires_pressure_and_useful_threshold(snapshot_game, distance, threshold, expected):
    from ai_policy import use_scarce_magic
    from spell_system import restore_spellbook
    game, add_unit = snapshot_game
    wizard = add_unit('wizard', 1)
    restore_spellbook(wizard.unit.model, [], 1)
    enemy = add_unit('enemy', 2)
    enemy.bodyNP.setY(distance)
    game.fsm.spellInstanceToCast = SimpleNamespace(casting_value=threshold)
    assert use_scarce_magic(game, wizard, 'Casting') is expected


def test_ai_retains_actual_shard_inventory_until_useful(snapshot_game):
    from magic_items import install_inventory, magic_roll_bonus
    from spell_system import restore_spellbook
    game, add_unit = snapshot_game
    wizard = add_unit('wizard', 1, model('Mage', ''))
    wizard.game = game
    wizard.unit.roster_metadata = {'roster_selections': [{'ref': 'wizard', 'type': 'unit'}]}
    restore_spellbook(wizard.unit.model, [], 1)
    enemy = add_unit('enemy', 2)
    enemy.bodyNP.setY(40)
    item = install_inventory(wizard, [{'name': 'Wyrdstone Shard', 'category': 'Arcane Items',
        'selection_ref': 'wizard/shard', 'owner_ref': 'wizard'}])[0]
    game.aiControls = lambda member: True
    game.fsm.spellInstanceToCast = SimpleNamespace(casting_value=8)
    assert asyncio.run(magic_roll_bonus(game, wizard, 'Casting')) == 0
    assert not item.uses
    enemy.bodyNP.setY(12)
    assert asyncio.run(magic_roll_bonus(game, wizard, 'Casting')) == 1
    assert item.uses
    assert asyncio.run(magic_roll_bonus(game, wizard, 'Casting')) == 0


def test_loading_waits_for_committed_ai_command(controller, monkeypatch):
    from persistence import load_game_state
    game = controller.game
    game.AIplayer2 = controller
    controller._command_running = True
    read_save = Mock(side_effect=AssertionError('must not read or restore mid-command'))
    monkeypatch.setattr('persistence._read_save_file', read_save)
    load_game_state(game, 'unused.json')
    read_save.assert_not_called()
    assert not hasattr(game, 'battleLoadGeneration')


@pytest.mark.parametrize('player', [1, 2])
@pytest.mark.parametrize('remember_side', [False, True])
def test_snapshot_joined_character_ownership_and_world_transform(
        snapshot_game, player, remember_side):
    game, add_unit = snapshot_game
    host = add_unit('host', player)
    character = add_unit('character', player)
    host.bodyNP.setPos(20, 10, 0)
    host.bodyNP.setH(65)
    character.bodyNP.reparentTo(host.bodyNP)
    character.bodyNP.setPos(1, 2, 0)
    character.bodyNP.setH(15)
    character.hostUnit = host
    if remember_side:
        character._player = player
    (game.player1Units if player == 1 else game.player2Units).remove(character)

    captured = GameState.from_game(game).get_unit_by_name('character')

    assert captured['player'] == player
    assert captured['position'] == pytest.approx(tuple(character.bodyNP.getPos(game.render)))
    assert captured['heading'] == pytest.approx(character.bodyNP.getH(game.render))
    assert captured['host_name'] == 'host'


@pytest.mark.parametrize('phase', ['MovementPhase', 'ShootingPhase', 'CombatPhase'])
def test_joined_character_is_retained_but_not_an_independent_actor(snapshot_game, phase):
    game, add_unit = snapshot_game
    game.fsm.state = phase
    host = add_unit('host', 2)
    character = add_unit('character', 2)
    enemy = add_unit('enemy', 1)
    character.hostUnit = host
    character.isInCombat = True
    character.isInCombatWith = [enemy]
    host.hasMovedThisTurn = host.hasAttackedThisTurn = True
    state = GameState.from_game(game)
    state.get_unit_by_name('character')['ranged'] = True
    state.get_unit_by_name('enemy')['position'] = (0, 20, 0)

    assert [unit['name'] for unit in state.get_player_units(2)] == ['host', 'character']
    assert [unit['name'] for unit in state.get_independent_units(2)] == ['host']
    tree = MinimaxTree(None)
    assert all(action.unit_name != 'character' for action in tree._generate_possible_actions(state))
    optimized = OptimizedMinimaxTree(None)
    assert all(action.unit_name != 'character' for action in optimized._generate_tactical_actions(state))
    ai = EnhancedAI(game, game.player2Units, game.player1Units, use_minimax=False)
    try:
        assert ai._heuristic_decision(state).action_type == 'end_phase'
    finally:
        ai.helper1.ignoreAll()


def test_snapshot_omits_destroyed_and_detached_units(snapshot_game):
    game, add_unit = snapshot_game
    add_unit('living', 1)
    add_unit('dead', 1).unit.nmodels = 0
    add_unit('removed', 2).bodyNP.removeNode()

    assert [unit['name'] for unit in GameState.from_game(game).units] == ['living']


def test_snapshot_rejects_unknown_ownership(snapshot_game):
    game, add_unit = snapshot_game
    add_unit('orphan', 1)
    game.player1Units.clear()

    with pytest.raises(ValueError, match='cannot determine owner of orphan'):
        GameState.from_game(game)


def test_snapshot_uses_mount_movement(snapshot_game):
    game, add_unit = snapshot_game
    rider = model('Silver Helm', '')
    rider.attach_mount(model('Elven Steed', ''))
    add_unit('knight', 2, rider)

    captured = GameState.from_game(game).get_unit_by_name('knight')

    assert rider.get_movement(0) == 9
    assert captured['M'] == captured['ground_movement'] == 9
    assert captured['fly_movement'] == 0
    assert not captured['is_flying']


@pytest.mark.parametrize('mode', ['fly', 'ground'])
def test_snapshot_distinguishes_flight_and_ground_movement(snapshot_game, mode):
    game, add_unit = snapshot_game
    profile = model('Silver Helm', '')
    profile.characteristics['M'] = 5
    profile._base_characteristics['M'] = 5
    profile.special_rules.append({'name': 'Fly (10)', 'fly': True, 'fly_movement': 10})
    profile.flight_mode = mode
    add_unit('flyer', 1, profile)

    captured = GameState.from_game(game).get_unit_by_name('flyer')

    assert captured['ground_movement'] == 5
    assert captured['fly_movement'] == 10
    assert captured['M'] == (10 if mode == 'fly' else 5)
    assert captured['is_flying'] == (mode == 'fly')


def test_snapshot_preserves_zero_movement(snapshot_game):
    game, add_unit = snapshot_game
    member = add_unit('immobile', 1)
    member.unit.model.characteristics['M'] = 0

    assert GameState.from_game(game).get_unit_by_name('immobile')['M'] == 0


@pytest.mark.parametrize('phase,index', [
    ('MovementPhase', 1), ('ReserveMovePhase', 2), ('DeployPhase', 0),
    ('SpellPhase', 0), ('BattleEnded', 3)])
def test_snapshot_preserves_actual_window_and_turn_counters(snapshot_game, phase, index):
    game, add_unit = snapshot_game
    game.fsm.state = phase
    game.fsm.currentPhaseIndex = index

    state = GameState.from_game(game)
    copied = state.clone()

    assert state.current_phase == copied.current_phase == phase
    assert state.current_phase_index == index
    assert state.charge_stage == copied.charge_stage == 'declarations'
    assert state.rounds_completed == copied.rounds_completed == (1, 0)
    assert state.current_round == 0
    game.roundCounter.currentRoundPlayer[0] = 4
    assert state.rounds_completed == (1, 0)


def test_snapshot_and_clone_do_not_share_live_combat_lists(snapshot_game):
    game, add_unit = snapshot_game
    member = add_unit('friendly', 1)
    enemy = add_unit('enemy', 2)
    member.isInCombatWith = [enemy]
    member.isInCombatFlank = ['flank']
    state = GameState.from_game(game)
    state.score = 123
    copied = state.clone()
    captured = state.get_unit_by_name('friendly')
    simulated = copied.get_unit_by_name('friendly')
    simulated['isInCombatWith'].clear()
    simulated['isInCombatFlank'].append('rear')
    simulated['position'] = (50, 50, 0)
    simulated['nmodels'] = 1

    assert captured['isInCombatWith'] == ['enemy']
    assert captured['isInCombatFlank'] == ['flank']
    assert captured['position'] == tuple(member.bodyNP.getPos(game.render))
    assert captured['nmodels'] == member.unit.nmodels == 5
    assert member.isInCombatWith == [enemy]
    assert member.isInCombatFlank == ['flank']
    assert copied.score is None


def test_snapshot_refreshes_classification_after_profile_change(snapshot_game):
    game, add_unit = snapshot_game
    member = add_unit('changing-profile', 1)
    member.unit.model.characteristics.update(WS=3, S=3, T=3, A=1, Ld=7, W=1)
    first = GameState.from_game(game).get_unit_by_name('changing-profile')
    member.unit.model.attach_mount(model('Elven Steed', ''))
    member.unit.model.characteristics.update(WS=6, S=6, A=5)

    second = GameState.from_game(game).get_unit_by_name('changing-profile')

    assert first['unit_type'] == 'basic'
    assert second['unit_type'] == 'hammer'
    assert second['support_role'] == 'fast'


@pytest.fixture
def controller(snapshot_game):
    game, add_unit = snapshot_game
    game.fsm.state = 'StrategyPhase'
    game.fsm.currentPhaseIndex = 0
    game.fsm.nextPhase = Mock()
    game.chargeStage = None
    game.save_game_state = Mock()
    game.analyzer = SimpleNamespace(get_strategy_report=Mock(return_value='AI test report'))
    ai = EnhancedAI(game, game.player2Units, game.player1Units, use_minimax=False)
    yield ai
    ai.helper1.ignoreAll()


@pytest.mark.parametrize('player', [1, 2])
def test_controller_heuristic_only_completes_phase(controller, player):
    controller.player_num = player
    controller.game.roundCounter.current_player = player

    action = asyncio.run(controller.take_turn())

    assert action.action_type == 'end_phase'
    controller.game.fsm.nextPhase.assert_called_once()
    assert not controller._turn_running


@pytest.mark.parametrize('phase', ['StrategyPhase', 'MovementPhase', 'ShootingPhase', 'CombatPhase', 'DeployPhase'])
def test_controller_does_not_act_on_opponents_turn(controller, phase):
    controller.game.fsm.state = phase
    controller.game.roundCounter.current_player = 1
    controller.make_decision = AsyncMock(return_value=GameAction('end_phase', 'system'))
    controller.execute_action = AsyncMock()
    controller.deployUnits = Mock()

    assert asyncio.run(controller.take_turn()) is None

    controller.make_decision.assert_not_awaited()
    controller.execute_action.assert_not_awaited()
    controller.deployUnits.assert_not_called()
    controller.game.save_game_state.assert_not_called()
    controller.game.fsm.nextPhase.assert_not_called()


@pytest.mark.parametrize('phase', ['SpellPhase', 'MakeChoice', 'BattleEnded', 'CampaignPhase'])
def test_controller_does_not_drive_special_windows(controller, phase):
    controller.game.fsm.state = phase
    controller.make_decision = AsyncMock(return_value=GameAction('end_phase', 'system'))

    assert asyncio.run(controller.take_turn()) is None

    controller.make_decision.assert_not_awaited()
    controller.game.save_game_state.assert_not_called()
    controller.game.fsm.nextPhase.assert_not_called()


def test_controller_missing_decision_stops_without_advancing(controller):
    controller.make_decision = AsyncMock(return_value=None)
    controller.execute_action = AsyncMock()

    assert asyncio.run(controller.take_turn()) is None

    controller.execute_action.assert_not_awaited()
    controller.game.fsm.nextPhase.assert_not_called()
    assert not controller._turn_running


@pytest.mark.parametrize('failure', ['decision', 'execution', 'advance'])
@pytest.mark.parametrize('error_type', [RuntimeError, asyncio.CancelledError])
def test_controller_failure_always_releases_running_lock(controller, failure, error_type):
    controller.make_decision = AsyncMock(return_value=GameAction('end_phase', 'system'))
    controller.execute_action = AsyncMock(return_value=ActionOutcome('completed'))
    failing = {'decision': controller.make_decision, 'execution': controller.execute_action,
               'advance': controller.game.fsm.nextPhase}[failure]
    failing.side_effect = error_type('injected failure')

    with pytest.raises(error_type, match='injected failure'):
        asyncio.run(controller.take_turn())

    assert not controller._turn_running
    failing.side_effect = None
    assert asyncio.run(controller.take_turn()).action_type == 'end_phase'


@pytest.mark.parametrize('change', ['phase', 'player', 'round', 'charge-stage', 'reload', 'disabled'])
def test_controller_discards_decision_after_context_change(controller, change):
    game = controller.game

    async def decide():
        if change == 'phase':
            game.fsm.state = 'MovementPhase'
        elif change == 'player':
            game.roundCounter.current_player = 1
        elif change == 'round':
            game.roundCounter.currentRoundPlayer[1] += 1
        elif change == 'charge-stage':
            game.chargeStage = 'resolving'
        elif change == 'reload':
            game.battleLoadGeneration = 1
        else:
            controller.active = False
        return GameAction('end_phase', 'system')

    controller.make_decision = AsyncMock(side_effect=decide)
    controller.execute_action = AsyncMock()

    assert asyncio.run(controller.take_turn()) is None

    controller.execute_action.assert_not_awaited()
    game.fsm.nextPhase.assert_not_called()
    assert not controller._turn_running


def test_controller_does_not_advance_replacement_phase_after_execution(controller):
    controller.make_decision = AsyncMock(return_value=GameAction('end_phase', 'system'))

    async def execute(action):
        controller.game.fsm.state = 'ShootingPhase'

    controller.execute_action = AsyncMock(side_effect=execute)

    assert asyncio.run(controller.take_turn()) is None

    controller.game.fsm.nextPhase.assert_not_called()
    assert not controller._turn_running


@pytest.mark.parametrize('busy', ['awaitingChoice', 'magicBusy', 'resolvingCombat', 'restoringBattle'])
def test_controller_waits_for_existing_resolution(controller, busy):
    setattr(controller.game, busy, True)
    controller.make_decision = AsyncMock(return_value=GameAction('end_phase', 'system'))

    assert asyncio.run(controller.take_turn()) is None

    controller.make_decision.assert_not_awaited()
    controller.game.fsm.nextPhase.assert_not_called()


def test_controller_reentrant_call_does_not_clear_existing_lock(controller):
    controller._turn_running = True
    controller.make_decision = AsyncMock()

    assert asyncio.run(controller.take_turn()) is None

    controller.make_decision.assert_not_awaited()
    assert controller._turn_running


def test_executor_awaits_its_movement_command(controller, snapshot_game):
    game, add_unit = snapshot_game
    member = add_unit('mover', 2)
    game.fsm.state = 'MovementPhase'
    game.chargeStage = 'remaining'
    game.pathTowardsMouse = Mock()

    async def move(unit, *, wait_for_completion=False):
        assert wait_for_completion
        assert unit is member
        unit.hasMovedThisTurn = True

    game.moveUnit = AsyncMock(side_effect=move)
    outcome = asyncio.run(controller.execute_action(
        GameAction('move', 'mover', {'target_x': 5, 'target_y': 6})))

    assert outcome.status == 'completed'
    game.moveUnit.assert_awaited_once_with(member, wait_for_completion=True)
    game.pathTowardsMouse.assert_called_once_with(member, 5, 6)


@pytest.mark.parametrize('error_type', [RuntimeError, asyncio.CancelledError])
def test_executor_clears_command_lock_on_highlight_failure(controller, snapshot_game, error_type):
    game, add_unit = snapshot_game
    add_unit('mover', 2)
    game.fsm.state = 'MovementPhase'
    game.chargeStage = 'remaining'
    controller._highlight_acting_unit = Mock(side_effect=error_type('highlight failure'))
    controller._unhighlight_acting_unit = Mock()
    game.moveUnit = Mock()

    with pytest.raises(error_type, match='highlight failure'):
        asyncio.run(controller.execute_action(
            GameAction('move', 'mover', {'target_x': 5, 'target_y': 6})))

    assert not controller._command_running
    controller._unhighlight_acting_unit.assert_called_once()
    game.moveUnit.assert_not_called()


def test_executor_rejected_move_does_not_spend_allowance(controller, snapshot_game):
    game, add_unit = snapshot_game
    member = add_unit('mover', 2)
    member.request = Mock()
    game.fsm.state = 'MovementPhase'
    game.chargeStage = 'remaining'
    game.pathTowardsMouse = Mock()
    game.moveUnit = Mock(return_value=None)

    outcome = asyncio.run(controller.execute_action(
        GameAction('move', 'mover', {'target_x': 5, 'target_y': 6})))

    assert outcome.status == 'rejected'
    assert not member.hasMovedThisTurn
    member.request.assert_not_called()


@pytest.mark.parametrize('accepted', [False, True])
def test_executor_shooting_reports_real_outcome(controller, snapshot_game, accepted):
    game, add_unit = snapshot_game
    member = add_unit('shooter', 2)
    add_unit('target', 1)
    game.fsm.state = 'ShootingPhase'

    async def shoot(attacker, defender):
        if accepted:
            attacker.hasAttackedThisTurn = True
        return accepted

    game.shootAt = AsyncMock(side_effect=shoot)
    outcome = asyncio.run(controller.execute_action(
        GameAction('shoot', 'shooter', {'target': 'target'})))

    assert outcome.status == ('completed' if accepted else 'rejected')
    assert member.hasAttackedThisTurn == accepted
    game.shootAt.assert_awaited_once()


def test_executor_rejects_wrong_owner_without_committing(controller, snapshot_game):
    game, add_unit = snapshot_game
    add_unit('opponent', 1)
    game.fsm.state = 'MovementPhase'
    game.moveUnit = Mock()
    game.pathTowardsMouse = Mock()

    outcome = asyncio.run(controller.execute_action(
        GameAction('move', 'opponent', {'target_x': 5, 'target_y': 6})))

    assert outcome.status == 'rejected'
    game.moveUnit.assert_not_called()
    game.pathTowardsMouse.assert_not_called()


def test_autoplay_advances_only_owned_enabled_windows(controller):
    controller.take_turn = AsyncMock()
    controller.game.roundCounter.current_player = 1
    assert asyncio.run(controller.autoplay_step()) is False
    controller.take_turn.assert_not_awaited()
    controller.game.roundCounter.current_player = 2
    controller.automatic = False
    assert asyncio.run(controller.autoplay_step()) is False
    controller.take_turn.assert_not_awaited()
    controller.automatic = True
    assert asyncio.run(controller.autoplay_step()) is True
    controller.take_turn.assert_awaited_once()


def test_autoplay_pauses_on_stall_without_skipping(controller):
    controller.take_turn = AsyncMock()
    for attempt in range(3):
        assert asyncio.run(controller.autoplay_step()) is True
    assert not controller.active
    assert asyncio.run(controller.autoplay_step()) is False
    controller.game.fsm.nextPhase.assert_not_called()


def test_autoplay_exception_pauses_and_preserves_phase(controller):
    controller.take_turn = AsyncMock(side_effect=ValueError('broken command'))
    with pytest.raises(ValueError, match='broken command'):
        asyncio.run(controller.autoplay_step())
    assert not controller.active
    controller.game.fsm.nextPhase.assert_not_called()


def test_failed_command_pauses_without_advancing_or_retrying(controller):
    controller.make_decision = AsyncMock(return_value=GameAction('attack', 'unit'))
    controller.execute_action = AsyncMock(return_value=ActionOutcome('failed', 'combat resolver error'))
    asyncio.run(controller.take_turn())
    assert not controller.active
    assert controller.pause_reason == 'combat resolver error'
    assert controller.execute_action.await_count == 1
    controller.game.fsm.nextPhase.assert_not_called()


def test_spell_construction_is_shared_without_spending_attempt(snapshot_game):
    from spell_system import build_spell
    game, add_unit = snapshot_game
    caster = add_unit('wizard', 1, model('Mage', ''))
    caster.spellsCastThisTurn = []
    caster.unit.model.spells = {'Fireball': {'name': 'Fireball', 'casting_value': 8,
                                          'range': 24, 'phase': 'shooting', 'type': 'Magic Missile'}}
    game.fsm.endOfTurnSpells = []

    spell = build_spell(game, caster, 'Fireball', allow_catalogue=False)

    assert spell.caster is caster
    assert spell.selection_key == 'Fireball'
    assert spell.casting_value == 8
    assert spell.spell_range == 24
    assert caster.spellsCastThisTurn == []
    assert build_spell(game, caster, 'missing', allow_catalogue=False) is None