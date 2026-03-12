"""
Game state persistence — save and load functionality.

All functions accept the game (MyApp) instance as their first argument
so they can be imported and called without subclassing.
"""

import json
from datetime import datetime


def save_game_state(game, filename=None):
    """
    Serialize the current game state to a JSON file.

    Args:
        game: The MyApp game instance.
        filename: Target filename. Auto-generated with timestamp if omitted.

    Returns:
        The filename that was written.
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"savegame_{timestamp}.json"

    game_state = {
        'current_phase': game.fsm.phases[game.fsm.currentPhaseIndex],
        'current_phase_index': game.fsm.currentPhaseIndex,
        'current_round': game.roundCounter.currentRoundPlayer,
        'current_player': game.roundCounter.current_player,
        'max_rounds': game.roundCounter.max_rounds,
        'ai_player2_active': game.AIplayer2.active,
        'units': [],
    }

    for unit in game.units:
        unit_data = {
            'name': unit.unitName,
            'position': list(unit.bodyNP.getPos()),
            'heading': unit.bodyNP.getH(),
            'pitch': unit.bodyNP.getP(),
            'roll': unit.bodyNP.getR(),
            'state': unit.state,
            'color': list(unit.color),
            'isInCombat': unit.isInCombat,
            'hasMovedThisTurn': unit.hasMovedThisTurn,
            'hasAttackedThisTurn': unit.hasAttackedThisTurn,
            'attemptedRallyThisTurn': unit.attemptedRallyThisTurn,
            'isDeployed': unit.isDeployed,
            'nmodels': unit.unit.nmodels,
            'files': unit.unit.files,
            'ranks': unit.unit.ranks,
            'points_cost': unit.unit.model.characteristics.get('Points', 0) * unit.unit.nmodels,
            'characteristics': unit.unit.model.characteristics,
            'armor_save': unit.unit.model.armor_save,
            'charging': unit.unit.model.charging,
            'player': 1 if unit in game.player1Units else 2,
            'isInCombatWith': [u.unitName for u in unit.isInCombatWith],
            'isInCombatFlank': unit.isInCombatFlank,
        }

        if unit.unit.model.equipedWeapon:
            unit_data['equipped_weapon'] = unit.unit.model.equipedWeapon['name']
        else:
            unit_data['equipped_weapon'] = None

        game_state['units'].append(unit_data)

    with open(filename, 'w') as f:
        json.dump(game_state, f, indent=2)

    print(f"Game saved to {filename}")
    return filename


def load_game_state(game, filename):
    """
    Restore game state from a JSON save file.

    Args:
        game: The MyApp game instance.
        filename: Path to the save file.
    """
    with open(filename, 'r') as f:
        game_state = json.load(f)

    # Restore FSM state
    game.fsm.currentPhaseIndex = game_state['current_phase_index']
    game.fsm.request(game_state['current_phase'])

    # Restore round counter
    game.roundCounter.currentRoundPlayer = game_state['current_round']
    game.roundCounter.current_player = game_state['current_player']
    if game.roundCounter.current_player == 1:
        game.roundCounter.enterPlayerOne()
    else:
        game.roundCounter.enterPlayerTwo()
    game.roundCounter.max_rounds = game_state['max_rounds']
    game.roundCounter.update_round_display()

    # Restore AI settings
    game.AIplayer2.active = game_state['ai_player2_active']

    unit_map = {unit.unitName: unit for unit in game.units}

    # First pass: restore individual unit state
    for unit_data in game_state['units']:
        unit_name = unit_data['name']
        if unit_name not in unit_map:
            continue

        unit = unit_map[unit_name]

        unit.bodyNP.setPos(*unit_data['position'])
        unit.bodyNP.setH(unit_data['heading'])
        unit.bodyNP.setP(unit_data['pitch'])
        unit.bodyNP.setR(unit_data['roll'])

        unit.request(unit_data['state'])

        unit.isInCombat = unit_data['isInCombat']
        unit.hasMovedThisTurn = unit_data['hasMovedThisTurn']
        unit.hasAttackedThisTurn = unit_data['hasAttackedThisTurn']
        unit.attemptedRallyThisTurn = unit_data['attemptedRallyThisTurn']
        unit.isDeployed = unit_data['isDeployed']

        unit.unit.nmodels = unit_data['nmodels']
        unit.unit.files = unit_data['files']
        unit.unit.ranks = unit_data['ranks']

        unit.unit.model.characteristics = unit_data['characteristics']
        unit.unit.model.armor_save = unit_data['armor_save']
        unit.unit.model.charging = unit_data['charging']

        if unit_data['equipped_weapon']:
            unit.unit.model.equip_weapon(unit_data['equipped_weapon'])

        unit.isInCombatWith = []
        unit.isInCombatFlank = []

    # Second pass: restore combat relationships
    for unit_data in game_state['units']:
        unit_name = unit_data['name']
        if unit_name not in unit_map:
            continue

        unit = unit_map[unit_name]
        for combat_unit_name in unit_data['isInCombatWith']:
            if combat_unit_name in unit_map:
                unit.isInCombatWith.append(unit_map[combat_unit_name])
        unit.isInCombatFlank = unit_data['isInCombatFlank']
        unit.updateTextNode()

    print(f"Game loaded from {filename}")
    game.debugTextUnit.setText(f"Loaded: {filename}")

    # Print analysis for both players
    for player_num in (1, 2):
        evaluation = game.analyzer.evaluate_overall_state(player_num=player_num)
        print(f"Player {player_num} Assessment: {evaluation['assessment']}")
        print(f"Total Score: {evaluation['total_score']:.1f}")
        strategy = game.analyzer.suggest_strategy(player_num=player_num)
        print(f"Suggested Strategy: {strategy}")
