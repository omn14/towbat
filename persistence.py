"""
Game state persistence — save and load functionality.

All functions accept the game (MyApp) instance as their first argument
so they can be imported and called without subclassing.
"""

import json
import os
import shutil
from datetime import datetime


def _clean_weapon(weapon):
    """Return a JSON-safe copy of a weapon dict, dropping coded (callable) rules."""
    safe = {}
    for key, value in weapon.items():
        if callable(value):
            continue
        try:
            json.dumps(value)
        except TypeError:
            continue
        safe[key] = value
    return safe


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
            # Enough to reconstruct the unit on load if it is missing.
            'model_name': unit.unit.model.name,
            'weapons': [_clean_weapon(w) for w in unit.unit.model.weapons.values()],
            'mount': (unit.unit.model.get_mount().name
                      if unit.unit.model.is_mounted() else None),
        }

        if unit.unit.model.equipedWeapon:
            unit_data['equipped_weapon'] = unit.unit.model.equipedWeapon['name']
        else:
            unit_data['equipped_weapon'] = None

        game_state['units'].append(unit_data)

    # Serialize fully before touching disk so a failure here can never
    # corrupt the target file (the crash that prompted this safeguard).
    payload = json.dumps(game_state, indent=2)

    # Keep a backup of the last good save to fall back to on corruption.
    if os.path.exists(filename):
        try:
            shutil.copyfile(filename, filename + '.bak')
        except OSError as exc:
            print(f"[persistence] could not back up {filename}: {exc}")

    # Atomic write: write a temp file, flush to disk, then rename into place.
    tmp = filename + '.tmp'
    with open(tmp, 'w') as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, filename)

    print(f"Game saved to {filename}")
    return filename


def _read_save_file(path):
    """Load and validate a save file. Returns the dict, or None if unusable."""
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"[persistence] save file '{path}' is corrupted: {exc}")
        return None
    # Basic structural validation so a truncated/partial file is rejected.
    if not isinstance(data, dict) or 'units' not in data or 'current_phase' not in data:
        print(f"[persistence] save file '{path}' is missing required data.")
        return None
    return data


def load_game_state(game, filename):
    """
    Restore game state from a JSON save file.

    Args:
        game: The MyApp game instance.
        filename: Path to the save file.
    """
    # Fall back to the last-good backup if the main file is corrupt/missing.
    game_state = _read_save_file(filename)
    if game_state is None:
        game_state = _read_save_file(filename + '.bak')
        if game_state is not None:
            print(f"[persistence] loaded backup save '{filename}.bak' instead.")

    if game_state is None:
        message = f"Load failed: '{filename}' is corrupted or missing."
        print(f"[persistence] {message}")
        if getattr(game, 'debugTextUnit', None):
            game.debugTextUnit.setText(message)
        return

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

    # Recreate any saved units that are missing from the current scene (e.g. a
    # cannon added after the initial army load) so a load fully restores them.
    existing_names = {unit.unitName for unit in game.units}
    for unit_data in game_state['units']:
        if unit_data['name'] in existing_names:
            continue
        chars = unit_data.get('characteristics', {})
        # Resolve the base model name (older saves lack 'model_name').
        base_name = (unit_data.get('model_name') or chars.get('Model')
                     or chars.get('Unit') or unit_data['name'])
        spec = {
            'name': base_name,
            'nmodels': unit_data['nmodels'],
            'files': unit_data['files'],
            'ranks': unit_data['ranks'],
            'mount': unit_data.get('mount'),
            'weapons': unit_data.get('weapons', []),
        }
        game._create_unit(spec, unit_data.get('player', 1), unit_data['name'])

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
