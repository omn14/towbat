"""
Game state persistence — save and load functionality.

All functions accept the game (MyApp) instance as their first argument
so they can be imported and called without subclassing.
"""

import json
import os
import shutil
from datetime import datetime

from characters import join_unit
from spell_system import load_spells, save_spells


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
        'current_phase': (game.fsm.getCurrentOrNextState()
                          or game.fsm.phases[game.fsm.currentPhaseIndex]),
        'current_phase_index': game.fsm.currentPhaseIndex,
        'current_round': game.roundCounter.currentRoundPlayer,
        'current_player': game.roundCounter.current_player,
        'max_rounds': game.roundCounter.max_rounds,
        'ai_player2_active': game.AIplayer2.active,
        'spells_in_play': save_spells(game),
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
            'chargedThisTurn': getattr(unit, 'chargedThisTurn', False),
            'countsAsChargedNextTurn': getattr(unit, 'countsAsChargedNextTurn', False),
            'chargeDistance': getattr(unit, 'chargeDistance', 0.0),
            'cannotChargeThisTurn': getattr(unit, 'cannotChargeThisTurn', False),
            'panicTestedThisPhase': getattr(unit, 'panicTestedThisPhase', False),
            'fledThisPhase': getattr(unit, 'fledThisPhase', False),
            'usedStubborn': getattr(unit, 'usedStubborn', False),
            'spellsCastThisTurn': list(getattr(unit, 'spellsCastThisTurn', [])),
            'cannotCastThisTurn': getattr(unit, 'cannotCastThisTurn', False),
            'isDisrupted': getattr(unit, 'isDisrupted', False),
            'isGeneral': getattr(unit, 'isGeneral', False),
            'isBSB': getattr(unit, 'isBSB', False),
            'woundsOnModel': getattr(unit, 'woundsOnModel', 0),
            'startOfBattleModels': getattr(unit, 'startOfBattleModels', unit.unit.nmodels),
            'startOfPhaseModels': getattr(unit, 'startOfPhaseModels', unit.unit.nmodels),
            'isDeployed': unit.isDeployed,
            'nmodels': unit.unit.nmodels,
            'files': unit.unit.files,
            'ranks': unit.unit.ranks,
            'points_cost': unit.unit.model.characteristics.get('Points', 0) * unit.unit.nmodels,
            'characteristics': unit.unit.model.characteristics,
            'armor_save': unit.unit.model.armor_save,
            'armour': list(getattr(unit.unit.model, 'armour', []) or []),
            'charging': unit.unit.model.charging,
            'player': getattr(unit, '_player', 1 if unit in game.player1Units else 2),
            'isInCombatWith': [u.unitName for u in unit.isInCombatWith],
            'isInCombatFlank': unit.isInCombatFlank,
            # Enough to reconstruct the unit on load if it is missing.
            'model_name': unit.unit.model.name,
            'weapons': [_clean_weapon(w) for w in unit.unit.model.weapons.values()],
            'spells': [_clean_weapon(s) for s in
                       getattr(unit.unit.model, 'spells', {}).values()],
            'wizard_level': unit.unit.model.wizard_level(0),
            'mount': (unit.unit.model.get_mount().name
                      if unit.unit.model.is_mounted() else None),
            # Character joined to this unit's front rank, if any.
            'joined_character': (unit.joinedCharacter.unitName
                                 if getattr(unit, 'joinedCharacter', None) else None),
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

    # Remove any current units that aren't in the save (e.g. units destroyed
    # after this save was taken) so a load reflects the saved roster exactly.
    saved_names = {unit_data['name'] for unit_data in game_state['units']}
    for unit in list(game.units):
        if unit.unitName in saved_names:
            continue
        try:
            game.world.removeRigidBody(unit.bodyNP.node())
        except Exception:
            pass
        try:
            unit.bodyNP.removeNode()
        except Exception:
            pass
        try:
            unit.model.removeNode()
        except Exception:
            pass
        game.units.remove(unit)
        if unit in game.player1Units:
            game.player1Units.remove(unit)
        if unit in game.player2Units:
            game.player2Units.remove(unit)

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
            'spells': unit_data.get('spells', []),
            'wizard_level': unit_data.get('wizard_level'),
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
        unit.chargedThisTurn = unit_data.get('chargedThisTurn', False)
        unit.countsAsChargedNextTurn = unit_data.get('countsAsChargedNextTurn', False)
        unit.chargeDistance = unit_data.get('chargeDistance', 0.0)
        unit.cannotChargeThisTurn = unit_data.get('cannotChargeThisTurn', False)
        unit.panicTestedThisPhase = unit_data.get('panicTestedThisPhase', False)
        unit.fledThisPhase = unit_data.get('fledThisPhase', False)
        unit.usedStubborn = unit_data.get('usedStubborn', False)
        # A spell attempted after the save was taken has not been attempted in
        # the state being loaded, so the allowance has to come back with it.
        unit.spellsCastThisTurn = list(unit_data.get('spellsCastThisTurn', []))
        unit.cannotCastThisTurn = unit_data.get('cannotCastThisTurn', False)
        unit.isDisrupted = unit_data.get('isDisrupted', False)
        # Saves written before the General was tracked keep the load-time nomination.
        unit.isGeneral = unit_data.get('isGeneral', getattr(unit, 'isGeneral', False))
        unit.isBSB = unit_data.get('isBSB', getattr(unit, 'isBSB', False))
        unit.woundsOnModel = unit_data.get('woundsOnModel', 0)
        unit.startOfBattleModels = unit_data.get('startOfBattleModels', unit.unit.nmodels)
        unit.startOfPhaseModels = unit_data.get('startOfPhaseModels', unit.unit.nmodels)
        unit.isDeployed = unit_data['isDeployed']

        unit.unit.nmodels = unit_data['nmodels']
        unit.unit.files = unit_data['files']
        unit.unit.ranks = unit_data['ranks']

        unit.unit.model.characteristics = unit_data['characteristics']
        unit.unit.model.armor_save = unit_data['armor_save']
        unit.unit.model.armour = list(unit_data.get('armour', []) or [])
        unit.unit.model.charging = unit_data['charging']

        if unit_data['equipped_weapon']:
            unit.unit.model.equip_weapon(unit_data['equipped_weapon'])

        # A coded spell keeps its class; saves only carry the data.
        for spell in unit_data.get('spells') or []:
            known = unit.unit.model.spells.get(spell['name'])
            if known is None:
                unit.unit.model.spells[spell['name']] = dict(spell)
            else:
                for key, value in spell.items():
                    known.setdefault(key, value)

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

    # Third pass: re-join characters to their host units.
    for unit_data in game_state['units']:
        char_name = unit_data.get('joined_character')
        host = unit_map.get(unit_data['name'])
        character = unit_map.get(char_name) if char_name else None
        if host is not None and character is not None:
            join_unit(game, character, host)

    # Spells still in play: a hex, a ward or a vortex outlives the turn it was
    # cast in, so it has to come back or the save silently ends it.
    for spell in list(game.fsm.endOfTurnSpells):
        spell.endSpell()
    game.fsm.endOfTurnSpells = []
    for spell in list(getattr(game, 'remainsInPlay', [])):
        spell.endSpell()
    game.remainsInPlay = []
    load_spells(game, game_state.get('spells_in_play'), unit_map)

    # Each model sits on the terrain surface, not at its unit's own Z. That
    # offset is derived rather than saved, so a unit restored onto a hill would
    # otherwise stand at ground level, inside the hill.
    for unit in game.units:
        if not unit.model.isEmpty():
            game.movement.alignModelsToHillNormal(unit)

    print(f"Game loaded from {filename}")
    game.debugTextUnit.setText(f"Loaded: {filename}")

    # Print analysis for both players
    for player_num in (1, 2):
        evaluation = game.analyzer.evaluate_overall_state(player_num=player_num)
        print(f"Player {player_num} Assessment: {evaluation['assessment']}")
        print(f"Total Score: {evaluation['total_score']:.1f}")
        strategy = game.analyzer.suggest_strategy(player_num=player_num)
        print(f"Suggested Strategy: {strategy}")
