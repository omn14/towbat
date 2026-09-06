"""
Game state persistence — save and load functionality.

All functions accept the game (MyApp) instance as their first argument
so they can be imported and called without subclassing.
"""

import copy
import json
import os
import shutil
from datetime import datetime

from challenges import Challenge
from characters import detach_character, join_unit
from models import model as Model
from spell_system import load_spells, save_spells, restore_spellbook


# ── Where saves live ──────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(_HERE, 'saves')


def save_path(filename) -> str:
    """Where a save by that name lives.

    A bare name goes in saves/. A name that already carries a directory is
    taken as given, so a caller that knows its own path is not second-guessed.
    """
    if os.path.isabs(filename) or os.path.dirname(filename):
        return filename
    return os.path.join(SAVE_DIR, filename)


def list_saves() -> list:
    """Save file names in saves/, most recently written first.

    `.bak` and `.tmp` companions are left out: neither is a save the player
    chose to make.
    """
    try:
        names = [n for n in os.listdir(SAVE_DIR) if n.endswith('.json')]
    except OSError:
        return []
    return sorted(names, reverse=True,
                  key=lambda n: os.path.getmtime(os.path.join(SAVE_DIR, n)))


def save_label(name) -> str:
    """A save's name and when it was written, for the load menu."""
    stem = name[:-5] if name.endswith('.json') else name
    try:
        when = datetime.fromtimestamp(os.path.getmtime(save_path(name)))
    except OSError:
        return stem
    return f"{stem}   {when.strftime('%d %b %H:%M')}"


# ── Interface preferences ─────────────────────────────────────────────
# Kept apart from save games: these follow the player, not the battle.
SETTINGS_FILE = os.path.join(_HERE, 'settings.json')


def load_settings() -> dict:
    """Interface preferences, or an empty dict if there are none yet."""
    try:
        with open(SETTINGS_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_setting(key, value) -> None:
    """Update one preference, leaving the rest of the file alone."""
    data = load_settings()
    data[key] = value
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(data, f, indent=4)
    except OSError as exc:
        # A preference that will not save must not stop the game.
        print(f"[Settings] could not write {SETTINGS_FILE}: {exc}")


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
    filename = save_path(filename)

    game_state = {
        'current_phase': (game.fsm.getCurrentOrNextState()
                          or game.fsm.phases[game.fsm.currentPhaseIndex]),
        'current_phase_index': game.fsm.currentPhaseIndex,
        'current_round': game.roundCounter.currentRoundPlayer,
        'current_player': game.roundCounter.current_player,
        'max_rounds': game.roundCounter.max_rounds,
        'ai_player2_active': game.AIplayer2.active,
        'spells_in_play': save_spells(game),
        # A challenge outlives the turn it was issued in (To The Death!, p. 211).
        'challenges': [
            {'challenger': c.challenger.unitName if c.challenger else None,
             'host': c.host.unitName if c.host else None,
             'accepter': c.accepter.unitName if c.accepter else None,
             'accepter_host': c.accepter_host.unitName if c.accepter_host else None,
             'refused': c.refused,
             'rounds': c.rounds}
            for c in (getattr(game, 'challenges', None) or [])],
        'units': [],
    }

    for unit in game.units:
        unit_data = {
            'name': unit.unitName,
            # The army list's rules, not the catalogue's: Skirmishers, Fire &
            # Flee and the rest live on the roster, so a unit rebuilt from a
            # save has no way to find them again.
            'special_rules': list(
                unit.unit.model.characteristics.get('Special Rules') or []),
            'position': list(unit.bodyNP.getPos()),
            'heading': unit.bodyNP.getH(),
            'pitch': unit.bodyNP.getP(),
            'roll': unit.bodyNP.getR(),
            'state': unit.state,
            'color': list(unit.color),
            'isInCombat': unit.isInCombat,
            'hasMovedThisTurn': unit.hasMovedThisTurn,
            'marchedThisTurn': getattr(unit, 'marchedThisTurn', False),
            'hasAttackedThisTurn': unit.hasAttackedThisTurn,
            'standAndShootWounds': getattr(unit, 'standAndShootWounds', 0),
            'attemptedRallyThisTurn': unit.attemptedRallyThisTurn,
            'chargedThisTurn': getattr(unit, 'chargedThisTurn', False),
            'countsAsChargedNextTurn': getattr(unit, 'countsAsChargedNextTurn', False),
            'chargeDistance': getattr(unit, 'chargeDistance', 0.0),
            'cannotChargeThisTurn': getattr(unit, 'cannotChargeThisTurn', False),
            'cannotPursueThisTurn': getattr(unit, 'cannotPursueThisTurn', False),
            # The manoeuvre allowance: without these a reload refunds the half
            # Movement a manoeuvre cost and lifts the one-per-move limit.
            'moveSpentThisTurn': getattr(unit, 'moveSpentThisTurn', 0.0),
            'manoeuvreThisTurn': getattr(unit, 'manoeuvreThisTurn', None),
            'redressDelta': getattr(unit, 'redressDelta', 0),
            'panicTestedThisPhase': getattr(unit, 'panicTestedThisPhase', False),
            'fledThisPhase': getattr(unit, 'fledThisPhase', False),
            'usedStubborn': getattr(unit, 'usedStubborn', False),
            'spellsCastThisTurn': list(getattr(unit, 'spellsCastThisTurn', [])),
            'boundSpellPhases': list(getattr(unit, 'boundSpellPhases', [])),
            'cannotCastThisTurn': getattr(unit, 'cannotCastThisTurn', False),
            'isDisrupted': getattr(unit, 'isDisrupted', False),
            'isGeneral': getattr(unit, 'isGeneral', False),
            'isBSB': getattr(unit, 'isBSB', False),
            'woundsOnModel': getattr(unit, 'woundsOnModel', 0),
            'startOfBattleModels': getattr(unit, 'startOfBattleModels', unit.unit.nmodels),
            'startOfPhaseModels': getattr(unit, 'startOfPhaseModels', unit.unit.nmodels),
            'startOfPhaseEngaged': getattr(unit, 'startOfPhaseEngaged', False),
            'roundsFought': getattr(unit, 'roundsFought', 0),
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
            'mount_special_rules': (list(unit.unit.model.get_mount().characteristics.get('Special Rules') or [])
                                    if unit.unit.model.is_mounted() else []),
            # Character joined to this unit's front rank, if any.
            'joined_character': (unit.joinedCharacter.unitName
                                 if getattr(unit, 'joinedCharacter', None) else None),
            'retiredFromCombat': bool(getattr(unit, 'retiredFromCombat', False)),
        }

        if unit.unit.model.equipedWeapon:
            unit_data['equipped_weapon'] = unit.unit.model.equipedWeapon['name']
        else:
            unit_data['equipped_weapon'] = None

        game_state['units'].append(unit_data)

    # Serialize fully before touching disk so a failure here can never
    # corrupt the target file (the crash that prompted this safeguard).
    payload = json.dumps(game_state, indent=2)

    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)

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
        filename: Name of a save in saves/, or a path to one.
    """
    path = save_path(filename)
    # A save left behind in the old location still loads.
    if not os.path.exists(path) and os.path.exists(filename):
        path = filename
    # Fall back to the last-good backup if the main file is corrupt/missing.
    game_state = _read_save_file(path)
    if game_state is None:
        game_state = _read_save_file(path + '.bak')
        if game_state is not None:
            print(f"[persistence] loaded backup save '{path}.bak' instead.")

    if game_state is None:
        message = f"Load failed: '{filename}' is corrupted or missing."
        print(f"[persistence] {message}")
        messenger.send('hud-log', [message, 'morale'])
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
    # Unparent first: deleting an old host must not delete a character the save keeps.
    for host in list(game.units):
        character = getattr(host, 'joinedCharacter', None)
        if character is not None:
            character.bodyNP.wrtReparentTo(host.bodyNP.getParent())
            detach_character(host)
            game.world.attachRigidBody(character.bodyNP.node())
            host.layOutRanks()
            host.rebuildFootprint()
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
            'mount_special_rules': unit_data.get('mount_special_rules', []),
            'weapons': unit_data.get('weapons', []),
            'spells': unit_data.get('spells', []),
            'wizard_level': unit_data.get('wizard_level'),
        }
        game._create_unit(spec, unit_data.get('player', 1), unit_data['name'])

    # Restore the army list's rules. A recreated unit has only its catalogue
    # profile, and a surviving one may still be carrying a rule the *previous*
    # save granted, so the saved list replaces rather than adds to what is
    # there.
    by_name = {u.unitName: u for u in game.units}
    for unit_data in game_state['units']:
        unit = by_name.get(unit_data['name'])
        if unit is not None:
            game.applyDataRules(unit.unit.model, unit_data.get('special_rules'),
                                replace=True)
            mount = unit.unit.model.get_mount()
            if 'mount' in unit_data:
                mount_name = unit_data['mount']
                if mount_name is None:
                    unit.unit.model.special_rules = [
                        r for r in unit.unit.model.special_rules
                        if not (isinstance(r, dict) and r.get('tag') == 'mount')]
                    mount = None
                elif mount is None or mount.name != mount_name:
                    mount = Model(mount_name, '')
                    unit.unit.model.attach_mount(mount)
            if mount is not None and 'mount_special_rules' in unit_data:
                game.applyDataRules(mount, unit_data['mount_special_rules'], replace=True)
                mount._base_characteristics = copy.deepcopy(mount.characteristics)

    unit_map = {unit.unitName: unit for unit in game.units}
    # Lists are mutated in place because the AI holds references to them.
    game.player1Units[:] = []
    game.player2Units[:] = []
    for data in game_state['units']:
        member = unit_map.get(data['name'])
        if member is not None:
            member._player = data.get('player', 1)
            (game.player1Units if member._player == 1 else game.player2Units).append(member)

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
        unit.marchedThisTurn = unit_data.get('marchedThisTurn', False)
        unit.hasAttackedThisTurn = unit_data['hasAttackedThisTurn']
        unit.standAndShootWounds = unit_data.get('standAndShootWounds', 0)
        unit.attemptedRallyThisTurn = unit_data['attemptedRallyThisTurn']
        unit.chargedThisTurn = unit_data.get('chargedThisTurn', False)
        unit.countsAsChargedNextTurn = unit_data.get('countsAsChargedNextTurn', False)
        unit.chargeDistance = unit_data.get('chargeDistance', 0.0)
        unit.cannotChargeThisTurn = unit_data.get('cannotChargeThisTurn', False)
        unit.cannotPursueThisTurn = unit_data.get('cannotPursueThisTurn', False)
        unit.moveSpentThisTurn = unit_data.get('moveSpentThisTurn', 0.0)
        unit.manoeuvreThisTurn = unit_data.get('manoeuvreThisTurn', None)
        unit.redressDelta = unit_data.get('redressDelta', 0)
        unit.panicTestedThisPhase = unit_data.get('panicTestedThisPhase', False)
        unit.fledThisPhase = unit_data.get('fledThisPhase', False)
        unit.usedStubborn = unit_data.get('usedStubborn', False)
        # A spell attempted after the save was taken has not been attempted in
        # the state being loaded, so the allowance has to come back with it.
        unit.spellsCastThisTurn = list(unit_data.get('spellsCastThisTurn', []))
        unit.boundSpellPhases = list(unit_data.get('boundSpellPhases', []))
        unit.cannotCastThisTurn = unit_data.get('cannotCastThisTurn', False)
        unit.isDisrupted = unit_data.get('isDisrupted', False)
        # Saves written before the General was tracked keep the load-time nomination.
        unit.isGeneral = unit_data.get('isGeneral', getattr(unit, 'isGeneral', False))
        unit.isBSB = unit_data.get('isBSB', getattr(unit, 'isBSB', False))
        unit.woundsOnModel = unit_data.get('woundsOnModel', 0)
        unit.startOfBattleModels = unit_data.get('startOfBattleModels', unit.unit.nmodels)
        unit.startOfPhaseModels = unit_data.get('startOfPhaseModels', unit.unit.nmodels)
        unit.startOfPhaseEngaged = unit_data.get('startOfPhaseEngaged', False)
        unit.roundsFought = unit_data.get('roundsFought', 0)
        unit.isDeployed = unit_data['isDeployed']

        unit.unit.nmodels = unit_data['nmodels']
        unit.unit.files = unit_data['files']
        unit.unit.ranks = unit_data['ranks']

        unit.unit.model.characteristics = unit_data['characteristics']
        # A save is the source of truth for the profile it stores. Without this
        # the first reset_characteristics() after a combat reverts to the bare
        # catalogue entry and quietly drops whatever the roster gave the model.
        unit.unit.model._base_characteristics = copy.deepcopy(
            unit_data['characteristics'])
        unit.unit.model.armor_save = unit_data['armor_save']
        unit.unit.model.armour = list(unit_data.get('armour', []) or [])
        unit.unit.model.charging = unit_data['charging']

        if unit_data['equipped_weapon']:
            unit.unit.model.equip_weapon(unit_data['equipped_weapon'])

        restore_spellbook(unit.unit.model, unit_data.get('spells', []),
                          unit_data.get('wizard_level', 0))

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

    # A model that refused a challenge stays hidden, so its retirement is
    # restored after joining — join_unit puts it back in the front rank.
    for unit_data in game_state['units']:
        unit = unit_map.get(unit_data['name'])
        if unit is not None:
            unit.retiredFromCombat = bool(unit_data.get('retiredFromCombat'))
    for unit in game.units:
        if getattr(unit, 'joinedCharacter', None) is not None:
            unit.placeCharacter()

    # A challenge outlives the turn it was issued in (To The Death!, p. 211).
    game.challenges = []
    for saved in game_state.get('challenges') or []:
        challenger = unit_map.get(saved.get('challenger'))
        if challenger is None:
            continue
        challenge = Challenge(challenger, unit_map.get(saved.get('host')),
                              unit_map.get(saved.get('accepter')),
                              unit_map.get(saved.get('accepter_host')))
        challenge.refused = bool(saved.get('refused'))
        challenge.rounds = int(saved.get('rounds', 0))
        game.challenges.append(challenge)

    # Spells still in play: a hex, a ward or a vortex outlives the turn it was
    # cast in, so it has to come back or the save silently ends it.
    for spell in list(game.fsm.endOfTurnSpells):
        spell.endSpell()
    game.fsm.endOfTurnSpells = []
    for spell in list(getattr(game, 'remainsInPlay', [])):
        spell.endSpell()
    game.remainsInPlay = []
    load_spells(game, game_state.get('spells_in_play'), unit_map)
    game.roundCounter.apply_selection_masks()

    # Each model sits on the terrain surface, not at its unit's own Z. That
    # offset is derived rather than saved, so a unit restored onto a hill would
    # otherwise stand at ground level, inside the hill.
    for unit in game.units:
        if not unit.model.isEmpty():
            game.movement.alignModelsToHillNormal(unit)

    print(f"Game loaded from {filename}")
    messenger.send('hud-log', [f"Loaded: {filename}", 'info'])

    # Print analysis for both players
    for player_num in (1, 2):
        evaluation = game.analyzer.evaluate_overall_state(player_num=player_num)
        print(f"Player {player_num} Assessment: {evaluation['assessment']}")
        print(f"Total Score: {evaluation['total_score']:.1f}")
        strategy = game.analyzer.suggest_strategy(player_num=player_num)
        print(f"Suggested Strategy: {strategy}")
