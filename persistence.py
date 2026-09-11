"""
Game state persistence — save and load functionality.

All functions accept the game (MyApp) instance as their first argument
so they can be imported and called without subclassing.
"""

import copy
from magic_items import save_inventory, restore_inventory
import json
import os
import shutil
from datetime import datetime

from battlescribe import get_catalogue, STAT_KEYS
from challenges import Challenge
from characters import detach_character, join_unit
from models import model as Model
from rules_log import battle_log
from special_rules import apply_rule_keywords
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


def _save_profile_state(profile):
    """Keep effective stats separate from the roster baseline, including split profiles."""
    parts = {}
    for tag in ('mount', 'crew', 'beasts'):
        part = getattr(profile, f'get_{tag}', lambda: None)()
        parts[tag] = None if part is None else {
            'name': part.name,
            'count': profile.part_count(tag) if tag != 'mount' else 1,
            **_save_profile_state(part),
        }
    return {
        'characteristics': copy.deepcopy(profile.characteristics),
        'profile_weapons': [_clean_weapon(weapon) for weapon in profile.weapons.values()],
        'profile_equipped_weapon': (profile.equipedWeapon or {}).get('name'),
        'profile_armour': list(getattr(profile, 'armour', [])),
        'profile_armor_save': profile.armor_save,
        'base_characteristics': copy.deepcopy(
            getattr(profile, '_base_characteristics', None) or profile.characteristics),
        'profile_parts': parts,
    }


def _restore_profile_state(profile, data):
    """Legacy saves have one authoritative profile; new saves retain both copies."""
    apply_rule_keywords(profile, data['characteristics'].get('Special Rules', []), replace=True)
    profile.characteristics = copy.deepcopy(data['characteristics'])
    profile._base_characteristics = copy.deepcopy(
        data.get('base_characteristics') or data['characteristics'])
    if 'profile_weapons' in data:
        profile.special_rules = [rule for rule in profile.special_rules if rule is not profile.equipedWeapon]
        profile.equipedWeapon = None
        profile.weapons = {}
        for weapon in data['profile_weapons']:
            if not profile.give_weapon(weapon['name']):
                profile.weapons[weapon['name']] = copy.deepcopy(weapon)
        equipped = data.get('profile_equipped_weapon')
        if equipped:
            profile.equip_weapon(equipped)
        profile.armour = list(data.get('profile_armour', []))
        profile.armor_save = data.get('profile_armor_save', profile.armor_save)
    for tag, record in data.get('profile_parts', {}).items():
        if tag not in ('mount', 'crew', 'beasts'):
            continue
        if record is None:
            profile.special_rules = [rule for rule in profile.special_rules
                                     if not (isinstance(rule, dict) and rule.get('tag') == tag)]
            continue
        part = getattr(profile, f'get_{tag}')()
        if part is None or part.name != record['name']:
            part = Model(record['name'], '')
        _restore_profile_state(part, record)
        if tag == 'mount':
            profile.attach_mount(part)
        else:
            getattr(profile, f'attach_{tag}')(part, record.get('count', 1))


def save_game_state(game, filename=None):
    """
    Serialize the current game state to a JSON file.

    Args:
        game: The MyApp game instance.
        filename: Target filename. Auto-generated with timestamp if omitted.

    Returns:
        The filename that was written.
    """
    from charge_declarations import save_declarations
    from drilled import move_pending
    if getattr(game, 'magicBusy', False) is True or getattr(game, 'castingSpell', False) is True:
        battle_log('Finish magic resolution before saving a battle.', 'info')
        return None
    if move_pending(game):
        battle_log('Finish the Drilled movement choice before saving.', 'info')
        return None
    if getattr(game, 'chargeStage', None) in ('resolving', 'blocked'):
        battle_log('Save unavailable during interrupted or active charge resolution.', 'info')
        return None
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"savegame_{timestamp}.json"
    filename = save_path(filename)

    game_state = {
        'current_phase': (game.fsm.getCurrentOrNextState()
                          or game.fsm.phases[game.fsm.currentPhaseIndex]),
        'current_phase_index': game.fsm.currentPhaseIndex,
        'charge_stage': getattr(game, 'chargeStage', None),
        'charge_declarations': save_declarations(game),
        'current_round': game.roundCounter.currentRoundPlayer,
        'current_player': game.roundCounter.current_player,
        'max_rounds': game.roundCounter.max_rounds,
        'deployment_stage': getattr(game, 'deploymentStage', 'ordinary'),
        'scout_deploy_first': getattr(game, 'scoutDeployFirst', None),
        'first_finished_deploying': getattr(game, 'firstFinishedDeploying', None),
        'vanguard_first': getattr(game, 'vanguardFirst', None),
        'vanguard_active': getattr(game, 'vanguardActive', None),
        'ai_player2_active': game.AIplayer2.active,
        'strategy_command_done': getattr(game, 'strategyCommandDone', True),
        'fated_dispel_turns': getattr(game, 'fatedDispelTurns', {}),
        'dispel_blocked_turns': getattr(game, 'dispelBlockedTurns', {}),
        'conjuration_done_turn': getattr(game, 'conjurationDoneTurn', None),
        'captured_standards': copy.deepcopy(getattr(game, 'capturedStandards', [])),
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

    terrain = getattr(game, 'terrain_manager', None)
    if terrain is not None:
        spell_pieces = [getattr(spell, 'piece', None)
                        for spell in getattr(game, 'remainsInPlay', [])]
        game_state['terrain'] = terrain.to_records(exclude=spell_pieces)

    for unit in game.units:
        unit_data = {
            'name': unit.unitName,
            'command': copy.deepcopy(getattr(unit.unit, 'command', [])),
            'command_profiles': {key: _save_profile_state(profile) for key, profile in
                                 getattr(unit.unit, 'command_models', {}).items()},
            'roster_metadata': copy.deepcopy(getattr(unit.unit, 'roster_metadata', {})),
            'magic_item_inventory': save_inventory(unit),
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
            'reserveDoneTurn': getattr(unit, 'reserveDoneTurn', None),
            'reserveMovementTurn': getattr(unit, 'reserveMovementTurn', None),
            'reserveMovementBlocked': getattr(unit, 'reserveMovementBlocked', None),
            'reserveMoveOriginal': getattr(unit, 'reserveMoveOriginal', None),
            'marchTestResult': (getattr(unit, 'marchTestResult', None)
                                if getattr(unit, 'marchTestResult', None) != 'pending' else None),
            'hasAttackedThisTurn': unit.hasAttackedThisTurn,
            'standAndShootWounds': getattr(unit, 'standAndShootWounds', 0),
            'attemptedRallyThisTurn': unit.attemptedRallyThisTurn,
            'usedRallyingCry': getattr(unit, 'usedRallyingCry', False),
            'chargedThisTurn': getattr(unit, 'chargedThisTurn', False),
            'counterChargeTurn': getattr(unit, 'counterChargeTurn', None),
            'lileathUsedTurn': getattr(unit, 'lileathUsedTurn', None),
            'dispelBlockedTurn': getattr(unit, 'dispelBlockedTurn', None),
            'chargeAttempts': getattr(unit, 'chargeAttempts', 0),
            'chargeAttemptPending': getattr(unit, 'chargeAttemptPending', False),
            'firstChargePending': getattr(unit, 'firstChargePending', False),
            'firstChargeDisruptedBy': list(getattr(unit, 'firstChargeDisruptedBy', [])),
            'firstChargeDisruptedNextTurnBy': list(getattr(unit, 'firstChargeDisruptedNextTurnBy', [])),
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
            'usedShieldwall': getattr(unit, 'usedShieldwall', False),
            'wasChargedThisTurn': getattr(unit, 'wasChargedThisTurn', False),
            'countsAsChargeTargetNextTurn': getattr(unit, 'countsAsChargeTargetNextTurn', False),
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
            'scoutDeploymentChoice': getattr(unit, 'scoutDeploymentChoice', None),
            'deployedAsScouts': getattr(unit, 'deployedAsScouts', False),
            'vanguardDone': getattr(unit, 'vanguardDone', False),
            'madeVanguardMove': getattr(unit, 'madeVanguardMove', False),
            'nmodels': unit.unit.nmodels,
            'files': unit.unit.files,
            'ranks': unit.unit.ranks,
            'skirmish_layout': (unit.savedSkirmishLayout()
                                if hasattr(unit, 'savedSkirmishLayout') else None),
            'points_cost': unit.unit.model.characteristics.get('Points', 0) * unit.unit.nmodels,
            **_save_profile_state(unit.unit.model),
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


def _repair_missing_profiles(unit_records):
    """Recover failed catalogue lookups, never replace saved characteristic values."""
    for record in unit_records:
        saved = record.get('characteristics', {})
        if any(key in saved for key in STAT_KEYS):
            continue
        name = record.get('model_name') or saved.get('Model') or saved.get('Unit')
        if not name:
            continue
        profile = get_catalogue().characteristics(name)
        if profile is None:
            continue
        profile.update(saved)
        record['characteristics'] = profile
        message = (f"Restored missing profile for {record['name']} ({name}) from catalogue: "
               f"S{profile.get('S')} T{profile.get('T')} W{profile.get('W')}; "
               f"{profile.get('Troop Type')}. Saved rules and battle state retained.")
        print(f"[persistence] {message}")
        battle_log(message, 'info')


def load_game_state(game, filename):
    """
    Restore game state from a JSON save file.

    Args:
        game: The MyApp game instance.
        filename: Name of a save in saves/, or a path to one.
    """
    from drilled import move_pending
    if move_pending(game):
        battle_log('Finish the Drilled movement choice before loading.', 'info')
        return
    if getattr(game, 'chargeStage', None) == 'resolving':
        battle_log('Finish charge resolution before loading a battle.', 'info')
        return
    if getattr(game, 'spellGenerationBusy', False) is True:
        print('[persistence] Finish the spell-generation choice before loading a battle.')
        return
    if getattr(game, 'magicBusy', False) is True or getattr(game, 'castingSpell', False) is True:
        battle_log('Finish magic resolution before loading a battle.', 'info')
        return
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

    _repair_missing_profiles(game_state['units'])
    from spell_effects import active_spells, end_effect
    for spell in active_spells(game):
        end_effect(spell, 'reloading battle')
    game.fsm.endOfTurnSpells = []
    game.remainsInPlay = []

    editor = getattr(game, 'skirmishEditor', None)
    if editor is not None:
        editor.close(resume=False)

    # Restore FSM state
    game.fsm.currentPhaseIndex = game_state['current_phase_index']
    game.restoringBattle = True
    try:
        game.fsm.request(game_state['current_phase'])
    finally:
        game.restoringBattle = False

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
        unit.reserveDoneTurn = unit_data.get('reserveDoneTurn')
        unit.reserveMovementTurn = unit_data.get('reserveMovementTurn')
        unit.reserveMovementBlocked = unit_data.get('reserveMovementBlocked')
        unit.reserveMoveOriginal = unit_data.get('reserveMoveOriginal')
        unit.marchTestResult = unit_data.get('marchTestResult')
        unit.hasAttackedThisTurn = unit_data['hasAttackedThisTurn']
        unit.standAndShootWounds = unit_data.get('standAndShootWounds', 0)
        unit.attemptedRallyThisTurn = unit_data['attemptedRallyThisTurn']
        unit.usedRallyingCry = unit_data.get('usedRallyingCry', False)
        unit.chargedThisTurn = unit_data.get('chargedThisTurn', False)
        unit.counterChargeTurn = unit_data.get('counterChargeTurn')
        unit.lileathUsedTurn = unit_data.get('lileathUsedTurn')
        unit.dispelBlockedTurn = unit_data.get('dispelBlockedTurn')
        unit.chargeAttempts = unit_data.get('chargeAttempts', 1)
        unit.chargeAttemptPending = unit_data.get('chargeAttemptPending', False)
        unit.firstChargePending = unit_data.get('firstChargePending', False)
        unit.firstChargeDisruptedBy = list(unit_data.get('firstChargeDisruptedBy', []))
        unit.firstChargeDisruptedNextTurnBy = list(unit_data.get('firstChargeDisruptedNextTurnBy', []))
        if 'chargeAttempts' not in unit_data:
            from first_charge import has_first_charge
            from rules_log import rule_skipped
            if has_first_charge(unit):
                rule_skipped('First Charge', unit, 'legacy save has no charge history; benefit treated as spent')
        unit.countsAsChargedNextTurn = unit_data.get('countsAsChargedNextTurn', False)
        unit.chargeDistance = unit_data.get('chargeDistance', 0.0)
        unit.cannotChargeThisTurn = unit_data.get('cannotChargeThisTurn', False)
        unit.cannotPursueThisTurn = unit_data.get('cannotPursueThisTurn', False)
        unit.moveSpentThisTurn = unit_data.get('moveSpentThisTurn', 0.0)
        unit.formedSkirmishPreview = None
        unit.formedSkirmishCharge = None
        unit.formedSkirmishCache = None
        unit.manoeuvreThisTurn = unit_data.get('manoeuvreThisTurn', None)
        unit.redressDelta = unit_data.get('redressDelta', 0)
        unit.panicTestedThisPhase = unit_data.get('panicTestedThisPhase', False)
        unit.fledThisPhase = unit_data.get('fledThisPhase', False)
        unit.usedStubborn = unit_data.get('usedStubborn', False)
        unit.usedShieldwall = unit_data.get('usedShieldwall', False)
        unit.wasChargedThisTurn = unit_data.get('wasChargedThisTurn', False)
        unit.countsAsChargeTargetNextTurn = unit_data.get('countsAsChargeTargetNextTurn', False)
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
        unit.scoutDeploymentChoice = unit_data.get('scoutDeploymentChoice')
        unit.deployedAsScouts = unit_data.get('deployedAsScouts', False)
        unit.vanguardDone = unit_data.get('vanguardDone', False)
        unit.madeVanguardMove = unit_data.get('madeVanguardMove', False)

        formation_changed = (unit.unit.nmodels, unit.unit.files, unit.unit.ranks) != (
            unit_data['nmodels'], unit_data['files'], unit_data['ranks'])
        unit.unit.nmodels = unit_data['nmodels']
        unit.unit.files = unit_data['files']
        unit.unit.ranks = unit_data['ranks']

        _restore_profile_state(unit.unit.model, unit_data)
        from roster_runtime import apply_roster_ownership
        apply_roster_ownership(unit.unit, {'command': unit_data.get('command', [])})
        unit.unit.roster_metadata = copy.deepcopy(unit_data.get('roster_metadata', {}))
        restore_inventory(unit, unit_data.get('magic_item_inventory', []))
        for key, record in unit_data.get('command_profiles', {}).items():
            if key in unit.unit.command_models:
                _restore_profile_state(unit.unit.command_models[key], record)
        unit.unit.model.armor_save = unit_data['armor_save']
        unit.unit.model.armour = list(unit_data.get('armour', []) or [])
        unit.unit.model.charging = unit_data['charging']

        if unit_data['equipped_weapon']:
            unit.unit.model.equip_weapon(unit_data['equipped_weapon'])

        restore_spellbook(unit.unit.model, unit_data.get('spells', []),
                          unit_data.get('wizard_level', 0))

        if hasattr(unit, 'model') and hasattr(unit, 'layOutRanks'):
            children = list(unit.model.getChildren())
            formation_changed = formation_changed or len(children) != unit.unit.nmodels
            while children and len(children) < unit.unit.nmodels:
                children.append(children[0].copyTo(unit.model))
            for child in children[unit.unit.nmodels:]:
                child.removeNode()
            unit.layOutRanks()
            if formation_changed:
                unit.rebuildFootprint()

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

    for unit_data in game_state['units']:
        unit = unit_map.get(unit_data['name'])
        if (unit is not None and hasattr(unit, 'restoreSkirmishLayout')
                and (unit.isSkirmisher or unit.unit.model.is_skirmisher())):
            unit.restoreSkirmishLayout(unit_data.get('skirmish_layout'))

    # A challenge outlives the turn it was issued in (To The Death!, p. 211).
    game.challenges = []
    from command_groups import champions
    challenge_models = dict(unit_map)
    for member in game.units:
        challenge_models.update({champion.unitName: champion for champion in
                                 champions(member, include_retired=True)})
    for saved in game_state.get('challenges') or []:
        challenger = challenge_models.get(saved.get('challenger'))
        if challenger is None:
            continue
        challenge = Challenge(challenger, unit_map.get(saved.get('host')),
                              challenge_models.get(saved.get('accepter')),
                              unit_map.get(saved.get('accepter_host')))
        challenge.refused = bool(saved.get('refused'))
        challenge.rounds = int(saved.get('rounds', 0))
        game.challenges.append(challenge)

    # Spells still in play: a hex, a ward or a vortex outlives the turn it was
    # cast in, so it has to come back or the save silently ends it.
    if 'terrain' in game_state and getattr(game, 'terrain_manager', None) is not None:
        game.terrain_manager.clear()
        game.terrain_manager.load_records(game_state['terrain'])
    load_spells(game, game_state.get('spells_in_play'), unit_map)
    # Restore after phase entry and rebuilding units; neither may re-roll or
    # advance a half-finished Scout deployment. Older saves deployed normally.
    game.deploymentStage = game_state.get('deployment_stage', 'ordinary')
    game.scoutDeployFirst = game_state.get('scout_deploy_first')
    game.firstFinishedDeploying = game_state.get('first_finished_deploying')
    game.vanguardFirst = game_state.get('vanguard_first')
    game.vanguardActive = game_state.get('vanguard_active')
    game.strategyCommandDone = game_state.get('strategy_command_done', True)
    game.fatedDispelTurns = game_state.get('fated_dispel_turns', {})
    game.dispelBlockedTurns = game_state.get('dispel_blocked_turns', {})
    game.conjurationDoneTurn = game_state.get('conjuration_done_turn')
    game.magicBusy = False
    game.capturedStandards = copy.deepcopy(game_state.get('captured_standards', []))
    game.rallyingCryBusy = False
    from charge_declarations import restore_declarations
    restore_declarations(game, game_state, unit_map)
    game.roundCounter.apply_selection_masks()
    if game_state['current_phase'] == 'ReserveMovePhase':
        from reserve_move import prepare
        prepare(game)
    if game_state['current_phase'] == 'DeployPhase':
        from deployPhase import refresh_deployment
        refresh_deployment(game)
        if (game.roundCounter.current_player == 2 and game.AIplayer2.active
            and game.deploymentStage != 'vanguard'):
            game.AIplayer2.deployUnits()

    # Each model sits on the terrain surface, not at its unit's own Z. That
    # offset is derived rather than saved, so a unit restored onto a hill would
    # otherwise stand at ground level, inside the hill.
    for unit in game.units:
        if not unit.model.isEmpty():
            game.movement.alignModelsToHillNormal(unit)

    print(f"Game loaded from {filename}")
    messenger.send('hud-log', [f"Loaded: {filename}", 'info'])

    # Print analysis for both players
    if game_state['current_phase'] == 'DeployPhase':
        from spell_generation import begin_spell_generation
        begin_spell_generation(game)

    for player_num in (1, 2):
        evaluation = game.analyzer.evaluate_overall_state(player_num=player_num)
        print(f"Player {player_num} Assessment: {evaluation['assessment']}")
        print(f"Total Score: {evaluation['total_score']:.1f}")
        strategy = game.analyzer.suggest_strategy(player_num=player_num)
        print(f"Suggested Strategy: {strategy}")
