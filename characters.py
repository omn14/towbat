"""
Character-joins-unit mechanic.

A character model can be deployed onto a friendly unit to join its front rank.
While joined the character moves with the unit, fights as part of the fighting
rank (replacing one normal model), and shoots with the unit if it carries a
missile weapon. The character is kept as its own ``unitGraphics`` (so it retains
its own profile, weapons and wounds) but is taken out of independent play and
tracked on the host via ``host.joinedCharacter`` and a runtime special rule.
"""

from panda3d.core import Point3, BitMask32

# Runtime-only special-rule tag used to find a host's joined character during
# combat and shooting (mirrors the 'mount' tag pattern).
JOIN_TAG = 'joined_character'


def is_character(unit_graphics) -> bool:
    """True if this unitGraphics is a character (catalogue Category)."""
    u = getattr(unit_graphics, 'unit', None)
    ch = u.model.characteristics if u and u.model else {}
    return str(ch.get('Category', '')).strip().lower() == 'characters'


def get_joined_character(host):
    """Return the host's joined character unitGraphics, or None."""
    return next(iter(get_joined_characters(host)), None)


def get_joined_characters(host):
    """All attached characters, with compatibility for older saves and fixtures."""
    members = getattr(host, 'joinedCharacters', None)
    if members is not None:
        return list(members)
    member = getattr(host, 'joinedCharacter', None)
    return [member] if member is not None else []


def rank_placements(count, files, width, depth, characters, command_slots=(), preferred=None):
    """Reserve full base areas; incompatible bases flank the front rank (p. 207)."""
    import math
    files = max(1, files)
    occupied = set(command_slots)
    placements = {}
    flanks = ['right', 'left']
    preferred = preferred or {}
    for identity, base_width, base_depth, retired in characters:
        columns, rows = round(base_width / width), round(base_depth / depth)
        fits = (columns >= 1 and rows >= 1 and columns <= files
                and math.isclose(columns * width, base_width, abs_tol=1e-4)
                and math.isclose(rows * depth, base_depth, abs_tol=1e-4))
        candidates = sorted(range(max(0, files - columns + 1)),
                            key=lambda column: (abs(column + (columns - 1) / 2 - files // 2), column)) if fits else []
        if fits and identity in preferred:
            candidates = [preferred[identity]]
        chosen = None
        if not retired:
            for candidate in candidates:
                start_row, column = divmod(candidate, files)
                cells = {(start_row + row) * files + column + offset for row in range(rows)
                         for offset in range(columns)}
                front_characters = sum(not record['rear'] for record in placements.values()) + 1
                front_reserved = sum(cell < files for cell in occupied | cells)
                ordinary_front = min(count, files - front_reserved + sum(slot < files for slot in command_slots))
                if (column + columns <= files and not cells.intersection(occupied)
                        and (start_row > 0 or ordinary_front >= front_characters)):
                    chosen = candidate, cells
                    break
        if chosen is not None:
            candidate, cells = chosen
            start_row, column = divmod(candidate, files)
            occupied.update(cells)
            placements[identity] = dict(slot=candidate, cells=sorted(cells), rear=False, adjacent=False,
                                        x=column * width + (base_width - width) / 2,
                                        y=-start_row * depth - (base_depth - depth) / 2,
                                        width=base_width, depth=base_depth)
        elif (not fits and not retired and flanks
              and sum(slot < files for slot in command_slots) < min(files, count)
              and sum(not record['rear'] for record in placements.values()) + 1
                  <= min(count, files - sum(cell < files and cell not in command_slots for cell in occupied))):
            side = flanks.pop(0)
            position = files * width - width / 2 + base_width / 2 if side == 'right' else -(width + base_width) / 2
            placements[identity] = dict(slot=files - 1 if side == 'right' else 0, cells=[], rear=False, adjacent=True,
                                        x=position, y=-(base_depth - depth) / 2,
                                        width=base_width, depth=base_depth)
        else:
            placements[identity] = dict(slot=None, cells=[], rear=True, adjacent=not fits,
                                        width=base_width, depth=base_depth)
    ordinary = list(command_slots)
    slot = 0
    while len(ordinary) < count:
        if slot not in occupied:
            ordinary.append(slot)
        slot += 1
    bottom = max([depth / 2] + [(slot // files + .5) * depth for slot in ordinary]
                 + [-record['y'] + record['depth'] / 2 for record in placements.values()
                    if not record['rear']])
    for record in placements.values():
        if record['rear']:
            record.update(slot=math.ceil((bottom - depth / 2) / depth) * files,
                          x=(files - 1) * width / 2, y=-bottom - record['depth'] / 2)
            bottom += record['depth']
    return ordinary, placements


def has_joined_character(host) -> bool:
    return get_joined_character(host) is not None


async def move_through_ranks(game, hosts):
    """Optional fighting-rank moves, inactive player's characters first (pp. 208-209)."""
    from combat_contacts import CombatContactSnapshot
    from command_groups import command_positions
    from rules_log import rule_log, rule_skipped
    hosts = list(dict.fromkeys(hosts))
    ordered = sorted(hosts, key=lambda host: side_of(game, host) == game.roundCounter.current_player)
    for host, character in [(host, member) for host in ordered for member in get_joined_characters(host)]:
        if character.unit.nmodels <= 0 or getattr(character, 'retiredFromCombat', False):
            continue
        enemies = [enemy for enemy in host.isInCombatWith if enemy in hosts and enemy.unit.nmodels > 0]
        if not enemies:
            continue
        snapshot = CombatContactSnapshot([host, *enemies])
        _, slots, _, initial, _ = snapshot.formations[id(host)]
        character_index = initial + get_joined_characters(host).index(character)
        positions = [snapshot.positions(host, enemy)[1] for enemy in enemies]
        if len(slots) <= character_index:
            continue
        current = getattr(character, 'formationSlot', host.characterSlot)
        if any(group[character_index].fighting for group in positions):
            rule_skipped('Moving Through the Ranks', character,
                         f'already in a fighting rank of {host.unit.name}, slot {current + 1} (p. 209)')
            continue
        occupied = set(command_positions(host).values())
        candidates = sorted({slots[index] for group in positions for index, place in enumerate(group)
                             if place.fighting and index < initial and slots[index] not in occupied})
        from command_groups import living_command
        descriptions = [(member.unitName, member.modelWidth, member.modelHeight,
                         getattr(member, 'retiredFromCombat', False)) for member in get_joined_characters(host)]
        preferred = {member.unitName: member.combatSlot for member in get_joined_characters(host)
                     if getattr(member, 'combatSlot', None) is not None}
        legal = []
        for destination in candidates:
            _, planned = rank_placements(host.unit.nmodels, host.unit.files, host.modelWidth, host.modelHeight,
                descriptions, [command_positions(host)[id(entry)] for entry in living_command(host)],
                {**preferred, character.unitName: destination})
            if planned[character.unitName]['slot'] == destination and not planned[character.unitName]['rear']:
                legal.append(destination)
        candidates = legal
        options = {f'Rank {slot // host.unit.files + 1}, file {slot % host.unit.files + 1}': slot
                   for slot in candidates}
        if not options:
            rule_skipped('Moving Through the Ranks', character,
                         f'{host.unit.name}: no fighting-rank slot free of command models (pp. 198, 209)')
            continue
        selected = (next(iter(options)) if game.aiControls(host) else
                    await game.makeChoiceNew(['Stay in place', *options], Point3(0, 0, 10), owner=host,
                                             prompt=f'{character.unit.name}: move through the ranks?'))
        if selected not in options:
            rule_skipped('Moving Through the Ranks', character,
                         f'stays in slot {current + 1}; declines {len(options)} fighting-rank positions (p. 209)')
            continue
        previous, destination = current, options[selected]
        if getattr(character, 'combatReturnSlot', None) is None:
            character.combatReturnSlot = previous
        character.combatSlot = destination
        if character is get_joined_character(host):
            host.characterCombatReturnSlot = character.combatReturnSlot
        host.layOutRanks()
        host.placeCharacter()
        rule_log('Moving Through the Ranks', character,
                 f'{host.unit.name}: slot {previous + 1} -> {destination + 1}, {selected.lower()} (pp. 208-209)')


def return_through_ranks(host):
    """Return to the previous formation position when the combat ends (p. 208)."""
    members = [member for member in get_joined_characters(host) if getattr(member, 'combatReturnSlot', None) is not None]
    if not members and getattr(host, 'characterCombatReturnSlot', None) is None:
        return
    for member in members:
        member.combatSlot = None
        member.combatReturnSlot = None
    host.characterCombatReturnSlot = None
    host.layOutRanks()
    host.placeCharacter()
    for character in members:
        from rules_log import rule_log
        rule_log('Moving Through the Ranks', character,
                 f'{host.unit.name} no longer engaged; returns to slot {character.formationSlot + 1} (p. 208)')


def same_player(game, a, b) -> bool:
    """True if both unitGraphics belong to the same player."""
    return ((a in game.player1Units and b in game.player1Units) or
            (a in game.player2Units and b in game.player2Units))


def ai_controls_player(game, player):
    controller = getattr(game, f'AIplayer{player}', None)
    return controller is not None and controller.active


def side_of(game, unit, default: int | None = 1):
    """Which player *unit* fights for, 1 or 2.

    Joining takes a character out of both player lists, so membership alone
    answers this wrongly for the very models that cast most of the spells.
    """
    if unit in game.player1Units:
        return 1
    if unit in game.player2Units:
        return 2
    side = getattr(unit, '_player', None)
    if side in (1, 2):
        return side
    host = getattr(unit, 'hostUnit', None)
    return side_of(game, host, default) if host is not None else default


def friendly_units(game, unit):
    """The units on *unit*'s own side."""
    return game.player1Units if side_of(game, unit) == 1 else game.player2Units


def enemy_units(game, unit):
    """The units opposing *unit*."""
    return game.player2Units if side_of(game, unit) == 1 else game.player1Units


def effective_troop_type(member):
    """Mounted characters use their mount's troop category (Rulebook pp. 204-205)."""
    profile = member.unit.model
    mount = profile.get_mount()
    return (mount or profile).troop_type()


def remaining_move_reason(game, unit):
    """Joining and voluntary departure take place in Remaining Moves (p. 207)."""
    from chaos_gifts import succumbed
    if getattr(getattr(game, 'fsm', None), 'state', None) != 'MovementPhase':
        return 'only available in Remaining Moves'
    if getattr(game, 'chargeStage', None) not in (None, 'remaining'):
        return 'finish charge declarations and charge moves first'
    if side_of(game, unit, None) != game.roundCounter.current_player:
        return 'not the active player\'s unit'
    if not getattr(unit, 'isDeployed', False) or unit.unit.nmodels <= 0:
        return 'unit is not on the battlefield'
    if getattr(unit, 'state', None) != 'Idle' or getattr(unit, 'isInCombat', False):
        return 'unit is fleeing, engaged, or unable to make an ordinary move'
    if getattr(unit, 'hasMovedThisTurn', False) or getattr(unit, 'joinedMovementLocked', False):
        return 'unit has moved or was joined during Remaining Moves'
    if succumbed(unit):
        return 'unit succumbed to Stupidity'
    if any(getattr(unit, flag, False) for flag in ('chargedThisTurn', 'fledThisPhase', 'freePivot')):
        return 'unit has charged, fled, or has an unfinished pivot'
    if any(getattr(member, 'freePivot', None) for member in getattr(game, 'units', [])):
        return 'finish the pending pivot first'
    if any(getattr(game, flag, False) for flag in ('awaitingChoice', 'magicBusy', '_drilledMoveActive')):
        return 'finish the current action first'
    return None


def join_reason(game, character, host, *, movement=False):
    """Compatibility shared by deployment and Remaining Moves (pp. 167, 179, 185, 191, 195, 207)."""
    import troop_types
    from special_rules import is_ethereal
    if character is host or not is_character(character) or is_character(host):
        return 'a character may join a regiment, not another character'
    if getattr(character, 'hostUnit', None) is not None:
        return 'character is already attached to a unit'
    if side_of(game, character, None) is None or side_of(game, character, None) != side_of(game, host, None):
        return 'cannot join an enemy unit'
    if host.unit.nmodels <= 0 or getattr(host, 'state', None) == 'IsFleeing' or getattr(host, 'isInCombat', False):
        return 'host is destroyed, fleeing, or engaged'
    character_type, host_type = effective_troop_type(character), effective_troop_type(host)
    if any(troop_types.has_rule(kind, 'Lumbering') for kind in (character_type, host_type)):
        return 'Lumbering models cannot join or be joined (p. 195)'
    if troop_types.has_rule(host_type, 'Clumsy') and not troop_types.has_rule(character_type, 'Clumsy'):
        return 'a Clumsy host requires a Clumsy character (p. 191)'
    if getattr(host, 'isSkirmisher', False) and character_type != host_type:
        return f'Skirmishers require matching troop subcategories: {character_type} / {host_type} (p. 185)'
    own = {rule.get('name', '').casefold() for rule in character.unit.model.special_rules}
    other = {rule.get('name', '').casefold() for rule in host.unit.model.special_rules}
    for name in ('Loner', 'Unbreakable'):
        if (name.casefold() in own) != (name.casefold() in other):
            return f'both character and host must have {name}, or neither'
    if is_ethereal(character.unit.model) != is_ethereal(host.unit.model):
        return 'both character and host must be Ethereal, or neither'
    if 'sons of caledor' in other and not (getattr(character, 'isGeneral', False) or 'blood of caledor' in own):
        return 'Sons of Caledor require the General or Blood of Caledor'
    if 'chracian warriors' in other and not (getattr(character, 'isGeneral', False) or 'chracian hunter' in own
            or character.unit.model.name.casefold() in ('korhil lionmane', 'chracian chieftain')):
        return 'Chracian Warriors require an eligible character'
    if movement:
        if not getattr(host, 'isDeployed', False):
            return 'host is not deployed on the battlefield'
        return remaining_move_reason(game, character)
    return None


def leave_reason(game, character):
    """Leave before the host moves; a host unable to move cannot release a character (p. 207)."""
    host = getattr(character, 'hostUnit', None)
    if host is None:
        return 'character is not attached to a unit'
    if getattr(character, 'hasMovedThisTurn', False):
        return 'character has already moved this turn'
    if getattr(host, 'moveSpentThisTurn', 0) > 0:
        return 'host has already spent movement on a manoeuvre'
    return remaining_move_reason(game, host)


def join_unit(game, character, host, *, restoring=False) -> bool:
    """Join the front rank, displacing ordinary models to the rear (Rulebook p. 207)."""
    from rules_log import rule_log, rule_skipped
    reason = None if restoring else join_reason(game, character, host)
    if reason:
        rule_skipped('Characters & Units', character, f'cannot join {host.unit.name}: {reason}')
        return False
    character_rules = {rule.get('name', '').casefold() for rule in character.unit.model.special_rules}
    host_rules = {rule.get('name', '').casefold() for rule in host.unit.model.special_rules}
    if 'sons of caledor' in host_rules:
        rule_log('Sons of Caledor', character, f'may join {host.unit.name}: '
                 + ('army General' if getattr(character, 'isGeneral', False) else 'Blood of Caledor'))
    if 'chracian warriors' in host_rules:
        rule_log('Chracian Warriors', character,
                 f'may join {host.unit.name}: General={bool(getattr(character, "isGeneral", False))}, '
                 f'Chracian Hunter={"chracian hunter" in character_rules}, '
                 f'profile={character.unit.model.name} (FoF p. 163; FAQ v1.5.3)')
    host.joinedCharacters = [*get_joined_characters(host), character]
    host.joinedCharacter = host.joinedCharacters[0]
    character.hostUnit = host
    character.isDeployed = True
    character.unit.model._joined_skirmish = host.isSkirmisher
    character.isSkirmisher = host.isSkirmisher
    if character.isSkirmisher and not character.skirmishLayout:
        character._arrange_skirmish_blob()
    rule_log('Characters & Formations', character,
             f'joins {host.unit.name}: adopts its {"Skirmish" if host.isSkirmisher else "formed"} '
             'formation; no independent movement (p. 205)')

    # Runtime marker so combat/shooting can discover the character generically.
    hm = host.unit.model
    hm.special_rules.append({'name': 'Joined Character', 'tag': JOIN_TAG,
                             'characterUnit': character.unit,
                             'characterGraphics': character})

    # Take the character out of the physics world and independent selection,
    # then parent it under the host so it follows all movement/rotation.
    try:
        game.world.removeRigidBody(character.bodyNP.node())
    except Exception:
        pass
    character.bodyNP.setCollideMask(BitMask32.allOff())
    # Parented to the body, not to host.model: casualties are taken by deleting
    # the last child of host.model, which would otherwise delete the character.
    character.bodyNP.reparentTo(host.bodyNP)
    character.bodyNP.setHpr(0, 0, 0)
    character.model.setColor(character.color)

    # The character stands in the middle of the front rank, and the unit's own
    # models close up around it, so the one it displaces ends up at the back.
    host.characterSlot = max(1, host.unit.files) // 2
    host.characterCombatReturnSlot = getattr(host.joinedCharacter, 'combatReturnSlot', None)
    if not restoring:
        character.combatSlot = character.combatReturnSlot = None
    character.joinedPosition = None
    front_rank = host.model.getPos()
    host.layOutRanks()
    host.rebuildFootprint()
    if (not restoring and not getattr(game, 'restoringBattle', False)
            and not (host.isSkirmisher and not host.skirmishCombat)):
        shift = front_rank - host.model.getPos()
        host.bodyNP.setPos(host.bodyNP, shift)
        host.bodyNP.node().setTransformDirty()
    host.placeCharacter()
    placement = getattr(host, 'characterPlacements', {}).get(character.unitName)
    if placement is not None:
        location = 'rear: no legal front-rank room' if placement['rear'] else ('front-rank flank' if placement['adjacent'] else 'front rank')
        rule_log('Positioning Characters', character,
                 f'{host.unit.name}: {character.modelWidth * 25.4:g}x{character.modelHeight * 25.4:g} mm base, '
                 f'{len(placement["cells"])} grid cells, {location}; command models retained (p. 207)')
    if getattr(game, 'movement', None) is not None:
        game.movement.alignModelsToHillNormal(host)
        game.movement.alignModelsToHillNormal(character)

    # Remember the character's side before it leaves the player lists so a save
    # can still record which player it belongs to.
    character._player = 1 if character in game.player1Units else 2
    for lst in (game.player1Units, game.player2Units):
        if character in lst:
            lst.remove(character)
    from spell_effects import active_spells
    for spell in active_spells(game):
        if hasattr(spell, 'on_join'):
            spell.on_join(character, host)
    return True


def on_host_removed(game, host):
    """Clean up a joined character when its host unit is destroyed."""
    from spell_effects import caster_removed
    caster_removed(game, host)
    characters = get_joined_characters(host)
    host.joinedCharacters = []
    host.joinedCharacter = None
    # The character's nodes are children of the host body and are torn down with
    # it; just drop it from the game's unit tracking.
    for character in characters:
        character.hostUnit = None
        character.unit.model._joined_skirmish = None
        character.unit.nmodels = 0
        for members in (game.player1Units, game.player2Units):
            if character in members:
                members.remove(character)
        if not character.bodyNP.isEmpty():
            character.bodyNP.removeNode()
        if character in game.units:
            game.units.remove(character)


def detach_character(host, character=None):
    """Forget the character a host was carrying, rank marker and rule with it."""
    if host is None:
        return
    character = character if character is not None else get_joined_character(host)
    host.joinedCharacters = [member for member in get_joined_characters(host) if member is not character]
    host.joinedCharacter = next(iter(host.joinedCharacters), None)
    first = host.joinedCharacter
    host.characterSlot = getattr(first, 'formationSlot', None)
    host.characterCombatReturnSlot = getattr(first, 'combatReturnSlot', None)
    host.skirmishCharacterPosition = getattr(first, 'joinedPosition', None)
    if hasattr(host, 'characterPlacements'):
        host.characterPlacements.pop(getattr(character, 'unitName', None), None)
    hm = host.unit.model
    hm.special_rules = [r for r in hm.special_rules
                        if not (isinstance(r, dict) and r.get('tag') == JOIN_TAG
                                and (r.get('characterGraphics') is character
                                     or r.get('characterUnit') is getattr(character, 'unit', None)))]
    if character is not None:
        character.hostUnit = None
        character.combatSlot = character.combatReturnSlot = None
        character.unit.model._joined_skirmish = None
        if hasattr(character, 'isSkirmisher'):
            character.isSkirmisher = character.unit.model.is_skirmisher()
            from rules_log import rule_log
            rule_log('Characters & Formations', character,
                     f'leaves {host.unit.name}: resumes '
                     f'{"Skirmish" if character.isSkirmisher else "own formed"} formation (p. 205)')
        from spell_effects import refresh_self_spells
        refresh_self_spells(character)
    return character


def release_character(game, character):
    """Restore independent scene ownership without changing the base's world position (p. 207)."""
    host = character.hostUnit
    owner = side_of(game, host)
    character.bodyNP.wrtReparentTo(host.bodyNP.getParent())
    detach_character(host, character)
    character._player = owner
    character.skirmishCombat = False
    character.layOutRanks()
    character.rebuildFootprint()
    side = game.player1Units if owner == 1 else game.player2Units
    if character not in side:
        side.append(character)
    if getattr(game, 'roundCounter', None) is not None:
        game.roundCounter.apply_selection_masks()
    return character


def slay_character(game, character):
    """Take a slain character off the board and out of its host's rank.

    The reverse of joining: the host closes its ranks over the gap. Not the
    same path as an ordinary casualty — a joined character has no rigid body
    of its own (joining took it out of the physics world) and its nodes hang
    under the host, so `removeModelsFromUnit` would remove a body twice and
    leave the host pointing at a destroyed model.
    """
    from spell_effects import caster_removed
    caster_removed(game, character)
    host = getattr(character, 'hostUnit', None)
    if (host is not None and getattr(host, 'isSkirmisher', False)
            and not getattr(host, 'skirmishCombat', False)):
        from scouts import model_base_boxes
        from skirmish import targeted_replacement
        from rules_log import rule_log
        boxes = model_base_boxes(host)
        replacement = targeted_replacement(boxes, host.unit.nmodels + get_joined_characters(host).index(character),
                                           range(host.unit.nmodels))
        if replacement is not None:
            position = character.bodyNP.getPos(host.bodyNP)
            record = host.skirmishLayout[replacement]
            record['x'], record['y'] = position.x, position.y
            host.applySkirmishLayout()
            rule_log('Skirmishers', host,
                     f'model {record["id"] + 1} replaces slain {character.unit.name}; '
                     'remaining models stay coherent (p. 184, FAQ v1.5.3)')
    detach_character(host, character)
    if not character.model.isEmpty():
        character.model.removeNode()
    if not character.bodyNP.isEmpty():
        character.bodyNP.removeNode()
    if character in game.units:
        game.units.remove(character)
    for lst in (game.player1Units, game.player2Units):
        if character in lst:
            lst.remove(character)
    if host is not None and not host.model.isEmpty():
        host.layOutRanks()
        host.rebuildFootprint()
