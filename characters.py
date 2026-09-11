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
    return getattr(host, 'joinedCharacter', None)


def has_joined_character(host) -> bool:
    return get_joined_character(host) is not None


async def move_through_ranks(game, hosts):
    """Optional fighting-rank moves, inactive player's characters first (pp. 208-209)."""
    from combat_contacts import CombatContactSnapshot
    from command_groups import command_positions
    from rules_log import rule_log, rule_skipped
    hosts = list(dict.fromkeys(hosts))
    ordered = sorted(hosts, key=lambda host: side_of(game, host) == game.roundCounter.current_player)
    for host in ordered:
        character = get_joined_character(host)
        if character is None or character.unit.nmodels <= 0 or getattr(character, 'retiredFromCombat', False):
            continue
        enemies = [enemy for enemy in host.isInCombatWith if enemy in hosts and enemy.unit.nmodels > 0]
        if not enemies:
            continue
        snapshot = CombatContactSnapshot([host, *enemies])
        _, slots, _, initial, _ = snapshot.formations[id(host)]
        positions = [snapshot.positions(host, enemy)[1] for enemy in enemies]
        if len(slots) <= initial:
            continue
        if any(group[initial].fighting for group in positions):
            rule_skipped('Moving Through the Ranks', character,
                         f'already in a fighting rank of {host.unit.name}, slot {host.characterSlot + 1} (p. 209)')
            continue
        occupied = set(command_positions(host).values())
        candidates = sorted({slots[index] for group in positions for index, place in enumerate(group)
                             if place.fighting and index < initial and slots[index] not in occupied})
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
                         f'stays in slot {host.characterSlot + 1}; declines {len(options)} fighting-rank positions (p. 209)')
            continue
        previous, destination = host.characterSlot, options[selected]
        if getattr(host, 'characterCombatReturnSlot', None) is None:
            host.characterCombatReturnSlot = previous
        children = list(host.model.getChildren())
        displaced = children[slots.index(destination)]
        displaced.setPos(previous % host.unit.files * host.modelWidth,
                         -(previous // host.unit.files) * host.modelHeight, 0)
        host.characterSlot = destination
        host.placeCharacter()
        rule_log('Moving Through the Ranks', character,
                 f'{host.unit.name}: slot {previous + 1} -> {destination + 1}, {selected.lower()} (pp. 208-209)')


def return_through_ranks(host):
    """Return to the previous formation position when the combat ends (p. 208)."""
    previous = getattr(host, 'characterCombatReturnSlot', None)
    if previous is None:
        return
    host.characterSlot = previous
    host.layOutRanks()
    host.characterCombatReturnSlot = None
    host.placeCharacter()
    character = get_joined_character(host)
    if character is not None:
        from rules_log import rule_log
        rule_log('Moving Through the Ranks', character,
                 f'{host.unit.name} no longer engaged; returns to slot {host.characterSlot + 1} (p. 208)')


def same_player(game, a, b) -> bool:
    """True if both unitGraphics belong to the same player."""
    return ((a in game.player1Units and b in game.player1Units) or
            (a in game.player2Units and b in game.player2Units))


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


def join_unit(game, character, host) -> bool:
    """Attach *character* to the front rank of *host*. Returns True on success."""
    if character is host or has_joined_character(host) or is_character(host):
        return False
    from rules_log import rule_log, rule_skipped
    if any(member.unit.model.troop_type_rule('Lumbering') for member in (character, host)):
        rule_skipped('Lumbering', character,
                     f'cannot join {host.unit.name}: Lumbering models cannot join or be joined (p. 195)')
        return False
    character_rules = {rule.get('name', '').casefold() for rule in character.unit.model.special_rules}
    host_rules = {rule.get('name', '').casefold() for rule in host.unit.model.special_rules}
    if ('loner' in character_rules) != ('loner' in host_rules):
        rule_skipped('Loner', character, f'cannot join {host.unit.name}: both must have Loner (p. 172)')
        return False
    if 'sons of caledor' in host_rules:
        if not getattr(character, 'isGeneral', False) and 'blood of caledor' not in character_rules:
            rule_skipped('Sons of Caledor', character,
                         f'cannot join {host.unit.name}: requires the General or Blood of Caledor (FoF p. 170)')
            return False
        rule_log('Sons of Caledor', character, f'may join {host.unit.name}: '
                 + ('army General' if getattr(character, 'isGeneral', False) else 'Blood of Caledor'))
    from special_rules import is_ethereal
    if is_ethereal(character.unit.model) != is_ethereal(host.unit.model):
        from rules_log import rule_skipped
        rule_skipped('Ethereal', character, f'cannot join {host.unit.name}: both must be Ethereal or neither (p. 167)')
        return False

    host.joinedCharacter = character
    character.hostUnit = host
    character.isDeployed = True

    # Runtime marker so combat/shooting can discover the character generically.
    hm = host.unit.model
    hm.special_rules = [r for r in hm.special_rules
                        if not (isinstance(r, dict) and r.get('tag') == JOIN_TAG)]
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
    host.characterCombatReturnSlot = None
    host.skirmishCharacterPosition = None
    host.layOutRanks()
    host.placeCharacter()
    host.rebuildFootprint()

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
    character = get_joined_character(host)
    if character is None:
        return
    host.joinedCharacter = None
    character.hostUnit = None
    # The character's nodes are children of the host body and are torn down with
    # it; just drop it from the game's unit tracking.
    if character in game.units:
        game.units.remove(character)


def detach_character(host):
    """Forget the character a host was carrying, rank marker and rule with it."""
    if host is None:
        return
    character = get_joined_character(host)
    host.joinedCharacter = None
    host.characterSlot = None
    host.characterCombatReturnSlot = None
    host.skirmishCharacterPosition = None
    hm = host.unit.model
    hm.special_rules = [r for r in hm.special_rules
                        if not (isinstance(r, dict) and r.get('tag') == JOIN_TAG)]
    if character is not None:
        character.hostUnit = None
        from spell_effects import refresh_self_spells
        refresh_self_spells(character)
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
        replacement = targeted_replacement(boxes, len(boxes) - 1,
                                           range(host.unit.nmodels))
        if replacement is not None:
            position = character.bodyNP.getPos(host.bodyNP)
            record = host.skirmishLayout[replacement]
            record['x'], record['y'] = position.x, position.y
            host.applySkirmishLayout()
            rule_log('Skirmishers', host,
                     f'model {record["id"] + 1} replaces slain {character.unit.name}; '
                     'remaining models stay coherent (p. 184, FAQ v1.5.3)')
    detach_character(host)
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
