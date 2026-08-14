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


def same_player(game, a, b) -> bool:
    """True if both unitGraphics belong to the same player."""
    return ((a in game.player1Units and b in game.player1Units) or
            (a in game.player2Units and b in game.player2Units))


def join_unit(game, character, host) -> bool:
    """Attach *character* to the front rank of *host*. Returns True on success."""
    if character is host or has_joined_character(host) or is_character(host):
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
    character.bodyNP.reparentTo(host.bodyNP)
    character.bodyNP.setPos(Point3(0, host.unitHeight * 0.5, 0))  # front-centre
    character.bodyNP.setHpr(0, 0, 0)
    character.model.setColor(character.color)

    # Remember the character's side before it leaves the player lists so a save
    # can still record which player it belongs to.
    character._player = 1 if character in game.player1Units else 2
    for lst in (game.player1Units, game.player2Units):
        if character in lst:
            lst.remove(character)
    return True


def on_host_removed(game, host):
    """Clean up a joined character when its host unit is destroyed."""
    character = get_joined_character(host)
    if character is None:
        return
    host.joinedCharacter = None
    character.hostUnit = None
    # The character's nodes are children of the host body and are torn down with
    # it; just drop it from the game's unit tracking.
    if character in game.units:
        game.units.remove(character)
