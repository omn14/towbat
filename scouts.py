"""Scouts deployment and eligibility (Rulebook p. 177; Official FAQ v1.5.3).

Distances are between model bases, not regiment centres or the empty corners
of a skirmish blob. One world unit is one inch.
"""

from characters import side_of
from psychology import _box_corners, obb_distance

SCOUT_CLEARANCE = 12.0
BOARD_HALF_WIDTH = 36.0
BOARD_HALF_DEPTH = 24.0


def has_scouts(unit):
    # Split-profile grants apply to the whole model (pp. 192, 194, 204),
    # unlike a joined character, who must qualify separately.
    pending, seen, own = [unit], set(), False
    while pending:
        source = pending.pop()
        profile = getattr(getattr(source, 'unit', source), 'model', source)
        if id(profile) in seen:
            continue
        seen.add(id(profile))
        for rule in getattr(profile, 'special_rules', []):
            if not isinstance(rule, dict):
                continue
            own = own or bool(rule.get('scouts'))
            if rule.get('tag') == 'mount' and rule.get('mountUnit') is not None:
                pending.append(rule['mountUnit'])
            if (rule.get('tag') in ('crew', 'beasts') and rule.get('partUnit') is not None
                    and rule.get('count', 1) > 0):
                pending.append(rule['partUnit'])
    joined = getattr(unit, 'joinedCharacter', None)
    return bool(own and (joined is None or has_scouts(joined)))


def scout_charge_blocked(game, unit):
    """Only the owning player's first turn is barred, regardless of turn order."""
    owner = side_of(game, unit, default=None)
    if owner is None:
        return False
    joined = getattr(unit, 'joinedCharacter', None)
    deployed = (getattr(unit, 'deployedAsScouts', False)
                or getattr(joined, 'deployedAsScouts', False))
    return bool(deployed and game.roundCounter.currentRoundPlayer[owner - 1] == 0)


def scouts_block_vanguard(unit):
    """FAQ: late-deployed Scouts cannot subsequently make a Vanguard move."""
    joined = getattr(unit, 'joinedCharacter', None)
    return bool(getattr(unit, 'deployedAsScouts', False)
                or getattr(joined, 'deployedAsScouts', False))


def model_base_boxes(unit):
    """World XY boxes for live bases, including a joined character's own base.

    Use the formation nodes for centres but the database base dimensions for
    edges: mesh bounds include weapons and visual hill tilts are not facings.
    """
    if unit.unit.nmodels <= 0 or unit.bodyNP.isEmpty():
        return []
    root = unit.bodyNP.getTop()
    scale = unit.bodyNP.getScale(root)
    hx, hy = unit.modelWidth * abs(scale.x) / 2, unit.modelHeight * abs(scale.y) / 2
    heading = unit.bodyNP.getH(root)
    boxes = []
    for child in list(unit.model.getChildren())[:unit.unit.nmodels]:
        pos = child.getPos(root)
        boxes.append((pos.x, pos.y, hx, hy, heading))
    joined = getattr(unit, 'joinedCharacter', None)
    if joined is not None:
        boxes.extend(model_base_boxes(joined))
    return boxes


def nearest_enemy(game, unit, boxes=None):
    boxes = model_base_boxes(unit) if boxes is None else boxes
    owner = side_of(game, unit, default=None)
    best, enemy = float('inf'), None
    for other in game.units:
        if (other is unit or getattr(other, 'hostUnit', None) is not None
                or not other.isDeployed or other.unit.nmodels <= 0
                or side_of(game, other, default=None) in (None, owner)):
            continue
        enemy_boxes = model_base_boxes(other)
        for a in boxes:
            for b in enemy_boxes:
                distance = obb_distance(a, b)
                if distance < best:
                    best, enemy = distance, other
    return best, enemy


def placement_error(game, unit, *, scouting=False, ignore=None):
    """A quiet preview/drop validator; the caller logs only a refused drop.

    The own-zone exception sometimes assumed for Scouts does NOT exist (FAQ).
    Impassable pieces currently use their bounding rectangle conservatively.
    """
    boxes = model_base_boxes(unit)
    if not boxes:
        return 'No living models to deploy.'
    owner = side_of(game, unit, default=None)
    if owner is None:
        return 'Cannot determine the deploying player.'
    ymin, ymax = (-24.0, -12.0) if owner == 1 else (12.0, 24.0)
    for box in boxes:
        for x, y in _box_corners(*box):
            if abs(x) > BOARD_HALF_WIDTH or abs(y) > BOARD_HALF_DEPTH:
                return 'Every model base must be completely on the battlefield.'
            if not scouting and not ymin <= y <= ymax:
                return 'Every model base must be completely inside its deployment zone.'

    if scouting:
        distance, enemy = nearest_enemy(game, unit, boxes)
        if distance <= SCOUT_CLEARANCE:
            return (f'Nearest enemy {enemy.unit.name}: {distance:.3f}" base-to-base; '
                    f'Scouts require more than {SCOUT_CLEARANCE:g}" (also in their own zone).')

    for other in game.units:
        if (other is unit or other is ignore or not other.isDeployed
                or getattr(other, 'hostUnit', None) is not None):
            continue
        if any(obb_distance(a, b) <= 0 for a in boxes for b in model_base_boxes(other)):
            return f'Model bases overlap {other.unit.name}.'

    terrain = getattr(game, 'terrain_manager', None)
    for piece in getattr(terrain, 'terrain_pieces', []):
        if piece.is_impassable:
            box = (piece.center.x, piece.center.y, piece.width / 2, piece.height / 2, 0)
            if any(obb_distance(a, box) <= 0 for a in boxes):
                return f'Cannot deploy in impassable {piece.terrain_type} terrain.'
    return None


def deployment_candidates(game, player):
    """Undeployed units eligible for the current deployment step."""
    scouting = getattr(game, 'deploymentStage', 'ordinary') == 'scouts'
    units = game.player1Units if player == 1 else game.player2Units
    return [u for u in units if not u.isDeployed
            and (getattr(u, 'scoutDeploymentChoice', None) == 'scouts') == scouting]