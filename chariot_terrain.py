"""Unit-specific linear obstacle blocking for Iron Shod Wheels (p. 194)."""

from contextlib import contextmanager

from collision_masks import CollisionMask as CM


def linear_impassable(piece, unit):
    from special_rules import unit_is_ethereal
    return (getattr(piece, 'linear_obstacle', False)
            and unit.unit.model.troop_type_rule('Iron Shod Wheels')
            and not unit_is_ethereal(unit))


@contextmanager
def obstacle_masks(game, unit, pass_over=False):
    changed = []
    if not pass_over:
        for piece in getattr(getattr(game, 'terrain_manager', None), 'terrain_pieces', []):
            node = getattr(piece, 'ghost_np', None)
            if node is not None and not node.isEmpty() and linear_impassable(piece, unit):
                changed.append((node, node.getCollideMask()))
                node.setCollideMask(node.getCollideMask() | CM.TERRAIN_IMPASSABLE)
    try:
        yield
    finally:
        for node, mask in changed:
            node.setCollideMask(mask)