"""Lileath's Blessing (Forces of Fantasy p. 185; Magic FAQ v1.5.3)."""

from panda3d.core import Vec3

from magic_items import current_turn
from rules_log import rule_log, rule_skipped


async def reroll_casting(spell, dice, outcome, result):
    caster, game = spell.caster, spell.game
    if caster is None or game is None:
        return dice
    if not any(isinstance(rule, dict) and rule.get('name') == "Lileath's Blessing"
               for rule in caster.unit.model.special_rules):
        return dice
    detail = f'{spell.name}: dice {dice}, result {result} vs {spell.casting_value}+'
    token = current_turn(game)
    if spell.bound:
        reason = 'Bound spell; bearer cannot apply personal casting rules (p. 109)'
    elif outcome != 'failed':
        reason = f'{outcome}; only failed Casting rolls qualify, never a Miscast'
    elif getattr(caster, 'lileathUsedTurn', None) == token and token is not None:
        reason = 'already used this turn'
    elif token is None:
        reason = 'turn identity unavailable; cannot track once-per-turn use'
    else:
        reason = None
    if reason:
        rule_skipped("Lileath's Blessing", caster, f'{detail}; {reason}')
        return dice
    if not game.aiControls(caster):
        choice = await game.makeChoiceNew(
            ['Re-roll', 'Keep'], Vec3(0, 0, 10), owner=caster,
            prompt=f"{caster.unit.name}: Lileath's Blessing?", detail=detail)
        if choice != 'Re-roll':
            rule_skipped("Lileath's Blessing", caster, f'{detail}; keeps failure, blessing remains available')
            return dice
    caster.lileathUsedTurn = token
    total, replacement = await spell._roll_casting_dice()
    rule_log("Lileath's Blessing", caster,
             f'{detail} -> re-roll {replacement} = {total}; replacement stands, used for this turn')
    return replacement