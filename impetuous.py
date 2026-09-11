"""Compulsory charge declarations (Rulebook p. 172, amended; FAQ v1.5.3)."""

from panda3d.core import Vec3

from characters import enemy_units
from first_charge import begin_charge_attempt
from flight import compulsory_mode, compulsory_preview
from formed_skirmish_charge import preview_charge, route_allowance, route_to_model
from psychology import active_character, leadership_passed, reroll_leadership
from rules_log import rule_log, rule_skipped
from scouts import model_base_boxes, scout_charge_blocked
from skirmish_visibility import model_can_see
from special_rules import max_charge_range, unit_has_swiftstride
from vanguard import vanguard_charge_blocked


def has_impetuous(unit):
    """One or more models, including a joined character, suffices (p. 172)."""
    for member in (unit, getattr(unit, 'joinedCharacter', None)):
        if member is None:
            continue
        profile = member.unit.model
        for part in (profile, profile.get_mount(), profile.get_crew(), profile.get_beasts()):
            if part is not None and any(isinstance(rule, dict) and rule.get('name') == 'Impetuous'
                                        for rule in part.special_rules):
                return True
    return False


@compulsory_preview
def legal_targets(game, unit):
    """Use front-arc sight, range including wheels, and the shared route planner.

    Marching Column prevents the move, not its declaration (p. 101; FAQ v1.5.3).
    """
    if (unit.state != 'Idle' or unit.hasMovedThisTurn or unit.moveSpentThisTurn
            or unit.isInCombat or unit.cannotChargeThisTurn or not unit.isDeployed
            or getattr(unit, 'hostUnit', None) is not None
            or unit.bodyNP.isEmpty() or unit.unit.nmodels <= 0
            or scout_charge_blocked(game, unit) or vanguard_charge_blocked(game, unit)):
        return []
    result = []
    origin = unit.bodyNP.getPos()
    source = game.psychology._unit_box(unit)
    pieces = game.terrain_manager.terrain_pieces
    for target in enemy_units(game, unit):
        if not target.isDeployed or target.bodyNP.isEmpty() or target.unit.nmodels <= 0:
            continue
        if unit.isSkirmisher:
            continue
        if target.isSkirmisher:
            preview = preview_charge(game, unit, target)
            if preview.error is None:
                result.append((target, preview.route, preview.route.target_index))
            continue
        others = [box for member in game.units if member not in (unit, target)
                  and member.isDeployed and not member.bodyNP.isEmpty()
                  and getattr(member, 'hostUnit', None) is None
                  for box in model_base_boxes(member)]
        target_box = game.psychology._unit_box(target)
        opaque = [(piece.center.x, piece.center.y, piece.width / 2, piece.height / 2, 0)
                  for piece in pieces if piece.blocks_line_of_sight
                  and not piece.contains(origin) and not piece.contains(target.bodyNP.getPos())]
        if not model_can_see(source, [target_box], [*others, *opaque], facing=source[4]):
            continue
        obstacles = [*others, *((piece.center.x, piece.center.y, piece.width / 2, piece.height / 2, 0)
                                for piece in pieces if piece.is_impassable)]
        route = route_to_model([source], [target_box], 0, origin, obstacles)
        if route is not None:
            maximum = max_charge_range(route_allowance(game, unit, route), unit_has_swiftstride(unit))
            if route.distance <= maximum + 1e-5:
                result.append((target, route, None))
    return sorted(result, key=lambda candidate: candidate[1].distance)


async def complete_declarations(game):
    """Test once while the resolver owns the phase, before any reactions (p. 172)."""
    from charge_declarations import ChargeDeclaration
    active = game.player1Units if game.roundCounter.current_player == 1 else game.player2Units
    for unit in list(active):
        if not has_impetuous(unit):
            continue
        if unit.isSkirmisher:
            rule_skipped('Impetuous', unit, 'LEFTOVER: compulsory loose-formation charge selection is not supported')
            continue
        declared = next((entry for entry in game.chargeDeclarations if entry.charger is unit), None)
        targets = [] if declared else legal_targets(game, unit)
        if declared is None and not targets:
            rule_skipped('Impetuous', unit, 'no legal charge target; no Leadership test or compulsory charge')
            continue
        from warband import leadership_for_test
        leadership, general = leadership_for_test(game.psychology, unit, 'Impetuous')
        joined = active_character(unit)
        if joined is not None:
            leadership = max(leadership, int(joined.unit.model.characteristics.get('Ld', leadership)))
        dice = await game.rollLeadershipDice()
        dice = await reroll_leadership(game, unit, 'Impetuous', dice, leadership, game.rollLeadershipDice)
        passed = leadership_passed(sum(dice), leadership)
        rule_log('Impetuous', unit,
                 f'2D6={sum(dice)} vs Ld {leadership}'
                 + (f' (Inspiring Presence: {general.unit.name})' if general else '')
                 + (' -> PASS; may act normally' if passed else ' -> FAIL; must declare a charge'))
        if declared is not None:
            declared.compulsory = not passed
            continue
        if passed:
            continue
        selected = targets[0]
        if len(targets) > 1 and not game.aiControls(unit):
            options = {f'{index}: {target.unit.name} ({route.distance:.2f}")': (target, route, target_index)
                       for index, (target, route, target_index) in enumerate(targets, 1)}
            label = await game.makeChoiceNew(list(options), Vec3(-20, 0, 10), owner=unit,
                                            prompt=f'{unit.unit.name}: compulsory Impetuous charge')
            selected = options.get(label, selected)
        target, route, target_index = selected
        mode = compulsory_mode(game, unit)
        for member in game.movement.movementParticipants(unit):
            member.unit.model.flight_mode = mode
        if unit.unit.model.can_fly():
            rule_log('Fly', unit, f'compulsory Impetuous charge uses {mode}: '
                     f'greatest available M{game.movement.movementAllowance(unit):g} (FAQ v1.5.3)')
        facing = tuple(unit.bodyNP.getHpr())
        entry = ChargeDeclaration(unit, target, tuple(unit.bodyNP.getPos()), facing,
                                  tuple(route.destination), (route.heading + route.wheel, 0, 0),
                                  tuple(route.destination), route.distance,
                                  target_index=target_index, compulsory=True)
        begin_charge_attempt(unit)
        unit.hasMovedThisTurn = True
        unit.marchedThisTurn = unit.wouldMarch = False
        game.chargeDeclarations.append(entry)
        rule_log('Impetuous', unit, f'compulsory charge at {target.unit.name}: '
                 f'{route.distance:.2f}" including wheel; added before charge reactions')