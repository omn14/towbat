"""Movement-phase charge declarations before charge moves (Rulebook pp. 119-121, 167)."""

from dataclasses import dataclass


@dataclass
class ChargeDeclaration:
    charger: object
    defender: object
    origin: tuple
    facing: tuple
    contact_position: tuple
    contact_facing: tuple
    destination: tuple
    distance: float
    reaction: str | None = None
    flank_angle: tuple | None = None
    route: object = None
    target_index: int | None = None
    preview: object = None
    compulsory: bool = False
    charge_dice: object = None


def save_declarations(game):
    return [dict(charger=entry.charger.unitName, defender=entry.defender.unitName,
                 origin=entry.origin, facing=entry.facing,
                 contact_position=entry.contact_position, contact_facing=entry.contact_facing,
                 destination=entry.destination, distance=entry.distance, target_index=entry.target_index,
                 compulsory=entry.compulsory)
            for entry in getattr(game, 'chargeDeclarations', [])]


def restore_declarations(game, state, unit_map):
    game.chargeDeclarations = []
    for record in state.get('charge_declarations', []):
        charger, defender = unit_map.get(record['charger']), unit_map.get(record['defender'])
        if charger is not None and defender is not None:
            values = {key: tuple(record[key]) for key in
                      ('origin', 'facing', 'contact_position', 'contact_facing', 'destination')}
            game.chargeDeclarations.append(ChargeDeclaration(charger, defender, **values,
                                                             distance=record['distance'],
                                                             target_index=record.get('target_index'),
                                                             compulsory=record.get('compulsory', False)))
    stage = state.get('charge_stage')
    if stage is None and state.get('current_phase') == 'MovementPhase':
        stage = 'remaining'
    set_stage(game, stage)


def begin_declarations(game):
    game.chargeDeclarations = []
    set_stage(game, 'declarations')


def set_stage(game, stage):
    game.chargeStage = stage
    from direct.showbase.MessengerGlobal import messenger
    messenger.send('hud-charge-stage', [stage])


def ordinary_move_allowed(game):
    from rules_log import battle_log
    if (game.fsm.state == 'MovementPhase'
            and getattr(game, 'chargeStage', None) in ('declarations', 'resolving', 'blocked')):
        battle_log('Resolve Charges before Remaining Moves.', 'info')
        return False
    return True


def collecting(game):
    return (getattr(game, 'chargeStage', None) == 'declarations'
            and game.fsm.state == 'MovementPhase')


async def choose_reactions(game, entries):
    """Choose one reaction target per defender after all declarations (pp. 120, 167)."""
    from panda3d.core import Vec3
    from magic_items import current_turn
    from rules_log import rule_log

    defenders = []
    for entry in entries:
        if not any(member is entry.defender for member in defenders):
            defenders.append(entry.defender)
        entry.reaction = 'hold'
    for defender in defenders:
        incoming = [entry for entry in entries if entry.defender is defender]
        if (defender.bodyNP.isEmpty() or defender.unit.nmodels <= 0
                or defender.state == 'InCombat'):
            continue
        spent = getattr(defender, 'counterChargeTurn', None)
        if spent is not None and spent == current_turn(game):
            continue
        options = {'hold': None, 'flee': incoming[0]}
        for index, entry in enumerate(incoming, 1):
            origin, facing = Vec3(*entry.origin), Vec3(*entry.facing)
            suffix = f'{index}: {entry.charger.unit.name}'
            if game.combat.counterChargeOption(defender, entry.charger, origin, facing):
                options[f'counter charge {suffix}'] = entry
            shoot = game.combat.standAndShootOption(defender, entry.charger, origin, facing)
            if shoot:
                options[f'stand & shoot {suffix}'] = entry
                if game.combat.fireAndFleeOption(defender, entry.charger, shoot):
                    options[f'fire & flee {suffix}'] = entry
        if defender.state == 'IsFleeing':
            choice = 'flee'
        elif game.aiControls(defender):
            choice = next((label for label in options if label.startswith('counter charge ')),
                          next((label for label in options if label.startswith('stand & shoot ')), 'hold'))
        else:
            choice = await game.makeChoiceNew(
                list(options), Vec3(20, 0, 10), owner=defender,
                prompt=f'{defender.unit.name}: charge reaction',
                detail='\n'.join(f'{index}: {entry.charger.unit.name}'
                                 for index, entry in enumerate(incoming, 1)))
        selected = options[choice]
        if selected is not None:
            selected.reaction = next(action for action in ('counter charge', 'stand & shoot', 'fire & flee', 'flee')
                                     if choice.startswith(action))
        rule_log('Charge Reactions', defender,
                 f'{len(incoming)} declared charge(s): {choice} (pp. 120, 167)')


async def flee_reaction(game, defender, incoming, *, fire_and_flee=False):
    """Flee once from the strongest declared charger, before charge rolls (pp. 120, 133)."""
    from panda3d.core import AsyncFuture, Vec3
    from post_combat import fire_and_flee_roll, flee_roll, flees_from
    from special_rules import board_edge_distance
    from psychology import unit_strength_total
    from rules_log import rule_log, rule_skipped

    if getattr(defender, 'fledThisPhase', False):
        rule_skipped('The Limits of Endurance', defender, 'already fled this phase; 0" and no additional pivot (p. 133)')
        return
    candidates = [(entry.charger, unit_strength_total(entry.charger)) for entry in incoming]
    source = flees_from(candidates)
    position = defender.bodyNP.getPos()
    bonus = await game.combat.swiftstrideChoice(
        defender, 'flee', distance_to_edge=board_edge_distance(position.x, position.y))
    dice, rolls = await game.combat.rullTerninger(3 if bonus else 2, bonus)
    try:
        distance = (fire_and_flee_roll if fire_and_flee else flee_roll)(rolls)
    finally:
        for die in dice:
            die.remove(game.world)
    direction = Vec3(position - source.bodyNP.getPos())
    direction.z = 0
    direction.normalize()
    defender.fledThisPhase = True
    defender.request('IsFleeing')
    rule_log('Fire & Flee' if fire_and_flee else 'Flee', defender,
             f'{len(incoming)} chargers; flees from {source.unit.name} '
             f'(US {unit_strength_total(source)}), dice {rolls} -> {distance}" (p. 120)')
    finished = AsyncFuture()
    game.psychology._start_flee_move(defender, direction, distance, 'flee', lambda: finished.set_result(None))
    await finished


async def resolve_declarations(game):
    """Close declarations once; no ordinary move or second resolver may interleave."""
    from rules_log import battle_log
    from panda3d.core import Vec3
    from first_charge import finish_charge_attempt

    if not collecting(game):
        return
    set_stage(game, 'resolving')
    entries = game.chargeDeclarations
    counter_defenders = []
    completed = False
    try:
        from impetuous import complete_declarations
        await complete_declarations(game)
        await choose_reactions(game, entries)
        for entry in entries:
            if entry.reaction == 'counter charge':
                counter_defenders.append(entry.defender)
                await game.combat.counterChargeInterval(
                    entry.charger, entry.defender, Vec3(*entry.origin), Vec3(*entry.facing), defer_charge=True)
                entry.reaction = 'hold'
            elif entry.reaction in ('stand & shoot', 'fire & flee'):
                fire_and_flee = entry.reaction == 'fire & flee'
                shoot = game.combat.standAndShootOption(
                    entry.defender, entry.charger, Vec3(*entry.origin), Vec3(*entry.facing))
                if shoot:
                    await game.combat.standAndShoot(entry.defender, entry.charger, shoot.weapon, shoot.distance,
                                                  target_boxes=getattr(shoot, 'target_boxes', None))
                if fire_and_flee:
                    await flee_reaction(game, entry.defender,
                                        [pending for pending in entries if pending.defender is entry.defender],
                                        fire_and_flee=True)
                entry.reaction = 'hold'
            elif entry.reaction == 'flee':
                await flee_reaction(game, entry.defender,
                                    [pending for pending in entries if pending.defender is entry.defender])
                entry.reaction = 'hold'
        while entries:
            entry = entries[0]
            if len(entries) > 1 and not game.aiControls(entry.charger):
                options = {f'{index}: {pending.charger.unit.name} -> {pending.defender.unit.name}': pending
                           for index, pending in enumerate(entries, 1)}
                selected = await game.makeChoiceNew(list(options), Vec3(-20, 0, 10), owner=entry.charger,
                                                    prompt='Move which charge next?')
                entry = options[selected]
            await game.combat.resolveDeclaredCharge(entry)
            entries.remove(entry)
        completed = True
    finally:
        for defender in counter_defenders:
            finish_charge_attempt(defender)
        if completed:
            set_stage(game, 'remaining')
            battle_log('Charge moves complete. Remaining Moves.', 'info')
        else:
            set_stage(game, 'blocked')
            battle_log('Charge resolution interrupted; reload the declaration-stage save before continuing.', 'info')


def queue_charge(game, charger, defender, origin, facing):
    """Reserve a declared charger at its starting pose, not its contact preview (p. 119)."""
    if any(entry.charger is charger for entry in game.chargeDeclarations):
        return None
    entry = ChargeDeclaration(
        charger, defender, tuple(origin), tuple(facing),
        tuple(charger.bodyNP.getPos()), tuple(charger.bodyNP.getHpr()),
        tuple(game.playerNP.getPos()), float(game.moveArceDistance))
    preview = getattr(charger, 'formedSkirmishCharge', None)
    if preview is not None:
        entry.target_index = preview.route.target_index
    game.chargeDeclarations.append(entry)
    charger.bodyNP.setPos(*entry.origin)
    charger.bodyNP.setHpr(*entry.facing)
    charger.bodyNP.node().setTransformDirty()
    charger.hasMovedThisTurn = True
    charger.isChargingMove = False
    charger.marchedThisTurn = charger.wouldMarch = False
    game.autoCharge = game.autoHold = False
    return entry