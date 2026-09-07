"""Rallying Cry in the Command sub-phase (Rulebook pp. 117, 175, 202)."""

from characters import side_of
from panda3d.core import Vec3
from psychology import (PsychologySystem, _stat_int, command_range,
                        is_battle_standard_unit, is_character_unit, obb_distance)
from rules_log import battle_log, rule_log, rule_skipped


def has_rallying_cry(unit):
    profile = unit.unit.model
    return bool(getattr(profile, 'is_rallying_cry', lambda: False)())


def command_radius(character):
    """Ordinary characters use Ld inches, General/BSB use 12/18 (p. 202)."""
    if getattr(character, 'isGeneral', False) or is_battle_standard_unit(character):
        return command_range(character)
    return float(max(0, _stat_int(character.unit.model.characteristics, 'Ld', 0)))


def source_reason(game, character):
    """Non-logging eligibility query; log only when an action is attempted."""
    if not is_character_unit(character) or not has_rallying_cry(character):
        return 'requires a character with Rallying Cry'
    if game.fsm.state != 'StrategyPhase' or getattr(game, 'strategyCommandDone', False):
        return 'only available during the Command sub-phase'
    if side_of(game, character, default=None) != game.roundCounter.current_player:
        return 'not this character\'s turn'
    if character.bodyNP.isEmpty() or character.unit.nmodels <= 0:
        return 'character is not on the battlefield'
    host = getattr(character, 'hostUnit', None) or character
    if host.state == 'IsFleeing':
        return 'character or joined unit is fleeing'
    if getattr(host, 'isInCombat', False) or host.state == 'InCombat':
        return 'character or joined unit is engaged in combat'
    if getattr(character, 'retiredFromCombat', False):
        return 'character has retired from the fighting rank'
    if getattr(character, 'usedRallyingCry', False):
        return 'already used Rallying Cry this Command sub-phase'
    return None


def target_distance(character, target):
    return obb_distance(PsychologySystem._unit_box(character),
                        PsychologySystem._unit_box(target))


def target_reason(game, character, target):
    reason = source_reason(game, character)
    if reason is not None:
        return reason
    if target.bodyNP.isEmpty() or target.unit.nmodels <= 0:
        return 'target is not on the battlefield'
    if side_of(game, target, default=None) != side_of(game, character, default=None):
        return 'target is not friendly'
    if getattr(target, 'hostUnit', None) is not None:
        return 'nominate the fleeing unit, not its joined character'
    if target.state != 'IsFleeing' or getattr(target, 'isInCombat', False):
        return 'target is not a fleeing, unengaged unit'
    distance, radius = target_distance(character, target), command_radius(character)
    if distance > radius:
        return f'target {distance:.2f}" away exceeds Command range {radius:g}"'
    return None


def command_characters(game):
    return [unit for unit in game.units
            if is_character_unit(unit) and has_rallying_cry(unit)
            and side_of(game, unit, default=None) == game.roundCounter.current_player]


def begin_command(game):
    """Start a fresh Command window, not a return from spell selection (p. 117)."""
    characters = command_characters(game)
    for character in characters:
        character.usedRallyingCry = False
    game.strategyCommandDone = not bool(characters)
    game.rallyingCryBusy = False
    if characters:
        battle_log('Strategy: Command', 'info')


def finish_command(game):
    if getattr(game, 'rallyingCryBusy', False) or getattr(game, 'awaitingChoice', False):
        return False
    for character in command_characters(game):
        if not getattr(character, 'usedRallyingCry', False):
            rule_skipped('Rallying Cry', character,
                         source_reason(game, character) or 'Command ended without a nomination')
    game.strategyCommandDone = True
    battle_log('Strategy: Conjuration / Rally', 'info')
    return True


async def use_rallying_cry(game, character, target):
    """The TARGET makes the Rally test; no extra character test (p. 175).

    The FAQ's personal-test restriction does not add a test to this rule.
    Failure leaves the normal Rally sub-phase attempt available.
    """
    if getattr(game, 'rallyingCryBusy', False):
        return False
    reason = target_reason(game, character, target)
    if reason is not None:
        rule_skipped('Rallying Cry', character, reason)
        return False
    character.usedRallyingCry = True
    game.rallyingCryBusy = True
    selected = game.unitToMove
    try:
        distance, radius = target_distance(character, target), command_radius(character)
        rule_log('Rallying Cry', character,
                 f'nominates {target.unitName}, {distance:.2f}" away within {radius:g}" Command range; '
                 'one use spent; target makes an immediate Rally test')
        game.unitToMove = target
        game.refreshSelectedUnit()
        rallied = await game.rallyUnit(target, command=True)
        rule_log('Rallying Cry', character,
                 f'{target.unitName}: ' + ('rallied; no charge this turn' if rallied else
                 'Rally failed; may still attempt normal Rally'))
        return rallied
    finally:
        game.rallyingCryBusy = False
        game.unitToMove = selected
        game.refreshSelectedUnit()


async def choose_rallying_cry(game, selected):
    character = selected if has_rallying_cry(selected) and is_character_unit(selected) else (
        getattr(selected, 'joinedCharacter', None))
    if character is None or not has_rallying_cry(character):
        battle_log('Command: this unit has no character with Rallying Cry.', 'info')
        return
    reason = source_reason(game, character)
    if reason is not None:
        rule_skipped('Rallying Cry', character, reason)
        return
    friendlies = game.player1Units if side_of(game, character) == 1 else game.player2Units
    targets = [unit for unit in friendlies if target_reason(game, character, unit) is None]
    if not targets:
        rule_skipped('Rallying Cry', character,
                     f'no fleeing friendly unit within {command_radius(character):g}" Command range')
        return
    if game.aiControls(character):
        target = max(targets, key=lambda unit: unit.unit.nmodels)
    else:
        choices = {f'{index + 1}. {unit.unitName}': unit for index, unit in enumerate(targets)}
        descriptions = {label: f'{target_distance(character, unit):.2f}" away; '
                        f'{unit.unit.nmodels} models; Ld {game.psychology.leadership_of(unit)[0]}'
                        for label, unit in choices.items()}
        choice = await game.makeChoiceNew(
            list(choices), Vec3(0, 0, 10), owner=character, cancellable=True,
            prompt=f'{character.unitName}: Rallying Cry', descriptions=descriptions,
            detail=f'Command range {command_radius(character):g}"')
        target = choices.get(choice)
        if target is None:
            rule_skipped('Rallying Cry', character, 'nomination cancelled; use retained')
            return
    await use_rallying_cry(game, character, target)


async def ai_command(game):
    for character in command_characters(game):
        await choose_rallying_cry(game, character)
    finish_command(game)