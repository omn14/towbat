"""Flight selection and compulsory ground movement (Rulebook p. 170)."""

from contextlib import contextmanager
from functools import wraps

from rules_log import rule_log, rule_skipped


def selectable(game, unit):
    from characters import side_of
    from free_pivot import pending
    return (unit is not None and unit.unit.model.can_fly()
            and game.fsm.state in ('MovementPhase', 'ReserveMovePhase')
            and side_of(game, unit, None) == game.roundCounter.current_player
            and unit.state == 'Idle' and not unit.hasMovedThisTurn
            and not getattr(game, 'awaitingChoice', False) and not pending(game)
            and getattr(game, 'chargeStage', None) not in ('resolving', 'blocked')
            and not any(entry.charger is unit for entry in getattr(game, 'chargeDeclarations', [])))


def set_mode(game, unit, mode):
    if mode not in ('fly', 'ground') or not selectable(game, unit):
        if unit is not None:
            rule_skipped('Fly', unit, 'movement mode cannot change after committing a move or charge')
        return False
    for member in game.movement.movementParticipants(unit):
        member.unit.model.flight_mode = mode
    movement = game.movement.movementAllowance(unit)
    rule_log('Fly', unit, f'chooses {mode} movement: M{movement:g} (p. 170)')
    game.setGroundOverlay(False)
    game.refreshSelectedUnit()
    return True


def compulsory_mode(game, unit):
    """Use the greatest available Movement for compulsory charges (p. 170; FAQ v1.5.3)."""
    profiles = [member.unit.model for member in game.movement.movementParticipants(unit)]
    if not all(profile.can_fly() and not getattr(profile, '_groundMovement', False) for profile in profiles):
        return 'ground'
    return ('fly' if min(profile.get_fly_movement() for profile in profiles)
            >= min(profile.get_movement() for profile in profiles) else 'ground')


def compulsory_preview(method):
    @wraps(method)
    def preview(game, unit, *args, **kwargs):
        profiles = [member.unit.model for member in game.movement.movementParticipants(unit)]
        previous = [getattr(profile, 'flight_mode', 'fly') for profile in profiles]
        mode = compulsory_mode(game, unit)
        for profile in profiles:
            profile.flight_mode = mode
        try:
            return method(game, unit, *args, **kwargs)
        finally:
            for profile, value in zip(profiles, previous):
                profile.flight_mode = value
    return preview


def refresh_controls(game, unit):
    from direct.gui.DirectGui import DirectRadioButton, DGG
    import gui_theme as theme

    buttons = getattr(game, 'flightButtons', None)
    if buttons is None:
        game.flightModeChoice = ['fly']
        buttons = []
        for index, mode in enumerate(('fly', 'ground')):
            button = DirectRadioButton(
                parent=game.a2dTopLeft, text=mode.title(), variable=game.flightModeChoice,
                value=[mode], text_font=theme.get_font(), text_fg=theme.BTN_TEXT,
                text_scale=.030, text_pos=(.14, -.023), indicatorValue=0,
                frameColor=theme.BTN_RED, frameSize=(0, .30, -.05, .012),
                relief=DGG.FLAT, pos=(.74 + index * .31, 0, -.07),
                command=lambda selected=mode: set_mode(game, game.unitToMove, selected))
            buttons.append(button)
        for button in buttons:
            button.setOthers(buttons)
        game.flightButtons = buttons
    game.flightModeChoice[0] = getattr(unit.unit.model, 'flight_mode', 'fly') if unit else 'fly'
    for button in buttons:
        button.setIndicatorValue()
        button.show() if selectable(game, unit) else button.hide()


@contextmanager
def grounded(unit):
    """Following up and pursuing cannot use Fly, even across terrain (p. 170)."""
    model = unit.unit.model
    previous = getattr(model, '_groundMovement', False)
    if model.is_flying():
        rule_log('Fly', unit, 'Follow Up/pursuit must use ground movement; cannot pass over obstacles (p. 170)')
    model._groundMovement = True
    try:
        yield
    finally:
        model._groundMovement = previous


def ground_pursuit(method):
    @wraps(method)
    async def resolve(resolver, unit, *args, **kwargs):
        with grounded(unit):
            return await method(resolver, unit, *args, **kwargs)
    return resolve