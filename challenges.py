"""
Challenges — Rulebook p. 210-211.

A duel fought inside a combat between two characters. The rules with no
Panda3D in them live here; `combat_resolution` runs the exchange and the fight.

Duellists direct all attacks at each other and cannot be targeted by outside
attacks (p. 211). Their Initiative steps still share the combat's clock because
incidental Miscast damage can cross between the duel and the surrounding units.
"""

from characters import get_joined_character, get_joined_characters, is_character
from command_groups import Champion

# Overkill is capped at five bonus points (p. 211).
MAX_OVERKILL = 5


class KingGuard(Champion):
    """An ordinary White Lion nominated for a duel, not a promotion (FoF p. 163)."""

    def __init__(self, host, entry):
        super().__init__(host, entry, host.unit.model, 'kings_guard')
        self.unitName = f'{host.unitName}::kings_guard'
        self.source_command = None

    @property
    def guard_index(self):
        return self.command_entry.get('index', 0)

    @guard_index.setter
    def guard_index(self, value):
        self.command_entry['index'] = value

    def mark_slain(self):
        self.command_entry['active'] = False
        self.unit.nmodels = 0
        if self.source_command is not None:
            self.source_command['active'] = False


def guard_fighters(host):
    """Selected and retired ordinary duellists, separate from command (FoF p. 163)."""
    retired = getattr(host.unit, 'retired_king_guards', None)
    if retired is None:
        retired = [KingGuard(host, entry) for entry in getattr(host.unit, 'retired_guard_states', [])]
        host.unit.retired_king_guards = retired
    current = getattr(host.unit, 'king_guard_fighter', None)
    return [*retired, *([current] if current is not None else [])]


def king_guard(host, *, restore=False):
    """One representative of the identical eligible rank-and-file models (FoF p. 163)."""
    if not restore:
        general = next((member for member in get_joined_characters(host)
                if getattr(member, 'isGeneral', False)), None)
        if (general is None or general.unit.nmodels <= 0 or not getattr(general, 'isGeneral', False)
                or is_retired(general) or not any(rule.get('name') == "King's Guard"
                                                 for rule in host.unit.model.special_rules)):
            return None
    from command_groups import living_command
    command = living_command(host)
    fighters = guard_fighters(host)
    retired_indices = {fighter.guard_index for fighter in fighters if fighter.retiredFromCombat}
    available = [index for index in range(host.unit.nmodels)
                 if index not in retired_indices
                 and (index >= len(command) or command[index].get('role') != 'champion')]
    if not available:
        return None
    fighter = getattr(host.unit, 'king_guard_fighter', None)
    if fighter is not None and fighter.retiredFromCombat and not restore:
        host.unit.retired_king_guards.append(fighter)
        host.unit.retired_guard_states = [guard.command_entry for guard in host.unit.retired_king_guards]
        host.unit.king_guard_state = None
        fighter = None
    if fighter is None or not fighter.command_entry.get('active', True):
        entry = getattr(host.unit, 'king_guard_state', None) or {}
        if not entry.get('active', True):
            entry = {}
        entry = {'active': True, 'wounds': 0, **entry}
        fighter = KingGuard(host, entry)
        host.unit.king_guard_fighter = fighter
        host.unit.king_guard_state = entry
    enemies = [enemy for enemy in getattr(host, 'isInCombatWith', []) if enemy.unit.nmodels > 0]
    if getattr(host, 'isInCombat', False) and enemies and hasattr(host, 'modelWidth'):
        from combat_contacts import CombatContactSnapshot
        contacts = CombatContactSnapshot([host, *enemies])
        eligible = []
        for index in available:
            fighter.guard_index = index
            if contacts.can_challenge(host, fighter, enemies):
                eligible.append(index)
        available = eligible
    if not available:
        return None
    fighter.guard_index = next((index for index in available if index >= len(command)), available[0])
    fighter.source_command = command[fighter.guard_index] if fighter.guard_index < len(command) else None
    return fighter


class Challenge:
    """One duel, from the moment it is issued until it resolves.

    Carries across turns: if both survive and the combat continues, so does
    the challenge (To The Death!, p. 211).
    """

    def __init__(self, challenger, host, accepter=None, accepter_host=None):
        self.challenger = challenger
        self.host = host
        self.accepter = accepter
        self.accepter_host = accepter_host
        self.refused = False
        self.retired = None
        self.rounds = 0

    @property
    def answered(self) -> bool:
        return self.accepter is not None

    def participants(self):
        return [m for m in (self.challenger, self.accepter) if m is not None]

    def involves(self, model) -> bool:
        return model is not None and model in self.participants()

    def opponent_of(self, model):
        if model is self.challenger:
            return self.accepter
        if model is self.accepter:
            return self.challenger
        return None

    def hosts(self):
        return [h for h in (self.host, self.accepter_host) if h is not None]


def duellists(unit):
    """Living, non-retired character/champion candidates (pp. 199, 210)."""
    if unit is None:
        return []
    if getattr(unit.unit, 'nmodels', 0) <= 0:
        return []
    if is_character(unit):
        candidates = [] if is_retired(unit) else [unit]
    else:
        candidates = []
        candidates.extend(joined for joined in get_joined_characters(unit)
                          if joined.unit.nmodels > 0 and not is_retired(joined))
        from command_groups import champions
        candidates.extend(champions(unit))
        guard = king_guard(unit)
        if guard is not None and not is_retired(guard):
            candidates.append(guard)
    enemies = [enemy for enemy in getattr(unit, 'isInCombatWith', []) if enemy.unit.nmodels > 0]
    if getattr(unit, 'isInCombat', False) and enemies and hasattr(unit, 'modelWidth'):
        from combat_contacts import CombatContactSnapshot
        contacts = CombatContactSnapshot([unit, *enemies])
        candidates = [candidate for candidate in candidates if contacts.can_challenge(unit, candidate, enemies)]
    return candidates


def duellist(unit):
    """Default participant for legacy callers and AI (pp. 199, 210)."""
    return next(iter(duellists(unit)), None)


def is_retired(model) -> bool:
    """True if *model* has retired from combat by refusing a challenge."""
    return bool(getattr(model, 'retiredFromCombat', False))


def can_issue(unit) -> bool:
    """Whether *unit* has a model that can issue a challenge (p. 210)."""
    return duellist(unit) is not None


def can_accept(unit) -> bool:
    """Whether *unit* has a model that can accept (p. 210).

    A unit with no character leaves the challenge unanswered.
    """
    return duellist(unit) is not None


def surrounded(unit) -> bool:
    """Whether *unit* is engaged in all four of its arcs (p. 211).

    The engine records only front, flank and rear per engagement — it cannot
    tell a left flank from a right one — so this asks for a front, a rear and
    two flanks, which is as close as the recorded facings come.
    """
    facings = list(getattr(unit, 'isInCombatFlank', None) or [])
    return ('front' in facings and 'rear' in facings
            and facings.count('flank') >= 2)


def refusal_barred(model, host):
    """Why *model* cannot refuse a challenge, or None if it may (p. 211)."""
    if host is None or model is host:
        return "it is not part of a unit"
    if getattr(host.unit, 'nmodels', 0) <= 1:
        return "it is the last model in its unit"
    if surrounded(host):
        return "its unit is engaged in all four arcs"
    return None


def may_refuse(model, host) -> bool:
    return refusal_barred(model, host) is None


def wounds_remaining(model) -> int:
    """Wounds *model* has left before this round's blows land."""
    unit = getattr(model, 'unit', None)
    chars = unit.model.characteristics if unit is not None else {}
    try:
        total = int(chars.get('W', 1))
    except (TypeError, ValueError):
        total = 1
    return max(0, total - int(getattr(model, 'woundsOnModel', 0)))


def overkill_bonus(unsaved_wounds: int, wounds_left: int) -> int:
    """Combat result for cutting a rival down with room to spare (p. 211).

    Each unsaved wound beyond what the loser had left is worth a point, to a
    maximum of five. Nothing is earned unless the rival actually falls.
    """
    if unsaved_wounds < wounds_left:
        return 0
    return max(0, min(unsaved_wounds - wounds_left, MAX_OVERKILL))


def find_challenge(game, *units):
    """A live challenge involving any of *units*, or None."""
    for challenge in getattr(game, 'challenges', None) or []:
        for unit in units:
            if unit is not None and unit in challenge.hosts():
                return challenge
    return None


def add_challenge(game, challenge):
    if getattr(game, 'challenges', None) is None:
        game.challenges = []
    game.challenges.append(challenge)
    from rules_log import rule_log
    for participant in challenge.participants():
        if isinstance(participant, KingGuard):
            rule_log("King's Guard", participant.command_host,
                     f'General joined: one {participant.unit.name} fights the challenge with its normal '
                     f'A{participant.unit.model.characteristics.get("A", 1)} and W'
                     f'{participant.unit.model.characteristics.get("W", 1)} (FoF p. 163)')


def end_challenge(game, challenge):
    """Take a challenge out of play once a duellist falls or the combat ends."""
    challenges = getattr(game, 'challenges', None) or []
    if challenge in challenges:
        challenges.remove(challenge)
