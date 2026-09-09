"""Ordinary command groups, distinct from characters and BSBs (pp. 198-201)."""

from copy import deepcopy
from types import SimpleNamespace

from rules_log import rule_log, rule_skipped


def regiment(member):
    return getattr(member, 'unit', member)


def install_command(member, selections):
    """Promotions occupy existing bodies, not extra models (Rulebook p. 198)."""
    group = regiment(member)
    group.command = deepcopy(selections)
    group.command_fighters = {}
    for entry in group.command:
        entry.setdefault('active', True)
        entry.setdefault('wounds', 0)


def living_command(member):
    """Preserve champions, then standards, then musicians (pp. 199-201)."""
    group = regiment(member)
    entries = getattr(group, 'command', [])
    priority = {'champion': 0, 'standard_bearer': 1, 'musician': 2}
    active = [entry for entry in entries if entry.get('active', True)]
    active.sort(key=lambda entry: priority.get(entry.get('role'), 3))
    return active[:max(0, group.nmodels)]


def has_command(member, role, *, front=False):
    group = regiment(member)
    entries = living_command(member)
    if front:
        slots = command_positions(member)
        entries = [entry for entry in entries if slots.get(id(entry), group.files) < group.files]
    return any(entry.get('role') == role for entry in entries)


def command_positions(member, files=None):
    """Standards take centre; other command lead from adjacent slots (p. 198)."""
    group = regiment(member)
    files = max(1, group.files if files is None else files)
    entries = living_command(member)
    priority = {'standard_bearer': 0, 'champion': 1, 'musician': 2}
    ordered = sorted(entries, key=lambda entry: priority.get(entry.get('role'), 3))
    slots = sorted(range(max(group.nmodels, len(entries))),
                   key=lambda slot: (slot // files, abs(slot % files - files // 2), slot))
    positions = {}
    for entry in ordered:
        if entry.get('retired', False):
            continue
        positions[id(entry)] = slots.pop(0)
    for entry in ordered:
        if not entry.get('retired', False):
            continue
        slot = max(slots) if slots and max(slots) >= files else max(files, group.nmodels)
        positions[id(entry)] = slot
        if slot in slots:
            slots.remove(slot)
    return positions


def remove_command_casualties(member, *, log=True):
    """Bank losses so later restores cannot resurrect command (pp. 199-201)."""
    survivors = living_command(member)
    for entry in getattr(regiment(member), 'command', []):
        if entry.get('active', True) and not any(entry is kept for kept in survivors):
            entry['active'] = False
            if log:
                rule_log('Command Group', member,
                         f"{entry.get('name', entry['role'])} lost; "
                         f"{regiment(member).nmodels} models remain")


def standard_bonus(members, *, log=False):
    """A side's ordinary standard bonus is separate from its BSB (pp. 153, 200)."""
    bearers = [member for member in members if has_command(member, 'standard_bearer')]
    if bearers and log:
        rule_log('Standard Bearer', bearers[0],
                 f"{len(bearers)} surviving unit standard(s) -> +1 combat result")
    return int(bool(bearers))


def musician_bonus(first, second, first_score, second_score, *, log=False):
    """Only break a final tied score; opposing musicians cancel (p. 201)."""
    first_music = [member for member in first if has_command(member, 'musician', front=True)]
    second_music = [member for member in second if has_command(member, 'musician', front=True)]
    if first_score != second_score or bool(first_music) == bool(second_music):
        if log:
            reason = (f"scores {first_score}:{second_score} are not tied" if first_score != second_score
                      else "both sides have musicians; bonuses cancel")
            for member in first_music + second_music:
                rule_skipped('Musician', member, reason)
        return 0, 0
    winner = (first_music or second_music)[0]
    if log:
        rule_log('Musician', winner,
                 f"scores tied {first_score}:{second_score}; no opposing musician -> +1 combat result")
    return int(bool(first_music)), int(bool(second_music))


def musician_leadership(member, leadership, context, *, log=False):
    """Steadying Rhythm and Quick Time, capped at Ld 10 (Rulebook p. 201)."""
    if context not in {'rally', 'march'} or not has_command(member, 'musician'):
        return leadership
    improved = min(10, leadership + 1) if leadership < 10 else leadership
    if log:
        if improved != leadership:
            rule_log('Musician', member, f"{context} Leadership {leadership} -> {improved}")
        else:
            rule_skipped('Musician', member, f"{context} Leadership {leadership} already at least 10")
    return improved


class Champion:
    """A promoted body with personal wounds, not a joined character (p. 199)."""

    def __init__(self, host, entry, profile, key):
        self.command_host = host
        self.command_entry = entry
        self.unitName = f'{host.unitName}::command::{key}'
        self.unit = SimpleNamespace(name=profile.name, model=profile, nmodels=1, files=1, ranks=1)

    @property
    def woundsOnModel(self):
        return self.command_entry.get('wounds', 0)

    @woundsOnModel.setter
    def woundsOnModel(self, value):
        self.command_entry['wounds'] = value

    @property
    def retiredFromCombat(self):
        return self.command_entry.get('retired', False)

    @retiredFromCombat.setter
    def retiredFromCombat(self, value):
        self.command_entry['retired'] = value


def champions(member, *, include_retired=False):
    group = regiment(member)
    fighters = getattr(group, 'command_fighters', {})
    result = []
    for index, entry in enumerate(getattr(group, 'command', [])):
        if entry not in living_command(member) or entry.get('role') != 'champion':
            continue
        if entry.get('retired', False) and not include_retired:
            continue
        key = entry.get('selection_ref', str(index))
        profile = getattr(group, 'command_models', {}).get(key)
        if profile is None:
            continue
        if key not in fighters:
            fighters[key] = Champion(member, entry, profile, key)
        result.append(fighters[key])
    group.command_fighters = fighters
    return result


def capture_standard(game, loser, winner):
    """Destroyed in combat or run down: an irrecoverable 50 VP trophy (pp. 200, 286)."""
    from characters import side_of
    trophies = getattr(game, 'capturedStandards', None)
    if trophies is None:
        trophies = game.capturedStandards = []
    for index, entry in enumerate(getattr(regiment(loser), 'command', [])):
        if entry.get('role') != 'standard_bearer' or not entry.get('active', True) or entry.get('captured'):
            continue
        entry['active'] = False
        entry['captured'] = True
        entry['captured_by'] = side_of(game, winner)
        trophies.append({'unit': loser.unitName, 'selection_ref': entry.get('selection_ref', str(index)),
                         'captured_by': entry['captured_by'], 'victory_points': 50})
        rule_log('Trophies of War', winner,
                 f'{loser.unitName} standard captured permanently -> +50 Victory Points (pp. 200, 286)')