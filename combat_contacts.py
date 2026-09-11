"""Model-base fighting and supporting reach (Rulebook pp. 145-147, FAQ v1.5.3)."""

from dataclasses import dataclass

from psychology import obb_distance

CONTACT_EPSILON = 0.01


@dataclass(frozen=True)
class FightingPosition:
    index: int
    distance: float
    contact: bool
    fighting: bool
    supporting: bool

    def attacks(self, characteristic, movement, *, support=False, count=1):
        if self.contact:
            return characteristic * count
        if self.distance <= movement + CONTACT_EPSILON and (self.fighting or (support and self.supporting)):
            return min(1, characteristic) * count
        return 0


def fighting_positions(boxes, enemies, files, *, facing='front', press=False, slots=None):
    """Classify rows/files from actual contact; support is front-only (pp. 145, 190)."""
    files = max(1, files)
    slots = list(range(len(boxes))) if slots is None else slots
    distances = [min((obb_distance(box, enemy) for enemy in enemies), default=float('inf')) for box in boxes]
    rows = [slot // files for slot in slots]
    columns = [slot % files for slot in slots]
    lines = columns if facing in ('left', 'right') else rows
    touched = {lines[index] for index, distance in enumerate(distances) if distance <= CONTACT_EPSILON}
    inward = -1 if facing in ('rear', 'right') else 1
    fighting = touched | {line + inward for line in touched} if press else touched
    support = {line + 1 for line in fighting} - fighting if facing == 'front' else set()
    return [FightingPosition(index, distances[index], distances[index] <= CONTACT_EPSILON,
                             line in fighting, line in support)
            for index, line in enumerate(lines)]


class CombatContactSnapshot:
    """Keep pre-casualty base positions while rendered removals are deferred (p. 146)."""

    def __init__(self, hosts):
        from command_groups import living_command
        from scouts import model_base_boxes
        self.formations = {}
        for host in hosts:
            boxes = model_base_boxes(host)
            children = list(host.model.getChildren())[:host.unit.nmodels]
            files = max(1, host.unit.files)
            slots = [round(-child.getY() / host.modelHeight) * files
                     + round(child.getX() / host.modelWidth) for child in children]
            joined = getattr(host, 'joinedCharacter', None)
            if joined is not None and len(boxes) > len(children):
                slots.append(getattr(host, 'characterSlot', 0) or 0)
            command = {index: entry for index, entry in enumerate(living_command(host))}
            self.formations[id(host)] = (boxes, slots, command, len(children), joined)

    def positions(self, host, target):
        boxes, slots, _, _, _ = self.formations[id(host)]
        enemy_boxes = self.formations[id(target)][0]
        facing = 'front'
        try:
            facing = host.isInCombatFlank[host.isInCombatWith.index(target)]
        except (AttributeError, ValueError, IndexError):
            pass
        if facing == 'flank':
            facing = 'left' if target.bodyNP.getPos(host.bodyNP).x < 0 else 'right'
        press = host.unit.model.troop_type_rule('Press of Battle') and not getattr(host, 'chargedThisTurn', False)
        return facing, fighting_positions(boxes, enemy_boxes, host.unit.files, facing=facing,
                                           press=press, slots=slots)

    def can_challenge(self, host, candidate, enemies):
        """A candidate must be within or adjacent to a fighting rank (p. 210)."""
        _, slots, command, initial, joined = self.formations[id(host)]
        if candidate is joined:
            index = initial if len(slots) > initial else None
        elif candidate is host:
            index = 0 if slots else None
        else:
            index = next((index for index, entry in command.items()
                          if entry is getattr(candidate, 'command_entry', None)), None)
        if index is None:
            return False
        files = max(1, host.unit.files)
        for enemy in enemies:
            facing, positions = self.positions(host, enemy)
            lines = [slot % files if facing in ('left', 'right') else slot // files for slot in slots]
            if any(place.fighting and abs(lines[index] - lines[place.index]) <= 1 for place in positions):
                return True
        return False

    def attacks(self, part, models, challenge=None):
        """Full A in contact, one per out-of-contact part, no supporting mounts."""
        from battleFunctions import attack_characteristic
        from rules_log import rule_log, rule_skipped
        host, profile = part.host, part.profile
        boxes, slots, command, initial, joined = self.formations[id(host)]
        charged = bool(getattr(host, 'chargedThisTurn', False))
        _, positions = self.positions(host, part.target)
        champion_indices = {index for index, entry in command.items() if entry.get('role') == 'champion'}
        duelling = {id(getattr(member, 'command_entry', None)) for member in challenge.participants()} if challenge else set()
        lost_champions = sum(not command[index].get('active', True) for index in champion_indices)
        casualties = max(0, initial - models - lost_champions)
        ordinary = [index for index in range(initial) if index not in champion_indices]
        ordinary.sort(key=lambda index: (not positions[index].fighting, not positions[index].supporting, slots[index]))
        excluded = set(ordinary[:casualties])
        excluded.update(index for index in champion_indices if not command[index].get('active', True)
                        or command[index].get('retired', False) or id(command[index]) in duelling)
        support = part.role in ('main', 'champion', 'character') and profile.fights_in_extra_rank(charged=charged)
        movement_profile = joined.unit.model if part.role == 'character' and joined is not None else host.unit.model
        movement = movement_profile.get_movement()
        characteristic = attack_characteristic(profile, charged=charged,
                                               inches=float(getattr(host, 'chargeDistance', 0) or 0))
        count = part.count
        if part.role == 'main':
            indices = [index for index in range(initial) if index not in champion_indices]
        elif part.role == 'champion':
            indices = [index for index in champion_indices if command[index] is part.entry]
        elif part.role == 'character':
            indices = [initial] if joined is not None and joined.unit.nmodels > 0 and len(boxes) > initial else []
            if part.fighter is not joined:
                support = False
                count = part.fighter.unit.nmodels
        else:
            indices = list(range(initial))
        quotas = {index: positions[index].attacks(characteristic, movement, support=support, count=count)
                  for index in indices if index not in excluded}
        total = sum(quotas.values())
        contacts = sum(positions[index].contact for index in quotas)
        weapon = profile.equipedWeapon or {}
        if any('fight in extra rank' in str(rule).casefold() for rule in weapon.get('special_rules', [])):
            supporting = sum(attacks for index, attacks in quotas.items() if positions[index].supporting)
            logger = rule_log if supporting else rule_skipped
            logger('Fight in Extra Rank', host,
                   f'{profile.name}, {weapon.get("name")}: charged={charged}, ground M{movement:g}; '
                   f'{supporting} supporting attacks within reach after casualties (pp. 145, 169, 215)')
        if host.unit.model.troop_type_rule('Press of Battle'):
            _, ordinary_positions = self.positions_without_press(host, part.target)
            extra = sum(attacks for index, attacks in quotas.items()
                        if positions[index].fighting and not ordinary_positions[index].fighting)
            logger = rule_log if extra else rule_skipped
            logger('Press of Battle', host,
                   f'{profile.name}: charged={charged}, ground M{movement:g}; '
                   f'{extra} attacks from the additional fighting rank after casualties (p. 190)')
        logger = rule_log if total else rule_skipped
        logger('Fighting Rank', host,
               f'{profile.name} ({part.role}): {contacts} bases in contact, ground M{movement:g}, '
               f'{casualties} earlier ordinary casualties -> {total} attacks (pp. 145-146)')
        return total

    def positions_without_press(self, host, target):
        facing, _ = self.positions(host, target)
        boxes, slots, _, _, _ = self.formations[id(host)]
        return facing, fighting_positions(boxes, self.formations[id(target)][0], host.unit.files,
                                           facing=facing, slots=slots)