"""Selected High Magic Enchantments (Rulebook pp. 107-108, 168, 207, 329)."""

from panda3d.core import Point3

from battleFunctions import attack_characteristic, ward_save_value
from characters import side_of
from psychology import PsychologySystem, _box_corners, _polys_overlap, obb_distance
from rules_log import rule_log, rule_skipped
from spell_effects import active_spells, end_effect, register
from spell_system import Spell


def unit_members(target):
    host = getattr(target, 'hostUnit', None) or target
    joined = getattr(host, 'joinedCharacter', None)
    return [host, joined] if joined is not None else [host]


def profiles_for(member):
    """A unit grant includes command and split profiles (pp. 192, 194, 207)."""
    profiles = [member.unit.model, *getattr(member.unit, 'command_models', {}).values()]
    for profile in list(profiles):
        for tag in ('mount', 'crew', 'beasts'):
            part = getattr(profile, f'get_{tag}')()
            if part is not None and part not in profiles:
                profiles.append(part)
    return profiles


class UnitEnchantmentSpell(Spell):
    spell_type = 'Enchantment'
    allows_engaged = False
    grant = {}

    def target_reason(self, target):
        if not hasattr(target, 'unit') or self.caster is None or self.game is None:
            return 'a caster and friendly unit are required'
        if side_of(self.game, self.caster, None) != side_of(self.game, target, None):
            return 'only friendly units may be targeted'
        host = getattr(target, 'hostUnit', None) or target
        if host.bodyNP.isEmpty() or host.unit.nmodels <= 0 or not getattr(host, 'isDeployed', True):
            return 'target is not on the battlefield'
        if getattr(host, 'isInCombat', False) and not self.allows_engaged:
            return 'this spell cannot target a unit engaged in combat'
        if self.caster in unit_members(host):
            return None
        caster_box = PsychologySystem._unit_box(self.caster)
        target_box = PsychologySystem._unit_box(host)
        distance = obb_distance(caster_box, target_box)
        if distance > self.RANGE + 1e-6:
            return f'target {distance:.2f}" away exceeds range {self.RANGE}"'
        if not self.caster.unit.model.has_all_round_vision():
            top = self.caster.bodyNP.getTop()
            points = [self.caster.bodyNP.getRelativePoint(top, Point3(horizontal, vertical, 0))
                      for horizontal, vertical in _box_corners(*target_box)]
            reach = max(abs(point.x) + abs(point.y) for point in points) + 1
            half_width, half_depth = caster_box[2:4]
            arc = [(-half_width, half_depth), (half_width, half_depth),
                   (half_width + reach, half_depth + reach),
                   (-half_width - reach, half_depth + reach)]
            if not _polys_overlap([(point.x, point.y) for point in points], arc):
                return 'target is outside the caster\'s vision arc'
        return None

    def canTarget(self, target):
        reason = self.target_reason(target)
        if reason:
            rule_skipped(self.name, self.caster, f'{reason} (pp. 107-108, 329)')
        return reason is None

    def mark_targets(self, mask):
        """Enchantments need a vision arc, not an unobstructed shooting ray (p. 108)."""
        found = False
        for target in self.game.units:
            if getattr(target, 'hostUnit', None) is not None or self.target_reason(target):
                continue
            target.model.setColor(1, 0, 1, 1)
            target.bodyNP.setCollideMask(mask)
            found = True
        return found

    async def apply(self, target):
        self.attach(target, 1)

    def attach(self, target, ticks):
        self.affected_unit = getattr(target, 'hostUnit', None) or target
        self.affected_members = unit_members(target)
        self.ticks_remaining = ticks
        self.grants = []
        if self.duration_list is None and self.game is not None:
            self.duration_list = self.game.fsm.endOfTurnSpells
        register(self, self.affected_unit, duration='end_turn')
        self.refresh()

    def refresh(self):
        wanted = []
        for member in self.affected_members:
            for profile in profiles_for(member):
                if profile not in wanted:
                    wanted.append(profile)
        for profile, rule in list(self.grants):
            if profile not in wanted:
                profile.special_rules[:] = [entry for entry in profile.special_rules if entry is not rule]
                self.grants.remove((profile, rule))
        for profile in wanted:
            if any(existing is profile for existing, _ in self.grants):
                continue
            before = self.value(profile)
            rule = dict(self.grant, name=self.name)
            profile.special_rules.append(rule)
            self.grants.append((profile, rule))
            after = self.value(profile)
            report = rule_log if before != after else rule_skipped
            report(self.name, self.affected_unit,
                   f'{profile.name}: {self.stat_name} {before} -> {after}; '
                   'until end of this turn; same spell does not stack (p. 329)')

    def on_join(self, character, host):
        """Joining spreads an existing unit spell without renewing its duration (p. 207)."""
        if not any(member in self.affected_members for member in (host, character)):
            return
        for member in (host, character):
            if member not in self.affected_members:
                self.affected_members.append(member)
        self.refresh()

    def remove_from(self, members, reason):
        self.affected_members = [member for member in self.affected_members if member not in members]
        if not self.affected_members:
            end_effect(self, reason)
        else:
            self.affected_unit = self.affected_members[0]
            self.refresh()
            rule_log(self.name, self.affected_unit, f'{reason}; separate recipients keep their remaining duration')

    def save_effect(self):
        return {'members': [member.unitName for member in self.affected_members
                            if not member.bodyNP.isEmpty()]}

    def save_target(self):
        return next((member for member in self.affected_members if not member.bodyNP.isEmpty()), None)

    def load_effect(self, data, unit_map):
        self.affected_members = [unit_map[name] for name in data['members'] if name in unit_map]
        self.refresh()

    def endSpell(self):
        end_effect(self, 'effect removed')

    def remove_effect(self):
        for profile, rule in getattr(self, 'grants', []):
            profile.special_rules[:] = [entry for entry in profile.special_rules if entry is not rule]
        self.grants = []


class FuryOfKhaineSpell(UnitEnchantmentSpell):
    """Extra Attacks (+1), including engaged targets, until turn end (p. 329)."""
    RANGE = 12
    allows_engaged = True
    grant = {'extra_attacks': 1}
    stat_name = 'A'
    value = staticmethod(attack_characteristic)


class ShieldOfSapherySpell(UnitEnchantmentSpell):
    """A 5+ Ward replaces previous Enchantments, not Hexes or native rules (p. 329)."""
    RANGE = 18
    grant = {'ward': 5}
    stat_name = 'Ward value (0 = none)'
    value = staticmethod(ward_save_value)

    async def apply(self, target):
        members = unit_members(target)
        for spell in active_spells(self.game):
            previous = getattr(spell, 'affected_unit', None)
            recipients = getattr(spell, 'affected_members', unit_members(previous) if previous is not None else [])
            if (spell is not self and getattr(spell, 'spell_type', None) == 'Enchantment'
                    and previous is not None
                    and any(member in members for member in recipients)):
                reason = f'replaced by {self.name} on {target.unit.name} (p. 329)'
                if hasattr(spell, 'remove_from'):
                    spell.remove_from(members, reason)
                else:
                    end_effect(spell, reason)
        await super().apply(target)


HIGH_MAGIC = {'Fury of Khaine': FuryOfKhaineSpell, 'Shield of Saphery': ShieldOfSapherySpell}