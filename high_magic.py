"""Selected High Magic effects (Rulebook pp. 107-108, 168, 207, 329)."""

from panda3d.core import Point3, Vec3

from battleFunctions import attack_characteristic, resolve_magic_hits, ward_save_value
from characters import is_character, side_of
from models import roll_dice_expr
from psychology import PsychologySystem, _box_corners, _polys_overlap, obb_distance
from rules_log import rule_log, rule_skipped
from spell_effects import active_spells, end_effect, register
from spell_system import Spell
from toHitAndToWound import stat_value
from tempest import TempestSpell


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
    duration = 'end_turn'
    duration_text = 'end of this turn'
    rule_page = 'p. 329'

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
        register(self, self.affected_unit, duration=self.duration)
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
                                         f'until {self.duration_text}; same spell does not stack ({self.rule_page})')

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
                reason = f'replaced by {self.name} on {target.unit.name} ({self.rule_page})'
                if hasattr(spell, 'remove_from'):
                    spell.remove_from(members, reason)
                else:
                    end_effect(spell, reason)
        await super().apply(target)


class CourageOfAenarionSpell(ShieldOfSapherySpell):
    """Unbreakable, replacing older Enchantments; Remains in Play (FoF p. 186)."""
    RANGE = 15
    allows_engaged = True
    grant = {'Unbreakable': True}
    duration = 'remains'
    duration_text = 'the Remains in Play spell ends'
    stat_name = 'Unbreakable'
    rule_page = 'Forces of Fantasy p. 186'

    @staticmethod
    def value(profile):
        return any(rule.get('Unbreakable') for rule in profile.special_rules)


class DrainMagicSpell(Spell):
    """Enemy Wizards within 24 inches add 2 to casting values (Rulebook p. 328)."""
    spell_type = 'Hex'
    targets_self = True

    async def apply(self, target):
        self.attach(self.caster, 1)

    def attach(self, target, ticks):
        self.ticks_remaining = ticks
        register(self, self.caster, duration='remains')

    def affects(self, caster):
        if (self.caster is None or caster is None or self.caster.bodyNP.isEmpty()
                or caster.bodyNP.isEmpty() or getattr(self.caster, 'retiredFromCombat', False)):
            return False
        return (side_of(self.game, caster, None) != side_of(self.game, self.caster, None)
                and caster.unit.model.is_wizard()
                and obb_distance(PsychologySystem._unit_box(caster),
                                 PsychologySystem._unit_box(self.caster)) <= 24 + 1e-6)

    def endSpell(self):
        end_effect(self, 'effect removed')

    def remove_effect(self):
        pass


def drained_casting_value(spell):
    """Same-name auras are not cumulative; range is checked when casting (p. 328)."""
    sources = [effect for effect in active_spells(spell.game)
               if isinstance(effect, DrainMagicSpell)]
    affected = any(effect.affects(spell.caster) for effect in sources)
    value = spell.casting_value + (2 if affected else 0)
    if sources:
        report = rule_log if affected else rule_skipped
        report('Drain Magic', spell.caster,
               f'{spell.name}: casting value {spell.casting_value}+ -> {value}+; '
               'enemy Wizard within 24 inches required; duplicate auras do not stack (p. 328)')
    return value


class WalkBetweenWorldsSpell(UnitEnchantmentSpell):
    """Self/host Ethereal and Reserve Move until next Start of Turn (p. 329; Magic FAQ)."""
    spell_type = 'Conveyance'
    targets_self = True
    self_scope = True
    allows_engaged = True
    RANGE = 0
    grant = {'ethereal': True, 'reserve_move': True}
    duration = 'next_start'
    duration_text = 'caster\'s next Start of Turn'
    stat_name = 'Ethereal/Reserve Move'

    @staticmethod
    def value(profile):
        from reserve_move import has_reserve_move
        from special_rules import is_ethereal
        return is_ethereal(profile), has_reserve_move(profile)

    def target_reason(self, target):
        if self.caster is None or target is not self.caster:
            return 'Self spell: only its caster may be targeted'
        host = getattr(self.caster, 'hostUnit', None) or self.caster
        if getattr(self.caster, 'retiredFromCombat', False):
            return 'a retired Wizard cannot cast (Magic FAQ v1.5.3)'
        if getattr(host, 'state', None) == 'IsFleeing':
            return 'a fleeing Wizard or host cannot cast'
        return super().target_reason(target)

    def attach(self, target, ticks):
        if self.caster is None:
            return
        spells = getattr(self.caster, '_self_spells', [])
        if self not in spells:
            self.caster._self_spells = [*spells, self]
        super().attach(self.caster, ticks)

    def refresh(self):
        previous = list(getattr(self, 'affected_members', []))
        self.affected_members = [self.caster]
        host = getattr(self.caster, 'hostUnit', None)
        if host is not None and not getattr(self.caster, 'retiredFromCombat', False):
            self.affected_members.append(host)
        self.affected_unit = self.caster
        super().refresh()
        removed = [member.unit.name for member in previous if member not in self.affected_members]
        if removed:
            rule_log(self.name, self.caster, f'host benefit removed from {", ".join(removed)}; '
                     'caster no longer supplies its Self spell to that unit (p. 329; Magic FAQ v1.5.3)')

    def on_join(self, character, host):
        if character is self.caster:
            self.refresh()

    def load_effect(self, data, unit_map):
        self.refresh()

    def save_target(self):
        return self.caster

    def remove_effect(self):
        super().remove_effect()
        if self.caster is not None:
            self.caster._self_spells = [spell for spell in getattr(self.caster, '_self_spells', []) if spell is not self]


def visible_spell_target(game, caster, target):
    """Planar base/terrain LOS using the engine's shared sight geometry (p. 103)."""
    from scouts import model_base_boxes
    from skirmish_visibility import model_can_see
    observers, targets = model_base_boxes(caster), model_base_boxes(target)
    if not observers or not targets:
        return False
    observer = observers[0]
    blockers = [box for member in game.units
                if getattr(member, 'hostUnit', None) is None and member.isDeployed
                for box in model_base_boxes(member)
                if box != observer and box not in targets]
    for piece in getattr(getattr(game, 'terrain_manager', None), 'terrain_pieces', []):
        if piece.blocks_line_of_sight and not piece.contains(Point3(*observer[:2], 0)):
            blockers.append((piece.center.x, piece.center.y, piece.width / 2, piece.height / 2, 0))
    facing = None if caster.unit.model.has_all_round_vision() else observer[4]
    return model_can_see(observer, targets, blockers, facing=facing)


class VaulsUnmakingSpell(Spell):
    """Permanently disable a chosen item on a visible enemy character (FoF p. 186)."""
    spell_type = 'Hex'
    RANGE = 12

    def target_reason(self, target):
        if not is_character(target):
            return 'only enemy characters may be targeted'
        if (self.caster is None or target not in self.game.units or target.unit.nmodels <= 0
                or target.bodyNP.isEmpty() or not target.isDeployed):
            return 'target must be a living character on the battlefield'
        if side_of(self.game, self.caster, None) == side_of(self.game, target, None):
            return 'target must be an enemy character'
        distance = obb_distance(PsychologySystem._unit_box(self.caster), PsychologySystem._unit_box(target))
        if distance > self.RANGE + 1e-6:
            return f'target {distance:.2f} inches away exceeds 12 inches'
        if not visible_spell_target(self.game, self.caster, target):
            return 'caster cannot draw line of sight in its vision arc'
        return None

    def canTarget(self, target):
        reason = self.target_reason(target)
        if reason:
            rule_skipped(self.name, self.caster, reason)
        return reason is None

    async def choose_target(self):
        options = {member.unitName: member for member in self.game.units if self.target_reason(member) is None}
        if not options:
            rule_skipped(self.name, self.caster, f'no legal visible enemy within {self.RANGE} inches')
            return None
        choice = await self.game.makeChoiceNew(list(options), Vec3(0, 0, 10),
                    owner=self.caster, cancellable=True, prompt=f'{self.name}: target')
        return options.get(choice)

    async def apply(self, target):
        from magic_items import disable_item, inventory
        options = {f'{index + 1}: {item.name}': item for index, item in enumerate(inventory(target))
                   if not item.destroyed and item.disabled_reason is None}
        if not options:
            rule_skipped(self.name, target, 'no usable magic item remains to unmake')
            return
        choice = await self.game.makeChoiceNew(list(options), Vec3(0, 0, 10),
                    owner=self.caster, prompt=f'{self.name}: choose an item on {target.unit.name}')
        if choice in options:
            item = options[choice]
            disable_item(target, item, f'{self.name}: unusable for the remainder of the game')
            rule_log(self.name, target, f'{item.name}: all item effects disabled for the rest of the game (FoF p. 186)')


class FieryConvocationSpell(VaulsUnmakingSpell):
    """Scatter a 5-inch S4 AP-2 Flaming template over enemies (Rulebook pp. 95, 329)."""
    spell_type = 'Magic Missile'
    RANGE = 18

    def target_reason(self, target):
        if (self.caster is None or target not in self.game.units or target.unit.nmodels <= 0
                or target.bodyNP.isEmpty() or not target.isDeployed
                or getattr(target, 'hostUnit', None) is not None):
            return 'target must be a living enemy unit on the battlefield'
        if side_of(self.game, self.caster, None) == side_of(self.game, target, None):
            return 'target must be an enemy unit'
        if target.isInCombat:
            return 'a Magic Missile cannot target a unit engaged in combat'
        distance = obb_distance(PsychologySystem._unit_box(self.caster), PsychologySystem._unit_box(target))
        if distance > self.RANGE + 1e-6:
            return f'target {distance:.2f} inches away exceeds 18 inches'
        if not visible_spell_target(self.game, self.caster, target):
            return 'target is outside the vision arc or line of sight'
        return None

    async def apply(self, target):
        from spell_templates import scatter_template, fiery_template
        center = target.bodyNP.getPos(target.bodyNP.getTop())
        center, scatter = scatter_template(center, roll_dice_expr('D3+1'))
        rule_log(self.name, self.caster,
                 f'5-inch template: {scatter}; final centre ({center.x:.2f}, {center.y:.2f}) (p. 95)')
        await fiery_template(self, center)


class CorporealUnmakingSpell(Spell):
    """D3 S5 Assailment hits; only Ward saves are allowed (Rulebook p. 329).

    Joined Wizards target their host's enemies from the fighting rank (p. 207,
    Magic FAQ v1.5.3). Initiative windows isolate challenge allocation (p. 211).
    """

    spell_type = 'Assailment'

    def target_reason(self, target):
        if self.caster is None or self.game is None or not hasattr(target, 'unit'):
            return 'a caster and enemy combat unit are required'
        window = getattr(self.game, 'assailmentWindow', None)
        if window is not None and window['caster'] is self.caster:
            return None if target in window['targets'] and target.unit.nmodels > 0 else 'not an eligible combat target'
        return 'Assailment is only available when the Wizard fights at Initiative (p. 108)'

    def canTarget(self, target):
        reason = self.target_reason(target)
        if reason:
            rule_skipped(self.name, self.caster, f'{reason} (pp. 107, 211, 329)')
        return reason is None

    def mark_targets(self, mask):
        """Combat-range spells target engagements, not shooting rays (p. 107)."""
        found = False
        for target in self.game.units:
            if self.target_reason(target):
                continue
            target.model.setColor(1, 0, 1, 1)
            target.bodyNP.setCollideMask(mask)
            found = True
        return found

    async def apply(self, target):
        hits = roll_dice_expr('D3')
        window = getattr(self.game, 'assailmentWindow', None)
        if window is not None and window['caster'] is self.caster:
            from assailment import resolve_hits
            resolve_hits(self, target, hits, 5, 0, allow_armour=False, allow_regeneration=False)
            return
        wounds, saves, unsaved = resolve_magic_hits(target.unit, hits, 5, 0,
                                                   allow_armour=False, allow_regeneration=False)
        rule_log(self.name, self.caster,
                 f'{target.unit.name}: D3 -> {hits} automatic magical S5 hits -> {wounds} wounds, '
                 f'{saves} Ward saves, {unsaved} unsaved; no armour or Regeneration (p. 329)')
        if unsaved:
            window = getattr(self.game, 'assailmentWindow', None)
            if window is not None and window['caster'] is self.caster:
                window['damage'](target, unsaved)
                return
            per_model = max(1, stat_value(target.unit.model.characteristics.get('W'), 1))
            remaining = max(0, target.unit.nmodels * per_model - getattr(target, 'woundsOnModel', 0))
            credited = min(unsaved, remaining)
            host = getattr(self.caster, 'hostUnit', None) or self.caster
            host.assailmentWounds = getattr(host, 'assailmentWounds', 0) + credited
            rule_log(self.name, host, f'{credited} wounds banked for combat result '
                     f'({remaining} target Wounds remaining; no excess-wound credit, p. 151)')
            self.game.movement.applyWounds(target, unsaved)


class HandOfKhaineSpell(CorporealUnmakingSpell):
    """One selected enemy model suffers one S4 hit, no armour (FoF p. 186)."""
    single_model = True

    async def apply(self, target):
        from assailment import resolve_hits
        resolve_hits(self, target, 1, 4, 0, allow_armour=False)


HIGH_MAGIC = {'Fury of Khaine': FuryOfKhaineSpell, 'Shield of Saphery': ShieldOfSapherySpell,
              'Walk Between Worlds': WalkBetweenWorldsSpell,
              'Corporeal Unmaking': CorporealUnmakingSpell,
              'Courage of Aenarion': CourageOfAenarionSpell, 'Drain Magic': DrainMagicSpell,
              "Vaul's Unmaking": VaulsUnmakingSpell, 'Fiery Convocation': FieryConvocationSpell,
              'Tempest': TempestSpell, 'Hand of Khaine': HandOfKhaineSpell}