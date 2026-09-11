"""Awaited, bearer-owned combat spell choices at Initiative (Rulebook pp. 108, 146, 211)."""

from panda3d.core import Vec3

from rules_log import rule_log, rule_skipped


def single_model_targets(targets, challenge):
    from command_groups import champions
    if challenge is not None and any(challenge.involves(target) for target in targets):
        return targets
    result = []
    for host in targets:
        protected = champions(host, include_retired=True)
        if host.unit.nmodels > len(protected):
            result.append(host)
        result.extend(champion for champion in protected if not champion.retiredFromCombat
                      and not (challenge and challenge.involves(champion)))
        joined = getattr(host, 'joinedCharacter', None)
        if joined is not None and not getattr(joined, 'retiredFromCombat', False) and not (
                challenge and challenge.involves(joined)):
            result.append(joined)
    return result


async def cast_at_initiative(game, caster, targets, damage, *, challenge=None, miscast_damage=None):
    from spell_system import may_attempt, spell_class
    from magic_items import item_spell_available
    from chaos_gifts import succumbed
    if caster is None or getattr(caster, 'retiredFromCombat', False):
        return
    if succumbed(caster):
        rule_skipped('Assailment', caster, 'succumbed to Stupidity; cannot cast (p. 178)')
        return
    profile = caster.unit.model
    if not profile.is_wizard() and not any(record.get('bound') for record in profile.spells.values()):
        return
    previous = getattr(game, 'assailmentWindow', None)
    busy = getattr(game, 'magicBusy', False)
    game.assailmentWindow = dict(caster=caster, targets=targets, damage=damage, challenge=challenge,
                                miscast_damage=miscast_damage)
    game.magicBusy = True
    try:
        while targets:
            if caster.unit.nmodels <= 0 or (hasattr(caster, 'bodyNP') and caster.bodyNP.isEmpty()):
                rule_skipped('Assailment', caster, 'Wizard slain; no further casting attempts')
                return
            choices = []
            for name, record in profile.spells.items():
                if record.get('type') != 'Assailment' or spell_class(record.get('name', name)) is None:
                    continue
                spent = getattr(caster, 'cannotCastThisTurn', False)
                if record.get('bound'):
                    available = (not spent and 'combat' not in getattr(caster, 'boundSpellPhases', [])
                                 and item_spell_available(caster, record))
                else:
                    cast = getattr(caster, 'spellsCastThisTurn', [])
                    available = may_attempt(cast, name, profile.wizard_level(1), spent)
                if available:
                    choices.append(name)
            if not choices:
                return
            if game.aiControls(caster):
                name = choices[0]
            else:
                name = await game.makeChoiceNew([*choices, 'No more spells'], Vec3(0, 0, 10), owner=caster,
                                               prompt=f'{caster.unit.name}: Assailment at Initiative')
            if name not in choices:
                rule_skipped('Assailment', caster, 'declines remaining spells; ordinary attacks continue')
                return
            record = profile.spells[name]
            spell_name = record.get('name', name)
            spell = spell_class(spell_name)(spell_name, record['casting_value'], game=game, caster=caster)
            spell.selection_key = name
            spell.wizard_level = profile.wizard_level(1)
            spell.bound = record.get('bound', False)
            spell.power_level = record.get('power_level', 0)
            spell.spell_range = 'Combat'
            offered = single_model_targets(targets, challenge) if getattr(spell, 'single_model', False) else targets
            game.assailmentWindow['targets'] = offered
            legal = [target for target in offered if spell.canTarget(target)]
            if not legal:
                rule_skipped(name, caster, 'no surviving legal target at this Initiative')
                return
            if len(legal) == 1 or game.aiControls(caster):
                target = legal[0]
            else:
                options = {target.unitName: target for target in legal}
                selected = await game.makeChoiceNew(list(options), Vec3(0, 0, 10), owner=caster,
                                                   prompt=f'{name}: target')
                target = options.get(selected)
                if target is None:
                    return
            if spell.bound:
                caster.boundSpellPhases = [*getattr(caster, 'boundSpellPhases', []), 'combat']
            else:
                caster.spellsCastThisTurn = [*getattr(caster, 'spellsCastThisTurn', []), name]
            rule_log('Assailment', caster, f'{name} resolves at the Wizard\'s Initiative before ordinary attacks (p. 108)')
            await spell.spellFunction(target)
            if spell.no_more_spells:
                caster.cannotCastThisTurn = True
    finally:
        game.assailmentWindow = previous
        game.magicBusy = busy


def drain_steps(steps):
    """Synchronous compatibility for combat callers with no spell choices."""
    while True:
        try:
            next(steps)
        except StopIteration as finished:
            return finished.value


def resolve_hits(spell, target, hits, strength, ap, *, allow_armour=True, allow_regeneration=True):
    """Regeneration recovers wounds but does not erase combat result (p. 176)."""
    from battleFunctions import resolve_magic_hits
    regenerated = []
    wounds, saves, unsaved = resolve_magic_hits(target.unit, hits, strength, ap,
        allow_armour=allow_armour, allow_regeneration=allow_regeneration, regenerated=regenerated)
    rule_log(spell.name, spell.caster, f'{target.unit.name}: {hits} automatic magical S{strength} '
             f'AP-{ap} hits -> {wounds} wounds, {saves} saved, {unsaved} unsaved; '
             f'armour {"allowed" if allow_armour else "prohibited"}, '
             f'Regeneration {"allowed" if allow_regeneration else "prohibited"}; '
             f'{len(regenerated)} regenerated wounds still count toward combat result (pp. 151, 176)')
    window = getattr(spell.game, 'assailmentWindow', None)
    if window is not None and window['caster'] is spell.caster:
        window['damage'](target, unsaved, len(regenerated))
    elif unsaved:
        spell.game.movement.applyWounds(target, unsaved)
    return unsaved