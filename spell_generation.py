"""Pre-deployment spell generation (Rulebook pp. 106, 319; Forces of Fantasy p. 186)."""

from copy import deepcopy
import random

from panda3d.core import Vec3

from battlescribe import slugify, spell_key
from magic_items import known_spell_count
from rules_log import dice_roll, rule_log, rule_skipped
from spell_system import restore_spellbook, spell_class


SAPHERY_SIGNATURES = {'hand_of_khaine', 'courage_of_aenarion', 'vauls_unmaking'}


def spell_record(entry):
    return deepcopy({key: value for key, value in entry.items() if key != 'class'})


def spell_tables(member):
    """Saphery alternatives are signatures even when the export numbers them 1-3."""
    metadata = member.unit.roster_metadata
    pool = metadata.get('spell_pool', []) + [entry for entry in member.unit.model.spells.values()
                                            if not entry.get('bound')]
    unique = {entry['name']: spell_record(entry) for entry in pool}
    numbered = {}
    signatures = []
    keywords = member.unit.model.characteristics.get('Special Rules', []) or []
    saphery = any(slugify(name) == 'lore_of_saphery' for name in keywords)
    for entry in unique.values():
        if slugify(entry['name']) in SAPHERY_SIGNATURES:
            if saphery:
                signatures.append(entry)
        elif entry.get('number') is None:
            signatures.append(entry)
        elif entry.get('number') in range(1, 7):
            number = entry['number']
            if number in numbered:
                raise ValueError('Spell pool contains more than one numbered lore')
            numbered[number] = entry
    if set(numbered) != set(range(1, 7)):
        raise ValueError('Spell generation requires one complete numbered lore (1-6)')
    normal = [entry for entry in signatures if slugify(entry['name']) not in SAPHERY_SIGNATURES]
    if len(normal) != 1:
        raise ValueError('Spell generation requires exactly one normal signature spell')
    return numbered, signatures


def start_generation(member):
    """Bank dice before offering a choice; resuming never re-rolls saved results."""
    metadata = member.unit.roster_metadata
    state = metadata.get('spell_generation')
    if state is not None:
        return state
    numbered, signatures = spell_tables(member)
    known = [spell_record(entry) for entry in member.unit.model.spells.values() if not entry.get('bound')]
    count = known_spell_count(member, log=True)
    if count < len(known) or count > len(numbered):
        raise ValueError(f'Cannot generate {count} known spells from this pool and existing selections')
    state = {'known': known, 'generated': [], 'rolls': [], 'signature': None,
             'stage': 'signature', 'complete': False, 'count': count,
             'signatures': deepcopy(signatures)}
    metadata['spell_generation'] = state
    chosen = {entry['name'] for entry in known}
    while len(chosen) < count:
        roll = random.randint(1, 6)
        dice_roll([roll])
        state['rolls'].append(roll)
        spell = numbered[roll]
        if spell['name'] in chosen:
            rule_skipped('Spell Generation', member, f'D6={roll}: {spell["name"]} already known; re-roll duplicate (p. 319)')
            continue
        state['generated'].append(deepcopy(spell))
        chosen.add(spell['name'])
        rule_log('Spell Generation', member, f'D6={roll} -> {spell["name"]}; {len(chosen)}/{count} known spells')
    return state


def generation_reference(member, state):
    """Read-only profiles from the eligible lore and the saved generation result."""
    numbered, signatures = spell_tables(member)
    known_names = {entry['name'] for entry in state['known']}
    generated_names = {entry['name'] for entry in state['generated']}
    remaining = [entry for entry in numbered.values()
                 if entry['name'] not in known_names | generated_names]
    groups = [('Already known', state['known']), ('Generated', state['generated']),
              ('Not generated', sorted(remaining, key=lambda entry: entry['number'])),
              ('Signature option', [entry for entry in signatures
                                    if entry['name'] not in known_names | generated_names])]
    reference = []
    for status, entries in groups:
        for entry in entries:
            selected = state.get('signature')
            display_status = ('Selected signature' if selected and selected['name'] == entry['name']
                              else status)
            value = entry.get('casting_value')
            casting = f'{value}+' if isinstance(value, (int, float)) else str(value or 'Not recorded')
            reach = entry.get('range', 'Not recorded')
            if isinstance(reach, (int, float)):
                reach = f'{reach}"'
            phase = str(entry.get('phase') or 'Not recorded').title()
            details = [entry['name'], display_status,
                       f'Type: {entry.get("type") or "Not recorded"}',
                       f'Casting value: {casting}    Range: {reach}', f'Phase: {phase}',
                       '', entry.get('effect') or 'Effect text not recorded in this roster.']
            if spell_class(entry['name']) is None:
                details += ['', 'Engine effect: not implemented']
            reference.append({'name': entry['name'], 'status': display_status,
                              'detail': '\n'.join(details)})
    return reference


async def generate_spells(game, member):
    """One optional signature swap, including Saphery alternatives (FoF p. 186)."""
    metadata = member.unit.roster_metadata
    if not metadata.get('spell_generation_pending'):
        return True
    try:
        state = start_generation(member)
    except ValueError as error:
        rule_skipped('Spell Generation', member, str(error))
        return False
    if state['complete']:
        metadata['spell_generation_pending'] = False
        return True
    generated = state['generated']
    if generated and state['stage'] == 'signature':
        known_names = {entry['name'] for entry in state['known'] + generated}
        options = ['Keep spells'] + [entry['name'] for entry in state['signatures']
                                               if entry['name'] not in known_names]
        selected = options[0] if game.aiControls(member) else await game.makeChoiceNew(
            options, Vec3(0, 0, 10), owner=member,
            prompt=f'{member.unit.name}: signature spell?',
            detail='Generated: ' + ', '.join(entry['name'] for entry in generated),
            reference=generation_reference(member, state))
        if selected not in options:
            return False
        if selected != options[0]:
            state['signature'] = next(deepcopy(entry) for entry in state['signatures'] if entry['name'] == selected)
            state['stage'] = 'replace'
        else:
            state['stage'] = 'finish'
            rule_skipped('Signature Spell', member, 'keeps all randomly generated spells; no substitution')
    if state['stage'] == 'replace':
        options = [entry['name'] for entry in generated]
        selected = options[0] if game.aiControls(member) else await game.makeChoiceNew(
            options, Vec3(0, 0, 10), owner=member,
            prompt=f'{member.unit.name}: replace which spell?', detail=f'New spell: {state["signature"]["name"]}',
            reference=generation_reference(member, state))
        if selected not in options:
            return False
        generated[options.index(selected)] = deepcopy(state['signature'])
        rule_log('Lore of Saphery' if slugify(state['signature']['name']) in SAPHERY_SIGNATURES else 'Signature Spell',
                 member, f'{selected} -> {state["signature"]["name"]}; one substitution, {state["count"]} known spells')
        state['stage'] = 'finish'
    known = state['known'] + generated
    from magic_items import bind_generated_spells
    bind_generated_spells(member, known)
    bound = [entry for entry in member.unit.model.spells.values() if entry.get('bound')]
    restore_spellbook(member.unit.model, bound + known, member.unit.model.wizard_level())
    state['complete'] = True
    metadata['spell_generation_pending'] = False
    for entry in known:
        if spell_class(entry['name']) is None:
            rule_skipped('Spell Generation', member, f'{entry["name"]} is known but its spell effect is not implemented')
    rule_log('Spell Generation', member, f'final spellbook: {", ".join(spell_key(entry) for entry in known)}; '
             f'Wizard Level {member.unit.model.wizard_level()} unchanged')
    return True


def pending_wizards(game):
    return [member for member in game.units
            if getattr(member.unit, 'roster_metadata', {}).get('spell_generation_pending')]


async def prepare_spellbooks(game):
    """Resolve one side at a time, with each player choosing their Wizard order (p. 319)."""
    from characters import side_of

    game.spellGenerationBusy = True
    try:
        for player in (1, 2):
            while pending := [member for member in pending_wizards(game) if side_of(game, member) == player]:
                member = pending[0]
                if len(pending) > 1 and not game.aiControls(member):
                    choices = {candidate.unitName: candidate for candidate in pending}
                    selected = await game.makeChoiceNew(list(choices), Vec3(0, 0, 10), owner=member,
                                                        prompt=f'Player {player}: generate spells for which Wizard?')
                    if selected not in choices:
                        return False
                    member = choices[selected]
                if not await generate_spells(game, member):
                    return False
        return not pending_wizards(game)
    finally:
        game.spellGenerationBusy = False


def begin_spell_generation(game):
    """Schedule before deployment, or resume saved choices after restoration."""
    if getattr(game, 'restoringBattle', False) is True or getattr(game, 'spellGenerationBusy', False) is True:
        return
    if not pending_wizards(game):
        return
    game.spellGenerationBusy = True

    async def prepare():
        complete = await prepare_spellbooks(game)
        if complete and game.fsm.state == 'DeployPhase':
            from deployPhase import refresh_deployment
            refresh_deployment(game)

    game.taskMgr.add(prepare(), 'spellGenerationTask')