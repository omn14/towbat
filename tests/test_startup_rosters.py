"""Fresh converted armies must reach generation, item display and their first cast."""

import asyncio
from pathlib import Path
from random import Random
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from panda3d.core import getModelPath, loadPrcFileData

from battleFunctions import attack_characteristic
from choiceFunctions import Choice
from game import MyApp
from high_magic import FuryOfKhaineSpell
from magic_items import inventory, resolve_bearer, save_inventory
from persistence import load_game_state, save_game_state
from spell_effects import end_turn
from spell_generation import begin_spell_generation, pending_wizards
from spell_system import Spell
from tests.test_shieldwall_scene import combat_tasks


def test_default_startup_generates_spells_and_displays_purchased_items(tmp_path, capsys):
    loadPrcFileData('', 'window-type offscreen\nwin-size 1280 720\naudio-library-name null')
    getModelPath().appendDirectory(str(Path(__file__).resolve().parents[1]))
    scheduled = []

    def capture_generation(game):
        with patch.object(game.taskMgr, 'add') as schedule:
            begin_spell_generation(game)
        scheduled.extend(call.args[0] for call in schedule.call_args_list
                         if call.args[1] == 'spellGenerationTask')

    with patch('spell_generation.begin_spell_generation', side_effect=capture_generation):
        app = MyApp()
    try:
        mat = app.ground.getShaderInput('matTex').getTexture()
        assert (mat.getXSize(), mat.getYSize()) == (4096, 4096)
        assert app.magicBusy is False
        assert app.campaign_map is None
        assert not hasattr(app, 'country_model') and not hasattr(app, 'cloud_plane')
        assert not app.taskMgr.hasTaskNamed('update_campaign_terrain')
        assert not app.taskMgr.hasTaskNamed('update_cloud_time')
        assert [piece.terrain_type for piece in app.terrain_manager.terrain_pieces] == [
            'forest', 'forest', 'hill', 'hill', 'river', 'house', 'house']
        assert len(app.units) == 10
        assert all(not member.model.hasPythonTag('test_base_model') for member in app.units)
        assert [member.unit.nmodels for member in app.player1Units] == [1, 6, 5, 3, 1]
        assert [member.unit.nmodels for member in app.player2Units] == [1, 4, 5, 10, 5]
        assert app.fsm.state == 'DeployPhase' and app.spellGenerationBusy
        assert len(scheduled) == 1
        mage, = pending_wizards(app)
        assert mage.unit.model.wizard_level() == 2 and not mage.unit.model.spells

        def choose(options, position, cancellable, descriptions, prompt, detail, **kwargs):
            assert set(options) == {'Keep spells', 'Drain Magic', "Vaul's Unmaking",
                                    'Hand of Khaine', 'Courage of Aenarion'}
            reference = kwargs['reference']
            assert len(reference) == 10
            assert sum(entry['status'] == 'Generated' for entry in reference) == 3
            assert sum(entry['status'] == 'Not generated' for entry in reference) == 3
            choice = Choice(options, position, cancellable, descriptions, prompt, detail,
                            reference=reference)
            missed = next(entry for entry in reference if entry['status'] == 'Not generated')
            choice._inspect_reference(missed['name'])
            assert 'Casting value:' in choice.reference_text.getText()
            assert 'Engine effect: not implemented' not in choice.reference_text.getText()
            assert not choice.choiceMade and choice.choice is None
            app.graphicsEngine.renderFrame()
            app.graphicsEngine.renderFrame()
            assert app.screenshot(str(tmp_path / 'startup-spells.png'), defaultFilename=False)
            choice._pick('Keep spells')
            return choice

        with patch('game.Choice', side_effect=choose) as choices, \
            combat_tasks(app) as run, \
            patch('choiceFunctions.taskMgr', app.taskMgr, create=True):
            run(scheduled[0])
        choices.assert_called_once()
        assert not app.spellGenerationBusy and not pending_wizards(app)
        assert len(mage.unit.model.spells) == 3 and mage.unit.model.wizard_level() == 2

        purchased = {item.name: (member, item) for member in app.units for item in inventory(member)}
        assert set(purchased) == {'Silvery Wand', 'Helm Of Courage', 'The Banner Of The Bold'}
        for name, (member, item) in purchased.items():
            assert resolve_bearer(member, item) is not None
            app.showSelectedUnit(member)
            details = app.unitDetailLines(member)
            offset = next(index for index, line in enumerate(details) if line == f'Item: {name}')
            app.hud.scroll_details(offset - app.hud._detail_offset)
            assert name in '\n'.join(label.getText() for label in app.hud._detail_labels)
        hero, _ = purchased['Helm Of Courage']
        assert hero.unit.model.armor_save == 5 and hero.unit.model.effective_armour_save() == 4
        app.graphicsEngine.renderFrame()
        app.graphicsEngine.renderFrame()
        assert app.screenshot(str(tmp_path / 'startup-items.png'), defaultFilename=False)
        check_first_fury_cast(app, 'Pass', applied=True)
        check_first_fury_cast(app, 'Fated dispel', applied=False)
        ready = check_matchup_roundtrip(app, tmp_path)
        for cavalry in ('Silver Helm', 'Dragon Prince'):
            check_matchup_cavalry(app, ready, tmp_path, cavalry)
        check_matchup_specialists(app, ready, tmp_path)
        output = capsys.readouterr().out
        for rule in ('Silvery Wand', 'First Charge', 'Counter Charge', 'Iron Shod Wheels', 'Dangerous Terrain', 'Fear'):
            assert rule in output
    finally:
        for coroutine in scheduled:
            coroutine.close()
        app.destroy()


def check_first_fury_cast(app, dispel, *, applied):
    armies = {member.unit.model.name: member for member in app.units}
    mage, princes = armies['Mage'], armies['Dragon Prince']
    for member in app.units:
        member.isDeployed = True
    mage.bodyNP.setPos(0, -15, 0)
    mage.bodyNP.setH(0)
    princes.bodyNP.setPos(0, -5, 0)
    original = attack_characteristic(princes.unit.model)
    spell = FuryOfKhaineSpell('Fury of Khaine', 9, wizard_level=2, game=app, caster=mage)

    async def choose(options, *args, owner, **kwargs):
        assert app.magicBusy is True
        assert owner in app.player2Units
        assert dispel in options
        return dispel

    with patch.object(app, 'aiControls', return_value=False), \
            patch.object(app, 'makeChoiceNew', AsyncMock(side_effect=choose)) as choice, \
            patch.object(Spell, '_roll_casting_dice', AsyncMock(side_effect=[
                (10, [6, 4]), (12, [6, 6])])) as dice:
        asyncio.run(spell.spellFunction(princes))
    choice.assert_awaited_once()
    assert spell.casting == 11
    assert dice.await_count == (1 if applied else 2)
    assert attack_characteristic(princes.unit.model) == original + int(applied)
    assert (spell in app.fsm.endOfTurnSpells) is applied
    assert app.magicBusy is False
    end_turn(app)


def check_matchup_roundtrip(app, tmp_path):
    def identity():
        records = {}
        for member in app.units:
            profile = member.unit.model
            profiles = {'main': profile}
            profiles.update({tag: part for tag in ('mount', 'crew', 'beasts')
                             if (part := getattr(profile, f'get_{tag}')()) is not None})
            records[profile.name] = {
                'models': member.unit.nmodels,
                'points': app.victoryRoster[member.unitName]['points'],
                'command': [(entry['role'], entry.get('selection_ref')) for entry in getattr(member.unit, 'command', [])],
                'items': save_inventory(member),
                'weapons': {tag: sorted(part.weapons) for tag, part in profiles.items()},
                'parts': {tag: profile.part_count(tag) for tag in ('crew', 'beasts')},
                'spells': sorted(profile.spells),
                'level': profile.wizard_level(),
            }
        return records

    expected = identity()
    assert len(expected) == 10
    assert [sum(record['points'] for record in app.victoryRoster.values() if record['player'] == player)
            for player in (1, 2)] == [500, 500]
    for side, units in enumerate((app.player1Units, app.player2Units)):
        for index, member in enumerate(units):
            member.isDeployed = True
            member.bodyNP.setPos(-20 + index * 10, 12 if side else -12, 0)
            member.bodyNP.setH(180 if side else 0)
    app.fsm.request('MovementPhase')
    app.chargeStage = 'remaining'
    path = save_game_state(app, str(tmp_path / 'he-chaos-500-ready.json'))
    assert path is not None
    for _ in range(2):
        load_game_state(app, path)
        assert identity() == expected
        assert not pending_wizards(app) and not app.spellGenerationBusy and not app.magicBusy
    app.graphicsEngine.renderFrame()
    app.graphicsEngine.renderFrame()
    assert app.screenshot(str(tmp_path / 'he-chaos-500-ready.png'), defaultFilename=False)
    return path


def check_matchup_cavalry(app, ready, tmp_path, cavalry):
    from tests.test_counter_charge_scene import declared_charge
    from battleFunctions import simulate_battle
    from charge_declarations import resolve_declarations
    from victory_points import calculate
    app, charger, defender, origin, facing, contact = declared_charge((app, ready), cavalry=cavalry)
    app.AIplayer2.active = False
    with combat_tasks(app) as run, patch.object(app, 'aiControls', return_value=True), \
            patch.object(app.combat, 'rullTerninger', AsyncMock(side_effect=[([], [3]), ([], [6, 6])])), \
            patch.object(app.combat, 'swiftstrideChargeChoice', AsyncMock(return_value=False)):
        run(app.combat.chargeAndChargeReaction(charger, contact, origin, facing, SimpleNamespace(done='done')))
        if app.chargeDeclarations:
            run(resolve_declarations(app))
    assert charger.isInCombatWith == [defender] and defender.isInCombatWith == [charger]
    assert charger.chargedThisTurn and defender.chargedThisTurn
    assert charger.firstChargeDisruptedBy and defender.firstChargeDisruptedBy
    app.fsm.request('CombatPhase')
    app.unitToMove = charger
    generator = Random(27)
    with combat_tasks(app) as run, patch.object(app, 'aiControls', return_value=True), \
            patch.object(app, 'makeChoiceNew', AsyncMock(side_effect=lambda options, *args, **kwargs: options[0])), \
            patch.object(app, 'rollLeadershipDice', AsyncMock(return_value=[1, 1])), \
            patch.object(app.combat, 'rollBreakDice', AsyncMock(return_value=[1, 1])), \
            patch('battleFunctions.random.randint', side_effect=generator.randint), \
            patch('combat_resolution.simulate_battle', wraps=simulate_battle) as fights:
        run(app.combat._verySimpleBattleInner(SimpleNamespace(done='done')))
    profiles = {call.args[0].model.name for call in fights.call_args_list}
    assert charger.unit.model.name in profiles and defender.unit.model.name in profiles
    assert charger.unit.model.get_mount().name in profiles
    assert charger.hasAttackedThisTurn and defender.hasAttackedThisTurn
    result = calculate(app)
    assert result['complete']
    stem = cavalry.lower().replace(' ', '-')
    saved = save_game_state(app, str(tmp_path / f'he-chaos-500-{stem}-after-combat.json'))
    assert saved is not None
    load_game_state(app, saved)
    assert calculate(app) == result
    app.roundCounter.max_rounds = 6
    app.roundCounter.currentRoundPlayer = [6, 5]
    app.roundCounter.request('PlayerTwo')
    app.fsm.nextPhase()
    assert app.fsm.state == 'BattleEnded'
    assert app.battleResult['scores'] == result['scores']
    app.graphicsEngine.renderFrame()
    app.graphicsEngine.renderFrame()
    assert app.screenshot(str(tmp_path / f'he-chaos-500-{stem}-result.png'), defaultFilename=False)


def check_matchup_specialists(app, ready, tmp_path):
    from direct.interval.IntervalGlobal import Sequence
    from panda3d.core import Point3, Vec3
    from battleFunctions import impact_hit_report, resolve_impact_hits, simulate_battle
    from combat_profiles import combat_profiles
    from fear import test_fear
    from flight import grounded
    from tests.test_faction_rules_scene import members
    from tests.test_combat_contacts_scene import edge_contact
    load_game_state(app, ready)
    app.terrain_manager.clear()
    roster = members(app)
    skycutter, warriors, horsemen = (roster[name] for name in ('Lothern Skycutter', 'Chaos Warrior', 'Marauder Horsemen'))
    assert not horsemen.unit.model.missile_weapon()
    option = app.combat.standAndShootOption(horsemen, skycutter)
    assert not option and not app.combat.fireAndFleeOption(horsemen, skycutter, option)
    profile = skycutter.unit.model
    assert profile.get_crew().firing_bs() == 4 and profile.part_count('crew') == 3
    for owner in (profile, profile.get_crew()):
        owner.equip_weapon('Shortbow')
    skycutter.bodyNP.setPos(0, -10, 0)
    skycutter.bodyNP.setH(0)
    warriors.bodyNP.setPos(0, 0, 0)
    generator = Random(73)
    with combat_tasks(app) as run, patch('game.simulate_battle', wraps=simulate_battle) as shots, \
            patch.object(app, 'shootingAnimation', AsyncMock()), \
            patch('battleFunctions.random.randint', side_effect=generator.randint):
        run(app.shootAt(skycutter, warriors))
    assert shots.call_count == 1
    assert shots.call_args.args[0].nmodels == 3
    assert shots.call_args.args[0].model is profile.get_crew()
    assert 'S5 AP-2' in impact_hit_report(skycutter.unit, warriors.unit)[0]
    with patch('battleFunctions.random.randint', side_effect=generator.randint):
        hits, wounds, saves, unsaved = resolve_impact_hits(skycutter.unit, warriors.unit)
    assert 2 <= hits <= 4 and 0 <= unsaved <= wounds <= hits and saves + unsaved == wounds
    app.movement.applyWounds(warriors, unsaved)
    parts = combat_profiles(skycutter, warriors)
    assert {part.role for part in parts} == {'crew', 'beasts'}
    assert next(part for part in parts if part.role == 'crew').count == 3
    assert next(part for part in parts if part.role == 'beasts').profile.name == 'Swiftfeather Roc'
    knights = roster['Chaos Knight']
    edge_contact(skycutter, knights)
    skycutter.chargedThisTurn = True
    skycutter.chargeDistance = 4
    skycutter.roundsFought = 1
    app.attackers, app.defenders = [skycutter], [knights]
    app.attackSequence = Sequence()
    app.combat._pendingWounds = {}
    app.combat._combatStartModels = {id(member.unit): member.unit.nmodels for member in (skycutter, knights)}
    removals = Sequence()
    with combat_tasks(app) as run, patch.object(app, 'aiControls', return_value=True), \
            patch('combat_resolution.simulate_battle', wraps=simulate_battle) as melee, \
            patch('battleFunctions.random.randint', side_effect=generator.randint):
        run(app.combat.resolveCombatWithSpells(None, removals))
        async def apply_casualties():
            await removals
        run(apply_casualties())
    assert {call.args[0].model.name for call in melee.call_args_list} == {'Sea Guard Crew', 'Swiftfeather Roc'}
    skycutter.bodyNP.setPos(0, -10, 0)
    terrain = app.terrain_manager.add_terrain('forest', Point3(0, -10, 0), 4, 4)
    with patch('terrain_system.random.randint') as dice:
        assert app.movement.dangerousTerrainTests(skycutter, Vec3(0, -18, 0), Vec3(0, -2, 0), features=[terrain]) == 0
    dice.assert_not_called()
    with grounded(skycutter), patch('terrain_system.random.randint', side_effect=[1, 2]):
        assert app.movement.dangerousTerrainTests(skycutter, Vec3(0, -18, 0), Vec3(0, -2, 0), features=[terrain]) == 2
    assert skycutter.unit.nmodels == 1 and skycutter.woundsOnModel == 2
    with patch('fear.random.randint') as dice:
        assert asyncio.run(test_fear(app, horsemen, [skycutter], 'combat chosen'))
    dice.assert_not_called()
    assert warriors.unit.nmodels >= 2
    app.movement.applyWounds(warriors, warriors.unit.nmodels - 2)
    with patch.object(app, 'makeChoiceNew', AsyncMock(return_value='Keep')), \
            patch('fear.random.randint', return_value=6):
        assert not asyncio.run(test_fear(app, warriors, [skycutter], 'combat chosen'))
    saved = save_game_state(app, str(tmp_path / 'he-chaos-500-specialists.json'))
    assert saved is not None
    load_game_state(app, saved)
    assert members(app)['Lothern Skycutter'].woundsOnModel == 2
    assert members(app)['Chaos Warrior'].fearFailed