"""Fresh converted armies must reach generation, item display and their first cast."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

from panda3d.core import getModelPath, loadPrcFileData

from battleFunctions import attack_characteristic
from choiceFunctions import Choice
from game import MyApp
from high_magic import FuryOfKhaineSpell
from magic_items import inventory, resolve_bearer
from spell_effects import end_turn
from spell_generation import begin_spell_generation, pending_wizards
from spell_system import Spell
from tests.test_shieldwall_scene import combat_tasks


def test_default_startup_generates_spells_and_displays_purchased_items(tmp_path):
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
        assert app.magicBusy is False
        assert len(app.units) == 10
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