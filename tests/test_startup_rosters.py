"""The default converted armies must reach generation and item display at startup."""

from pathlib import Path
from unittest.mock import patch

from panda3d.core import getModelPath, loadPrcFileData

from choiceFunctions import Choice
from game import MyApp
from magic_items import inventory, resolve_bearer
from spell_generation import begin_spell_generation, pending_wizards
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
            assert 'Engine effect: not implemented' in choice.reference_text.getText()
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
    finally:
        for coroutine in scheduled:
            coroutine.close()
        app.destroy()