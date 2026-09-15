"""Choice descriptions must not overlap the answers, even after wrapping."""

import asyncio
import json
import math
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from direct.showbase.ShowBase import ShowBase
from panda3d.core import Filename, Point3, getModelPath, loadPrcFileData

from battlescribe import get_catalogue, spell_key
from choiceFunctions import Choice
from spell_system import spell_readout
from spell_generation import generation_reference, start_generation
from tests.test_spell_generation import mage_with_pool


@pytest.fixture(scope='module', params=[(1280, 720), (1024, 768), (720, 960)])
def display(request):
    width, height = request.param
    loadPrcFileData('', f'window-type offscreen\nwin-size {width} {height}\naudio-library-name null')
    getModelPath().appendDirectory(Filename.fromOsSpecific(str(Path(__file__).resolve().parents[1])))
    base = ShowBase()
    yield base
    base.destroy()


@pytest.fixture
def dialog(display):
    dialogs = []

    def create(*args, **kwargs):
        choice = Choice(*args, pos=(0, 0, 0), **kwargs)
        dialogs.append(choice)
        return choice

    yield create
    for choice in dialogs:
        asyncio.run(choice.cleanup())


def bound_fireball():
    return dict(get_catalogue().spell('Fireball'), bound=True, power_level=1,
                source='Ruby Ring of Ruin')


def button_top(button):
    return button.getZ() + button['frameSize'][3] * button.getSz()


def test_bound_fireball_description_stays_above_button(dialog):
    spell = bound_fireball()
    key = spell_key(spell)
    choice = dialog([key], cancellable=True,
                    descriptions={key: spell_readout(key, spell)},
                    prompt='Captain of the Empire: cast which spell?')
    choice._showDetail(key)
    assert choice.detail.textNode.getNumRows() > 1
    bottom, _ = choice.detail.getTightBounds(choice.panel)
    assert bottom.z >= max(button_top(b) for b in choice.buttons) + 0.01


def test_all_spell_hover_text_fits_without_moving_buttons(dialog):
    spells = [*get_catalogue().lore('Battle Magic'), bound_fireball()]
    descriptions = {spell_key(s): spell_readout(spell_key(s), s) for s in spells}
    choice = dialog(list(descriptions), cancellable=True, descriptions=descriptions)
    positions = [tuple(b.getPos()) for b in choice.buttons]
    panel_position = tuple(choice.panel.getPos())
    for name in descriptions:
        choice._showDetail(name)
        bottom, _ = choice.detail.getTightBounds(choice.panel)
        assert bottom.z >= max(button_top(b) for b in choice.buttons) + 0.01, name
        assert [tuple(b.getPos()) for b in choice.buttons] == positions
        assert tuple(choice.panel.getPos()) == panel_position
    choice._showDetail(None)
    assert choice.detail.getText() == ''
    assert [tuple(b.getPos()) for b in choice.buttons] == positions
    assert choice.panel.getZ() <= 1 - choice.PAD


def test_wrapped_description_not_just_explicit_newlines_gets_space(dialog):
    description = 'A long effect with conditions and modifiers. ' * 12
    choice = dialog(['Cast'], descriptions={'Cast': description})
    choice._showDetail('Cast')
    assert choice.detail.textNode.getNumRows() > 4
    bottom, _ = choice.detail.getTightBounds(choice.panel)
    assert bottom.z >= button_top(choice.buttons[0]) + 0.01


def test_plain_choice_does_not_reserve_an_empty_description_panel(dialog):
    choice = dialog(['Yes', 'No'])
    assert choice.detail is None
    assert -choice.panel['frameSize'][2] < 0.3


def test_hud_empty_label_restores_scale_without_measuring():
    from hud import HUD

    label = Mock()
    label.getText.return_value = ''
    label.textNode.getWidth.side_effect = AssertionError('Empty text has no layout bounds')
    HUD._fit(None, label, 0.019, 0.4)
    label.textNode.getWidth.assert_not_called()
    label.setScale.assert_called_once_with(0.019)


def test_hud_startup_detail_measurements_remain_finite(display):
    import gui_theme
    from hud import HUD

    label = gui_theme.styled_text(parent=display.aspect2d, scale=0.019)
    details = ['', 'Ld 8 from Mage Unit', 'Hand Weapon',
               'Wizard Level 2: 2 casting(s) left', 'Item: Silvery Wand']
    try:
        for cycle in range(100):
            for text in details:
                label.setText(text)
                try:
                    HUD._fit(None, label, 0.019, 0.4)
                    if text:
                        measured = label.textNode.getWidth()
                        assert math.isfinite(measured) and measured >= 0, repr(text)
                    else:
                        assert label.getScale() == (0.019, 0.019)
                except AssertionError as error:
                    raise AssertionError(f'HUD text measurement: cycle={cycle}, text={text!r}, '
                                         f'transform={label.textNode.getTransform()}') from error
                assert all(math.isfinite(component) and component > 0 for component in label.getScale())
                assert not label.textNode.getTransform().isNan(), repr(text)
                assert not label.getMat().isNan(), repr(text)
            display.graphicsEngine.renderFrame()
    finally:
        label.destroy()


@pytest.mark.parametrize('remains_in_play', [False, True])
def test_spell_choice_buttons_fit_and_render(display, dialog, tmp_path, remains_in_play):
    if remains_in_play:
        choice = dialog(['Keep', 'Keep this player turn', 'End spell'],
                        prompt='Courage of Aenarion: Remains in Play', detail='Caster: Mage')
        selected = 1
    else:
        choice = dialog(['Keep spells', 'Drain Magic', "Vaul's Unmaking",
                         'Courage of Aenarion', 'Hand of Khaine'],
                        prompt='Mage: signature spell?',
                        detail='Generated: Walk Between Worlds, Fiery Convocation, Shield of Saphery')
        selected = 2
    for button in choice.buttons:
        text = button.component('text0')
        node = text.textNode
        left, right, lower, upper = button['frameSize']
        bottom = node.getTransform().xformPoint(Point3(node.getLeft(), 0, node.getBottom()))
        top = node.getTransform().xformPoint(Point3(node.getRight(), 0, node.getTop()))
        assert bottom.x >= left and top.x <= right, button['text']
        assert bottom.z >= lower and top.z <= upper, button['text']
    display.graphicsEngine.renderFrame()
    display.graphicsEngine.renderFrame()
    filename = 'remains-in-play.png' if remains_in_play else 'spell-generation.png'
    assert display.screenshot(Filename.fromOsSpecific(str(tmp_path / filename)).getFullpath(), defaultFilename=False)
    with patch.object(display.taskMgr, 'add') as schedule:
        choice.buttons[selected]['command'](*choice.buttons[selected]['extraArgs'])
    asyncio.run(schedule.call_args.args[0])
    assert choice.choice == ('Keep this player turn' if remains_in_play else "Vaul's Unmaking")
    assert choice.choiceMade


def test_generation_reference_browsing_is_not_an_answer_and_stays_bounded(display, dialog, tmp_path):
    mage = mage_with_pool()
    with open(Path(__file__).resolve().parents[1] / 'strategy_armies/my_army_he.json') as source:
        roster = json.load(source)
    mage.unit.roster_metadata['spell_pool'] = next(
        entry['spell_pool'] for entry in roster['units'] if entry.get('spell_pool'))
    with patch('spell_generation.random.randint', side_effect=[1, 2, 6]):
        state = start_generation(mage)
    reference = generation_reference(mage, state)
    choice = dialog(['Keep spells', 'Drain Magic', "Vaul's Unmaking",
                     'Courage of Aenarion', 'Hand of Khaine'], reference=reference,
                    prompt='Mage: signature spell?',
                    detail='Generated: Walk Between Worlds, Fiery Convocation, Shield of Saphery')
    buttons_before = [tuple(button.getPos()) for button in choice.buttons]
    frame_before = choice.panel['frameSize']
    assert 'Walk Between Worlds' in choice.reference_text.getText()
    for button, entry in zip(choice.reference_buttons, reference):
        button['command'](*button['extraArgs'])
        assert choice.reference_text.getText() == entry['detail']
        assert not choice.choiceMade and choice.choice is None
        assert [tuple(button.getPos()) for button in choice.buttons] == buttons_before
        assert choice.panel['frameSize'] == frame_before
        assert choice.reference_reader.verticalScroll['value'] == 0
    left, right, lower, upper = choice.panel['frameSize']
    assert -display.getAspectRatio() <= left and right <= display.getAspectRatio()
    assert -1 <= choice.panel.getZ() + lower < choice.panel.getZ() + upper <= 1
    display.graphicsEngine.renderFrame()
    display.graphicsEngine.renderFrame()
    for view in choice.reference_views:
        assert view.getZ() + view['frameSize'][2] > max(button_top(button) for button in choice.buttons)
        assert view.horizontalScroll.isHidden() or view.horizontalScroll.isStashed()
    choice._inspect_reference('Tempest')
    assert 'Not generated' in choice.reference_text.getText()
    display.graphicsEngine.renderFrame()
    display.graphicsEngine.renderFrame()
    assert display.screenshot(Filename.fromOsSpecific(str(tmp_path / 'generation-reference.png')).getFullpath(), defaultFilename=False)
    choice._inspect_reference("Vaul's Unmaking")
    display.graphicsEngine.renderFrame()
    display.graphicsEngine.renderFrame()
    assert display.screenshot(Filename.fromOsSpecific(str(tmp_path / 'generation-effect.png')).getFullpath(), defaultFilename=False)
    reference[-1]['detail'] = 'Long effect. ' * 350 + '\nLast effect line.'
    choice._inspect_reference(reference[-1]['name'])
    assert choice.reference_reader['canvasSize'][2] < choice.reference_reader['frameSize'][2]
    choice.reference_reader.verticalScroll['value'] = 1
    assert choice.reference_text.getText().endswith('Last effect line.')
    display.graphicsEngine.renderFrame()
    display.graphicsEngine.renderFrame()
    bottom, _ = choice.reference_text.getTightBounds(choice.reference_reader)
    assert bottom.z >= choice.reference_reader['frameSize'][2]
    choice._showDetail("Vaul's Unmaking")
    assert "Vaul's Unmaking" in choice.reference_text.getText()
    assert not choice.choiceMade and choice.reference_reader.verticalScroll['value'] == 0
    with patch.object(display.taskMgr, 'add') as schedule:
        choice.buttons[0]['command'](*choice.buttons[0]['extraArgs'])
    asyncio.run(schedule.call_args.args[0])
    assert choice.choiceMade and choice.choice == 'Keep spells'
    assert not choice.reference_views and not choice.reference_buttons