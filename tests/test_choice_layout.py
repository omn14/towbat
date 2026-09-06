"""Choice descriptions must not overlap the answers, even after wrapping."""

import asyncio
from pathlib import Path

import pytest
from direct.showbase.ShowBase import ShowBase
from panda3d.core import getModelPath, loadPrcFileData

from battlescribe import get_catalogue, spell_key
from choiceFunctions import Choice
from spell_system import spell_readout


@pytest.fixture(scope='module')
def display():
    loadPrcFileData('', 'window-type offscreen\nwin-size 1280 720\naudio-library-name null')
    getModelPath().appendDirectory(str(Path(__file__).resolve().parents[1]))
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