"""Battle log scrolling, measured against a real HUD.

The HUD needs a ShowBase, so it sits behind the same boundary as
`harness_align.py`: run it directly rather than under pytest.

    python tests/harness_battle_log.py

It prints the text node's position at each scroll extreme and writes two
screenshots, because the failure this was written for is a visual one — the
log used to clamp at the top of its page, which pushed the newest lines out
through the bottom where nothing showed them.
"""

import os
import sys

from panda3d.core import loadPrcFileData, getModelPath, PNMImage, Point2, Point3

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

loadPrcFileData("", "window-type offscreen\nwin-size 1280 720\n"
                    "audio-library-name null")
getModelPath().appendDirectory(ROOT)

from direct.showbase.ShowBase import ShowBase   # noqa: E402

base = ShowBase()

import hud   # noqa: E402


def z(h):
    """The text node's z; OnscreenText keeps a flat (x, z) position."""
    return round(h._log_text.getPos()[-1], 4)


class _Pointer:
    """Stands in for the mouse: the wheel only scrolls over the log page."""

    def __init__(self):
        self.at = None

    def hasMouse(self):
        return self.at is not None

    def getMouse(self):
        return Point2(*self.at)

    def over(self, hud_, inside=True):
        """Put the pointer in the middle of the log page, or well off it."""
        top = hud_.LOG_TOP_V if hud_._vertical else hud_.LOG_TOP
        bottom = hud_.LOG_BOTTOM_V if hud_._vertical else hud_.LOG_BOTTOM
        lo = base.render2d.getRelativePoint(hud_._log_anchor,
                                            Point3(0, 0, bottom))
        hi = base.render2d.getRelativePoint(hud_._log_anchor,
                                            Point3(hud_._log_w, 0, top))
        self.at = ((lo.getX() + hi.getX()) / 2, (lo.getZ() + hi.getZ()) / 2) \
            if inside else (0.0, 0.5)


def main(shots='/tmp'):
    h = hud.HUD()
    base.graphicsEngine.renderFrame()
    top = h.LOG_TOP_V if h._vertical else h.LOG_TOP
    bottom = h.LOG_BOTTOM_V if h._vertical else h.LOG_BOTTOM
    pointer = _Pointer()
    base.mouseWatcherNode = pointer

    print(f"page runs {bottom} .. {top}")
    print(f"empty            : z {z(h)}  (must be finite: an empty TextNode "
          f"measures NaN and poisons the transform)")

    for i in range(30):
        h.log(f"Rule {i} — State Missile Trooper Unit: a long line that wraps "
              f"inside the battle log page")
    base.graphicsEngine.renderFrame()
    print(f"30 entries       : z {z(h)}  scrollable {h._log_max_scroll:.3f}")

    pointer.over(h)
    print(f"pointer on page  : {h.pointer_over_log()}")
    h.scroll_log(3)
    previous_position = z(h)
    h.log('An incoming wrapped event ' * 30)
    assert z(h) == previous_position
    print(f"back three lines : z {z(h)}  scroll {h._log_scroll:.3f}")
    h.scroll_log(1000)
    print(f"oldest           : z {z(h)}  scroll {h._log_scroll:.3f}  "
          f"(z should equal the page top, {top})")
    _shot(shots, 'oldest')
    h.scroll_log(-1000)
    print(f"newest           : z {z(h)}  scroll {h._log_scroll:.3f}")
    _shot(shots, 'newest')

    pointer.over(h, inside=False)
    before = z(h)
    h.scroll_log(5)
    print(f"pointer off page : unchanged {z(h) == before}  "
          f"(the wheel belongs to the camera there)")
    print(f"clipped to page  : {h._log_text.hasScissor()}")

    pointer.over(h)
    h.clear_log()
    print(f"cleared          : z {z(h)}  scroll {h._log_scroll:.3f}")

    from rules_log import log_scope
    for index in range(220):
        with log_scope(round=3, player=1, phase='CombatPhase', combat='Princes vs Knights', initiative=9):
            h.log(f'Dragon Princes: {index} attacks -> 4 hits -> 1 wound -> 0 slain',
                  'combat', 'Dragon Princes', 'Armour 3+ with AP-2 -> 5+; roll [5]: saved')
    h.open_history()
    history = h._history
    _pump_events()
    history.set_details(True)
    _pump_events()
    assert len(history.displayed) == 100
    history.scroll_lines(5)
    _pump_events()
    before = history.text.getPos()
    h.log('A very long incoming event ' * 30)
    _pump_events()
    assert history.text.getPos() == before
    history.page(-1)
    _pump_events()
    assert history.displayed[-1].sequence < h._journal.sequence
    history.set_subject('Missing unit')
    _pump_events()
    assert not history.displayed
    history.set_subject('Dragon Princes')
    _pump_events()
    assert len(history.displayed) == 100
    assert history.units.get() == 'Dragon Princes'
    history.scrollbar['value'] = .5
    _pump_events()
    assert abs(history.scroll / history.max_scroll - .5) < .001
    history.latest()
    _pump_events()
    assert history.scroll == 0 and history.frozen is None
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from unittest.mock import patch
    import json
    with TemporaryDirectory() as directory, patch('battle_log_view.Path', return_value=Path(directory)):
        history.export()
        exported = json.loads(next(Path(directory).glob('*.json')).read_text())
        assert len(exported) == len(h._journal.entries)
        assert 'Armour 3+' in next(Path(directory).glob('*.txt')).read_text()
    with patch('battle_log_view.shutil.which', return_value=None):
        history.copy()
        assert 'Clipboard unavailable' in history.status.getText()
    with patch('battle_log_view.shutil.which', return_value='/usr/bin/wl-copy'), \
            patch('battle_log_view.subprocess.run') as copy:
        history.copy()
        assert copy.call_args.kwargs['input'] == h._journal.export()
    history.redraw()
    _shot(shots, 'history')
    from panda3d.core import FrameBufferProperties, WindowProperties, GraphicsPipe
    properties = WindowProperties.size(720, 960)
    buffer = base.graphicsEngine.makeOutput(base.pipe, 'portrait-history', -10,
        FrameBufferProperties.getDefault(), properties, GraphicsPipe.BFRefuseWindow,
        base.win.getGsg(), base.win)
    region = buffer.makeDisplayRegion()
    region.setCamera(base.cam2d)
    base.setAspectRatio(720 / 960)
    h._layout()
    _pump_events()
    assert history.width < .75
    for button, _, _ in history.controls:
        frame = button['frameSize']
        assert abs(button.getX()) + max(abs(frame[0]), abs(frame[1])) < history.width
    base.graphicsEngine.renderFrame()
    image = PNMImage()
    buffer.getScreenshot(image)
    image.write(os.path.join(shots, 'battle_log_history_portrait.png'))
    base.graphicsEngine.removeWindow(buffer)
    base.setAspectRatio(1280 / 720)
    h._layout()
    history.set_mode('Debug')
    history.latest()
    _pump_events()
    assert history.scroll == 0
    h.close_history()
    from rules_log import battle_log
    battle_log('Charge moves complete. Remaining Moves.', 'info')
    _pump_events()
    assert h._history is None
    assert h._journal.entries[-1].text == 'Charge moves complete. Remaining Moves.'
    state = h.snapshot()
    h.destroy()
    h = hud.HUD(orientation=hud.HUD.VERTICAL)
    h.restore(state)
    assert h._journal.sequence == state['entries'][-1].sequence
    pointer.over(h)
    assert h.pointer_over_log()
    h.show_tab('rules')
    assert not h.pointer_over_log()
    h.show_tab('log')
    h.open_history()
    _pump_events()
    assert h._history.mode.get() == 'Debug'
    _shot(shots, 'history_vertical')
    import asyncio
    from choiceFunctions import Choice
    choice = Choice(['Hold'], (0, 0, 0), prompt='Chaos Knights: charge reaction')
    choice.choice = 'Hold'
    choice.choiceMade = True
    asyncio.run(choice.cleanup())
    asyncio.run(choice.cleanup())
    messages = [entry for entry in h._journal.entries if entry.text == 'Chaos Knights: charge reaction: Hold']
    assert len(messages) == 1 and messages[0].category == 'debug'
    h.destroy()


def _pump_events():
    """A redraw must not keep generating deferred scrollbar adjustments."""
    import faulthandler
    faulthandler.dump_traceback_later(5, exit=True)
    try:
        for frame in range(2):
            base.eventMgr.doEvents()
            base.graphicsEngine.renderFrame()
    finally:
        faulthandler.cancel_dump_traceback_later()


def _shot(directory, name):
    base.graphicsEngine.renderFrame()
    base.graphicsEngine.renderFrame()
    img = PNMImage()
    base.win.getScreenshot(img)
    path = os.path.join(directory, f"battle_log_{name}.png")
    img.write(path)
    print(f"                   wrote {path}")


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '/tmp')
