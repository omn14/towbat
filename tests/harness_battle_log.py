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
