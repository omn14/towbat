"""The in-game question box.

A prompt with a row of answers, drawn in screen space over the board. This was
a row of clickable cubes standing on the table until the challenge rules put a
long question on it, and it became plain that the player was being asked to
choose without being told what the question was.

The interface is unchanged for callers: build one, add `mouseActivate` as a
task, and read `choice` once `choiceMade` goes true.
"""

from direct.gui.DirectGui import DGG, DirectButton, DirectFrame
from direct.showbase.DirectObject import DirectObject
from panda3d.core import TextNode, TransparencyAttrib

import gui_theme


class Choice:
    # Panel geometry, in aspect2d units.
    HALF_W = 0.5
    ROW_H = 0.115
    LINE_H = 0.055
    PAD = 0.022
    BORDER = 0.012
    # Three across leaves each button wide enough for a two-word answer like
    # 'Great Weapon' on one line; at four they wrapped and spilled over the row
    # beneath.
    MAX_PER_ROW = 3
    HEADER_H = 0.072
    TITLE_SCALE = 0.038
    BTN_SCALE = 0.042

    def __init__(self, choices, pos, cancellable=False, descriptions=None,
                 prompt=None, detail=None):
        self.num_choices = len(choices)
        self.choices = choices
        self.choiceMade = False
        self.choice = None
        # Kept for callers that still speak the old language of hit boxes.
        self.hitbox = None
        self.boxes = []
        self.buttons = []
        self.hovered = None
        # Hovering a choice shows its blurb; a spell is unplayable if you
        # cannot read what it does before committing to it.
        self.descriptions = descriptions or {}
        self.detail = None
        self.shown = None
        self.panel = None
        self._build(list(choices), prompt, detail, cancellable)
        self.helper1 = DirectObject()
        if cancellable:
            self.helper1.accept('mouse3', self.onCancel)

    # ─── Building ────────────────────────────────────────────────────────

    def _build(self, names, prompt, detail, cancellable):
        rows = -(-len(names) // self.MAX_PER_ROW) or 1
        inset = self.HALF_W - self.BORDER
        # Laid out downwards from a top edge at zero, because the height of the
        # title bar is not known until its text has been wrapped and measured.
        # The panel is resized to fit and re-centred once the cursor lands.
        self.panel = DirectFrame(
            parent=base.aspect2d, relief=DGG.FLAT,
            frameColor=gui_theme.PARCHMENT_DARK,
            frameSize=(-self.HALF_W, self.HALF_W, -1, 0))
        self.panel.setTransparency(TransparencyAttrib.MAlpha)
        self.panel.setBin('gui-popup', 0)
        sheet = DirectFrame(parent=self.panel, relief=DGG.FLAT,
                            frameColor=(1, 1, 1, 1),
                            frameTexture=gui_theme.TEX_PARCHMENT,
                            frameSize=(-inset, inset, -1, -self.BORDER))

        z = -self.BORDER
        # The question has to read against parchment, and a title bar is what
        # tells a panel from the board behind it.
        bar = DirectFrame(parent=self.panel, relief=DGG.FLAT,
                          frameColor=gui_theme.RED_WAX,
                          frameSize=(-inset, inset, z - self.HEADER_H, z))
        title = gui_theme.styled_text(
            text=(prompt or "Choose"), parent=self.panel, pos=(0, 0),
            scale=self.TITLE_SCALE, fg=gui_theme.GOLD, align=TextNode.ACenter,
            wordwrap=(inset * 2 - 0.05) / self.TITLE_SCALE)
        node = title.textNode
        line = node.getLineHeight() * self.TITLE_SCALE
        text_h = max(line, node.getHeight() * self.TITLE_SCALE)
        header_h = max(self.HEADER_H, text_h + 0.03)
        bar['frameSize'] = (-inset, inset, z - header_h, z)
        # The anchor is the first line's baseline, so a two-line title has to be
        # pushed up by half a block to sit centred in its bar.
        title.setPos(0, z - header_h / 2.0 + text_h / 2.0 - line * 0.78)
        z -= header_h + self.PAD

        if detail:
            gui_theme.styled_text(
                text=detail, parent=self.panel, pos=(0, z - 0.026), scale=0.032,
                fg=gui_theme.INK, align=TextNode.ACenter,
                wordwrap=(inset * 2 - 0.04) / 0.032)
            z -= self.LINE_H

        if self.descriptions:
            self.detail = gui_theme.styled_text(
                text="", parent=self.panel, pos=(0, z - 0.024), scale=0.03,
                fg=gui_theme.INK_FADED, align=TextNode.ACenter,
                wordwrap=(inset * 2 - 0.04) / 0.03)
            z -= self.LINE_H

        for i, name in enumerate(names):
            row, col = divmod(i, self.MAX_PER_ROW)
            across = min(self.MAX_PER_ROW, len(names) - row * self.MAX_PER_ROW)
            width = (inset * 2 - 0.03) / across
            x = (col - (across - 1) / 2.0) * width
            self.buttons.append(self._button(
                name, (x, 0, z - self.ROW_H * (row + 0.62)),
                width * 0.94, primary=(i == 0)))
        z -= self.ROW_H * rows

        if cancellable:
            gui_theme.styled_text(
                text="right-click to cancel", parent=self.panel,
                pos=(0, z - 0.024), scale=0.024,
                fg=gui_theme.INK_FADED, align=TextNode.ACenter)
            z -= 0.04

        total = -z + self.PAD
        self.panel['frameSize'] = (-self.HALF_W, self.HALF_W, -total, 0)
        sheet['frameSize'] = (-inset, inset, -total + self.BORDER, -self.BORDER)
        self.panel.setPos(0, 0, 0.30 + total / 2.0)

    def _button(self, name, pos, width, primary=False):
        """One answer. The first is the affirmative, and is dressed as such."""
        half = width / 2.0 / self.BTN_SCALE
        colour = gui_theme.BTN_RED if primary else gui_theme.BTN_NEUTRAL
        hover = (gui_theme.BTN_RED_HOVER if primary
                 else gui_theme.BTN_NEUTRAL_HOVER)
        # Shown capitalised without touching the value the caller compares.
        label = name.replace('\n', ' ')
        label = label[:1].upper() + label[1:]
        btn = DirectButton(
            parent=self.panel, text=label,
            text_font=gui_theme.get_font(), text_fg=gui_theme.BTN_TEXT,
            text_align=TextNode.ACenter, text_pos=(0, -0.32),
            text_wordwrap=half * 1.85, text_scale=0.88,
            scale=self.BTN_SCALE, pos=pos, relief=DGG.FLAT,
            frameColor=colour, frameSize=(-half, half, -0.75, 1.15),
            command=self._pick, extraArgs=[name])
        btn.bind(DGG.ENTER, lambda *_: self._enter(name, btn, hover))
        btn.bind(DGG.EXIT, lambda *_: self._leave(name, btn, colour))
        return btn

    # ─── Hovering ────────────────────────────────────────────────────────

    def _enter(self, name, btn, colour):
        btn['frameColor'] = colour
        self.hovered = name
        self._showDetail(name)

    def _leave(self, name, btn, colour):
        btn['frameColor'] = colour
        if self.hovered == name:
            self.hovered = None
            self._showDetail(None)

    def _showDetail(self, name):
        if self.detail is None or name == self.shown:
            return
        self.shown = name
        self.detail.setText(self.descriptions.get(name, ""))

    # ─── Answering ───────────────────────────────────────────────────────

    def _pick(self, name):
        print(f"Choice selected: {name}")
        self.choice = name
        taskMgr.add(self.cleanup())

    def onCancel(self):
        """Right-click closes the menu without choosing; the caller sees None."""
        print("Choice cancelled")
        self.choice = None
        taskMgr.add(self.cleanup())

    def mouseActivate(self, task):
        """Wait for an answer; the buttons report themselves now."""
        return task.done if self.choiceMade else task.cont

    async def cleanup(self):
        self.choiceMade = True
        self.helper1.ignore('mouse3')
        for btn in self.buttons:
            btn.destroy()
        self.buttons = []
        if self.detail is not None:
            self.detail.destroy()
            self.detail = None
        if self.panel is not None:
            self.panel.destroy()
            self.panel = None
