"""On-screen HUD: one full-width command bar along the bottom of the screen.

The rule trace produced by ``rules_log`` is the engine's only evidence that a
rule fired, and until now it went to the console alone. The log panel here is
that same stream made visible while the game is being played.

The bar is *drawn over* the board rather than shrinking the camera's display
region. The picking code extrudes raw mouse coordinates straight through
``base.camLens``, which assumes the region covers the whole window; shrinking
it would silently offset every pick. Nothing is lost by overlaying, because
DirectGui regions suppress mouse-button events, so a click on the bar cannot
reach a unit behind it.

Sections are laid out as fractions of the bar's width and re-flowed on
``window-event``, so the bar survives any window shape rather than being 16:9
only.
"""

from collections import deque

from direct.gui.DirectGui import DirectButton, DirectFrame, DGG
from direct.gui.OnscreenText import OnscreenText
from direct.showbase.DirectObject import DirectObject
from panda3d.core import (TextNode, TextProperties, TextPropertiesManager,
                          TransparencyAttrib)

import gui_theme as T
import rules_log


# Log category -> (inline text-properties name, colour).
# The pages are parchment, so these are ink tints rather than the bright
# on-black colours the corner panels used; the same hues, darkened to read.
_CATEGORY_COLOURS = {
    'rule':   ('log_rule',   (0.46, 0.32, 0.04, 1.0)),
    'skip':   ('log_skip',   (0.48, 0.43, 0.34, 1.0)),
    'dice':   ('log_dice',   (0.16, 0.28, 0.46, 1.0)),
    'combat': ('log_combat', T.INK),
    'morale': ('log_morale', (0.62, 0.10, 0.08, 1.0)),
    'good':   ('log_good',   (0.13, 0.40, 0.13, 1.0)),
    'info':   ('log_info',   T.INK),
}

_PHASE_ON = 'hud_phase_on'
_PHASE_OFF = 'hud_phase_off'
_PHASE_ON_COLOUR = (0.52, 0.13, 0.10, 1.0)
_PHASE_OFF_COLOUR = (0.42, 0.36, 0.26, 0.65)

# State chips on the regiment page.
_CHIP_COLOURS = {
    'bad':  ('chip_bad',  (0.62, 0.10, 0.08, 1.0)),
    'good': ('chip_good', (0.13, 0.40, 0.13, 1.0)),
    'note': ('chip_note', (0.46, 0.32, 0.04, 1.0)),
}

_STAT_HI = (0.13, 0.40, 0.13, 1.0)
_STAT_LO = (0.62, 0.10, 0.08, 1.0)

# Models bar: green while the unit is safe, amber past the 50% fall-back
# threshold, red past the 25% heavy-casualties one.
_BAR_FULL = (0.25, 0.45, 0.18, 1.0)
_BAR_MID = (0.72, 0.52, 0.10, 1.0)


def _register_properties():
    """Register the inline colours once per process."""
    tpm = TextPropertiesManager.getGlobalPtr()
    entries = list(_CATEGORY_COLOURS.values()) + list(_CHIP_COLOURS.values())
    entries.append((_PHASE_ON, _PHASE_ON_COLOUR))
    entries.append((_PHASE_OFF, _PHASE_OFF_COLOUR))
    for name, colour in entries:
        if not tpm.hasProperties(name):
            tp = TextProperties()
            tp.setTextColor(*colour)
            tpm.setProperties(name, tp)


def _markup(prop: str, text: str) -> str:
    """Wrap *text* in an inline text-properties run."""
    return f"\1{prop}\1{text}\2"


class HUD(DirectObject):
    """Screen furniture. Built once, updated by ``setText``."""

    TRACK = ['DeployPhase', 'StrategyPhase', 'MovementPhase',
             'ShootingPhase', 'CombatPhase']
    LABELS = {'DeployPhase': 'DEPLOY', 'StrategyPhase': 'STRATEGY',
              'MovementPhase': 'MOVEMENT', 'ShootingPhase': 'SHOOTING',
              'CombatPhase': 'COMBAT'}
    # Detours that are not steps of the turn sequence.
    ASIDES = {'SpellPhase': 'CASTING', 'MakeChoice': 'CHOOSING',
              'CampaignPhase': 'CAMPAIGN'}

    LOG_ENTRIES = 6
    LOG_SCALE = 0.026
    LOG_TOP = 0.350
    LOG_BOTTOM = 0.048
    # The ledger measures down from its section top, so its bounds are negative.
    LOG_TOP_V = -0.074
    LOG_BOTTOM_V = -0.402

    # ── Bar geometry ────────────────────────────────────────────────
    BAR_H = 0.46            # aspect2d is 2 units tall, so this is 23% of it
    LEDGER_W = 0.62         # and 2*aspect wide, so this is the ledger's slice
    GUTTER = 0.014
    HEAD_Z = 0.408          # section headings
    RULE_Z = 0.390          # the gold rule under them

    HORIZONTAL = 'horizontal'
    VERTICAL = 'vertical'

    # Section spans as fractions of the bar's length, in reading order: left to
    # right along the bottom bar, top to bottom down the ledger.
    SECTIONS_H = {
        'rail':       (0.000, 0.050),
        'regiment':   (0.050, 0.315),
        'log':        (0.315, 0.560),
        'centre':     (0.560, 0.790),
        'rules':      (0.790, 0.838),
        'objectives': (0.838, 0.930),
        'end':        (0.930, 1.000),
    }
    SECTIONS_V = {
        'turn':       (0.000, 0.058),
        'regiment':   (0.058, 0.348),
        'phase':      (0.348, 0.508),
        'dice':       (0.508, 0.622),
        'tabs':       (0.622, 0.838),
        'controls':   (0.838, 0.905),
        'end':        (0.905, 1.000),
    }

    # Regiment page. Column positions are fractions across the section, so the
    # stat table stays aligned without needing a fixed-width face.
    STAT_KEYS = ('M', 'WS', 'BS', 'S', 'T', 'W', 'I', 'A', 'Ld')
    STAT_AVERAGE = {'M': 4, 'WS': 3, 'BS': 3, 'S': 3, 'T': 3,
                    'W': 1, 'I': 3, 'A': 1, 'Ld': 7}
    PORTRAIT = 0.24         # side of the square portrait slot
    TEXT_X = 0.28           # where the text column starts, past the portrait
    COL_FIRST = 0.32
    COL_LAST = 0.94
    DETAIL_LINES = 2
    DETAIL_TOP = 0.114
    DETAIL_STEP = 0.030
    BAR_Z = 0.138           # models bar
    BAR_HEIGHT = 0.024
    MODELS_Z = 0.178
    CHIPS_Z = 0.052

    DICE_SLOTS = 5
    DICE_SIZE = 0.085
    PHASE_SCALE = 0.022

    # Hover tooltip. Screen space, so it can be measured and kept on screen;
    # as world-space text on the unit it simply ran off the bottom edge.
    TIP_SCALE = 0.030
    TIP_PAD = 0.020
    TIP_GAP = 0.015
    # Stat-table column stop. The theme font is proportional, so Unit._stat_table
    # separates its columns with tabs; this is the width they snap to. Wide
    # enough for the broadest cell a stat can produce ("10+" at 1.54 units).
    TIP_TAB_WIDTH = 1.9

    def __init__(self, orientation=HORIZONTAL):
        DirectObject.__init__(self)
        _register_properties()

        self.orientation = orientation
        self._vertical = orientation == self.VERTICAL
        self.SECTIONS = self.SECTIONS_V if self._vertical else self.SECTIONS_H

        self._entries = deque(maxlen=self.LOG_ENTRIES)
        self._active_phase = self.TRACK[0]
        self._visible = True
        self._collapsed = False
        # Set by the vertical build; the horizontal one has no equivalents.
        self._phase_rows = {}
        self._dice_total = None
        self._tabs = {}
        self._turn_state = None
        self._dice_state = []

        font = T.get_font()
        self._font = font

        # Every positioned child is registered in _flex as a fraction across
        # its section, and re-placed by _layout when the window changes shape.
        self._pages = {}
        self._flex = []
        self._stretch = []
        self._log_x = 0.0
        self._log_section = 'tabs' if self._vertical else 'log'

        if self._vertical:
            self._bar = DirectFrame(
                parent=base.a2dRightCenter,
                frameTexture=T.TEX_COMMAND_BAR,
                frameColor=(1, 1, 1, 1),
                frameSize=(-self.LEDGER_W, 0, -1, 1),
                relief=DGG.FLAT,
            )
        else:
            self._bar = DirectFrame(
                parent=base.a2dBottomCenter,
                frameTexture=T.TEX_COMMAND_BAR,
                frameColor=(1, 1, 1, 1),
                frameSize=(-1, 1, 0, self.BAR_H),
                relief=DGG.FLAT,
            )
        self._bar.setTransparency(TransparencyAttrib.MAlpha)
        self._widgets = [self._bar]

        if self._vertical:
            self._build_turn_v(font)
            self._build_regiment_v(font)
            self._build_phase_v(font)
            self._build_dice_v(font)
            self._build_tabs_v(font)
            self._build_controls_v(font)
            self._build_end(font, -0.098, 0.086)
            self._build_collapse(font)
        else:
            self._build_rail(font)
            self._build_regiment(font)
            self._build_log(font)
            self._build_centre(font)
            self._build_rules(font)
            self._build_objectives(font)
            self._build_end(font, 0.210, 0.105)
        self._build_tooltip(font)

        self._layout()
        self.accept('window-event', self._on_window)

        self.set_phase(self._active_phase)
        self._redraw_log()
        self.set_dice([])

        self.accept('hud-turn', self.set_turn)
        self.accept('hud-phase', self.set_phase)
        self.accept('hud-log', self.log)
        self.accept('hud-unit', self.show_unit)
        self.accept('hud-dice', self.set_dice)
        rules_log.add_listener(self._on_rule)

    # ─── Layout ───────────────────────────────────────────────────────

    def _section(self, name, page=True, z0=0.034, z1=None):
        """Anchor node for one section, with an optional parchment page."""
        z1 = self.RULE_Z if z1 is None else z1
        anchor = self._bar.attachNewNode(f'sec-{name}')
        frame = None
        if page:
            frame = DirectFrame(
                parent=anchor, frameTexture=T.TEX_PARCHMENT,
                frameColor=(1, 1, 1, 1), relief=DGG.FLAT,
                frameSize=(0, 1, z0, z1))
            frame.setTransparency(TransparencyAttrib.MAlpha)
        self._pages[name] = (anchor, frame, z0, z1)
        return anchor

    def _place(self, node, section, frac_x, z):
        """Put *node* at a fraction across its section. Re-applied on resize."""
        self._flex.append((node, section, frac_x, z))
        return node

    def _span(self, frame, section, f0, f1, z):
        """A frame whose width spans *f0*..*f1* of its section."""
        self._stretch.append((frame, section, f0, f1, z))
        return frame

    def _label(self, parent, section, frac_x, z, scale, colour,
               align=TextNode.ALeft, text=''):
        node = T.styled_text(text=text, pos=(0, z), scale=scale, fg=colour,
                             align=align, parent=parent, font=self._font)
        return self._place(node, section, frac_x, z)

    def _heading(self, parent, section, text):
        """Small-caps gold section title with a rule beneath it."""
        self._label(parent, section, 0.5, self.HEAD_Z, 0.026, T.GOLD,
                    TextNode.ACenter, text)
        rule = DirectFrame(parent=parent, frameColor=T.SEPARATOR,
                           frameSize=(0, 1, -0.0015, 0.0015))
        return self._span(rule, section, 0.0, 1.0, self.RULE_Z + 0.012)

    def _section_width(self, name, total):
        """Space across the bar's short axis: the section's own slice when it
        tiles left to right, the whole ledger when sections stack downwards."""
        if self._vertical:
            return self.LEDGER_W - 2 * self.GUTTER
        f0, f1 = self.SECTIONS[name]
        return total * (f1 - f0) - 2 * self.GUTTER

    def _section_length(self, name, total):
        """Space along the bar's long axis."""
        f0, f1 = self.SECTIONS[name]
        return total * (f1 - f0) - 2 * self.GUTTER

    def _on_window(self, win=None):
        self._layout()

    def _layout(self):
        """Re-flow for the current window shape."""
        a = base.getAspectRatio()
        if self._vertical:
            # The ledger runs the full height, so its long axis is aspect2d's
            # fixed two units and nothing about it moves with the aspect ratio.
            total = 2.0
            self._bar['frameSize'] = (-self.LEDGER_W, 0, -1, 1)
        else:
            total = 2.0 * a
            self._bar['frameSize'] = (-a, a, 0, self.BAR_H)

        for name, (anchor, frame, z0, z1) in self._pages.items():
            f0, _ = self.SECTIONS[name]
            cross = self._section_width(name, total)
            if self._vertical:
                anchor.setPos(-self.LEDGER_W + self.GUTTER, 0,
                              1 - total * f0 - self.GUTTER)
                if frame is not None:
                    frame['frameSize'] = (
                        0, cross, -self._section_length(name, total), 0)
            else:
                anchor.setPos(-a + total * f0 + self.GUTTER, 0, 0)
                if frame is not None:
                    frame['frameSize'] = (0, cross, z0, z1)

        for frame, name, f0, f1, z in self._stretch:
            width = self._section_width(name, total)
            size = frame['frameSize']
            frame['frameSize'] = (0, (f1 - f0) * width, size[2], size[3])
            frame.setPos(f0 * width, 0, z)

        for node, name, frac, z in self._flex:
            x = frac * self._section_width(name, total)
            # OnscreenText overrides setPos with a flat (x, z) signature.
            if isinstance(node, OnscreenText):
                node.setPos(x, z)
            else:
                node.setPos(x, 0, z)

        log_w = self._section_width(self._log_section, total)
        self._log_x = 0.02 * log_w
        self._log_text['wordwrap'] = (log_w - 0.04) / self.LOG_SCALE
        self._fit_phase()
        self._redraw_log()

    def _fit_phase(self):
        """Shrink the phase track if the five labels are wider than the section.

        The track is one text node rather than five chips, so it cannot wrap;
        on a narrow window it would otherwise run out over the neighbouring
        panels. The vertical ledger lists the phases instead, and needs none
        of this.
        """
        if self._vertical:
            return
        width = self._section_width('centre', 2.0 * base.getAspectRatio()) * 0.94
        natural = self._phase.textNode.getWidth() * self.PHASE_SCALE
        scale = self.PHASE_SCALE
        if natural > width:
            scale *= width / natural
        self._phase.setScale(scale)

    # ─── Sections ─────────────────────────────────────────────────────

    def _slot(self, parent, w, h, centred=False):
        """An empty framed slot, reserved for art that is not drawn yet."""
        size = ((-w / 2, w / 2, -h / 2, h / 2) if centred else (0, w, 0, h))
        frame = DirectFrame(parent=parent, frameTexture=T.TEX_SLOT,
                            frameColor=(1, 1, 1, 1), relief=DGG.FLAT,
                            frameSize=size)
        frame.setTransparency(TransparencyAttrib.MAlpha)
        return frame

    def _build_rail(self, font):
        """Two navigation slots down the left edge — army list and spellbook."""
        anchor = self._section('rail', page=False)
        for text, z in (('ARMY', 0.310), ('BOOK', 0.130)):
            self._place(self._slot(anchor, 0.13, 0.13, centred=True),
                        'rail', 0.5, z)
            self._label(anchor, 'rail', 0.5, z - 0.095, 0.019, T.CREAM,
                        TextNode.ACenter, text)


    def _build_regiment(self, font):
        anchor = self._section('regiment')
        self._heading(anchor, 'regiment', 'SELECTED REGIMENT')

        self._portrait = self._slot(anchor, self.PORTRAIT, self.PORTRAIT)
        self._place(self._portrait, 'regiment', 0.02, 0.140)

        self._card_name = self._label(anchor, 'regiment', self.TEXT_X, 0.330,
                                      0.036, T.GOLD)
        self._card_sub = self._label(anchor, 'regiment', self.TEXT_X, 0.294,
                                     0.022, T.INK_FADED)

        step = (self.COL_LAST - self.COL_FIRST) / (len(self.STAT_KEYS) - 1)
        self._stat_values = {}
        self._regiment_static = []
        for i, key in enumerate(self.STAT_KEYS):
            frac = self.COL_FIRST + i * step
            self._regiment_static.append(
                self._label(anchor, 'regiment', frac, 0.252, 0.020,
                            T.PARCHMENT_DARK, TextNode.ACenter, key))
            self._stat_values[key] = self._label(
                anchor, 'regiment', frac, 0.212, 0.030, T.INK,
                TextNode.ACenter)

        self._regiment_static.append(
            self._label(anchor, 'regiment', self.TEXT_X, self.MODELS_Z, 0.020,
                        T.PARCHMENT_DARK, text='MODELS'))
        self._card_models = self._label(anchor, 'regiment', self.COL_LAST,
                                        self.MODELS_Z, 0.022, T.INK,
                                        TextNode.ARight)

        # The bar starts past the portrait, so it does not run underneath it.
        self._bar_back = self._span(
            DirectFrame(parent=anchor, frameColor=(0.30, 0.24, 0.16, 0.85),
                        frameSize=(0, 1, 0, self.BAR_HEIGHT)),
            'regiment', self.TEXT_X, self.COL_LAST, self.BAR_Z)
        self._card_bar = self._span(
            DirectFrame(parent=anchor, frameColor=_BAR_FULL,
                        frameSize=(0, 1, 0, self.BAR_HEIGHT)),
            'regiment', self.TEXT_X, self.COL_LAST, self.BAR_Z)

        # 50% splits flee from fall back and 25% is the heavy-casualties Panic
        # threshold; both decide what happens next, so they are marked.
        span = self.COL_LAST - self.TEXT_X
        self._bar_ticks = []
        for fraction, colour in ((0.50, T.INK), (0.25, (0.4, 0.05, 0.05, 1))):
            tick = DirectFrame(
                parent=anchor, frameColor=colour,
                frameSize=(0, 0.004, -0.004, self.BAR_HEIGHT + 0.004))
            self._bar_ticks.append((tick, fraction))
            self._place(tick, 'regiment', self.TEXT_X + fraction * span,
                        self.BAR_Z)

        self._detail_labels = [
            self._label(anchor, 'regiment', 0.02,
                        self.DETAIL_TOP - i * self.DETAIL_STEP, 0.021, T.INK)
            for i in range(self.DETAIL_LINES)]
        self._card_chips = self._label(anchor, 'regiment', 0.02,
                                       self.CHIPS_Z, 0.021, T.INK)
        self._set_regiment_visible(False)

    def _build_log(self, font):
        anchor = self._section('log')
        self._heading(anchor, 'log', 'BATTLE LOG')
        self._log_text = self._label(anchor, 'log', 0.02, self.LOG_TOP,
                                     self.LOG_SCALE, T.INK)
        self._log_text['wordwrap'] = 30

    def _build_centre(self, font):
        anchor = self._section('centre')
        self._heading(anchor, 'centre', 'RECENT DICE')

        self._dice_slots, self._dice_values = [], []
        span = 1.0 / self.DICE_SLOTS
        for i in range(self.DICE_SLOTS):
            frac = span * (i + 0.5)
            slot = self._slot(anchor, self.DICE_SIZE, self.DICE_SIZE,
                              centred=True)
            self._place(slot, 'centre', frac, 0.300)
            self._dice_slots.append(slot)
            self._dice_values.append(
                self._label(anchor, 'centre', frac, 0.286, 0.034, T.CREAM,
                            TextNode.ACenter))

        self._label(anchor, 'centre', 0.5, 0.206, 0.022, T.PARCHMENT_DARK,
                    TextNode.ACenter, 'TURN PHASE')
        self._phase = self._label(anchor, 'centre', 0.5, 0.152,
                                  self.PHASE_SCALE, T.CREAM, TextNode.ACenter)
        self._turn = self._label(anchor, 'centre', 0.5, 0.096, 0.030, T.GOLD,
                                 TextNode.ACenter)
        self._round = self._label(anchor, 'centre', 0.5, 0.054, 0.022,
                                  T.INK_FADED, TextNode.ACenter)

    def _build_rules(self, font):
        anchor = self._section('rules', page=False)
        self._place(self._slot(anchor, 0.13, 0.19, centred=True),
                    'rules', 0.5, 0.265)
        self._label(anchor, 'rules', 0.5, 0.135, 0.019, T.CREAM,
                    TextNode.ACenter, 'GAME')
        self._label(anchor, 'rules', 0.5, 0.105, 0.019, T.CREAM,
                    TextNode.ACenter, 'RULES')

    def _build_objectives(self, font):
        anchor = self._section('objectives')
        self._heading(anchor, 'objectives', 'OBJECTIVES')
        # No objectives system in the engine yet, so the panel reserves the
        # space rather than inventing a readout.
        self._place(self._slot(anchor, 0.20, 0.30, centred=True),
                    'objectives', 0.5, 0.210)

    def _build_end(self, font, z, scale):
        anchor = self._section('end', page=False)
        # Advances the phase; the FSM owns the turn sequence, so the bar posts
        # an intent rather than reaching into it.
        self._end_btn = DirectButton(
            parent=anchor, text='END\nPHASE', text_font=font,
            text_fg=T.BTN_TEXT, text_scale=0.42, text_pos=(0, 0.16),
            frameTexture=T.TEX_BTN_ROUND, frameColor=(1, 1, 1, 1),
            frameSize=(-1, 1, -1, 1), relief=DGG.FLAT, scale=scale,
            command=lambda: messenger.send('hud-end-phase'))
        self._end_btn.setTransparency(TransparencyAttrib.MAlpha)
        self._end_btn.bind(
            DGG.WITHIN,
            lambda *_: self._end_btn.__setitem__('frameTexture',
                                                 T.TEX_BTN_ROUND_HOVER))
        self._end_btn.bind(
            DGG.WITHOUT,
            lambda *_: self._end_btn.__setitem__('frameTexture',
                                                 T.TEX_BTN_ROUND))
        self._place(self._end_btn, 'end', 0.5, z)


    # ─── Vertical ledger ──────────────────────────────────────────────
    # Sections stack downwards, so these z offsets are measured down from each
    # section's own top rather than up from the floor of the bar.

    V_STAT_ROWS = (('M', 'T'), ('WS', 'W'), ('BS', 'A'), ('S', 'Ld'))

    def _build_turn_v(self, font):
        anchor = self._section('turn')
        self._turn = self._label(anchor, 'turn', 0.04, -0.036, 0.024, T.GOLD)
        self._round = self._label(anchor, 'turn', 0.96, -0.036, 0.019,
                                  T.INK_FADED, TextNode.ARight)
        self._aside = self._label(anchor, 'turn', 0.5, -0.068, 0.017,
                                  _PHASE_ON_COLOUR, TextNode.ACenter)

    def _build_regiment_v(self, font):
        anchor = self._section('regiment')
        self._heading_v(anchor, 'regiment', 'SELECTED REGIMENT')

        self._portrait = self._slot(anchor, 0.155, 0.155)
        self._place(self._portrait, 'regiment', 0.04, -0.235)

        self._card_name = self._label(anchor, 'regiment', 0.24, -0.118,
                                      0.026, T.GOLD)
        self._card_sub = self._label(anchor, 'regiment', 0.24, -0.152,
                                     0.019, T.INK_FADED)

        self._regiment_static = []
        self._regiment_static.append(
            self._label(anchor, 'regiment', 0.24, -0.190, 0.019,
                        T.PARCHMENT_DARK, text='Models'))
        self._card_models = self._label(anchor, 'regiment', 0.96, -0.190,
                                        0.021, T.INK, TextNode.ARight)

        self._bar_back = self._span(
            DirectFrame(parent=anchor, frameColor=(0.30, 0.24, 0.16, 0.85),
                        frameSize=(0, 1, 0, self.BAR_HEIGHT)),
            'regiment', 0.24, 0.96, -0.226)
        self._card_bar = self._span(
            DirectFrame(parent=anchor, frameColor=_BAR_FULL,
                        frameSize=(0, 1, 0, self.BAR_HEIGHT)),
            'regiment', 0.24, 0.96, -0.226)
        self._bar_ticks = []
        for fraction, colour in ((0.50, T.INK), (0.25, (0.4, 0.05, 0.05, 1))):
            tick = DirectFrame(
                parent=anchor, frameColor=colour,
                frameSize=(0, 0.004, -0.004, self.BAR_HEIGHT + 0.004))
            self._bar_ticks.append((tick, fraction))
            self._place(tick, 'regiment', 0.24 + fraction * 0.72, -0.226)

        # Two columns of four, which fits the ledger where a nine-across strip
        # would not.
        self._stat_values = {}
        for row, (left, right) in enumerate(self.V_STAT_ROWS):
            z = -0.290 - row * 0.036
            for col, key in ((0, left), (1, right)):
                x0 = 0.04 + col * 0.49
                self._regiment_static.append(
                    self._label(anchor, 'regiment', x0 + 0.03, z, 0.019,
                                T.PARCHMENT_DARK, TextNode.ACenter, key))
                self._stat_values[key] = self._label(
                    anchor, 'regiment', x0 + 0.40, z, 0.022, T.INK,
                    TextNode.ARight)

        self._detail_labels = [
            self._label(anchor, 'regiment', 0.04, -0.452 - i * 0.030,
                        0.019, T.INK)
            for i in range(self.DETAIL_LINES)]
        self._card_chips = self._label(anchor, 'regiment', 0.04, -0.516,
                                       0.019, T.INK)
        self._set_regiment_visible(False)

    def _heading_v(self, parent, section, text):
        """Section title with its rule, measured down from the section top."""
        self._label(parent, section, 0.5, -0.040, 0.024, T.GOLD,
                    TextNode.ACenter, text)
        rule = DirectFrame(parent=parent, frameColor=T.SEPARATOR,
                           frameSize=(0, 1, -0.0015, 0.0015))
        return self._span(rule, section, 0.0, 1.0, -0.058)

    def _build_phase_v(self, font):
        anchor = self._section('phase')
        self._heading_v(anchor, 'phase', 'TURN PHASE')
        self._phase_rows = {}
        for i, name in enumerate(self.TRACK):
            z = -0.092 - i * 0.046
            self._place(self._slot(anchor, 0.034, 0.034), 'phase', 0.05, z)
            row = DirectFrame(parent=anchor, frameColor=(0.62, 0.50, 0.20, 0.0),
                              frameSize=(0, 1, -0.008, 0.036))
            self._span(row, 'phase', 0.02, 0.98, z)
            label = self._label(anchor, 'phase', 0.16, z + 0.007, 0.020,
                                T.INK_FADED, text=self.LABELS[name])
            self._phase_rows[name] = (row, label)

    def _build_dice_v(self, font):
        anchor = self._section('dice')
        self._heading_v(anchor, 'dice', 'RECENT DICE')
        self._dice_slots, self._dice_values = [], []
        span = 1.0 / self.DICE_SLOTS
        for i in range(self.DICE_SLOTS):
            frac = span * (i + 0.5)
            slot = self._slot(anchor, 0.072, 0.072, centred=True)
            self._place(slot, 'dice', frac, -0.128)
            self._dice_slots.append(slot)
            self._dice_values.append(
                self._label(anchor, 'dice', frac, -0.140, 0.030, T.CREAM,
                            TextNode.ACenter))
        self._dice_total = self._label(anchor, 'dice', 0.5, -0.196, 0.021,
                                       T.INK, TextNode.ACenter)

    def _build_tabs_v(self, font):
        anchor = self._section('tabs')
        names = (('log', 'BATTLE LOG'), ('rules', 'GAME RULES'),
                 ('objectives', 'OBJECTIVES'))
        self._tabs = {}
        self._tab_panels = {}
        for i, (key, text) in enumerate(names):
            btn = DirectButton(
                parent=anchor, text=text, text_font=font,
                text_fg=T.CREAM, text_scale=0.016, text_pos=(0, -0.006),
                frameColor=(0.30, 0.24, 0.16, 0.85),
                frameSize=(-0.098, 0.098, -0.018, 0.018),
                relief=DGG.FLAT,
                command=self.show_tab, extraArgs=[key])
            self._place(btn, 'tabs', 0.17 + i * 0.33, -0.026)
            self._tabs[key] = btn

        # One panel behind all three tabs; only the log has anything to say.
        self._log_text = self._label(anchor, 'tabs', 0.03, -0.070,
                                     self.LOG_SCALE, T.INK)
        self._log_text['wordwrap'] = 30
        self._tab_panels['log'] = [self._log_text]
        for key, z in (('rules', -0.230), ('objectives', -0.230)):
            slot = self._slot(anchor, 0.22, 0.22, centred=True)
            self._place(slot, 'tabs', 0.5, z)
            self._tab_panels[key] = [slot]
        self.show_tab('log')

    def _build_controls_v(self, font):
        anchor = self._section('controls', page=False)
        for i, text in enumerate(('ARMY', 'SPELLBOOK')):
            frac = 0.27 + i * 0.46
            self._place(self._slot(anchor, 0.16, 0.070, centred=True),
                        'controls', frac, -0.044)
            self._label(anchor, 'controls', frac, -0.052, 0.017, T.CREAM,
                        TextNode.ACenter, text)

    def _build_collapse(self, font):
        """Tab on the ledger's outer edge that folds it away."""
        self._collapse_btn = DirectButton(
            parent=self._bar, text='>', text_font=font, text_fg=T.CREAM,
            text_scale=0.9, text_pos=(0, -0.32),
            frameColor=(0.30, 0.24, 0.16, 0.95),
            frameSize=(-0.5, 0.5, -1.2, 1.2), relief=DGG.FLAT, scale=0.045,
            pos=(-self.LEDGER_W - 0.021, 0, 0),
            command=self.toggle_collapse)

    def toggle_collapse(self):
        """Slide the ledger off the edge, leaving only its handle."""
        self._collapsed = not self._collapsed
        self._bar.setX(self.LEDGER_W if self._collapsed else 0)
        self._collapse_btn['text'] = '<' if self._collapsed else '>'
        messenger.send('hud-layout-changed')

    def show_tab(self, key):
        """Raise one of the ledger's tabbed panels."""
        for name, nodes in self._tab_panels.items():
            for node in nodes:
                node.show() if name == key else node.hide()
        for name, btn in self._tabs.items():
            btn['frameColor'] = ((0.52, 0.42, 0.26, 0.95) if name == key
                                 else (0.30, 0.24, 0.16, 0.85))

    # ─── Hover tooltip ────────────────────────────────────────────────

    def _build_tooltip(self, font):
        self._tip_frame = DirectFrame(
            parent=base.aspect2d,
            frameColor=(0.10, 0.08, 0.06, 0.94),
            frameSize=(0, 0, 0, 0),
            relief=DGG.FLAT,
        )
        self._tip_frame.setTransparency(TransparencyAttrib.MAlpha)
        self._tip_frame.setBin('gui-popup', 0)
        self._tip_text = T.styled_text(
            text='', pos=(0, 0), scale=self.TIP_SCALE, fg=T.CREAM,
            align=TextNode.ALeft, parent=self._tip_frame, font=font)
        self._tip_text.textNode.setTabWidth(self.TIP_TAB_WIDTH)
        self._tip_frame.hide()
        self._widgets.append(self._tip_frame)

    def show_tooltip(self, text: str, x: float, y: float):
        """Show *text* with its top-left corner at the cursor (*x*, *y* in
        aspect2d coordinates), nudged just far enough to stay on screen."""
        if not text:
            self.hide_tooltip()
            return

        self._tip_text.setText(text)
        node = self._tip_text.textNode
        s = self.TIP_SCALE
        left, right = node.getLeft() * s, node.getRight() * s
        bottom, top = node.getBottom() * s, node.getTop() * s
        pad = self.TIP_PAD
        self._tip_frame['frameSize'] = (left - pad, right + pad,
                                        bottom - pad, top + pad)

        # Put the top-left of the text block at the cursor, offset by a gap so
        # the pointer does not sit on the first line.
        px = x + self.TIP_GAP - left
        pz = y - self.TIP_GAP - top

        # Then shift by the least amount that brings it back on screen, so the
        # anchor stays as near the cursor as it can.
        limit_x, limit_y = base.getAspectRatio(), 1.0
        x0, x1 = px + left - pad, px + right + pad
        y0, y1 = pz + bottom - pad, pz + top + pad
        if x1 > limit_x:
            px -= x1 - limit_x
        if x0 < -limit_x:
            px += -limit_x - x0
        if y0 < -limit_y:
            pz += -limit_y - y0
        if y1 > limit_y:
            pz -= y1 - limit_y

        self._tip_frame.setPos(px, 0, pz)
        self._tip_frame.show()

    def hide_tooltip(self):
        self._tip_frame.hide()

    # ─── Regiment page ────────────────────────────────────────────────

    def _set_regiment_visible(self, visible: bool):
        """Blank the readouts when nothing is selected.

        The parchment and the heading stay: hiding the whole page would leave
        a hole in the middle of the bar.
        """
        nodes = [self._card_name, self._card_sub, self._card_models,
                 self._card_chips, self._card_bar, self._bar_back]
        nodes += self._detail_labels
        nodes += self._regiment_static
        nodes += list(self._stat_values.values())
        nodes += [tick for tick, _ in self._bar_ticks]
        for node in nodes:
            node.show() if visible else node.hide()

    def show_unit(self, info: dict):
        """Fill the regiment page. *info* is built by the caller, which owns
        the rules; the page only lays it out."""
        self._set_regiment_visible(True)
        self._card_name.setText(info.get('name', ''))

        bits = []
        if info.get('troop_type'):
            bits.append(info['troop_type'])
        if info.get('us'):
            bits.append(f"US {info['us']}")
        if info.get('files'):
            bits.append(f"{info['files']}x{info['ranks']}")
        bits.append(f"Save {info.get('save') or 'none'}")
        if info.get('ward'):
            bits.append(f"Ward {info['ward']}")
        if info.get('rank_bonus'):
            bits.append(f"Rank +{info['rank_bonus']}")
        self._card_sub.setText("   ".join(bits))

        stats = info.get('stats') or {}
        for key, node in self._stat_values.items():
            value = stats.get(key, '-')
            node.setText(str(value))
            average = self.STAT_AVERAGE.get(key)
            try:
                numeric = int(value)
            except (TypeError, ValueError):
                node['fg'] = T.INK
                continue
            node['fg'] = (_STAT_HI if numeric > average
                          else _STAT_LO if numeric < average else T.INK)

        details = (info.get('details') or [])[:self.DETAIL_LINES]
        for i, node in enumerate(self._detail_labels):
            node.setText(details[i] if i < len(details) else '')

        models = info.get('models', 0)
        start = max(1, info.get('start_models', models) or 1)
        self._card_models.setText(f"{models} / {start}")
        fraction = max(0.0, min(1.0, models / start))
        self._card_bar.setScale(max(fraction, 0.001), 1, 1)
        self._card_bar['frameColor'] = (
            T.RED_WAX if fraction <= 0.25 else
            _BAR_MID if fraction <= 0.5 else _BAR_FULL)

        chips = info.get('chips') or []
        self._card_chips.setText("  ".join(
            _markup(_CHIP_COLOURS.get(tone, _CHIP_COLOURS['note'])[0],
                    f"[{text}]")
            for text, tone in chips))

    def clear_unit(self):
        self._set_regiment_visible(False)

    # ─── Recent dice ──────────────────────────────────────────────────

    def set_dice(self, values):
        """Show the faces of the last roll to settle, newest roll only.

        Unused slots stay on screen dimmed rather than disappearing, so the
        strip does not change width between a 2D6 and a 5D6 roll.
        """
        values = [v for v in list(values)[-self.DICE_SLOTS:] if v is not None]
        self._dice_state = values
        for i, node in enumerate(self._dice_values):
            node.setText(str(values[i]) if i < len(values) else '')
        for i, slot in enumerate(self._dice_slots):
            slot.setColorScale(1, 1, 1, 1.0 if i < len(values) else 0.35)
        if self._dice_total is not None:
            self._dice_total.setText(f"Total: {sum(values)}" if values else '')


    # ─── Turn / phase ─────────────────────────────────────────────────

    def set_turn(self, player: int, round_no: int, max_rounds: int):
        # Kept so a rebuild in the other orientation can restore it.
        self._turn_state = (player, round_no, max_rounds)
        self._turn.setText(f"PLAYER {player}")
        self._round.setText(f"Round {round_no} / {max_rounds}")

    def set_phase(self, phase: str):
        """Light the current step of the turn sequence."""
        if phase in self.TRACK:
            self._active_phase = phase
        aside = self.ASIDES.get(phase)

        if self._phase_rows:
            for name, (row, label) in self._phase_rows.items():
                on = name == self._active_phase
                row['frameColor'] = ((0.62, 0.50, 0.20, 0.55) if on
                                     else (0.62, 0.50, 0.20, 0.0))
                label['fg'] = _PHASE_ON_COLOUR if on else T.INK_FADED
            self._aside.setText(f"[{aside}]" if aside else '')
            return

        chips = [_markup(_PHASE_ON if name == self._active_phase else _PHASE_OFF,
                         self.LABELS[name])
                 for name in self.TRACK]
        line = _markup(_PHASE_OFF, '   ').join(chips)

        if aside:
            line += _markup(_PHASE_ON, f"   [{aside}]")
        self._phase.setText(line)
        self._fit_phase()

    # ─── Battle log ───────────────────────────────────────────────────

    def log(self, text: str, category: str = 'info'):
        """Post one line to the battle log."""
        if category not in _CATEGORY_COLOURS:
            category = 'info'
        self._entries.append((category, text))
        self._redraw_log()

    def clear_log(self):
        self._entries.clear()
        self._redraw_log()

    def _on_rule(self, kind: str, rule: str, subject: str, detail: str):
        if kind == 'skipped':
            self.log(f"{rule} — {subject}: not claimed ({detail})", 'skip')
        else:
            self.log(f"{rule} — {subject}: {detail}", 'rule')

    def _redraw_log(self):
        lines = []
        for category, text in self._entries:
            prop = _CATEGORY_COLOURS[category][0]
            lines.append(_markup(prop, f"\u2022 {text}"))
        self._log_text.setText('\n'.join(lines))

        # Grow the block upward from the bottom of the page, so the newest
        # line always sits in the same place however much the lines wrap.
        height = self._log_text.textNode.getHeight() * self.LOG_SCALE
        top = self.LOG_TOP_V if self._vertical else self.LOG_TOP
        bottom = self.LOG_BOTTOM_V if self._vertical else self.LOG_BOTTOM
        self._log_text.setPos(self._log_x, min(bottom + height, top))

    # ─── Chrome visibility ────────────────────────────────────────────

    def view_shift(self):
        """How far the 3D view must move for the board to sit in the middle of
        what the HUD leaves: fractions of the window, rightwards then upwards.

        The bar covers the bottom of the screen and the ledger the right-hand
        side, so which axis is displaced depends on the orientation. The caller
        turns this into a lens film offset.
        """
        if self._collapsed:
            return (0.0, 0.0)
        if self._vertical:
            # aspect2d is 2*aspect wide, so the ledger's share of the width is
            # LEDGER_W / (2*aspect), and the board centres half that to the left.
            return (-self.LEDGER_W / (4.0 * base.getAspectRatio()), 0.0)
        return (0.0, self.BAR_H / 4.0)

    def snapshot(self):
        """The state a rebuild in the other orientation has to carry over."""
        return {'entries': list(self._entries),
                'phase': self._active_phase,
                'turn': self._turn_state,
                'dice': self._dice_state,
                'collapsed': self._collapsed}

    def restore(self, state):
        self._entries.extend(state.get('entries') or ())
        self._redraw_log()
        if state.get('phase'):
            self.set_phase(state['phase'])
        if state.get('turn'):
            self.set_turn(*state['turn'])
        self.set_dice(state.get('dice') or [])
        if state.get('collapsed') and self._vertical:
            self.toggle_collapse()

    def toggle(self):
        """Hide or show all 2D chrome (for screenshots)."""
        self._visible = not self._visible
        for widget in self._widgets:
            widget.show() if self._visible else widget.hide()

    def destroy(self):
        rules_log.remove_listener(self._on_rule)
        self.ignoreAll()
        for widget in self._widgets:
            widget.destroy()
        self._widgets = []
