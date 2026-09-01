"""On-screen HUD: turn banner, phase track and the battle log.

The rule trace produced by ``rules_log`` is the engine's only evidence that a
rule fired, and until now it went to the console alone. The log panel here is
that same stream made visible while the game is being played.

Everything is anchored to Panda3D's aspect2d corner nodes rather than fixed
aspect2d coordinates, so the layout survives any window shape.
"""

from collections import deque

from direct.gui.DirectGui import DirectFrame, DGG
from direct.showbase.DirectObject import DirectObject
from panda3d.core import (TextNode, TextProperties, TextPropertiesManager,
                          TransparencyAttrib)

import gui_theme as T
import rules_log


# Log category -> (inline text-properties name, colour).
_CATEGORY_COLOURS = {
    'rule':   ('log_rule',   T.GOLD),
    'skip':   ('log_skip',   (0.58, 0.55, 0.48, 1.0)),
    'dice':   ('log_dice',   (0.81, 0.92, 1.00, 1.0)),
    'combat': ('log_combat', T.CREAM),
    'morale': ('log_morale', (1.00, 0.54, 0.54, 1.0)),
    'good':   ('log_good',   (0.43, 0.88, 0.43, 1.0)),
    'info':   ('log_info',   T.CREAM),
}

_PHASE_ON = 'hud_phase_on'
_PHASE_OFF = 'hud_phase_off'

# State chips on the unit card.
_CHIP_COLOURS = {
    'bad':  ('chip_bad',  (1.00, 0.45, 0.45, 1.0)),
    'good': ('chip_good', (0.45, 0.90, 0.45, 1.0)),
    'note': ('chip_note', T.GOLD),
}

_STAT_HI = (0.45, 0.90, 0.45, 1.0)
_STAT_LO = (1.00, 0.45, 0.45, 1.0)


def _register_properties():
    """Register the inline colours once per process."""
    tpm = TextPropertiesManager.getGlobalPtr()
    entries = list(_CATEGORY_COLOURS.values()) + list(_CHIP_COLOURS.values())
    entries.append((_PHASE_ON, T.GOLD))
    entries.append((_PHASE_OFF, (0.92, 0.85, 0.68, 0.40)))
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

    LOG_ENTRIES = 8
    LOG_SCALE = 0.032
    LOG_LEFT = -1.00
    LOG_TOP = 0.500
    LOG_BOTTOM = 0.075
    LOG_TEXT_WIDTH = 0.94

    # Unit card. Columns are placed by hand and centred, which keeps the stat
    # table aligned without a fixed-width face.
    STAT_KEYS = ('M', 'WS', 'BS', 'S', 'T', 'W', 'I', 'A', 'Ld')
    STAT_AVERAGE = {'M': 4, 'WS': 3, 'BS': 3, 'S': 3, 'T': 3,
                    'W': 1, 'I': 3, 'A': 1, 'Ld': 7}
    CARD_LEFT = 0.06
    CARD_TOP = 0.78
    CARD_BOTTOM = 0.06
    COL_FIRST = 0.11
    COL_LAST = 0.97
    BAR_LEFT = 0.08
    BAR_WIDTH = 0.92
    BAR_Z = 0.163
    BAR_HEIGHT = 0.026
    MODELS_Z = 0.215
    CHIPS_Z = 0.093
    DETAIL_LINES = 5
    DETAIL_TOP = 0.455
    DETAIL_STEP = 0.046

    # Hover tooltip. Screen space, so it can be measured and kept on screen;
    # as world-space text on the unit it simply ran off the bottom edge.
    TIP_SCALE = 0.030
    TIP_PAD = 0.020
    TIP_GAP = 0.015

    def __init__(self):
        DirectObject.__init__(self)
        _register_properties()

        self._entries = deque(maxlen=self.LOG_ENTRIES)
        self._active_phase = self.TRACK[0]
        self._visible = True

        font = T.get_font()

        # ── Turn banner (top left) ────────────────────────────────────
        self._turn = T.styled_text(
            text='', pos=(0.03, -0.10), scale=0.055, fg=T.GOLD,
            align=TextNode.ALeft, parent=base.a2dTopLeft, font=font)
        self._round = T.styled_text(
            text='', pos=(0.03, -0.155), scale=0.036, fg=T.CREAM,
            align=TextNode.ALeft, parent=base.a2dTopLeft, font=font)

        # ── Phase track (top centre) ──────────────────────────────────
        self._phase = T.styled_text(
            text='', pos=(0, -0.095), scale=0.034, fg=T.CREAM,
            align=TextNode.ACenter, parent=base.a2dTopCenter, font=font)

        # ── Battle log (bottom right) ─────────────────────────────────
        self._log_frame = DirectFrame(
            parent=base.a2dBottomRight,
            frameColor=T.PANEL_BG,
            frameSize=(-1.05, -0.03, 0.04, 0.62),
            relief=DGG.FLAT,
        )
        self._log_frame.setTransparency(TransparencyAttrib.MAlpha)

        T.styled_text(
            text='BATTLE LOG', pos=(self.LOG_LEFT, 0.555), scale=0.032,
            fg=T.GOLD, align=TextNode.ALeft, parent=self._log_frame,
            font=font, mayChange=False)
        self._log_rule = DirectFrame(
            parent=self._log_frame, frameColor=T.SEPARATOR,
            frameSize=(self.LOG_LEFT, -0.08, -0.002, 0.002),
            pos=(0, 0, 0.538),
        )

        self._log_text = T.styled_text(
            text='', pos=(self.LOG_LEFT, self.LOG_TOP), scale=self.LOG_SCALE,
            fg=T.CREAM, align=TextNode.ALeft, parent=self._log_frame,
            font=font, wordwrap=(self.LOG_TEXT_WIDTH / self.LOG_SCALE))

        self._widgets = [self._turn, self._round, self._phase, self._log_frame]

        self._build_unit_card(font)
        self._build_tooltip(font)

        self.set_phase(self._active_phase)
        self._redraw_log()

        self.accept('hud-turn', self.set_turn)
        self.accept('hud-phase', self.set_phase)
        self.accept('hud-log', self.log)
        self.accept('hud-unit', self.show_unit)
        rules_log.add_listener(self._on_rule)

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

    # ─── Unit card ────────────────────────────────────────────────────

    def _build_unit_card(self, font):
        self._card = DirectFrame(
            parent=base.a2dBottomLeft,
            frameColor=T.PANEL_BG,
            frameSize=(0.03, 1.05, self.CARD_BOTTOM, self.CARD_TOP),
            relief=DGG.FLAT,
        )
        self._card.setTransparency(TransparencyAttrib.MAlpha)
        self._widgets.append(self._card)

        def label(pos, scale, colour, align=TextNode.ALeft, text='',
                  parent=None):
            return T.styled_text(text=text, pos=pos, scale=scale, fg=colour,
                                 align=align, parent=parent or self._card,
                                 font=font)

        self._card_name = label((self.CARD_LEFT, 0.715), 0.052, T.GOLD)
        self._card_sub = label((self.CARD_LEFT, 0.667), 0.030, T.CREAM)

        step = (self.COL_LAST - self.COL_FIRST) / (len(self.STAT_KEYS) - 1)
        self._stat_values = {}
        for i, key in enumerate(self.STAT_KEYS):
            x = self.COL_FIRST + i * step
            label((x, 0.600), 0.028, T.GOLD, TextNode.ACenter, key)
            self._stat_values[key] = label((x, 0.538), 0.042, T.CREAM,
                                           TextNode.ACenter)

        self._detail_labels = [
            label((self.CARD_LEFT, self.DETAIL_TOP - i * self.DETAIL_STEP),
                  0.028, T.CREAM)
            for i in range(self.DETAIL_LINES)]

        # The bar and chips ride up when a unit has fewer detail lines, so an
        # unmounted footslogger's card has no hole in the middle of it.
        self._card_lower = self._card.attachNewNode('card-lower')
        label((self.BAR_LEFT, self.MODELS_Z), 0.026, T.GOLD, text='MODELS',
              parent=self._card_lower)
        self._card_models = label((self.BAR_LEFT + self.BAR_WIDTH,
                                   self.MODELS_Z), 0.028, T.CREAM,
                                  TextNode.ARight, parent=self._card_lower)

        DirectFrame(parent=self._card_lower,
                    frameColor=(0.05, 0.04, 0.03, 0.9),
                    frameSize=(0, self.BAR_WIDTH, 0, self.BAR_HEIGHT),
                    pos=(self.BAR_LEFT, 0, self.BAR_Z))
        self._card_bar = DirectFrame(
            parent=self._card_lower, frameColor=T.GOLD,
            frameSize=(0, self.BAR_WIDTH, 0, self.BAR_HEIGHT),
            pos=(self.BAR_LEFT, 0, self.BAR_Z))
        # 50% splits flee from fall back, 25% is the heavy-casualties Panic
        # threshold; both decide what happens next, so they are marked.
        for fraction, colour in ((0.50, T.CREAM), (0.25, (0.8, 0.2, 0.2, 1))):
            DirectFrame(parent=self._card_lower, frameColor=colour,
                        frameSize=(0, 0.004, -0.004, self.BAR_HEIGHT + 0.004),
                        pos=(self.BAR_LEFT + self.BAR_WIDTH * fraction, 0,
                             self.BAR_Z))

        self._card_chips = label((self.CARD_LEFT, self.CHIPS_Z), 0.030, T.CREAM,
                                 parent=self._card_lower)
        self._card.hide()

    def show_unit(self, info: dict):
        """Fill the unit card. *info* is built by the caller, which owns the
        rules; the card only lays it out."""
        self._card.show()
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
                node['fg'] = T.CREAM
                continue
            node['fg'] = (_STAT_HI if numeric > average
                          else _STAT_LO if numeric < average else T.CREAM)

        details = (info.get('details') or [])[:self.DETAIL_LINES]
        for i, node in enumerate(self._detail_labels):
            node.setText(details[i] if i < len(details) else '')
        # The card shrinks to its content rather than leaving a hole where a
        # footslogger has no mount or weapon line.
        shift = (self.DETAIL_LINES - len(details)) * self.DETAIL_STEP
        self._card_lower.setZ(shift)
        self._card['frameSize'] = (0.03, 1.05,
                                   self.CARD_BOTTOM + shift, self.CARD_TOP)

        models = info.get('models', 0)
        start = max(1, info.get('start_models', models) or 1)
        self._card_models.setText(f"{models} / {start}")
        fraction = max(0.0, min(1.0, models / start))
        self._card_bar.setScale(max(fraction, 0.001), 1, 1)
        self._card_bar['frameColor'] = (
            (0.8, 0.2, 0.2, 1) if fraction <= 0.25 else
            (0.85, 0.65, 0.2, 1) if fraction <= 0.5 else T.GOLD)

        chips = info.get('chips') or []
        self._card_chips.setText("  ".join(
            _markup(_CHIP_COLOURS.get(tone, _CHIP_COLOURS['note'])[0],
                    f"[{text}]")
            for text, tone in chips))

    def clear_unit(self):
        self._card.hide()

    # ─── Turn / phase ─────────────────────────────────────────────────

    def set_turn(self, player: int, round_no: int, max_rounds: int):
        self._turn.setText(f"PLAYER {player}")
        self._round.setText(f"Round {round_no} / {max_rounds}")

    def set_phase(self, phase: str):
        """Light the current step of the turn sequence."""
        if phase in self.TRACK:
            self._active_phase = phase

        chips = [_markup(_PHASE_ON if name == self._active_phase else _PHASE_OFF,
                         self.LABELS[name])
                 for name in self.TRACK]
        line = _markup(_PHASE_OFF, '   ').join(chips)

        aside = self.ASIDES.get(phase)
        if aside:
            line += _markup(_PHASE_ON, f"   [{aside}]")
        self._phase.setText(line)

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

        # Grow the block upward from the bottom of the panel, so the newest
        # line always sits in the same place however much the lines wrap.
        height = self._log_text.textNode.getHeight() * self.LOG_SCALE
        top = min(self.LOG_BOTTOM + height, self.LOG_TOP)
        self._log_text.setPos(self.LOG_LEFT, top)

    # ─── Chrome visibility ────────────────────────────────────────────

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
