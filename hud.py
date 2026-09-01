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


def _register_properties():
    """Register the inline colours once per process."""
    tpm = TextPropertiesManager.getGlobalPtr()
    entries = list(_CATEGORY_COLOURS.values())
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

    LOG_ENTRIES = 9
    LOG_SCALE = 0.030
    LOG_LEFT = -0.74
    LOG_TOP = 0.495
    LOG_BOTTOM = 0.075
    LOG_TEXT_WIDTH = 0.68

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
            frameSize=(-0.78, -0.02, 0.04, 0.62),
            relief=DGG.FLAT,
        )
        self._log_frame.setTransparency(TransparencyAttrib.MAlpha)

        T.styled_text(
            text='BATTLE LOG', pos=(self.LOG_LEFT, 0.555), scale=0.030,
            fg=T.GOLD, align=TextNode.ALeft, parent=self._log_frame,
            font=font, mayChange=False)
        self._log_rule = DirectFrame(
            parent=self._log_frame, frameColor=T.SEPARATOR,
            frameSize=(self.LOG_LEFT, -0.06, -0.002, 0.002),
            pos=(0, 0, 0.535),
        )

        self._log_text = T.styled_text(
            text='', pos=(self.LOG_LEFT, self.LOG_TOP), scale=self.LOG_SCALE,
            fg=T.CREAM, align=TextNode.ALeft, parent=self._log_frame,
            font=font, wordwrap=(self.LOG_TEXT_WIDTH / self.LOG_SCALE))

        self._widgets = [self._turn, self._round, self._phase, self._log_frame]

        self.set_phase(self._active_phase)
        self._redraw_log()

        self.accept('hud-turn', self.set_turn)
        self.accept('hud-phase', self.set_phase)
        self.accept('hud-log', self.log)
        rules_log.add_listener(self._on_rule)

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
