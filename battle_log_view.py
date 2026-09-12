"""Paged, bounded rendering for the battle journal."""

from datetime import datetime
from pathlib import Path
import shutil
import subprocess

from direct.gui.DirectGui import DirectButton, DirectCheckButton, DirectOptionMenu, DirectSlider, DGG
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import Point3, TextNode

import gui_theme as T


class BattleLogView:
    PAGE_SIZE = 100
    SCALE = .034
    TOP = .39
    BOTTOM = -.57

    def __init__(self, hud):
        self.hud = hud
        self.end = None
        self.scroll = 0.0
        self.max_scroll = 0.0
        self.details = False
        self.subject = 'All units'
        self.frozen = None
        self._syncing_scroll = False
        self.panel = T.styled_panel((-1, 1, -.83, .83), parent=aspect2d, texture=T.TEX_PARCHMENT)
        self.panel.setBin('gui-popup', 30)
        self.controls = []
        self.title = T.styled_text('BATTLE HISTORY', parent=self.panel, pos=(0, .72),
                                   scale=.048, align=TextNode.ACenter, fg=T.INK, shadow=None)
        self.text = OnscreenText(parent=self.panel, text='', font=T.get_font(),
                                 scale=self.SCALE, align=TextNode.ALeft, fg=T.INK,
                                 mayChange=True)
        self.status = T.styled_text('', parent=self.panel, pos=(0, -.76), scale=.025,
                                    align=TextNode.ACenter, fg=T.INK, shadow=None)
        self.context = T.styled_text('', parent=self.panel, pos=(0, .425), scale=.024,
                         align=TextNode.ACenter, fg=T.INK, shadow=None)
        self.scrollbar = DirectSlider(parent=self.panel, range=(0, 1), value=0,
                           orientation=DGG.VERTICAL, pageSize=.1,
                           frameSize=(-.009, .009, self.BOTTOM, self.TOP),
                           thumb_frameSize=(-.016, .016, -.045, .045),
                           command=self.drag_scroll)
        self.mode = DirectOptionMenu(parent=self.panel, items=list(hud._journal.MODES),
                                     initialitem=hud._journal.MODES.index(hud._log_mode),
                                     scale=.036, command=self.set_mode, text_font=T.get_font(),
                                     frameColor=T.CREAM, text_fg=T.INK)
        subjects = sorted({entry.subject for entry in hud._journal.entries if entry.subject})
        self.units = DirectOptionMenu(parent=self.panel, items=['All units', *subjects],
                                      scale=.03, command=self.set_subject, text_font=T.get_font(),
                                      frameColor=T.CREAM, text_fg=T.INK)
        self.detail = DirectCheckButton(parent=self.panel, text='Roll details', scale=.035,
                                        text_font=T.get_font(), command=self.set_details,
                                        frameColor=T.CREAM, text_fg=T.INK)
        self._button('Close', .72, .74, hud.close_history)
        self._button('Older', -.75, -.65, lambda: self.page(-1))
        self._button('Newer', -.40, -.65, lambda: self.page(1))
        self._button('Latest', -.05, -.65, self.latest)
        self._button('Copy', .30, -.65, self.copy)
        self._button('Export', .70, -.65, self.export)
        self.layout()

    def _button(self, label, fraction, vertical, command):
        button = DirectButton(parent=self.panel, text=label, text_font=T.get_font(),
                              text_scale=.030, text_pos=(0, -.01),
                              frameSize=(-.10, .10, -.026, .026),
                              frameColor=T.CREAM,
                              text_fg=T.INK, relief=DGG.FLAT, command=command)
        self.controls.append((button, fraction, vertical))

    def layout(self):
        self.width = min(1.35, base.getAspectRatio() - .05)
        self.panel['frameSize'] = (-self.width, self.width, -.83, .83)
        for button, fraction, vertical in self.controls:
            button.setPos(fraction * self.width, 0, vertical)
        self.mode.setPos(-self.width + .05, 0, .61)
        self.units.setPos(-self.width + .05, 0, .49)
        self.detail.setPos(self.width - .25, 0, .60)
        self.scrollbar.setPos(self.width - .03, 0, 0)
        self.text['wordwrap'] = (2 * self.width - .16) / self.SCALE
        self.context['wordwrap'] = (2 * self.width - .12) / .024
        self.status['wordwrap'] = (2 * self.width - .1) / .025
        self.redraw()

    def set_mode(self, mode):
        self.hud._log_mode = mode
        if hasattr(self, 'mode'):
            self.mode.set(mode, fCommand=0)
        self.hud._log_scroll = 0
        self.hud._redraw_log()
        self.latest()

    def set_subject(self, subject):
        self.subject = subject
        if hasattr(self, 'units') and subject in self.units['items']:
            self.units.set(subject, fCommand=0)
        self.latest()

    def set_details(self, enabled):
        self.details = bool(enabled)
        if hasattr(self, 'detail'):
            self.detail['indicatorValue'] = self.details
            self.detail.setIndicatorValue()
        self.redraw()

    def latest(self):
        self.end = None
        self.scroll = 0
        self.frozen = None
        if hasattr(self, 'width'):
            self.redraw()

    def page(self, direction):
        entries = self.hud._journal.visible(self.hud._log_mode, self.subject)
        end = len(entries) if self.end is None else self.end
        if self.frozen:
            last = self.frozen[-1].sequence
            end = sum(entry.sequence <= last for entry in entries)
        self.end = min(len(entries), max(min(len(entries), self.PAGE_SIZE),
                                         end + direction * self.PAGE_SIZE))
        if self.end == len(entries):
            self.end = None
        self.scroll = 0
        self.frozen = None
        self.redraw()
        if self.end is not None:
            self.frozen = list(self.displayed)
        if direction < 0:
            self.scroll = self.max_scroll
            self.redraw()

    def redraw(self):
        if not hasattr(self, 'width'):
            return
        subjects = ['All units', *sorted({entry.subject for entry in self.hud._journal.entries if entry.subject})]
        if subjects != self.units['items']:
            self.units['items'] = subjects
            self.units.set(self.subject if self.subject in subjects else 'All units', fCommand=0)
        if self.frozen is None:
            entries = self.hud._journal.visible(self.hud._log_mode, self.subject)
            end = len(entries) if self.end is None else min(self.end, len(entries))
            entries = entries[max(0, end - self.PAGE_SIZE):end]
        else:
            entries = self.frozen
        self.displayed = entries
        self.text.setText(self.hud.log_text(entries, details=self.details))
        height = self.text.textNode.getHeight() * self.SCALE if entries else 0
        self.max_scroll = max(0, height - (self.TOP - self.BOTTOM))
        self.scroll = min(self.scroll, self.max_scroll)
        self.text.setPos(-self.width + .05, self.BOTTOM + height - self.scroll)
        self.text.setScissor(self.panel, Point3(-self.width + .04, 0, self.BOTTOM),
                            Point3(self.width - .07, 0, self.TOP))
        first = entries[0].sequence if entries else 0
        last = entries[-1].sequence if entries else 0
        self.status.setText(f'Events {first}-{last} | {len(self.hud._journal.entries)} retained')
        heading = 'No matching events'
        if entries:
            heading = entries[0].heading if all(entry.group == entries[0].group for entry in entries) else (
                'Multiple rounds, phases or combats')
        self.context.setText(heading)
        self._syncing_scroll = True
        self.scrollbar['value'] = self.scroll / self.max_scroll if self.max_scroll else 0
        self._syncing_scroll = False

    def drag_scroll(self):
        if self._syncing_scroll or not hasattr(self, 'scrollbar') or not hasattr(self, 'displayed'):
            return
        self.scroll = self.scrollbar['value'] * self.max_scroll
        self.frozen = list(self.displayed) if self.scroll > 0 or self.end is not None else None
        self.redraw()

    def scroll_lines(self, lines):
        self.scroll = max(0, min(self.max_scroll, self.scroll + lines * self.SCALE * 1.1))
        self.frozen = list(self.displayed) if self.scroll > 0 or self.end is not None else None
        self.redraw()

    def pointer_over(self, mouse):
        point = self.panel.getRelativePoint(base.render2d, Point3(mouse.getX(), 0, mouse.getY()))
        return -self.width <= point.x <= self.width and self.BOTTOM <= point.z <= self.TOP

    def export(self):
        try:
            folder = Path('logs')
            folder.mkdir(exist_ok=True)
            path = folder / f'battle-{datetime.now():%Y%m%d-%H%M%S-%f}'
            path.with_suffix('.txt').write_text(self.hud._journal.export(), encoding='utf-8')
            path.with_suffix('.json').write_text(self.hud._journal.export(structured=True), encoding='utf-8')
            self.status.setText(f'Exported {path}.txt and .json')
        except OSError as error:
            self.status.setText(f'Export failed: {error}')

    def copy(self):
        command = next((command for command in (['wl-copy'], ['xclip', '-selection', 'clipboard'],
                                                 ['xsel', '--clipboard', '--input'])
                        if shutil.which(command[0])), None)
        if command is None:
            self.status.setText('Clipboard unavailable: install wl-clipboard or xclip; Export is available.')
            return
        try:
            subprocess.run(command, input=self.hud._journal.export(), text=True,
                           check=True, timeout=2, capture_output=True)
            self.status.setText('Copied full retained history')
        except (OSError, subprocess.SubprocessError):
            self.status.setText('Clipboard unavailable. Export is available.')

    def destroy(self):
        self.panel.destroy()