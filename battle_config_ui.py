"""Native Panda3D preset editor; no battle state exists until Start Game."""

from copy import deepcopy
from pathlib import Path

from direct.gui.DirectGui import (DGG, DirectButton, DirectCheckButton, DirectEntry,
                                  DirectFrame, DirectOptionMenu, DirectScrolledFrame)
from direct.showbase.DirectObject import DirectObject
from panda3d.core import TextNode

import gui_theme as theme
from battle_config import (CUSTOM_DEPLOYMENT_MAPS, DEFAULT_PRESET, DEPLOYMENT_MAPS, LANDMARK_PROPERTIES,
                           MIRRORABLE_MAPS, OBJECTIVE_LAYOUTS, REED_FENS_MAP, REED_FENS_PRESET, ConfigError, _number,
                           load_config, save_config, validate_activation, validate_config)


FIELDS = {
    'Battle': [
        ('points_limit', 'Points limit', 'integer', None),
        ('battlefield.width', 'Board width (inches)', 'number', None),
        ('battlefield.depth', 'Board depth (inches)', 'number', None),
        ('game.rounds', 'Rounds', 'integer', None),
        ('deployment.map', 'Deployment map', 'menu', ('random', *DEPLOYMENT_MAPS, *CUSTOM_DEPLOYMENT_MAPS)),
        ('deployment.mirror', 'Alternate deployment', 'boolean', None),
        ('battlefield.show_boundary', 'Show playable boundary', 'boolean', None),
        ('battlefield.show_deployment', 'Show deployment zones', 'boolean', None),
        ('seed', 'Setup seed (optional)', 'seed', None),
    ],
    'Terrain': [
        ('terrain.method', 'Placement method', 'menu', ('alternating', 'scattered', 'fixed')),
        ('terrain.feature_count', 'Terrain features', 'integer', None),
        ('terrain.recommended_max_span', 'Recommended span (inches)', 'number', None),
        ('terrain.centre_clearance', 'Centre clearance (inches)', 'number', None),
        ('terrain.opponent_feature_clearance', 'Opponent clearance (inches)', 'number', None),
        ('terrain.objective_clearance', 'Objective clearance (inches)', 'number', None),
    ],
    'Objectives': [
        ('objectives.layout', 'Objective layout', 'menu', ('random', *OBJECTIVE_LAYOUTS, 'none')),
        ('objectives.landmark_property', 'Landmark property', 'menu', ('random', *LANDMARK_PROPERTIES)),
        ('objectives.trove_base_mm', 'Trove base (mm)', 'number', None),
        ('objectives.landmark_base_mm', 'Landmark base (mm)', 'number', None),
        ('objectives.control_distance', 'Control distance (inches)', 'number', None),
        ('objectives.minimum_unit_strength', 'Minimum Unit Strength', 'integer', None),
    ],
    'Scoring': [
        ('scoring.trove_per_player_turn', 'Trove per player turn', 'integer', None),
        ('scoring.landmark_per_player_turn', 'Landmark per player turn', 'integer', None),
        ('scoring.general', 'General', 'integer', None),
        ('scoring.captured_standard', 'Captured standard', 'integer', None),
        ('scoring.battle_standard_bearer', 'Battle Standard Bearer', 'integer', None),
    ],
    'Muster': [
        ('army.minimum_units', 'Minimum qualifying units', 'integer', None),
        ('army.maximum_unit_strength', 'Maximum Unit Strength', 'integer', None),
        ('army.maximum_character_fraction', 'Character selection cap (%)', 'percent', None),
        ('army.maximum_core_fraction', 'Core selection cap (%)', 'percent', None),
        ('army.maximum_special_fraction', 'Special selection cap (%)', 'percent', None),
        ('army.maximum_rare_mercenary_fraction', 'Rare / mercenary cap (%)', 'percent', None),
        ('army.restricted_options_allowance', 'Restricted options allowance', 'integer', None),
    ],
    'Modules': [
        ('optional_rules.secondary_objectives.raid_and_burn', 'Raid & Burn', 'option', None),
        ('optional_rules.secondary_objectives.baggage_carts', 'Baggage Carts', 'option', None),
        ('optional_rules.secret_objectives', 'Secret objectives (unavailable)', 'unavailable', None),
        ('optional_rules.random_happenings.disruptive_weather', 'Disruptive Weather (unavailable)', 'unavailable_option', None),
        ('optional_rules.random_happenings.wilderness_terrain', 'Wilderness Terrain (unavailable)', 'unavailable_option', None),
        ('optional_rules.random_happenings.chaos_of_war', 'Chaos of War (unavailable)', 'unavailable_option', None),
        ('optional_rules.battle_march_magic_items', 'Magic-item module (paused)', 'unavailable', None),
        ('game.time_limit_minutes', 'Time limit (unavailable)', 'time', None),
    ],
}


MAP_FIELDS = ('battlefield.width', 'battlefield.depth', 'objectives.layout',
              *(field[0] for field in FIELDS['Terrain']))


def _value(config, path):
    value = config
    for part in path.split('.'):
        value = part in value if isinstance(value, list) else value[part]
    return value


def _set_value(config, path, value):
    parts = path.split('.')
    target = config
    for part in parts[:-1]:
        target = target[part]
    if isinstance(target, list):
        if value and parts[-1] not in target:
            target.append(parts[-1])
        elif not value and parts[-1] in target:
            target.remove(parts[-1])
    else:
        target[parts[-1]] = value


class BattleConfigScreen(DirectObject):
    def __init__(self, game, config, path, seed, on_start):
        self.game = game
        self.config = validate_config(config)
        self.path = Path(path or DEFAULT_PRESET).expanduser()
        self.seed = seed
        self.on_start = on_start
        self.started = False
        self.closed = False
        self.tab = 'Battle'
        self.values = {}
        self.controls = {}
        self.menu_choices = {}
        self.official_map_values = None
        self.load_values()
        self.root = DirectFrame(parent=game.aspect2d, sortOrder=1000,
                                frameTexture=theme.TEX_PARCHMENT, frameColor=(1, 1, 1, 1))
        self.title = theme.styled_text('Battle March', parent=self.root, scale=.075,
                                       fg=theme.INK, shadow=None)
        self.subtitle = theme.styled_text("General's Companion", parent=self.root, scale=.03,
                                          fg=theme.INK_FADED, shadow=None)
        self.path_entry = DirectEntry(parent=self.root, initialText=str(self.path), scale=.033,
                                      text_font=theme.get_font(), text_fg=theme.ENTRY_FG,
                                      frameColor=theme.ENTRY_BG, numLines=1, overflow=True)
        self.load_button = self.button('Load', self.reload, .25)
        self.save_button = self.button('Save Config', self.save, .42)
        self.start_button = self.button('Start Game', self.start, .45, primary=True)
        self.exit_button = self.button('Exit', self.exit, .22)
        self.status = theme.styled_text('', parent=self.root, scale=.03,
                                        fg=theme.RED_WAX, shadow=None)
        self.tabs = {name: self.button(name, lambda name=name: self.select_tab(name), .36)
                     for name in FIELDS}
        self.scroll = DirectScrolledFrame(parent=self.root, frameColor=(0, 0, 0, 0),
                                          relief=DGG.FLAT, borderWidth=(0, 0),
                                          scrollBarWidth=.035, autoHideScrollBars=True)
        self.scroll.horizontalScroll.hide()
        self.labels = []
        self.accept('aspectRatioChanged', self.layout)
        self.accept('wheel_up', self.scroll_by, [-.13])
        self.accept('wheel_down', self.scroll_by, [.13])
        self.accept('escape', self.exit)
        self.layout()

    def button(self, label, command, width, primary=False):
        return DirectButton(parent=self.root, text=label, text_font=theme.get_font(),
                            text_scale=.033, text_pos=(0, -.012), text_fg=theme.CREAM,
                            frameSize=(-width / 2, width / 2, -.045, .045),
                            frameColor=theme.BTN_GREEN if primary else theme.BTN_NEUTRAL,
                            relief=DGG.FLAT, command=command)

    def load_values(self):
        for fields in FIELDS.values():
            for path, label, kind, options in fields:
                value = self.seed if kind == 'seed' else _value(self.config, path)
                if kind in ('integer', 'number', 'percent', 'seed', 'time'):
                    value = '' if value is None else str(value * 100 if kind == 'percent' else value)
                self.values[path] = value

    def capture(self):
        for path, control in self.controls.items():
            if isinstance(control, DirectEntry):
                self.values[path] = control.get()
            elif isinstance(control, DirectCheckButton):
                self.values[path] = bool(control['indicatorValue'])
            else:
                self.values[path] = control['items'].index(control.get())
                choices = self.menu_choices[path]
                self.values[path] = choices[self.values[path]]

    def draft(self):
        self.capture()
        config = deepcopy(self.config)
        seed = None
        for fields in FIELDS.values():
            for path, label, kind, options in fields:
                value = self.values[path]
                try:
                    if kind in ('seed', 'time') and not value.strip():
                        value = None
                    elif kind in ('integer', 'seed'):
                        value = int(value)
                    elif kind in ('number', 'percent', 'time'):
                        value = float(value) / (100 if kind == 'percent' else 1)
                except ValueError as error:
                    raise ConfigError(f'{label}: enter a valid {"integer" if kind in ("integer", "seed") else "number"}') from error
                if kind == 'seed':
                    seed = value
                    if seed is not None:
                        _number(seed, 'Setup seed', 0, 2 ** 53 - 1, integer=True)
                else:
                    _set_value(config, path, value)
        return validate_config(config), seed

    def select_tab(self, name):
        self.capture()
        self.tab = name
        self.build_fields()

    def layout(self):
        self.capture()
        aspect = self.game.getAspectRatio()
        self.root.setScale(1 / min(aspect, 1))
        self.root['frameSize'] = (-aspect, aspect, -1, 1)
        self.width = min(3.2, 2 * aspect - .12)
        left, right = -self.width / 2, self.width / 2
        self.title.setPos(left, .865)
        self.subtitle.setPos(left, .80)
        self.path_entry.setPos(left, 0, .67)
        self.path_entry['width'] = (self.width - .35) / .033
        self.path_entry['frameSize'] = (0, (self.width - .35) / .033, -.4, 1.2)
        self.load_button.setPos(right - .125, 0, .685)
        tab_width = self.width / len(self.tabs)
        for index, button in enumerate(self.tabs.values()):
            button['frameSize'] = (-tab_width / 2 + .006, tab_width / 2 - .006, -.045, .045)
            button['text_scale'] = .028
            button.setPos(left + tab_width * (index + .5), 0, .52)
        self.scroll['frameSize'] = (left, right, -.61, .43)
        self.status.setPos(left, -.70)
        self.status['wordwrap'] = self.width / .03
        self.exit_button.setPos(left + .11, 0, -.90)
        self.save_button.setPos(right - .71, 0, -.90)
        self.start_button.setPos(right - .225, 0, -.90)
        self.build_fields()

    def build_fields(self):
        for control in self.controls.values():
            control.destroy()
        for label in self.labels:
            label.destroy()
        self.controls.clear()
        self.menu_choices.clear()
        self.labels.clear()
        for name, button in self.tabs.items():
            button['frameColor'] = theme.GREEN_BANNER if name == self.tab else theme.BTN_NEUTRAL
        fields = FIELDS[self.tab]
        left, right = -self.width / 2 + .015, self.width / 2 - .07
        input_width = min(.72, self.width * .43)
        input_left = right - input_width
        self.scroll['canvasSize'] = (-self.width / 2, self.width / 2 - .04,
                                    min(-.61, .39 - len(fields) * .145), .43)
        self.scroll.verticalScroll['value'] = 0
        for index, (path, label, kind, choices) in enumerate(fields):
            height = .34 - index * .145
            self.labels.append(theme.styled_text(label, parent=self.scroll.getCanvas(),
                               pos=(left, height), scale=.032, fg=theme.INK, shadow=None,
                               wordwrap=max(5, (input_left - left - .045) / .032)))
            value = self.values[path]
            if kind in ('boolean', 'option', 'unavailable', 'unavailable_option'):
                control = DirectCheckButton(parent=self.scroll.getCanvas(), text='',
                                             pos=(right - .04, 0, height + .012), scale=.045,
                                             indicatorValue=int(value), boxBorder=.08,
                                             boxPlacement='left')
                if kind.startswith('unavailable'):
                    control['state'] = DGG.NORMAL if value else DGG.DISABLED
                    control['command'] = lambda checked, control=control: control.configure(
                        state=DGG.NORMAL if checked else DGG.DISABLED)
                if path == 'deployment.mirror':
                    control['state'] = (DGG.NORMAL if self.values['deployment.map'] in ('random', *MIRRORABLE_MAPS)
                                        else DGG.DISABLED)
            elif kind == 'menu':
                if self.values['deployment.map'] != REED_FENS_MAP:
                    choices = tuple(choice for choice in choices if choice not in ('fixed', 'none'))
                self.menu_choices[path] = choices
                items = [choice if choice in CUSTOM_DEPLOYMENT_MAPS else choice.replace('_', ' ').title()
                         for choice in choices]
                text_options = ({'text_scale': .8, 'text_wordwrap': (input_width - .09) / (.033 * .8),
                                 'text_pos': (.45, .35), 'item_text_scale': .8,
                                 'item_text_wordwrap': (input_width - .09) / (.033 * .8)}
                                if path == 'deployment.map' else {'text_scale': 1, 'text_pos': (.45, -.06)})
                control = DirectOptionMenu(parent=self.scroll.getCanvas(), items=items,
                                            initialitem=choices.index(value), text_font=theme.get_font(),
                                            scale=.033, text_fg=theme.CREAM,
                                            frameColor=theme.ENTRY_BG, relief=DGG.FLAT,
                                            frameSize=(0, (input_width - .05) / .033, -.91, 1.36),
                                            text_align=TextNode.ALeft,
                                            popupMarker_scale=.6, popupMarkerBorder=(.2, .1),
                                            item_text_font=theme.get_font(), item_text_fg=theme.CREAM,
                                            item_frameColor=theme.ENTRY_BG, item_pad=(.3, .25),
                                            item_relief=DGG.FLAT, highlightColor=theme.BTN_GREEN,
                                            pos=(input_left, 0, height), **text_options)
                control.popupMenu.reparentTo(self.root)
                control.cancelFrame.reparentTo(self.root)
                if path == 'deployment.map':
                    control['command'] = lambda selected: self.map_changed()
            else:
                control = DirectEntry(parent=self.scroll.getCanvas(), initialText=value,
                                       pos=(input_left, 0, height), scale=.033,
                                       width=input_width / .033, numLines=1,
                                       text_font=theme.get_font(), text_fg=theme.ENTRY_FG,
                                       frameColor=theme.ENTRY_BG)
            if self.values['deployment.map'] == REED_FENS_MAP and (path in MAP_FIELDS or path.startswith('objectives.')):
                control['state'] = DGG.DISABLED
            self.controls[path] = control

    def map_changed(self):
        previous = self.values['deployment.map']
        self.capture()
        selected = self.values['deployment.map']
        if selected == REED_FENS_MAP and previous != selected:
            self.official_map_values = {path: self.values[path] for path in MAP_FIELDS}
            self.official_source = deepcopy(self.config['source'])
            preset = load_config(REED_FENS_PRESET)
            for path in MAP_FIELDS:
                value = _value(preset, path)
                self.values[path] = str(value) if isinstance(value, (int, float)) else value
            self.config['source'] = preset['source']
        elif previous == REED_FENS_MAP and selected != previous:
            if self.official_map_values is not None:
                self.values.update(self.official_map_values)
                self.config['source'] = self.official_source
            else:
                preset = load_config()
                for path in MAP_FIELDS:
                    value = _value(preset, path)
                    self.values[path] = str(value) if isinstance(value, (int, float)) else value
                self.config['source'] = preset['source']
        if self.values['deployment.map'] not in ('random', *MIRRORABLE_MAPS):
            self.values['deployment.mirror'] = False
        self.build_fields()

    def scroll_by(self, amount):
        scrollbar = self.scroll.verticalScroll
        scrollbar['value'] = min(1, max(0, scrollbar['value'] + amount))

    def message(self, text, error=False):
        self.status['fg'] = theme.RED_WAX if error else theme.INK
        self.status.setText(text)

    def save(self):
        if self.started or self.closed:
            return False
        try:
            config, seed = self.draft()
            path = self.path_entry.get().strip()
            if not path:
                raise ConfigError('Config file: enter a JSON path')
            save_config(path, config)
        except (ConfigError, OSError) as error:
            self.message(str(error), error=True)
            return False
        self.config, self.seed, self.path = config, seed, Path(path).expanduser()
        self.message(f'Saved {self.path.name}')
        return True

    def reload(self):
        if self.started or self.closed:
            return
        try:
            path = Path(self.path_entry.get().strip()).expanduser()
            config = load_config(path)
        except (ConfigError, OSError) as error:
            self.message(str(error), error=True)
            return
        self.config, self.path = config, path
        self.official_map_values = None
        self.load_values()
        self.build_fields()
        self.message(f'Loaded {path.name}')

    def start(self):
        if self.started or self.closed:
            return
        try:
            config, seed = self.draft()
            validate_activation(config)
        except ConfigError as error:
            self.message(str(error), error=True)
            return
        if self.save():
            self.started = True
            self.on_start(config, seed)

    def exit(self):
        self.destroy()
        self.game.userExit()

    def destroy(self):
        if self.closed:
            return
        self.closed = True
        self.ignoreAll()
        for label in self.labels:
            label.destroy()
        for label in (self.title, self.subtitle, self.status):
            label.destroy()
        self.root.destroy()