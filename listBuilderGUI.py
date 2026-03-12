"""
Panda3D GUI-based Interactive Army List Builder
Three-panel layout: Available Units | Army Roster | Unit Details
"""

from direct.showbase.ShowBase import ShowBase
from direct.gui.DirectGui import *
from panda3d.core import TextNode, TransparencyAttrib
import os
import json
from models import model
from units import unit


class ArmyListBuilderGUI:
    def __init__(self, base_app):
        self.base = base_app
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.available_units = {}
        self.army_list = []
        self.points_budget = 2000
        self.gui_elements = []
        self.selected_army_idx = None

        # Sub-element lists for selective panel refresh
        self._middle_panel_elements = []
        self._right_panel_elements = []
        self._popup_elements = []

        self.load_available_units()
        self.current_screen = None

        # Show main menu by default
        self.show_main_menu()

    def load_available_units(self):
        """Load all available units from army_units/<faction>/ JSON files"""
        army_units_dir = os.path.join(self.base_dir, 'army_units')
        self.factions = {}

        for root, _dirs, files in os.walk(army_units_dir):
            for json_file in sorted(files):
                if not json_file.endswith('_characteristics.json'):
                    continue
                full_path = os.path.join(root, json_file)
                try:
                    with open(full_path, 'r') as f:
                        data = json.load(f)
                        unit_name = data.get('Model', 'Unknown')
                        faction = os.path.basename(root).replace('_', ' ').title()
                        self.available_units[unit_name] = {
                            'file': full_path,
                            'faction': faction,
                            'characteristics': data
                        }
                        self.factions.setdefault(faction, []).append(unit_name)
                except Exception as e:
                    print(f"Error loading {full_path}: {e}")

        for faction in self.factions:
            self.factions[faction].sort()

        print(f"Loaded {len(self.available_units)} unit types across {len(self.factions)} factions")

    def _unit_pts(self, unit_name: str) -> int:
        """Return the points cost per model for a unit (0 if not defined)."""
        info = self.available_units.get(unit_name, {})
        return int(info.get('characteristics', {}).get('Points', 0) or 0)

    def _army_total_pts(self) -> int:
        """Return the total points cost of the current army list."""
        return sum(u.get('points_cost', 0) for u in self.army_list)

    def _army_remaining_pts(self) -> int:
        """Return remaining points in the budget."""
        return self.points_budget - self._army_total_pts()

    def clear_screen(self):
        """Remove all GUI elements"""
        for element in self.gui_elements:
            element.destroy()
        self.gui_elements = []
        self._middle_panel_elements = []
        self._right_panel_elements = []
        self._clear_popup()

    def _clear_popup(self):
        """Destroy popup overlay elements"""
        for element in self._popup_elements:
            element.destroy()
        self._popup_elements = []

    def _clear_right_panel(self):
        """Destroy right-panel elements only"""
        for element in self._right_panel_elements:
            element.destroy()
        self._right_panel_elements = []

    # ─── Main Menu ────────────────────────────────────────────────────

    def show_main_menu(self):
        """Display the main menu"""
        self.clear_screen()
        self.current_screen = "main_menu"
        self.selected_army_idx = None

        bg_frame = DirectFrame(
            frameSize=(-1.2, 1.2, -0.95, 0.95),
            frameColor=(0.05, 0.05, 0.1, 0.9),
            pos=(0, 0, 0), relief=DGG.SUNKEN, borderWidth=(0.01, 0.01))
        self.gui_elements.append(bg_frame)

        title_shadow = OnscreenText(
            text="ARMY LIST BUILDER", pos=(0.01, 0.84), scale=0.13,
            fg=(0, 0, 0, 0.5), align=TextNode.ACenter, mayChange=False)
        self.gui_elements.append(title_shadow)

        title = OnscreenText(
            text="ARMY LIST BUILDER", pos=(0, 0.85), scale=0.13,
            fg=(1, 0.85, 0.2, 1), align=TextNode.ACenter, mayChange=False)
        self.gui_elements.append(title)

        line_frame = DirectFrame(
            frameSize=(-0.8, 0.8, -0.005, 0.005),
            frameColor=(0.8, 0.6, 0.1, 1), pos=(0, 0, 0.65))
        self.gui_elements.append(line_frame)

        subtitle = OnscreenText(
            text="Build Your Warhammer Army", pos=(0, 0.55), scale=0.06,
            fg=(0.7, 0.7, 0.8, 1), align=TextNode.ACenter, mayChange=False)
        self.gui_elements.append(subtitle)

        spent = self._army_total_pts()
        remaining = self.points_budget - spent
        budget_color = (0.2, 1, 0.2, 1) if remaining >= 0 else (1, 0.3, 0.3, 1)
        budget_info = OnscreenText(
            text=f"Budget: {spent} / {self.points_budget} pts  ({remaining} remaining)",
            pos=(0, 0.44), scale=0.055, fg=budget_color,
            align=TextNode.ACenter, mayChange=False)
        self.gui_elements.append(budget_info)

        button_data = [
            ("Build Army",         self.show_army_builder,  0.22, (0.2, 0.4, 0.3, 1)),
            ("Set Points Budget",  self.show_budget_screen, 0.04, (0.45, 0.35, 0.05, 1)),
            ("Save Army List",     self.show_save_screen,  -0.14, (0.3, 0.4, 0.5, 1)),
            ("Load Army List",     self.show_load_screen,  -0.32, (0.5, 0.4, 0.2, 1)),
            ("Exit List Builder",  self.exit_builder,      -0.50, (0.5, 0.2, 0.2, 1)),
        ]

        for text, command, y_pos, color in button_data:
            btn = DirectButton(
                text=text, text_font=None, text_scale=0.9, scale=0.075,
                pos=(0, 0, y_pos), command=command,
                frameSize=(-4.2, 4.2, -0.6, 1.1), text_fg=(1, 1, 1, 1),
                frameColor=color, relief=DGG.RAISED, borderWidth=(0.015, 0.015))
            self.gui_elements.append(btn)

    # ─── Three-Panel Army Builder ─────────────────────────────────────

    def show_army_builder(self):
        """Main three-panel army builder screen."""
        self.clear_screen()
        self.current_screen = "army_builder"
        self.selected_army_idx = None

        # Full background
        bg = DirectFrame(
            frameSize=(-1.8, 1.8, -1.0, 1.0),
            frameColor=(0.06, 0.06, 0.10, 1.0), pos=(0, 0, 0))
        self.gui_elements.append(bg)

        # ── Top bar ──
        top_bar = DirectFrame(
            frameSize=(-1.8, 1.8, -0.005, 0.12),
            frameColor=(0.12, 0.10, 0.18, 1), pos=(0, 0, 0.88))
        self.gui_elements.append(top_bar)

        title = OnscreenText(
            text="ARMY BUILDER", pos=(-1.55, 0.90), scale=0.065,
            fg=(1, 0.85, 0.2, 1), align=TextNode.ALeft, mayChange=False)
        self.gui_elements.append(title)

        self._pts_label = OnscreenText(
            text="", pos=(0, 0.90), scale=0.055,
            fg=(0.3, 1, 0.4, 1), align=TextNode.ACenter, mayChange=True)
        self.gui_elements.append(self._pts_label)
        self._refresh_pts_label()

        back_btn = DirectButton(
            text="< Menu", text_scale=0.9, scale=0.055,
            pos=(1.45, 0, 0.90), command=self.show_main_menu,
            frameSize=(-2.0, 2.0, -0.6, 1.1), frameColor=(0.35, 0.25, 0.25, 1),
            text_fg=(1, 1, 1, 1), relief=DGG.RAISED, borderWidth=(0.01, 0.01))
        self.gui_elements.append(back_btn)

        # ── Column headers ──
        for text, x in [("AVAILABLE UNITS", -1.10), ("YOUR ARMY", 0.0), ("UNIT DETAILS", 1.10)]:
            lbl = OnscreenText(
                text=text, pos=(x, 0.82), scale=0.048,
                fg=(0.85, 0.75, 0.5, 1), align=TextNode.ACenter, mayChange=False)
            self.gui_elements.append(lbl)

        for x_center in [-1.10, 0.0, 1.10]:
            sep = DirectFrame(
                frameSize=(-0.45, 0.45, -0.002, 0.002),
                frameColor=(0.5, 0.4, 0.2, 0.6), pos=(x_center, 0, 0.79))
            self.gui_elements.append(sep)

        # Build the three panels
        self._build_left_panel()
        self._build_middle_panel()
        self._build_right_panel_placeholder()

    def _refresh_pts_label(self):
        """Update the points budget label on the builder top bar."""
        spent = self._army_total_pts()
        remaining = self.points_budget - spent
        color = (0.3, 1, 0.4, 1) if remaining >= 0 else (1, 0.4, 0.4, 1)
        self._pts_label.setText(
            f"{spent} / {self.points_budget} pts  ({remaining} remaining)")
        self._pts_label['fg'] = color

    # ── LEFT PANEL: available units ───────────────────────────────────

    def _build_left_panel(self):
        """Scrollable list of all available units grouped by faction."""
        panel_bg = DirectFrame(
            frameSize=(-0.55, 0.55, -0.88, 0.78),
            frameColor=(0.08, 0.08, 0.13, 0.9),
            pos=(-1.10, 0, 0), relief=DGG.SUNKEN, borderWidth=(0.008, 0.008))
        self.gui_elements.append(panel_bg)

        # Calculate canvas height
        total_rows = 0
        for faction in sorted(self.factions.keys()):
            total_rows += 1
            total_rows += len(self.factions[faction])
        row_h = 0.08
        canvas_h = total_rows * row_h + 0.15

        scroll = DirectScrolledFrame(
            canvasSize=(-0.50, 0.46, -canvas_h, 0),
            frameSize=(-0.53, 0.53, -0.86, 0.76),
            pos=(-1.10, 0, 0),
            scrollBarWidth=0.03, frameColor=(0, 0, 0, 0),
            verticalScroll_scrollSize=0.06,
            verticalScroll_thumb_frameColor=(0.4, 0.35, 0.2, 0.8),
            verticalScroll_incButton_frameColor=(0.3, 0.25, 0.15, 0.8),
            verticalScroll_decButton_frameColor=(0.3, 0.25, 0.15, 0.8))
        self.gui_elements.append(scroll)

        canvas = scroll.getCanvas()
        y = -0.04

        for faction in sorted(self.factions.keys()):
            # Faction header
            DirectFrame(
                frameSize=(-0.49, 0.45, -0.035, 0.035),
                frameColor=(0.22, 0.18, 0.08, 0.9),
                pos=(0, 0, y), parent=canvas)
            DirectLabel(
                text=f"  {faction}", text_scale=0.042, text_align=TextNode.ALeft,
                pos=(-0.48, 0, y - 0.012), frameColor=(0, 0, 0, 0),
                text_fg=(1, 0.85, 0.3, 1), parent=canvas)
            y -= row_h

            for unit_name in self.factions[faction]:
                pts = self._unit_pts(unit_name)
                pts_str = f" [{pts}pts]" if pts else ""

                DirectFrame(
                    frameSize=(-0.49, 0.45, -0.035, 0.035),
                    frameColor=(0.12, 0.12, 0.17, 0.5),
                    pos=(0, 0, y), parent=canvas)

                btn = DirectButton(
                    text=f"{unit_name}{pts_str}",
                    text_scale=0.034, text_align=TextNode.ALeft,
                    text_pos=(-0.44, -0.01),
                    pos=(0, 0, y),
                    command=self._show_add_popup, extraArgs=[unit_name],
                    frameSize=(-0.49, 0.33, -0.035, 0.035),
                    frameColor=(0, 0, 0, 0), text_fg=(0.8, 0.85, 0.9, 1),
                    relief=DGG.FLAT, parent=canvas)
                btn.bind(DGG.ENTER, lambda evt, b=btn: b.setColorScale(1.3, 1.3, 1.0, 1))
                btn.bind(DGG.EXIT,  lambda evt, b=btn: b.setColorScale(1, 1, 1, 1))

                add_btn = DirectButton(
                    text="+", text_scale=0.035, text_pos=(0, -0.01),
                    pos=(0.39, 0, y),
                    command=self._show_add_popup, extraArgs=[unit_name],
                    frameSize=(-0.04, 0.04, -0.03, 0.03),
                    frameColor=(0.2, 0.55, 0.2, 1), text_fg=(1, 1, 1, 1),
                    relief=DGG.RAISED, borderWidth=(0.003, 0.003), parent=canvas)

                y -= row_h

    # ── MIDDLE PANEL: army roster ─────────────────────────────────────

    def _build_middle_panel(self):
        """Scrollable list of units currently in the army."""
        for e in self._middle_panel_elements:
            e.destroy()
        self._middle_panel_elements = []

        panel_bg = DirectFrame(
            frameSize=(-0.45, 0.45, -0.88, 0.78),
            frameColor=(0.08, 0.07, 0.12, 0.9),
            pos=(0, 0, 0), relief=DGG.SUNKEN, borderWidth=(0.008, 0.008))
        self.gui_elements.append(panel_bg)
        self._middle_panel_elements.append(panel_bg)

        if not self.army_list:
            empty_lbl = OnscreenText(
                text="No units yet.\nAdd units from\nthe left panel.",
                pos=(0, 0.2), scale=0.05, fg=(0.5, 0.5, 0.55, 1),
                align=TextNode.ACenter, mayChange=False)
            self.gui_elements.append(empty_lbl)
            self._middle_panel_elements.append(empty_lbl)
            return

        row_h = 0.10
        canvas_h = len(self.army_list) * row_h + 0.1

        scroll = DirectScrolledFrame(
            canvasSize=(-0.42, 0.38, -canvas_h, 0),
            frameSize=(-0.43, 0.43, -0.86, 0.76),
            pos=(0, 0, 0),
            scrollBarWidth=0.03, frameColor=(0, 0, 0, 0),
            verticalScroll_scrollSize=0.06,
            verticalScroll_thumb_frameColor=(0.35, 0.25, 0.45, 0.8),
            verticalScroll_incButton_frameColor=(0.25, 0.2, 0.35, 0.8),
            verticalScroll_decButton_frameColor=(0.25, 0.2, 0.35, 0.8))
        self.gui_elements.append(scroll)
        self._middle_panel_elements.append(scroll)

        canvas = scroll.getCanvas()
        y = -0.05

        for idx, army_unit in enumerate(self.army_list):
            is_selected = (idx == self.selected_army_idx)
            bg_color = (0.28, 0.22, 0.42, 0.85) if is_selected else (
                (0.14, 0.12, 0.20, 0.6) if idx % 2 == 0 else (0.11, 0.10, 0.17, 0.6))

            DirectFrame(
                frameSize=(-0.41, 0.37, -0.045, 0.045),
                frameColor=bg_color, pos=(0, 0, y), parent=canvas)

            pts_cost = army_unit.get('points_cost', 0)
            row_btn = DirectButton(
                text=f"{idx+1}. {army_unit['name']}  ({pts_cost}pts)",
                text_scale=0.036, text_align=TextNode.ALeft,
                text_pos=(-0.38, -0.005),
                pos=(0, 0, y),
                command=self._select_army_unit, extraArgs=[idx],
                frameSize=(-0.41, 0.27, -0.045, 0.045),
                frameColor=(0, 0, 0, 0),
                text_fg=(0.95, 0.85, 1, 1) if is_selected else (0.8, 0.75, 0.9, 1),
                relief=DGG.FLAT, parent=canvas)
            row_btn.bind(DGG.ENTER, lambda evt, b=row_btn: b.setColorScale(1.2, 1.2, 1.0, 1))
            row_btn.bind(DGG.EXIT,  lambda evt, b=row_btn: b.setColorScale(1, 1, 1, 1))

            sub = f"{army_unit['nmodels']}mdl  {army_unit['files']}x{army_unit['ranks']}"
            DirectLabel(
                text=sub, text_scale=0.028, text_align=TextNode.ALeft,
                pos=(-0.38, 0, y - 0.028), frameColor=(0, 0, 0, 0),
                text_fg=(0.55, 0.6, 0.7, 1), parent=canvas)

            rem_btn = DirectButton(
                text="X", text_scale=0.03, text_pos=(0, -0.008),
                pos=(0.33, 0, y),
                command=self._remove_army_unit, extraArgs=[idx],
                frameSize=(-0.03, 0.03, -0.025, 0.025),
                frameColor=(0.6, 0.15, 0.15, 1), text_fg=(1, 1, 1, 1),
                relief=DGG.RAISED, borderWidth=(0.003, 0.003), parent=canvas)

            y -= row_h

        # Summary bar
        total_models = sum(u['nmodels'] for u in self.army_list)
        summary_lbl = OnscreenText(
            text=f"{len(self.army_list)} units | {total_models} models",
            pos=(0, -0.91), scale=0.04, fg=(0.6, 0.65, 0.75, 1),
            align=TextNode.ACenter, mayChange=False)
        self.gui_elements.append(summary_lbl)
        self._middle_panel_elements.append(summary_lbl)

    def _select_army_unit(self, idx):
        """Select a unit in the army roster and show its details."""
        self.selected_army_idx = idx
        self._build_middle_panel()
        self._show_unit_detail(idx)

    def _remove_army_unit(self, idx):
        """Remove a unit from the army and refresh the panels."""
        if 0 <= idx < len(self.army_list):
            self.army_list.pop(idx)
            if self.selected_army_idx is not None:
                if self.selected_army_idx == idx:
                    self.selected_army_idx = None
                elif self.selected_army_idx > idx:
                    self.selected_army_idx -= 1
            self._refresh_pts_label()
            self._build_middle_panel()
            self._clear_right_panel()
            if self.selected_army_idx is not None:
                self._show_unit_detail(self.selected_army_idx)
            else:
                self._build_right_panel_placeholder()

    # ── RIGHT PANEL: unit details ─────────────────────────────────────

    def _build_right_panel_placeholder(self):
        """Placeholder shown when no unit is selected."""
        self._clear_right_panel()

        panel_bg = DirectFrame(
            frameSize=(-0.45, 0.55, -0.88, 0.78),
            frameColor=(0.08, 0.08, 0.12, 0.9),
            pos=(1.10, 0, 0), relief=DGG.SUNKEN, borderWidth=(0.008, 0.008))
        self.gui_elements.append(panel_bg)
        self._right_panel_elements.append(panel_bg)

        placeholder = OnscreenText(
            text="Select a unit in\nyour army to view\nits details here.",
            pos=(1.10, 0.1), scale=0.05, fg=(0.45, 0.45, 0.5, 1),
            align=TextNode.ACenter, mayChange=False)
        self.gui_elements.append(placeholder)
        self._right_panel_elements.append(placeholder)

    def _show_unit_detail(self, idx):
        """Show full details for the selected army unit in the right panel."""
        self._clear_right_panel()

        if idx < 0 or idx >= len(self.army_list):
            self._build_right_panel_placeholder()
            return

        army_unit = self.army_list[idx]
        unit_name = army_unit['name']
        info = self.available_units.get(unit_name, {})
        stats = info.get('characteristics', {})

        # Panel background
        panel_bg = DirectFrame(
            frameSize=(-0.45, 0.55, -0.88, 0.78),
            frameColor=(0.08, 0.08, 0.12, 0.9),
            pos=(1.10, 0, 0), relief=DGG.SUNKEN, borderWidth=(0.008, 0.008))
        self.gui_elements.append(panel_bg)
        self._right_panel_elements.append(panel_bg)

        def _add(elem):
            self.gui_elements.append(elem)
            self._right_panel_elements.append(elem)

        # Unit name
        _add(OnscreenText(
            text=unit_name, pos=(1.10, 0.70), scale=0.058,
            fg=(1, 0.88, 0.3, 1), align=TextNode.ACenter, mayChange=False))

        # Faction
        _add(OnscreenText(
            text=army_unit.get('faction', ''), pos=(1.10, 0.63), scale=0.04,
            fg=(0.6, 0.65, 0.7, 1), align=TextNode.ACenter, mayChange=False))

        _add(DirectFrame(
            frameSize=(-0.38, 0.38, -0.002, 0.002),
            frameColor=(0.5, 0.4, 0.2, 0.5), pos=(1.10, 0, 0.59)))

        # Type
        unit_type = stats.get('Type', 'Unknown')
        _add(OnscreenText(
            text=f"Type: {unit_type}", pos=(0.72, 0.53), scale=0.04,
            fg=(0.7, 0.8, 0.9, 1), align=TextNode.ALeft, mayChange=False))

        # Stats table
        _add(OnscreenText(
            text="M   WS  BS   S   T   W   I   A   Ld",
            pos=(0.72, 0.44), scale=0.036,
            fg=(0.85, 0.75, 0.5, 1), align=TextNode.ALeft, mayChange=False))

        stat_keys = ['M', 'WS', 'BS', 'S', 'T', 'W', 'I', 'A', 'Ld']
        vals = "  ".join(str(stats.get(k, '?')).rjust(2) for k in stat_keys)
        _add(OnscreenText(
            text=vals, pos=(0.72, 0.39), scale=0.038,
            fg=(0.9, 0.95, 1, 1), align=TextNode.ALeft, mayChange=False))

        # ── Configuration display ──
        _add(DirectFrame(
            frameSize=(-0.38, 0.38, -0.002, 0.002),
            frameColor=(0.5, 0.4, 0.2, 0.3), pos=(1.10, 0, 0.33)))

        _add(OnscreenText(
            text="CONFIGURATION", pos=(1.10, 0.27), scale=0.042,
            fg=(0.85, 0.75, 0.5, 1), align=TextNode.ACenter, mayChange=False))

        for label_text, value, yp in [
            ("Models:", str(army_unit['nmodels']), 0.19),
            ("Files (Width):", str(army_unit['files']), 0.12),
            ("Ranks (Depth):", str(army_unit['ranks']), 0.05),
        ]:
            _add(OnscreenText(
                text=label_text, pos=(0.72, yp), scale=0.04,
                fg=(0.75, 0.8, 0.85, 1), align=TextNode.ALeft, mayChange=False))
            _add(OnscreenText(
                text=value, pos=(1.42, yp), scale=0.042,
                fg=(1, 1, 1, 1), align=TextNode.ARight, mayChange=False))

        # Points
        _add(DirectFrame(
            frameSize=(-0.38, 0.38, -0.002, 0.002),
            frameColor=(0.5, 0.4, 0.2, 0.3), pos=(1.10, 0, -0.03)))

        pts_per = self._unit_pts(unit_name)
        total_cost = army_unit.get('points_cost', 0)

        _add(OnscreenText(
            text=f"Points/model: {pts_per}", pos=(0.72, -0.09), scale=0.04,
            fg=(0.7, 0.8, 0.7, 1), align=TextNode.ALeft, mayChange=False))
        _add(OnscreenText(
            text=f"Total cost: {total_cost} pts", pos=(0.72, -0.16), scale=0.045,
            fg=(0.4, 1, 0.5, 1), align=TextNode.ALeft, mayChange=False))

        # ── Edit controls ──
        _add(DirectFrame(
            frameSize=(-0.38, 0.38, -0.002, 0.002),
            frameColor=(0.5, 0.4, 0.2, 0.3), pos=(1.10, 0, -0.25)))

        _add(OnscreenText(
            text="EDIT UNIT", pos=(1.10, -0.31), scale=0.042,
            fg=(0.85, 0.75, 0.5, 1), align=TextNode.ACenter, mayChange=False))

        self._edit_entries = {}
        for label_text, default_val, yp in [
            ("Models:", army_unit['nmodels'], -0.40),
            ("Files:",  army_unit['files'],   -0.50),
            ("Ranks:",  army_unit['ranks'],   -0.60),
        ]:
            _add(OnscreenText(
                text=label_text, pos=(0.72, yp), scale=0.04,
                fg=(0.75, 0.8, 0.85, 1), align=TextNode.ALeft, mayChange=False))
            entry = DirectEntry(
                text="", scale=0.045, pos=(1.15, 0, yp),
                initialText=str(default_val), numLines=1, width=4,
                frameColor=(0.18, 0.22, 0.18, 1), text_fg=(1, 1, 1, 1))
            _add(entry)
            self._edit_entries[label_text] = entry

        apply_btn = DirectButton(
            text="Apply Changes", text_scale=0.9, scale=0.05,
            pos=(1.10, 0, -0.73),
            command=self._apply_unit_edits, extraArgs=[idx],
            frameSize=(-3.5, 3.5, -0.7, 1.2),
            frameColor=(0.2, 0.55, 0.25, 1), text_fg=(1, 1, 1, 1),
            relief=DGG.RAISED, borderWidth=(0.01, 0.01))
        _add(apply_btn)

        rem_btn = DirectButton(
            text="Remove Unit", text_scale=0.9, scale=0.05,
            pos=(1.10, 0, -0.82),
            command=self._remove_army_unit, extraArgs=[idx],
            frameSize=(-3.5, 3.5, -0.7, 1.2),
            frameColor=(0.55, 0.15, 0.15, 1), text_fg=(1, 1, 1, 1),
            relief=DGG.RAISED, borderWidth=(0.01, 0.01))
        _add(rem_btn)

    def _apply_unit_edits(self, idx):
        """Apply edited models/files/ranks to the army unit."""
        if idx < 0 or idx >= len(self.army_list):
            return
        try:
            nmodels = int(self._edit_entries["Models:"].get())
            files = int(self._edit_entries["Files:"].get())
            ranks = int(self._edit_entries["Ranks:"].get())
            if nmodels < 1 or files < 1 or ranks < 1:
                return

            unit_name = self.army_list[idx]['name']
            pts_per = self._unit_pts(unit_name)
            new_cost = nmodels * pts_per

            old_cost = self.army_list[idx].get('points_cost', 0)
            if pts_per > 0 and (self._army_total_pts() - old_cost + new_cost) > self.points_budget:
                self._show_builder_message("Not enough points!")
                return

            self.army_list[idx]['nmodels'] = nmodels
            self.army_list[idx]['files'] = files
            self.army_list[idx]['ranks'] = ranks
            self.army_list[idx]['points_cost'] = new_cost

            self._refresh_pts_label()
            self._build_middle_panel()
            self._show_unit_detail(idx)
        except ValueError:
            pass

    # ── Add-unit popup ────────────────────────────────────────────────

    def _show_add_popup(self, unit_name):
        """Centred popup to configure models/files/ranks before adding a unit."""
        self._clear_popup()

        overlay = DirectFrame(
            frameSize=(-2, 2, -1.5, 1.5),
            frameColor=(0, 0, 0, 0.55), pos=(0, 0, 0), sortOrder=50)
        overlay['state'] = DGG.NORMAL
        self._popup_elements.append(overlay)
        self.gui_elements.append(overlay)

        popup = DirectFrame(
            frameSize=(-0.55, 0.55, -0.50, 0.50),
            frameColor=(0.10, 0.12, 0.18, 0.97),
            pos=(0, 0, 0.05), relief=DGG.RAISED,
            borderWidth=(0.012, 0.012), sortOrder=51)
        self._popup_elements.append(popup)
        self.gui_elements.append(popup)

        OnscreenText(
            text=f"Add: {unit_name}", pos=(0, 0.42), scale=0.06,
            fg=(1, 0.88, 0.3, 1), align=TextNode.ACenter, mayChange=False,
            parent=popup)

        pts = self._unit_pts(unit_name)
        OnscreenText(
            text=f"{pts} pts/model" if pts else "Free",
            pos=(0, 0.34), scale=0.042,
            fg=(0.6, 0.8, 0.6, 1), align=TextNode.ACenter, mayChange=False,
            parent=popup)

        entries = {}
        for label_text, default, yp in [
            ("Number of Models:", "10", 0.20),
            ("Files (Width):",    "5",  0.06),
            ("Ranks (Depth):",    "2", -0.08),
        ]:
            OnscreenText(
                text=label_text, pos=(-0.45, yp), scale=0.045,
                fg=(0.85, 0.9, 0.85, 1), align=TextNode.ALeft,
                mayChange=False, parent=popup)
            ent = DirectEntry(
                text="", scale=0.055, pos=(0.18, 0, yp),
                initialText=default, numLines=1, width=5,
                frameColor=(0.18, 0.25, 0.18, 1), text_fg=(1, 1, 1, 1),
                parent=popup)
            entries[label_text] = ent

        DirectButton(
            text="Add to Army", text_scale=0.9, scale=0.06,
            pos=(0, 0, -0.28), parent=popup,
            command=self._confirm_add_unit, extraArgs=[unit_name, entries],
            frameSize=(-3.2, 3.2, -0.7, 1.2),
            frameColor=(0.2, 0.6, 0.25, 1), text_fg=(1, 1, 1, 1),
            relief=DGG.RAISED, borderWidth=(0.012, 0.012))

        DirectButton(
            text="Cancel", text_scale=0.9, scale=0.05,
            pos=(0, 0, -0.42), parent=popup,
            command=self._clear_popup,
            frameSize=(-2.5, 2.5, -0.7, 1.2),
            frameColor=(0.4, 0.25, 0.25, 1), text_fg=(1, 1, 1, 1),
            relief=DGG.RAISED, borderWidth=(0.01, 0.01))

    def _confirm_add_unit(self, unit_name, entries):
        """Validate inputs and add the unit to the army."""
        try:
            nmodels = int(entries["Number of Models:"].get())
            files = int(entries["Files (Width):"].get())
            ranks = int(entries["Ranks (Depth):"].get())
            if nmodels < 1 or files < 1 or ranks < 1:
                return

            pts_per = self._unit_pts(unit_name)
            cost = nmodels * pts_per
            remaining = self._army_remaining_pts()
            if pts_per > 0 and cost > remaining:
                self._show_builder_message(
                    f"Not enough points!\n{nmodels}x {unit_name} = {cost} pts\n"
                    f"Only {remaining} pts left.")
                return

            army_unit = {
                'name': unit_name,
                'faction': self.available_units[unit_name].get('faction', ''),
                'nmodels': nmodels,
                'files': files,
                'ranks': ranks,
                'points_cost': cost,
                'json_file': self.available_units[unit_name]['file']
            }
            self.army_list.append(army_unit)

            self._clear_popup()
            self._refresh_pts_label()
            self._build_middle_panel()
        except ValueError:
            pass

    def _show_builder_message(self, text):
        """Show a brief message overlay on top of the builder screen."""
        self._clear_popup()

        overlay = DirectFrame(
            frameSize=(-2, 2, -1.5, 1.5),
            frameColor=(0, 0, 0, 0.5), pos=(0, 0, 0), sortOrder=50)
        overlay['state'] = DGG.NORMAL
        self._popup_elements.append(overlay)
        self.gui_elements.append(overlay)

        msg_frame = DirectFrame(
            frameSize=(-0.6, 0.6, -0.25, 0.25),
            frameColor=(0.12, 0.10, 0.18, 0.97),
            pos=(0, 0, 0), relief=DGG.RAISED,
            borderWidth=(0.012, 0.012), sortOrder=51)
        self._popup_elements.append(msg_frame)
        self.gui_elements.append(msg_frame)

        OnscreenText(
            text=text, pos=(0, 0.05), scale=0.05,
            fg=(1, 0.8, 0.3, 1), align=TextNode.ACenter, mayChange=False,
            wordwrap=20, parent=msg_frame)

        DirectButton(
            text="OK", text_scale=0.9, scale=0.055,
            pos=(0, 0, -0.16), parent=msg_frame,
            command=self._clear_popup,
            frameSize=(-2, 2, -0.7, 1.2),
            frameColor=(0.3, 0.3, 0.4, 1), text_fg=(1, 1, 1, 1),
            relief=DGG.RAISED, borderWidth=(0.01, 0.01))
    
    def show_save_screen(self):
        """Show screen for saving army list"""
        self.clear_screen()
        self.current_screen = "save"
        
        if not self.army_list:
            self.show_message("Cannot save an empty army list!", self.show_main_menu)
            return
        
        # Title
        title = OnscreenText(
            text="SAVE ARMY LIST",
            pos=(0, 0.7),
            scale=0.1,
            fg=(1, 1, 0, 1),
            align=TextNode.ACenter
        )
        self.gui_elements.append(title)
        
        # Filename entry
        filename_label = OnscreenText(
            text="Enter filename:",
            pos=(0, 0.4),
            scale=0.07,
            fg=(1, 1, 1, 1),
            align=TextNode.ACenter
        )
        self.gui_elements.append(filename_label)
        
        filename_entry = DirectEntry(
            text="",
            scale=0.08,
            pos=(-0.4, 0, 0.2),
            initialText="my_army",
            numLines=1,
            width=10,
            frameColor=(0.3, 0.3, 0.3, 1)
        )
        self.gui_elements.append(filename_entry)
        
        # Save button
        save_btn = DirectButton(
            text="Save",
            scale=0.08,
            pos=(0, 0, -0.1),
            command=self.save_army_list,
            extraArgs=[filename_entry],
            frameSize=(-2, 2, -0.5, 1),
            frameColor=(0, 0.6, 0, 1)
        )
        self.gui_elements.append(save_btn)
        
        # Cancel button
        cancel_btn = DirectButton(
            text="Cancel",
            scale=0.07,
            pos=(0, 0, -0.4),
            command=self.show_main_menu,
            frameSize=(-2, 2, -0.5, 1)
        )
        self.gui_elements.append(cancel_btn)
    
    def save_army_list(self, filename_entry):
        """Save the army list to a file"""
        filename = filename_entry.get().strip()
        if not filename:
            self.show_message("Please enter a filename!")
            return

        filepath = os.path.join(self.base_dir, f"{filename}.json")

        try:
            save_data = {'budget': self.points_budget, 'units': self.army_list}
            with open(filepath, 'w') as f:
                json.dump(save_data, f, indent=4)
            self.show_message(f"Army list saved to {filename}.json!", self.show_main_menu)
        except Exception as e:
            self.show_message(f"Error saving: {e}")
    
    def show_load_screen(self):
        """Show screen for loading army list"""
        self.clear_screen()
        self.current_screen = "load"
        
        # Title
        title = OnscreenText(
            text="LOAD ARMY LIST",
            pos=(0, 0.7),
            scale=0.1,
            fg=(1, 1, 0, 1),
            align=TextNode.ACenter
        )
        self.gui_elements.append(title)
        
        # Filename entry
        filename_label = OnscreenText(
            text="Enter filename:",
            pos=(0, 0.4),
            scale=0.07,
            fg=(1, 1, 1, 1),
            align=TextNode.ACenter
        )
        self.gui_elements.append(filename_label)
        
        filename_entry = DirectEntry(
            text="",
            scale=0.08,
            pos=(-0.4, 0, 0.2),
            initialText="my_army",
            numLines=1,
            width=10,
            frameColor=(0.3, 0.3, 0.3, 1)
        )
        self.gui_elements.append(filename_entry)
        
        # Load button
        load_btn = DirectButton(
            text="Load",
            scale=0.08,
            pos=(0, 0, -0.1),
            command=self.load_army_list_file,
            extraArgs=[filename_entry],
            frameSize=(-2, 2, -0.5, 1),
            frameColor=(0, 0.6, 0, 1)
        )
        self.gui_elements.append(load_btn)
        
        # Cancel button
        cancel_btn = DirectButton(
            text="Cancel",
            scale=0.07,
            pos=(0, 0, -0.4),
            command=self.show_main_menu,
            frameSize=(-2, 2, -0.5, 1)
        )
        self.gui_elements.append(cancel_btn)
    
    def load_army_list_file(self, filename_entry):
        """Load an army list from a file"""
        filename = filename_entry.get().strip()
        if not filename:
            self.show_message("Please enter a filename!")
            return

        filepath = os.path.join(self.base_dir, f"{filename}.json")

        if not os.path.exists(filepath):
            self.show_message(f"File {filename}.json not found!")
            return

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            # Support both new {budget, units} format and legacy list format
            if isinstance(data, dict) and 'units' in data:
                self.points_budget = data.get('budget', self.points_budget)
                self.army_list = data['units']
            else:
                self.army_list = data
            self.show_message(f"Army list loaded from {filename}.json!", self.show_main_menu)
        except Exception as e:
            self.show_message(f"Error loading: {e}")

    def load_from_file(self, filepath):
        """Silently load an army list directly from a file path (no dialog)."""
        if not filepath or not os.path.exists(filepath):
            print(f"[ListBuilder] File not found: {filepath}")
            return
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            if isinstance(data, dict) and 'units' in data:
                self.points_budget = data.get('budget', self.points_budget)
                self.army_list = data['units']
            else:
                self.army_list = data
            print(f"[ListBuilder] Loaded {len(self.army_list)} units from {filepath}")
        except Exception as e:
            print(f"[ListBuilder] Error loading {filepath}: {e}")

    def show_budget_screen(self):
        """Show screen to set the shared points budget"""
        self.clear_screen()
        self.current_screen = "budget"

        bg_frame = DirectFrame(
            frameSize=(-1.0, 1.0, -0.72, 0.72),
            frameColor=(0.05, 0.08, 0.05, 0.95),
            pos=(0, 0, 0),
            relief=DGG.SUNKEN,
            borderWidth=(0.01, 0.01)
        )
        self.gui_elements.append(bg_frame)

        title = OnscreenText(
            text="SET POINTS BUDGET",
            pos=(0, 0.62),
            scale=0.1,
            fg=(1, 0.85, 0.2, 1),
            align=TextNode.ACenter
        )
        self.gui_elements.append(title)

        line = DirectFrame(
            frameSize=(-0.7, 0.7, -0.003, 0.003),
            frameColor=(0.8, 0.6, 0.1, 1),
            pos=(0, 0, 0.53)
        )
        self.gui_elements.append(line)

        info = OnscreenText(
            text="Both players agree on a shared limit.\nRecruit units with your allotted points — spend wisely!",
            pos=(0, 0.40),
            scale=0.055,
            fg=(0.8, 0.85, 0.8, 1),
            align=TextNode.ACenter
        )
        self.gui_elements.append(info)

        current_label = OnscreenText(
            text=f"Current budget: {self.points_budget} pts  |  Spent: {self._army_total_pts()} pts",
            pos=(0, 0.22),
            scale=0.062,
            fg=(0.4, 1, 0.55, 1),
            align=TextNode.ACenter
        )
        self.gui_elements.append(current_label)

        entry_label = OnscreenText(
            text="New budget:",
            pos=(-0.42, 0.05),
            scale=0.06,
            fg=(0.9, 0.9, 0.9, 1),
            align=TextNode.ALeft
        )
        self.gui_elements.append(entry_label)

        budget_entry = DirectEntry(
            text="",
            scale=0.07,
            pos=(0.12, 0, 0.05),
            initialText=str(self.points_budget),
            numLines=1,
            width=7,
            frameColor=(0.2, 0.3, 0.2, 1),
            text_fg=(1, 1, 1, 1)
        )
        self.gui_elements.append(budget_entry)

        preset_label = OnscreenText(
            text="Quick presets:",
            pos=(-0.42, -0.12),
            scale=0.052,
            fg=(0.75, 0.75, 0.75, 1),
            align=TextNode.ALeft
        )
        self.gui_elements.append(preset_label)

        for pts, x_off in [(500, -0.52), (1000, -0.19), (1500, 0.14), (2000, 0.47)]:
            preset_btn = DirectButton(
                text=str(pts),
                text_scale=0.9,
                scale=0.063,
                pos=(x_off, 0, -0.25),
                command=budget_entry.set,
                extraArgs=[str(pts)],
                frameSize=(-2.2, 2.2, -0.6, 1.1),
                frameColor=(0.25, 0.35, 0.25, 1),
                text_fg=(0.9, 1, 0.9, 1),
                relief=DGG.RAISED,
                borderWidth=(0.01, 0.01)
            )
            self.gui_elements.append(preset_btn)

        confirm_btn = DirectButton(
            text="Confirm",
            text_scale=0.9,
            scale=0.08,
            pos=(0, 0, -0.44),
            command=self._apply_budget,
            extraArgs=[budget_entry],
            frameSize=(-3.0, 3.0, -0.6, 1.1),
            frameColor=(0.2, 0.6, 0.2, 1),
            text_fg=(1, 1, 1, 1),
            relief=DGG.RAISED,
            borderWidth=(0.015, 0.015)
        )
        self.gui_elements.append(confirm_btn)

        back_btn = DirectButton(
            text="< Back",
            text_scale=0.9,
            scale=0.07,
            pos=(0, 0, -0.62),
            command=self.show_main_menu,
            frameSize=(-2.5, 2.5, -0.6, 1.1),
            frameColor=(0.3, 0.3, 0.4, 1),
            text_fg=(1, 1, 1, 1),
            relief=DGG.RAISED,
            borderWidth=(0.01, 0.01)
        )
        self.gui_elements.append(back_btn)

    def _apply_budget(self, entry):
        """Apply the new points budget from the entry widget."""
        try:
            val = int(entry.get().strip())
            if val < 1:
                self.show_message("Budget must be at least 1 point!")
                return
            self.points_budget = val
            self.show_message(f"Points budget set to {val} pts!", self.show_main_menu)
        except ValueError:
            self.show_message("Please enter a valid whole number!")

    def show_message(self, message, next_command=None):
        """Show a message dialog"""
        self.clear_screen()
        
        # Message text
        msg_text = OnscreenText(
            text=message,
            pos=(0, 0.2),
            scale=0.08,
            fg=(1, 1, 1, 1),
            align=TextNode.ACenter,
            wordwrap=20
        )
        self.gui_elements.append(msg_text)
        
        # OK button
        ok_btn = DirectButton(
            text="OK",
            scale=0.08,
            pos=(0, 0, -0.2),
            command=next_command if next_command else self.show_main_menu,
            frameSize=(-2, 2, -0.5, 1)
        )
        self.gui_elements.append(ok_btn)
    
    def exit_builder(self):
        """Exit the list builder and return to game"""
        self.clear_screen()
        print("Exiting Army List Builder")
        # If you want to return to the game, you can add logic here
    
    def hide(self):
        """Hide all GUI elements without destroying them"""
        for element in self.gui_elements:
            element.hide()
    
    def show(self):
        """Show all GUI elements"""
        for element in self.gui_elements:
            element.show()


# Standalone test application
class ListBuilderApp(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        
        # Set background color - dark mystical blue/purple
        self.setBackgroundColor(0.08, 0.06, 0.15, 1)
        
        # Disable mouse camera control
        self.disableMouse()
        
        # Create the list builder GUI
        self.list_builder = ArmyListBuilderGUI(self)
        
        # Add ESC key to exit
        self.accept('escape', self.userExit)
        
        print("="*60)
        print("WARHAMMER ARMY LIST BUILDER - GUI")
        print("="*60)
        print("Army List Builder GUI initialized")
        print("Navigate through menus to build your army")
        print("Press ESC to exit")
        print("="*60)


if __name__ == "__main__":
    app = ListBuilderApp()
    app.run()
