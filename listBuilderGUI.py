"""
Panda3D GUI-based Interactive Army List Builder
Integrates with the Warhammer game using DirectGUI components
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
        self.available_units = {}
        self.army_list = []
        self.gui_elements = []
        
        self.load_available_units()
        self.current_screen = None
        
        # Show main menu by default
        self.show_main_menu()
    
    def load_available_units(self):
        """Load all available units from JSON characteristic files"""
        json_files = [f for f in os.listdir('.') if f.endswith('_characteristics.json')]
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    unit_name = data.get('Model', 'Unknown')
                    self.available_units[unit_name] = {
                        'file': json_file,
                        'characteristics': data
                    }
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
        
        print(f"Loaded {len(self.available_units)} unit types")
    
    def clear_screen(self):
        """Remove all GUI elements"""
        for element in self.gui_elements:
            element.destroy()
        self.gui_elements = []
    
    def show_main_menu(self):
        """Display the main menu"""
        self.clear_screen()
        self.current_screen = "main_menu"
        
        # Title
        title = OnscreenText(
            text="ARMY LIST BUILDER",
            pos=(0, 0.85),
            scale=0.12,
            fg=(1, 1, 0, 1),
            align=TextNode.ACenter,
            mayChange=False
        )
        self.gui_elements.append(title)
        
        # Subtitle
        subtitle = OnscreenText(
            text="Build Your Warhammer Army",
            pos=(0, 0.70),
            scale=0.06,
            fg=(0.8, 0.8, 0.8, 1),
            align=TextNode.ACenter,
            mayChange=False
        )
        self.gui_elements.append(subtitle)
        
        # Menu buttons
        button_data = [
            ("View Available Units", self.show_unit_browser, 0.4),
            ("Add Unit to Army", self.show_add_unit_screen, 0.2),
            ("View Current Army", self.show_army_list_screen, 0.0),
            ("Save Army List", self.show_save_screen, -0.2),
            ("Load Army List", self.show_load_screen, -0.4),
            ("Exit List Builder", self.exit_builder, -0.7)
        ]
        
        for text, command, y_pos in button_data:
            btn = DirectButton(
                text=text,
                scale=0.08,
                pos=(0, 0, y_pos),
                command=command,
                frameSize=(-4, 4, -0.5, 1),
                text_fg=(1, 1, 1, 1),
                relief=DGG.RAISED,
                borderWidth=(0.02, 0.02)
            )
            self.gui_elements.append(btn)
    
    def show_unit_browser(self):
        """Display all available units in a scrolled list"""
        self.clear_screen()
        self.current_screen = "unit_browser"
        
        # Title
        title = OnscreenText(
            text="AVAILABLE UNITS",
            pos=(0, 0.9),
            scale=0.1,
            fg=(1, 1, 0, 1),
            align=TextNode.ACenter
        )
        self.gui_elements.append(title)
        
        # Create scrolled frame for units
        frame = DirectScrolledFrame(
            canvasSize=(-1, 1, -len(self.available_units) * 0.15, 0.1),
            frameSize=(-1, 1, -0.7, 0.7),
            pos=(0, 0, 0),
            scrollBarWidth=0.04,
            frameColor=(0.2, 0.2, 0.2, 0.8)
        )
        self.gui_elements.append(frame)
        
        # Add unit entries
        sorted_units = sorted(self.available_units.keys())
        y_pos = 0
        
        for unit_name in sorted_units:
            stats = self.available_units[unit_name]['characteristics']
            
            # Unit name
            name_text = DirectLabel(
                text=unit_name,
                text_scale=0.05,
                text_align=TextNode.ALeft,
                pos=(-0.9, 0, y_pos),
                frameColor=(0, 0, 0, 0),
                text_fg=(1, 1, 1, 1),
                parent=frame.getCanvas()
            )
            
            # Stats text
            stats_str = f"M:{stats.get('M','?')} WS:{stats.get('WS','?')} BS:{stats.get('BS','?')} " \
                       f"S:{stats.get('S','?')} T:{stats.get('T','?')} W:{stats.get('W','?')} " \
                       f"I:{stats.get('I','?')} A:{stats.get('A','?')} Ld:{stats.get('Ld','?')}"
            
            stats_text = DirectLabel(
                text=stats_str,
                text_scale=0.04,
                text_align=TextNode.ALeft,
                pos=(-0.9, 0, y_pos - 0.06),
                frameColor=(0, 0, 0, 0),
                text_fg=(0.7, 0.7, 0.7, 1),
                parent=frame.getCanvas()
            )
            
            y_pos -= 0.15
        
        # Back button
        back_btn = DirectButton(
            text="Back to Main Menu",
            scale=0.07,
            pos=(0, 0, -0.85),
            command=self.show_main_menu,
            frameSize=(-3, 3, -0.5, 1)
        )
        self.gui_elements.append(back_btn)
    
    def show_add_unit_screen(self):
        """Show screen for adding a unit"""
        self.clear_screen()
        self.current_screen = "add_unit"
        
        # Title
        title = OnscreenText(
            text="ADD UNIT TO ARMY",
            pos=(0, 0.9),
            scale=0.1,
            fg=(1, 1, 0, 1),
            align=TextNode.ACenter
        )
        self.gui_elements.append(title)
        
        # Instructions
        instructions = OnscreenText(
            text="Select a unit type:",
            pos=(0, 0.75),
            scale=0.06,
            fg=(0.9, 0.9, 0.9, 1),
            align=TextNode.ACenter
        )
        self.gui_elements.append(instructions)
        
        # Create scrolled frame for unit selection
        sorted_units = sorted(self.available_units.keys())
        
        # Calculate canvas height based on number of units
        canvas_height = len(sorted_units) * 0.1
        
        unit_frame = DirectScrolledFrame(
            canvasSize=(-0.4, 0.4, -canvas_height, 0),
            frameSize=(-0.5, 0.5, -0.55, 0.65),
            frameColor=(0.2, 0.2, 0.2, 0.8),
            pos=(0, 0, 0),
            scrollBarWidth=0.04,
            verticalScroll_scrollSize=0.1,
            verticalScroll_thumb_relief=DGG.RAISED,
            verticalScroll_incButton_relief=DGG.RAISED,
            verticalScroll_decButton_relief=DGG.RAISED
        )
        self.gui_elements.append(unit_frame)
        
        # Add unit buttons to the scrolled frame
        y_pos = -0.05
        for unit_name in sorted_units:
            btn = DirectButton(
                text=unit_name,
                text_scale=0.045,
                text_align=TextNode.ALeft,
                text_pos=(-0.35, -0.01),
                pos=(0, 0, y_pos),
                command=self.show_unit_config_screen,
                extraArgs=[unit_name],
                frameSize=(-0.38, 0.38, -0.04, 0.04),
                frameColor=(0.3, 0.3, 0.3, 1),
                text_fg=(1, 1, 1, 1),
                relief=DGG.RAISED,
                borderWidth=(0.005, 0.005),
                parent=unit_frame.getCanvas()
            )
            y_pos -= 0.1
        
        # Back button
        back_btn = DirectButton(
            text="Back",
            scale=0.07,
            pos=(0, 0, -0.85),
            command=self.show_main_menu,
            frameSize=(-2, 2, -0.5, 1)
        )
        self.gui_elements.append(back_btn)
    
    def show_unit_config_screen(self, unit_name):
        """Show configuration screen for a selected unit"""
        self.clear_screen()
        self.current_screen = "unit_config"
        
        # Title
        title = OnscreenText(
            text=f"CONFIGURE: {unit_name}",
            pos=(0, 0.9),
            scale=0.08,
            fg=(1, 1, 0, 1),
            align=TextNode.ACenter
        )
        self.gui_elements.append(title)
        
        # Show unit stats
        stats = self.available_units[unit_name]['characteristics']
        stats_str = f"M:{stats.get('M','?')} WS:{stats.get('WS','?')} BS:{stats.get('BS','?')} " \
                   f"S:{stats.get('S','?')} T:{stats.get('T','?')} W:{stats.get('W','?')} " \
                   f"I:{stats.get('I','?')} A:{stats.get('A','?')} Ld:{stats.get('Ld','?')}"
        
        stats_label = OnscreenText(
            text=stats_str,
            pos=(0, 0.75),
            scale=0.05,
            fg=(0.8, 0.8, 0.8, 1),
            align=TextNode.ACenter
        )
        self.gui_elements.append(stats_label)
        
        # Number of models entry
        models_label = OnscreenText(
            text="Number of Models:",
            pos=(-0.5, 0.5),
            scale=0.06,
            fg=(1, 1, 1, 1),
            align=TextNode.ALeft
        )
        self.gui_elements.append(models_label)
        
        models_entry = DirectEntry(
            text="",
            scale=0.07,
            pos=(0.3, 0, 0.5),
            initialText="10",
            numLines=1,
            width=5,
            frameColor=(0.3, 0.3, 0.3, 1)
        )
        self.gui_elements.append(models_entry)
        
        # Files (width) entry
        files_label = OnscreenText(
            text="Files (Width):",
            pos=(-0.5, 0.3),
            scale=0.06,
            fg=(1, 1, 1, 1),
            align=TextNode.ALeft
        )
        self.gui_elements.append(files_label)
        
        files_entry = DirectEntry(
            text="",
            scale=0.07,
            pos=(0.3, 0, 0.3),
            initialText="5",
            numLines=1,
            width=5,
            frameColor=(0.3, 0.3, 0.3, 1)
        )
        self.gui_elements.append(files_entry)
        
        # Ranks (depth) entry
        ranks_label = OnscreenText(
            text="Ranks (Depth):",
            pos=(-0.5, 0.1),
            scale=0.06,
            fg=(1, 1, 1, 1),
            align=TextNode.ALeft
        )
        self.gui_elements.append(ranks_label)
        
        ranks_entry = DirectEntry(
            text="",
            scale=0.07,
            pos=(0.3, 0, 0.1),
            initialText="2",
            numLines=1,
            width=5,
            frameColor=(0.3, 0.3, 0.3, 1)
        )
        self.gui_elements.append(ranks_entry)
        
        # Add button
        add_btn = DirectButton(
            text="Add to Army",
            scale=0.08,
            pos=(0, 0, -0.3),
            command=self.add_configured_unit,
            extraArgs=[unit_name, models_entry, files_entry, ranks_entry],
            frameSize=(-3, 3, -0.5, 1),
            frameColor=(0, 0.6, 0, 1)
        )
        self.gui_elements.append(add_btn)
        
        # Back button
        back_btn = DirectButton(
            text="Cancel",
            scale=0.07,
            pos=(0, 0, -0.6),
            command=self.show_add_unit_screen,
            frameSize=(-2, 2, -0.5, 1)
        )
        self.gui_elements.append(back_btn)
    
    def add_configured_unit(self, unit_name, models_entry, files_entry, ranks_entry):
        """Add the configured unit to the army list"""
        try:
            nmodels = int(models_entry.get())
            files = int(files_entry.get())
            ranks = int(ranks_entry.get())
            
            if nmodels < 1 or files < 1 or ranks < 1:
                self.show_message("Error: All values must be positive!")
                return
            
            # Add to army list
            army_unit = {
                'name': unit_name,
                'nmodels': nmodels,
                'files': files,
                'ranks': ranks,
                'json_file': self.available_units[unit_name]['file']
            }
            
            self.army_list.append(army_unit)
            self.show_message(f"Added {unit_name} to army!", self.show_main_menu)
            
        except ValueError:
            self.show_message("Error: Please enter valid numbers!")
    
    def show_army_list_screen(self):
        """Display the current army list"""
        self.clear_screen()
        self.current_screen = "army_list"
        
        # Title
        title = OnscreenText(
            text="CURRENT ARMY LIST",
            pos=(0, 0.9),
            scale=0.1,
            fg=(1, 1, 0, 1),
            align=TextNode.ACenter
        )
        self.gui_elements.append(title)
        
        if not self.army_list:
            empty_msg = OnscreenText(
                text="Army list is empty.\nAdd some units!",
                pos=(0, 0.3),
                scale=0.08,
                fg=(0.8, 0.8, 0.8, 1),
                align=TextNode.ACenter
            )
            self.gui_elements.append(empty_msg)
        else:
            # Calculate canvas height based on number of units
            canvas_height = len(self.army_list) * 0.18
            
            # Create scrolled frame for army list
            frame = DirectScrolledFrame(
                canvasSize=(-0.95, 0.95, -canvas_height, 0),
                frameSize=(-1, 1, -0.6, 0.7),
                pos=(0, 0, 0),
                scrollBarWidth=0.04,
                frameColor=(0.2, 0.2, 0.2, 0.8),
                verticalScroll_scrollSize=0.1
            )
            self.gui_elements.append(frame)
            
            # Add army units
            y_pos = -0.05
            for idx, army_unit in enumerate(self.army_list):
                # Unit info
                unit_text = f"{idx+1}. {army_unit['name']}"
                name_label = DirectLabel(
                    text=unit_text,
                    text_scale=0.05,
                    text_align=TextNode.ALeft,
                    pos=(-0.9, 0, y_pos),
                    frameColor=(0, 0, 0, 0),
                    text_fg=(1, 1, 1, 1),
                    parent=frame.getCanvas()
                )
                
                # Configuration
                config_text = f"Models: {army_unit['nmodels']} | Formation: {army_unit['files']}x{army_unit['ranks']}"
                config_label = DirectLabel(
                    text=config_text,
                    text_scale=0.04,
                    text_align=TextNode.ALeft,
                    pos=(-0.9, 0, y_pos - 0.06),
                    frameColor=(0, 0, 0, 0),
                    text_fg=(0.7, 0.7, 0.7, 1),
                    parent=frame.getCanvas()
                )
                
                # Remove button
                remove_btn = DirectButton(
                    text="Remove",
                    text_scale=0.028,
                    text_pos=(0, -0.006),
                    pos=(0.68, 0, y_pos - 0.03),
                    command=self.remove_unit_from_army,
                    extraArgs=[idx],
                    frameSize=(-0.7, 0.7, -0.08, 0.10),
                    frameColor=(0.7, 0, 0, 1),
                    relief=DGG.RAISED,
                    borderWidth=(0.003, 0.003),
                    parent=frame.getCanvas()
                )
                
                y_pos -= 0.18
            
            # Summary
            total_models = sum(u['nmodels'] for u in self.army_list)
            summary = OnscreenText(
                text=f"Total Units: {len(self.army_list)} | Total Models: {total_models}",
                pos=(0, -0.75),
                scale=0.06,
                fg=(0.8, 1, 0.8, 1),
                align=TextNode.ACenter
            )
            self.gui_elements.append(summary)
        
        # Back button
        back_btn = DirectButton(
            text="Back to Main Menu",
            scale=0.07,
            pos=(0, 0, -0.85),
            command=self.show_main_menu,
            frameSize=(-3, 3, -0.5, 1)
        )
        self.gui_elements.append(back_btn)
    
    def remove_unit_from_army(self, idx):
        """Remove a unit from the army list"""
        if 0 <= idx < len(self.army_list):
            removed = self.army_list.pop(idx)
            self.show_army_list_screen()  # Refresh the display
    
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
        
        filename = f"{filename}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(self.army_list, f, indent=4)
            self.show_message(f"Army list saved to {filename}!", self.show_main_menu)
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
        
        filename = f"{filename}.json"
        
        if not os.path.exists(filename):
            self.show_message(f"File {filename} not found!")
            return
        
        try:
            with open(filename, 'r') as f:
                self.army_list = json.load(f)
            self.show_message(f"Army list loaded from {filename}!", self.show_main_menu)
        except Exception as e:
            self.show_message(f"Error loading: {e}")
    
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
        
        # Set background color
        self.setBackgroundColor(0.1, 0.1, 0.15, 1)
        
        # Disable mouse camera control
        self.disableMouse()
        
        # Create the list builder GUI
        self.list_builder = ArmyListBuilderGUI(self)
        
        # Add ESC key to exit
        self.accept('escape', self.userExit)
        
        print("Army List Builder GUI initialized")
        print("Press ESC to exit")


if __name__ == "__main__":
    app = ListBuilderApp()
    app.run()
