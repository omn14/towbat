"""
Example showing how to integrate the Army List Builder GUI into your existing game
"""

from direct.showbase.ShowBase import ShowBase
from listBuilderGUI import ArmyListBuilderGUI


class GameWithListBuilder(ShowBase):
    """
    Example of integrating the list builder into an existing Panda3D application
    """
    def __init__(self):
        ShowBase.__init__(self)
        
        # Your existing game setup...
        self.setBackgroundColor(0.1, 0.1, 0.15, 1)
        self.disableMouse()
        
        # Initialize the list builder GUI (hidden by default)
        self.list_builder = None
        self.list_builder_active = False
        
        # Set up key bindings
        self.accept('l', self.toggle_list_builder)  # Press 'L' to toggle list builder
        self.accept('escape', self.handle_escape)
        
        print("Game initialized")
        print("Press 'L' to open Army List Builder")
        print("Press ESC to exit")
    
    def toggle_list_builder(self):
        """Toggle the list builder GUI on/off"""
        if not self.list_builder_active:
            # Open list builder
            if self.list_builder is None:
                self.list_builder = ArmyListBuilderGUI(self)
            else:
                self.list_builder.show()
            
            self.list_builder_active = True
            print("List Builder opened")
            
            # You might want to pause game logic here
            # self.pause_game()
        else:
            # Close list builder
            if self.list_builder:
                self.list_builder.hide()
            
            self.list_builder_active = False
            print("List Builder closed")
            
            # Resume game logic
            # self.resume_game()
    
    def handle_escape(self):
        """Handle ESC key - close list builder if open, otherwise exit"""
        if self.list_builder_active:
            self.toggle_list_builder()
        else:
            self.userExit()
    
    def get_army_list(self):
        """Get the current army list from the builder"""
        if self.list_builder:
            return self.list_builder.army_list
        return []
    
    def create_units_from_army_list(self):
        """
        Example: Create actual game units from the army list
        Call this after the player has finished building their army
        """
        if not self.list_builder:
            print("No army list available")
            return []
        
        army_list = self.list_builder.army_list
        created_units = []
        
        for army_unit in army_list:
            # Here you would create actual unit objects
            # using your existing units.py and models.py classes
            print(f"Creating unit: {army_unit['name']} with {army_unit['nmodels']} models")
            
            # Example (adjust to your actual unit creation code):
            # model_obj = model(army_unit['name'], "some_url")
            # unit_obj = unit(
            #     army_unit['name'],
            #     model_obj,
            #     army_unit['nmodels'],
            #     army_unit['files'],
            #     army_unit['ranks']
            # )
            # created_units.append(unit_obj)
        
        return created_units


# INTEGRATION WITH YOUR game.py:
# 
# 1. Add import at the top:
#    from listBuilderGUI import ArmyListBuilderGUI
# 
# 2. In your game class __init__, add:
#    self.list_builder = None
#    self.list_builder_active = False
#    self.accept('l', self.toggle_list_builder)
# 
# 3. Add the toggle_list_builder method to your game class
# 
# 4. Optionally add a method to create units from the army list


if __name__ == "__main__":
    app = GameWithListBuilder()
    app.run()
