"""
Interactive Army List Builder
Allows players to select and configure units for their Warhammer army
"""

import os
import json
from models import model
from units import unit

class ArmyListBuilder:
    def __init__(self):
        self.available_units = {}
        self.army_list = []
        self.load_available_units()
        
    def load_available_units(self):
        """Load all available units from JSON characteristic files"""
        json_files = [f for f in os.listdir('.') if f.endswith('_characteristics.json')]
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    unit_name = data.get('Model', 'Unknown')
                    # Store the json filename for later use
                    self.available_units[unit_name] = {
                        'file': json_file,
                        'characteristics': data
                    }
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
        
        print(f"Loaded {len(self.available_units)} unit types")
    
    def display_available_units(self):
        """Display all available units with their stats"""
        print("\n" + "="*80)
        print("AVAILABLE UNITS")
        print("="*80)
        
        sorted_units = sorted(self.available_units.keys())
        for idx, unit_name in enumerate(sorted_units, 1):
            stats = self.available_units[unit_name]['characteristics']
            print(f"{idx:2d}. {unit_name:30s} | M:{stats.get('M','?'):2s} WS:{stats.get('WS','?'):2s} "
                  f"BS:{stats.get('BS','?'):2s} S:{stats.get('S','?'):2s} T:{stats.get('T','?'):2s} "
                  f"W:{stats.get('W','?'):2s} I:{stats.get('I','?'):2s} A:{stats.get('A','?'):2s} "
                  f"Ld:{stats.get('Ld','?'):2s}")
        print("="*80)
        
    def display_army_list(self):
        """Display the current army list"""
        print("\n" + "="*80)
        print("CURRENT ARMY LIST")
        print("="*80)
        
        if not self.army_list:
            print("(Empty - no units added yet)")
        else:
            for idx, army_unit in enumerate(self.army_list, 1):
                print(f"{idx:2d}. {army_unit['name']:30s} | "
                      f"Models: {army_unit['nmodels']:3d} | "
                      f"Formation: {army_unit['files']}x{army_unit['ranks']} "
                      f"(Files x Ranks)")
        
        print("="*80)
        print(f"Total units: {len(self.army_list)}")
        print(f"Total models: {sum(u['nmodels'] for u in self.army_list)}")
        print("="*80)
    
    def add_unit_to_army(self):
        """Add a unit to the army list"""
        self.display_available_units()
        
        try:
            choice = input("\nEnter unit number to add (or 'c' to cancel): ").strip()
            if choice.lower() == 'c':
                return
            
            choice_idx = int(choice) - 1
            sorted_units = sorted(self.available_units.keys())
            
            if choice_idx < 0 or choice_idx >= len(sorted_units):
                print("Invalid selection!")
                return
            
            unit_name = sorted_units[choice_idx]
            
            # Get unit configuration
            print(f"\nConfiguring: {unit_name}")
            nmodels = int(input("Number of models: ").strip())
            files = int(input("Number of files (width): ").strip())
            ranks = int(input("Number of ranks (depth): ").strip())
            
            if nmodels < 1 or files < 1 or ranks < 1:
                print("Invalid configuration - all values must be positive!")
                return
            
            if nmodels < files * ranks:
                print(f"Warning: {nmodels} models insufficient for {files}x{ranks} formation!")
                proceed = input("Add anyway? (y/n): ").strip().lower()
                if proceed != 'y':
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
            print(f"\n✓ Added {unit_name} to army list!")
            
        except ValueError:
            print("Invalid input!")
        except Exception as e:
            print(f"Error: {e}")
    
    def remove_unit_from_army(self):
        """Remove a unit from the army list"""
        if not self.army_list:
            print("\nArmy list is empty!")
            return
        
        self.display_army_list()
        
        try:
            choice = input("\nEnter unit number to remove (or 'c' to cancel): ").strip()
            if choice.lower() == 'c':
                return
            
            choice_idx = int(choice) - 1
            
            if choice_idx < 0 or choice_idx >= len(self.army_list):
                print("Invalid selection!")
                return
            
            removed_unit = self.army_list.pop(choice_idx)
            print(f"\n✓ Removed {removed_unit['name']} from army list!")
            
        except ValueError:
            print("Invalid input!")
        except Exception as e:
            print(f"Error: {e}")
    
    def save_army_list(self):
        """Save the army list to a file"""
        if not self.army_list:
            print("\nCannot save an empty army list!")
            return
        
        filename = input("\nEnter filename to save (without extension): ").strip()
        if not filename:
            print("Invalid filename!")
            return
        
        filename = f"{filename}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(self.army_list, f, indent=4)
            print(f"\n✓ Army list saved to {filename}")
        except Exception as e:
            print(f"Error saving: {e}")
    
    def load_army_list(self):
        """Load an army list from a file"""
        filename = input("\nEnter filename to load (without extension): ").strip()
        if not filename:
            print("Invalid filename!")
            return
        
        filename = f"{filename}.json"
        
        if not os.path.exists(filename):
            print(f"File {filename} not found!")
            return
        
        try:
            with open(filename, 'r') as f:
                self.army_list = json.load(f)
            print(f"\n✓ Army list loaded from {filename}")
            self.display_army_list()
        except Exception as e:
            print(f"Error loading: {e}")
    
    def view_unit_details(self):
        """View detailed stats for a specific unit"""
        self.display_available_units()
        
        try:
            choice = input("\nEnter unit number to view details (or 'c' to cancel): ").strip()
            if choice.lower() == 'c':
                return
            
            choice_idx = int(choice) - 1
            sorted_units = sorted(self.available_units.keys())
            
            if choice_idx < 0 or choice_idx >= len(sorted_units):
                print("Invalid selection!")
                return
            
            unit_name = sorted_units[choice_idx]
            stats = self.available_units[unit_name]['characteristics']
            
            print("\n" + "="*60)
            print(f"UNIT DETAILS: {unit_name}")
            print("="*60)
            for key, value in stats.items():
                print(f"{key:15s}: {value}")
            print("="*60)
            
        except ValueError:
            print("Invalid input!")
        except Exception as e:
            print(f"Error: {e}")
    
    def main_menu(self):
        """Display and handle the main menu"""
        while True:
            print("\n" + "="*60)
            print("ARMY LIST BUILDER - MAIN MENU")
            print("="*60)
            print("1. View Available Units")
            print("2. View Unit Details")
            print("3. Add Unit to Army")
            print("4. Remove Unit from Army")
            print("5. View Current Army List")
            print("6. Save Army List")
            print("7. Load Army List")
            print("8. Clear Army List")
            print("9. Exit")
            print("="*60)
            
            choice = input("\nEnter your choice (1-9): ").strip()
            
            if choice == '1':
                self.display_available_units()
            elif choice == '2':
                self.view_unit_details()
            elif choice == '3':
                self.add_unit_to_army()
            elif choice == '4':
                self.remove_unit_from_army()
            elif choice == '5':
                self.display_army_list()
            elif choice == '6':
                self.save_army_list()
            elif choice == '7':
                self.load_army_list()
            elif choice == '8':
                confirm = input("\nAre you sure you want to clear the army list? (y/n): ").strip().lower()
                if confirm == 'y':
                    self.army_list = []
                    print("\n✓ Army list cleared!")
            elif choice == '9':
                print("\nExiting Army List Builder. Goodbye!")
                break
            else:
                print("\nInvalid choice! Please enter a number from 1-9.")
            
            # Pause before showing menu again
            if choice != '9':
                input("\nPress Enter to continue...")


def main():
    """Main entry point for the list builder"""
    print("\n" + "="*60)
    print("WARHAMMER ARMY LIST BUILDER")
    print("="*60)
    print("Welcome! Build your army by selecting units.")
    print("="*60)
    
    builder = ArmyListBuilder()
    builder.main_menu()


if __name__ == "__main__":
    main()
