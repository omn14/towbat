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
        self.factions = {}  # faction_name -> [unit_name, ...]
        self.army_list = []
        self.load_available_units()
        
    def load_available_units(self):
        """Load all available units from JSON characteristic files in army_units/"""
        army_units_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'army_units')
        
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

        # Sort unit lists within each faction
        for faction in self.factions:
            self.factions[faction].sort()

        print(f"Loaded {len(self.available_units)} unit types across {len(self.factions)} factions")
    
    def display_available_units(self):
        """Display all available units grouped by faction"""
        print("\n" + "="*80)
        print("AVAILABLE UNITS")
        print("="*80)

        idx = 0
        unit_index_map = []  # flat list for selection
        for faction in sorted(self.factions.keys()):
            print(f"\n  -- {faction} --")
            for unit_name in self.factions[faction]:
                idx += 1
                unit_index_map.append(unit_name)
                stats = self.available_units[unit_name]['characteristics']
                print(f"{idx:2d}. {unit_name:30s} | M:{stats.get('M','?'):2s} WS:{stats.get('WS','?'):2s} "
                      f"BS:{stats.get('BS','?'):2s} S:{stats.get('S','?'):2s} T:{stats.get('T','?'):2s} "
                      f"W:{stats.get('W','?'):2s} I:{stats.get('I','?'):2s} A:{stats.get('A','?'):2s} "
                      f"Ld:{stats.get('Ld','?'):2s}")
        print("="*80)
        return unit_index_map
        
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
    
    def _select_faction(self):
        """Prompt the user to select a faction. Returns faction name or None."""
        sorted_factions = sorted(self.factions.keys())
        print("\n" + "="*60)
        print("SELECT FACTION")
        print("="*60)
        for idx, faction in enumerate(sorted_factions, 1):
            count = len(self.factions[faction])
            print(f"{idx:2d}. {faction}  ({count} units)")
        print("="*60)

        try:
            choice = input("\nEnter faction number (or 'c' to cancel): ").strip()
            if choice.lower() == 'c':
                return None
            choice_idx = int(choice) - 1
            if choice_idx < 0 or choice_idx >= len(sorted_factions):
                print("Invalid selection!")
                return None
            return sorted_factions[choice_idx]
        except ValueError:
            print("Invalid input!")
            return None

    def add_unit_to_army(self):
        """Add a unit to the army list (faction-first workflow)"""
        faction = self._select_faction()
        if faction is None:
            return

        # Display units for the selected faction
        faction_units = self.factions[faction]
        print(f"\n" + "="*60)
        print(f"{faction.upper()} UNITS")
        print("="*60)
        for idx, unit_name in enumerate(faction_units, 1):
            stats = self.available_units[unit_name]['characteristics']
            print(f"{idx:2d}. {unit_name:30s} | M:{stats.get('M','?'):2s} WS:{stats.get('WS','?'):2s} "
                  f"BS:{stats.get('BS','?'):2s} S:{stats.get('S','?'):2s} T:{stats.get('T','?'):2s} "
                  f"W:{stats.get('W','?'):2s} I:{stats.get('I','?'):2s} A:{stats.get('A','?'):2s} "
                  f"Ld:{stats.get('Ld','?'):2s}")
        print("="*60)

        try:
            choice = input("\nEnter unit number to add (or 'c' to cancel): ").strip()
            if choice.lower() == 'c':
                return

            choice_idx = int(choice) - 1
            if choice_idx < 0 or choice_idx >= len(faction_units):
                print("Invalid selection!")
                return

            unit_name = faction_units[choice_idx]
            
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
                'faction': faction,
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
        unit_index_map = self.display_available_units()

        try:
            choice = input("\nEnter unit number to view details (or 'c' to cancel): ").strip()
            if choice.lower() == 'c':
                return

            choice_idx = int(choice) - 1

            if choice_idx < 0 or choice_idx >= len(unit_index_map):
                print("Invalid selection!")
                return

            unit_name = unit_index_map[choice_idx]
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
