"""
Campaign Map module for Panda3D terrain rendering.
Loads PNG heightmaps as terrain and applies texture overlays.
"""

from direct.fsm.FSM import FSM
from panda3d.core import (
    GeoMipTerrain, 
    Texture, 
    TextureStage,
    PNMImage,
    Filename,
    Point3,
    Vec3,
    Vec4,
    NodePath,
    Shader,
    TransparencyAttrib,
    BitMask32,
    ColorWriteAttrib
)
from panda3d.bullet import (
    BulletRigidBodyNode,
    BulletHeightfieldShape,
    BulletTriangleMesh,
    BulletTriangleMeshShape,
    ZUp
)


class CampaignMap:
    """
    A campaign map class that loads terrain from a PNG heightmap
    and applies a texture to the terrain surface.
    """
    
    def __init__(self, base, name="campaign_terrain"):
        """
        Initialize the campaign map.
        
        Args:
            base: The Panda3D ShowBase instance
            name: Name identifier for the terrain
        """
        self.base = base
        self.name = name
        self.terrain = None
        self.terrain_root = None
        self.heightmap_image = None
        self.texture = None
        self.texture_stage = None
        self.collision_node = None
        
        # Terrain configuration
        self.block_size = 64  # Size of terrain blocks
        self.height_scale = 50.0  # Vertical scale multiplier
        self.horizontal_scale = 1.0  # Horizontal scale multiplier
        self.near_distance = 50  # LOD near distance
        self.far_distance = 200  # LOD far distance
        
    def load_heightmap(self, heightmap_path, height_scale=None, horizontal_scale=None):
        """
        Load a PNG file as a heightmap terrain.
        
        Args:
            heightmap_path: Path to the PNG heightmap file
            height_scale: Optional vertical scale multiplier
            horizontal_scale: Optional horizontal scale multiplier
            
        Returns:
            NodePath to the terrain root node
        """
        if height_scale is not None:
            self.height_scale = height_scale
        if horizontal_scale is not None:
            self.horizontal_scale = horizontal_scale
            
        # Create GeoMipTerrain
        self.terrain = GeoMipTerrain(self.name)
        
        # Load the heightmap image
        self.terrain.setHeightfield(Filename(heightmap_path))
        
        # Store the heightmap image for height queries
        self.heightmap_image = PNMImage()
        self.heightmap_image.read(Filename(heightmap_path))
        
        # Configure terrain LOD
        self.terrain.setBlockSize(self.block_size)
        self.terrain.setNear(self.near_distance)
        self.terrain.setFar(self.far_distance)
        
        # Enable automatic LOD updates based on focal point
        self.terrain.setFocalPoint(self.base.camera)
        
        # Set minimum level of detail (0 = highest detail)
        self.terrain.setMinLevel(0)
        
        # Bruteforce mode generates entire terrain at once (better for smaller terrains)
        # Set to False for large terrains with dynamic LOD
        self.terrain.setBruteforce(True)
        
        # Generate the terrain geometry
        self.terrain.generate()
        
        # Get the root node and parent it to render
        self.terrain_root = self.terrain.getRoot()
        self.terrain_root.reparentTo(self.base.render)
        
        # Apply scaling
        self.terrain_root.setScale(
            self.horizontal_scale, 
            self.horizontal_scale, 
            self.height_scale
        )
        
        # Center the terrain (optional - comment out if you want origin at corner)
        terrain_size = self.get_terrain_size()
        # self.terrain_root.setPos(-terrain_size[0] / 2, -terrain_size[1] / 2, 0)
        
        return self.terrain_root
    
    def set_texture(self, texture_path, wrap_mode=True):
        """
        Apply a texture to the terrain.
        
        Args:
            texture_path: Path to the PNG texture file
            wrap_mode: If True, texture will repeat; if False, it will clamp
            
        Returns:
            The applied Texture object
        """
        if self.terrain_root is None:
            raise RuntimeError("Must load heightmap before setting texture")
        
        # Load the texture
        self.texture = self.base.loader.loadTexture(texture_path)
        
        if self.texture is None:
            raise ValueError(f"Failed to load texture: {texture_path}")
        
        # Configure texture wrapping
        if wrap_mode:
            self.texture.setWrapU(Texture.WM_repeat)
            self.texture.setWrapV(Texture.WM_repeat)
        else:
            self.texture.setWrapU(Texture.WM_clamp)
            self.texture.setWrapV(Texture.WM_clamp)
        
        # Create a texture stage for the base texture
        self.texture_stage = TextureStage('terrain_texture')
        self.texture_stage.setMode(TextureStage.M_modulate)
        
        # Apply texture to terrain
        self.terrain_root.setTexture(self.texture_stage, self.texture)
        
        # Set texture scale based on terrain size for proper UV mapping
        terrain_size = self.get_terrain_size()
        # Adjust these values to control texture tiling
        self.terrain_root.setTexScale(self.texture_stage, 1, 1)
        
        return self.texture
    
    def add_detail_texture(self, texture_path, scale=10.0):
        """
        Add a detail texture that tiles across the terrain.
        
        Args:
            texture_path: Path to the detail texture PNG
            scale: How many times the texture repeats across the terrain
            
        Returns:
            The detail TextureStage
        """
        if self.terrain_root is None:
            raise RuntimeError("Must load heightmap before adding detail texture")
        
        detail_tex = self.base.loader.loadTexture(texture_path)
        detail_tex.setWrapU(Texture.WM_repeat)
        detail_tex.setWrapV(Texture.WM_repeat)
        
        detail_stage = TextureStage('detail_texture')
        detail_stage.setMode(TextureStage.M_modulate)
        detail_stage.setSort(1)  # Render after base texture
        
        self.terrain_root.setTexture(detail_stage, detail_tex)
        self.terrain_root.setTexScale(detail_stage, scale, scale)
        
        return detail_stage
    
    def get_terrain_size(self):
        """
        Get the size of the terrain in world units.
        
        Returns:
            Tuple of (width, depth) after scaling
        """
        if self.heightmap_image is None:
            return (0, 0)
        
        width = self.heightmap_image.getXSize() * self.horizontal_scale
        depth = self.heightmap_image.getYSize() * self.horizontal_scale
        
        return (width, depth)
    
    def get_height_at(self, x, y):
        """
        Get the terrain height at a specific world position.
        
        Args:
            x: X coordinate in world space
            y: Y coordinate in world space
            
        Returns:
            Height value at the given position, or 0 if outside terrain
        """
        if self.heightmap_image is None:
            return 0
        
        # Convert world coordinates to heightmap pixel coordinates
        px = int(x / self.horizontal_scale)
        py = int(y / self.horizontal_scale)
        
        # Bounds check
        if (px < 0 or px >= self.heightmap_image.getXSize() or
            py < 0 or py >= self.heightmap_image.getYSize()):
            return 0
        
        # Get grayscale value (0-1) and scale by height
        gray = self.heightmap_image.getGray(px, py)
        return gray * self.height_scale
    
    def update(self):
        """
        Update terrain LOD. Call this each frame for dynamic LOD terrain.
        Only needed if setBruteforce(False) was used.
        """
        if self.terrain is not None:
            self.terrain.update()
    
    def create_collision_mesh(self, bullet_world):
        """
        Create a Bullet physics collision mesh for the terrain.
        
        Args:
            bullet_world: The BulletWorld instance to add the collision to
            
        Returns:
            NodePath to the collision node
        """
        if self.terrain_root is None:
            raise RuntimeError("Must load heightmap before creating collision mesh")
        
        # Create triangle mesh from terrain geometry
        mesh = BulletTriangleMesh()
        
        # Iterate through terrain geometry and add to mesh
        for geom_node in self.terrain_root.findAllMatches('**/+GeomNode'):
            geom_node_obj = geom_node.node()
            for i in range(geom_node_obj.getNumGeoms()):
                geom = geom_node_obj.getGeom(i)
                mesh.addGeom(geom, True, geom_node.getTransform(self.terrain_root))
        
        # Create collision shape
        shape = BulletTriangleMeshShape(mesh, dynamic=False)
        
        # Create rigid body node
        self.collision_node = BulletRigidBodyNode('terrain_collision')
        self.collision_node.addShape(shape)
        self.collision_node.setMass(0)  # Static object
        
        # Create NodePath and attach to render
        collision_np = self.base.render.attachNewNode(self.collision_node)
        collision_np.setTransform(self.terrain_root.getTransform())
        
        # Add to bullet world
        bullet_world.attachRigidBody(self.collision_node)
        
        return collision_np
    
    def contryCollision(self, bullet_world, contryNP):
        if self.terrain_root is None:
            raise RuntimeError("Must load heightmap before creating collision mesh")
        
        # Create triangle mesh from terrain geometry
        mesh = BulletTriangleMesh()
        
        # Iterate through terrain geometry and add to mesh
        for geom_node in contryNP.findAllMatches('**/+GeomNode'):
            geom_node_obj = geom_node.node()
            for i in range(geom_node_obj.getNumGeoms()):
                geom = geom_node_obj.getGeom(i)
                mesh.addGeom(geom, True, geom_node.getTransform(contryNP))
        
        # Create collision shape
        shape = BulletTriangleMeshShape(mesh, dynamic=False)
        
        # Create rigid body node
        name=contryNP.getName()
        collision_node = BulletRigidBodyNode(f'{name}_collision')
        collision_node.addShape(shape)
        collision_node.setMass(0)  # Static object
        
        # Create NodePath and attach to render
        collision_np = self.base.render.attachNewNode(collision_node)
        collision_np.setTransform(contryNP.getTransform())
        
        # Add to bullet world
        bullet_world.attachRigidBody(collision_node)
        
        return collision_np
    
    def set_position(self, x, y, z):
        """Set the terrain position in world space."""
        if self.terrain_root:
            self.terrain_root.setPos(x, y, z)
    
    def set_scale(self, sx, sy, sz):
        """Set the terrain scale."""
        if self.terrain_root:
            self.terrain_root.setScale(sx, sy, sz)
            self.horizontal_scale = sx
            self.height_scale = sz
    
    def hide(self):
        """Hide the terrain."""
        if self.terrain_root:
            self.terrain_root.hide()
    
    def show(self):
        """Show the terrain."""
        if self.terrain_root:
            self.terrain_root.show()
    
    def destroy(self):
        """Clean up and remove the terrain from the scene."""
        if self.terrain_root:
            self.terrain_root.removeNode()
            self.terrain_root = None
        self.terrain = None
        self.heightmap_image = None
        self.texture = None


class CountryFSM(FSM):
    """
    Finite State Machine for managing country selection and states.
    Dynamically creates states for each country in the provided country model.
    """
    
    def __init__(self, country_model, base=None):
        """
        Initialize the CountryFSM.
        
        Args:
            country_model: NodePath containing country children
            base: Optional ShowBase instance for rendering operations
        """
        FSM.__init__(self, 'CountryFSM')
        self.country_model = country_model
        self.base = base
        self.countries = {}
        self.current_country_np = None
        self.highlighted_neighbors = []  # Track currently highlighted neighbors
        self.visited_countries = set()  # Track countries that have been visited/clicked
        
        # Dictionary to store which countries border each other
        # Format: {country_name: [list of neighboring country names]}
        self.borders = {}
        
        # Load the country color GLSL shader
        self.country_shader = Shader.load(
            Shader.SL_GLSL,
            vertex="shaders/country_color.vert",
            fragment="shaders/country_color.frag"
        )
        
        # Define colors for different states (r, g, b, blend_strength)
        self.selected_color = Vec4(0.0, 0.8, 0.2, 0.7)
        self.neighbor_color = Vec4(0.2, 0.5, 0.8, 0.6)  # Blue for neighbors
        self.hover_color = Vec4(0.8, 0.8, 0.0, 0.5)
        self.visited_color = Vec4(0.3, 0.6, 0.3, 0.6)  # Darker green for visited
        self.visited_color = self.selected_color
        
        # Blend mode: 0=multiply, 1=overlay, 2=additive
        self.color_blend_mode = 2.0
        
        # Generate a unique default color per country using golden-ratio hue spacing
        self.country_default_colors = {}
        children = list(country_model.getChildren())
        golden_ratio = 0.618033988749895
        base_hue = 0.0  # Starting hue
        
        # Store reference to each country NodePath by name
        for i, child in enumerate(children):
            country_name = child.getName()
            self.countries[country_name] = child
            self.borders[country_name] = []  # Initialize empty borders list
            
            # Generate a unique hue, then convert HSV -> RGB
            hue = (base_hue + i * golden_ratio) % 1.0
            saturation = 0.5
            value = 0.6
            color_rgb = self._hsv_to_rgb(hue, saturation, value)
            default_color = Vec4(color_rgb[0], color_rgb[1], color_rgb[2], 0.4)
            self.country_default_colors[country_name] = default_color
            
            # Apply shader and set unique default color
            child.setShader(self.country_shader)
            child.setShaderInput("countryColor", default_color)
            child.setShaderInput("colorBlendMode", self.color_blend_mode)
            child.setShaderInput("animState", 0.0)
            child.setShaderInput("countryTime", 0.0)
            
            print(f"Registered country state: {country_name} with color {default_color}")
        
        # Start a per-frame task to feed time to the shader
        if self.base is not None:
            self.base.taskMgr.add(self._update_country_time, "update_country_time")
    
    @staticmethod
    def _hsv_to_rgb(h, s, v):
        """Convert HSV (0-1 range) to RGB (0-1 range)."""
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return (r, g, b)
    
    def _get_default_color(self, country_name):
        """Get the unique default color for a country."""
        return self.country_default_colors.get(country_name, Vec4(0.1, 0.1, 0.1, 0.3))
    
    def _update_country_time(self, task):
        """Per-frame task to update the countryTime uniform on all countries."""
        t = task.time
        for country_np in self.countries.values():
            country_np.setShaderInput("countryTime", t)
        return task.cont
    
    def _apply_country_color(self, country_np, color, anim_state=0.0):
        """
        Apply a color and animation state to a country via the GLSL shader.
        
        Args:
            country_np: NodePath of the country
            color: Vec4 color (r, g, b, blend_strength)
            anim_state: 0=idle, 1=selected pulse, 2=neighbor wave, 3=hover shimmer
        """
        country_np.setShaderInput("countryColor", color)
        country_np.setShaderInput("animState", float(anim_state))
    
    def enterNone(self):
        """Enter the None/idle state - no country selected."""
        print("FSM: Entering None state")
        # Reset all countries to default appearance, but keep visited countries colored
        for country_name, country_np in self.countries.items():
            if country_name in self.visited_countries:
                self._apply_country_color(country_np, self.visited_color, anim_state=0.0)
            else:
                self._apply_country_color(country_np, self._get_default_color(country_name), anim_state=0.0)
        self.current_country_np = None
        self.highlighted_neighbors = []
    
    def exitNone(self):
        """Exit the None/idle state."""
        print("FSM: Exiting None state")
    
    def selectCountry(self, country_name):
        """
        Transition to a specific country state.
        Only allows transitions to neighboring countries or from None state.
        Prevents revisiting already visited countries.
        
        Args:
            country_name: Name of the country to select
        """
        if country_name not in self.countries:
            print(f"Warning: Country '{country_name}' not found")
            return
        
        # Check if country has already been visited
        if country_name in self.visited_countries:
            print(f"Cannot select {country_name}: Already visited")
            # Visual feedback for visited country
            if country_name in self.countries:
                invalid_np = self.countries[country_name]
                # Could add a brief red flash or pulse effect here
            return
        
        current_state = self.state
        
        # Allow transition if:
        # 1. Currently in None state (first selection)
        # 2. Target country is a neighbor of current country AND not visited
        if current_state == 'None':
            self.request(country_name)
        elif country_name in self.borders.get(current_state, []):
            print(f"Transitioning from {current_state} to neighboring {country_name}")
            self.request(country_name)
        else:
            print(f"Cannot transition: {country_name} is not adjacent to {current_state}")
            # Optionally provide visual feedback that the transition is invalid
            if country_name in self.countries:
                # Flash the country to indicate it's not accessible
                invalid_np = self.countries[country_name]
                original_color = invalid_np.getColor()
                invalid_np.setColor(Vec4(1.0, 0.0, 0.0, 0.5))  # Red flash
                # You could add a task to reset color after a delay
    
    def deselectCountry(self):
        """Deselect the current country and return to None state."""
        self.request('None')
    
    def __getattr__(self, name):
        """
        Dynamically handle enter/exit methods for country states.
        This allows the FSM to work with any country names without
        explicitly defining methods for each one.
        """
        # Check if this is an enter method for a known country
        if name.startswith('enter'):
            country_name = name[5:]  # Remove 'enter' prefix
            if country_name in self.countries:
                return lambda: self._enterCountry(country_name)
        
        # Check if this is an exit method for a known country
        elif name.startswith('exit'):
            country_name = name[4:]  # Remove 'exit' prefix
            if country_name in self.countries:
                return lambda: self._exitCountry(country_name)
        
        # If not a state method, raise AttributeError
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def _enterCountry(self, country_name):
        """Generic enter method for any country state."""
        print(f"FSM: Entering {country_name} state")
        self.current_country_np = self.countries[country_name]
        # Mark this country as visited
        self.visited_countries.add(country_name)
        # Highlight the selected country via shader with pulsing animation
        self._apply_country_color(self.current_country_np, self.selected_color, anim_state=1.0)
        self.current_country_np.setTransparency(TransparencyAttrib.MAlpha)
        self.current_country_np.setBin("transparent", 60)
        
        # Highlight neighboring countries (only unvisited ones)
        self.highlighted_neighbors = []
        if country_name in self.borders:
            for neighbor_name in self.borders[country_name]:
                # Only highlight if not visited
                if neighbor_name in self.countries and neighbor_name not in self.visited_countries:
                    neighbor_np = self.countries[neighbor_name]
                    self._apply_country_color(neighbor_np, self.neighbor_color, anim_state=2.0)
                    neighbor_np.setTransparency(TransparencyAttrib.MAlpha)
                    neighbor_np.setBin("transparent", 55)
                    self.highlighted_neighbors.append(neighbor_name)
                    print(f"  Highlighted neighbor: {neighbor_name}")
                elif neighbor_name in self.visited_countries:
                    print(f"  Skipping visited neighbor: {neighbor_name}")
        
        # You can add custom behavior here, such as:
        # - Displaying country information
        # - Playing sounds
        # - Triggering animations
        # - Updating UI elements
    
    def _exitCountry(self, country_name):
        """Generic exit method for any country state."""
        print(f"FSM: Exiting {country_name} state")
        
        if self.current_country_np:
            # Set to visited color instead of default (no animation)
            self._apply_country_color(self.current_country_np, self.visited_color, anim_state=0.0)
            self.current_country_np.setBin("transparent", 50)
        
        # Reset highlighted neighbors - check if they're visited
        for neighbor_name in self.highlighted_neighbors:
            if neighbor_name in self.countries:
                if neighbor_name in self.visited_countries:
                    self._apply_country_color(self.countries[neighbor_name], self.visited_color, anim_state=0.0)
                else:
                    self._apply_country_color(self.countries[neighbor_name], self._get_default_color(neighbor_name), anim_state=0.0)
                self.countries[neighbor_name].setBin("transparent", 50)
        
        self.highlighted_neighbors = []
    
    def hoverCountry(self, country_name):
        """Temporarily highlight a country on hover with shimmer animation."""
        if country_name in self.countries and country_name != self.state:
            self._apply_country_color(self.countries[country_name], self.hover_color, anim_state=3.0)
    
    def unhoverCountry(self, country_name):
        """Remove hover highlight from a country."""
        if country_name in self.countries and country_name != self.state:
            # Restore appropriate color and animation based on status
            if country_name in self.visited_countries:
                self._apply_country_color(self.countries[country_name], self.visited_color, anim_state=0.0)
            elif country_name in self.highlighted_neighbors:
                # Restore neighbor wave animation
                self._apply_country_color(self.countries[country_name], self.neighbor_color, anim_state=2.0)
            else:
                self._apply_country_color(self.countries[country_name], self._get_default_color(country_name), anim_state=0.0)
    
    def getCurrentCountry(self):
        """Get the currently selected country name."""
        return self.state if self.state != 'None' else None
    
    def getCountryNodePath(self, country_name):
        """Get the NodePath for a specific country."""
        return self.countries.get(country_name)
    
    def getAllCountries(self):
        """Get a list of all country names."""
        return list(self.countries.keys())
    
    def setBorders(self, country_name, neighboring_countries):
        """
        Define which countries border a specific country.
        
        Args:
            country_name: Name of the country
            neighboring_countries: List of country names that border this country
        """
        if country_name in self.countries:
            self.borders[country_name] = neighboring_countries
            print(f"Set borders for {country_name}: {neighboring_countries}")
        else:
            print(f"Warning: Cannot set borders for unknown country '{country_name}'")
    
    def addBorder(self, country1, country2, bidirectional=True):
        """
        Add a border between two countries.
        
        Args:
            country1: First country name
            country2: Second country name
            bidirectional: If True, adds border in both directions (default)
        """
        if country1 in self.countries and country2 in self.countries:
            if country2 not in self.borders[country1]:
                self.borders[country1].append(country2)
            
            if bidirectional and country1 not in self.borders[country2]:
                self.borders[country2].append(country1)
            
            print(f"Added border: {country1} <-> {country2}" if bidirectional else f"Added border: {country1} -> {country2}")
        else:
            print(f"Warning: Cannot add border between unknown countries")
    
    def getBorders(self, country_name):
        """
        Get the list of neighboring countries.
        
        Args:
            country_name: Name of the country
            
        Returns:
            List of neighboring country names
        """
        return self.borders.get(country_name, [])
    
    def isNeighbor(self, country1, country2):
        """
        Check if two countries are neighbors.
        
        Args:
            country1: First country name
            country2: Second country name
            
        Returns:
            True if countries are neighbors, False otherwise
        """
        return country2 in self.borders.get(country1, [])
    
    def getAvailableMoves(self, country_name=None):
        """
        Get list of unvisited neighboring countries that can be moved to.
        
        Args:
            country_name: Name of the country (defaults to current state)
            
        Returns:
            List of unvisited neighboring country names
        """
        if country_name is None:
            country_name = self.state
        
        if country_name == 'None' or country_name not in self.borders:
            return []
        
        # Filter out visited countries from neighbors
        available = [n for n in self.borders[country_name] 
                    if n not in self.visited_countries and n in self.countries]
        return available
    
    def hasAvailableMoves(self, country_name=None):
        """
        Check if there are any unvisited neighbors to move to.
        
        Args:
            country_name: Name of the country (defaults to current state)
            
        Returns:
            True if there are unvisited neighbors, False otherwise
        """
        return len(self.getAvailableMoves(country_name)) > 0
    
    def clearVisitedCountries(self, reset_state=True):
        """
        Clear the visited countries set and reset all colors to default.
        
        Args:
            reset_state: If True, also returns FSM to None state (default: True)
        """
        self.visited_countries.clear()
        for country_name, country_np in self.countries.items():
            self._apply_country_color(country_np, self._get_default_color(country_name), anim_state=0.0)
        
        if reset_state and self.state != 'None':
            self.request('None')
        
        print("Cleared all visited countries")
    
    def getVisitedCountries(self):
        """Get the set of visited country names."""
        return self.visited_countries.copy()
    
    def isVisited(self, country_name):
        """Check if a country has been visited."""
        return country_name in self.visited_countries
    
    def markAsVisited(self, country_name):
        """Manually mark a country as visited."""
        if country_name in self.countries:
            self.visited_countries.add(country_name)
            if country_name != self.state:  # Don't change if currently selected
                self._apply_country_color(self.countries[country_name], self.visited_color, anim_state=0.0)
            print(f"Marked {country_name} as visited")
        else:
            print(f"Warning: Cannot mark unknown country '{country_name}' as visited")
    
    def setCountryColor(self, country_name, color):
        """
        Set a custom color for a specific country via the GLSL shader.
        
        Args:
            country_name: Name of the country
            color: Vec4 color (r, g, b, blend_strength)
        """
        if country_name in self.countries:
            self._apply_country_color(self.countries[country_name], color)
            print(f"Set custom color for {country_name}: {color}")
        else:
            print(f"Warning: Cannot set color for unknown country '{country_name}'")
    
    def setBlendMode(self, mode):
        """
        Set the color blend mode for all countries.
        
        Args:
            mode: 0 = multiply, 1 = overlay, 2 = additive
        """
        self.color_blend_mode = float(mode)
        for country_np in self.countries.values():
            country_np.setShaderInput("colorBlendMode", self.color_blend_mode)


# Example usage (for testing)
if __name__ == "__main__":
    from direct.showbase.ShowBase import ShowBase
    from panda3d.core import AmbientLight, DirectionalLight, Vec4
    from panda3d.bullet import BulletWorld
    from panda3d.core import Vec3
    import random
    from panda3d.core import AntialiasAttrib
    from panda3d.core import loadPrcFileData
    from panda3d.core import TransparencyAttrib, ColorWriteAttrib
    from panda3d.core import Shader

    # Configure before ShowBase creation
    loadPrcFileData("", """
        framebuffer-multisample 1
        multisamples 4
        framebuffer-hardware 1
        framebuffer-software 0
    """)
    
    class TerrainDemo(ShowBase):
        def __init__(self):
            ShowBase.__init__(self)
            
            # Create campaign map
            self.campaign_map = CampaignMap(self)
            
            # Load heightmap (replace with your heightmap path)
            self.campaign_map.load_heightmap("assets/textures/wals_dem_resized.png",height_scale=25)
            
            # Apply texture (replace with your texture path)
            self.campaign_map.set_texture("assets/textures/wals_tex_resized.png")
            
            # Position camera to view terrain
            self.disable_mouse()
            self.camera.setPos(1000, -1000, 1500)
            self.camera.lookAt(500, 750, 0)
            self.camera.lookAt(1025, 0, 0)
            #self.camera.lookAt(self.campaign_map.terrain_root)
            self.accept("m", self.enMo)  # Press 'm' to enable mouse control
            # Move terrain so center is at origin (0,0,0)
            terrain_size = self.campaign_map.get_terrain_size()
            print("Terrain size:", terrain_size)
            self.campaign_map.set_position(-1024 / 2, -2048 / 2, 0)
            
            # Set up lighting
            
            # Ambient light for overall scene illumination
            ambient_light = AmbientLight('ambient')
            ambient_light.setColor(Vec4(0.3, 0.3, 0.3, 1))
            ambient_np = self.render.attachNewNode(ambient_light)
            self.render.setLight(ambient_np)
            
            # Directional light (sun)
            directional_light = DirectionalLight('sun')
            directional_light.setColor(Vec4(0.8, 0.8, 0.7, 1))
            directional_np = self.render.attachNewNode(directional_light)
            directional_np.setHpr(45, -45, 0)
            self.render.setLight(directional_np)

            # Update task for dynamic LOD
            self.taskMgr.add(self.update_terrain, "update_terrain")

            self.country = self.loader.loadModel("models/blender/maps1.bam")
            self.country.reparentTo(self.render)
            self.camera.lookAt(self.country)

            # Create Bullet physics world
            self.bullet_world = BulletWorld()
            self.bullet_world.setGravity(Vec3(0, 0, -9.81))

            # Load shader
            cloud_shader = Shader.load(
                Shader.SL_GLSL,
                vertex="cloud.vert.txt",
                fragment="cloud.frag.txt"
            )
            
            # Store nodes with cloud shader for time updates
            self.cloud_nodes = []

            # Create collision meshes for terrain and country
            #self.campaign_map.create_collision_mesh(self.bullet_world)
            # Create a plane above the terrain for cloud shader
            cloud_plane = self.loader.loadModel("models/box")  # Simple plane/box model
            cloud_plane.setScale(1000, 2000, 1)  # Scale to cover terrain area
            cloud_plane.setPos(-512, -1024, 20)  # Position above terrain
            cloud_plane.setShader(cloud_shader)
            cloud_plane.setShaderInput("customTime", 0.0)
            cloud_plane.setShaderInput("cloudColor", Vec4(1.0, 1.0, 1.0, 1.0))
            cloud_plane.setShaderInput("skyColor", Vec4(0.5, 0.7, 0.9, 0.1))
            cloud_plane.setShaderInput("cloudCoverage", 0.5)  # 0.0 = no clouds, 1.0 = full coverage
            cloud_plane.setTransparency(TransparencyAttrib.MAlpha)
            cloud_plane.setBin("transparent", 100)
            cloud_plane.reparentTo(self.render)
            self.cloud_nodes.append(cloud_plane)
            for childe in self.country.getChildren():
                print(childe.getName())
                np = self.campaign_map.contryCollision(self.bullet_world, childe)
                
                # Setup transparency and rendering for texture stage blending
                childe.setTransparency(TransparencyAttrib.MAlpha)
                childe.setBin("transparent", 50)
                childe.setDepthTest(False)
                #childe.setDepthWrite(False)
                """ childe.setShader(cloud_shader)
                # Set shader inputs
                childe.setShaderInput("customTime", 0.0)  # Initialize time
                childe.setShaderInput("cloudColor", Vec4(1.0, 1.0, 1.0, 1.0))  # White
                childe.setShaderInput("skyColor", Vec4(0.5, 0.7, 0.9, 0.3))    # Light blue
                childe.setTransparency(TransparencyAttrib.MAlpha)
                childe.setBin("transparent", 10)
                # Store reference for updates
                self.cloud_nodes.append(childe) """
            #self.campaign_map.contryCollision(self.bullet_world, self.country.find("**/Plane.002"))
            
            # Initialize Country FSM
            self.country_fsm = CountryFSM(self.country, self)
            self.country_fsm.request('None')  # Start in None state
            print(f"Available countries: {self.country_fsm.getAllCountries()}")
            
            # Define country borders (adjacency relationships)
            # Replace these with your actual country names and their neighbors
            # Example format:
            # self.country_fsm.addBorder("CountryA", "CountryB")  # Makes A and B neighbors
            
            # Get all countries for easy reference
            all_countries = self.country_fsm.getAllCountries()
            
            # Example: Define borders automatically or manually
            # You can define borders like this:
            # self.country_fsm.addBorder("Plane", "Plane.001")
            # self.country_fsm.addBorder("Plane.001", "Plane.002")
            # self.country_fsm.addBorder("Plane.002", "Plane.003")
            # etc...
            
            #self.country_fsm.addBorder("Plane.008", "Plane.007")
            #self.country_fsm.addBorder("Plane.008", "Plane.011")
            #base.toggleTexture()
            
            
            self.country_fsm.setBorders("Plane.005", ["Plane.007", "Plane.006", "Plane.009"])
            self.country_fsm.setBorders("Plane.006", ["Plane.005", "Plane.009", "Plane.010", "Plane.011", "Plane.007"])
            self.country_fsm.setBorders("Plane.007", ["Plane.005", "Plane.006", "Plane.011", "Plane.008"])
            self.country_fsm.setBorders("Plane.008", ["Plane.007", "Plane.011"])
            self.country_fsm.setBorders("Plane.009", ["Plane.005", "Plane.006", "Plane.010", "Plane.013"])
            self.country_fsm.setBorders("Plane.010", ["Plane.006", "Plane.009", "Plane.012", "Plane.013", "Plane.011"])
            self.country_fsm.setBorders("Plane.011", ["Plane.006", "Plane.007", "Plane.008", "Plane.010", "Plane.012", "Plane.016"])
            self.country_fsm.setBorders("Plane.012", ["Plane.010", "Plane.011","Plane.013", "Plane.016", "Plane.015"])
            self.country_fsm.setBorders("Plane.013", ["Plane.009", "Plane.010", "Plane.012", "Plane.015", "Plane.014"])
            self.country_fsm.setBorders("Plane.014", ["Plane.013", "Plane.015"])
            self.country_fsm.setBorders("Plane.015", ["Plane.012", "Plane.013", "Plane.014", "Plane.016", "Plane.017", "Plane.018"])
            self.country_fsm.setBorders("Plane.016", ["Plane.011", "Plane.012", "Plane.015", "Plane.017"])
            self.country_fsm.setBorders("Plane.017", ["Plane.015", "Plane.016", "Plane.019"])
            self.country_fsm.setBorders("Plane.018", ["Plane.015"])
            self.country_fsm.setBorders("Plane.019", ["Plane.017"])
            """ # Or use a more flexible approach:
            if len(all_countries) >= 2:
                # Example: Connect first few countries in a chain for demonstration
                for i in range(min(len(all_countries) - 1, 10)):
                    self.country_fsm.addBorder(all_countries[i], all_countries[i + 1])
                    print(f"Auto-connected: {all_countries[i]} <-> {all_countries[i + 1]}") """
            
            print("\\nBorder configuration complete. Selected country will highlight its neighbors in blue.")
            print("You can only transition to neighboring (blue-highlighted) countries.\\n")

            # Add physics update task
            self.taskMgr.add(self.update_physics, "update_physics")
            
            # Add cloud time update task
            self.taskMgr.add(self.update_cloud_time, "update_cloud_time")

            self.accept("mouse1", self.mouseClick)
            self.accept("mouse3", self.deselect)  # Right-click to deselect
            #mask_model.setBin("background", 10)
            #self.campaign_map.terrain_root.setAttrib(ColorWriteAttrib.make(ColorWriteAttrib.C_off))

            # Target model uses normal depth testing
            #self.campaign_map.terrain_root.setBin("opaque", 20)
            # Depth test will naturally occlude where mask is closer
            #self.render.setAntialias(AntialiasAttrib.MAuto)
        
        def update_terrain(self, task):
            self.campaign_map.update()
            return task.cont
        
        def enMo(self):
            self.enableMouse()

        def update_physics(self, task):
            dt = globalClock.getDt()
            self.bullet_world.doPhysics(dt)
            return task.cont
        
        def update_cloud_time(self, task):
            # Update time uniform for all cloud shader nodes
            for node in self.cloud_nodes:
                node.setShaderInput("customTime", task.time*0.1)
            return task.cont
        
        def mouseClick(self):
            if self.mouseWatcherNode.hasMouse():
                mpos = self.mouseWatcherNode.getMouse()
                print("Mouse position:", mpos)
                pMouse = base.mouseWatcherNode.getMouse()
                pFrom = Point3()
                pTo = Point3()
                base.camLens.extrude(pMouse, pFrom, pTo)
                # Transform to global coordinates
                pFrom = render.getRelativePoint(base.cam, pFrom)
                pTo = render.getRelativePoint(base.cam, pTo)
                result = self.bullet_world.rayTestClosest(pFrom, pTo, BitMask32.bit(1))

                if result.hasHit():
                    hit_node_name = result.getNode().getName()
                    print(f"Hit: {hit_node_name} at {result.getHitPos()}")
                    
                    # Extract country name from collision node name (removes '_collision' suffix)
                    country_name = hit_node_name.split("_")[0]
                    
                    # Use FSM to select the country
                    self.country_fsm.selectCountry(country_name)
                    
                    # Old manual color code (now handled by FSM):
                    # np=self.render.find("**/"+country_name)
                    # np.setColor(random.random(), random.random(), random.random(), 0.3)
                    # np.setTransparency(TransparencyAttrib.MAlpha)
                    # np.setBin("transparent", 50)
                    # np.setDepthTest(False)
                else:
                    print("No collision detected")
                    # Deselect if clicking empty space
                    self.country_fsm.deselectCountry()
        
        def deselect(self):
            """Handle right-click to deselect country."""
            print("Deselecting country")
            self.country_fsm.deselectCountry()
    
    # Run demo
    demo = TerrainDemo()
    demo.run()
