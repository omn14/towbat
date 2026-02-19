"""
Campaign Map module for Panda3D terrain rendering.
Loads PNG heightmaps as terrain and applies texture overlays.
"""

from panda3d.core import (
    GeoMipTerrain, 
    Texture, 
    TextureStage,
    PNMImage,
    Filename,
    Point3,
    Vec3,
    NodePath,
    Shader,
    TransparencyAttrib
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


# Example usage (for testing)
if __name__ == "__main__":
    from direct.showbase.ShowBase import ShowBase
    from panda3d.core import AmbientLight, DirectionalLight, Vec4
    
    class TerrainDemo(ShowBase):
        def __init__(self):
            ShowBase.__init__(self)
            
            # Create campaign map
            self.campaign_map = CampaignMap(self)
            
            # Load heightmap (replace with your heightmap path)
            self.campaign_map.load_heightmap("assets/textures/wals_dem.png",height_scale=25)
            
            # Apply texture (replace with your texture path)
            self.campaign_map.set_texture("assets/textures/wals_tex.png")
            
            # Position camera to view terrain
            self.disable_mouse()
            self.camera.setPos(1000, -1000, 1500)
            self.camera.lookAt(500, 750, 0)
            #self.camera.lookAt(self.campaign_map.terrain_root)
            self.accept("m", self.enMo)  # Press 'm' to enable mouse control
            # Move terrain so center is at origin (0,0,0)
            terrain_size = self.campaign_map.get_terrain_size()
            self.campaign_map.set_position(-terrain_size[0] / 2, -terrain_size[1] / 2, 0)
            
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
        
        def update_terrain(self, task):
            self.campaign_map.update()
            return task.cont
        
        def enMo(self):
            self.enableMouse()
    
    # Run demo
    demo = TerrainDemo()
    demo.run()
