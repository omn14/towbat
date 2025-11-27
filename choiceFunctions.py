from panda3d.core import Vec3, CardMaker, Point3, BitMask32
from panda3d.bullet import BulletRigidBodyNode, BulletBoxShape
from direct.showbase.DirectObject import DirectObject

class Choice:
    def __init__(self, num_choices,pos):
        self.num_choices = num_choices
        self.choiceMade = False
        self.boxes = []
        for i in range(num_choices):
            loc=pos+Vec3(i*4,0,0)
            box=self.create_bullet_rigidbody_cube(None, location=loc, size=2.0, name=f"ChoiceCube_{i}")
            self.boxes.append(box)
        taskMgr.add(self.mouseActivate, "mouseActivateTask")
        self.helper1 = DirectObject()
        self.helper1.accept('mouse1', self.onMouseClick)
        #self.old = messenger.whoAccepts('mouse1')
        #base.accept("mouse1", self.onMouseClick)

    def cleanup(self):
        taskMgr.remove("mouseActivateTask")
        self.helper1.ignore('mouse1')
        for box in self.boxes:
            base.world.removeRigidBody(box.node())
            box.removeNode()
        

    def onMouseClick(self):
        print("Mouse clicked")
        if self.hitbox:
            print(f"Choice selected: {self.hitbox.getName()}")
            self.cleanup()
            
    def mouseActivate(self,task):
        #print("Choice activated")
        if base.mouseWatcherNode.hasMouse():
            # Get mouse position in normalized device coordinates
            pMouse = base.mouseWatcherNode.getMouse()
            pFrom = Point3()
            pTo = Point3()
            base.camLens.extrude(pMouse, pFrom, pTo)
            
            # Transform to global coordinates
            pFrom = render.getRelativePoint(base.cam, pFrom)
            pTo = render.getRelativePoint(base.cam, pTo)

            # Perform ray test
            result = base.world.rayTestClosest(pFrom, pTo, BitMask32.bit(31))
            
            if result.hasHit():
                hit_node = result.getNode()
                self.hitbox=hit_node
                #print(f"Choice selected: {hit_node.getName()}")
            else:
                self.hitbox=None
        return task.cont
    
    def create_bullet_rigidbody_cube(self, world, location=Vec3(0, 0, 0), size=1.0, name="BulletCube"):
        """Create a cube with Panda3D bullet physics rigidbody properties"""
        
        # Create cube geometry
        cube_geom = CardMaker(name)
        cube_geom.setFrame(-size/2, size/2, -size/2, size/2)
        cube_node = cube_geom.generate()
        
        # Create visual model
        cube_model = render.attachNewNode(cube_node)
        #cube_model.setPos(location[0], location[1], location[2])
        cube_model.setScale(size)
        
        # Create bullet collision shape
        shape = BulletBoxShape(Vec3(size/2, size/2, size/2))
        
        # Create rigidbody
        rigidbody = BulletRigidBodyNode(name)
        rigidbody.setMass(0)
        rigidbody.addShape(shape)
        rigidbody.setIntoCollideMask(BitMask32.bit(31))  # Set collide mask
        
        # Configure bullet-like properties
        #rigidbody.setFriction(0.2)
        #rigidbody.setRestitution(0.8)
        
        # Attach to scene
        rigidbody_np = render.attachNewNode(rigidbody)
        rigidbody_np.setPos(location[0], location[1], location[2])
        
        # Add to bullet world
        base.world.attachRigidBody(rigidbody)

        cube_model.reparentTo(rigidbody_np)
        
        return rigidbody_np