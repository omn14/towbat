from panda3d.core import Vec3, CardMaker, Point3, BitMask32
from panda3d.bullet import BulletRigidBodyNode, BulletBoxShape
from direct.showbase.DirectObject import DirectObject
from panda3d.core import TextNode
from direct.interval.IntervalGlobal import Parallel, Sequence, LerpPosInterval

import gui_theme
from collision_masks import CollisionMask as CM

class Choice:
    def __init__(self, choices, pos, cancellable=False, descriptions=None):
        self.num_choices = len(choices)
        self.choices = choices
        self.choiceMade = False
        self.choice = None
        self.hitbox = None
        self.boxes = []
        # Hovering a choice shows its blurb; a spell is unplayable if you
        # cannot read what it does before committing to it.
        self.descriptions = descriptions or {}
        self.detail = None
        self.shown = None
        if self.descriptions:
            self.detail = gui_theme.styled_text(
                text="", pos=(0.0, -0.55), scale=0.045,
                fg=gui_theme.CREAM, align=TextNode.ACenter)
        for i, c in enumerate(self.choices):
            loc=pos+Vec3(i*16,0,20)
            box=self.create_bullet_rigidbody_cube(None, location=loc, size=8.0, name=c)
            self.boxes.append(box)
        #self.ma = taskMgr.add(self.mouseActivate, "mouseActivateTask")
        self.helper1 = DirectObject()
        self.helper1.accept('mouse1', self.onMouseClick)
        if cancellable:
            self.helper1.accept('mouse3', self.onCancel)
        #self.old = messenger.whoAccepts('mouse1')
        #base.accept("mouse1", self.onMouseClick)

    async def cleanup(self):
        #taskMgr.remove("mouseActivateTask")
        self.choiceMade = True
        self.helper1.ignore('mouse1')
        self.helper1.ignore('mouse3')
        if self.detail is not None:
            self.detail.destroy()
            self.detail = None
        
        for box in self.boxes:
            if box.isEmpty():
                continue
            if self.hitbox and box.node() == self.hitbox:
                moveInterval = LerpPosInterval(box, 1.0, box.getPos()+Vec3(0,0,20))
                await moveInterval
            if not box.isEmpty():
                base.world.removeRigidBody(box.node())
                box.removeNode()
        

    def onMouseClick(self):
        print("Mouse clicked in choice function")
        if self.hitbox:
            print(f"Choice selected: {self.hitbox.getName()}")
            self.choice = self.hitbox.getName()
            #base.messenger.send('choice-made', [self.hitbox.getName()])
            taskMgr.add(self.cleanup())
    def onCancel(self):
        """Right-click closes the menu without choosing; the caller sees None."""
        print("Choice cancelled")
        self.choice = None
        self.hitbox = None
        taskMgr.add(self.cleanup())

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
            # The menu cubes have their own bit: the board-edge ghosts are all
            # named 'Ghost', and picking one would be read as a choice.
            result = base.world.rayTestClosest(pFrom, pTo, CM.MENU_CHOICE)
            
            if result.hasHit():
                hit_node = result.getNode()
                self.hitbox=hit_node
                #print(f"Choice selected: {hit_node.getName()}")
            else:
                self.hitbox=None
            self._showDetail(self.hitbox.getName() if self.hitbox else None)
            if self.choiceMade:
                return task.done    
        return task.cont

    def _showDetail(self, name):
        if self.detail is None or name == self.shown:
            return
        self.shown = name
        self.detail.setText(self.descriptions.get(name, ""))
    
    def create_bullet_rigidbody_cube(self, world, location=Vec3(0, 0, 0), size=1.0, name="BulletCube"):
        """Create a cube with Panda3D bullet physics rigidbody properties"""
        
        # Create cube geometry
        cube_geom = CardMaker(name)
        #cube_geom.setFrame(-size/2, size/2, -size/2, size/2)
        cube_node = cube_geom.generate()
        
        # Create visual model
        cube_model = render.attachNewNode(cube_node)
        #cube_model.setPos(location[0], location[1], location[2])
        cube_model.setPos(-Vec3(size/2, size/2, size/2))
        cube_model.setScale(size)
        
        # Create bullet collision shape
        shape = BulletBoxShape(Vec3(size/2, size/2, size/2))
        
        # Create rigidbody
        rigidbody = BulletRigidBodyNode(name)
        rigidbody.setMass(0)
        rigidbody.addShape(shape)
        rigidbody.setIntoCollideMask(CM.MENU_CHOICE)  # Set collide mask
        
        # Configure bullet-like properties
        #rigidbody.setFriction(0.2)
        #rigidbody.setRestitution(0.8)
        
        # Attach to scene
        rigidbody_np = render.attachNewNode(rigidbody)
        rigidbody_np.setPos(location[0], location[1], location[2])
        
        # Create and attach text node
        text = TextNode('text')
        text.setText(name)
        text.setAlign(TextNode.ACenter)
        # Wrapped to the spacing between boxes, in text units before the scale
        # below: a label longer than its own box used to run across its
        # neighbour's and the two read as one line of nonsense.
        text.setWordwrap(7)
        textNodePath = rigidbody_np.attachNewNode(text)
        textNodePath.setScale(2)
        textNodePath.setPos(0, 0, size)  # Position above the rigid body
        textNodePath.setBillboardPointEye()  # Make it face the camera
        
        # Add to bullet world
        base.world.attachRigidBody(rigidbody)

        cube_model.reparentTo(rigidbody_np)
        
        return rigidbody_np