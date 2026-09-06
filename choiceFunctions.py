from panda3d.core import (Vec3, CardMaker, Point3, BitMask32, TransparencyAttrib,
                         TextNode)
from panda3d.bullet import BulletRigidBodyNode, BulletBoxShape
from direct.showbase.DirectObject import DirectObject
from direct.interval.IntervalGlobal import Parallel, Sequence, LerpPosInterval

import gui_theme
from collision_masks import CollisionMask as CM


def _plaque(parent, size, texture, colour):
    """A billboarded card standing in for a menu panel.

    Lit off on purpose: these are interface, not scenery, and the board's own
    lighting had them reading as grey slate in the shade.
    """
    card = CardMaker('plaque')
    card.setFrame(-size / 2.0, size / 2.0, -size / 2.0, size / 2.0)
    np = parent.attachNewNode(card.generate())
    tex = loader.loadTexture(texture) if texture else None
    if tex is not None:
        np.setTexture(tex)
    np.setColor(*colour)
    np.setTransparency(TransparencyAttrib.MAlpha)
    np.setLightOff(1)
    # These are interface: what is on top is decided by the render bin, not by
    # depth. Left to depth, a panel drew over its own lettering, which is
    # coplanar with it, and left the words a ghost of themselves.
    np.setDepthWrite(False)
    np.setBillboardPointEye()
    return np


def _inscription(parent, text, scale, colour, wordwrap=None):
    """Text in the medieval hand, unlit and drawn over its panel."""
    node = TextNode('inscription')
    node.setText(text)
    node.setAlign(TextNode.ACenter)
    node.setTextColor(*colour)
    node.setShadow(0.05, 0.05)
    node.setShadowColor(*gui_theme.SHADOW)
    font = gui_theme.get_font()
    if font is not None:
        node.setFont(font)
    if wordwrap:
        node.setWordwrap(wordwrap)
    np = parent.attachNewNode(node)
    np.setScale(scale)
    np.setLightOff(1)
    np.setDepthWrite(False)
    np.setDepthTest(False)
    return np


class Choice:
    # Board units between box centres, and how many will fit across the screen
    # before the row wraps. A long row used to run off to the right and the
    # player had to zoom out to see the choices they had.
    SPACING = 16
    MAX_PER_ROW = 5
    BOX_SIZE = 8.0

    def __init__(self, choices, pos, cancellable=False, descriptions=None,
                 prompt=None):
        self.num_choices = len(choices)
        self.choices = choices
        self.choiceMade = False
        self.choice = None
        self.hitbox = None
        self.boxes = []
        self.plaques = {}
        self.hovered = None
        self.prompt = None
        # Hovering a choice shows its blurb; a spell is unplayable if you
        # cannot read what it does before committing to it.
        self.descriptions = descriptions or {}
        self.detail = None
        self.shown = None
        if self.descriptions:
            self.detail = gui_theme.styled_text(
                text="", pos=(0.0, -0.55), scale=0.045,
                fg=gui_theme.CREAM, align=TextNode.ACenter)
        names = list(self.choices)
        rows = -(-len(names) // self.MAX_PER_ROW) or 1
        for i, c in enumerate(names):
            row, col = divmod(i, self.MAX_PER_ROW)
            across = min(self.MAX_PER_ROW, len(names) - row * self.MAX_PER_ROW)
            loc = pos + Vec3((col - (across - 1) / 2.0) * self.SPACING,
                             -row * self.SPACING, 20)
            box = self.create_bullet_rigidbody_cube(
                None, location=loc, size=self.BOX_SIZE, name=c)
            self.boxes.append(box)
        if prompt:
            self.prompt = self._makePrompt(
                prompt, pos + Vec3(0, self.SPACING * 0.5, 20))
        #self.ma = taskMgr.add(self.mouseActivate, "mouseActivateTask")
        self.helper1 = DirectObject()
        self.helper1.accept('mouse1', self.onMouseClick)
        if cancellable:
            self.helper1.accept('mouse3', self.onCancel)
        #self.old = messenger.whoAccepts('mouse1')
        #base.accept("mouse1", self.onMouseClick)

    def _makePrompt(self, text, location):
        """The question on a banner above the boxes.

        Dark ground and pale lettering rather than the parchment used
        elsewhere: the banner floats over whatever the board happens to show,
        so it cannot rely on what is behind it.
        """
        np = render.attachNewNode('choicePrompt')
        np.setPos(location)
        label = _inscription(np, text, 2.4, gui_theme.GOLD, wordwrap=30)
        label.setBillboardPointEye()
        # Measured rather than guessed from the character count, which left a
        # banner half as wide again as the words on it.
        lo, hi = label.getTightBounds()
        banner = _plaque(np, 1.0, None, gui_theme.DARK_BG)
        banner.setScale(hi.getX() - lo.getX() + 5.0, 1.0,
                        hi.getZ() - lo.getZ() + 3.0)
        banner.setZ((hi.getZ() + lo.getZ()) / 2.0)
        # Both are transparent, so depth testing will not separate them: the
        # banner drew over its own lettering and left it a ghost of itself.
        banner.setBin('fixed', 10)
        label.setBin('fixed', 20)
        return np

    def _setHovered(self, name):
        """Light the panel under the pointer, so a choice can be aimed at."""
        if name == self.hovered:
            return
        for key, lit in ((self.hovered, False), (name, True)):
            plaque = self.plaques.get(key)
            if key is None or plaque is None or plaque.isEmpty():
                continue
            plaque.setColor(*(gui_theme.GOLD if lit else (1, 1, 1, 1)))
            plaque.setScale(self.BOX_SIZE * (1.87 if lit else 1.7), 1.0,
                            self.BOX_SIZE * (0.99 if lit else 0.9))
        self.hovered = name

    async def cleanup(self):
        #taskMgr.remove("mouseActivateTask")
        self.choiceMade = True
        self.helper1.ignore('mouse1')
        self.helper1.ignore('mouse3')
        if self.detail is not None:
            self.detail.destroy()
            self.detail = None
        if self.prompt is not None:
            self.prompt.removeNode()
            self.prompt = None
        
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
            self._setHovered(self.hitbox.getName() if self.hitbox else None)
            if self.choiceMade:
                return task.done    
        return task.cont

    def _showDetail(self, name):
        if self.detail is None or name == self.shown:
            return
        self.shown = name
        self.detail.setText(self.descriptions.get(name, ""))
    
    def create_bullet_rigidbody_cube(self, world, location=Vec3(0, 0, 0), size=1.0, name="BulletCube"):
        """A clickable menu panel: a Bullet box to pick, a plaque to read."""
        
        # Create cube geometry
        cube_geom = CardMaker(name)
        cube_geom.setFrame(-0.5, 0.5, -0.5, 0.5)
        cube_node = cube_geom.generate()
        
        # Create visual model
        cube_model = render.attachNewNode(cube_node)
        #cube_model.setPos(location[0], location[1], location[2])
        cube_model.setPos(0, 0, 0)
        # Wider than tall, so a two-word label sits on the plaque rather than
        # hanging off both ends of it. The click box matches.
        cube_model.setScale(size * 1.7, 1.0, size * 0.9)
        # Lit off and turned to face the camera: these are interface, not
        # scenery, and the board's own lighting had them reading as grey slate.
        cube_model.setTexture(loader.loadTexture(gui_theme.TEX_PARCHMENT))
        cube_model.setColor(1, 1, 1, 1)
        cube_model.setTransparency(TransparencyAttrib.MAlpha)
        cube_model.setLightOff(1)
        cube_model.setBillboardPointEye()
        cube_model.setBin('fixed', 10)
        self.plaques[name] = cube_model
        
        # Create bullet collision shape
        shape = BulletBoxShape(Vec3(size*0.85, size/2, size*0.45))
        
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
        # Wrapped to the panel's own width: a label longer than its plaque used
        # to run across its neighbour's and the two read as one line of nonsense.
        textNodePath = _inscription(rigidbody_np, name, size * 0.24,
                                    gui_theme.INK, wordwrap=6)
        textNodePath.setPos(0, -0.2, size * 0.06)
        textNodePath.setBillboardPointEye()
        textNodePath.setBin('fixed', 20)
        
        # Add to bullet world
        base.world.attachRigidBody(rigidbody)

        cube_model.reparentTo(rigidbody_np)
        
        return rigidbody_np