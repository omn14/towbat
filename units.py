from models import *
from direct.fsm.FSM import FSM
from panda3d.bullet import BulletBoxShape, BulletRigidBodyNode
from panda3d.core import Point3, TextNode, BitMask32

class unit:
    def __init__(self, name: str, model: model, nmodels: int, files: int, ranks: int):
        self.name = name
        self.nmodels = nmodels
        self.model = model
        self.files = files
        self.ranks = ranks

class unitGraphics(FSM):
    def __init__(self,Game, name, modelpath, unit=None, scale=1.0,BulletWorld=None,color=(1,0,0,1),bitmask=1):
        FSM.__init__(self, 'unitFSM')
        self.game=Game
        self.unitName=name
        self.unit =unit
        self.world=BulletWorld
        self.color=color
        self.bitmask=bitmask
        print(f"Creating unit graphics for {self.unitName} with model {modelpath}")
        self.model = loader.loadModel(modelpath)
        print(f"Model {modelpath} loaded for unit {self.unitName}")
        self.model.setScale(scale)
        self.model.setColor(self.color)
        self.model.reparentTo(render)
        self.unitWidth=abs(self.model.getTightBounds()[1][0]-self.model.getTightBounds()[0][0])
        self.unitHeight=abs(self.model.getTightBounds()[1][1]-self.model.getTightBounds()[0][1])
        print(f"Unit Width: {self.unitWidth}, Unit Height: {self.unitHeight}")
        #self.model.ls()
        children = self.model.getChildren()
        #children.sort(key=lambda np: np.getName())
        #children[-1].removeNode()
        ranks=self.unit.ranks
        files=self.unit.files
        if self.unit.nmodels!=len(children):
            diffnmodel=self.unit.nmodels-len(children)
            for i in range(diffnmodel):
                clone=children[0].copyTo(self.model)
                children.append(clone)
        
        while len(children)>self.unit.nmodels:
            children[-1].removeNode()
            children = self.model.getChildren()

        self.modelWidth=abs(children[0].getTightBounds()[1][0]-children[0].getTightBounds()[0][0])
        self.modelHeight=abs(children[0].getTightBounds()[1][1]-children[0].getTightBounds()[0][1])
        print(f"Model Width: {self.modelWidth}, Model Height: {self.modelHeight}")

        self.request('Idle')

        for i, child in enumerate(children):
            row = i // files
            col = i % files
            print(f"Positioning child {child.getName()} at row {row}, col {col}")
            p=Point3(col * (self.modelWidth ),-row * (self.modelHeight ), 0)
            pp=p-Point3(self.unitWidth*2, -self.modelHeight/2,0)
            child.setPos(p)
            #child.setPos((col - (files - 1) / 2) * (self.modelWidth / files), (row - (ranks - 1) / 2) * (self.modelHeight / ranks), 0)

        #self.unitWidth=abs(self.model.getTightBounds()[1][0]-self.model.getTightBounds()[0][0])
        #self.unitHeight=abs(self.model.getTightBounds()[1][1]-self.model.getTightBounds()[0][1])

        #self.model.setPos(self.unitWidth*2, -self.modelHeight/2,0)


        #self.model.setPos(35,0,0)
        self.setUpCollisions()

        #children[-1].removeNode()

        self.isInCombat=False
        self.isInCombatWith=[]
        self.isInCombatFlank=[]
        self.hasMovedThisTurn=False
        self.hasAttackedThisTurn=False
        self.attemptedRallyThisTurn=False
        self.endedInUnit=False
        self.madePursuitChoice=False
        text=f"{self.isInCombat}\n{self.hasMovedThisTurn}\n{self.hasAttackedThisTurn}"
        
        """ self.text_node = OnscreenText(
            text=text,
            scale=scale,
            fg=color,
            align=0,  # Center alignment
            mayChange=True
        ) """
        self.text = TextNode('node name')
        self.text.setText(text)
        #self.text_node = self.model.attachNewNode(self.text)
        self.text_node = self.bodyNP.attachNewNode(self.text)
        self.text_node.setPos(self.unitWidth/3, self.unitHeight, 5)
        self.text_node.setScale(0.1)
        self.text_node.setBillboardPointEye(-5, fixed_depth=True)
        self.text_node.setBin("fixed", 0)
        self.text_node.setDepthWrite(False)
        self.text_node.setDepthTest(False)
        self.text_node.hide()

        
    def updateTextNode(self):
        text=f"In Combat: {self.isInCombat}\nMoved This Turn: {self.hasMovedThisTurn}\nAttacked This Turn: {self.hasAttackedThisTurn}"
        row = f"In Combat: {self.isInCombat}\n"
        row += f"Moved This Turn: {self.hasMovedThisTurn}\n"
        row += f"Attacked This Turn: {self.hasAttackedThisTurn}"
        row += f"\nEngaged With: {[unit.unitName for unit in self.isInCombatWith]}"
        row += f"\nFlanks: {self.isInCombatFlank}"
        row += f"\nState: {self.state}"
        self.text.setText(row)

    def setUpCollisions(self):
        if self.world:
            # Estimate radius from bounding box
            #self.model.clearBounds()
            #self.model.calcTightBounds(render)
            print(f"Setting up collisions for unit: {self.unitName}")
            bounds = self.model.getTightBounds()
            print(f"Unit {self.unitName} bounding box: {bounds}")
            
            

            # Create box shape from bounding box dimensions
            box_size = bounds[1] - bounds[0]
            shape = BulletBoxShape(box_size * 0.5)  # BulletBoxShape takes half-extents
            body = BulletRigidBodyNode('UnitCollision-' + self.unitName)
            body.addShape(shape)
            body.setMass(0)  # Static object
            #body = BulletCharacterControllerNode(shape, 0.4, 'UnitCollision-' + self.unitName)
            
            
            self.bodyNP = render.attachNewNode(body)
            self.bodyNPfront = self.bodyNP.attachNewNode("front")
            self.bodyNPfront.setPos(0, box_size.y * 0.45, 0)  # Front point
            self.bodyNPback = self.bodyNP.attachNewNode("back")
            self.bodyNPback.setPos(0, -box_size.y * 0.45, 0)  # Back point
            self.bodyNP.setCollideMask(BitMask32.bit(1))
            self.world.attachRigidBody(body)
            #self.world.attachCharacter(self.bodyNP.node())

            self.model.node().setName('Model-' + self.unitName)
            self.model.reparentTo(self.bodyNP)
            self.bodyNP.setScale(1.0)
            #self.unitWidth=abs(self.model.getTightBounds()[1][0]-self.model.getTightBounds()[0][0])*self.bodyNP.getScale().x
            #self.unitHeight=abs(self.model.getTightBounds()[1][1]-self.model.getTightBounds()[0][1])*self.bodyNP.getScale().y
            self.unitWidth=box_size.x*self.bodyNP.getScale().x
            self.unitHeight=box_size.y*self.bodyNP.getScale().y
            self.model.setPos(-box_size.x/2+self.modelWidth/2, box_size.y/2-self.modelHeight/2,0)
            #self.model.flattenLight()
    
    def enterIdle(self):
        print(f"{self.unitName} is idle! state")
        self.hasMovedThisTurn=False
        taskMgr.doMethodLater(0.1, self.updateTextNode, "updateTextNode",extraArgs=[],appendTask=False)

    def enterMoved(self):
        print(f"{self.unitName} is in moved state!")
        self.hasMovedThisTurn=True
        messenger.send('unit-move-complete')
        taskMgr.doMethodLater(0.1, self.updateTextNode, "updateTextNode",extraArgs=[],appendTask=False)

    def enterInCombat(self):
        print(f"{self.unitName} is in combat state!")
        self.isInCombat=True
        messenger.send('unit-move-complete')
        taskMgr.doMethodLater(0.1, self.updateTextNode, "updateTextNode",extraArgs=[], appendTask=False)
    
    def exitInCombat(self):
        print(f"{self.unitName} is exiting combat state!")
        self.isInCombat=False
        self.isInCombatWith=[]
        self.isInCombatFlank=[]
        taskMgr.doMethodLater(0.1, self.updateTextNode, "updateTextNode",extraArgs=[], appendTask=False)

    def enterIsFleeing(self):
        print(f"{self.unitName} is in fleeing state!")
        self.attemptedRallyThisTurn=False
        taskMgr.doMethodLater(0.1, self.updateTextNode, "updateTextNode",extraArgs=[], appendTask=False)
        pass

    def enterIsPursuing(self):
        print(f"{self.unitName} is in pursuing state!")
        taskMgr.doMethodLater(0.1, self.updateTextNode, "updateTextNode",extraArgs=[], appendTask=False)
        pass
