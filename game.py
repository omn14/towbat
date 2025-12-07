from direct.showbase.ShowBase import ShowBase
from panda3d.core import Plane, PlaneNode, Point3, Vec2, Vec3, Vec4, BitMask32
from panda3d.core import CardMaker
from panda3d.bullet import BulletWorld, BulletPlaneShape, BulletRigidBodyNode, BulletTriangleMesh, BulletTriangleMeshShape, BulletBoxShape
from direct.interval.LerpInterval import LerpPosInterval, LerpPosHprInterval
from direct.interval.IntervalGlobal import Sequence, ProjectileInterval
from direct.interval.FunctionInterval import Func
from panda3d.core import Shader

from shaders.chargedistshaders import *
from panda3d.core import Texture
from panda3d.core import DirectionalLight, AmbientLight
from panda3d.core import MeshDrawer, NodePath
from panda3d.core import TextNode
import math
from panda3d.bullet import BulletSphereShape, BulletRigidBodyNode
from panda3d.bullet import BulletDebugNode
from direct.directutil import Mopath
from direct.interval.MopathInterval import MopathInterval
from panda3d.core import NurbsCurveEvaluator, NurbsCurveResult
from panda3d.core import NurbsCurve
from panda3d.core import GraphicsPipe


from panda3d.bullet import BulletCharacterControllerNode
from panda3d.bullet import BulletCapsuleShape
from panda3d.bullet import ZUp
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import LQuaterniond, LVector3d
from direct.fsm.FSM import FSM
from panda3d.bullet import BulletRigidBodyNode, BulletBoxShape
from panda3d.core import Vec3, BitMask32
from panda3d.core import LineSegs
#from panda3d.core import AsyncFuture

from models import *
from units import *
from toHitAndToWound import *
from battleFunctions import *

from dice import *
from choiceFunctions import *
from ClassOutOfBounds import *

#import charge_impact_effect
from direct.particles.ParticleEffect import ParticleEffect
from direct.interval.IntervalGlobal import Parallel
from panda3d.core import GraphicsOutput, Camera, OrthographicLens, RenderState
from panda3d.core import Texture, FrameBufferProperties, WindowProperties
from panda3d.core import CardMaker, TransparencyAttrib
from panda3d.core import RenderState, TextureStage

class gameFSM(FSM):
    def __init__(self, Game):
        FSM.__init__(self, 'GameFSM')
        self.game = Game
        
        self.endPhaseCube = self.createMenuCollisionCube("endPhase",Point3(5,20,3))
        
        self.phases = ['StrategyPhase', 'MovementPhase', 'ShootingPhase', 'CombatPhase']
        self.currentPhaseIndex=0

        self.game.debugText.setText(f"Current phase: {self.phases[self.currentPhaseIndex]}")
        self.request(self.phases[self.currentPhaseIndex])

        self.menuCubes= base.camera.findAllMatches("**/*MenuCube")
        print(self.menuCubes)


        self.accept('mouse1', self.mouseMenuCollide)

    def createMenuCollisionCube(self, name='StrategyPhase',pos=Point3(5, 20, 0)):
        # Create a Bullet collision cube (visible + physics) for debugging

        half_extents = Vec3(1, 1, 1)  # Half sizes (so cube is 2x2x2)
        shape = BulletBoxShape(half_extents)
        cube_node = BulletRigidBodyNode(name)
        cube_node.setMass(0)  # 0 = static; set >0 to make it dynamic
        cube_node.addShape(shape)

        cubeNP = base.camera.attachNewNode(cube_node)
        cubeNP.setPos(pos)  # Raise slightly above ground
        cubeNP.setCollideMask(BitMask32.bit(2))  # Set collision mask
        cubeNP.setName(name)

        # Attach to existing Bullet world (created in MyApp.setup_bullet)
        base.world.attachRigidBody(cube_node)

        # Optional visible geometry
        try:
            model = loader.loadModel('models/box')
            model.reparentTo(cubeNP)
            model.setScale(2)  # Box model is unit-sized; scale to match 2x2x2
            model.setPos(-1,-1,-1)
        except Exception:
            pass
        return cubeNP

    def mouseMenuCollide(self):
        
        if base.mouseWatcherNode.hasMouse():
            x = base.mouseWatcherNode.getMouseX()
            y = base.mouseWatcherNode.getMouseY()
            #print(x,y)
            #surface.set_shader_input("pos", Vec3(base.mouseWatcherNode.getMouseX(),0,base.mouseWatcherNode.getMouseY())*4)
            #pFrom = Point3(0, 0, 0)
            #pTo = Point3(10, 0, 0)

            # Get to and from pos in camera coordinates
            pMouse = base.mouseWatcherNode.getMouse()
            pFrom = Point3()
            pTo = Point3()
            base.camLens.extrude(pMouse, pFrom, pTo)

            # Transform to global coordinates
            pFrom = render.getRelativePoint(base.cam, pFrom)
            pTo = render.getRelativePoint(base.cam, pTo)

            result = base.world.rayTestClosest(pFrom, pTo,BitMask32.bit(2))

            if result.hasHit():
                """ print(result.hasHit())
                print(result.getHitPos())
                print(result.getHitNormal())
                print(result.getHitFraction())
                print(result.getNode()) """
                
                self.currentPhaseIndex = (self.currentPhaseIndex + 1) % len(self.phases)
                #self.game.debugText.setText(f"Current phase: {self.phases[self.currentPhaseIndex]}")
                self.request(self.phases[self.currentPhaseIndex])

    def enterStrategyPhase(self):
        self.game.debugText.setText(f"Current phase: {self.phases[self.currentPhaseIndex]}")
        print("Entering Strategy Phase")
        

    def exitStrategyPhase(self):
        print("Exiting Strategy Phase")
        self.game.ignore('mouse1')
        

    def enterMovementPhase(self):
        print("Entering Movement Phase")
        self.game.debugText.setText(f"Current phase: {self.phases[self.currentPhaseIndex]}")
        self.game.accept('mouse1', self.game.setActiveUnit,[self.game.taskLoopPathTowardsMouse, "taskLoopPathTowardsMouse"])
        
        

    def exitMovementPhase(self):
        print("Exiting Movement Phase")
        taskMgr.remove("taskLoopPathTowardsMouse")
        self.cleanup()
        self.game.ignore('mouse1')
        self.game.ground.setShaderInput("isActive", False)
        self.game.boundries.contactTest(self.game.boundries.northBoundry,180,Vec3(0,-0.1,0))
        self.game.boundries.contactTest(self.game.boundries.southBoundry,0,Vec3(0,0.1,0))
        self.game.boundries.contactTest(self.game.boundries.westBoundry,270,Vec3(0.1,0,0))
        self.game.boundries.contactTest(self.game.boundries.eastBoundry,90,Vec3(-0.1,0,0))

    def enterShootingPhase(self):
        print("Entering Shooting Phase")
        self.game.debugText.setText(f"Current phase: {self.phases[self.currentPhaseIndex]}")
        
        
        self.game.accept('mouse1', self.game.setActiveUnit,[self.game.taskShootingArcUpdate, "taskShootingArcUpdate"])

        

    def exitShootingPhase(self):
        print("Exiting Shooting Phase")
        self.game.ignore('mouse1')
        self.cleanup()
        self.game.ground.setShaderInput("isActive", False)
        taskMgr.remove("taskShootingTrajectoryDrawLine")
        

    def enterCombatPhase(self):
        print("Entering Combat Phase")
        self.game.debugText.setText(f"Current phase: {self.phases[self.currentPhaseIndex]}")
        self.game.accept('mouse1', self.game.setActiveUnit,[self.game.taskStartCombat, "taskStartCombat"])


    def exitCombatPhase(self):
        print("Exiting Combat Phase")
        self.game.ignore('mouse1')

    def enterMakeChoice(self):
        print("Entering Make Choice Phase")
        self.game.debugText.setText(f"Current phase: MakeChoice")
        self.game.accept('mouse1', self.game.makeChoiceSelection)
    
    def exitMakeChoice(self):
        print("Exiting Make Choice Phase")
        self.game.ignore('mouse1')

    def cleanup(self):
        
        for unit in self.game.units:
            unit.model.setColor(unit.color)
            unit.bodyNP.setCollideMask(BitMask32.bit(unit.bitmask))
            unit.hasMovedThisTurn=False
            unit.updateTextNode()



class unitGraphics():
    def __init__(self,name, modelpath, unit=None, scale=1.0,BulletWorld=None,color=(1,0,0,1),bitmask=1):
        self.unitName=name
        self.unit =unit
        self.world=BulletWorld
        self.color=color
        self.bitmask=bitmask
        self.model = loader.loadModel(modelpath)
        self.model.setScale(scale)
        self.model.setColor(self.color)
        self.model.reparentTo(render)
        self.unitWidth=abs(self.model.getTightBounds()[1][0]-self.model.getTightBounds()[0][0])
        self.unitHeight=abs(self.model.getTightBounds()[1][1]-self.model.getTightBounds()[0][1])
        print(f"Unit Width: {self.unitWidth}, Unit Height: {self.unitHeight}")
        self.model.ls()
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
        self.modelWidth=abs(children[0].getTightBounds()[1][0]-children[0].getTightBounds()[0][0])
        self.modelHeight=abs(children[0].getTightBounds()[1][1]-children[0].getTightBounds()[0][1])
        print(f"Model Width: {self.modelWidth}, Model Height: {self.modelHeight}")

        for i, child in enumerate(children):
            row = i // files
            col = i % files
            print(f"Positioning child {child.getName()} at row {row}, col {col}")
            p=Point3(col * (self.modelWidth ),-row * (self.modelHeight ), 0)
            pp=p-Point3(self.unitWidth*2, -self.modelHeight/2,0)
            child.setPos(p)
            #child.setPos((col - (files - 1) / 2) * (self.modelWidth / files), (row - (ranks - 1) / 2) * (self.modelHeight / ranks), 0)

        self.unitWidth=abs(self.model.getTightBounds()[1][0]-self.model.getTightBounds()[0][0])
        self.unitHeight=abs(self.model.getTightBounds()[1][1]-self.model.getTightBounds()[0][1])

        #self.model.setPos(self.unitWidth*2, -self.modelHeight/2,0)


        #self.model.setPos(35,0,0)
        self.setUpCollisions()

        #children[-1].removeNode()

        self.isInCombat=False
        self.isInCombatWith=[]
        self.isInCombatFlank=[]
        self.hasMovedThisTurn=False
        self.hasAttackedThisTurn=False
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
        self.text.setText(row)

    def setUpCollisions(self):
        if self.world:
            # Estimate radius from bounding box
            #self.model.clearBounds()
            #self.model.calcTightBounds(render)
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
            self.unitWidth=abs(self.model.getTightBounds()[1][0]-self.model.getTightBounds()[0][0])*self.bodyNP.getScale().x
            self.unitHeight=abs(self.model.getTightBounds()[1][1]-self.model.getTightBounds()[0][1])*self.bodyNP.getScale().y
            self.model.setPos(-box_size.x/2+self.modelWidth/2, box_size.y/2-self.modelHeight/2,0)
            #self.model.flattenLight()





class MyApp(ShowBase):
    def __init__(self):
        super().__init__()

        # Disable default camera controls
        #self.disableMouse()
        base.enableParticles()
        

        # Create a flat plane using CardMaker
        cm = CardMaker("ground")
        cm.setFrame(-50, 50, -50, 50)  # 200x200 plane
        self.ground = self.render.attachNewNode(cm.generate())
        self.ground.setPos(0, 0, 0)
        self.ground.setHpr(0, -90, 0)
        self.ground.setColor(0, 1, 0, 1)  # Set plane color to green (RGBA)
        tex = self.loader.loadTexture('maps/noise.rgb')
        #tex = self.loader.loadTexture('maps/panda_head.rgb')
        self.ground.setTexture(tex)
        self.groundSizeboundingbox=self.ground.getTightBounds()

        # Directional light
        dlight = DirectionalLight('dlight')
        dlight.setColor((0.8, 0.8, 0.7, 1))
        dlnp = self.render.attachNewNode(dlight)
        dlnp.setHpr(-45, -60, 0)
        self.render.setLight(dlnp)

        # Ambient light
        alight = AmbientLight('alight')
        alight.setColor((0.2, 0.2, 0.3, 1))
        alnp = self.render.attachNewNode(alight)
        self.render.setLight(alnp)

              

        #lol
        self.arcPoint=Vec2(0.55,0.55)
        self.arcPointRotation=0

        

        # Make a copy of the smiley model and position it differently
        self.smiley_copy = self.loader.loadModel('models/smiley')
        self.smiley_copy.reparentTo(self.render)
        self.smiley_copy.setPos(-50, 0, 0)
        self.smiley_copy.setScale(2)

        # Position the camera above the plane, looking straight down
        self.disableMouse()
        self.camera.setPos(0, -75, 150)
        self.camera.lookAt(self.ground)
        #self.enableMouse()
        #self.camera.setP(-90)  # Pitch downwards
        self.setup_shader()
        self.setup_bullet()
        #self.accept('mouse1', self.upAndDown)
        #self.accept('mouse1', self.setActiveUnit,[self.taskLoopPathTowardsMouse, "taskLoopPathTowardsMouse"])
        self.accept('q-up', self.pathTowardsMouse)
        self.accept('w-up', self.startTaskFunction,[self.taskLoopPathTowardsMouse, "taskLoopPathTowardsMouse"])
        
        self.debugTextUnit = self.setup_text_node(text="Debug Info", pos=(-1.3, -0.9), scale=0.05, color=(1, 1, 0, 1))
        self.debugTextUnit.setText("Debug Info test")

        self.debugTextInfo = self.setup_text_node(text="Debug Info", pos=(0.7, -0.8), scale=0.05, color=(1, 1, 0, 1))
        self.moveArceDistance = 0
        self.debugTextInfo.setText("Debug Arch test")

        self.numsPoints=0
        self.unitHitPos=Point3(0,0,0)

        self.units = []
        url_man_at_arm = "https://www.newrecruit.eu/wiki/tow/warhammer-the-old-world/kingdom-of-bretonnia/3ddf-271a-aaec-73eb/man-at-arms"
        man_at_arm = model("Man_at_Arm", url_man_at_arm)
        man_at_arm.armor_save = 7
        man_at_arm_unit = unit("Man_at_Arm Unit", man_at_arm, 10,5,2)
        self.bretBowmen = unitGraphics('BretBowmen','models/bret_bowmen.bam',man_at_arm_unit, scale=1.0, BulletWorld=self.world, color=(1,0,0,1))
        self.bretBowmen.bodyNP.setPos(25,35,0)
        self.bretBowmen.bodyNP.setH(180)
        self.units.append(self.bretBowmen)


        url_night_goblin = "https://www.newrecruit.eu/wiki/tow/warhammer-the-old-world/orc-and-goblin-tribes/f241-11e2-3771-3b16/night-goblin"
        night_goblin = NightGoblin("Night Goblin", url_night_goblin)
        night_goblin.armor_save = 7 
        night_goblin_unit = unit("Night Goblin Unit", night_goblin, 30,10,3)
        self.goblins = unitGraphics('Goblins','models/goblin_archers.bam',night_goblin_unit, scale=1.0, BulletWorld=self.world, color=(0,1,0,1))
        self.goblins.bodyNP.setPos(0,-20,0)
        self.goblins.unit.model.equip_weapon('short bow')
        self.units.append(self.goblins)
        self.unitToMove=self.bretBowmen
        
        url_goblin_wolf_rider = "https://www.newrecruit.eu/wiki/warhammer-armies-project/warhammer-armies-project/orcs-%26-goblins/9e93-cbcd-9787-baaa/goblin-wolf-rider"
        url_giant_wolf = "https://www.newrecruit.eu/wiki/warhammer-armies-project/warhammer-armies-project/orcs-%26-goblins/2b89-9731-8924-f606/giant-wolf"
        giant_wolf = GiantWolf("Giant Wolf", url_giant_wolf)
        giant_wolf_unit = unit("Giant Wolf Unit", giant_wolf, 15,5,3)
        goblin_wolf_rider = GoblinWolfRider("Goblin Wolf Rider", url_goblin_wolf_rider, mountUnit=giant_wolf_unit)
        goblin_wolf_rider_unit = unit("Goblin Wolf Rider Unit", goblin_wolf_rider, 15,5,3)
        self.goblinWolfRiders = unitGraphics('GoblinWolfRiders','models/goblin_wolfriders.bam',goblin_wolf_rider_unit, scale=1.0, BulletWorld=self.world, color=(0,1,0,1))
        self.goblinWolfRiders.bodyNP.setPos(-20,-30,0)
        #self.goblinWolfRiders.bodyNP.setH(90)
        self.units.append(self.goblinWolfRiders)
        
        
        self.accept('mouse3', self.moveUnit,[self.unitToMove])



        self.debugText = self.setup_text_node(text="Debug Info", pos=(-1.3, 0.9), scale=0.05, color=(1, 1, 0, 1))
        self.debugText.setText("Debug Info test")
        self.boundries = OutOfBounds()
        self.fsm = gameFSM(self)
        ###Shooting scenario testing
        if 0:
            self.fsm.currentPhaseIndex=2
            self.fsm.request(self.fsm.phases[self.fsm.currentPhaseIndex])
            self.goblins.bodyNP.setPos(0,0,0)
            #self.drawProjectileTrajectory(Point3(0,0,0), Point3(10,10,0))
            self.unitToMove=self.goblins

        if 0:
            self.fsm.currentPhaseIndex=1
            self.fsm.request(self.fsm.phases[self.fsm.currentPhaseIndex])
            self.goblins.bodyNP.setPos(0,-30,0)
            self.goblinWolfRiders.bodyNP.setPos(0,-40,0)
            #self.drawProjectileTrajectory(Point3(0,0,0), Point3(10,10,0))
            self.unitToMove=self.goblins
        if 0:
            self.fsm.currentPhaseIndex=1
            self.fsm.request(self.fsm.phases[self.fsm.currentPhaseIndex])
            self.goblins.bodyNP.setPos(0,-13,0)
            self.goblinWolfRiders.bodyNP.setPos(10,-15,0)
            self.bretBowmen.bodyNP.setPos(0,13,0)
            #self.drawProjectileTrajectory(Point3(0,0,0), Point3(10,10,0))
            self.unitToMove=self.goblins

        if 1:
            self.fsm.currentPhaseIndex=1
            self.fsm.request(self.fsm.phases[self.fsm.currentPhaseIndex])
            self.goblins.bodyNP.setPos(0,-3,0)
            self.goblinWolfRiders.bodyNP.setPos(10,5,0)
            self.goblinWolfRiders.bodyNP.setH(90)
            self.bretBowmen.bodyNP.setPos(0,5,0)
            #self.drawProjectileTrajectory(Point3(0,0,0), Point3(10,10,0))
            self.unitToMove=self.goblins

        if 0:
            self.fsm.currentPhaseIndex=3
            self.fsm.request(self.fsm.phases[self.fsm.currentPhaseIndex])
            self.goblins.bodyNP.setPos(0,-3,0)
            self.goblinWolfRiders.bodyNP.setPos(10,5,0)
            self.goblinWolfRiders.bodyNP.setH(90)
            self.bretBowmen.bodyNP.setPos(0,5,0)
            #self.drawProjectileTrajectory(Point3(0,0,0), Point3(10,10,0))
            self.unitToMove=self.goblins
            self.bretBowmen.isInCombatWith.append(self.goblins)
            self.bretBowmen.isInCombatWith.append(self.goblinWolfRiders)
            self.bretBowmen.isInCombatFlank.append('front')
            self.bretBowmen.isInCombatFlank.append('front')
            self.bretBowmen.isInCombat=True
            self.goblins.isInCombatWith.append(self.bretBowmen)
            self.goblins.isInCombat=True
            self.goblins.isInCombatFlank.append('front')
            self.goblinWolfRiders.isInCombatWith.append(self.bretBowmen)
            self.goblinWolfRiders.isInCombat=True
            self.goblinWolfRiders.isInCombatFlank.append('left')


        self.rectangleLine = self.drawRectangle(center=Point3(0, 0, 1), width=72, height=48, color=Vec4(1, 1, 0, 1))
        
        self.deploymentLine = self.drawRectangle(center=Point3(0, 0, .5), width=72, height=24, color=Vec4(1, 1, 1, 1))



        self.taskMgr.add(self.mouseHoverUnit, "mouseHoverUnit")
        #self.p = charge_impact_effect.ChargeImpactEffect(parent=render)
        self.p = ParticleEffect()
        self.p.loadConfig("particles/whburst2.ptf")

        self.p_miss = ParticleEffect()
        self.p_miss.loadConfig("particles/whmiss.ptf")

        # load the ball model
        self.ball = loader.loadModel("smiley")
        self.ball.reparentTo(render)
        self.ball.setPos(-15,0,0)

        # setup the projectile interval
        self.trajectory = ProjectileInterval(self.ball, duration=1,
                                            endPos=Point3(15,0, 0))
        
        self.mousePosOnGround=Point3(0,0,0)

        self.bakeTextures(self.ground)
    

    def bakeTextures(self, target_np, texture_size=512, name_suffix="_baked"):
        tex = Texture()
        tex.setMinfilter(Texture.FTLinear)
        tex.setMagfilter(Texture.FTLinear)

        win_props = WindowProperties.size(256, 256)
        fb_props = FrameBufferProperties()
        fb_props.setRgbColor(True)
        fb_props.setDepthBits(24)

        buffer = base.graphicsEngine.makeOutput(
            base.pipe, "rtBuffer", -2,
            fb_props, win_props,
            GraphicsPipe.BFRefuseWindow, base.win.getGsg(), base.win
        )

        buffer.addRenderTexture(tex, GraphicsOutput.RTMBindOrCopy)

        rt_scene = NodePath("rt-scene-root")
        # Create the same directional light as the main scene
        dlight = DirectionalLight('rt-dlight')
        dlight.setColor((0.8, 0.8, 0.7, 1))
        dlnp = rt_scene.attachNewNode(dlight)
        dlnp.setHpr(-45, -60, 0)
        rt_scene.setLight(dlnp)

        # Create the same ambient light as the main scene
        alight = AmbientLight('rt-alight')
        alight.setColor((0.2, 0.2, 0.3, 1))
        alnp = rt_scene.attachNewNode(alight)
        rt_scene.setLight(alnp)
        # reparent any models you want baked into the texture under rt_scene
        #model_copy = original_model.copyTo(rt_scene)
        model_copy = target_np.copyTo(rt_scene)

        rt_camera = base.makeCamera(buffer)
        rt_camera.reparentTo(rt_scene)
        #rt_camera.setPos(0, -10, 5)
        rt_camera.setPos(0, 0, 300)
        rt_camera.lookAt(model_copy)
        min_pt, max_pt = model_copy.getTightBounds()
        size = max_pt - min_pt
        width = size.x
        height = size.z   # or size.y depending on orientation

        # Get texture dimensions
        texture = model_copy.getTexture()
        if texture:
            width_pixels = texture.getXSize()
            height_pixels = texture.getYSize()
            print(texture)
        else:
            width_pixels = texture_size
            height_pixels = texture_size

        lens = OrthographicLens(); lens.setFilmSize(100, 100); rt_camera.node().setLens(lens)
        center = (min_pt + max_pt) * 0.5
        rt_camera.setPos(center.x, center.y , center.z + 10)
        rt_camera.lookAt(center)
        """ pivot = NodePath('bake-pivot')
        model_copy.reparentTo(pivot)
        model_copy.setPos(-center)  # center it at origin
        rt_camera.reparentTo(pivot)
        rt_camera.setPos(0, -10, 0)
        rt_camera.lookAt(0, 0, 0) """
        
        base.graphicsEngine.renderFrame()  # ensure at least one frame is rendered
        #tex.write(name_suffix + ".png")  # optional: save to disk for inspection

        # if you need the same shader, leave the shader/material assignments as-is

        #target_np.setTexture(tex, 1)         # stage 1 (or 0 if you prefer)
        target_np.setTransparency(TransparencyAttrib.MAlpha)  # if the shader outputs alpha
        # If you want to drive a custom shader input:
        target_np.setShaderInput("bakedMap", tex)
        
        return tex

    def applyBakedTexture(self, node, baked_texture):
        """
        Applies a baked texture to a node, replacing its current appearance.
        
        Args:
            node: The NodePath to apply the texture to
            baked_texture: The baked texture to apply
        """
        # Clear existing shaders and textures
        node.clearShader()
        node.clearTexture()
        
        # Apply the baked texture
        node.setTexture(baked_texture)
        
        # Set up simple texture rendering
        ts = TextureStage.getDefault()
        node.setTexture(ts, baked_texture)
        
        # Enable transparency if the baked texture has alpha
        if baked_texture.getNumComponents() == 4:
            node.setTransparency(TransparencyAttrib.MAlpha)

    def bakeAndApply(self, node, texture_size=512):
        """
        Convenience method to bake a node's appearance and immediately apply it.
        
        Args:
            node: The NodePath to bake and apply to
            texture_size: Size of the baked texture
            
        Returns:
            Texture: The baked texture that was applied
        """
        baked_texture = self.bakeTextures(node, texture_size)
        if baked_texture:
            self.applyBakedTexture(node, baked_texture)
        return baked_texture

    def drawProjectileTrajectory(self,startPos,endPos,n=20):
        # Remove existing trajectory line if it exists
        if hasattr(self, 'trajectoryLine'):
            self.trajectoryLine.removeNode()
        
        line_segs = LineSegs()
        line_segs.setColor(1, 0, 0, 1)  # Red color for the trajectory
        line_segs.setThickness(2.0)

        for i in range(n + 1):
            t = i / n
            # Simple linear interpolation for demonstration; replace with actual trajectory calculation
            x = (1 - t) * startPos.x + t * endPos.x
            y = (1 - t) * startPos.y + t * endPos.y
            z = (1 - t) * startPos.z + t * endPos.z + (4 * t * (1 - t)) * 10  # Adding a parabolic arc

            if i == 0:
                line_segs.moveTo(x, y, z)
            else:
                line_segs.drawTo(x, y, z)

        trajectory_node = line_segs.create()
        self.trajectoryLine = render.attachNewNode(trajectory_node)
        self.trajectoryLine.setName("ProjectileTrajectory")

        return self.trajectoryLine
        
    def spawnProjectiles(self,n,startPos,endPos):
        # setup the projectile interval
        self.trajectories = []
        self.projectiles = []
        for i in range(n):
            ball = loader.loadModel("smiley")
            ball.reparentTo(render)
            ball.setPos(startPos+Vec3(random.uniform(-2,2),random.uniform(-2,2),0))
            self.projectiles.append(ball)
            pos = endPos + Vec3(random.uniform(-2,2),random.uniform(-2,2),0)
            duration = random.uniform(0.9, 1.1)
            trajectory = ProjectileInterval(ball, duration=duration,
                                            endPos=pos)
            self.trajectories.append(trajectory)
        # Create a Parallel interval to run all trajectories simultaneously
        parallel_trajectories = Parallel(*self.trajectories)
        #parallel_trajectories.start()
        #for trajectory in self.trajectories:
        #    trajectory.start()
        return parallel_trajectories

    def startTaskFunction(self,taskfunction,taskname):
        if taskMgr.hasTaskNamed(taskname):
            taskMgr.remove(taskname)
        taskMgr.add(taskfunction, taskname)
        return

    def taskLoopPathTowardsMouse(self, task):
        if self.unitToMove.isInCombat:
            print("Unit is in combat, cant move.")
            return task.done
        
        self.pathTowardsMouse(self.unitToMove)
        return task.cont
    
    def taskShootingArcUpdate(self, task):
        for unit in self.units:
            unit.model.setColor(unit.color)
            unit.bodyNP.setCollideMask(BitMask32.bit(unit.bitmask))
        if self.unitToMove.isInCombat:
            print("Unit is in combat, cant shoot.")
            return task.done
        print("equiped weapon is: ",self.unitToMove.unit.model.equipedWeapon)
        if self.unitToMove.unit.model.equipedWeapon is None:# or not self.unitToMove.unit.model.equippedWeapon.is_ranged:
            print("Unit has no equiped weapon equipped, cant shoot.")
            return task.done
        r=False
        print(self.unitToMove.unit.model.weapons)
        for weapon in self.unitToMove.unit.model.weapons:
            print("checking weapon: ",self.unitToMove.unit.model.weapons.get(weapon))
            if self.unitToMove.unit.model.weapons.get(weapon).get('tag') == 'ranged':
                r=True
                self.unitToMove.unit.model.equip_weapon(weapon)
                print("Equipping weapon: ",weapon)
        if not r:
            print("Unit has no ranged weapon, cant shoot.")
            return task.done
            if not self.unitToMove.unit.model.equipedWeapon.get('tag') == 'ranged':
                print("Unit has no ranged weapon equipped, cant shoot.")
                return task.done
        self.shootingArcPoints = self.shootingArc(self.unitToMove.bodyNP.getPos(render), 
                                                       num_points=80, rotationangle=self.unitToMove.bodyNP.getH()+45)
        self.ground.setShaderInput("polygonpoints", self.shootingArcPoints)
        self.ground.setShaderInput("isActive", True)
        if not taskMgr.hasTaskNamed("taskShootingTrajectoryDrawLine"):
            taskMgr.add(self.taskShootingTrajectoryDrawLine, "taskShootingTrajectoryDrawLine")
        self.checkArrows()
        return task.done
    
    def taskStartCombat(self, task):
        if self.unitToMove.isInCombat:
            #self.verySimpleBattle(self.unitToMove.bodyNP, self.unitToMove.isInCombatWith.bodyNP, "front")
            #base.messenger.toggleVerbose()
            #taskMgr.add(self.verySimpleBattle, "verySimpleBattle", extraArgs=[self.unitToMove.bodyNP, self.unitToMove.isInCombatWith.bodyNP, self.unitToMove.isInCombatFlank], appendTask=True)
            
            self.taskMgr.add(self.verySimpleBattleStart)
        return task.done
    
    def checkIfInsidePolygon(self, point, polygonPoints):
        # Ray-casting algorithm to determine if point is inside polygon
        n = len(polygonPoints)
        inside = False

        x, y = point.x, point.y
        p1x, p1y = polygonPoints[0].x, polygonPoints[0].y

        for i in range(n + 1):
            p2x, p2y = polygonPoints[i % n].x, polygonPoints[i % n].y
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def taskShootingTrajectoryDrawLine(self, task):
        if self.checkIfInsidePolygon(self.mousePosOnGround, self.coordsToWorld(self.shootingArcPoints)):
            self.trajectoryLine = self.drawProjectileTrajectory(self.unitToMove.bodyNP.getPos(), self.mousePosOnGround)
        return task.cont
    
    def coordsToWorld(self, points):
        worldPoints = []
        for point in points:
            point = point * 2
            point -= Vec2(1,1)
            point = point * 50
            worldPoints.append(Point3(point.x, point.y, 0))
        return worldPoints

    def checkArrows(self):
        for point in self.shootingArcPoints:
            point = point * 2
            point -= Vec2(1,1)
            point = point * 50
            pFrom = self.unitToMove.bodyNP.getPos(render)
            pTo = Point3(point.x, point.y, pFrom.z)

            result = self.world.rayTestClosest(pFrom, pTo, BitMask32.bit(1))

            if result.hasHit():
                print(result.hasHit())
                print(result.getHitPos())
                print(result.getHitNormal())
                print(result.getHitFraction())
                print(result.getNode().getChildren())
                for c in result.getNode().getChildren():
                    print(c.getName())
                    if "Model" in c.getName():
                        np = NodePath.anyPath(c)
                        np.setColor(1,0,1,1)
                        NodePath.anyPath(result.getNode()).setCollideMask(BitMask32.bit(3))
                        #self.toCleanup.append(np)

    def cameraShake(self, intensity=1.0, duration=0.5):
        original_pos = self.camera.getPos()

        def shake_task(task):
            elapsed = task.time
            if elapsed < duration:
                offset_x = (random.uniform(-1, 1) * intensity)
                offset_y = (random.uniform(-1, 1) * intensity)
                offset_z = (random.uniform(-1, 1) * intensity)
                self.camera.setPos(original_pos + Vec3(offset_x, offset_y, offset_z))
                return task.cont
            else:
                self.camera.setPos(original_pos)
                return task.done

        self.taskMgr.add(shake_task, "cameraShakeTask")
    
    def mouseHoverUnit(self,task):
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
            result = self.world.rayTestClosest(pFrom, pTo, BitMask32.allOn())
            #self.debug_ray(pFrom, pTo)
            if result.hasHit():
                hit_node = result.getNode()
                # Check if hit node is a unit
                #if isinstance(hit_node, BulletRigidBodyNode):
                self.mousePosOnGround = result.getHitPos()
                #self.trajectoryLine = self.drawProjectileTrajectory(self.unitToMove.bodyNP.getPos(), result.getHitPos())
                #self.drawProjectileTrajectory(Point3(-15,-5,0), Point3(-15,5,0), n=50)
                if True:
                    node_name = hit_node.getName()
                    #print(node_name)
                    if node_name.startswith('UnitCollision-'):
                        unit_name = node_name.replace('UnitCollision-', '')
                        # Set the active unit based on which was clicked
                        for unit in self.units:
                            if unit_name == unit.unitName:
                                hovered_unit = unit
                                #print(f"Hovered unit: {unit.unitName}")
                                unit.text_node.show()
                    else:
                        for unit in self.units:
                            unit.text_node.hide()
        
        return task.cont
                        

    def setActiveUnit(self,taskfunction,taskname):
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
            result = self.world.rayTestClosest(pFrom, pTo, BitMask32.bit(1))
            
            if result.hasHit():
                hit_node = result.getNode()
                print("Mouse click hit:",result.getHitPos())

                """ self.terninger=[]
                for i in range(5):
                    terning = Dice(self.world, position=result.getHitPos()+Vec3(i,0,10), size=1.0)
                    self.terninger.append(terning)
                for terning in self.terninger:
                    terning.roll()
                taskMgr.add(checkDice, "checkDiceTask", extraArgs=[self.terninger], appendTask=True) """
                #print(self.fsm.state)
                
                
                if True:
                    node_name = hit_node.getName()
                    if node_name.startswith('UnitCollision-'):
                        unit_name = node_name.replace('UnitCollision-', '')
                        # Set the active unit based on which was clicked
                        for unit in self.units:
                            if unit_name == unit.unitName:
                                self.unitToMove = unit
                                print(f"Selected unit: {unit.unitName}")
                        """ if unit_name == self.bretBowmen.unitName:
                            self.unitToMove = self.bretBowmen
                            print(f"Selected unit: {self.bretBowmen.unitName}")
                        elif unit_name == self.goblins.unitName:
                            self.unitToMove = self.goblins
                            print(f"Selected unit: {self.goblins.unitName}") """
                        self.accept('mouse3', self.moveUnit,[self.unitToMove])
                        #self.startTaskFunction(self.taskLoopPathTowardsMouse, "taskLoopPathTowardsMouse")
                        self.startTaskFunction(taskfunction, taskname)
                        self.debugTextUnit.setText(f"Selected unit: {self.unitToMove.unitName}\nStats: {self.unitToMove.unit.model.characteristics}")

            result2 = self.world.rayTestClosest(pFrom, pTo, BitMask32.bit(3))    
            if result2.hasHit():
                total_hits = 0
                selected_unit = self.getSelectedUnit(result2.getNode())
                attacker = self.unitToMove.unit
                defender = selected_unit.unit
                print(attacker.name, "shooting an arrow at",defender.name)
                #attacker.model.equip_weapon('short bow')
                attacks, total_hits, suffered_wounds,  saves_made, total_wounds = simulate_battle(attacker, defender,charge=False)
                self.printBattleResults(self.unitToMove, selected_unit, attacks, total_hits, suffered_wounds, saves_made, total_wounds)
                #unit.model.setColor(unit.color)
                #unit.bodyNP.setCollideMask(BitMask32.bit(unit.bitmask))
                self.unitToMove.bodyNP.setCollideMask(BitMask32.bit(4))
                selected_unit.bodyNP.setCollideMask(BitMask32.bit(4))
                self.shootingAnimation(self.unitToMove,selected_unit,total_wounds)
                """ for c in result2.getNode().getChildren():
                    #print(c.getName())
                    if "Model" in c.getName():
                        np = NodePath.anyPath(c)
                        np.setColor(0,1,0,1)
                        NodePath.anyPath(result2.getNode()).setCollideMask(BitMask32.bit(1)) """

    def shootingAnimation(self,attackerUnit,defenderUnit,total_wounds):
        
        #self.p.start(parent=render, renderParent=render)
        self.p.setPos(defenderUnit.bodyNP.getPos())
        

        #self.p_miss.start(parent=render, renderParent=render)
        self.p_miss.setPos(defenderUnit.bodyNP.getPos())

        self.cameraShake(intensity=0.5, duration=0.3)
        
        
        parTra = self.spawnProjectiles(5,attackerUnit.bodyNP.getPos(),defenderUnit.bodyNP.getPos())
        seq = Sequence(parTra,
                       Func(self.p.start, parent=render, renderParent=render),
                       Func(taskMgr.doMethodLater, 4.0, lambda task: self.p.disable(), 'stopParticles'),
                       Func(self.p_miss.start, parent=render, renderParent=render),
                       Func(self.removeModelsFromUnit, defenderUnit, total_wounds),
                       Func(taskMgr.doMethodLater, 4.0, lambda task: self.p_miss.disable(), 'stopMissParticles')
                       )
        seq.start()
        taskMgr.remove("taskShootingTrajectoryDrawLine")

    def getSelectedUnit(self,cnode):
        #if isinstance(cnode, BulletRigidBodyNode):
        if True:
            node_name = cnode.getName()
            if node_name.startswith('UnitCollision-'):
                unit_name = node_name.replace('UnitCollision-', '')
                # Set the active unit based on which was clicked
                for unit in self.units:
                    if unit_name == unit.unitName:
                        selected = unit
                        print(f"Selected unit: {unit.unitName}")
                """ if unit_name == self.bretBowmen.unitName:
                    selected = self.bretBowmen
                    print(f"Selected unit: {self.bretBowmen.unitName}")
                elif unit_name == self.goblins.unitName:
                    selected = self.goblins
                    print(f"Selected unit: {self.goblins.unitName}") """
        return selected

    
    def setup_text_node(self, text="", pos=(0, 0.9), scale=0.07, color=(1, 1, 1, 1)):
        """
        Creates and returns a text node for displaying text on screen.
        
        Args:
            text: The text to display
            pos: (x, y) position in aspect2d coordinates (-1 to 1)
            scale: Text scale
            color: Text color as (r, g, b, a) tuple
        
        Returns:
            TextNode object that can be updated with .setText()
        """
        
        text_node = OnscreenText(
            text=text,
            pos=pos,
            scale=scale,
            fg=color,
            align=0,  # Center alignment
            mayChange=True
        )
        return text_node

    def setup_shader(self):
        #surface = self.render.find("**/ground")
        surface = self.ground
        shader = Shader.load(Shader.SL_GLSL, "shaders/c2.vert", "shaders/c1.frag")
        surface.setShader(shader)
        surface.setShaderInput("pos", Vec3(0,0,0))
        # Define polygon points for the shader
        self.polygonpoints = []
        num_points = 6  # Example: hexagon
        radius = 0.5
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            self.polygonpoints.append(Vec2(x, y))
        surface.setShaderInput("polygonpoints", self.polygonpoints)
        surface.setShaderInput("isActive", False)
        self.polygonpoints = []

    def setup_bullet(self):
        self.world = BulletWorld()
        self.world.setGravity(Vec3(0, 0, -9.81))
        shape = BulletPlaneShape(Vec3(0, 0, 1), 0)
        node = BulletRigidBodyNode('Ground')
        node.addShape(shape)
        np = render.attachNewNode(node)
        np.setCollideMask(BitMask32.bit(1))
        self.world.attachRigidBody(node)
        """ mesh = BulletTriangleMesh()

        for geomNP in render.findAllMatches('**/+GeomNode'):
            print("fant node")
            geomNode = geomNP.node()
            ts = geomNP.getTransform(np)
            #print(ts)
            for geom in geomNode.getGeoms():
                mesh.addGeom(geom, ts=ts)
                #print(geom)
        #lol

        worldNP = render.attachNewNode('World')
        body = BulletRigidBodyNode('grid')
        shape = BulletTriangleMeshShape(mesh, False)
        bodyNP = worldNP.attachNewNode(body)
        bodyNP.node().addShape(shape)
        bodyNP.setCollideMask(BitMask32.allOn()) 
        self.world.attachRigidBody(bodyNP.node())"""

        # Show Bullet debug nodes
        debugNode = BulletDebugNode('Debug')
        debugNode.showWireframe(True)
        debugNode.showConstraints(True)
        debugNode.showBoundingBoxes(False)
        debugNode.showNormals(True)
        debugNP = render.attachNewNode(debugNode)
        
        self.world.setDebugNode(debugNP.node())
        debugNP.show()
        # Add simple Bullet collision geometry (sphere) to smiley_copy

        # Estimate radius from bounding box
        bounds = self.smiley_copy.getTightBounds()
        center = (bounds[0] + bounds[1]) * 0.5
        radius = max((bounds[1] - bounds[0]).length() * 0.5, 0.1)
        radius *= 1.5  # Adjust scale factor as needed

        smiley_copy_shape = BulletSphereShape(radius)
        smiley_copy_body = BulletRigidBodyNode('SmileyCopy')
        smiley_copy_body.addShape(smiley_copy_shape)
        smiley_copy_np = self.smiley_copy.attachNewNode(smiley_copy_body)
        smiley_copy_np.setCollideMask(BitMask32.bit(1))
        self.world.attachRigidBody(smiley_copy_body)
        """ 
        # Add a simple Bullet character controller for the player
        height = .01
        radius = 0.4
        shape = BulletCapsuleShape(radius, height - 2*radius, ZUp)

        playerNode = BulletCharacterControllerNode(shape, 0.4, 'Player')
        #self.playerNP = self.worldNP.attachNewNode(playerNode)
        self.playerNP = render.attachNewNode(playerNode)
        self.playerNP.setPos(-2, 0, 14)
        self.playerNP.setH(45)
        self.playerNP.setCollideMask(BitMask32.bit(6))
        #self.playerNP.setKinematic(True)

        self.world.attachCharacter(self.playerNP.node()) """
        self.playerNP = render.attachNewNode("player")
        # Add a task to update Bullet physics every frame
        self.taskMgr.add(self.update_physics, "update_physics")

    def update_physics(self,task):
            dt = globalClock.getDt()
            self.world.doPhysics(dt)
            
            return task.cont
    def move_node_smoothly(self, node, target_pos, duration=1.0):
        interval = LerpPosInterval(node, duration, target_pos,blendType='easeInOut')
        mySequence = Sequence(interval)
        mySequence.start()

    def draw_circle(self, center=Point3(0, 0, 0), radius=5, segments=32, color=(1, 0, 0, 1)):

        # Create MeshDrawer and NodePath if not already present
        if not hasattr(self, 'mesh_drawer'):
            self.mesh_drawer = MeshDrawer()
            self.mesh_drawer.setBudget(1000)
            #self.mesh_drawer_np = NodePath(self.mesh_drawer.create())
            self.mesh_drawer_node = self.mesh_drawer.getRoot()
            self.mesh_drawer_node.reparentTo(self.render)

        self.mesh_drawer.begin(base.cam, self.render)
        angle_step = 2 * 3.14159265 / segments
        for i in range(segments + 1):
            angle = i * angle_step
            x = center.x + radius * math.cos(angle)
            y = center.y + radius * math.sin(angle)
            z = center.z
            # Draw line segments between consecutive points (crossSegment)
            if i > 0:
                prev_angle = (i - 1) * angle_step
                prev_x = center.x + radius * math.cos(prev_angle)
                prev_y = center.y + radius * math.sin(prev_angle)
                prev_z = center.z
                self.mesh_drawer.segment(Point3(x, y, z),
                    Point3(prev_x, prev_y, prev_z), Vec4(0,0,1,1), .1, color
                )
        self.mesh_drawer.end()

    def draw_arc(self, center=Point3(0, 0, 0), radius=5, remainingmove=5,  start_angle=0, end_angle=90, segments=32, color=(1, 0, 0, 1)):
        # Create MeshDrawer and NodePath if not already present
        if not hasattr(self, 'mesh_drawer'):
            self.mesh_drawer = MeshDrawer()
            self.mesh_drawer.setBudget(1000)
            
            self.mesh_drawer_node = self.mesh_drawer.getRoot()
            self.mesh_drawer_node.setTwoSided(True)
            self.mesh_drawer_node.reparentTo(self.smiley)

        self.mesh_drawer.begin(base.cam, self.render)
        angle_step = (end_angle - start_angle) * (math.pi / 180) / segments
        center_color = Vec4(*color)
        for i in range(segments):
            angle1 = start_angle * (math.pi / 180) + i * angle_step
            angle2 = start_angle * (math.pi / 180) + (i + 1) * angle_step
            x1 = center.x + radius * math.cos(angle1)
            y1 = center.y + radius * math.sin(angle1)
            z1 = center.z
            x2 = center.x + radius * math.cos(angle2)
            y2 = center.y + radius * math.sin(angle2)
            z2 = center.z
            # Draw triangle from center to arc segment
            self.mesh_drawer.tri(
                center, center_color, (0, 0),
                Point3(x1, y1, z1), center_color, (1, 0),
                Point3(x2, y2, z2), center_color, (0, 1)
            )
        # Calculate the direction of the end of the arc
        angle_rad = end_angle * (math.pi / 180)
        dir_x = math.cos(angle_rad)
        dir_y = math.sin(angle_rad)

        # Rectangle parameters
        rect_length = radius  # Length along the arc direction
        rect_width = remainingmove   # Width perpendicular to the arc

        # Calculate rectangle corners
        half_length = rect_length / 2
        half_width = rect_width / 2

        rect_center_x = center.x + radius / 2 * dir_x - half_width * dir_x
        rect_center_y = center.y + radius / 2 * dir_y + half_width * dir_y
        rect_center_z = center.z

        

        # Direction vectors
        forward = Vec3(dir_x, dir_y, 0).normalized()
        right = Vec3(-dir_y, dir_x, 0).normalized()

        corner1 = Point3(rect_center_x + forward.x * half_length + right.x * half_width,
                         rect_center_y + forward.y * half_length + right.y * half_width,
                         rect_center_z)
        corner2 = Point3(rect_center_x + forward.x * half_length - right.x * half_width,
                         rect_center_y + forward.y * half_length - right.y * half_width,
                         rect_center_z)
        corner3 = Point3(rect_center_x - forward.x * half_length - right.x * half_width,
                         rect_center_y - forward.y * half_length - right.y * half_width,
                         rect_center_z)
        corner4 = Point3(rect_center_x - forward.x * half_length + right.x * half_width,
                         rect_center_y - forward.y * half_length + right.y * half_width,
                         rect_center_z)

        rect_color = Vec4(*color)

        # Draw the rectangle as two triangles (correct winding order)
        self.mesh_drawer.tri(corner1, rect_color, (0, 0), corner3, rect_color, (1, 1), corner2, rect_color, (1, 0))
        self.mesh_drawer.tri(corner3, rect_color, (1, 1), corner1, rect_color, (0, 0), corner4, rect_color, (0, 1))
        self.mesh_drawer.end()

        mesh = BulletTriangleMesh()

        for geomNP in self.mesh_drawer_node.findAllMatches('**/+GeomNode'):
            print("fant node i mesh drawer")
            geomNode = geomNP.node()
            ts = geomNP.getTransform(self.mesh_drawer_node)
            #print(ts)
            for geom in geomNode.getGeoms():
                mesh.addGeom(geom, ts=ts)
                #print(geom)
        #lol
        body = BulletRigidBodyNode('movearea')
        shape = BulletTriangleMeshShape(mesh, False)
        body.addShape(shape)
        # Detach any existing BulletRigidBodyNode children from mesh_drawer_node
        for child in self.mesh_drawer_node.getChildren():
            if child.node().isOfType(BulletRigidBodyNode.getClassType()):
                self.world.removeRigidBody(child.node())
                child.detachNode()
        bodyNP = self.mesh_drawer_node.attachNewNode(body)
        bodyNP.node().setMass(0)
        bodyNP.setCollideMask(BitMask32.allOn())
        self.world.attachRigidBody(bodyNP.node())
        #self.world.doPhysics(0.01)
         
        # Calculate points along the arc's centerline
        motionpath_points = []
        for i in range(segments + 1):
            angle = start_angle * (math.pi / 180) + i * angle_step
            x = center.x + (radius - rect_width / 2) * math.cos(angle)
            y = center.y + (radius - rect_width / 2) * math.sin(angle)
            z = center.z
            motionpath_points.append({
                "pos": Point3(x, y, z),
                "hpr": (math.degrees(angle), 0, 0)
            })
        
        # Add points along the rectangle's centerline (straight out from arc end)
        # Centerline starts at the end of the arc and goes straight in the arc's end direction
        for i in range(1, 6):
            t = i / 5.0
            x = center.x + radius * math.cos(angle_rad) + t * rect_width * dir_x
            y = center.y + radius * math.sin(angle_rad) + t * rect_width * dir_y
            z = center.z
            motionpath_points.append({
            "pos": Point3(-x, y, z),
            "hpr": (math.degrees(angle_rad), 0, 0)
            })

    
        
        


    def check_bullet_collision(self, node_a, node_b):
        """
        Checks for Bullet collision between two Panda3D NodePaths with BulletRigidBodyNode attached.
        Returns True if they are colliding, False otherwise.
        """
        # Get Bullet nodes
        body_a = node_a.find("**/+BulletRigidBodyNode")
        body_b = node_b.find("**/+BulletRigidBodyNode")
        if body_a.isEmpty() or body_b.isEmpty():
            print("One or both nodes do not have a BulletRigidBodyNode attached.")
            return False

        # Get Bullet shapes
        bullet_node_a = body_a.node()
        bullet_node_b = body_b.node()

        # Use BulletWorld contactTestPair
        result = self.world.contactTestPair(bullet_node_a, bullet_node_b)
        print("Collision check result:")
        print(result.getNumContacts())
        return result.getNumContacts() > 0

    

    def upAndDown(self):
        
        #time = task.time
        #surface.setZ(0+sin(time)*3)
        if base.mouseWatcherNode.hasMouse():
            x = base.mouseWatcherNode.getMouseX()
            y = base.mouseWatcherNode.getMouseY()
            #print(x,y)
            #surface.set_shader_input("pos", Vec3(base.mouseWatcherNode.getMouseX(),0,base.mouseWatcherNode.getMouseY())*4)
            #pFrom = Point3(0, 0, 0)
            #pTo = Point3(10, 0, 0)

            # Get to and from pos in camera coordinates
            pMouse = base.mouseWatcherNode.getMouse()
            pFrom = Point3()
            pTo = Point3()
            base.camLens.extrude(pMouse, pFrom, pTo)

            # Transform to global coordinates
            pFrom = render.getRelativePoint(base.cam, pFrom)
            pTo = render.getRelativePoint(base.cam, pTo)

            result = self.world.rayTestClosest(pFrom, pTo)

            print(result.hasHit())
            print(result.getHitPos())
            print(result.getHitNormal())
            print(result.getHitFraction())
            print(result.getNode())
            #surface.set_shader_input("pos", result.getHitPos())

            #self.smiley.setPos(result.getHitPos() + Vec3(0,0,0))
            self.goblins.bodyNP.setPos(result.getHitPos())
            #self.move_node_smoothly(self.smiley, result.getHitPos() + Vec3(0,0,0.1), duration=0.5)
            #dist = (self.smiley.getPos() - self.smiley_copy.getPos()).length()
            #print(f"Distance between smilies: {dist}")

            
            groundSizeboundingbox=self.ground.getTightBounds()
            print(groundSizeboundingbox)
            self.ground.set_shader_input("pos", result.getHitPos()/abs(groundSizeboundingbox[0][0]))
            print(self.goblins.model.getTightBounds())
            unitWidth=abs(self.goblins.model.getTightBounds()[1][0]-self.goblins.model.getTightBounds()[0][0])
            unitHeight=abs(self.goblins.model.getTightBounds()[1][1]-self.goblins.model.getTightBounds()[0][1])
            print(f"Unit Width: {unitWidth}, Unit Height: {unitHeight}")
            self.ground.set_shader_input("unitSize", Vec3(unitWidth, unitHeight, 0))

            

            """ 
            #self.draw_circle(center=Point3(0, 0, 5), radius=10, segments=64, color=(1, 1, 1, 1))
            self.draw_arc(center=Point3(0,0, 0), radius=self.unitHeight/2, remainingmove=5, start_angle=0, end_angle=45, segments=64, color=(1, 1, 1, 1))
            #self.mesh_drawer_node.reparentTo(self.smiley)
            print(self.smiley.ls())
            self.mesh_drawer_node.setPos(Vec3(-self.unitWidth/4, -self.unitHeight/4, 0))
            self.mesh_drawer_node.setZ(0)
            self.mesh_drawer_node.setHpr(90,0,0)
            collision = self.check_bullet_collision(self.mesh_drawer_node, self.smiley_copy)
            #self.mesh_drawer_node.hide() 
            """
            return
    
    def shootingArc(self, origo, num_points=40, rotationangle=30):
        points =[]
        origo=(origo/50 +1)*0.5
        origo   = Vec2(origo.x,origo.y)
        points.append(origo)

        arcmax = math.pi/2

        for i in range(0,num_points):
            angle = arcmax * i / (num_points - 1)
            x = 0.5 * math.cos(angle) 
            y = 0.5 * math.sin(angle)
            points.append(origo+Vec2(x,y))
            points[-1] = self.rotatePoint(points[-1], rotationangle, origo=origo)

        for n in range(len(points),num_points+3):
            points.append(points[-1])
        return points

    def pointArc(self,origo, num_points=40, mouse_pos=None,rotationangle=-21,width=0.5,height=0.5,movedistance=8):
        points =[]
        forward= Vec2(-math.sin(math.radians(rotationangle)), math.cos(math.radians(rotationangle)))*height/2.0
        origo = origo + forward
        #origo   = Vec2(0.55,0.55)
        #origo= origo+Vec2(math.cos(math.radians(rotationangle))*height/2,-math.sin(math.radians(rotationangle))*height/2)
        points.append(origo)

        arcmax = math.pi/2

        #rotationangle= 30
        midpoint=Vec2(width*math.cos(math.radians(rotationangle) ), width*math.sin(math.radians(rotationangle) ))*0.5+origo
        midpoint_unit_vector = midpoint - origo
        midpoint_mouse_vector = mouse_pos - midpoint if mouse_pos else Vec2(1,1)
        maxMoveDistance=movedistance
        #movedistance = min(movedistance, midpoint_mouse_vector.length())
        print(midpoint_unit_vector,midpoint_mouse_vector,midpoint_unit_vector.normalized().dot(midpoint_mouse_vector.normalized()))
        print("before mouse pos:", mouse_pos, "midpoint:", midpoint)
        filipped=False
        if midpoint_unit_vector.normalized().dot(midpoint_mouse_vector.normalized()) > 0:
            #rotationangle += 180
            print("flipped")
            filipped=True
            vinkel=rotationangle+90
            mouse_pos = self.mirrorPointArc([mouse_pos], mirror_vec=Vec2(math.cos(math.radians(vinkel)), math.sin(math.radians(vinkel))), origin=midpoint)[0]
            print("after mouse pos:", mouse_pos)
            #mouse_pos = self.rotatePoint(mouse_pos, 90, origo=midpoint)

        quarternion = LQuaterniond()
        quarternion.set_from_axis_angle(rotationangle, LVector3d(0,0,1))
        print(quarternion.getForward())
        behind= quarternion.getForward().dot(LVector3d(midpoint_mouse_vector.normalized().x, midpoint_mouse_vector.normalized().y,0))
        print(behind)
        nums=0
        if behind < 0 and abs(behind) > 0.8:
            print("behind arc")
            if abs(behind) > 0.8:
                angle = 0
                print("right behind")
                x = width * math.cos(0) 
                y = width * math.sin(0)
                points.append(origo+Vec2(x,y))
                points[-1] = self.rotatePoint(points[-1], rotationangle, origo=origo)
                points.append(points[-1]+Vec2(-math.sin(math.radians(rotationangle)),math.cos(math.radians(rotationangle)))*movedistance/4)
                points.append(points[0]+Vec2(-math.sin(math.radians(rotationangle)),math.cos(math.radians(rotationangle)))*movedistance/4)
                for i, p in enumerate(points):
                    points[i] = p-Vec2(quarternion.getForward().x, quarternion.getForward().y)*movedistance/4
                points = points[2:] + points[:2]

                #return points
        elif behind < 0 and abs(behind) < 0.8:
            print("somewhat behind arc")
            angle = 0
            print("right behind")
            x = width * math.cos(0) 
            y = width * math.sin(0)
            points.append(origo+Vec2(x,y))
            points[-1] = self.rotatePoint(points[-1], rotationangle, origo=origo)
            points.append(points[-1])
            points.append(points[-1]+Vec2(-math.sin(math.radians(rotationangle)),math.cos(math.radians(rotationangle)))*height)
            points.append(points[-1])
            points.append(points[0]+Vec2(-math.sin(math.radians(rotationangle)),math.cos(math.radians(rotationangle)))*height)
            #for i, p in enumerate(points):
            print(points)
            for i in [2,3]:
                points[i] = points[i]+Vec2(quarternion.getRight().x, quarternion.getRight().y)*movedistance/4 
                #points[i] -= Vec2(quarternion.getForward().x, quarternion.getForward().y)*height    
            print(points)
            for i in [1,4]:
                points[i] = points[i]+Vec2(quarternion.getRight().x, quarternion.getRight().y)*(movedistance/4 -width)
                #points[i] -= Vec2(quarternion.getForward().x, quarternion.getForward().y)*height    
            print(points)

            for i in range(len(points)):
                #points[i] = p-Vec2(quarternion.getRight().x, quarternion.getRight().y)*movedistance/4   
                #   
                #print(Vec2(quarternion.getForward().x, quarternion.getForward().y)*height/2)
                #print(points[i])
                points[i] = points[i]-Vec2(quarternion.getForward().x, quarternion.getForward().y)*height/2
                #print(points[i])
            print(points)
            points = [points[-1]] + points[:-1]
            print(points)
            
            for i in [-1,-2]:
                print(points[i])
            vinkel=rotationangle+90
            filipped = not filipped
        
        
            

        else:
            for i in range(0,num_points):
                angle = arcmax * i / num_points
                x = width * math.cos(angle) 
                y = width * math.sin(angle)
                points.append(origo+Vec2(x,y))

                points[-1] = self.rotatePoint(points[-1], rotationangle, origo=origo)

                vector=points[-1]-origo
                pointmid=origo+vector*.5
                vectormouse=mouse_pos - pointmid if mouse_pos else Vec2(1,1)
                #print(f"vectormouse: {vectormouse}, vector: {vector}, dot: {vectormouse.dot(vector)}")
                nums=i
                if abs(vectormouse.dot(vector)) < .001:
                    
                    break
                if width * angle >= movedistance:
                    print("break because width*angle >= movedistance:", width*angle, movedistance)
                    break

            #if movedistance 
            #if movedistance + angle * width/2 > maxMoveDistance:
            print(movedistance, angle*width, vectormouse.length()+angle*width)
            movedistance = min(movedistance, vectormouse.length()+angle*width)
            if movedistance - angle*width > 0:
                movedistance -= angle*width
            else:
                movedistance = 0

            print("final movedistance:", movedistance,movedistance*2*50,(movedistance+angle*width)*2*50)
            self.moveArceDistance = (movedistance+angle*width)*2*50
            self.debugTextInfo.setText(f"Arc distance: {(self.moveArceDistance):.1f} ")
            #movedistance = min(movedistance, angle*width)
            #points.append(points[-1]+Vec2(-math.sin(angle+math.radians(rotationangle)),math.cos(angle+math.radians(rotationangle)))*0.2)
            points.append(points[-1]+Vec2(-math.sin(angle+math.radians(rotationangle)),math.cos(angle+math.radians(rotationangle)))*movedistance)
            #points[-1] = self.rotatePoint(points[-1], rotationangle)
            #points.append(points[0]+Vec2(-math.sin(angle+math.radians(rotationangle)),math.cos(angle+math.radians(rotationangle)))*0.2)
            points.append(points[0]+Vec2(-math.sin(angle+math.radians(rotationangle)),math.cos(angle+math.radians(rotationangle)))*movedistance)
            #points[-1] = self.rotatePoint(points[-1], rotationangle)
            #print(points)
        nums=len(points)
        midpointfront = (points[-1] + points[-2]) * 0.5
        midpointfront = (points[nums-1] + points[nums-2]) * 0.5
        self.numsPoints=nums
        for n in range(len(points),num_points+3):
            points.append(points[-1])
        
        #print(len(points))

        if filipped:
            mirrored_points = self.mirrorPointArc(points, mirror_vec=Vec2(math.cos(math.radians(vinkel)), math.sin(math.radians(vinkel))), origin=midpoint)
            #self.arcPoint=mirrored_points[nums+2]
            #self.arcPoint=midpointfront
            self.arcPoint=(mirrored_points[nums-1] + mirrored_points[nums-2]) * 0.5
            self.arcPointRotation=math.degrees(-angle)
            return mirrored_points
        
        #print(points)
        #self.arcPoint=(points[nums+1]+points[nums])*0.5
        self.arcPoint=midpointfront
        self.arcPointRotation=math.degrees(angle)

        
        return points

    def mirrorPointArc(self, points, mirror_vec, origin):
        mirrored_points = []
        # Mirror about a vector (not necessarily passing through (0,0); use 'origin' as the base point)
        # To mirror about an arbitrary vector, you need to specify the vector direction.
        # Here, let's add an optional argument for the mirror vector:
        # Usage: mirrorPointArc(points, mirror_vec=Vec2(1,0))
        #mirror_vec = getattr(self, 'mirror_vec', Vec2(1, 0))  # Default to x-axis if not set
        #mirror_vec = mirror_vec
        mirror_vec = mirror_vec.normalized()
        #print("Mirroring with vector:", mirror_vec)
        for p in points:
            # Vector from origin to point
            #print("Original point:", p)
            v = p - origin
            # Project v onto mirror_vec
            #print(v.dot(mirror_vec))
            #print(v.normalized().dot(mirror_vec))
            proj =  (mirror_vec * v.normalized().dot(mirror_vec)) * v.length()
            # Perpendicular component
            #print("v:", v, "proj:", proj)
            perp = v - proj
            #print("perp:", perp)
            # Mirror: subtract twice the perpendicular component
            mirrored_v = p -  perp * 2
            mirrored_points.append(Vec2(mirrored_v.x, mirrored_v.y))
        """ for p in points:
            mirrored_points.append(Vec2(0.5 - (p.x - 0.5), p.y)) """
        return mirrored_points
    

    
    def rotatePoint(self, point, angle_degrees, origo=Vec2(0.25,0.25)):
        angle_radians = math.radians(angle_degrees)
        cos_angle = math.cos(angle_radians)
        sin_angle = math.sin(angle_radians)
        x = point.x - origo.x
        y = point.y - origo.y
        # Counter-clockwise rotation
        x_rotated = x * cos_angle - y * sin_angle
        y_rotated = x * sin_angle + y * cos_angle
        return Vec2(x_rotated + origo.x, y_rotated + origo.y)
    
    def meshPointArc(self, origo, num_points=40, mouse_pos=None, rotationangle=-21):
        if not hasattr(self, 'mesh_drawer'):
            self.mesh_drawer = MeshDrawer()
            self.mesh_drawer.setBudget(1000)
            
            self.mesh_drawer_node = self.mesh_drawer.getRoot()
            self.mesh_drawer_node.setTwoSided(True)
            self.mesh_drawer_node.reparentTo(self.smiley)

        #self.mesh_drawer.begin(base.cam, self.render)
        points = self.pointArc(origo, num_points, mouse_pos, rotationangle)
        mesh = BulletTriangleMesh()

        for i in range(1, len(points)-2):
            p0 = Point3(points[0].x, points[0].y, 0)
            p1 = Point3(points[i].x, points[i].y, 0)
            p2 = Point3(points[i+1].x, points[i+1].y, 0)
            mesh.add_triangle(p0, p1, p2)

        body = BulletRigidBodyNode('arcarea')
        shape = BulletTriangleMeshShape(mesh, False)
        body.addShape(shape)
        # Detach any existing BulletRigidBodyNode children from mesh_drawer_node
        for child in self.mesh_drawer_node.getChildren():
            if child.node().isOfType(BulletRigidBodyNode.getClassType()):
                self.world.removeRigidBody(child.node())
                child.detachNode()
        bodyNP = self.mesh_drawer_node.attachNewNode(body)
        bodyNP.setHpr(90,0,0)
        bodyNP.node().setMass(0)
        bodyNP.setCollideMask(BitMask32.allOn())
        self.world.attachRigidBody(bodyNP.node())
        return points

    def pathTowardsMouse(self,unit):
        
        if base.mouseWatcherNode.hasMouse():
            self.unitToMove=unit
            x = base.mouseWatcherNode.getMouseX()
            y = base.mouseWatcherNode.getMouseY()
            #print(x,y)
            #surface.set_shader_input("pos", Vec3(base.mouseWatcherNode.getMouseX(),0,base.mouseWatcherNode.getMouseY())*4)
            #pFrom = Point3(0, 0, 0)
            #pTo = Point3(10, 0, 0)

            # Get to and from pos in camera coordinates
            pMouse = base.mouseWatcherNode.getMouse()
            pFrom = Point3()
            pTo = Point3()
            base.camLens.extrude(pMouse, pFrom, pTo)

            # Transform to global coordinates
            pFrom = render.getRelativePoint(base.cam, pFrom)
            pTo = render.getRelativePoint(base.cam, pTo)

            result = self.world.rayTestClosest(pFrom, pTo, BitMask32.bit(1))

            print(result.hasHit())
            print(result.getHitPos())
            print(result.getHitNormal())
            print(result.getHitFraction())
            print(result.getNode())
            #surface.set_shader_input("pos", result.getHitPos())

            #self.smiley.setPos(result.getHitPos() + Vec3(0,0,2))
            #self.move_node_smoothly(self.smiley, result.getHitPos() + Vec3(0,0,0.1), duration=0.5)

            groundSizeboundingbox=self.ground.getTightBounds()
            print(groundSizeboundingbox)
            pos=result.getHitPos()/abs(groundSizeboundingbox[0][0])
            self.ground.set_shader_input("pos", pos)
            #self.polygonpoints = []
            pos += Vec3(1, 1, 1)
            pos *= 0.5
            """ self.polygonpoints.insert(0, Vec2(pos.x, pos.y))
            if len(self.polygonpoints) > 6:
                self.polygonpoints.pop() """
            
            

            unitwidth=unit.unitWidth/abs(groundSizeboundingbox[0][0])/2

            unitheight=unit.unitHeight/abs(groundSizeboundingbox[0][1])/2

            unitrotation=unit.bodyNP.getH()

            unitposxy=Vec2(unit.bodyNP.getX()/abs(groundSizeboundingbox[0][0]), unit.bodyNP.getY()/abs(groundSizeboundingbox[0][1]))
            unitposxy += Vec2(1,1)
            unitposxy *= 0.5

            #unitposxy += Vec2(-math.cos(math.radians(unitrotation))*unitwidth*0.5, -math.sin(math.radians(unitrotation))*unitheight*0.5)
            
            unitposxy.x += -math.cos(math.radians(unitrotation))*unitwidth*0.5
            unitposxy.y += -math.sin(math.radians(unitrotation))*unitwidth*0.5

            #pos.x = -math.cos(math.radians(unitrotation))*unitwidth*0.5
            #pos.y = -math.sin(math.radians(unitrotation))*unitwidth*0.5 

            print(f"unitposxy: {unitposxy} smileypos: {unit.bodyNP.getPos()} groundbb: {groundSizeboundingbox}")

            move = int(self.unitToMove.unit.model.characteristics['M'])*2
            for rule in self.unitToMove.unit.model.special_rules:
                if rule.get('mountUnit'):
                    mountmove= int(rule['mountUnit'].model.characteristics['M'])*2
                    move = max(move, mountmove)
            print("Unit move:", move)
            

            self.polygonpoints = self.pointArc(origo=unitposxy, num_points=80, mouse_pos=Vec2(pos.x, pos.y),
                                               width=unitwidth,height=unitheight, rotationangle=unit.bodyNP.getH(),
                                               movedistance=move/(2*abs(groundSizeboundingbox[0][1])))
            #self.polygonpoints = self.mirrorPointArc(self.polygonpoints)

            
            #self.playerNP.setPos(result.getHitPos()+Vec3(10,10,0))
            #self.playerNP.node().setLinearMovement(Vec3(10,10,0), True)
            p1 = (self.polygonpoints[self.numsPoints-3]*2-1)*50
            p2 = (self.polygonpoints[self.numsPoints-2]*2-1)*50
            p3 = (self.polygonpoints[0]*2-1)*50
            p4 = (self.polygonpoints[self.numsPoints-1]*2-1)*50
            self.world.doPhysics(0.016)
            cont=self.world.rayTestClosest(Point3(p1.x, p1.y, 0.25), Point3(p2.x, p2.y, 0.25),BitMask32.allOn())
            
            #self.debug_ray(Point3(p1.x, p1.y, .9), Point3(p2.x, p2.y, .9))
            cont2=self.world.rayTestClosest(Point3(p3.x, p3.y, 0.25), Point3(p4.x, p4.y, 0.25),BitMask32.allOn())
            ve2 = ((Vec2(p2.x - p1.x, p2.y - p1.y)/50+1)*.5).normalized()
            ve4 = Vec2(p4.x - p3.x, p4.y - p3.y).normalized()
            print("Cont2:", cont2.hasHit(), cont2.getHitPos())
            print("Cont:", cont.hasHit(), cont.getHitPos())
            
            #closest_dist=0
            closest_dist = float('inf')
            closest_pos = None
            if cont.hasHit():
                # Check which contact point is closest to smiley
                
                
                hit_pos = cont.getHitPos()
                #dist = (hit_pos - self.smiley.getPos()).length()
                dist = (hit_pos - Point3(p1.x, p1.y, 0.1)).length()
                if dist < closest_dist:
                    closest_dist = dist
                    closest_pos = hit_pos
            if cont2.hasHit():
                # Check which contact point is closest to smiley
                hit_pos = cont2.getHitPos()
                dist = (hit_pos - Point3(p3.x, p3.y, 0.1)).length()
                if dist < closest_dist:
                    closest_dist = dist
                    closest_pos = hit_pos

            
            p1_5 = (p1 + p3) * 0.5
            p2_5 = (p2 + p4) * 0.5
            cont3=self.world.rayTestClosest(Point3(p1_5.x, p1_5.y, 0.25), Point3(p2_5.x, p2_5.y, 0.25),BitMask32.allOn())
            print("Cont3:", cont3.hasHit(), cont3.getHitPos())
            if cont3.hasHit():
                # Check which contact point is closest to smiley
                hit_pos = cont3.getHitPos()
                dist = (hit_pos - Point3(p1_5.x, p1_5.y, 0.1)).length()
                if dist < closest_dist:
                    closest_dist = dist
                    closest_pos = hit_pos

            if closest_pos:
                self.unitHitPos = closest_pos
                self.playerNP.setPos(closest_pos)

                newmove = closest_dist+math.radians(abs(self.arcPointRotation))*unit.unitWidth
                print("New move distance:", newmove, "closest dist:", closest_dist, "arc rotation:", self.arcPointRotation)
                self.polygonpoints = self.pointArc(origo=unitposxy, num_points=80, mouse_pos=Vec2(pos.x, pos.y),
                                                width=unitwidth,height=unitheight, rotationangle=unit.bodyNP.getH(),
                                                movedistance=newmove/(2*abs(groundSizeboundingbox[0][1])))

            self.ground.setShaderInput("polygonpoints", self.polygonpoints)
            self.ground.setShaderInput("isActive", True)
            
            return

    def debug_ray(self, pFrom, pTo):
        # Create a line to visualize the ray
        line = LineSegs()
        line.setColor(1, 0, 0, 1)  # Red line
        line.moveTo(pFrom)
        line.drawTo(pTo)
        
        line_node = render.attachNewNode(line.create())
        # Remove after 2 seconds
        #self.taskMgr.doMethodLater(2.0, line_node.removeNode, "remove-debug-ray")

    def drawRectangle(self, center=Point3(0, 0, 0), width=5, height=3, color=(1, 0, 0, 1)):
        """
        Draws a rectangle using LineSegs.
        
        Args:
            center: Center position of the rectangle
            width: Width of the rectangle
            height: Height of the rectangle
            color: Color of the rectangle as (r, g, b, a) tuple
        
        Returns:
            NodePath of the created rectangle
        """
        
        
        line_segs = LineSegs()
        line_segs.setColor(*color)
        line_segs.setThickness(2.0)
        
        # Calculate corner points
        half_width = width / 2
        half_height = height / 2
        
        corners = [
            Point3(center.x - half_width, center.y - half_height, center.z),  # Bottom-left
            Point3(center.x + half_width, center.y - half_height, center.z),  # Bottom-right
            Point3(center.x + half_width, center.y + half_height, center.z),  # Top-right
            Point3(center.x - half_width, center.y + half_height, center.z),  # Top-left
        ]
        
        # Draw the rectangle by connecting corners
        line_segs.moveTo(corners[0])
        for i in range(1, len(corners)):
            line_segs.drawTo(corners[i])
        # Close the rectangle
        line_segs.drawTo(corners[0])
        
        rectangle_node = line_segs.create()
        rectangleLine = render.attachNewNode(rectangle_node)
        rectangleLine.setName("Rectangle")
        
        return rectangleLine

    def moveUnit(self, unit):
        taskMgr.remove("taskLoopPathTowardsMouse")
        if unit.hasMovedThisTurn:
            print("Unit has already moved this turn.")
            return
        print("Moving unit to arc point...")
        pos = self.arcPoint
        print("Normalized position:", pos)
        pos=pos*2
        print("Moving unit to arc point:", self.arcPoint)
        pos -= Vec2(1,1)
        
        print("Normalized position:", pos)
        #pos.x *= abs(self.ground.getTightBounds()[0][0])
        pos.x *= 50
        pos.y *= 50
        print("Calculated position:", pos)
        oposUnit=unit.bodyNP.getPos()
        orotUnit=unit.bodyNP.getHpr()
        unit.bodyNP.setPos(pos.x , pos.y , 0)
        unit.bodyNP.setH(unit.bodyNP.getH() + self.arcPointRotation)
        unit.bodyNP.setPos(unit.bodyNPback.getPos(render))
        #self.checkUnitContact(unit)
        c = self.checkUnitContactSmall(unit)
        if c:
            taskMgr.add(self.chargeAndChargeReaction, "chargeAndChargeReactionTask",extraArgs=[unit, c,oposUnit, orotUnit],appendTask=True)
            #self.getFlankFromContact(unit, c)
        
        self.bakeTextures(self.ground)

    async def chargeAndChargeReaction(self,unit,c,oposUnit, orotUnit,task):
        chargeYesNo = ["Yes", "No"]
        self.cyn = Choice(chargeYesNo, Vec3(-20,0,10))
        self.cyn.ma = taskMgr.add(self.cyn.mouseActivate, "mouseActivateTask")
        self.ignore('mouse1')
        print("Waiting for choice...")
        await self.cyn.ma
        self.accept('mouse1', self.setActiveUnit,[self.taskLoopPathTowardsMouse, "taskLoopPathTowardsMouse"])
        print("event recieced")
        cynchoice = self.cyn.choice
        
        print('Event delivered with args:', self.cyn.choice)
        del self.cyn

        if cynchoice == "Yes":
            print("Charging into combat...")


            chargeReaction = ["hold", "flee"]
            self.cyn = Choice(chargeReaction, Vec3(20,0,10))
            self.cyn.ma = taskMgr.add(self.cyn.mouseActivate, "mouseActivateTask")
            self.ignore('mouse1')
            print("Waiting for choice...")
            await self.cyn.ma
            self.accept('mouse1', self.setActiveUnit,[self.taskLoopPathTowardsMouse, "taskLoopPathTowardsMouse"])
            print("event received")
            crchoice = self.cyn.choice
            
            print('Event delivered with args:', self.cyn.choice)
            del self.cyn
            defenderNP = render.find(f"**/{c.getNode1().getName()}")
            if crchoice == "hold":
                #chargeSequence = Sequence()

                print("Defender holds position.")
                
                angleToRotate = self.getFlankFromContact(unit, c)

                
                unit.hasMovedThisTurn=True

                unit.updateTextNode()
                unit.bodyNP.setCollideMask(BitMask32.bit(4))
                taskMgr.add(self.chargeInterval,"chargeIntervalTask", extraArgs=[unit, defenderNP, angleToRotate,oposUnit, orotUnit], appendTask=False)
            elif crchoice == "flee":
                angleToRotate = self.getFlankFromContact(unit, c)
                print("Defender flees!")
                taskMgr.add(self.fleeInterval,"fleeIntervalTask", extraArgs=[unit, defenderNP, angleToRotate,oposUnit, orotUnit], appendTask=False)
                fleeDirection = defenderNP.getPos() - unit.bodyNP.getPos()
                storeRotation = defenderNP.getHpr()
                defenderNP.lookAt(defenderNP.getPos() + fleeDirection)
                fleeRotation = defenderNP.getHpr()
                defenderNP.setHpr(storeRotation)

                
        else:
            print("Charge cancelled.")
            unit.bodyNP.setPos(oposUnit)
            unit.bodyNP.setHpr(orotUnit)
            self.startTaskFunction(self.taskLoopPathTowardsMouse, "taskLoopPathTowardsMouse")
            
        return task.done

    def checkUnitContactSmall(self, unit):
        contacts = self.world.contactTest(unit.bodyNP.node(), BitMask32.allOn())
        for contact in contacts.getContacts():
            print("Contact with:", contact.getNode0().getName(), contact.getNode1().getName())
            
            mpoint = contact.getManifoldPoint()
            print(mpoint.getDistance())
            print(mpoint.getAppliedImpulse())
            print(mpoint.getPositionWorldOnA())
            print(mpoint.getPositionWorldOnB())
            print(mpoint.getLocalPointA())
            print(mpoint.getLocalPointB())
            if 'UnitCollision-' in contact.getNode1().getName():
                print("Unit collision detected!")
                return contact
        return None
    
    async def fleeInterval(self, unit, defenderNP, angleToRotate,oposUnit, orotUnit):
        self.terningerCharge=[]
        for i in range(2):
            terning = Dice(self.world, position=Vec3(-20+i*2,0,10), size=1.0)
            self.terningerCharge.append(terning)
        for terning in self.terningerCharge:
            terning.roll()
        chtask=taskMgr.add(checkDice, "checkDiceTaskCharge", extraArgs=[self.terningerCharge], appendTask=True)
        
        

        self.terningerFlee=[]
        for i in range(2):
            terning = Dice(self.world, position=Vec3(20+i*2,0,10), size=1.0)
            self.terningerFlee.append(terning)
        for terning in self.terningerFlee:
            terning.roll()
        await taskMgr.add(checkDice, "checkDiceTaskFlee", extraArgs=[self.terningerFlee], appendTask=True)
        await chtask
        chdice = []
        for terning in self.terningerCharge:
            chdice.append(terning.currentValue)
        print("Charge dice results:", chdice)
        fldice = []
        for terning in self.terningerFlee:
            fldice.append(terning.currentValue)
        print("Flee dice results:", fldice)
        contactPos=unit.bodyNP.getPos()
        contactRot=unit.bodyNP.getHpr()
        self.zax = loader.loadModel("models/zup-axis")
        self.z2= loader.loadModel("models/zup-axis")
        self.z2.reparentTo(render)
        self.z2.setPos(oposUnit)
        #self.zax.reparentTo(render)
        
        shape = unit.bodyNP.node().getShape(0)
        if isinstance(shape, BulletBoxShape):
            half_extents = shape.getHalfExtentsWithMargin()
            width = half_extents.x 
            height = half_extents.y 
            print(f"Defender unit width: {width}")
        parent = unit.bodyNP.getParent()
        newnode = render.attachNewNode(f"Temp-{unit.unitName}")
        unit.bodyNP.setPos(oposUnit)
        unit.bodyNP.setHpr(orotUnit)
        #unit.bodyNP.setHpr(Vec3(0,0,0))

        newnode.reparentTo(unit.bodyNP)

        rot = LRotationf()
        rot.setHpr(unit.bodyNP.getHpr())
        #fwd=rot.getForward()
        rgt = rot.getRight()
        dire = contactPos - oposUnit
        angle_between = rgt.dot(dire.normalized())
        if angle_between >=0:
            sign = 1
        else:
            sign = -1
        
        #newnode.setPos(unit.bodyNP.getPos()+Vec3(width,height,0))
        newnode.setPos(Vec3(width*sign/unit.bodyNP.getScale().x,height/unit.bodyNP.getScale().y,0))
        newnode.wrtReparentTo(render)

        unit.bodyNP.setHpr(orotUnit)
        unit.bodyNP.wrtReparentTo(newnode)
        self.zax.reparentTo(newnode)
        # Rotate the new node smoothly to align with defender
        print("rotate from to",newnode.getHpr(), contactRot)
        newnode_hpr = newnode.getHpr()
        # Ensure all angles are positive (0-360 range)
        positive_h = newnode_hpr.x % 360
        positive_p = newnode_hpr.y % 360  
        positive_r = newnode_hpr.z % 360

        if positive_h > 180:
            positive_h -= 360

        if positive_h - contactRot.x > 180:
            contactRot = Vec3(contactRot.x + 360, contactRot.y, contactRot.z)
        newnode.setHpr(positive_h, positive_p, positive_r)

        print("rotate from to",newnode.getHpr(), contactRot)

        wheel1Angle = contactRot.x - orotUnit.x
        print("wheel1Angle:", wheel1Angle)
        newnode.setHpr(contactRot)
        #wheel1Pos = unit.bodyNP.getPos()
        wheel1Pos = newnode.getPos(render)
        


        if 1:
            # Calculate distance to move forward to reach contactPos
            #current_pos = unit.bodyNP.getPos(render)
            direction = self.playerNP.getPos() - wheel1Pos
            

            #math.radians(positive_h)*width

            wdistance = abs(math.radians(wheel1Angle)*width*2)
            #cdistance = direction.length()
            cdistance = self.moveArceDistance - wdistance

            #return distance  # Add a small buffer
            print ("Calculated distance to move forward:", cdistance,wdistance,width*2)

        chdist = int(unit.unit.model.characteristics['M']) + max(chdice)
        for rule in unit.unit.model.special_rules:
            if rule.get('mountUnit'):
                chdist= int(rule['mountUnit'].model.characteristics['M'])+ max(chdice)
        fldist = sum(fldice)
        print("Charge distance:", chdist)
        print("Flee distance:", fldist)
        if chdist < wdistance:
            angle = math.degrees(chdist/width)
            contactRot = Vec3(orotUnit.x + angle, contactRot.y, contactRot.z)*wheel1Angle/abs(wheel1Angle)



        
        newnode.setHpr(positive_h, positive_p, positive_r)

        rotation_interval = LerpPosHprInterval(
            newnode, 
            duration=1.5, 
            pos=newnode.getPos(),
            hpr=contactRot,
            blendType='easeInOut'
        )
        await rotation_interval
        if chdist < wdistance:
            unit.bodyNP.wrtReparentTo(parent)
            """ for terning in self.terningerCharge:
                terning.remove(self.world) """
            #return
        #unit.bodyNP.wrtReparentTo(parent)
        ocdistance=cdistance
        if chdist < wdistance+cdistance:
            cdistance = chdist - wdistance
            cdistance = max(cdistance, 0)

        angle = contactRot.x
        vector = Vec2(-math.sin(math.radians(angle)), math.cos(math.radians(angle)))
        print((contactPos - newnode.getPos()).normalized()*cdistance)
        print(cdistance,"aplpied to vector:", Vec3(vector.x, vector.y, 0),wdistance, chdist)
        cmove = chdist - wdistance
        pos_interval = LerpPosHprInterval(
            #unit.bodyNP, 
            newnode,
            duration=1.5, 
            #pos=contactPos,
            #pos=contactPos-direction.normalized()*3,
            #pos=wheel1Pos + direction.normalized()*(cdistance+wdistance),
            #pos=wheel1Pos + direction.normalized(),
            #pos=wheel1Pos + direction.normalized()*cdistance,
            pos=wheel1Pos + Vec3(vector.x, vector.y, 0)*cmove,
            #pos=ppp,
            hpr=contactRot,
            blendType='easeInOut'
        )
        

        #await pos_interval
        #unit.bodyNP.wrtReparentTo(parent)
        
        

        rotation_interval = LerpPosHprInterval(
            defenderNP, 
            duration=0.5, 
            pos=defenderNP.getPos(),
            hpr=contactRot,
            blendType='easeInOut'
        )
        await rotation_interval
        """ angle = contactRot.x
        vector = Vec2(-math.sin(math.radians(angle)), math.cos(math.radians(angle)))

        pos_interval = LerpPosHprInterval(
            unit.bodyNP, 
            duration=1.5, 
            pos=unit.bodyNP.getPos() + Vec3(vector.x, vector.y, 0)*20,
            hpr=contactRot,
            blendType='easeInOut'
        )
        """
        #await pos_interval
        angle = contactRot.x
        vector = Vec2(-math.sin(math.radians(angle)), math.cos(math.radians(angle)))
        pos_interval2 = LerpPosHprInterval(
            defenderNP, 
            duration=1.5, 
            pos=defenderNP.getPos() + Vec3(vector.x, vector.y, 0)*fldist,
            hpr=contactRot,
            blendType='easeInOut'
        )

        #await pos_interval2

        par = Parallel(
            pos_interval,
            pos_interval2
        )

        await par
        for terning in self.terningerCharge:
                terning.remove(self.world)
        for terning in self.terningerFlee:
                terning.remove(self.world)
        unit.bodyNP.wrtReparentTo(parent)
        
        defenderUnit=self.getSelectedUnit(defenderNP.node())
        print("Checking if fleeing unit is caught...")
        print(defenderNP.getCollideMask())
        print(unit.bodyNP.getCollideMask())
        cont = self.checkUnitContactSmall(unit)
        if cont:
            print("Fleeing Unit caught, and are slayed!")
            self.world.removeRigidBody(defenderUnit.bodyNP.node())
            defenderUnit.model.removeNode()
            defenderUnit.bodyNP.removeNode()
            self.units.remove(defenderUnit)



        return 
    
    async def chargeInterval(self, unit, defenderNP, angleToRotate,oposUnit, orotUnit):
        self.terninger=[]
        for i in range(2):
            terning = Dice(self.world, position=Vec3(0,0,10), size=1.0)
            self.terninger.append(terning)
        for terning in self.terninger:
            terning.roll()
        await taskMgr.add(checkDice, "checkDiceTask", extraArgs=[self.terninger], appendTask=True)
        
        chdice = []
        for terning in self.terninger:
            chdice.append(terning.currentValue)
        print("Charge dice results:", chdice)
        contactPos=unit.bodyNP.getPos()
        contactRot=unit.bodyNP.getHpr()
        self.zax = loader.loadModel("models/zup-axis")
        self.z2= loader.loadModel("models/zup-axis")
        self.z2.reparentTo(render)
        self.z2.setPos(oposUnit)
        #self.zax.reparentTo(render)
        
        shape = unit.bodyNP.node().getShape(0)
        if isinstance(shape, BulletBoxShape):
            half_extents = shape.getHalfExtentsWithMargin()
            width = half_extents.x 
            height = half_extents.y 
            print(f"Defender unit width: {width}")
        parent = unit.bodyNP.getParent()
        newnode = render.attachNewNode(f"Temp-{unit.unitName}")
        unit.bodyNP.setPos(oposUnit)
        unit.bodyNP.setHpr(orotUnit)
        #unit.bodyNP.setHpr(Vec3(0,0,0))

        newnode.reparentTo(unit.bodyNP)

        rot = LRotationf()
        rot.setHpr(unit.bodyNP.getHpr())
        #fwd=rot.getForward()
        rgt = rot.getRight()
        dire = contactPos - oposUnit
        angle_between = rgt.dot(dire.normalized())
        if angle_between >=0:
            sign = 1
        else:
            sign = -1
        
        #newnode.setPos(unit.bodyNP.getPos()+Vec3(width,height,0))
        newnode.setPos(Vec3(width*sign/unit.bodyNP.getScale().x,height/unit.bodyNP.getScale().y,0))
        newnode.wrtReparentTo(render)

        unit.bodyNP.setHpr(orotUnit)
        unit.bodyNP.wrtReparentTo(newnode)
        self.zax.reparentTo(newnode)
        # Rotate the new node smoothly to align with defender
        print("rotate from to",newnode.getHpr(), contactRot)
        newnode_hpr = newnode.getHpr()
        # Ensure all angles are positive (0-360 range)
        positive_h = newnode_hpr.x % 360
        positive_p = newnode_hpr.y % 360  
        positive_r = newnode_hpr.z % 360

        if positive_h > 180:
            positive_h -= 360
        
        if positive_h < 0:
            positive_h += 360

        if positive_h - contactRot.x > 180:
            contactRot = Vec3(contactRot.x + 360, contactRot.y, contactRot.z)

        newnode.setHpr(positive_h, positive_p, positive_r)
        print("rotate from to",newnode.getHpr(), contactRot)

        wheel1Angle = contactRot.x - orotUnit.x
        print("wheel1Angle:", wheel1Angle)
        newnode.setHpr(contactRot)
        #wheel1Pos = unit.bodyNP.getPos()
        wheel1Pos = newnode.getPos(render)
        


        if 1:
            # Calculate distance to move forward to reach contactPos
            #current_pos = unit.bodyNP.getPos(render)
            direction = self.playerNP.getPos() - wheel1Pos
            

            #math.radians(positive_h)*width

            wdistance = abs(math.radians(wheel1Angle)*width*2)
            #cdistance = direction.length()
            cdistance = self.moveArceDistance - wdistance

            #return distance  # Add a small buffer
            print ("Calculated distance to move forward:", cdistance,wdistance,width*2)

        chdist = int(unit.unit.model.characteristics['M']) + max(chdice)
        for rule in unit.unit.model.special_rules:
            if rule.get('mountUnit'):
                chdist= int(rule['mountUnit'].model.characteristics['M'])+ max(chdice)
        print("Charge distance:", chdist)
        if chdist < wdistance:
            angle = math.degrees(chdist/width)
            contactRot = Vec3(orotUnit.x + angle, contactRot.y, contactRot.z)*wheel1Angle/abs(wheel1Angle)



        
        newnode.setHpr(positive_h, positive_p, positive_r)

        rotation_interval = LerpPosHprInterval(
            newnode, 
            duration=0.5, 
            pos=newnode.getPos(),
            hpr=contactRot,
            blendType='easeInOut'
        )
        await rotation_interval
        if chdist < wdistance:
            unit.bodyNP.wrtReparentTo(parent)
            for terning in self.terninger:
                terning.remove(self.world)
            print("Charge distance less than wheel distance, returning.")
            return
        #unit.bodyNP.wrtReparentTo(parent)
        ocdistance=cdistance
        if chdist < wdistance+cdistance:
            cdistance = chdist - wdistance

        
        angle = contactRot.x
        vector = Vec2(-math.sin(math.radians(angle)), math.cos(math.radians(angle)))
        print((contactPos - newnode.getPos()).normalized()*cdistance)
        cmove = min(chdist, cdistance)
        pos_interval = LerpPosHprInterval(
            #unit.bodyNP, 
            newnode,
            duration=0.5, 
            #pos=contactPos,
            #pos=contactPos-direction.normalized()*3,
            #pos=wheel1Pos + direction.normalized()*(cdistance+wdistance),
            #pos=wheel1Pos + direction.normalized(),
            #pos=wheel1Pos + direction.normalized()*cdistance,
            pos=wheel1Pos + Vec3(vector.x, vector.y, 0)*cmove,
            #pos=ppp,
            hpr=contactRot,
            blendType='easeInOut'
        )
        

        await pos_interval
        unit.bodyNP.wrtReparentTo(parent)
        if chdist < self.moveArceDistance:
            #unit.bodyNP.setCollideMask(BitMask32.bit(unit.bitmask))
            for terning in self.terninger:
                terning.remove(self.world)
            print("Charge distance less than total distance, returning.", chdist, wdistance,ocdistance, self.moveArceDistance)
            return
        parent = unit.bodyNP.getParent()
        newnode = render.attachNewNode(f"Temp-{unit.unitName}")
        newnode.setPos(self.playerNP.getPos())
        newnode.setHpr(unit.bodyNP.getHpr())
        unit.bodyNP.wrtReparentTo(newnode)
        # Rotate the new node smoothly to align with defender

        
        finalHpr = (newnode.getH() + angleToRotate, newnode.getP(), newnode.getR())
        print("Final HPR:", finalHpr)
        print("Angle to rotate:", angleToRotate)
        print("Current HPR before final rotation:", newnode.getHpr())
        rotation_interval = LerpPosHprInterval(
            newnode, 
            duration=0.5, 
            pos=newnode.getPos(),
            #hpr=(newnode.getH() + angleToRotate, newnode.getP(), newnode.getR()),
            hpr=finalHpr,
            blendType='easeInOut'
        )
        await rotation_interval
        unit.bodyNP.wrtReparentTo(parent)
        newnode.removeNode()
        """ sequence = Sequence(
            rotation_interval,
            Func(unit.bodyNP.wrtReparentTo, parent),
            Func(newnode.removeNode)
            #Func(self.verySimpleBattle, unit.bodyNP, defenderNP, "front")
        ) """


        unit.isInCombat=True
        #unit.bodyNP.setCollideMask(BitMask32.bit(4))
        
        defenderUnit=self.getSelectedUnit(defenderNP.node())
        unit.isInCombatWith.append(defenderUnit)
        unit.isInCombatFlank.append("front")
        defenderUnit.isInCombatWith.append(unit)
        defenderUnit.isInCombat=True
        defenderUnit.isInCombatFlank.append("front")
        #defenderUnit.bodyNP.setCollideMask(BitMask32.bit(4))
        unit.updateTextNode()
        defenderUnit.updateTextNode()
        for terning in self.terninger:
            terning.remove(self.world)
        del self.terninger
        return 

    def getFlankFromContact(self, unit, contact):
        print("Unit collision detected!")
        # Handle unit collision (e.g., stop movement, apply damage, etc.)
        angleAttacker = unit.bodyNP.getH()
        defenderNP = render.find(f"**/{contact.getNode1().getName()}")
        angleDefender = defenderNP.getH()
        print(f"contact position in defender coordsystem: {self.playerNP.getPos(defenderNP)}")
        hitloc = self.playerNP.getPos(defenderNP) 

        shape = contact.getNode1().getShape(0)
        if isinstance(shape, BulletBoxShape):
            half_extents = shape.getHalfExtentsWithMargin()
            width = half_extents.x * unit.bodyNP.getScale().x 
            height = half_extents.y * unit.bodyNP.getScale().y 
            print(f"Defender unit width: {width}, height: {height}")

        angleToRotate = angleDefender - angleAttacker
        print(f"Attacker angle: {angleAttacker}, Defender angle: {angleDefender}")
        print(f"Rotating attacker by {angleToRotate} degrees to face defender.")
        angleToRotate = (angleToRotate ) % 360   # Normalize to [-180, 180]
        print(f"normalized {angleToRotate} degrees to face defender.")
        print(f"Hit location in defender coords: {hitloc}")
        """ rot = LRotationf()
        rot.setHpr(defenderNP.getHpr())
        fwd=rot.getForward()
        fwd2 = Vec2(fwd.x, fwd.y)
        print("Defender forward vector:", fwd)
        vec = hitloc - defenderNP.getPos()
        vec2 = Vec2(vec.x, vec.y)
        print("Vector from defender to hit location:", vec) """
        #angle_between = math.acos(fwd2.normalized().dot(vec2.normalized())) * (180.0 / math.pi)
        angle_between = math.acos(Vec3(0,1,0).dot(hitloc.normalized())) * (180.0 / math.pi)
        print("Angle between forward and hit location vector:", angle_between)
        frontArcAngle = 90 - math.atan2(height, width) * (180.0 / math.pi)
        print("Front arc angle:", frontArcAngle)

        #if abs(hitloc.x*unit.bodyNP.getScale().x - width) < .03:
        if angle_between > frontArcAngle and hitloc.x > 0:
            print("Hit on right side of defender")
            print(f"Initial angle to rotate: {angleToRotate}")
            #angleToRotate = (angleToRotate + 90) % 360 - 180
            if angleToRotate < 0:
                angleToRotate += 90
            if angleToRotate > 90:
                angleToRotate = 360-90- angleToRotate
                angleToRotate *= -1
            
            print(f"Adjusted angle to rotate: {angleToRotate}")
        #elif abs(hitloc.x*unit.bodyNP.getScale().x + width) < .03:
        elif angle_between > frontArcAngle and hitloc.x < 0:
            print("Hit on left side of defender")
            print(f"Initial angle to rotate: {angleToRotate}")
            #angleToRotate = (angleToRotate - 90) % 360 - 180
            if angleToRotate > 90:
                angleToRotate -= 90
            else:
                angleToRotate = 90 - angleToRotate
                angleToRotate *= -1
            print(f"Adjusted angle to rotate: {angleToRotate}")
        #elif abs(hitloc.y*unit.bodyNP.getScale().y - height) < .03:
        elif angle_between < frontArcAngle:
            print("Hit front side of defender")
            print(f"Initial angle to rotate: {angleToRotate}")
            #angleToRotate = (angleToRotate + 180) % 180
            if angleToRotate > 90:
                angleToRotate -= 180
                #angleToRotate *= -1
            print(f"Adjusted angle to rotate: {angleToRotate}")
        #elif abs(hitloc.y*unit.bodyNP.getScale().y + height) < .03:
        elif angle_between > frontArcAngle+90:
            print("Hit rear side of defender")
            print(f"Initial angle to rotate: {angleToRotate}")
            #angleToRotate = (angleToRotate + 180) % 90
            
            if angleToRotate > 90:
                #angleToRotate -= 90
                angleToRotate = (360 -angleToRotate) * -1
                #angleToRotate *= -1
            """ if angleToRotate < -90:
                angleToRotate += 90
                angleToRotate *= -1 """
            
        else:
            print("Hit i dont know where")
        return angleToRotate
    
    

    

    def printBattleResults(self, attackerUnit, defenderUnit, attacks, total_hits, suffered_wounds,  saves_made, total_wounds):
        print(f"Battle results for {attackerUnit.unit.name} attacking with weapon {attackerUnit.unit.model.equipedWeapon.get('name')}:")
        print(f"Total hits by {attackerUnit.unit.name} on {defenderUnit.unit.name}: {total_hits}")
        print(f"suffered wounds by {attackerUnit.unit.name} on {defenderUnit.unit.name}: {suffered_wounds}")
        print(f"Saves made by {defenderUnit.unit.name}: {saves_made}")
        print(f"Total wounds by {attackerUnit.unit.name} on {defenderUnit.unit.name}: {total_wounds}")

    def waitForChoice(self, choice,function,task):
        pass

    async def verySimpleBattleStart(self,task):
        #await messenger.future("choice-made")
        #await self.weaponChoise.helper1.messenger.future("choice-made")
        #await messenger.future("mouse1")
        weps =self.unitToMove.unit.model.weapons
        self.weaponChoise = Choice(weps, Vec3(0,0,10))
        self.weaponChoise.ma = taskMgr.add(self.weaponChoise.mouseActivate, "mouseActivateTask")
        self.ignore('mouse1')
        print("Waiting for choice...")
        await self.weaponChoise.ma
        self.accept('mouse1', self.setActiveUnit,[self.taskStartCombat, "taskStartCombat"])
        print("event recieced")
        wepchoice = self.weaponChoise.choice
        self.unitToMove.unit.model.equip_weapon(wepchoice)
        print('Event delivered with args:', self.weaponChoise.choice)
        del self.weaponChoise
        #messenger.send("start-attack-sequence")
        taskMgr.add(self.verySimpleBattle, "verySimpleBattleTask")
        return task.done
    
    async def makeChoice(self, choice):
        choice.ma = taskMgr.add(choice.mouseActivate, "mouseActivateTask")
        self.ignore('mouse1')
        print("Waiting for choice...")
        await choice.ma
        self.accept('mouse1', self.setActiveUnit,[self.taskStartCombat, "taskStartCombat"])
        print("event recieced")
        selected_choice = choice.choice
        print('Event delivered with args:', choice.choice)
        return

    async def verySimpleBattle(self,task):
        print("Starting very simple battle...")
        attacker = self.unitToMove.bodyNP
        defender = self.unitToMove.isInCombatWith[0].bodyNP
        flank = self.unitToMove.isInCombatFlank[0]
        engagedWith = [x.unitName for x in self.unitToMove.isInCombatWith]
        print("Attacker:", attacker.node().getName())
        print("engaged in battle with:", engagedWith)
        print("on flanks:", self.unitToMove.isInCombatFlank)
        choice = Choice(engagedWith, Vec3(0,0,10))
        battleChoice = taskMgr.add(self.makeChoice, "makeChoiceTask", extraArgs=[choice], appendTask=False)
        await battleChoice
        selected_choice = choice.choice
        del choice
        print(f"Selected choice: {selected_choice}")
        for unit in self.unitToMove.isInCombatWith:
            if unit.unitName == selected_choice:
                defender = unit.bodyNP
                break
        print(f"{attacker.node().getName()} attacks {defender.node().getName()} in {flank}!")
        attackerUnit=self.getSelectedUnit(attacker.node())
        defenderUnit=self.getSelectedUnit(defender.node())
        defender_nmodels = defenderUnit.unit.nmodels
        print(f"{attackerUnit.unit.name} attacks {defenderUnit.unit.name} in {flank}!")
        if attackerUnit.unit.model.equipedWeapon.get('tag') == 'ranged':
            print("attacker unit has ranged weapon equiped, switch to melee weapon for combat.")
            attackerUnit.unit.model.equip_weapon('hand weapon')
        if defenderUnit.unit.model.equipedWeapon.get('tag') == 'ranged':
            print("defender unit has ranged weapon equiped, switch to melee weapon for combat.")
            defenderUnit.unit.model.equip_weapon('hand weapon')

        apos=attacker.getPos()
        back_int = LerpPosHprInterval(
            attacker,
            duration=0.5,
            pos=attacker.getPos() - (defender.getPos() - attacker.getPos()).normalized() * 2,
            hpr=attacker.getHpr(),
            blendType='easeInOut'
        )
        forward_int = LerpPosHprInterval(
            attacker,
            duration=0.5,
            pos=apos,
            hpr=attacker.getHpr(),
            blendType='easeInOut'
        )
        self.attackSequence = Sequence(
            back_int,
            forward_int
        )
        

        attacks, total_hits, suffered_wounds,  saves_made, total_wounds = simulate_battle(attackerUnit.unit, defenderUnit.unit,charge=True)
        self.printBattleResults(attackerUnit, defenderUnit, attacks, total_hits, suffered_wounds,  saves_made, total_wounds)
        defenderUnit.unit.nmodels-=total_wounds
        attacker_score = total_wounds
        #self.removeModelsFromUnit(defenderUnit,total_wounds)
        self.attackSequence.append(Func(self.removeModelsFromUnit, defenderUnit, total_wounds))

        for unit in self.unitToMove.isInCombatWith:
            defender=unit.bodyNP
            defenderUnit=self.getSelectedUnit(defender.node())
            apos=defender.getPos()
            back_int = LerpPosHprInterval(
                defender,
                duration=0.5,
                pos=defender.getPos() - (attacker.getPos() - defender.getPos()).normalized() * 2,
                hpr=defender.getHpr(),
                blendType='easeInOut'
            )
            forward_int = LerpPosHprInterval(
                defender,
                duration=0.5,
                pos=apos,
                hpr=defender.getHpr(),
                blendType='easeInOut'
            )
            self.attackSequence.append(back_int)
            self.attackSequence.append(forward_int)

            attacks, total_hits, suffered_wounds,  saves_made, total_wounds = simulate_battle(defenderUnit.unit, attackerUnit.unit,charge=False)
            self.printBattleResults(defenderUnit, attackerUnit, attacks, total_hits, suffered_wounds,  saves_made, total_wounds)
            attackerUnit.unit.nmodels-=total_wounds
            #self.removeModelsFromUnit(attackerUnit,total_wounds)
            self.attackSequence.append(Func(self.removeModelsFromUnit, attackerUnit, total_wounds))
        
        defender_score = total_wounds
        #defenderUnit.unit.nmodels=defender_nmodels
        print(f"Attacker score: {attacker_score}, Defender score: {defender_score}")
        await self.attackSequence
        self.attackSequence2 = Sequence()
        if attacker_score-99 < defender_score:
            #attacker loses
            #self.attackSequence.append(Func(self.fallBack, defender, attacker))
            loserUnit = attackerUnit
            winnerUnit = defenderUnit
            #direction=self.fleeDirectionMultUnits(attackerUnit,[self.getSelectedUnit(u.bodyNP.node()) for u in attackerUnit.isInCombatWith])
            #self.attackSequence2.append(Func(self.fallBack, attacker,direction))
        else:
            #defender loses
            loserUnit = defenderUnit
            winnerUnit = attackerUnit
            #direction=self.fleeDirectionMultUnits(defenderUnit,[self.getSelectedUnit(u.bodyNP.node()) for u in defenderUnit.isInCombatWith])
            #self.attackSequence.append(Func(self.fallBack, defender,direction))
            #self.attackSequence.append(Func(self.fallBack, attacker, defender))
        
        direction=self.fleeDirectionMultUnits(loserUnit,[self.getSelectedUnit(u.bodyNP.node()) for u in loserUnit.isInCombatWith])
        persuitDiceTasks=[]
        persuitDiceDices=[]
        persuingUnit=[]
        
        
        for unit in loserUnit.isInCombatWith:
            persuitOrNot = [unit.unitName+'\nPersuit', unit.unitName+'\nRestrain']
            choice = Choice(persuitOrNot, Vec3(0,0,10))
            battleChoice = taskMgr.add(self.makeChoice, "makeChoiceTask", extraArgs=[choice], appendTask=False)
            await battleChoice
            selected_choice = choice.choice
            del choice
            print(f"Selected choice: {selected_choice}")
            if selected_choice == persuitOrNot[0]:
                print(f"{defenderUnit.unit.name} chooses to pursue!")
                #pursuit
                #self.attackSequence2.append(Func(self.fallBack, unit.bodyNP,direction))
                persuingUnit.append(unit)
                """ 
                terningerPersuit=[]
                for i in range(2):
                    terning = Dice(self.world, position=unit.bodyNP.getPos() + Vec3(-20+i*4,0,10), size=1.0)
                    terningerPersuit.append(terning)
                
                persuitDiceDices.append(terningerPersuit)
                """


        

        persuingUnit.append(loserUnit)

        for i, unit in enumerate(persuingUnit):
            terningerPersuit=[]
            for j in range(2):
                terning = Dice(self.world, position=unit.bodyNP.getPos() + Vec3(-20+j*4,0,10), size=1.0)
                terningerPersuit.append(terning)
            for terning in terningerPersuit:
                terning.roll()
        

            persuitDiceTasks.append(taskMgr.add(checkDice, "checkDiceTaskPersuit"+str(loserUnit.unitName), extraArgs=[terningerPersuit], appendTask=True))
            persuitDiceDices.append(terningerPersuit)


        for task in persuitDiceTasks:
            await task
        for i, unit in enumerate(persuingUnit):
            persuitDices = persuitDiceDices[i]
            persuit_results = [terning.currentValue for terning in persuitDices]
            persuit_score = sum(persuit_results)
            print(f"Persuit dice results for {unit.unit.name}: {persuit_results}, total: {persuit_score}")
            
            print(f"{unit.unit.name} successfully pursues the fleeing unit!")
            self.attackSequence2.append(Func(self.fallBack, unit.bodyNP,direction,length=persuit_score*1.0))
            
        for dices in persuitDiceDices:
            for terning in dices:
                terning.remove(self.world)


        #self.attackSequence2.append(Func(self.fallBack, loserUnit.bodyNP,direction))
        if not self.attackSequence.isPlaying():
            self.attackSequence2.start()
    


    def fleeDirectionMultUnits(self, loser,winners):
        winPos=Vec3(0,0,0)
        lPos=loser.bodyNP.getPos()
        for w in winners:
            winPos+=w.bodyNP.getPos()
        winPos=winPos/len(winners)
        ldir=[]
        for w in winners:
            dir = (lPos - w.bodyNP.getPos()).normalized()
            ldir.append(dir)
        finalDir=Vec3(0,0,0)
        for d in ldir:
            finalDir+=d
        return finalDir.normalized()


    
    def centerOfModels(self, unit):
        coords=[]
        for child in unit.model.getChildren():
            coords.append(child.getPos(render))
        
        min_x = min(pos.x for pos in coords)
        max_x = max(pos.x for pos in coords)
        min_y = min(pos.y for pos in coords)
        max_y = max(pos.y for pos in coords)
        min_z = min(pos.z for pos in coords)
        max_z = max(pos.z for pos in coords)

        center = Point3((min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2)
        return center

    def getCenterOfUnit(self, unit):
        bounds = unit.model.getTightBounds()
        center = (bounds[0] + bounds[1]) * 0.5
        # Convert from local model coordinates to world coordinates
        world_center = unit.model.getPos(render) + center
        return world_center
        
    def removeModelsFromUnit(self, unit,models_to_remove):
        #unit.unit.nmodels = max(0, unit.nmodels - num_models)
        cildren = unit.model.getChildren()
        models_to_remove = min(len(cildren), models_to_remove)
        #unit.model.ls()
        for i in range(models_to_remove):
            #cildren[-1*(i+1)].removeNode()
            cildren = unit.model.getChildren()
            print(f"Removing model {cildren[-1].getName()} from unit {unit.unit.name}")
            cildren[-1].removeNode()
        
        cildren = unit.model.getChildren()
        if len(cildren) == 0:
            print(f"All models removed from unit {unit.unit.name}. Removing unit from game.")
            if self.attackSequence.isPlaying():
                print("Pausing attack sequence")
                self.attackSequence.pause()
                print(self.attackSequence.isPlaying())
            #self.attackSequence.finish()
            self.world.removeRigidBody(unit.bodyNP.node())
            unit.bodyNP.removeNode()
            unit.model.removeNode()
            self.units.remove(unit)
            
            return
        self.world.removeRigidBody(unit.bodyNP.node())
        for shape in unit.bodyNP.node().shapes:
            unit.bodyNP.node().removeShape(shape)
        bounds = unit.model.getTightBounds()
        box_size = bounds[1] - bounds[0]
        shape = BulletBoxShape(box_size * 0.5)  # BulletBoxShape takes half-extents
        #body = BulletRigidBodyNode('UnitCollision-' + self.unitName)
        unit.bodyNP.node().addShape(shape)
        unit.bodyNP.node().setMass(0)  # Static object
        self.world.attachRigidBody(unit.bodyNP.node())
        
        
        unit.model.setPos(0,0,0)
        
        unit.model.setPos(-box_size.x/2+unit.modelWidth/2, box_size.y/2-unit.modelHeight/2,0)


        
        #unit.setUpCollisions()

    def fallBack(self, loser,direction,length=10.0):
        if loser.isEmpty():
            print("looser is destryed, no fallback")
            w=self.getSelectedUnit(winner.node())
            w.isInCombat=False
            w.isInCombatWith=[]
            w.isInCombatFlank=[]
            #TODO: winner overruns
            return
        
        print(f"{loser.node().getName()} falls back!")

        loserPos=loser.getPos()
        #direction = (loserPos - winnerPos).normalized()
        #length = 15.0  # Adjust as needed
        newPos = loserPos + direction * length
        oldHpr=loser.getHpr()
        loser.lookAt(loserPos + direction)
        fleeHpr=loser.getHpr()
        newHpr=loser.getHpr()
        loser.setHpr(oldHpr)
        rotate_interval = LerpPosHprInterval(
            loser, 
            duration=0.5, 
            pos=loser.getPos(),
            hpr=fleeHpr,
            blendType='easeInOut'
        )
        move_interval = LerpPosInterval(
            loser, 
            duration=1.0, 
            pos=newPos,
            blendType='easeInOut'
        )
        rotate_interval2 = LerpPosHprInterval(
            loser, 
            duration=0.5, 
            pos=newPos,
            hpr=newHpr,
            blendType='easeInOut'
        )
        sequence = Sequence(
            rotate_interval,
            move_interval,
            rotate_interval2,
            #Func(self.persuitMove, winner, loser)
        )
        sequence.start()

    def fallBack2(self, winner, loser):
        if loser.isEmpty():
            print("looser is destryed, no fallback")
            w=self.getSelectedUnit(winner.node())
            w.isInCombat=False
            w.isInCombatWith=[]
            w.isInCombatFlank=[]
            #TODO: winner overruns
            return
        if winner is None or loser is None or winner.isEmpty() or loser.isEmpty():
            print("Error: winner or loser is None or empty.")
            return
        print(f"{winner.node().getName()} is victorious!")
        print(f"{loser.node().getName()} falls back!")

        winnerPos=winner.getPos()
        loserPos=loser.getPos()
        direction = (loserPos - winnerPos).normalized()
        fallback_distance = 15.0  # Adjust as needed
        newPos = loserPos + direction * fallback_distance
        oldHpr=loser.getHpr()
        loser.lookAt(winnerPos)
        fleeHpr=loser.getHpr()-Vec3(180,0,0)
        newHpr=loser.getHpr()
        loser.setHpr(oldHpr)
        rotate_interval = LerpPosHprInterval(
            loser, 
            duration=0.5, 
            pos=loser.getPos(),
            hpr=fleeHpr,
            blendType='easeInOut'
        )
        move_interval = LerpPosInterval(
            loser, 
            duration=1.0, 
            pos=newPos,
            blendType='easeInOut'
        )
        rotate_interval2 = LerpPosHprInterval(
            loser, 
            duration=0.5, 
            pos=newPos,
            hpr=newHpr,
            blendType='easeInOut'
        )
        sequence = Sequence(
            rotate_interval,
            move_interval,
            rotate_interval2,
            Func(self.persuitMove, winner, loser)
        )
        sequence.start()

    def persuitMove(self, winner, loser):
        if loser.isEmpty():
            "looser is destryed, no pursuit"
            #TODO: winner overruns
            return
        if winner is None or loser is None or winner.isEmpty() or loser.isEmpty():
            print("Error: winner or loser is None or empty.")
            return
        print(f"{winner.node().getName()} pursues {loser.node().getName()}!")
        winnerPos=winner.getPos()
        loserPos=loser.getPos()
        oldhpr=winner.getHpr()
        winner.lookAt(loserPos)
        pursueHpr=winner.getHpr()
        winner.setHpr(oldhpr)
        rotate_interval = LerpPosHprInterval(
            winner, 
            duration=0.5, 
            pos=winner.getPos(),
            hpr=pursueHpr,
            blendType='easeInOut'
        )
        direction = (loserPos - winnerPos).normalized()
        pursue_distance = 15.0  # Adjust as needed
        newPos = winnerPos + direction * pursue_distance
        move_interval = LerpPosInterval(
            winner, 
            duration=1.0, 
            pos=newPos,
            blendType='easeInOut'
        )
        sequence = Sequence(
            rotate_interval,
            move_interval
        )
        sequence.start()

app = MyApp()
app.run()