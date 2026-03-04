# ─── Standard Library ───────────────────────────────────────────────────────
import math
import json

# ─── Panda3D Core ────────────────────────────────────────────────────────────
from panda3d.core import (
    Plane, PlaneNode, Point3, Vec2, Vec3, Vec4, BitMask32, TransformState,
    CardMaker, Shader, Texture, TextNode, NodePath, MeshDrawer, LineSegs,
    DirectionalLight, AmbientLight, NurbsCurveEvaluator, NurbsCurveResult,
    GraphicsPipe, GraphicsOutput, LQuaterniond, LVector3d,
    OrthographicLens, Camera, RenderState, TextureStage,
    FrameBufferProperties, WindowProperties, TransparencyAttrib,
    PStatClient, loadPrcFileData,
)

# ─── Panda3D Bullet Physics ─────────────────────────────────────────────────
from panda3d.bullet import (
    BulletWorld, BulletPlaneShape, BulletRigidBodyNode,
    BulletTriangleMesh, BulletTriangleMeshShape, BulletBoxShape,
    BulletSphereShape, BulletDebugNode,
    BulletCharacterControllerNode, BulletCapsuleShape, ZUp,
)

# ─── Panda3D Direct ─────────────────────────────────────────────────────────
from direct.showbase.ShowBase import ShowBase
from direct.task.Task import Task
from direct.interval.LerpInterval import LerpPosInterval, LerpPosHprInterval
from direct.interval.IntervalGlobal import Sequence, ProjectileInterval, Wait, Parallel
from direct.interval.FunctionInterval import Func
from direct.gui.OnscreenText import OnscreenText
from direct.particles.ParticleEffect import ParticleEffect

# ─── Shaders ─────────────────────────────────────────────────────────────────
from shaders.chargedistshaders import *

# ─── Project Modules ─────────────────────────────────────────────────────────
from models import *
from units import *
from toHitAndToWound import *
from battleFunctions import *
from dice import *
from choiceFunctions import *
from ClassOutOfBounds import *
from ClassRoundCounter import *
from ClassAI import *
from rulesFunctions import *
from deployPhase import *
from gameStateAnalyzer import *
from listBuilderGUI import ArmyListBuilderGUI
from campaignMap import CampaignMap, CountryFSM

# ─── Extracted Subsystems ────────────────────────────────────────────────────
from game_fsm import GamePhaseFSM
from spell_system import DevilsVisitSpell, RaiseDeadSpell
from persistence import save_game_state, load_game_state
from combat_resolution import CombatResolver

# ─── Config ──────────────────────────────────────────────────────────────────
loadPrcFileData('', 'show-frame-rate-meter true')


class MyApp(ShowBase):

    # ─── Initialization ──────────────────────────────────────────────────────

    def __init__(self):
        super().__init__()

        # Enable PStats profiling
        #PStatClient.connect()
        
        # Disable default camera controls
        #self.disableMouse()
        base.enableParticles()
        self.signal = False
        self.autoCharge=False
        self.autoRoll=False
        self.autoHold=False
        self.unitCopies = []
        

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
        self.setup_campaign_map()
        self.accept('q-up', self.pathTowardsMouse)
        self.accept('w-up', self.startTaskFunction,[self.taskLoopPathTowardsMouse, "taskLoopPathTowardsMouse"])
        
        self.accept('f5', self.save_game_state, ['quicksave.json'])  # F5 to quick save
        self.accept('f9', self.load_game_state, ['quicksave.json'])  # F9 to quick load
        self.accept('wheel_up', self.zoomIn)  # Mouse wheel forward zooms in
        self.accept('wheel_down', self.zoomOut)  # Mouse wheel backward zooms out
        self.analyzer = GameStateAnalyzer(self)

        self.debugTextUnit = self.setup_text_node(text="Debug Info", pos=(-1.3, -0.9), scale=0.05, color=(1, 1, 0, 1))
        self.debugTextUnit.setText("Debug Info test")

        self.debugTextInfo = self.setup_text_node(text="Debug Info", pos=(0.7, -0.8), scale=0.05, color=(1, 1, 0, 1))
        self.moveArceDistance = 0
        self.debugTextInfo.setText("Debug Arch test")

        self.diceInfoText = self.setup_text_node(text="Dice Info", pos=(-0.7, 0.55), scale=0.05, color=(1, 1, 0, 1))

        self.numsPoints=0
        self.unitHitPos=Point3(0,0,0)

        
        self.units = []
        self.player1Units = []
        self.player2Units = []

        # Spell definitions for wizard units
        spells = {
            'Raise Dead': {
                'description': 'Allows the Necromancer to raise fallen units as Zombies.',
                'casting_value': 7,
                'range': 12,
                'effect': 'Raises a fallen unit within range as a Zombie under the Necromancer\'s control.',
                'phase': 'strategy',
                'class': RaiseDeadSpell
            },
            'Deathly Chill': {
                'description': 'Inflicts a chilling effect on enemy units, reducing their movement.',
                'casting_value': 6,
                'range': 18,
                'effect': 'Reduces the movement characteristic of enemy units within range by 2 for one turn.',
                'phase': 'shooting'
            },
            'Devils visit': {
                'description': 'Increase ally movement',
                'casting_value': 6,
                'range': 18,
                'effect': 'increases the movement characteristic of ally',
                'phase': 'strategy',
                'class': DevilsVisitSpell
            }
        }

        self.load_player1_army("strategy_armies/gunline.json")
        self.load_player2_army("strategy_armies/horde_rush.json")


        self.unitToMove=self.player1Units[0]
        self.accept('mouse3', self.moveUnit,[self.unitToMove])
        #self.messenger.toggleVerbose()
        self.roundCounter = RoundCounter(self,6)

        self.debugText = self.setup_text_node(text="Debug Info", pos=(-1.3, 0.9), scale=0.05, color=(1, 1, 0, 1))
        self.debugText.setText("Debug Info test")
        self.boundries = OutOfBounds(self)
        """ self.AIplayer2 = ClassAI(self, self.player2Units, self.player1Units)
        self.AIplayer2.active = True """
        
        from aiMinimaxIntegration import EnhancedAI

        # Replace: self.AIplayer2 = ClassAI(...)
        self.AIplayer2 = EnhancedAI(
            self, self.player2Units, self.player1Units,
            player_num=2, use_minimax=True, minimax_depth=19
        )
        self.AIplayer2.tree.stop_after_n_returns = 1
        async def auppp():
            for unit in self.player2Units:
                action = await taskMgr.add(self.AIplayer2.take_turn())
                if action.action_type == 'end_phase':
                    break
        self.accept('a-up', lambda: taskMgr.add(self.AIplayer2.take_turn()))
        #self.accept('a-up', lambda: taskMgr.add(auppp()))

        # In your game class __init__:
        self.list_builder = None
        self.list_builder_active = False
        self.accept('l', self.toggle_list_builder)

        self.setActiveUnitTask=self.taskLoopStrategy
        self.setActiveUnitTaskName="taskLoopStrategy"

        self.fsm = GamePhaseFSM(self)
        self.combat = CombatResolver(self)
        self.accept('c', self.toggle_campaign_map)

        self.fsm.request("DeployPhase")

        self.rectangleLine = self.drawRectangle(center=Point3(0, 0, 1), width=72, height=48, color=Vec4(1, 1, 0, 1))
        self.deploymentLine = self.drawRectangle(center=Point3(0, 0, .5), width=72, height=24, color=Vec4(1, 1, 1, 1))

        self.z2= loader.loadModel("models/zup-axis")
        self.z2.reparentTo(render)
        #self.z2.setPos(oposUnit)

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

        
    
    # ─── Army Loading ─────────────────────────────────────────────────────

    def load_army_from_json(self, filename, player_num=1, start_pos=Point3(0, -20, 0), spacing=12):
        """
        Load army units from a JSON file created by the list builder
        
        Args:
            filename: Path to the JSON army list file
            player_num: 1 or 2, determines which player's army this is
            start_pos: Starting position for the first unit
            spacing: Horizontal spacing between units
        
        Returns:
            List of created unitGraphics objects
        """
        # Create a mapping from unit names (from characteristics JSON) to model paths and classes
        unit_model_mapping = {
            'Man at Arms': {'path': 'models/bret_bowmen.bam', 'class': model, 'color': (1, 0, 0, 1)},
            'Man_at_Arm': {'path': 'models/bret_bowmen.bam', 'class': model, 'color': (1, 0, 0, 1)},
            'Mounted Knight of the Realm': {'path': 'models/bret_knight.bam', 'class': MountedKnightOfTheRealm, 'color': (1, 0, 0, 1)},
            'Jade Lancer': {'path': 'models/jade_lancer.bam', 'class': JadeLancer, 'color': (1, 1, 0, 1)},
            'Jade Warrior': {'path': 'models/jade_warrior.bam', 'class': JadeWarrior, 'color': (1, 1, 0, 1)},
            'Night Goblin': {'path': 'models/goblin_archers.bam', 'class': NightGoblin, 'color': (0, 1, 0, 1)},
            'Goblin Wolf Rider': {'path': 'models/goblin_wolfriders.bam', 'class': GoblinWolfRider, 'color': (0, 1, 0, 1)},
            'Black Knight': {'path': 'models/black_knights.bam', 'class': BlackKnight, 'color': (0, 0, 1, 1)},
            'Zombie': {'path': 'models/zombies.bam', 'class': Zombie, 'color': (0, 0, 1, 1)},
            'Orc Boyz': {'path': 'models/goblin_archers.bam', 'class': model, 'color': (0, 1, 0, 1)},
            'Orc Boy': {'path': 'models/goblin_archers.bam', 'class': model, 'color': (0, 1, 0, 1)},
            'Black Orc': {'path': 'models/goblin_archers.bam', 'class': model, 'color': (0, 1, 0, 1)},
            'Necromancer': {'path': 'models/zombies.bam', 'class': Necromancer, 'color': (0, 0, 1, 1)},
            'Saurus Warrior': {'path': 'models/jade_warrior.bam', 'class': model, 'color': (1, 1, 0, 1)},
            'Pegasus Knight': {'path': 'models/bret_knight.bam', 'class': model, 'color': (1, 0, 0, 1)},
            'Dire Wolf': {'path': 'models/dire_wolves.bam', 'class': DireWolf, 'color': (0, 0, 1, 1)},
        }
        
        # Load the JSON file
        try:
            with open(filename, 'r') as f:
                army_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: File {filename} not found!")
            return []
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in {filename}!")
            return []
        
        created_units = []
        current_x = start_pos.x
        
        for idx, army_unit_data in enumerate(army_data):
            unit_name = army_unit_data['name']
            nmodels = army_unit_data['nmodels']
            files = army_unit_data['files']
            ranks = army_unit_data['ranks']
            
            # Get model info from mapping, default to generic if not found
            model_info = unit_model_mapping.get(unit_name, {
                'path': 'models/jade_warrior.bam',
                'class': model,
                'color': (0.5, 0.5, 0.5, 1)
            })
            
            try:
                # Create model instance
                model_class = model_info['class']
                if model_class == model:
                    # Basic model
                    model_instance = model(unit_name, "")
                elif model_class in [JadeLancer, MountedKnightOfTheRealm, GoblinWolfRider, BlackKnight]:
                    # Mounted units need mount units
                    # For simplicity, create basic mounts
                    mount_model = model(f"{unit_name} Mount", "")
                    mount_unit = unit(f"{unit_name} Mount Unit", mount_model, nmodels, files, ranks)
                    model_instance = model_class(unit_name, "", mountUnit=mount_unit)
                else:
                    # Other special classes
                    model_instance = model_class(unit_name, "")
                
                model_instance.armor_save = 7  # Default armor save
                
                # Create unit instance
                unit_instance = unit(f"{unit_name} Unit", model_instance, nmodels, files, ranks)
                
                # Create unit graphics
                # Include player_num in the name so P1 and P2 units with the
                # same model type get unique collision-node / lookup names.
                graphics_name = f"P{player_num}_{unit_name.replace(' ', '')}{idx}"
                unit_graphics = unitGraphics(
                    self,
                    graphics_name,
                    model_info['path'],
                    unit_instance,
                    scale=1.0,
                    BulletWorld=self.world,
                    color=model_info['color']
                )
                
                # Position the unit
                unit_graphics.bodyNP.setPos(current_x, start_pos.y, start_pos.z)
                
                # Add to appropriate lists
                self.units.append(unit_graphics)
                if player_num == 1:
                    self.player1Units.append(unit_graphics)
                else:
                    self.player2Units.append(unit_graphics)
                
                created_units.append(unit_graphics)
                
                # Update position for next unit
                current_x += spacing
                
                print(f"Loaded unit: {unit_name} ({nmodels} models, {files}x{ranks})")
                
            except Exception as e:
                print(f"Error creating unit {unit_name}: {e}")
                continue
        
        print(f"Successfully loaded {len(created_units)} units from {filename}")
        return created_units

    # ─── Texture Baking ──────────────────────────────────────────────────

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

    # ─── Projectiles & Visual Effects ────────────────────────────────────

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

    # ─── Task Management & Phase Loops ────────────────────────────────────

    def startTaskFunction(self,taskfunction,taskname):
        if taskMgr.hasTaskNamed(taskname):
            taskMgr.remove(taskname)
        taskMgr.add(taskfunction, taskname)
        return

    def freeReformUnit(self, unit,task):
        c = self.checkUnitContactSmall(unit)
        contact=False
        if c:
            print("Unit is in contact with another unit, cannot reform here.")
            unit.model.setColor(.6,0.6,0.6,1)
            contact=True
        else:
            #return task.cont
            unit.model.setColor(unit.color)
            
        if base.mouseWatcherNode.hasMouse():
            mousePos = base.mouseWatcherNode.getMouse()
            pFrom = Point3()
            pTo = Point3()
            base.camLens.extrude(mousePos, pFrom, pTo)

            # Transform to global coordinates
            pFrom = render.getRelativePoint(base.cam, pFrom)
            pTo = render.getRelativePoint(base.cam, pTo)

            result = base.world.rayTestClosest(pFrom, pTo)

            if result.hasHit():
                hitPos = result.getHitPos()
                unit.bodyNP.lookAt(hitPos)
                #unit.hasMovedThisTurn=True
                #unit.updateTextNode()
        if self.signal and not contact:
            self.signal = False
            return task.done
        self.signal = False
        return task.cont
    
    def giveSignal(self):
        self.signal = True
        return

    async def rallyUnit(self, unit):
        # Placeholder for rallying logic
        Ld=int(unit.unit.model.characteristics['Ld'])
        print("losing unit original LD:", Ld)
        terningerLd=[]
        for i in range(2):
            terning = Dice(self.world, position=Vec3(20+i*2,0,10), size=1.0,color=(1,0,0,1))
            terningerLd.append(terning)
        for terning in terningerLd:
            terning.roll()
        await taskMgr.add(checkDice, "checkDiceTaskFlee", extraArgs=[terningerLd], appendTask=True)
        ldDice = []
        for terning in terningerLd:
            ldDice.append(terning.currentValue)
        leadership_score = sum(ldDice)
        for terning in terningerLd:
            terning.remove(self.world)
        print("Leadership dice results for fleeing unit:", ldDice, "sum:", leadership_score)
        if leadership_score <= Ld+99:
            print(f"Rallying unit: {unit.unit.name}")
            self.ignore('mouse1')
            self.accept('mouse1', self.giveSignal)
            await taskMgr.add(self.freeReformUnit, "freeReformUnitTask", extraArgs=[unit], appendTask=True)
            print(f"Unit {unit.unit.name} has rallied successfully.")
            self.ignore('mouse1')
            self.accept('mouse1', self.setActiveUnit,[self.taskLoopStrategy, "taskLoopStrategy"])
            unit.request("Idle")
        unit.attemptedRallyThisTurn=True
        return

    # ─── Phase Task Loops ─────────────────────────────────────────────────

    def taskLoopDeploy(self, task):
        #base.messenger.toggleVerbose()
        if allUnitsDeployed(self.units):
            print("All units deployed, moving to next phase.")
            self.fsm.request("StrategyPhase")
            return task.done
        if self.unitToMove.isDeployed:
            print("Unit is already deployed, cannot move.")
            return task.done
        self.ignore('mouse1')
        movetask = taskMgr.add(taskMoveUnit, "taskMoveUnit", extraArgs=[self,self.unitToMove], appendTask=True)
        self.accept('mouse1', endMoveUnit, [self,movetask])
        return task.done


    def taskLoopStrategy(self, task):
        # Placeholder for strategy phase logic
        if self.unitToMove.state == "IsFleeing" and not self.unitToMove.attemptedRallyThisTurn:
            if not taskMgr.hasTaskNamed("rallyUnitTask"):
                print("Attempt to rally fleeing unit.")
                taskMgr.add(self.rallyUnit(self.unitToMove), "rallyUnitTask")
            #return task.done

        if any(rule.get('wizard',False) for rule in self.unitToMove.unit.model.special_rules):
            print("Wizard phase logic here.")
            self.fsm.request("SpellPhase")

        return task.done

    def taskLoopPathTowardsMouse(self, task):
        if self.unitToMove.state != "Idle":
            print("Unit is not idle, cannot move.")
            return task.done
        
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
    
    async def taskMagicArcUpdate(self, task):
        for unit in self.units:
            unit.model.setColor(unit.color)
            unit.bodyNP.setCollideMask(BitMask32.bit(unit.bitmask))
        if self.unitToMove.isInCombat:
            print("Unit is in combat, cant shoot.")
            return task.done
        if not any(rule.get('wizard', False) for rule in self.unitToMove.unit.model.special_rules):
            print("Unit is not a wizard, cant cast.")
            return task.done
        """ print("equiped weapon is: ",self.unitToMove.unit.model.equipedWeapon)
        if self.unitToMove.unit.model.equipedWeapon is None:# or not self.unitToMove.unit.model.equippedWeapon.is_ranged:
            print("Unit has no equiped weapon equipped, cant shoot.")
            return task.done """
        r=False
        print(self.unitToMove.unit.model.spells)
        spellChoices = []
        spellClasses = []
        for spell in self.unitToMove.unit.model.spells:
            print("checking spells: ",self.unitToMove.unit.model.spells.get(spell))
            if self.unitToMove.unit.model.spells.get(spell).get('phase') == 'strategy':
                spellChoices.append(spell)
                spellClasses.append(self.unitToMove.unit.model.spells.get(spell).get('class'))
                r=True
        if not r:
            print("Unit has no strategy phase spells, cant cast.")
            return task.done
        
        
        spellchoice = await taskMgr.add(self.makeChoiceNew(spellChoices, Vec3(-20,0,10)))

        print("Chosen spell: ", spellchoice)
        index = spellChoices.index(spellchoice)
        self.fsm.activeSpell = self.unitToMove.unit.model.spells.get(spellchoice)
        self.fsm.spellClassToCast = spellClasses[index]
        self.fsm.spellInstanceToCast = self.fsm.spellClassToCast(spellchoice, self.fsm.activeSpell.get('casting_value',12),self.fsm.endOfTurnSpells)
        #self.fsm.endOfTurnSpells.append(self.fsm.spellInstanceToCast)

        radius = self.fsm.activeSpell.get('range',18)
        #radius = self.coordsToLocal([Vec2(radius,0)])[0].x
        radius = radius/(2*50)


        self.shootingArcPoints = self.shootingArc(self.unitToMove.bodyNP.getPos(render), 
                                                       num_points=80, rotationangle=self.unitToMove.bodyNP.getH()+45, radius=radius)
        self.ground.setShaderInput("polygonpoints", self.shootingArcPoints)
        self.ground.setShaderInput("isActive", True)
        if not taskMgr.hasTaskNamed("taskShootingTrajectoryDrawLine"):
            taskMgr.add(self.taskShootingTrajectoryDrawLine, "taskShootingTrajectoryDrawLine")
        self.checkArrows(BitMask32.bit(5))
        return task.done
    
    def taskStartCombat(self, task):
        if not self.unitToMove.isInCombat:
            print("Unit is not in combat, cant start combat.")
            return task.done
        if self.unitToMove.hasAttackedThisTurn:
            print("Unit has already attacked this turn, cant attack again.")
            return task.done
        if self.unitToMove.isInCombat and not self.unitToMove.hasAttackedThisTurn:
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
    
    def coordsToLocal(self, points):
        localPoints = []
        for point in points:
            point = point / 50
            point += Vec2(1,1)
            point = point / 2
            localPoints.append(Point3(point.x, point.y, 0))
        return localPoints

    def checkArrows(self,mask=BitMask32.bit(3)):
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
                        NodePath.anyPath(result.getNode()).setCollideMask(mask)
                        #self.toCleanup.append(np)

    # ─── Camera & UI ──────────────────────────────────────────────────────

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
                        

    async def setActiveUnit(self,taskfunction,taskname):
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
            
            result3 = self.world.rayTestClosest(pFrom, pTo, BitMask32.bit(5))    
            if result3.hasHit():
                selected_unit = self.getSelectedUnit(result3.getNode())
                print("Selected magic target:",selected_unit.unit.name)
                # Implement spell casting logic here
                #await taskMgr.add(self.raiseDead(selected_unit))
                #await taskMgr.add(self.fsm.spellFunctionToCast(selected_unit))
                await self.fsm.spellInstanceToCast.spellFunction(selected_unit)

                #selected_unit.bodyNP.setCollideMask(BitMask32.bit(1))
                #self.checkArrows(BitMask32.bit(1))
                if self.roundCounter.current_player == 1:
                    self.roundCounter.request('PlayerOne')
                else:
                    self.roundCounter.request('PlayerTwo')
                self.fsm.request("StrategyPhase")

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

    # ─── Unit Selection & Interaction ─────────────────────────────────────

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

    # ─── Campaign Map ─────────────────────────────────────────────────────

    def toggle_campaign_map(self):
        """Toggle between campaign map and battle view."""
        if self.fsm.state == 'CampaignPhase':
            # Return to whatever battle phase we were in
            self.fsm.request(self.fsm.phases[self.fsm.currentPhaseIndex])
        else:
            self.fsm.request('CampaignPhase')

    def setup_campaign_map(self):
        """Initialize campaign map components (hidden initially)."""
        # Create campaign map terrain
        self.campaign_map = CampaignMap(self)
        self.campaign_map.load_heightmap("assets/textures/wals_dem_resized.png", height_scale=25)
        self.campaign_map.set_texture("assets/textures/wals_tex_resized.png")
        # Offset far to the right so it doesn't overlap the battle board
        self.campaign_offset_x = 2000
        self.campaign_map.set_position(self.campaign_offset_x - 1024 / 2, -2048 / 2, 0)

        # Load country overlay model
        self.country_model = self.loader.loadModel("models/blender/maps1.bam")
        self.country_model.setPos(self.campaign_offset_x, 0, 0)
        self.country_model.reparentTo(self.render)

        # Create collision meshes for each country child
        for child in self.country_model.getChildren():
            self.campaign_map.contryCollision(self.world, child)
            child.setTransparency(TransparencyAttrib.MAlpha)
            child.setBin("transparent", 50)
            child.setDepthTest(False)

        # Initialize country FSM with shader-based coloring
        self.country_fsm = CountryFSM(self.country_model, self)
        self.country_fsm.request('None')

        # Define country borders
        self.country_fsm.setBorders("Plane.005", ["Plane.007", "Plane.006", "Plane.009"])
        self.country_fsm.setBorders("Plane.006", ["Plane.005", "Plane.009", "Plane.010", "Plane.011", "Plane.007"])
        self.country_fsm.setBorders("Plane.007", ["Plane.005", "Plane.006", "Plane.011", "Plane.008"])
        self.country_fsm.setBorders("Plane.008", ["Plane.007", "Plane.011"])
        self.country_fsm.setBorders("Plane.009", ["Plane.005", "Plane.006", "Plane.010", "Plane.013"])
        self.country_fsm.setBorders("Plane.010", ["Plane.006", "Plane.009", "Plane.012", "Plane.013", "Plane.011"])
        self.country_fsm.setBorders("Plane.011", ["Plane.006", "Plane.007", "Plane.008", "Plane.010", "Plane.012", "Plane.016"])
        self.country_fsm.setBorders("Plane.012", ["Plane.010", "Plane.011", "Plane.013", "Plane.016", "Plane.015"])
        self.country_fsm.setBorders("Plane.013", ["Plane.009", "Plane.010", "Plane.012", "Plane.015", "Plane.014"])
        self.country_fsm.setBorders("Plane.014", ["Plane.013", "Plane.015"])
        self.country_fsm.setBorders("Plane.015", ["Plane.012", "Plane.013", "Plane.014", "Plane.016", "Plane.017", "Plane.018"])
        self.country_fsm.setBorders("Plane.016", ["Plane.011", "Plane.012", "Plane.015", "Plane.017"])
        self.country_fsm.setBorders("Plane.017", ["Plane.015", "Plane.016", "Plane.019"])
        self.country_fsm.setBorders("Plane.018", ["Plane.015"])
        self.country_fsm.setBorders("Plane.019", ["Plane.017"])

        # Load cloud shader
        self.cloud_shader = Shader.load(
            Shader.SL_GLSL,
            vertex="cloud.vert.txt",
            fragment="cloud.frag.txt"
        )

        # Create cloud plane above terrain
        self.cloud_nodes = []
        self.cloud_plane = self.loader.loadModel("models/box")
        self.cloud_plane.setScale(1000, 2000, 1)
        self.cloud_plane.setPos(self.campaign_offset_x - 512, -1024, 20)
        self.cloud_plane.setShader(self.cloud_shader)
        self.cloud_plane.setShaderInput("customTime", 0.0)
        self.cloud_plane.setShaderInput("cloudColor", Vec4(1.0, 1.0, 1.0, 1.0))
        self.cloud_plane.setShaderInput("skyColor", Vec4(0.5, 0.7, 0.9, 0.1))
        self.cloud_plane.setShaderInput("cloudCoverage", 0.5)
        self.cloud_plane.setTransparency(TransparencyAttrib.MAlpha)
        self.cloud_plane.setBin("transparent", 100)
        self.cloud_plane.reparentTo(self.render)
        self.cloud_nodes.append(self.cloud_plane)

        # Hide everything campaign-related initially
        self.campaign_map.hide()
        self.country_model.hide()
        self.cloud_plane.hide()

        print(f"Campaign map initialized. Available countries: {self.country_fsm.getAllCountries()}")

    def update_campaign_terrain(self, task):
        """Update terrain LOD for campaign map."""
        self.campaign_map.update()
        return task.cont

    def update_cloud_time(self, task):
        """Update time uniform for cloud shader nodes."""
        for node in self.cloud_nodes:
            node.setShaderInput("customTime", task.time * 0.1)
        return task.cont

    def campaign_mouse_click(self):
        """Handle mouse click on campaign map for country selection."""
        if self.mouseWatcherNode.hasMouse():
            pMouse = self.mouseWatcherNode.getMouse()
            pFrom = Point3()
            pTo = Point3()
            self.camLens.extrude(pMouse, pFrom, pTo)
            pFrom = render.getRelativePoint(self.cam, pFrom)
            pTo = render.getRelativePoint(self.cam, pTo)
            result = self.world.rayTestClosest(pFrom, pTo, BitMask32.bit(1))

            if result.hasHit():
                hit_node_name = result.getNode().getName()
                print(f"Campaign hit: {hit_node_name} at {result.getHitPos()}")
                country_name = hit_node_name.split("_")[0]
                self.country_fsm.selectCountry(country_name)
            else:
                print("No country hit")
                self.country_fsm.deselectCountry()

    def campaign_deselect(self):
        """Handle right-click to deselect country on campaign map."""
        self.country_fsm.deselectCountry()

    # ─── Shader & Physics Setup ───────────────────────────────────────────

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
        self.debugNP = render.attachNewNode(debugNode)
        
        self.world.setDebugNode(self.debugNP.node())
        self.debugNP.show()
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
            self.world.doPhysics(dt*2.0, 10, 0.008)
            #self.world.doPhysics(dt, 10, 0.008)
            
            return task.cont
    def move_node_smoothly(self, node, target_pos, duration=1.0):
        interval = LerpPosInterval(node, duration, target_pos,blendType='easeInOut')
        mySequence = Sequence(interval)
        mySequence.start()

    # ─── Drawing Helpers (circles, arcs, rectangles) ─────────────────────

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

    
    def shootingArc(self, origo, num_points=40, rotationangle=30,radius=0.15):
        points =[]
        origo=(origo/50 +1)*0.5
        origo   = Vec2(origo.x,origo.y)
        points.append(origo)

        arcmax = math.pi/2

        for i in range(0,num_points):
            angle = arcmax * i / (num_points - 1)
            x = radius * math.cos(angle) 
            y = radius * math.sin(angle)
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
                if abs(vectormouse.normalized().dot(vector)) < .001:
                    
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

    # ─── Movement & Pathfinding ───────────────────────────────────────────

    def pathTowardsMouse(self,unit,x=None,y=None):
        if not base.mouseWatcherNode.hasMouse():
            return
        if base.mouseWatcherNode.hasMouse() and x is None and y is None:
            self.unitToMove=unit
            x = base.mouseWatcherNode.getMouseX()
            y = base.mouseWatcherNode.getMouseY()
            pMouse = base.mouseWatcherNode.getMouse()
            pFrom = Point3()
            pTo = Point3()
            base.camLens.extrude(pMouse, pFrom, pTo)

            # Transform to global coordinates
            pFrom = render.getRelativePoint(base.cam, pFrom)
            pTo = render.getRelativePoint(base.cam, pTo)
        else:# x is not None and y is not None:
            self.unitToMove=unit
            pFrom = Point3()
            #pFrom = render.getRelativePoint(base.cam, pFrom)
            pFrom = Point3(x, y, 10)
            pTo = Point3(x, y, -10)
            #pTo = render.getRelativePoint(base.cam, pTo)
            
        if True:
            print(f"pFrom: {pFrom}, pTo: {pTo}")
            
            #print(x,y)
            #surface.set_shader_input("pos", Vec3(base.mouseWatcherNode.getMouseX(),0,base.mouseWatcherNode.getMouseY())*4)
            #pFrom = Point3(0, 0, 0)
            #pTo = Point3(10, 0, 0)

            # Get to and from pos in camera coordinates
            #pFrom = render.getRelativePoint(base.cam, pFrom)

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

            """ for rule in self.unitToMove.unit.model.special_rules:
                if rule.get('move'):
                    #print("Unit is flatfooted, cannot move.")
                    rule['move'](self.unitToMove.unit.model)
            
            for rule in self.unitToMove.unit.model.special_rules:
                if rule.get('mountUnit'):
                    for ruleM in rule['mountUnit'].model.special_rules:
                        if ruleM.get('move'):
                            ruleM['move'](rule['mountUnit'].model)

            M=str(self.unitToMove.unit.model.characteristics['M'])
            if M.isdigit():
                M = int(M)
            else:
                print(f"Warning: M value '{M}' is not a number, defaulting to 1")
                M = 0
            move = M*2
            if unit.state == "IsPursuing":
                move = 21

            for rule in self.unitToMove.unit.model.special_rules:
                if rule.get('mountUnit'):
                    mountmove= int(rule['mountUnit'].model.characteristics['M'])*2
                    move = max(move, mountmove)
            print("Unit move:", move)
            
            self.unitToMove.unit.model.reset_characteristics()
            for rule in self.unitToMove.unit.model.special_rules:
                if rule.get('mountUnit'):
                    rule['mountUnit'].model.reset_characteristics() """

            M=str(self.unitToMove.unit.model.characteristics['M'])
            if M.isdigit():
                M = int(M)
            else:
                print(f"Warning: M value '{M}' is not a number, defaulting to 1")
                M = 0
            move = M+6
            if unit.state == "IsPursuing":
                move = 21

            for rule in self.unitToMove.unit.model.special_rules:
                if rule.get('mountUnit'):
                    mountmove= int(rule['mountUnit'].model.characteristics['M'])+6
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
            #self.world.doPhysics(0.016)
            closest_dist = float('inf')
            closest_pos = None
            
            frac,closest_pos_frac,tsTo = self.sweepTestRot(unit,p3,self.arcPointRotation)
            if frac < 1.0:
                self.arcPointRotation *= frac
                closest_dist = 0
                closest_pos = closest_pos_frac

            else:
                dire=(Vec3(p2.x, p2.y, .9) - Vec3(p1.x, p1.y, .9) ).normalized()
                le=(Vec3(p2.x, p2.y, .9) - Vec3(p1.x, p1.y, .9) ).length()
                #le=move/(2*abs(groundSizeboundingbox[0][1]))
                #le-=math.radians(abs(self.arcPointRotation))*unit.unitWidth
                frac,closest_pos_frac = self.sweepTestDir(unit,tsTo,dire,le)
                if frac < 1.0:
                    closest_dist = le*frac
                    closest_pos = closest_pos_frac

            


            if closest_pos:
                self.unitHitPos = closest_pos
                self.playerNP.setPos(closest_pos)
                #self.z2.setPos(closest_pos + Vec3(0,0,0.5))

                newmove = closest_dist+math.radians(abs(self.arcPointRotation))*unit.unitWidth
                print("New move distance:", newmove, "closest dist:", closest_dist, "arc rotation:", self.arcPointRotation)
                self.polygonpoints = self.pointArc(origo=unitposxy, num_points=80, mouse_pos=Vec2(pos.x, pos.y),
                                                width=unitwidth,height=unitheight, rotationangle=unit.bodyNP.getH(),
                                                movedistance=newmove/(2*abs(groundSizeboundingbox[0][1])))

                self.ground.setShaderInput("polygonpoints", self.polygonpoints)
                self.ground.setShaderInput("isActive", True)
                return

            M=str(self.unitToMove.unit.model.characteristics['M'])
            modifyer=1
            modifyerM=1
            for rule in self.unitToMove.unit.model.special_rules:
                if rule.get('move'):
                    #print("Unit is flatfooted, cannot move.")
                    modifyer=rule['move']
                    #M = str(int(int(M) * modifyer))
            
            for rule in self.unitToMove.unit.model.special_rules:
                if rule.get('mountUnit'):
                    for ruleM in rule['mountUnit'].model.special_rules:
                        if ruleM.get('move'):
                            #ruleM['move'](rule['mountUnit'].model)
                            modifyerM=ruleM['move']


            
            if M.isdigit():
                M = int(M)
            else:
                print(f"Warning: M value '{M}' is not a number, defaulting to 1")
                M = 0
            move = M*2
            print("Unit move:", move)
            move = move * modifyer
            

            for rule in self.unitToMove.unit.model.special_rules:
                if rule.get('mountUnit'):
                    mountmove= modifyerM*int(rule['mountUnit'].model.characteristics['M'])*2
                    move = max(move, mountmove)
            if unit.state == "IsPursuing":
                move = 21
            
            
            print("Modified unit move:", move)
            
            """ self.unitToMove.unit.model.reset_characteristics()
            for rule in self.unitToMove.unit.model.special_rules:
                if rule.get('mountUnit'):
                    rule['mountUnit'].model.reset_characteristics() """
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
            #self.world.doPhysics(0.016)
            closest_dist = float('inf')
            closest_pos = None
            
            frac,closest_pos_frac,tsTo = self.sweepTestRot(unit,p3,self.arcPointRotation)
            if frac < 1.0:
                self.arcPointRotation *= frac
                closest_dist = 0
                closest_pos = closest_pos_frac

            else:
                dire=(Vec3(p2.x, p2.y, .9) - Vec3(p1.x, p1.y, .9) ).normalized()
                le=(Vec3(p2.x, p2.y, .9) - Vec3(p1.x, p1.y, .9) ).length()
                #le=move/(2*abs(groundSizeboundingbox[0][1]))
                #le-=math.radians(abs(self.arcPointRotation))*unit.unitWidth
                frac,closest_pos_frac = self.sweepTestDir(unit,tsTo,dire,le)
                if frac < 1.0:
                    closest_dist = le*frac
                    closest_pos = closest_pos_frac

            


            if closest_pos:
                self.unitHitPos = closest_pos
                self.playerNP.setPos(closest_pos)
                #self.z2.setPos(closest_pos + Vec3(0,0,0.5))

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

    # ─── Unit Movement Execution ──────────────────────────────────────────

    def moveUnit(self, unit):
        if taskMgr.hasTaskNamed("taskLoopPathTowardsMouse"):
            taskMgr.remove("taskLoopPathTowardsMouse")
            
        if unit.state != "Idle":
            print("Unit is not idle")
            if unit.state != "IsPursuing":
                print("Unit is not pursuing, cannot move.")
                return
        
        
        
        
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
            defenderNP = render.find(f"**/{c.getNode1().getName()}")
            defenderUnit=self.getSelectedUnit(defenderNP.node())

            if unit.state == "IsPursuing":
                pass
            else:
                if defenderUnit in self.player1Units:
                    if unit in self.player1Units:
                        print("Both units belong to Player 1, cannot enter combat.")
                        direction = unit.bodyNP.getPos() - defenderNP.getPos()
                        direction.normalize()
                        self.fallBackContactTest(unit.bodyNP, direction*.3)
                        unit.request("Moved")
                        return
                if defenderUnit in self.player2Units:
                    if unit in self.player2Units:
                        print("Both units belong to Player 2, cannot enter combat.")
                        direction = unit.bodyNP.getPos() - defenderNP.getPos()
                        direction.normalize()
                        self.fallBackContactTest(unit.bodyNP, direction*.3)
                        unit.request("Moved")
                        return

            taskMgr.add(self.chargeAndChargeReaction, extraArgs=[unit, c,oposUnit, orotUnit],appendTask=True)
            #self.getFlankFromContact(unit, c)
            unit.model.setColor(.7,0.7,0.7,1)
            copiedUnit=unit.bodyNP.copyTo(render)
            self.unitCopies.append(copiedUnit)
            unit.model.setColor(unit.color)

            #copyiedUnit.setColor(1,0,0,1)
            copiedUnit.setColor(.7,0.7,0.7,1)
            copiedUnit.setPos(oposUnit)
            copiedUnit.setHpr(orotUnit)
        else:
            unit.request("Moved")
        self.bakeTextures(self.ground)

    async def makeChoiceNew(self, choices, position):
        cyn = Choice(choices, position)
        cyn.ma = taskMgr.add(cyn.mouseActivate, "mouseActivateTask")
        self.ignore('mouse1')
        print("Waiting for choice...")
        if self.roundCounter.current_player == 2 and self.AIplayer2.active:
            #cynchoice = chargeYesNo[0]
            await Task.pause(1.0)
            cyn.hitbox = cyn.boxes[0].node()
            
            cyn.onMouseClick()
        else:
            await cyn.ma
        #self.accept('mouse1', self.setActiveUnit,[self.taskLoopPathTowardsMouse, "taskLoopPathTowardsMouse"])
        self.accept('mouse1', self.setActiveUnit,[self.setActiveUnitTask, self.setActiveUnitTaskName])
        print("event received")
        cynchoice = cyn.choice
        
        print('Event delivered with args:', cyn.choice)
        del cyn
        return cynchoice
    
    async def makeChoice(self, choice):
        choice.ma = taskMgr.add(choice.mouseActivate, "mouseActivateTask")
        self.ignore('mouse1')
        print("Waiting for choice...")
        await choice.ma
        self.accept('mouse1', self.setActiveUnit,[self.setActiveUnitTask, self.setActiveUnitTaskName])
        print("event received")
        selected_choice = choice.choice
        print('Event delivered with args:', choice.choice)
        return

    # ─── Combat Resolution (delegates to CombatResolver) ─────────────────

    async def chargeAndChargeReaction(self, unit, c, oposUnit, orotUnit, task):
        return await self.combat.chargeAndChargeReaction(unit, c, oposUnit, orotUnit, task)

    def checkUnitContactSmall(self, unit):
        return self.combat.checkUnitContactSmall(unit)

    async def fleeInterval(self, unit, defenderNP, angleToRotate, oposUnit, orotUnit):
        return await self.combat.fleeInterval(unit, defenderNP, angleToRotate, oposUnit, orotUnit)

    async def rullTerninger(self, antall):
        return await self.combat.rullTerninger(antall)

    async def chargeInterval(self, unit, defenderNP, angleToRotate, oposUnit, orotUnit, flank, chdice=None):
        return await self.combat.chargeInterval(unit, defenderNP, angleToRotate, oposUnit, orotUnit, flank, chdice)

    def getFlankFromContact(self, unit, contact):
        return self.combat.getFlankFromContact(unit, contact)

    def printBattleResults(self, attackerUnit, defenderUnit, attacks, total_hits,
                           suffered_wounds, saves_made, total_wounds):
        self.combat.printBattleResults(attackerUnit, defenderUnit, attacks, total_hits,
                                       suffered_wounds, saves_made, total_wounds)

    async def verySimpleBattleStart(self, task):
        return await self.combat.verySimpleBattleStart(task)

    async def verySimpleBattle(self, task):
        return await self.combat.verySimpleBattle(task)

    async def GiveGroundFromCombat(self, loserUnit):
        return await self.combat.GiveGroundFromCombat(loserUnit)

    async def FBIGFromCombat(self, loserUnit):
        return await self.combat.FBIGFromCombat(loserUnit)

    async def fleeFromCombat(self, loserUnit):
        return await self.combat.fleeFromCombat(loserUnit)

    # ─── Flee, Pursuit & Rally ────────────────────────────────────────────

    def checkFleeCaught(self, fleeUnit, pursuerUnit,task):
        #fleeUnit.bodyNP.setCollideMask(BitMask32.bit(2))
        #pursuerUnit.bodyNP.setCollideMask(BitMask32.bit(2))
        #pursuerUnit.bodyNP.node().setCcdMotionThreshold(1e-7)
        #pursuerUnit.bodyNP.node().setCcdSweptSphereRadius(0.50)
        if fleeUnit.bodyNP.isEmpty() or pursuerUnit.bodyNP.isEmpty():
            print("One of the units is already removed, stopping flee catch check.")
            return task.done
        pursuerUnit.bodyNP.node().setTransformDirty()
        fleeUnit.bodyNP.node().setTransformDirty()
        result = self.world.contactTestPair(fleeUnit.bodyNP.node(), pursuerUnit.bodyNP.node())
        print("Checking flee contact between", fleeUnit.unit.name, "and", pursuerUnit.unit.name)
        print(fleeUnit.bodyNP.node().pickDirtyFlag())
        print(result.getNumContacts(), result.getContacts())
        for contact in result.getContacts():
            print("Contact detected between fleeing unit and pursuer!")
            self.world.removeRigidBody(fleeUnit.bodyNP.node())
            fleeUnit.model.removeNode()
            fleeUnit.bodyNP.removeNode()
            self.units.remove(fleeUnit)
            if fleeUnit in self.player1Units:
                self.player1Units.remove(fleeUnit)
            if fleeUnit in self.player2Units:
                self.player2Units.remove(fleeUnit)
            #messenger.send('unit-move-complete')
            return task.done
        return task.cont


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
            for u in unit.isInCombatWith:
                u.request("Idle")
            self.world.removeRigidBody(unit.bodyNP.node())
            unit.bodyNP.removeNode()
            unit.model.removeNode()
            self.units.remove(unit)
            if unit in self.player1Units:
                self.player1Units.remove(unit)
            if unit in self.player2Units:
                self.player2Units.remove(unit)
            
            
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

    #def sweepTest(self, shape,startPos,startHpr, endPos,endHpr):
    def sweepTest(self, unit, direction,length):
        startPos=unit.bodyNP.getPos()
        Hpr=unit.bodyNP.getHpr()
        unit.bodyNP.lookAt(startPos + direction)
        nHpr = unit.bodyNP.getHpr()
        unit.bodyNP.setHpr(Hpr)


        tsFrom = TransformState.makePosHpr(startPos, nHpr)
        tsTo = TransformState.makePosHpr(startPos + direction * length, nHpr)
        shape = unit.bodyNP.node().getShape(0)
        #shape = BulletSphereShape(0.5)
        penetration = 0.0
        omasks=[]
        for u in self.units:
            omasks.append(u.bodyNP.getCollideMask())
            u.bodyNP.setCollideMask(BitMask32.bit(9))
        unit.bodyNP.setCollideMask(BitMask32.bit(30))
        for u in unit.isInCombatWith:
            u.bodyNP.setCollideMask(BitMask32.bit(30))
        #self.mountedKnightOfTheRealm.bodyNP.setCollideMask(BitMask32.bit(9))
        result = base.world.sweepTestClosest(shape, tsFrom, tsTo,BitMask32.bit(9))
        #unit.setCollideMask(BitMask32.bit(1))
        for i,u in enumerate(self.units):
            u.bodyNP.setCollideMask(omasks[i])
        if result.hasHit():
            print(result.hasHit())
            print(result.getHitPos())
            print(result.getHitNormal())
            print(result.getHitFraction())
            print(result.getNode())
            self.z2.setPos(result.getHitPos())
            print("sweep test topos:", result.getToPos())
            print(startPos + direction * length)
            return result.getHitFraction()
        return 1.0
    
    def sweepTestRot(self, unit, point,angle):
        startPos=unit.bodyNP.getPos()
        Hpr=unit.bodyNP.getHpr()
        shape = unit.bodyNP.node().getShape(0)

        newxy= self.rotatePoint(Vec2(startPos.x, startPos.y), angle, point)
        
        #unit.bodyNP.lookAt(startPos + direction)
        nHpr = Hpr + Vec3(angle,0,0)
        #unit.bodyNP.setHpr(Hpr)


        tsFrom = TransformState.makePosHpr(startPos, Hpr)
        tsTo = TransformState.makePosHpr(Vec3(newxy[0], newxy[1], startPos.z), nHpr)
        
        #shape = BulletSphereShape(0.5)
        penetration = 0.0
        omasks=[]
        for u in self.units:
            omasks.append(u.bodyNP.getCollideMask())
            u.bodyNP.setCollideMask(BitMask32.bit(9))
        unit.bodyNP.setCollideMask(BitMask32.bit(30))
        """ for u in unit.isInCombatWith:
            u.bodyNP.setCollideMask(BitMask32.bit(30)) """
        #self.mountedKnightOfTheRealm.bodyNP.setCollideMask(BitMask32.bit(9))
        result = base.world.sweepTestClosest(shape, tsFrom, tsTo,BitMask32.bit(9))
        #unit.setCollideMask(BitMask32.bit(1))
        for i,u in enumerate(self.units):
            u.bodyNP.setCollideMask(omasks[i])
        if result.hasHit():
            print(result.hasHit())
            print(result.getHitPos())
            print(result.getHitNormal())
            print(result.getHitFraction())
            print(result.getNode())
            #self.z2.setPos(result.getHitPos())
            print("sweep test topos:", result.getToPos())
            
            return result.getHitFraction(),result.getHitPos(),tsTo
        return 1.0,None,tsTo
    
    def sweepTestDir(self, unit, tsFrom, direction,length):
        
        #tsFrom = TransformState.makePosHpr(startPos, nHpr)
        tsTo = TransformState.makePosHpr(tsFrom.getPos() + direction * length, tsFrom.getHpr())

        shape = unit.bodyNP.node().getShape(0)
               
        #shape = BulletSphereShape(0.5)
        penetration = 0.0
        omasks=[]
        for u in self.units:
            omasks.append(u.bodyNP.getCollideMask())
            u.bodyNP.setCollideMask(BitMask32.bit(9))
        unit.bodyNP.setCollideMask(BitMask32.bit(30))
        """ for u in unit.isInCombatWith:
            u.bodyNP.setCollideMask(BitMask32.bit(30)) """
        #self.mountedKnightOfTheRealm.bodyNP.setCollideMask(BitMask32.bit(9))
        result = base.world.sweepTestClosest(shape, tsFrom, tsTo,BitMask32.bit(9))
        #unit.setCollideMask(BitMask32.bit(1))
        for i,u in enumerate(self.units):
            u.bodyNP.setCollideMask(omasks[i])
        if result.hasHit():
            print(result.hasHit())
            print(result.getHitPos())
            print(result.getHitNormal())
            print(result.getHitFraction())
            print(result.getNode())
            #self.z2.setPos(result.getHitPos())
            print("sweep test topos:", result.getToPos())
            
            return result.getHitFraction(),result.getHitPos()
        return 1.0,None
    
    def fallBackContactTest(self, unitNP,moveVec=Vec3(0,0,0)):
        unit = self.getSelectedUnit(unitNP.node())
        print("fallBackContactTest called for unit:", unit.unit.name)
        unit.bodyNP.setCollideMask(BitMask32.bit(1))
        for us in self.units:
            us.bodyNP.node().setTransformDirty()
        for u in unit.isInCombatWith:
            u.bodyNP.node().setTransformDirty()
        ghost = unit.bodyNP.node()
        
        result = base.world.contactTest(ghost)
        unit.bodyNP.setCollideMask(unit.bitmask)
        for contact in result.getContacts():
            node_name = contact.getNode1().getName()
            if node_name.startswith('UnitCollision-') :
                print(contact.getNode0())
                print(contact.getNode1())

                mpoint = contact.getManifoldPoint()
                print(mpoint.getDistance())
                print(mpoint.getAppliedImpulse())
                print(mpoint.getPositionWorldOnA())
                print(mpoint.getPositionWorldOnB())
                
                print(mpoint.getLocalPointA())
                print(mpoint.getLocalPointB())
                #np=render.find(f"**/{contact.getNode1().getName()}")
                #selected_unit = self.game.getSelectedUnit(contact.getNode1())

                contact_unit = self.getSelectedUnit(contact.getNode1())
                #contact_unit.bodyNP.node().setTransformDirty()
                if contact_unit in unit.isInCombatWith:
                    print("Contact with unit in combat, no fallback movement applied.")
                    continue
                self.z2.setPos(unit.bodyNP.getPos() + mpoint.getLocalPointA())
                selected_unit = unit
                """ if selected_unit.state == 'InCombat':
                    print("Unit in combat, cannot be moved in bounds again now!")
                    return """
                """ if selected_unit.state == 'IsFleeing':
                    print("Unit is fleeing out of the battle field, it is destroyed!")
                    base.world.removeRigidBody(selected_unit.bodyNP.node())
                    self.game.units.remove(selected_unit)
                    if selected_unit in self.game.player1Units:
                        self.game.player1Units.remove(selected_unit)
                    if selected_unit in self.game.player2Units:
                        self.game.player2Units.remove(selected_unit)
                    selected_unit.bodyNP.removeNode()
                    selected_unit.model.removeNode()
                    return """
                np=unit.bodyNP
                #np.setHpr(Vec3(H,0,0))
                cpos=Vec3(np.getPos())
                np.setPos(cpos+moveVec)
                return self.fallBackContactTest(unitNP,moveVec)
        
    def fallBack(self, loser,direction,length=10.0,rally=False,GG=False,flee=False):
        if loser.isEmpty():
            print("looser is destryed, no fallback")
            """ w=self.getSelectedUnit(winner.node())
            w.isInCombat=False
            w.isInCombatWith=[]
            w.isInCombatFlank=[] """
            #TODO: winner overruns
            return
        
        print(f"{loser.node().getName()} falls back!")
        
        
        loserPos=loser.getPos()
        newPos = loserPos + direction * length
        oldHpr=loser.getHpr()
        if not GG:
            loser.lookAt(loserPos + direction)
        #self.sweepTest(loser.node().getShape(0), loser.getPos(), loser.getHpr(), loser.getPos() + direction * length, loser.getHpr())
        #self.sweepTest(loser, direction, length)
        
        fleeHpr=loser.getHpr()
        if rally:
            r=Vec3(180,0,0)
        else:
            r=Vec3(0,0,0)
        newHpr=loser.getHpr() + r
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

        loser.setPos(newPos)
        bpos=loser.getPos()
        loser.setHpr(newHpr)
        if rally or flee:
            self.fallBackContactTest(loser,direction*.1)
        
        else:
            self.fallBackContactTest(loser,-direction*.1)

        newPos = loser.getPos()
        if (newPos - bpos).length() > 1.1:
            print("Adjusted fallback position due to collision:", bpos,newPos)
            loserUnit = self.getSelectedUnit(loser.node())
            loserUnit.endedInUnit=True

        loser.setPos(loserPos)
        loser.setHpr(oldHpr)

        move_interval2 = LerpPosInterval(
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
            move_interval2,
            #Func(self.fallBackContactTest, loser,direction),
            rotate_interval2,
            #Func(self.pursuitMove, winner, loser)
        )
        sequence.start()
        #if rally:
        #    self.fallBackContactTest(loser,direction)
        #return sequence

    def fallBack2(self, loser,direction,length=10.0,rally=False,GG=False,flee=False):
        if loser.isEmpty():
            print("looser is destryed, no fallback")
            """ w=self.getSelectedUnit(winner.node())
            w.isInCombat=False
            w.isInCombatWith=[]
            w.isInCombatFlank=[] """
            #TODO: winner overruns
            return
        
        print(f"{loser.node().getName()} falls back!")
        
        
        loserPos=loser.getPos()
        newPos = loserPos + direction * length
        oldHpr=loser.getHpr()
        if not GG:
            loser.lookAt(loserPos + direction)
        #self.sweepTest(loser.node().getShape(0), loser.getPos(), loser.getHpr(), loser.getPos() + direction * length, loser.getHpr())
        #self.sweepTest(loser, direction, length)
        
        fleeHpr=loser.getHpr()
        if rally:
            r=Vec3(180,0,0)
        else:
            r=Vec3(0,0,0)
        newHpr=loser.getHpr() + r
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

        loser.setPos(newPos)
        bpos=loser.getPos()
        loser.setHpr(newHpr)
        if rally or flee:
            self.fallBackContactTest(loser,direction*.1)
        
        else:
            self.fallBackContactTest(loser,-direction*.1)

        newPos = loser.getPos()
        if (newPos - bpos).length() > 1.1:
            print("Adjusted fallback position due to collision:", bpos,newPos)
            loserUnit = self.getSelectedUnit(loser.node())
            loserUnit.endedInUnit=True

        loser.setPos(loserPos)
        loser.setHpr(oldHpr)

        move_interval2 = LerpPosInterval(
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
            move_interval2,
            #Func(self.fallBackContactTest, loser,direction),
            rotate_interval2,
            #Func(self.pursuitMove, winner, loser)
        )
        #sequence.start()
        #if rally:
        #    self.fallBackContactTest(loser,direction)
        return sequence

    

    def pursuitMove(self, winner, loser):
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

    

    # ─── Persistence ──────────────────────────────────────────────────────

    def save_game_state(self, filename=None):
        """Delegate to persistence module."""
        return save_game_state(self, filename)

    def load_game_state(self, filename):
        """Delegate to persistence module."""
        load_game_state(self, filename)

    # ─── Camera Zoom & Controls ───────────────────────────────────────────

    def zoomIn(self):
        # Move camera closer (towards Y=0 from Y=-75)
        rot = LRotationf()
        rot.setHpr(self.camera.getHpr())
        fwd = rot.getForward()
        
        self.camera.setPos(self.camera.getPos() + fwd * 5)
    
    def zoomOut(self):
        # Move camera farther away (towards Y=-200 from Y=-75)
        rot = LRotationf()
        rot.setHpr(self.camera.getHpr())
        fwd = rot.getForward()
        
        self.camera.setPos(self.camera.getPos() - fwd * 5)


    

    # Add this method:
    # ─── List Builder & Army Management UI ────────────────────────────────

    def toggle_list_builder(self):
        if not self.list_builder_active:
            if self.list_builder is None:
                self.list_builder = ArmyListBuilderGUI(self)
            else:
                self.list_builder.show()
            self.list_builder_active = True
        else:
            self.list_builder.hide()
            self.list_builder_active = False
    
    def load_player1_army(self, filename="my_army.json"):
        """Load player 1's army from a file"""
        print(f"Loading Player 1 army from {filename}...")
        units = self.load_army_from_json(filename, player_num=1, start_pos=Point3(-20, -25, 0), spacing=12)
        if units:
            print(f"Player 1 army loaded: {len(units)} units")
        return units
    
    def load_player2_army(self, filename="my_army.json"):
        """Load player 2's army from a file"""
        print(f"Loading Player 2 army from {filename}...")
        units = self.load_army_from_json(filename, player_num=2, start_pos=Point3(-20, 25, 0), spacing=12)
        if units:
            print(f"Player 2 army loaded: {len(units)} units")
            # Set heading to face player 1
            for unit in units:
                unit.bodyNP.setH(180)
        return units

app = MyApp()
app.run()