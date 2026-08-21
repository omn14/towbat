# ─── Standard Library ───────────────────────────────────────────────────────
import math
import json
import os

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
from special_rules import build_special_rules
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
from collision_masks import CollisionMask as CM

# ─── Extracted Subsystems ────────────────────────────────────────────────────
from game_fsm import GamePhaseFSM
from spell_system import DevilsVisitSpell, RaiseDeadSpell
from persistence import save_game_state, load_game_state
from characters import JOIN_TAG
from combat_resolution import CombatResolver
from movement_system import MovementSystem
from terrain_system import TerrainManager, sees_over
from psychology import (PsychologySystem, select_general, select_battle_standard,
                       command_range)
from tutorial_system import TutorialManager
from cannon_fire import CannonFire
from bombardment import Bombardment
from debug_tools import DebugTools, debug_enabled
import gui_theme

# ─── Config ──────────────────────────────────────────────────────────────────
loadPrcFileData('', 'show-frame-rate-meter true')

# Weapon ranges are written in inches; the board is three world units per inch.
WORLD_UNITS_PER_INCH = 3.0


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
        self.speedMultiplier = 2.0
        

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
        """ self.smiley_copy = self.loader.loadModel('models/smiley')
        self.smiley_copy.reparentTo(self.render)
        self.smiley_copy.setPos(-50, 0, 0)
        self.smiley_copy.setScale(2) """

        # Position the camera above the plane, looking straight down
        self.disableMouse()
        self.camera.setPos(0, -75, 150)
        self.camera.lookAt(self.ground)
        #self.enableMouse()
        #self.camera.setP(-90)  # Pitch downwards
        self.setup_shader()
        self.setup_bullet()
        self.setup_campaign_map()

        # ── Terrain ───────────────────────────────────────────────────────
        self.terrain_manager = TerrainManager(self)
        terrain_map = "maps/sample_terrain.json"
        if os.path.exists(terrain_map):
            self.terrain_manager.load_from_json(terrain_map)
        else:
            # Fallback layout if the map file is missing.
            self.terrain_manager.add_terrain('forest', Point3(10, 2, 0.1), 14, 10)
            self.terrain_manager.add_terrain('hill',   Point3(-20, -5, 0.1), 12, 8)

        self.accept('q-up', self.pathTowardsMouse)
        self.accept('w-up', self.startTaskFunction,[self.taskLoopPathTowardsMouse, "taskLoopPathTowardsMouse"])
        
        self.accept('f5', self.save_game_state, ['quicksave.json'])  # F5 to quick save
        self.accept('f6', self.toggle_campaign_map)  # F6 toggles the campaign map
        self.accept('f9', self.load_game_state, ['quicksave.json'])  # F9 to quick load
        self.accept('f10', self.load_game_state, ['previous_phase.json'])  # F10 to load previous phase
        self.accept('f7', self.terrain_manager.toggle_debug)  # F7 toggles river water-detection band
        self.accept('wheel_up', self.zoomIn)  # Mouse wheel forward zooms in
        self.accept('wheel_down', self.zoomOut)  # Mouse wheel backward zooms out
        self.analyzer = GameStateAnalyzer(self)

        self.awaitingChoice = False
        self.resolvingCombat = False
        self.debugTextUnit = self.setup_text_node(text="Debug Info", pos=(-1.3, -0.9), scale=0.05, color=gui_theme.HINT_FG)
        self.debugTextUnit.setText("Debug Info test")

        self.debugTextInfo = self.setup_text_node(text="Debug Info", pos=(0.7, -0.8), scale=0.05, color=gui_theme.HINT_FG)
        self.moveArceDistance = 0
        self.debugTextInfo.setText("Debug Arch test")

        self.diceInfoText = self.setup_text_node(text="Dice Info", pos=(-0.7, 0.55), scale=0.05, color=gui_theme.GOLD)

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

        #self.load_player1_army("strategy_armies/gunline.json")
        #self.load_player2_army("strategy_armies/horde_rush.json")

        #self.load_player1_army("strategy_armies/hammer_and_anvil.json")
        #self.p1army="strategy_armies/orc_and_goblin_horde.json"
        self.p1army="strategy_armies/Bm_army.json"
        self.load_player1_army(self.p1army)
        self.load_player2_army("strategy_armies/vampire_counts_legion.json")


        self.unitToMove=self.player1Units[0]
        self.accept('mouse3', self.moveUnit,[self.unitToMove])
        #self.messenger.toggleVerbose()
        self.roundCounter = RoundCounter(self,16)

        self.debugText = self.setup_text_node(text="Debug Info", pos=(-1.3, 0.9), scale=0.05, color=gui_theme.CREAM)
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
        # TEMP: disable P2 AI so player 2 deploys and acts manually.
        self.AIplayer2.active = False
        async def auppp():
            for unit in self.player2Units:
                action = await taskMgr.add(self.AIplayer2.take_turn())
                if action.action_type == 'end_phase':
                    break
        #self.accept('a-up', lambda: taskMgr.add(self.AIplayer2.take_turn()))
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
        self.movement = MovementSystem(self)
        self.psychology = PsychologySystem(self)
        self.tutorial = TutorialManager(self)
        self.cannon = CannonFire(self)
        self.bombard = Bombardment(self)
        # Developer tools; inert unless WH_DEBUG is set or --debug is passed.
        self.debug_tools = DebugTools(self) if debug_enabled() else None
        self.accept('t', self.start_tutorial)
        # Debug: force a Panic test on the selected unit (Phase 0 wiring).
        self.accept('shift-p', lambda: self.psychology.panic_test(self.unitToMove, cause="debug"))

        self.fsm.request("DeployPhase")

        self.rectangleLine = self.drawRectangle(center=Point3(0, 0, 1), width=72, height=48, color=Vec4(1, 1, 0, 1))
        self.deploymentLine = self.drawRectangle(center=Point3(0, 0, .5), width=72, height=24, color=Vec4(1, 1, 1, 1))

        #self.z2= loader.loadModel("models/zup-axis")
        #self.z2.reparentTo(render)
        
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
        self.ball.setPos(-15,0,-100)

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
        # Load the JSON file
        try:
            with open(filename, 'r') as f:
                raw = json.load(f)
        except FileNotFoundError:
            print(f"Error: File {filename} not found!")
            return []
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in {filename}!")
            return []

        # Support list-builder format: {"budget": N, "units": [...]}
        if isinstance(raw, dict) and 'units' in raw:
            army_data = raw['units']
        else:
            army_data = raw

        created_units = []
        current_x = start_pos.x
        RESERVE_GAP = 1.0  # small gap between densely-packed reserve units
        for idx, army_unit_data in enumerate(army_data):
            unit_name = army_unit_data['name']
            # Include player_num in the name so P1 and P2 units with the same
            # model type get unique collision-node / lookup names.
            graphics_name = f"P{player_num}_{unit_name.replace(' ', '')}{idx}"
            unit_graphics = self._create_unit(army_unit_data, player_num, graphics_name)
            if unit_graphics is None:
                continue
            # Pack units edge-to-edge by their actual width, starting at start_pos.x.
            w = getattr(unit_graphics, 'unitWidth', spacing) or spacing
            unit_graphics.bodyNP.setPos(current_x + w / 2, start_pos.y, start_pos.z)
            created_units.append(unit_graphics)
            current_x += w + RESERVE_GAP
            print(f"Loaded unit: {unit_name} ({army_unit_data['nmodels']} models, "
                  f"{army_unit_data['files']}x{army_unit_data['ranks']})")

        print(f"Successfully loaded {len(created_units)} units from {filename}")
        return created_units

    # ─── Unit Construction ────────────────────────────────────────────────

    # Unit display name → model path, class and default colour.
    UNIT_MODEL_MAPPING = {
        # ── Bretonnia ──────────────────────────────────────────────
        'Man at Arms':                  {'path': 'models/bret_bowmen.bam',      'class': model,                  'color': (1, 0, 0, 1)},
        'Man_at_Arm':                   {'path': 'models/bret_bowmen.bam',      'class': model,                  'color': (1, 0, 0, 1)},
        'Mounted Knight of the Realm':  {'path': 'models/bret_knight.bam',      'class': MountedKnightOfTheRealm,'color': (1, 0, 0, 1)},
        'Pegasus Knight':               {'path': 'models/bret_knight.bam',      'class': PegasusKnight,          'color': (1, 0, 0, 1)},
        'Grail Knight':                 {'path': 'models/bret_knight.bam',      'class': GrailKnight,            'color': (1, 0, 0, 1)},
        'Peasant Bowman':               {'path': 'models/bret_bowmen.bam',      'class': PeasantBowman,          'color': (1, 0, 0, 1)},
        'Battle Pilgrim':               {'path': 'models/bret_bowmen.bam',      'class': BattlePilgrim,          'color': (1, 0, 0, 1)},
        # ── Grand Cathay ───────────────────────────────────────────
        'Jade Warrior':                 {'path': 'models/jade_warrior.bam',     'class': JadeWarrior,            'color': (1, 1, 0, 1)},
        'Jade Lancer':                  {'path': 'models/jade_lancer.bam',      'class': JadeLancer,             'color': (1, 1, 0, 1)},
        'Peasant Spearman':             {'path': 'models/jade_warrior.bam',     'class': PeasantSpearman,        'color': (1, 1, 0, 1)},
        'Iron Hail Gunner':             {'path': 'models/jade_warrior.bam',     'class': IronHailGunner,         'color': (1, 1, 0, 1)},
        # ── Orc & Goblin Tribes ────────────────────────────────────
        'Night Goblin':                 {'path': 'models/goblin_archers.bam',   'class': NightGoblin,            'color': (0, 1, 0, 1)},
        'Goblin Wolf Rider':            {'path': 'models/goblin_wolfriders.bam','class': GoblinWolfRider,        'color': (0, 1, 0, 1)},
        'Orc Boyz':                     {'path': 'models/goblin_archers.bam',   'class': OrcBoyz,                'color': (0, 1, 0, 1)},
        'Orc Boy':                      {'path': 'models/goblin_archers.bam',   'class': OrcBoyz,                'color': (0, 1, 0, 1)},
        'Black Orc':                    {'path': 'models/goblin_archers.bam',   'class': BlackOrc,               'color': (0, 1, 0, 1)},
        'Orc Boar Boy':                 {'path': 'models/goblin_wolfriders.bam','class': OrcBoarBoy,             'color': (0, 1, 0, 1)},
        'Boar Boy':                     {'path': 'models/goblin_wolfriders.bam','class': OrcBoarBoy,             'color': (0, 1, 0, 1)},
        'Wolf Rider':                   {'path': 'models/goblin_wolfriders.bam','class': GoblinWolfRider,        'color': (0, 1, 0, 1)},
        'Troll':                        {'path': 'models/goblin_archers.bam',   'class': Troll,                  'color': (0, 1, 0, 1)},
        # ── Vampire Counts ─────────────────────────────────────────
        'Black Knight':                 {'path': 'models/black_knights.bam',    'class': BlackKnight,            'color': (0, 0, 1, 1)},
        'Zombie':                       {'path': 'models/zombies.bam',          'class': Zombie,                 'color': (0, 0, 1, 1)},
        'Dire Wolf':                    {'path': 'models/dire_wolves.bam',      'class': DireWolf,               'color': (0, 0, 1, 1)},
        'Necromancer':                  {'path': 'models/zombies.bam',          'class': Necromancer,            'color': (0, 0, 1, 1)},
        'Skeleton Warrior':             {'path': 'models/zombies.bam',          'class': SkeletonWarrior,        'color': (0, 0, 1, 1)},
        'Crypt Ghoul':                  {'path': 'models/zombies.bam',          'class': CryptGhoul,             'color': (0, 0, 1, 1)},
        'Grave Guard':                  {'path': 'models/black_knights.bam',    'class': GraveGuard,             'color': (0, 0, 1, 1)},
        # ── Lizardmen ──────────────────────────────────────────────
        'Saurus Warrior':               {'path': 'models/jade_warrior.bam',     'class': SaurusWarrior,          'color': (0, 1, 1, 1)},
        'Skink':                        {'path': 'models/goblin_archers.bam',   'class': Skink,                  'color': (0, 1, 1, 1)},
        'Temple Guard':                 {'path': 'models/jade_warrior.bam',     'class': TempleGuard,            'color': (0, 1, 1, 1)},
        'Cold One Rider':               {'path': 'models/jade_lancer.bam',      'class': ColdOneRider,           'color': (0, 1, 1, 1)},
    }

    # Every unit in a player's army shares one colour.
    PLAYER_COLORS = {1: (1, 0, 0, 1), 2: (0, 0, 1, 1)}

    # Default mount per rider name, used when the army data names no mount.
    MOUNT_NAME_MAP = {
        'Jade Lancer':                 'Cathayan Warhorse',
        'Mounted Knight of the Realm': 'Bretonnian Warhorse',
        'Goblin Wolf Rider':           'Giant Wolf',
        'Wolf Rider':                  'Giant Wolf',
        'Black Knight':                'Skeletal Steed',
        'Pegasus Knight':              'Barded Pegasus',
        'Grail Knight':                'Bretonnian Warhorse',
        'Orc Boar Boy':                'War Boar',
        'Boar Boy':                    'War Boar',
        'Cold One Rider':              'Cold One',
    }

    def _create_unit(self, army_unit_data, player_num, graphics_name):
        """Build one unitGraphics from an army-list entry and add it to the scene.

        Shared by the army loader and save-game restore. Returns the
        unitGraphics, or None if creation failed.
        """
        mounted_classes = [JadeLancer, MountedKnightOfTheRealm, GoblinWolfRider, BlackKnight,
                           PegasusKnight, GrailKnight, OrcBoarBoy, ColdOneRider]
        unit_name = army_unit_data['name']
        nmodels = army_unit_data['nmodels']
        files = army_unit_data['files']
        ranks = army_unit_data['ranks']
        model_info = self.UNIT_MODEL_MAPPING.get(unit_name, {
            'path': 'models/jade_warrior.bam', 'class': model, 'color': (0.5, 0.5, 0.5, 1)})
        player_color = self.PLAYER_COLORS.get(player_num, (0.5, 0.5, 0.5, 1))
        try:
            model_class = model_info['class']
            # Prefer the mount named in the army data; fall back to the default map.
            mount_name = army_unit_data.get('mount') or self.MOUNT_NAME_MAP.get(unit_name)
            mount_unit = None
            if mount_name:
                mount_model = model(mount_name, "")
                mount_unit = unit(f"{mount_name} Unit", mount_model, nmodels, files, ranks)

            if model_class in mounted_classes:
                model_instance = model_class(unit_name, "", mountUnit=mount_unit)
            else:
                model_instance = model_class(unit_name, "")

            # Attach a data-driven mount to any unit that didn't already get one.
            # A chariot's draught beasts are listed as a mount in the roster but
            # are part of its own profile, so they must not be attached twice.
            if mount_unit is not None and not model_instance.is_mounted() \
                    and not model_instance.is_chariot():
                model_instance.attach_mount(mount_unit)

            # Equip data-driven weapons from the army list (e.g. imported rosters).
            for w in army_unit_data.get('weapons', []):
                wdict = dict(w)
                if wdict.get('tag') == 'ranged' and not wdict.get('ranged_strength'):
                    wdict['ranged_strength'] = model_instance.shooting_strength()
                name = wdict.get('name', 'weapon')
                # A class/catalogue weapon (e.g. a war machine's piece) is the
                # fresh source of truth; saved data only fills fields it lacks.
                if name in model_instance.weapons:
                    existing = model_instance.weapons[name]
                    for key, value in wdict.items():
                        existing.setdefault(key, value)
                else:
                    model_instance.weapons[name] = wdict

            # Merge data-driven special rules from the army list (e.g. the
            # Skirmishers upgrade on an imported roster) and wire their hooks.
            extra_rules = army_unit_data.get('special_rules') or []
            if extra_rules:
                current = model_instance.characteristics.get('Special Rules')
                current = list(current) if isinstance(current, list) else []
                for rname in extra_rules:
                    if rname and rname not in current:
                        current.append(rname)
                model_instance.characteristics['Special Rules'] = current
                have = {r.get('name') for r in model_instance.special_rules
                        if isinstance(r, dict)}
                for entry in build_special_rules(model_instance):
                    if isinstance(entry, dict) and entry.get('name') not in have:
                        model_instance.special_rules.append(entry)
                        have.add(entry.get('name'))

            # Derive the armour save from the roster's armour equipment.
            armour = army_unit_data.get('armour')
            if armour:
                model_instance.set_armour(armour)

            unit_instance = unit(f"{unit_name} Unit", model_instance, nmodels, files, ranks)
            unit_graphics = unitGraphics(
                self, graphics_name, model_info['path'], unit_instance,
                scale=1.0, BulletWorld=self.world, color=player_color)

            self.units.append(unit_graphics)
            if player_num == 1:
                self.player1Units.append(unit_graphics)
            else:
                self.player2Units.append(unit_graphics)
            return unit_graphics
        except Exception as e:
            print(f"Error creating unit {unit_name}: {e}")
            return None

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

    def drawProjectileTrajectory(self,startPos,endPos,n=20,color=(1,0,0,1)):
        # Remove existing trajectory line if it exists
        if hasattr(self, 'trajectoryLine'):
            self.trajectoryLine.removeNode()
        
        line_segs = LineSegs()
        line_segs.setColor(*color)  # trajectory colour (green short / red long range)
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
        balls = []
        for i in range(n):
            ball = loader.loadModel("smiley")
            ball.reparentTo(render)
            ball.setPos(startPos+Vec3(random.uniform(-2,2),random.uniform(-2,2),0))
            balls.append(ball)
            self.projectiles.append(ball)
            pos = endPos + Vec3(random.uniform(-2,2),random.uniform(-2,2),0)
            duration = random.uniform(0.9, 1.1)/self.speedMultiplier
            trajectory = ProjectileInterval(ball, duration=duration,
                                            endPos=pos)
            self.trajectories.append(trajectory)
        # Create a Parallel interval to run all trajectories simultaneously
        parallel_trajectories = Parallel(*self.trajectories)
        #parallel_trajectories.start()
        #for trajectory in self.trajectories:
        #    trajectory.start()
        return parallel_trajectories, balls

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
            # Greyed out signals the unit can't settle here; no per-frame spam.
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

            result = base.world.rayTestClosest(pFrom, pTo, BitMask32.bit(1))

            if result.hasHit():
                hitPos = result.getHitPos()
                unit.bodyNP.lookAt(hitPos)
                unit.bodyNP.setP(0)   # reform turns on the spot — keep upright
                unit.bodyNP.setR(0)
                #unit.hasMovedThisTurn=True
                #unit.updateTextNode()
        if self.signal and not contact:
            self.signal = False
            return task.done
        self.signal = False
        return task.cont
    
    def giveSignal(self):
        if self.awaitingChoice:
            return
        self.signal = True
        return

    def startFreeReform(self, unit, on_done=None):
        """Interactive free reform (rotate/reposition, left-click to confirm).
        Used by auto-rally after a Fall Back in Good Order.  Calls *on_done*
        once the reform is confirmed."""
        self._reformDone = on_done
        self.ignore('mouse1')
        self.accept('mouse1', self.giveSignal)
        taskMgr.add(self._freeReformTask, "freeReformUnitTask",
                    extraArgs=[unit], appendTask=True)

    def _freeReformTask(self, unit, task):
        result = self.freeReformUnit(unit, task)
        if result == task.done:
            self.ignore('mouse1')
            self.accept('mouse1', self.setActiveUnit,
                        [self.setActiveUnitTask, self.setActiveUnitTaskName])
            cb = getattr(self, '_reformDone', None)
            self._reformDone = None
            if cb:
                cb()
        return result

    async def rollLeadershipDice(self):
        """Roll the physical 2D6 of a Leadership test and return their values."""
        terningerLd = []
        for i in range(2):
            terning = Dice(self.world, position=Vec3(20+i*2,0,10), size=1.0,color=(1,0,0,1))
            terningerLd.append(terning)
        for terning in terningerLd:
            terning.roll()
        await taskMgr.add(checkDice, "checkDiceTaskFlee", extraArgs=[terningerLd], appendTask=True)
        ldDice = [terning.currentValue for terning in terningerLd]
        for terning in terningerLd:
            terning.remove(self.world)
        return ldDice

    async def rallyUnit(self, unit):
        # Attempts to rally a fleeing unit by testing against its Leadership characteristic and allowing a free reform on success
        Ld, general = self.psychology.leadership_of(unit)
        if general is not None:
            print(f"{unit.unit.name} rallies on the General's Leadership "
                  f"({general.unit.name}, Ld {Ld}) — Inspiring Presence.")
        ldDice = await self.rollLeadershipDice()
        leadership_score = sum(ldDice)
        print("Leadership dice results for fleeing unit:", ldDice, "sum:", leadership_score,
              "Ld:", Ld)
        bsb = self.psychology.battle_standard_of(unit)
        if leadership_score > Ld and bsb is not None:
            print(f"{unit.unit.name} re-rolls its failed Rally test "
                  f"(Hold Your Ground: {bsb.unit.name}).")
            ldDice = await self.rollLeadershipDice()
            leadership_score = sum(ldDice)
            print("Re-rolled Leadership dice:", ldDice, "sum:", leadership_score)
        if leadership_score <= Ld:
            print(f"Rallying unit: {unit.unit.name}")
            self.ignore('mouse1')
            self.accept('mouse1', self.giveSignal)
            await taskMgr.add(self.freeReformUnit, "freeReformUnitTask", extraArgs=[unit], appendTask=True)
            print(f"Unit {unit.unit.name} has rallied successfully.")
            self.ignore('mouse1')
            self.accept('mouse1', self.setActiveUnit,[self.taskLoopStrategy, "taskLoopStrategy"])
            unit.request("Idle")
        else:
            print(f"Unit {unit.unit.name} fails to rally and keeps fleeing.")
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
        # War-machine cannons use the dedicated Cannon Fire targeting flow.
        if self.cannon.is_cannon(self.unitToMove):
            self.cannon.begin_targeting(self.unitToMove)
            return task.done
        # Bombardment war machines (Mortar, etc.) use the blast-template flow.
        if self.bombard.is_bombardment(self.unitToMove):
            self.bombard.begin_targeting(self.unitToMove)
            return task.done
        if self.unitToMove.unit.model.equipedWeapon is None:# or not self.unitToMove.unit.model.equippedWeapon.is_ranged:
            print("Unit has no equiped weapon equipped, cant shoot.")
            return task.done
        r=False
        for weapon in self.unitToMove.unit.model.weapons:
            if self.unitToMove.unit.model.weapons.get(weapon).get('tag') == 'ranged':
                r=True
                self.unitToMove.unit.model.equip_weapon(weapon)
        if not r:
            print(f"[Shooting] {self.unitToMove.unit.name} has no ranged weapon.")
            return task.done
        _rw = self.unitToMove.unit.model.equipedWeapon
        _model = self.unitToMove.unit.model
        _all_round = _model.has_all_round_vision()
        _why = 'skirmisher' if _model.is_skirmisher() else 'firing platform'
        print(f"[Shooting] {self.unitToMove.unit.name} ready with {_rw.get('name')} "
              f"(R{_rw.get('ranged_range','?')} S{_rw.get('ranged_strength','?')} "
              f"AP-{_rw.get('ranged_AP', 0)}) | arc={f'360 ({_why})' if _all_round else '90'}")
        self.shootingArcPoints = self.shootingArc(self.unitToMove.bodyNP.getPos(render), 
                                                       num_points=80, 
                                                       rotationangle=self.unitToMove.bodyNP.getH()+45,
                                                       radius=self.unitToMove.unit.model.equipedWeapon.get('ranged_range',18)*3/100,
                                                       full_circle=_all_round)
        self.checkArrowsTerrain()
        self.setGroundOverlay(True, self.shootingArcPoints)
        # Half-range boundary: inside = short range, outside = long range (-1).
        half = self.unitToMove.unit.model.equipedWeapon.get('ranged_range', 18) * 1.5
        self.drawRangeRing(self.unitToMove.bodyNP.getPos(), half)
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
        spellChoices = []
        spellClasses = []
        for spell in self.unitToMove.unit.model.spells:
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
                                                       num_points=80, rotationangle=self.unitToMove.bodyNP.getH()+45, radius=radius,
                                                       full_circle=self.unitToMove.unit.model.has_all_round_vision())
        self.setGroundOverlay(True, self.shootingArcPoints)
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

    def targetUnderMouse(self, mask=BitMask32.bit(3)):
        """The targetable unit the mouse is over, or None."""
        if not base.mouseWatcherNode.hasMouse():
            return None
        pMouse = base.mouseWatcherNode.getMouse()
        pFrom, pTo = Point3(), Point3()
        base.camLens.extrude(pMouse, pFrom, pTo)
        pFrom = render.getRelativePoint(base.cam, pFrom)
        pTo = render.getRelativePoint(base.cam, pTo)
        result = self.world.rayTestClosest(pFrom, pTo, mask)
        return self.getSelectedUnit(result.getNode()) if result.hasHit() else None

    def taskShootingTrajectoryDrawLine(self, task):
        # Aim at whatever targetable unit is under the mouse, even where the arc
        # was clipped short of it — a unit on a hill is seen over the ones in
        # front of it, so the arc alone would not show what is being aimed at.
        target = self.targetUnderMouse()
        if target is not None:
            aim = target.bodyNP.getPos()
        elif self.checkIfInsidePolygon(self.mousePosOnGround,
                                       self.coordsToWorld(self.shootingArcPoints)):
            aim = self.mousePosOnGround
        else:
            return task.cont
        weapon = self.unitToMove.unit.model.equipedWeapon or {}
        half = weapon.get('ranged_range', 0) * 1.5
        dist = (self.unitToMove.bodyNP.getPos() - aim).length()
        long_range = bool(half and dist > half)
        color = (1, 0.35, 0.35, 1) if long_range else (0.35, 1, 0.35, 1)
        self.trajectoryLine = self.drawProjectileTrajectory(
            self.unitToMove.bodyNP.getPos(), aim, color=color)
        readout = "LONG RANGE  (-1 To Hit)" if long_range else "Short range"
        if target is not None:
            readout = f"{target.unit.name}  —  {readout}"
        self.debugTextInfo.setText(readout)
        return task.cont

    def drawRangeRing(self, center, radius, segments=64, color=(1, 1, 0, 0.8)):
        """Draw the half-range boundary ring for the aiming unit."""
        if getattr(self, 'rangeRing', None):
            self.rangeRing.removeNode()
        ls = LineSegs()
        ls.setColor(*color)
        ls.setThickness(2.0)
        for i in range(segments + 1):
            a = 2 * math.pi * i / segments
            x = center.x + radius * math.cos(a)
            y = center.y + radius * math.sin(a)
            if i == 0:
                ls.moveTo(x, y, center.z + 0.2)
            else:
                ls.drawTo(x, y, center.z + 0.2)
        self.rangeRing = render.attachNewNode(ls.create())
        self.rangeRing.setName("HalfRangeRing")
        return self.rangeRing
    
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

    def losBlockUnit(self, from_pos, to_pos, candidates):
        """Return the point where line of sight is blocked by an intervening
        unit (its far edge, so the blocker stays targetable but anything behind
        it is hidden), or None.  *candidates* is a precomputed list of
        (x, y, radius_sq) footprints to keep this cheap inside the arc loop."""
        if not candidates:
            return None
        dx = to_pos.x - from_pos.x
        dy = to_pos.y - from_pos.y
        length = math.hypot(dx, dy)
        steps = max(int(length / 0.5), 2)
        blocker = None        # first intervening unit the sight line enters
        exit_point = None     # last sample still inside it (its far edge)
        for i in range(1, steps + 1):   # skip i=0 (the shooter's own position)
            t = i / steps
            sx = from_pos.x + dx * t
            sy = from_pos.y + dy * t
            inside = False
            for c in candidates:
                cx, cy, r2 = c
                if (sx - cx) ** 2 + (sy - cy) ** 2 <= r2:
                    if blocker is None:
                        blocker = c
                    if c is blocker:
                        inside = True
            if blocker is not None:
                if inside:
                    exit_point = Point3(sx, sy, 0)   # advance through the blocker
                else:
                    return exit_point                # past its far edge → block
        # Sight line ended within the blocker (it's the target) → not blocked.
        return None

    def checkArrowsTerrain(self,mask=BitMask32.bit(3)):
        shooter = self.unitToMove
        pFrom = Point3(shooter.bodyNP.getX(), shooter.bodyNP.getY(), 0)
        # On a hill the shooter is elevated: it sees over units on lower ground,
        # but woods (and other hills) still block sight. A unit must be
        # *entirely* on the hill to claim the benefit (Official FAQ 1.5.3).
        hill = self.movement.hillUnderUnit(shooter)
        # Precompute unit-footprint blockers once.
        candidates = []
        for unit in self.units:
            if unit is shooter:
                continue
            up = unit.bodyNP.getPos()
            if hill is not None:
                on_hills = [t for t in self.terrain_manager.get_all_terrain_at(up)
                            if t.terrain_type == 'hill']
                if not on_hills:
                    continue                      # seen over from the high ground
                # On the same hill, only a unit nearer its top blocks.
                if hill in on_hills and sees_over(shooter.bodyNP.getPos(), up,
                                                  hill.center):
                    continue
            radius = max(getattr(unit, 'unitWidth', 3.0),
                         getattr(unit, 'unitHeight', 3.0)) / 2.0
            candidates.append((up.x, up.y, radius * radius))
        for n,point in enumerate(self.shootingArcPoints):
            point = point * 2
            point -= Vec2(1,1)
            point = point * 50
            pTo = Point3(point.x, point.y, 0)
            # Clip where the line of sight is first blocked — by LoS-blocking
            # terrain (forest/hill) or an intervening unit, whichever is nearer.
            terrain_block = self.terrain_manager.los_block_point(pFrom, pTo)
            unit_block = self.losBlockUnit(pFrom, pTo, candidates)
            block = None
            if terrain_block is not None and unit_block is not None:
                dt = (terrain_block - pFrom).lengthSquared()
                du = (unit_block - pFrom).lengthSquared()
                block = terrain_block if dt <= du else unit_block
            else:
                block = terrain_block or unit_block
            if block is not None:
                hxy = self.coordsToLocal([Vec2(block.x, block.y)])[0]
                self.shootingArcPoints[n] = Vec2(hxy.x, hxy.y)
        
                
        """ if not hit:
            print(f"no targets in shooting arc for unit {self.unitToMove.unit.name}")
            #self.ground.setShaderInput("isActive", False)
            if taskMgr.hasTaskNamed("taskShootingTrajectoryDrawLine"):
                taskMgr.remove("taskShootingTrajectoryDrawLine") """
    
    def checkArrows(self,mask=BitMask32.bit(3)):
        hit = False
        for point in self.shootingArcPoints:
            point = point * 2
            point -= Vec2(1,1)
            point = point * 50
            pFrom = self.unitToMove.bodyNP.getPos(render)
            pTo = Point3(point.x, point.y, pFrom.z)

            result = self.world.rayTestClosest(pFrom, pTo, BitMask32.bit(1))

            if result.hasHit():
                for c in result.getNode().getChildren():
                    if "Model" in c.getName():
                        np = NodePath.anyPath(c)
                        np.setColor(1,0,1,1)
                        NodePath.anyPath(result.getNode()).setCollideMask(mask)
                        #self.toCleanup.append(np)
                        hit = True
        hit = self.markHillTargets(mask) or hit
        if not hit:
            print(f"[Shooting] no targets in {self.unitToMove.unit.name}'s arc.")
            #self.ground.setShaderInput("isActive", False)
            if taskMgr.hasTaskNamed("taskShootingTrajectoryDrawLine"):
                taskMgr.remove("taskShootingTrajectoryDrawLine")

    def markHillTargets(self, mask=BitMask32.bit(3)):
        """Vantage Point: a unit entirely on a hill is seen across or through
        other units, so it stays targetable even where the arc was clipped
        short of it (Rulebook p. 271). Only units are seen over; a wood or
        another hill in the way still blocks.
        """
        shooter = self.unitToMove
        weapon = shooter.unit.model.equipedWeapon or {}
        reach = (weapon.get('ranged_range') or 0) * WORLD_UNITS_PER_INCH
        if reach <= 0:
            return False
        origin = shooter.bodyNP.getPos()
        pFrom = Point3(origin.x, origin.y, 0)
        marked = False
        for unit in self.units:
            if unit is shooter or unit.bodyNP.isEmpty():
                continue
            up = unit.bodyNP.getPos()
            if (up - origin).length() > reach:
                continue
            if not self.movement.entirelyOnHill(unit):
                continue
            pTo = Point3(up.x, up.y, 0)
            block = self.terrain_manager.los_block_point(pFrom, pTo)
            if block is not None and \
               (block - pFrom).lengthSquared() < (pTo - pFrom).lengthSquared():
                continue
            for c in unit.bodyNP.node().getChildren():
                if "Model" in c.getName():
                    NodePath.anyPath(c).setColor(1, 0, 1, 1)
            unit.bodyNP.setCollideMask(mask)
            marked = True
        return marked

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
            result = self.world.rayTestClosest(pFrom, pTo, CM.HOVER_PICK)
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
        if self.awaitingChoice:
            return
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
                self.shootAt(self.unitToMove, selected_unit)
                """ attacker = self.unitToMove.unit
                defender = selected_unit.unit
                print(attacker.name, "shooting an arrow at",defender.name)
                #attacker.model.equip_weapon('short bow')
                attacks, total_hits, suffered_wounds,  saves_made, total_wounds = simulate_battle(attacker, defender,charge=False)
                self.printBattleResults(self.unitToMove, selected_unit, attacks, total_hits, suffered_wounds, saves_made, total_wounds)
                #unit.model.setColor(unit.color)
                #unit.bodyNP.setCollideMask(BitMask32.bit(unit.bitmask))
                self.unitToMove.bodyNP.setCollideMask(BitMask32.bit(4))
                selected_unit.bodyNP.setCollideMask(BitMask32.bit(4))
                self.shootingAnimation(self.unitToMove,selected_unit,total_wounds) """
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

    def shootAt(self, attackerUnit, defenderUnit):
        attacker = attackerUnit.unit
        defender = defenderUnit.unit
        weapon = attacker.model.equipedWeapon or {}
        # Long range (beyond half the weapon's range) imposes -1 To Hit.
        _half = weapon.get('ranged_range', 0) * 1.5
        _dist = (attackerUnit.bodyNP.getPos() - defenderUnit.bodyNP.getPos()).length()
        attacker.model.at_long_range = bool(_half and _dist > _half)
        # US1 Skirmisher target imposes -1 To Hit on the shooter.
        attacker.model.target_skirmisher = (defender.model.is_skirmisher()
                                             and defender.model.unit_strength() == 1)
        _tag = 'LONG RANGE, -1 To Hit' if attacker.model.at_long_range else 'short range'
        # Vantage Point: a unit entirely on a hill fires with one extra rank.
        extra_ranks = 1 if self.movement.entirelyOnHill(attackerUnit) else 0
        print(f"\n[Shooting] {attacker.name} -> {defender.name} | "
              f"{weapon.get('name', 'weapon')} | range {_dist:.0f}\" ({_tag})"
              f"{'  | on a hill: +1 firing rank' if extra_ranks else ''}")
        # A joined character with a missile weapon replaces one unit shooter and
        # fires with its own profile.
        joinedRule = next((r for r in attacker.model.special_rules
                           if isinstance(r, dict) and r.get('tag') == JOIN_TAG), None)
        charShooter = None
        if joinedRule:
            cu = joinedRule['characterUnit']
            cw = cu.model.equipedWeapon
            if cw and cw.get('tag') == 'ranged':
                charShooter = cu
        origFiles = attacker.files
        if charShooter and origFiles > 1:
            attacker.files -= 1
        attacks, total_hits, suffered_wounds,  saves_made, total_wounds = simulate_battle(
            attacker, defender, charge=False, extra_ranks=extra_ranks)
        attacker.files = origFiles
        self.printBattleResults(attackerUnit, defenderUnit, attacks, total_hits, suffered_wounds, saves_made, total_wounds)
        if charShooter:
            c_attacks, c_hits, c_suffered, c_saves, c_wounds = simulate_battle(charShooter, defender, charge=False)
            self.printBattleResults(attackerUnit, defenderUnit, c_attacks, c_hits, c_suffered, c_saves, c_wounds)
            total_wounds += c_wounds
        attackerUnit.bodyNP.setCollideMask(BitMask32.bit(4))
        defenderUnit.bodyNP.setCollideMask(BitMask32.bit(4))
        attackerUnit.hasAttackedThisTurn = True
        taskMgr.add(self.shootingAnimation(attackerUnit, defenderUnit, total_wounds))

    async def shootingAnimation(self,attackerUnit,defenderUnit,total_wounds):
        
        #self.p.start(parent=render, renderParent=render)
        self.p.setPos(defenderUnit.bodyNP.getPos())
        

        #self.p_miss.start(parent=render, renderParent=render)
        self.p_miss.setPos(defenderUnit.bodyNP.getPos())

        self.cameraShake(intensity=0.5, duration=0.3)
        
        
        parTra, balls = self.spawnProjectiles(5,attackerUnit.bodyNP.getPos(),defenderUnit.bodyNP.getPos())
        """ seq = Sequence(parTra,
                       Func(self.p.start, parent=render, renderParent=render),
                       Func(taskMgr.doMethodLater, 4.0, lambda task: self.p.disable(), 'stopParticles'),
                       Func(self.p_miss.start, parent=render, renderParent=render),
                       #Func(self.removeModelsFromUnit, defenderUnit, total_wounds),
                       Func(taskMgr.doMethodLater, 4.0, lambda task: self.p_miss.disable(), 'stopMissParticles'),
                       #Func(taskMgr.doMethodLater, 3.0, lambda task: base.messenger.send('unit-move-complete'), 'unitmovecompletetask')
                       )
         """
        #seq.start()
        self.p.start(parent=render, renderParent=render)
        self.p_miss.start(parent=render, renderParent=render)
        await parTra
        self.applyWounds(defenderUnit, total_wounds)
        # Heavy casualties from shooting can trigger a Panic test.
        self.psychology.check_heavy_casualties(defenderUnit, 'shooting', attacker=attackerUnit)
        await Task.pause(2.0 / self.speedMultiplier)
        self.p.disable()
        self.p_miss.disable()
        for ball in balls:
            ball.removeNode()
        messenger.send('unit-move-complete')
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
        Uses the shared medieval theme font and shadow.
        
        Args:
            text: The text to display
            pos: (x, y) position in aspect2d coordinates (-1 to 1)
            scale: Text scale
            color: Text color as (r, g, b, a) tuple
        
        Returns:
            TextNode object that can be updated with .setText()
        """
        return gui_theme.styled_text(
            text=text, pos=pos, scale=scale, fg=color,
            align=TextNode.ACenter,
        )

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
                country_name = hit_node_name.split("_")[0]
                self.country_fsm.selectCountry(country_name)
            else:
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

    def setGroundOverlay(self, active, points=None):
        """Set the movement/shooting range overlay on the ground card and
        broadcast it to terrain so the indicator wraps over hills/water."""
        if points is not None:
            self.ground.setShaderInput("polygonpoints", points)
        self.ground.setShaderInput("isActive", active)
        if hasattr(self, 'terrain_manager'):
            self.terrain_manager.set_move_overlay(active, points)

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
        #self.debugNP.show()

        """ # Add simple Bullet collision geometry (sphere) to smiley_copy

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
        self.world.attachRigidBody(smiley_copy_body) """
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

    # ─── Drawing / Movement / Sweep (delegates to MovementSystem) ───────

    def draw_circle(self, center=Point3(0, 0, 0), radius=5, segments=32, color=(1, 0, 0, 1)):
        return self.movement.draw_circle(center, radius, segments, color)

    def draw_arc(self, center=Point3(0, 0, 0), radius=5, remainingmove=5,
                 start_angle=0, end_angle=90, segments=32, color=(1, 0, 0, 1)):
        return self.movement.draw_arc(center, radius, remainingmove, start_angle, end_angle, segments, color)

    def check_bullet_collision(self, node_a, node_b):
        return self.movement.check_bullet_collision(node_a, node_b)

    def shootingArc(self, origo, num_points=40, rotationangle=30, radius=0.15, full_circle=False):
        return self.movement.shootingArc(origo, num_points, rotationangle, radius, full_circle)

    def pointArc(self, origo, num_points=40, mouse_pos=None, rotationangle=-21,
                 width=0.5, height=0.5, movedistance=8):
        return self.movement.pointArc(origo, num_points, mouse_pos, rotationangle, width, height, movedistance)

    def mirrorPointArc(self, points, mirror_vec, origin):
        return self.movement.mirrorPointArc(points, mirror_vec, origin)

    def rotatePoint(self, point, angle_degrees, origo=Vec2(0.25, 0.25)):
        return self.movement.rotatePoint(point, angle_degrees, origo)

    def meshPointArc(self, origo, num_points=40, mouse_pos=None, rotationangle=-21):
        return self.movement.meshPointArc(origo, num_points, mouse_pos, rotationangle)

    def pathTowardsMouse(self, unit, x=None, y=None):
        return self.movement.pathTowardsMouse(unit, x, y)

    def debug_ray(self, pFrom, pTo):
        return self.movement.debug_ray(pFrom, pTo)

    def drawRectangle(self, center=Point3(0, 0, 0), width=5, height=3, color=(1, 0, 0, 1)):
        return self.movement.drawRectangle(center, width, height, color)

    def moveUnit(self, unit):
        if self.awaitingChoice:
            return
        self.movement.moveUnit(unit)

    async def makeChoiceNew(self, choices, position):
        cyn = Choice(choices, position)
        cyn.ma = taskMgr.add(cyn.mouseActivate, "mouseActivateTask")
        self.awaitingChoice = True
        self.ignore('mouse1')
        if self.roundCounter.current_player in [1, 2] and self.AIplayer2.active:
            #cynchoice = chargeYesNo[0]
            await Task.pause(1.0)
            # AI auto-selects first choice — use first key for dicts, first element for lists
            first_choice = next(iter(cyn.choices))
            cyn.choice = first_choice
            cyn.choiceMade = True
            taskMgr.remove("mouseActivateTask")
            taskMgr.add(cyn.cleanup())
        else:
            await cyn.ma
        #self.accept('mouse1', self.setActiveUnit,[self.taskLoopPathTowardsMouse, "taskLoopPathTowardsMouse"])
        self.awaitingChoice = False
        self.accept('mouse1', self.setActiveUnit,[self.setActiveUnitTask, self.setActiveUnitTaskName])
        cynchoice = cyn.choice
        
        del cyn
        return cynchoice
    
    """ async def makeChoice(self, choice):
        choice.ma = taskMgr.add(choice.mouseActivate, "mouseActivateTask")
        self.ignore('mouse1')
        print("Waiting for choice...")
        await choice.ma
        self.accept('mouse1', self.setActiveUnit,[self.setActiveUnitTask, self.setActiveUnitTaskName])
        print("event received")
        selected_choice = choice.choice
        print('Event delivered with args:', choice.choice)
        return
 """
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

    # ─── Flee, Pursuit & Rally (delegates to MovementSystem) ────────────

    def checkFleeCaught(self, fleeUnit, pursuerUnit, task):
        return self.movement.checkFleeCaught(fleeUnit, pursuerUnit, task)

    def fleeDirectionMultUnits(self, loser, winners):
        return self.movement.fleeDirectionMultUnits(loser, winners)

    def centerOfModels(self, unit):
        return self.movement.centerOfModels(unit)

    def getCenterOfUnit(self, unit):
        return self.movement.getCenterOfUnit(unit)

    def removeModelsFromUnit(self, unit, models_to_remove):
        self.movement.removeModelsFromUnit(unit, models_to_remove)

    def applyWounds(self, unit, wounds):
        self.movement.applyWounds(unit, wounds)

    def sweepTest(self, unit, direction, length):
        return self.movement.sweepTest(unit, direction, length)

    def sweepTestRot(self, unit, point, angle):
        return self.movement.sweepTestRot(unit, point, angle)

    def sweepTestDir(self, unit, tsFrom, direction, length):
        return self.movement.sweepTestDir(unit, tsFrom, direction, length)

    def fallBackContactTest(self, unitNP, moveVec=Vec3(0, 0, 0)):
        return self.movement.fallBackContactTest(unitNP, moveVec)

    def fallBack(self, loser, direction, length=10.0, rally=False, GG=False, flee=False):
        self.movement.fallBack(loser, direction, length, rally, GG, flee)

    def fallBack2(self, loser, direction, length=10.0, rally=False, GG=False, flee=False):
        return self.movement.fallBack2(loser, direction, length, rally, GG, flee)

    def pursuitMove(self, winner, loser):
        self.movement.pursuitMove(winner, loser)

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
            # Populate the list builder with the current player 1 army file
            if hasattr(self, 'p1army') and self.p1army:
                self.list_builder.load_from_file(self.p1army)
                self.list_builder.show_main_menu()
            self.list_builder_active = True
        else:
            self.list_builder.hide()
            self.list_builder_active = False
    
    def load_player1_army(self, filename="my_army.json"):
        """Load player 1's army from a file"""
        print(f"Loading Player 1 army from {filename}...")
        units = self.load_army_from_json(filename, player_num=1, start_pos=Point3(-48, -38, 0), spacing=12)
        if units:
            print(f"Player 1 army loaded: {len(units)} units")
            self.nominate_general(units, 1)
        return units
    
    def load_player2_army(self, filename="my_army.json"):
        """Load player 2's army from a file"""
        print(f"Loading Player 2 army from {filename}...")
        units = self.load_army_from_json(filename, player_num=2, start_pos=Point3(-48, 38, 0), spacing=12)
        if units:
            print(f"Player 2 army loaded: {len(units)} units")
            self.nominate_general(units, 2)
            # Set heading to face player 1
            for unit in units:
                unit.bodyNP.setH(180)
        return units

    def nominate_general(self, units, player_num):
        """Pick the army General (highest-Leadership character) and the Battle
        Standard Bearer before deployment. Neither is replaced once slain, so
        this runs once per army load."""
        bsb = select_battle_standard(units)
        general = select_general(units)
        if general is None:
            print(f"Player {player_num} has no character to lead — no General.")
        else:
            ld = general.unit.model.characteristics.get('Ld', '?')
            print(f"Player {player_num} General: {general.unit.name} (Ld {ld}), "
                  f"Command range {command_range(general):.0f}\"")
        if bsb is not None:
            print(f"Player {player_num} Battle Standard: {bsb.unit.name}, "
                  f"Command range {command_range(bsb):.0f}\"")
        return general

    def set_player_army(self, army_list, player_num, budget=2000):
        """Replace a player's on-table army with a new list (from the list builder)."""
        path = f"strategy_armies/player{player_num}_army.json"
        with open(path, 'w') as f:
            json.dump({'budget': budget, 'units': army_list}, f, indent=4)

        # Tear down existing units for this player. Mutate the list in place so
        # references held elsewhere (e.g. the AI) stay valid.
        player_units = self.player1Units if player_num == 1 else self.player2Units
        for u in list(player_units):
            try:
                self.world.removeRigidBody(u.bodyNP.node())
            except Exception:
                pass
            try:
                u.model.removeNode()
                u.bodyNP.removeNode()
            except Exception:
                pass
            if u in self.units:
                self.units.remove(u)
        player_units.clear()

        # Load the new army (load_army_from_json appends in place).
        if player_num == 1:
            self.p1army = path
            self.load_player1_army(path)
        else:
            self.p2army = path
            self.load_player2_army(path)

        # Keep the active unit and its binding valid.
        if self.player1Units:
            self.unitToMove = self.player1Units[0]
            self.accept('mouse3', self.moveUnit, [self.unitToMove])
        return len(player_units)

    # ─── Tutorial ─────────────────────────────────────────────────────

    def start_tutorial(self, filepath='tutorials/tutorial_basics.json'):
        """Launch the tutorial scenario system."""
        self.tutorial.start(filepath)

app = MyApp()
app.run()