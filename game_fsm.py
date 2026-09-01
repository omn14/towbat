"""
Game Phase Finite State Machine.

Manages transitions between game phases: Deploy, Strategy, Movement,
Shooting, Combat, Spell, Campaign, and MakeChoice.
"""

from direct.fsm.FSM import FSM
from panda3d.core import Point3, Vec3, BitMask32, TransformState
from panda3d.bullet import BulletBoxShape, BulletRigidBodyNode
from direct.interval.LerpInterval import LerpHprInterval
from direct.interval.IntervalGlobal import Sequence
from direct.interval.FunctionInterval import Func


class GamePhaseFSM(FSM):
    """Controls the game's turn phase flow via a finite state machine."""

    PHASES = ['StrategyPhase', 'MovementPhase', 'ShootingPhase', 'CombatPhase']

    def __init__(self, game):
        FSM.__init__(self, 'GameFSM')
        self.game = game

        self.end_of_turn_spells = []
        self._cube_cooldown = False

        self.end_phase_cube = self._create_menu_collision_cube(
            "endPhase", Point3(5, 20, 3)
        )

        self.current_phase_index = 0

        self.request(self.PHASES[self.current_phase_index])

        self.menu_cubes = base.camera.findAllMatches("**/*MenuCube")

        # The cube rides the camera, but Bullet only re-reads a body's transform
        # when that body is flagged dirty, and moving an ancestor doesn't flag it.
        taskMgr.add(self._sync_menu_cube, "syncMenuCube")

        self.accept('mouse1', self._on_menu_click)

    # ─── Convenience properties for backward compatibility ──────────

    def request(self, request, *args):
        """Announce every transition, however the phase was entered.

        Phases are requested directly from half a dozen places as well as
        through nextPhase(), so hooking the transition is the only way the
        HUD sees all of them.
        """
        result = FSM.request(self, request, *args)
        messenger.send('hud-phase', [self.state or request])
        return result

    @property
    def phases(self):
        return self.PHASES

    @property
    def currentPhaseIndex(self):
        return self.current_phase_index

    @currentPhaseIndex.setter
    def currentPhaseIndex(self, value):
        self.current_phase_index = value

    @property
    def endOfTurnSpells(self):
        return self.end_of_turn_spells

    @endOfTurnSpells.setter
    def endOfTurnSpells(self, value):
        self.end_of_turn_spells = value

    # ─── Menu Collision ─────────────────────────────────────────────

    def _create_menu_collision_cube(self, name='StrategyPhase', pos=Point3(5, 20, 0)):
        """Create a Bullet collision cube attached to the camera for UI interaction."""
        half_extents = Vec3(1, 1, 1)
        shape = BulletBoxShape(half_extents)
        cube_node = BulletRigidBodyNode(name)
        cube_node.setMass(0)
        cube_node.addShape(shape)

        cube_np = base.camera.attachNewNode(cube_node)
        cube_np.setPos(pos)
        cube_np.setCollideMask(BitMask32.bit(2))
        cube_np.setName(name)

        base.world.attachRigidBody(cube_node)

        try:
            model = loader.loadModel('models/box')
            model.reparentTo(cube_np)
            model.setScale(2)
            model.setPos(-1, -1, -1)
        except Exception:
            pass
        return cube_np

    def _sync_menu_cube(self, task):
        self.end_phase_cube.node().setTransformDirty()
        return task.cont

    def _on_menu_click(self):
        """Handle click on the phase-advance menu cube."""
        if not base.mouseWatcherNode.hasMouse():
            return
        # A click belongs to an open choice menu, not to the phase cube.
        if getattr(self.game, 'awaitingChoice', False):
            return

        pMouse = base.mouseWatcherNode.getMouse()
        pFrom = Point3()
        pTo = Point3()
        base.camLens.extrude(pMouse, pFrom, pTo)

        pFrom = render.getRelativePoint(base.cam, pFrom)
        pTo = render.getRelativePoint(base.cam, pTo)

        result = base.world.rayTestClosest(pFrom, pTo, BitMask32.bit(2))

        if result.hasHit():
            if self._cube_cooldown:
                return
            self._cube_cooldown = True

            # Spin the cube for visual feedback
            cube = self.end_phase_cube
            start_hpr = cube.getHpr()
            spin = LerpHprInterval(
                cube,
                duration=0.4,
                hpr=start_hpr + Vec3(0, 0, 360),
                startHpr=start_hpr,
                blendType='easeInOut',
            )
            seq = Sequence(
                spin,
                Func(self._clear_cube_cooldown),
            )
            seq.start()

            self.nextPhase()

    def _clear_cube_cooldown(self):
        """Re-enable phase cube interaction after the spin animation."""
        self._cube_cooldown = False

    def nextPhase(self):
        """Advance to the next phase in the cycle."""
        """ units = self.game.player2Units if self.game.roundCounter.current_player == 2 else self.game.player1Units
        for unit in units:
            self.game.fallBackContactTest(unit.bodyNP)
        units = self.game.player1Units if self.game.roundCounter.current_player == 1 else self.game.player2Units
        for unit in units:
            self.game.fallBackContactTest(unit.bodyNP) """
        self.current_phase_index = (
            (self.current_phase_index + 1) % len(self.PHASES)
        )
        self.request(self.PHASES[self.current_phase_index])
        # Notify tutorial system of the phase change
        messenger.send('tutorial-phase-change', [self.PHASES[self.current_phase_index]])
    # ─── Phase Enter/Exit Handlers ──────────────────────────────────

    def enterDeployPhase(self):
        print("Entering Deploy Phase")
        messenger.send('tutorial-phase-change', ['DeployPhase'])
        self.game.boundary_ghost = BulletRigidBodyNode('deployZone')

        dep_width = 72
        dep_height = 12
        box_w = 20
        box_h = 50

        self.game.boundary_ghost.addShape(
            BulletBoxShape(Vec3(box_w, 100, 10)),
            TransformState.makePos(Point3(dep_width / 2 + box_w, 0, 0))
        )
        self.game.boundary_ghost.addShape(
            BulletBoxShape(Vec3(box_w, 100, 10)),
            TransformState.makePos(Point3(-dep_width / 2 - box_w, 0, 0))
        )
        self.game.boundary_ghost.addShape(
            BulletBoxShape(Vec3(dep_width / 2, box_h, 10)),
            TransformState.makePos(Point3(0, dep_height / 2 + box_h, 0))
        )
        self.game.boundary_ghost.addShape(
            BulletBoxShape(Vec3(dep_width / 2, box_h, 10)),
            TransformState.makePos(Point3(0, -dep_height / 2 - box_h, 0))
        )

        self.game.boundary_np = render.attachNewNode(self.game.boundary_ghost)
        self.game.boundary_np.setCollideMask(BitMask32.bit(11))
        self.game.boundary_np.setPos(0, -dep_height - dep_height / 2, 0)
        base.world.attachRigidBody(self.game.boundary_ghost)

        self.game.setActiveUnitTask = self.game.taskLoopDeploy
        self.game.setActiveUnitTaskName = "taskLoopDeploy"
        self.game.accept(
            'mouse1', self.game.setActiveUnit,
            [self.game.setActiveUnitTask, self.game.setActiveUnitTaskName]
        )

    def exitDeployPhase(self):
        base.world.removeRigidBody(self.game.boundary_ghost)
        self.game.boundary_np.removeNode()
        # Ensure turn starts with player 1 after deployment
        self.game.roundCounter.request('PlayerOne')

    def enterStrategyPhase(self):
        self.current_phase_index = 0
        self.game.setActiveUnitTask = self.game.taskLoopStrategy
        self.game.setActiveUnitTaskName = "taskLoopStrategy"
        self.game.accept(
            'mouse1', self.game.setActiveUnit,
            [self.game.setActiveUnitTask, self.game.setActiveUnitTaskName]
        )
        print("Entering Strategy Phase")
        self.game.setGroundOverlay(False)
        # Start of Turn: a Magical Vortex drifts before anything else happens.
        for spell in list(getattr(self.game, 'remainsInPlay', [])):
            spell.scatter(self.game)
        for unit in self.game.units:
            unit.hasAttackedThisTurn = False
            unit.panicTestedThisPhase = False
            unit.fledThisPhase = False
            unit.startOfPhaseModels = unit.unit.nmodels
            if unit.state != "InCombat" and unit.state != "IsFleeing":
                unit.hasMovedThisTurn = False
                unit.attemptedRallyThisTurn = False
                unit.cannotChargeThisTurn = False
                unit.request("Idle")
            unit.updateTextNode()

    def exitStrategyPhase(self):
        self.game.ignore('mouse1')
        if taskMgr.hasTaskNamed("taskLoopStrategy"):
            taskMgr.remove("taskLoopStrategy")

    def enterMovementPhase(self):
        print("Entering Movement Phase")
        for unit in self.game.units:
            unit.panicTestedThisPhase = False
            unit.fledThisPhase = False
            unit.startOfPhaseModels = unit.unit.nmodels
        self.game.setActiveUnitTask = self.game.taskLoopPathTowardsMouse
        self.game.setActiveUnitTaskName = "taskLoopPathTowardsMouse"
        self.game.accept(
            'mouse1', self.game.setActiveUnit,
            [self.game.setActiveUnitTask, self.game.setActiveUnitTaskName]
        )

    def exitMovementPhase(self):
        taskMgr.remove("taskLoopPathTowardsMouse")
        self._cleanup_phase()
        self.game.ignore('mouse1')
        # The charge move is over — clear the Panic exemption.
        for unit in self.game.units:
            unit.isChargingMove = False
        self.game.boundries.contactTest(
            self.game.boundries.northBoundry, 180, Vec3(0, -0.1, 0)
        )
        self.game.boundries.contactTest(
            self.game.boundries.southBoundry, 0, Vec3(0, 0.1, 0)
        )
        self.game.boundries.contactTest(
            self.game.boundries.westBoundry, 270, Vec3(0.1, 0, 0)
        )
        self.game.boundries.contactTest(
            self.game.boundries.eastBoundry, 90, Vec3(-0.1, 0, 0)
        )
        for u in self.game.unitCopies:
            u.removeNode()
        self.game.unitCopies = []

    def enterShootingPhase(self):
        print("Entering Shooting Phase")
        for unit in self.game.units:
            unit.panicTestedThisPhase = False
            unit.fledThisPhase = False
            unit.startOfPhaseModels = unit.unit.nmodels
        self.game.setActiveUnitTask = self.game.taskShootingArcUpdate
        self.game.setActiveUnitTaskName = "taskShootingArcUpdate"
        self.game.accept(
            'mouse1', self.game.setActiveUnit,
            [self.game.setActiveUnitTask, self.game.setActiveUnitTaskName]
        )

    def exitShootingPhase(self):
        self.game.ignore('mouse1')
        self._cleanup_phase()
        self.game.setGroundOverlay(False)
        taskMgr.remove("taskShootingTrajectoryDrawLine")
        if getattr(self.game, 'cannon', None):
            self.game.cannon.cleanup()
        if getattr(self.game, 'bombard', None):
            self.game.bombard.cleanup()
        if getattr(self.game, 'rangeRing', None):
            self.game.rangeRing.removeNode()
            self.game.rangeRing = None

    def enterCombatPhase(self):
        print("Entering Combat Phase")
        self.game.setActiveUnitTask = self.game.taskStartCombat
        self.game.setActiveUnitTaskName = "taskStartCombat"
        self.game.accept(
            'mouse1', self.game.setActiveUnit,
            [self.game.setActiveUnitTask, self.game.setActiveUnitTaskName]
        )

        for unit in self.game.units:
            if unit.state == "InCombat":
                unit.hasAttackedThisTurn = False
            unit.panicTestedThisPhase = False
            unit.fledThisPhase = False
            # Combat-start size, for the nearby-friend-destroyed US>=5 gate.
            unit.startOfPhaseModels = unit.unit.nmodels
            # Pursuit into a New Combat asks whether the enemy was *already*
            # fighting when the phase began, which pursuits themselves change.
            unit.startOfPhaseEngaged = unit.isInCombat
            self.game.movement.updateDisrupted(unit)

    def exitCombatPhase(self):
        self.game.ignore('mouse1')
        self.game.roundCounter.next_turn()
        self.game.roundCounter.update_round_display()
        # The charge bonus lasts only the turn of the charge.
        for unit in self.game.units:
            # A pursuer that caught a unit which fell back counts as charging in
            # the turn that locked combat is fought, which is the next one.
            unit.chargedThisTurn = unit.countsAsChargedNextTurn
            unit.countsAsChargedNextTurn = False
            if not unit.chargedThisTurn:
                unit.chargeDistance = 0.0
            unit.cannotPursueThisTurn = False
            # A Wizard's casting allowance is per turn. Resetting it on entry
            # to the Strategy phase would refill it every time a spell sent the
            # game back there.
            unit.spellsCastThisTurn = []
            unit.cannotCastThisTurn = False
        for spell in list(self.end_of_turn_spells):
            spell.ticks_remaining -= 1
            if spell.ticks_remaining <= 0:
                spell.endSpell()
                self.end_of_turn_spells.remove(spell)
        for u in self.game.unitCopies:
            u.removeNode()
        self.game.unitCopies = []

    def enterMakeChoice(self):
        print("Entering Make Choice Phase")
        self.game.accept('mouse1', self.game.makeChoiceSelection)

    def exitMakeChoice(self):
        self.game.ignore('mouse1')

    def enterSpellPhase(self):
        print("Entering Spell Phase")
        self.activeSpell = None
        self.spellFunctionToCast = None
        # Casting is a detour from whichever phase asked for it; a spell's type
        # decides the phase it may be cast in, so remember where to go back to.
        self.phaseBeforeSpell = self.PHASES[self.current_phase_index]
        taskMgr.add(self.game.taskMagicArcUpdate, "taskMagicArcUpdate")
        self.game.setActiveUnitTask = self.game.taskMagicArcUpdate
        self.game.setActiveUnitTaskName = "taskMagicArcUpdate"
        self.game.accept(
            'mouse1', self.game.setActiveUnit,
            [self.game.setActiveUnitTask, self.game.setActiveUnitTaskName]
        )

    def exitSpellPhase(self):
        self.activeSpell = None
        self.spellFunctionToCast = None
        self.game.ignore('mouse1')
        self._cleanup_phase()
        self.game.setGroundOverlay(False)
        taskMgr.remove("taskMagicArcUpdate")
        taskMgr.remove("taskShootingTrajectoryDrawLine")
        if getattr(self.game, 'rangeRing', None):
            self.game.rangeRing.removeNode()
            self.game.rangeRing = None
        # The aim line is only drawn once a spell has been chosen.
        if getattr(self.game, 'trajectoryLine', None) is not None:
            self.game.trajectoryLine.removeNode()

    def enterCampaignPhase(self):
        """Show campaign map, hide battle scene."""
        print("Entering Campaign Phase")
        self.game.debugNP.hide()

        self._saved_cam_pos = self.game.camera.getPos()
        self._saved_cam_hpr = self.game.camera.getHpr()

        self.game.ground.hide()
        for u in self.game.units:
            u.bodyNP.hide()

        self.game.campaign_map.show()
        self.game.country_model.show()
        self.game.cloud_plane.show()

        self.game.camera.setPos(
            self.game.campaign_offset_x - 500, -1500, 1200
        )
        self.game.camera.lookAt(self.game.country_model)

        self.game.taskMgr.add(
            self.game.update_campaign_terrain, "update_campaign_terrain"
        )
        self.game.taskMgr.add(
            self.game.update_cloud_time, "update_cloud_time"
        )

        self.game.ignore('mouse1')
        self.ignore('mouse1')
        self.game.accept('mouse1', self.game.campaign_mouse_click)
        self.game.accept('mouse3', self.game.campaign_deselect)
        self.game.accept('m', self.game.enableMouse)

    def exitCampaignPhase(self):
        """Hide campaign map, restore battle scene."""
        self.game.debugNP.show()

        self.game.campaign_map.hide()
        self.game.country_model.hide()
        self.game.cloud_plane.hide()

        self.game.taskMgr.remove("update_campaign_terrain")
        self.game.taskMgr.remove("update_cloud_time")

        self.game.ground.show()
        for u in self.game.units:
            u.bodyNP.show()

        self.game.disableMouse()
        self.game.camera.setPos(self._saved_cam_pos)
        self.game.camera.setHpr(self._saved_cam_hpr)

        self.game.ignore('mouse1')
        self.game.ignore('mouse3')
        self.game.ignore('m')
        self.accept('mouse1', self._on_menu_click)
        self.game.accept(
            'mouse1', self.game.setActiveUnit,
            [self.game.setActiveUnitTask, self.game.setActiveUnitTaskName]
        )

    # ─── Helpers ────────────────────────────────────────────────────

    def _cleanup_phase(self):
        """Reset unit visuals after a phase ends."""
        for unit in self.game.units:
            unit.model.setColor(unit.color)
            unit.endedInUnit = False
            unit.updateTextNode()
