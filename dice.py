from panda3d.core import Vec3
from panda3d.bullet import BulletRigidBodyNode, BulletBoxShape
from direct.showbase.ShowBase import ShowBase
import random
from panda3d.core import LRotationf, LColor, Material

from rules_log import dice_roll


# Which number shows when a given body axis points at the sky.
_FACES = {('up', 1): 6, ('up', -1): 1,
          ('forward', 1): 2, ('forward', -1): 5,
          ('right', 1): 3, ('right', -1): 4}

# Below this the die is not lying flat -- cocked on another die or on the rim.
COCKED_DOT = 0.75


def face_up(q):
    """Return (number, how square-on it is) for a die with orientation *q*.

    The face is whichever body axis points most nearly at the sky. Each axis
    used to be tested against a fixed threshold instead, and a die that
    cleared none of them kept the value the *previous* roll had left in it,
    so a cocked die quietly reported an old number rather than a wrong one.
    """
    best, best_dot = None, 0.0
    for name, vec in (('up', q.getUp()), ('forward', q.getForward()),
                      ('right', q.getRight())):
        dot = vec.normalized().dot(Vec3(0, 0, 1))
        if abs(dot) > abs(best_dot):
            best, best_dot = name, dot
    if best is None:
        return None, 0.0
    return _FACES[(best, 1 if best_dot > 0 else -1)], abs(best_dot)


def checkDice(allDice,task):
    """Task to check the status of all dice in the scene."""
    hpr = [None] * len(allDice)
    for n,dice in enumerate(allDice):
        #if dice.node.isActive() and dice.node.getLinearVelocity().length() < 0.1 and dice.node.getAngularVelocity().length() < 0.1:
        if not dice.node.isActive():
            # Dice is stationary
            hpr[n] = dice.np.getHpr()

    #print("Dice orientations:", hpr)
    if all(h is not None for h in hpr):
        values = []
        for i, o in enumerate(hpr):
            q = LRotationf()
            q.setHpr(o)
            face, flatness = face_up(q)
            if flatness < COCKED_DOT:
                # Read it anyway -- the alternative is no number at all -- but
                # say so, because it is the one result worth distrusting.
                print(f"[Dice] die {i} is cocked ({flatness:.2f} square-on), "
                      f"reading nearest face {face}")
            allDice[i].currentValue = face
            values.append(face)
        print(f"All dice have settled: {values}")
        # The one point where a roll's faces are known, whoever threw it.
        dice_roll(values)
        return task.done
    return task.cont

class Dice:
    def __init__(self, world, position=(0, 0, 5), size=1.0,color=(1,1,1,1),
                 body_color=None):
        """
        Initialize a 6-sided dice cube with Bullet physics.
        
        Args:
            world: BulletWorld instance
            position: Initial position as tuple (x, y, z)
            size: Size of the dice cube
            color: colour *scale* applied over the model (a tint)
            body_color: repaints the die body outright, keeping the pips white
        """
        self.size = size
        self.currentValue = None  # To store the result after rolling
        
        # Create bullet rigid body node
        self.node = BulletRigidBodyNode('Dice')
        self.node.setMass(10.0)
        self.node.setFriction(2.5)
        self.node.setDeactivationTime(0.05)

        # Higher values = stops faster:
        #self.node.setLinearDamping(0.9)
        #self.node.setAngularDamping(0.9)

        self.node.setLinearSleepThreshold(0.1)   # Very sensitive
        self.node.setAngularSleepThreshold(0.1)
        
        # Add box shape for physics collision
        shape = BulletBoxShape(Vec3(size / 2, size / 2, size / 2))
        self.node.addShape(shape)
        
        # Attach to world
        self.np = render.attachNewNode(self.node)
        self.np.setPos(*position)
        world.attachRigidBody(self.node)
        
        # Load or create visual model
        #self.model = loader.loadModel('models/box')
        self.model = loader.loadModel('models/bdie.bam')
        self.model.setScale(size)
        self.model.reparentTo(self.np)
        self.model.setColorScale(*color)
        if body_color is not None:
            self.paint_body(body_color)
        #self.model.setPos(-Vec3(size / 2, size / 2, size / 2))
        #self.zup = loader.loadModel('models/zup-axis')
        #self.zup.reparentTo(self.model)

    def paint_body(self, colour):
        """Recolour the die body. The model carries one material per part, so
        the near-grey pip material is left alone and stays white."""
        for mat in self.model.findAllMaterials():
            base = mat.getBaseColor()
            channels = (base.getX(), base.getY(), base.getZ())
            if max(channels) - min(channels) < 0.15:   # neutral: the pips
                continue
            painted = Material(mat)
            painted.setBaseColor(LColor(*colour))
            self.model.replaceMaterial(mat, painted)
    
    def roll(self, force=10):
        """Apply random force and torque to simulate rolling."""
        force_vec = Vec3(
            random.uniform(-force, force),
            #random.uniform(-force, force),
            random.uniform(5, force)*4,
            random.uniform(5, force)*-2
        )
        torque_vec = Vec3(
            random.uniform(-10, 10),
            random.uniform(-10, 10),
            random.uniform(-10, 10)
        )
        self.node.applyCentralImpulse(force_vec)
        self.node.applyTorqueImpulse(torque_vec)
    
    def get_position(self):
        """Return current position of the dice."""
        return self.np.getPos()
    
    def remove(self, world):
        """Remove dice from world and scene."""
        world.removeRigidBody(self.node)
        self.np.removeNode()


# Old World Artillery Dice faces: 2, 4, 6, 8, 10 and Misfire.
ARTILLERY_FACES = {1: 'Misfire', 2: 2, 3: 4, 4: 6, 5: 8, 6: 10}


class ArtilleryDice(Dice):
    """A physics die whose settled d6 face maps to an Artillery Dice value."""

    def artillery_value(self):
        """Return the artillery result ('Misfire' or an int) after settling."""
        return ARTILLERY_FACES.get(self.currentValue, 'Misfire')


# Scatter Dice: four arrow faces and two 'Hit!' faces.
SCATTER_FACES = {1: 'Hit!', 2: 'Hit!', 3: 'Arrow', 4: 'Arrow', 5: 'Arrow', 6: 'Arrow'}


class ScatterDice(Dice):
    """A physics die whose settled d6 face maps to a Scatter Dice result."""

    def scatter_value(self):
        """Return 'Hit!' or 'Arrow' after settling."""
        return SCATTER_FACES.get(self.currentValue, 'Arrow')