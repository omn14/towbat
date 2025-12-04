from panda3d.core import Vec3
from panda3d.bullet import BulletRigidBodyNode, BulletBoxShape
from direct.showbase.ShowBase import ShowBase
import random
from panda3d.core import LRotationf


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
        print("All dice have settled.")
        for i, o in enumerate(hpr):
            q = LRotationf()
            q.setHpr(o)
            #print(f"Orientation HPR: {o} -> Quat: {q}")
            if q.getUp().normalized().dot(Vec3(0,0,1)) > 0.9:
                print(6,"Z is up")
                allDice[i].currentValue = 6
            if q.getUp().normalized().dot(Vec3(0,0,-1)) > 0.9:
                print(1,"Z is down")
                allDice[i].currentValue   = 1
            if q.getForward().normalized().dot(Vec3(0,0,1)) > 0.9:
                print(2,"Y is up")
                allDice[i].currentValue = 2
            if q.getForward().normalized().dot(Vec3(0,0,-1)) > 0.9:
                print(5,"Y is down")
                allDice[i].currentValue = 5
            if q.getRight().normalized().dot(Vec3(0,0,1)) > 0.9:
                print(3,"X is up")
                allDice[i].currentValue = 3
            if q.getRight().normalized().dot(Vec3(0,0,-1)) > 0.9:
                print(4,"X is down")
                allDice[i].currentValue = 4
            
            #print(q.getUp().normalized(), q.getForward().normalized(), q.getRight().normalized())
        #for dice in allDice:
        #    dice.remove(base.world)
        return task.done
    return task.cont

class Dice:
    def __init__(self, world, position=(0, 0, 5), size=1.0):
        """
        Initialize a 6-sided dice cube with Bullet physics.
        
        Args:
            world: BulletWorld instance
            position: Initial position as tuple (x, y, z)
            size: Size of the dice cube
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
        #self.model.setPos(-Vec3(size / 2, size / 2, size / 2))
        #self.zup = loader.loadModel('models/zup-axis')
        #self.zup.reparentTo(self.model)
    
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