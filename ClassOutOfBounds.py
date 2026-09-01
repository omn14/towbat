



from panda3d.bullet import BulletBoxShape, BulletGhostNode
from panda3d.core import Vec3, BitMask32

from collision_masks import CollisionMask as CM
from models import BOARD_DEPTH, BOARD_WIDTH




class OutOfBounds:

    def __init__(self,Game):
        print("OutOfBounds initialized")
        self.game = Game
        self.mask = CM.OUT_OF_BOUNDS
        # Walls sit just outside the board, so their inner faces are the edge.
        t = 5
        self.northBoundry=self.boundry((0,BOARD_DEPTH/2+t,0),Vec3(BOARD_WIDTH/2, t, t))
        self.southBoundry=self.boundry((0,-BOARD_DEPTH/2-t,0),Vec3(BOARD_WIDTH/2, t, t))
        # The side walls overrun north and south so the corners are covered.
        self.westBoundry=self.boundry((-BOARD_WIDTH/2-t,0,0),Vec3(t, BOARD_DEPTH/2+10, t))
        self.eastBoundry=self.boundry((BOARD_WIDTH/2+t,0,0),Vec3(t, BOARD_DEPTH/2+10, t))
        
        #self.northBoundry=self.boundry((0,48/2-12,11))

        #taskMgr.add(self.checkGhost, 'checkGhost',extraArgs=[self.northBoundry], appendTask=True)

    def boundry(self, position, shp):
        
        shape = BulletBoxShape(shp)

        ghost = BulletGhostNode('Ghost')
        ghost.addShape(shape)
        ghostNP = render.attachNewNode(ghost)
        ghostNP.setPos(*position)
        
        ghostNP.setCollideMask(self.mask)  # Set collide mask to match rigid bodies

        base.world.attachGhost(ghost)
        return ghostNP

    def checkGhost(self, boundry, task):
        ghost = boundry.node()
        print(ghost.getNumOverlappingNodes())
        for node in ghost.getOverlappingNodes():
            print(node)
            

        return task.cont
    
    def contactTest(self, boundry,H,moveVec=Vec3(0,0,0)):
        boundry.setCollideMask(BitMask32.bit(1))
        ghost = boundry.node()
        
        result = base.world.contactTest(ghost)
        boundry.setCollideMask(self.mask)
        for contact in result.getContacts():
            node_name = contact.getNode1().getName()
            if node_name.startswith('UnitCollision-'):
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
                selected_unit = self.game.getSelectedUnit(contact.getNode1())
                if selected_unit.state == 'InCombat':
                    print("Unit in combat, cannot be moved in bounds again now!")
                    return
                if selected_unit.state == 'IsFleeing':
                    print("Unit is fleeing out of the battle field, it is destroyed!")
                    base.world.removeRigidBody(selected_unit.bodyNP.node())
                    self.game.units.remove(selected_unit)
                    if selected_unit in self.game.player1Units:
                        self.game.player1Units.remove(selected_unit)
                    if selected_unit in self.game.player2Units:
                        self.game.player2Units.remove(selected_unit)
                    selected_unit.bodyNP.removeNode()
                    selected_unit.model.removeNode()
                    return
                np=selected_unit.bodyNP
                np.setHpr(Vec3(H,0,0))
                cpos=Vec3(np.getPos())
                np.setPos(cpos+moveVec)
                return self.contactTest(boundry,H,moveVec)
        
    

