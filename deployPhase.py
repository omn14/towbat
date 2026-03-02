from panda3d.core import Point3, BitMask32
import random

def taskMoveUnit(game,unit,task):
    #game.ignore('mouse1')
    if game.roundCounter.current_player == 2 and game.AIplayer2.active:
        #pxy = (random.uniform(-22, 22), random.uniform(7.5, 15))
        pxy = (random.uniform(-36, 36), random.uniform(12, 24))
    else:
        pxy = getMouseXY()

    if pxy is None:
        return task.cont
    x, y = pxy
    #print("Mouse position during move phase:", x, y)
    unit.bodyNP.setPos(x,y,0)
    
    # Notify Bullet that the transform has changed
    unit.bodyNP.node().setTransformDirty()
    outBounds = False
    c = game.checkUnitContactSmall(unit)
    if c:
        print("Unit is in contact with another unit, cannot deploy here.")
        outBounds = True
    unit.bodyNP.node().setTransformDirty()
    if not is_shape_inside(base.world, unit.bodyNP.node(), game.boundary_ghost):
        print("Unit is out of bounds, cannot deploy here.")
        outBounds = True
    if outBounds:
        unit.model.setColor(.6,0.6,0.6,1)
    else:
        unit.model.setColor(unit.color)

    
    if game.roundCounter.current_player == 2 and game.AIplayer2.active:
        if outBounds:
            return task.cont

        endMoveUnit(game,"dummy")
        return task.done
    return task.cont

def endMoveUnit(game,taskToEnd):
    print("Ending move unit task")
    outBounds = False
    c = game.checkUnitContactSmall(game.unitToMove)
    if c:
        print("Unit is in contact with another unit, cannot deploy here.")
        outBounds = True
    game.unitToMove.bodyNP.node().setTransformDirty()
    if not is_shape_inside(base.world, game.unitToMove.bodyNP.node(), game.boundary_ghost):
        print("Unit is out of bounds, cannot deploy here.")
        outBounds = True
    if outBounds:
        game.unitToMove.model.setColor(.6,0.6,0.6,1)
        return
    else:
        game.unitToMove.model.setColor(game.unitToMove.color)
    taskMgr.remove(taskToEnd)
    game.unitToMove.isDeployed=True
    game.accept('mouse1', game.setActiveUnit,[game.setActiveUnitTask, game.setActiveUnitTaskName])
    depH=12
    if game.roundCounter.current_player == 2:
        if not allUnitsDeployed(game.player1Units):
            game.roundCounter.request('PlayerOne')
            #game.boundary_np.setPos(0, -7.5-7.5/2, 0)
            game.boundary_np.setPos(0, -depH-depH/2, 0)

    else:
        if not allUnitsDeployed(game.player2Units):
            game.roundCounter.request('PlayerTwo')
            game.boundary_np.setPos(0, depH+depH/2, 0)
            if game.roundCounter.current_player == 2 and game.AIplayer2.active:
                game.AIplayer2.deployUnits()
            
    if allUnitsDeployed(game.units):
        print("All units deployed, moving to next phase.")
        game.fsm.request("StrategyPhase")
    
    

def getMouseXY():
    if base.mouseWatcherNode.hasMouse():
        pMouse = base.mouseWatcherNode.getMouse()
        pFrom = Point3()
        pTo = Point3()
        base.camLens.extrude(pMouse, pFrom, pTo)
        # Transform to global coordinates
        pFrom = render.getRelativePoint(base.cam, pFrom)
        pTo = render.getRelativePoint(base.cam, pTo)
        result = base.world.rayTestClosest(pFrom, pTo, BitMask32.bit(1))
        if result.hasHit():
            hitPos = result.getHitPos()
            xx = hitPos.getX()
            yy = hitPos.getY()
            return (xx, yy)
        else:
            return None
    else:
        return None
    
def allUnitsDeployed(units):
    for unit in units:
        if not unit.isDeployed:
            return False
    return True

def is_fully_contained(test_node, boundary_ghost):
    """Check if test_node is fully within boundary_ghost"""
    overlapping = boundary_ghost.getOverlappingNodes()
    
    if test_node not in overlapping:
        return False
    
    # For full containment, check bounding box
    test_bounds = test_node.getBounds()
    boundary_bounds = boundary_ghost.getBounds()
    
    return boundary_bounds.contains(test_bounds)

def is_shape_inside(world, inner_node, outer_ghost):
    """Check if inner_node is completely inside outer_ghost"""
    # Get all contact points
    result = world.contactTest(inner_node)
    
    # If there are contacts with the outer boundary, it's not fully inside
    for contact in result.getContacts():
        
        if contact.getNode0() == outer_ghost or contact.getNode1() == outer_ghost:
            """ r = world.contactTestPair(contact.getNode0(), contact.getNode1())
            # Check penetration depth - negative means outside
            for c in r.getContacts():
                 print(f"Contact distance: {c.getManifoldPoint().getDistance()}")
                 #print(f"Contact points: {c.getManifoldPoint().getPositionWorldOnA()}, {c.getManifoldPoint().getPositionWorldOnB()}")
                 #print(f"Contact normal: {c.getManifoldPoint().getNormalWorldOnB()}")
                 if c.getManifoldPoint().getDistance() > 0:
                    return False """
            return False
            
    return True