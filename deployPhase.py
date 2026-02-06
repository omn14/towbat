from panda3d.core import Point3, BitMask32


def taskMoveUnit(game,unit,task):
    #game.ignore('mouse1')
    pxy = getMouseXY()
    if pxy is None:
        return task.cont
    x, y = pxy
    #print("Mouse position during move phase:", x, y)
    unit.bodyNP.setPos(x,y,0)
    c = game.checkUnitContactSmall(unit)
    if c:
        unit.model.setColor(.6,0.6,0.6,1)
    else:
        unit.model.setColor(unit.color)
    return task.cont

def endMoveUnit(game,taskToEnd):
    print("Ending move unit task")
    c = game.checkUnitContactSmall(game.unitToMove)
    if c:
        print("Unit is in contact with another unit, cannot reform here.")
        return
    taskMgr.remove(taskToEnd)
    game.accept('mouse1', game.setActiveUnit,[game.setActiveUnitTask, game.setActiveUnitTaskName])
    if game.roundCounter.current_player == 2:
        game.roundCounter.request('PlayerOne')
    else:
        game.roundCounter.request('PlayerTwo')
    
    

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