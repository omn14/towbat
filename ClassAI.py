

from direct.showbase.DirectObject import DirectObject

class ClassAI:
    def __init__(self,Game,playerUnits,enemyUnits):
        self.game=Game
        self.playerUnits=playerUnits
        self.enemyUnits=enemyUnits
        
        self.waitTask="waitTask"
        self._move_complete = False
        self.helper1 = DirectObject()
        self.helper1.accept('unit-move-complete', self.endLoopWaitForMoveComplete)


    async def takeMoveTurn(self):
        for unit in self.playerUnits:
            if unit.state == "Idle":
                print(f"AI controlling unit: {unit.unit.name}")
                # Simple AI logic: Move forward by 1 unit
                #await taskMgr.add(self.moveTowardsClosestEnemy, f"moveTowardsClosestEnemy-{unit.unit.name}", extraArgs=[unit], appendTask=False)
                self.moveTowardsClosestEnemy(unit)
                #await messenger.future('unit-move-complete')
                await taskMgr.add(self.loopWaitForMoveComplete, self.waitTask, extraArgs=[unit], appendTask=True)
                #await self.helper1.future('unit-move-complete')
        return
    
    async def takeCombatTurn(self):
        for unit in self.playerUnits:
            if unit.state == "InCombat":
                print(f"AI controlling combat for unit: {unit.unit.name}")
                # Simple AI logic: Attack the first enemy in combat
                if unit.hasAttackedThisTurn == False:
                    self.game.unitToMove=unit
                    taskMgr.add(self.game.taskStartCombat,"taskStartCombat", extraArgs=[], appendTask=True)
                    self._move_complete = False
                    await taskMgr.add(self.loopWaitForMoveComplete, self.waitTask, extraArgs=[unit], appendTask=True)
                    #await self.helper1.future('unit-move-complete')
        return
    
    def loopWaitForMoveComplete(self,unit,task):
        print(f"Waiting for move complete for unit: {unit.unit.name}")
        if self._move_complete:
            self._move_complete = False
            print(f"signal recieved for unit: {unit.unit.name}")
            return task.done
        return task.cont

    def endLoopWaitForMoveComplete(self):
        self._move_complete = True

    def moveTowardsClosestEnemy(self,unit):
        own_pos=unit.bodyNP.getPos()
        closest_enemy=None
        closest_dist=float('inf')
        for enemy in self.enemyUnits:
            enemy_pos=enemy.bodyNP.getPos()
            dist=(enemy_pos-own_pos).length()
            if dist<closest_dist:
                closest_dist=dist
                closest_enemy=enemy
        if closest_enemy:
            self.game.unitToMove=unit
            closest_enemy_pos=closest_enemy.bodyNP.getPos()
            self.game.pathTowardsMouse(unit,closest_enemy_pos.x,closest_enemy_pos.y)
            #taskMgr.doMethodLater(0.5, self.game.moveUnit, "moveUnit", extraArgs=[unit], appendTask=False)
            #await taskMgr.add(self.game.taskLoopPathTowardsMouse,"taskLoopPathTowardsMouse", extraArgs=[closest_enemy_pos.x,closest_enemy_pos.y], appendTask=True)
            #taskMgr.add(self.game.taskLoopPathTowardsMouse,"taskLoopPathTowardsMouse", extraArgs=[], appendTask=True)
            #await taskMgr.add(self.game.pathTowardsMouse,"pathTowardsMouse",extraArgs=[unit,closest_enemy_pos.x,closest_enemy_pos.y], appendTask=False)
            self.game.moveUnit(unit)
            


            print(f"{unit.unit.name} moved towards {closest_enemy.unit.name}")
        return

    