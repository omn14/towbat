"""Exercise formed-to-Skirmisher charge scheduling with Panda's real task manager."""

from types import SimpleNamespace
from unittest.mock import patch

from direct.task.Task import TaskManager
from panda3d.core import AsyncTaskManager, NodePath, Vec3

from movement_system import MovementSystem


def test_formed_charge_task_is_named_and_forwards_original_state():
    manager = TaskManager()
    manager.mgr = AsyncTaskManager('charge-scheduling-test')
    calls = []

    async def charge(attacker, contact, position, heading, task, defender=None):
        calls.append((attacker, contact, position, heading, task.getName(), defender))
        return task.done

    defender = object()
    attacker = SimpleNamespace(state='Idle', hasMovedThisTurn=False,
                               bodyNP=NodePath('Skycutter'), isChargingMove=False,
                               formedSkirmishPreview=SimpleNamespace(target=defender))
    attacker.bodyNP.setPos(3, -8, 0)
    attacker.bodyNP.setH(37)
    origin, heading = Vec3(attacker.bodyNP.getPos()), Vec3(attacker.bodyNP.getHpr())
    movement = MovementSystem.__new__(MovementSystem)
    movement.game = SimpleNamespace(combat=SimpleNamespace(chargeAndChargeReaction=charge),
                                    skirmMoveGhost=None)
    try:
        with patch('movement_system.taskMgr', manager, create=True), \
                patch('movement_system.in_vanguard', return_value=False), \
                patch('skirmish_ui.clear_plot_preview'):
            movement.moveUnit(attacker)
        scheduled = manager.getTasks()
        assert len(scheduled) == 1 and scheduled[0].hasName()
        assert attacker.isChargingMove and not calls
        attacker.bodyNP.setPos(30, 30, 0)
        attacker.bodyNP.setH(90)
        manager.step()
        assert calls == [(attacker, None, origin, heading, 'chargeAndChargeReaction', defender)]
        assert not manager.hasTaskNamed('chargeAndChargeReaction')
    finally:
        for task in manager.getTasks():
            manager.remove(task)
        attacker.bodyNP.removeNode()