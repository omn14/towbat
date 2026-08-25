"""Offscreen harness for fleeing around impassable terrain (Rulebook p. 133).

A centre-line test said a 10 degree pivot cleared a house and the unit walked
straight into it, so this builds a real Bullet unit box and a real impassable
terrain body, runs the real `impassableAhead` / `fleeAroundImpassable`, and
prints what each candidate direction is actually measured to do -- including
whether the direction finally chosen leaves the unit clear along its whole
swept width.

    source .venv/bin/activate && python tests/harness_flee_terrain.py
"""

import math
import os
import sys

from panda3d.core import loadPrcFileData

loadPrcFileData("", "window-type offscreen")
loadPrcFileData("", "audio-library-name null")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from direct.showbase.ShowBase import ShowBase  # noqa: E402
from panda3d.bullet import (BulletBoxShape, BulletRigidBodyNode,  # noqa: E402
                            BulletWorld)
from panda3d.core import BitMask32, TransformState, Vec3  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from collision_masks import CollisionMask as CM  # noqa: E402
from combat_resolution import CombatResolver  # noqa: E402
from post_combat import detour_angles, turn_direction  # noqa: E402


def make_box(name, world, width, depth, pos, heading, mask):
    body = BulletRigidBodyNode(name)
    body.addShape(BulletBoxShape(Vec3(width / 2, depth / 2, 1.0)))
    body.setMass(0)
    np = base.render.attachNewNode(body)
    np.setPos(pos)
    np.setH(heading)
    np.setCollideMask(mask)
    world.attachRigidBody(body)
    return np


def scenario(label, unit_w, unit_d, unit_pos, house_pos, house_w, house_d,
             direction, distance):
    base_world = BulletWorld()
    unitNP = make_box('UnitCollision-Peasants', base_world, unit_w, unit_d,
                      unit_pos, 0, CM.SWEEP_TARGET)
    make_box('Terrain-House', base_world, house_w, house_d, house_pos, 0,
             CM.TERRAIN_IMPASSABLE)

    unit = SimpleNamespace(
        bodyNP=unitNP,
        unitName='Peasants',
        unit=SimpleNamespace(name='Peasant Soldier Unit',
                             model=SimpleNamespace(is_flying=lambda: False)),
    )

    resolver = CombatResolver.__new__(CombatResolver)
    movement = _Movement(base_world)
    resolver.game = SimpleNamespace(movement=movement, units=[unit])

    print(f"\n=== {label}")
    print(f"  unit {unit_w:.1f}x{unit_d:.1f} at "
          f"({unit_pos.x:.1f},{unit_pos.y:.1f}), house {house_w:.1f}x{house_d:.1f} "
          f"at ({house_pos.x:.1f},{house_pos.y:.1f}), fleeing {distance:.1f}\"")

    for turn in detour_angles(max_turn=90.0, step=15.0):
        x, y = turn_direction((direction.x, direction.y), turn)
        blocked = resolver.impassableAhead(unit, Vec3(x, y, 0), distance)
        print(f"    turn {turn:+6.1f} deg -> {'BLOCKED' if blocked else 'clear'}")

    chosen = resolver.fleeAroundImpassable(unit, direction, distance)
    still = resolver.impassableAhead(unit, chosen, distance)
    turned = math.degrees(math.atan2(chosen.y, chosen.x)
                          - math.atan2(direction.y, direction.x))
    print(f"  chosen: turned {turned:+.0f} deg -> "
          f"{'STILL BLOCKED' if still else 'clear'}")


class _Movement:
    """Just enough of MovementSystem for the sweep under test."""

    def __init__(self, world):
        self.world = world

    def sweepTestDir(self, unit, tsFrom, direction, length, mask=None,
                     pass_over=None):
        tsTo = TransformState.makePosHpr(tsFrom.getPos() + direction * length,
                                         tsFrom.getHpr())
        shape = unit.bodyNP.node().getShape(0)
        result = self.world.sweepTestClosest(shape, tsFrom, tsTo, mask)
        if result.hasHit():
            return result.getHitFraction(), result.getHitPos()
        return 1.0, None


if __name__ == '__main__':
    base = ShowBase()

    # The game that failed: a 12-model peasant block fleeing east into a house
    # three inches ahead. A 10 degree pivot was reported as clear.
    scenario("house dead ahead, 3\" flee",
             unit_w=5.0, unit_d=2.0, unit_pos=Vec3(-10, 2, 0),
             house_pos=Vec3(-5, 2, 0), house_w=6.0, house_d=6.0,
             direction=Vec3(1, 0.05, 0).normalized(), distance=3.0)

    # Clipping one corner: the way past is a small turn, and it should be the
    # small turn that gets taken.
    scenario("house off to one side",
             unit_w=5.0, unit_d=2.0, unit_pos=Vec3(-10, 2, 0),
             house_pos=Vec3(-5, 6, 0), house_w=6.0, house_d=6.0,
             direction=Vec3(1, 0.05, 0).normalized(), distance=3.0)

    # Nothing in the way at all: no detour should be taken.
    scenario("open ground",
             unit_w=5.0, unit_d=2.0, unit_pos=Vec3(-10, 2, 0),
             house_pos=Vec3(30, 30, 0), house_w=6.0, house_d=6.0,
             direction=Vec3(1, 0.05, 0).normalized(), distance=3.0)
