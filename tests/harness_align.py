"""Offscreen harness for the post-combat wheel-to-align (Rulebook p. 157).

Nothing about the align is visible from a unit test and describing it in words
has proved unreliable, so this builds two real Bullet-backed unit boxes, runs
the real `contactPointOn` and the same pivot maths `alignToEnemy` animates,
and prints the numbers: where the pivot landed, whether it stayed put through
the turn, and whether the two end up face to face and touching.

    source .venv/bin/activate && python tests/harness_align.py
"""

import math
import os
import sys

from panda3d.core import loadPrcFileData

loadPrcFileData("", "window-type offscreen")
loadPrcFileData("", "audio-library-name null")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from direct.showbase.ShowBase import ShowBase  # noqa: E402
from panda3d.bullet import BulletBoxShape, BulletRigidBodyNode, BulletWorld  # noqa: E402
from panda3d.core import Point3, Vec3  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from combat_resolution import CombatResolver  # noqa: E402


def make_unit(name, world, width, depth, pos, heading):
    """A stand-in for a unit: one box body, centred on its node, scale 1."""
    body = BulletRigidBodyNode('UnitCollision-' + name)
    body.addShape(BulletBoxShape(Vec3(width / 2, depth / 2, 0.5)))
    body.setMass(0)
    np = base.render.attachNewNode(body)
    np.setPos(pos)
    np.setH(heading)
    world.attachRigidBody(body)
    return np


def corners(np):
    half = np.node().getShape(0).getHalfExtentsWithMargin()
    return [base.render.getRelativePoint(np, Point3(sx * half.x, sy * half.y, 0))
            for sx, sy in ((-1, 1), (1, 1), (1, -1), (-1, -1))]


def gap_between(a, b):
    """Smallest distance from any corner of *a* to *b*'s box, negative if
    inside. Crude, but enough to tell contact from a gap."""
    half = b.node().getShape(0).getHalfExtentsWithMargin()
    best = None
    for c in corners(a):
        local = b.getRelativePoint(base.render, c)
        dx = abs(local.x) - half.x
        dy = abs(local.y) - half.y
        d = (math.hypot(max(dx, 0), max(dy, 0)) if dx > 0 or dy > 0
             else max(dx, dy))
        best = d if best is None else min(best, d)
    return best


def pivot_rotate(np, pivot, angle):
    """What `alignToEnemy` animates: turn about *pivot*, no translation."""
    parent = np.getParent()
    node = base.render.attachNewNode("pivot")
    node.setPos(pivot)
    node.setHpr(np.getHpr())
    np.wrtReparentTo(node)
    node.setH(node.getH() + angle)
    np.wrtReparentTo(parent)
    node.removeNode()


def report(label, attacker, defender, angle):
    resolver = CombatResolver.__new__(CombatResolver)
    wrapper = SimpleNamespace(bodyNP=attacker)
    pivot = resolver.contactPointOn(wrapper, defender)
    local = attacker.getRelativePoint(base.render, pivot)
    half = attacker.node().getShape(0).getHalfExtentsWithMargin()

    # A marker riding along with the attacker says where the pivot ended up.
    marker = attacker.attachNewNode("pivot-marker")
    marker.setPos(attacker.getRelativePoint(base.render, pivot))

    print(f"\n=== {label}")
    print(f"  half extents {half.x:.2f} x {half.y:.2f}")
    print(f"  pivot local {local.x:+.2f} {local.y:+.2f}  "
          f"(on edge: x={abs(abs(local.x) - half.x) < 1e-3}, "
          f"y={abs(abs(local.y) - half.y) < 1e-3})")
    print(f"  gap before  {gap_between(attacker, defender):+.3f}")

    pivot_rotate(attacker, pivot, angle)
    print(f"  turned {angle:+.1f} deg -> H {attacker.getH():.1f}, "
          f"defender H {defender.getH():.1f}")
    print(f"  pivot drifted {(marker.getPos(base.render) - pivot).length():.3f}")
    print(f"  gap after wheel {gap_between(attacker, defender):+.3f}")

    marker.removeNode()


def main():
    ShowBase()

    # A defender facing -Y, so an attacker coming up from below strikes its
    # front; the attacker has overrun in at an angle and stopped on contact.
    world = BulletWorld()
    defender = make_unit("D", world, 6.0, 2.0, Point3(0, 0, 0), 180)
    attacker = make_unit("A", world, 5.0, 2.0, Point3(-1.5, -3.0, 0), 25)
    report("25 deg off square", attacker, defender, -25)

    world2 = BulletWorld()
    d2 = make_unit("D2", world2, 6.0, 2.0, Point3(0, 0, 0), 180)
    a2 = make_unit("A2", world2, 5.0, 2.0, Point3(-5.0, -2.4, 0), 15)
    report("clipping the front-left corner", a2, d2, -15)

    world3 = BulletWorld()
    d3 = make_unit("D3", world3, 6.0, 2.0, Point3(0, 0, 0), 180)
    a3 = make_unit("A3", world3, 5.0, 2.0, Point3(1.0, -3.2, 0), -40)
    report("40 deg the other way", a3, d3, 40)


if __name__ == "__main__":
    main()
