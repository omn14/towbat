"""Movement system — pathfinding, arc/sweep calculations, fallback & pursuit.

Extracted from game.py.  The ``MovementSystem`` class wraps the game
instance and exposes every method that was previously defined inline on
``MyApp``.  Panda3D globals (``render``, ``taskMgr``, ``base``, etc.)
are used directly; all other game state is accessed via ``self.game``.
"""

import math
from collision_masks import CollisionMask as CM
from characters import on_host_removed
from special_rules import max_charge_range, unit_has_swiftstride
from toHitAndToWound import stat_value

from panda3d.core import (
    Vec2, Vec3, Vec4, Point3,
    LRotationf, LQuaterniond, LVector3d,
    TransformState, BitMask32,
    LineSegs, MeshDrawer, NodePath,
)
from panda3d.bullet import (
    BulletRigidBodyNode, BulletBoxShape,
    BulletTriangleMesh, BulletTriangleMeshShape,
)
from direct.interval.IntervalGlobal import (
    LerpPosHprInterval, LerpPosInterval, Sequence, Func, Wait,
)


class MovementSystem:
    """Encapsulates movement, pathfinding, sweep tests, and fall-back logic."""

    def __init__(self, game):
        self.game = game

    # ─── Drawing Helpers (circles, arcs, rectangles) ─────────────────────

    def draw_circle(self, center=Point3(0, 0, 0), radius=5, segments=32, color=(1, 0, 0, 1)):

        # Create MeshDrawer and NodePath if not already present
        if not hasattr(self.game, 'mesh_drawer'):
            self.game.mesh_drawer = MeshDrawer()
            self.game.mesh_drawer.setBudget(1000)
            #self.game.mesh_drawer_np = NodePath(self.game.mesh_drawer.create())
            self.game.mesh_drawer_node = self.game.mesh_drawer.getRoot()
            self.game.mesh_drawer_node.reparentTo(render)

        self.game.mesh_drawer.begin(base.cam, render)
        angle_step = 2 * 3.14159265 / segments
        for i in range(segments + 1):
            angle = i * angle_step
            x = center.x + radius * math.cos(angle)
            y = center.y + radius * math.sin(angle)
            z = center.z
            # Draw line segments between consecutive points (crossSegment)
            if i > 0:
                prev_angle = (i - 1) * angle_step
                prev_x = center.x + radius * math.cos(prev_angle)
                prev_y = center.y + radius * math.sin(prev_angle)
                prev_z = center.z
                self.game.mesh_drawer.segment(Point3(x, y, z),
                    Point3(prev_x, prev_y, prev_z), Vec4(0,0,1,1), .1, color
                )
        self.game.mesh_drawer.end()

    def draw_arc(self, center=Point3(0, 0, 0), radius=5, remainingmove=5,  start_angle=0, end_angle=90, segments=32, color=(1, 0, 0, 1)):
        # Create MeshDrawer and NodePath if not already present
        if not hasattr(self.game, 'mesh_drawer'):
            self.game.mesh_drawer = MeshDrawer()
            self.game.mesh_drawer.setBudget(1000)
            
            self.game.mesh_drawer_node = self.game.mesh_drawer.getRoot()
            self.game.mesh_drawer_node.setTwoSided(True)
            self.game.mesh_drawer_node.reparentTo(self.game.smiley)

        self.game.mesh_drawer.begin(base.cam, render)
        angle_step = (end_angle - start_angle) * (math.pi / 180) / segments
        center_color = Vec4(*color)
        for i in range(segments):
            angle1 = start_angle * (math.pi / 180) + i * angle_step
            angle2 = start_angle * (math.pi / 180) + (i + 1) * angle_step
            x1 = center.x + radius * math.cos(angle1)
            y1 = center.y + radius * math.sin(angle1)
            z1 = center.z
            x2 = center.x + radius * math.cos(angle2)
            y2 = center.y + radius * math.sin(angle2)
            z2 = center.z
            # Draw triangle from center to arc segment
            self.game.mesh_drawer.tri(
                center, center_color, (0, 0),
                Point3(x1, y1, z1), center_color, (1, 0),
                Point3(x2, y2, z2), center_color, (0, 1)
            )
        # Calculate the direction of the end of the arc
        angle_rad = end_angle * (math.pi / 180)
        dir_x = math.cos(angle_rad)
        dir_y = math.sin(angle_rad)

        # Rectangle parameters
        rect_length = radius  # Length along the arc direction
        rect_width = remainingmove   # Width perpendicular to the arc

        # Calculate rectangle corners
        half_length = rect_length / 2
        half_width = rect_width / 2

        rect_center_x = center.x + radius / 2 * dir_x - half_width * dir_x
        rect_center_y = center.y + radius / 2 * dir_y + half_width * dir_y
        rect_center_z = center.z

        

        # Direction vectors
        forward = Vec3(dir_x, dir_y, 0).normalized()
        right = Vec3(-dir_y, dir_x, 0).normalized()

        corner1 = Point3(rect_center_x + forward.x * half_length + right.x * half_width,
                         rect_center_y + forward.y * half_length + right.y * half_width,
                         rect_center_z)
        corner2 = Point3(rect_center_x + forward.x * half_length - right.x * half_width,
                         rect_center_y + forward.y * half_length - right.y * half_width,
                         rect_center_z)
        corner3 = Point3(rect_center_x - forward.x * half_length - right.x * half_width,
                         rect_center_y - forward.y * half_length - right.y * half_width,
                         rect_center_z)
        corner4 = Point3(rect_center_x - forward.x * half_length + right.x * half_width,
                         rect_center_y - forward.y * half_length + right.y * half_width,
                         rect_center_z)

        rect_color = Vec4(*color)

        # Draw the rectangle as two triangles (correct winding order)
        self.game.mesh_drawer.tri(corner1, rect_color, (0, 0), corner3, rect_color, (1, 1), corner2, rect_color, (1, 0))
        self.game.mesh_drawer.tri(corner3, rect_color, (1, 1), corner1, rect_color, (0, 0), corner4, rect_color, (0, 1))
        self.game.mesh_drawer.end()

        mesh = BulletTriangleMesh()

        for geomNP in self.game.mesh_drawer_node.findAllMatches('**/+GeomNode'):
            geomNode = geomNP.node()
            ts = geomNP.getTransform(self.game.mesh_drawer_node)
            #print(ts)
            for geom in geomNode.getGeoms():
                mesh.addGeom(geom, ts=ts)
                #print(geom)
        #lol
        body = BulletRigidBodyNode('movearea')
        shape = BulletTriangleMeshShape(mesh, False)
        body.addShape(shape)
        # Detach any existing BulletRigidBodyNode children from mesh_drawer_node
        for child in self.game.mesh_drawer_node.getChildren():
            if child.node().isOfType(BulletRigidBodyNode.getClassType()):
                self.game.world.removeRigidBody(child.node())
                child.detachNode()
        bodyNP = self.game.mesh_drawer_node.attachNewNode(body)
        bodyNP.node().setMass(0)
        bodyNP.setCollideMask(BitMask32.allOn())
        self.game.world.attachRigidBody(bodyNP.node())
        #self.game.world.doPhysics(0.01)
         
        # Calculate points along the arc's centerline
        motionpath_points = []
        for i in range(segments + 1):
            angle = start_angle * (math.pi / 180) + i * angle_step
            x = center.x + (radius - rect_width / 2) * math.cos(angle)
            y = center.y + (radius - rect_width / 2) * math.sin(angle)
            z = center.z
            motionpath_points.append({
                "pos": Point3(x, y, z),
                "hpr": (math.degrees(angle), 0, 0)
            })
        
        # Add points along the rectangle's centerline (straight out from arc end)
        # Centerline starts at the end of the arc and goes straight in the arc's end direction
        for i in range(1, 6):
            t = i / 5.0
            x = center.x + radius * math.cos(angle_rad) + t * rect_width * dir_x
            y = center.y + radius * math.sin(angle_rad) + t * rect_width * dir_y
            z = center.z
            motionpath_points.append({
            "pos": Point3(-x, y, z),
            "hpr": (math.degrees(angle_rad), 0, 0)
            })

    
        
        


    def check_bullet_collision(self, node_a, node_b):
        """
        Checks for Bullet collision between two Panda3D NodePaths with BulletRigidBodyNode attached.
        Returns True if they are colliding, False otherwise.
        """
        # Get Bullet nodes
        body_a = node_a.find("**/+BulletRigidBodyNode")
        body_b = node_b.find("**/+BulletRigidBodyNode")
        if body_a.isEmpty() or body_b.isEmpty():
            return False

        # Get Bullet shapes
        bullet_node_a = body_a.node()
        bullet_node_b = body_b.node()

        # Use BulletWorld contactTestPair
        result = self.game.world.contactTestPair(bullet_node_a, bullet_node_b)
        return result.getNumContacts() > 0

    
    def shootingArc(self, origo, num_points=40, rotationangle=30,radius=0.15, full_circle=False):
        points =[]
        origo=(origo/50 +1)*0.5
        origo   = Vec2(origo.x,origo.y)
        # Skirmishers have a 360° arc; everyone else a 90° front cone.
        arcmax = 2 * math.pi if full_circle else math.pi/2
        if not full_circle:
            points.append(origo)   # pie-wedge apex at the shooter

        for i in range(0,num_points):
            angle = arcmax * i / (num_points - 1)
            x = radius * math.cos(angle) 
            y = radius * math.sin(angle)
            points.append(origo+Vec2(x,y))
            points[-1] = self.rotatePoint(points[-1], rotationangle, origo=origo)

        for n in range(len(points),num_points+3):
            points.append(points[-1])
        return points

    def pointArc(self,origo, num_points=40, mouse_pos=None,rotationangle=-21,width=0.5,height=0.5,movedistance=8,sidemove=None):
        points =[]
        sidestep_point = None
        # A sideways move halves the Movement characteristic, so it has its own
        # allowance rather than being a fraction of the forward one.
        if sidemove is None:
            sidemove = movedistance / 4
        forward= Vec2(-math.sin(math.radians(rotationangle)), math.cos(math.radians(rotationangle)))*height/2.0
        origo = origo + forward
        #origo   = Vec2(0.55,0.55)
        #origo= origo+Vec2(math.cos(math.radians(rotationangle))*height/2,-math.sin(math.radians(rotationangle))*height/2)
        points.append(origo)

        arcmax = math.pi/2

        #rotationangle= 30
        midpoint=Vec2(width*math.cos(math.radians(rotationangle) ), width*math.sin(math.radians(rotationangle) ))*0.5+origo
        midpoint_unit_vector = midpoint - origo
        midpoint_mouse_vector = mouse_pos - midpoint if mouse_pos else Vec2(1,1)
        maxMoveDistance=movedistance
        raw_mouse_pos = mouse_pos   # kept unmirrored, so a sidestep knows its side
        #movedistance = min(movedistance, midpoint_mouse_vector.length())
        filipped=False
        if midpoint_unit_vector.normalized().dot(midpoint_mouse_vector.normalized()) > 0:
            #rotationangle += 180
            filipped=True
            vinkel=rotationangle+90
            mouse_pos = self.mirrorPointArc([mouse_pos], mirror_vec=Vec2(math.cos(math.radians(vinkel)), math.sin(math.radians(vinkel))), origin=midpoint)[0]
            #mouse_pos = self.rotatePoint(mouse_pos, 90, origo=midpoint)

        quarternion = LQuaterniond()
        quarternion.set_from_axis_angle(rotationangle, LVector3d(0,0,1))
        behind= quarternion.getForward().dot(LVector3d(midpoint_mouse_vector.normalized().x, midpoint_mouse_vector.normalized().y,0))
        nums=0
        if behind < 0 and abs(behind) > 0.8:
            if abs(behind) > 0.8:
                angle = 0
                x = width * math.cos(0) 
                y = width * math.sin(0)
                points.append(origo+Vec2(x,y))
                points[-1] = self.rotatePoint(points[-1], rotationangle, origo=origo)
                points.append(points[-1]+Vec2(-math.sin(math.radians(rotationangle)),math.cos(math.radians(rotationangle)))*movedistance/4)
                points.append(points[0]+Vec2(-math.sin(math.radians(rotationangle)),math.cos(math.radians(rotationangle)))*movedistance/4)
                for i, p in enumerate(points):
                    points[i] = p-Vec2(quarternion.getForward().x, quarternion.getForward().y)*movedistance/4
                points = points[2:] + points[:2]
                self.game.moveArceDistance = .9
                self.game.debugTextInfo.setText(f"Arc distance: {(self.game.moveArceDistance):.1f} ")

                #return points
        elif behind < 0 and abs(behind) < 0.8:
            angle = 0
            # Move Sideways: half the Movement characteristic, facing unchanged
            # (Rulebook p. 125).
            right = Vec2(quarternion.getRight().x, quarternion.getRight().y)
            frontMid = origo + right * (width / 2.0)   # origo is the front-left corner
            lateral = (raw_mouse_pos - frontMid).dot(right) if raw_mouse_pos else 0.0
            lateral = max(-sidemove, min(sidemove, lateral))
            reach = abs(lateral)
            x = width * math.cos(0) 
            y = width * math.sin(0)
            points.append(origo+Vec2(x,y))
            points[-1] = self.rotatePoint(points[-1], rotationangle, origo=origo)
            points.append(points[-1])
            points.append(points[-1]+Vec2(-math.sin(math.radians(rotationangle)),math.cos(math.radians(rotationangle)))*height)
            points.append(points[-1])
            points.append(points[0]+Vec2(-math.sin(math.radians(rotationangle)),math.cos(math.radians(rotationangle)))*height)
            for i in [2,3]:
                points[i] = points[i]+Vec2(quarternion.getRight().x, quarternion.getRight().y)*reach 
            for i in [1,4]:
                points[i] = points[i]+Vec2(quarternion.getRight().x, quarternion.getRight().y)*(reach -width)

            for i in range(len(points)):
                # The band is built forward of the front edge; pull it back a
                # whole depth so it covers the unit's own footprint instead.
                points[i] = points[i]-Vec2(quarternion.getForward().x, quarternion.getForward().y)*height
            points = [points[-1]] + points[:-1]

            vinkel=rotationangle+90
            filipped = not filipped
            # arcPoint is where the front-edge centre lands; the move is purely
            # along the unit's right vector, with no forward component.
            sidestep_point = frontMid + right * lateral
            self.game.moveArceDistance = reach * 100
            self.game.debugTextInfo.setText(
                f"Sidestep: {self.game.moveArceDistance:.1f} of {sidemove * 100:.1f} ")
        
        
            

        else:
            for i in range(0,num_points):
                angle = arcmax * i / num_points
                x = width * math.cos(angle) 
                y = width * math.sin(angle)
                points.append(origo+Vec2(x,y))

                points[-1] = self.rotatePoint(points[-1], rotationangle, origo=origo)

                vector=points[-1]-origo
                pointmid=origo+vector*.5
                vectormouse=mouse_pos - pointmid if mouse_pos else Vec2(1,1)
                #print(f"vectormouse: {vectormouse}, vector: {vector}, dot: {vectormouse.dot(vector)}")
                nums=i
                if abs(vectormouse.normalized().dot(vector)) < .001:
                    
                    break
                if width * angle >= movedistance:
                    break

            movedistance = min(movedistance, vectormouse.length()+angle*width)
            if movedistance - angle*width > 0:
                movedistance -= angle*width
            else:
                movedistance = 0

            self.game.moveArceDistance = (movedistance+angle*width)*2*50
            self.game.debugTextInfo.setText(f"Arc distance: {(self.game.moveArceDistance):.1f} ")
            #movedistance = min(movedistance, angle*width)
            #points.append(points[-1]+Vec2(-math.sin(angle+math.radians(rotationangle)),math.cos(angle+math.radians(rotationangle)))*0.2)
            points.append(points[-1]+Vec2(-math.sin(angle+math.radians(rotationangle)),math.cos(angle+math.radians(rotationangle)))*movedistance)
            #points[-1] = self.rotatePoint(points[-1], rotationangle)
            #points.append(points[0]+Vec2(-math.sin(angle+math.radians(rotationangle)),math.cos(angle+math.radians(rotationangle)))*0.2)
            points.append(points[0]+Vec2(-math.sin(angle+math.radians(rotationangle)),math.cos(angle+math.radians(rotationangle)))*movedistance)
            #points[-1] = self.rotatePoint(points[-1], rotationangle)
            #print(points)
        nums=len(points)
        midpointfront = (points[-1] + points[-2]) * 0.5
        midpointfront = (points[nums-1] + points[nums-2]) * 0.5
        self.game.numsPoints=nums
        for n in range(len(points),num_points+3):
            points.append(points[-1])
        
        #print(len(points))

        if filipped:
            points = self.mirrorPointArc(points, mirror_vec=Vec2(math.cos(math.radians(vinkel)), math.sin(math.radians(vinkel))), origin=midpoint)
            self.game.arcPoint=(points[nums-1] + points[nums-2]) * 0.5
            self.game.arcPointRotation=math.degrees(-angle)
        else:
            self.game.arcPoint=midpointfront
            self.game.arcPointRotation=math.degrees(angle)

        if sidestep_point is not None:
            self.game.arcPoint = sidestep_point
            self.game.arcPointRotation = 0.0

        return points

    def mirrorPointArc(self, points, mirror_vec, origin):
        mirrored_points = []
        # Mirror about a vector (not necessarily passing through (0,0); use 'origin' as the base point)
        # To mirror about an arbitrary vector, you need to specify the vector direction.
        # Here, let's add an optional argument for the mirror vector:
        # Usage: mirrorPointArc(points, mirror_vec=Vec2(1,0))
        #mirror_vec = getattr(self, 'mirror_vec', Vec2(1, 0))  # Default to x-axis if not set
        #mirror_vec = mirror_vec
        mirror_vec = mirror_vec.normalized()
        #print("Mirroring with vector:", mirror_vec)
        for p in points:
            # Vector from origin to point
            #print("Original point:", p)
            v = p - origin
            # Project v onto mirror_vec
            #print(v.dot(mirror_vec))
            #print(v.normalized().dot(mirror_vec))
            proj =  (mirror_vec * v.normalized().dot(mirror_vec)) * v.length()
            # Perpendicular component
            #print("v:", v, "proj:", proj)
            perp = v - proj
            #print("perp:", perp)
            # Mirror: subtract twice the perpendicular component
            mirrored_v = p -  perp * 2
            mirrored_points.append(Vec2(mirrored_v.x, mirrored_v.y))
        """ for p in points:
            mirrored_points.append(Vec2(0.5 - (p.x - 0.5), p.y)) """
        return mirrored_points
    

    
    def rotatePoint(self, point, angle_degrees, origo=Vec2(0.25,0.25)):
        angle_radians = math.radians(angle_degrees)
        cos_angle = math.cos(angle_radians)
        sin_angle = math.sin(angle_radians)
        x = point.x - origo.x
        y = point.y - origo.y
        # Counter-clockwise rotation
        x_rotated = x * cos_angle - y * sin_angle
        y_rotated = x * sin_angle + y * cos_angle
        return Vec2(x_rotated + origo.x, y_rotated + origo.y)
    
    def meshPointArc(self, origo, num_points=40, mouse_pos=None, rotationangle=-21):
        if not hasattr(self.game, 'mesh_drawer'):
            self.game.mesh_drawer = MeshDrawer()
            self.game.mesh_drawer.setBudget(1000)
            
            self.game.mesh_drawer_node = self.game.mesh_drawer.getRoot()
            self.game.mesh_drawer_node.setTwoSided(True)
            self.game.mesh_drawer_node.reparentTo(self.game.smiley)

        #self.game.mesh_drawer.begin(base.cam, render)
        points = self.pointArc(origo, num_points, mouse_pos, rotationangle)
        mesh = BulletTriangleMesh()

        for i in range(1, len(points)-2):
            p0 = Point3(points[0].x, points[0].y, 0)
            p1 = Point3(points[i].x, points[i].y, 0)
            p2 = Point3(points[i+1].x, points[i+1].y, 0)
            mesh.add_triangle(p0, p1, p2)

        body = BulletRigidBodyNode('arcarea')
        shape = BulletTriangleMeshShape(mesh, False)
        body.addShape(shape)
        # Detach any existing BulletRigidBodyNode children from mesh_drawer_node
        for child in self.game.mesh_drawer_node.getChildren():
            if child.node().isOfType(BulletRigidBodyNode.getClassType()):
                self.game.world.removeRigidBody(child.node())
                child.detachNode()
        bodyNP = self.game.mesh_drawer_node.attachNewNode(body)
        bodyNP.setHpr(90,0,0)
        bodyNP.node().setMass(0)
        bodyNP.setCollideMask(BitMask32.allOn())
        self.game.world.attachRigidBody(bodyNP.node())
        return points

    # ─── Movement & Pathfinding ───────────────────────────────────────────

    def pathTerrainModifier(self, unit, from_pos, to_pos) -> int:
        """Movement characteristic modifier from terrain on the move path.

        Difficult terrain is -1 to Movement whether the unit starts in it,
        passes through it or ends in it (Rulebook p. 135). Uses the terrain
        field, so the penalty matches the shape the player can see.
        """
        tm = getattr(self.game, 'terrain_manager', None)
        if tm is None:
            return 0
        mod = 0
        for t in tm.get_terrain_between(from_pos, to_pos):
            mod = min(mod, t.movement_modifier)
        return mod

    def pathTowardsMouse(self,unit,x=None,y=None):
        if not base.mouseWatcherNode.hasMouse():
            return
        if base.mouseWatcherNode.hasMouse() and x is None and y is None:
            self.game.unitToMove=unit
            x = base.mouseWatcherNode.getMouseX()
            y = base.mouseWatcherNode.getMouseY()
            pMouse = base.mouseWatcherNode.getMouse()
            pFrom = Point3()
            pTo = Point3()
            base.camLens.extrude(pMouse, pFrom, pTo)

            # Transform to global coordinates
            pFrom = render.getRelativePoint(base.cam, pFrom)
            pTo = render.getRelativePoint(base.cam, pTo)
        else:# x is not None and y is not None:
            self.game.unitToMove=unit
            pFrom = Point3()
            #pFrom = render.getRelativePoint(base.cam, pFrom)
            pFrom = Point3(x, y, 10)
            pTo = Point3(x, y, -10)
            #pTo = render.getRelativePoint(base.cam, pTo)
            
        if True:
            #print(x,y)
            #surface.set_shader_input("pos", Vec3(base.mouseWatcherNode.getMouseX(),0,base.mouseWatcherNode.getMouseY())*4)
            #pFrom = Point3(0, 0, 0)
            #pTo = Point3(10, 0, 0)

            # Get to and from pos in camera coordinates
            #pFrom = render.getRelativePoint(base.cam, pFrom)

            result = self.game.world.rayTestClosest(pFrom, pTo, BitMask32.bit(1))

            # Skirmishers move as a loose group: free 360° translation, no wheel.
            if getattr(unit, 'isSkirmisher', False) and result.hasHit():
                self._skirmishMovePreview(unit, result.getHitPos())
                return


            #self.game.smiley.setPos(result.getHitPos() + Vec3(0,0,2))
            #self.game.move_node_smoothly(self.game.smiley, result.getHitPos() + Vec3(0,0,0.1), duration=0.5)

            groundSizeboundingbox=self.game.ground.getTightBounds()
            pos=result.getHitPos()/abs(groundSizeboundingbox[0][0])
            self.game.ground.set_shader_input("pos", pos)
            #self.game.polygonpoints = []
            pos += Vec3(1, 1, 1)
            pos *= 0.5
            """ self.game.polygonpoints.insert(0, Vec2(pos.x, pos.y))
            if len(self.game.polygonpoints) > 6:
                self.game.polygonpoints.pop() """
            
            

            unitwidth=unit.unitWidth/abs(groundSizeboundingbox[0][0])/2

            unitheight=unit.unitHeight/abs(groundSizeboundingbox[0][1])/2

            unitrotation=unit.bodyNP.getH()

            unitposxy=Vec2(unit.bodyNP.getX()/abs(groundSizeboundingbox[0][0]), unit.bodyNP.getY()/abs(groundSizeboundingbox[0][1]))
            unitposxy += Vec2(1,1)
            unitposxy *= 0.5

            #unitposxy += Vec2(-math.cos(math.radians(unitrotation))*unitwidth*0.5, -math.sin(math.radians(unitrotation))*unitheight*0.5)
            
            unitposxy.x += -math.cos(math.radians(unitrotation))*unitwidth*0.5
            unitposxy.y += -math.sin(math.radians(unitrotation))*unitwidth*0.5

            #pos.x = -math.cos(math.radians(unitrotation))*unitwidth*0.5
            #pos.y = -math.sin(math.radians(unitrotation))*unitwidth*0.5 

            """ for rule in self.game.unitToMove.unit.model.special_rules:
                if rule.get('move'):
                    #print("Unit is flatfooted, cannot move.")
                    rule['move'](self.game.unitToMove.unit.model)
            
            for rule in self.game.unitToMove.unit.model.special_rules:
                if rule.get('mountUnit'):
                    for ruleM in rule['mountUnit'].model.special_rules:
                        if ruleM.get('move'):
                            ruleM['move'](rule['mountUnit'].model)

            M=str(self.game.unitToMove.unit.model.characteristics['M'])
            if M.isdigit():
                M = int(M)
            else:
                print(f"Warning: M value '{M}' is not a number, defaulting to 1")
                M = 0
            move = M*2
            if unit.state == "IsPursuing":
                move = 21

            for rule in self.game.unitToMove.unit.model.special_rules:
                if rule.get('mountUnit'):
                    mountmove= int(rule['mountUnit'].model.characteristics['M'])*2
                    move = max(move, mountmove)
            print("Unit move:", move)
            
            self.game.unitToMove.unit.model.reset_characteristics()
            for rule in self.game.unitToMove.unit.model.special_rules:
                if rule.get('mountUnit'):
                    rule['mountUnit'].model.reset_characteristics() """

            # Mounted units always move using their mount's Movement.
            # Flyers use their Fly Movement characteristic instead.
            _model = self.game.unitToMove.unit.model
            _flying = _model.is_flying()
            M = _model.get_fly_movement(default=0) if _flying else _model.get_movement(default=0)
            # ── Terrain penalty ── (flyers pass over terrain freely)
            terrainMod = 0
            if not _flying and result.hasHit():
                terrainMod = self.pathTerrainModifier(
                    unit, unit.bodyNP.getPos(), result.getHitPos())
            M = max(1, M + terrainMod)
            # This arc is the one kept when the path runs into a unit, so it is
            # the charge-declaration range rather than a march.
            move = max_charge_range(M, unit_has_swiftstride(unit))
            if unit.state == "IsPursuing":
                move = 21

            self.game.polygonpoints = self.pointArc(origo=unitposxy, num_points=80, mouse_pos=Vec2(pos.x, pos.y),
                                               width=unitwidth,height=unitheight, rotationangle=unit.bodyNP.getH(),
                                               movedistance=move/(2*abs(groundSizeboundingbox[0][1])),
                                               sidemove=(M/2)/(2*abs(groundSizeboundingbox[0][1])))
            #self.game.polygonpoints = self.mirrorPointArc(self.game.polygonpoints)

            
            #self.game.playerNP.setPos(result.getHitPos()+Vec3(10,10,0))
            #self.game.playerNP.node().setLinearMovement(Vec3(10,10,0), True)
            p1 = (self.game.polygonpoints[self.game.numsPoints-3]*2-1)*50
            p2 = (self.game.polygonpoints[self.game.numsPoints-2]*2-1)*50
            p3 = (self.game.polygonpoints[0]*2-1)*50
            p4 = (self.game.polygonpoints[self.game.numsPoints-1]*2-1)*50
            #self.game.world.doPhysics(0.016)
            closest_dist = float('inf')
            closest_pos = None
            
            frac,closest_pos_frac,tsTo = self.sweepTestRot(unit,p3,self.game.arcPointRotation)
            if frac < 1.0:
                self.game.arcPointRotation *= frac
                closest_dist = 0
                closest_pos = closest_pos_frac

            else:
                dire=(Vec3(p2.x, p2.y, .9) - Vec3(p1.x, p1.y, .9) ).normalized()
                le=(Vec3(p2.x, p2.y, .9) - Vec3(p1.x, p1.y, .9) ).length()
                #le=move/(2*abs(groundSizeboundingbox[0][1]))
                #le-=math.radians(abs(self.game.arcPointRotation))*unit.unitWidth
                frac,closest_pos_frac = self.sweepTestDir(unit,tsTo,dire,le)
                if frac < 1.0:
                    closest_dist = le*frac
                    closest_pos = closest_pos_frac

            


            if closest_pos:
                self.game.unitHitPos = closest_pos
                self.game.playerNP.setPos(closest_pos)
                #self.game.z2.setPos(closest_pos + Vec3(0,0,0.5))

                newmove = closest_dist+math.radians(abs(self.game.arcPointRotation))*unit.unitWidth
                self.game.polygonpoints = self.pointArc(origo=unitposxy, num_points=80, mouse_pos=Vec2(pos.x, pos.y),
                                                width=unitwidth,height=unitheight, rotationangle=unit.bodyNP.getH(),
                                                movedistance=newmove/(2*abs(groundSizeboundingbox[0][1])),
                                                sidemove=(M/2)/(2*abs(groundSizeboundingbox[0][1])))

                self.game.setGroundOverlay(True, self.game.polygonpoints)
                return

            modifyer=1
            modifyerM=1
            for rule in self.game.unitToMove.unit.model.special_rules:
                if rule.get('move'):
                    #print("Unit is flatfooted, cannot move.")
                    modifyer=rule['move']
                    #M = str(int(int(M) * modifyer))

            for rule in self.game.unitToMove.unit.model.special_rules:
                if rule.get('mountUnit'):
                    for ruleM in rule['mountUnit'].model.special_rules:
                        if ruleM.get('move'):
                            #ruleM['move'](rule['mountUnit'].model)
                            modifyerM=ruleM['move']

            # Mounted units always move using their mount's Movement.
            # Flyers use their Fly Movement characteristic instead.
            _model = self.game.unitToMove.unit.model
            _flying = _model.is_flying()
            M = _model.get_fly_movement(default=0) if _flying else _model.get_movement(default=0)
            M = max(1, M + terrainMod)   # difficult terrain: -1 Movement, min 1
            move = M*2
            move = move * (modifyerM if _model.is_mounted() else modifyer)
            if unit.state == "IsPursuing":
                move = 21

            move = int(move)
            
            """ self.game.unitToMove.unit.model.reset_characteristics()
            for rule in self.game.unitToMove.unit.model.special_rules:
                if rule.get('mountUnit'):
                    rule['mountUnit'].model.reset_characteristics() """
            self.game.polygonpoints = self.pointArc(origo=unitposxy, num_points=80, mouse_pos=Vec2(pos.x, pos.y),
                                               width=unitwidth,height=unitheight, rotationangle=unit.bodyNP.getH(),
                                               movedistance=move/(2*abs(groundSizeboundingbox[0][1])),
                                               sidemove=(M/2)/(2*abs(groundSizeboundingbox[0][1])))
            #self.game.polygonpoints = self.mirrorPointArc(self.game.polygonpoints)

            
            #self.game.playerNP.setPos(result.getHitPos()+Vec3(10,10,0))
            #self.game.playerNP.node().setLinearMovement(Vec3(10,10,0), True)
            p1 = (self.game.polygonpoints[self.game.numsPoints-3]*2-1)*50
            p2 = (self.game.polygonpoints[self.game.numsPoints-2]*2-1)*50
            p3 = (self.game.polygonpoints[0]*2-1)*50
            p4 = (self.game.polygonpoints[self.game.numsPoints-1]*2-1)*50
            #self.game.world.doPhysics(0.016)
            closest_dist = float('inf')
            closest_pos = None
            
            frac,closest_pos_frac,tsTo = self.sweepTestRot(unit,p3,self.game.arcPointRotation)
            if frac < 1.0:
                self.game.arcPointRotation *= frac
                closest_dist = 0
                closest_pos = closest_pos_frac

            else:
                dire=(Vec3(p2.x, p2.y, .9) - Vec3(p1.x, p1.y, .9) ).normalized()
                le=(Vec3(p2.x, p2.y, .9) - Vec3(p1.x, p1.y, .9) ).length()
                #le=move/(2*abs(groundSizeboundingbox[0][1]))
                #le-=math.radians(abs(self.game.arcPointRotation))*unit.unitWidth
                frac,closest_pos_frac = self.sweepTestDir(unit,tsTo,dire,le)
                if frac < 1.0:
                    closest_dist = le*frac
                    closest_pos = closest_pos_frac

            


            if closest_pos:
                self.game.unitHitPos = closest_pos
                self.game.playerNP.setPos(closest_pos)
                #self.game.z2.setPos(closest_pos + Vec3(0,0,0.5))

                newmove = closest_dist+math.radians(abs(self.game.arcPointRotation))*unit.unitWidth
                self.game.polygonpoints = self.pointArc(origo=unitposxy, num_points=80, mouse_pos=Vec2(pos.x, pos.y),
                                                width=unitwidth,height=unitheight, rotationangle=unit.bodyNP.getH(),
                                                movedistance=newmove/(2*abs(groundSizeboundingbox[0][1])),
                                                sidemove=(M/2)/(2*abs(groundSizeboundingbox[0][1])))

            self.game.setGroundOverlay(True, self.game.polygonpoints)

            return

    def debug_ray(self, pFrom, pTo):
        # Create a line to visualize the ray
        line = LineSegs()
        line.setColor(1, 0, 0, 1)  # Red line
        line.moveTo(pFrom)
        line.drawTo(pTo)
        
        line_node = render.attachNewNode(line.create())
        # Remove after 2 seconds
        #self.game.taskMgr.doMethodLater(2.0, line_node.removeNode, "remove-debug-ray")

    def drawRectangle(self, center=Point3(0, 0, 0), width=5, height=3, color=(1, 0, 0, 1)):
        """
        Draws a rectangle using LineSegs.
        
        Args:
            center: Center position of the rectangle
            width: Width of the rectangle
            height: Height of the rectangle
            color: Color of the rectangle as (r, g, b, a) tuple
        
        Returns:
            NodePath of the created rectangle
        """
        
        
        line_segs = LineSegs()
        line_segs.setColor(*color)
        line_segs.setThickness(2.0)
        
        # Calculate corner points
        half_width = width / 2
        half_height = height / 2
        
        corners = [
            Point3(center.x - half_width, center.y - half_height, center.z),  # Bottom-left
            Point3(center.x + half_width, center.y - half_height, center.z),  # Bottom-right
            Point3(center.x + half_width, center.y + half_height, center.z),  # Top-right
            Point3(center.x - half_width, center.y + half_height, center.z),  # Top-left
        ]
        
        # Draw the rectangle by connecting corners
        line_segs.moveTo(corners[0])
        for i in range(1, len(corners)):
            line_segs.drawTo(corners[i])
        # Close the rectangle
        line_segs.drawTo(corners[0])
        
        rectangle_node = line_segs.create()
        rectangleLine = render.attachNewNode(rectangle_node)
        rectangleLine.setName("Rectangle")
        
        return rectangleLine

    # ─── Unit Movement Execution ──────────────────────────────────────────

    def _destOnUnit(self, unit, endp) -> bool:
        """True if placing *unit* at *endp* would overlap another unit's body.

        Used by the flyer preview to decide between flying over (open ground)
        and charging (landing on top of a model).
        """
        body = unit.bodyNP
        saved = body.getPos()
        body.setPos(endp.x, endp.y, saved.z)
        body.node().setTransformDirty()
        contact = self.game.checkUnitContactSmall(unit)
        body.setPos(saved)
        body.node().setTransformDirty()
        return contact is not None

    def _skirmishMovePreview(self, unit, target):
        """Free-move preview for Skirmishers: straight-line translation up to
        the move allowance in any direction (no wheel), with a circular range
        indicator.  Sets arcPoint (normalised) + arcPointRotation=0 for moveUnit.
        """
        half = abs(self.game.ground.getTightBounds()[0][0]) or 50.0
        cur = unit.bodyNP.getPos()
        _model = unit.unit.model
        # Flyers use their Fly Movement characteristic instead of M.
        m = _model.get_fly_movement(0) if _model.is_flying() else _model.get_movement(0)
        maxmove = 21.0 if unit.state == "IsPursuing" else m * 2.0

        d = Vec3(target.x - cur.x, target.y - cur.y, 0)
        dist = d.length()
        dirn = d / dist if dist > 1e-6 else Vec3(0, 1, 0)
        clamped = min(dist, maxmove)
        endp = cur + dirn * clamped

        # Non-flyers always stop at a blocking body.  Flyers pass over units,
        # but if the preview lands on top of a unit it becomes a charge, so we
        # sweep to stop against that unit and let moveUnit set up the charge.
        if not _model.is_flying() or self._destOnUnit(unit, endp):
            frac, _hit = self.sweepTestDir(unit, unit.bodyNP.getTransform(),
                                           dirn, clamped, pass_over=False)
            if frac < 1.0:
                clamped *= frac
            endp = cur + dirn * clamped

        self.game.arcPoint = Vec2((endp.x / half + 1) * 0.5, (endp.y / half + 1) * 0.5)
        self.game.arcPointRotation = 0
        self.game.unitHitPos = endp
        self.game.playerNP.setPos(endp)
        # Straight-line distance travelled, for the charge-roll check.
        self.game.moveArceDistance = clamped

        # Circular move-range indicator centred on the unit.
        self.game.polygonpoints = self.shootingArc(
            cur, num_points=80, radius=maxmove / (half * 2.0), full_circle=True)
        self.game.setGroundOverlay(True, self.game.polygonpoints)

        # Ghost footprint showing where the unit will end up.
        if getattr(self.game, 'skirmMoveGhost', None):
            self.game.skirmMoveGhost.removeNode()
        self.game.skirmMoveGhost = self.drawRectangle(
            center=Point3(endp.x, endp.y, 0.3),
            width=unit.unitWidth, height=unit.unitHeight,
            color=(0.4, 1.0, 0.4, 1.0))

    def moveUnit(self, unit):
        if taskMgr.hasTaskNamed("taskLoopPathTowardsMouse"):
            taskMgr.remove("taskLoopPathTowardsMouse")
        # Clear the skirmisher destination ghost once the move is committed.
        if getattr(self.game, 'skirmMoveGhost', None):
            self.game.skirmMoveGhost.removeNode()
            self.game.skirmMoveGhost = None
            
        if unit.state != "Idle":
            if unit.state != "IsPursuing":
                print("Unit is not pursuing, cannot move.")
                return
        
        
        
        
        if unit.hasMovedThisTurn:
            print("Unit has already moved this turn.")
            return
        
        pos = self.game.arcPoint
        pos=pos*2
        pos -= Vec2(1,1)
        #pos.x *= abs(self.game.ground.getTightBounds()[0][0])
        pos.x *= 50
        pos.y *= 50
        oposUnit=unit.bodyNP.getPos()
        orotUnit=unit.bodyNP.getHpr()
        unit.bodyNP.setPos(pos.x , pos.y , 0)
        # Skirmishers translate freely (no wheel, no back-pivot); formed units
        # wheel about their rear as before.
        if not getattr(unit, 'isSkirmisher', False):
            unit.bodyNP.setH(unit.bodyNP.getH() + self.game.arcPointRotation)
            unit.bodyNP.setPos(unit.bodyNPback.getPos(render))
        #self.game.checkUnitContact(unit)
        c = self.game.checkUnitContactSmall(unit)
        
        if c:
            defenderNP = render.find(f"**/{c.getNode1().getName()}")
            defenderUnit=self.game.getSelectedUnit(defenderNP.node())

            if unit.state == "IsPursuing":
                pass
            else:
                if defenderUnit in self.game.player1Units:
                    if unit in self.game.player1Units:
                        print("Both units belong to Player 1, cannot enter combat.")
                        direction = unit.bodyNP.getPos() - defenderNP.getPos()
                        direction.normalize()
                        self.fallBackContactTest(unit.bodyNP, direction*.3)
                        unit.request("Moved")
                        return
                if defenderUnit in self.game.player2Units:
                    if unit in self.game.player2Units:
                        print("Both units belong to Player 2, cannot enter combat.")
                        direction = unit.bodyNP.getPos() - defenderNP.getPos()
                        direction.normalize()
                        self.fallBackContactTest(unit.bodyNP, direction*.3)
                        unit.request("Moved")
                        return

            taskMgr.add(self.game.chargeAndChargeReaction, extraArgs=[unit, c,oposUnit, orotUnit],appendTask=True)
            unit.isChargingMove = True   # exempt from Panic while making the charge
            #self.game.getFlankFromContact(unit, c)
            unit.model.setColor(.7,0.7,0.7,1)
            copiedUnit=unit.bodyNP.copyTo(render)
            self.game.unitCopies.append(copiedUnit)
            unit.model.setColor(unit.color)

            #copyiedUnit.setColor(1,0,0,1)
            copiedUnit.setColor(.7,0.7,0.7,1)
            copiedUnit.setPos(oposUnit)
            copiedUnit.setHpr(orotUnit)
        else:
            unit.request("Moved")
            self.alignModelsToHillNormal(unit)
        self.game.bakeTextures(self.game.ground)

    # ─── Flee, Pursuit & Rally ────────────────────────────────────────────

    def checkFleeCaught(self, fleeUnit, pursuerUnit,task):
        #fleeUnit.bodyNP.setCollideMask(BitMask32.bit(2))
        #pursuerUnit.bodyNP.setCollideMask(BitMask32.bit(2))
        #pursuerUnit.bodyNP.node().setCcdMotionThreshold(1e-7)
        #pursuerUnit.bodyNP.node().setCcdSweptSphereRadius(0.50)
        if fleeUnit.bodyNP.isEmpty() or pursuerUnit.bodyNP.isEmpty():
            print("One of the units is already removed, stopping flee catch check.")
            return task.done
        pursuerUnit.bodyNP.node().setTransformDirty()
        fleeUnit.bodyNP.node().setTransformDirty()
        result = self.game.world.contactTestPair(fleeUnit.bodyNP.node(), pursuerUnit.bodyNP.node())
        for contact in result.getContacts():
            print("Contact detected between fleeing unit and pursuer!")
            self.game.world.removeRigidBody(fleeUnit.bodyNP.node())
            fleeUnit.model.removeNode()
            fleeUnit.bodyNP.removeNode()
            self.game.units.remove(fleeUnit)
            if fleeUnit in self.game.player1Units:
                self.game.player1Units.remove(fleeUnit)
            if fleeUnit in self.game.player2Units:
                self.game.player2Units.remove(fleeUnit)
            messenger.send('unit-move-complete')
            return task.done
        return task.cont


    def fleeDirectionMultUnits(self, loser,winners):
        
        winPos=Vec3(0,0,0)
        lPos=loser.bodyNP.getPos()
        for w in winners:
            winPos+=w.bodyNP.getPos()
        winPos=winPos/len(winners)
        ldir=[]
        for w in winners:
            dir = (lPos - w.bodyNP.getPos()).normalized()
            ldir.append(dir)
        finalDir=Vec3(0,0,0)
        for d in ldir:
            finalDir+=d
        return finalDir.normalized()


    
    def centerOfModels(self, unit):
        coords=[]
        for child in unit.model.getChildren():
            coords.append(child.getPos(render))
        
        min_x = min(pos.x for pos in coords)
        max_x = max(pos.x for pos in coords)
        min_y = min(pos.y for pos in coords)
        max_y = max(pos.y for pos in coords)
        min_z = min(pos.z for pos in coords)
        max_z = max(pos.z for pos in coords)

        center = Point3((min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2)
        return center

    def getCenterOfUnit(self, unit):
        bounds = unit.model.getTightBounds()
        center = (bounds[0] + bounds[1]) * 0.5
        # Convert from local model coordinates to world coordinates
        world_center = unit.model.getPos(render) + center
        return world_center
        
    def applyWounds(self, unit, wounds):
        """Turn unsaved wounds into slain models.

        Multi-wound models (a chariot has 6) soak several wounds each, and the
        leftovers stay on the wounded model rather than carrying to the next.
        """
        if wounds <= 0:
            return
        W = max(1, stat_value(unit.unit.model.characteristics.get('W'), 1))
        if W == 1:
            self.removeModelsFromUnit(unit, wounds)
            return
        pool = getattr(unit, 'woundsOnModel', 0) + wounds
        slain, unit.woundsOnModel = divmod(pool, W)
        print(f"{unit.unit.name}: {wounds} wound(s) -> {slain} slain "
              f"({unit.woundsOnModel}/{W} on the wounded model)")
        if slain:
            self.removeModelsFromUnit(unit, slain)

    def removeModelsFromUnit(self, unit, models_to_remove):
        # Unit may have already been fully removed by another simultaneous combat
        if unit not in self.game.units:
            print(f"removeModelsFromUnit: {unit.unitName} is no longer in game, skipping.")
            return
        if unit.model.isEmpty() or unit.bodyNP.isEmpty():
            print(f"removeModelsFromUnit: {unit.unitName} NodePath already empty, skipping.")
            return
        # Capture Unit Strength and footprint before removal, for the
        # nearby-friend-destroyed Panic test.  US uses the unit's size at the
        # start of this phase/combat (casualties are removed in stages, so the
        # destroying call can see very few models).
        pre_models = len(unit.model.getChildren())
        sop = getattr(unit, 'startOfPhaseModels', pre_models) or pre_models
        us_before = unit.unit.model.unit_strength() * max(pre_models, sop)
        friendly_side = (self.game.player1Units if unit in self.game.player1Units
                         else self.game.player2Units)
        _dp = unit.bodyNP.getPos()
        death_box = (_dp.x, _dp.y,
                     getattr(unit, 'unitWidth', 2.0) / 2.0,
                     getattr(unit, 'unitHeight', 2.0) / 2.0,
                     unit.bodyNP.getH())
        cildren = unit.model.getChildren()
        models_to_remove = min(len(cildren), models_to_remove)
        #unit.model.ls()
        for i in range(models_to_remove):
            #cildren[-1*(i+1)].removeNode()
            cildren = unit.model.getChildren()
            cildren[-1].removeNode()
        
        cildren = unit.model.getChildren()
        # Keep the logical model count in sync with the surviving models.
        unit.unit.nmodels = len(cildren)
        if len(cildren) == 0:
            print(f"All models removed from unit {unit.unit.name}. Removing unit from game.")
            try:
                if self.game.attackSequence.isPlaying():
                    self.game.attackSequence.pause()
            except AttributeError:
                pass
            #self.game.attackSequence.finish()
            for u in unit.isInCombatWith:
                u.request("Idle")
            #messenger.send('unit-move-complete')
            on_host_removed(self.game, unit)
            self.game.world.removeRigidBody(unit.bodyNP.node())
            unit.bodyNP.removeNode()
            unit.model.removeNode()
            self.game.units.remove(unit)
            if unit in self.game.player1Units:
                self.game.player1Units.remove(unit)
            if unit in self.game.player2Units:
                self.game.player2Units.remove(unit)
            
            # A destroyed unit of US>=5 panics nearby friends.
            if getattr(self.game, 'psychology', None):
                self.game.psychology.on_unit_destroyed(death_box, friendly_side, us_before)
            return
        self.game.world.removeRigidBody(unit.bodyNP.node())
        for shape in unit.bodyNP.node().shapes:
            unit.bodyNP.node().removeShape(shape)
        if unit.model.isEmpty():
            return
        box_size = unit.footprintSize()
        shape = BulletBoxShape(box_size * 0.5)  # BulletBoxShape takes half-extents
        unit.bodyNP.node().addShape(shape)
        unit.bodyNP.node().setMass(0)  # Static object
        self.game.world.attachRigidBody(unit.bodyNP.node())
        unit.applyFootprint(box_size)

    # ─── Sweep Tests ──────────────────────────────────────────────────────

    #def sweepTest(self, shape,startPos,startHpr, endPos,endHpr):
    def sweepTest(self, unit, direction,length):
        startPos=unit.bodyNP.getPos()
        Hpr=unit.bodyNP.getHpr()
        unit.bodyNP.lookAt(startPos + direction)
        nHpr = unit.bodyNP.getHpr()
        unit.bodyNP.setHpr(Hpr)


        tsFrom = TransformState.makePosHpr(startPos, nHpr)
        tsTo = TransformState.makePosHpr(startPos + direction * length, nHpr)
        shape = unit.bodyNP.node().getShape(0)
        #shape = BulletSphereShape(0.5)
        penetration = 0.0
        omasks=[]
        for u in self.game.units:
            omasks.append(u.bodyNP.getCollideMask())
            u.bodyNP.setCollideMask(BitMask32.bit(9))
        unit.bodyNP.setCollideMask(BitMask32.bit(30))
        for u in unit.isInCombatWith:
            u.bodyNP.setCollideMask(BitMask32.bit(30))
        #self.game.mountedKnightOfTheRealm.bodyNP.setCollideMask(BitMask32.bit(9))
        result = base.world.sweepTestClosest(shape, tsFrom, tsTo,BitMask32.bit(9))
        #unit.setCollideMask(BitMask32.bit(1))
        for i,u in enumerate(self.game.units):
            u.bodyNP.setCollideMask(omasks[i])
        if result.hasHit():
            return result.getHitFraction()
        return 1.0
    
    def sweepTestRot(self, unit, point,angle,mask=BitMask32.bit(9),pass_over=None):
        # Flyers pass over other units: skip the unit-sweep (bit 9) hit test
        # unless the caller forces detection with pass_over=False.
        if pass_over is None:
            pass_over = unit.unit.model.is_flying()
        if mask == BitMask32.bit(9) and pass_over:
            mask = BitMask32.allOff()
        startPos=unit.bodyNP.getPos()
        Hpr=unit.bodyNP.getHpr()
        shape = unit.bodyNP.node().getShape(0)

        newxy= self.rotatePoint(Vec2(startPos.x, startPos.y), angle, point)
        
        #unit.bodyNP.lookAt(startPos + direction)
        nHpr = Hpr + Vec3(angle,0,0)
        #unit.bodyNP.setHpr(Hpr)


        tsFrom = TransformState.makePosHpr(startPos, Hpr)
        tsTo = TransformState.makePosHpr(Vec3(newxy[0], newxy[1], startPos.z), nHpr)
        
        #shape = BulletSphereShape(0.5)
        penetration = 0.0
        omasks=[]
        for u in self.game.units:
            omasks.append(u.bodyNP.getCollideMask())
            u.bodyNP.setCollideMask(BitMask32.bit(9))
        unit.bodyNP.setCollideMask(BitMask32.bit(30))
        """ for u in unit.isInCombatWith:
            u.bodyNP.setCollideMask(BitMask32.bit(30)) """
        #self.game.mountedKnightOfTheRealm.bodyNP.setCollideMask(BitMask32.bit(9))
        result = base.world.sweepTestClosest(shape, tsFrom, tsTo,mask)
        #unit.setCollideMask(BitMask32.bit(1))
        for i,u in enumerate(self.game.units):
            u.bodyNP.setCollideMask(omasks[i])
        if result.hasHit():
            return result.getHitFraction(),result.getHitPos(),tsTo
        return 1.0,None,tsTo
    
    def sweepTestDir(self, unit, tsFrom, direction,length,mask=BitMask32.bit(9),pass_over=None):
        # Flyers pass over other units: skip the unit-sweep (bit 9) hit test
        # unless the caller forces detection with pass_over=False.
        if pass_over is None:
            pass_over = unit.unit.model.is_flying()
        if mask == BitMask32.bit(9) and pass_over:
            mask = BitMask32.allOff()
        
        #tsFrom = TransformState.makePosHpr(startPos, nHpr)
        tsTo = TransformState.makePosHpr(tsFrom.getPos() + direction * length, tsFrom.getHpr())

        shape = unit.bodyNP.node().getShape(0)
               
        #shape = BulletSphereShape(0.5)
        penetration = 0.0
        omasks=[]
        for u in self.game.units:
            omasks.append(u.bodyNP.getCollideMask())
            u.bodyNP.setCollideMask(BitMask32.bit(9))
        unit.bodyNP.setCollideMask(BitMask32.bit(30))
        """ for u in unit.isInCombatWith:
            u.bodyNP.setCollideMask(BitMask32.bit(30)) """
        #self.game.mountedKnightOfTheRealm.bodyNP.setCollideMask(BitMask32.bit(9))
        #result = base.world.sweepTestClosest(shape, tsFrom, tsTo,BitMask32.bit(9))
        result = base.world.sweepTestClosest(shape, tsFrom, tsTo,mask)
        #unit.setCollideMask(BitMask32.bit(1))
        for i,u in enumerate(self.game.units):
            u.bodyNP.setCollideMask(omasks[i])
        if result.hasHit():
            return result.getHitFraction(),result.getHitPos()
        return 1.0,None

    # ─── Fallback & Contact Resolution ────────────────────────────────────

    def fallBackContactTest(self, unitNP,moveVec=Vec3(0,0,0)):
        unit = self.game.getSelectedUnit(unitNP.node())
        #unit.bodyNP.setCollideMask(BitMask32.bit(1))
        for us in self.game.units:
            if not us.bodyNP.isEmpty():
                us.bodyNP.node().setTransformDirty()
        for u in unit.isInCombatWith:
            if not u.bodyNP.isEmpty():
                u.bodyNP.node().setTransformDirty()
        ghost = unit.bodyNP.node()
        
        result = base.world.contactTest(ghost)
        #unit.bodyNP.setCollideMask(unit.bitmask)
        for contact in result.getContacts():
            node_name = contact.getNode1().getName()
            if node_name.startswith('UnitCollision-') :
                mpoint = contact.getManifoldPoint()
                #np=render.find(f"**/{contact.getNode1().getName()}")
                #selected_unit = self.game.getSelectedUnit(contact.getNode1())

                contact_unit = self.game.getSelectedUnit(contact.getNode1())
                #contact_unit.bodyNP.node().setTransformDirty()
                """ if contact_unit in unit.isInCombatWith:
                    print("Contact with unit in combat, no fallback movement applied.")
                    continue """
                #self.game.z2.setPos(unit.bodyNP.getPos() + mpoint.getLocalPointA())
                selected_unit = unit
                """ if selected_unit.state == 'InCombat':
                    print("Unit in combat, cannot be moved in bounds again now!")
                    return """
                """ if selected_unit.state == 'IsFleeing':
                    print("Unit is fleeing out of the battle field, it is destroyed!")
                    base.world.removeRigidBody(selected_unit.bodyNP.node())
                    self.game.units.remove(selected_unit)
                    if selected_unit in self.game.player1Units:
                        self.game.player1Units.remove(selected_unit)
                    if selected_unit in self.game.player2Units:
                        self.game.player2Units.remove(selected_unit)
                    selected_unit.bodyNP.removeNode()
                    selected_unit.model.removeNode()
                    return """
                np=unit.bodyNP
                #np.setHpr(Vec3(H,0,0))
                cpos=Vec3(np.getPos())
                
                if moveVec==Vec3(0,0,0):
                    #moveVec=mpoint.getLocalPointA() * 1.1
                    #moveVec=mpoint.getNormalWorldOnB() * (mpoint.getDistance() * 1.1)
                    moveVec=unit.bodyNP.getPos() - contact_unit.bodyNP.getPos()
                    moveVec.normalize()
                    moveVec*=0.1
                    moveVec.z=0
                np.setPos(cpos+moveVec)
                return self.fallBackContactTest(unitNP,moveVec)
        
    def fallBack(self, loser,direction,length=10.0,rally=False,GG=False,flee=False):
        if loser.isEmpty():
            print("looser is destryed, no fallback")
            """ w=self.game.getSelectedUnit(winner.node())
            w.isInCombat=False
            w.isInCombatWith=[]
            w.isInCombatFlank=[] """
            #TODO: winner overruns
            return
        
        print(f"{loser.node().getName()} falls back!")
        
        
        loserPos=loser.getPos()
        newPos = loserPos + direction * length
        oldHpr=loser.getHpr()
        if not GG:
            loser.lookAt(loserPos + direction)
        #self.sweepTest(loser.node().getShape(0), loser.getPos(), loser.getHpr(), loser.getPos() + direction * length, loser.getHpr())
        #self.sweepTest(loser, direction, length)
        
        fleeHpr=loser.getHpr()
        if rally:
            r=Vec3(180,0,0)
        else:
            r=Vec3(0,0,0)
        newHpr=loser.getHpr() + r
        loser.setHpr(oldHpr)
        rotate_interval = LerpPosHprInterval(
            loser, 
            duration=0.5, 
            pos=loser.getPos(),
            hpr=fleeHpr,
            blendType='easeInOut'
        )
        move_interval = LerpPosInterval(
            loser, 
            duration=1.0, 
            pos=newPos,
            blendType='easeInOut'
        )

        loser.setPos(newPos)
        bpos=loser.getPos()
        loser.setHpr(newHpr)
        if rally or flee:
            self.fallBackContactTest(loser,direction*.1)
        
        else:
            self.fallBackContactTest(loser,-direction*.1)

        newPos = loser.getPos()
        if (newPos - bpos).length() > 1.1:
            print("Adjusted fallback position due to collision:", bpos,newPos)
            loserUnit = self.game.getSelectedUnit(loser.node())
            loserUnit.endedInUnit=True

        loser.setPos(loserPos)
        loser.setHpr(oldHpr)

        move_interval2 = LerpPosInterval(
            loser, 
            duration=1.0, 
            pos=newPos,
            blendType='easeInOut'
        )

        rotate_interval2 = LerpPosHprInterval(
            loser, 
            duration=0.5, 
            pos=newPos,
            hpr=newHpr,
            blendType='easeInOut'
        )
        sequence = Sequence(
            rotate_interval,
            move_interval,
            move_interval2,
            #Func(self.fallBackContactTest, loser,direction),
            rotate_interval2,
            #Func(self.pursuitMove, winner, loser)
        )
        sequence.start()
        #if rally:
        #    self.fallBackContactTest(loser,direction)
        #return sequence

    def fallBack2(self, loser,direction,length=10.0,rally=False,GG=False,flee=False):
        if loser.isEmpty():
            print("looser is destryed, no fallback")
            """ w=self.game.getSelectedUnit(winner.node())
            w.isInCombat=False
            w.isInCombatWith=[]
            w.isInCombatFlank=[] """
            #TODO: winner overruns
            return
        
        print(f"{loser.node().getName()} falls back!")
        
        
        loserPos=loser.getPos()
        newPos = loserPos + direction * length
        oldHpr=loser.getHpr()
        if not GG:
            loser.lookAt(loserPos + direction)
        #self.sweepTest(loser.node().getShape(0), loser.getPos(), loser.getHpr(), loser.getPos() + direction * length, loser.getHpr())
        #self.sweepTest(loser, direction, length)
        
        fleeHpr=loser.getHpr()
        if rally:
            r=Vec3(180,0,0)
        else:
            r=Vec3(0,0,0)
        newHpr=loser.getHpr() + r
        loser.setHpr(oldHpr)
        rotate_interval = LerpPosHprInterval(
            loser, 
            duration=0.5, 
            pos=loser.getPos(),
            hpr=fleeHpr,
            blendType='easeInOut'
        )
        move_interval = LerpPosInterval(
            loser, 
            duration=1.0, 
            pos=newPos,
            blendType='easeInOut'
        )

        loser.setPos(newPos)
        bpos=loser.getPos()
        loser.setHpr(newHpr)
        if rally or flee:
            self.fallBackContactTest(loser,direction*.1)
        
        else:
            self.fallBackContactTest(loser,-direction*.1)

        newPos = loser.getPos()
        if (newPos - bpos).length() > 1.1:
            print("Adjusted fallback position due to collision:", bpos,newPos)
            loserUnit = self.game.getSelectedUnit(loser.node())
            loserUnit.endedInUnit=True

        loser.setPos(loserPos)
        loser.setHpr(oldHpr)

        move_interval2 = LerpPosInterval(
            loser, 
            duration=1.0, 
            pos=newPos,
            blendType='easeInOut'
        )

        rotate_interval2 = LerpPosHprInterval(
            loser, 
            duration=0.5, 
            pos=newPos,
            hpr=newHpr,
            blendType='easeInOut'
        )
        sequence = Sequence(
            rotate_interval,
            move_interval,
            move_interval2,
            #Func(self.fallBackContactTest, loser,direction),
            rotate_interval2,
            #Func(self.pursuitMove, winner, loser)
        )
        #sequence.start()
        #if rally:
        #    self.fallBackContactTest(loser,direction)
        return sequence

    

    def pursuitMove(self, winner, loser):
        if loser.isEmpty():
            "looser is destryed, no pursuit"
            #TODO: winner overruns
            return
        if winner is None or loser is None or winner.isEmpty() or loser.isEmpty():
            print("Error: winner or loser is None or empty.")
            return
        print(f"{winner.node().getName()} pursues {loser.node().getName()}!")
        winnerPos=winner.getPos()
        loserPos=loser.getPos()
        oldhpr=winner.getHpr()
        winner.lookAt(loserPos)
        pursueHpr=winner.getHpr()
        winner.setHpr(oldhpr)
        rotate_interval = LerpPosHprInterval(
            winner, 
            duration=0.5, 
            pos=winner.getPos(),
            hpr=pursueHpr,
            blendType='easeInOut'
        )
        direction = (loserPos - winnerPos).normalized()
        pursue_distance = 15.0  # Adjust as needed
        newPos = winnerPos + direction * pursue_distance
        move_interval = LerpPosInterval(
            winner, 
            duration=1.0, 
            pos=newPos,
            blendType='easeInOut'
        )
        sequence = Sequence(
            rotate_interval,
            move_interval
        )
        sequence.start()
    
    def unitOnHill(self, unit):
        """Return True if *unit* is standing on a hill terrain piece."""
        if not hasattr(self.game, 'terrain_manager'):
            return False
        terrain = self.game.terrain_manager.get_terrain_at(unit.bodyNP.getPos())
        return terrain is not None and terrain.terrain_type == 'hill'

    def alignModelsToHillNormal(self, unit):
        """Sit every individual model of *unit* on the terrain topography:
        snap its Z to the hill/forest surface height and tilt it to the
        surface normal.  Uses the analytic terrain height (not a physics
        raycast) so it works regardless of collision state."""
        tm = getattr(self.game, 'terrain_manager', None)
        if tm is None:
            return

        up = Vec3(0, 0, 1)
        for child in unit.model.getChildren():
            world_pos = child.getPos(render)

            # Snap the model's world Z onto the terrain surface.
            surf = tm.get_surface_height(world_pos)
            child.setPos(render, world_pos.x, world_pos.y, surf)

            # Tilt to match the surface normal (flat where there's no slope).
            normal = tm.get_surface_normal(world_pos)
            if normal.almostEqual(up, 0.001):
                child.setP(0)
                child.setR(0)
                continue

            axis = up.cross(normal)
            axis.normalize()
            dot = max(-1.0, min(1.0, up.dot(normal)))
            angle_deg = math.degrees(math.acos(dot))
            quat = LRotationf(axis, angle_deg)
            child.setQuat(child.getParent(), quat)
