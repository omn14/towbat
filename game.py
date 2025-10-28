from direct.showbase.ShowBase import ShowBase
from panda3d.core import Plane, PlaneNode, Point3, Vec2, Vec3, Vec4, BitMask32
from panda3d.core import CardMaker
from panda3d.bullet import BulletWorld, BulletPlaneShape, BulletRigidBodyNode, BulletTriangleMesh, BulletTriangleMeshShape
from direct.interval.LerpInterval import LerpPosInterval, LerpPosHprInterval
from direct.interval.IntervalGlobal import Sequence
from direct.interval.FunctionInterval import Func
from panda3d.core import Shader

from shaders.chargedistshaders import *
from panda3d.core import Texture
from panda3d.core import DirectionalLight, AmbientLight
from panda3d.core import MeshDrawer, NodePath
import math
from panda3d.bullet import BulletSphereShape, BulletRigidBodyNode
from panda3d.bullet import BulletDebugNode
from direct.directutil import Mopath
from direct.interval.MopathInterval import MopathInterval
from panda3d.core import NurbsCurveEvaluator, NurbsCurveResult
from panda3d.core import NurbsCurve



class MyApp(ShowBase):
    def __init__(self):
        super().__init__()

        # Disable default camera controls
        #self.disableMouse()

        # Create a flat plane using CardMaker
        cm = CardMaker("ground")
        cm.setFrame(-25, 25, -25, 25)  # 200x200 plane
        self.ground = self.render.attachNewNode(cm.generate())
        self.ground.setPos(0, 0, 0)
        self.ground.setHpr(0, -90, 0)
        self.ground.setColor(0, 1, 0, 1)  # Set plane color to green (RGBA)
        tex = self.loader.loadTexture('maps/noise.rgb')
        self.ground.setTexture(tex)

        # Directional light
        dlight = DirectionalLight('dlight')
        dlight.setColor((0.8, 0.8, 0.7, 1))
        dlnp = self.render.attachNewNode(dlight)
        dlnp.setHpr(-45, -60, 0)
        self.render.setLight(dlnp)

        # Ambient light
        alight = AmbientLight('alight')
        alight.setColor((0.2, 0.2, 0.3, 1))
        alnp = self.render.attachNewNode(alight)
        self.render.setLight(alnp)

        # Load a smiley model and position it above the ground
        self.smiley = self.loader.loadModel('models/goblin_archers.bam')
        self.smiley.reparentTo(self.render)
        self.smiley.setPos(0, 50, 0)
        self.smiley.setScale(2)
        print(self.smiley.getTightBounds())
        self.unitWidth=abs(self.smiley.getTightBounds()[1][0]-self.smiley.getTightBounds()[0][0])
        self.unitHeight=abs(self.smiley.getTightBounds()[1][1]-self.smiley.getTightBounds()[0][1])

        # Make a copy of the smiley model and position it differently
        self.smiley_copy = self.loader.loadModel('models/smiley')
        self.smiley_copy.reparentTo(self.render)
        self.smiley_copy.setPos(0, 0, 0)
        self.smiley_copy.setScale(2)

        # Position the camera above the plane, looking straight down
        self.camera.setPos(0, 300, 300)
        self.camera.lookAt(self.ground)
        #self.camera.setP(-90)  # Pitch downwards
        self.setup_shader()
        self.setup_bullet()
        self.accept('mouse1', self.upAndDown)
        self.accept('q-up', self.pathTowardsMouse)


    

    def setup_shader(self):
        #surface = self.render.find("**/ground")
        surface = self.ground
        shader = Shader.load(Shader.SL_GLSL, "shaders/c2.vert", "shaders/c1.frag")
        surface.setShader(shader)
        surface.setShaderInput("pos", Vec3(0,0,0))
        # Define polygon points for the shader
        self.polygonpoints = []
        num_points = 6  # Example: hexagon
        radius = 0.5
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            self.polygonpoints.append(Vec2(x, y))
        surface.setShaderInput("polygonpoints", self.polygonpoints)
        self.polygonpoints = []

    def setup_bullet(self):
        self.world = BulletWorld()
        shape = BulletPlaneShape(Vec3(0, 0, 1), 1)
        node = BulletRigidBodyNode('Ground')
        node.addShape(shape)
        np = render.attachNewNode(node)
        self.world.attachRigidBody(node)
        """ mesh = BulletTriangleMesh()

        for geomNP in render.findAllMatches('**/+GeomNode'):
            print("fant node")
            geomNode = geomNP.node()
            ts = geomNP.getTransform(np)
            #print(ts)
            for geom in geomNode.getGeoms():
                mesh.addGeom(geom, ts=ts)
                #print(geom)
        #lol

        worldNP = render.attachNewNode('World')
        body = BulletRigidBodyNode('grid')
        shape = BulletTriangleMeshShape(mesh, False)
        bodyNP = worldNP.attachNewNode(body)
        bodyNP.node().addShape(shape)
        bodyNP.setCollideMask(BitMask32.allOn()) 
        self.world.attachRigidBody(bodyNP.node())"""

        # Show Bullet debug nodes
        debugNode = BulletDebugNode('Debug')
        debugNode.showWireframe(True)
        debugNode.showConstraints(True)
        debugNode.showBoundingBoxes(True)
        debugNode.showNormals(True)
        debugNP = render.attachNewNode(debugNode)
        
        self.world.setDebugNode(debugNP.node())
        debugNP.show()
        # Add simple Bullet collision geometry (sphere) to smiley_copy

        # Estimate radius from bounding box
        bounds = self.smiley_copy.getTightBounds()
        center = (bounds[0] + bounds[1]) * 0.5
        radius = max((bounds[1] - bounds[0]).length() * 0.5, 0.1)
        radius *= 0.5  # Adjust scale factor as needed

        smiley_copy_shape = BulletSphereShape(radius)
        smiley_copy_body = BulletRigidBodyNode('SmileyCopy')
        smiley_copy_body.addShape(smiley_copy_shape)
        smiley_copy_np = self.smiley_copy.attachNewNode(smiley_copy_body)
        smiley_copy_np.setCollideMask(BitMask32.allOn())
        self.world.attachRigidBody(smiley_copy_body)

        # Add a task to update Bullet physics every frame
        

        self.taskMgr.add(self.update_physics, "update_physics")

    def update_physics(self,task):
            dt = globalClock.getDt()
            self.world.doPhysics(dt)
            
            return task.cont
    def move_node_smoothly(self, node, target_pos, duration=1.0):
        interval = LerpPosInterval(node, duration, target_pos,blendType='easeInOut')
        mySequence = Sequence(interval)
        mySequence.start()

    def draw_circle(self, center=Point3(0, 0, 0), radius=5, segments=32, color=(1, 0, 0, 1)):

        # Create MeshDrawer and NodePath if not already present
        if not hasattr(self, 'mesh_drawer'):
            self.mesh_drawer = MeshDrawer()
            self.mesh_drawer.setBudget(1000)
            #self.mesh_drawer_np = NodePath(self.mesh_drawer.create())
            self.mesh_drawer_node = self.mesh_drawer.getRoot()
            self.mesh_drawer_node.reparentTo(self.render)

        self.mesh_drawer.begin(base.cam, self.render)
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
                self.mesh_drawer.segment(Point3(x, y, z),
                    Point3(prev_x, prev_y, prev_z), Vec4(0,0,1,1), .1, color
                )
        self.mesh_drawer.end()

    def draw_arc(self, center=Point3(0, 0, 0), radius=5, remainingmove=5,  start_angle=0, end_angle=90, segments=32, color=(1, 0, 0, 1)):
        # Create MeshDrawer and NodePath if not already present
        if not hasattr(self, 'mesh_drawer'):
            self.mesh_drawer = MeshDrawer()
            self.mesh_drawer.setBudget(1000)
            
            self.mesh_drawer_node = self.mesh_drawer.getRoot()
            self.mesh_drawer_node.setTwoSided(True)
            self.mesh_drawer_node.reparentTo(self.smiley)

        self.mesh_drawer.begin(base.cam, self.render)
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
            self.mesh_drawer.tri(
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
        self.mesh_drawer.tri(corner1, rect_color, (0, 0), corner3, rect_color, (1, 1), corner2, rect_color, (1, 0))
        self.mesh_drawer.tri(corner3, rect_color, (1, 1), corner1, rect_color, (0, 0), corner4, rect_color, (0, 1))
        self.mesh_drawer.end()

        mesh = BulletTriangleMesh()

        for geomNP in self.mesh_drawer_node.findAllMatches('**/+GeomNode'):
            print("fant node i mesh drawer")
            geomNode = geomNP.node()
            ts = geomNP.getTransform(self.mesh_drawer_node)
            #print(ts)
            for geom in geomNode.getGeoms():
                mesh.addGeom(geom, ts=ts)
                #print(geom)
        #lol
        body = BulletRigidBodyNode('movearea')
        shape = BulletTriangleMeshShape(mesh, False)
        body.addShape(shape)
        # Detach any existing BulletRigidBodyNode children from mesh_drawer_node
        for child in self.mesh_drawer_node.getChildren():
            if child.node().isOfType(BulletRigidBodyNode.getClassType()):
                self.world.removeRigidBody(child.node())
                child.detachNode()
        bodyNP = self.mesh_drawer_node.attachNewNode(body)
        bodyNP.node().setMass(0)
        bodyNP.setCollideMask(BitMask32.allOn())
        self.world.attachRigidBody(bodyNP.node())
        #self.world.doPhysics(0.01)
         
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
            print("One or both nodes do not have a BulletRigidBodyNode attached.")
            return False

        # Get Bullet shapes
        bullet_node_a = body_a.node()
        bullet_node_b = body_b.node()

        # Use BulletWorld contactTestPair
        result = self.world.contactTestPair(bullet_node_a, bullet_node_b)
        print("Collision check result:")
        print(result.getNumContacts())
        return result.getNumContacts() > 0

    

    def upAndDown(self):
        
        #time = task.time
        #surface.setZ(0+sin(time)*3)
        if base.mouseWatcherNode.hasMouse():
            x = base.mouseWatcherNode.getMouseX()
            y = base.mouseWatcherNode.getMouseY()
            #print(x,y)
            #surface.set_shader_input("pos", Vec3(base.mouseWatcherNode.getMouseX(),0,base.mouseWatcherNode.getMouseY())*4)
            #pFrom = Point3(0, 0, 0)
            #pTo = Point3(10, 0, 0)

            # Get to and from pos in camera coordinates
            pMouse = base.mouseWatcherNode.getMouse()
            pFrom = Point3()
            pTo = Point3()
            base.camLens.extrude(pMouse, pFrom, pTo)

            # Transform to global coordinates
            pFrom = render.getRelativePoint(base.cam, pFrom)
            pTo = render.getRelativePoint(base.cam, pTo)

            result = self.world.rayTestClosest(pFrom, pTo)

            print(result.hasHit())
            print(result.getHitPos())
            print(result.getHitNormal())
            print(result.getHitFraction())
            print(result.getNode())
            #surface.set_shader_input("pos", result.getHitPos())

            self.smiley.setPos(result.getHitPos() + Vec3(0,0,2))
            #self.move_node_smoothly(self.smiley, result.getHitPos() + Vec3(0,0,0.1), duration=0.5)
            dist = (self.smiley.getPos() - self.smiley_copy.getPos()).length()
            print(f"Distance between smilies: {dist}")

            
            groundSizeboundingbox=self.ground.getTightBounds()
            print(groundSizeboundingbox)
            self.ground.set_shader_input("pos", result.getHitPos()/abs(groundSizeboundingbox[0][0]))
            print(self.smiley.getTightBounds())
            unitWidth=abs(self.smiley.getTightBounds()[1][0]-self.smiley.getTightBounds()[0][0])
            unitHeight=abs(self.smiley.getTightBounds()[1][1]-self.smiley.getTightBounds()[0][1])
            print(f"Unit Width: {unitWidth}, Unit Height: {unitHeight}")
            self.ground.set_shader_input("unitSize", Vec3(unitWidth, unitHeight, 0))
            """ 
            #self.draw_circle(center=Point3(0, 0, 5), radius=10, segments=64, color=(1, 1, 1, 1))
            self.draw_arc(center=Point3(0,0, 0), radius=self.unitHeight/2, remainingmove=5, start_angle=0, end_angle=45, segments=64, color=(1, 1, 1, 1))
            #self.mesh_drawer_node.reparentTo(self.smiley)
            print(self.smiley.ls())
            self.mesh_drawer_node.setPos(Vec3(-self.unitWidth/4, -self.unitHeight/4, 0))
            self.mesh_drawer_node.setZ(0)
            self.mesh_drawer_node.setHpr(90,0,0)
            collision = self.check_bullet_collision(self.mesh_drawer_node, self.smiley_copy)
            #self.mesh_drawer_node.hide() 
            """
            return
    

    def pointArc(self,origo, num_points=40, mouse_pos=None,rotationangle=0):
        width=0.5

        points =[]
        #origo   = Vec2(0.55,0.55)
        points.append(origo)

        arcmax = math.pi/2

        rotationangle= 0
        midpoint=Vec2(width*math.cos(math.radians(rotationangle) ), width*math.sin(math.radians(rotationangle) ))*0.5+origo
        midpoint_unit_vector = midpoint - origo
        midpoint_mouse_vector = mouse_pos - midpoint if mouse_pos else Vec2(1,1)
        print(midpoint_unit_vector,midpoint_mouse_vector,midpoint_unit_vector.dot(midpoint_mouse_vector))
        print("before mouse pos:", mouse_pos, "midpoint:", midpoint)
        filipped=False
        if midpoint_unit_vector.dot(midpoint_mouse_vector) > 0:
            #rotationangle += 180
            print("flipped")
            filipped=True
            mouse_pos = self.mirrorPointArc([mouse_pos], mirror_vec=Vec2(0, 10), origin=midpoint)[0]
            print("after mouse pos:", mouse_pos)
            #mouse_pos = self.rotatePoint(mouse_pos, 90, origo=midpoint)

            

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
            if abs(vectormouse.dot(vector)) < .01:
                break
        points.append(points[-1]+Vec2(-math.sin(angle+math.radians(rotationangle)),math.cos(angle+math.radians(rotationangle)))*0.4)
        #points[-1] = self.rotatePoint(points[-1], rotationangle)
        points.append(points[0]+Vec2(-math.sin(angle+math.radians(rotationangle)),math.cos(angle+math.radians(rotationangle)))*0.4)
        #points[-1] = self.rotatePoint(points[-1], rotationangle)

        for i in range(len(points),num_points+3):
            points.append(points[-1])
        
        print(len(points))

        if filipped:
            mirrored_points = self.mirrorPointArc(points, mirror_vec=Vec2(0, 10), origin=midpoint)
            return mirrored_points

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
        print("Mirroring with vector:", mirror_vec)
        for p in points:
            # Vector from origin to point
            print("Original point:", p)
            v = p - origin
            # Project v onto mirror_vec
            print(v.dot(mirror_vec))
            print(v.normalized().dot(mirror_vec))
            proj =  (mirror_vec * v.normalized().dot(mirror_vec)) * v.length()
            # Perpendicular component
            print("v:", v, "proj:", proj)
            perp = v - proj
            print("perp:", perp)
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

    def pathTowardsMouse(self):
        
        #time = task.time
        #surface.setZ(0+sin(time)*3)
        if base.mouseWatcherNode.hasMouse():
            x = base.mouseWatcherNode.getMouseX()
            y = base.mouseWatcherNode.getMouseY()
            #print(x,y)
            #surface.set_shader_input("pos", Vec3(base.mouseWatcherNode.getMouseX(),0,base.mouseWatcherNode.getMouseY())*4)
            #pFrom = Point3(0, 0, 0)
            #pTo = Point3(10, 0, 0)

            # Get to and from pos in camera coordinates
            pMouse = base.mouseWatcherNode.getMouse()
            pFrom = Point3()
            pTo = Point3()
            base.camLens.extrude(pMouse, pFrom, pTo)

            # Transform to global coordinates
            pFrom = render.getRelativePoint(base.cam, pFrom)
            pTo = render.getRelativePoint(base.cam, pTo)

            result = self.world.rayTestClosest(pFrom, pTo)

            print(result.hasHit())
            print(result.getHitPos())
            print(result.getHitNormal())
            print(result.getHitFraction())
            print(result.getNode())
            #surface.set_shader_input("pos", result.getHitPos())

            #self.smiley.setPos(result.getHitPos() + Vec3(0,0,2))
            #self.move_node_smoothly(self.smiley, result.getHitPos() + Vec3(0,0,0.1), duration=0.5)

            groundSizeboundingbox=self.ground.getTightBounds()
            print(groundSizeboundingbox)
            pos=result.getHitPos()/abs(groundSizeboundingbox[0][0])
            self.ground.set_shader_input("pos", pos)
            #self.polygonpoints = []
            pos += Vec3(1, 1, 1)
            pos *= 0.5
            self.polygonpoints.insert(0, Vec2(pos.x, pos.y))
            if len(self.polygonpoints) > 6:
                self.polygonpoints.pop()
            self.polygonpoints = self.pointArc(origo=Vec2(0.25, 0.25), num_points=40, mouse_pos=Vec2(pos.x, pos.y))
            #self.polygonpoints = self.mirrorPointArc(self.polygonpoints)

            self.ground.setShaderInput("polygonpoints", self.polygonpoints)
            return





app = MyApp()
app.run()