from direct.showbase.ShowBase import ShowBase
from panda3d.core import Plane, PlaneNode, Point3, Vec3, BitMask32
from panda3d.core import CardMaker
from panda3d.bullet import BulletWorld, BulletPlaneShape, BulletRigidBodyNode, BulletTriangleMesh, BulletTriangleMeshShape


class MyApp(ShowBase):
    def __init__(self):
        super().__init__()

        # Disable default camera controls
        #self.disableMouse()

        # Create a flat plane using CardMaker
        cm = CardMaker("ground")
        cm.setFrame(-10, 10, -10, 10)  # 20x20 plane
        ground = self.render.attachNewNode(cm.generate())
        ground.setPos(0, 50, 0)
        ground.setHpr(0, 0, 0)
        ground.setColor(0, 1, 0, 1)  # Set plane color to green (RGBA)

        # Load a smiley model and position it above the ground
        self.smiley = self.loader.loadModel('models/smiley')
        self.smiley.reparentTo(self.render)
        self.smiley.setPos(0, 50, 2)
        self.smiley.setScale(2)

        # Position the camera above the plane, looking straight down
        self.camera.setPos(0, -300, 0)
        self.camera.lookAt(ground)
        #self.camera.setP(-90)  # Pitch downwards

        self.setup_bullet()
        self.accept('mouse1', self.upAndDown)


    def setup_bullet(self):
        self.world = BulletWorld()
        shape = BulletPlaneShape(Vec3(0, 0, 1), 1)
        node = BulletRigidBodyNode('Ground')
        node.addShape(shape)
        np = render.attachNewNode(node)
        mesh = BulletTriangleMesh()

        for geomNP in render.findAllMatches('**/+GeomNode'):
            print("fant node")
            geomNode = geomNP.node()
            ts = geomNP.getTransform(np)
            print(ts)
            for geom in geomNode.getGeoms():
                mesh.addGeom(geom, ts=ts)
                print(geom)
        #lol

        worldNP = render.attachNewNode('World')
        body = BulletRigidBodyNode('grid')
        shape = BulletTriangleMeshShape(mesh, False)
        bodyNP = worldNP.attachNewNode(body)
        bodyNP.node().addShape(shape)
        bodyNP.setCollideMask(BitMask32.allOn())
        self.world.attachRigidBody(bodyNP.node())
    
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

        return 

app = MyApp()
app.run()