"""Catalogue-footprint display bases, in the engine's inch-sized world units."""

from functools import lru_cache
import random

from panda3d.core import (Geom, GeomNode, GeomTriangles, GeomVertexData,
                         GeomVertexFormat, GeomVertexReader, GeomVertexWriter, Material,
                         NodePath, PNMImage, Texture, Vec3)

BASE_HEIGHT = 2.8 / 25.4
BEVEL_INSET = 1.1 / 25.4
BAKED_PLINTH_ASSETS = frozenset({
    'jade_warrior', 'jade_lancer', 'bret_bowmen', 'bret_knight',
    'goblin_archers', 'goblin_wolfriders', 'black_knights', 'zombies', 'dire_wolves',
})


@lru_cache(maxsize=64)
def _without_plinth(geometry, transform):
    """Shipped BAM plinths and the generated cart end at z=.07 world units."""
    reader = GeomVertexReader(geometry.getVertexData(), 'vertex')
    matrix = transform.getMat()
    heights = [matrix.xformPoint(reader.getData3()).z
               for _ in range(geometry.getVertexData().getNumRows())]
    retained = GeomTriangles(Geom.UHStatic)
    for primitive in geometry.getPrimitives():
        triangles = primitive.decompose()
        for index in range(triangles.getNumPrimitives()):
            start = triangles.getPrimitiveStart(index)
            indices = [triangles.getVertex(start + corner) for corner in range(3)]
            if max(heights[vertex] for vertex in indices) > .07002:
                retained.addVertices(*indices)
    result = geometry.makeCopy()
    result.clearPrimitives()
    result.addPrimitive(retained)
    return result


@lru_cache(maxsize=1)
def ground_texture():
    rng = random.Random('miniature-base-earth')
    image = PNMImage(128, 128, 3)
    for row in range(128):
        for column in range(128):
            grain = rng.uniform(-.075, .075)
            stone = rng.random() < .055
            moss = rng.random() < .16
            color = ((.42, .43, .37) if stone else (.23, .28, .13) if moss else (.30, .28, .23))
            image.setXel(column, row, *(max(0, min(1, channel + grain)) for channel in color))
    texture = Texture('miniature-base-earth')
    texture.load(image)
    texture.setWrapU(Texture.WMRepeat)
    texture.setWrapV(Texture.WMRepeat)
    texture.setMinfilter(Texture.FTLinearMipmapLinear)
    texture.setMagfilter(Texture.FTLinear)
    texture.setAnisotropicDegree(4)
    return texture


def _surface(name, faces, color):
    data = GeomVertexData(name, GeomVertexFormat.getV3n3c4t2(), Geom.UHStatic)
    vertices = GeomVertexWriter(data, 'vertex')
    normals = GeomVertexWriter(data, 'normal')
    colors = GeomVertexWriter(data, 'color')
    texcoords = GeomVertexWriter(data, 'texcoord')
    triangles = GeomTriangles(Geom.UHStatic)
    for face in faces:
        start = vertices.getWriteRow()
        normal = (Vec3(*face[1]) - Vec3(*face[0])).cross(Vec3(*face[2]) - Vec3(*face[0]))
        normal.normalize()
        for position in face:
            vertices.addData3(*position)
            normals.addData3(normal)
            colors.addData4(*color)
            texcoords.addData2(position[0], position[1])
        triangles.addVertices(start, start + 1, start + 2)
        triangles.addVertices(start, start + 2, start + 3)
    geom = Geom(data)
    geom.addPrimitive(triangles)
    node = GeomNode(name)
    node.addGeom(geom)
    return node


@lru_cache(maxsize=128)
def _base_template(width, depth):
    if width <= 0 or depth <= 0:
        raise ValueError('Miniature base dimensions must be positive')
    inset = min(BEVEL_INSET, min(width, depth) * .08)
    edge = min(.45 / 25.4, inset * .4)

    def rectangle(shrink, height):
        half_width, half_depth = width / 2 - shrink, depth / 2 - shrink
        return [(-half_width, -half_depth, height), (half_width, -half_depth, height),
                (half_width, half_depth, height), (-half_width, half_depth, height)]

    bottom = rectangle(0, 0)
    lip = rectangle(0, .45 / 25.4)
    shoulder = rectangle(inset, BASE_HEIGHT)
    inner = rectangle(inset + edge, BASE_HEIGHT)
    rim_faces = [list(reversed(bottom))]
    for lower, upper in ((bottom, lip), (lip, shoulder), (shoulder, inner)):
        for index in range(4):
            following = (index + 1) % 4
            rim_faces.append([lower[index], lower[following], upper[following], upper[index]])
    root = NodePath('bevelled-base')
    root.setColorOff(1)
    root.setColorScale(1, 1, 1, 1, 1)
    rim = root.attachNewNode(_surface('base-rim', rim_faces, (.115, .125, .12, 1)))
    rim.setTextureOff(1)
    plastic = Material('charcoal-plastic')
    plastic.setDiffuse((.115, .125, .12, 1))
    plastic.setAmbient((.115, .125, .12, 1))
    plastic.setSpecular((.22, .22, .22, 1))
    plastic.setShininess(24)
    rim.setMaterial(plastic, 1)
    top = root.attachNewNode(_surface('base-ground', [inner], (1, 1, 1, 1)))
    earth = Material('matte-earth')
    earth.setDiffuse((1, 1, 1, 1))
    earth.setAmbient((.8, .8, .8, 1))
    earth.setSpecular((0, 0, 0, 1))
    top.setMaterial(earth, 1)
    top.setTexture(ground_texture(), 2)
    root.setPythonTag('base_dimensions', (width, depth))
    return root


def make_base(width, depth):
    """Independent base nodes sharing immutable geometry and a tiled earth texture."""
    return _base_template(float(width), float(depth)).copyTo(NodePath())


def base_figure(figure, width, depth, *, baked_plinth=False):
    """Wrap one source miniature, retaining one direct child per casualty slot."""
    parent = figure.getParent()
    wrapper = parent.attachNewNode(figure.getName() + '-based')
    figure.reparentTo(wrapper)
    figure.setPos(0, 0, 0)
    if baked_plinth:
        for mesh in figure.findAllMatches('**/+GeomNode'):
            node = mesh.node()
            transform = mesh.getTransform(wrapper)
            for index in range(node.getNumGeoms()):
                node.setGeom(index, _without_plinth(node.getGeom(index), transform))
    lower, upper = figure.getTightBounds(wrapper)
    size = upper - lower
    inset = min(BEVEL_INSET, min(width, depth) * .08)
    fit = min(1, (width - 2 * inset) / max(size.x, .001),
              (depth - 2 * inset) / max(size.y, .001))
    figure.setScale(figure.getScale() * fit)
    lower, upper = figure.getTightBounds(wrapper)
    figure.setPos(-(lower.x + upper.x) / 2, -(lower.y + upper.y) / 2,
                  BASE_HEIGHT - (.07 * fit if baked_plinth else lower.z))
    make_base(width, depth).reparentTo(wrapper)
    return wrapper