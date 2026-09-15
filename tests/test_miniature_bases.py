"""Exact footprints and renderable bevels independent of a full battle scene."""

from pathlib import Path

import pytest
from panda3d.core import ColorAttrib, GeomVertexReader, NodePath, Loader, TextureAttrib

from miniature_bases import BASE_HEIGHT, make_base


@pytest.mark.parametrize('size_mm', [(20, 20), (25, 25), (30, 30), (25, 50),
                                     (30, 60), (40, 60), (50, 50), (50, 100),
                                     (60, 100), (100, 150)])
def test_base_footprint_matches_catalogue_dimensions(size_mm):
    width, depth = (dimension / 25.4 for dimension in size_mm)
    base = make_base(width, depth)
    lower, upper = base.getTightBounds()
    assert tuple(lower) == pytest.approx((-width / 2, -depth / 2, 0))
    assert tuple(upper) == pytest.approx((width / 2, depth / 2, BASE_HEIGHT))
    rim = base.find('**/base-rim').node().getGeom(0)
    normals = GeomVertexReader(rim.getVertexData(), 'normal')
    assert any(0 < normals.getData3().z < 1 for _ in range(rim.getVertexData().getNumRows()))
    ground = base.find('**/base-ground')
    top_lower, top_upper = ground.getTightBounds(base)
    assert top_lower.x > lower.x and top_upper.x < upper.x
    assert top_lower.y > lower.y and top_upper.y < upper.y
    assert ground.hasTexture()
    assert ground.getNetState().getAttrib(TextureAttrib).getNumOnStages() == 1


def test_base_instances_do_not_share_transforms_or_parentage():
    first, second = make_base(1, 2), make_base(1, 2)
    parent = NodePath('unit')
    first.reparentTo(parent)
    first.setPos(3, 4, 5)
    assert tuple(second.getPos()) == (0, 0, 0)
    assert second.getParent() != parent


def test_player_tint_does_not_override_base_finish():
    parent = NodePath('player-colored-figure')
    parent.setColor(1, 0, 0, 1)
    parent.setColorScale(.8, .9, .85, 1)
    base = make_base(1, 1)
    base.reparentTo(parent)
    rim = base.find('**/base-rim')
    assert rim.getNetState().getAttrib(ColorAttrib).getColorType() == ColorAttrib.TVertex
    assert tuple(rim.getMaterial().getDiffuse()) == pytest.approx((.115, .125, .12, 1))


def test_missing_catalogue_size_keeps_one_consistent_artwork_footprint():
    from types import SimpleNamespace
    from unittest.mock import patch
    from panda3d.core import CardMaker
    from units import unitGraphics
    import units

    figures = NodePath('custom-figures')
    for extent in (1, 2):
        card = CardMaker('custom-artwork')
        card.setFrame(-extent, extent, -extent, extent)
        figure = figures.attachNewNode(card.generate())
        figure.setP(-90)
    owner = SimpleNamespace(unit=SimpleNamespace(model=SimpleNamespace(
        name='Unknown profile', get_base_size=lambda: None)))
    loader = SimpleNamespace(loadModel=lambda filename: figures)
    with patch.object(units, 'loader', loader, create=True):
        result = unitGraphics.loadFigureModel(owner, 'custom-model.bam')
    assert len(result.getChildren()) == 2
    for base in result.findAllMatches('**/bevelled-base'):
        assert base.getPythonTag('base_dimensions') == pytest.approx((2, 2))


@pytest.mark.parametrize('profile,mount,asset', [
    ('Elven Spearman', None, 'jade_warrior'),
    ('Noble', None, 'jade_warrior'),
    ('Noble', 'Elven Steed', 'bret_knight'),
    ('Lothern Skycutter', None, 'jade_warrior'),
    ('Chaos Knight', 'Chaos Steed', 'black_knights'),
    ('Baggage Cart', None, 'jade_warrior'),
    ('Elven Spearman', None, 'bret_bowmen'),
    ('Elven Spearman', None, 'goblin_archers'),
    ('Elven Spearman', None, 'zombies'),
    ('Noble', 'Elven Steed', 'jade_lancer'),
    ('Noble', 'Elven Steed', 'goblin_wolfriders'),
    ('Dire Wolf', None, 'dire_wolves'),
])
def test_shared_loader_bases_each_figure_at_resolved_profile_size(profile, mount, asset):
    from types import SimpleNamespace
    from unittest.mock import patch
    from models import model
    from units import unitGraphics
    import units

    fighter = model(profile, '')
    if mount:
        fighter.attach_mount(SimpleNamespace(model=model(mount, '')))
    owner = SimpleNamespace(unit=SimpleNamespace(model=fighter), color=(.8, .2, .15, 1))
    loader = SimpleNamespace(loadModel=lambda filename: NodePath(Loader.getGlobalPtr().loadSync(filename)))
    with patch.object(units, 'loader', loader, create=True):
        figures = unitGraphics.loadFigureModel(owner, Path(__file__).resolve().parents[1] / f'models/{asset}.bam')
    width, depth = (dimension / 25.4 for dimension in fighter.get_base_size())
    for figure in figures.getChildren():
        bases = figure.findAllMatches('**/bevelled-base')
        assert len(bases) == 1
        lower, upper = bases[0].getTightBounds(figure)
        assert tuple(upper - lower) == pytest.approx((width, depth, BASE_HEIGHT))
        body = next(child for child in figure.getChildren() if child.getName() != 'bevelled-base')
        body_lower, body_upper = body.getTightBounds(figure)
        assert body_lower.z <= BASE_HEIGHT + .01
        assert body_upper.z > BASE_HEIGHT
        for mesh in body.findAllMatches('**/+GeomNode'):
            matrix = mesh.getMat(figure)
            for geometry in mesh.node().getGeoms():
                reader = GeomVertexReader(geometry.getVertexData(), 'vertex')
                heights = [matrix.xformPoint(reader.getData3()).z
                           for _ in range(geometry.getVertexData().getNumRows())]
                for primitive in geometry.getPrimitives():
                    triangles = primitive.decompose()
                    for index in range(triangles.getNumPrimitives()):
                        start = triangles.getPrimitiveStart(index)
                        assert max(heights[triangles.getVertex(start + corner)] for corner in range(3)) > BASE_HEIGHT
        assert body_lower.x >= -width / 2 - 1e-5 and body_upper.x <= width / 2 + 1e-5
        assert body_lower.y >= -depth / 2 - 1e-5 and body_upper.y <= depth / 2 + 1e-5


def test_offscreen_based_miniatures_are_visible_and_seated(tmp_path):
    from types import SimpleNamespace
    from direct.showbase.ShowBase import ShowBase
    from panda3d.core import (AmbientLight, CardMaker, DirectionalLight, Filename,
                             OrthographicLens, PNMImage, loadPrcFileData)
    from models import model
    from units import unitGraphics

    loadPrcFileData('', 'window-type offscreen\nwin-size 1280 800\naudio-library-name null\nmultisamples 4\nframebuffer-multisample 1')
    app = ShowBase()
    try:
        app.setBackgroundColor(.19, .215, .225, 1)
        ambient = AmbientLight('ambient')
        ambient.setColor((.48, .48, .48, 1))
        app.render.setLight(app.render.attachNewNode(ambient))
        key = DirectionalLight('key')
        key.setColor((.95, .91, .82, 1))
        key_node = app.render.attachNewNode(key)
        key_node.setHpr(-35, -55, 0)
        app.render.setLight(key_node)
        fill = DirectionalLight('fill')
        fill.setColor((.3, .38, .45, 1))
        fill_node = app.render.attachNewNode(fill)
        fill_node.setHpr(125, -30, 0)
        app.render.setLight(fill_node)
        ground = CardMaker('ground')
        ground.setFrame(-24, 24, -24, 24)
        floor = app.render.attachNewNode(ground.generate())
        floor.setP(-90)
        floor.setZ(-.015)
        floor.setColor(.24, .275, .28, 1)
        samples = [((25, 25), (-5, 0)), ((30, 60), (-2.5, 0)),
                   ((50, 50), (.3, 0)), ((50, 100), (3.8, 0))]
        preview = app.render.attachNewNode('base-lineup')
        for size_mm, position in samples:
            base = make_base(*(dimension / 25.4 for dimension in size_mm))
            base.reparentTo(preview)
            base.setPos(*position, 0)
        for index, (profile, mount, asset) in enumerate([
                ('Elven Spearman', None, 'jade_warrior'),
                ('Noble', 'Elven Steed', 'bret_knight'),
                ('Chaos Knight', 'Chaos Steed', 'black_knights')]):
            fighter = model(profile, '')
            if mount:
                fighter.attach_mount(SimpleNamespace(model=model(mount, '')))
            owner = SimpleNamespace(unit=SimpleNamespace(model=fighter), color=(.8, .2, .15, 1))
            figures = unitGraphics.loadFigureModel(owner, Path(__file__).resolve().parents[1] / f'models/{asset}.bam')
            figures.reparentTo(preview)
            figures.setPos(-4 + index * 4, 5, 0)
            figures.setColor((.58, .17, .12, 1) if index != 1 else (.14, .32, .52, 1))
            for figure in list(figures.getChildren())[1:]:
                figure.removeNode()
        lens = OrthographicLens()
        lens.setFilmSize(15.5, 9.7)
        app.cam.node().setLens(lens)
        app.camera.setPos(10, -15, 14)
        app.camera.lookAt(0, 2.3, .25)

        def snapshot(name):
            app.graphicsEngine.renderFrame()
            app.graphicsEngine.renderFrame()
            image = PNMImage()
            assert app.win.getScreenshot(image)
            colors = {tuple(image.getXel(column, row)) for row in range(0, image.getYSize(), 8)
                      for column in range(0, image.getXSize(), 8)}
            assert len(colors) > 50
            assert image.write(Filename.fromOsSpecific(str(tmp_path / f'bases-{name}.png')))

        snapshot('overview')
        app.camera.setPos(0, 2.3, 24)
        app.camera.lookAt(0, 2.3, 0)
        snapshot('top')
        preview.removeNode()
        fighter = model('Elven Spearman', '')
        owner = SimpleNamespace(unit=SimpleNamespace(model=fighter), color=(.8, .2, .15, 1))
        regiment = unitGraphics.loadFigureModel(owner, Path(__file__).resolve().parents[1] / 'models/jade_warrior.bam')
        regiment.reparentTo(app.render)
        regiment.setColor(.55, .16, .12, 1)
        children = list(regiment.getChildren())
        for index in range(15 - len(children)):
            children.append(children[index % len(children)].copyTo(regiment))
        width, depth = (dimension / 25.4 for dimension in fighter.get_base_size())
        for index, child in enumerate(children):
            child.setPos((index % 5 - 2) * width, (index // 5 - 1) * depth, 0)
        assert len(regiment.getChildren()) == len(regiment.findAllMatches('**/bevelled-base')) == 15
        lens.setFilmSize(8, 5)
        app.camera.setPos(7, -12, 10)
        app.camera.lookAt(0, 0, .55)
        snapshot('regiment')
        print(f'Base preview screenshots: {tmp_path}')
    finally:
        app.destroy()