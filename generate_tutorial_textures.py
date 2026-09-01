"""Generate medieval-themed textures for the tutorial UI.

v2 -- much grittier: heavy stains, ink splatters, cracked edges,
embossed 3-D bevels, fibrous grain, and burnt corners.

Run once to produce PNG files in assets/textures/tutorial/.
"""

import os
import random
import math
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'assets', 'textures', 'tutorial')
os.makedirs(OUT_DIR, exist_ok=True)


# ── Low-level helpers ────────────────────────────────────────────────

def _noise(w, h, base_color, variance=18, seed=42):
    """Per-pixel colour noise around *base_color*."""
    rng = random.Random(seed)
    img = Image.new('RGBA', (w, h))
    px = img.load()
    for y in range(h):
        for x in range(w):
            r = max(0, min(255, base_color[0] + rng.randint(-variance, variance)))
            g = max(0, min(255, base_color[1] + rng.randint(-variance, variance)))
            b = max(0, min(255, base_color[2] + rng.randint(-variance, variance)))
            px[x, y] = (r, g, b, base_color[3] if len(base_color) > 3 else 255)
    return img


def _fiber_grain(img, seed=55, density=0.12, alpha=15):
    """Add subtle horizontal fiber streaks (like real parchment grain)."""
    rng = random.Random(seed)
    w, h = img.size
    overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for _ in range(int(h * density)):
        y = rng.randint(0, h - 1)
        x0 = rng.randint(0, w // 3)
        x1 = rng.randint(w // 2, w)
        c = rng.choice([(80, 60, 30, alpha), (60, 45, 20, alpha),
                        (100, 80, 50, alpha)])
        draw.line([(x0, y), (x1, y)], fill=c, width=1)
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=0.8))
    return Image.alpha_composite(img, overlay)


def _stain(img, n=8, seed=99, min_alpha=10, max_alpha=35):
    """Heavy overlapping stain blobs -- coffee rings, water damage, age spots."""
    rng = random.Random(seed)
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = img.size
    for _ in range(n):
        cx, cy = rng.randint(0, w), rng.randint(0, h)
        rx = rng.randint(15, w // 2)
        ry = rng.randint(15, h // 2)
        alpha = rng.randint(min_alpha, max_alpha)
        color = rng.choice([
            (110, 80, 40, alpha), (90, 60, 25, alpha),
            (70, 50, 30, alpha),  (130, 95, 50, alpha),
            (50, 35, 20, alpha),
        ])
        draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=color)
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=10))
    return Image.alpha_composite(img, overlay)


def _ink_splatter(img, n=12, seed=123):
    """Tiny dark ink dots and splatters."""
    rng = random.Random(seed)
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = img.size
    for _ in range(n):
        cx, cy = rng.randint(0, w), rng.randint(0, h)
        r = rng.randint(1, 5)
        alpha = rng.randint(20, 60)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                     fill=(30, 20, 10, alpha))
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=0.5))
    return Image.alpha_composite(img, overlay)


def _edge_darken(img, border=20, strength=55):
    """Heavy vignette -- burnt / charred edge effect."""
    w, h = img.size
    px = img.load()
    for y in range(h):
        for x in range(w):
            dx = min(x, w - 1 - x)
            dy = min(y, h - 1 - y)
            d = min(dx, dy)
            if d < border:
                factor = 1.0 - (strength / 255.0) * (1.0 - d / border)
                r, g, b, a = px[x, y]
                px[x, y] = (int(r * factor), int(g * factor),
                            int(b * factor), a)
    return img


def _corner_burn(img, radius=60, strength=70, seed=200):
    """Extra darkening in corners (fire damage)."""
    rng = random.Random(seed)
    w, h = img.size
    overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    corners = [(0, 0), (w, 0), (0, h), (w, h)]
    for cx, cy in corners:
        r = radius + rng.randint(-10, 20)
        a = strength + rng.randint(-10, 10)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                     fill=(20, 12, 5, a))
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=radius * 0.6))
    return Image.alpha_composite(img, overlay)


def _bevel(img, light=35, dark=50, width=4):
    """3-D raised bevel: top/left lighter, bottom/right darker."""
    w, h = img.size
    px = img.load()
    for i in range(width):
        t = 1.0 - i / width  # taper
        l_add = int(light * t)
        d_sub = int(dark * t)
        # Top edge
        for x in range(i, w - i):
            r, g, b, a = px[x, i]
            if a > 0:
                px[x, i] = (min(255, r + l_add), min(255, g + l_add),
                            min(255, b + l_add), a)
        # Left edge
        for y in range(i, h - i):
            r, g, b, a = px[i, y]
            if a > 0:
                px[i, y] = (min(255, r + l_add), min(255, g + l_add),
                            min(255, b + l_add), a)
        # Bottom edge
        for x in range(i, w - i):
            r, g, b, a = px[x, h - 1 - i]
            if a > 0:
                px[x, h - 1 - i] = (max(0, r - d_sub), max(0, g - d_sub),
                                    max(0, b - d_sub), a)
        # Right edge
        for y in range(i, h - i):
            r, g, b, a = px[w - 1 - i, y]
            if a > 0:
                px[w - 1 - i, y] = (max(0, r - d_sub), max(0, g - d_sub),
                                    max(0, b - d_sub), a)
    return img


def _ornament_line(draw, y, w, color, thickness=2):
    """Horizontal ornamental line with diamond accents."""
    draw.line([(0, y), (w, y)], fill=color, width=thickness)
    spacing = 32
    ds = 4
    for cx in range(spacing // 2, w, spacing):
        pts = [(cx, y - ds), (cx + ds, y), (cx, y + ds), (cx - ds, y)]
        draw.polygon(pts, fill=color)


def _embossed_border(draw, w, h, inset=6, color_light=(220, 200, 140, 200),
                     color_dark=(80, 60, 30, 200), thickness=2):
    """Draw an inner embossed rectangular border for a framed look."""
    # Outer dark
    draw.rectangle([inset, inset, w - inset - 1, h - inset - 1],
                   outline=color_dark, width=thickness)
    # Inner light (offset gives 3-D ridge)
    i2 = inset + thickness + 1
    draw.rectangle([i2, i2, w - i2 - 1, h - i2 - 1],
                   outline=color_light, width=1)


# ── 1. Parchment (objectives / main panels) ─────────────────────────

def make_parchment(w=512, h=512):
    img = _noise(w, h, (210, 188, 148, 240), variance=15, seed=42)
    img = _fiber_grain(img, seed=55, density=0.15, alpha=18)
    img = _stain(img, n=10, seed=77, min_alpha=12, max_alpha=40)
    img = _ink_splatter(img, n=15, seed=111)
    img = _edge_darken(img, border=30, strength=55)
    img = _corner_burn(img, radius=80, strength=60)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.8))
    img = _bevel(img, light=40, dark=55, width=5)
    draw = ImageDraw.Draw(img)
    _embossed_border(draw, w, h, inset=8,
                     color_light=(190, 165, 100, 180),
                     color_dark=(90, 65, 30, 200))
    _ornament_line(draw, 18, w, (145, 115, 60, 200), thickness=2)
    _ornament_line(draw, h - 19, w, (145, 115, 60, 200), thickness=2)
    path = os.path.join(OUT_DIR, 'parchment.png')
    img.save(path)
    print(f'  Saved {path}')
    return path


# ── 2. Dark vellum (hint scroll) ────────────────────────────────────

def make_vellum(w=512, h=160):
    img = _noise(w, h, (32, 26, 20, 215), variance=10, seed=101)
    img = _fiber_grain(img, seed=66, density=0.1, alpha=12)
    img = _stain(img, n=5, seed=202, min_alpha=8, max_alpha=25)
    img = _ink_splatter(img, n=8, seed=333)
    img = _edge_darken(img, border=14, strength=30)
    img = _corner_burn(img, radius=40, strength=45, seed=210)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.6))
    img = _bevel(img, light=22, dark=40, width=4)
    draw = ImageDraw.Draw(img)
    gold = (195, 165, 55, 190)
    dark = (80, 60, 25, 160)
    draw.line([(0, 3), (w, 3)], fill=gold, width=2)
    draw.line([(0, 5), (w, 5)], fill=dark, width=1)
    draw.line([(0, h - 4), (w, h - 4)], fill=gold, width=2)
    draw.line([(0, h - 6), (w, h - 6)], fill=dark, width=1)
    path = os.path.join(OUT_DIR, 'vellum.png')
    img.save(path)
    print(f'  Saved {path}')
    return path


# ── 3. Ornamental border strip ──────────────────────────────────────

def make_border(w=512, h=32):
    img = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 6, w, h - 6], fill=(130, 98, 48, 235))
    draw.rectangle([0, 8, w, h - 8], fill=(175, 145, 75, 225))
    # Noise overlay for metallic texture
    tex = _noise(w, h - 16, (170, 140, 70, 40), variance=20, seed=444)
    base_strip = img.crop((0, 8, w, h - 8))
    img.paste(Image.alpha_composite(base_strip, tex), (0, 8))
    draw = ImageDraw.Draw(img)
    gold = (210, 180, 55, 255)
    dark = (90, 65, 30, 255)
    spacing = 20
    for cx in range(spacing // 2, w, spacing):
        cy = h // 2
        s = 5
        pts = [(cx, cy - s), (cx + s, cy), (cx, cy + s), (cx - s, cy)]
        draw.polygon(pts, fill=gold, outline=dark)
    draw.line([(0, 6), (w, 6)], fill=dark, width=1)
    draw.line([(0, h - 7), (w, h - 7)], fill=dark, width=1)
    path = os.path.join(OUT_DIR, 'border.png')
    img.save(path)
    print(f'  Saved {path}')
    return path


# ── 4. Button textures ──────────────────────────────────────────────

def _make_btn(w, h, body_color, outline_color, highlight_color, seed, name):
    """Gritty, beveled button with leather/metal texture."""
    img = _noise(w, h, body_color, variance=12, seed=seed)
    img = _stain(img, n=3, seed=seed + 10, min_alpha=6, max_alpha=20)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.6))
    # Mask to rounded rectangle
    mask = Image.new('L', (w, h), 0)
    md = ImageDraw.Draw(mask)
    r = 8
    md.rounded_rectangle([2, 2, w - 3, h - 3], radius=r, fill=255)
    img.putalpha(mask)
    # Bevel for 3-D
    img = _bevel(img, light=30, dark=45, width=3)
    draw = ImageDraw.Draw(img)
    # Outline
    draw.rounded_rectangle([2, 2, w - 3, h - 3], radius=r,
                           outline=outline_color, width=2)
    # Top highlight gradient
    for yy in range(4, h // 3):
        a = int(highlight_color[3] * (1.0 - yy / (h / 3)))
        draw.line([(4, yy), (w - 5, yy)],
                  fill=(highlight_color[0], highlight_color[1],
                        highlight_color[2], a), width=1)
    path = os.path.join(OUT_DIR, f'{name}.png')
    img.save(path)
    print(f'  Saved {path}')
    return path


def make_button(w=256, h=64):
    return _make_btn(w, h,
                     body_color=(50, 88, 40, 240),
                     outline_color=(170, 140, 50, 255),
                     highlight_color=(120, 160, 100, 50),
                     seed=500, name='button')


def make_button_hover(w=256, h=64):
    return _make_btn(w, h,
                     body_color=(65, 115, 50, 245),
                     outline_color=(215, 185, 65, 255),
                     highlight_color=(145, 190, 120, 60),
                     seed=510, name='button_hover')


def make_button_red(w=256, h=64):
    return _make_btn(w, h,
                     body_color=(130, 30, 25, 240),
                     outline_color=(170, 140, 50, 255),
                     highlight_color=(180, 60, 50, 50),
                     seed=520, name='button_red')


# ── 5. Victory panel background ─────────────────────────────────────

def make_victory_panel(w=512, h=400):
    img = _noise(w, h, (200, 180, 140, 245), variance=14, seed=303)
    img = _fiber_grain(img, seed=77, density=0.12, alpha=16)
    img = _stain(img, n=8, seed=404, min_alpha=14, max_alpha=38)
    img = _ink_splatter(img, n=10, seed=505)
    img = _edge_darken(img, border=35, strength=55)
    img = _corner_burn(img, radius=70, strength=65, seed=606)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.8))
    img = _bevel(img, light=35, dark=50, width=5)
    draw = ImageDraw.Draw(img)
    _embossed_border(draw, w, h, inset=8,
                     color_light=(185, 155, 90, 190),
                     color_dark=(85, 60, 28, 210))
    gold = (175, 145, 55, 220)
    dark = (95, 70, 30, 200)
    _ornament_line(draw, 18, w, gold)
    _ornament_line(draw, 24, w, dark, thickness=1)
    _ornament_line(draw, h - 19, w, gold)
    _ornament_line(draw, h - 25, w, dark, thickness=1)
    draw.line([(10, 18), (10, h - 19)], fill=gold, width=2)
    draw.line([(w - 11, 18), (w - 11, h - 19)], fill=gold, width=2)
    path = os.path.join(OUT_DIR, 'victory_panel.png')
    img.save(path)
    print(f'  Saved {path}')
    return path


# ── 6. Command bar (the bottom HUD frame) ───────────────────────────

def make_command_bar(w=1024, h=256):
    """Dark oak backing for the bottom command bar.

    Stretched across the full width rather than tiled, so the grain is drawn
    the whole way and there is no seam to line up.
    """
    img = _noise(w, h, (46, 33, 21, 255), variance=9, seed=707)
    rng = random.Random(717)
    # Grain goes through a composited overlay: ImageDraw writes RGBA verbatim
    # rather than blending, so drawing these lines straight on would punch
    # translucent stripes through the bar and show the board behind it.
    grain = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    gd = ImageDraw.Draw(grain)
    for _ in range(h):
        y = rng.randint(0, h - 1)
        shade = rng.randint(-20, 15)
        gd.line([(0, y), (w, y)],
                fill=(max(0, 46 + shade), max(0, 33 + shade),
                      max(0, 21 + shade), 45),
                width=rng.choice([1, 1, 2]))
    img = Image.alpha_composite(img, grain)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.7))
    img = _stain(img, n=7, seed=727, min_alpha=10, max_alpha=30)
    img = _corner_burn(img, radius=120, strength=55, seed=737)
    draw = ImageDraw.Draw(img)
    gold = (168, 138, 58, 255)
    dark = (24, 16, 9, 255)
    # Gold beading along the top edge only: the bottom of the bar is the
    # bottom of the screen, so a rule there reads as a gap.
    draw.line([(0, 1), (w, 1)], fill=dark, width=3)
    draw.line([(0, 4), (w, 4)], fill=gold, width=3)
    draw.line([(0, 8), (w, 8)], fill=dark, width=2)
    # The bar is the floor of the screen: it has to hide the board behind it.
    img.putalpha(255)
    path = os.path.join(OUT_DIR, 'command_bar.png')
    img.save(path)
    print(f'  Saved {path}')
    return path


# ── 7. Empty framed slot (portrait, crest, phase icon) ──────────────

def make_slot(w=192, h=192):
    """A recessed empty frame, for art that has not been drawn yet.

    Corner ticks and a faint diagonal cross mark it as a reserved slot rather
    than a panel someone forgot to fill.
    """
    img = _noise(w, h, (34, 26, 17, 240), variance=8, seed=808)
    img = _stain(img, n=3, seed=818, min_alpha=8, max_alpha=22)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.6))
    img = _bevel(img, light=12, dark=44, width=3)
    # Composited, not drawn straight on, for the same reason as the bar grain.
    cross = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    cd = ImageDraw.Draw(cross)
    cd.line([(10, 10), (w - 11, h - 11)], fill=(120, 98, 48, 55), width=1)
    cd.line([(w - 11, 10), (10, h - 11)], fill=(120, 98, 48, 55), width=1)
    img = Image.alpha_composite(img, cross)
    draw = ImageDraw.Draw(img)
    _embossed_border(draw, w, h, inset=3,
                     color_light=(152, 124, 56, 205),
                     color_dark=(70, 52, 24, 225))
    tick, off = 14, 6
    gold = (190, 158, 70, 230)
    for cx, sx in ((off, 1), (w - off - 1, -1)):
        for cy, sy in ((off, 1), (h - off - 1, -1)):
            draw.line([(cx, cy), (cx + sx * tick, cy)], fill=gold, width=2)
            draw.line([(cx, cy), (cx, cy + sy * tick)], fill=gold, width=2)
    img.putalpha(255)
    path = os.path.join(OUT_DIR, 'slot.png')
    img.save(path)
    print(f'  Saved {path}')
    return path


# ── 8. Round button (End Turn) ──────────────────────────────────────

def _make_round_btn(d, body_color, rim_color, highlight, seed, name):
    img = _noise(d, d, body_color, variance=12, seed=seed)
    img = _stain(img, n=4, seed=seed + 10, min_alpha=8, max_alpha=24)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.7))

    # Domed highlight over the top of the disc. Composited rather than drawn
    # straight on: ImageDraw writes RGBA verbatim instead of blending, so a
    # translucent fill would punch a hole in the disc instead of lightening it.
    r = d / 2
    glow = Image.new('RGBA', (d, d), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    for yy in range(4, int(r)):
        a = int(highlight[3] * (1.0 - yy / r))
        half = math.sqrt(max(0.0, r * r - (r - yy) ** 2))
        gd.line([(r - half + 3, yy), (r + half - 4, yy)],
                fill=(highlight[0], highlight[1], highlight[2], a), width=1)
    glow = glow.filter(ImageFilter.GaussianBlur(radius=4))
    img = Image.alpha_composite(img, glow)

    mask = Image.new('L', (d, d), 0)
    ImageDraw.Draw(mask).ellipse([2, 2, d - 3, d - 3], fill=255)
    img.putalpha(mask)

    draw = ImageDraw.Draw(img)
    draw.ellipse([2, 2, d - 3, d - 3], outline=rim_color, width=5)
    draw.ellipse([9, 9, d - 10, d - 10], outline=(40, 26, 12, 255), width=2)
    path = os.path.join(OUT_DIR, f'{name}.png')
    img.save(path)
    print(f'  Saved {path}')
    return path


def make_round_button(d=256):
    return _make_round_btn(d, (122, 26, 22, 255), (170, 140, 52, 255),
                           (235, 140, 110, 90), 909, 'button_round')


def make_round_button_hover(d=256):
    return _make_round_btn(d, (158, 36, 28, 255), (215, 182, 72, 255),
                           (255, 175, 140, 110), 919, 'button_round_hover')


# ── Generate all ─────────────────────────────────────────────────────

if __name__ == '__main__':
    print('Generating tutorial textures (v2 -- gritty)...')
    make_parchment()
    make_vellum()
    make_border()
    make_button()
    make_button_hover()
    make_button_red()
    make_victory_panel()
    make_command_bar()
    make_slot()
    make_round_button()
    make_round_button_hover()
    print('Done!')
