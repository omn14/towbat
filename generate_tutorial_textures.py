"""Generate medieval-themed textures for the tutorial UI.

Run once to produce PNG files in assets/textures/tutorial/.
"""

import os
import random
import math
from PIL import Image, ImageDraw, ImageFilter

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'assets', 'textures', 'tutorial')
os.makedirs(OUT_DIR, exist_ok=True)


def _noise(w, h, base_color, variance=18, seed=42):
    """Return an Image with per-pixel noise around *base_color*."""
    rng = random.Random(seed)
    img = Image.new('RGBA', (w, h))
    pixels = img.load()
    for y in range(h):
        for x in range(w):
            r = max(0, min(255, base_color[0] + rng.randint(-variance, variance)))
            g = max(0, min(255, base_color[1] + rng.randint(-variance, variance)))
            b = max(0, min(255, base_color[2] + rng.randint(-variance, variance)))
            pixels[x, y] = (r, g, b, base_color[3] if len(base_color) > 3 else 255)
    return img


def _stain(img, n=5, seed=99):
    """Add random translucent stain blobs to simulate aged parchment."""
    rng = random.Random(seed)
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = img.size
    for _ in range(n):
        cx = rng.randint(0, w)
        cy = rng.randint(0, h)
        rx = rng.randint(20, w // 3)
        ry = rng.randint(20, h // 3)
        alpha = rng.randint(8, 25)
        color = rng.choice([
            (120, 90, 50, alpha),
            (100, 70, 30, alpha),
            (80, 60, 40, alpha),
        ])
        draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=color)
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=12))
    return Image.alpha_composite(img, overlay)


def _edge_darken(img, border=16, strength=40):
    """Darken edges to create a vignette / burned edge look."""
    w, h = img.size
    pixels = img.load()
    for y in range(h):
        for x in range(w):
            dx = min(x, w - 1 - x)
            dy = min(y, h - 1 - y)
            d = min(dx, dy)
            if d < border:
                factor = 1.0 - (strength / 255.0) * (1.0 - d / border)
                r, g, b, a = pixels[x, y]
                pixels[x, y] = (int(r * factor), int(g * factor), int(b * factor), a)
    return img


def _ornament_line(draw, y, w, color, thickness=2):
    """Draw a horizontal ornamental line with small diamond accents."""
    draw.line([(0, y), (w, y)], fill=color, width=thickness)
    spacing = 40
    diamond_size = 4
    for cx in range(spacing, w, spacing):
        pts = [(cx, y - diamond_size), (cx + diamond_size, y),
               (cx, y + diamond_size), (cx - diamond_size, y)]
        draw.polygon(pts, fill=color)


# ── 1. Parchment (objectives panel background) ──────────────────────

def make_parchment(w=512, h=512):
    img = _noise(w, h, (215, 195, 155, 240), variance=12, seed=42)
    img = _stain(img, n=6, seed=77)
    img = _edge_darken(img, border=24, strength=35)
    img = img.filter(ImageFilter.GaussianBlur(radius=1))
    draw = ImageDraw.Draw(img)
    # Top & bottom ornamental lines
    _ornament_line(draw, 6, w, (150, 120, 70, 200), thickness=2)
    _ornament_line(draw, h - 7, w, (150, 120, 70, 200), thickness=2)
    path = os.path.join(OUT_DIR, 'parchment.png')
    img.save(path)
    print(f'  Saved {path}')
    return path


# ── 2. Dark vellum (hint scroll background) ─────────────────────────

def make_vellum(w=512, h=128):
    img = _noise(w, h, (35, 30, 25, 210), variance=8, seed=101)
    img = _stain(img, n=3, seed=202)
    img = _edge_darken(img, border=12, strength=20)
    img = img.filter(ImageFilter.GaussianBlur(radius=1))
    draw = ImageDraw.Draw(img)
    # Gold-ish trim lines
    gold = (200, 170, 60, 180)
    draw.line([(0, 2), (w, 2)], fill=gold, width=2)
    draw.line([(0, h - 3), (w, h - 3)], fill=gold, width=2)
    path = os.path.join(OUT_DIR, 'vellum.png')
    img.save(path)
    print(f'  Saved {path}')
    return path


# ── 3. Ornamental border strip (horizontal trim) ────────────────────

def make_border(w=512, h=32):
    img = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # Base bar
    draw.rectangle([0, 8, w, h - 8], fill=(140, 105, 55, 230))
    # Inner highlight
    draw.rectangle([0, 10, w, h - 10], fill=(180, 150, 80, 220))
    # Diamond pattern
    gold = (210, 180, 60, 255)
    dark = (100, 75, 35, 255)
    spacing = 24
    for cx in range(spacing // 2, w, spacing):
        cy = h // 2
        s = 5
        pts = [(cx, cy - s), (cx + s, cy), (cx, cy + s), (cx - s, cy)]
        draw.polygon(pts, fill=gold, outline=dark)
    # Edge lines
    draw.line([(0, 8), (w, 8)], fill=dark, width=1)
    draw.line([(0, h - 9), (w, h - 9)], fill=dark, width=1)
    path = os.path.join(OUT_DIR, 'border.png')
    img.save(path)
    print(f'  Saved {path}')
    return path


# ── 4. Button texture ───────────────────────────────────────────────

def make_button(w=256, h=64):
    img = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # Rounded rectangle body
    r = 8
    draw.rounded_rectangle([2, 2, w - 3, h - 3], radius=r,
                           fill=(55, 95, 45, 240),
                           outline=(180, 150, 60, 255), width=2)
    # Inner subtle highlight
    draw.rounded_rectangle([4, 4, w - 5, h // 2], radius=r,
                           fill=(70, 120, 55, 60))
    path = os.path.join(OUT_DIR, 'button.png')
    img.save(path)
    print(f'  Saved {path}')
    return path


def make_button_hover(w=256, h=64):
    img = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    r = 8
    draw.rounded_rectangle([2, 2, w - 3, h - 3], radius=r,
                           fill=(70, 120, 55, 245),
                           outline=(220, 190, 70, 255), width=2)
    draw.rounded_rectangle([4, 4, w - 5, h // 2], radius=r,
                           fill=(90, 145, 70, 70))
    path = os.path.join(OUT_DIR, 'button_hover.png')
    img.save(path)
    print(f'  Saved {path}')
    return path


def make_button_red(w=256, h=64):
    img = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    r = 8
    draw.rounded_rectangle([2, 2, w - 3, h - 3], radius=r,
                           fill=(140, 35, 30, 240),
                           outline=(180, 150, 60, 255), width=2)
    draw.rounded_rectangle([4, 4, w - 5, h // 2], radius=r,
                           fill=(170, 55, 45, 60))
    path = os.path.join(OUT_DIR, 'button_red.png')
    img.save(path)
    print(f'  Saved {path}')
    return path


# ── 5. Victory panel background ─────────────────────────────────────

def make_victory_panel(w=512, h=400):
    img = _noise(w, h, (205, 185, 145, 245), variance=10, seed=303)
    img = _stain(img, n=4, seed=404)
    img = _edge_darken(img, border=30, strength=45)
    img = img.filter(ImageFilter.GaussianBlur(radius=1))
    draw = ImageDraw.Draw(img)
    gold = (180, 150, 60, 220)
    dark = (100, 75, 35, 200)
    # Top ornamental border
    _ornament_line(draw, 8, w, gold)
    _ornament_line(draw, 14, w, dark, thickness=1)
    # Bottom ornamental border
    _ornament_line(draw, h - 9, w, gold)
    _ornament_line(draw, h - 15, w, dark, thickness=1)
    # Side lines
    draw.line([(6, 8), (6, h - 9)], fill=gold, width=2)
    draw.line([(w - 7, 8), (w - 7, h - 9)], fill=gold, width=2)
    path = os.path.join(OUT_DIR, 'victory_panel.png')
    img.save(path)
    print(f'  Saved {path}')
    return path


# ── Generate all ─────────────────────────────────────────────────────

if __name__ == '__main__':
    print('Generating tutorial textures...')
    make_parchment()
    make_vellum()
    make_border()
    make_button()
    make_button_hover()
    make_button_red()
    make_victory_panel()
    print('Done!')
