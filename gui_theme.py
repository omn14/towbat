"""Shared medieval GUI theme constants, helpers and font loader.

Import from here in game.py, tutorial_system.py, ClassRoundCounter.py,
listBuilderGUI.py, etc. to keep a consistent visual style.
"""

from panda3d.core import TextNode, TransparencyAttrib
from direct.gui.OnscreenText import OnscreenText
from direct.gui.OnscreenImage import OnscreenImage
from direct.gui.DirectGui import DirectButton, DirectFrame, DirectLabel, DGG

# ── Colour palette ────────────────────────────────────────────────────────
PARCHMENT       = (0.85, 0.75, 0.55, 0.92)
PARCHMENT_DARK  = (0.55, 0.45, 0.30, 0.95)
GOLD            = (0.90, 0.75, 0.20, 1.0)
INK             = (0.15, 0.10, 0.05, 1.0)
INK_FADED       = (0.35, 0.28, 0.18, 1.0)
RED_WAX         = (0.65, 0.12, 0.12, 1.0)
GREEN_BANNER    = (0.18, 0.38, 0.15, 0.95)
HINT_BG         = (0.12, 0.10, 0.08, 0.80)
HINT_FG         = (0.95, 0.88, 0.65, 1.0)
SHADOW          = (0.05, 0.03, 0.02, 0.7)
CREAM           = (0.92, 0.85, 0.68, 1.0)
DARK_BG         = (0.08, 0.06, 0.04, 0.95)
PANEL_BG        = (0.12, 0.10, 0.08, 0.92)
SEPARATOR       = (0.50, 0.40, 0.20, 0.6)
BTN_TEXT         = (0.90, 0.82, 0.60, 1.0)    # warm aged text on buttons
BTN_GREEN       = (0.20, 0.42, 0.18, 1.0)
BTN_GREEN_HOVER = (0.28, 0.55, 0.24, 1.0)
BTN_RED         = (0.55, 0.12, 0.10, 1.0)
BTN_RED_HOVER   = (0.70, 0.18, 0.14, 1.0)
BTN_NEUTRAL     = (0.40, 0.32, 0.20, 1.0)
BTN_NEUTRAL_HOVER = (0.52, 0.42, 0.28, 1.0)
ENTRY_BG        = (0.18, 0.15, 0.10, 0.9)
ENTRY_FG        = CREAM

# ── Fonts ─────────────────────────────────────────────────────────────────
# One face for everything, for want of a second one. Panda3D's bundled
# cmss12/cmtt12 are OT1-encoded Computer Modern: '+' renders as a minus and
# '>' as an inverted question mark, which is fatal in a UI full of "4+" saves.
# A body sans has to be added as a TTF asset before the split is worth making.
FONT_PATH = 'fonts/MedievalSharp.ttf'

# ── Texture paths ─────────────────────────────────────────────────────────
TEX_DIR         = 'assets/textures/tutorial/'
TEX_PARCHMENT   = TEX_DIR + 'parchment.png'
TEX_VELLUM      = TEX_DIR + 'vellum.png'
TEX_BORDER      = TEX_DIR + 'border.png'
TEX_BUTTON      = TEX_DIR + 'button.png'
TEX_BUTTON_HOVER = TEX_DIR + 'button_hover.png'
TEX_BUTTON_RED  = TEX_DIR + 'button_red.png'
TEX_VICTORY     = TEX_DIR + 'victory_panel.png'
TEX_COMMAND_BAR = TEX_DIR + 'command_bar.png'
TEX_SLOT        = TEX_DIR + 'slot.png'
TEX_BTN_ROUND   = TEX_DIR + 'button_round.png'
TEX_BTN_ROUND_HOVER = TEX_DIR + 'button_round_hover.png'

# ── Cached font reference (loaded lazily, after ShowBase exists) ─────────
_med_font = None


def _smooth(font, pixels_per_unit=128):
    """Mipmapped, anisotropic filtering so text stays crisp when scaled."""
    from panda3d.core import SamplerState
    font.setPixelsPerUnit(pixels_per_unit)
    font.setMinfilter(SamplerState.FT_linear_mipmap_linear)
    font.setMagfilter(SamplerState.FT_linear)
    font.setAnisotropicDegree(4)
    return font


def load_medieval_font():
    """Load and cache the MedievalSharp font.  Call once *after* ShowBase init."""
    global _med_font
    _med_font = loader.loadFont(FONT_PATH)   # noqa: F821 (panda3d global)
    _med_font.setPageSize(1024, 1024)
    return _smooth(_med_font, 256)


def get_font():
    """Return the cached font, loading it if necessary."""
    if _med_font is None:
        return load_medieval_font()
    return _med_font


# ── Helper: textured frame ───────────────────────────────────────────────
def tex_frame(image_path, parent=None, pos=(0, 0, 0), scale=(1, 1, 1)):
    """OnscreenImage used as a textured background panel."""
    img = OnscreenImage(image=image_path, pos=pos, scale=scale, parent=parent)
    img.setTransparency(TransparencyAttrib.MAlpha)
    return img


# ── Helper: textured button ──────────────────────────────────────────────
def tex_button(text, pos, command, parent=None,
               normal=None, hover=None,
               scale=0.06, pad=(0.5, 0.25), font=None):
    """DirectButton with texture backgrounds."""
    if normal is None:
        normal = TEX_BUTTON
    if hover is None:
        hover = TEX_BUTTON_HOVER
    if font is None:
        font = get_font()
    btn = DirectButton(
        text=text,
        text_font=font,
        text_fg=BTN_TEXT,
        scale=scale,
        pos=pos,
        command=command,
        parent=parent,
        frameTexture=normal,
        frameColor=(1, 1, 1, 1),
        pad=pad,
        relief=DGG.FLAT,
    )
    btn.setTransparency(TransparencyAttrib.MAlpha)
    return btn


# ── Helper: styled OnscreenText (HUD) ───────────────────────────────────
def styled_text(text="", pos=(0, 0), scale=0.05, fg=CREAM,
                align=TextNode.ALeft, shadow=SHADOW,
                parent=None, mayChange=True, wordwrap=None, font=None):
    """Create an OnscreenText node with the medieval font and colours."""
    if font is None:
        font = get_font()
    kwargs = dict(
        text=text, pos=pos, scale=scale, fg=fg,
        align=align, shadow=shadow, font=font,
        mayChange=mayChange,
    )
    if parent is not None:
        kwargs['parent'] = parent
    if wordwrap is not None:
        kwargs['wordwrap'] = wordwrap
    return OnscreenText(**kwargs)


# ── Helper: styled DirectFrame panel ────────────────────────────────────
def styled_panel(frameSize, pos=(0, 0, 0), parent=None, texture=None, color=None):
    """Create a DirectFrame panel, optionally with a texture."""
    if color is None:
        color = (1, 1, 1, 1) if texture else PANEL_BG
    frame = DirectFrame(
        frameColor=color,
        frameSize=frameSize,
        pos=pos,
        parent=parent,
    )
    if texture:
        frame['frameTexture'] = texture
    frame.setTransparency(TransparencyAttrib.MAlpha)
    return frame


# ── Helper: flat-colour button (for list builder etc.) ───────────────────
def flat_button(text, pos, command, parent=None,
                bg=BTN_NEUTRAL, hover_bg=BTN_NEUTRAL_HOVER,
                scale=0.06, pad=(0.5, 0.25), font=None, text_fg=CREAM):
    """DirectButton with flat medieval-coloured backgrounds."""
    if font is None:
        font = get_font()
    btn = DirectButton(
        text=text,
        text_font=font,
        text_fg=text_fg,
        scale=scale,
        pos=pos,
        command=command,
        parent=parent,
        frameColor=bg,
        pad=pad,
        relief=DGG.FLAT,
    )
    # Simple hover colour swap
    btn.bind(DGG.WITHIN, lambda *_: btn.setColor(*hover_bg))
    btn.bind(DGG.WITHOUT, lambda *_: btn.setColor(1, 1, 1, 1))
    return btn
