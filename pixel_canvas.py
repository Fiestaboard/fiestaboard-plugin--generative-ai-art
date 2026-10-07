"""Full-colour pixel art for LED pixel-matrix displays (the ``canvas`` variable).

FiestaBoard pages can hold *pixel canvases* (core PR #2228 and follow-ups):
drawings that a pixel-matrix board such as a Divoom Pixoo 64 paints at full
RGB, beside or under the page's text. A plugin feeds one through a variable
whose manifest metadata says ``"format": "canvas"``; its value is a *content
object*::

    {"size": [64, 64], "background": "#140c2a",
     "palette": {"a": "#ffffff"},             # only when "pixels" is present
     "shapes": [{"type": "gradient", ...}, {"type": "circle", ...}, ...],
     "pixels": ["....a...", ...]}             # optional, drawn last

This module owns everything about that object for this plugin:

- :func:`pixel_target` decides whether the bound board is a pixel display and
  at what size to draw (the display's pixel size when the core reports it,
  else 64 x 64). It reads ``self.board.display`` defensively so it works on
  cores with or without the ``pixels`` display feature.
- :func:`canvas_system_prompt` asks the model for a scene in the canvas
  vocabulary with any ``#rrggbb`` colour, in pixel coordinates.
- :func:`scene_to_content` turns whatever the model returned -- canvas shapes,
  or this plugin's older scene primitives (stripes, rings, wave, noise,
  triangle, mirror, normalised 0..1 coordinates, colour letters) -- into a
  content object that is clamped to core's limits. It never emits anything
  core would refuse: a shape that cannot be repaired is dropped.
- :func:`tiles_content` wraps a split-flap tile piece as a content object, so
  the variable is a valid canvas on every board.
- :func:`sample_grid` derives the board-colour tile art from a canvas, so a
  pixel display costs one model call for both ``canvas`` and ``art``.
- :func:`fallback_content` is the deterministic scene used when generation
  fails and there is no earlier canvas for that display.

Nothing here imports FiestaBoard core except, optionally, core's own canvas
validator, used as a last check when the core has one.
"""

from __future__ import annotations

import json
import logging
import math
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests

try:
    from .source import (
        COLOR_LETTERS,
        MAX_ATTEMPTS,
        MAX_DESCRIPTION_CHARS,
        MAX_THEME_CHARS,
        ArtRequestError,
        ArtValidationError,
        _loads_with_salvage,
        _strip_fences,
        _text,
    )
except ImportError:  # imported as a top-level module (see __init__.py)
    from source import (  # type: ignore[no-redef]
        COLOR_LETTERS,
        MAX_ATTEMPTS,
        MAX_DESCRIPTION_CHARS,
        MAX_THEME_CHARS,
        ArtRequestError,
        ArtValidationError,
        _loads_with_salvage,
        _strip_fences,
        _text,
    )

try:  # Core's own validator, when the core has the canvas engine.
    from src.canvas.models import CanvasContent as _CoreCanvasContent
except Exception:  # noqa: BLE001 -- any older core: our own checks are the contract
    _CoreCanvasContent = None

logger = logging.getLogger(__name__)

# --- Core's limits (src/canvas/models.py) ------------------------------------

MAX_CONTENT_SIZE = 128
MAX_CANVAS_SHAPES = 256
MAX_COORD = 1024
MAX_LINE_WIDTH = 16
MAX_POLYGON_POINTS = 256
MAX_TEXT_LENGTH = 256
MAX_PALETTE_KEYS = 62

# --- This plugin's choices ---------------------------------------------------

#: Drawing size when the display does not report its pixel size (a Pixoo 64).
DEFAULT_CANVAS_SIZE: Tuple[int, int] = (64, 64)
#: Shapes the prompt allows. Enough for real detail; ~40 tokens a shape keeps
#: a full reply inside :data:`CANVAS_MAX_TOKENS`.
MAX_PROMPT_SHAPES = 60
CANVAS_MAX_TOKENS = 3000
#: Expansion bounds for the shortcut primitives.
MAX_BANDS = 32
MAX_RINGS = 24
WAVE_SAMPLES = 32
RADIAL_STEPS = 12
MAX_NOISE_DENSITY = 0.5

#: Board colours as the LED renderer paints them (core ``BOARD_COLORS``).
BOARD_RGB: Dict[str, Tuple[int, int, int]] = {
    "red": (0xEB, 0x40, 0x34),
    "orange": (0xF5, 0xA6, 0x23),
    "yellow": (0xF8, 0xE7, 0x1C),
    "green": (0x7E, 0xD3, 0x21),
    "blue": (0x4A, 0x90, 0xD9),
    "violet": (0x9B, 0x59, 0xB6),
    "white": (0xFF, 0xFF, 0xFF),
    "black": (0x1A, 0x1A, 0x1A),
}
#: Tile letter -> board colour name ("R" -> "red").
LETTER_NAMES: Dict[str, str] = {letter: marker.strip("{}") for letter, marker in COLOR_LETTERS.items()}
_NAME_LETTERS: Dict[str, str] = {name: letter for letter, name in LETTER_NAMES.items()}

_HEX6 = re.compile(r"#[0-9a-fA-F]{6}")
_HEX3 = re.compile(r"#[0-9a-fA-F]{3}")
_PALETTE_KEYS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

_X_FIELDS = ("x", "w", "cx", "rx", "x1", "x2")
_Y_FIELDS = ("y", "h", "cy", "ry", "y1", "y2", "baseline", "amplitude")


# ---------------------------------------------------------------------------
# Which displays, at what size
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PixelTarget:
    """A pixel display to draw for: the content size and whether it is mono."""

    width: int
    height: int
    mono: bool = False

    @property
    def size(self) -> Tuple[int, int]:
        return (self.width, self.height)


def _pixel_size(width: Any, height: Any) -> Tuple[int, int]:
    """The display's pixel size fitted into core's content limit, else the default."""
    ok = all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in (width, height))
    if not ok:
        return DEFAULT_CANVAS_SIZE
    longest = max(width, height)
    if longest <= MAX_CONTENT_SIZE:
        return (width, height)
    factor = MAX_CONTENT_SIZE / longest
    return (max(1, round(width * factor)), max(1, round(height * factor)))


def pixel_target(board: Any) -> Optional[PixelTarget]:
    """The pixel display *board* is, or ``None`` for anything else.

    A display that supports the ``"pixels"`` feature is one (core with pixel
    canvases). On a core whose ``supports()`` does not know that feature yet
    (it raises), an RGB LED matrix is treated as one. Its pixel size comes
    from ``display.width`` / ``display.height`` when present.
    """
    display = getattr(board, "display", None) if board is not None else None
    if display is None:
        return None
    is_pixels = False
    supports = getattr(display, "supports", None)
    if callable(supports):
        try:
            is_pixels = bool(supports("pixels"))
        except Exception:  # noqa: BLE001 -- an older core: unknown feature name
            is_pixels = False
    color = getattr(display, "color", None)
    if not is_pixels:
        is_pixels = getattr(display, "technology", None) == "led_matrix" and color == "rgb"
    if not is_pixels:
        return None
    width, height = _pixel_size(getattr(display, "width", None), getattr(display, "height", None))
    return PixelTarget(width, height, mono=color == "mono")


def canvas_key(board: Any, canvas: Any, target: PixelTarget) -> str:
    """Cache key for a canvas: the board's own key (or grid + display) plus the size."""
    base = getattr(board, "key", None) if board is not None else None
    if not isinstance(base, str) or not base:
        base = f"{canvas.cols}x{canvas.rows}"
        display_key = getattr(getattr(board, "display", None), "key", None)
        if isinstance(display_key, str) and display_key:
            base += f"|{display_key}"
    return f"{base}|{target.width}x{target.height}"


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


def canvas_system_prompt(target: PixelTarget, extra_instructions: str = "") -> str:
    """The system prompt for a full-colour scene on *target*."""
    w, h = target.size
    cx, cy = w // 2, h // 2
    colour_rule = (
        "COLOURS: any \"#rrggbb\" hex colour. Use a rich, deliberate palette: shading, "
        "highlights, glow and depth, not just a handful of flat colours."
    )
    if target.mono:
        colour_rule += (
            "\nThis display is MONOCHROME: it lights a pixel when its colour is bright and leaves it "
            "dark otherwise, so compose with brightness (bright on dark) rather than hue."
        )
    prompt = f"""You are a pixel artist composing for a full-colour LED pixel-matrix display \
of exactly {w} × {h} pixels. Every pixel is one RGB LED. This is not a tile board with a \
fixed palette: draw real pixel art.

You do NOT emit pixels one by one. Describe the picture as a SCENE of shapes and the \
display software rasterises it.

COORDINATES are pixels: x = 0 is the left edge and x = {w} the right edge; y = 0 is the \
top and y = {h} the bottom (y grows downward). Numbers may have decimals.

{colour_rule}

RULES (non-negotiable):
1. Output ONLY valid JSON — no markdown fences, no prose before or after.
2. "shapes" is painted in order, back to front: later shapes cover earlier ones.
3. Use between 6 and {MAX_PROMPT_SHAPES} shapes. Layer them: background forms first, then \
mid-ground, then small details and highlights.
4. Use "gradient" for skies, light and depth.
5. Use "text" sparingly — only when the theme needs a word, at most one short word; the \
letters are tiny.
6. The piece must have intentional visual structure — a focal point, balance, rhythm — \
and evoke the theme, not just name it.

SHAPE VOCABULARY (fill and stroke are optional; give at least one):
{{"type": "rect", "x": 0, "y": {cy}, "w": {w}, "h": {h - cy}, "fill": "#2d1b4e", "stroke": "#ffffff"}}
{{"type": "circle", "cx": {cx}, "cy": {cy}, "r": {max(1, min(w, h) // 8)}, "fill": "#ffd27f"}}
{{"type": "ellipse", "cx": {cx}, "cy": {cy}, "rx": {max(1, w // 4)}, "ry": {max(1, h // 8)}, "fill": "#88ccff"}}
{{"type": "line", "x1": 0, "y1": {cy}, "x2": {w}, "y2": {cy}, "stroke": "#ffb36b", "width": 1}}
{{"type": "polygon", "points": [[0, {h}], [{cx}, {cy}], [{w}, {h}]], "fill": "#1d3b2a"}}
{{"type": "gradient", "x": 0, "y": 0, "w": {w}, "h": {cy}, "from": "#1b1446", "to": "#f2994a", "angle": 90}}
   (angle in degrees: 0 = left→right, 90 = top→bottom; x/y/w/h default to the whole picture)
{{"type": "text", "x": 1, "y": 1, "text": "HI", "color": "#ffffff", "font": "3x5"}}

SHORTCUTS (also in pixels; expanded into shapes for you):
{{"type": "stripes", "colors": ["#223355", "#335577"], "direction": "vertical|horizontal|diagonal", "count": 6}}
   (direction is the way the colour changes: vertical = bands stacked top to bottom)
{{"type": "rings", "colors": ["#ff5e62", "#ff9966"], "cx": {cx}, "cy": {cy}, "count": 5}}
{{"type": "wave", "color": "#3a7bd5", "baseline": {cy}, "amplitude": {max(1, h // 10)}, \
"frequency": 2, "phase": 0, "thickness": 3, "fill": "band|above|below"}}
{{"type": "noise", "colors": ["#ffffff"], "density": 0.05, "x": 0, "y": 0, "w": {w}, "h": {cy}}}
   (scattered single pixels, e.g. stars or sparkle; drawn on top of everything)

OUTPUT FORMAT (strict JSON, no other text):
{{
  "theme": "<the theme in a few words>",
  "description": "<one sentence describing the visual impression>",
  "title": "<a short title, at most 16 characters>",
  "background": "#rrggbb",
  "mirror": "none|horizontal|vertical|both",
  "shapes": [ <6 to {MAX_PROMPT_SHAPES} shapes from the vocabulary above> ]
}}"""
    extra = (extra_instructions or "").strip()
    if extra:
        prompt += "\n" + extra
    return prompt


# ---------------------------------------------------------------------------
# Colours and numbers
# ---------------------------------------------------------------------------


def _colour(value: Any) -> Optional[str]:
    """A colour core accepts (``#rgb``, ``#rrggbb`` or a board name), or ``None``.

    Colour letters (``R``, ``O`` ... ``K``) from the tile vocabulary become
    board colour names. Anything else -- including transparent -- is ``None``.
    """
    if not isinstance(value, str):
        return None
    text = value.strip()
    lowered = text.lower()
    if lowered in BOARD_RGB:
        return lowered
    if len(text) == 1 and text.upper() in LETTER_NAMES:
        return LETTER_NAMES[text.upper()]
    if _HEX6.fullmatch(text) or _HEX3.fullmatch(text):
        return lowered
    return None


def _rgb(value: Any) -> Optional[Tuple[int, int, int]]:
    """RGB for a colour :func:`_colour` accepts, else ``None``."""
    colour = _colour(value)
    if colour is None:
        return None
    if colour in BOARD_RGB:
        return BOARD_RGB[colour]
    digits = colour[1:]
    if len(digits) == 3:
        digits = "".join(ch * 2 for ch in digits)
    return (int(digits[0:2], 16), int(digits[2:4], 16), int(digits[4:6], 16))


def _hex(rgb: Sequence[float]) -> str:
    return "#" + "".join(f"{max(0, min(255, int(round(c)))):02x}" for c in rgb)


def _lerp_colour(colours: Sequence[str], t: float) -> str:
    """The colour at *t* (0..1) along evenly spaced *colours*."""
    rgbs = [_rgb(c) for c in colours]
    if len(rgbs) == 1:
        return _hex(rgbs[0])
    t = max(0.0, min(1.0, t)) * (len(rgbs) - 1)
    i = min(int(t), len(rgbs) - 2)
    f = t - i
    a, b = rgbs[i], rgbs[i + 1]
    return _hex([a[k] + (b[k] - a[k]) * f for k in range(3)])


def _num(value: Any) -> Optional[float]:
    """A finite number clamped to core's coordinate range, tidied for JSON."""
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except ValueError:
            return None
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    value = float(round(max(-MAX_COORD, min(MAX_COORD, float(value))), 2))
    return int(value) if value.is_integer() else value


def _plain(value: Any, fallback: float) -> float:
    """A bare float for internal arithmetic."""
    number = _num(value)
    return fallback if number is None else float(number)


def _colours(value: Any) -> List[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [c for c in (_colour(v) for v in value) if c is not None]


# ---------------------------------------------------------------------------
# Shape cleaning: exactly core's fields, clamped to core's limits
# ---------------------------------------------------------------------------

_SHAPE_FIELDS: Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]] = {
    #            required numbers               optional numbers          required colours   optional colours
    "rect": (("x", "y", "w", "h"), (), (), ("fill", "stroke")),
    "circle": (("cx", "cy", "r"), (), (), ("fill", "stroke")),
    "ellipse": (("cx", "cy", "rx", "ry"), (), (), ("fill", "stroke")),
    "line": (("x1", "y1", "x2", "y2"), ("width",), ("stroke",), ()),
    "polygon": ((), (), (), ("fill", "stroke")),
    "text": (("x", "y"), (), ("color",), ()),
    "gradient": ((), ("x", "y", "w", "h", "angle"), ("from", "to"), ()),
}


def _clean(shape: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """*shape* as a canvas shape core accepts, or ``None`` if it cannot be one.

    Only the fields core defines are kept (core forbids extras), so a model
    cannot smuggle in ``if`` / ``foreach`` expressions or anything else.
    """
    kind = shape.get("type")
    spec = _SHAPE_FIELDS.get(kind)
    if spec is None:
        return None
    required_nums, optional_nums, required_colours, optional_colours = spec
    out: Dict[str, Any] = {"type": kind}
    for name in required_nums:
        value = _num(shape.get(name))
        if value is None:
            return None
        out[name] = value
    for name in optional_nums:
        value = _num(shape.get(name)) if shape.get(name) is not None else None
        if value is not None:
            out[name] = value
    for name in required_colours:
        colour = _colour(shape.get(name))
        if colour is None:
            return None
        out[name] = colour
    for name in optional_colours:
        colour = _colour(shape.get(name))
        if colour is not None:
            out[name] = colour

    if kind in ("rect", "circle", "ellipse", "polygon") and "fill" not in out and "stroke" not in out:
        return None
    if kind == "line":
        out["width"] = int(max(1, min(MAX_LINE_WIDTH, round(out.get("width", 1)))))
    if kind == "polygon":
        points = _points(shape.get("points"))
        if points is None:
            return None
        out["points"] = points
    if kind == "text":
        text = shape.get("text")
        if not isinstance(text, str):
            return None
        # Braces could form a {{…}} expression, which core would evaluate
        # against the page's variables. A model's text is never an expression.
        text = text.replace("{", "").replace("}", "")[:MAX_TEXT_LENGTH]
        if not text.strip():
            return None
        out["text"] = text
        out["font"] = shape.get("font") if shape.get("font") in ("3x5", "5x7") else "3x5"
    return out


def _points(raw: Any) -> Optional[List[List[float]]]:
    if not isinstance(raw, list):
        return None
    points: List[List[float]] = []
    for item in raw[:MAX_POLYGON_POINTS]:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            x, y = _num(item[0]), _num(item[1])
            if x is not None and y is not None:
                points.append([x, y])
    return points if len(points) >= 3 else None


# ---------------------------------------------------------------------------
# Legacy / shortcut primitives -> canvas shapes (pixel space)
# ---------------------------------------------------------------------------


def _with_fill(shape: Dict[str, Any], kind: str, *fields: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"type": kind}
    for name in fields:
        if name in shape:
            out[name] = shape[name]
    out["fill"] = shape.get("fill") or shape.get("color")
    if "stroke" in shape:
        out["stroke"] = shape["stroke"]
    return out


def _gradient(shape: Dict[str, Any], w: int, h: int) -> List[Dict[str, Any]]:
    if "from" in shape or "to" in shape:
        keep = ("x", "y", "w", "h", "from", "to", "angle")
        return [{"type": "gradient", **{k: shape[k] for k in keep if k in shape}}]
    colours = _colours(shape.get("colors"))
    if not colours:
        return []
    if len(colours) == 1:
        return [{"type": "rect", "x": 0, "y": 0, "w": w, "h": h, "fill": colours[0]}]
    direction = str(shape.get("direction") or "vertical").strip().lower()
    n = len(colours) - 1
    if direction in ("horizontal", "x", "left-right", "lr"):
        return [
            {"type": "gradient", "x": w * i / n, "y": 0, "w": w / n, "h": h,
             "from": colours[i], "to": colours[i + 1], "angle": 0}
            for i in range(n)
        ]
    if direction in ("diagonal", "diag", "tlbr"):
        return [{"type": "gradient", "from": colours[0], "to": colours[-1], "angle": 45}]
    if direction in ("antidiagonal", "anti-diagonal", "trbl", "diagonal-reverse"):
        return [{"type": "gradient", "from": colours[0], "to": colours[-1], "angle": 315}]
    if direction in ("radial", "circular", "center", "centre"):
        reach = math.hypot(w / 2, h / 2)
        return [
            {"type": "circle", "cx": w / 2, "cy": h / 2, "r": reach * (RADIAL_STEPS - k) / RADIAL_STEPS,
             "fill": _lerp_colour(colours, (RADIAL_STEPS - k) / RADIAL_STEPS)}
            for k in range(RADIAL_STEPS)
        ]
    return [
        {"type": "gradient", "x": 0, "y": h * i / n, "w": w, "h": h / n,
         "from": colours[i], "to": colours[i + 1], "angle": 90}
        for i in range(n)
    ]


def _count(value: Any, fallback: int, hi: int) -> int:
    number = _num(value)
    return fallback if number is None else max(1, min(hi, int(number)))


def _stripes(shape: Dict[str, Any], w: int, h: int) -> List[Dict[str, Any]]:
    colours = _colours(shape.get("colors")) or ["red", "orange"]
    count = _count(shape.get("count"), max(2, len(colours) * 2), MAX_BANDS)
    direction = str(shape.get("direction") or "vertical").strip().lower()
    out: List[Dict[str, Any]] = []
    for i in range(count):
        colour = colours[i % len(colours)]
        if direction in ("horizontal", "x", "left-right", "lr"):
            out.append({"type": "rect", "x": w * i / count, "y": 0, "w": w / count, "h": h, "fill": colour})
        elif direction in ("diagonal", "diag", "tlbr", "antidiagonal", "anti-diagonal", "trbl"):
            # The band between x/w + y/h = 2i/count and 2(i+1)/count.
            c0, c1 = 2 * i / count, 2 * (i + 1) / count
            points = [[c0 * w, 0], [c1 * w, 0], [0, c1 * h], [0, c0 * h]]
            if direction not in ("diagonal", "diag", "tlbr"):
                points = [[x, h - y] for x, y in points]
            out.append({"type": "polygon", "points": points, "fill": colour})
        else:
            out.append({"type": "rect", "x": 0, "y": h * i / count, "w": w, "h": h / count, "fill": colour})
    return out


def _rings(shape: Dict[str, Any], w: int, h: int) -> List[Dict[str, Any]]:
    colours = _colours(shape.get("colors")) or ["blue", "white"]
    count = _count(shape.get("count"), max(2, len(colours) * 2), MAX_RINGS)
    cx = _plain(shape.get("cx"), w / 2)
    cy = _plain(shape.get("cy"), h / 2)
    reach = max(math.hypot(cx - x, cy - y) for x in (0, w) for y in (0, h)) or 1.0
    return [
        {"type": "circle", "cx": cx, "cy": cy, "r": reach * (k + 1) / count, "fill": colours[k % len(colours)]}
        for k in range(count - 1, -1, -1)
    ]


def _wave(shape: Dict[str, Any], w: int, h: int) -> List[Dict[str, Any]]:
    colour = shape.get("fill") if _colour(shape.get("fill")) else shape.get("color", "blue")
    baseline = _plain(shape.get("baseline"), h / 2)
    amplitude = _plain(shape.get("amplitude"), h * 0.15)
    frequency = max(0.0, min(16.0, _plain(shape.get("frequency"), 1.5)))
    phase = _plain(shape.get("phase"), 0.0)
    half = max(_plain(shape.get("thickness"), h * 0.12), 1.0) / 2
    mode = str(shape.get("fill", "band")).strip().lower()
    xs = [w * i / WAVE_SAMPLES for i in range(WAVE_SAMPLES + 1)]
    crest = [baseline + amplitude * math.sin(math.tau * frequency * x / max(w, 1) + phase) for x in xs]
    if mode == "below":
        points = [[x, y] for x, y in zip(xs, crest)] + [[w, h], [0, h]]
    elif mode == "above":
        points = [[x, y] for x, y in zip(xs, crest)] + [[w, 0], [0, 0]]
    else:
        points = [[x, y - half] for x, y in zip(xs, crest)] + [[x, y + half] for x, y in reversed(list(zip(xs, crest)))]
    return [{"type": "polygon", "points": points, "fill": colour}]


def _mirror_x(shape: Dict[str, Any], w: int) -> Optional[Dict[str, Any]]:
    kind = shape["type"]
    out = dict(shape)
    if kind == "rect":
        out["x"] = w - shape["x"] - shape["w"]
    elif kind in ("circle", "ellipse"):
        out["cx"] = w - shape["cx"]
    elif kind == "line":
        out["x1"], out["x2"] = w - shape["x1"], w - shape["x2"]
    elif kind == "polygon":
        out["points"] = [[w - x, y] for x, y in shape["points"]]
    elif kind == "gradient":
        if "x" in shape or "w" in shape:
            out["x"] = w - shape.get("x", 0) - shape.get("w", w)
        out["angle"] = (180 - shape.get("angle", 90)) % 360
    else:  # text reads backwards when mirrored: leave it single
        return None
    return out


def _mirror_y(shape: Dict[str, Any], h: int) -> Optional[Dict[str, Any]]:
    kind = shape["type"]
    out = dict(shape)
    if kind == "rect":
        out["y"] = h - shape["y"] - shape["h"]
    elif kind in ("circle", "ellipse"):
        out["cy"] = h - shape["cy"]
    elif kind == "line":
        out["y1"], out["y2"] = h - shape["y1"], h - shape["y2"]
    elif kind == "polygon":
        out["points"] = [[x, h - y] for x, y in shape["points"]]
    elif kind == "gradient":
        if "y" in shape or "h" in shape:
            out["y"] = h - shape.get("y", 0) - shape.get("h", h)
        out["angle"] = (-shape.get("angle", 90)) % 360
    else:
        return None
    return out


def _mirrored(shapes: List[Dict[str, Any]], mode: Any, w: int, h: int) -> List[Dict[str, Any]]:
    """Each shape followed by its reflections.

    Interleaving (shape, its reflections, next shape, ...) rather than
    appending all reflections at the end is what makes the result exactly
    symmetric: every pixel and its mirror image are last covered by the same
    shape index.
    """
    name = str(mode or "none").strip().lower()
    across = name in ("horizontal", "both", "left-right", "lr")
    down = name in ("vertical", "both", "top-bottom", "tb")
    if not (across or down):
        return shapes
    out: List[Dict[str, Any]] = []
    for shape in shapes:
        variants = [shape]
        if across:
            variants.append(_mirror_x(shape, w))
        if down:
            variants.append(_mirror_y(shape, h))
        if across and down:
            flipped = _mirror_x(shape, w)
            variants.append(_mirror_y(flipped, h) if flipped else None)
        for variant in variants:
            if variant is not None:
                tidy = _clean(variant)
                if tidy is not None:
                    out.append(tidy)
    return out


# ---------------------------------------------------------------------------
# Noise -> pixel rows
# ---------------------------------------------------------------------------


class _PixelLayer:
    def __init__(self, w: int, h: int):
        self.w, self.h = w, h
        self.cells = [["."] * w for _ in range(h)]
        self.palette: Dict[str, str] = {}

    def key_for(self, colour: str) -> Optional[str]:
        for key, value in self.palette.items():
            if value == colour:
                return key
        if len(self.palette) >= MAX_PALETTE_KEYS:
            return None
        key = _PALETTE_KEYS[len(self.palette)]
        self.palette[key] = colour
        return key

    def noise(self, shape: Dict[str, Any]) -> None:
        colours = _colours(shape.get("colors")) or ["white"]
        density = max(0.0, min(MAX_NOISE_DENSITY, _plain(shape.get("density"), 0.15)))
        x0 = int(max(0, min(self.w, _plain(shape.get("x"), 0))))
        y0 = int(max(0, min(self.h, _plain(shape.get("y"), 0))))
        x1 = int(max(x0, min(self.w, x0 + _plain(shape.get("w"), self.w))))
        y1 = int(max(y0, min(self.h, y0 + _plain(shape.get("h"), self.h))))
        # Seeded from the shape itself: the same reply always gives the same
        # stars, so a cached piece and its re-render agree.
        rng = random.Random(json.dumps(shape, sort_keys=True, default=str))
        keys = [self.key_for(c) for c in colours]
        keys = [k for k in keys if k is not None]
        if not keys:
            return
        for y in range(y0, y1):
            for x in range(x0, x1):
                if rng.random() < density:
                    self.cells[y][x] = keys[rng.randrange(len(keys))]

    def mirror(self, mode: Any) -> None:
        name = str(mode or "none").strip().lower()
        if name in ("horizontal", "both", "left-right", "lr"):
            for row in self.cells:
                for x in range(self.w // 2):
                    row[self.w - 1 - x] = row[x]
        if name in ("vertical", "both", "top-bottom", "tb"):
            for y in range(self.h // 2):
                self.cells[self.h - 1 - y] = list(self.cells[y])

    def rows(self) -> List[str]:
        """Pixel rows with trailing transparency trimmed (smaller pages.json)."""
        rows = ["".join(row).rstrip(".") for row in self.cells]
        while rows and not rows[-1]:
            rows.pop()
        return rows


# ---------------------------------------------------------------------------
# Scene -> content
# ---------------------------------------------------------------------------


def _coordinate_values(shapes: Sequence[Any]) -> List[float]:
    values: List[float] = []
    for shape in shapes:
        if not isinstance(shape, dict):
            continue
        for name in _X_FIELDS + _Y_FIELDS + ("r",):
            number = _num(shape.get(name))
            if number is not None:
                values.append(float(number))
        for point in shape.get("points") or []:
            if isinstance(point, (list, tuple)):
                values.extend(float(v) for v in (_num(p) for p in point[:2]) if v is not None)
    return values


def _is_normalised(shapes: Sequence[Any]) -> bool:
    """Whether a scene is in the tile prompt's 0..1 space rather than pixels.

    True when every coordinate lies in [0, 1] and at least one is a fraction:
    a pixel-space scene squeezed into the top-left pixel is not a real reply.
    """
    values = _coordinate_values(shapes)
    return bool(values) and all(0 <= v <= 1 for v in values) and any(0 < v < 1 for v in values)


def _scale(shape: Dict[str, Any], w: int, h: int) -> Dict[str, Any]:
    out = dict(shape)
    for name in _X_FIELDS:
        if _num(shape.get(name)) is not None:
            out[name] = float(_num(shape[name])) * w
    for name in _Y_FIELDS:
        if _num(shape.get(name)) is not None:
            out[name] = float(_num(shape[name])) * h
    if _num(shape.get("r")) is not None:
        out["r"] = float(_num(shape["r"])) * min(w, h)
    if _num(shape.get("thickness")) is not None:
        span = h if str(shape.get("type", "")).lower() == "wave" else min(w, h)
        out["thickness"] = float(_num(shape["thickness"])) * span
    if isinstance(shape.get("points"), list):
        out["points"] = [
            [float(_num(p[0]) or 0) * w, float(_num(p[1]) or 0) * h]
            for p in shape["points"]
            if isinstance(p, (list, tuple)) and len(p) >= 2
        ]
    return out


def _expand(shape: Dict[str, Any], w: int, h: int) -> List[Dict[str, Any]]:
    """Canvas-shaped dicts for one model shape (before cleaning)."""
    kind = str(shape.get("type", "")).strip().lower()
    if kind in ("rect", "rectangle"):
        return [_with_fill(shape, "rect", "x", "y", "w", "h")]
    if kind == "circle":
        if "r" in shape:
            return [_with_fill(shape, "circle", "cx", "cy", "r")]
        return [_with_fill(shape, "ellipse", "cx", "cy", "rx", "ry")]
    if kind == "ellipse":
        return [_with_fill(shape, "ellipse", "cx", "cy", "rx", "ry")]
    if kind in ("polygon", "triangle"):
        return [_with_fill(shape, "polygon", "points")]
    if kind == "line":
        return [{
            "type": "line", "x1": shape.get("x1"), "y1": shape.get("y1"), "x2": shape.get("x2"),
            "y2": shape.get("y2"), "stroke": shape.get("stroke", shape.get("color")),
            "width": shape.get("width", shape.get("thickness", 1)),
        }]
    if kind == "text":
        return [dict(shape, type="text")]
    if kind == "gradient":
        return _gradient(shape, w, h)
    if kind in ("stripes", "bands"):
        return _stripes(shape, w, h)
    if kind == "rings":
        return _rings(shape, w, h)
    if kind == "wave":
        return _wave(shape, w, h)
    return []


def _colours_in(content: Dict[str, Any]) -> set:
    found = {_rgb(content.get("background"))}
    for shape in content.get("shapes", []):
        for name in ("fill", "stroke", "color", "from", "to"):
            if name in shape:
                found.add(_rgb(shape[name]))
    found.update(_rgb(v) for v in content.get("palette", {}).values())
    found.discard(None)
    return found


def _core_check(content: Dict[str, Any]) -> None:
    """Core's own validator as a last word, when this core has one."""
    if _CoreCanvasContent is None:
        return
    try:
        _CoreCanvasContent.model_validate(content)
    except Exception as exc:  # noqa: BLE001 -- pydantic or anything else: never emit it
        raise ArtValidationError(f"Canvas content refused by core: {exc}") from exc


def scene_to_content(scene: Dict[str, Any], target: PixelTarget) -> Dict[str, Any]:
    """A content object for *scene* at *target*'s size, clamped to core's limits.

    Accepts canvas shapes, this plugin's older primitives and normalised
    (0..1) coordinates. Raises :class:`ArtValidationError` when nothing
    drawable is left or the picture would be a single colour.
    """
    w, h = target.size
    raw_shapes = scene.get("shapes") if isinstance(scene.get("shapes"), list) else []
    normalised = _is_normalised(raw_shapes)
    layer = _PixelLayer(w, h)
    shapes: List[Dict[str, Any]] = []
    for raw in raw_shapes:
        if not isinstance(raw, dict):
            continue
        shape = _scale(raw, w, h) if normalised else raw
        if str(shape.get("type", "")).strip().lower() == "noise":
            layer.noise(shape)
            continue
        for expanded in _expand(shape, w, h):
            tidy = _clean(expanded)
            if tidy is not None:
                shapes.append(tidy)

    mirror = scene.get("mirror")
    shapes = _mirrored(shapes, mirror, w, h)[:MAX_CANVAS_SHAPES]
    layer.mirror(mirror)
    pixels = layer.rows()

    content: Dict[str, Any] = {
        "size": [w, h],
        "background": _colour(scene.get("background")) or "#000000",
    }
    if shapes:
        content["shapes"] = shapes
    if pixels:
        content["palette"] = dict(layer.palette)
        content["pixels"] = pixels
    if not shapes and not pixels:
        raise ArtValidationError("Scene has no drawable shapes")
    if len(_colours_in(content)) < 2:
        raise ArtValidationError("Scene uses fewer than 2 colours")
    _core_check(content)
    return content


def tiles_content(grid: Sequence[Sequence[str]]) -> Dict[str, Any]:
    """A tile piece (rows of colour letters) as a content object: one pixel per tile."""
    rows = ["".join(cell if cell in LETTER_NAMES else "K" for cell in row)[:MAX_CONTENT_SIZE]
            for row in list(grid)[:MAX_CONTENT_SIZE]]
    width = max((len(row) for row in rows), default=1) or 1
    rows = [row.ljust(width, "K") for row in rows] or ["K"]
    used = sorted({ch for row in rows for ch in row})
    return {
        "size": [width, len(rows)],
        "background": "black",
        "palette": {letter: LETTER_NAMES[letter] for letter in used},
        "pixels": rows,
    }


def _grid_content(raw: List[Any]) -> Dict[str, Any]:
    rows = [
        [(cell.strip()[:1].upper() if isinstance(cell, str) and cell.strip() else "K") for cell in row]
        for row in raw
        if isinstance(row, list) and row
    ]
    if not rows:
        raise ArtValidationError("'grid' contains no usable rows")
    return tiles_content(rows)


def parse_canvas_response(raw: Any, target: PixelTarget) -> Dict[str, Any]:
    """Parse a model reply into ``theme``, ``description``, ``title`` and ``content``."""
    cleaned = _strip_fences(raw.strip() if isinstance(raw, str) else "")
    parsed = _loads_with_salvage(cleaned)
    if not isinstance(parsed, dict):
        raise ArtValidationError("Response is not a JSON object")
    if isinstance(parsed.get("shapes"), list) and parsed["shapes"]:
        content = scene_to_content(parsed, target)
    elif isinstance(parsed.get("grid"), list) and parsed["grid"]:
        content = _grid_content(parsed["grid"])
    else:
        raise ArtValidationError("Response contains neither scene 'shapes' nor a 'grid'")
    return {
        "theme": _text(parsed.get("theme"), MAX_THEME_CHARS),
        "description": _text(parsed.get("description"), MAX_DESCRIPTION_CHARS),
        "title": _text(parsed.get("title"), MAX_CONTENT_SIZE),
        "content": content,
    }


# ---------------------------------------------------------------------------
# Fallback scene
# ---------------------------------------------------------------------------

FALLBACK_THEME = "twilight hills"
FALLBACK_DESCRIPTION = "A low sun setting behind violet hills."


def fallback_content(target: PixelTarget) -> Dict[str, Any]:
    """A fixed twilight landscape at *target*'s size. Always valid; never random."""
    w, h = target.size
    scene = [
        {"type": "gradient", "from": "#1b1446", "to": "#f2994a", "angle": 90},
        {"type": "circle", "cx": 0.68 * w, "cy": 0.56 * h, "r": 0.14 * min(w, h), "fill": "#ffd36e"},
        {"type": "polygon", "fill": "#3b2a5a", "points": [
            [0, h], [0, 0.7 * h], [0.25 * w, 0.5 * h], [0.5 * w, 0.68 * h],
            [0.75 * w, 0.45 * h], [w, 0.62 * h], [w, h]]},
        {"type": "polygon", "fill": "#1d1430", "points": [
            [0, h], [0, 0.85 * h], [0.35 * w, 0.72 * h], [0.7 * w, 0.88 * h], [w, 0.78 * h], [w, h]]},
    ]
    return {
        "size": [w, h],
        "background": "#0b0820",
        "shapes": [tidy for tidy in (_clean(s) for s in scene) if tidy is not None],
    }


# ---------------------------------------------------------------------------
# Canvas -> board tiles
# ---------------------------------------------------------------------------


def _nearest_letter(rgb: Tuple[float, float, float]) -> str:
    best = min(BOARD_RGB, key=lambda name: sum((BOARD_RGB[name][k] - rgb[k]) ** 2 for k in range(3)))
    return _NAME_LETTERS[best]


def _inside_polygon(points: Sequence[Sequence[float]], x: float, y: float) -> bool:
    inside = False
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            if x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
                inside = not inside
    return inside


def _segment_distance(x: float, y: float, s: Dict[str, Any]) -> float:
    x1, y1, x2, y2 = s["x1"], s["y1"], s["x2"], s["y2"]
    dx, dy = x2 - x1, y2 - y1
    length_sq = dx * dx + dy * dy
    t = 0.0 if length_sq == 0 else max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / length_sq))
    return math.hypot(x - (x1 + t * dx), y - (y1 + t * dy))


def _colour_at(shape: Dict[str, Any], x: float, y: float, w: int, h: int, reach: float):
    """The colour *shape* paints at content point ``(x, y)``, or ``None``."""
    kind = shape.get("type")
    fill = _rgb(shape.get("fill"))
    if kind == "rect" and fill:
        x0, x1 = sorted((shape["x"], shape["x"] + shape["w"]))
        y0, y1 = sorted((shape["y"], shape["y"] + shape["h"]))
        return fill if x0 <= x < x1 and y0 <= y < y1 else None
    if kind == "circle" and fill:
        return fill if (x - shape["cx"]) ** 2 + (y - shape["cy"]) ** 2 <= shape["r"] ** 2 else None
    if kind == "ellipse" and fill and shape["rx"] and shape["ry"]:
        d = ((x - shape["cx"]) / shape["rx"]) ** 2 + ((y - shape["cy"]) / shape["ry"]) ** 2
        return fill if d <= 1 else None
    if kind == "polygon" and fill:
        return fill if _inside_polygon(shape["points"], x, y) else None
    if kind == "line":
        near = max(shape.get("width", 1) / 2, reach)
        return _rgb(shape.get("stroke")) if _segment_distance(x, y, shape) <= near else None
    if kind == "gradient":
        gx, gy = shape.get("x", 0), shape.get("y", 0)
        gw, gh = shape.get("w", w), shape.get("h", h)
        x0, x1 = sorted((gx, gx + gw))
        y0, y1 = sorted((gy, gy + gh))
        if not (x0 <= x < x1 and y0 <= y < y1):
            return None
        a, b = _rgb(shape.get("from")), _rgb(shape.get("to"))
        if a is None or b is None:
            return None
        angle = math.radians(shape.get("angle", 90))
        ux, uy = math.cos(angle), math.sin(angle)
        corners = [cx * ux + cy * uy for cx in (x0, x1) for cy in (y0, y1)]
        lo, hi = min(corners), max(corners)
        t = 0.0 if hi == lo else (x * ux + y * uy - lo) / (hi - lo)
        return tuple(a[k] + (b[k] - a[k]) * t for k in range(3))
    return None  # stroke-only outlines and text are too fine for a tile


def sample_grid(content: Dict[str, Any], rows: int, cols: int) -> List[List[str]]:
    """Tile art for a *rows* x *cols* board, sampled from *content* at cell centres.

    Each tile takes the nearest board colour to what the canvas paints at its
    centre. Lines are widened to half a tile so they survive the sampling.
    """
    w, h = content.get("size") or DEFAULT_CANVAS_SIZE
    background = _rgb(content.get("background")) or (0, 0, 0)
    palette = content.get("palette") or {}
    pixels = content.get("pixels") or []
    shapes = content.get("shapes") or []
    cell_w, cell_h = w / cols, h / rows
    reach = max(cell_w, cell_h) / 2
    grid: List[List[str]] = []
    for r in range(rows):
        y = (r + 0.5) * cell_h
        row: List[str] = []
        for c in range(cols):
            x = (c + 0.5) * cell_w
            colour = background
            for shape in shapes:
                painted = _colour_at(shape, x, y, w, h, reach)
                if painted is not None:
                    colour = painted
            py, px = int(y), int(x)
            if py < len(pixels) and px < len(pixels[py]) and pixels[py][px] != ".":
                colour = _rgb(palette.get(pixels[py][px])) or colour
            row.append(_nearest_letter(colour))
        grid.append(row)
    return grid


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def canvas_messages(generator: Any, theme: str, target: PixelTarget) -> List[Dict[str, str]]:
    """The chat messages for one full-colour piece."""
    system = generator.custom_system_prompt or canvas_system_prompt(target, generator.extra_instructions)
    user = (
        f"Compose a pixel-art piece with this theme: {theme}\n\n"
        "Remember: output ONLY the JSON object described above."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def generate_pixel_piece(generator: Any, canvas: Any, target: PixelTarget) -> Optional[Dict[str, Any]]:
    """One model call (plus retries) -> a full-colour canvas and the tile art derived from it.

    Returns ``theme``, ``description``, ``title``, ``canvas`` (content
    object), ``grid``, ``lines`` and ``art``; or ``None`` when every attempt
    failed (a transport failure is not retried, as for tile art).
    """
    theme = generator._pick_theme()
    for attempt in range(MAX_ATTEMPTS):
        try:
            raw = generator.complete_messages(canvas_messages(generator, theme, target), CANVAS_MAX_TOKENS)
            parsed = parse_canvas_response(raw, target)
        except (requests.RequestException, ArtRequestError) as exc:
            logger.error("API request failed: %s", exc)
            return None
        except ArtValidationError as exc:
            logger.warning(
                "Pixel art validation failed (attempt %d/%d) for theme '%s' at %dx%d: %s",
                attempt + 1, MAX_ATTEMPTS, theme, target.width, target.height, exc,
            )
            theme = generator._pick_theme(exclude=theme)
            continue

        content = parsed["content"]
        grid = sample_grid(content, canvas.art_rows, canvas.cols)
        title = (parsed["title"] or parsed["theme"])[: canvas.cols] if canvas.has_title_row else ""
        lines = generator.render_lines(grid, title, canvas)
        return {
            "theme": parsed["theme"],
            "description": parsed["description"],
            "title": title,
            "canvas": content,
            "grid": grid,
            "lines": lines,
            "art": "\n".join(lines),
        }

    logger.error("Pixel art generation failed after %d attempts at %dx%d", MAX_ATTEMPTS, target.width, target.height)
    return None
