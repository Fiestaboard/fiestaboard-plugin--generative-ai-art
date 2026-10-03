"""Art generation logic for the Generative AI Art FiestaBoard plugin.

The plugin renders onto whatever board it is bound to: a Flagship (22x6), a
Note (15x3), or a note array anywhere from 15x3 to 120x24 (which is what a
FiestaPanel is). Nothing here owns a list of board sizes; every dimension
arrives as a :class:`Canvas` built from ``self.board`` by the plugin.

Two emission strategies, chosen from the cell count alone:

``grid`` (small boards, at most :data:`PER_CELL_MAX_CELLS` cells)
    The model emits one colour letter per cell. Fine for a Flagship (132
    cells) or a Note (45): the response is a few hundred tokens and the
    model gets per-tile control.

``scene`` (everything larger)
    The model emits a *compact scene description* -- a background plus an
    ordered list of shapes in normalised 0..1 coordinates -- and this module
    rasterises it to ``cols x rows``. Token cost is O(1) in board size
    instead of O(cells), and any aspect ratio is native by construction
    because the scene is described in a unit square that is mapped onto the
    board. This is what makes a 120x24 panel (2,880 cells) possible at all:
    per-cell emission there would be ~6,000 output tokens, past the point
    where a small default model reliably finishes, and a truncated response
    is a hard parse failure.

Response handling is deliberately *repair-or-retry* rather than
all-or-nothing. A grid that is one row short, or a few cells wider than the
board, is resampled onto the canvas; only a response too damaged to be art
(mostly-invalid cells, fewer than two colours, unparseable even after
salvage) costs a retry.
"""

from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import requests

logger = logging.getLogger(__name__)

# Request timeout in seconds
REQUEST_TIMEOUT = 60

# Color alphabet: single letter → FiestaBoard color marker
# K = blacK (not B, to avoid B=blue ambiguity)
COLOR_LETTERS: Dict[str, str] = {
    "R": "{red}",
    "O": "{orange}",
    "Y": "{yellow}",
    "G": "{green}",
    "B": "{blue}",
    "V": "{violet}",
    "W": "{white}",
    "K": "{black}",
}

VALID_COLORS = set(COLOR_LETTERS.keys())

#: Painted where a scene declares no background and wherever no shape covers.
DEFAULT_BACKGROUND = "K"

#: The board assumed when no board is bound (``self.board is None``): unit
#: tests and legacy callers both hit that, and the platform contract is
#: "treat it as a Flagship", not "crash". This is the ONLY place a board
#: size is written down in this plugin.
DEFAULT_BOARD_ROWS = 6
DEFAULT_BOARD_COLS = 22

#: Above this many cells, per-cell emission stops being viable. A cell costs
#: roughly four output tokens as a quoted letter in a JSON array, so 240
#: cells is already ~1k tokens; a 30x12 panel (360) is ~1.5k and a 120x24
#: panel (2,880) is ~12k -- well past what a small default model finishes.
#: Boards at or below this use ``grid``; everything above uses ``scene``.
#: 240 covers a Flagship (132), a Note (45) and a 15x12 array (180).
PER_CELL_MAX_CELLS = 240

#: Attempts per piece. Each retry picks a different theme, because a theme
#: the model cannot express is a common cause of a malformed response.
MAX_ATTEMPTS = 3

# --- Repair tolerances -----------------------------------------------------
# A response within these bounds is repaired onto the canvas; outside them it
# is too damaged to be the art anyone asked for, so it costs a retry.

#: A grid may be half to double the expected size on either axis and still be
#: resampled. Beyond that the model clearly did not compose for this board.
SIZE_TOLERANCE = 0.5

#: Fraction of cells that may be unreadable before the response is discarded.
MAX_INVALID_CELL_RATIO = 0.25

#: Fraction of rows that may be badly sized before the response is discarded.
MAX_DAMAGED_ROW_RATIO = 1.0 / 3.0

# --- Token budgets ---------------------------------------------------------
# Never left implicit: an unset max_tokens means the endpoint's default, and
# a truncated JSON array is a parse failure, not a smaller picture.

#: A scene is a fixed handful of shapes regardless of board size.
SCENE_MAX_TOKENS = 1200

#: Roughly what one `"B", ` costs, plus room for the JSON envelope.
GRID_TOKENS_PER_CELL = 4
GRID_TOKEN_OVERHEAD = 300
MIN_MAX_TOKENS = 512
MAX_MAX_TOKENS = 4096

#: Metadata bounds, kept equal to ``variables.simple.*.max_length`` in
#: manifest.json (``test_manifest.py`` asserts they stay in step). Truncating
#: here is what keeps the manifest's declared lengths honest whatever the
#: model returns.
MAX_THEME_CHARS = 50
MAX_DESCRIPTION_CHARS = 100

# Built-in theme pool — varied enough to produce very different compositions
BUILTIN_THEMES = [
    # --- Geometric / structural ---
    "concentric rings expanding from the center",
    "diagonal color gradient from corner to corner",
    "bold geometric chevrons or zigzags",
    "symmetrical mandala-like radial pattern",
    "horizontal banded color field with a focal accent",
    "vertical columns of alternating color families",
    "color-block abstraction inspired by Mondrian",
    "checkerboard variant with irregular cell sizes",
    "two large opposing diagonal color fields",
    "staircase terracing pattern from bottom-left",
    "diamond lattice with alternating fill colors",
    "isometric cube illusion using three tones",
    "op-art concentric squares with high contrast",
    "bold offset stripes at 45 degrees",
    "herringbone weave pattern",
    "interlocking brick or basketweave pattern",
    "radial spoke pattern emanating from one corner",
    "plaid or tartan: horizontal and vertical bands crossing",
    "Bauhaus-style composition: rectangles, circles, lines",
    "triangle grid — large and small contrasting triangles",
    "pixel grid of 2×2 color blocks in a structured palette",
    # --- Nature / landscape ---
    "abstract mountain silhouette with sky gradient",
    "aurora borealis: vertical curtains of shifting color",
    "sunset horizon with layered warm and cool bands",
    "desert dunes: warm earth tones with cool shadow bands",
    "cityscape silhouette: dark blocks on gradient sky",
    "pixel landscape: ground, horizon, sky layers",
    "underwater: light shafts descending through blue-green",
    "storm cell: dark vortex with electric accent colors",
    "volcanic landscape: black ground, orange glow, dark sky",
    "coral reef cross-section: layered warm and cool patches",
    "autumn forest floor: scattered warm patches on dark ground",
    "tidal pool reflection: mirrored bands with shimmer accent",
    "twilight sky fading from orange at the horizon to deep violet",
    "cloud formations: soft rounded masses on a gradient sky",
    "glacier: cool blues and whites with deep crevasse shadows",
    "night sky with a bright horizon glow",
    # --- Texture / pattern ---
    "scattered irregular patches like a mosaic",
    "wave interference pattern — overlapping sine-like curves",
    "stained glass: irregular polygons in contrasting colors",
    "circuit board: right-angle traces on a dark ground",
    "random walk path of a bright color on a neutral field",
    "heatmap-style gradient: cool edges, warm center",
    "sparse pointillist dots on a contrasting background",
    "blurred organic blob cluster in the center",
    "lava-lamp: rounded blobs of warm color on cool ground",
    "topographic contour lines on a single-hue gradient",
    "woven textile: tight alternating thread colors",
    "noise field: irregular dithered color patches",
    "kaleidoscope mirror reflection",
    "seed-of-life sacred geometry circles",
    # --- Abstract / painterly ---
    "flowing river of a single bright color on a dark ground",
    "bold typographic-inspired abstract shapes",
    "horizontal banded color field — Mark Rothko-style",
    "abstract expressionist color field: large loose brushstroke blocks",
    "color spectrum arc from cool to warm",
    "neon sign glow: bright lines and halos on dark background",
    "patchwork quilt: irregular rectangles in a warm palette",
    "butterfly wing symmetry: mirrored left-right composition",
    "color gradient snake path winding across the board",
    "bold racing stripes: two or three wide parallel bands",
    "Japanese wave pattern: repeating curved arcs in blue",
    "retro video-game sprite: bright pixel art on dark background",
]


class ArtValidationError(Exception):
    """Raised when the LLM response cannot be turned into a valid art grid."""


# ---------------------------------------------------------------------------
# Canvas
# ---------------------------------------------------------------------------


class ArtRequestError(Exception):
    """A request that never produced a reply worth parsing.

    Raised by a ``complete`` callable handed to :class:`ArtGenerator` (the
    plugin's bridge to FiestaBoard's AI providers). Treated exactly like a
    transport failure: no retry with another theme.
    """


@dataclass(frozen=True)
class Canvas:
    """The board being composed for. Every dimension in this module comes
    from one of these; there are no board-size literals on a layout path.

    Built by the plugin from ``self.board``, so a Flagship, a Note and any
    note array from 15x3 to 120x24 are all just different ``rows``/``cols``.
    """

    rows: int
    cols: int
    show_title: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "rows", max(1, int(self.rows)))
        object.__setattr__(self, "cols", max(1, int(self.cols)))

    @classmethod
    def default(cls, show_title: bool = False) -> "Canvas":
        """The canvas used when no board is bound: a Flagship."""
        return cls(rows=DEFAULT_BOARD_ROWS, cols=DEFAULT_BOARD_COLS, show_title=show_title)

    @property
    def has_title_row(self) -> bool:
        """Whether a row is actually spent on a title.

        A one-row board has no row to spare, so ``show_title`` is honoured
        only where there is more than one row -- otherwise the title would
        consume the entire piece.
        """
        return self.show_title and self.rows > 1

    @property
    def art_rows(self) -> int:
        """Rows available to the art itself."""
        return self.rows - 1 if self.has_title_row else self.rows

    @property
    def cells(self) -> int:
        return self.art_rows * self.cols

    @property
    def mode(self) -> str:
        """``"grid"`` (per-cell) or ``"scene"`` (described and rasterised)."""
        return "grid" if self.cells <= PER_CELL_MAX_CELLS else "scene"

    @property
    def key(self) -> str:
        """Cache key: two boards with these dimensions render interchangeably."""
        return f"{self.cols}x{self.rows}{'+title' if self.has_title_row else ''}"


# ---------------------------------------------------------------------------
# Scene rasteriser
# ---------------------------------------------------------------------------


def _letter(value: Any, fallback: str = DEFAULT_BACKGROUND) -> str:
    """Coerce *value* to a valid colour letter, or return *fallback*."""
    if isinstance(value, str) and value:
        candidate = value.strip()[:1].upper()
        if candidate in VALID_COLORS:
            return candidate
    return fallback


def _letters(value: Any, fallback: Sequence[str] = ("W", "K")) -> List[str]:
    """Coerce *value* to a non-empty list of valid colour letters."""
    if isinstance(value, str):
        value = [value]
    out: List[str] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                candidate = item.strip()[:1].upper()
                if candidate in VALID_COLORS:
                    out.append(candidate)
    return out or list(fallback)


def _number(value: Any, fallback: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Coerce *value* to a float clamped to ``[lo, hi]``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return fallback
    return max(lo, min(hi, float(value)))


def _count(value: Any, fallback: int, lo: int = 1, hi: int = 64) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return fallback
    return max(lo, min(hi, int(value)))


def _axis(u: float, v: float, direction: Any) -> float:
    """Project a normalised point onto a 0..1 parameter for *direction*."""
    name = str(direction or "vertical").strip().lower()
    if name in ("horizontal", "x", "left-right", "lr"):
        return u
    if name in ("diagonal", "diag", "tlbr"):
        return (u + v) / 2.0
    if name in ("antidiagonal", "anti-diagonal", "trbl", "diagonal-reverse"):
        return (u + (1.0 - v)) / 2.0
    if name in ("radial", "circular", "center", "centre"):
        # Distance from the centre of the unit square, normalised so the
        # corners land exactly at 1.0 whatever the board's aspect ratio.
        return min(1.0, math.hypot(u - 0.5, v - 0.5) / math.sqrt(0.5))
    return v


def _pick(colors: Sequence[str], t: float) -> str:
    """Band index for parameter *t* over *colors*."""
    n = len(colors)
    return colors[min(n - 1, max(0, int(t * n)))]


def _span(start: float, extent: float, count: int) -> Tuple[int, int]:
    """Half-open cell range covering ``[start, start+extent]`` of *count* cells.

    Rounded outwards to at least one cell, so a shape thinner than a tile
    still paints rather than vanishing on a small board -- the reason a Note
    does not render as a blank rectangle when a scene was composed loosely.
    """
    lo = int(math.floor(start * count))
    hi = int(math.ceil((start + extent) * count))
    lo = max(0, min(count - 1, lo))
    hi = max(lo + 1, min(count, hi))
    return lo, hi


def _each_cell(rows: int, cols: int):
    for r in range(rows):
        v = (r + 0.5) / rows
        for c in range(cols):
            yield r, c, (c + 0.5) / cols, v


def _paint_gradient(grid: List[List[str]], shape: Dict[str, Any], rng: random.Random) -> None:
    colors = _letters(shape.get("colors"), ("B", "V"))
    direction = shape.get("direction")
    rows, cols = len(grid), len(grid[0])
    for r, c, u, v in _each_cell(rows, cols):
        grid[r][c] = _pick(colors, _axis(u, v, direction))


def _paint_stripes(grid: List[List[str]], shape: Dict[str, Any], rng: random.Random) -> None:
    colors = _letters(shape.get("colors"), ("R", "O"))
    direction = shape.get("direction")
    rows, cols = len(grid), len(grid[0])
    bands = _count(shape.get("count"), max(2, len(colors) * 2), lo=1, hi=max(rows, cols))
    for r, c, u, v in _each_cell(rows, cols):
        index = int(_axis(u, v, direction) * bands)
        grid[r][c] = colors[index % len(colors)]


def _paint_rect(grid: List[List[str]], shape: Dict[str, Any], rng: random.Random) -> None:
    color = _letter(shape.get("color"), "W")
    rows, cols = len(grid), len(grid[0])
    x = _number(shape.get("x"), 0.25)
    y = _number(shape.get("y"), 0.25)
    w = _number(shape.get("w"), 0.5, lo=0.0, hi=1.0)
    h = _number(shape.get("h"), 0.5, lo=0.0, hi=1.0)
    c0, c1 = _span(x, w, cols)
    r0, r1 = _span(y, h, rows)
    for r in range(r0, r1):
        for c in range(c0, c1):
            grid[r][c] = color


def _paint_ellipse(grid: List[List[str]], shape: Dict[str, Any], rng: random.Random) -> None:
    color = _letter(shape.get("color"), "W")
    rows, cols = len(grid), len(grid[0])
    cx = _number(shape.get("cx"), 0.5)
    cy = _number(shape.get("cy"), 0.5)
    # Floor the radii at half a cell so an ellipse never rounds away to nothing.
    rx = max(_number(shape.get("rx"), 0.3, lo=0.0), 0.5 / cols)
    ry = max(_number(shape.get("ry"), 0.3, lo=0.0), 0.5 / rows)
    for r, c, u, v in _each_cell(rows, cols):
        if ((u - cx) / rx) ** 2 + ((v - cy) / ry) ** 2 <= 1.0:
            grid[r][c] = color


def _paint_rings(grid: List[List[str]], shape: Dict[str, Any], rng: random.Random) -> None:
    colors = _letters(shape.get("colors"), ("B", "W"))
    rows, cols = len(grid), len(grid[0])
    cx = _number(shape.get("cx"), 0.5)
    cy = _number(shape.get("cy"), 0.5)
    bands = _count(shape.get("count"), max(2, len(colors) * 2), lo=1, hi=max(rows, cols))
    # Normalise by the farthest corner so the rings always reach the edge.
    furthest = max(
        math.hypot(cx - corner_x, cy - corner_y)
        for corner_x in (0.0, 1.0)
        for corner_y in (0.0, 1.0)
    ) or 1.0
    for r, c, u, v in _each_cell(rows, cols):
        d = math.hypot(u - cx, v - cy) / furthest
        grid[r][c] = colors[int(d * bands) % len(colors)]


def _paint_triangle(grid: List[List[str]], shape: Dict[str, Any], rng: random.Random) -> None:
    color = _letter(shape.get("color"), "W")
    raw = shape.get("points")
    if not isinstance(raw, list) or len(raw) < 3:
        return
    points: List[Tuple[float, float]] = []
    for item in raw[:3]:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            points.append((_number(item[0], 0.5), _number(item[1], 0.5)))
    if len(points) != 3:
        return

    def side(px: float, py: float, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return (b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0])

    rows, cols = len(grid), len(grid[0])
    for r, c, u, v in _each_cell(rows, cols):
        d1 = side(u, v, points[0], points[1])
        d2 = side(u, v, points[1], points[2])
        d3 = side(u, v, points[2], points[0])
        if not ((d1 < 0 or d2 < 0 or d3 < 0) and (d1 > 0 or d2 > 0 or d3 > 0)):
            grid[r][c] = color


def _paint_line(grid: List[List[str]], shape: Dict[str, Any], rng: random.Random) -> None:
    color = _letter(shape.get("color"), "W")
    rows, cols = len(grid), len(grid[0])
    x1 = _number(shape.get("x1"), 0.0)
    y1 = _number(shape.get("y1"), 0.0)
    x2 = _number(shape.get("x2"), 1.0)
    y2 = _number(shape.get("y2"), 1.0)
    half = max(_number(shape.get("thickness"), 0.08, lo=0.0) / 2.0, 0.5 / max(rows, cols))
    dx, dy = x2 - x1, y2 - y1
    length_sq = dx * dx + dy * dy
    for r, c, u, v in _each_cell(rows, cols):
        if length_sq == 0:
            distance = math.hypot(u - x1, v - y1)
        else:
            t = max(0.0, min(1.0, ((u - x1) * dx + (v - y1) * dy) / length_sq))
            distance = math.hypot(u - (x1 + t * dx), v - (y1 + t * dy))
        if distance <= half:
            grid[r][c] = color


def _paint_wave(grid: List[List[str]], shape: Dict[str, Any], rng: random.Random) -> None:
    color = _letter(shape.get("color"), "B")
    rows, cols = len(grid), len(grid[0])
    baseline = _number(shape.get("baseline"), 0.5)
    amplitude = _number(shape.get("amplitude"), 0.15, lo=0.0)
    frequency = _number(shape.get("frequency"), 1.5, lo=0.0, hi=16.0)
    phase = _number(shape.get("phase"), 0.0, lo=-math.tau, hi=math.tau)
    half = max(_number(shape.get("thickness"), 0.12, lo=0.0) / 2.0, 0.5 / rows)
    fill = str(shape.get("fill", "band")).strip().lower()
    for r, c, u, v in _each_cell(rows, cols):
        crest = baseline + amplitude * math.sin(math.tau * frequency * u + phase)
        if fill == "below" and v >= crest:
            grid[r][c] = color
        elif fill == "above" and v <= crest:
            grid[r][c] = color
        elif fill not in ("below", "above") and abs(v - crest) <= half:
            grid[r][c] = color


def _paint_noise(grid: List[List[str]], shape: Dict[str, Any], rng: random.Random) -> None:
    colors = _letters(shape.get("colors"), ("W", "K"))
    density = _number(shape.get("density"), 0.15, lo=0.0, hi=1.0)
    rows, cols = len(grid), len(grid[0])
    for r in range(rows):
        for c in range(cols):
            if rng.random() < density:
                grid[r][c] = colors[rng.randrange(len(colors))]


_PAINTERS: Dict[str, Callable[[List[List[str]], Dict[str, Any], random.Random], None]] = {
    "gradient": _paint_gradient,
    "stripes": _paint_stripes,
    "bands": _paint_stripes,
    "rect": _paint_rect,
    "rectangle": _paint_rect,
    "ellipse": _paint_ellipse,
    "circle": _paint_ellipse,
    "rings": _paint_rings,
    "triangle": _paint_triangle,
    "line": _paint_line,
    "wave": _paint_wave,
    "noise": _paint_noise,
}

#: The shape names the prompt advertises, in the order it lists them.
SCENE_SHAPES: Tuple[str, ...] = (
    "gradient",
    "stripes",
    "rect",
    "ellipse",
    "rings",
    "triangle",
    "line",
    "wave",
    "noise",
)


def _mirror(grid: List[List[str]], mode: Any) -> None:
    """Apply symmetry to *grid* in place. Unknown modes are a no-op."""
    name = str(mode or "none").strip().lower()
    rows, cols = len(grid), len(grid[0])
    if name in ("horizontal", "both", "left-right", "lr"):
        for r in range(rows):
            for c in range(cols // 2):
                grid[r][cols - 1 - c] = grid[r][c]
    if name in ("vertical", "both", "top-bottom", "tb"):
        for r in range(rows // 2):
            grid[rows - 1 - r] = list(grid[r])


def rasterise(scene: Dict[str, Any], canvas: Canvas, rng: Optional[random.Random] = None) -> List[List[str]]:
    """Rasterise a scene description onto *canvas*.

    The scene lives in a unit square; every shape is evaluated at each cell's
    normalised centre, so the same description fills a 15x3 Note and a 120x24
    panel equally -- the composition stretches to the board instead of being
    cropped or padded. Shapes paint back to front.
    """
    rng = rng or random.Random()
    rows, cols = canvas.art_rows, canvas.cols
    grid = [[_letter(scene.get("background"), DEFAULT_BACKGROUND)] * cols for _ in range(rows)]

    shapes = scene.get("shapes")
    if not isinstance(shapes, list):
        shapes = []
    for shape in shapes:
        if not isinstance(shape, dict):
            continue
        painter = _PAINTERS.get(str(shape.get("type", "")).strip().lower())
        if painter is None:
            logger.debug("Skipping unknown scene shape: %r", shape.get("type"))
            continue
        painter(grid, shape, rng)

    _mirror(grid, scene.get("mirror"))
    return grid


# ---------------------------------------------------------------------------
# JSON salvage
# ---------------------------------------------------------------------------


def _safe_cut_points(text: str) -> List[Tuple[int, List[str]]]:
    """Positions where *text* could be cut and closed into valid JSON.

    Each entry is ``(index, closers)``: cutting at ``index`` and appending
    ``closers`` balances every bracket still open there.
    """
    points: List[Tuple[int, List[str]]] = []
    stack: List[str] = []
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
                points.append((index + 1, list(stack)))
            continue
        if char == '"':
            in_string = True
        elif char in "[{":
            stack.append("]" if char == "[" else "}")
        elif char in "]}":
            if stack:
                stack.pop()
            points.append((index + 1, list(stack)))
        elif char == ",":
            points.append((index, list(stack)))
    return points


def _loads_with_salvage(text: str) -> Any:
    """Parse *text* as JSON, repairing a truncated tail if need be.

    A response cut off mid-array is the classic large-board failure: the
    model ran out of tokens and the whole piece is lost to a
    ``JSONDecodeError``. Closing the open brackets at the last complete
    value recovers a shorter-but-real composition, which the grid repair
    below then resamples onto the board.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        first_error = exc

    for index, closers in reversed(_safe_cut_points(text)[-64:]):
        candidate = text[:index].rstrip().rstrip(",")
        try:
            parsed = json.loads(candidate + "".join(reversed(closers)))
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            logger.warning("Recovered a truncated JSON response by closing it at offset %d", index)
            return parsed

    raise ArtValidationError(f"Invalid JSON: {first_error}") from first_error


# ---------------------------------------------------------------------------
# Grid repair
# ---------------------------------------------------------------------------


def _resample(seq: Sequence[Any], n: int) -> List[Any]:
    """Nearest-neighbour resample of *seq* to exactly *n* entries."""
    length = len(seq)
    if length == n:
        return list(seq)
    return [seq[min(length - 1, int(i * length / n))] for i in range(n)]


def _repair_grid(raw: Any, canvas: Canvas) -> List[List[str]]:
    """Coerce a model-emitted grid onto *canvas*, or raise if it is too damaged.

    Repairs (silent, logged): rows or columns off by a tolerated amount are
    resampled; an unreadable cell takes the row's dominant colour. Retries
    (raised): a grid outside :data:`SIZE_TOLERANCE`, one with more than
    :data:`MAX_INVALID_CELL_RATIO` unreadable cells, or one whose rows are
    mostly the wrong length -- those are not "a bit off", they are a model
    that did not compose for this board.
    """
    if not isinstance(raw, list):
        raise ArtValidationError(f"'grid' must be a list, got {type(raw).__name__}")

    rows_in: List[List[Any]] = [row for row in raw if isinstance(row, list) and row]
    if not rows_in:
        raise ArtValidationError("'grid' contains no usable rows")

    target_rows, target_cols = canvas.art_rows, canvas.cols
    if not (target_rows * SIZE_TOLERANCE <= len(rows_in) <= target_rows / SIZE_TOLERANCE):
        raise ArtValidationError(
            f"Expected about {target_rows} rows, got {len(rows_in)} — too far off to repair"
        )

    damaged = 0
    invalid = 0
    total = 0
    cleaned: List[List[str]] = []
    for row in rows_in:
        row_damaged = not (target_cols * SIZE_TOLERANCE <= len(row) <= target_cols / SIZE_TOLERANCE)
        letters: List[Optional[str]] = []
        for cell in row:
            total += 1
            letter = cell.strip()[:1].upper() if isinstance(cell, str) and cell.strip() else ""
            if letter in VALID_COLORS:
                letters.append(letter)
            else:
                invalid += 1
                letters.append(None)
        dominant = _dominant(letters)
        if dominant is None:
            # Not a single readable cell in this row; treat it as damaged and
            # let the row-level budget below decide whether to retry.
            row_damaged = True
            dominant = DEFAULT_BACKGROUND
        damaged += int(row_damaged)
        cleaned.append([letter or dominant for letter in letters])

    if total and invalid / total > MAX_INVALID_CELL_RATIO:
        raise ArtValidationError(
            f"{invalid}/{total} cells are not valid colour codes — response is not usable art"
        )
    if damaged / len(rows_in) > MAX_DAMAGED_ROW_RATIO:
        raise ArtValidationError(
            f"{damaged}/{len(rows_in)} rows are the wrong width for a {target_cols}-column board"
        )

    if invalid or damaged or len(rows_in) != target_rows:
        logger.info(
            "Repaired art grid onto %dx%d (rows in: %d, damaged rows: %d, invalid cells: %d)",
            target_cols,
            target_rows,
            len(rows_in),
            damaged,
            invalid,
        )

    resampled = _resample(cleaned, target_rows)
    return [_resample(row, target_cols) for row in resampled]


def _dominant(letters: Sequence[Optional[str]]) -> Optional[str]:
    """Most common readable colour in *letters*, or ``None`` if there is none."""
    counts: Dict[str, int] = {}
    for letter in letters:
        if letter:
            counts[letter] = counts.get(letter, 0) + 1
    if not counts:
        return None
    return max(counts, key=lambda key: counts[key])


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class ArtGenerator:
    """Generates colour-tile art for a board using an OpenAI-compatible LLM.

    The generator holds no geometry: every call takes the :class:`Canvas` it
    is composing for, so one generator serves every board a user owns and no
    cache of it can serve one board's frame to another.

    Args:
        base_url: Base URL for the chat completions endpoint
                  (e.g. ``https://api.openai.com/v1``).
        api_key: Bearer token / API key for authentication.
        model: Model name (e.g. ``gpt-4o-mini``).
        temperature: Sampling temperature (0–2).
        themes: Optional list of custom theme strings. Falls back to
                ``BUILTIN_THEMES`` when empty or ``None``.
        extra_instructions: Additional text appended to the system prompt.
        custom_system_prompt: Replaces the built-in system prompt entirely.
        show_title: Reserve the bottom board row for a title.
        complete: Optional ``complete(messages, temperature=, max_tokens=)``
                  returning the reply text. When given it replaces the HTTP
                  call (``base_url`` and ``api_key`` are then unused); the
                  plugin passes a bridge to FiestaBoard's AI providers. It
                  raises :class:`ArtRequestError` when no reply came back.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float,
        themes: Optional[List[str]] = None,
        extra_instructions: str = "",
        custom_system_prompt: str = "",
        show_title: bool = False,
        complete: Optional[Callable[..., str]] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.themes = themes if themes else BUILTIN_THEMES
        self.extra_instructions = extra_instructions.strip()
        self.custom_system_prompt = custom_system_prompt.strip()
        self.show_title = show_title
        self.complete = complete

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def canvas_for(self, rows: Optional[int], cols: Optional[int]) -> Canvas:
        """Build the canvas for a board of *rows* x *cols*.

        ``None`` for either means "no board is bound", which the platform
        contract says to treat as a Flagship.
        """
        if rows is None or cols is None:
            return Canvas.default(show_title=self.show_title)
        return Canvas(rows=rows, cols=cols, show_title=self.show_title)

    def generate(self, canvas: Optional[Canvas] = None) -> Optional[Dict[str, Any]]:
        """Generate a single art piece for *canvas*.

        Up to :data:`MAX_ATTEMPTS` calls; each retry picks a different theme.
        A response that can be repaired onto the board is used rather than
        retried, so a model that is slightly off-spec still puts art on the
        wall.

        Returns:
            Dict with ``theme``, ``description``, ``grid``, ``title``,
            ``lines`` (one string per board row) and ``art`` (those lines
            joined with newlines), or ``None`` if every attempt failed.
        """
        canvas = canvas or Canvas.default(show_title=self.show_title)
        theme = self._pick_theme()

        for attempt in range(MAX_ATTEMPTS):
            try:
                parsed = self._validate_and_parse(self._call_api(theme, canvas), canvas)
            except (requests.RequestException, ArtRequestError) as exc:
                # A transport failure will not be fixed by a different theme,
                # and the caller has a per-geometry fallback piece for exactly
                # this case.
                logger.error("API request failed: %s", exc)
                return None
            except ArtValidationError as exc:
                logger.warning(
                    "Art validation failed (attempt %d/%d) for theme '%s' on %s: %s",
                    attempt + 1,
                    MAX_ATTEMPTS,
                    theme,
                    canvas.key,
                    exc,
                )
                theme = self._pick_theme(exclude=theme)
                continue

            lines = self.render_lines(parsed["grid"], parsed.get("title", ""), canvas)
            return {
                "theme": parsed["theme"],
                "description": parsed["description"],
                "grid": parsed["grid"],
                "title": parsed.get("title", ""),
                "lines": lines,
                "art": "\n".join(lines),
            }

        logger.error("Art generation failed after %d attempts on %s", MAX_ATTEMPTS, canvas.key)
        return None

    def build_default_system_prompt(self, canvas: Optional[Canvas] = None) -> str:
        """Return the auto-generated system prompt for *canvas*.

        Exposed publicly so callers can show the user what will be sent.
        Defaults to the Flagship canvas when no board is given.
        """
        canvas = canvas or Canvas.default(show_title=self.show_title)
        if canvas.mode == "scene":
            prompt = self._scene_prompt(canvas)
        else:
            prompt = self._grid_prompt(canvas)
        if self.extra_instructions:
            prompt += "\n" + self.extra_instructions
        return prompt

    def render_lines(self, grid: List[List[str]], title: str, canvas: Canvas) -> List[str]:
        """Convert a canvas-sized grid to board rows.

        Each colour is one tile, so a row is exactly ``canvas.cols`` tiles
        wide however many characters the markers take. The title row, when
        enabled, is centred and hard-trimmed to the board width -- a long
        title is shortened, never allowed to overflow the board.
        """
        lines = ["".join(COLOR_LETTERS[cell] for cell in row) for row in grid]
        if canvas.has_title_row:
            lines.append(title.upper()[: canvas.cols].center(canvas.cols))
        return lines

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _pick_theme(self, exclude: Optional[str] = None) -> str:
        """Return a random theme from the pool, optionally excluding one."""
        choices = [t for t in self.themes if t != exclude]
        if not choices:
            choices = self.themes
        return random.choice(choices)

    @staticmethod
    def _color_table() -> str:
        return "\n".join(
            f"  {letter} = {marker.strip('{}')}" for letter, marker in COLOR_LETTERS.items()
        )

    def _title_clause(self, canvas: Canvas) -> Tuple[str, str, str]:
        """``(task suffix, rule line, JSON field)`` for the optional title row."""
        if not canvas.has_title_row:
            return "", "", ""
        return (
            " The bottom board row is reserved for a short title you will provide.",
            f'\n- Include a "title" field: a short label for the piece, at most {canvas.cols} characters.',
            f',\n  "title": "<short label, max {canvas.cols} characters>"',
        )

    def _grid_prompt(self, canvas: Canvas) -> str:
        """Per-cell prompt, used for boards small enough to afford one."""
        task_suffix, title_rule, title_field = self._title_clause(canvas)
        art_rows = canvas.art_rows
        return f"""You are a generative-art composer for a physical split-flap display.
The display uses ONLY the following 8 colors, identified by single capital letters:

{self._color_table()}

Your task is to compose an abstract art piece filling a {art_rows}-row × \
{canvas.cols}-column grid of colored tiles.{task_suffix}

RULES (non-negotiable):
1. Output ONLY valid JSON — no markdown fences, no prose before or after.
2. The "grid" array must contain EXACTLY {art_rows} rows.
3. Every row must contain EXACTLY {canvas.cols} elements.
4. Every element must be one of: R O Y G B V W K
5. Use AT LEAST 2 distinct colors and AT MOST 6 distinct colors per piece.
6. The piece must have intentional visual structure — a focal point, balance, \
rhythm, or clear compositional logic. It must look like deliberate art, \
not random noise.{title_rule}

AESTHETIC GUIDANCE:
- The board is {canvas.cols} tiles wide and {art_rows} tall; compose for that \
aspect ratio rather than for a square.
- Choose 2–4 dominant colors and 0–2 accent colors.
- Avoid checkerboard patterns unless the theme calls for it.
- Think in regions, gradients, curves, or geometric forms rather than \
per-cell random choices.
- The piece should evoke the theme visually, not just use its name as a label.

OUTPUT FORMAT (strict JSON, no other text):
{{
  "theme": "<the theme in a few words>",
  "description": "<one sentence describing the visual impression>",
  "grid": [
    ["{canvas.cols} single-letter codes for row 1"],
    ...
    ["{canvas.cols} single-letter codes for row {art_rows}"]
  ]{title_field}
}}"""

    def _scene_prompt(self, canvas: Canvas) -> str:
        """Compact scene prompt, used for boards too large for per-cell art.

        The model never sees a cell count it has to fill: it describes the
        composition once, in normalised coordinates, and this module
        rasterises. That is what keeps a 120x24 panel a ~1k-token request
        instead of a ~12k-token one that gets truncated.
        """
        task_suffix, title_rule, title_field = self._title_clause(canvas)
        art_rows = canvas.art_rows
        return f"""You are a generative-art composer for a physical split-flap display.
The display uses ONLY the following 8 colors, identified by single capital letters:

{self._color_table()}

The board is {art_rows} rows × {canvas.cols} columns of colored tiles — an \
aspect ratio of roughly {canvas.cols}:{art_rows} (width:height). You do NOT \
emit individual tiles. You describe the composition as a SCENE and the display \
software rasterises it onto the board.{task_suffix}

Coordinates are NORMALISED: x = 0 is the left edge, x = 1 the right edge, \
y = 0 the top, y = 1 the bottom, whatever the board's real size. Compose \
deliberately for the stated aspect ratio — a wide short board wants a \
horizontal composition, a tall narrow one a vertical composition.

RULES (non-negotiable):
1. Output ONLY valid JSON — no markdown fences, no prose before or after.
2. "shapes" is an ordered list painted back to front: later shapes cover earlier ones.
3. Use between 2 and 10 shapes. Fewer, larger shapes read better than many small ones.
4. Every color is one of: R O Y G B V W K
5. Use AT LEAST 2 distinct colors and AT MOST 6 distinct colors per piece.
6. Every coordinate, size, thickness and density is a number between 0 and 1.
7. The piece must have intentional visual structure — a focal point, balance, \
rhythm, or clear compositional logic.{title_rule}

SHAPE VOCABULARY (use only these types):
{{"type": "gradient", "colors": ["B", "V", "R"], "direction": "vertical|horizontal|diagonal|radial"}}
{{"type": "stripes", "colors": ["R", "O"], "direction": "vertical|horizontal|diagonal", "count": 6}}
{{"type": "rect", "color": "Y", "x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}}
{{"type": "ellipse", "color": "W", "cx": 0.5, "cy": 0.5, "rx": 0.2, "ry": 0.3}}
{{"type": "rings", "colors": ["B", "W"], "cx": 0.5, "cy": 0.5, "count": 5}}
{{"type": "triangle", "color": "G", "points": [[0.0, 1.0], [0.5, 0.2], [1.0, 1.0]]}}
{{"type": "line", "color": "W", "x1": 0.0, "y1": 0.1, "x2": 1.0, "y2": 0.9, "thickness": 0.06}}
{{"type": "wave", "color": "B", "baseline": 0.6, "amplitude": 0.15, "frequency": 2, \
"phase": 0, "thickness": 0.1, "fill": "band|above|below"}}
{{"type": "noise", "colors": ["K", "V"], "density": 0.15}}

OUTPUT FORMAT (strict JSON, no other text):
{{
  "theme": "<the theme in a few words>",
  "description": "<one sentence describing the visual impression>",
  "background": "<one color letter painted before any shape>",
  "mirror": "none|horizontal|vertical|both",
  "shapes": [ <2 to 10 shapes from the vocabulary above> ]{title_field}
}}"""

    def _build_messages(self, theme: str, canvas: Canvas) -> List[Dict[str, str]]:
        """Build the messages list for the chat completions request."""
        if self.custom_system_prompt:
            system = self.custom_system_prompt
        else:
            system = self.build_default_system_prompt(canvas)

        user = (
            f"Compose an art piece with this theme: {theme}\n\n"
            "Remember: output ONLY the JSON object described above."
        )

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    @staticmethod
    def max_tokens_for(canvas: Canvas) -> int:
        """Output-token budget for *canvas*.

        Always sent. Left unset, the endpoint's own default decides, and a
        response that stops mid-array is a parse failure rather than a
        smaller picture. A scene is a constant handful of shapes; a grid
        scales with the cells it has to name.
        """
        if canvas.mode == "scene":
            return SCENE_MAX_TOKENS
        estimate = canvas.cells * GRID_TOKENS_PER_CELL + GRID_TOKEN_OVERHEAD
        return max(MIN_MAX_TOKENS, min(MAX_MAX_TOKENS, estimate))

    # ------------------------------------------------------------------
    # API + parsing
    # ------------------------------------------------------------------

    def _call_api(self, theme: str, canvas: Canvas) -> str:
        """POST to the chat completions endpoint and return the content string."""
        if self.complete is not None:
            return self.complete(
                self._build_messages(theme, canvas),
                temperature=self.temperature,
                max_tokens=self.max_tokens_for(canvas),
            )
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": self._build_messages(theme, canvas),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens_for(canvas),
        }

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()

        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ArtValidationError(f"Unexpected API response structure: {exc}") from exc

        return content

    def _validate_and_parse(self, raw: str, canvas: Optional[Canvas] = None) -> Dict[str, Any]:
        """Parse an LLM response into a canvas-sized piece.

        Accepts either emission format regardless of which one was asked
        for -- a model that returns a grid when a scene was requested (or the
        reverse) is still giving us a composition, and discarding it would
        cost a retry for nothing.

        Returns a dict with ``theme``, ``description``, ``grid`` (exactly
        ``canvas.art_rows`` x ``canvas.cols`` colour letters) and, when the
        canvas has a title row, ``title``. Raises
        :class:`ArtValidationError` when the response is past repair.
        """
        canvas = canvas or Canvas.default(show_title=self.show_title)

        cleaned = _strip_fences(raw.strip() if isinstance(raw, str) else "")
        parsed = _loads_with_salvage(cleaned)
        if not isinstance(parsed, dict):
            raise ArtValidationError("Response is not a JSON object")

        has_grid = isinstance(parsed.get("grid"), list) and parsed["grid"]
        has_scene = isinstance(parsed.get("shapes"), list) and parsed["shapes"]
        if not has_grid and not has_scene:
            raise ArtValidationError(
                "Response contains neither a 'grid' of cells nor a list of scene 'shapes'"
            )

        # Prefer the format this canvas asked for when the model sent both.
        if has_scene and (canvas.mode == "scene" or not has_grid):
            grid = rasterise(parsed, canvas)
        else:
            grid = _repair_grid(parsed["grid"], canvas)

        distinct = {cell for row in grid for cell in row}
        if len(distinct) < 2:
            raise ArtValidationError(
                f"Art uses only {len(distinct)} color(s); minimum is 2"
            )

        result: Dict[str, Any] = {
            "theme": _text(parsed.get("theme"), MAX_THEME_CHARS),
            "description": _text(parsed.get("description"), MAX_DESCRIPTION_CHARS),
            "grid": grid,
        }
        if canvas.has_title_row:
            title = parsed.get("title")
            if not isinstance(title, str) or not title.strip():
                # A missing title is not worth a retry: the piece is fine and
                # the theme is a perfectly good label for it.
                title = result["theme"]
            result["title"] = title.strip()[: canvas.cols]
        return result


def _strip_fences(text: str) -> str:
    """Remove an accidental markdown code fence around *text*."""
    if not text.startswith("```"):
        return text
    lines = text.splitlines()[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _text(value: Any, limit: int) -> str:
    """Coerce *value* to a trimmed string of at most *limit* characters.

    Bounded here so the manifest's declared ``max_length`` stays true
    whatever the model decides to write.
    """
    if not isinstance(value, str):
        return ""
    return value.strip()[:limit]
