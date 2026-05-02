#!/usr/bin/env python3
"""Generate screenshot images for the Generative AI Art plugin docs.

Renders several hand-crafted sample art grids as PNG images, saved to
docs/black/ and docs/ (the main board-display.png).

Usage:
    python3 scripts/generate_images.py
"""

import sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw
except ImportError:
    print("Pillow is required: pip install Pillow")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Colour palette — matches official FiestaBoard hex values
# ---------------------------------------------------------------------------
COLOR_HEX = {
    "R": "#eb4034",  # red
    "O": "#f5a623",  # orange
    "Y": "#f8e71c",  # yellow
    "G": "#7ed321",  # green
    "B": "#4a90d9",  # blue
    "V": "#9b59b6",  # violet
    "W": "#ffffff",  # white
    "K": "#1a1a1a",  # black
}

# ---------------------------------------------------------------------------
# Sample art grids  (6 rows × 22 cols for Flagship)
# ---------------------------------------------------------------------------

# 1. Aurora borealis — vertical curtains of violet/green on deep blue
AURORA = [
    list("BVVBGGGBBVVBGGBVVBGGBV"),
    list("VVBGGGBVVBGGGBVVBGGGBV"),
    list("BGGGBVVBGGGVVBBGGGVVBB"),
    list("GGGVVBBGGGVVBGGGGVVBBG"),
    list("BBVVBGGGBBVVBGGBBVVBGG"),
    list("BBBVVBBBBBVVBBBBBVVBBB"),
]

# 2. Sunset — warm layered bands fading to night
SUNSET = [
    list("BBBBBBBBBBBBBBBBBBBBBB"),
    list("BBBBBVVVVVVVVVVVBBBBB"),  # wait — need exactly 22 cols
    list("VVVVVVRRRRRRRRRVVVVVV"),
    list("RRRRRRRROOOOOOORRRRRRR"),
    list("OOOOOOOOOYYYOOOOOOOOOO"),
    list("YYYYYYYYYYYYYYYYYYYYYYY"),
]

# Fix row lengths
def _fix(grid):
    return [row[:22] for row in grid]

SUNSET = _fix(SUNSET)
# Rebuild properly
SUNSET = [
    list("BBBBBBBBBBBBBBBBBBBBBB"),
    list("BBBBBBBVVVVVVVVBBBBBBB"),
    list("VVVVVVVVRRRRRRRVVVVVVV"),
    list("RRRRRRRROOOOOOORRRRRRR") + ["R"],
    list("OOOOOOOOOYYYYOOOOOOOOO") + ["O"],
    list("YYYYYYYYYYYYYYYYYYYYYYY")[:22],
]

# 3. Mondrian-style geometric colour-block
MONDRIAN = [
    list("RRRRRRRRRBBBBBBBBBBBBR"),  # 22? Let me be precise
]

def make_grid(rows_str_list):
    """Build a grid from a list of exactly-22-char strings."""
    return [list(row) for row in rows_str_list]


AURORA = make_grid([
    "BVVBGGGBBVVBGGBVVBGGBV",
    "VVBGGGBVVBGGGBVVBGGGBV",
    "BGGGBVVBGGGVVBBGGGVVBB",
    "GGGVVBBGGGVVBGGGGVVBBG",
    "BBVVBGGGBBVVBGGBBVVBGG",
    "BBBVVBBBBBVVBBBBBVVBBB",
])

SUNSET = make_grid([
    "BBBBBBBBBBBBBBBBBBBBBB",
    "BBBBBBBVVVVVVVBBBBBBBB",
    "VVVVVVVVRRRRRRRVVVVVVV",
    "RRRRRRRROOOOOOORRRRRRRR"[:22],
    "OOOOOOOOYYYYYYYYOOOOOOO"[:22],
    "YYYYYYYYYYYYYYYYYYYYYYYY"[:22],
])

MONDRIAN = make_grid([
    "RRRRRRRRRRBBBBBBBBBBBB",
    "RRRRRRRRRRBBBBBBBBBBBB",
    "RRRRRRRRRRKKKKKKKKKKKK",
    "YYYYYYYYYYKKKKKKKKKKKK",
    "YYYYYYYYYYYYYYYYYYYYYYY"[:22],
    "YYYYYYYYYYYYYYYYYYYYYYY"[:22],
])

# 4. Mountain silhouette — cool sky, dark peaks, warm ground
MOUNTAIN = make_grid([
    "BBBBBBBBBBBBBBBBBBBBBB",
    "BBBBBBBBBBBBBBBBBBBBBB",
    "BBBBBBBKKKBBBBBKBBBBBB",
    "BBBBKKKKKKKKKKKKKBBBBB",
    "BKKKKKKKKKKKKKKKKKKKBB",
    "GGGGGGGGGGGGGGGGGGGGGG",
])

# 5. Concentric rings from center
RINGS = make_grid([
    "KKKKKKKKKKKKKKKKKKKKKK",
    "KBBBBBBBBBBBBBBBBBBBBK",
    "KBVVVVVVVVVVVVVVVVVBK",
    "KBVGGGGGGGGGGGGGGGVBK"[:22],
    "KBVVVVVVVVVVVVVVVVVBK",
    "KBBBBBBBBBBBBBBBBBBBBK",
])

# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

def hex_to_rgb(hex_color: str):
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def render_grid(grid, tile_size=40, gap=3, bg="#0d0d0d"):
    """Render a colour grid as a PIL Image."""
    rows = len(grid)
    cols = max(len(r) for r in grid)

    w = cols * tile_size + (cols - 1) * gap
    h = rows * tile_size + (rows - 1) * gap

    img = Image.new("RGB", (w, h), color=hex_to_rgb(bg))
    draw = ImageDraw.Draw(img)

    m_top, m_bot, m_h = 3, 4, 1  # tile margins

    for ri, row in enumerate(grid):
        for ci, code in enumerate(row):
            if code not in COLOR_HEX:
                continue
            x = ci * (tile_size + gap)
            y = ri * (tile_size + gap)

            cx = x + m_h
            cy = y + m_top
            cw = tile_size - m_h * 2
            ch = tile_size - (m_top + m_bot)
            rgb = hex_to_rgb(COLOR_HEX[code])

            draw.rectangle([cx, cy, cx + cw - 1, cy + ch - 1], fill=rgb)

            # Highlights / shadows
            draw.rectangle([cx, cy, cx + cw - 1, cy + 1],
                           fill=tuple(min(255, c + 30) for c in rgb))
            draw.rectangle([cx, cy, cx + 1, cy + ch - 1],
                           fill=tuple(min(255, c + 20) for c in rgb))
            draw.rectangle([cx, cy + ch - 2, cx + cw - 1, cy + ch - 1],
                           fill=tuple(max(0, c - 40) for c in rgb))
            draw.rectangle([cx + cw - 2, cy, cx + cw - 1, cy + ch - 1],
                           fill=tuple(max(0, c - 30) for c in rgb))

            # Centre split-flap crease
            center_y = cy + ch // 2
            draw.rectangle([cx, center_y, cx + cw - 1, center_y],
                           fill=tuple(max(0, c - 20) for c in rgb))

    return img


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SAMPLES = [
    ("aurora",    AURORA,   "Aurora Borealis"),
    ("sunset",    SUNSET,   "Sunset"),
    ("mondrian",  MONDRIAN, "Mondrian"),
    ("mountain",  MOUNTAIN, "Mountain Silhouette"),
    ("rings",     RINGS,    "Concentric Rings"),
]

def main():
    out_root = Path(__file__).parent.parent / "docs"
    black_dir = out_root / "black"
    black_dir.mkdir(parents=True, exist_ok=True)

    for name, grid, label in SAMPLES:
        img = render_grid(grid)
        path = black_dir / f"generative-ai-art-{name}.png"
        img.save(path)
        print(f"  Saved {path}")

    # board-display.png — use the sunset as the hero shot
    hero = render_grid(SUNSET)
    hero_path = out_root / "board-display.png"
    hero.save(hero_path)
    print(f"  Saved {hero_path}")

    print("Done.")


if __name__ == "__main__":
    main()
