"""Manifest checks that keep declared bounds honest against the code.

The page editor sizes templates from ``max_length``, so a variable that
renders longer than it declares makes the editor's fit warnings wrong. These
tests tie each declared bound to the constant the code actually enforces.
"""

import json
from pathlib import Path

import pytest
from src.devices import MAX_NOTES_PER_AXIS, NOTE_COLS, NOTE_ROWS
from src.plugins.previews import validate_previews
from src.text_to_board import count_tiles

from plugins.generative_ai_art.source import (
    MAX_DESCRIPTION_CHARS,
    MAX_THEME_CHARS,
    SCENE_SHAPES,
    _PAINTERS,
    ArtGenerator,
    Canvas,
)

_MANIFEST_PATH = Path(__file__).resolve().parent.parent / "manifest.json"
MANIFEST = json.loads(_MANIFEST_PATH.read_text())
SIMPLE = MANIFEST["variables"]["simple"]

#: The largest board FiestaBoard supports: an 8x8 note array.
MAX_BOARD_COLS = MAX_NOTES_PER_AXIS * NOTE_COLS
MAX_BOARD_ROWS = MAX_NOTES_PER_AXIS * NOTE_ROWS


def test_art_max_length_matches_the_largest_board():
    """``art`` is one tile per cell plus the newlines between rows."""
    expected = MAX_BOARD_COLS * MAX_BOARD_ROWS + (MAX_BOARD_ROWS - 1)
    assert SIMPLE["art"]["max_length"] == expected


def test_art_at_the_largest_board_fits_its_declared_length():
    generator = ArtGenerator(
        base_url="https://api.example.com/v1", api_key="sk-test", model="m", temperature=1.0
    )
    canvas = Canvas(rows=MAX_BOARD_ROWS, cols=MAX_BOARD_COLS)
    grid = [["B" if (r + c) % 2 else "R" for c in range(canvas.cols)] for r in range(canvas.art_rows)]
    art = "\n".join(generator.render_lines(grid, "", canvas))
    assert count_tiles(art) == SIMPLE["art"]["max_length"]


@pytest.mark.parametrize(
    "name,constant",
    [("theme", MAX_THEME_CHARS), ("description", MAX_DESCRIPTION_CHARS)],
)
def test_metadata_bounds_match_the_code(name, constant):
    assert SIMPLE[name]["max_length"] == constant


def test_no_board_size_setting():
    """The platform knows the board; one config must serve every board."""
    assert "device_type" not in MANIFEST["settings_schema"]["properties"]
    names = {var["name"] for var in MANIFEST["env_vars"]}
    assert "GENERATIVE_AI_ART_DEVICE_TYPE" not in names


def test_previews_are_valid_and_cover_note_array():
    assert validate_previews(MANIFEST["previews"]) == []
    shapes = {p.get("device_type") for p in MANIFEST["previews"]}
    assert {"flagship", "note", "note_array"} <= shapes


def test_every_advertised_shape_is_paintable():
    """The prompt may only offer shapes the rasteriser can actually draw."""
    generator = ArtGenerator(
        base_url="https://api.example.com/v1", api_key="sk-test", model="m", temperature=1.0
    )
    prompt = generator.build_default_system_prompt(Canvas(rows=24, cols=120))
    for shape in SCENE_SHAPES:
        assert shape in _PAINTERS, f"{shape} is advertised but has no painter"
        assert f'"type": "{shape}"' in prompt, f"{shape} is missing from the scene prompt"
