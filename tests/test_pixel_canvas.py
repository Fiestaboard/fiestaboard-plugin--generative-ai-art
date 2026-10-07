"""Full-colour pixel art for LED pixel-matrix displays: the ``canvas`` variable.

The model calls are mocked (``requests.post``); nothing here touches the
network. Where FiestaBoard core ships the canvas engine (``src.canvas``), every
emitted content object is also validated, resolved and rasterised by core's
own code, so "valid" means what the board will actually accept. On a core
without it those extra checks are skipped and the structural checks still run.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from src.devices import BoardContext
from src.text_to_board import count_tiles

from plugins.generative_ai_art import MAX_REMEMBERED_PIECES, GenerativeAiArtPlugin
from plugins.generative_ai_art.pixel_canvas import (
    CANVAS_MAX_TOKENS,
    DEFAULT_CANVAS_SIZE,
    MAX_CANVAS_SHAPES,
    MAX_PROMPT_SHAPES,
    PixelTarget,
    canvas_key,
    canvas_system_prompt,
    fallback_content,
    pixel_target,
    sample_grid,
    scene_to_content,
    tiles_content,
)
from plugins.generative_ai_art.source import ArtValidationError, Canvas

try:  # core with the pixel-canvas engine (FiestaBoard PR #2228 and later)
    from src.canvas.evaluate import resolve_content
    from src.canvas.models import CanvasContent
    from src.canvas.raster import render_content
except ImportError:  # an older core: structural checks only
    CanvasContent = None

POST = "plugins.generative_ai_art.source.requests.post"

# ---------------------------------------------------------------------------
# Displays and boards
# ---------------------------------------------------------------------------


class PixelDisplay:
    """A display as core PR 3 describes it: feature ``pixels`` + pixel size."""

    technology = "led_matrix"
    color = "rgb"

    def __init__(self, width=None, height=None, color="rgb", key="divoom_pixoo64|led_3x5|rgb|fill|3x5"):
        if width is not None:
            self.width = width
        if height is not None:
            self.height = height
        self.color = color
        self.key = key

    def supports(self, feature):
        if feature == "pixels":
            return True
        return feature == "rgb" and self.color == "rgb"


class OldCoreDisplay:
    """A display from a core before PR 3: ``supports('pixels')`` raises."""

    def __init__(self, technology="led_matrix", color="rgb", key="divoom_pixoo64|led_3x5|rgb|fill|3x5"):
        self.technology = technology
        self.color = color
        self.key = key

    def supports(self, feature):
        if feature == "pixels":
            raise ValueError("Unknown display feature 'pixels'")
        return False


def make_board(device_type, rows, cols, display):
    """A BoardContext carrying *display*.

    Cores without ``BoardContext.display`` (9.13 and earlier) get a stand-in
    with the same attributes, playing the board a newer core would bind, so
    the plugin's behaviour is still exercised on the core CI checks out.
    """
    try:
        return BoardContext(device_type=device_type, rows=rows, cols=cols, display=display)
    except TypeError:
        return SimpleNamespace(device_type=device_type, rows=rows, cols=cols, display=display,
                               width=cols, height=rows)


def pixel_board(width=64, height=64, rows=8, cols=10, **kwargs):
    return make_board("panel", rows, cols, PixelDisplay(width, height, **kwargs))


SPLIT_FLAP = BoardContext.from_device_type("flagship")

# ---------------------------------------------------------------------------
# Model responses
# ---------------------------------------------------------------------------

PIXEL_SCENE = {
    "theme": "desert dusk",
    "description": "A violet dusk over dunes with a low golden sun.",
    "title": "DUSK",
    "background": "#140c2a",
    "shapes": [
        {"type": "gradient", "x": 0, "y": 0, "w": 64, "h": 40, "from": "#2b1055", "to": "#ff7e5f", "angle": 90},
        {"type": "circle", "cx": 44, "cy": 30, "r": 8, "fill": "#ffd36e"},
        {"type": "polygon", "points": [[0, 64], [0, 44], [18, 34], [34, 46], [50, 38], [64, 44], [64, 64]],
         "fill": "#3a1f2b"},
        {"type": "line", "x1": 0, "y1": 50, "x2": 64, "y2": 50, "stroke": "#ffb36b", "width": 1},
        {"type": "noise", "colors": ["#ffffff"], "density": 0.05, "x": 0, "y": 0, "w": 64, "h": 24},
    ],
}

GRID = {
    "theme": "test theme",
    "description": "A test composition",
    "grid": [["B" if (r + c) % 2 == 0 else "R" for c in range(22)] for r in range(6)],
}


def _response(payload):
    response = MagicMock(status_code=200)
    content = payload if isinstance(payload, str) else json.dumps(payload)
    response.json.return_value = {"choices": [{"message": {"content": content}}]}
    response.raise_for_status.return_value = None
    return response


def _plugin(**config) -> GenerativeAiArtPlugin:
    plugin = GenerativeAiArtPlugin({"id": "generative_ai_art", "name": "x", "version": "1.0.0"})
    plugin.config = {
        "enabled": True,
        "api_key": "sk-test",
        "api_base_url": "https://api.example.com/v1",
        "model": "gpt-4o-mini",
        "temperature": 1.0,
        "refresh_seconds": 300,
        **config,
    }
    return plugin


def _fetch(plugin, board, payload=PIXEL_SCENE):
    with patch(POST, return_value=_response(payload)) as post:
        with plugin._bound_board(board):
            result = plugin.fetch_data()
    return result, post


def _system_prompt(post) -> str:
    return post.call_args.kwargs["json"]["messages"][0]["content"]


# ---------------------------------------------------------------------------
# Content validity: structural, plus core's own validator when it exists
# ---------------------------------------------------------------------------


def assert_valid_content(content, size=None):
    """Structural checks always; core's validate + resolve + rasterise when present."""
    assert isinstance(content, dict)
    json.loads(json.dumps(content))  # plain JSON all the way down
    assert set(content) <= {"size", "background", "palette", "shapes", "pixels"}
    w, h = content["size"]
    assert 1 <= w <= 128 and 1 <= h <= 128
    if size is not None:
        assert content["size"] == list(size)
    assert len(content.get("shapes", [])) <= 256
    assert len(content.get("palette", {})) <= 62
    assert len(content.get("pixels", [])) <= 128
    for shape in content.get("shapes", []):
        assert shape["type"] in {"rect", "circle", "ellipse", "line", "polygon", "text", "gradient"}
        for value in shape.values():
            assert not (isinstance(value, str) and "{{" in value), "no template expressions from a model"
    if CanvasContent is not None:
        model = CanvasContent.model_validate(content)
        resolved, issues = resolve_content(model, {})
        assert issues == []
        rgba = render_content(resolved, w, h)
        assert len(rgba) == w * h * 4


def distinct_colours(content) -> int:
    """Distinct opaque colours core actually paints (None without core)."""
    if CanvasContent is None:
        return None
    resolved, _ = resolve_content(CanvasContent.model_validate(content), {})
    w, h = content["size"]
    rgba = render_content(resolved, w, h)
    return len({rgba[i:i + 4] for i in range(0, len(rgba), 4) if rgba[i + 3]})


# ---------------------------------------------------------------------------
# Which displays get full-colour pixel art
# ---------------------------------------------------------------------------


class TestPixelTarget:
    def test_display_supporting_pixels_is_a_pixel_target_at_its_size(self):
        target = pixel_target(pixel_board(32, 16))
        assert target == PixelTarget(width=32, height=16)

    def test_rgb_led_matrix_on_an_older_core_falls_back_to_64_square(self):
        board = make_board("panel", 8, 10, OldCoreDisplay())
        assert pixel_target(board) == PixelTarget(*DEFAULT_CANVAS_SIZE)
        assert DEFAULT_CANVAS_SIZE == (64, 64)

    def test_real_core_display_profile_rgb_led_is_a_pixel_target(self):
        profile = pytest.importorskip("src.outputs.display_profile")
        board = make_board("panel", 8, 10, profile.DisplayProfile(technology="led_matrix", color="rgb"))
        target = pixel_target(board)
        assert target is not None and target.size == (64, 64)

    @pytest.mark.parametrize(
        "display",
        [None, OldCoreDisplay(technology="split_flap", color="tiles"), OldCoreDisplay(color="mono"),
         OldCoreDisplay(technology="screen", color="rgb")],
        ids=["no-display", "split-flap", "mono-led-old-core", "screen"],
    )
    def test_other_displays_are_not_pixel_targets(self, display):
        board = make_board("panel", 8, 10, display)
        assert pixel_target(board) is None

    def test_no_board_is_not_a_pixel_target(self):
        assert pixel_target(None) is None

    def test_mono_pixel_display_is_a_mono_target(self):
        target = pixel_target(pixel_board(64, 32, color="mono"))
        assert target.mono and target.size == (64, 32)

    def test_oversized_display_is_scaled_into_the_content_limit_keeping_aspect(self):
        assert pixel_target(pixel_board(256, 64)).size == (128, 32)

    @pytest.mark.parametrize("w,h", [(0, 64), (-5, 10), ("64", 64), (True, 64), (None, 64)])
    def test_unusable_pixel_size_falls_back_to_default(self, w, h):
        assert pixel_target(pixel_board(w, h)).size == DEFAULT_CANVAS_SIZE


# ---------------------------------------------------------------------------
# The canvas variable on pixel displays
# ---------------------------------------------------------------------------


class TestCanvasOnPixelDisplays:
    def test_canvas_is_valid_full_colour_content_at_the_display_size(self):
        result, post = _fetch(_plugin(), pixel_board(64, 64))
        assert result.available
        content = result.data["canvas"]
        assert_valid_content(content, size=(64, 64))
        assert post.call_count == 1  # one model call feeds both canvas and art
        types = [shape["type"] for shape in content["shapes"]]
        assert types[:4] == ["gradient", "circle", "polygon", "line"]
        assert content["background"] == "#140c2a"
        assert content["pixels"], "noise becomes scattered pixels"
        colours = distinct_colours(content)
        if colours is not None:
            assert colours > 8, "full colour, not the 8 board colours"

    def test_canvas_size_follows_the_display(self):
        result, _ = _fetch(_plugin(), pixel_board(32, 16))
        assert result.data["canvas"]["size"] == [32, 16]

    def test_canvas_size_defaults_to_64_square_when_the_display_has_no_size(self):
        board = make_board("panel", 8, 10, OldCoreDisplay())
        result, post = _fetch(_plugin(), board)
        assert result.data["canvas"]["size"] == [64, 64]
        assert "64 × 64" in _system_prompt(post)

    def test_prompt_is_for_a_full_colour_led_pixel_display(self):
        _, post = _fetch(_plugin(), pixel_board(32, 16))
        prompt = _system_prompt(post)
        assert "LED" in prompt
        assert "32 × 16" in prompt
        assert "#rrggbb" in prompt
        assert "split-flap" not in prompt
        assert f"{MAX_PROMPT_SHAPES}" in prompt
        assert post.call_args.kwargs["json"]["max_tokens"] == CANVAS_MAX_TOKENS

    def test_mono_display_prompt_says_monochrome(self):
        _, post = _fetch(_plugin(), pixel_board(64, 32, color="mono"))
        assert "monochrome" in _system_prompt(post).lower()

    def test_extra_instructions_reach_the_canvas_prompt(self):
        _, post = _fetch(_plugin(extra_instructions="favour teal"), pixel_board())
        assert _system_prompt(post).rstrip().endswith("favour teal")

    def test_art_still_fills_the_board_on_a_pixel_display(self):
        board = pixel_board(64, 64, rows=8, cols=10)
        result, _ = _fetch(_plugin(), board)
        lines = result.formatted_lines
        assert len(lines) == 8
        assert all(count_tiles(line) == 10 for line in lines)
        assert result.data["art"] == "\n".join(lines)
        assert result.data["theme"] == "desert dusk"

    def test_art_on_a_pixel_display_honours_the_title_row(self):
        result, _ = _fetch(_plugin(show_title=True), pixel_board(64, 64, rows=8, cols=10))
        assert len(result.formatted_lines) == 8
        assert result.formatted_lines[-1].strip() == "DUSK"

    def test_a_grid_reply_on_a_pixel_display_still_yields_a_canvas(self):
        """A custom prompt written for tiles may answer with a grid."""
        result, _ = _fetch(_plugin(custom_system_prompt="tiles please"), pixel_board(), payload=GRID)
        content = result.data["canvas"]
        assert_valid_content(content, size=(22, 6))
        assert content["pixels"][0] == "BRBRBRBRBRBRBRBRBRBRBR"

    def test_custom_system_prompt_is_used_on_pixel_displays(self):
        _, post = _fetch(_plugin(custom_system_prompt="my own prompt"), pixel_board())
        assert _system_prompt(post) == "my own prompt"


# ---------------------------------------------------------------------------
# Prompts are display-aware for tile art too
# ---------------------------------------------------------------------------


class TestDisplayAwareTilePrompts:
    def test_split_flap_prompt_says_split_flap(self):
        _, post = _fetch(_plugin(), SPLIT_FLAP, payload=GRID)
        assert "physical split-flap display" in _system_prompt(post)

    def test_no_display_is_treated_as_split_flap(self):
        board = make_board("flagship", 6, 22, None)
        _, post = _fetch(_plugin(), board, payload=GRID)
        assert "physical split-flap display" in _system_prompt(post)

    @pytest.mark.parametrize(
        "display,word",
        [(OldCoreDisplay(color="mono"), "LED"), (OldCoreDisplay(technology="screen", color="rgb"), "screen")],
        ids=["mono-led", "screen"],
    )
    def test_non_split_flap_tile_prompt_names_the_real_display(self, display, word):
        board = make_board("panel", 6, 22, display)
        _, post = _fetch(_plugin(), board, payload=GRID)
        prompt = _system_prompt(post)
        assert "split-flap" not in prompt
        assert word in prompt

    def test_large_scene_prompt_is_display_aware_too(self):
        board = make_board("panel", 24, 120, OldCoreDisplay(color="mono"))
        scene = {"theme": "t", "description": "d", "background": "K",
                 "shapes": [{"type": "rect", "color": "W", "x": 0, "y": 0, "w": 0.5, "h": 1}]}
        _, post = _fetch(_plugin(), board, payload=scene)
        prompt = _system_prompt(post)
        assert "SCENE" in prompt and "split-flap" not in prompt


# ---------------------------------------------------------------------------
# Split-flap boards: art unchanged, canvas built from the tiles
# ---------------------------------------------------------------------------


class TestSplitFlapUnchanged:
    def test_art_is_the_tile_piece_exactly_as_before(self):
        plugin = _plugin()
        result, post = _fetch(plugin, SPLIT_FLAP, payload=GRID)
        generator = plugin._generator
        canvas = Canvas(rows=6, cols=22)
        expected = generator.render_lines(GRID["grid"], "", canvas)
        assert result.formatted_lines == expected
        assert result.data["art"] == "\n".join(expected)
        assert _system_prompt(post) == generator.build_default_system_prompt(canvas)
        assert post.call_args.kwargs["json"]["max_tokens"] == generator.max_tokens_for(canvas)

    def test_canvas_on_split_flap_is_the_tile_art_as_pixels(self):
        result, _ = _fetch(_plugin(), SPLIT_FLAP, payload=GRID)
        content = result.data["canvas"]
        assert_valid_content(content, size=(22, 6))
        assert content["pixels"] == ["".join(row) for row in GRID["grid"]]
        assert content["palette"]["B"] == "blue" and content["palette"]["R"] == "red"

    def test_canvas_from_tiles_excludes_the_title_row(self):
        grid = GRID["grid"][:5]
        result, _ = _fetch(_plugin(show_title=True), SPLIT_FLAP, payload={**GRID, "grid": grid, "title": "HI"})
        assert result.data["canvas"]["size"] == [22, 5]

    def test_largest_panel_tiles_fit_the_content_limits(self):
        grid = [["B" if (r * c) % 3 else "Y" for c in range(128)] for r in range(96)]
        assert_valid_content(tiles_content(grid), size=(128, 96))

    def test_split_flap_failure_without_a_piece_is_still_unavailable(self):
        plugin = _plugin()
        with patch(POST, return_value=_response("not json at all")):
            with plugin._bound_board(SPLIT_FLAP):
                result = plugin.fetch_data()
        assert not result.available
        assert result.data is None


# ---------------------------------------------------------------------------
# Legacy scene primitives become canvas shapes
# ---------------------------------------------------------------------------

T = PixelTarget(64, 64)


def _convert(*shapes, **scene):
    content = scene_to_content({"background": "#000000", "shapes": list(shapes), **scene}, T)
    assert_valid_content(content, size=(64, 64))
    return content


class TestLegacyPrimitives:
    # "direction" is the way the colour changes, as in the tile rasteriser
    # (source._axis): vertical stripes are bands stacked top to bottom.
    def test_vertical_stripes_become_full_width_bands(self):
        content = _convert({"type": "stripes", "colors": ["#ff0000", "#00ff00"], "direction": "vertical", "count": 4})
        rects = [s for s in content["shapes"] if s["type"] == "rect"]
        assert len(rects) == 4
        assert [r["fill"] for r in rects] == ["#ff0000", "#00ff00"] * 2
        assert [(r["y"], r["h"], r["w"]) for r in rects] == [(0, 16, 64), (16, 16, 64), (32, 16, 64), (48, 16, 64)]

    def test_horizontal_stripes_become_full_height_bars(self):
        content = _convert({"type": "stripes", "colors": ["R", "O"], "direction": "horizontal", "count": 2})
        assert [(s["x"], s["w"], s["h"]) for s in content["shapes"]] == [(0, 32, 64), (32, 32, 64)]
        assert [s["fill"] for s in content["shapes"]] == ["red", "orange"]

    def test_diagonal_stripes_become_polygons(self):
        content = _convert({"type": "stripes", "colors": ["#111111", "#eeeeee"], "direction": "diagonal", "count": 6})
        assert {s["type"] for s in content["shapes"]} == {"polygon"}
        assert len(content["shapes"]) == 6

    def test_rings_become_concentric_circles_largest_first(self):
        content = _convert({"type": "rings", "colors": ["#0000ff", "#ffffff"], "cx": 32, "cy": 32, "count": 4})
        circles = content["shapes"]
        assert [s["type"] for s in circles] == ["circle"] * 4
        radii = [s["r"] for s in circles]
        assert radii == sorted(radii, reverse=True)
        assert all((s["cx"], s["cy"]) == (32, 32) for s in circles)

    def test_triangle_becomes_a_polygon(self):
        content = _convert({"type": "triangle", "color": "#00aa00", "points": [[0, 64], [32, 8], [64, 64]]})
        assert content["shapes"] == [{"type": "polygon", "points": [[0, 64], [32, 8], [64, 64]], "fill": "#00aa00"}]

    @pytest.mark.parametrize("fill,edge_y", [("below", 64), ("above", 0)])
    def test_filled_wave_becomes_a_polygon_closed_to_the_edge(self, fill, edge_y):
        content = _convert({"type": "wave", "color": "#0088ff", "baseline": 40, "amplitude": 6,
                            "frequency": 2, "fill": fill})
        (poly,) = content["shapes"]
        assert poly["type"] == "polygon" and poly["fill"] == "#0088ff"
        assert [64, edge_y] in poly["points"] and [0, edge_y] in poly["points"]
        ys = [p[1] for p in poly["points"] if p[1] != edge_y]
        assert min(ys) < 40 < max(ys)

    def test_band_wave_is_a_ribbon_polygon(self):
        content = _convert({"type": "wave", "color": "#0088ff", "baseline": 32, "amplitude": 4,
                            "frequency": 1, "thickness": 4})
        (poly,) = content["shapes"]
        assert poly["type"] == "polygon" and len(poly["points"]) >= 20

    def test_noise_becomes_seeded_pixels_inside_its_region(self):
        shape = {"type": "noise", "colors": ["#ffffff", "#ffee00"], "density": 0.2, "x": 0, "y": 0, "w": 64, "h": 16}
        first = _convert(shape)
        assert first == _convert(shape), "deterministic for the same scene"
        rows = first["pixels"]
        assert len(rows) <= 16 and all(len(row) <= 64 for row in rows)  # trailing transparency trimmed
        lit = [(y, x) for y, row in enumerate(rows) for x, ch in enumerate(row) if ch != "."]
        assert lit and all(y < 16 for y, _ in lit)
        assert set(first["palette"].values()) == {"#ffffff", "#ffee00"}

    def test_mirror_horizontal_adds_reflected_copies(self):
        content = _convert({"type": "rect", "x": 4, "y": 10, "w": 8, "h": 8, "fill": "#ff0000"}, mirror="horizontal")
        assert [(s["x"], s["y"]) for s in content["shapes"]] == [(4, 10), (52, 10)]

    def test_mirror_both_is_symmetric_when_rendered(self):
        content = _convert(
            # Edges chosen so no edge crosses a pixel centre exactly: the
            # rasteriser's half-open tie rule is not mirror-symmetric.
            {"type": "polygon", "points": [[0, 0], [30, 5], [12, 25]], "fill": "#ff0000"},
            {"type": "circle", "cx": 12, "cy": 40, "r": 6, "fill": "#00ff00"},
            {"type": "gradient", "x": 0, "y": 0, "w": 20, "h": 20, "from": "#000080", "to": "#ffff00", "angle": 90},
            mirror="both",
        )
        assert len(content["shapes"]) == 12
        if CanvasContent is not None:
            resolved, _ = resolve_content(CanvasContent.model_validate(content), {})
            rgba = render_content(resolved, 64, 64)
            px = lambda x, y: rgba[(y * 64 + x) * 4:(y * 64 + x) * 4 + 4]  # noqa: E731
            for y in range(64):
                for x in range(64):
                    assert px(x, y) == px(63 - x, y) == px(x, 63 - y)

    def test_legacy_multi_colour_gradient_becomes_banded_gradients(self):
        content = _convert({"type": "gradient", "colors": ["#000000", "#ff0000", "#ffffff"], "direction": "vertical"})
        grads = content["shapes"]
        assert [g["type"] for g in grads] == ["gradient", "gradient"]
        assert [(g["from"], g["to"], g["angle"]) for g in grads] == [
            ("#000000", "#ff0000", 90), ("#ff0000", "#ffffff", 90)]

    def test_legacy_radial_gradient_becomes_nested_circles(self):
        content = _convert({"type": "gradient", "colors": ["#000000", "#ffffff"], "direction": "radial"})
        assert {s["type"] for s in content["shapes"]} == {"circle"}

    def test_legacy_rect_ellipse_line_take_their_colour_field(self):
        content = _convert(
            {"type": "rectangle", "color": "Y", "x": 1, "y": 1, "w": 5, "h": 5},
            {"type": "ellipse", "color": "#123456", "cx": 30, "cy": 30, "rx": 9, "ry": 4},
            {"type": "circle", "color": "#654321", "cx": 30, "cy": 30, "rx": 3, "ry": 3},
            {"type": "line", "color": "W", "x1": 0, "y1": 0, "x2": 63, "y2": 63, "thickness": 3},
        )
        rect, ellipse, circle, line = content["shapes"]
        assert rect == {"type": "rect", "x": 1, "y": 1, "w": 5, "h": 5, "fill": "yellow"}
        assert ellipse["fill"] == "#123456" and ellipse["type"] == "ellipse"
        assert circle["type"] == "ellipse" and circle["fill"] == "#654321"
        assert line == {"type": "line", "x1": 0, "y1": 0, "x2": 63, "y2": 63, "stroke": "white", "width": 3}

    def test_a_normalised_legacy_scene_is_scaled_to_pixels(self):
        content = scene_to_content(
            {"background": "K", "shapes": [{"type": "rect", "color": "R", "x": 0.25, "y": 0.5, "w": 0.5, "h": 0.25}]},
            PixelTarget(64, 32),
        )
        assert content["shapes"] == [{"type": "rect", "x": 16, "y": 16, "w": 32, "h": 8, "fill": "red"}]
        assert content["background"] == "black"


# ---------------------------------------------------------------------------
# Clamping to core's limits
# ---------------------------------------------------------------------------


class TestLimits:
    def test_shape_count_is_capped(self):
        shapes = [{"type": "rect", "x": i % 64, "y": i // 64, "w": 1, "h": 1, "fill": "#ff0000"} for i in range(400)]
        content = _convert(*shapes)
        assert len(content["shapes"]) == MAX_CANVAS_SHAPES == 256

    def test_mirror_never_exceeds_the_shape_cap(self):
        shapes = [{"type": "rect", "x": 1, "y": 1, "w": 1, "h": 1, "fill": "#ff0000"}] * 200
        assert len(_convert(*shapes, mirror="both")["shapes"]) == 256

    def test_coordinates_are_clamped(self):
        content = _convert({"type": "circle", "cx": 1e9, "cy": -1e9, "r": 5e6, "fill": "#fff"})
        assert content["shapes"][0] == {"type": "circle", "cx": 1024, "cy": -1024, "r": 1024, "fill": "#fff"}

    def test_line_width_is_clamped(self):
        content = _convert({"type": "line", "x1": 0, "y1": 0, "x2": 9, "y2": 9, "stroke": "#fff", "width": 50})
        assert content["shapes"][0]["width"] == 16

    def test_text_is_bounded_and_cannot_carry_expressions(self):
        content = _convert({"type": "text", "x": 1, "y": 1, "text": "{{secrets.key}}" + "A" * 1000,
                            "color": "#ffffff", "font": "huge"})
        text = content["shapes"][0]
        assert len(text["text"]) <= 256 and "{{" not in text["text"] and "}}" not in text["text"]
        assert text["font"] in ("3x5", "5x7")

    def test_polygon_points_are_capped_and_short_ones_dropped(self):
        many = [[i % 64, (i * 7) % 64] for i in range(400)]
        content = _convert(
            {"type": "polygon", "points": many, "fill": "#ff0000"},
            {"type": "polygon", "points": [[0, 0], [1, 1]], "fill": "#00ff00"},
        )
        assert len(content["shapes"]) == 1 and len(content["shapes"][0]["points"]) == 256

    def test_bad_shapes_and_colours_are_dropped_not_emitted(self):
        content = _convert(
            {"type": "sparkle", "x": 1},
            {"type": "rect", "x": "lots", "y": 0, "w": 5, "h": 5, "fill": "#ff0000"},
            {"type": "rect", "x": 0, "y": 0, "w": 5, "h": 5, "fill": "chartreuse-ish"},
            {"type": "rect", "x": 0, "y": 0, "w": 5, "h": 5, "fill": "#00ff00", "if": "{{x}}", "foreach": "{{y}}"},
            "not a shape",
        )
        assert content["shapes"] == [{"type": "rect", "x": 0, "y": 0, "w": 5, "h": 5, "fill": "#00ff00"}]

    def test_unknown_background_becomes_black(self):
        assert _convert({"type": "rect", "x": 0, "y": 0, "w": 5, "h": 5, "fill": "#fff"},
                        background="nope")["background"] == "#000000"

    def test_a_scene_with_nothing_drawable_is_rejected(self):
        with pytest.raises(ArtValidationError):
            scene_to_content({"background": "#000000", "shapes": [{"type": "sparkle"}]}, T)

    def test_a_single_colour_scene_is_rejected(self):
        with pytest.raises(ArtValidationError):
            scene_to_content({"background": "#000000",
                              "shapes": [{"type": "rect", "x": 0, "y": 0, "w": 5, "h": 5, "fill": "#000000"}]}, T)

    def test_the_prompt_advertises_only_the_size_and_vocabulary(self):
        prompt = canvas_system_prompt(PixelTarget(48, 24))
        for shape in ("rect", "circle", "ellipse", "line", "polygon", "gradient", "text",
                      "stripes", "rings", "wave", "noise"):
            assert f'"type": "{shape}"' in prompt
        assert "48 × 24" in prompt


# ---------------------------------------------------------------------------
# Failure fallback and per-display caching
# ---------------------------------------------------------------------------


class TestFallbackAndCache:
    def test_failure_returns_the_last_good_canvas_for_that_display(self):
        plugin = _plugin()
        board = pixel_board(64, 64)
        good, _ = _fetch(plugin, board)
        bad, _ = _fetch(plugin, board, payload="garbage")
        assert bad.available
        assert bad.data["canvas"] == good.data["canvas"]
        assert bad.data["_fallback"] is True
        assert bad.formatted_lines == good.formatted_lines

    def test_failure_with_no_history_returns_the_deterministic_fallback_scene(self):
        plugin = _plugin()
        result, _ = _fetch(plugin, pixel_board(48, 32), payload="garbage")
        assert result.available
        assert result.error  # recorded, as on every failure
        assert result.data["_fallback"] is True
        assert result.data["canvas"] == fallback_content(PixelTarget(48, 32))
        assert_valid_content(result.data["canvas"], size=(48, 32))
        assert all(count_tiles(line) == 10 for line in result.formatted_lines)

    def test_transport_failure_is_recorded_and_falls_back(self):
        import requests

        plugin = _plugin()
        with patch(POST, side_effect=requests.ConnectionError("down")):
            with plugin._bound_board(pixel_board()):
                result = plugin.fetch_data()
        assert result.available and result.data["_fallback"] is True
        assert "failed" in result.error

    def test_fallback_scene_is_valid_at_every_size(self):
        for size in [(64, 64), (32, 16), (128, 32), (1, 1), (16, 128)]:
            assert_valid_content(fallback_content(PixelTarget(*size)), size=size)

    def test_a_failure_on_one_display_never_serves_another_displays_canvas(self):
        plugin = _plugin()
        _fetch(plugin, pixel_board(64, 64))
        other, _ = _fetch(plugin, pixel_board(32, 16), payload="garbage")
        assert other.data["canvas"]["size"] == [32, 16]
        assert other.data["canvas"] == fallback_content(PixelTarget(32, 16))

    def test_cache_key_includes_the_board_key_and_size(self):
        # BoardContext.key (newer cores) folds in the grid and the display key.
        board = SimpleNamespace(key="panel:10x8|divoom_pixoo64", rows=8, cols=10, display=PixelDisplay())
        key = canvas_key(board, Canvas(rows=8, cols=10), PixelTarget(64, 64))
        assert key == "panel:10x8|divoom_pixoo64|64x64"
        smaller = canvas_key(board, Canvas(rows=8, cols=10), PixelTarget(32, 32))
        assert smaller != key

    def test_cache_key_without_a_board_key_uses_grid_and_display(self):
        board = SimpleNamespace(rows=8, cols=10, display=SimpleNamespace(key="pixoo"))
        assert canvas_key(board, Canvas(rows=8, cols=10), PixelTarget(64, 64)) == "10x8|pixoo|64x64"
        bare = SimpleNamespace(rows=8, cols=10, display=None)
        assert canvas_key(bare, Canvas(rows=8, cols=10), PixelTarget(64, 64)) == "10x8|64x64"

    def test_remembered_canvases_are_bounded(self):
        plugin = _plugin()
        for width in range(1, MAX_REMEMBERED_PIECES + 6):
            _fetch(plugin, pixel_board(width, 64))
        assert len(plugin._last_canvases) == MAX_REMEMBERED_PIECES

    def test_config_change_forgets_remembered_canvases(self):
        plugin = _plugin()
        _fetch(plugin, pixel_board())
        plugin.on_config_change({}, plugin.config)
        assert plugin._last_canvases == {}


# ---------------------------------------------------------------------------
# Deriving tile art from a canvas
# ---------------------------------------------------------------------------


class TestSampleGrid:
    def test_samples_shapes_to_the_nearest_board_colour(self):
        content = {
            "size": [20, 10], "background": "#000000",
            "shapes": [
                {"type": "rect", "x": 0, "y": 0, "w": 10, "h": 10, "fill": "#ee2211"},
                {"type": "circle", "cx": 15, "cy": 5, "r": 3, "fill": "#2266dd"},
            ],
        }
        grid = sample_grid(content, rows=2, cols=4)
        assert grid == [["R", "R", "K", "K"], ["R", "R", "K", "K"]]

    def test_samples_gradients_polygons_lines_and_pixels(self):
        content = {
            "size": [10, 10], "background": "#000000",
            "palette": {"a": "#ffff00"},
            "shapes": [
                {"type": "gradient", "from": "#ffffff", "to": "#000000", "angle": 90},
                {"type": "polygon", "points": [[0, 10], [5, 5], [10, 10]], "fill": "#00ff00"},
                {"type": "line", "x1": 0, "y1": 0, "x2": 10, "y2": 0, "stroke": "#ff0000", "width": 2},
            ],
            "pixels": ["", "", "", "", "", "", "", "", "", "........aa"],
        }
        grid = sample_grid(content, rows=10, cols=10)
        assert grid[0][5] == "R"        # the line along the top
        assert grid[1][0] == "W"        # light end of the white→black gradient
        assert grid[9][5] == "G"        # inside the polygon
        assert grid[9][9] == "Y"        # a pixel row drawn last
        assert grid[8][0] == "K"        # dark end of the gradient

    def test_stroke_only_and_text_shapes_do_not_flood_cells(self):
        content = {"size": [10, 10], "background": "#000000", "shapes": [
            {"type": "rect", "x": 0, "y": 0, "w": 10, "h": 10, "stroke": "#ffffff"},
            {"type": "text", "x": 0, "y": 0, "text": "HI", "color": "#ffffff"},
        ]}
        assert sample_grid(content, rows=1, cols=1) == [["K"]]


# ---------------------------------------------------------------------------
# Manifest: the variable contract and the LED demo page
# ---------------------------------------------------------------------------

from pathlib import Path  # noqa: E402

MANIFEST = json.loads((Path(__file__).resolve().parent.parent / "manifest.json").read_text())


def test_canvas_variable_is_declared_with_the_canvas_format():
    spec = MANIFEST["variables"]["simple"]["canvas"]
    assert spec["format"] == "canvas"
    assert spec["type"] == "object"
    assert "max_length" not in spec  # an object, not board text
    example = json.loads(spec["example"])
    assert_valid_content(example)


def test_minor_version_bump_for_the_canvas_variable():
    assert MANIFEST["version"] == "1.6.0"


@pytest.mark.parametrize("device_type", ["flagship", "note"])
def test_demo_pages_carry_a_full_bleed_canvas_over_the_tile_art(device_type):
    """Core accepts only flagship/note demo keys, and resolves an LED panel
    to the first one, so the canvas rides on those demos: an LED pixel board
    paints it over everything; boards (and cores) without canvases ignore it
    and show the tile art from the template underneath."""
    demo = MANIFEST["demo"][device_type]
    (canvas,) = demo["canvases"]
    assert canvas == {
        "id": "art",
        "area": {"row": 1, "col": 1, "rows": 96, "cols": 128},  # the largest grid; core clamps
        "bleed": ["all"],
        "text": "hide",
        "source": "{{generative_ai_art.canvas}}",
    }
    assert demo["template"][0] == "{{generative_ai_art.art}}"
    if CanvasContent is not None:
        from src.canvas.models import Canvas as CoreCanvas

        CoreCanvas.model_validate(canvas)


def test_demo_keys_are_ones_core_accepts():
    assert set(MANIFEST["demo"]) <= {"flagship", "note"}


# ---------------------------------------------------------------------------
# Edge cases of parsing, conversion and failure reporting
# ---------------------------------------------------------------------------


class TestEdges:
    @pytest.mark.parametrize(
        "reply",
        ["[1, 2, 3]", '{"theme": "x"}', '{"grid": [[], "row"]}'],
        ids=["not-an-object", "no-shapes-or-grid", "grid-without-rows"],
    )
    def test_unusable_replies_are_retried_then_fall_back(self, reply):
        plugin = _plugin()
        with patch(POST, return_value=_response(reply)) as post:
            with plugin._bound_board(pixel_board()):
                result = plugin.fetch_data()
        assert post.call_count == 3  # MAX_ATTEMPTS, a new theme each time
        assert result.data["_fallback"] is True

    @pytest.mark.parametrize("direction,angle", [("diagonal", 45), ("antidiagonal", 315)])
    def test_diagonal_legacy_gradients_keep_their_ends(self, direction, angle):
        content = _convert({"type": "gradient", "colors": ["#000000", "#808080", "#ffffff"], "direction": direction})
        assert content["shapes"] == [{"type": "gradient", "from": "#000000", "to": "#ffffff", "angle": angle}]

    def test_horizontal_legacy_gradient_bands_left_to_right(self):
        content = _convert({"type": "gradient", "colors": ["#000000", "#ff0000", "#ffffff"], "direction": "horizontal"})
        assert [(g["x"], g["w"], g["angle"]) for g in content["shapes"]] == [(0, 32, 0), (32, 32, 0)]

    def test_single_colour_legacy_gradient_is_a_flat_fill(self):
        content = _convert({"type": "gradient", "colors": "#ff0000"})
        assert content["shapes"] == [{"type": "rect", "x": 0, "y": 0, "w": 64, "h": 64, "fill": "#ff0000"}]

    def test_legacy_gradient_without_colours_is_dropped(self):
        content = _convert({"type": "gradient", "colors": ["nope"]},
                           {"type": "rect", "x": 0, "y": 0, "w": 5, "h": 5, "fill": "#ffffff"})
        assert [s["type"] for s in content["shapes"]] == ["rect"]

    def test_antidiagonal_stripes_are_flipped(self):
        down = _convert({"type": "stripes", "colors": ["#111111", "#eeeeee"], "direction": "diagonal", "count": 2})
        up = _convert({"type": "stripes", "colors": ["#111111", "#eeeeee"], "direction": "antidiagonal", "count": 2})
        assert up["shapes"][0]["points"] == [[x, 64 - y] for x, y in down["shapes"][0]["points"]]

    def test_mirror_reflects_lines_and_strokes(self):
        content = _convert(
            {"type": "line", "x1": 2, "y1": 3, "x2": 10, "y2": 20, "stroke": "#ffffff"},
            {"type": "ellipse", "cx": 10, "cy": 12, "rx": 4, "ry": 2, "fill": "#00ff00", "stroke": "#ff0000"},
            mirror="both",
        )
        lines = [s for s in content["shapes"] if s["type"] == "line"]
        assert [(s["x1"], s["y1"], s["x2"], s["y2"]) for s in lines] == [
            (2, 3, 10, 20), (62, 3, 54, 20), (2, 61, 10, 44), (62, 61, 54, 44)]
        assert all(s["stroke"] == "#ff0000" for s in content["shapes"] if s["type"] == "ellipse")

    def test_mirror_leaves_text_single(self):
        content = _convert(
            {"type": "rect", "x": 0, "y": 0, "w": 5, "h": 5, "fill": "#ff0000"},
            {"type": "text", "x": 2, "y": 2, "text": "HI", "color": "#ffffff"},
            mirror="horizontal",
        )
        assert [s["type"] for s in content["shapes"]] == ["rect", "rect", "text"]

    def test_mirror_makes_noise_symmetric(self):
        content = _convert({"type": "noise", "colors": ["#ffffff"], "density": 0.3}, mirror="both")
        rows = [row.ljust(64, ".") for row in content["pixels"]]
        rows += ["." * 64] * (64 - len(rows))
        assert rows == [row[::-1] for row in rows] == rows[::-1]

    def test_normalised_wave_rings_and_polygons_scale_with_the_display(self):
        content = scene_to_content({"background": "K", "shapes": [
            {"type": "wave", "color": "B", "baseline": 0.5, "amplitude": 0.1, "thickness": 0.25, "frequency": 1},
            {"type": "circle", "color": "W", "cx": 0.5, "cy": 0.5, "r": 0.25},
            {"type": "triangle", "color": "R", "points": [[0, 1], [0.5, 0], [1, 1]]},
        ]}, PixelTarget(64, 32))
        wave, circle, triangle = content["shapes"]
        ys = [p[1] for p in wave["points"]]
        assert min(ys) >= 16 - 3.2 - 4 - 0.01 and max(ys) <= 16 + 3.2 + 4 + 0.01  # thickness 0.25 x 32
        assert circle == {"type": "circle", "cx": 32, "cy": 16, "r": 8, "fill": "white"}
        assert triangle["points"] == [[0, 32], [32, 0], [64, 32]]

    def test_numeric_strings_are_read_as_numbers(self):
        content = _convert({"type": "rect", "x": "4", "y": " 5.5 ", "w": 3, "h": 3, "fill": "#ffffff"})
        assert (content["shapes"][0]["x"], content["shapes"][0]["y"]) == (4, 5.5)

    def test_palette_never_exceeds_core_limit(self):
        colours = [f"#{i:06x}" for i in range(1, 80)]
        content = _convert({"type": "noise", "colors": colours, "density": 0.5})
        assert len(content["palette"]) == 62

    def test_ellipses_sample_into_tiles(self):
        content = {"size": [10, 10], "background": "#000000",
                   "shapes": [{"type": "ellipse", "cx": 5, "cy": 5, "rx": 3, "ry": 2, "fill": "#ffffff"}]}
        assert sample_grid(content, rows=1, cols=1) == [["W"]]
        assert sample_grid(content, rows=2, cols=2) == [["K", "K"], ["K", "K"]]

    def test_provider_not_set_up_is_reported_on_pixel_displays_too(self):
        errors = pytest.importorskip("src.ai.plugin_api")
        plugin = GenerativeAiArtPlugin({"id": "generative_ai_art", "name": "x", "version": "1.0.0"})
        plugin.config = {"enabled": True, "refresh_seconds": 300}
        plugin.ai_complete = MagicMock(side_effect=errors.AINotConfiguredError("AI is off"))
        with plugin._bound_board(pixel_board()):
            result = plugin.fetch_data()
        assert not result.available
        assert "Settings → AI Providers" in result.error

    def test_other_provider_failures_fall_back_with_the_reason(self):
        plugin = GenerativeAiArtPlugin({"id": "generative_ai_art", "name": "x", "version": "1.0.0"})
        plugin.config = {"enabled": True, "refresh_seconds": 300}
        plugin.ai_complete = MagicMock(side_effect=RuntimeError("provider exploded"))
        with plugin._bound_board(pixel_board()):
            result = plugin.fetch_data()
        assert result.available and result.data["_fallback"] is True
        assert "provider exploded" in result.error

    def test_provider_reply_feeds_the_canvas(self):
        reply = SimpleNamespace(text=json.dumps(PIXEL_SCENE), model="provider-model")
        plugin = GenerativeAiArtPlugin({"id": "generative_ai_art", "name": "x", "version": "1.0.0"})
        plugin.config = {"enabled": True, "refresh_seconds": 300}
        plugin.ai_complete = MagicMock(return_value=reply)
        with plugin._bound_board(pixel_board()):
            result = plugin.fetch_data()
        assert result.data["model"] == "provider-model"
        assert_valid_content(result.data["canvas"], size=(64, 64))
        assert plugin.ai_complete.call_args.kwargs["max_tokens"] == CANVAS_MAX_TOKENS
