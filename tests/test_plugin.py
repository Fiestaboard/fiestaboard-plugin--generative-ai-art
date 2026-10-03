"""Tests for generative_ai_art plugin."""

import json
import random
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest
from src.devices import BoardContext
from src.text_to_board import count_tiles

from plugins.generative_ai_art import GenerativeAiArtPlugin
from plugins.generative_ai_art.source import (
    BUILTIN_THEMES,
    MAX_DESCRIPTION_CHARS,
    MAX_THEME_CHARS,
    PER_CELL_MAX_CELLS,
    SCENE_MAX_TOKENS,
    ArtGenerator,
    ArtValidationError,
    Canvas,
    rasterise,
)

FLAGSHIP = BoardContext.from_device_type("flagship")
NOTE = BoardContext.from_device_type("note")


def note_array(notes_wide: int, notes_tall: int) -> BoardContext:
    return BoardContext(device_type="note_array", rows=notes_tall * 3, cols=notes_wide * 15)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_grid(rows: int, cols: int, color: str = "B") -> list:
    """Build a uniform grid for testing."""
    return [[color] * cols for _ in range(rows)]


def _make_grid_two_colors(rows: int, cols: int) -> list:
    """Build a grid with two alternating colors."""
    return [["B" if (r + c) % 2 == 0 else "R" for c in range(cols)] for r in range(rows)]


def _grid_response(
    theme: str = "test theme",
    rows: int = 6,
    cols: int = 22,
    title: Optional[str] = None,
) -> str:
    """Return a valid per-cell JSON response string."""
    d: dict = {
        "theme": theme,
        "description": "A test composition",
        "grid": _make_grid_two_colors(rows, cols),
    }
    if title is not None:
        d["title"] = title
    return json.dumps(d)


SCENE = {
    "theme": "twilight ridge",
    "description": "A dark ridge under a violet sky",
    "background": "K",
    "shapes": [
        {"type": "gradient", "colors": ["V", "B", "O"], "direction": "vertical"},
        {"type": "triangle", "color": "K", "points": [[0.0, 1.0], [0.4, 0.3], [0.8, 1.0]]},
    ],
}


def _scene_response(title: Optional[str] = None) -> str:
    payload = dict(SCENE)
    if title is not None:
        payload["title"] = title
    return json.dumps(payload)


def _generator(**kwargs) -> ArtGenerator:
    defaults = dict(
        base_url="https://api.example.com/v1",
        api_key="sk-test",
        model="gpt-4o-mini",
        temperature=1.0,
    )
    defaults.update(kwargs)
    return ArtGenerator(**defaults)


# ---------------------------------------------------------------------------
# Canvas: every dimension in the plugin comes from one of these
# ---------------------------------------------------------------------------

class TestCanvas:
    def test_default_is_a_flagship(self):
        canvas = Canvas.default()
        assert (canvas.rows, canvas.cols) == (6, 22)

    @pytest.mark.parametrize(
        "rows,cols,expected_mode",
        [
            (6, 22, "grid"),      # flagship, 132 cells
            (3, 15, "grid"),      # note, 45 cells
            (12, 15, "grid"),     # 1x4 array, 180 cells
            (3, 120, "scene"),    # 8x1 array, 360 cells
            (12, 30, "scene"),    # 65" panel, 360 cells
            (24, 120, "scene"),   # 8x8 array, 2880 cells
        ],
    )
    def test_mode_follows_cell_count_not_device_type(self, rows, cols, expected_mode):
        assert Canvas(rows=rows, cols=cols).mode == expected_mode

    def test_threshold_is_the_boundary(self):
        assert Canvas(rows=1, cols=PER_CELL_MAX_CELLS).mode == "grid"
        assert Canvas(rows=1, cols=PER_CELL_MAX_CELLS + 1).mode == "scene"

    def test_title_row_comes_out_of_the_art(self):
        canvas = Canvas(rows=6, cols=22, show_title=True)
        assert canvas.has_title_row is True
        assert canvas.art_rows == 5

    def test_single_row_board_keeps_all_of_itself_for_art(self):
        """A one-row board has no row to spare for a title."""
        canvas = Canvas(rows=1, cols=15, show_title=True)
        assert canvas.has_title_row is False
        assert canvas.art_rows == 1

    def test_key_distinguishes_geometries_with_the_same_device_type(self):
        assert Canvas(rows=12, cols=30).key != Canvas(rows=3, cols=120).key

    def test_degenerate_dimensions_are_clamped_not_crashed(self):
        canvas = Canvas(rows=0, cols=-5)
        assert (canvas.rows, canvas.cols) == (1, 1)


# ---------------------------------------------------------------------------
# Per-cell grid: repair rather than discard
# ---------------------------------------------------------------------------

class TestGridRepair:
    def setup_method(self):
        self.gen = _generator()
        self.canvas = Canvas(rows=6, cols=22)

    def test_valid_flagship_grid(self):
        result = self.gen._validate_and_parse(_grid_response(rows=6, cols=22), self.canvas)
        assert result["theme"] == "test theme"
        assert len(result["grid"]) == 6
        assert all(len(row) == 22 for row in result["grid"])

    def test_valid_note_grid(self):
        canvas = Canvas(rows=3, cols=15)
        result = self.gen._validate_and_parse(_grid_response(rows=3, cols=15), canvas)
        assert len(result["grid"]) == 3
        assert all(len(row) == 15 for row in result["grid"])

    def test_strips_markdown_fences(self):
        raw = f"```json\n{_grid_response()}\n```"
        assert self.gen._validate_and_parse(raw, self.canvas)["theme"] == "test theme"

    def test_short_grid_is_resampled_not_discarded(self):
        """One row missing out of six must not throw away the other five."""
        raw = json.dumps({"theme": "t", "description": "d", "grid": _make_grid_two_colors(5, 22)})
        grid = self.gen._validate_and_parse(raw, self.canvas)["grid"]
        assert len(grid) == 6
        assert all(len(row) == 22 for row in grid)

    def test_narrow_rows_are_resampled(self):
        raw = json.dumps({"theme": "t", "description": "d", "grid": _make_grid_two_colors(6, 18)})
        grid = self.gen._validate_and_parse(raw, self.canvas)["grid"]
        assert all(len(row) == 22 for row in grid)

    def test_overlong_rows_are_resampled_down(self):
        raw = json.dumps({"theme": "t", "description": "d", "grid": _make_grid_two_colors(6, 30)})
        grid = self.gen._validate_and_parse(raw, self.canvas)["grid"]
        assert all(len(row) == 22 for row in grid)

    def test_a_few_bad_cells_are_coerced(self):
        grid = _make_grid_two_colors(6, 22)
        grid[0][0] = "Z"
        grid[1][3] = None
        raw = json.dumps({"theme": "t", "description": "d", "grid": grid})
        repaired = self.gen._validate_and_parse(raw, self.canvas)["grid"]
        assert all(cell in set("ROYGBVWK") for row in repaired for cell in row)

    def test_lowercase_letters_are_accepted(self):
        grid = [[cell.lower() for cell in row] for row in _make_grid_two_colors(6, 22)]
        raw = json.dumps({"theme": "t", "description": "d", "grid": grid})
        assert self.gen._validate_and_parse(raw, self.canvas)["grid"][0][0] == "B"

    def test_mostly_invalid_cells_are_rejected(self):
        grid = [["Z"] * 22 for _ in range(6)]
        grid[0] = ["B"] * 22
        raw = json.dumps({"theme": "t", "description": "d", "grid": grid})
        with pytest.raises(ArtValidationError, match="not valid colour codes"):
            self.gen._validate_and_parse(raw, self.canvas)

    def test_wildly_wrong_row_count_is_rejected(self):
        raw = json.dumps({"theme": "t", "description": "d", "grid": _make_grid_two_colors(1, 22)})
        with pytest.raises(ArtValidationError, match="too far off to repair"):
            self.gen._validate_and_parse(raw, self.canvas)

    def test_wildly_wrong_row_widths_are_rejected(self):
        raw = json.dumps({"theme": "t", "description": "d", "grid": _make_grid_two_colors(6, 4)})
        with pytest.raises(ArtValidationError, match="wrong width"):
            self.gen._validate_and_parse(raw, self.canvas)

    def test_mono_color_rejected(self):
        raw = json.dumps({"theme": "t", "description": "d", "grid": _make_grid(6, 22, "B")})
        with pytest.raises(ArtValidationError, match="minimum is 2"):
            self.gen._validate_and_parse(raw, self.canvas)

    def test_no_content_at_all_is_rejected(self):
        raw = json.dumps({"theme": "t", "description": "d"})
        with pytest.raises(ArtValidationError, match="neither a 'grid'"):
            self.gen._validate_and_parse(raw, self.canvas)

    def test_grid_of_wrong_type_is_rejected(self):
        raw = json.dumps({"theme": "t", "description": "d", "grid": "BBBB"})
        with pytest.raises(ArtValidationError, match="neither a 'grid'"):
            self.gen._validate_and_parse(raw, self.canvas)

    def test_unparseable_response_is_rejected(self):
        with pytest.raises(ArtValidationError, match="Invalid JSON"):
            self.gen._validate_and_parse("not json at all", self.canvas)

    def test_non_object_response_is_rejected(self):
        with pytest.raises(ArtValidationError, match="not a JSON object"):
            self.gen._validate_and_parse("[1, 2, 3]", self.canvas)

    def test_missing_metadata_is_filled_in_rather_than_retried(self):
        raw = json.dumps({"grid": _make_grid_two_colors(6, 22)})
        result = self.gen._validate_and_parse(raw, self.canvas)
        assert result["theme"] == ""
        assert result["description"] == ""

    def test_overlong_metadata_is_truncated_to_its_declared_bound(self):
        raw = json.dumps({
            "theme": "t" * 500,
            "description": "d" * 500,
            "grid": _make_grid_two_colors(6, 22),
        })
        result = self.gen._validate_and_parse(raw, self.canvas)
        assert len(result["theme"]) == MAX_THEME_CHARS
        assert len(result["description"]) == MAX_DESCRIPTION_CHARS

    def test_all_valid_colors_accepted(self):
        valid_colors = list("ROYGBVWK")
        grid = []
        for r in range(6):
            c1 = valid_colors[r % 8]
            c2 = valid_colors[(r + 1) % 8]
            grid.append([c1 if i % 2 == 0 else c2 for i in range(22)])
        raw = json.dumps({"theme": "t", "description": "d", "grid": grid})
        assert self.gen._validate_and_parse(raw, self.canvas) is not None


class TestTruncatedJsonSalvage:
    """A response cut off mid-array is the classic large-board failure."""

    def test_truncated_grid_is_recovered(self):
        canvas = Canvas(rows=6, cols=22)
        full = _grid_response(rows=6, cols=22)
        truncated = full[: full.index('"grid"') + 400]
        grid = _generator()._validate_and_parse(truncated, canvas)["grid"]
        assert len(grid) == 6
        assert all(len(row) == 22 for row in grid)

    def test_truncated_scene_is_recovered(self):
        canvas = Canvas(rows=24, cols=120)
        full = _scene_response()
        truncated = full[:-12]
        result = _generator()._validate_and_parse(truncated, canvas)
        assert len(result["grid"]) == 24

    def test_unsalvageable_text_still_raises(self):
        """Cut before any complete value, there is nothing to close."""
        with pytest.raises(ArtValidationError, match="Invalid JSON"):
            _generator()._validate_and_parse('{"grid": ', Canvas(rows=6, cols=22))

    def test_salvage_hands_a_too_short_grid_to_the_repair_budget(self):
        """Salvage recovers JSON; repair still decides whether it is art."""
        with pytest.raises(ArtValidationError, match="too far off to repair"):
            _generator()._validate_and_parse('{"grid": [["B"', Canvas(rows=6, cols=22))


# ---------------------------------------------------------------------------
# Scene rasteriser: O(1) tokens, any aspect ratio
# ---------------------------------------------------------------------------

class TestRasteriser:
    @pytest.mark.parametrize(
        "rows,cols",
        [(6, 22), (3, 15), (12, 30), (12, 15), (3, 120), (24, 120)],
    )
    def test_fills_exactly_the_canvas(self, rows, cols):
        canvas = Canvas(rows=rows, cols=cols)
        grid = rasterise(SCENE, canvas, random.Random(1))
        assert len(grid) == canvas.art_rows
        assert all(len(row) == cols for row in grid)
        assert all(cell in set("ROYGBVWK") for row in grid for cell in row)

    def test_background_shows_where_nothing_is_painted(self):
        grid = rasterise({"background": "W", "shapes": []}, Canvas(rows=3, cols=5))
        assert grid == [["W"] * 5] * 3

    def test_unknown_background_falls_back_to_black(self):
        grid = rasterise({"background": "zzz", "shapes": []}, Canvas(rows=2, cols=2))
        assert grid == [["K", "K"], ["K", "K"]]

    def test_unknown_shape_is_skipped_not_fatal(self):
        grid = rasterise(
            {"background": "K", "shapes": [{"type": "hypercube"}, {"type": "rect", "color": "W"}]},
            Canvas(rows=4, cols=4),
        )
        assert "W" in {cell for row in grid for cell in row}

    def test_non_dict_shape_is_skipped(self):
        grid = rasterise({"background": "K", "shapes": ["nonsense", 5]}, Canvas(rows=2, cols=2))
        assert grid == [["K", "K"], ["K", "K"]]

    def test_shapes_paint_back_to_front(self):
        scene = {
            "background": "K",
            "shapes": [
                {"type": "rect", "color": "R", "x": 0, "y": 0, "w": 1, "h": 1},
                {"type": "rect", "color": "B", "x": 0, "y": 0, "w": 1, "h": 1},
            ],
        }
        grid = rasterise(scene, Canvas(rows=3, cols=3))
        assert grid == [["B"] * 3] * 3

    def test_vertical_gradient_runs_top_to_bottom(self):
        scene = {"shapes": [{"type": "gradient", "colors": ["R", "B"], "direction": "vertical"}]}
        grid = rasterise(scene, Canvas(rows=4, cols=2))
        assert grid[0] == ["R", "R"]
        assert grid[-1] == ["B", "B"]

    def test_horizontal_gradient_runs_left_to_right(self):
        scene = {"shapes": [{"type": "gradient", "colors": ["R", "B"], "direction": "horizontal"}]}
        grid = rasterise(scene, Canvas(rows=2, cols=4))
        assert grid[0][0] == "R"
        assert grid[0][-1] == "B"

    def test_radial_gradient_centres_the_first_colour(self):
        scene = {"shapes": [{"type": "gradient", "colors": ["W", "K"], "direction": "radial"}]}
        grid = rasterise(scene, Canvas(rows=5, cols=5))
        assert grid[2][2] == "W"
        assert grid[0][0] == "K"

    def test_stripes_repeat(self):
        scene = {"shapes": [
            {"type": "stripes", "colors": ["R", "B"], "direction": "horizontal", "count": 4}
        ]}
        grid = rasterise(scene, Canvas(rows=1, cols=4))
        assert grid[0] == ["R", "B", "R", "B"]

    def test_rect_covers_the_requested_fraction(self):
        scene = {"background": "K", "shapes": [
            {"type": "rect", "color": "W", "x": 0.0, "y": 0.0, "w": 0.5, "h": 1.0}
        ]}
        grid = rasterise(scene, Canvas(rows=2, cols=10))
        assert grid[0][:5] == ["W"] * 5
        assert grid[0][5:] == ["K"] * 5

    def test_tiny_rect_still_paints_at_least_one_cell(self):
        """A shape thinner than a tile must not vanish on a small board."""
        scene = {"background": "K", "shapes": [
            {"type": "rect", "color": "W", "x": 0.4, "y": 0.4, "w": 0.001, "h": 0.001}
        ]}
        grid = rasterise(scene, Canvas(rows=3, cols=3))
        assert "W" in {cell for row in grid for cell in row}

    def test_ellipse_is_centred(self):
        scene = {"background": "K", "shapes": [
            {"type": "ellipse", "color": "W", "cx": 0.5, "cy": 0.5, "rx": 0.2, "ry": 0.2}
        ]}
        grid = rasterise(scene, Canvas(rows=7, cols=7))
        assert grid[3][3] == "W"
        assert grid[0][0] == "K"

    def test_rings_alternate_outwards(self):
        scene = {"shapes": [
            {"type": "rings", "colors": ["W", "K"], "cx": 0.5, "cy": 0.5, "count": 4}
        ]}
        grid = rasterise(scene, Canvas(rows=9, cols=9))
        assert grid[4][4] == "W"
        assert len({cell for row in grid for cell in row}) == 2

    def test_triangle_fills_its_interior(self):
        scene = {"background": "K", "shapes": [
            {"type": "triangle", "color": "W", "points": [[0.0, 1.0], [0.5, 0.0], [1.0, 1.0]]}
        ]}
        grid = rasterise(scene, Canvas(rows=5, cols=5))
        assert grid[4][2] == "W"
        assert grid[0][0] == "K"

    def test_malformed_triangle_is_ignored(self):
        scene = {"background": "K", "shapes": [{"type": "triangle", "color": "W", "points": [[0, 0]]}]}
        grid = rasterise(scene, Canvas(rows=3, cols=3))
        assert grid == [["K"] * 3] * 3

    def test_line_paints_a_diagonal(self):
        scene = {"background": "K", "shapes": [
            {"type": "line", "color": "W", "x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0,
             "thickness": 0.05}
        ]}
        grid = rasterise(scene, Canvas(rows=6, cols=6))
        assert grid[0][0] == "W"
        assert grid[5][5] == "W"
        assert grid[0][5] == "K"

    def test_degenerate_line_is_a_dot(self):
        scene = {"background": "K", "shapes": [
            {"type": "line", "color": "W", "x1": 0.5, "y1": 0.5, "x2": 0.5, "y2": 0.5,
             "thickness": 0.4}
        ]}
        grid = rasterise(scene, Canvas(rows=5, cols=5))
        assert grid[2][2] == "W"

    @pytest.mark.parametrize("fill,top,bottom", [("below", "K", "W"), ("above", "W", "K")])
    def test_wave_fill_direction(self, fill, top, bottom):
        scene = {"background": "K", "shapes": [
            {"type": "wave", "color": "W", "baseline": 0.5, "amplitude": 0.0,
             "frequency": 1, "fill": fill}
        ]}
        grid = rasterise(scene, Canvas(rows=4, cols=3))
        assert grid[0][0] == top
        assert grid[-1][0] == bottom

    def test_wave_band_hugs_the_baseline(self):
        scene = {"background": "K", "shapes": [
            {"type": "wave", "color": "W", "baseline": 0.5, "amplitude": 0.0,
             "frequency": 1, "thickness": 0.1, "fill": "band"}
        ]}
        grid = rasterise(scene, Canvas(rows=9, cols=3))
        assert grid[4][0] == "W"
        assert grid[0][0] == "K"

    def test_noise_is_bounded_by_density(self):
        scene = {"background": "K", "shapes": [
            {"type": "noise", "colors": ["W"], "density": 1.0}
        ]}
        grid = rasterise(scene, Canvas(rows=4, cols=4), random.Random(3))
        assert grid == [["W"] * 4] * 4

    def test_horizontal_mirror_makes_the_halves_match(self):
        scene = {
            "background": "K",
            "mirror": "horizontal",
            "shapes": [{"type": "rect", "color": "W", "x": 0.0, "y": 0.0, "w": 0.25, "h": 1.0}],
        }
        grid = rasterise(scene, Canvas(rows=2, cols=8))
        assert grid[0][:2] == ["W", "W"]
        assert grid[0][-2:] == ["W", "W"]

    def test_vertical_mirror_makes_the_halves_match(self):
        scene = {
            "background": "K",
            "mirror": "vertical",
            "shapes": [{"type": "rect", "color": "W", "x": 0.0, "y": 0.0, "w": 1.0, "h": 0.25}],
        }
        grid = rasterise(scene, Canvas(rows=8, cols=2))
        assert grid[0] == ["W", "W"]
        assert grid[-1] == ["W", "W"]

    def test_unknown_mirror_mode_is_a_noop(self):
        scene = {
            "background": "K",
            "mirror": "sideways",
            "shapes": [{"type": "rect", "color": "W", "x": 0.0, "y": 0.0, "w": 0.25, "h": 1.0}],
        }
        grid = rasterise(scene, Canvas(rows=2, cols=8))
        assert grid[0][-1] == "K"

    def test_same_scene_reads_natively_on_opposite_aspect_ratios(self):
        """A 15x12 board and a 120x3 board are both note arrays and nothing else alike."""
        tall = rasterise(SCENE, Canvas(rows=12, cols=15), random.Random(1))
        wide = rasterise(SCENE, Canvas(rows=3, cols=120), random.Random(1))
        assert (len(tall), len(tall[0])) == (12, 15)
        assert (len(wide), len(wide[0])) == (3, 120)


# ---------------------------------------------------------------------------
# Prompt building and token budget
# ---------------------------------------------------------------------------

class TestPrompts:
    def test_small_board_gets_the_per_cell_prompt(self):
        prompt = _generator().build_default_system_prompt(Canvas(rows=6, cols=22))
        assert "EXACTLY 6 rows" in prompt
        assert "EXACTLY 22 elements" in prompt
        assert '"shapes"' not in prompt

    def test_note_prompt_states_note_dimensions(self):
        prompt = _generator().build_default_system_prompt(Canvas(rows=3, cols=15))
        assert "EXACTLY 3 rows" in prompt
        assert "EXACTLY 15 elements" in prompt

    def test_large_board_gets_the_scene_prompt(self):
        prompt = _generator().build_default_system_prompt(Canvas(rows=24, cols=120))
        assert "NORMALISED" in prompt
        assert '"shapes"' in prompt
        # Never asks the model to enumerate 2,880 cells.
        assert "EXACTLY 24 rows" not in prompt

    def test_scene_prompt_states_the_aspect_ratio(self):
        prompt = _generator().build_default_system_prompt(Canvas(rows=3, cols=120))
        assert "120:3" in prompt

    def test_default_prompt_with_no_canvas_is_the_flagship_prompt(self):
        assert "EXACTLY 6 rows" in _generator().build_default_system_prompt()

    def test_show_title_reserves_a_row_in_the_prompt(self):
        gen = _generator(show_title=True)
        prompt = gen.build_default_system_prompt(Canvas(rows=6, cols=22, show_title=True))
        assert "EXACTLY 5 rows" in prompt
        assert '"title"' in prompt

    def test_no_title_field_without_show_title(self):
        prompt = _generator().build_default_system_prompt(Canvas(rows=6, cols=22))
        assert '"title"' not in prompt

    def test_extra_instructions_are_appended(self):
        gen = _generator(extra_instructions="favour cool colours")
        assert gen.build_default_system_prompt().endswith("favour cool colours")

    def test_custom_system_prompt_replaces_the_default(self):
        gen = _generator(custom_system_prompt="just do it")
        messages = gen._build_messages("a theme", Canvas(rows=6, cols=22))
        assert messages[0]["content"] == "just do it"

    def test_max_tokens_is_constant_for_scene_boards(self):
        assert ArtGenerator.max_tokens_for(Canvas(rows=24, cols=120)) == SCENE_MAX_TOKENS
        assert ArtGenerator.max_tokens_for(Canvas(rows=3, cols=120)) == SCENE_MAX_TOKENS

    def test_max_tokens_scales_with_cells_for_grid_boards(self):
        small = ArtGenerator.max_tokens_for(Canvas(rows=3, cols=15))
        large = ArtGenerator.max_tokens_for(Canvas(rows=12, cols=15))
        assert small <= large

    @patch("plugins.generative_ai_art.source.requests.post")
    def test_max_tokens_is_always_sent(self, mock_post):
        mock_post.return_value = _response(_grid_response())
        _generator().generate(Canvas(rows=6, cols=22))
        assert "max_tokens" in mock_post.call_args.kwargs["json"]


# ---------------------------------------------------------------------------
# Rendering board rows
# ---------------------------------------------------------------------------

class TestRenderLines:
    def test_single_cell_red(self):
        gen = _generator()
        assert gen.render_lines([["R"]], "", Canvas(rows=1, cols=1)) == ["{red}"]

    def test_all_color_codes_rendered(self):
        lines = _generator().render_lines([list("ROYGBVWK")], "", Canvas(rows=1, cols=8))
        for marker in ["{red}", "{orange}", "{yellow}", "{green}",
                       "{blue}", "{violet}", "{white}", "{black}"]:
            assert marker in lines[0]

    @pytest.mark.parametrize("rows,cols", [(6, 22), (3, 15), (12, 30), (3, 120), (24, 120)])
    def test_every_row_is_exactly_the_board_width_in_tiles(self, rows, cols):
        canvas = Canvas(rows=rows, cols=cols)
        grid = _make_grid_two_colors(canvas.art_rows, cols)
        lines = _generator().render_lines(grid, "", canvas)
        assert len(lines) == rows
        assert all(count_tiles(line) == cols for line in lines)

    def test_title_row_is_centred_and_uppercased(self):
        canvas = Canvas(rows=6, cols=22, show_title=True)
        lines = _generator(show_title=True).render_lines(
            _make_grid_two_colors(5, 22), "aurora", canvas
        )
        assert len(lines) == 6
        assert lines[-1] == "AURORA".center(22)

    def test_overlong_title_is_trimmed_to_the_board(self):
        canvas = Canvas(rows=3, cols=15, show_title=True)
        lines = _generator(show_title=True).render_lines(
            _make_grid_two_colors(2, 15), "a" * 40, canvas
        )
        assert count_tiles(lines[-1]) == 15

    def test_no_title_row_without_show_title(self):
        lines = _generator().render_lines(
            _make_grid_two_colors(6, 22), "ignored", Canvas(rows=6, cols=22)
        )
        assert len(lines) == 6


# ---------------------------------------------------------------------------
# generate(): retries and fallbacks
# ---------------------------------------------------------------------------

def _response(content: str) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"choices": [{"message": {"content": content}}]}
    return mock_resp


class TestGenerate:
    @patch("plugins.generative_ai_art.source.requests.post")
    def test_retries_with_a_different_theme_after_a_bad_response(self, mock_post):
        mock_post.side_effect = [_response("garbage"), _response(_grid_response())]
        piece = _generator().generate(Canvas(rows=6, cols=22))
        assert piece is not None
        assert mock_post.call_count == 2

    @patch("plugins.generative_ai_art.source.requests.post")
    def test_gives_up_after_the_attempt_budget(self, mock_post):
        mock_post.return_value = _response("garbage")
        assert _generator().generate(Canvas(rows=6, cols=22)) is None
        assert mock_post.call_count == 3

    @patch("plugins.generative_ai_art.source.requests.post")
    def test_network_error_does_not_retry(self, mock_post):
        import requests as req_module
        mock_post.side_effect = req_module.RequestException("timeout")
        assert _generator().generate(Canvas(rows=6, cols=22)) is None
        assert mock_post.call_count == 1

    @patch("plugins.generative_ai_art.source.requests.post")
    def test_malformed_api_envelope_is_a_validation_failure(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"nope": True}
        mock_post.return_value = mock_resp
        assert _generator().generate(Canvas(rows=6, cols=22)) is None

    @patch("plugins.generative_ai_art.source.requests.post")
    def test_scene_response_is_accepted_on_a_small_board(self, mock_post):
        """The parser takes either format, so an off-format response is not wasted."""
        mock_post.return_value = _response(_scene_response())
        piece = _generator().generate(Canvas(rows=3, cols=15))
        assert piece is not None
        assert piece["art"].count("\n") == 2

    @patch("plugins.generative_ai_art.source.requests.post")
    def test_grid_response_is_accepted_on_a_large_board(self, mock_post):
        mock_post.return_value = _response(_grid_response(rows=24, cols=120))
        piece = _generator().generate(Canvas(rows=24, cols=120))
        assert piece is not None
        assert len(piece["lines"]) == 24

    def test_theme_exclusion_falls_back_when_the_pool_has_one_entry(self):
        gen = _generator(themes=["only theme"])
        assert gen._pick_theme(exclude="only theme") == "only theme"


# ---------------------------------------------------------------------------
# Plugin: validate_config
# ---------------------------------------------------------------------------

class TestPluginValidateConfig:
    def setup_method(self, method):
        self.manifest = {
            "id": "generative_ai_art",
            "name": "Test",
            "version": "1.1.0",
            "description": "",
            "author": "",
            "min_refresh_seconds": 300,
            "settings_schema": {
                "type": "object",
                "properties": {
                    "refresh_seconds": {"type": "integer", "default": 1800, "minimum": 300}
                },
            },
        }

    def _plugin(self):
        return GenerativeAiArtPlugin(self.manifest)

    def test_valid_config_no_errors(self, base_config):
        assert self._plugin().validate_config(base_config) == []

    def test_missing_api_key_is_allowed_for_sign_in(self, base_config):
        """Sign in with OpenRouter replaces the key, so it is no longer required."""
        base_config.pop("api_key")
        assert self._plugin().validate_config(base_config) == []

    def test_empty_api_key_is_allowed_for_sign_in(self, base_config):
        base_config["api_key"] = ""
        assert self._plugin().validate_config(base_config) == []

    def test_legacy_device_type_is_ignored_not_rejected(self, base_config):
        """An existing user's config must keep working after the setting is gone."""
        base_config["device_type"] = "note"
        assert self._plugin().validate_config(base_config) == []

    def test_nonsense_legacy_device_type_is_also_ignored(self, base_config):
        base_config["device_type"] = "mega"
        assert self._plugin().validate_config(base_config) == []

    def test_temperature_out_of_range(self, base_config):
        base_config["temperature"] = 3.5
        assert any("temperature" in e for e in self._plugin().validate_config(base_config))

    def test_bad_base_url(self, base_config):
        base_config["api_base_url"] = "ftp://bad"
        assert any("url" in e.lower() for e in self._plugin().validate_config(base_config))

    def test_refresh_below_minimum(self, base_config):
        base_config["refresh_seconds"] = 60
        assert any("refresh" in e.lower() for e in self._plugin().validate_config(base_config))


# ---------------------------------------------------------------------------
# Plugin: fetch_data (mocked HTTP)
# ---------------------------------------------------------------------------

class TestPluginFetchData:
    def _make_plugin(self, base_config, manifest):
        plugin = GenerativeAiArtPlugin(manifest)
        plugin.config = base_config
        return plugin

    @patch("plugins.generative_ai_art.source.requests.post")
    def test_successful_fetch(self, mock_post, base_config, sample_manifest):
        mock_post.return_value = _response(_scene_response())
        result = self._make_plugin(base_config, sample_manifest).fetch_data()

        assert result.available is True
        assert result.error is None
        for key in ("art", "theme", "description", "model", "generated_at"):
            assert key in result.data

    @patch("plugins.generative_ai_art.source.requests.post")
    def test_art_string_has_color_markers(self, mock_post, base_config, sample_manifest):
        mock_post.return_value = _response(_scene_response())
        result = self._make_plugin(base_config, sample_manifest).fetch_data()
        assert "{" in result.data["art"]

    @patch("plugins.generative_ai_art.source.requests.post")
    def test_formatted_lines_are_emitted_for_the_whole_board(
        self, mock_post, base_config, sample_manifest
    ):
        mock_post.return_value = _response(_scene_response())
        plugin = self._make_plugin(base_config, sample_manifest)
        result = plugin.get_data(FLAGSHIP)
        assert result.formatted_lines is not None
        assert len(result.formatted_lines) == 6
        assert all(count_tiles(line) == 22 for line in result.formatted_lines)

    @pytest.mark.parametrize(
        "board",
        [FLAGSHIP, NOTE, note_array(2, 4), note_array(1, 4), note_array(8, 1), note_array(8, 8)],
        ids=["flagship", "note", "panel-30x12", "tall-15x12", "wide-120x3", "max-120x24"],
    )
    @patch("plugins.generative_ai_art.source.requests.post")
    def test_art_is_sized_to_the_bound_board(
        self, mock_post, board, base_config, sample_manifest
    ):
        mock_post.return_value = _response(_scene_response())
        plugin = self._make_plugin(base_config, sample_manifest)
        result = plugin.get_data(board)
        lines = result.data["art"].split("\n")
        assert len(lines) == board.rows
        assert all(count_tiles(line) == board.cols for line in lines)

    @patch("plugins.generative_ai_art.source.requests.post")
    def test_unbound_board_renders_as_a_flagship(self, mock_post, base_config, sample_manifest):
        mock_post.return_value = _response(_scene_response())
        plugin = self._make_plugin(base_config, sample_manifest)
        result = plugin.get_data(None)
        assert len(result.data["art"].split("\n")) == 6

    @patch("plugins.generative_ai_art.source.requests.post")
    def test_get_formatted_display_follows_the_bound_board(
        self, mock_post, base_config, sample_manifest
    ):
        mock_post.return_value = _response(_scene_response())
        plugin = self._make_plugin(base_config, sample_manifest)
        with plugin._bound_board(note_array(8, 1)):
            lines = plugin.get_formatted_display()
        assert len(lines) == 3
        assert all(count_tiles(line) == 120 for line in lines)

    @patch("plugins.generative_ai_art.source.requests.post")
    def test_fallback_on_api_error(self, mock_post, base_config, sample_manifest):
        """After a successful fetch, an API error returns the last good piece."""
        import requests as req_module

        mock_post.side_effect = [
            _response(_scene_response()),
            req_module.RequestException("timeout"),
        ]
        plugin = self._make_plugin(base_config, sample_manifest)

        first = plugin.fetch_data()
        assert first.available is True

        second = plugin.fetch_data()
        assert second.available is True
        assert second.data["art"] == first.data["art"]
        assert second.data["_fallback"] is True

    @patch("plugins.generative_ai_art.source.requests.post")
    def test_fallback_never_serves_another_boards_frame(
        self, mock_post, base_config, sample_manifest
    ):
        """A Flagship piece must not be handed to a Note when generation fails."""
        import requests as req_module

        mock_post.return_value = _response(_scene_response())
        plugin = self._make_plugin(base_config, sample_manifest)
        with plugin._bound_board(FLAGSHIP):
            assert plugin.fetch_data().available is True

        mock_post.side_effect = req_module.RequestException("timeout")
        mock_post.return_value = None
        with plugin._bound_board(NOTE):
            result = plugin.fetch_data()
        # No Note piece has ever succeeded, so there is nothing to fall back to.
        assert result.available is False

    @patch("plugins.generative_ai_art.source.requests.post")
    def test_fallback_is_per_geometry(self, mock_post, base_config, sample_manifest):
        import requests as req_module

        mock_post.return_value = _response(_scene_response())
        plugin = self._make_plugin(base_config, sample_manifest)
        for board in (FLAGSHIP, NOTE):
            with plugin._bound_board(board):
                plugin.fetch_data()

        mock_post.side_effect = req_module.RequestException("timeout")
        mock_post.return_value = None
        with plugin._bound_board(NOTE):
            result = plugin.fetch_data()
        lines = result.data["art"].split("\n")
        assert len(lines) == NOTE.rows
        assert all(count_tiles(line) == NOTE.cols for line in lines)

    @patch("plugins.generative_ai_art.source.requests.post")
    def test_unavailable_when_no_history_and_failure(self, mock_post, base_config, sample_manifest):
        import requests as req_module

        mock_post.side_effect = req_module.RequestException("timeout")
        assert self._make_plugin(base_config, sample_manifest).fetch_data().available is False

    def test_unavailable_when_no_api_key(self, sample_manifest):
        plugin = GenerativeAiArtPlugin(sample_manifest)
        plugin.config = {"api_key": ""}
        result = plugin.fetch_data()
        assert result.available is False
        assert "api key" in result.error.lower()

    @patch("plugins.generative_ai_art.source.requests.post")
    def test_config_change_resets_generator_and_history(
        self, mock_post, base_config, sample_manifest
    ):
        mock_post.return_value = _response(_scene_response())
        plugin = self._make_plugin(base_config, sample_manifest)
        plugin.fetch_data()
        plugin.on_config_change(base_config, {**base_config, "model": "other"})
        assert plugin._generator is None
        assert plugin._last_pieces == {}

    @patch("plugins.generative_ai_art.source.requests.post")
    def test_remembered_pieces_are_bounded(self, mock_post, base_config, sample_manifest):
        mock_post.return_value = _response(_scene_response())
        plugin = self._make_plugin(base_config, sample_manifest)
        for width in range(1, 25):
            with plugin._bound_board(note_array(min(width, 8), 1)):
                plugin.fetch_data()
            with plugin._bound_board(BoardContext(device_type="note_array", rows=3, cols=15 + width)):
                plugin.fetch_data()
        assert len(plugin._last_pieces) <= 16

    @patch("plugins.generative_ai_art.source.requests.post")
    def test_show_title_uses_the_bottom_row_of_any_board(
        self, mock_post, base_config, sample_manifest
    ):
        mock_post.return_value = _response(_scene_response(title="OCEAN"))
        base_config["show_title"] = True
        plugin = self._make_plugin(base_config, sample_manifest)
        result = plugin.get_data(note_array(2, 4))
        lines = result.data["art"].split("\n")
        assert len(lines) == 12
        assert lines[-1].strip() == "OCEAN"
        assert all(count_tiles(line) == 30 for line in lines)


# ---------------------------------------------------------------------------
# Theme pool
# ---------------------------------------------------------------------------

class TestBuiltinThemes:
    def test_builtin_themes_is_nonempty(self):
        assert len(BUILTIN_THEMES) >= 20

    def test_builtin_themes_all_strings(self):
        for theme in BUILTIN_THEMES:
            assert isinstance(theme, str) and theme.strip()
