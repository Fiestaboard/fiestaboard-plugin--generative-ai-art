"""Authored pixels retain their exact positions through the native canvas path."""
import json

import pytest

from plugins.generative_ai_art.pixel_canvas import PixelTarget, parse_canvas_response
from plugins.generative_ai_art.source import ArtValidationError

from .test_pixel_canvas import _fetch, _plugin, pixel_board


def parse(scene, width=4, height=2):
    return parse_canvas_response(json.dumps(scene), PixelTarget(width, height))["content"]


def test_authored_rows_are_native_size_with_literal_rgb_colors():
    content = parse({"palette": {"a": "#12ab34", "b": "#F0A"}, "pixels": ["a..b", ".a"]})
    assert content["size"] == [4, 2]
    assert content["palette"][content["pixels"][0][0]] == "#12ab34"
    assert content["palette"][content["pixels"][0][3]] == "#f0a"
    assert content["pixels"][0][1:3] == ".."
    assert content["pixels"][1][0] == "."


@pytest.mark.parametrize("fields", [
    {"palette": {"a": "#fff"}},
    {"pixels": ["a"]},
    {"palette": {".": "#fff"}, "pixels": ["."]},
    {"palette": {"ab": "#fff"}, "pixels": ["a"]},
    {"palette": {"é": "#fff"}, "pixels": ["é"]},
    {"palette": {"a": "{{secret.value}}"}, "pixels": ["a"]},
    {"palette": {"a": "red"}, "pixels": ["a"]},
    {"palette": {"a": 123}, "pixels": ["a"]},
    {"palette": [], "pixels": ["a"]},
    {"palette": {"a": "#fff"}, "pixels": "a"},
    {"palette": {"a": "#fff"}, "pixels": [["a"]]},
    {"palette": {"a": "#fff"}, "pixels": ["b"]},
    {"palette": {"a": "#fff"}, "pixels": ["aaaaa"]},
    {"palette": {"a": "#fff"}, "pixels": ["a", "a", "a"]},
    {"palette": {str(i): "#fff" for i in range(63)}, "pixels": ["a"]},
])
def test_bad_authored_data_is_rejected_even_when_shapes_are_valid(fields):
    scene = {"shapes": [{"type": "rect", "x": 0, "y": 0, "w": 4, "h": 2, "fill": "#123456"}], **fields}
    with pytest.raises(ArtValidationError):
        parse(scene)


def test_authored_pixels_overlay_noise_without_recoloring_it():
    from plugins.generative_ai_art.pixel_canvas import scene_to_content

    noise = {"type": "noise", "colors": ["#ffffff"], "density": 0.5}
    base = scene_to_content({"shapes": [noise]}, PixelTarget(8, 8))
    content = parse({"shapes": [noise], "palette": {"a": "#12ab34"}, "pixels": ["a"]}, 8, 8)
    assert content["palette"][content["pixels"][0][0]] == "#12ab34"
    # Every other noise pixel keeps its color; transparent authored space leaves it alone.
    for y, row in enumerate(base["pixels"]):
        for x, key in enumerate(row):
            if (x, y) != (0, 0) and key != ".":
                assert content["palette"][content["pixels"][y][x]] == "#ffffff"


def test_combined_palette_overflow_rejects_instead_of_dropping_color():
    keys = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    palette = {key: f"#{index + 1:06x}" for index, key in enumerate(keys)}
    with pytest.raises(ArtValidationError):
        parse({"shapes": [{"type": "noise", "colors": ["#ffffff"], "density": 0.5}],
               "palette": palette, "pixels": [keys]}, 64, 64)


def test_all_62_palette_entries_fit_and_authored_coordinates_ignore_mirror():
    keys = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    palette = {key: f"#{index + 1:06x}" for index, key in enumerate(keys)}
    content = parse({"palette": palette, "pixels": [keys], "mirror": "both"}, 64, 64)
    assert len(content["palette"]) == 62
    assert content["pixels"] == [keys]
    assert content["palette"] == palette


def test_transparent_only_rows_are_not_accepted_as_artwork():
    with pytest.raises(ArtValidationError):
        parse({"palette": {}, "pixels": ["...."]})


def test_pixel_reply_flows_through_plugin_with_one_model_call():
    result, post = _fetch(_plugin(), pixel_board(32, 16), payload={
        "theme": "corner", "palette": {"a": "#12ab34"}, "pixels": ["a"],
    })
    assert post.call_count == 1
    assert result.data["canvas"]["size"] == [32, 16]
    assert result.data["canvas"]["palette"][result.data["canvas"]["pixels"][0][0]] == "#12ab34"
    assert not result.data.get("_fallback")


@pytest.mark.parametrize("font,gap,padding", [("3x5", "gap", 0), ("5x7", "fill", 1)])
def test_full_bleed_native_pixels_reach_exact_panel_positions(font, gap, padding):
    pytest.importorskip("src.canvas")
    from src.canvas.layer import render_canvases_on_grid
    from src.canvas.models import Canvas
    from src.led import LedLayoutOptions, LedMatrixSpec, layout_message, rasterize
    from src.led.matrix import grid_layout

    rows = ["." * 64 for _ in range(64)]
    rows[0] = "a" + "." * 62 + "b"
    rows[31] = "." * 17 + "a" + "." * 46
    rows[63] = "b" + "." * 62 + "a"
    content = parse({"background": "#010203", "palette": {"a": "#12ab34", "b": "#ef5678"},
                     "pixels": rows, "shapes": [
                         {"type": "rect", "x": 16, "y": 30, "w": 3, "h": 3, "fill": "#203040"},
                     ]}, 64, 64)
    canvas = Canvas.model_validate({"id": "art", "area": {"row": 1, "col": 1, "rows": 96, "cols": 128},
                                    "bleed": ["all"], "scale": 1, "content": content})
    layers, issues = render_canvases_on_grid([canvas], grid_layout(64, 64, font), {})
    assert not issues
    assert (layers[0].x, layers[0].y, layers[0].w, layers[0].h) == (0, 0, 64, 64)
    result = rasterize(layout_message("UNDERLYING TEXT", LedMatrixSpec(64, 64, font),
                                     LedLayoutOptions(tile_gap=gap, block_padding=padding), layers=layers))
    expected = bytearray(bytes([1, 2, 3]) * 4096)
    for y in range(30, 33):
        for x in range(16, 19):
            offset = (y * 64 + x) * 3
            expected[offset:offset + 3] = bytes([32, 48, 64])
    for x, y, rgb in [(0, 0, [18, 171, 52]), (63, 0, [239, 86, 120]),
                      (17, 31, [18, 171, 52]), (0, 63, [239, 86, 120]), (63, 63, [18, 171, 52])]:
        offset = (y * 64 + x) * 3
        expected[offset:offset + 3] = bytes(rgb)
    assert result.pixels == bytes(expected)


@pytest.mark.parametrize("raw", [
    '{"palette":{"a":"#12ab34"},"pixels":["a", "a',
    '{"shapes":[{"type":"rect","x":0,"y":0,"w":4,"h":2,"fill":"#123456"}],"pixels":["a',
])
def test_truncated_authored_rows_are_rejected_not_salvaged_as_partial_art(raw):
    with pytest.raises(ArtValidationError):
        parse_canvas_response(raw, PixelTarget(4, 2))


@pytest.mark.parametrize("raw", [
    r'{"palett\u0065":{"a":"#12ab34"},"pix\u0065ls":["a", "a',
    r'{"shapes":[{"type":"rect","x":0,"y":0,"w":4,"h":2,"fill":"#123456"}],"pix\u0065ls":["a',
])
def test_truncated_authored_rows_with_escaped_keys_are_not_salvaged(raw):
    with pytest.raises(ArtValidationError):
        parse_canvas_response(raw, PixelTarget(4, 2))


def test_legacy_scene_salvage_ignores_key_like_text_inside_values():
    raw = r'{"description":"a label: \"pixels\": example", "shapes":[{"type":"rect","x":0,"y":0,"w":4,"h":2,"fill":"#123456"},'
    content = parse_canvas_response(raw, PixelTarget(4, 2))["content"]
    assert content["shapes"] == [{"type": "rect", "x": 0, "y": 0, "w": 4, "h": 2, "fill": "#123456"}]
