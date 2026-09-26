"""Board-geometry conformance for the Generative AI Art plugin.

Imports the shared suite from FiestaBoard core (on ``PYTHONPATH`` in CI), so
"supports every board" means exactly what it means for core's own plugins:
a Flagship, a Note, and note arrays from 15x3 to 120x24 -- which is what a
FiestaPanel is.
"""

import json
from pathlib import Path

import pytest
from src.plugins.geometry_conformance import assert_board_conformance

from plugins.generative_ai_art import GenerativeAiArtPlugin
from plugins.generative_ai_art import source as art_source

_MANIFEST_PATH = Path(__file__).resolve().parent.parent / "manifest.json"
MANIFEST = json.loads(_MANIFEST_PATH.read_text())

#: The conformance suite reads flat ``max_lengths``; this manifest declares the
#: same bounds per variable under ``variables.simple``. Projecting them across
#: is what makes DECLARED_LENGTH_EXCEEDED a live check rather than a no-op --
#: without it the suite would never compare a rendered variable to its bound.
CONFORMANCE_MANIFEST = {
    **MANIFEST,
    "max_lengths": {
        name: spec["max_length"]
        for name, spec in MANIFEST["variables"]["simple"].items()
        if isinstance(spec.get("max_length"), int)
    },
}


# A scene, not a grid: the parser accepts either format on any board, so one
# stub serves every geometry the suite renders -- including the 120x24 panel
# where per-cell emission is not viable in the first place. Deterministic
# (no "noise" shape) so a failure is reproducible.
STUB_SCENE = {
    "theme": "twilight ridge",
    "description": "A dark ridge line under a violet-to-orange sky.",
    "title": "TWILIGHT",
    "background": "K",
    "mirror": "none",
    "shapes": [
        {"type": "gradient", "colors": ["V", "B", "O"], "direction": "vertical"},
        {"type": "ellipse", "color": "Y", "cx": 0.75, "cy": 0.3, "rx": 0.07, "ry": 0.18},
        {"type": "triangle", "color": "K", "points": [[0.0, 1.0], [0.35, 0.3], [0.7, 1.0]]},
        {"type": "wave", "color": "G", "baseline": 0.9, "amplitude": 0.04,
         "frequency": 2, "thickness": 0.2, "fill": "below"},
    ],
}

CONFIG = {
    "enabled": True,
    "api_key": "sk-test",
    "api_base_url": "https://api.openai.com/v1",
    "model": "gpt-4o-mini",
    "temperature": 1.2,
    "refresh_seconds": 300,
}


class _StubResponse:
    """The slice of ``requests.Response`` the generator actually touches."""

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"choices": [{"message": {"content": json.dumps(STUB_SCENE)}}]}


def _stub_post(*args, **kwargs) -> _StubResponse:
    return _StubResponse()


def make_plugin() -> GenerativeAiArtPlugin:
    """A fresh, configured plugin. Network is stubbed by the fixture below."""
    plugin = GenerativeAiArtPlugin(MANIFEST)
    plugin.config = dict(CONFIG)
    return plugin


@pytest.fixture
def no_network(monkeypatch):
    monkeypatch.setattr(art_source.requests, "post", _stub_post)


def test_renders_on_every_board_shape(no_network):
    report = assert_board_conformance(
        make_plugin,
        manifest=CONFORMANCE_MANIFEST,
        strict_growth=True,
        require_note_array_preview=True,
    )
    # The plugin emits whole-board content, so the suite must have had rows to
    # measure. A green run with nothing measured would prove nothing.
    assert not any(w.code == "NO_BOARD_OUTPUT" for w in report.warnings), report.summary()


def test_renders_with_a_title_row_on_every_board_shape(no_network):
    """show_title spends a board row; the art must still fit what is left."""

    def factory() -> GenerativeAiArtPlugin:
        plugin = make_plugin()
        plugin.config = {**CONFIG, "show_title": True}
        return plugin

    assert_board_conformance(
        factory,
        manifest=CONFORMANCE_MANIFEST,
        strict_growth=True,
        require_note_array_preview=True,
    )
