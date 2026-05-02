"""Tests for generative_ai_art plugin."""

import json
from unittest.mock import MagicMock, patch

import pytest

from plugins.generative_ai_art import GenerativeAiArtPlugin
from plugins.generative_ai_art.source import (
    BUILTIN_THEMES,
    ArtGenerator,
    ArtValidationError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_grid(rows: int, cols: int, color: str = "B") -> list:
    """Build a uniform grid for testing."""
    return [[color] * cols for _ in range(rows)]


def _make_grid_two_colors(rows: int, cols: int) -> list:
    """Build a grid with two alternating colors."""
    grid = []
    for r in range(rows):
        row = []
        for c in range(cols):
            row.append("B" if (r + c) % 2 == 0 else "R")
        grid.append(row)
    return grid


def _valid_response(theme: str = "test theme", rows: int = 6, cols: int = 22) -> str:
    """Return a valid JSON response string."""
    return json.dumps({
        "theme": theme,
        "description": "A test composition",
        "grid": _make_grid_two_colors(rows, cols),
    })


# ---------------------------------------------------------------------------
# ArtGenerator._validate_and_parse
# ---------------------------------------------------------------------------

class TestArtValidation:
    """Tests for ArtGenerator._validate_and_parse."""

    def setup_method(self):
        self.gen = ArtGenerator(
            base_url="https://api.example.com/v1",
            api_key="sk-test",
            model="gpt-4o-mini",
            temperature=1.0,
            device_type="flagship",
        )

    def test_valid_flagship_grid(self):
        raw = _valid_response(rows=6, cols=22)
        result = self.gen._validate_and_parse(raw)
        assert result["theme"] == "test theme"
        assert len(result["grid"]) == 6
        assert len(result["grid"][0]) == 22

    def test_valid_note_grid(self):
        gen = ArtGenerator(
            base_url="https://api.example.com/v1",
            api_key="sk-test",
            model="gpt-4o-mini",
            temperature=1.0,
            device_type="note",
        )
        raw = _valid_response(rows=3, cols=15)
        result = gen._validate_and_parse(raw)
        assert len(result["grid"]) == 3
        assert len(result["grid"][0]) == 15

    def test_strips_markdown_fences(self):
        inner = _valid_response(rows=6, cols=22)
        raw = f"```json\n{inner}\n```"
        result = self.gen._validate_and_parse(raw)
        assert result["theme"] == "test theme"

    def test_wrong_row_count(self):
        raw = json.dumps({
            "theme": "t",
            "description": "d",
            "grid": _make_grid_two_colors(4, 22),  # wrong rows
        })
        with pytest.raises(ArtValidationError, match="Expected 6 rows"):
            self.gen._validate_and_parse(raw)

    def test_wrong_col_count(self):
        raw = json.dumps({
            "theme": "t",
            "description": "d",
            "grid": _make_grid_two_colors(6, 20),  # wrong cols
        })
        with pytest.raises(ArtValidationError, match="expected 22 columns"):
            self.gen._validate_and_parse(raw)

    def test_invalid_color_letter(self):
        grid = _make_grid_two_colors(6, 22)
        grid[0][0] = "Z"  # invalid
        raw = json.dumps({"theme": "t", "description": "d", "grid": grid})
        with pytest.raises(ArtValidationError, match="Invalid color"):
            self.gen._validate_and_parse(raw)

    def test_mono_color_rejected(self):
        """A grid with only one color should fail (minimum 2 required)."""
        raw = json.dumps({
            "theme": "t",
            "description": "d",
            "grid": _make_grid(6, 22, "B"),  # all blue
        })
        with pytest.raises(ArtValidationError, match="minimum is 2"):
            self.gen._validate_and_parse(raw)

    def test_missing_key_grid(self):
        raw = json.dumps({"theme": "t", "description": "d"})
        with pytest.raises(ArtValidationError, match="Missing required key: 'grid'"):
            self.gen._validate_and_parse(raw)

    def test_invalid_json(self):
        with pytest.raises(ArtValidationError, match="Invalid JSON"):
            self.gen._validate_and_parse("not json at all")

    def test_all_valid_colors_accepted(self):
        """All 8 valid color codes should be accepted."""
        valid_colors = list("ROYGBVWK")
        # Build grid cycling through all 8 colors using two different colors per row
        grid = []
        for r in range(6):
            c1 = valid_colors[r % 8]
            c2 = valid_colors[(r + 1) % 8]
            # Ensure c1 != c2
            if c1 == c2:
                c2 = valid_colors[(r + 2) % 8]
            row = [c1 if i % 2 == 0 else c2 for i in range(22)]
            grid.append(row)
        raw = json.dumps({"theme": "t", "description": "d", "grid": grid})
        result = self.gen._validate_and_parse(raw)
        assert result is not None


# ---------------------------------------------------------------------------
# ArtGenerator._grid_to_art_string
# ---------------------------------------------------------------------------

class TestGridToArtString:
    """Tests for ArtGenerator._grid_to_art_string."""

    def setup_method(self):
        self.gen = ArtGenerator(
            base_url="https://api.example.com/v1",
            api_key="sk-test",
            model="test",
            temperature=1.0,
            device_type="flagship",
        )

    def test_single_cell_red(self):
        result = self.gen._grid_to_art_string([["R"]])
        assert result == "{red}"

    def test_row_of_two_colors(self):
        result = self.gen._grid_to_art_string([["B", "R"]])
        assert result == "{blue}{red}"

    def test_two_rows_newline_separated(self):
        result = self.gen._grid_to_art_string([["B", "B"], ["R", "R"]])
        assert result == "{blue}{blue}\n{red}{red}"

    def test_all_color_codes_rendered(self):
        row = list("ROYGBVWK")
        result = self.gen._grid_to_art_string([row])
        for marker in ["{red}", "{orange}", "{yellow}", "{green}",
                       "{blue}", "{violet}", "{white}", "{black}"]:
            assert marker in result

    def test_flagship_grid_line_count(self):
        grid = _make_grid_two_colors(6, 22)
        result = self.gen._grid_to_art_string(grid)
        assert result.count("\n") == 5  # 6 rows → 5 newlines

    def test_note_grid_line_count(self):
        gen = ArtGenerator(
            base_url="https://api.example.com/v1",
            api_key="sk-test",
            model="test",
            temperature=1.0,
            device_type="note",
        )
        grid = _make_grid_two_colors(3, 15)
        result = gen._grid_to_art_string(grid)
        assert result.count("\n") == 2  # 3 rows → 2 newlines


# ---------------------------------------------------------------------------
# Plugin: validate_config
# ---------------------------------------------------------------------------

class TestPluginValidateConfig:
    """Tests for GenerativeAiArtPlugin.validate_config."""

    def setup_method(self, method):
        self.manifest = {
            "id": "generative_ai_art",
            "name": "Test",
            "version": "1.0.0",
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
        errors = self._plugin().validate_config(base_config)
        assert errors == []

    def test_missing_api_key(self, base_config):
        base_config.pop("api_key")
        errors = self._plugin().validate_config(base_config)
        assert any("api key" in e.lower() for e in errors)

    def test_empty_api_key(self, base_config):
        base_config["api_key"] = ""
        errors = self._plugin().validate_config(base_config)
        assert any("api key" in e.lower() for e in errors)

    def test_invalid_device_type(self, base_config):
        base_config["device_type"] = "mega"
        errors = self._plugin().validate_config(base_config)
        assert any("device_type" in e for e in errors)

    def test_valid_note_device_type(self, base_config):
        base_config["device_type"] = "note"
        errors = self._plugin().validate_config(base_config)
        assert errors == []

    def test_history_size_too_large(self, base_config):
        base_config["history_size"] = 9999
        errors = self._plugin().validate_config(base_config)
        assert any("history_size" in e for e in errors)

    def test_history_size_zero(self, base_config):
        base_config["history_size"] = 0
        errors = self._plugin().validate_config(base_config)
        assert any("history_size" in e for e in errors)

    def test_temperature_out_of_range(self, base_config):
        base_config["temperature"] = 3.5
        errors = self._plugin().validate_config(base_config)
        assert any("temperature" in e for e in errors)

    def test_bad_base_url(self, base_config):
        base_config["api_base_url"] = "ftp://bad"
        errors = self._plugin().validate_config(base_config)
        assert any("url" in e.lower() for e in errors)

    def test_refresh_below_minimum(self, base_config):
        base_config["refresh_seconds"] = 60
        errors = self._plugin().validate_config(base_config)
        assert any("refresh" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# Plugin: fetch_data (mocked HTTP)
# ---------------------------------------------------------------------------

class TestPluginFetchData:
    """Tests for GenerativeAiArtPlugin.fetch_data with mocked HTTP."""

    def _make_plugin(self, base_config, manifest):
        plugin = GenerativeAiArtPlugin(manifest)
        plugin.config = base_config
        return plugin

    def _mock_response(self, rows: int = 6, cols: int = 22, theme: str = "ocean waves") -> MagicMock:
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": _valid_response(theme=theme, rows=rows, cols=cols)
                    }
                }
            ]
        }
        return mock_resp

    @patch("plugins.generative_ai_art.source.requests.post")
    def test_successful_fetch(self, mock_post, base_config, sample_manifest):
        mock_post.return_value = self._mock_response()
        plugin = self._make_plugin(base_config, sample_manifest)

        result = plugin.fetch_data()

        assert result.available is True
        assert result.error is None
        assert "art" in result.data
        assert "theme" in result.data
        assert "description" in result.data
        assert "model" in result.data
        assert "generated_at" in result.data

    @patch("plugins.generative_ai_art.source.requests.post")
    def test_art_string_has_color_markers(self, mock_post, base_config, sample_manifest):
        mock_post.return_value = self._mock_response()
        plugin = self._make_plugin(base_config, sample_manifest)
        result = plugin.fetch_data()

        assert "{blue}" in result.data["art"] or "{red}" in result.data["art"]

    @patch("plugins.generative_ai_art.source.requests.post")
    def test_history_grows(self, mock_post, base_config, sample_manifest):
        mock_post.return_value = self._mock_response()
        plugin = self._make_plugin(base_config, sample_manifest)

        plugin.fetch_data()
        plugin.fetch_data()
        plugin.fetch_data()

        assert len(plugin._history) == 3

    @patch("plugins.generative_ai_art.source.requests.post")
    def test_fallback_on_api_error(self, mock_post, base_config, sample_manifest):
        """After a successful fetch, an API error should return the last good piece."""
        import requests as req_module

        # First call succeeds, second raises
        mock_post.side_effect = [
            self._mock_response(),
            req_module.RequestException("timeout"),
        ]
        plugin = self._make_plugin(base_config, sample_manifest)

        first = plugin.fetch_data()
        assert first.available is True

        second = plugin.fetch_data()
        # Should fall back to last known piece
        assert second.available is True
        assert second.data["art"] == first.data["art"]

    @patch("plugins.generative_ai_art.source.requests.post")
    def test_unavailable_when_no_history_and_failure(self, mock_post, base_config, sample_manifest):
        """With empty history and API failure the plugin should be unavailable."""
        import requests as req_module

        mock_post.side_effect = req_module.RequestException("timeout")
        plugin = self._make_plugin(base_config, sample_manifest)

        result = plugin.fetch_data()
        assert result.available is False

    def test_unavailable_when_no_api_key(self, sample_manifest):
        plugin = GenerativeAiArtPlugin(sample_manifest)
        plugin.config = {"api_key": ""}
        result = plugin.fetch_data()
        assert result.available is False
        assert "api key" in result.error.lower()

    @patch("plugins.generative_ai_art.source.requests.post")
    def test_note_device_type(self, mock_post, base_config, sample_manifest):
        mock_post.return_value = self._mock_response(rows=3, cols=15)
        base_config["device_type"] = "note"
        plugin = self._make_plugin(base_config, sample_manifest)

        result = plugin.fetch_data()
        assert result.available is True
        # Note grid has 3 rows → 2 newlines in the art string
        assert result.data["art"].count("\n") == 2


# ---------------------------------------------------------------------------
# Plugin: history persistence (mocked filesystem)
# ---------------------------------------------------------------------------

class TestPluginHistoryPersistence:
    """Tests for history save/load behaviour."""

    @patch("plugins.generative_ai_art.source.requests.post")
    def test_history_not_saved_when_persist_false(self, mock_post, base_config, sample_manifest, tmp_path):
        mock_post.return_value = MagicMock(
            raise_for_status=MagicMock(),
            json=MagicMock(return_value={
                "choices": [{"message": {"content": _valid_response()}}]
            }),
        )
        base_config["persist_history"] = False
        plugin = GenerativeAiArtPlugin(sample_manifest)
        plugin.config = base_config
        plugin.fetch_data()

        # Nothing should be written
        import plugins.generative_ai_art as plugin_mod
        history_file = plugin_mod._HISTORY_FILE
        assert not history_file.exists()

    @patch("plugins.generative_ai_art.source.requests.post")
    def test_history_loaded_on_init(self, mock_post, tmp_path, sample_manifest):
        """Pre-seeded history on disk should be loaded when the generator is first built."""
        import plugins.generative_ai_art as plugin_mod

        pre_records = [
            {
                "theme": "old theme",
                "description": "old desc",
                "art": "{blue}",
                "model": "gpt-4o-mini",
                "generated_at": "2026-01-01T00:00:00",
            }
        ]

        history_dir = tmp_path / "data" / "plugins" / "generative_ai_art"
        history_dir.mkdir(parents=True)
        history_file = history_dir / "history.json"
        history_file.write_text(json.dumps(pre_records))

        mock_post.return_value = MagicMock(
            raise_for_status=MagicMock(),
            json=MagicMock(return_value={
                "choices": [{"message": {"content": _valid_response()}}]
            }),
        )

        original_file = plugin_mod._HISTORY_FILE
        original_dir = plugin_mod._HISTORY_DIR
        try:
            plugin_mod._HISTORY_FILE = history_file
            plugin_mod._HISTORY_DIR = history_dir

            plugin = GenerativeAiArtPlugin(sample_manifest)
            plugin.config = {
                "api_key": "sk-test",
                "api_base_url": "https://api.openai.com/v1",
                "model": "gpt-4o-mini",
                "device_type": "flagship",
                "temperature": 1.2,
                "refresh_seconds": 1800,
                "history_size": 100,
                "persist_history": True,
            }

            # Trigger generator build (which loads history)
            plugin._get_generator()

            assert len(plugin._history) == 1
            assert plugin._history[0]["theme"] == "old theme"
        finally:
            plugin_mod._HISTORY_FILE = original_file
            plugin_mod._HISTORY_DIR = original_dir


# ---------------------------------------------------------------------------
# Theme pool
# ---------------------------------------------------------------------------

class TestBuiltinThemes:
    def test_builtin_themes_is_nonempty(self):
        assert len(BUILTIN_THEMES) >= 20

    def test_builtin_themes_all_strings(self):
        for theme in BUILTIN_THEMES:
            assert isinstance(theme, str) and theme.strip()
