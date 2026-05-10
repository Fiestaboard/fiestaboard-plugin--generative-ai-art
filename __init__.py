"""Generative AI Art plugin for FiestaBoard.

Generates full-screen abstract art for your split-flap display by calling
an OpenAI-compatible chat completions endpoint.  Each piece is a unique
colour-tile composition using the board's 8-colour palette.

Supports:
- Flagship display (6 rows × 22 cols)
- Note display (3 rows × 15 cols)
- Any OpenAI v1-compatible endpoint (OpenAI, OpenRouter, Ollama, etc.)
"""

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.plugins.base import PluginBase, PluginResult

try:
    from .source import ArtGenerator
except ImportError:
    # Fallback for when __init__.py is imported as a top-level module
    # (e.g., during pytest package setup before the package context is known).
    from source import ArtGenerator  # type: ignore[no-redef]

logger = logging.getLogger(__name__)


class GenerativeAiArtPlugin(PluginBase):
    """Generative AI Art plugin.

    On each ``fetch_data`` call the plugin asks the configured LLM to compose
    a new art piece for the board dimensions and returns it as template
    variables.  If the LLM call fails the plugin falls back to the most
    recently successful piece so the board keeps displaying something.
    """

    def __init__(self, manifest: Dict[str, Any]) -> None:
        super().__init__(manifest)
        self._generator: Optional[ArtGenerator] = None
        self._last_piece: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # PluginBase interface
    # ------------------------------------------------------------------

    @property
    def plugin_id(self) -> str:
        return "generative_ai_art"

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate plugin configuration."""
        errors: List[str] = []

        if not config.get("api_key"):
            errors.append("API key is required")

        base_url = config.get("api_base_url", "")
        if base_url and not (
            base_url.startswith("http://") or base_url.startswith("https://")
        ):
            errors.append("API base URL must start with http:// or https://")

        device_type = config.get("device_type", "flagship")
        if device_type not in ("flagship", "note"):
            errors.append("device_type must be 'flagship' or 'note'")

        temperature = config.get("temperature", 1.2)
        if not isinstance(temperature, (int, float)) or not (0 <= temperature <= 2):
            errors.append("temperature must be a number between 0 and 2")

        errors.extend(self._validate_refresh_seconds(config))
        return errors

    def on_config_change(
        self, old_config: Dict[str, Any], new_config: Dict[str, Any]
    ) -> None:
        """Reset generator when API settings change."""
        self._generator = None
        logger.debug("GenerativeAiArtPlugin config updated, generator reset")

    def fetch_data(self) -> PluginResult:
        """Generate a new art piece and return it as plugin data.

        On LLM failure returns the last known good piece if available, or
        marks the plugin as unavailable if no piece has been generated yet.
        """
        cfg = self.config
        if not cfg.get("api_key"):
            return PluginResult(available=False, error="API key not configured")

        generator = self._get_generator()
        if generator is None:
            return PluginResult(available=False, error="Art generator could not be initialised")

        piece = generator.generate()

        if piece is None:
            logger.warning("Art generation failed; trying last-piece fallback")
            return self._fallback_result("Art generation failed; showing last known piece")

        record: Dict[str, Any] = {
            "theme": piece["theme"],
            "description": piece["description"],
            "art": piece["art"],
            "model": cfg.get("model", "unknown"),
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self._last_piece = record

        return PluginResult(
            available=True,
            data=self._record_to_data(record),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_generator(self) -> Optional[ArtGenerator]:
        """Return (or lazily create) the ArtGenerator for the current config."""
        if self._generator is not None:
            return self._generator

        cfg = self.config
        api_key = cfg.get("api_key", "")
        if not api_key:
            return None

        themes = cfg.get("themes") or []
        self._generator = ArtGenerator(
            base_url=cfg.get("api_base_url", "https://api.openai.com/v1"),
            api_key=api_key,
            model=cfg.get("model", "gpt-4o-mini"),
            temperature=float(cfg.get("temperature", 1.2)),
            device_type=cfg.get("device_type", "flagship"),
            themes=themes if isinstance(themes, list) else [],
            extra_instructions=cfg.get("extra_instructions", ""),
            custom_system_prompt=cfg.get("custom_system_prompt", ""),
            show_title=bool(cfg.get("show_title", False)),
        )

        return self._generator

    def _fallback_result(self, error: str) -> PluginResult:
        """Return the most recently generated piece, or unavailable if none."""
        if self._last_piece is not None:
            logger.info("Returning last-known art piece as fallback")
            data = self._record_to_data(self._last_piece)
            data["_fallback"] = True
            return PluginResult(available=True, data=data)

        return PluginResult(available=False, error=error)

    @staticmethod
    def _record_to_data(record: Dict[str, Any]) -> Dict[str, Any]:
        """Convert an internal piece record to the public data dict."""
        return {
            "art": record.get("art", ""),
            "theme": record.get("theme", ""),
            "description": record.get("description", ""),
            "model": record.get("model", ""),
            "generated_at": record.get("generated_at", ""),
        }


# Export the plugin class
Plugin = GenerativeAiArtPlugin
