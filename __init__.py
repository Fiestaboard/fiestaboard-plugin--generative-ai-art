"""Generative AI Art plugin for FiestaBoard.

Generates full-screen abstract art for your split-flap display by calling
an OpenAI-compatible chat completions endpoint.  Each piece is a unique
colour-tile composition using the board's 8-colour palette.

Supports:
- Flagship display (6 rows × 22 cols)
- Note display (3 rows × 15 cols)
- Any OpenAI v1-compatible endpoint (OpenAI, OpenRouter, Ollama, etc.)
- Ring-buffer history of the last N generated pieces
- Optional persistence of history to disk so it survives restarts
"""

import json
import logging
import os
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from src.plugins.base import PluginBase, PluginResult

try:
    from .source import ArtGenerator
except ImportError:
    # Fallback for when __init__.py is imported as a top-level module
    # (e.g., during pytest package setup before the package context is known).
    from source import ArtGenerator  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

# Where to persist history relative to the process working directory.
# Matches the /data/ mount used by the FiestaBoard docker-compose setup.
_HISTORY_DIR = Path("data") / "plugins" / "generative_ai_art"
_HISTORY_FILE = _HISTORY_DIR / "history.json"

# Maximum pieces allowed in the history JSON file regardless of config.
_ABSOLUTE_MAX_HISTORY = 500


class GenerativeAiArtPlugin(PluginBase):
    """Generative AI Art plugin.

    On each ``fetch_data`` call the plugin asks the configured LLM to compose
    a new art piece for the board dimensions, stores it in a fixed-length
    history ring, optionally persists history to disk, and returns the latest
    piece as template variables.

    If the LLM call fails (network error, invalid response, etc.) the plugin
    falls back to the most recently successful piece so the board keeps
    displaying something.
    """

    def __init__(self, manifest: Dict[str, Any]) -> None:
        super().__init__(manifest)
        self._generator: Optional[ArtGenerator] = None
        self._history: Deque[Dict[str, Any]] = deque(maxlen=100)

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

        history_size = config.get("history_size", 100)
        if not isinstance(history_size, int) or not (1 <= history_size <= _ABSOLUTE_MAX_HISTORY):
            errors.append(f"history_size must be an integer between 1 and {_ABSOLUTE_MAX_HISTORY}")

        temperature = config.get("temperature", 1.2)
        if not isinstance(temperature, (int, float)) or not (0 <= temperature <= 2):
            errors.append("temperature must be a number between 0 and 2")

        errors.extend(self._validate_refresh_seconds(config))
        return errors

    def on_config_change(
        self, old_config: Dict[str, Any], new_config: Dict[str, Any]
    ) -> None:
        """Reset generator when API settings change; resize history ring if needed."""
        # Rebuild generator so next fetch picks up new credentials / model
        self._generator = None

        # Resize history deque if history_size changed
        new_size = int(new_config.get("history_size", 100))
        new_size = max(1, min(new_size, _ABSOLUTE_MAX_HISTORY))
        if self._history.maxlen != new_size:
            self._history = deque(list(self._history), maxlen=new_size)

        logger.debug("GenerativeAiArtPlugin config updated, generator reset")

    def fetch_data(self) -> PluginResult:
        """Generate a new art piece and return it as plugin data.

        If ``paused`` is True or ``pin_offset`` is non-zero the plugin returns
        a piece from history without calling the LLM.  If history is empty it
        falls through to generation so the board always has something to show.

        On LLM failure returns the last known good piece if available, or marks
        the plugin as unavailable if history is empty.
        """
        cfg = self.config
        if not cfg.get("api_key"):
            return PluginResult(available=False, error="API key not configured")

        generator = self._get_generator()
        if generator is None:
            return PluginResult(available=False, error="Art generator could not be initialised")

        paused = bool(cfg.get("paused", False))
        pin_offset = int(cfg.get("pin_offset", 0))

        if (paused or pin_offset > 0) and self._history:
            hist = list(self._history)
            idx = min(pin_offset, len(hist) - 1)
            return PluginResult(available=True, data=self._record_to_data(hist[-(idx + 1)]))

        piece = generator.generate()

        if piece is None:
            logger.warning("Art generation failed; trying history fallback")
            return self._fallback_result("Art generation failed; showing last known piece")

        # Successful — record and persist
        record: Dict[str, Any] = {
            "theme": piece["theme"],
            "description": piece["description"],
            "art": piece["art"],
            "model": cfg.get("model", "unknown"),
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self._history.append(record)

        if cfg.get("persist_history", True):
            self._save_history()

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
        )

        # Load persisted history the first time we build the generator
        if cfg.get("persist_history", True) and not self._history:
            new_size = int(cfg.get("history_size", 100))
            new_size = max(1, min(new_size, _ABSOLUTE_MAX_HISTORY))
            self._history = deque(maxlen=new_size)
            self._load_history()

        return self._generator

    def _fallback_result(self, error: str) -> PluginResult:
        """Return the most recently generated piece, or unavailable if none."""
        if self._history:
            last = self._history[-1]
            logger.info("Returning last-known art piece as fallback")
            data = self._record_to_data(last)
            data["_fallback"] = True
            return PluginResult(available=True, data=data)

        return PluginResult(available=False, error=error)

    @staticmethod
    def _record_to_data(record: Dict[str, Any]) -> Dict[str, Any]:
        """Convert an internal history record to the public data dict."""
        return {
            "art": record.get("art", ""),
            "theme": record.get("theme", ""),
            "description": record.get("description", ""),
            "model": record.get("model", ""),
            "generated_at": record.get("generated_at", ""),
        }

    # ------------------------------------------------------------------
    # History persistence
    # ------------------------------------------------------------------

    def _save_history(self) -> None:
        """Write current history to disk as JSON, silently catching all errors."""
        try:
            _HISTORY_DIR.mkdir(parents=True, exist_ok=True)
            payload = list(self._history)
            tmp = _HISTORY_FILE.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
            tmp.replace(_HISTORY_FILE)
            logger.debug("Saved %d history entries to %s", len(payload), _HISTORY_FILE)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to save art history: %s", exc)

    def _load_history(self) -> None:
        """Load history from disk into the deque, silently catching all errors."""
        if not _HISTORY_FILE.exists():
            return
        try:
            with _HISTORY_FILE.open("r", encoding="utf-8") as fh:
                records = json.load(fh)
            if not isinstance(records, list):
                return
            # Only keep records that have the expected shape
            loaded = 0
            for rec in records:
                if isinstance(rec, dict) and "art" in rec and "theme" in rec:
                    self._history.append(rec)
                    loaded += 1
            logger.info("Loaded %d art history entries from %s", loaded, _HISTORY_FILE)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load art history: %s", exc)


# Export the plugin class
Plugin = GenerativeAiArtPlugin
