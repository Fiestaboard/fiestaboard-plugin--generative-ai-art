"""Generative AI Art plugin for FiestaBoard.

Generates full-screen abstract art for your split-flap display by calling
an OpenAI-compatible chat completions endpoint.  Each piece is a unique
colour-tile composition using the board's 8-colour palette.

The board is never configured: the platform binds ``self.board`` for the
render and every dimension is derived from it, so one configuration serves
every board the user owns —

- Flagship (22 × 6)
- Note (15 × 3)
- Note array / FiestaPanel, anything from 15 × 3 to 120 × 24

Works with any OpenAI v1-compatible endpoint (OpenAI, OpenRouter, Ollama, …).
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.plugins.base import PluginBase, PluginResult

try:
    from .source import ArtGenerator, Canvas
except ImportError:
    # Fallback for when __init__.py is imported as a top-level module
    # (e.g., during pytest package setup before the package context is known).
    from source import ArtGenerator, Canvas  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

#: How many per-geometry fallback pieces to keep. A user can own several
#: boards; keeping one piece each means an outage on a Note never puts a
#: 120-column frame on it. Bounded so a long-lived process cannot grow a
#: cache entry per geometry it has ever seen.
MAX_REMEMBERED_PIECES = 16


class GenerativeAiArtPlugin(PluginBase):
    """Generative AI Art plugin.

    On each ``fetch_data`` call the plugin asks the configured LLM to compose
    a new art piece **for the board currently bound to ``self.board``** and
    returns it both as ``formatted_lines`` (whole-board content) and as
    template variables.  If the LLM call fails the plugin falls back to the
    most recently successful piece *for that same geometry* so the board
    keeps displaying something that actually fits it.
    """

    def __init__(self, manifest: Dict[str, Any]) -> None:
        super().__init__(manifest)
        # The generator holds no geometry — the canvas is passed per call —
        # so a single instance is safe to share across every board.
        self._generator: Optional[ArtGenerator] = None
        # Fallback pieces keyed by canvas geometry. Keying matters: without
        # it an outage on one board serves another board's frame, which is
        # either an overflow or a mostly-blank panel.
        self._last_pieces: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # PluginBase interface
    # ------------------------------------------------------------------

    @property
    def plugin_id(self) -> str:
        return "generative_ai_art"

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate plugin configuration.

        Note there is deliberately no board-size setting to validate: the
        platform knows which board it is rendering, and a user with a
        Flagship and a panel must not have to pick one.  A ``device_type``
        left over from an older config is ignored rather than rejected.
        """
        errors: List[str] = []

        if not config.get("api_key"):
            errors.append("API key is required")

        base_url = config.get("api_base_url", "")
        if base_url and not (
            base_url.startswith("http://") or base_url.startswith("https://")
        ):
            errors.append("API base URL must start with http:// or https://")

        temperature = config.get("temperature", 1.2)
        if not isinstance(temperature, (int, float)) or not (0 <= temperature <= 2):
            errors.append("temperature must be a number between 0 and 2")

        errors.extend(self._validate_refresh_seconds(config))
        return errors

    def on_config_change(
        self, old_config: Dict[str, Any], new_config: Dict[str, Any]
    ) -> None:
        """Reset generator and remembered pieces when settings change."""
        self._generator = None
        self._last_pieces.clear()
        logger.debug("GenerativeAiArtPlugin config updated, generator reset")

    def fetch_data(self) -> PluginResult:
        """Generate a new art piece sized to the bound board.

        On LLM failure returns the last known good piece *for this geometry*
        if there is one, or marks the plugin as unavailable.
        """
        cfg = self.config
        canvas = self.canvas()

        if not cfg.get("api_key"):
            return PluginResult(available=False, error="API key not configured")

        generator = self._get_generator()
        if generator is None:
            return PluginResult(
                available=False, error="Art generator could not be initialised"
            )

        piece = generator.generate(canvas)

        if piece is None:
            logger.warning(
                "Art generation failed for %s; trying last-piece fallback", canvas.key
            )
            return self._fallback_result(
                canvas, "Art generation failed; showing last known piece"
            )

        record: Dict[str, Any] = {
            "theme": piece["theme"],
            "description": piece["description"],
            "art": piece["art"],
            "lines": piece["lines"],
            "model": cfg.get("model", "unknown"),
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self._remember(canvas, record)

        return PluginResult(
            available=True,
            data=self._record_to_data(record),
            formatted_lines=list(record["lines"]),
        )

    def get_formatted_display(self) -> List[str]:
        """Board rows for the currently bound board.

        Not called by the platform today, but it is the documented contract,
        so it is held to the same bounds as ``formatted_lines``: it renders
        for ``self.board``, never for a remembered other board.  Routed
        through ``get_data`` so it shares the per-geometry cache rather than
        spending a second generation on every call.
        """
        canvas = self.canvas()
        result = self.get_data(self.board)
        if not result.available or not result.formatted_lines:
            return []
        return list(result.formatted_lines)[: canvas.rows]

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def canvas(self) -> Canvas:
        """The canvas for the board being rendered.

        ``self.board`` is ``None`` outside a board-scoped render (unit tests,
        legacy callers); the platform contract is to assume a Flagship rather
        than to crash, which :meth:`Canvas.default` does.
        """
        show_title = bool(self.config.get("show_title", False))
        board = self.board
        if board is None:
            return Canvas.default(show_title=show_title)
        return Canvas(rows=board.rows, cols=board.cols, show_title=show_title)

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
            themes=themes if isinstance(themes, list) else [],
            extra_instructions=cfg.get("extra_instructions", ""),
            custom_system_prompt=cfg.get("custom_system_prompt", ""),
            show_title=bool(cfg.get("show_title", False)),
        )

        return self._generator

    def _remember(self, canvas: Canvas, record: Dict[str, Any]) -> None:
        """Store *record* as the fallback piece for *canvas*'s geometry."""
        self._last_pieces[canvas.key] = record
        while len(self._last_pieces) > MAX_REMEMBERED_PIECES:
            self._last_pieces.pop(next(iter(self._last_pieces)))

    def _fallback_result(self, canvas: Canvas, error: str) -> PluginResult:
        """Return the last piece generated *for this geometry*, if any.

        A piece from a different geometry is never reused: it would either
        overflow the board or leave most of it blank.
        """
        record = self._last_pieces.get(canvas.key)
        if record is not None:
            logger.info("Returning last-known art piece for %s as fallback", canvas.key)
            data = self._record_to_data(record)
            data["_fallback"] = True
            return PluginResult(
                available=True,
                data=data,
                formatted_lines=list(record.get("lines") or []),
            )

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
