"""Generative AI Art plugin for FiestaBoard.

Generates full-screen abstract art for your board by calling an AI model.
On a split-flap board each piece is a colour-tile composition in the board's
8-colour palette (``art``). On an LED pixel-matrix display (a Divoom Pixoo 64,
say) the plugin draws full-colour pixel art instead and publishes it as the
``canvas`` variable -- a pixel-canvas content object a page canvas can show --
with ``art`` derived from the same picture. ``canvas`` is a valid content
object on every board (built from the tiles on split-flap boards).

The board is never configured: the platform binds ``self.board`` for the
render and every dimension is derived from it, so one configuration serves
every board the user owns —

- Flagship (22 × 6)
- Note (15 × 3)
- Note array / FiestaPanel, anything from 15 × 3 to 120 × 24

How it reaches a model:

- **FiestaBoard's AI providers** (the default, FiestaBoard 9.11.0+): the
  provider picked in ``ai_provider`` (blank = FiestaBot's default) through
  ``self.ai_complete``, so every provider and sign-in set up in Settings →
  AI Providers works and the plugin carries no AI setup of its own.
- **A separate API key** (``api_key`` + ``api_base_url`` + ``model``): the
  original OpenAI-compatible path, unchanged, and always used when a key is
  saved. It also keeps working on cores that predate ``ai_complete``.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from src.plugins.base import PluginBase, PluginResult

try:
    from .pixel_canvas import (
        FALLBACK_DESCRIPTION,
        FALLBACK_THEME,
        PixelTarget,
        canvas_key,
        fallback_content,
        generate_pixel_piece,
        pixel_target,
        sample_grid,
        tiles_content,
    )
    from .source import REQUEST_TIMEOUT, ArtGenerator, ArtRequestError, Canvas, medium_for
except ImportError:
    # Fallback for when __init__.py is imported as a top-level module
    # (e.g., during pytest package setup before the package context is known).
    from pixel_canvas import (  # type: ignore[no-redef]
        FALLBACK_DESCRIPTION,
        FALLBACK_THEME,
        PixelTarget,
        canvas_key,
        fallback_content,
        generate_pixel_piece,
        pixel_target,
        sample_grid,
        tiles_content,
    )
    from source import REQUEST_TIMEOUT, ArtGenerator, ArtRequestError, Canvas, medium_for  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

#: How many per-geometry fallback pieces to keep. A user can own several
#: boards; keeping one piece each means an outage on a Note never puts a
#: 120-column frame on it. Bounded so a long-lived process cannot grow a
#: cache entry per geometry it has ever seen.
MAX_REMEMBERED_PIECES = 16

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"

NEEDS_NEWER_CORE = (
    "Update FiestaBoard to use its AI providers, or paste an API key in this plugin's settings."
)
NOT_CONFIGURED = (
    "Set up AI in Settings → AI Providers, or paste an API key in this plugin's settings. ({})"
)
REJECTED = "Reconnect the AI provider in Settings → AI Providers. ({})"


def _ai_error_classes() -> Tuple[type, ...]:
    """``(AINotConfiguredError, AIRejectedError)``, or empty on a core before 9.11.0."""
    try:
        from src.plugins.base import AINotConfiguredError, AIRejectedError
    except ImportError:
        return ()
    return AINotConfiguredError, AIRejectedError


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
        self._generator_key: Optional[Tuple[str, ...]] = None
        # What the last provider call left behind: the error it raised and
        # the model that answered.
        self._ai_failure: Optional[BaseException] = None
        self._ai_model_used: str = ""
        # Fallback pieces keyed by canvas geometry. Keying matters: without
        # it an outage on one board serves another board's frame, which is
        # either an overflow or a mostly-blank panel.
        self._last_pieces: Dict[str, Dict[str, Any]] = {}
        # Full-colour pieces for pixel displays, keyed by board/display key
        # plus the drawing size (see pixel_canvas.canvas_key).
        self._last_canvases: Dict[str, Dict[str, Any]] = {}

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

        # No API key is not an error: FiestaBoard's AI providers are used
        # instead, and fetch_data reports when none is set up.

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
        self._last_canvases.clear()
        logger.debug("GenerativeAiArtPlugin config updated, generator reset")

    def fetch_data(self) -> PluginResult:
        """Generate a new art piece sized to the bound board.

        On LLM failure returns the last known good piece *for this geometry*
        if there is one, or marks the plugin as unavailable.
        """
        cfg = self.config
        canvas = self.canvas()

        api_key = cfg.get("api_key", "")
        if api_key:
            # The original path, byte for byte: a saved key always wins.
            model = cfg.get("model", DEFAULT_MODEL)
            generator = self._get_generator(
                ("key", cfg.get("api_base_url", DEFAULT_BASE_URL), api_key, model)
            )
        else:
            if not callable(getattr(self, "ai_complete", None)):  # core before 9.11.0
                return PluginResult(available=False, error=NEEDS_NEWER_CORE)
            generator = self._get_generator(("ai",))

        self._ai_failure = None
        model_used = cfg.get("model", "unknown") if api_key else None

        target = pixel_target(self.board)
        if target is not None:
            return self._fetch_pixel_art(generator, canvas, target, model_used)

        piece = generator.generate(canvas)

        if piece is None and self._ai_failure is not None:
            refused = self._refusal()
            if refused is not None:
                return refused
            failure = self._ai_failure
            logger.warning("AI provider call failed for %s: %s", canvas.key, failure)
            return self._fallback_result(canvas, f"Art generation failed: {failure}")

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
            "canvas": tiles_content(piece["grid"]),
            "model": self._ai_model_used if model_used is None else model_used,
            "generated_at": _now(),
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
        medium = medium_for(getattr(board, "display", None))
        return Canvas(rows=board.rows, cols=board.cols, show_title=show_title, medium=medium)

    # ------------------------------------------------------------------
    # Pixel displays
    # ------------------------------------------------------------------

    def _fetch_pixel_art(
        self, generator: ArtGenerator, canvas: Canvas, target: PixelTarget, model_used: Optional[str]
    ) -> PluginResult:
        """One full-colour piece for a pixel display: ``canvas`` plus derived ``art``.

        On failure: the last good piece for this display and size, else a
        fixed fallback scene -- never an empty or invalid canvas. A provider
        that is not set up (or refused) is reported exactly as for tile art.
        """
        key = canvas_key(self.board, canvas, target)
        piece = generate_pixel_piece(generator, canvas, target)

        if piece is None:
            refused = self._refusal()
            if refused is not None:
                return refused
            if self._ai_failure is not None:
                error = f"Art generation failed: {self._ai_failure}"
            else:
                error = "Art generation failed; showing a fallback piece"
            logger.warning("Pixel art generation failed for %s: %s", key, error)
            return self._pixel_fallback(generator, canvas, target, key, error)

        record: Dict[str, Any] = {
            "theme": piece["theme"],
            "description": piece["description"],
            "art": piece["art"],
            "lines": piece["lines"],
            "canvas": piece["canvas"],
            "model": self._ai_model_used if model_used is None else model_used,
            "generated_at": _now(),
        }
        self._last_canvases[key] = record
        while len(self._last_canvases) > MAX_REMEMBERED_PIECES:
            self._last_canvases.pop(next(iter(self._last_canvases)))
        return PluginResult(available=True, data=self._record_to_data(record), formatted_lines=list(record["lines"]))

    def _pixel_fallback(
        self, generator: ArtGenerator, canvas: Canvas, target: PixelTarget, key: str, error: str
    ) -> PluginResult:
        record = self._last_canvases.get(key)
        if record is None:
            content = fallback_content(target)
            grid = sample_grid(content, canvas.art_rows, canvas.cols)
            lines = generator.render_lines(grid, FALLBACK_THEME, canvas)
            record = {
                "theme": FALLBACK_THEME,
                "description": FALLBACK_DESCRIPTION,
                "art": "\n".join(lines),
                "lines": lines,
                "canvas": content,
                "model": "",
                "generated_at": "",
            }
        else:
            logger.info("Returning last-known pixel art for %s as fallback", key)
        data = self._record_to_data(record)
        data["_fallback"] = True
        return PluginResult(available=True, data=data, error=error, formatted_lines=list(record["lines"]))

    def _refusal(self) -> Optional[PluginResult]:
        """The not-set-up / rejected result for the last provider failure, if it was one."""
        failure = self._ai_failure
        not_configured = _ai_error_classes()
        if failure is None or not not_configured:
            return None
        if isinstance(failure, not_configured[0]):
            return PluginResult(available=False, error=NOT_CONFIGURED.format(failure))
        if isinstance(failure, not_configured[1]):
            return PluginResult(available=False, error=REJECTED.format(failure))
        return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ai_reply(self, messages: List[Dict[str, str]], *, temperature: float, max_tokens: int) -> str:
        """Ask FiestaBoard's AI providers; the bridge handed to ArtGenerator.

        ``self.ai_complete`` is looked up on every call, never captured.  Any
        failure is kept for ``fetch_data`` to explain and re-raised as an
        :class:`ArtRequestError`, which the generator does not retry.
        """
        cfg = self.config
        if self._provider_protocol(cfg.get("ai_provider") or None) == "anthropic":
            # Anthropic refuses anything above 1.0, and this plugin defaults
            # to 1.2. Newer cores cap it themselves; 9.11 to 9.13 pass it on.
            temperature = min(temperature, 1.0)
        try:
            result = self.ai_complete(
                messages,
                provider_id=cfg.get("ai_provider") or None,
                model=cfg.get("ai_model") or None,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=float(REQUEST_TIMEOUT),
            )
        except Exception as exc:  # noqa: BLE001 — fetch_data must never raise
            self._ai_failure = exc
            raise ArtRequestError(str(exc)) from exc
        self._ai_model_used = str(getattr(result, "model", "") or "")
        return str(getattr(result, "text", result))

    def _provider_protocol(self, provider_id: Optional[str]) -> Optional[str]:
        """The protocol of *provider_id* (``None`` = FiestaBot's default), if known."""
        try:
            listed = self.ai_providers() or []
        except Exception:  # noqa: BLE001 — unreadable settings: let ai_complete say why
            return None
        for provider in listed:
            chosen = provider.get("id") == provider_id if provider_id else provider.get("default")
            if chosen:
                return provider.get("protocol")
        return None

    def _get_generator(self, key: Tuple[str, ...]) -> ArtGenerator:
        """Return the ArtGenerator for *key*, rebuilt when it changes.

        *key* is ``("key", base_url, api_key, model)`` for a saved API key, or
        ``("ai",)`` for FiestaBoard's AI providers.
        """
        if self._generator is not None and self._generator_key == key:
            return self._generator

        cfg = self.config
        themes = cfg.get("themes") or []
        if key[0] == "key":
            _, base_url, api_key, model = key
            connection: Dict[str, Any] = {"base_url": base_url, "api_key": api_key, "model": model}
        else:
            connection = {"base_url": "", "api_key": "", "model": "", "complete": self._ai_reply}
        self._generator_key = key
        self._generator = ArtGenerator(
            temperature=float(cfg.get("temperature", 1.2)),
            themes=themes if isinstance(themes, list) else [],
            extra_instructions=cfg.get("extra_instructions", ""),
            custom_system_prompt=cfg.get("custom_system_prompt", ""),
            show_title=bool(cfg.get("show_title", False)),
            **connection,
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
            "canvas": record.get("canvas"),
        }


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


# Export the plugin class
Plugin = GenerativeAiArtPlugin
