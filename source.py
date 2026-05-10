"""Art generation logic for the Generative AI Art FiestaBoard plugin.

Handles communication with an OpenAI-compatible chat completions endpoint,
builds art prompts, validates LLM responses, and converts grid data into
FiestaBoard color-marker strings.
"""

import json
import logging
import random
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# Request timeout in seconds
REQUEST_TIMEOUT = 60

# Color alphabet: single letter → FiestaBoard color marker
# K = blacK (not B, to avoid B=blue ambiguity)
COLOR_LETTERS: Dict[str, str] = {
    "R": "{red}",
    "O": "{orange}",
    "Y": "{yellow}",
    "G": "{green}",
    "B": "{blue}",
    "V": "{violet}",
    "W": "{white}",
    "K": "{black}",
}

VALID_COLORS = set(COLOR_LETTERS.keys())

# Built-in theme pool — varied enough to produce very different compositions
BUILTIN_THEMES = [
    # --- Geometric / structural ---
    "concentric rings expanding from the center",
    "diagonal color gradient from corner to corner",
    "bold geometric chevrons or zigzags",
    "symmetrical mandala-like radial pattern",
    "horizontal banded color field with a focal accent",
    "vertical columns of alternating color families",
    "color-block abstraction inspired by Mondrian",
    "checkerboard variant with irregular cell sizes",
    "two large opposing diagonal color fields",
    "staircase terracing pattern from bottom-left",
    "diamond lattice with alternating fill colors",
    "isometric cube illusion using three tones",
    "op-art concentric squares with high contrast",
    "bold offset stripes at 45 degrees",
    "herringbone weave pattern",
    "interlocking brick or basketweave pattern",
    "radial spoke pattern emanating from one corner",
    "plaid or tartan: horizontal and vertical bands crossing",
    "Bauhaus-style composition: rectangles, circles, lines",
    "triangle grid — large and small contrasting triangles",
    "pixel grid of 2×2 color blocks in a structured palette",
    # --- Nature / landscape ---
    "abstract mountain silhouette with sky gradient",
    "aurora borealis: vertical curtains of shifting color",
    "sunset horizon with layered warm and cool bands",
    "desert dunes: warm earth tones with cool shadow bands",
    "cityscape silhouette: dark blocks on gradient sky",
    "pixel landscape: ground, horizon, sky layers",
    "underwater: light shafts descending through blue-green",
    "storm cell: dark vortex with electric accent colors",
    "volcanic landscape: black ground, orange glow, dark sky",
    "coral reef cross-section: layered warm and cool patches",
    "autumn forest floor: scattered warm patches on dark ground",
    "tidal pool reflection: mirrored bands with shimmer accent",
    "twilight sky fading from orange at the horizon to deep violet",
    "cloud formations: soft rounded masses on a gradient sky",
    "glacier: cool blues and whites with deep crevasse shadows",
    "night sky with a bright horizon glow",
    # --- Texture / pattern ---
    "scattered irregular patches like a mosaic",
    "wave interference pattern — overlapping sine-like curves",
    "stained glass: irregular polygons in contrasting colors",
    "circuit board: right-angle traces on a dark ground",
    "random walk path of a bright color on a neutral field",
    "heatmap-style gradient: cool edges, warm center",
    "sparse pointillist dots on a contrasting background",
    "blurred organic blob cluster in the center",
    "lava-lamp: rounded blobs of warm color on cool ground",
    "topographic contour lines on a single-hue gradient",
    "woven textile: tight alternating thread colors",
    "noise field: irregular dithered color patches",
    "kaleidoscope mirror reflection",
    "seed-of-life sacred geometry circles",
    # --- Abstract / painterly ---
    "flowing river of a single bright color on a dark ground",
    "bold typographic-inspired abstract shapes",
    "horizontal banded color field — Mark Rothko-style",
    "abstract expressionist color field: large loose brushstroke blocks",
    "color spectrum arc from cool to warm",
    "neon sign glow: bright lines and halos on dark background",
    "patchwork quilt: irregular rectangles in a warm palette",
    "butterfly wing symmetry: mirrored left-right composition",
    "color gradient snake path winding across the board",
    "bold racing stripes: two or three wide parallel bands",
    "Japanese wave pattern: repeating curved arcs in blue",
    "retro video-game sprite: bright pixel art on dark background",
]


class ArtValidationError(Exception):
    """Raised when the LLM response cannot be parsed into a valid art grid."""


class ArtGenerator:
    """Generates color-tile art grids using an OpenAI-compatible LLM.

    Args:
        base_url: Base URL for the chat completions endpoint
                  (e.g. ``https://api.openai.com/v1``).
        api_key: Bearer token / API key for authentication.
        model: Model name (e.g. ``gpt-4o-mini``).
        temperature: Sampling temperature (0–2).
        device_type: ``"flagship"`` (6×22) or ``"note"`` (3×15).
        themes: Optional list of custom theme strings. Falls back to
                ``BUILTIN_THEMES`` when empty or ``None``.
        extra_instructions: Additional text appended to the system prompt.
    """

    DEVICE_DIMENSIONS = {
        "flagship": (6, 22),
        "note": (3, 15),
    }

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float,
        device_type: str = "flagship",
        themes: Optional[List[str]] = None,
        extra_instructions: str = "",
        custom_system_prompt: str = "",
        show_title: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.device_type = device_type if device_type in self.DEVICE_DIMENSIONS else "flagship"
        self.rows, self.cols = self.DEVICE_DIMENSIONS[self.device_type]
        self.themes = themes if themes else BUILTIN_THEMES
        self.extra_instructions = extra_instructions.strip()
        self.custom_system_prompt = custom_system_prompt.strip()
        self.show_title = show_title

    @property
    def _art_rows(self) -> int:
        """Rows used for the color art grid (excludes title row when enabled)."""
        return self.rows - 1 if self.show_title else self.rows

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> Optional[Dict[str, Any]]:
        """Generate a single art piece.

        Attempts up to two calls; the second uses a different theme if the
        first response fails validation.

        Returns:
            Dict with keys ``theme``, ``description``, ``grid`` (list of
            list of single-letter color codes), and ``art`` (FiestaBoard
            color-marker string), or ``None`` if both attempts fail.
        """
        theme = self._pick_theme()

        for attempt in range(2):
            try:
                raw = self._call_api(theme)
                parsed = self._validate_and_parse(raw)
                return {
                    "theme": parsed["theme"],
                    "description": parsed["description"],
                    "grid": parsed["grid"],
                    "art": self._grid_to_art_string(parsed["grid"], parsed.get("title", "")),
                }
            except ArtValidationError as exc:
                logger.warning(
                    "Art validation failed (attempt %d/2) for theme '%s': %s",
                    attempt + 1,
                    theme,
                    exc,
                )
                # Pick a different theme for retry
                theme = self._pick_theme(exclude=theme)
            except requests.RequestException as exc:
                logger.error("API request failed: %s", exc)
                return None

        logger.error("Art generation failed after 2 attempts")
        return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _pick_theme(self, exclude: Optional[str] = None) -> str:
        """Return a random theme from the pool, optionally excluding one."""
        choices = [t for t in self.themes if t != exclude]
        if not choices:
            choices = self.themes
        return random.choice(choices)

    def build_default_system_prompt(self) -> str:
        """Return the auto-generated system prompt for the current configuration.

        Exposed publicly so callers can show the user what will be sent.
        """
        color_table = "\n".join(
            f"  {letter} = {marker.strip('{}')}"
            for letter, marker in COLOR_LETTERS.items()
        )

        art_rows = self._art_rows

        if self.show_title:
            task_line = (
                f"Your task is to compose an abstract art piece that fills the top {art_rows} rows "
                f"of a {self.rows}-row × {self.cols}-column grid. "
                f"The bottom row is reserved for a short title you will provide."
            )
            title_rule = f"\n7. Include a \"title\" field: a short label for the piece, maximum {self.cols} characters."
            title_json_field = f',\n  "title": "<short label, max {self.cols} characters>"'
        else:
            task_line = (
                f"Your task is to compose a full-screen abstract art piece that fills every cell "
                f"of the {self.rows}-row × {self.cols}-column grid."
            )
            title_rule = ""
            title_json_field = ""

        prompt = f"""You are a generative-art composer for a physical split-flap display.
The display uses ONLY the following 8 colors, identified by single capital letters:

{color_table}

{task_line}

RULES (non-negotiable):
1. Output ONLY valid JSON — no markdown fences, no prose before or after.
2. The "grid" array must contain EXACTLY {art_rows} rows.
3. Every row must contain EXACTLY {self.cols} elements.
4. Every element must be one of: R O Y G B V W K
5. Use AT LEAST 2 distinct colors and AT MOST 6 distinct colors per piece.
6. The piece must have intentional visual structure — a focal point, balance, \
rhythm, or clear compositional logic. It must look like deliberate art, \
not random noise.{title_rule}

AESTHETIC GUIDANCE:
- Choose 2–4 dominant colors and 0–2 accent colors.
- Avoid checkerboard patterns unless the theme calls for it.
- Think in regions, gradients, curves, or geometric forms rather than \
per-cell random choices.
- The piece should evoke the theme visually, not just use its name as a label.

OUTPUT FORMAT (strict JSON, no other text):
{{
  "theme": "<the theme in a few words>",
  "description": "<one sentence describing the visual impression>",
  "grid": [
    ["{self.cols} single-letter codes for row 1"],
    ...
    ["{self.cols} single-letter codes for row {art_rows}"]
  ]{title_json_field}
}}"""

        if self.extra_instructions:
            prompt += "\n" + self.extra_instructions

        return prompt

    def _build_messages(self, theme: str) -> List[Dict[str, str]]:
        """Build the messages list for the chat completions request."""
        system = self.custom_system_prompt if self.custom_system_prompt else self.build_default_system_prompt()

        user = (
            f"Compose an art piece with this theme: {theme}\n\n"
            "Remember: output ONLY the JSON object described above."
        )

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def _call_api(self, theme: str) -> str:
        """POST to the chat completions endpoint and return the content string."""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": self._build_messages(theme),
            "temperature": self.temperature,
        }

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()

        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ArtValidationError(
                f"Unexpected API response structure: {exc}"
            ) from exc

        return content

    def _validate_and_parse(self, raw: str) -> Dict[str, Any]:
        """Parse and validate the raw LLM output string.

        Returns the parsed dict if valid; raises ``ArtValidationError``
        otherwise.
        """
        # Strip any accidental markdown fences
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            # Remove opening fence (```json or ```)
            lines = lines[1:]
            # Remove closing fence
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ArtValidationError(f"Invalid JSON: {exc}") from exc

        if not isinstance(parsed, dict):
            raise ArtValidationError("Response is not a JSON object")

        # Validate required keys
        required_keys = ("theme", "description", "grid")
        for key in required_keys:
            if key not in parsed:
                raise ArtValidationError(f"Missing required key: '{key}'")

        if self.show_title:
            if "title" not in parsed:
                raise ArtValidationError("Missing required key: 'title'")
            title = parsed["title"]
            if not isinstance(title, str):
                raise ArtValidationError("'title' must be a string")
            if len(title) > self.cols:
                raise ArtValidationError(
                    f"Title is {len(title)} characters but board is only {self.cols} columns wide"
                )

        grid = parsed["grid"]
        if not isinstance(grid, list):
            raise ArtValidationError("'grid' must be a list")

        art_rows = self._art_rows
        if len(grid) != art_rows:
            raise ArtValidationError(
                f"Expected {art_rows} rows, got {len(grid)}"
            )

        seen_colors: set = set()
        for row_idx, row in enumerate(grid):
            if not isinstance(row, list):
                raise ArtValidationError(
                    f"Row {row_idx} is not a list (got {type(row).__name__})"
                )
            if len(row) != self.cols:
                raise ArtValidationError(
                    f"Row {row_idx}: expected {self.cols} columns, got {len(row)}"
                )
            for col_idx, cell in enumerate(row):
                if cell not in VALID_COLORS:
                    raise ArtValidationError(
                        f"Invalid color '{cell}' at [{row_idx}][{col_idx}]. "
                        f"Valid: {sorted(VALID_COLORS)}"
                    )
                seen_colors.add(cell)

        if len(seen_colors) < 2:
            raise ArtValidationError(
                f"Art uses only {len(seen_colors)} color(s); minimum is 2"
            )

        return parsed

    def _grid_to_art_string(self, grid: List[List[str]], title: str = "") -> str:
        """Convert a validated single-letter grid to a FiestaBoard marker string.

        Rows are joined with newlines. When show_title is enabled, a centered
        title row is appended as the last line.

        Returns:
            Multi-line string using ``{color}`` markers, with an optional
            plain-text title as the final row.
        """
        rows = []
        for row in grid:
            rows.append("".join(COLOR_LETTERS[cell] for cell in row))
        if self.show_title:
            rows.append(title.upper().center(self.cols))
        return "\n".join(rows)
