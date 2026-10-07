# Generative AI Art Plugin

Full-screen abstract art composed by your FiestaBoard AI provider for whatever board it lands on, from a Note to an LED pixel display.

![Generative AI Art Display](./docs/board-display.png)

**→ [Setup Guide](./docs/SETUP.md)**

## Overview

Each refresh the plugin asks the model for a new piece, following a rotating set of 63 artistic themes (concentric rings, mountain silhouettes, aurora borealis, Mondrian-style blocks, and many more). On a split-flap board the piece is a colour-tile composition in the board's 8-colour palette. On an LED pixel-matrix display such as a Divoom Pixoo 64 it is full-colour pixel art, drawn into a page **canvas**.

It uses your FiestaBoard AI provider: whatever you set up in **Settings → AI Providers** (an OpenAI-compatible, Anthropic or OpenAI Responses endpoint, a pasted key, or a sign-in with OpenRouter, Hugging Face or ChatGPT). A separate API key for any OpenAI-compatible endpoint is optional.

### How it adapts to the board

There is no board-size setting. The platform tells the plugin which board it is rendering, and everything is derived from that, so one configuration serves every board you own.

| Board | What the model is asked for |
|---|---|
| Note (15 × 3), Flagship (22 × 6), 15 × 12 array — up to 240 tiles | **per-cell**: one colour letter per tile |
| Anything larger, up to a 128 × 96 panel | **scene**: shapes in normalised `0..1` coordinates, rasterised to the tiles by the plugin |
| An LED pixel-matrix display (RGB) | **pixel scene**: shapes in pixel coordinates with any `#rrggbb` colour, returned as a canvas |

The tile scene format is a background plus an ordered list of shapes (gradients, stripes, rectangles, ellipses, rings, triangles, lines, waves, noise) with an optional mirror for symmetry. Because it lives in a unit square, the request costs the same for a 15 × 3 Note as for a 120 × 24 panel, and any aspect ratio is native.

On a pixel display the model draws at the display's own pixel size (64 × 64 when the board does not report one) with up to 60 shapes: rectangles, circles, ellipses, lines, polygons, gradients and an occasional word of text, plus stripes, rings, waves and noise as shortcuts the plugin expands. The plugin clamps the result to FiestaBoard's canvas limits (at most 256 shapes and 128 × 128) and drops anything it cannot repair, so the canvas is always valid. One model call feeds both variables: the tile `art` on a pixel display is sampled from the same picture.

A response that is slightly off-spec (a row short, a few unreadable cells, a reply truncated mid-array) is repaired rather than thrown away.

## Template Variables

### Display

| Variable | Description |
|---|---|
| `generative_ai_art.art` | Full-board colour pattern, sized to the board being rendered (one tile per cell, rows separated by newlines) |
| `generative_ai_art.canvas` | Full-colour pixel art as a canvas content object (`"format": "canvas"`). Use it as a page canvas's `source`. On a split-flap board it is the tile art, one pixel per tile |

### Metadata

| Variable | Description |
|---|---|
| `generative_ai_art.theme` | Theme of the current piece |
| `generative_ai_art.description` | One-sentence artist's description |
| `generative_ai_art.model` | Model that generated the piece |
| `generative_ai_art.generated_at` | ISO timestamp |

The `canvas` value looks like this (trimmed):

```json
{
  "size": [64, 64],
  "background": "#140c2a",
  "shapes": [
    {"type": "gradient", "x": 0, "y": 0, "w": 64, "h": 40, "from": "#2b1055", "to": "#ff7e5f", "angle": 90},
    {"type": "circle", "cx": 44, "cy": 30, "r": 8, "fill": "#ffd36e"},
    {"type": "polygon", "points": [[0, 64], [0, 44], [18, 34], [34, 46], [50, 38], [64, 44], [64, 64]], "fill": "#3a1f2b"}
  ],
  "palette": {"a": "#ffffff"},
  "pixels": ["..........a.....", "...a........"]
}
```

## Example Templates

### Split-flap boards

The whole board is the art: one variable on the first line, with wrapping on. It fills row 1 and wraps down as far as the board goes.

```text
{{generative_ai_art.art}}
```

### LED pixel displays: full-bleed canvas

On a pixel display, add a canvas that covers the whole page and takes its drawing from the plugin. The area names the largest grid FiestaBoard supports; it is clamped to your board. `"bleed": ["all"]` takes the drawing to the panel's edge, and `"text": "hide"` blanks the text underneath.

```json
{
  "template": ["{{generative_ai_art.art}}"],
  "canvases": [
    {
      "id": "art",
      "area": {"row": 1, "col": 1, "rows": 96, "cols": 128},
      "bleed": ["all"],
      "text": "hide",
      "source": "{{generative_ai_art.canvas}}"
    }
  ]
}
```

The tile art stays in the template, so the same page still shows art on a board that does not draw canvases. The plugin's demo pages are built this way.

### LED pixel displays: art with a title line

Leave the first row for text and let the canvas fill the rest. With `"text": "flow"`, text keeps to the cells the canvas leaves free.

```json
{
  "template": ["{{generative_ai_art.theme}}"],
  "line_metadata": [{"alignment": "center", "wrap": false}],
  "canvases": [
    {
      "id": "art",
      "area": {"row": 2, "col": 1, "rows": 95, "cols": 128},
      "bleed": ["left", "right", "bottom"],
      "text": "flow",
      "source": "{{generative_ai_art.canvas}}"
    }
  ]
}
```

Canvases need a FiestaBoard with pixel-canvas support: the canvas engine (FiestaBoard PR #2228) and canvases on pages (its follow-up, branch `feat/canvas-pages`). On an older FiestaBoard the plugin works as before, and `canvas` is simply not drawn.

## Configuration

Set up a provider once in **Settings → AI Providers**, then enable the plugin. That is all it needs.

| Setting | Type | Default | Description |
|---|---|---|---|
| `ai_provider` | string | `""` | Which FiestaBoard AI provider composes the art. Empty means FiestaBot's default provider. |
| `ai_model` | string | `""` | Model for that provider. Empty means the provider's default model. |
| `api_key` | string | — | **Use a separate API key (optional).** When set it always wins, and the plugin calls `api_base_url` with `model` directly instead of your AI providers. Use any value (e.g. `"ollama"`) for local endpoints that don't need auth. |
| `api_base_url` | string | `https://api.openai.com/v1` | With a separate API key: base URL for the chat completions endpoint. |
| `model` | string | `gpt-4o-mini` | With a separate API key: model name. |
| `temperature` | number 0–2 | `1.2` | Sampling temperature. 1.0–1.4 works well for art; Anthropic providers get at most 1.0. |
| `refresh_seconds` | integer ≥300 | `1800` | How often to generate a new piece (minimum 5 minutes). |
| `themes` | string[] | `[]` | Custom theme list. Leave empty to use the 63 built-in themes. |
| `extra_instructions` | string | `""` | Extra instructions appended to the built-in prompt, on every board (e.g. `"favour cool colours"`). Ignored when `custom_system_prompt` is set. |
| `custom_system_prompt` | string | `""` | Replaces the entire system prompt, on every board including pixel displays. A reply in either the tile format (`grid` or `shapes`) or the pixel-scene format is accepted. Call `generator.build_default_system_prompt(canvas)` to see the default tile prompt, or `pixel_canvas.canvas_system_prompt(target)` for the pixel one. |
| `show_title` | boolean | `false` | Spend the bottom board row on a short centred title. Applies to `art`; `canvas` never includes a title. |

If the board shows "Set up AI in Settings → AI Providers…", AI is turned off or no provider is set up. "Reconnect the AI provider…" means the provider refused its key or sign-in. On a FiestaBoard older than 9.11.0 the plugin asks you to "Update FiestaBoard to use its AI providers, or paste an API key"; a separate API key keeps working there.

Configs saved before 1.5.0 keep their `api_key`, `api_base_url` and `model`, and keep using them.

### Environment-variable overrides

These set the separate API key path.

| Variable | Default |
|---|---|
| `GENERATIVE_AI_ART_API_KEY` | — |
| `GENERATIVE_AI_ART_API_BASE_URL` | `https://api.openai.com/v1` |
| `GENERATIVE_AI_ART_MODEL` | `gpt-4o-mini` |
| `GENERATIVE_AI_ART_TEMPERATURE` | `1.2` |
| `GENERATIVE_AI_ART_REFRESH_SECONDS` | `1800` |

## Features

- Uses your FiestaBoard AI provider (FiestaBoard 9.11.0+): FiestaBot's default, or the one you pick
- Optional separate API key for any OpenAI v1-compatible endpoint (OpenAI, OpenRouter, Ollama, LM Studio, etc.)
- Renders on every board FiestaBoard supports, with nothing to configure:
  - **Flagship** — 22 × 6
  - **Note** — 15 × 3
  - **Note array** — anything from 15 × 3 to 120 × 24
  - **FiestaPanel** — any grid from 15 × 3 up to 128 × 96
  - **LED pixel displays** — full-colour pixel art at the display's pixel size, through the `canvas` variable
- Prompts name the real display: split-flap, LED matrix or screen
- Graceful fallback: if the model call fails, the board keeps showing the last successful piece **for that board** — a Note is never handed a Flagship's frame, and a pixel display is never handed another display's canvas. A pixel display with no earlier piece shows a fixed twilight landscape rather than nothing
- Uses only the board's 8 colours on split-flap boards — at least 2 and at most 6 per piece — so compositions stay readable on physical hardware
- Configurable refresh interval, temperature and theme list

### Development

```bash
git clone https://github.com/Fiestaboard/fiestaboard-plugin--generative-ai-art
cd fiestaboard-plugin--generative-ai-art

# The repo ships the plugins/generative_ai_art import shim; add the top-level one
ln -s . generative_ai_art

# Run tests (adjust the FiestaBoard path as needed)
PYTHONPATH="$PWD:/path/to/FiestaBoard" pytest tests/ -v
```

Where the FiestaBoard checkout has the canvas engine (`src/canvas`), the tests also validate and rasterise every canvas with core's own code.

## Author

FiestaBoard Team. MIT licensed — see [LICENSE](LICENSE).
