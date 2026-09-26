# fiestaboard-plugin--generative-ai-art

A [FiestaBoard](https://github.com/Fiestaboard/FiestaBoard) plugin that generates full-screen abstract art for your split-flap display using any OpenAI-compatible LLM.

Each refresh the plugin asks the model to compose a unique colour-tile composition using the board's 8-colour palette, following a rotating set of 63 artistic themes (concentric rings, mountain silhouettes, aurora borealis, Mondrian-style blocks, and many more).

## Features

- Works with any OpenAI v1-compatible endpoint — OpenAI, OpenRouter, Ollama, LM Studio, etc.
- Renders on **every board FiestaBoard supports**, with nothing to configure:
  - **Flagship** — 22 × 6
  - **Note** — 15 × 3
  - **Note array / FiestaPanel** — anything from 15 × 3 to 120 × 24
- Graceful fallback: if the LLM call fails, the board keeps showing the last successful piece
  **for that board** — a Note is never handed a Flagship's frame
- Configurable refresh interval, temperature, and custom theme list

## How it adapts to the board

There is no board-size setting. The platform tells the plugin which board it
is rendering, and every dimension is derived from that, so one configuration
serves every board you own. Two emission strategies are picked from the cell
count alone:

| Board | Cells | Strategy |
|---|---|---|
| Note (15 × 3), Flagship (22 × 6), 15 × 12 array | ≤ 240 | **per-cell** — the model names one colour per tile |
| Anything larger, up to a 120 × 24 panel | > 240 | **scene** — the model describes the composition and the plugin rasterises it |

The scene format is a background plus an ordered list of shapes (gradients,
stripes, rectangles, ellipses, rings, triangles, lines, waves, noise) in
normalised `0..1` coordinates, with an optional mirror for symmetry. Because
the scene lives in a unit square that is mapped onto the board, the request
costs the same number of tokens for a 15 × 3 Note as for a 120 × 24 panel
(2,880 tiles) and **any aspect ratio is native**: a 15 × 12 array and a
120 × 3 array are both note arrays and nothing else alike, and both get a
composition made for their shape rather than a cropped Flagship.

Per-cell emission is kept for small boards because it gives the model finer
control where the token cost is affordable. Whichever format the model
actually returns is accepted on any board, and a response that is slightly
off-spec (a row short, a few unreadable cells, a reply truncated mid-array)
is repaired onto the board rather than thrown away.

## Configuration

| Setting | Type | Default | Description |
|---|---|---|---|
| `api_key` *(required)* | string | — | API key. Use any value (e.g. `"ollama"`) for local endpoints that don't need auth. |
| `api_base_url` | string | `https://api.openai.com/v1` | Base URL for the chat completions endpoint. |
| `model` | string | `gpt-4o-mini` | Model name. |
| `temperature` | number 0–2 | `1.2` | Sampling temperature. 1.0–1.4 works well for art. |
| `refresh_seconds` | integer ≥300 | `1800` | How often to generate a new piece (minimum 5 minutes). |
| `themes` | string[] | `[]` | Custom theme list. Leave empty to use the 63 built-in themes. |
| `extra_instructions` | string | `""` | Extra instructions appended to the built-in system prompt (e.g. `"favour cool colours"`). Ignored when `custom_system_prompt` is set. |
| `custom_system_prompt` | string | `""` | Override the entire system prompt sent to the LLM. Leave blank to use the built-in prompt. Call `generator.build_default_system_prompt(canvas)` to inspect the default for a given board. A custom prompt is used unchanged on every board, so write it board-agnostically. |

### Environment-variable overrides

| Variable | Default |
|---|---|
| `GENERATIVE_AI_ART_API_KEY` | — |
| `GENERATIVE_AI_ART_API_BASE_URL` | `https://api.openai.com/v1` |
| `GENERATIVE_AI_ART_MODEL` | `gpt-4o-mini` |
| `GENERATIVE_AI_ART_TEMPERATURE` | `1.2` |
| `GENERATIVE_AI_ART_REFRESH_SECONDS` | `1800` |

## Template variables

| Variable | Description |
|---|---|
| `generative_ai_art.art` | Full-board colour pattern, sized to the board being rendered |
| `generative_ai_art.theme` | Theme of the current piece |
| `generative_ai_art.description` | One-sentence artist's description |
| `generative_ai_art.model` | Model that generated the piece |
| `generative_ai_art.generated_at` | ISO timestamp |

## Installation

### Quick start with Ollama

```yaml
# docker-compose.override.yml
services:
  fiestaboard:
    environment:
      GENERATIVE_AI_ART_API_KEY: "ollama"
      GENERATIVE_AI_ART_API_BASE_URL: "http://ollama:11434/v1"
      GENERATIVE_AI_ART_MODEL: "llama3.2"
      GENERATIVE_AI_ART_REFRESH_SECONDS: "900"
```

### OpenAI

```yaml
services:
  fiestaboard:
    environment:
      GENERATIVE_AI_ART_API_KEY: "sk-..."
      GENERATIVE_AI_ART_MODEL: "gpt-4o-mini"
```

### OpenRouter

```yaml
services:
  fiestaboard:
    environment:
      GENERATIVE_AI_ART_API_KEY: "sk-or-..."
      GENERATIVE_AI_ART_API_BASE_URL: "https://openrouter.ai/api/v1"
      GENERATIVE_AI_ART_MODEL: "anthropic/claude-3-haiku"
```

## Development

```bash
git clone https://github.com/Fiestaboard/fiestaboard-plugin--generative-ai-art
cd fiestaboard-plugin--generative-ai-art

# Create the test import shim (one-time setup)
mkdir -p plugins && ln -sf .. plugins/generative_ai_art

# Run tests (adjust FiestaBoard path as needed)
PYTHONPATH="/path/to/plugin:/path/to/FiestaBoard" pytest tests/ -v
```

## Colour palette

The board has exactly 8 colours. The plugin uses single-letter codes internally:

| Code | Colour |
|---|---|
| `R` | Red |
| `O` | Orange |
| `Y` | Yellow |
| `G` | Green |
| `B` | Blue |
| `V` | Violet |
| `W` | White |
| `K` | Black |

Each piece must use at least 2 distinct colours and no more than 6, keeping compositions readable on physical hardware.

## License

MIT — see [LICENSE](LICENSE).
