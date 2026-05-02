# fiestaboard-plugin--generative-ai-art

A [FiestaBoard](https://github.com/Fiestaboard/FiestaBoard) plugin that generates full-screen abstract art for your split-flap display using any OpenAI-compatible LLM.

Each refresh the plugin asks the model to compose a unique colour-tile composition using the board's 8-colour palette, following a rotating set of artistic themes (concentric rings, mountain silhouettes, aurora borealis, Mondrian-style blocks, and 26 others).

## Features

- Works with any OpenAI v1-compatible endpoint — OpenAI, OpenRouter, Ollama, LM Studio, etc.
- Supports both display sizes:
  - **Flagship** — 6 rows × 22 columns
  - **Note** — 3 rows × 15 columns
- Ring-buffer history of the last N generated pieces (default 100)
- Optional persistence to disk so the history survives restarts
- Graceful fallback: if the LLM call fails, the board keeps showing the last successful piece
- Configurable refresh interval, temperature, and custom theme list

## Configuration

| Setting | Type | Default | Description |
|---|---|---|---|
| `api_key` *(required)* | string | — | API key. Use any value (e.g. `"ollama"`) for local endpoints that don't need auth. |
| `api_base_url` | string | `https://api.openai.com/v1` | Base URL for the chat completions endpoint. |
| `model` | string | `gpt-4o-mini` | Model name. |
| `device_type` | `flagship` \| `note` | `flagship` | Target display size. |
| `temperature` | number 0–2 | `1.2` | Sampling temperature. 1.0–1.4 works well for art. |
| `refresh_seconds` | integer ≥300 | `1800` | How often to generate a new piece (minimum 5 minutes). |
| `history_size` | integer 1–500 | `100` | Number of past pieces to remember. |
| `persist_history` | boolean | `true` | Save history to `data/plugins/generative_ai_art/history.json`. |
| `themes` | string[] | `[]` | Custom theme list. Leave empty to use the 30 built-in themes. |
| `extra_instructions` | string | `""` | Extra instructions appended to the system prompt (e.g. `"favour cool colours"`). |

### Environment-variable overrides

| Variable | Default |
|---|---|
| `GENERATIVE_AI_ART_API_KEY` | — |
| `GENERATIVE_AI_ART_API_BASE_URL` | `https://api.openai.com/v1` |
| `GENERATIVE_AI_ART_MODEL` | `gpt-4o-mini` |
| `GENERATIVE_AI_ART_DEVICE_TYPE` | `flagship` |
| `GENERATIVE_AI_ART_TEMPERATURE` | `1.2` |
| `GENERATIVE_AI_ART_REFRESH_SECONDS` | `1800` |
| `GENERATIVE_AI_ART_HISTORY_SIZE` | `100` |

## Template variables

| Variable | Description |
|---|---|
| `generative_ai_art.art` | Full-board colour pattern, ready for display |
| `generative_ai_art.theme` | Theme of the current piece |
| `generative_ai_art.description` | One-sentence artist's description |
| `generative_ai_art.model` | Model that generated the piece |
| `generative_ai_art.generated_at` | ISO timestamp |
| `generative_ai_art.history_count` | Number of pieces in history |

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
      GENERATIVE_AI_ART_DEVICE_TYPE: "flagship"
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
