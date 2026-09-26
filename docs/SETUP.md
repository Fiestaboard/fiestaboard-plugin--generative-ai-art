# Generative AI Art Setup Guide

Each refresh, the plugin asks an LLM to compose a full-screen abstract art piece for your split-flap display using the board's 8-colour palette. Every piece is unique — rotating through 63 built-in themes like aurora borealis, Mondrian-style blocks, mountain silhouettes, and more. The piece is composed for whatever board it lands on: there is no display-size setting to get wrong.

![Generative AI Art — Sunset](./board-display.png)

| Aurora Borealis | Mountain Silhouette | Concentric Rings |
|---|---|---|
| ![Aurora](./black/generative-ai-art-aurora.png) | ![Mountain](./black/generative-ai-art-mountain.png) | ![Rings](./black/generative-ai-art-rings.png) |

## Overview

**What it does:**
- Generates unique full-screen colour art on every refresh
- Uses any OpenAI v1-compatible endpoint (OpenAI, Ollama, OpenRouter, LM Studio, etc.)
- Fills **any** board: Flagship (22 × 6), Note (15 × 3), or a note array /
  FiestaPanel anywhere from 15 × 3 up to 120 × 24 — nothing to configure
- Falls back gracefully — if the LLM is unavailable the last piece *for that
  board* stays on screen
- 63 built-in artistic themes, or supply your own

### Board sizes

The plugin reads the board it is rendering on and derives everything from it,
so the same configuration serves a Note, a Flagship and a wall-sized panel at
the same time.

For small boards (up to 240 tiles — a Note, a Flagship, a 15 × 12 array) the
model names one colour per tile. Above that, asking for 2,880 individual
tiles would be a ~12,000-token reply that most models truncate, so the model
instead returns a compact **scene description** — regions, gradients, shapes
in normalised `0..1` coordinates — which the plugin rasterises to the board.
That request costs the same whatever the board's size, and it composes for
the board's real aspect ratio, so a wide-short 120 × 3 array and a
tall-narrow 15 × 12 array each get art made for their shape.

**Use Cases:**
- Ambient art display that changes throughout the day
- Showcase your local LLM's creative capabilities
- Rotating colour piece for a gallery wall or office

## Prerequisites

- ✅ An OpenAI-compatible API endpoint
- ✅ An API key (use any value, e.g. `"ollama"`, for local endpoints)

## Quick Setup

### Option A: OpenAI

```yaml
# docker-compose.override.yml
services:
  fiestaboard:
    environment:
      GENERATIVE_AI_ART_API_KEY: "sk-..."
      GENERATIVE_AI_ART_MODEL: "gpt-4o-mini"
```

### Option B: Ollama (local, free)

```yaml
services:
  fiestaboard:
    environment:
      GENERATIVE_AI_ART_API_KEY: "ollama"
      GENERATIVE_AI_ART_API_BASE_URL: "http://ollama:11434/v1"
      GENERATIVE_AI_ART_MODEL: "llama3.2"
```

### Option C: OpenRouter

```yaml
services:
  fiestaboard:
    environment:
      GENERATIVE_AI_ART_API_KEY: "sk-or-..."
      GENERATIVE_AI_ART_API_BASE_URL: "https://openrouter.ai/api/v1"
      GENERATIVE_AI_ART_MODEL: "anthropic/claude-3-haiku"
```

## Configuration Reference

| Setting | Default | Description |
|---|---|---|
| `api_key` *(required)* | — | API key for your endpoint. |
| `api_base_url` | `https://api.openai.com/v1` | Base URL for the chat completions endpoint. |
| `model` | `gpt-4o-mini` | Model to use for art generation. |
| `temperature` | `1.2` | Sampling temperature (0–2). 1.0–1.4 gives varied, artistic results. |
| `refresh_seconds` | `1800` | How often to generate a new piece (minimum 300 s / 5 min). |
| `themes` | `[]` | Custom theme list. Empty = use the 63 built-in themes. |
| `extra_instructions` | `""` | Additional instructions appended to the prompt (e.g. `"favour cool colours"`). |

## Template Variables

| Variable | Description |
|---|---|
| `generative_ai_art.art` | Full-board colour pattern, sized to the board — use this in your page template |
| `generative_ai_art.theme` | Theme of the current piece |
| `generative_ai_art.description` | One-sentence artist's description |
| `generative_ai_art.model` | Model that generated the piece |
| `generative_ai_art.generated_at` | ISO timestamp of generation |

## Page Template

The whole board is the art, so the template is a single variable on the first
line with wrapping enabled — it fills row 1 and wraps down as far as the board
goes:

```
{{generative_ai_art.art}}
```

Turn on **Show Title** to spend the bottom row on a short centred label
instead; the art then fills every row above it, on any board.

## Tips

- **Temperature 1.0–1.4** gives the best balance of creativity and structure. Higher values produce more chaotic results.
- **Shorter refresh intervals** (e.g. 300 s) work well with fast local models; longer intervals (e.g. 3600 s) are better for rate-limited API endpoints.
- Use **`extra_instructions`** to steer the palette — e.g. `"use only warm colours"` or `"favour dark, moody compositions"`.
- Any OpenAI-compatible local model works; smaller models (7B–13B parameters) produce surprisingly good results.
