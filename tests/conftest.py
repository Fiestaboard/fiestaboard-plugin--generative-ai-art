"""Plugin test fixtures."""

import pytest


@pytest.fixture
def sample_manifest():
    """Minimal manifest for plugin initialisation."""
    return {
        "id": "generative_ai_art",
        "name": "Generative AI Art",
        "version": "1.1.0",
        "description": "Test",
        "author": "FiestaBoard Team",
        "min_refresh_seconds": 300,
        "settings_schema": {
            "type": "object",
            "properties": {
                "refresh_seconds": {
                    "type": "integer",
                    "default": 300,
                    "minimum": 300,
                }
            },
        },
    }


@pytest.fixture
def base_config():
    """Minimal valid plugin configuration."""
    return {
        "enabled": True,
        "api_key": "sk-test",
        "api_base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "device_type": "flagship",
        "temperature": 1.2,
        "refresh_seconds": 300,
    }
