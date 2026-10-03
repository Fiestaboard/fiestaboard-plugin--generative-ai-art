"""Sign in with OpenRouter, beside the pasted API key."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from plugins.generative_ai_art import OPENROUTER_BASE_URL, GenerativeAiArtPlugin

MANIFEST = json.loads((Path(__file__).parent.parent / "manifest.json").read_text())

GRID = {
    "theme": "test theme",
    "description": "A test composition",
    "grid": [["B" if (r + c) % 2 == 0 else "R" for c in range(22)] for r in range(6)],
}


def _ok():
    response = MagicMock(status_code=200)
    response.json.return_value = {"choices": [{"message": {"content": json.dumps(GRID)}}]}
    response.raise_for_status.return_value = None
    return response


def _status(code):
    response = MagicMock(status_code=code)
    response.raise_for_status.side_effect = requests.HTTPError(f"{code}", response=response)
    return response


def _plugin(config, token=None):
    plugin = GenerativeAiArtPlugin(MANIFEST)
    plugin.config = {"enabled": True, "refresh_seconds": 300, **config}
    plugin.get_oauth_token = lambda: token
    plugin.report_oauth_rejected = MagicMock(return_value=None)
    return plugin


# ── manifest ────────────────────────────────────────────────────────────────


def test_oauth_block_is_a_valid_openrouter_key_exchange():
    from src.oauth.provider import parse_provider_block, validate_provider_block

    block = MANIFEST["oauth"]
    assert validate_provider_block(block) == []
    assert "client_secret" not in block
    provider = parse_provider_block(block, MANIFEST["name"])
    assert provider.flows == ("key_exchange",)
    assert block["authorization_url"] == "https://openrouter.ai/auth"
    assert block["token_url"] == "https://openrouter.ai/api/v1/auth/keys"


def test_manifest_requires_a_core_that_knows_key_exchange():
    assert MANIFEST["fiestaboard_version"] == ">=9.9.0"


def test_api_key_is_no_longer_required_but_keeps_its_name():
    schema = MANIFEST["settings_schema"]
    assert "api_key" not in schema.get("required", [])
    assert {"api_key", "api_base_url", "model"} <= set(schema["properties"])


# ── which credential is used ────────────────────────────────────────────────


def test_pasted_api_key_wins_and_is_sent_unchanged():
    plugin = _plugin(
        {"api_key": "sk-test", "api_base_url": "http://localhost:11434/v1", "model": "llama3"},
        token="test_openrouter_key",
    )
    with patch("plugins.generative_ai_art.source.requests.post", return_value=_ok()) as post:
        result = plugin.fetch_data()
    assert result.available is True
    assert post.call_args.args[0] == "http://localhost:11434/v1/chat/completions"
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer sk-test"
    assert post.call_args.kwargs["json"]["model"] == "llama3"


def test_existing_key_only_config_uses_the_old_defaults():
    """A pre-1.5.0 config with just a key still goes to OpenAI with gpt-4o-mini."""
    plugin = _plugin({"api_key": "sk-test"}, token="test_openrouter_key")
    with patch("plugins.generative_ai_art.source.requests.post", return_value=_ok()) as post:
        assert plugin.fetch_data().available is True
    assert post.call_args.args[0] == "https://api.openai.com/v1/chat/completions"
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer sk-test"
    assert post.call_args.kwargs["json"]["model"] == "gpt-4o-mini"


def test_pasted_key_on_openrouter_keeps_the_model_name_as_typed():
    """The openai/ prefix is for sign-in only; a pasted OpenRouter key is untouched."""
    plugin = _plugin(
        {"api_key": "sk-or-test", "api_base_url": OPENROUTER_BASE_URL, "model": "gpt-4o-mini"},
    )
    with patch("plugins.generative_ai_art.source.requests.post", return_value=_ok()) as post:
        plugin.fetch_data()
    assert post.call_args.kwargs["json"]["model"] == "gpt-4o-mini"
    plugin.report_oauth_rejected.assert_not_called()


def test_signed_in_key_goes_to_openrouter():
    plugin = _plugin({"model": "gpt-4o-mini"}, token="test_openrouter_key")
    with patch("plugins.generative_ai_art.source.requests.post", return_value=_ok()) as post:
        result = plugin.fetch_data()
    assert result.available is True
    assert post.call_args.args[0] == f"{OPENROUTER_BASE_URL}/chat/completions"
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer test_openrouter_key"
    # A bare OpenAI model name gets OpenRouter's vendor prefix.
    assert post.call_args.kwargs["json"]["model"] == "openai/gpt-4o-mini"


def test_signed_in_keeps_an_openrouter_model_slug():
    plugin = _plugin({"model": "anthropic/claude-sonnet-4"}, token="test_openrouter_key")
    with patch("plugins.generative_ai_art.source.requests.post", return_value=_ok()) as post:
        plugin.fetch_data()
    assert post.call_args.kwargs["json"]["model"] == "anthropic/claude-sonnet-4"


def test_signed_in_ignores_a_saved_openai_base_url():
    plugin = _plugin({"api_base_url": "https://api.openai.com/v1"}, token="test_openrouter_key")
    with patch("plugins.generative_ai_art.source.requests.post", return_value=_ok()) as post:
        plugin.fetch_data()
    assert post.call_args.args[0].startswith(OPENROUTER_BASE_URL)


def test_neither_key_nor_sign_in_makes_no_request():
    plugin = _plugin({}, token=None)
    with patch("plugins.generative_ai_art.source.requests.post") as post:
        result = plugin.fetch_data()
    assert result.available is False
    assert "sign in" in result.error.lower()
    assert "api key" in result.error.lower()
    post.assert_not_called()


def test_older_core_without_oauth_still_works_with_a_key():
    plugin = GenerativeAiArtPlugin(MANIFEST)
    plugin.config = {"enabled": True, "api_key": "sk-test", "refresh_seconds": 300}
    with patch.object(GenerativeAiArtPlugin, "get_oauth_token", None, create=True):
        with patch("plugins.generative_ai_art.source.requests.post", return_value=_ok()):
            assert plugin.fetch_data().available is True
        with patch("plugins.generative_ai_art.source.requests.post") as post:
            plugin.config = {"enabled": True, "refresh_seconds": 300}
            assert plugin.fetch_data().available is False
            post.assert_not_called()


def test_token_lookup_failure_is_not_a_crash():
    plugin = _plugin({})

    def boom():
        raise RuntimeError("oauth service down")

    plugin.get_oauth_token = boom
    result = plugin.fetch_data()
    assert result.available is False


def test_new_token_is_used_on_the_next_fetch():
    plugin = _plugin({}, token="test_key_one")
    with patch("plugins.generative_ai_art.source.requests.post", return_value=_ok()) as post:
        plugin.fetch_data()
        plugin.get_oauth_token = lambda: "test_key_two"
        plugin.fetch_data()
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer test_key_two"


# ── rejection ───────────────────────────────────────────────────────────────


def test_rejected_sign_in_is_reported_and_asks_to_sign_in_again():
    plugin = _plugin({}, token="test_openrouter_key")
    with patch("plugins.generative_ai_art.source.requests.post", return_value=_status(401)) as post:
        result = plugin.fetch_data()
    plugin.report_oauth_rejected.assert_called_once()
    assert post.call_count == 1
    assert result.available is False
    assert "sign in again" in result.error.lower()


def test_rejected_sign_in_retries_once_with_a_replacement_key():
    plugin = _plugin({}, token="test_old_key")
    plugin.report_oauth_rejected = MagicMock(return_value="test_new_key")
    with patch(
        "plugins.generative_ai_art.source.requests.post", side_effect=[_status(401), _ok()]
    ) as post:
        result = plugin.fetch_data()
    assert result.available is True
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer test_new_key"


def test_pasted_key_401_is_never_reported():
    plugin = _plugin({"api_key": "sk-test"}, token="test_openrouter_key")
    with patch("plugins.generative_ai_art.source.requests.post", return_value=_status(401)):
        result = plugin.fetch_data()
    plugin.report_oauth_rejected.assert_not_called()
    assert result.available is False


@pytest.mark.parametrize("code", [403, 429, 500])
def test_other_errors_are_not_reported_as_rejection(code):
    plugin = _plugin({}, token="test_openrouter_key")
    with patch("plugins.generative_ai_art.source.requests.post", return_value=_status(code)):
        plugin.fetch_data()
    plugin.report_oauth_rejected.assert_not_called()


def test_rejection_on_a_core_without_report_hook():
    plugin = _plugin({}, token="test_openrouter_key")
    del plugin.report_oauth_rejected
    with patch.object(GenerativeAiArtPlugin, "report_oauth_rejected", None, create=True):
        with patch("plugins.generative_ai_art.source.requests.post", return_value=_status(401)):
            result = plugin.fetch_data()
    assert result.available is False
    assert "sign in again" in result.error.lower()


# ── validation ──────────────────────────────────────────────────────────────


def test_config_without_api_key_is_valid_for_sign_in():
    plugin = GenerativeAiArtPlugin(MANIFEST)
    assert plugin.validate_config({"enabled": True, "refresh_seconds": 300}) == []
