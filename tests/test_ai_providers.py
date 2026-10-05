"""FiestaBoard's AI providers, beside the plugin's own (legacy) API key."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from plugins.generative_ai_art import (
    NEEDS_NEWER_CORE,
    GenerativeAiArtPlugin,
)
from src.ai.plugin_api import (
    AICompletion,
    AINotConfiguredError,
    AIProviderError,
    AIRejectedError,
)

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


def _completion(text=None, model="provider-model"):
    return AICompletion(text=text or json.dumps(GRID), model=model, provider_id="p1")


def _plugin(config, complete=None):
    plugin = GenerativeAiArtPlugin(MANIFEST)
    plugin.config = {"enabled": True, "refresh_seconds": 300, **config}
    if complete is not None:
        plugin.ai_complete = complete
    return plugin


# ── manifest ────────────────────────────────────────────────────────────────


def test_no_plugin_level_oauth_sign_in():
    assert "oauth" not in MANIFEST


def test_manifest_requires_a_core_with_ai_complete():
    assert MANIFEST["fiestaboard_version"] == ">=9.11.0"


def test_provider_picker_uses_the_core_ai_providers_source():
    props = MANIFEST["settings_schema"]["properties"]
    field = props["ai_provider"]
    assert field["type"] == "string"
    assert field["default"] == ""
    assert field["ui:widget"] == "remote-options"
    assert field["ui:options"]["options_id"] == "ai_providers"
    assert props["ai_model"]["type"] == "string"
    assert props["ai_model"].get("default", "") == ""


def test_provider_fields_come_first_and_legacy_settings_keep_their_names():
    schema = MANIFEST["settings_schema"]
    names = list(schema["properties"])
    assert names.index("ai_provider") < names.index("api_key")
    assert {"api_key", "api_base_url", "model"} <= set(names)
    assert "api_key" not in schema.get("required", [])
    assert schema["properties"]["api_key"]["secret"] is True


def test_manifest_passes_core_validation():
    from src.plugins.manifest import (
        settings_schema_ui_warnings,
        validate_manifest,
        validate_settings_schema_ui,
    )

    assert validate_manifest(MANIFEST)[0] is True
    assert validate_settings_schema_ui(MANIFEST["settings_schema"]) == []
    assert settings_schema_ui_warnings(MANIFEST["settings_schema"]) == []


def test_ai_providers_options_are_answered_by_core():
    """The plugin does not override get_options, so core's picker is reached."""
    assert "get_options" not in GenerativeAiArtPlugin.__dict__


# ── a saved API key wins, byte for byte ─────────────────────────────────────


def test_pasted_api_key_wins_and_is_sent_unchanged():
    complete = MagicMock()
    plugin = _plugin(
        {
            "api_key": "sk-test",
            "api_base_url": "http://localhost:11434/v1",
            "model": "llama3",
            "ai_provider": "p1",
        },
        complete,
    )
    with patch("plugins.generative_ai_art.source.requests.post", return_value=_ok()) as post:
        result = plugin.fetch_data()
    assert result.available is True
    complete.assert_not_called()
    assert post.call_args.args[0] == "http://localhost:11434/v1/chat/completions"
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer sk-test"
    assert post.call_args.kwargs["json"]["model"] == "llama3"
    assert result.data["model"] == "llama3"


def test_existing_key_only_config_uses_the_old_defaults():
    plugin = _plugin({"api_key": "sk-test"}, MagicMock())
    with patch("plugins.generative_ai_art.source.requests.post", return_value=_ok()) as post:
        assert plugin.fetch_data().available is True
    assert post.call_args.args[0] == "https://api.openai.com/v1/chat/completions"
    assert post.call_args.kwargs["json"]["model"] == "gpt-4o-mini"
    assert post.call_args.kwargs["json"]["temperature"] == 1.2


def test_pasted_openrouter_key_keeps_the_model_name_as_typed():
    plugin = _plugin(
        {"api_key": "sk-or", "api_base_url": "https://openrouter.ai/api/v1", "model": "gpt-4o-mini"}
    )
    with patch("plugins.generative_ai_art.source.requests.post", return_value=_ok()) as post:
        plugin.fetch_data()
    assert post.call_args.kwargs["json"]["model"] == "gpt-4o-mini"


def test_api_key_works_on_an_older_core_without_ai_complete():
    plugin = _plugin({"api_key": "sk-test"})
    plugin.ai_complete = None  # what getattr sees on a core before 9.11.0
    with patch("plugins.generative_ai_art.source.requests.post", return_value=_ok()):
        assert plugin.fetch_data().available is True


# ── FiestaBoard's AI providers ──────────────────────────────────────────────


def test_no_key_uses_the_default_provider_and_its_model():
    complete = MagicMock(return_value=_completion())
    plugin = _plugin({}, complete)
    with patch("plugins.generative_ai_art.source.requests.post") as post:
        result = plugin.fetch_data()
    post.assert_not_called()
    assert result.available is True
    assert len(result.formatted_lines) == 6
    kwargs = complete.call_args.kwargs
    assert kwargs["provider_id"] is None
    assert kwargs["model"] is None
    assert kwargs["temperature"] == 1.2
    assert kwargs["max_tokens"] > 0
    messages = complete.call_args.args[0]
    assert [m["role"] for m in messages] == ["system", "user"]
    assert result.data["model"] == "provider-model"


def test_picked_provider_and_model_are_passed_through():
    complete = MagicMock(return_value=_completion())
    plugin = _plugin(
        {"ai_provider": "openrouter", "ai_model": "anthropic/claude-3-haiku", "temperature": 0.9},
        complete,
    )
    plugin.fetch_data()
    kwargs = complete.call_args.kwargs
    assert kwargs["provider_id"] == "openrouter"
    assert kwargs["model"] == "anthropic/claude-3-haiku"
    assert kwargs["temperature"] == 0.9


def test_provider_path_ignores_the_legacy_model_and_base_url():
    complete = MagicMock(return_value=_completion())
    plugin = _plugin({"api_base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini"}, complete)
    plugin.fetch_data()
    assert complete.call_args.kwargs["model"] is None


def test_unparseable_art_is_retried_through_the_same_provider():
    complete = MagicMock(return_value=_completion(text="not json"))
    plugin = _plugin({}, complete)
    plugin.fetch_data()
    assert complete.call_count == 3


def test_older_core_without_key_asks_to_update_or_paste_a_key():
    plugin = _plugin({})
    plugin.ai_complete = None
    with patch("plugins.generative_ai_art.source.requests.post") as post:
        result = plugin.fetch_data()
    post.assert_not_called()
    assert result.available is False
    assert result.error == NEEDS_NEWER_CORE
    assert "Update FiestaBoard to use its AI providers, or paste an API key" in result.error


def test_ai_not_configured_is_unavailable_with_the_reason():
    complete = MagicMock(side_effect=AINotConfiguredError("AI is turned off."))
    result = _plugin({}, complete).fetch_data()
    assert result.available is False
    assert "Settings → AI Providers" in result.error
    assert "AI is turned off." in result.error
    assert complete.call_count == 1


def test_rejected_provider_asks_to_reconnect():
    complete = MagicMock(side_effect=AIRejectedError("Sign in to OpenRouter again."))
    result = _plugin({}, complete).fetch_data()
    assert result.available is False
    assert "Sign in to OpenRouter again." in result.error
    assert complete.call_count == 1


def test_provider_error_falls_back_to_the_last_piece_for_this_board():
    complete = MagicMock(return_value=_completion())
    plugin = _plugin({}, complete)
    first = plugin.fetch_data()
    complete.side_effect = AIProviderError("Provider unreachable.")
    second = plugin.fetch_data()
    assert second.available is True
    assert second.data["_fallback"] is True
    assert second.formatted_lines == first.formatted_lines


def test_provider_error_without_a_piece_is_unavailable():
    complete = MagicMock(side_effect=AIProviderError("Provider unreachable."))
    result = _plugin({}, complete).fetch_data()
    assert result.available is False
    assert "Provider unreachable." in result.error


def test_a_newly_set_ai_complete_is_used_on_the_next_fetch():
    plugin = _plugin({}, MagicMock(return_value=_completion(model="m1")))
    assert plugin.fetch_data().data["model"] == "m1"
    plugin.ai_complete = MagicMock(return_value=_completion(model="m2"))
    assert plugin.fetch_data().data["model"] == "m2"


def test_switching_to_a_pasted_key_stops_using_the_providers():
    complete = MagicMock(return_value=_completion())
    plugin = _plugin({}, complete)
    plugin.fetch_data()
    old = dict(plugin.config)
    plugin.config = {**old, "api_key": "sk-test"}
    plugin.on_config_change(old, plugin.config)
    with patch("plugins.generative_ai_art.source.requests.post", return_value=_ok()) as post:
        plugin.fetch_data()
    assert post.called
    assert complete.call_count == 1


def test_config_without_api_key_is_valid():
    assert _plugin({}).validate_config({"enabled": True, "refresh_seconds": 300}) == []


def test_unexpected_error_is_not_a_crash():
    complete = MagicMock(side_effect=RuntimeError("boom"))
    result = _plugin({}, complete).fetch_data()
    assert result.available is False


@pytest.mark.parametrize("value", ["", None])
def test_blank_provider_means_default(value):
    complete = MagicMock(return_value=_completion())
    _plugin({"ai_provider": value, "ai_model": value}, complete).fetch_data()
    assert complete.call_args.kwargs["provider_id"] is None
    assert complete.call_args.kwargs["model"] is None


# ── through core's real ai_complete (no mock of the plugin API) ─────────────

PROVIDER = {"id": "p1", "name": "Test", "protocol": "openai", "base_url": "https://example.test/v1",
            "api_key": "k", "default_model": "core-model"}


def _core_plugin(config, block):
    """A plugin whose ai_complete is core's own, with settings and HTTP stubbed."""
    plugin = _plugin(config)
    patches = [patch("src.ai.plugin_api._providers_block", return_value=block)]
    return plugin, patches


def _run(plugin, patches, reply=None):
    from contextlib import ExitStack

    async def post(provider, payload, **kwargs):
        post.calls.append((provider, payload))
        return {"choices": [{"message": {"content": reply or json.dumps(GRID)}}]}

    post.calls = []
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        stack.enter_context(patch("src.ai.plugin_api._post_chat_completion", post))
        http = stack.enter_context(patch("plugins.generative_ai_art.source.requests.post"))
        result = plugin.fetch_data()
    return result, post.calls, http


@pytest.mark.parametrize(
    "block, reason",
    [
        ({"enabled": False, "providers": [PROVIDER]}, "not enabled"),
        ({"enabled": True, "providers": []}, "No AI providers"),
        ({}, "not enabled"),
    ],
)
def test_core_ai_disabled_or_missing_provider_is_unavailable(block, reason):
    plugin, patches = _core_plugin({}, block)
    result, calls, http = _run(plugin, patches)
    assert result.available is False
    assert "Settings → AI Providers" in result.error
    assert reason in result.error
    assert calls == []
    http.assert_not_called()


def test_core_picked_provider_that_was_deleted_is_unavailable():
    plugin, patches = _core_plugin({"ai_provider": "gone"}, {"enabled": True, "providers": [PROVIDER]})
    result, calls, _ = _run(plugin, patches)
    assert result.available is False
    assert "'gone' not found" in result.error
    assert calls == []


def test_core_provider_path_end_to_end():
    plugin, patches = _core_plugin({"temperature": 0.9}, {"enabled": True, "providers": [PROVIDER]})
    result, calls, http = _run(plugin, patches)
    assert result.available is True
    assert len(result.formatted_lines) == 6
    assert result.data["model"] == "core-model"
    http.assert_not_called()
    (provider, payload), = calls
    assert provider["id"] == "p1"
    assert payload["model"] == "core-model"
    assert payload["temperature"] == 0.9
    assert payload["messages"][0]["role"] == "system"


def test_core_saved_key_never_touches_the_providers():
    plugin, patches = _core_plugin({"api_key": "sk-test"}, {"enabled": True, "providers": [PROVIDER]})
    from contextlib import ExitStack

    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        complete = stack.enter_context(patch("src.ai.plugin_api.complete"))
        post = stack.enter_context(
            patch("plugins.generative_ai_art.source.requests.post", return_value=_ok())
        )
        result = plugin.fetch_data()
    assert result.available is True
    complete.assert_not_called()
    assert post.call_args.args[0] == "https://api.openai.com/v1/chat/completions"


# ── Anthropic accepts temperature 0-1 (cores before FiestaBoard caps it) ───


def _providers(*protocols):
    return lambda: [
        {"id": f"p{i}", "protocol": proto, "default": i == 0} for i, proto in enumerate(protocols)
    ]


def test_anthropic_provider_gets_temperature_capped_at_one():
    complete = MagicMock(return_value=_completion())
    plugin = _plugin({"ai_provider": "p1", "temperature": 1.2}, complete)
    plugin.ai_providers = _providers("openai", "anthropic")
    plugin.fetch_data()
    assert complete.call_args.kwargs["temperature"] == 1.0


def test_default_anthropic_provider_gets_temperature_capped_at_one():
    complete = MagicMock(return_value=_completion())
    plugin = _plugin({"temperature": 1.4}, complete)
    plugin.ai_providers = _providers("anthropic", "openai")
    plugin.fetch_data()
    assert complete.call_args.kwargs["temperature"] == 1.0


def test_other_protocols_keep_the_configured_temperature():
    complete = MagicMock(return_value=_completion())
    plugin = _plugin({"ai_provider": "p1", "temperature": 1.2}, complete)
    plugin.ai_providers = _providers("anthropic", "openai")
    plugin.fetch_data()
    assert complete.call_args.kwargs["temperature"] == 1.2


def test_unreadable_providers_keep_the_configured_temperature():
    complete = MagicMock(return_value=_completion())
    plugin = _plugin({"temperature": 1.2}, complete)
    plugin.ai_providers = MagicMock(side_effect=RuntimeError("test unreadable"))
    plugin.fetch_data()
    assert complete.call_args.kwargs["temperature"] == 1.2
