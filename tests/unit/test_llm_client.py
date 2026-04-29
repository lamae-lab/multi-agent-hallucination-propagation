"""Tests for LLM client."""

from unittest.mock import MagicMock, patch

import pytest

from src.llm import config
from src.llm.client import (
    AuthenticationError,
    LLMError,
    chat,
    get_active_provider,
    list_providers,
)


def test_get_active_provider_returns_config_value():
    original = config.ACTIVE_PROVIDER
    try:
        config.ACTIVE_PROVIDER = "ilaas"
        assert get_active_provider() == "ilaas"
        config.ACTIVE_PROVIDER = "ollama"
        assert get_active_provider() == "ollama"
    finally:
        config.ACTIVE_PROVIDER = original


def test_list_providers_returns_all_three():
    providers = list_providers()
    assert "ollama" in providers
    assert "ilaas" in providers
    assert "openai" in providers


def test_chat_raises_value_error_for_invalid_provider():
    original = config.ACTIVE_PROVIDER
    try:
        config.ACTIVE_PROVIDER = "nonexistent"
        with pytest.raises(ValueError, match="Unknown provider 'nonexistent'"):
            chat([{"role": "user", "content": "test"}])
    finally:
        config.ACTIVE_PROVIDER = original


@patch("src.llm.client.requests.post")
def test_chat_returns_response_content(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "Hello!"}}]
    }
    mock_post.return_value = mock_response

    original = config.ACTIVE_PROVIDER
    try:
        config.ACTIVE_PROVIDER = "ilaas"
        result = chat([{"role": "user", "content": "hi"}])
        assert result == "Hello!"
    finally:
        config.ACTIVE_PROVIDER = original


@patch("src.llm.client.requests.post")
def test_chat_sends_correct_payload(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "ok"}}]
    }
    mock_post.return_value = mock_response

    original = config.ACTIVE_PROVIDER
    try:
        config.ACTIVE_PROVIDER = "ilaas"
        chat(
            [{"role": "user", "content": "test"}],
            temperature=0.5,
            max_tokens=100,
        )

        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs["json"]
        assert payload["model"] == config.PROVIDERS["ilaas"]["model"]
        assert payload["messages"] == [{"role": "user", "content": "test"}]
        assert payload["temperature"] == 0.5
        assert payload["max_tokens"] == 100
        assert payload["stream"] is False
    finally:
        config.ACTIVE_PROVIDER = original


@patch("src.llm.client.requests.post")
def test_chat_uses_model_override(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "ok"}}]
    }
    mock_post.return_value = mock_response

    original = config.ACTIVE_PROVIDER
    try:
        config.ACTIVE_PROVIDER = "ilaas"
        chat([{"role": "user", "content": "test"}], model="custom-model")
        payload = mock_post.call_args.kwargs["json"]
        assert payload["model"] == "custom-model"
    finally:
        config.ACTIVE_PROVIDER = original


@patch("src.llm.client.requests.post")
def test_chat_sends_bearer_auth(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "ok"}}]
    }
    mock_post.return_value = mock_response

    original = config.ACTIVE_PROVIDER
    try:
        config.ACTIVE_PROVIDER = "ilaas"
        chat([{"role": "user", "content": "test"}])
        headers = mock_post.call_args.kwargs["headers"]
        expected_key = config.PROVIDERS["ilaas"]["api_key"]
        assert headers["Authorization"] == f"Bearer {expected_key}"
    finally:
        config.ACTIVE_PROVIDER = original


@patch("src.llm.client.requests.post")
def test_chat_raises_auth_error_on_401(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_post.return_value = mock_response

    original = config.ACTIVE_PROVIDER
    try:
        config.ACTIVE_PROVIDER = "ilaas"
        with pytest.raises(AuthenticationError, match="Authentication failed"):
            chat([{"role": "user", "content": "test"}])
    finally:
        config.ACTIVE_PROVIDER = original


@patch("src.llm.client.requests.post")
def test_chat_raises_auth_error_on_403(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 403
    mock_post.return_value = mock_response

    original = config.ACTIVE_PROVIDER
    try:
        config.ACTIVE_PROVIDER = "ilaas"
        with pytest.raises(AuthenticationError, match="Authentication failed"):
            chat([{"role": "user", "content": "test"}])
    finally:
        config.ACTIVE_PROVIDER = original


@patch("src.llm.client.requests.post")
def test_chat_raises_llm_error_on_500(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    mock_post.return_value = mock_response

    original = config.ACTIVE_PROVIDER
    try:
        config.ACTIVE_PROVIDER = "ilaas"
        with pytest.raises(LLMError, match="HTTP 500"):
            chat([{"role": "user", "content": "test"}])
    finally:
        config.ACTIVE_PROVIDER = original


@patch("src.llm.client.requests.post")
def test_chat_raises_llm_error_on_bad_response_format(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"unexpected": "format"}
    mock_response.text = '{"unexpected": "format"}'
    mock_post.return_value = mock_response

    original = config.ACTIVE_PROVIDER
    try:
        config.ACTIVE_PROVIDER = "ilaas"
        with pytest.raises(LLMError, match="Unexpected response format"):
            chat([{"role": "user", "content": "test"}])
    finally:
        config.ACTIVE_PROVIDER = original


@patch("src.llm.client.requests.post")
def test_chat_raises_connection_error_on_unreachable(mock_post):
    import requests

    mock_post.side_effect = requests.ConnectionError("Connection refused")

    original = config.ACTIVE_PROVIDER
    try:
        config.ACTIVE_PROVIDER = "ollama"
        with pytest.raises(ConnectionError, match="Cannot reach provider 'ollama'"):
            chat([{"role": "user", "content": "test"}])
    finally:
        config.ACTIVE_PROVIDER = original


@patch("src.llm.client.requests.post")
def test_chat_raises_connection_error_on_timeout(mock_post):
    import requests

    mock_post.side_effect = requests.Timeout("Read timed out")

    original = config.ACTIVE_PROVIDER
    try:
        config.ACTIVE_PROVIDER = "ollama"
        with pytest.raises(ConnectionError, match="timed out"):
            chat([{"role": "user", "content": "test"}])
    finally:
        config.ACTIVE_PROVIDER = original


def test_runtime_provider_switching():
    """Verify SC-003: switching ACTIVE_PROVIDER takes effect immediately."""
    original = config.ACTIVE_PROVIDER
    try:
        config.ACTIVE_PROVIDER = "ollama"
        assert get_active_provider() == "ollama"
        config.ACTIVE_PROVIDER = "ilaas"
        assert get_active_provider() == "ilaas"
        config.ACTIVE_PROVIDER = "openai"
        assert get_active_provider() == "openai"
    finally:
        config.ACTIVE_PROVIDER = original
