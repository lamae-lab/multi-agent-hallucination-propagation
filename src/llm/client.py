"""Unified LLM client for multi-provider chat completions.

All providers use the OpenAI-compatible /v1/chat/completions endpoint.
The active provider is determined by ACTIVE_PROVIDER in config.py.
"""

import requests

from src.llm import config


class LLMError(Exception):
    """Raised for unexpected LLM provider errors."""


class AuthenticationError(LLMError):
    """Raised when API key is invalid or expired."""


def get_active_provider() -> str:
    """Return the name of the currently active provider."""
    return config.ACTIVE_PROVIDER


def list_providers() -> list[str]:
    """Return the names of all configured providers."""
    return list(config.PROVIDERS.keys())


def chat(
    messages: list[dict],
    *,
    model: str | None = None,
    temperature: float = 0,
    max_tokens: int | None = None,
) -> str:
    """Send a chat completion request to the active provider.

    Args:
        messages: List of message dicts with 'role' and 'content' keys.
        model: Override the provider's default model.
        temperature: Sampling temperature. Default 0 (deterministic).
        max_tokens: Maximum tokens in completion.

    Returns:
        The content of the assistant's response message.

    Raises:
        ValueError: If the active provider is not recognized.
        ConnectionError: If the provider is unreachable.
        AuthenticationError: If the API key is invalid (HTTP 401/403).
        LLMError: For any other provider error.
    """
    provider_name = config.ACTIVE_PROVIDER

    if provider_name not in config.PROVIDERS:
        raise ValueError(
            f"Unknown provider '{provider_name}'. "
            f"Available providers: {', '.join(config.PROVIDERS.keys())}"
        )

    provider = config.PROVIDERS[provider_name]
    url = f"{provider['base_url']}/chat/completions"
    headers = {
        "Authorization": f"Bearer {provider['api_key']}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model or provider["model"],
        "messages": messages,
        "stream": False,
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    data = _send_request(provider_name, provider, url, headers, payload)

    try:
        msg = data["choices"][0]["message"]
        content = msg.get("content") or ""
        # Reasoning models (e.g. gpt-oss-120b) may put output in reasoning_content
        if not content and msg.get("reasoning_content"):
            content = msg["reasoning_content"]
        return content
    except (KeyError, IndexError, ValueError) as e:
        raise LLMError(
            f"Unexpected response format from provider '{provider_name}'. "
            f"Expected choices[0].message.content but got: {data}"
        ) from e


def chat_raw(
    messages: list[dict],
    *,
    tools: list[dict] | None = None,
    model: str | None = None,
    temperature: float = 0,
    max_tokens: int | None = None,
) -> dict:
    """Send a chat completion request and return the full assistant message.

    Unlike chat(), this returns the complete message dict including tool_calls.

    Args:
        messages: List of message dicts with 'role' and 'content' keys.
        tools: List of tool definitions in OpenAI format.
        model: Override the provider's default model.
        temperature: Sampling temperature. Default 0 (deterministic).
        max_tokens: Maximum tokens in completion.

    Returns:
        The full choices[0].message dict from the API response.
    """
    provider_name = config.ACTIVE_PROVIDER

    if provider_name not in config.PROVIDERS:
        raise ValueError(
            f"Unknown provider '{provider_name}'. "
            f"Available providers: {', '.join(config.PROVIDERS.keys())}"
        )

    provider = config.PROVIDERS[provider_name]
    url = f"{provider['base_url']}/chat/completions"
    headers = {
        "Authorization": f"Bearer {provider['api_key']}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model or provider["model"],
        "messages": messages,
        "stream": False,
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    data = _send_request(provider_name, provider, url, headers, payload)

    try:
        return data["choices"][0]["message"]
    except (KeyError, IndexError, ValueError) as e:
        raise LLMError(
            f"Unexpected response format from provider '{provider_name}'. "
            f"Expected choices[0].message but got: {data}"
        ) from e


def _send_request(provider_name, provider, url, headers, payload):
    """Send HTTP request to provider and return parsed JSON response."""
    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=provider.get("timeout", 120),
            verify=provider.get("verify_ssl", True),
        )
    except requests.ConnectionError:
        raise ConnectionError(
            f"Cannot reach provider '{provider_name}' at {provider['base_url']}. "
            f"Check that the server is running and the URL is correct."
        )
    except requests.Timeout:
        raise ConnectionError(
            f"Request to provider '{provider_name}' at {provider['base_url']} "
            f"timed out after {provider.get('timeout', 120)}s."
        )

    if response.status_code in (401, 403):
        raise AuthenticationError(
            f"Authentication failed for provider '{provider_name}'. "
            f"Check that the API key is valid. (HTTP {response.status_code})"
        )

    if response.status_code != 200:
        raise LLMError(
            f"Provider '{provider_name}' returned HTTP {response.status_code}: "
            f"{response.text}"
        )

    return response.json()
