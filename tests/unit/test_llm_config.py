"""Tests for LLM provider configuration."""

from src.llm import config


def test_active_provider_is_valid():
    assert config.ACTIVE_PROVIDER in config.PROVIDERS


def test_all_three_providers_defined():
    assert "ollama" in config.PROVIDERS
    assert "ilaas" in config.PROVIDERS
    assert "openai" in config.PROVIDERS


def test_provider_has_required_keys():
    required_keys = {"base_url", "api_key", "model", "timeout"}
    for name, provider in config.PROVIDERS.items():
        missing = required_keys - set(provider.keys())
        assert not missing, f"Provider '{name}' missing keys: {missing}"


def test_provider_base_urls_are_strings():
    for name, provider in config.PROVIDERS.items():
        assert isinstance(provider["base_url"], str), f"Provider '{name}' base_url not a string"
        assert provider["base_url"].startswith("http"), f"Provider '{name}' base_url invalid"


def test_provider_api_keys_are_nonempty():
    for name, provider in config.PROVIDERS.items():
        assert isinstance(provider["api_key"], str), f"Provider '{name}' api_key not a string"
        assert len(provider["api_key"]) > 0, f"Provider '{name}' api_key is empty"


def test_provider_models_are_nonempty():
    for name, provider in config.PROVIDERS.items():
        assert isinstance(provider["model"], str), f"Provider '{name}' model not a string"
        assert len(provider["model"]) > 0, f"Provider '{name}' model is empty"


def test_ollama_base_url():
    assert "192.168.50.29" in config.PROVIDERS["ollama"]["base_url"]
    assert "11434" in config.PROVIDERS["ollama"]["base_url"]


def test_ilaas_base_url():
    assert "llm.ilaas.fr" in config.PROVIDERS["ilaas"]["base_url"]


def test_openai_base_url():
    assert "api.openai.com" in config.PROVIDERS["openai"]["base_url"]
