"""LLM Provider Configuration.

Switch the active provider by changing ACTIVE_PROVIDER below.
All providers use the OpenAI-compatible /v1/chat/completions endpoint.
"""

# ---------------------------------------------------------------------------
# Change this single variable to switch providers: "ollama", "ilaas", "openai"
# ---------------------------------------------------------------------------
ACTIVE_PROVIDER = "ilaas"

# ---------------------------------------------------------------------------
# Provider definitions
# ---------------------------------------------------------------------------
PROVIDERS = {
    "ollama": {
        "base_url": "http://192.168.50.29:11434/v1",
        "api_key": "ollama",  # Required by client but ignored by Ollama
        "model": "qwen2.5:14b",
        "timeout": 120,
    },
    "ilaas": {
        "base_url": "https://llm.ilaas.fr/v1",
        "api_key": "REDACTED_KEY",
        "model": "gpt-oss-120b",
        "timeout": 120,
        "verify_ssl": False,  # ILAAS cert revocation check fails on Windows
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        # For OpenRouter: base_url = "https://openrouter.ai/api/v1"
        "api_key": "YOUR_KEY_HERE",
        "model": "gpt-4o",
        "timeout": 120,
    },
}
